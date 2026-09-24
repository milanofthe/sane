//! A sparse LU as one program: the entry values are its parameter-pure
//! inputs and the right-hand side its main inputs, so a numeric
//! factorization is the prolog and a solve the main phase, on whichever
//! backend runs the tape.
//!
//! The pivot rows are fixed when the program is built ([`Plan`]) and
//! guarded; after a prolog, [`LuProgram::factored`] reads the guard and the
//! factors' finiteness from the state, and on a failure the consumer
//! rebuilds with [`LuProgram::repivot`] on the values. How often, and when
//! to hand a system to another solver instead, is the consumer's policy.

use crate::field::F64;
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};
use crate::tape::{input_index, Tape};

use super::{solve_planned, solve_supernodal_planned, supernodes, Pattern, Plan, SparseRows};

/// When the supernodal program (panels, dense kernels) replaces the scalar
/// one: the system has at least `min_n` unknowns (below that the analysis
/// is skipped), and panels at least `min_width` wide (the width from which
/// a panel's products are kernels) cover a `min_share` of them, or the
/// factorization predicts at least `min_flops` and such panels carry a
/// `min_flop_share` of them. The second is a large system whose flops
/// gather in a few wide separators (a mesh): its many narrow panels cost
/// more than scalar steps, and the kernels pay for that only at scale.
#[derive(Clone, Copy, Debug)]
pub struct Panels {
    pub min_n: usize,
    pub min_width: usize,
    pub min_share: f64,
    pub min_flops: usize,
    pub min_flop_share: f64,
}

impl Default for Panels {
    fn default() -> Self {
        Panels {
            min_n: 64,
            min_width: 8,
            min_share: 0.5,
            min_flops: 10_000_000,
            min_flop_share: 0.8,
        }
    }
}

/// The LU of one sparsity pattern along one [`Plan`], compiled.
pub struct LuProgram {
    n: usize,
    /// The distinct `(row, column)` positions of the values, in the order
    /// a caller supplies them.
    entries: Vec<(usize, usize)>,
    pattern: Pattern,
    plan: Plan,
    panels: Option<Panels>,
    /// Where entry `k` sits among the inputs, and the right-hand side of
    /// unknown `i` at `nnz + rhs_pos[i]`: the supernodal program orders
    /// both so its kernels read them in place.
    entry_pos: Vec<usize>,
    rhs_pos: Vec<usize>,
    tape: Tape,
    /// The guard's slot in the prolog state; `None` when every step has a
    /// single candidate row and the guard is the constant one.
    guard: Option<usize>,
    fill: usize,
    /// `(panels, widest)` of a supernodal program.
    supernodal: Option<(usize, usize)>,
}

impl LuProgram {
    /// The program for the values at `entries` (distinct positions of an
    /// `n` by `n` matrix) along `plan`, supernodal where `panels` says it
    /// pays.
    pub fn build(
        n: usize,
        entries: Vec<(usize, usize)>,
        plan: Plan,
        panels: Option<Panels>,
    ) -> LuProgram {
        let mut pattern: Pattern = vec![Vec::new(); n];
        for &(i, j) in &entries {
            pattern[i].push(j);
        }
        let nnz = entries.len();
        let sn = panels
            .filter(|p| n >= p.min_n)
            .map(|p| (p, supernodes(&pattern, &plan)))
            .filter(|(p, sn)| {
                let wide: usize = sn.widths().iter().filter(|&&w| w >= p.min_width).sum();
                wide as f64 >= p.min_share * n.max(1) as f64
                    || (plan.cost.flops >= p.min_flops
                        && sn.flop_share(p.min_width) >= p.min_flop_share)
            })
            .map(|(_, sn)| sn);
        let (entry_pos, rhs_pos) = match &sn {
            Some(sn) => (sn.value_order(&entries), sn.rhs_order()),
            None => ((0..nnz).collect(), (0..n).collect()),
        };
        // One symbol per input position: the entries, then the right-hand
        // side.
        let mut g: Graph<F64> = Graph::new();
        let mut syms: Vec<SymbolId> = Vec::with_capacity(nnz + n);
        let mut at: Vec<ExprId> = Vec::with_capacity(nnz + n);
        for p in 0..nnz + n {
            let e = g.sym(&format!("lu.{p}"));
            let Node::Symbol(s) = *g.node(e) else {
                unreachable!("sym is a symbol")
            };
            syms.push(s);
            at.push(e);
        }
        let mut rows: SparseRows = vec![Vec::new(); n];
        for (k, &(i, j)) in entries.iter().enumerate() {
            rows[i].push((j, at[entry_pos[k]]));
        }
        let b: Vec<ExprId> = (0..n).map(|i| at[nnz + rhs_pos[i]]).collect();
        let solved = match &sn {
            Some(sn) => solve_supernodal_planned(&mut g, &rows, &plan, sn, &b),
            None => solve_planned(&mut g, &rows, &plan, &b),
        };
        let unguarded = solved.pivots_ok == g.one();
        let mut roots = solved.x;
        roots.push(solved.pivots_ok);
        let mut pure = vec![true; nnz];
        pure.resize(nnz + n, false);
        let tape = Tape::compile_split(&g, &roots, &syms, &pure);
        // The guard depends on the values only: the prolog computes it and
        // leaves it in the state.
        let guard = (!unguarded).then(|| {
            let k = tape.outputs()[n];
            assert!(
                input_index(k).is_none() && (k as usize) < tape.state_len(),
                "the pivot guard is a prolog value"
            );
            k as usize
        });
        LuProgram {
            n,
            entries,
            pattern,
            plan,
            panels,
            entry_pos,
            rhs_pos,
            tape,
            guard,
            fill: solved.fill,
            supernodal: sn.map(|sn| (sn.n_panels(), sn.max_width())),
        }
    }

    /// The same system with the pivot rows a numeric elimination takes on
    /// the magnitudes `mags` of the values (in entry order).
    pub fn repivot(&self, mags: &[f64]) -> LuProgram {
        let at: rustc_hash::FxHashMap<(usize, usize), f64> = self
            .entries
            .iter()
            .copied()
            .zip(mags.iter().copied())
            .collect();
        let plan = self.plan.repivot(&self.pattern, |i, j| {
            at.get(&(i, j)).copied().unwrap_or(0.0)
        });
        LuProgram::build(self.n, self.entries.clone(), plan, self.panels)
    }

    pub fn tape(&self) -> &Tape {
        &self.tape
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn entries(&self) -> &[(usize, usize)] {
        &self.entries
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    /// Fill of the factorization.
    pub fn fill(&self) -> usize {
        self.fill
    }
    /// `(panels, widest panel)` when the program is supernodal.
    pub fn supernodal(&self) -> Option<(usize, usize)> {
        self.supernodal
    }
    /// Length of the input vector: the entries, then the right-hand side.
    pub fn input_len(&self) -> usize {
        self.entries.len() + self.n
    }

    /// Put the values (entry order) into `inputs`, each scaled by its row's
    /// `row_scale` when given.
    pub fn write_values(&self, values: &[f64], row_scale: Option<&[f64]>, inputs: &mut [f64]) {
        for (k, (&(i, _), &v)) in self.entries.iter().zip(values).enumerate() {
            inputs[self.entry_pos[k]] = row_scale.map_or(v, |s| v * s[i]);
        }
    }

    /// Put a right-hand side into `inputs`, scaled as the values were.
    pub fn write_rhs(&self, rhs: &[f64], row_scale: Option<&[f64]>, inputs: &mut [f64]) {
        let nnz = self.entries.len();
        for (i, &v) in rhs.iter().enumerate() {
            inputs[nnz + self.rhs_pos[i]] = row_scale.map_or(v, |s| v * s[i]);
        }
    }

    /// After a prolog over `inputs` into `work`: whether the pivot guard
    /// held and the values and factors are finite. All of it depends on the
    /// values only, so no substitution is needed to learn it.
    pub fn factored(&self, inputs: &[f64], work: &[f64]) -> bool {
        self.guard.is_none_or(|k| work[k] == 1.0)
            && inputs[..self.entries.len()].iter().all(|v| v.is_finite())
            && work[..self.tape.state_len()].iter().all(|v| v.is_finite())
    }

    /// The unknowns among a main phase's outputs.
    pub fn solution<'o>(&self, out: &'o [f64]) -> &'o [f64] {
        &out[..self.n]
    }
}
