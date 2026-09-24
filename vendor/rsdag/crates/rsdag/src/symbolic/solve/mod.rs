//! A linear solve as graph ops: the vertical integration of the solve.
//!
//! A simulator is segmented -- assemble, factor, solve, update -- with a data
//! hand-off between the stages and a generic sparse solver that must order,
//! allocate and dispatch at run time because it knows nothing about the
//! matrix until it sees it. Here the matrix is a graph whose sparsity pattern
//! is fixed at build time, so the elimination is a fixed sequence of ring
//! ops: a DAG like any other. [`lu_static`] builds it, [`StaticLu::solve_static`] the
//! two triangular solves, and [`newton_step`] composes residual, Jacobian,
//! factorization, solve and update into one expression per unknown, which
//! then compiles to one native function with the rest of the model.
//!
//! Measured against an outsourced multifrontal solve on a 512-unknown
//! circuit residual, the integrated step was about 30x faster: not from the
//! arithmetic, but from everything a generic solver does per call and this
//! form does once, at build time.
//!
//! Everything here is sparse, and the structure is worked out the way a
//! sparse solver works it out, before any arithmetic exists: the
//! Jacobian comes as [`SparseRows`], the [`Pattern`] is the columns of
//! each row, [`plan`] permutes it to block upper triangular form
//! ([`btf`]), orders every diagonal block for fill ([`amd`]) and predicts
//! the factorization's fill and flops from the elimination tree
//! ([`predict`]); [`solve_planned`] then emits one static LU per block in
//! Crout form (one dot product per factor entry, [`lu_static`]) and the
//! substitution across blocks. The build costs what the factorization's
//! fill costs, as for any direct solver, and a million unknowns with a
//! circuit-like pattern is a program of a few tens of ops per unknown.
//! Every flop of the factorization is a node, so a pattern with heavy fill
//! (a 2D mesh past a few thousand unknowns) is a large program, which is
//! what [`Plan::cost`] is for: below a few hundred flops per unknown the
//! graph wins, above it a sparse solver does. The elimination order and
//! the pivot rows are fixed at build time, as in a static solver, and
//! guarded: each step whose column has more than one structural
//! candidate checks at run time that its pivot row still dominates the
//! column by [`PIVOT_TOLERANCE`] ([`Solved::pivots_ok`]). When the guard
//! fails, [`Plan::repivot`] picks the rows a numeric elimination with
//! partial pivoting takes on the current values, and the program is
//! rebuilt: the static program's cost per step, a numeric solver's
//! pivoting across steps.

pub mod amd;
pub mod block;
pub mod btf;
pub mod num;
pub mod predict;
pub mod program;
pub mod supernodal;

pub use block::{block_pattern, solve_block_planned, solve_block_planned_sizes, Block, BlockRows};
pub use num::{Cx, Num};
pub use program::{LuProgram, Panels};
pub use supernodal::{solve_supernodal_planned, supernodes, Supernodes};

use rustc_hash::FxHashMap as HashMap;

pub use crate::autodiff::SparseRows;
use crate::field::Field;
use crate::graph::Graph;
use crate::node::ExprId;
pub use btf::Btf;
pub use predict::Cost;

/// The structural pattern of a square matrix: the columns present in each
/// row, ascending.
pub type Pattern = Vec<Vec<usize>>;

/// A static LU factorization as graph ops, in a given elimination order.
pub struct StaticLu {
    /// The elimination order: `order[k]` is the original index eliminated
    /// at step `k`.
    pub order: Vec<usize>,
    /// Unit-lower multipliers and upper entries, keyed by *permuted*
    /// position `(k, l)` (step indices). Only the structurally nonzero ones.
    pub factors: HashMap<(usize, usize), ExprId>,
    /// Structural nonzeros created by the elimination beyond the pattern.
    pub fill: usize,
    n: usize,
}

/// The pattern of sparse rows: the column of every entry.
pub fn pattern_of(m: &SparseRows) -> Pattern {
    m.iter()
        .map(|row| row.iter().map(|&(j, _)| j).collect())
        .collect()
}

/// A dense matrix of expressions as sparse rows: every entry that is not
/// the structural zero.
pub fn sparse_rows<K: Field>(g: &Graph<K>, m: &[Vec<ExprId>]) -> SparseRows {
    m.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|&(_, &e)| !g.is_zero(e))
                .map(|(j, &e)| (j, e))
                .collect()
        })
        .collect()
}

/// Static-pivot LU of a matrix of expressions in the given elimination
/// order, as graph ops, in Crout form: the factors, for a consumer that
/// solves several right-hand sides against one matrix. The solve of one
/// right-hand side is [`solve_guarded`].
///
/// At step `k` the pivot is `A[order[k]][order[k]]`. An entry is not
/// updated once per pivot; its updates are collected as `(l, u)` pairs and
/// it is finalized when its own step comes as `a - dot(ls, us)`: one dot
/// per factor entry, whatever the fill, which the backends fold in
/// registers. Values flow through the pattern's op sequence; a pivot that
/// is numerically zero at evaluation time gives an infinite multiplier,
/// exactly as an unpivoted elimination would.
pub fn lu_static<K: Field>(g: &mut Graph<K>, m: &SparseRows, order: &[usize]) -> StaticLu {
    let n = m.len();
    assert_eq!(order.len(), n, "one order entry per unknown");
    let mut pos = vec![0usize; n];
    for (k, &i) in order.iter().enumerate() {
        pos[i] = k;
    }
    // Original entries in permuted coordinates, the pending updates of
    // every entry, and row and column occupancy (original plus fill).
    let mut orig: HashMap<(usize, usize), ExprId> = HashMap::default();
    let mut pending: HashMap<(usize, usize), (Vec<ExprId>, Vec<ExprId>)> = HashMap::default();
    let mut in_row: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_col: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, row) in m.iter().enumerate() {
        for &(j, e) in row {
            let (r, c) = (pos[i], pos[j]);
            if orig.insert((r, c), e).is_none() {
                in_row[r].push(c);
                in_col[c].push(r);
            }
        }
    }
    /// An entry's value once every update it receives is known.
    // The value of an entry after the updates pending on it, kept so that a
    // second read (the right-hand side in back-substitution) sees it.
    fn finalize<K: Field>(
        g: &mut Graph<K>,
        orig: &mut HashMap<(usize, usize), ExprId>,
        pending: &mut HashMap<(usize, usize), (Vec<ExprId>, Vec<ExprId>)>,
        at: (usize, usize),
    ) -> ExprId {
        let v = match (orig.get(&at).copied(), pending.remove(&at)) {
            (Some(o), None) => o,
            (Some(o), Some((ls, us))) => {
                let d = g.dot(ls, us);
                g.sub(o, d)
            }
            (None, Some((ls, us))) => {
                let d = g.dot(ls, us);
                g.neg(d)
            }
            (None, None) => unreachable!("an occupied position has a value"),
        };
        orig.insert(at, v);
        v
    }
    let mut factors: HashMap<(usize, usize), ExprId> = HashMap::default();
    let mut fill = 0usize;
    for k in 0..n {
        assert!(
            orig.contains_key(&(k, k)) || pending.contains_key(&(k, k)),
            "a structurally nonzero diagonal in pivot position {k}"
        );
        let pivot = finalize(g, &mut orig, &mut pending, (k, k));
        factors.insert((k, k), pivot);
        let inv = g.recip(pivot);
        // Ascending, so the program is the same for the same pattern.
        let mut row_k: Vec<usize> = in_row[k].iter().copied().filter(|&c| c > k).collect();
        let mut col_k: Vec<usize> = in_col[k].iter().copied().filter(|&r| r > k).collect();
        row_k.sort_unstable();
        col_k.sort_unstable();
        let us: Vec<ExprId> = row_k
            .iter()
            .map(|&j| finalize(g, &mut orig, &mut pending, (k, j)))
            .collect();
        let ls: Vec<ExprId> = col_k
            .iter()
            .map(|&i| {
                let a = finalize(g, &mut orig, &mut pending, (i, k));
                g.mul(a, inv)
            })
            .collect();
        for (&j, &u) in row_k.iter().zip(&us) {
            factors.insert((k, j), u);
        }
        for (&i, &l) in col_k.iter().zip(&ls) {
            factors.insert((i, k), l);
        }
        for (&i, &l) in col_k.iter().zip(&ls) {
            for (&j, &u) in row_k.iter().zip(&us) {
                let entry = pending.entry((i, j)).or_default();
                if entry.0.is_empty() && !orig.contains_key(&(i, j)) {
                    fill += 1;
                    in_row[i].push(j);
                    in_col[j].push(i);
                }
                entry.0.push(l);
                entry.1.push(u);
            }
        }
    }
    StaticLu {
        order: order.to_vec(),
        factors,
        fill,
        n,
    }
}

impl StaticLu {
    /// Solve `A x = b` with the factors: forward through the unit lower
    /// part, backward through the upper, then undo the permutation. `b` is
    /// in original coordinates and so is the result.
    pub fn solve_static<K: Field>(&self, g: &mut Graph<K>, b: &[ExprId]) -> Vec<ExprId> {
        let n = self.n;
        assert_eq!(b.len(), n);
        let mut lower: Vec<Vec<(usize, ExprId)>> = vec![Vec::new(); n];
        let mut upper: Vec<Vec<(usize, ExprId)>> = vec![Vec::new(); n];
        for (&(i, j), &e) in &self.factors {
            if j < i {
                lower[i].push((j, e));
            } else if j > i {
                upper[i].push((j, e));
            }
        }
        // A fixed operand order keeps the program the same across builds.
        for row in lower.iter_mut().chain(upper.iter_mut()) {
            row.sort_by_key(|&(j, _)| j);
        }
        let mut y: Vec<ExprId> = (0..n).map(|k| b[self.order[k]]).collect();
        for i in 0..n {
            let mut acc = y[i];
            for &(j, l) in &lower[i] {
                let t = g.mul(l, y[j]);
                acc = g.sub(acc, t);
            }
            y[i] = acc;
        }
        let mut x = vec![g.zero(); n];
        for i in (0..n).rev() {
            let mut acc = y[i];
            for &(j, u) in &upper[i] {
                let t = g.mul(u, x[j]);
                acc = g.sub(acc, t);
            }
            x[i] = g.div(acc, self.factors[&(i, i)]);
        }
        // Back to original coordinates: unknown `order[k]` is `x[k]`.
        let mut out = vec![g.zero(); n];
        for (k, &i) in self.order.iter().enumerate() {
            out[i] = x[k];
        }
        out
    }
}

/// The structure of a solve, worked out once per pattern: the block
/// triangular form, an elimination order per diagonal block, and the
/// predicted cost.
#[derive(Clone, Debug)]
pub struct Plan {
    pub btf: Btf,
    /// Elimination order of each block, in the block's local indices:
    /// `orders[b][k]` is the column eliminated at step `k`.
    pub orders: Vec<Vec<usize>>,
    /// The row that pivots at each step of each block, in local indices.
    /// The transversal's diagonal by default; [`Plan::repivot`] replaces it
    /// with the rows a numeric elimination with partial pivoting takes.
    pub pivots: Vec<Vec<usize>>,
    /// Predicted fill and flops of the factorization, summed over blocks.
    pub cost: Cost,
}

impl Plan {
    pub fn n(&self) -> usize {
        self.btf.col_perm.len()
    }
    /// Predicted flops per unknown: the number to decide on.
    pub fn flops_per_unknown(&self) -> f64 {
        self.cost.flops as f64 / self.n().max(1) as f64
    }
}

/// Plan a solve on `pattern`: [`btf::block_triangular`], then [`amd::amd`]
/// on each block's symmetrized pattern, then [`predict::cost`]. `None`
/// when the pattern is structurally singular.
pub fn plan(pattern: &Pattern) -> Option<Plan> {
    let btf = btf::block_triangular(pattern)?;
    let n = pattern.len();
    let mut col_pos = vec![0usize; n];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        col_pos[j] = k;
    }
    let mut orders = Vec::with_capacity(btf.n_blocks());
    let mut cost = Cost::default();
    for b in 0..btf.n_blocks() {
        let range = btf.block(b);
        let lo = range.start;
        let size = range.len();
        // The block's own pattern in local indices, symmetrized.
        let mut local: Pattern = vec![Vec::new(); size];
        for k in range.clone() {
            let i = btf.row_perm[k];
            for &j in &pattern[i] {
                let c = col_pos[j];
                if range.contains(&c) {
                    local[k - lo].push(c - lo);
                }
            }
        }
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); size];
        for (r, row) in local.iter().enumerate() {
            for &c in row {
                if c != r {
                    adj[r].push(c);
                    adj[c].push(r);
                }
            }
        }
        for a in adj.iter_mut() {
            a.sort_unstable();
            a.dedup();
        }
        let order = if size <= 2 {
            (0..size).collect()
        } else {
            amd::amd(&adj)
        };
        let c = predict::cost(&local, &order);
        cost.fill += c.fill;
        cost.flops += c.flops;
        orders.push(order);
    }
    let pivots = orders.clone();
    Some(Plan {
        btf,
        orders,
        pivots,
        cost,
    })
}

impl Plan {
    /// The plan with the pivot rows a numeric elimination with partial
    /// pivoting takes on the values `value(row, col)` (original indices):
    /// what a consumer builds when the guard of [`solve_planned`] reports
    /// that the current pivot rows no longer hold. The structure and the
    /// elimination order stay; only which row pivots at each step changes.
    pub fn repivot(&self, pattern: &Pattern, value: impl Fn(usize, usize) -> f64) -> Plan {
        let btf = &self.btf;
        let n = pattern.len();
        let mut col_pos = vec![0usize; n];
        for (k, &j) in btf.col_perm.iter().enumerate() {
            col_pos[j] = k;
        }
        let mut pivots = Vec::with_capacity(btf.n_blocks());
        for b in 0..btf.n_blocks() {
            let range = btf.block(b);
            let lo = range.start;
            let size = range.len();
            // The block numerically, in local row and column indices.
            let mut a: HashMap<(usize, usize), f64> = HashMap::default();
            let mut in_col: Vec<Vec<usize>> = vec![Vec::new(); size];
            for k in range.clone() {
                let i = btf.row_perm[k];
                for &j in &pattern[i] {
                    let c = col_pos[j];
                    if range.contains(&c) {
                        a.insert((k - lo, c - lo), value(i, j));
                        in_col[c - lo].push(k - lo);
                    }
                }
            }
            // Threshold pivoting in the elimination order: at step k, among
            // the rows not yet taken whose magnitude in column `order[k]`
            // is within [`REPIVOT_CHOICE`] of the largest, the sparsest row,
            // lowest index first: the pivot quality of a partial pivot to a
            // factor of ten, the fill of a sparser choice, and a choice that
            // depends on the values and the pattern alone.
            let order = &self.orders[b];
            let mut taken = vec![false; size];
            let mut row_len = vec![0usize; size];
            for &(r, _) in a.keys() {
                row_len[r] += 1;
            }
            let mut rows = Vec::with_capacity(size);
            for &c in order {
                let mut open: Vec<usize> =
                    in_col[c].iter().copied().filter(|&r| !taken[r]).collect();
                open.sort_unstable();
                open.dedup();
                let mag = |r: usize| a.get(&(r, c)).copied().unwrap_or(0.0).abs();
                let largest = open.iter().map(|&r| mag(r)).fold(0.0f64, f64::max);
                let r = open
                    .iter()
                    .copied()
                    .filter(|&r| mag(r) >= REPIVOT_CHOICE * largest)
                    .min_by_key(|&r| (row_len[r], r))
                    .expect("a structurally nonsingular block");
                taken[r] = true;
                rows.push(r);
                // Eliminate column c from the rows still open, tracking fill.
                let piv = a[&(r, c)];
                let mut row_r: Vec<(usize, f64)> = a
                    .iter()
                    .filter(|(&(rr, cc), _)| rr == r && !taken_col(cc, order, c))
                    .map(|(&(_, cc), &v)| (cc, v))
                    .collect();
                row_r.sort_unstable_by_key(|&(cc, _)| cc);
                let col_c: Vec<usize> = open.iter().copied().filter(|&i| i != r).collect();
                for i in col_c {
                    let l = a[&(i, c)] / piv;
                    for &(cc, v) in &row_r {
                        let e = a.entry((i, cc)).or_insert_with(|| {
                            in_col[cc].push(i);
                            row_len[i] += 1;
                            0.0
                        });
                        *e -= l * v;
                    }
                }
            }
            pivots.push(rows);
        }
        Plan {
            btf: self.btf.clone(),
            orders: self.orders.clone(),
            pivots,
            cost: self.cost,
        }
    }
}

/// Whether column `cc` is eliminated at or before the step of `c`.
fn taken_col(cc: usize, order: &[usize], c: usize) -> bool {
    let pos = |x: usize| order.iter().position(|&o| o == x).unwrap_or(usize::MAX);
    pos(cc) <= pos(c)
}

/// Solve `A x = b` along a [`Plan`]: one static LU per diagonal block,
/// the blocks from the last to the first with the entries above the
/// diagonal blocks moved to the right-hand side as they are known. `b`
/// and the result are in original coordinates. Returns the unknowns and
/// the factorizations' actual fill.
pub fn solve_planned<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &[Vec<(usize, N)>],
    plan: &Plan,
    b: &[N],
) -> Solved<N> {
    let n = m.len();
    assert_eq!(b.len(), n);
    let btf = &plan.btf;
    let mut col_pos = vec![0usize; n];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        col_pos[j] = k;
    }
    // Permuted rows: entries as (permuted column, expr).
    let rows: Vec<Vec<(usize, N)>> = btf
        .row_perm
        .iter()
        .map(|&i| {
            let mut r: Vec<(usize, N)> = m[i].iter().map(|&(j, e)| (col_pos[j], e)).collect();
            r.sort_by_key(|&(c, _)| c);
            r
        })
        .collect();
    let mut x = vec![N::zero(g); n]; // permuted coordinates
    let mut fill = 0usize;
    let mut guards: Vec<ExprId> = Vec::new();
    for blk in (0..btf.n_blocks()).rev() {
        let range = btf.block(blk);
        let (lo, hi) = (range.start, range.end);
        let mut local: Vec<Vec<(usize, N)>> = Vec::with_capacity(hi - lo);
        let mut rhs: Vec<N> = Vec::with_capacity(hi - lo);
        for k in lo..hi {
            let mut row = Vec::new();
            let (mut ls, mut xs) = (Vec::new(), Vec::new());
            for &(c, e) in &rows[k] {
                if c >= hi {
                    ls.push(e);
                    xs.push(x[c]);
                } else if c >= lo {
                    row.push((c - lo, e));
                }
            }
            local.push(row);
            let bk = b[btf.row_perm[k]];
            rhs.push(if ls.is_empty() {
                bk
            } else {
                let d = N::dot(g, ls, xs);
                N::sub(g, bk, d)
            });
        }
        let (sol, gs, f) = solve_guarded(g, &local, &plan.orders[blk], &plan.pivots[blk], &rhs);
        fill += f;
        guards.extend(gs);
        for (k, v) in (lo..hi).zip(sol) {
            x[k] = v;
        }
    }
    let mut out = vec![N::zero(g); n];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        out[j] = x[k];
    }
    let pivots_ok = match guards.len() {
        0 => g.one(),
        1 => guards[0],
        _ => g.reduce(crate::node::ReduceOp::Min, guards),
    };
    Solved {
        x: out,
        pivots_ok,
        fill,
    }
}

/// The result of a planned solve.
pub struct Solved<N = ExprId> {
    /// The unknowns, one expression each, in original coordinates.
    pub x: Vec<N>,
    /// `1` while every pivot row still dominates its column by the
    /// threshold, `0` once a step would pivot elsewhere: the guard a
    /// consumer checks per evaluation, rebuilding with
    /// [`Plan::repivot`] when it fails.
    pub pivots_ok: ExprId,
    /// The factorizations' fill.
    pub fill: usize,
}

/// Below this ratio of a pivot's magnitude to the largest in its column,
/// the guard reports that the pivot rows should change (KLU's default).
pub const PIVOT_TOLERANCE: f64 = 1e-3;

/// [`Plan::repivot`] takes the sparsest row among those within this ratio
/// of the largest magnitude in the column.
pub const REPIVOT_CHOICE: f64 = 0.1;

/// The solve of one block: Gaussian elimination in the given column order
/// with the given pivot rows, over the matrix augmented by the right-hand
/// side, in Crout form (one dot per entry), plus a guard per step with
/// more than one structural candidate: whether the pivot row's magnitude
/// is at least [`PIVOT_TOLERANCE`] times the largest in its column. The
/// program is the static elimination's; the guard costs the column's
/// magnitudes, which the elimination computes anyway.
pub fn solve_guarded<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &[Vec<(usize, N)>],
    order: &[usize],
    pivots: &[usize],
    b: &[N],
) -> (Vec<N>, Vec<ExprId>, usize) {
    let n = m.len();
    assert_eq!(order.len(), n, "one order entry per unknown");
    assert_eq!(pivots.len(), n, "one pivot row per step");
    assert_eq!(b.len(), n);
    let mut cpos = vec![0usize; n];
    let mut rpos = vec![0usize; n];
    for k in 0..n {
        cpos[order[k]] = k;
        rpos[pivots[k]] = k;
    }
    let mut orig: HashMap<(usize, usize), N> = HashMap::default();
    let mut pending: HashMap<(usize, usize), (Vec<N>, Vec<N>)> = HashMap::default();
    let mut in_row: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_col: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (i, row) in m.iter().enumerate() {
        for &(j, e) in row {
            let (r, c) = (rpos[i], cpos[j]);
            if orig.insert((r, c), e).is_none() {
                in_row[r].push(c);
                in_col[c].push(r);
            }
        }
        orig.insert((rpos[i], n), b[i]);
        in_row[rpos[i]].push(n);
    }
    // The value of an entry after the updates pending on it, kept so that a
    // second read (the right-hand side in back-substitution) sees it.
    fn finalize<K: Field, N: Num>(
        g: &mut Graph<K>,
        orig: &mut HashMap<(usize, usize), N>,
        pending: &mut HashMap<(usize, usize), (Vec<N>, Vec<N>)>,
        at: (usize, usize),
    ) -> N {
        let v = match (orig.get(&at).copied(), pending.remove(&at)) {
            (Some(o), None) => o,
            (Some(o), Some((ls, us))) => {
                let d = N::dot(g, ls, us);
                N::sub(g, o, d)
            }
            (None, Some((ls, us))) => {
                let d = N::dot(g, ls, us);
                N::neg(g, d)
            }
            (None, None) => unreachable!("an occupied position has a value"),
        };
        orig.insert(at, v);
        v
    }
    let mut fill = 0usize;
    let mut upper: Vec<Vec<(usize, N)>> = vec![Vec::new(); n];
    let mut diag: Vec<N> = vec![N::zero(g); n];
    let mut guards: Vec<ExprId> = Vec::new();
    for k in 0..n {
        assert!(
            orig.contains_key(&(k, k)) || pending.contains_key(&(k, k)),
            "a structurally nonzero entry in pivot position {k}"
        );
        let pivot = finalize(g, &mut orig, &mut pending, (k, k));
        diag[k] = pivot;
        let inv = N::recip(g, pivot);
        let mut row_k: Vec<usize> = in_row[k].iter().copied().filter(|&c| c > k).collect();
        let mut col_k: Vec<usize> = in_col[k].iter().copied().filter(|&r| r > k).collect();
        row_k.sort_unstable();
        row_k.dedup();
        col_k.sort_unstable();
        col_k.dedup();
        let us: Vec<N> = row_k
            .iter()
            .map(|&j| finalize(g, &mut orig, &mut pending, (k, j)))
            .collect();
        for (&j, &u) in row_k.iter().zip(&us) {
            if j < n {
                upper[k].push((j, u));
            }
        }
        let below: Vec<N> = col_k
            .iter()
            .map(|&i| finalize(g, &mut orig, &mut pending, (i, k)))
            .collect();
        if !below.is_empty() {
            // The guard: the pivot dominates its column by the tolerance.
            let mut mags: Vec<ExprId> = below.iter().map(|&v| N::size(g, v)).collect();
            let largest = if mags.len() == 1 {
                mags.pop().unwrap()
            } else {
                g.reduce(crate::node::ReduceOp::Max, mags)
            };
            guards.push(N::guard(g, pivot, largest));
        }
        let ls: Vec<N> = below.iter().map(|&a| N::mul(g, a, inv)).collect();
        for (&i, &l) in col_k.iter().zip(&ls) {
            for (&j, &u) in row_k.iter().zip(&us) {
                let entry = pending.entry((i, j)).or_default();
                if entry.0.is_empty() && !orig.contains_key(&(i, j)) {
                    fill += 1;
                    in_row[i].push(j);
                    in_col[j].push(i);
                }
                entry.0.push(l);
                entry.1.push(u);
            }
        }
    }
    let mut x = vec![N::zero(g); n];
    for i in (0..n).rev() {
        let mut acc = finalize(g, &mut orig, &mut pending, (i, n));
        for &(j, u) in &upper[i] {
            let t = N::mul(g, u, x[j]);
            acc = N::sub(g, acc, t);
        }
        x[i] = N::div(g, acc, diag[i]);
    }
    let mut out = vec![N::zero(g); n];
    for (k, &j) in order.iter().enumerate() {
        out[j] = x[k];
    }
    (out, guards, fill)
}

/// One Newton step as expressions: `x - J^-1 F`, with `J` the Jacobian of
/// `f` with respect to `x`, solved along its [`plan`].
///
/// The result is an ordinary expression over the graph: it differentiates,
/// specializes, compiles and batches like the model it came from, and a
/// consumer that wants damping or a line search composes it from the same
/// pieces ([`lu_static`], [`StaticLu::solve_static`]).
///
/// `None` when the Jacobian is structurally singular: no pivot order
/// exists, whatever the values.
pub fn newton_step<K: Field>(
    g: &mut Graph<K>,
    f: &[ExprId],
    x: &[crate::node::SymbolId],
) -> Option<NewtonStep> {
    let jac = crate::autodiff::sparse_jacobian(g, f, x);
    let pattern = pattern_of(&jac);
    let plan = plan(&pattern)?;
    Some(newton_step_planned(g, f, x, &jac, &plan))
}

/// [`newton_step`] over a given Jacobian and plan: what a consumer calls
/// again with a [`Plan::repivot`]ed plan when the step's guard failed.
pub fn newton_step_planned<K: Field>(
    g: &mut Graph<K>,
    f: &[ExprId],
    x: &[crate::node::SymbolId],
    jac: &SparseRows,
    plan: &Plan,
) -> NewtonStep {
    let solved = solve_planned(g, jac, plan, f);
    let out = x
        .iter()
        .zip(&solved.x)
        .map(|(&s, &d)| {
            let xe = g.symbol_expr(s);
            g.sub(xe, d)
        })
        .collect();
    NewtonStep {
        x: out,
        pivots_ok: solved.pivots_ok,
        fill: solved.fill,
    }
}

/// A Newton step as expressions.
pub struct NewtonStep {
    /// The updated unknowns, one expression each.
    pub x: Vec<ExprId>,
    /// The solve's pivot guard (see [`Solved::pivots_ok`]).
    pub pivots_ok: ExprId,
    /// The factorizations' fill.
    pub fill: usize,
}
