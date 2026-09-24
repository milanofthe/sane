//! Fixed-pattern sparse LU for the engine's real linear solves: rsdag's
//! graph solve first, the rslab library where it declines.
//!
//! The solver's contract: the sparsity pattern of a Newton / stage /
//! homotopy matrix is fixed across a solve, so the symbolic work is done
//! **once** and every iteration runs only a numeric factorization over
//! refreshed values.
//!
//! The primary backend is the graph solve ([`GraphSystem`]): the static LU
//! of the pattern, along rsdag's block triangular form and fill-reducing
//! order, as one program over the entry values, its factorization the
//! parameter-pure prolog and its substitution the main pass, native code
//! where the build has it. The pivot rows are chosen on the first values
//! and guarded; a guard failure repivots the shared program on the values
//! (a few times per system, then the library takes the value sets the
//! guard rejects). A system beyond the program's range by rsdag's cost
//! predictor ([`GRAPH_LU_MAX_FLOPS_PER_UNKNOWN`], [`GRAPH_LU_MAX_FLOPS`],
//! [`GRAPH_LU_MAX_NNZ_PER_UNKNOWN`]) is the library's from the start, and
//! `Config::graph_solve` off selects the library everywhere.
//!
//! Three rslab backends serve three structural regimes there, chosen
//! automatically from the pattern's BTF analysis and a per-factorization
//! value-symmetry test:
//!
//! * **KLU** (BTF + per-block AMD + Gilbert-Peierls) for circuit-shaped
//!   patterns -- many BTF blocks / modest irreducible blocks. Its numeric-only
//!   `refactor` (frozen pattern + pivot sequence, no DFS, no pivot search)
//!   makes Newton iterations after the first factorization very cheap, and it
//!   provides the transpose solve the adjoint paths use.
//! * **Multifrontal LDLT** (`rslab::LdltSolver`, Bunch-Kaufman) when the
//!   system is large ([`LDLT_BLOCK_MIN`]) and the assembled values are
//!   *symmetric* -- MNA of R/L/C networks with independent sources, i.e. the
//!   power-grid regime. Half the fill/flops of any LU and rslab's most
//!   optimized kernel; checked per factorization (O(nnz) bitwise), so a
//!   nonlinear iterate that breaks symmetry transparently falls back to LU.
//! * **Multifrontal LU** (`rslab::LuSolver`, blocked SIMD kernels, scoped
//!   worker pool) for large *unsymmetric* single-block patterns crossing
//!   [`MF_BLOCK_MIN`], where scalar Gilbert-Peierls loses to blocked fronts
//!   (measured crossover ~1e4 unknowns on RC grids).
//!
//! Values are supplied in the caller's *entry order* (the tape's Jacobian
//! nonzeros followed by augmentation entries such as the gmin diagonal), with
//! duplicate `(row, col)` positions summed into one CSC slot -- the same
//! semantics faer's argsort provided. Row equilibration (the
//! `row_equilibration` solver trick) is the backend's built-in scaling: KLU's
//! row-max scaling when the trick is on, and the multifrontal path's own
//! equilibration (always on there); it is folded into factorization and
//! solves, so callers never scale right-hand sides.

use std::sync::{Arc, Mutex, OnceLock};

use rslab::{
    CscMatrix, FactorMethod, GeneralCsc, KluSettings, KluSolver, KluSymbolic, LdltSolver,
    LdltSymbolic, LuSolver, LuSymbolic, SolverSettings,
};

/// rslab's log records routed into SANE's logger. rslab's `Info` lines (one
/// per analysis and factorization: ordering picked, fill, threads, wall time)
/// are solver internals from SANE's point of view and land at `Debug`; its
/// warnings (a setting its path does not read) and errors keep their level.
/// The message carries an `rslab:` prefix and SANE's own stage path.
struct RslabLogBridge;

impl rslab::LogSink for RslabLogBridge {
    fn emit(&self, level: rslab::LogLevel, msg: &str) {
        let line = format!("rslab: {msg}");
        match level {
            rslab::LogLevel::Debug | rslab::LogLevel::Info => sane_core::log::debug(&line),
            rslab::LogLevel::Warning => sane_core::log::warning(&line),
            rslab::LogLevel::Error => sane_core::log::error(&line),
            rslab::LogLevel::Off => {}
        }
    }
}

/// SANE's threshold mirrored into rslab, so rslab formats only what SANE would
/// show: SANE `Debug` shows rslab's `Info` and `Debug`, anything above shows
/// only rslab's warnings and errors.
fn mirror_level(level: sane_core::log::LogLevel) {
    use sane_core::log::LogLevel as L;
    rslab::logging::set_level(match level {
        L::Debug => rslab::LogLevel::Debug,
        L::Info | L::Warning => rslab::LogLevel::Warning,
        L::Error => rslab::LogLevel::Error,
        L::Disabled => rslab::LogLevel::Off,
    });
}

/// Install the bridge once per process (called when the first DAE is compiled).
pub(crate) fn install_log_bridge() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        rslab::logging::set_sink(Box::new(RslabLogBridge));
        sane_core::log::on_level_change(mirror_level);
    });
}

/// Largest-BTF-block threshold above which the blocked multifrontal LU
/// replaces KLU for this pattern. Calibrated on RC grids on Apple M3: at ~8e3
/// the two are on par, at ~1.7e4 the multifrontal is ~1.4x faster, at 4e4
/// ~2.2x (KLU's numeric-only refactor narrows but does not close the gap).
const MF_BLOCK_MIN: usize = 10_000;

/// Dimension threshold above which a *symmetric* value set is factored by the
/// multifrontal Bunch-Kaufman LDLT instead (half the flops/fill of any LU, and
/// rslab's most optimized kernel). MNA systems of R/L/C networks with
/// independent sources are symmetric -- exactly the power-grid regime.
/// Calibrated on RC-grid Laplacians (Apple M3): at 1.6e3 KLU still wins
/// (1.5 vs 2.4 ms), at ~4e3 they tie, at 8.1e3 LDLT-MF wins 12.7 vs 16.6 ms,
/// at 4e4 it wins 71 vs 97 ms (unsymmetric LU) / 195 ms (KLU).
const LDLT_BLOCK_MIN: usize = 8_000;

/// The graph solve takes a pattern whose static LU costs at most this many
/// multiply-adds per unknown (rsdag's predictor, fill included): below it
/// the factorization as straight-line code beats a sparse LU library's
/// factor plus solve by an order of magnitude (circuit-like patterns land
/// at a few tens); above it the fill makes the program large and a
/// library's blocked kernels win.
pub(crate) const GRAPH_LU_MAX_FLOPS_PER_UNKNOWN: f64 = 400.0;
/// And at most this many multiply-adds in total, a bound on the program.
pub(crate) const GRAPH_LU_MAX_FLOPS: usize = 4_000_000;
/// A system denser than this many entries per unknown is declined before
/// its plan is even predicted (the prediction itself costs on a dense
/// pattern): a circuit pattern has a handful, a block-dense one such as
/// harmonic balance's Toeplitz blocks has hundreds.
const GRAPH_LU_MAX_NNZ_PER_UNKNOWN: usize = 16;
/// Factorizations between two repivots of a graph-solve program. The guard
/// says the pivot rows are no longer the ones partial pivoting would take;
/// a Newton path that alternates between regions must not rebuild the
/// program on every pass, so between repivots such a value set goes to a
/// library backend (its pivoting factorization, as a library does after a
/// failed numeric-only refactorization).
const GRAPH_LU_REPIVOT_GAP: u32 = 32;
/// Repivots a pattern's program takes over its lifetime: a few adapt the
/// pivot rows to the values a circuit actually has (the transversal's
/// diagonal is only a structural choice), after which the order is fixed,
/// as a library's is after its first factorization, and a value set the
/// order breaks down on goes to a library backend.
const GRAPH_LU_MAX_REPIVOTS: u32 = 4;

/// The LDLT factor settings: multifrontal schedule (measured fastest for the
/// mesh-class fronts this path is selected for), auto worker prediction.
fn ldlt_settings() -> SolverSettings {
    SolverSettings::default().with_method(FactorMethod::Multifrontal)
}

/// Symmetry side-structure over a CSC pattern: the transposed-slot pairing for
/// the O(nnz) per-factorization value-symmetry test, and the lower-triangle
/// skeleton + its symbolic analysis for the LDLT factorization.
struct LdltCand {
    /// Full-CSC slot -> slot of the transposed position (identity on the
    /// diagonal). Present only for structurally symmetric patterns.
    tpair: Vec<usize>,
    /// Lower-triangle entry m -> full-CSC slot it reads its value from.
    lower_from: Vec<usize>,
    lcol_ptr: Vec<usize>,
    lrow_idx: Vec<usize>,
    lsym: LdltSymbolic,
}

impl LdltCand {
    /// Build from a CSC pattern (values ignored). `None` if the pattern is
    /// structurally unsymmetric or the LDLT analysis fails.
    fn build(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Option<Self> {
        let tpair = transpose_pairs(n, col_ptr, row_idx)?;
        // Lower triangle (r >= c) in CSC order.
        let mut lcol_ptr = vec![0usize; n + 1];
        let mut lrow_idx: Vec<usize> = Vec::new();
        let mut lower_from: Vec<usize> = Vec::new();
        for c in 0..n {
            for k in col_ptr[c]..col_ptr[c + 1] {
                if row_idx[k] >= c {
                    lrow_idx.push(row_idx[k]);
                    lower_from.push(k);
                    lcol_ptr[c + 1] += 1;
                }
            }
        }
        for j in 0..n {
            lcol_ptr[j + 1] += lcol_ptr[j];
        }
        let skeleton = CscMatrix::<f64> {
            n,
            col_ptr: lcol_ptr.clone(),
            row_idx: lrow_idx.clone(),
            values: vec![1.0; lrow_idx.len()],
        };
        let lsym = LdltSymbolic::analyze(&skeleton).ok()?;
        Some(LdltCand {
            tpair,
            lower_from,
            lcol_ptr,
            lrow_idx,
            lsym,
        })
    }

    /// Bitwise value-symmetry test over the full CSC values. Conservative: a
    /// false negative only skips the LDLT fast path, never affects results.
    fn values_symmetric(&self, vals: &[f64]) -> bool {
        vals.iter()
            .enumerate()
            .all(|(k, &v)| v.to_bits() == vals[self.tpair[k]].to_bits())
    }

    /// Numeric LDLT factorization of the (symmetric) full-CSC values. `None`
    /// on a numerically rank-deficient matrix (caller falls through to LU).
    fn factor(&self, n: usize, vals: &[f64]) -> Option<LdltSolver<f64>> {
        let a = CscMatrix::<f64> {
            n,
            col_ptr: self.lcol_ptr.clone(),
            row_idx: self.lrow_idx.clone(),
            values: self.lower_from.iter().map(|&k| vals[k]).collect(),
        };
        self.lsym.factor(&a, &ldlt_settings()).ok()
    }
}

/// For each CSC slot `(r, c)`, the slot of `(c, r)` (per-column rows are
/// sorted, so a binary search per slot). `None` when any partner is missing --
/// a structurally unsymmetric pattern.
fn transpose_pairs(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Option<Vec<usize>> {
    // Column of each slot, for the reverse lookup.
    let mut col_of = vec![0usize; row_idx.len()];
    for c in 0..n {
        for k in col_ptr[c]..col_ptr[c + 1] {
            col_of[k] = c;
        }
    }
    let mut tpair = vec![0usize; row_idx.len()];
    for k in 0..row_idx.len() {
        let (r, c) = (row_idx[k], col_of[k]);
        let seg = &row_idx[col_ptr[r]..col_ptr[r + 1]];
        let off = seg.binary_search(&c).ok()?;
        tpair[k] = col_ptr[r] + off;
    }
    Some(tpair)
}

/// The symbolic analysis of a fixed pattern, in the backend the structure
/// selected.
enum Sym {
    Klu(KluSymbolic),
    // Boxed: the multifrontal analysis is much larger than the KLU one, and
    // patterns live for the whole compiled model.
    Mf(Box<LuSymbolic>),
}

/// rsdag's guarded static LU of a pattern as one program
/// ([`rsdag::symbolic::solve::LuProgram`]): a numeric refactorization is its
/// prolog, a solve its main phase, natively where the JIT is on.
struct GraphLu {
    prog: rsdag::symbolic::solve::LuProgram,
    #[cfg(feature = "jit")]
    native: Option<rsdag_jit::NativeTape>,
}

impl GraphLu {
    /// The program for the entries along `plan`, or `None` when the plan's
    /// cost is beyond the graph solve's range.
    fn build(n: usize, entries: &[(usize, usize)], plan: rsdag::symbolic::Plan) -> Option<Self> {
        if plan.flops_per_unknown() > GRAPH_LU_MAX_FLOPS_PER_UNKNOWN
            || plan.cost.flops > GRAPH_LU_MAX_FLOPS
        {
            return None;
        }
        Some(Self::compile(rsdag::symbolic::solve::LuProgram::build(
            n,
            entries.to_vec(),
            plan,
            Some(rsdag::symbolic::solve::Panels::default()),
        )))
    }

    fn compile(prog: rsdag::symbolic::solve::LuProgram) -> Self {
        let t0 = sane_core::time::Instant::now();
        #[cfg(feature = "jit")]
        let native = if crate::jit_enabled() {
            rsdag_jit::NativeTape::compile(prog.tape()).ok()
        } else {
            None
        };
        sane_core::log::debug(&format!(
            "graph solve{}: n={} nnz={} fill={} flops/unknown={:.0} program={} ops, native in {:.1} ms",
            match prog.supernodal() {
                Some((panels, widest)) => format!(" (supernodal, {panels} panels, widest {widest})"),
                None => String::new(),
            },
            prog.n(),
            prog.entries().len(),
            prog.fill(),
            prog.plan().flops_per_unknown(),
            prog.tape().n_ops(),
            t0.elapsed().as_secs_f64() * 1e3
        ));
        GraphLu {
            prog,
            #[cfg(feature = "jit")]
            native,
        }
    }

    /// The factorization: the prolog over the entry values.
    fn factor(&self, inputs: &[f64], work: &mut Vec<f64>) {
        #[cfg(feature = "jit")]
        if let Some(nt) = &self.native {
            nt.eval_prolog(inputs, work);
            return;
        }
        self.prog.tape().eval_prolog(inputs, work);
    }

    /// The substitution over a prepared prolog.
    fn substitute(&self, inputs: &[f64], work: &mut [f64], out: &mut Vec<f64>) {
        #[cfg(feature = "jit")]
        if let Some(nt) = &self.native {
            nt.eval_main(inputs, work, out);
            return;
        }
        self.prog.tape().eval_main(inputs, work, out);
    }
}

/// A sparse system as the graph solve sees it: `n` unknowns, the entries
/// `(row, col)` of the value slots in slot order (distinct positions; a
/// caller sums duplicates into a slot), the program shared by every
/// factorizer of the system, and its repivot budget.
pub struct GraphSystem {
    n: usize,
    entries: Vec<(usize, usize)>,
    pattern: rsdag::symbolic::Pattern,
    /// `Some(None)` once the system proved beyond the graph solve's range.
    program: Mutex<Option<Option<Arc<GraphLu>>>>,
    repivots: std::sync::atomic::AtomicU32,
}

impl GraphSystem {
    pub fn new(n: usize, entries: Vec<(usize, usize)>) -> Self {
        let mut pattern: rsdag::symbolic::Pattern = vec![Vec::new(); n];
        for &(i, j) in &entries {
            pattern[i].push(j);
        }
        GraphSystem {
            n,
            entries,
            pattern,
            program: Mutex::new(None),
            repivots: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// The transposed system over the same value slots: a program for
    /// `A^T x = b` fed with `A`'s values.
    pub fn transposed(&self) -> Self {
        let sys = Self::new(self.n, self.entries.iter().map(|&(i, j)| (j, i)).collect());
        sys
    }

    /// The program, built on first demand with the transversal's diagonal
    /// as the pivot rows (measured against choosing them on the first
    /// values: a Newton's starting point has the small conductances, and
    /// the diagonal of an MNA matrix is the better first guess); `None`
    /// when the graph solve is off or the system is beyond its range.
    fn program(&self) -> Option<Arc<GraphLu>> {
        if !sane_core::config().graph_solve {
            return None;
        }
        let mut slot = self.program.lock().unwrap();
        if let Some(g) = slot.as_ref() {
            return g.clone();
        }
        let dense = self.entries.len() > GRAPH_LU_MAX_NNZ_PER_UNKNOWN * self.n.max(1);
        let planned = if dense {
            None
        } else {
            rsdag::symbolic::plan(&self.pattern)
        };
        let built = planned
            .as_ref()
            .and_then(|plan| GraphLu::build(self.n, &self.entries, plan.clone()))
            .map(Arc::new);
        if built.is_none() {
            sane_core::log::debug(&format!(
                "graph solve: system n={} nnz={} beyond its range ({}), sparse LU library",
                self.n,
                self.entries.len(),
                match &planned {
                    Some(plan) => format!(
                        "{:.0} flops per unknown, {} in total",
                        plan.flops_per_unknown(),
                        plan.cost.flops
                    ),
                    None => "denser than a circuit pattern".to_string(),
                }
            ));
        }
        *slot = Some(built.clone());
        built
    }

    /// Take one of the system's repivots; `false` once they are spent.
    fn take_repivot(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.repivots
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < GRAPH_LU_MAX_REPIVOTS).then_some(n + 1)
            })
            .is_ok()
    }

    pub fn factorizer(self: &Arc<Self>) -> GraphFactorizer {
        GraphFactorizer {
            sys: self.clone(),
            lu: None,
            scratch: std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())),
            row_scale: Vec::new(),
            since_repivot: GRAPH_LU_REPIVOT_GAP,
        }
    }
}

/// One factorizer's state over a [`GraphSystem`]: the program it last
/// factored with, the prolog buffer and the input image (entries, then the
/// right-hand side).
pub struct GraphFactorizer {
    sys: Arc<GraphSystem>,
    lu: Option<Arc<GraphLu>>,
    /// The input image (entries, then the right-hand side), the prolog
    /// buffer and the outputs, behind a cell so a solve takes `&self`.
    scratch: std::cell::RefCell<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    /// Row scaling of the factored values (`1 / max |row|`, empty when the
    /// caller asked for none): applied to the entries before the
    /// factorization and to a right-hand side before the substitution.
    row_scale: Vec<f64>,
    /// Factorizations since the last repivot (see [`GRAPH_LU_REPIVOT_GAP`]).
    since_repivot: u32,
}

impl GraphFactorizer {
    /// Factor `values` (slot order): `Some(true)` factored, `Some(false)`
    /// this value set is a library backend's (a breakdown, or pivot rows the
    /// guard rejects between repivots), `None` when the graph solve does not
    /// serve this system. A guard failure or breakdown repivots the shared
    /// program on these values, within the system's budget and at most once
    /// per [`GRAPH_LU_REPIVOT_GAP`] factorizations.
    pub fn factor(&mut self, values: &[f64], row_scaling: bool) -> Option<bool> {
        let lu = match &self.lu {
            Some(lu) => lu.clone(),
            None => self.sys.program()?,
        };
        let n = self.sys.n;
        let nnz = values.len();
        debug_assert_eq!(nnz, self.sys.entries.len());
        let mut scratch = self.scratch.borrow_mut();
        let (inputs, work, _) = &mut *scratch;
        inputs.clear();
        inputs.resize(lu.prog.input_len(), 0.0);
        self.lu = Some(lu);
        self.row_scale.clear();
        if row_scaling {
            self.row_scale.resize(n, 0.0);
            for (k, &(i, _)) in self.sys.entries.iter().enumerate() {
                self.row_scale[i] = self.row_scale[i].max(values[k].abs());
            }
            for s in self.row_scale.iter_mut() {
                *s = if *s > 0.0 && s.is_finite() {
                    1.0 / *s
                } else {
                    1.0
                };
            }
        }
        self.since_repivot = self.since_repivot.saturating_add(1);
        for attempt in 0..2 {
            let lu = self.lu.as_ref().unwrap();
            // The entries at the program's positions (a repivoted program
            // has its own).
            let scale = row_scaling.then_some(&self.row_scale[..]);
            lu.prog.write_values(values, scale, inputs);
            lu.factor(inputs, work);
            // The guard and the factors' finiteness, from the prolog state.
            if lu.prog.factored(inputs, work) {
                return Some(true);
            }
            if attempt == 0 && self.since_repivot >= GRAPH_LU_REPIVOT_GAP && self.sys.take_repivot()
            {
                let mags: Vec<f64> = values.iter().map(|v| v.abs()).collect();
                let re = Arc::new(GraphLu::compile(lu.prog.repivot(&mags)));
                *self.sys.program.lock().unwrap() = Some(Some(re.clone()));
                self.lu = Some(re);
                self.since_repivot = 0;
                sane_core::log::debug(
                    "graph solve: pivot guard failed or factorization broke down, repivoted on the values",
                );
            } else {
                break;
            }
        }
        Some(false)
    }

    /// Solve against the last successful [`factor`](Self::factor).
    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let mut scratch = self.scratch.borrow_mut();
        let (inputs, work, out) = &mut *scratch;
        let lu = self.lu.as_ref().expect("factored");
        let scale = (!self.row_scale.is_empty()).then_some(&self.row_scale[..]);
        lu.prog.write_rhs(rhs, scale, inputs);
        lu.substitute(inputs, work, out);
        lu.prog.solution(out).to_vec()
    }
}

/// Distinct `(row, col)` positions of `(rows, cols)` in CSC order, and each
/// input entry's slot.
fn dedup_entries(
    n: usize,
    rows: &[usize],
    cols: &[usize],
) -> Option<(Vec<(usize, usize)>, Vec<usize>)> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_unstable_by_key(|&k| (cols[k], rows[k]));
    let mut entries: Vec<(usize, usize)> = Vec::with_capacity(rows.len());
    let mut slot = vec![0usize; rows.len()];
    for &k in &order {
        let (r, c) = (rows[k], cols[k]);
        if r >= n || c >= n {
            return None;
        }
        if entries.last() != Some(&(r, c)) {
            entries.push((r, c));
        }
        slot[k] = entries.len() - 1;
    }
    Some((entries, slot))
}

/// The graph systems of every triplet pattern factored one-shot so far
/// (forward and transposed), keyed by the pattern: an adjoint or a
/// sensitivity factors the same pattern at many points, and the programs
/// are the expensive part.
fn triplet_systems(n: usize, entries: &[(usize, usize)]) -> (Arc<GraphSystem>, Arc<GraphSystem>) {
    use std::hash::{Hash, Hasher};
    static CACHE: OnceLock<
        Mutex<rustc_hash::FxHashMap<u64, (Arc<GraphSystem>, Arc<GraphSystem>)>>,
    > = OnceLock::new();
    let mut h = rustc_hash::FxHasher::default();
    n.hash(&mut h);
    entries.hash(&mut h);
    let key = h.finish();
    let cache = CACHE.get_or_init(Default::default);
    let mut map = cache.lock().unwrap();
    map.entry(key)
        .or_insert_with(|| {
            let fwd = Arc::new(GraphSystem::new(n, entries.to_vec()));
            let tr = Arc::new(fwd.transposed());
            (fwd, tr)
        })
        .clone()
}

/// A one-shot factorization of `(row, col, value)` triplets that solves
/// forward and transposed: the graph solve's two programs over the pattern
/// (cached by pattern), the KLU library where the graph solve declines.
pub struct TripletLu {
    graph: Option<(GraphFactorizer, GraphFactorizer)>,
    klu: Option<KluSolver<f64>>,
}

impl TripletLu {
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        if let Some((fwd, _)) = self.graph.as_ref() {
            return Some(fwd.solve(b));
        }
        self.klu.as_ref()?.solve(b).ok()
    }

    pub fn solve_transpose(&self, b: &[f64]) -> Option<Vec<f64>> {
        if let Some((_, tr)) = self.graph.as_ref() {
            return Some(tr.solve(b));
        }
        self.klu.as_ref()?.solve_transpose(b).ok()
    }

    /// `k` right-hand sides, row-major (`rhs[i * k + a]`), solved one by
    /// one; the solutions in the same layout.
    pub fn solve_many(&self, rhs: &[f64], k: usize) -> Option<Vec<f64>> {
        if let Some((fwd, _)) = self.graph.as_ref() {
            let n = rhs.len() / k.max(1);
            let mut out = vec![0.0; rhs.len()];
            let mut col = vec![0.0; n];
            for a in 0..k {
                for i in 0..n {
                    col[i] = rhs[i * k + a];
                }
                let x = fwd.solve(&col);
                for i in 0..n {
                    out[i * k + a] = x[i];
                }
            }
            return Some(out);
        }
        self.klu.as_ref()?.solve_many(rhs, k).ok()
    }
}

/// One-shot factorization of triplets (duplicates summed) for forward and
/// transposed solves: the graph solve, the KLU library as the fallback.
/// `None` if the matrix is structurally or numerically singular.
pub fn factor_triplets_both(
    n: usize,
    rows: &[usize],
    cols: &[usize],
    values: &[f64],
) -> Option<TripletLu> {
    if sane_core::config().graph_solve {
        if let Some((entries, slot)) = dedup_entries(n, rows, cols) {
            let mut vals = vec![0.0; entries.len()];
            for (k, &v) in values.iter().enumerate() {
                vals[slot[k]] += v;
            }
            let (fs, ts) = triplet_systems(n, &entries);
            let (mut fwd, mut tr) = (fs.factorizer(), ts.factorizer());
            if fwd.factor(&vals, false) == Some(true) && tr.factor(&vals, false) == Some(true) {
                return Some(TripletLu {
                    graph: Some((fwd, tr)),
                    klu: None,
                });
            }
        }
    }
    factor_triplets_klu(n, rows, cols, values).map(|klu| TripletLu {
        graph: None,
        klu: Some(klu),
    })
}

/// Whether the pattern has a complete matching of rows to columns (a full
/// structural transversal): MC21-style augmenting-path search with a per-column
/// resume pointer, so the whole search stays near-linear on circuit patterns.
fn has_full_transversal(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> bool {
    let mut row_match = vec![usize::MAX; n]; // row -> column
    let mut col_match = vec![usize::MAX; n]; // column -> row
    let mut next = col_ptr[..n].to_vec(); // cheap-match resume pointer per column
    let mut visited = vec![usize::MAX; n]; // column -> search stamp
    let mut stack: Vec<(usize, usize)> = Vec::new(); // (column, entry index)
    for root in 0..n {
        stack.clear();
        stack.push((root, col_ptr[root]));
        visited[root] = root;
        let mut found = None;
        'search: while let Some(&mut (j, ref mut k)) = stack.last_mut() {
            // Cheap phase: an unmatched row directly in column j.
            while next[j] < col_ptr[j + 1] {
                let r = row_idx[next[j]];
                next[j] += 1;
                if row_match[r] == usize::MAX {
                    found = Some(r);
                    break 'search;
                }
            }
            // Deep phase: walk into the column matched to the next row.
            let mut pushed = false;
            while *k < col_ptr[j + 1] {
                let r = row_idx[*k];
                *k += 1;
                let jj = row_match[r];
                if visited[jj] != root {
                    visited[jj] = root;
                    stack.push((jj, col_ptr[jj]));
                    pushed = true;
                    break;
                }
            }
            if !pushed {
                stack.pop();
            }
        }
        let Some(mut r) = found else {
            return false;
        };
        // Augment along the stack: each column takes the row found below it.
        while let Some((j, _)) = stack.pop() {
            let prev = col_match[j];
            col_match[j] = r;
            row_match[r] = j;
            if prev == usize::MAX {
                break;
            }
            r = prev;
        }
    }
    true
}

/// A fixed sparse pattern with its symbolic analysis, reused across all
/// numeric factorizations of that pattern.
pub struct SparsePattern {
    n: usize,
    /// CSC pattern (deduplicated, per-column sorted).
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    /// Input entry `k` (the caller's entry order) -> CSC value slot.
    /// Duplicate positions map onto the same slot (contributions sum).
    slot: Vec<usize>,
    /// The unsymmetric analysis (KLU's block triangular form with the MC64
    /// transversal, or the LU path's row matching), computed from the values
    /// of the first numeric factorization and shared by every factorizer of
    /// the pattern: the DC drivers each hold their own `Refactorable`, and
    /// the analysis is the expensive part.
    sym: OnceLock<Sym>,
    /// LDLT candidacy: present when the pattern is large and structurally
    /// symmetric; each factorization then tests the values and takes the
    /// symmetric fast path when they match.
    ldlt: Option<LdltCand>,
    /// The pattern as the graph solve's system (see [`GraphSystem`]).
    graph: Arc<GraphSystem>,
}

impl SparsePattern {
    /// Analyze the pattern given as parallel `(rows, cols)` entry lists.
    /// `None` if the pattern is structurally singular (no complete matching),
    /// in which case the system has no unique solution for any value set and
    /// callers report a singular solve exactly as before.
    pub fn new(n: usize, rows: &[usize], cols: &[usize]) -> Option<Self> {
        debug_assert_eq!(rows.len(), cols.len());
        // Deduplicate into CSC: sort entry indices by (col, row), assign one
        // slot per distinct position, remember each entry's slot.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_unstable_by_key(|&k| (cols[k], rows[k]));
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_idx: Vec<usize> = Vec::with_capacity(rows.len());
        let mut slot = vec![0usize; rows.len()];
        let (mut prev_r, mut prev_c) = (usize::MAX, usize::MAX);
        for &k in &order {
            let (r, c) = (rows[k], cols[k]);
            if r >= n || c >= n {
                return None;
            }
            if r != prev_r || c != prev_c {
                row_idx.push(r);
                col_ptr[c + 1] += 1;
                (prev_r, prev_c) = (r, c);
            }
            slot[k] = row_idx.len() - 1;
        }
        for j in 0..n {
            col_ptr[j + 1] += col_ptr[j];
        }
        // The unsymmetric analysis depends on the values and runs at the
        // first numeric factorization (see `sym`). Structural singularity
        // does not, and is reported here, at compile time.
        if !has_full_transversal(n, &col_ptr, &row_idx) {
            return None;
        }
        // LDLT candidacy: large + structurally symmetric. Built once; the
        // per-factorization value test decides whether it is used.
        let ldlt = if n >= LDLT_BLOCK_MIN {
            LdltCand::build(n, &col_ptr, &row_idx)
        } else {
            None
        };
        let mut entries = Vec::with_capacity(row_idx.len());
        for j in 0..n {
            for k in col_ptr[j]..col_ptr[j + 1] {
                entries.push((row_idx[k], j));
            }
        }
        Some(SparsePattern {
            n,
            col_ptr,
            row_idx,
            slot,
            sym: OnceLock::new(),
            ldlt,
            graph: Arc::new(GraphSystem::new(n, entries)),
        })
    }

    /// Backend routing: KLU's BTF analysis is cheap and also yields the block
    /// structure; a pattern whose largest irreducible block crosses
    /// [`MF_BLOCK_MIN`] re-analyzes for the multifrontal backend instead.
    fn route(csc: &GeneralCsc<f64>) -> Option<Sym> {
        let klu = KluSymbolic::analyze(csc).ok()?;
        if klu.max_block_size() >= MF_BLOCK_MIN {
            if let Ok(mf) = LuSymbolic::analyze(csc) {
                return Some(Sym::Mf(Box::new(mf)));
            }
        }
        Some(Sym::Klu(klu))
    }

    /// A per-solve-loop factorizer over this pattern. On the KLU backend it
    /// uses the numeric-only `refactor` (frozen pattern + pivot sequence, no
    /// DFS, no pivot search) for every factorization after the first -- the
    /// KLU fast path for Newton iterations -- falling back to a full pivoting
    /// factor whenever the replay hits a vanished pivot. On the multifrontal
    /// backend every call is a numeric factorization over the shared analysis.
    pub fn factorizer(&self) -> Refactorable<'_> {
        Refactorable {
            pat: self,
            csc: self.scatter(&[]),
            graph: self.graph.factorizer(),
            graph_off: false,
            graph_active: false,
            klu: None,
            mf: None,
            ldlt: None,
            prev_vals: Vec::new(),
        }
    }

    /// Scatter caller-order `values` into a fresh CSC over this pattern
    /// (duplicates summed; an empty slice yields zero values).
    fn scatter(&self, values: &[f64]) -> GeneralCsc<f64> {
        debug_assert!(values.is_empty() || values.len() == self.slot.len());
        let mut vals = vec![0.0f64; self.row_idx.len()];
        for (k, &v) in values.iter().enumerate() {
            vals[self.slot[k]] += v;
        }
        GeneralCsc {
            n: self.n,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values: vals,
        }
    }
}

/// Reusable numeric factorization state over a [`SparsePattern`] for one solve
/// loop (see [`SparsePattern::factorizer`]). The convergence tests downstream
/// judge the true tape residual, so a stale KLU pivot order can only cost
/// iterations, never accuracy of the accepted solution.
pub struct Refactorable<'p> {
    pat: &'p SparsePattern,
    /// Scratch CSC (pattern fixed, values rewritten per factorization).
    csc: GeneralCsc<f64>,
    /// The graph backend, primed by the last successful factorization.
    graph: GraphFactorizer,
    /// The graph backend does not apply to this pattern (beyond its range):
    /// the library backends serve.
    graph_off: bool,
    /// The last successful factorization was the graph backend's; otherwise
    /// a library backend factored this value set (a breakdown the program
    /// could not pivot around) and solves.
    graph_active: bool,
    klu: Option<KluSolver<f64>>,
    mf: Option<LuSolver<f64>>,
    ldlt: Option<LdltSolver<f64>>,
    /// CSC values of the last *successful* factorization, for the identity
    /// fast path: a linear (or clamped/line-searched) iteration re-presents a
    /// bitwise-unchanged matrix, and the O(nnz) compare skips even the
    /// numeric refactor then. Empty until the first success.
    prev_vals: Vec<f64>,
}

impl Refactorable<'_> {
    /// Factor `values` (caller entry order, duplicates summed). `false` if the
    /// matrix is numerically singular (no valid factorization is retained).
    pub fn factor(&mut self, values: &[f64], row_scaling: bool) -> bool {
        debug_assert_eq!(values.len(), self.pat.slot.len());
        for v in self.csc.values.iter_mut() {
            *v = 0.0;
        }
        for (k, &v) in values.iter().enumerate() {
            self.csc.values[self.pat.slot[k]] += v;
        }
        dump_matrix(&self.csc);
        // Identity fast path: a bitwise-unchanged matrix (linear system, or a
        // clamped step that left the Jacobian untouched) keeps the existing
        // factors -- no refactor at all. NaN never compares equal, so a
        // poisoned value set always re-factors.
        if !self.prev_vals.is_empty() && self.prev_vals == self.csc.values {
            return true;
        }
        self.prev_vals.clear();
        self.graph_active = false;
        if !self.graph_off {
            match self.graph.factor(&self.csc.values, row_scaling) {
                Some(true) => {
                    self.graph_active = true;
                    self.prev_vals.clone_from(&self.csc.values);
                    return true;
                }
                // A breakdown the program could not pivot around: this value
                // set goes to a library backend, the program stays.
                Some(false) => {}
                None => self.graph_off = true,
            }
        }
        // Symmetric fast path (see SparsePattern::factor): LDLT numeric factor
        // over the shared analysis; on symmetry break or rank deficiency the
        // unsymmetric backends below take over for this value set.
        if let Some(cand) = &self.pat.ldlt {
            if cand.values_symmetric(&self.csc.values) {
                if let Some(f) = cand.factor(self.pat.n, &self.csc.values) {
                    self.ldlt = Some(f);
                    self.prev_vals.clone_from(&self.csc.values);
                    return true;
                }
            }
            self.ldlt = None;
        }
        let sym = match self.pat.sym.get() {
            Some(sym) => sym,
            None => match SparsePattern::route(&self.csc) {
                // A numerically singular first matrix leaves the slot empty,
                // so a later factorization with regular values analyzes anew.
                Some(s) => self.pat.sym.get_or_init(|| s),
                None => return false,
            },
        };
        match sym {
            Sym::Klu(sym) => {
                if let Some(s) = self.klu.as_mut() {
                    if s.refactor(&self.csc).is_ok() {
                        self.prev_vals.clone_from(&self.csc.values);
                        return true;
                    }
                }
                let settings = KluSettings::default().with_row_scaling(row_scaling);
                match sym.factor(&self.csc, &settings) {
                    Ok(s) => {
                        self.klu = Some(s);
                        self.prev_vals.clone_from(&self.csc.values);
                        true
                    }
                    Err(_) => {
                        self.klu = None;
                        false
                    }
                }
            }
            Sym::Mf(sym) => {
                self.mf = sym.factor(&self.csc, &SolverSettings::default()).ok();
                if self.mf.is_some() {
                    self.prev_vals.clone_from(&self.csc.values);
                }
                self.mf.is_some()
            }
        }
    }

    /// Solve against the last successful [`factor`](Self::factor).
    pub fn solve(&mut self, rhs: &[f64]) -> Option<Vec<f64>> {
        if self.graph_active {
            return Some(self.graph.solve(rhs));
        }
        if let Some(s) = &self.ldlt {
            return s.solve(rhs).ok();
        }
        match (&self.klu, &self.mf) {
            (Some(s), _) => s.solve(rhs).ok(),
            (_, Some(s)) => s.solve(rhs).ok(),
            _ => None,
        }
    }
}

/// One-shot **KLU** LU from triplets, for the adjoint / sensitivity paths that
/// need [`KluSolver::solve_transpose`] (and the batched multi-RHS solve) on
/// the same factorization. `None` on a singular matrix.
pub(crate) fn factor_triplets_klu(
    n: usize,
    rows: &[usize],
    cols: &[usize],
    values: &[f64],
) -> Option<KluSolver<f64>> {
    let a = GeneralCsc::from_triplets(n, rows, cols, values).ok()?;
    KluSolver::factor(&a, &KluSettings::default()).ok()
}

/// `SANE_DUMP_MATRIX=<dir>`: write the first assembled system matrix of each
/// process as Matrix Market (`<dir>/sane_<n>.mtx`, general real coordinate)
/// for solver benchmarks on real circuit matrices.
fn dump_matrix(csc: &GeneralCsc<f64>) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DONE: AtomicBool = AtomicBool::new(false);
    let Ok(dir) = std::env::var("SANE_DUMP_MATRIX") else {
        return;
    };
    if DONE.swap(true, Ordering::Relaxed) {
        return;
    }
    let path = std::path::Path::new(&dir).join(format!("sane_{}.mtx", csc.n));
    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix coordinate real general\n");
    out.push_str(&format!("{} {} {}\n", csc.n, csc.n, csc.row_idx.len()));
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            out.push_str(&format!(
                "{} {} {:e}\n",
                csc.row_idx[k] + 1,
                j + 1,
                csc.values[k]
            ));
        }
    }
    if let Err(e) = std::fs::write(&path, out) {
        eprintln!("SANE_DUMP_MATRIX: cannot write {}: {e}", path.display());
    } else {
        eprintln!("SANE_DUMP_MATRIX: wrote {}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_sums_duplicates_and_solves() {
        // 2x2 with a duplicated (0,0) entry: values 1.5 + 0.5 must sum to 2.
        // A = [[2, 1], [0, 3]], b = [5, 6] -> x = [1.5, 2].
        let rows = [0usize, 0, 0, 1];
        let cols = [0usize, 0, 1, 1];
        let pat = SparsePattern::new(2, &rows, &cols).expect("pattern");
        let mut fac = pat.factorizer();
        assert!(fac.factor(&[1.5, 0.5, 1.0, 3.0], true));
        let x = fac.solve(&[5.0, 6.0]).expect("solve");
        assert!(
            (x[0] - 1.5).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14,
            "{x:?}"
        );
    }

    #[test]
    fn refactorable_tracks_new_values() {
        let rows = [0usize, 0, 1, 1];
        let cols = [0usize, 1, 0, 1];
        let pat = SparsePattern::new(2, &rows, &cols).expect("pattern");
        let mut fac = pat.factorizer();
        assert!(fac.factor(&[2.0, 1.0, 0.0, 3.0], true));
        let x = fac.solve(&[5.0, 6.0]).expect("solve");
        assert!(
            (x[0] - 1.5).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14,
            "{x:?}"
        );
        // Same pattern, new values: the (refactored) solve must track them.
        assert!(fac.factor(&[4.0, 2.0, 0.0, 6.0], true));
        let x = fac.solve(&[10.0, 12.0]).expect("solve2");
        assert!(
            (x[0] - 1.5).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14,
            "{x:?}"
        );
        // Numerically singular values: factor reports failure.
        assert!(!fac.factor(&[1.0, 2.0, 2.0, 4.0], true));
    }

    #[test]
    fn structurally_singular_pattern_is_none() {
        // Empty column 1: no complete matching regardless of values.
        assert!(SparsePattern::new(2, &[0, 1], &[0, 0]).is_none());
    }

    #[test]
    fn numerically_singular_factor_is_none() {
        // Full pattern, rank 1 values.
        let rows = [0usize, 0, 1, 1];
        let cols = [0usize, 1, 0, 1];
        let pat = SparsePattern::new(2, &rows, &cols).expect("pattern");
        let mut fac = pat.factorizer();
        assert!(!fac.factor(&[1.0, 2.0, 2.0, 4.0], false));
        // The same pattern with regular values must factor.
        assert!(fac.factor(&[1.0, 2.0, 2.0, 5.0], false));
    }

    #[test]
    fn symmetric_large_pattern_takes_ldlt_and_matches() {
        // Symmetric tridiagonal above the LDLT threshold: the pattern must
        // carry the LDLT candidacy, factor through it, and solve correctly;
        // breaking symmetry in one value must fall back and still solve.
        let n = LDLT_BLOCK_MIN + 100;
        let (mut rows, mut cols) = (Vec::new(), Vec::new());
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                rows.push(i + 1);
                cols.push(i);
            }
        }
        let pat = SparsePattern::new(n, &rows, &cols).expect("pattern");
        assert!(
            pat.ldlt.is_some(),
            "structurally symmetric + large => LDLT candidate"
        );
        let sym_vals: Vec<f64> = rows
            .iter()
            .zip(&cols)
            .map(|(&r, &c)| if r == c { 4.0 } else { -1.0 })
            .collect();
        // Manufactured solution x = 1: b = A*1 (row sums).
        let mut b = vec![0.0; n];
        for (k, &v) in sym_vals.iter().enumerate() {
            b[rows[k]] += v;
        }
        let mut fac = pat.factorizer();
        assert!(fac.factor(&sym_vals, true));
        // A band pattern is the graph solve's; LDLT serves once the graph
        // solve declines a pattern.
        assert!(fac.graph_active || fac.ldlt.is_some());
        let x = fac.solve(&b).expect("solve");
        let err = x.iter().map(|v| (v - 1.0).abs()).fold(0.0f64, f64::max);
        assert!(err < 1e-10, "max err {err:e}");

        // Asymmetric values on the same pattern: must fall back and still solve.
        let mut asym = sym_vals.clone();
        asym[1] = -0.5; // one off-diagonal differs from its mirror
        let mut b2 = vec![0.0; n];
        for (k, &v) in asym.iter().enumerate() {
            b2[rows[k]] += v;
        }
        assert!(fac.factor(&asym, true));
        assert!(fac.ldlt.is_none(), "asymmetric values must not use LDLT");
        let x2 = fac.solve(&b2).expect("solve asym");
        let err2 = x2.iter().map(|v| (v - 1.0).abs()).fold(0.0f64, f64::max);
        assert!(err2 < 1e-10, "max err {err2:e}");

        // And symmetric values again: the same route as the first time.
        assert!(fac.factor(&sym_vals, true));
        assert!(fac.graph_active || fac.ldlt.is_some());
        let xr = fac.solve(&b).expect("solve sym again");
        assert!(xr.iter().map(|v| (v - 1.0).abs()).fold(0.0f64, f64::max) < 1e-10);
    }

    #[test]
    fn factor_triplets_matches_dense() {
        // 3x3 with an off-diagonal coupling; check against the hand solution.
        let rows = [0usize, 1, 2, 0, 2];
        let cols = [0usize, 1, 2, 1, 0];
        let vals = [2.0, 3.0, 4.0, 1.0, 0.5];
        let lu = factor_triplets_both(3, &rows, &cols, &vals).expect("factor");
        let b = [4.0, 9.0, 8.5];
        let x = lu.solve(&b).expect("solve");
        // A = [[2,1,0],[0,3,0],[0.5,0,4]] -> x = [0.5, 3, 2.0625]
        assert!((x[0] - 0.5).abs() < 1e-14);
        assert!((x[1] - 3.0).abs() < 1e-14);
        assert!((x[2] - 2.0625).abs() < 1e-14);
        // The KLU variant must agree and provide the transpose solve.
        let klu = factor_triplets_klu(3, &rows, &cols, &vals).expect("klu");
        let xk = klu.solve(&b).expect("klu solve");
        for i in 0..3 {
            assert!((xk[i] - x[i]).abs() < 1e-13);
        }
        let bt = [3.0, 10.0, 16.0]; // A^T x for x = [0.5, 3, 4] -> checks transpose path runs
        assert!(klu.solve_transpose(&bt).is_ok());
    }
}
