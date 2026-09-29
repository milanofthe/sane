//! Fixed-pattern sparse LU for the engine's real linear solves, on the
//! rslab library.
//!
//! The solver's contract: the sparsity pattern of a Newton / stage /
//! homotopy matrix is fixed across a solve, so the symbolic work is done
//! **once** and every iteration runs only a numeric factorization over
//! refreshed values, in place: each backend refactors into the factor of
//! the previous iteration and solves in work it keeps, so a warm Newton
//! step allocates next to nothing.
//!
//! Three rslab backends serve three structural regimes, chosen
//! automatically from the pattern's BTF analysis and a per-factorization
//! value-symmetry test:
//!
//! * **KLU** (BTF + per-block AMD + Gilbert-Peierls) for circuit-shaped
//!   patterns -- many BTF blocks / modest irreducible blocks. Its numeric-only
//!   `refactor` (frozen pattern + pivot sequence, no DFS, no pivot search)
//!   makes Newton iterations after the first factorization very cheap, and it
//!   provides the transpose solve the adjoint paths use.
//! * **Supernodal LDLT** (`rslab::LdltSolver`, Bunch-Kaufman) when the
//!   system is large ([`LDLT_BLOCK_MIN`]) and the assembled values are
//!   *symmetric* -- MNA of R/L/C networks with independent sources, i.e. the
//!   power-grid regime. Half the fill/flops of any LU and rslab's most
//!   optimized kernel; checked per factorization (O(nnz) bitwise), so a
//!   nonlinear iterate that breaks symmetry transparently falls back to LU.
//! * **Supernodal LU** (`rslab::LuSolver`, blocked SIMD kernels, scoped
//!   worker pool) for large *unsymmetric* single-block patterns crossing
//!   [`MF_BLOCK_MIN`], where scalar Gilbert-Peierls loses to blocked fronts
//!   (measured crossover ~1e4 unknowns on RC grids).
//!
//! Values are supplied in the caller's *entry order* (the tape's Jacobian
//! nonzeros followed by augmentation entries such as the gmin diagonal), with
//! duplicate `(row, col)` positions summed into one CSC slot -- the same
//! semantics faer's argsort provided. Row equilibration (the
//! `row_equilibration` solver trick) is the backend's built-in scaling: KLU's
//! row-max scaling when the trick is on, and the supernodal paths' own
//! equilibration (always on there); it is folded into factorization and
//! solves, so callers never scale right-hand sides.

use std::sync::OnceLock;

use rslab::{
    CscMatrix, GeneralCsc, KluSettings, KluSolver, KluSymbolic, LdltSolver, LdltSymbolic, LuSolver,
    LuSymbolic, SolveWork, SolverSettings,
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

/// Largest-BTF-block threshold above which the blocked supernodal LU
/// replaces KLU for this pattern. Calibrated on RC grids on Apple M3: at ~8e3
/// the two are on par, at ~1.7e4 the supernodal LU is ~1.4x faster, at 4e4
/// ~2.2x (KLU's numeric-only refactor narrows but does not close the gap).
const MF_BLOCK_MIN: usize = 10_000;

/// Dimension threshold above which a *symmetric* value set is factored by the
/// supernodal Bunch-Kaufman LDLT instead (half the flops/fill of any LU, and
/// rslab's most optimized kernel). MNA systems of R/L/C networks with
/// independent sources are symmetric -- exactly the power-grid regime.
/// Calibrated on RC-grid Laplacians (Apple M3): at 1.6e3 KLU still wins
/// (1.5 vs 2.4 ms), at ~4e3 they tie, at 8.1e3 LDLT-MF wins 12.7 vs 16.6 ms,
/// at 4e4 it wins 71 vs 97 ms (unsymmetric LU) / 195 ms (KLU).
const LDLT_BLOCK_MIN: usize = 8_000;

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
        let lsym = LdltSymbolic::analyze(&skeleton, &SolverSettings::default()).ok()?;
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

    /// The lower triangle's CSC, values to be written by [`factor`](Self::factor).
    fn lower(&self, n: usize) -> CscMatrix<f64> {
        CscMatrix {
            n,
            col_ptr: self.lcol_ptr.clone(),
            row_idx: self.lrow_idx.clone(),
            values: vec![0.0; self.lrow_idx.len()],
        }
    }

    /// Numeric LDLT factorization of the (symmetric) full-CSC values into
    /// `ldlt`, refactored in place when it holds an earlier factor of this
    /// pattern. `lower` is the lower triangle's CSC ([`lower`](Self::lower)),
    /// its values rewritten. `false` on a numerically rank-deficient matrix
    /// (the caller falls through to LU).
    fn factor(
        &self,
        vals: &[f64],
        lower: &mut CscMatrix<f64>,
        ldlt: &mut Option<LdltSolver<f64>>,
    ) -> bool {
        for (v, &k) in lower.values.iter_mut().zip(&self.lower_from) {
            *v = vals[k];
        }
        let opts = SolverSettings::default();
        in_place(
            ldlt,
            |s| self.lsym.refactor(lower, &opts, s),
            || self.lsym.factor(lower, &opts),
        )
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
/// Boxed: both analyses are large and of very different sizes.
enum Sym {
    Klu(Box<KluSymbolic>),
    Mf(Box<LuSymbolic>),
}

/// Factor into `slot` in place when it holds a factor of the same analysis,
/// fresh otherwise; `false` on a numerically singular matrix.
fn in_place<S, E>(
    slot: &mut Option<S>,
    refactor: impl FnOnce(&mut S) -> Result<(), E>,
    factor: impl FnOnce() -> Result<S, E>,
) -> bool {
    match slot {
        Some(s) => refactor(s).is_ok(),
        None => factor().map(|s| *slot = Some(s)).is_ok(),
    }
}

/// A one-shot KLU factorization of `(row, col, value)` triplets that solves
/// forward, transposed and for many right-hand sides on the same factors.
pub struct TripletLu {
    klu: KluSolver<f64>,
}

impl TripletLu {
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        self.klu.solve(b).ok()
    }

    pub fn solve_transpose(&self, b: &[f64]) -> Option<Vec<f64>> {
        self.klu.solve_transpose(b).ok()
    }

    /// `k` right-hand sides, column-major (`rhs[a * n + i]`); the solutions
    /// in the same layout.
    pub fn solve_many(&self, rhs: &[f64], k: usize) -> Option<Vec<f64>> {
        self.klu.solve_many(rhs, k).ok()
    }
}

/// One-shot factorization of triplets (duplicates summed) for the adjoint and
/// sensitivity paths. `None` if the matrix is structurally or numerically
/// singular.
pub fn factor_triplets(
    n: usize,
    rows: &[usize],
    cols: &[usize],
    values: &[f64],
) -> Option<TripletLu> {
    let a = GeneralCsc::from_triplets(n, rows, cols, values).ok()?;
    let klu = KluSolver::factor(&a, &KluSettings::default()).ok()?;
    Some(TripletLu { klu })
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
        Some(SparsePattern {
            n,
            col_ptr,
            row_idx,
            slot,
            sym: OnceLock::new(),
            ldlt,
        })
    }

    /// Backend routing: KLU's BTF analysis is cheap and also yields the block
    /// structure; a pattern whose largest irreducible block crosses
    /// [`MF_BLOCK_MIN`] re-analyzes for the supernodal backend instead.
    fn route(csc: &GeneralCsc<f64>) -> Option<Sym> {
        let klu = KluSymbolic::analyze(csc, &KluSettings::default()).ok()?;
        if klu.max_block_size() >= MF_BLOCK_MIN {
            if let Ok(mf) = LuSymbolic::analyze(csc, &SolverSettings::default()) {
                return Some(Sym::Mf(Box::new(mf)));
            }
        }
        Some(Sym::Klu(Box::new(klu)))
    }

    /// A per-solve-loop factorizer over this pattern. Every factorization
    /// after a backend's first refactors into that backend's factor in place:
    /// KLU's numeric-only `refactor` (frozen pattern + pivot sequence, no
    /// DFS, no pivot search), falling back to a full pivoting factor whenever
    /// the replay hits a vanished pivot; the supernodal paths' `refactor`
    /// over the shared analysis, into the buffers of the previous factor.
    pub fn factorizer(&self) -> Refactorable<'_> {
        Refactorable {
            pat: self,
            csc: GeneralCsc {
                n: self.n,
                col_ptr: self.col_ptr.clone(),
                row_idx: self.row_idx.clone(),
                values: vec![0.0; self.row_idx.len()],
            },
            lower: None,
            klu: None,
            mf: None,
            ldlt: None,
            active: Backend::None,
            prev_vals: Vec::new(),
            work: SolveWork::new(),
        }
    }
}

/// The backend holding the factor of the last successful factorization.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Backend {
    None,
    Ldlt,
    Klu,
    Mf,
}

/// Reusable numeric factorization state over a [`SparsePattern`] for one solve
/// loop (see [`SparsePattern::factorizer`]). The convergence tests downstream
/// judge the true tape residual, so a stale KLU pivot order can only cost
/// iterations, never accuracy of the accepted solution.
pub struct Refactorable<'p> {
    pat: &'p SparsePattern,
    /// Scratch CSC (pattern fixed, values rewritten per factorization).
    csc: GeneralCsc<f64>,
    /// The lower triangle the LDLT path factors, built on its first use.
    lower: Option<CscMatrix<f64>>,
    /// Each backend's factor, kept while another serves a value set so its
    /// next factorization is again in place.
    klu: Option<KluSolver<f64>>,
    mf: Option<LuSolver<f64>>,
    ldlt: Option<LdltSolver<f64>>,
    active: Backend,
    /// CSC values of the last *successful* factorization, for the identity
    /// fast path: a linear (or clamped/line-searched) iteration re-presents a
    /// bitwise-unchanged matrix, and the O(nnz) compare skips even the
    /// numeric refactor then. Empty until the first success.
    prev_vals: Vec<f64>,
    /// The solves' scratch, sized by the first.
    work: SolveWork<f64>,
}

impl Refactorable<'_> {
    /// Factor `values` (caller entry order, duplicates summed). `false` if the
    /// matrix is numerically singular (no valid factorization is retained).
    pub fn factor(&mut self, values: &[f64], row_scaling: bool) -> bool {
        debug_assert_eq!(values.len(), self.pat.slot.len());
        self.csc.values.fill(0.0);
        for (k, &v) in values.iter().enumerate() {
            self.csc.values[self.pat.slot[k]] += v;
        }
        // Identity fast path: a bitwise-unchanged matrix (linear system, or a
        // clamped step that left the Jacobian untouched) keeps the existing
        // factors -- no refactor at all. NaN never compares equal, so a
        // poisoned value set always re-factors.
        if !self.prev_vals.is_empty() && self.prev_vals == self.csc.values {
            return true;
        }
        self.prev_vals.clear();
        self.active = self.factor_values(row_scaling);
        if self.active != Backend::None {
            self.prev_vals.extend_from_slice(&self.csc.values);
        }
        self.active != Backend::None
    }

    /// The numeric factorization of `csc`'s values; the backend that took it.
    fn factor_values(&mut self, row_scaling: bool) -> Backend {
        let pat = self.pat;
        // Symmetric fast path: LDLT over the shared analysis; on symmetry
        // break or rank deficiency the unsymmetric backends below take over
        // for this value set.
        if let Some(cand) = &pat.ldlt {
            if cand.values_symmetric(&self.csc.values) {
                let lower = self.lower.get_or_insert_with(|| cand.lower(pat.n));
                if cand.factor(&self.csc.values, lower, &mut self.ldlt) {
                    return Backend::Ldlt;
                }
            }
        }
        let sym = match pat.sym.get() {
            Some(sym) => sym,
            None => match SparsePattern::route(&self.csc) {
                // A numerically singular first matrix leaves the slot empty,
                // so a later factorization with regular values analyzes anew.
                Some(s) => pat.sym.get_or_init(|| s),
                None => return Backend::None,
            },
        };
        let csc = &self.csc;
        let ok = match sym {
            // A replay that hits a vanished pivot factors afresh, pivoting.
            Sym::Klu(sym) => {
                self.klu.as_mut().is_some_and(|s| s.refactor(csc).is_ok()) || {
                    let settings = KluSettings::default().with_row_scaling(row_scaling);
                    self.klu = sym.factor(csc, &settings).ok();
                    self.klu.is_some()
                }
            }
            Sym::Mf(sym) => {
                let opts = SolverSettings::default();
                in_place(
                    &mut self.mf,
                    |s| sym.refactor(csc, &opts, s),
                    || sym.factor(csc, &opts),
                )
            }
        };
        match (ok, sym) {
            (false, _) => Backend::None,
            (true, Sym::Klu(_)) => Backend::Klu,
            (true, Sym::Mf(_)) => Backend::Mf,
        }
    }

    /// Solve against the last successful [`factor`](Self::factor) into `x`,
    /// in the factorizer's kept scratch: no allocation once it has served a
    /// solve. `false` without a factorization.
    pub fn solve_into(&mut self, rhs: &[f64], x: &mut [f64]) -> bool {
        dump_system(&self.csc, rhs);
        let w = &mut self.work;
        match self.active {
            Backend::Ldlt => self.ldlt.as_ref().map(|s| s.solve_into(rhs, x, w)),
            Backend::Klu => self.klu.as_ref().map(|s| s.solve_into(rhs, x, w)),
            Backend::Mf => self.mf.as_ref().map(|s| s.solve_into(rhs, x, w)),
            Backend::None => None,
        }
        .is_some_and(|r| r.is_ok())
    }

    /// [`solve_into`](Self::solve_into) a fresh vector.
    pub fn solve(&mut self, rhs: &[f64]) -> Option<Vec<f64>> {
        let mut x = vec![0.0; rhs.len()];
        self.solve_into(rhs, &mut x).then_some(x)
    }
}

/// A matrix entry [`dump_system`] can write in Matrix Market.
pub trait DumpValue: Copy {
    /// Matrix Market field name.
    const FIELD: &'static str;
    /// Append the value's Matrix Market text.
    fn push(self, out: &mut String);
}

impl DumpValue for f64 {
    const FIELD: &'static str = "real";
    fn push(self, out: &mut String) {
        out.push_str(&format!("{self:e}"));
    }
}

impl DumpValue for num_complex::Complex64 {
    const FIELD: &'static str = "complex";
    fn push(self, out: &mut String) {
        out.push_str(&format!("{:e} {:e}", self.re, self.im));
    }
}

/// `SANE_DUMP_MATRIX=<dir>`: write assembled systems as Matrix Market, for
/// solver benchmarks on real circuit matrices.
///
/// Each call writes `A` to `<dir>/sane_<n>_<k>.mtx` (general coordinate) and
/// the right-hand side to `<dir>/sane_<n>_<k>_b.mtx` (array), where `k` counts
/// the systems this process has written. `SANE_DUMP_LIMIT` caps that count
/// (default 1, the first system only). Real DC, Newton and transient systems
/// and complex AC systems use the same files and naming. Both variables are
/// read once per process.
pub fn dump_system<T: DumpValue>(csc: &GeneralCsc<T>, rhs: &[T]) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static WRITTEN: AtomicUsize = AtomicUsize::new(0);
    static TARGET: OnceLock<Option<(String, usize)>> = OnceLock::new();
    let target = TARGET.get_or_init(|| {
        let dir = std::env::var("SANE_DUMP_MATRIX").ok()?;
        let limit = std::env::var("SANE_DUMP_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        Some((dir, limit))
    });
    let Some((dir, limit)) = target else {
        return;
    };
    if csc.n == 0 {
        return;
    }
    let k = WRITTEN.fetch_add(1, Ordering::Relaxed);
    if k >= *limit {
        return;
    }
    let base = std::path::Path::new(&dir).join(format!("sane_{}_{k:03}", csc.n));
    let mut a = format!(
        "%%MatrixMarket matrix coordinate {} general\n{} {} {}\n",
        T::FIELD,
        csc.n,
        csc.n,
        csc.row_idx.len()
    );
    for j in 0..csc.n {
        for p in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            a.push_str(&format!("{} {} ", csc.row_idx[p] + 1, j + 1));
            csc.values[p].push(&mut a);
            a.push('\n');
        }
    }
    let mut b = format!(
        "%%MatrixMarket matrix array {} general\n{} 1\n",
        T::FIELD,
        rhs.len()
    );
    for &v in rhs {
        v.push(&mut b);
        b.push('\n');
    }
    let path = base.with_extension("mtx");
    let path_b = base.with_file_name(format!("sane_{}_{k:03}_b.mtx", csc.n));
    match std::fs::write(&path, a).and_then(|_| std::fs::write(&path_b, b)) {
        Ok(()) => eprintln!("SANE_DUMP_MATRIX: wrote {}", path.display()),
        Err(e) => eprintln!("SANE_DUMP_MATRIX: cannot write {}: {e}", path.display()),
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
        assert_eq!(fac.active, Backend::Ldlt);
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
        assert_ne!(
            fac.active,
            Backend::Ldlt,
            "asymmetric values must not use LDLT"
        );
        let x2 = fac.solve(&b2).expect("solve asym");
        let err2 = x2.iter().map(|v| (v - 1.0).abs()).fold(0.0f64, f64::max);
        assert!(err2 < 1e-10, "max err {err2:e}");

        // Symmetric values again: LDLT refactors in place, the same bits as
        // the first factorization.
        assert!(fac.factor(&sym_vals, true));
        assert_eq!(fac.active, Backend::Ldlt);
        let mut xr = vec![0.0; n];
        assert!(fac.solve_into(&b, &mut xr));
        assert!(x.iter().zip(&xr).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn factor_triplets_solves_forward_and_transposed() {
        // A = [[2,1,0],[0,3,0],[0.5,0,4]] with an off-diagonal coupling.
        let rows = [0usize, 1, 2, 0, 2];
        let cols = [0usize, 1, 2, 1, 0];
        let vals = [2.0, 3.0, 4.0, 1.0, 0.5];
        let lu = factor_triplets(3, &rows, &cols, &vals).expect("factor");
        let close =
            |x: &[f64], want: &[f64]| x.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-14);
        let x = lu.solve(&[4.0, 9.0, 8.5]).expect("solve");
        assert!(close(&x, &[0.5, 3.0, 2.0625]), "{x:?}");
        // A^T [0.5, 3, 4] = [3, 9.5, 16].
        let xt = lu.solve_transpose(&[3.0, 9.5, 16.0]).expect("transpose");
        assert!(close(&xt, &[0.5, 3.0, 4.0]), "{xt:?}");
    }
}
