//! Numeric solver for SANE DAEs: a damped Newton DC operating-point solve
//! with a **sparse** Jacobian and **sparse LU** (rslab's KLU), all in Rust.
//!
//! The residual and the nonzero Jacobian entries are evaluated by compiled
//! [`Tape`]s over reusable buffers (no per-iteration allocation, no symbol
//! hashing) -- which is what makes a Newton loop over a large circuit cheap.
//! The Jacobian is built sparsely (`dae::jacobian_x_coo`: only structurally
//! nonzero entries are differentiated, ~O(nnz)).
//!
//! Two things make the Newton loop scale to large nonlinear circuits:
//!
//! 1. **Symbolic reuse.** The block triangularization and fill-reducing
//!    ordering (KLU's BTF + per-block AMD) depend only on the Jacobian
//!    *pattern*, which is fixed across Newton iterations. We compute them once
//!    in [`CompiledDc::new`] and per iteration only refill the numeric values
//!    and run the numeric factorization (see [`sparse::SparsePattern`]). On
//!    large circuits the ordering is the per-iteration bottleneck, so this is
//!    the dominant speedup.
//! 2. **gmin homotopy.** Transistor circuits rarely converge from `x = 0`. When
//!    the plain Newton solve stalls we add a conductance `gmin` from every node
//!    to ground (making the system diagonally dominant), solve, then step
//!    `gmin` down to zero, each continuation point warm-started from the last.
//!    The diagonal is kept in the sparsity pattern unconditionally so injecting
//!    `gmin` never changes the pattern (and costs nothing when `gmin = 0`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Process-global count of sparse `dF/dx` factorizations run by `factor_fx` (the
/// adjoint / sensitivity path). Used to assert the "factor once, K right-hand
/// sides" contract of [`CompiledDc::hessian`] (#48) rather than one factorization
/// per parameter. Relaxed; exact only when a single analysis runs at a time.
static FACTOR_FX_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Read the [`FACTOR_FX_CALLS`] counter (for count-based sensitivity verification).
pub fn factor_fx_calls() -> usize {
    FACTOR_FX_CALLS.load(Ordering::Relaxed)
}

/// Reset the [`FACTOR_FX_CALLS`] counter to zero, returning the previous value.
pub fn reset_factor_fx_calls() -> usize {
    FACTOR_FX_CALLS.swap(0, Ordering::Relaxed)
}

/// Process-global count of HB state-waveform syntheses ([`hb`]'s `synth`, the per
/// unknown inverse-FFT AFT). Used to assert the "synth once per Newton iteration"
/// contract (#51): one synthesis per iteration rather than one for the residual
/// and one for the Jacobian. Relaxed; exact only when a single HB solve runs.
pub(crate) static HB_SYNTH_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Read the [`HB_SYNTH_CALLS`] counter (for count-based HB verification).
pub fn hb_synth_calls() -> usize {
    HB_SYNTH_CALLS.load(Ordering::Relaxed)
}

/// Reset the [`HB_SYNTH_CALLS`] counter to zero, returning the previous value.
pub fn reset_hb_synth_calls() -> usize {
    HB_SYNTH_CALLS.swap(0, Ordering::Relaxed)
}

use rsdag::{Crossing, Graph, SymbolId, Tape};
use sane_dae::Limit;
use sane_mna::SourceFn;

pub mod hb;
pub use delay::DelayHistory;
// Device controlling-voltage limiting (pnjlim/fetlim) is an internal solver
// step with no external consumers.
mod bundle;
pub(crate) mod limiting;
// Construction of a `CompiledDc` (Jacobian extraction, tape compilation).
mod compile;
// The DC operating-point solve and its continuation cascade.
mod dc;
mod delay;
mod esdirk32;
mod events;
mod newton;
mod stage;
mod trap;
// The hot-tape backend ladder (interpreter / specialized / native).
mod eval;
pub mod parallel;
// Schur-complement partitioning for large mostly-linear systems.
mod schur;
// Adjoint sensitivities and the second-order-adjoint Hessian.
mod sens;
pub(crate) mod sparse;
#[cfg(test)]
mod tests;
mod transient;
mod transient_adjoint;

#[cfg(feature = "jit")]
pub(crate) use eval::jit_enabled;
pub(crate) use eval::{PrologToken, StepEval};
use schur::{LinCache, Partition};
use sens::{CompiledHessian, HistJac, ParamJac};

use sane_core::constants::*;

/// Composable set of DC convergence aids ("solver tricks"). The robust DC solve
/// is a cascade -- plain damped Newton, then a sequence of continuation
/// fallbacks -- and each stage (plus the per-step shaping inside Newton) is an
/// independently toggleable trick here, rather than a hardcoded chain gated by
/// environment variables. [`CompiledDc::solve_dc`] runs the default set;
/// [`CompiledDc::solve_dc_with`] takes an explicit set (for benchmarking a
/// single trick, or enabling device limiting for hard FET-dense circuits).
#[derive(Clone, Copy, Debug)]
pub struct SolverTricks {
    /// Curve-aware per-device junction / channel limiting (`pnjlim` / `fetlim`)
    /// in the continuation correctors (the robust path). The fast-path Newton
    /// always limits when the devices declare limits: the Newton step is
    /// shortened as a whole to the largest fraction that keeps every limited
    /// junction within its bound (see `limiting::apply`), which is what takes
    /// a BJT mirror from 719 iterations through the cascade to 7 plain ones.
    /// Measured neutral in the correctors on the fixture corpus.
    pub device_limiting: bool,
    /// Backtracking line search on the residual norm.
    pub line_search: bool,
    /// Composite Newton step (Traub): after a full Newton step is accepted,
    /// take one more chord step on the factorization just built, from the
    /// residual at the new iterate (the residual-only tape, no Jacobian, no
    /// refactorization). Two residual evaluations and one solve buy an order
    /// of convergence: the pair is third-order per Jacobian instead of
    /// second. Taken only when it contracts the residual.
    pub composite_step: bool,
    /// gmin predictor-corrector homotopy fallback.
    pub gmin_continuation: bool,
    /// Source-stepping continuation fallback (ramps the independent sources).
    pub source_continuation: bool,
    /// Per-device companion continuation fallback (the models' linear form).
    pub companion_continuation: bool,
    /// Node-adaptive damped-Newton last-resort fallback (loads weak diagonals).
    pub node_adaptive: bool,
    /// Pseudo-transient relaxation fallback: when every static continuation is
    /// exhausted (typically stuck at a fold where the tracked branch ceases to
    /// exist), integrate the circuit's own dynamics over growing horizons and
    /// polish at the floor. The industry-standard last resort.
    pub pseudo_transient: bool,
    /// Schur-complement cached-factorization partitioning (a performance aid for
    /// large mostly-linear systems; off forces a plain sparse LU per iteration).
    pub partition: bool,
    /// Row-equilibrate the stage matrix (divide each row by its largest magnitude
    /// entry, KLU `scale=2`) before factorizing -- improves pivot quality and the
    /// conditioning of stiff transient solves. Off reverts to the raw matrix.
    pub row_equilibration: bool,
    /// Land transient steps exactly on independent-source waveform discontinuities
    /// (PWL corners, PULSE edges, EXP kinks) instead of integrating over them. Off
    /// lets the adaptive controller step freely (and smooth or reject across kinks).
    pub breakpoints: bool,
    /// Locate the crossings of the declared switching surfaces (Verilog-A
    /// `@(cross ...)`, switch thresholds) and land transient steps on them, so a
    /// hard mode change happens at a step boundary. Off lets the controller
    /// discover the discontinuity through rejects.
    pub events: bool,
    /// `.nodeset` stiff-pin phase: when a node-set is registered
    /// ([`set_nodeset`](CompiledDc::set_nodeset)), seed every cold solve with the
    /// symmetry-breaking pin phase. Gates the *data* in `self.nodeset` exactly as
    /// `device_limiting` gates `self.limits`; off ignores the node-set (for A/B
    /// benchmarking whether it helps). No effect when no node-set is registered.
    pub nodeset: bool,
}

impl Default for SolverTricks {
    fn default() -> Self {
        Self {
            // Off: measured worse on the fixture corpus once the solver-side
            // clamp was gone (60 ms -> 695 ms) -- this implementation limits in
            // the continuation correctors, where it over-throttles the very
            // sweeps that are meant to move. VACASK's equivalent lives in the
            // model (`$limit`), applied at the device's own scale on every
            // evaluation; SANE's `$limit` support is the place to take this
            // further, not the trick toggle.
            device_limiting: false,
            line_search: true,
            composite_step: true,
            gmin_continuation: true,
            source_continuation: true,
            companion_continuation: true,
            node_adaptive: true,
            pseudo_transient: true,
            partition: true,
            nodeset: true,
            row_equilibration: true,
            breakpoints: true,
            events: true,
        }
    }
}

/// Per-component DC convergence criterion (SPICE-style `reltol`/`abstol`/
/// `vntol`). A Newton iterate is converged when, for *every* unknown, both the
/// residual is below its absolute floor and the update is below
/// `reltol*|x| + floor` -- judged per component on the right physical scale (a
/// KCL row's residual is a current, a KVL row's a voltage) instead of one scalar
/// L2 norm that mixes nanoamps and kilovolts. The reltol part of SPICE's
/// residual test (a sum of the branch currents into a node) is intentionally
/// omitted: the engine does not accumulate per-branch currents, so the residual
/// uses the absolute floor only, and reltol carries on the update.
#[derive(Clone, Copy, Debug)]
pub struct Convergence {
    /// Relative update tolerance.
    pub reltol: f64,
    /// Absolute current floor (KCL-row residual, branch-current update).
    pub abstol: f64,
    /// Absolute voltage floor (KVL-row residual, node-voltage update).
    pub vntol: f64,
}

impl Default for Convergence {
    /// Tight engine-grade defaults (not SPICE's loose `reltol = 1e-3`): at least
    /// as strict as the previous scalar `1e-10` solve, just unit-aware.
    fn default() -> Self {
        Self {
            reltol: DC_RELTOL,
            abstol: DC_ABSTOL,
            vntol: DC_VNTOL,
        }
    }
}

impl Convergence {
    /// The classic loose SPICE defaults (`reltol 1e-3`, `abstol 1e-12 A`,
    /// `vntol 1e-6 V`): faster, SPICE-compatible, less accurate.
    pub fn spice() -> Self {
        Self {
            reltol: 1e-3,
            abstol: 1e-12,
            vntol: 1e-6,
        }
    }

    /// Map a legacy scalar tolerance onto the per-component criterion: the scalar
    /// becomes the absolute current/voltage floor, with the tight default reltol.
    /// Keeps the scalar `solve_dc(tol)` API behaving (slightly stricter, now with
    /// an update test) while routing it through the unified criterion.
    pub fn from_tol(tol: f64) -> Self {
        Self {
            reltol: DC_RELTOL,
            abstol: tol,
            vntol: tol,
        }
    }
}
/// The worker pool's thread count for the parallel passes (sweeps, the
/// harmonic-balance sampling): `0` restores the default, `n` asks for `n`
/// threads. Takes effect before the pool is first used (see
/// [`parallel`]).
pub fn set_parallelism(threads: usize) {
    sane_core::update_config(|c| c.threads = if threads == 0 { None } else { Some(threads) });
}

/// Where an input slot of the tape draws its value from.
enum InputSrc {
    X(usize),
    Xdot(usize),
    P(usize),
    T,
    /// Delay-history input `k`: the integrator-provided interpolated value of
    /// the delayed signal (see `delay::with_hist_values`).
    Hist(usize),
}

/// Precomputed sparse symbolic analysis (KLU BTF + per-block AMD over the
/// augmented pattern: Jacobian nonzeros followed by the full diagonal), reused
/// across Newton iterations. Values are refilled in the same entry order,
/// duplicates summed (see [`sparse::SparsePattern`]).
struct Symbolic {
    pattern: sparse::SparsePattern,
}

/// Transient integration method. The adaptive outer loop and the method's single
/// step (`esdirk32_step`) are split so further integrators can be added behind the
/// same interface.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum TransientMethod {
    /// ESDIRK32: 3rd-order, 4-stage, singly-diagonally-implicit (stages solved in
    /// sequence). Its modified-Newton stage solve -- per-stage device limiting and
    /// warm-starting -- is robust on hard, strongly-nonlinear circuits, where a
    /// fully-implicit coupled solve (RADAU IIA, evaluated and removed) is not.
    #[default]
    Esdirk32,
    /// Trapezoidal rule (SPICE `trap`): 2nd-order, A-stable, one implicit solve
    /// per step with a polynomial predictor as Newton start and a filtered
    /// predictor-corrector error estimate. The iteration-economy choice for
    /// switching / digital-style transients.
    Trap,
}

impl TransientMethod {
    /// Parse a method name (case-insensitive); `None` if unknown. The empty string
    /// selects the default ([`Self::Esdirk32`]).
    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "esdirk32" | "esdirk" => Some(Self::Esdirk32),
            "trap" | "trapezoidal" => Some(Self::Trap),
            _ => None,
        }
    }
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Esdirk32 => "ESDIRK32",
            Self::Trap => "TRAP",
        }
    }
}

/// Integration counters (logged at debug level), shared by the transient
/// integrators: steps taken / rejected, total inner Newton iterations, and
/// Jacobian (re)factorizations (= full `dF/dx` tape evals).
#[derive(Default)]
pub(crate) struct Stats {
    pub steps: usize,
    pub rejects: usize,
    pub iters: usize,
    pub refacs: usize,
    /// Switching events fired.
    pub events: usize,
    /// Steps retaken toward a located switching surface.
    pub event_steps: usize,
}

/// A switching event of a transient: surface `index` (see
/// [`CompiledDc::event_names`]) crossed at `t`; `direction` is `+1` when the
/// surface expression rose through zero, `-1` when it fell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransientEvent {
    pub t: f64,
    pub index: usize,
    pub direction: i8,
}

/// A DAE compiled for repeated numeric evaluation/solve.
pub struct CompiledDc {
    n: usize,
    nnz_x: usize,
    jx_rows: Vec<usize>,
    jx_cols: Vec<usize>,
    /// Position of each unknown's diagonal within the jacobian-x value array
    /// (`None` if structurally absent). Fixed by the sparsity pattern, so it is
    /// precomputed once here instead of rescanning the nonzeros every
    /// node-adaptive corrector call.
    diag_idx: Vec<Option<usize>>,
    jxd_rows: Vec<usize>,
    jxd_cols: Vec<usize>,
    /// residuals ++ jacobian-x nonzeros (one pass for a Newton step).
    tape_step: StepEval,
    /// residuals only (cheap line-search evaluations).
    tape_res: StepEval,
    /// jacobian-x' nonzeros.
    tape_jxd: StepEval,
    /// Compiled `dF/dp` (parameter Jacobian) for exact adjoint sensitivity, built
    /// lazily on first sensitivity / Hessian call (it is ~O(n^2)-ish to build at
    /// scale and only needed for sensitivity), via [`ensure_param_jac`].
    pjac: std::sync::OnceLock<ParamJac>,
    input_src: Vec<InputSrc>,
    /// Precomputed (buffer slot, source index) lists for [`Self::patch_inputs`].
    input_x_slots: Vec<(u32, u32)>,
    input_xdot_slots: Vec<(u32, u32)>,
    input_t_slots: Vec<u32>,
    input_hist_slots: Vec<(u32, u32)>,
    param_syms: Vec<SymbolId>,
    /// Lazily compiled `∂F/∂hist` (frequency-domain delay coupling), built by
    /// [`ensure_hist_jac`](Self::ensure_hist_jac).
    hjac: std::sync::OnceLock<HistJac>,
    /// Transport delays: the delayed-source unknown per delay (whose history
    /// the transient loop records and queries; the delay OUTPUT needs no
    /// position -- its residual `d - hist = 0` pins it through the tape's
    /// history input), plus the taus compiled over the parameter inputs
    /// (evaluated once per solve).
    delay_src: Vec<usize>,
    tape_tau: Option<Tape>,
    /// The switching surfaces as one tape over the step inputs (`None` when
    /// the DAE declares no events), their directions and names, and the
    /// events of the most recent transient (interior mutability like the
    /// gmin flags: the solve API is `&self`).
    tape_event: Option<Tape>,
    event_dirs: Vec<Crossing>,
    event_names: Vec<String>,
    last_events: std::sync::Mutex<Vec<TransientEvent>>,
    /// Per-param flag: is this the DC value of an independent source (a `V`/`I`
    /// element value, no `.` in the name)? Used by source stepping to ramp only
    /// the excitation, leaving component values fixed.
    source_mask: Vec<bool>,
    /// Reused sparse symbolic factorization (`None` only for degenerate
    /// patterns faer rejects, in which case we rebuild per iteration).
    symbolic: Option<Symbolic>,
    /// Per jx nonzero (aligned with `jx_rows`/`jx_cols`): does the entry vary
    /// with an unknown (a nonlinear/device entry) or is it constant (a linear,
    /// LTI element)? Computed once from the symbolic Jacobian. The DC Schur
    /// partition and harmonic balance both read it -- in HB a constant entry is
    /// frequency-diagonal `(G + jkw0*C)`, a variable one couples harmonics.
    jx_var: Vec<bool>,
    jxd_var: Vec<bool>,
    /// Linear/nonlinear partition of the Jacobian for cached-factorization
    /// (Schur-complement) solves. `None` when partitioning is not worthwhile
    /// (the nonlinear part is not a small fraction of the system).
    partition: Option<Partition>,
    /// Compiled second-order-adjoint Hessian (the Lagrangian second-derivative
    /// blocks as a reusable tape), so every Hessian is a leaf-value re-eval. Built
    /// lazily on first `hessian()` call (it is the heaviest, ~O(n^2) extract stage
    /// and is only needed for second-order sensitivity), via [`ensure_hessian`].
    chess: std::sync::OnceLock<CompiledHessian>,
    /// Base input symbols (x, differential xdot, params, t), kept so the lazy
    /// parameter-Jacobian and Hessian tapes can be compiled on demand.
    base_inputs: Vec<SymbolId>,
    /// Companion conductance network `(row, col, value)` from the device models
    /// (their linear `lambda = 0` form) -- the per-device homotopy continuation.
    companion: Vec<(usize, usize, f64)>,
    /// Reused symbolic factorization for the companion-augmented homotopy matrix
    /// `dF/dx + (1-lambda)*G_comp + gmin*I` (#52). Its pattern is the `dF/dx`
    /// nonzeros plus the companion `(row, col)` positions plus the full diagonal --
    /// fixed across the continuation, only the values change with `lambda`. Built
    /// lazily on first `companion_solve` (the fallback path most circuits never
    /// hit); `Some(None)` marks a degenerate pattern faer rejected.
    companion_symbolic: std::sync::OnceLock<Option<Symbolic>>,
    /// Per-device controlling-voltage limits (`pnjlim`/`fetlim`), mapped to
    /// global unknown indices, applied between iterates when the `device_limiting`
    /// trick is on. Empty for transformed DAEs (which carry no device limits).
    limits: Vec<Limit>,
    /// Number of leading node-voltage unknowns (the KCL rows `tape_iscale`
    /// splits into current terms).
    n_nodes: usize,
    /// What each unknown physically is. The convergence floors, the gmin
    /// shunts and the companion baseline are taken per kind: a device-internal
    /// node is a node (KCL row, volts) wherever it sits in the layout, a
    /// branch current is amperes, a state has no circuit unit.
    kinds: Vec<sane_dae::UnknownKind>,
    /// Tape over the individual current terms of every KCL node row (the additive
    /// summands of `residuals[0..n_nodes]`). Evaluating it and summing the
    /// magnitudes per node yields the SPICE node-current scale `sum|I_branch|`,
    /// used for the *relative* KCL convergence test (see `residual_relative_ok`).
    /// Built only for nonlinear circuits (the only ones that can reach the
    /// node-adaptive fallback); `None` for linear networks, which never use it.
    tape_iscale: Option<StepEval>,
    /// Per node row: `(offset, len)` slice into `tape_iscale`'s outputs giving
    /// that node's current terms. Length `n_nodes`.
    iscale_rows: Vec<(usize, usize)>,
    /// User `.nodeset` targets `(unknown_index, value)`. Empty by default. When
    /// set, every *cold-started* DC solve runs a stiff-pin phase first
    /// (symmetry breaking; see [`solve_dc_nodeset`](Self::solve_dc_nodeset)), so
    /// all analyses honor the node-set without per-call-site plumbing. Warm
    /// (non-empty `x0`) solves ignore it -- the warm start already supersedes it.
    nodeset: Vec<(usize, f64)>,
    /// The modular numerical-trick set this model honors across all analyses (DC
    /// cascade shaping, the linear-solver and transient aids). [`solve_dc`] uses it
    /// as the default trick set; the transient / stage factorization read it
    /// directly. Configure with [`set_tricks`](CompiledDc::set_tricks); override
    /// per call with [`solve_dc_with`](CompiledDc::solve_dc_with).
    tricks: SolverTricks,
    /// Independent-source stimulus shapes by element name (from the DAE), for
    /// transient breakpoints and the HB fundamental.
    sources: Vec<(String, SourceFn)>,
    /// Parameter name -> index into `p`, to resolve a source's time parameters
    /// (`"V1.pulse_td"`, ...) for breakpoints without re-borrowing the `Graph`.
    param_index: HashMap<String, usize>,
    /// Regularization flag for the most recent DC solve: the IEEE-754 bits of the
    /// held gmin when the operating point converged only under a raised shunt
    /// (gmin stepping could not reach the `GMIN_DC` floor), or `0` when the true
    /// floor was reached. Set by [`source_continuation`], reset at the top of
    /// [`solve_dc_conv_with`]; read via [`last_regularized_gmin`]. Drives the
    /// analysis layer's settle fallback, so nothing else may set it -- the
    /// second way a point can depend on gmin lives in `last_gmin_row` and is
    /// combined only where the two are reported. Interior mutability (an atomic)
    /// so the flag rides through the `&self` solve API without threading it
    /// through every `(x, conv, iters)` return (issue #54).
    last_gmin_hold: std::sync::atomic::AtomicU64,
    /// The worst gmin-dominated unknown of the most recent DC solve, or
    /// `usize::MAX` for none, alongside the IEEE-754 bits of its current share.
    /// Read via [`last_gmin_dominance`]; same interior-mutability reasoning.
    last_gmin_row: std::sync::atomic::AtomicUsize,
    last_gmin_share: std::sync::atomic::AtomicU64,
}
impl CompiledDc {
    pub fn dim(&self) -> usize {
        self.n
    }
    pub fn nnz(&self) -> usize {
        self.nnz_x
    }
    /// Number of parameters (columns of `dF/dp`), in `param_names` order.
    pub fn param_count(&self) -> usize {
        self.param_syms.len()
    }
    /// Parameter symbols, in column order.
    pub fn param_syms(&self) -> &[SymbolId] {
        &self.param_syms
    }
    pub fn param_names(&self, ctx: &Graph) -> Vec<String> {
        self.param_syms
            .iter()
            .map(|&s| ctx.symbol_name(s).to_string())
            .collect()
    }

    fn fill_inputs(&self, x: &[f64], xdot: &[f64], p: &[f64], t: f64, inputs: &mut Vec<f64>) {
        inputs.clear();
        for src in &self.input_src {
            inputs.push(match *src {
                InputSrc::X(i) => x.get(i).copied().unwrap_or(0.0),
                InputSrc::Xdot(i) => xdot.get(i).copied().unwrap_or(0.0),
                InputSrc::P(j) => p.get(j).copied().unwrap_or(0.0),
                InputSrc::T => t,
                InputSrc::Hist(k) => crate::delay::hist_value(k),
            });
        }
    }

    /// Patch only the state-dependent entries (x / xdot / t / history) of an
    /// input buffer a prior [`fill_inputs`](Self::fill_inputs) prepared. The
    /// parameter entries -- the bulk of a compact-model input vector (PSP103:
    /// ~800 of ~900) -- are constant over an integration, so the inner Newton
    /// loops skip re-copying them on every evaluation.
    fn patch_inputs(&self, x: &[f64], xdot: &[f64], t: f64, inputs: &mut [f64]) {
        debug_assert_eq!(inputs.len(), self.input_src.len(), "buffer not prepared");
        for &(dst, i) in &self.input_x_slots {
            inputs[dst as usize] = x.get(i as usize).copied().unwrap_or(0.0);
        }
        for &(dst, i) in &self.input_xdot_slots {
            inputs[dst as usize] = xdot.get(i as usize).copied().unwrap_or(0.0);
        }
        for &dst in &self.input_t_slots {
            inputs[dst as usize] = t;
        }
        for &(dst, k) in &self.input_hist_slots {
            inputs[dst as usize] = crate::delay::hist_value(k as usize);
        }
    }

    /// Residual `F(x, xdot, p, t)`.
    pub fn residual(&self, x: &[f64], xdot: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        self.fill_inputs(x, xdot, p, t, &mut inputs);
        self.tape_res.eval(&inputs, &mut work, &mut out);
        out
    }

    fn dense(rows: &[usize], cols: &[usize], data: &[f64], n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; n]; n];
        for (k, &v) in data.iter().enumerate() {
            m[rows[k]][cols[k]] = v;
        }
        m
    }

    /// Dense `dF/dx`.
    pub fn jacobian_x(&self, x: &[f64], xdot: &[f64], p: &[f64], t: f64) -> Vec<Vec<f64>> {
        let (_, _, data) = self.jacobian_x_sparse(x, xdot, p, t);
        Self::dense(&self.jx_rows, &self.jx_cols, &data, self.n)
    }
    /// Dense small-signal system matrix `dF/dx + GMIN_DC*I`: the Jacobian
    /// regularized with the same node-to-ground shunt the DC / transient solves
    /// carry, so a small-signal linearization (poles, zeros, AC, noise, MOR) is
    /// consistent with the operating point that was actually solved. It also
    /// keeps the pencil `G + sC` non-singular when an unknown is structurally
    /// decoupled at the bias (e.g. the branch current of a grounded thermal
    /// node in a self-heating compact model), which would make the raw `dF/dx`
    /// singular and the eigen/solve fail.
    pub fn system_matrix_dc(&self, x: &[f64], xdot: &[f64], p: &[f64], t: f64) -> Vec<Vec<f64>> {
        let mut g = self.jacobian_x(x, xdot, p, t);
        for i in 0..self.n {
            g[i][i] += GMIN_DC;
        }
        g
    }
    /// Dense `dF/dx'`.
    pub fn jacobian_xdot(&self, x: &[f64], xdot: &[f64], p: &[f64], t: f64) -> Vec<Vec<f64>> {
        let (_, _, data) = self.jacobian_xdot_sparse(x, xdot, p, t);
        Self::dense(&self.jxd_rows, &self.jxd_cols, &data, self.n)
    }
    /// Sparse `dF/dx` as `(rows, cols, values)`.
    pub fn jacobian_x_sparse(
        &self,
        x: &[f64],
        xdot: &[f64],
        p: &[f64],
        t: f64,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        self.fill_inputs(x, xdot, p, t, &mut inputs);
        self.tape_step.eval(&inputs, &mut work, &mut out);
        (
            self.jx_rows.clone(),
            self.jx_cols.clone(),
            out[self.n..].to_vec(),
        )
    }
    /// Sparse `dF/dx'` as `(rows, cols, values)`.
    pub fn jacobian_xdot_sparse(
        &self,
        x: &[f64],
        xdot: &[f64],
        p: &[f64],
        t: f64,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        self.fill_inputs(x, xdot, p, t, &mut inputs);
        self.tape_jxd.eval(&inputs, &mut work, &mut out);
        (self.jxd_rows.clone(), self.jxd_cols.clone(), out)
    }

    /// Sparse regularized small-signal conductance `G = dF/dx + GMIN_DC*I` at
    /// the operating point `(x, p)` (`xdot = 0`, `t = 0`), as triplets with the
    /// gmin shunt appended on every diagonal -- the same regularization the DC
    /// solve carried, so every small-signal analysis (AC, noise, symbolic
    /// reduction) linearizes the system that was actually solved. The single
    /// assembly point for the `A = G + jwC` builders.
    pub fn system_triplets_dc(&self, x: &[f64], p: &[f64]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let xdot = vec![0.0; self.n];
        let (mut r, mut c, mut v) = self.jacobian_x_sparse(x, &xdot, p, 0.0);
        for i in 0..self.n {
            r.push(i);
            c.push(i);
            v.push(GMIN_DC);
        }
        (r, c, v)
    }

    /// Reusable symbolic factorization for the **implicit-RK stage** linear system
    /// `J_stage = dF/dx + α·C` (`C = dF/dx'`, constant). The pattern is the union of
    /// the `dF/dx` and `dF/dx'` nonzeros plus the full diagonal (so a `gmin` shunt is
    /// free). `build_symbolic` sums duplicate `(i, j)` entries via the argsort, so the
    /// two blocks overlap freely and the value layout is simply the two blocks then
    /// the diagonal. Built once and reused across every stage and step (the pattern
    /// is invariant under the integrator).
    pub(crate) fn stage_symbolic(&self) -> Option<Symbolic> {
        let mut rows = self.jx_rows.clone();
        rows.extend_from_slice(&self.jxd_rows);
        let mut cols = self.jx_cols.clone();
        cols.extend_from_slice(&self.jxd_cols);
        Self::build_symbolic(self.n, &rows, &cols)
    }

    /// Factorize the stage matrix `dF/dx + α·C + gmin·I` into the transient's
    /// stage factorization cache `fac` (a [`sparse::Refactorable`] over the
    /// combined [`stage_symbolic`](Self::stage_symbolic) pattern). The
    /// modified-Newton integrator refreshes once per step and on convergence
    /// stalls; through the cache every refresh after the very first is a KLU
    /// numeric-only refactor (frozen pivot sequence, no DFS / pivot search)
    /// rather than a full pivoting factorization. Row equilibration (the
    /// `row_equilibration` trick) is the backend's built-in scaling, folded
    /// into factorization and solves. `false` on a singular stage matrix.
    pub(crate) fn factorize_stage(
        &self,
        fac: &mut sparse::Refactorable<'_>,
        dfdx_vals: &[f64],
        c_vals: &[f64],
        alpha: f64,
        gmin: f64,
        valbuf: &mut Vec<f64>,
    ) -> bool {
        let n = self.n;
        valbuf.clear();
        valbuf.extend_from_slice(dfdx_vals);
        valbuf.extend(c_vals.iter().map(|&c| c * alpha));
        valbuf.extend(std::iter::repeat_n(gmin, n));
        fac.factor(valbuf, self.tricks.row_equilibration)
    }

    /// Are transport delays present? Analyses without delay support guard on
    /// this and report a clear error instead of silently mis-solving.
    pub fn has_delays(&self) -> bool {
        !self.delay_src.is_empty()
    }

    /// Unknown indices of the delayed source signals, aligned with
    /// [`delay_taus`](Self::delay_taus).
    pub fn delay_sources(&self) -> &[usize] {
        &self.delay_src
    }

    /// The delay times, evaluated over the parameter vector.
    pub fn delay_taus(&self, p: &[f64]) -> Vec<f64> {
        match &self.tape_tau {
            None => Vec::new(),
            Some(tape) => {
                let (mut work, mut out) = (Vec::new(), Vec::new());
                tape.eval(p, &mut work, &mut out);
                out
            }
        }
    }

    /// Register `.nodeset` targets `(unknown_index, value)`. Once set, every
    /// *cold-started* DC solve (empty `x0`) runs the stiff-pin phase first, so all
    /// analyses honor the node-set without per-call-site plumbing. Pass an empty
    /// vector to clear.
    pub fn set_nodeset(&mut self, nodeset: Vec<(usize, f64)>) {
        self.nodeset = nodeset;
    }

    /// The modular trick set this model honors (default [`SolverTricks::default`]).
    pub fn tricks(&self) -> SolverTricks {
        self.tricks
    }

    /// Configure the modular trick set honored across analyses (DC default,
    /// transient / stage factorization). Per-call DC overrides still win via
    /// [`solve_dc_with`](CompiledDc::solve_dc_with).
    pub fn set_tricks(&mut self, tricks: SolverTricks) {
        self.tricks = tricks;
    }

    /// All independent-source waveform discontinuities in `(t0, t1]`, sorted and
    /// near-coincident corners merged -- the instants the transient integrator
    /// must land on exactly so a `C0` kink never falls *inside* a step (which would
    /// violate the smooth-LTE assumption). Empty when every source is smooth (e.g.
    /// a purely sinusoidal drive).
    /// What each unknown physically is, parallel to the unknown vector.
    pub fn unknown_kinds(&self) -> &[sane_dae::UnknownKind] {
        &self.kinds
    }

    /// The declared switching surfaces, by event index (`instance#k`).
    pub fn event_names(&self) -> &[String] {
        &self.event_names
    }

    /// The switching events of the most recent transient solve, in time order.
    pub fn last_transient_events(&self) -> Vec<TransientEvent> {
        self.last_events.lock().unwrap().clone()
    }

    pub fn transient_breakpoints(&self, p: &[f64], t0: f64, t1: f64) -> Vec<f64> {
        let mut bps = Vec::new();
        for (name, src) in &self.sources {
            let resolve = |suffix: &str| {
                self.param_index
                    .get(&format!("{name}.{suffix}"))
                    .and_then(|&i| p.get(i).copied())
            };
            src.breakpoints(resolve, t0, t1, &mut bps);
        }
        bps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Merge corners that coincide to within a hair of the span (e.g. a
        // zero-width rise makes two corners land on the same instant).
        let merge = (t1 - t0).abs() * 1e-9;
        bps.dedup_by(|a, b| (*a - *b).abs() <= merge);
        bps
    }

    /// The harmonic-balance fundamental (Hz): the tone of the first periodic source
    /// (a sine's frequency, a pulse train's `1/period`), or `None` if no source
    /// defines one.
    pub fn source_fundamental(&self, p: &[f64]) -> Option<f64> {
        self.sources.iter().find_map(|(name, src)| {
            let resolve = |suffix: &str| {
                self.param_index
                    .get(&format!("{name}.{suffix}"))
                    .and_then(|&i| p.get(i).copied())
            };
            src.fundamental(resolve)
        })
    }

    /// Transient solve of the DAE `F(x, x', t) = 0` over `t_eval` with the
    /// selected implicit-RK method (adaptive ESDIRK32; see [`TransientMethod`]).
    /// The mass matrix `C = dF/dx'` is constant (charge and flux enter linearly);
    /// every stage solve reuses the engine's device-limiting Newton machinery.
    ///
    /// `x0` is the initial state; if it does not match the system dimension a
    /// consistent DC operating point is computed and used instead. Returns the
    /// state at each time in `t_eval` (one inner vector per time point), or an
    /// error string if integration fails.
    pub fn solve_transient(
        &self,
        method: TransientMethod,
        p: &[f64],
        x0: &[f64],
        t_eval: &[f64],
        rtol: f64,
        atol: f64,
        dt_max: Option<f64>,
    ) -> Result<Vec<Vec<f64>>, String> {
        if t_eval.is_empty() {
            return Ok(Vec::new());
        }
        // The implicit-RK stage solve owns the device limiting / gmin / damping
        // the generic ODE backends lacked; see `TransientMethod`.
        self.solve_transient_irk(method, p, x0, t_eval, rtol, atol, dt_max)
    }
}
