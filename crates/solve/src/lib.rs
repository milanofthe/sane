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

use std::sync::{Arc, Mutex, MutexGuard};

use crate::limiting::Limit;
use rsdag::{Crossing, SymbolId, Tape};
use sane_core::stimulus::Stimulus;
use sane_core::Graph;
use sane_dae::Dae;

pub mod hb;
pub use delay::DelayHistory;
// Device controlling-voltage limiting (pnjlim/fetlim) is an internal solver
// step with no external consumers.
pub(crate) mod limiting;
// Construction of a `CompiledDc` (Jacobian extraction, tape compilation).
mod compile;
// The DC operating-point solve and its continuation cascade.
mod dc;
mod delay;
mod events;
mod index2;
mod newton;
mod program;
mod rosenbrock;
mod stage;
mod stage_matrix;
// The system at a point: currents, charges, Jacobians, small-signal matrices.
mod system;
// The hot-tape backend ladder (interpreter / specialized / native).
mod eval;
pub mod parallel;
// Schur-complement partitioning for large mostly-linear systems.
mod schur;
// Adjoint sensitivities and the second-order-adjoint Hessian.
mod sens;
pub(crate) mod sparse;
pub use sparse::{
    dump_system, factor_triplets, ComplexFactor, ComplexPattern, Csc, DumpValue, TripletLu,
    Triplets,
};
#[cfg(test)]
mod tests;
mod transient;
mod transient_sens;

pub use eval::Evaluator;
pub(crate) use eval::{Bound, StepEval};
use schur::LinCache;
pub use sens::NoiseAt;

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
    /// Curve-aware per-device junction / channel limiting (`pnjlim` /
    /// `fetlim`) in every Newton where the devices declare limits: the step is
    /// shortened as a whole to the largest fraction that keeps every limited
    /// junction within its bound (see `limiting::apply`), which is what takes
    /// a BJT mirror from 719 iterations through the cascade to 7 plain ones.
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
    /// symmetry-breaking pin phase. Gates the *data* in `self.dc.nodeset` exactly as
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
            device_limiting: true,
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
    /// The classic loose SPICE defaults (see [`SPICE_RELTOL`]): faster,
    /// SPICE-compatible, less accurate.
    #[cfg(test)]
    pub(crate) fn spice() -> Self {
        Self {
            reltol: SPICE_RELTOL,
            abstol: SPICE_ABSTOL,
            vntol: SPICE_VNTOL,
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
/// The thread count for the parallel work (the sweeps, the harmonic-balance
/// sampling, and in every solve the device instances of each evaluation):
/// `0` restores the default, `n` asks for `n` threads. Takes effect before
/// the pools are first used (see [`parallel`]).
pub fn set_parallelism(threads: usize) {
    sane_core::update_config(|c| c.threads = if threads == 0 { None } else { Some(threads) });
}

/// Where an input slot of the tape draws its value from.
enum InputSrc {
    X(usize),
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

/// Integration counters (logged at debug level): steps taken / rejected,
/// the system's evaluations (a Rosenbrock stage's, a restart's Newton
/// iteration's), and the stage matrix's factorizations.
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

/// A transient run: the state at every requested time, and the switching
/// events it crossed, in time order.
#[derive(Clone, Debug, Default)]
pub struct TransientRun {
    pub rows: Vec<Vec<f64>>,
    pub events: Vec<TransientEvent>,
}

/// A DAE compiled for repeated numeric evaluation/solve.
pub struct CompiledDc {
    /// The graph the programs compile from and the DAE they compile: what
    /// is built on first use (see [`sens::SensCache`]) builds over them.
    graph: Arc<Mutex<Graph>>,
    dae: Arc<Dae>,
    n: usize,
    nnz_x: usize,
    /// Per unknown, whether it is index-2 (see [`index2`]): a rate of the
    /// others, outside the transient's error control.
    index2: Vec<bool>,
    jx_rows: Vec<usize>,
    jx_cols: Vec<usize>,
    /// Position of each unknown's diagonal within the jacobian-x value array
    /// (`None` if structurally absent). Fixed by the sparsity pattern, so it is
    /// precomputed once here instead of rescanning the nonzeros every
    /// node-adaptive corrector call.
    diag_idx: Vec<Option<usize>>,
    jxd_rows: Vec<usize>,
    jxd_cols: Vec<usize>,
    /// The DC programs: the currents `I` (cheap line-search evaluations) and
    /// `I ++ G` (a Newton step), `G = dI/dx` in the jacobian-x pattern. A
    /// device body's charge model is not in them.
    tape_step_dc: StepEval,
    tape_res_dc: StepEval,
    /// The transient's (and harmonic balance's) programs: `I ++ Q` (the
    /// currents and the charges), and `I ++ Q ++ G ++ C` with `C = dQ/dx` in
    /// the jacobian-x' pattern.
    tape_tran_res: StepEval,
    tape_tran_step: StepEval,
    /// `C` alone (the state rates `x' = C⁻¹ f`).
    tape_c: StepEval,
    /// The rows' explicit time rates `dI/dt ++ dQ/dt` at the rows `dt_rows`
    /// (a charge's row offset by `n`), followed by `dI/dhist ++ dQ/dhist` at
    /// the `(row, delay)` of `dt_hist`; `None` without a time-dependent
    /// source or a delay.
    tape_dt: Option<StepEval>,
    dt_rows: Vec<usize>,
    dt_hist: Vec<(usize, usize)>,
    input_src: Vec<InputSrc>,
    /// Precomputed (buffer slot, source index) lists for [`Self::patch_inputs`].
    input_x_slots: Vec<(u32, u32)>,
    input_t_slots: Vec<u32>,
    input_hist_slots: Vec<(u32, u32)>,
    param_syms: Vec<SymbolId>,
    /// Their names, in column order.
    param_names: Vec<String>,
    /// Transport delays: what each delays (whose history the transient loop
    /// records and queries; the delay's output reads it through the tapes'
    /// history inputs), plus the taus compiled over the parameter inputs
    /// (evaluated once per solve).
    delay_src: crate::delay::DelaySources,
    tape_tau: Option<Tape>,
    /// The switching surfaces as one tape over the step inputs (`None` when
    /// the DAE declares no events), their directions and names, and the
    /// events of the most recent transient (interior mutability like the
    /// gmin flags: the solve API is `&self`).
    tape_event: Option<Tape>,
    event_dirs: Vec<Crossing>,
    event_names: Vec<String>,
    /// Per-param flag: is this the DC value of an independent source (a `V`/`I`
    /// element value, no `.` in the name)? Used by source stepping to ramp only
    /// the excitation, leaving component values fixed.
    source_mask: Vec<bool>,
    /// Per jx nonzero (aligned with `jx_rows`/`jx_cols`): does the entry vary
    /// with an unknown (a nonlinear/device entry) or is it constant (a linear,
    /// LTI element)? Computed once from the symbolic Jacobian. The DC Schur
    /// partition and harmonic balance both read it -- in HB a constant entry is
    /// frequency-diagonal `(G + jkw0*C)`, a variable one couples harmonics.
    jx_var: Vec<bool>,
    jxd_var: Vec<bool>,
    /// Base input symbols (x, params, t, delay histories), kept so the
    /// programs built on first use take the same inputs.
    base_inputs: Vec<SymbolId>,
    /// Which of them are parameter-pure (solve-constant): the prolog split
    /// of the lazily built programs.
    base_pure: Vec<bool>,
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
    /// The modular numerical-trick set this model honors across all analyses (DC
    /// cascade shaping, the linear-solver and transient aids). [`solve_dc`] uses it
    /// as the default trick set; the transient / stage factorization read it
    /// directly. Configure with [`set_tricks`](CompiledDc::set_tricks); override
    /// per call with [`solve_dc_with`](CompiledDc::solve_dc_with).
    tricks: SolverTricks,
    /// The sources' waveforms over the parameter columns, for the transient
    /// breakpoints and the HB fundamental.
    stimuli: Vec<Stimulus<Option<usize>>>,
    /// What the DC operating-point solve keeps.
    dc: dc::DcState,
    /// What the sensitivities build on first use.
    sens: sens::SensCache,
}
impl CompiledDc {
    pub fn dim(&self) -> usize {
        self.n
    }
    pub fn nnz(&self) -> usize {
        self.nnz_x
    }
    /// Number of parameters (columns of `dI/dp` and `dQ/dp`), in `param_names` order.
    pub(crate) fn param_count(&self) -> usize {
        self.param_syms.len()
    }
    /// Parameter symbols, in column order.
    pub fn param_syms(&self) -> &[SymbolId] {
        &self.param_syms
    }
    /// Parameter names, in column order.
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// The DAE this compiles.
    pub fn dae(&self) -> &Dae {
        &self.dae
    }

    /// The graph the DAE lives in, shared.
    pub fn context(&self) -> &Arc<Mutex<Graph>> {
        &self.graph
    }

    /// The graph, locked. The compiled system takes the lock itself when it
    /// builds a program on first use: hold none across its calls.
    pub fn graph(&self) -> MutexGuard<'_, Graph> {
        self.graph.lock().unwrap()
    }

    fn fill_inputs(&self, x: &[f64], p: &[f64], t: f64, inputs: &mut Vec<f64>) {
        inputs.clear();
        for src in &self.input_src {
            inputs.push(match *src {
                InputSrc::X(i) => x.get(i).copied().unwrap_or(0.0),
                InputSrc::P(j) => p.get(j).copied().unwrap_or(0.0),
                InputSrc::T => t,
                InputSrc::Hist(k) => crate::delay::hist_value(k),
            });
        }
    }

    /// Patch only the state-dependent entries (x / t / history) of an
    /// input buffer a prior [`fill_inputs`](Self::fill_inputs) prepared. The
    /// parameter entries -- the bulk of a compact-model input vector (PSP103:
    /// ~800 of ~900) -- are constant over an integration, so the inner Newton
    /// loops skip re-copying them on every evaluation. A program's own
    /// inputs after the system's (a pull's weights) stay as they are.
    fn patch_inputs(&self, x: &[f64], t: f64, inputs: &mut [f64]) {
        debug_assert!(inputs.len() >= self.input_src.len(), "buffer not prepared");
        for &(dst, i) in &self.input_x_slots {
            inputs[dst as usize] = x.get(i as usize).copied().unwrap_or(0.0);
        }
        for &dst in &self.input_t_slots {
            inputs[dst as usize] = t;
        }
        for &(dst, k) in &self.input_hist_slots {
            inputs[dst as usize] = crate::delay::hist_value(k as usize);
        }
    }

    /// Reusable symbolic factorization for the **implicit-RK stage** linear system
    /// `J_stage = G + α·C`. The pattern is the union of
    /// the `G` and `C` nonzeros plus the full diagonal (so a `gmin` shunt is
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

    /// Are transport delays present? Analyses without delay support guard on
    /// this and report a clear error instead of silently mis-solving.
    pub fn has_delays(&self) -> bool {
        self.delay_src.len() > 0
    }

    /// The number of transport delays.
    pub(crate) fn delay_count(&self) -> usize {
        self.delay_src.len()
    }

    /// The delayed signals at `(x, t)`, into `out`, aligned with
    /// [`delay_taus`](Self::delay_taus).
    pub(crate) fn delay_values(&self, x: &[f64], p: &[f64], t: f64, out: &mut Vec<f64>) {
        out.clear();
        match &self.delay_src {
            crate::delay::DelaySources::Unknowns(ix) => {
                out.extend(ix.iter().map(|&i| x.get(i).copied().unwrap_or(0.0)))
            }
            crate::delay::DelaySources::Exprs { n, tape, .. } => {
                let (mut inputs, mut work) = (Vec::new(), Vec::new());
                self.fill_inputs(x, p, t, &mut inputs);
                tape.eval(&inputs, &mut work, out);
                out.truncate(*n);
            }
        }
    }

    /// The rates of the delayed signals as the state moves at `xdot`, into
    /// `out`.
    pub(crate) fn delay_rates(
        &self,
        x: &[f64],
        xdot: &[f64],
        p: &[f64],
        t: f64,
        out: &mut Vec<f64>,
    ) {
        out.clear();
        match &self.delay_src {
            crate::delay::DelaySources::Unknowns(ix) => {
                out.extend(ix.iter().map(|&i| xdot.get(i).copied().unwrap_or(0.0)))
            }
            crate::delay::DelaySources::Exprs { n, tape, jac } => {
                let (mut inputs, mut work, mut vals) = (Vec::new(), Vec::new(), Vec::new());
                self.fill_inputs(x, p, t, &mut inputs);
                tape.eval(&inputs, &mut work, &mut vals);
                out.extend_from_slice(&vals[*n..2 * *n]);
                for (&(k, j), &g) in jac.iter().zip(&vals[2 * *n..]) {
                    out[k] += g * xdot.get(j).copied().unwrap_or(0.0);
                }
            }
        }
    }

    /// The gradient of the delayed signals w.r.t. the unknowns at `(x, t)`:
    /// `(delay, unknown, value)` triplets.
    pub fn delay_source_jac(&self, x: &[f64], p: &[f64], t: f64) -> Vec<(usize, usize, f64)> {
        match &self.delay_src {
            crate::delay::DelaySources::Unknowns(ix) => {
                ix.iter().enumerate().map(|(k, &i)| (k, i, 1.0)).collect()
            }
            crate::delay::DelaySources::Exprs { n, tape, jac } => {
                let (mut inputs, mut work, mut vals) = (Vec::new(), Vec::new(), Vec::new());
                self.fill_inputs(x, p, t, &mut inputs);
                tape.eval(&inputs, &mut work, &mut vals);
                (jac.iter().zip(&vals[2 * *n..]))
                    .map(|(&(k, j), &g)| (k, j, g))
                    .collect()
            }
        }
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
        self.dc.nodeset = nodeset;
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

    /// What each unknown physically is, parallel to the unknown vector.
    pub fn unknown_kinds(&self) -> &[sane_dae::UnknownKind] {
        &self.kinds
    }

    /// The declared switching surfaces, by event index (`instance#k`).
    pub fn event_names(&self) -> &[String] {
        &self.event_names
    }

    /// All independent-source waveform discontinuities in `(t0, t1]`, sorted and
    /// near-coincident corners merged -- the instants the transient integrator
    /// must land on exactly so a `C0` kink never falls *inside* a step (which would
    /// violate the smooth-LTE assumption). Empty when every source is smooth (e.g.
    /// a purely sinusoidal drive).
    pub fn transient_breakpoints(&self, p: &[f64], t0: f64, t1: f64) -> Vec<f64> {
        let value = |c: &Option<usize>| c.and_then(|j| p.get(j).copied());
        sane_core::stimulus::breakpoints(&self.stimuli, value, t0, t1)
    }

    /// The harmonic-balance fundamental (Hz): the tone of the first periodic source
    /// (a sine's frequency, a pulse train's `1/period`), or `None` if no source
    /// defines one.
    pub fn source_fundamental(&self, p: &[f64]) -> Option<f64> {
        let value = |c: &Option<usize>| c.and_then(|j| p.get(j).copied());
        self.stimuli.iter().find_map(|s| s.fundamental(value))
    }

    /// Transient solve of the DAE `I(x, t) + d/dt Q(x) = 0` over `t_eval` by
    /// adaptive Rodas4 (see [`transient`]). It integrates the charges
    /// `dQ/dt = -I(x, t)` (every row of the DAE reads `I + d/dt Q`), so a
    /// nonlinear capacitance is exact at every stage.
    ///
    /// `x0` is the initial state; if it does not match the system dimension a
    /// consistent DC operating point is computed and used instead. Returns the
    /// state at each time in `t_eval` (one inner vector per time point) and
    /// the switching events crossed, or an error string if integration fails.
    pub fn solve_transient(
        &self,
        p: &[f64],
        x0: &[f64],
        t_eval: &[f64],
        rtol: f64,
        atol: f64,
        dt_max: Option<f64>,
    ) -> Result<TransientRun, String> {
        crate::parallel::solve(|| {
            self.integrate(p, x0, t_eval, rtol, atol, dt_max, None)
                .map(|(run, _)| run)
        })
    }

    /// [`solve_transient`](Self::solve_transient) with the forward
    /// sensitivities of the unknowns `outputs` by the parameter columns
    /// `cols` (see [`transient_sens`]): per time, `outputs × cols` values,
    /// `d x[outputs[o]] / d p[cols[j]]` at `o * cols.len() + j`. From an
    /// operating point (`x0` empty, or the point itself with `dc_start`) the
    /// sensitivities start at the point's, from a given state at zero.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_transient_sensitivity(
        &self,
        p: &[f64],
        x0: &[f64],
        dc_start: bool,
        t_eval: &[f64],
        rtol: f64,
        atol: f64,
        dt_max: Option<f64>,
        cols: &[usize],
        outputs: &[usize],
    ) -> Result<(TransientRun, Vec<Vec<f64>>), String> {
        let req = transient::SensRequest {
            cols,
            outputs,
            dc_start,
        };
        crate::parallel::solve(|| {
            self.integrate(p, x0, t_eval, rtol, atol, dt_max, Some(&req))
                .map(|(run, sens)| (run, sens.unwrap_or_default()))
        })
    }
}
