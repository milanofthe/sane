//! The single global constants file for all of SANE. Every tunable numeric knob,
//! physical constant and algorithmic threshold lives here -- nothing is hardcoded
//! at its use site -- so the DAE assembly, device models, solver and analyses all
//! draw the same values. Grouped by concern below. Hosted in `sane_core` because
//! it is the one crate every other crate depends on.

// ===========================================================================
// Physics (SI units)
// ===========================================================================

/// Boltzmann constant `k_B` [J/K] (CODATA exact value).
pub const BOLTZMANN: f64 = 1.380649e-23;

/// Elementary charge `q` [C] (CODATA exact value).
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;

/// Thermal voltage coefficient `k_B / q` [V/K] (~8.617e-5).
pub const K_OVER_Q: f64 = BOLTZMANN / ELEMENTARY_CHARGE;

/// Zero degrees Celsius in Kelvin (the Celsius<->Kelvin offset).
pub const ZERO_CELSIUS_K: f64 = 273.15;

/// Nominal circuit temperature in Celsius (27 degC), the SPICE default `.temp`.
pub const TEMP_NOMINAL_C: f64 = 27.0;

/// Nominal circuit temperature [K] (27 degrees Celsius): the default operating
/// temperature seen by Verilog-A `$temperature` / `$vt` when no sweep is active.
pub const TEMP_NOMINAL_K: f64 = ZERO_CELSIUS_K + TEMP_NOMINAL_C;

/// Thermal voltage `kT/q` at the nominal temperature [V] (~25.865 mV).
pub const VT_NOMINAL: f64 = K_OVER_Q * TEMP_NOMINAL_K;

/// Reserved symbol name for the global circuit temperature [K]. Verilog-A
/// `$temperature` lowers to this shared symbol (rather than a constant) so a
/// temperature sweep can vary it; defaults to [`TEMP_NOMINAL_K`]. The `$` prefix
/// cannot collide with a node or `instance.param` name.
pub const TEMP_SYMBOL: &str = "$temp";

// ===========================================================================
// Device models
// ===========================================================================

/// Critical exponent for the overflow-safe exponential. Above `arg = EXP_VCRIT`
/// the junction exponential `exp(arg)` is continued by its tangent line
/// (value and first derivative matched at the knot), bounding its growth to
/// linear. This mirrors SPICE junction voltage limiting: it leaves normal
/// operating points (`arg < EXP_VCRIT`) exactly unchanged while preventing the
/// `exp` overflow that wrecks Newton and transient trial steps. `exp(40) ~ 2.4e17`,
/// far above any physical junction current yet far below `f64::MAX`.
pub const EXP_VCRIT: f64 = 40.0;

/// Default companion conductance for homotopy continuation (see
/// [`sane_device::DeviceModel::companion`]). A ~1 S conductance star makes the
/// `lambda = 0` linear network strongly regular (it dominates typical device
/// admittances), so the first continuation point is trivial.
pub const COMPANION_G: f64 = 1.0;

// ===========================================================================
// Verilog-A frontend (lowering / preprocessor)
// ===========================================================================

/// Cap on `for`-loop unrolling iterations (runaway guard).
pub const MAX_UNROLL: usize = 100_000;

/// Iterations a Verilog-A loop whose condition does not fold to a constant
/// unrolls, each gated by its condition. A loop over a parameter (`for (i = 0;
/// i < nf; ...)`) still running after these ends there, with an assertion
/// that it does for the parameter values in use; one over the solution
/// cannot be lowered.
pub const VA_LOOP_GATED_CAP: usize = 64;

/// Preprocessor: maximum macro-expansion / include nesting depth (recursion guard).
pub const PREPROCESSOR_MAX_DEPTH: usize = 256;

/// Diagnostics: the most include and macro-expansion contexts a span is
/// walked back through to its root file.
pub const DIAG_MAX_HOPS: usize = 32;

/// `$simparam("gmin")`: the minimum conductance a model asks the simulator
/// for, the DC solver's own (see [`GMIN_DC`]).
pub const VA_SIMPARAM_GMIN: f64 = GMIN_DC;

// ===========================================================================
// DC solver: damped Newton, limiting, partitioning
// ===========================================================================

/// Backtracking line-search: number of step-halvings tried per Newton step.
pub const LINE_SEARCH_TRIES: usize = 10;
/// Newton stall guard: the residual norm must fall by at least
/// `1 - NEWTON_STALL_FACTOR` over any `NEWTON_STALL_WINDOW` consecutive
/// iterations, else the iteration is stalled and returns early. A stalled
/// Newton (a floating device-internal node swinging by a volt while the
/// residual stays put, the ua741's `Q6.ci`) used to run out its full budget in
/// every corrector of the continuation cascade; the guard turns each of those
/// failures from the budget into a window. Sized on the corpus: a damped
/// Newton opens with genuine plateaus (a three-stage BJT amplifier sits at
/// its starting residual for 17 iterations, then converges in 20 more; the
/// ua741 gains 19% over one 16-iteration stretch), and a false trip costs the
/// whole cascade, so the window is wider than any converging plateau seen --
/// every converging run drops by more than an order of magnitude within 24
/// -- while the ua741's stall (under 8% over any 24) trips at iteration 58 of
/// its 100.
pub const NEWTON_STALL_WINDOW: usize = 24;
// Composite (Traub) step: taken after every full Newton step and kept when
// it contracts the residual. Gating it on the contraction regime (the
// quadratic phase its third-order bound rests on) was measured and declined:
// on the ua741 the chord steps that matter are the ones in the long flat
// opening phase, where every Newton step gains a percent and the chord on
// the frozen factorization gains another -- gated, that circuit falls from
// 61 iterations into a 600-iteration cascade, while the circuits the ungated
// chord hurts (a three-stage BJT amplifier, a CMOS OTA) stall early under
// `NEWTON_STALL_WINDOW` and cost 26 and 41 iterations through the cascade.
/// See [`NEWTON_STALL_WINDOW`].
pub const NEWTON_STALL_FACTOR: f64 = 0.95;

/// Line-search step reduction factor (alpha *= this).
pub const LINE_SEARCH_SHRINK: f64 = 0.5;

/// Device limiting (`pnjlim`): thermal voltage `kT/q` used in the logarithmic
/// junction-step limiting, the nominal one. The limiting is path-only -- it
/// never moves the converged operating point -- so this need not track a
/// per-device `temp`.
pub const LIMIT_VT: f64 = VT_NOMINAL;
/// Device limiting (`pnjlim`): a junction step larger than this many thermal
/// voltages is limited (SPICE3 `DEVpnjlim`).
pub const PNJLIM_STEP_VT: f64 = 2.0;
/// Device limiting (`pnjlim`): critical forward voltage above which the junction
/// step is logarithmically limited. SPICE derives it per junction as
/// `vcrit = Vt*ln(Vt/(sqrt(2)*Is))`, ~0.6-0.8 V for `Is` in 1e-14..1e-16. Only
/// logarithmically sensitive to `Is`, and path-only, so one representative value
/// suffices instead of a per-device threshold.
pub const LIMIT_VCRIT: f64 = 0.6;
/// Device limiting (`fetlim`): representative channel threshold voltage placing
/// the bounded-step region for FET `vgs`. Path-only, like [`LIMIT_VCRIT`].
pub const LIMIT_VTO: f64 = 0.5;
/// Device limiting (`fetlim`, SPICE3 `DEVfetlim`), in volts about the
/// threshold: the step a gate voltage may always take, its growth with the
/// overdrive, the overdrive past which the channel counts as strongly
/// inverted, the window about the threshold a step from below stops in, and
/// the overdrive a turning-on step is held to.
pub const FETLIM_STEP: f64 = 2.0;
pub const FETLIM_STEP_GAIN: f64 = 2.0;
pub const FETLIM_STRONG: f64 = 3.5;
pub const FETLIM_NEAR: f64 = 0.5;
pub const FETLIM_TURN_ON: f64 = 4.0;

/// Newton iteration budget of one point of every DC continuation: a gmin or
/// source step, a companion-homotopy point, a nodeset pin phase. Each starts
/// near the predicted solution, so it converges fast or not at all.
pub const CONTINUATION_MAX_ITER: usize = 100;

/// Linear-block caching (Schur complement): only partition systems at least
/// this large (below it, a plain sparse LU per iteration is already cheap).
pub const PARTITION_MIN_DIM: usize = 64;
/// Linear-block caching: only partition when the nonlinear block is at most
/// `dim / this` (i.e. the linear/parasitic part genuinely dominates), so the
/// cached factorization + small Schur solve pays off.
pub const PARTITION_MAX_NONLIN_FRAC: usize = 4;

/// DC convergence criterion (per-component, SPICE-style `reltol`/`abstol`/
/// `vntol`). The engine defaults are *tight* (engine-grade, not SPICE's loose
/// `reltol = 1e-3`): the per-component test replaces the old scalar L2 residual
/// norm so currents and voltages are judged on their own scales, without
/// loosening accuracy. A `Convergence::spice()` preset carries the classic loose
/// values for SPICE-compatible (faster) runs.
/// Relative tolerance on the Newton update `|dx_i| <= reltol*|x_i| + (abs floor)`.
pub const DC_RELTOL: f64 = 1e-7;
/// Absolute current floor: residual of a KCL (node) row, and update of a
/// branch-current unknown.
pub const DC_ABSTOL: f64 = 1e-12;
/// Absolute voltage floor: residual of a KVL (branch) row, and update of a
/// node-voltage unknown.
pub const DC_VNTOL: f64 = 1e-9;
/// The classic loose SPICE convergence defaults (`reltol`, `abstol` [A],
/// `vntol` [V]).
pub const SPICE_RELTOL: f64 = 1e-3;
pub const SPICE_ABSTOL: f64 = 1e-12;
pub const SPICE_VNTOL: f64 = 1e-6;

/// Operating-point solve settled by the analysis layer (deck paths): the Newton
/// residual tolerance and iteration cap passed to `solve_dc`. Distinct from the
/// per-row `DC_RELTOL/ABSTOL/VNTOL` floors above (this is the scalar gate the
/// analysis facade uses uniformly across DC/AC/transient/noise setup).
pub const DC_OP_TOL: f64 = 1e-10;
/// Newton iteration cap for the analysis-layer operating-point solve.
pub const DC_OP_MAXIT: usize = 100;

// ===========================================================================
// DC solver: homotopy / continuation cascade
// ===========================================================================

/// gmin continuation: starting conductance at `lambda = 0`, added to every
/// diagonal. Chosen large enough to dominate typical circuit conductances (a
/// 1 ohm shunt), making the first continuation point trivial.
pub const GMIN_START: f64 = 1.0;
/// Baseline shunt conductance kept on every node in the *final* operating-point
/// solve (matching ngspice's default `gmin = 1e-12`, which is never removed).
/// Regularizes the otherwise near-singular Jacobian of very-high-gain feedback
/// circuits (op-amps) without measurably shifting the solution.
pub const GMIN_DC: f64 = 1e-12;
/// Share of an unknown's linearized current that the `GMIN_DC` shunt may carry
/// before the operating point counts as set by the regularization rather than
/// by the circuit. Above this the reported value is the shunt's answer: a node
/// whose only path to ground is gmin sits at `i/GMIN_DC`, a number that says
/// nothing about the circuit. Ten percent is well clear of the rounding noise
/// of a healthy node (a 1 kOhm divider comes out at 2.5e-10) and still catches
/// the first decade of trouble.
pub const GMIN_DOMINANCE_FRAC: f64 = 0.1;
/// gmin step-down factor for the source-stepping fallback: after the source ramp
/// converges at `SOURCE_GMIN`, the shunt is reduced toward `GMIN_DC` one decade
/// at a time (warm-started), so a high-impedance node tracks into place instead
/// of oscillating on a single jump to the floor.
pub const GMIN_RAMP_DOWN: f64 = 0.1;
/// Classic gmin stepping (cascade fallback 1): the strong starting shunt under
/// full excitation. High enough that every node is dominated by its local
/// conductance (the relaxed solve converges from anywhere and lands in the
/// powered basin of a self-biased circuit), low enough that the adaptive
/// step-down reaches the floor in a handful of decades.
pub const GMIN_STEP_START: f64 = 1e-3;
/// gmin step-down: adaptive-ratio floor. A failed level is retried at half the
/// log-step (`ratio -> sqrt(ratio)`) from the last held point; once the ratio
/// exceeds this floor the remaining level is thinner than ~1% of a decade and
/// the step-down is genuinely stuck, so it stops refining and reports the
/// tightest held point.
pub const GMIN_STEP_RATIO_FLOOR: f64 = 0.98;
/// gmin step-down: total level budget across refinements. Six decades at the
/// coarsest ratio need six levels; the budget bounds the pathological case
/// where every level converges only after repeated refinement.
pub const GMIN_STEP_MAX_LEVELS: usize = 64;

/// Pseudo-transient continuation: initial pseudo-step. The first BE step's
/// anchor shunt is `1/PTC_H0 = 1e6` mho -- far above every circuit conductance
/// (converges from any point the static cascade left behind) without being so
/// stiff that nothing moves.
pub const PTC_H0: f64 = 1e-6;
/// Pseudo-transient continuation: pseudo-step growth after an accepted step.
pub const PTC_H_GROW: f64 = 10.0;
/// Pseudo-transient continuation: pseudo-step shrink after a failed step.
pub const PTC_H_SHRINK: f64 = 0.25;
/// Pseudo-transient continuation: give up when the pseudo-step underflows this.
pub const PTC_H_MIN: f64 = 1e-15;
/// Pseudo-transient continuation: anchor considered released at this step (the
/// pin conductance `1/h` is far below every circuit conductance).
pub const PTC_H_MAX: f64 = 1e15;
/// Pseudo-transient continuation: total BE-step budget.
pub const PTC_MAX_STEPS: usize = 80;

/// Nodeset pin conductance (`.nodeset`): the stiff shunt-to-target conductance
/// used in the *first* phase of a node-set solve. A large value (a 1 ohm spring
/// to the requested voltage) dominates typical circuit conductances, so phase 1
/// converges to a point close to the node-set values -- breaking the symmetry of
/// a bistable circuit onto one branch. Phase 2 removes the pin and re-solves
/// freely, so the converged operating point is unshifted (the pin shapes only
/// which root is found). Same scale as `GMIN_START`.
pub const GSET: f64 = 1.0;
/// Nodeset pin: per-row stiffness floor factor. `GSET` is sized for MNA node
/// rows (currents vs. a 1 S spring); a pinned row of a different physical scale
/// (a behavioral `idt` state row, say) can dwarf it and leave the pin soft. The
/// pin phase therefore rescales each pinned row's spring to
/// `max(GSET, GSET_ROW_SCALE * ||J_row(x0)||_inf)`, so the spring dominates
/// that row's own couplings regardless of units.
pub const GSET_ROW_SCALE: f64 = 10.0;

/// Source stepping: a modest shunt conductance held during the ramp to
/// regularize near-singular high-impedance nodes (mirror-loaded diff pairs) and
/// damp region-boundary chatter; removed for the final `lambda = 1` polish.
pub const SOURCE_GMIN: f64 = 1e-6;

/// Predictor-corrector continuation step control (shared by source / companion
/// continuation). The homotopy parameter `lambda` runs 0 -> 1 using the *exact*
/// path tangent `dx/dlambda` (one extra solve with the already-factored Jacobian)
/// to predict each next point, then corrects with Newton -- far fewer, larger,
/// robust steps than naive fixed-factor stepping.
/// Starting continuation step in `lambda`.
pub const HOMOTOPY_DLAM0: f64 = 0.1;
/// Grow the `lambda` step by this factor after a converged corrector.
pub const HOMOTOPY_GROW: f64 = 1.6;
/// Shrink the `lambda` step by this factor when the corrector fails.
pub const HOMOTOPY_SHRINK: f64 = 0.4;
/// Largest allowed `lambda` step.
pub const HOMOTOPY_DLAM_MAX: f64 = 0.25;
/// Smallest `lambda` step before giving up.
pub const HOMOTOPY_DLAM_MIN: f64 = 1e-4;
/// Hard cap on continuation points.
pub const HOMOTOPY_MAX_STEPS: usize = 400;
/// Failed correctors at `lambda = 1` before a continuation gives up. A path
/// that holds up to just below the end and fails at the end itself is not
/// cured by smaller steps: the end point's own Newton basin is the problem,
/// and the next homotopy in the cascade gets its turn.
pub const HOMOTOPY_ENDPOINT_TRIES: usize = 2;

/// Harmonic-balance source-stepping continuation (fixed-factor `lambda` ramp of
/// the drive amplitudes). Distinct from the predictor-corrector `HOMOTOPY_*`
/// knobs above because HB ramps amplitudes with simple grow/shrink stepping.
/// Initial `lambda` step.
pub const HB_CONT_DLAM0: f64 = 0.25;
/// Grow factor after an easy converged step.
pub const HB_CONT_GROW: f64 = 1.5;
/// Shrink factor when a step fails to converge.
pub const HB_CONT_SHRINK: f64 = 0.5;
/// Largest allowed `lambda` step.
pub const HB_CONT_DLAM_MAX: f64 = 0.5;
/// Smallest `lambda` step before giving up.
pub const HB_CONT_DLAM_MIN: f64 = 1e-3;
/// Newton iteration count at/below which a step is considered "easy" (grow).
pub const HB_CONT_EASY_ITERS: usize = 3;
/// Tolerance for treating `lambda` as having reached 1.
pub const HB_CONT_LAMBDA_EPS: f64 = 1e-9;

/// Node-adaptive damping (last-resort DC corrector): under-determined internal
/// nodes (e.g. a BSIM4 gate-internal node behind the gate resistance, whose DC
/// current is ~0) have a tiny diagonal Jacobian `|J_ii|`, so their Newton step is
/// huge and oscillates -- derailing the whole solve onto a non-physical root.
/// We load each such weak diagonal up to `ADAPT_DIAG_FRAC * max_i|J_ii|` (matrix
/// only, never the residual, so the converged `F + GMIN_DC*x = 0` is unshifted),
/// damping exactly those nodes' steps while leaving well-conditioned nodes at
/// full Newton. `1e-3` boosts the weak nodes by orders of magnitude while barely
/// touching strong ones.
pub const ADAPT_DIAG_FRAC: f64 = 1e-3;
/// Node-adaptive corrector: Newton iteration budget.
pub const ADAPT_MAX_ITER: usize = 300;

// ===========================================================================
// Transient integration (Rodas4, see `sane_solve::rosenbrock`)
// ===========================================================================

/// Transient integration: relative tolerance. SPICE-like (`reltol`); a
/// tighter value forces tiny steps across stiff junction switching.
pub const TRANSIENT_RTOL: f64 = 1e-4;
/// Transient integration: absolute tolerance. Acts as the floor near zero
/// crossings; too tight (e.g. 1e-9 on node voltages) stalls the step controller.
pub const TRANSIENT_ATOL: f64 = 1e-7;

/// Transient integration: smallest meaningful time increment as a fraction of the
/// total integration span `t_end - t_0`. Used both as the minimum step floor (the
/// step controller gives up below it) and as the end-of-span epsilon that folds a
/// hair-thin final leg into the previous one. Scale-free: a relative epsilon works
/// for spans from picoseconds to seconds.
pub const TRANSIENT_SPAN_EPS_FRAC: f64 = 1e-12;

/// The backward-Euler step that makes a caller's initial state consistent,
/// as a fraction of the first step (see `solve_transient`): short enough
/// that the charges stay where the state put them.
pub const TRANSIENT_IC_STEP_FRAC: f64 = 1e-3;

/// The first step of a transient without a step cap, as a fraction of its
/// span.
pub const TRANSIENT_H0_FRAC: f64 = 1e-2;
/// Breakpoints within this fraction of the span of each other are one.
pub const BREAKPOINT_MERGE_FRAC: f64 = 1e-9;
/// A step that would end short of a breakpoint by less than this many times
/// itself splits the rest into two halves instead (no sliver before it).
pub const BREAKPOINT_SPLIT: f64 = 1.9;
/// A gmin-held operating point is settled by a transient over this span [s]
/// before it is solved again.
pub const GMIN_SETTLE_SPAN: f64 = 0.06;
/// The stage Newton's tolerance floor in multiples of the machine epsilon
/// over `rtol` (Hairer and Wanner's `fnewt`).
pub const STAGE_TOL_FLOOR_EPS: f64 = 10.0;

/// Implicit Euler stage Newton (the consistent start and restarts): the
/// iteration budget.
pub const STAGE_MAX_ITER: usize = 25;
/// Implicit Euler stage Newton: the update tolerance, as a fraction of the
/// integration tolerance (`wn` is scaled by it), is Hairer-Wanner's (RADAU5
/// `fnewt`) `max(10 eps / rtol, min(STAGE_TOL_MAX, sqrt(rtol)))`: a
/// Newton error far below the local error buys nothing, and a stage solved
/// to the rounding of its charges cannot get below it.
pub const STAGE_TOL_MAX: f64 = 0.03;
/// Newton (DC and transient stage): a system whose every residual row is
/// within this many rounding units of the magnitudes it is computed from
/// (`|J| |x|`, and a stage's charge terms) is solved to machine precision,
/// whatever the update: rounding in the rows becomes a visible, meaningless
/// update along a weakly determined direction (a node held by gmin alone, a
/// charge difference `Q(X) - Q(xn)` of a large capacitor rounding at
/// `ulp(Q)`).
pub const NEWTON_ROUNDOFF: f64 = 64.0;
/// Implicit Euler stage modified Newton: refresh the frozen Jacobian when the
/// update fails to contract by at least this factor (convergence rate
/// θ = ‖Δₖ‖/‖Δₖ₋₁‖ too high).
pub const STAGE_STALL_THETA: f64 = 0.9;

/// Step-size controller: safety factor on the predicted step.
pub const STEP_SAFETY: f64 = 0.9;
/// Step-size controller: smallest allowed per-step rescale factor.
pub const STEP_SCALE_MIN: f64 = 0.1;
/// Step-size controller: largest allowed per-step rescale factor.
pub const STEP_SCALE_MAX: f64 = 2.0;
/// Step-size controller: error-norm floor to keep the controller well-defined
/// when the embedded estimate is (near) zero.
pub const STEP_ERR_FLOOR: f64 = 1e-16;
/// Step shrink factor applied when a step fails (a stage leaves the finite
/// numbers, a singular stage matrix).
pub const STEP_SHRINK: f64 = 0.25;
/// PI controller (Gustafsson): integral exponent `kI/p` on the current
/// error. With `kP` below, the pair smooths the accepted-step sequence and cuts
/// the reject rate of the pure I-controller (Hairer & Wanner IV.2 values).
pub const STEP_PI_KI: f64 = 0.7;
/// PI controller: proportional exponent `kP/p` on the previous accepted
/// error.
pub const STEP_PI_KP: f64 = 0.4;

/// Harmonic balance: a Newton step is retracted (backtracking) only when it
/// grew the residual max-norm by more than this factor. Newton is not
/// monotone near a solution of the harmonic residual (the max-norm over
/// harmonics can rise on a step that still converges), so a strict decrease
/// test stalls the continuation; this gate stops the divergence the undamped
/// iteration showed at large drive without touching the ordinary steps.
pub const HB_BACKTRACK_GROWTH: f64 = 10.0;

// ===========================================================================
// Transient events (switching surfaces)
// ===========================================================================

/// Event location: the crossing time is resolved to this fraction of the
/// bracketing step on the step's dense output.
pub const EVENT_TIME_TOL: f64 = 1e-6;
/// Event location: iteration cap of the regula-falsi root search per surface.
pub const EVENT_LOCATE_MAX_ITER: usize = 40;
/// Event location: how many times one step may be retaken toward a located
/// crossing before the candidate is accepted with the crossing inside.
pub const EVENT_RESTEP_MAX: usize = 4;
/// Step after an event: the pre-event step size times this (the post-event
/// mode is a fresh problem, like a source breakpoint).
pub const EVENT_RESTART_SHRINK: f64 = 0.25;
/// A landed-on surface re-arms once `|g|` exceeds this fraction of the largest
/// `|g|` seen (the trajectory has moved off the surface without crossing).
pub const EVENT_REARM_FRAC: f64 = 1e-3;
/// Consistent restart after a discontinuity: the implicit-Euler
/// reinitialisation step is this fraction of the pre-landing step -- long
/// enough to be past a surface located to `EVENT_TIME_TOL`, short enough that
/// the differential states do not move.
pub const EVENT_REINIT_FRAC: f64 = 4.0 * EVENT_TIME_TOL;

// ===========================================================================
// Analysis: pole/zero pencil tolerances
// ===========================================================================

/// Generalized-eigenvalue (pencil) infinity tolerance: a reciprocal eigenvalue
/// `mu` below this magnitude is treated as an infinite pole (the singular
/// reactive matrix `C` has a large null space, so most modes are algebraic, not
/// dynamic). Also the floor on `|mu|` in the standard reduction.
pub const PENCIL_INF_TOL: f64 = 1e-9;
/// Discard finite roots larger than this magnitude (numerical infinities).
pub const PENCIL_ROOT_MAX: f64 = 1e15;
/// A matrix's balancing (see `pz::balance`) rescales a row and column
/// only where it shrinks their norms below this fraction (LAPACK's `gebal`
/// takes 0.95).
pub const BALANCE_GAIN: f64 = 0.95;
/// The moves, as multiples of the larger of the shift and the pencil's own
/// rate `|M| / |N|`, a pencil's standard reduction makes off a shift the
/// pencil is singular at (a transfer that vanishes at DC has a zero at
/// `s = 0`): incommensurate, so a root at one is not at the next.
pub const PENCIL_SHIFTS: [f64; 3] = [1.0, -std::f64::consts::SQRT_2, std::f64::consts::E];
/// The most reductions (bands, and moves off a singular shift) a pencil's
/// roots are found in. Each band resolves about `1 / PENCIL_INF_TOL` times
/// the distance to the root nearest its shift, so a few cover every decade
/// up to [`PENCIL_ROOT_MAX`].
pub const PENCIL_BANDS_MAX: usize = 12;

/// Transport-delay DC relaxation: damped fixed point of the history inputs
/// onto their source values (`d = y` at DC). The damping keeps marginal
/// reflection loops (|Gamma| = 1: open / shorted ideal lines) contractive.
pub const DELAY_DC_MAX_ROUNDS: usize = 60;
/// Delay DC relaxation: the fraction of the step toward the delayed signals'
/// new values each round takes.
pub const DELAY_DC_DAMPING: f64 = 0.5;
/// Delay DC relaxation: the floor of its convergence test, relative.
pub const DELAY_DC_TOL_FLOOR: f64 = 1e-12;

// ===========================================================================
// Delay history
// ===========================================================================

/// Knots a history holds before it grows, by default.
pub const DELAY_HISTORY_KNOTS: usize = 64;
/// Knots a transient's history is given past what its step cap leaves in the
/// horizon: the short steps after each restart.
pub const DELAY_KNOT_SLACK: usize = 64;
/// The smallest ring a delayed signal's knots are kept in.
pub const DELAY_RING_MIN: usize = 8;
/// Knots a query walks from its cursor before it bisects instead.
pub const DELAY_PROBE_MAX: usize = 4;

// ===========================================================================
// Harmonic balance defaults
// ===========================================================================

/// Harmonics of the fundamental kept.
pub const HB_HARMONICS: usize = 8;
/// Time samples per harmonic of the transform (aliasing guard).
pub const HB_OVERSAMPLE: usize = 16;
/// Newton tolerance on the harmonic residual.
pub const HB_TOL: f64 = 1e-10;
/// Newton iteration budget.
pub const HB_MAX_ITER: usize = 60;
/// Time samples the transform evaluates together, one lane each.
pub const HB_LANES: usize = 4;

// ===========================================================================
// Small-signal analyses
// ===========================================================================

/// Frequencies per block of an AC sweep: one pivoting factorization each, a
/// unit of work for one worker.
pub const AC_BLOCK: usize = 64;
/// Two roots pair as complex conjugates when they agree to this, relative.
pub const PZ_CONJ_TOL: f64 = 1e-6;
/// The floor a magnitude is held to before its logarithm in decibels.
pub const DB_MAG_FLOOR: f64 = 1e-30;
/// The floor a divisor or pivot magnitude is held to: a guard against an
/// exact zero, far below any value of meaning.
pub const DIV_FLOOR: f64 = 1e-300;
/// Pruning: the relative contribution below which a branch is negligible.
pub const PRUNE_REL_TOL: f64 = 1e-3;
/// Pruning: the default frequency grid (Hz), from, to, points per decade.
pub const PRUNE_F_MIN: f64 = 1.0;
pub const PRUNE_F_MAX: f64 = 1e9;
pub const PRUNE_F_PER_DECADE: usize = 6;
/// The reference impedance of a port or a transmission line that does not
/// give one [Ohm].
pub const DEFAULT_Z0: f64 = 50.0;

// ===========================================================================
// Vector fitting (rational macromodels)
// ===========================================================================

/// Relative fit error the order search stops at.
pub const VF_TOL: f64 = 1e-4;
/// The most poles the order search tries.
pub const VF_MAX_POLES: usize = 40;
/// Passivity: the violation below which a fit counts as passive.
pub const VF_PASSIVITY_THRESHOLD: f64 = 1e-2;
/// Passivity enforcement: iteration budget.
pub const VF_PASSIVITY_ITERS: usize = 20;

// ===========================================================================
// Netlist and element defaults
// ===========================================================================

/// The delay of a lossless line that does not give one [s].
pub const TLINE_TD: f64 = 1e-9;
/// Segments a distributed line is discretized into, by default and at most.
pub const TLINE_SEGMENTS: usize = 16;
pub const TLINE_SEGMENTS_MAX: usize = 4096;
/// A distributed RC line's total series resistance and shunt capacitance
/// when the deck does not give them.
pub const URC_R: f64 = 1.0;
pub const URC_C: f64 = 1e-12;
/// The series resistance standing in for a line segment's zero inductance.
pub const LADDER_PLACEHOLDER_R: f64 = 1e-9;
/// MOSFET levels at most this are the native square law; above, a compact
/// model (the levels are read as numbers, the cut sits between 3 and 4).
pub const MOS_NATIVE_MAX_LEVEL: f64 = 3.5;
/// Nesting depth of subcircuit instances and of includes (recursion guard).
pub const SUBCKT_MAX_DEPTH: u32 = 64;
pub const INCLUDE_MAX_DEPTH: usize = 64;
/// A number this large or larger prints in exponent form rather than as an
/// integer.
pub const FMT_INT_MAX: f64 = 1e15;
/// The width and the period of a pulse that does not give them: longer than
/// any run, finite so the lowered waveform computes with it [s].
pub const PULSE_FOREVER: f64 = 1e30;
/// A switch's default on and off resistances and its hysteresis window.
pub const SWITCH_RON: f64 = 1.0;
pub const SWITCH_ROFF: f64 = 1e6;
pub const SWITCH_IH: f64 = 1e-6;

// ===========================================================================
// Sparse solves and threads
// ===========================================================================

/// Factorization work per factor entry from which the supernodal paths take
/// over from KLU (see `sane_solve::sparse`).
pub const SUPERNODAL_FLOPS_PER_ENTRY: f64 = 25.0;
/// Worker-thread count when `SANE_THREADS` is not set.
pub const DEFAULT_THREADS: usize = 4;

// ===========================================================================
// Progress reporting
// ===========================================================================

/// The shortest interval between two progress lines [s], the progress a new
/// line waits for, the bar's width in characters and the smoothing of the
/// rate estimate.
pub const PROGRESS_MIN_INTERVAL: f64 = 1.0;
pub const PROGRESS_LOG_EVERY: f64 = 0.2;
pub const PROGRESS_BAR_WIDTH: usize = 20;
pub const PROGRESS_EMA_ALPHA: f64 = 0.3;
/// The progress from which a time to completion is estimated.
pub const PROGRESS_ETA_MIN: f64 = 0.01;
/// A rate below this many iterations per second is shown per minute.
pub const PROGRESS_PER_MINUTE_BELOW: f64 = 0.1;

// ===========================================================================
// Web host (UI defaults)
// ===========================================================================

/// The AC sweep's default end frequency [Hz] and points, the transient's
/// output points.
pub const UI_AC_F_MAX: f64 = 1e8;
pub const UI_AC_POINTS: usize = 200;
pub const UI_TRAN_POINTS: usize = 400;
/// The graph view takes at most this many expression nodes (it lays out
/// only what is expanded, a big circuit starting collapsed).
pub const UI_GRAPH_MAX_NODES: usize = 200_000;
