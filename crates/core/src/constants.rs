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

/// Nominal circuit temperature [K] (27 degrees Celsius): the default operating
/// temperature seen by Verilog-A `$temperature` / `$vt` when no sweep is active.
pub const TEMP_NOMINAL_K: f64 = 300.15;

/// Zero degrees Celsius in Kelvin (the Celsius<->Kelvin offset).
pub const ZERO_CELSIUS_K: f64 = 273.15;

/// Nominal circuit temperature in Celsius (27 degC), the SPICE default `.temp`.
pub const TEMP_NOMINAL_C: f64 = 27.0;

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
/// `exp` overflow that wrecks Newton / BDF trial steps. `exp(40) ~ 2.4e17`,
/// far above any physical junction current yet far below `f64::MAX`.
pub const EXP_VCRIT: f64 = 40.0;

/// Thermal voltage kT/q at the nominal temperature (~300 K). SPICE derives this
/// from `temp`; decks never state it, so it is the default for junction `Vt`.
pub const THERMAL_VOLTAGE: f64 = 0.025852;

/// Stand-in for an infinite Early voltage (ngspice's default when `VAF`/`VAR`
/// are unspecified): large enough that `1/VAf` is numerically negligible, so the
/// base-width-modulation term vanishes and transport is ideal.
pub const EARLY_INF: f64 = 1e12;

/// Default companion conductance for homotopy continuation (see
/// [`sane_device::DeviceModel::companion`]). A ~1 S conductance star makes the
/// `lambda = 0` linear network strongly regular (it dominates typical device
/// admittances), so the first continuation point is trivial.
pub const COMPANION_G: f64 = 1.0;

// ===========================================================================
// Verilog-A frontend (lowering / preprocessor)
// ===========================================================================

/// Boltzmann constant over elementary charge (`k/q`) [V/K] as used in Verilog-A
/// `$vt` lowering. This is the historical literal baked into the lowered symbolic
/// expressions; it differs from the computed [`K_OVER_Q`] by ~1e-9 relative
/// (a handful of decimal digits), so it is kept as a distinct constant to keep
/// already-lowered models bit-identical.
pub const VERILOGA_K_OVER_Q: f64 = 8.617_333_262e-5;

/// Cap on `for`-loop unrolling iterations (runaway guard).
pub const MAX_UNROLL: usize = 100_000;

/// Cap on `while`-loop unrolling iterations. Bounded fixed-point/Newton loops in
/// compact models converge in a handful of steps; a loop with no static counter
/// bound hits this cap and errors (it cannot be lowered to a static DAG).
pub const WHILE_MAX_UNROLL: usize = 1_000;

/// Preprocessor: maximum macro-expansion / include nesting depth (recursion guard).
pub const PREPROCESSOR_MAX_DEPTH: usize = 256;

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
/// Stage-Newton tolerance (WRMS update norm) of the adjoint's fixed-grid
/// forward solve. Far tighter than the production `IRK_STAGE_TOL`: the
/// discrete adjoint assumes the stage residuals vanish EXACTLY, and whatever
/// Newton leaves behind leaks first-order error into the gradient.
pub const ADJOINT_STAGE_TOL: f64 = 1e-9;

/// Line-search step reduction factor (alpha *= this).
pub const LINE_SEARCH_SHRINK: f64 = 0.5;

/// Device limiting (`pnjlim`): thermal voltage `kT/q` used in the logarithmic
/// junction-step limiting (same nominal value as the device models' thermal
/// voltage). The limiting is path-only -- it never moves the converged operating
/// point -- so this need not track a per-device `temp`.
pub const LIMIT_VT: f64 = 0.025852;
/// Device limiting (`pnjlim`): critical forward voltage above which the junction
/// step is logarithmically limited. SPICE derives it per junction as
/// `vcrit = Vt*ln(Vt/(sqrt(2)*Is))`, ~0.6-0.8 V for `Is` in 1e-14..1e-16. Only
/// logarithmically sensitive to `Is`, and path-only, so one representative value
/// suffices instead of a per-device threshold.
pub const LIMIT_VCRIT: f64 = 0.6;
/// Device limiting (`fetlim`): representative channel threshold voltage placing
/// the bounded-step region for FET `vgs`. Path-only, like [`LIMIT_VCRIT`].
pub const LIMIT_VTO: f64 = 0.5;

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
/// gmin continuation: per-point Newton iteration budget (each point starts near
/// the predicted solution, so it should converge fast).
pub const GMIN_STEP_MAX_ITER: usize = 100;

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
/// Nodeset pin: per-phase Newton iteration budget.
pub const NODESET_MAX_ITER: usize = 100;

/// Source continuation: per-point Newton iteration budget.
pub const SOURCE_STEP_MAX_ITER: usize = 100;
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
// Transient: ESDIRK32 integration
// ===========================================================================

/// Transient (ESDIRK32) integration: relative tolerance. SPICE-like (`reltol`); a
/// tighter value forces tiny steps across stiff junction switching.
pub const TRANSIENT_RTOL: f64 = 1e-4;
/// Transient (ESDIRK32) integration: absolute tolerance. Acts as the floor near zero
/// crossings; too tight (e.g. 1e-9 on node voltages) stalls the step controller.
pub const TRANSIENT_ATOL: f64 = 1e-7;

/// Transient integration: smallest meaningful time increment as a fraction of the
/// total integration span `t_end - t_0`. Used both as the minimum step floor (the
/// step controller gives up below it) and as the end-of-span epsilon that folds a
/// hair-thin final leg into the previous one. Scale-free: a relative epsilon works
/// for spans from picoseconds to seconds.
pub const TRANSIENT_SPAN_EPS_FRAC: f64 = 1e-12;

// ===========================================================================
// Transient: ESDIRK32 implicit Runge-Kutta integrator
// ===========================================================================
// ESDIRK32: Kvaerno's ESDIRK 3/2 (A. Kvaerno, "Singly diagonally implicit
// Runge-Kutta methods with an explicit first stage", BIT 44, 2004). Four
// stages, explicit first stage, order 3 with an embedded order-2 estimate;
// BOTH the method and the embedded method are stiffly accurate (rows 4 and 3
// of `A` are `b` and `b̂`, `c_3 = c_4 = 1`) and L-stable, so the error
// estimate compares two solutions that each satisfy the algebraic constraints.
// Every abscissa lies in `[0, 1]`: no stage is evaluated beyond the step end,
// which is what lets a step land exactly on a breakpoint or a switching
// surface with all its stages in the old mode. γ is the L-stable root of
// `γ³ − 3γ² + 3γ/2 − 1/6 = 0`; the other entries are closed forms in γ
// (see `a_ij` in the solver), written out to double precision.

/// ESDIRK32 diagonal coefficient γ (the repeated SDIRK diagonal of the Butcher
/// `A`). Sets the Newton matrix shift `α = 1/(hγ)`.
pub const ESDIRK32_GAMMA: f64 = 0.435_866_521_508_459;
/// ESDIRK32 stage count (explicit first stage + 3 implicit stages).
pub const ESDIRK32_STAGES: usize = 4;
/// ESDIRK32 stage abscissae `c_i` (stage times `t + c_i·h`): `(0, 2γ, 1, 1)`.
pub const ESDIRK32_C: [f64; ESDIRK32_STAGES] = [0.0, 2.0 * ESDIRK32_GAMMA, 1.0, 1.0];
/// ESDIRK32 embedded error weights `b − b̂` (row 4 minus row 3 of `A`), used
/// by the step controller.
pub const ESDIRK32_TR: [f64; ESDIRK32_STAGES] = [
    -0.181_753_418_445_034_04,
    1.416_993_298_352_020_1,
    -1.671_106_401_415_445,
    ESDIRK32_GAMMA,
];

/// IRK inner stage Newton: per-stage iteration budget.
pub const IRK_STAGE_MAX_ITER: usize = 25;
/// IRK inner stage Newton: WRMS update tolerance for convergence.
pub const IRK_STAGE_TOL: f64 = 1e-3;
/// IRK modified-Newton: refresh the frozen Jacobian when the update fails to
/// contract by at least this factor (convergence rate θ = ‖Δₖ‖/‖Δₖ₋₁‖ too high).
/// Loose, so the frozen factorization is genuinely reused on stiff stages (a few
/// extra iterations cost less than a refactor + dF/dx eval).
pub const IRK_STALL_THETA: f64 = 0.9;

/// IRK step-size controller (fastsim/pathsim ARKODE-style): safety factor on the
/// predicted step.
pub const IRK_SAFETY_BETA: f64 = 0.9;
/// IRK step-size controller: smallest allowed per-step rescale factor.
pub const IRK_SCALE_MIN: f64 = 0.1;
/// IRK step-size controller: largest allowed per-step rescale factor.
pub const IRK_SCALE_MAX: f64 = 2.0;
/// IRK step-size controller: order of the embedded error estimate (`m + 1`,
/// `m = 2` for ESDIRK32), i.e. the exponent in the I-controller `β / err^(1/p)`.
pub const IRK_ERR_ORDER: f64 = 3.0;
/// IRK step-size controller: error-norm floor to keep the controller well-defined
/// when the embedded estimate is (near) zero.
pub const IRK_ERR_FLOOR: f64 = 1e-16;
/// IRK driver: step shrink factor applied when a stage Newton diverges (a too-large
/// step through a fast region; a smaller `h` recovers convergence).
pub const IRK_STEP_SHRINK: f64 = 0.25;
/// IRK PI controller (Gustafsson): integral exponent `kI/p` on the current
/// error. With `kP` below, the pair smooths the accepted-step sequence and cuts
/// the reject rate of the pure I-controller (Hairer & Wanner IV.2 values).
pub const IRK_PI_KI: f64 = 0.7;
/// IRK PI controller: proportional exponent `kP/p` on the previous accepted
/// error.
pub const IRK_PI_KP: f64 = 0.4;
/// Carry the stage factorization into the next step only when this step's
/// Newton needed at most this many iterations per implicit stage -- a slow
/// step means the frozen Jacobian has gone stale and the trade (saved
/// refactors vs. extra iterations) has flipped. Self-regulating: fast-moving
/// circuits refactor per step (old behavior), settled ones coast.
pub const IRK_CARRY_MAX_ITERS_PER_STAGE: usize = 3;

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

/// Transport-delay DC relaxation: damped fixed point of the history inputs
/// onto their source values (`d = y` at DC). The damping keeps marginal
/// reflection loops (|Gamma| = 1: open / shorted ideal lines) contractive.
pub const DELAY_DC_MAX_ROUNDS: usize = 60;
pub const DELAY_DC_DAMPING: f64 = 0.5;
