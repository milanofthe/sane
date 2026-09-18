<p align="center">
  <img src="assets/sane-logo.png" alt="SANE" width="400">
</p>

# SANE

**Symbolic Analog Network Engine** — symbolic and numeric circuit analysis. A
Rust core (hash-consed symbolic DAG, autodiff, native sparse solver) with a
Python binding. No C backend.

SANE extracts the differential-algebraic system `F(x, x', t) = 0` from a circuit
and analyzes it: DC operating point, transient, small-signal AC, poles/zeros,
noise, harmonic balance, and exact parameter sensitivity (first and second order)
for each of those — all by automatic differentiation of one symbolic DAG.
Inspired by Analog Insydes (Fraunhofer ITWM).

## Installation

Built from source with [maturin](https://github.com/PyO3/maturin); not on PyPI.

```
maturin develop --release -m crates/py/Cargo.toml          # builds the `sane` module
```

The native backend for the hot tapes is on by default; add
`--no-default-features` for an interpreter-only build, or keep it and disable
it per run with `SANE_JIT=0`.

Rust embedding (no Python): `cargo add sane-analysis` as a path/git dependency
pulls in the whole engine.

## Quick start

```python
import numpy as np
import sane

model = sane.Circuit.parse("""
    V1 in 0 5
    R1 in out 1k
    C1 out 0 1u
""").extract()                        # or sane.Model.from_netlist(...)

op  = model.operating_point()         # DC bias, labeled by node
v   = op["out"]                       # 5.0

ss  = model.small_signal("V1", "out") # linearize at the bias
ss.poles()                            # [-1000.+0.j]  (the RC pole)
mag_db, phase_deg = ss.bode(np.logspace(1, 5, 50))

traj = model.transient(np.linspace(0, 5e-3, 200))
traj["out"]                           # time series at node "out"

sens = op.sensitivity("out")          # exact dy/dp for every parameter (one adjoint solve)
sens.ranked()                         # [(param, rel. sensitivity), ...] most important first
```

Results are labeled by node and parameter name — no positional vectors. A result
object keeps its solved state, so derived analyses (`op.sensitivity(...)`) need
no re-solve. Parameters are read/written hierarchically: `model.X1.R2 = 1e3`,
`model.set("X1.R2", 1e3)`, `model.get("X1.R2")`.

## Netlist format

The SPICE frontend parses comments, continuation lines, `.param` with
`{expressions}`, `.subckt`/`X` hierarchy, `.model` with parameter aliasing, and
HSPICE single-quoted geometry expressions. Source waveforms: `DC`, `AC`, `SIN`,
`PULSE`, `EXP`, `PWL`.

| Directive | Effect |
|---|---|
| `.param name=expr` | named parameter, usable in `{expr}` |
| `.model name type(...)` | model card; `type` is a built-in or a bound Verilog-A module |
| `.subckt` / `X` | subcircuit definition / instance (flattened) |
| `.veriloga "file.va"` or inline `.veriloga ... .endveriloga` | load a Verilog-A module |
| `.model_alias level=N module` | bind a SPICE `level` to a Verilog-A module |
| `.temp` / `.option scale=...` | global temperature [°C] / drawn-to-meter scale |
| `.ic` / `.nodeset` | initial condition / DC symmetry-break hint |

### Devices

Every model is symbolic constitutive equations on the shared DAG, so Jacobians,
sensitivities, AC, harmonic balance, temperature and noise all come from the same
autodiff machinery.

| Netlist | Device | Model |
|---|---|---|
| `R` `C` `L` | resistor / capacitor / inductor | ideal linear; `R` carries `4kT/R` thermal noise |
| `K` | mutual inductance | coupling between two named inductors |
| `V` `I` | independent source | DC / AC plus `SIN` / `PULSE` / `EXP` / `PWL` waveforms |
| `E` `G` `F` `H` | controlled source | VCVS / VCCS / CCCS / CCVS |
| `B` | behavioral source | `V=` / `I=` expression: arithmetic, comparisons, `if(c,t,e)`, and `sin cos tan exp ln log sqrt sinh cosh tanh atan floor abs pow min max` (unknown function = parse error) |
| `S` `W` | controlled switch | smooth log-conductance window (`Vh`/`Ih`), hard threshold at `Vh=0`/`Ih=0`; the window edges are switching events the transient lands on |
| `D` | diode | exponential `Is(T)`, series `Rs`, junction + diffusion charge (`Cj0`/`TT`), reverse breakdown (`BV`), shot + flicker noise |
| `M` | MOSFET | level-1 square law + optional level-2/3 (body effect, DIBL, mobility), subthreshold, overlap/channel/bulk charge, thermal + flicker noise. `level>3` routes to a Verilog-A module via `.model_alias` |
| `Q` | BJT | Gummel-Poon: Early, high injection, leakage, series `Rb`/`Rc`/`Re`, junction + diffusion charge, shot + flicker noise |
| `J` | JFET | Shichman-Hodges; gate junctions + depletion charge |
| `Z` | MESFET | Statz / Raytheon (ngspice-compatible); gate junctions + depletion charge |
| `T` `O` `U` | transmission line | `T` exact lossless (Branin delay line); with `N=` (and for `O`/`U`) an N-segment lumped RLGC ladder |
| `N` | Verilog-A | compact models (BSIM / VBIC / PSP / EKV / HICUM / ...) compiled natively onto the DAG; parameter ranges enforced |

The nonlinear built-ins (`S`/`D`/`M`/`Q`/`J`/`Z`/`T` and the ideal transformer)
are themselves Verilog-A modules (`crates/veriloga/builtin/*.va`) and lower
through the same frontend as user compact models -- one device path, with the
template cache, instance batching, `$limit` collection and noise extraction
shared.

`$temp` [K] is a first-class symbolic global: thermal voltage `kT/q`, `Is(T)`
(with `Eg`/`XTI`), and mobility `~(T/Tnom)^-1.5` track it for `D`/`Q`/`M`/`J`/`Z`,
so `.temp` sweeps are physical and `d(metric)/dT` is exact. Noise sources (resistor
thermal, semiconductor shot/flicker, MOSFET channel-thermal, Verilog-A
`white_noise`/`flicker_noise`) are summed in one registry.

---

# Python API

`import sane`. The package exposes two classes — `Circuit` (build) and `Model`
(analyze) — plus labeled result objects and the symbolic engine. The Python layer
is a thin wrapper; orchestration, parameter store and analyses are in Rust
(`sane_analysis::Model`, raw handles at `sane._core`).

## Top-level functions

| Function | Returns | Purpose |
|---|---|---|
| `parse(netlist)` | `Circuit` | shorthand for `Circuit.parse` |
| `set_parallelism(threads)` | `None` | worker threads of the parallel sweeps (AC/noise over frequency, HB device sampling): `n`=n, `0`=the default (4). Set it before the first sweep |
| `set_log_level(level="info")` | `None` | native logging: `"debug"`/`"info"`/`"warning"`/`"error"`/`"off"` |
| `profile_begin()` / `profile_take()` | `None` / `list[(stage, ms)]` | collect per-stage timings |
| `reduced_netlist(netlist, transforms)` | `str` | apply graph transforms, emit a smaller netlist |
| `symbols(names)` | `Expr \| tuple[Expr]` | create symbols (space-separated) in a fresh `Context` |
| `jacobian(residuals, wrt)` | `list[list[Expr]]` | symbolic Jacobian |
| `sparsity(jac)` | `list[list[bool]]` | structural nonzero pattern |
| `compile_tape(roots, inputs)` | `Tape` | compile expressions to a flat evaluator |

Exported types: `Circuit`, `Model`, the result objects, `Context`, `Expr`,
`Tape`, and `GROUND_ALIASES` (`{"0","gnd","GND","Gnd","ground"}`).

## `Circuit`

Build from a netlist string or programmatically. Builder methods return `self`
for chaining. A node reference is a string name (or `0`/ground alias).

**Construction**
- `Circuit()` — empty circuit
- `Circuit.parse(netlist: str) -> Circuit` *(classmethod)* — parse a SPICE netlist
- `extract() -> Model` — lower to an analyzable `Model`

**Linear / controlled elements** — each returns `Circuit`
- `resistor(name, n1, n2, value=None)`
- `capacitor(name, n1, n2, value=None)`
- `inductor(name, n1, n2, value=None)`
- `voltage_source(name, n1, n2, dc=None)`
- `current_source(name, n1, n2, dc=None)`
- `vcvs(name, out_p, out_n, ctrl_p, ctrl_n, gain=None)` — voltage-controlled voltage source
- `vccs(name, out_p, out_n, ctrl_p, ctrl_n, gain=None)` — voltage-controlled current source
- `cccs(name, out_p, out_n, ctrl, gain=None)` — current-controlled current source
- `ccvs(name, out_p, out_n, ctrl, gain=None)` — current-controlled voltage source
- `mutual(name, l1, l2, k=None)` — couple two named inductors

**Nonlinear devices** — each returns `Circuit`; `**params` override model defaults
- `diode(name, anode, cathode, **params)`
- `mosfet(name, d, g, s, b=None, **params)` — bulk defaults to source
- `bjt(name, c, b, e, **params)`
- `vswitch(name, n1, n2, ctrl_p, ctrl_n, **params)`
- `cswitch(name, n1, n2, ctrl, **params)`

**Source waveforms** — attach to the most recently added source, return `Circuit`
- `sine(offset=0.0, amplitude=1.0, freq=None, omega=None)`
- `pulse(v1, v2, delay=0.0, rise=0.0, fall=0.0, width=0.0, period=0.0)`
- `exp(v1, v2, td1=0.0, tau1=0.0, td2=0.0, tau2=0.0)`
- `pwl(points)` — `points: list[(t, v)]`

**Introspection**
- `node_names: list[str]` — by internal node id (0 = ground)
- `values: dict[str, float]` — bound values by symbol name (`"R1"`, `"D1.Is"`)
- `elements: list[dict]` — `{"name","kind","nodes","control"}`
- `couplings: list[dict]` — `{"name","inductors"}`
- `device_count: int`

## `Model` — analyses

`Model.from_netlist(netlist) -> Model` is shorthand for
`Circuit.parse(netlist).extract()`.

Most analyses accept `values=None` (a `dict[str, float]` of per-call parameter
overrides) and `x0=None` (a starting state vector). Node/output references
resolve in order: unknown name → node name → source branch current.

| Method | Returns | Notes |
|---|---|---|
| `operating_point(values=None, x0=None, tol=1e-10, max_iter=100, nodeset=None, reltol=None, abstol=None, vntol=None)` | `OperatingPoint` | DC solve; `nodeset: dict[ref, V]` stiff-pins for symmetry breaking; per-component `reltol`/`abstol`/`vntol` |
| `transient(t, values=None, x0=None, rtol=1e-4, atol=1e-7, dt_max=None)` | `Trajectory` | ESDIRK32; `x0` defaults to the DC point; `dt_max` caps the adaptive step |
| `transient_events()` | `list[(name, t, dir)]` | the switching events of the last transient: surface `instance#k`, crossing time, `+1` rising / `-1` falling |
| `small_signal(input, output, values=None, x0=None)` | `SmallSignal` | linearize at the bias; poles/zeros/AC/sensitivity |
| `ac(input, output, freqs_hz, values=None, x0=None)` | `AcResponse` | AC response with the full derivative API |
| `ac_transfer(input, output, freqs_hz, values=None)` | `ndarray \| None` | symbolic transfer; exact for linear, no OP solve |
| `harmonic_balance(f0=0.0, harmonics=8, values=None, x0=None, continuation=None, tol=1e-10, max_iter=60, oversample=16, samples=None)` | `HarmonicBalance` | periodic steady state; `f0<=0` infers from a `SIN` source |
| `noise(output, fstart, fstop, points=50, values=None, x0=None)` | `NoiseSpectrum` | output-referred PSD [V/√Hz] |
| `temp_sweep(output, tstart, tstop, points=50, values=None)` | `TempSweep` | output vs temperature [°C], warm-started |
| `state_space(input, output, values=None, x0=None)` | `StateSpace` | descriptor `(E, A, B, C, D)` at the OP |
| `model_reduce(input, output, order, fstart, fstop, points=50, values=None, x0=None)` | `ReducedModel` | dominant-pole MOR keeping `order` poles |
| `optimize(targets, tunables, iters=50, values=None)` | `Optimization` | Levenberg-Marquardt over exact sensitivities; `targets: dict[output, value]`, `tunables: list[str]` |

**Sensitivity** (exact, via autodiff — no finite differences)

| Method | Returns | Notes |
|---|---|---|
| `sensitivity(output, values=None, x0=None, t=0.0)` | `Sensitivity` | DC `dy/dp` over all parameters (one adjoint solve) |
| `hessian(output, wrt, values=None, x0=None, t=0.0)` | `ndarray` | exact second order over the `wrt` subset (dense symmetric) |
| `transient_sensitivity(params, t, rtol=1e-4, atol=1e-7, values=None)` | `dict[str, Trajectory]` | forward `dx(t)/dp` per parameter |
| `ac_sensitivity(input, param, output, freqs_hz, values=None, x0=None)` | `ndarray` | `dH/dp(f)` including the OP shift (complex per frequency) |
| `pole_sensitivity(input, param, values=None, x0=None)` | `list[(pole, dpole/dp)]` | exact `dλ/dp` |
| `zero_sensitivity(input, output, param, values=None, x0=None)` | `list[(zero, dzero/dp)]` | exact `dz/dp` |

## `Model` — parameters & introspection

**Parameters** (hierarchical, by name or path)
- `model.X1.R2` / `model["X1.R2"]` — read leaf value or a group proxy
- `model.X1.R2 = 1e3` / `model["X1.R2"] = 1e3` — write
- `get(name) -> float`, `set(name, value) -> None`
- `update(mapping=None, **kw) -> None` — bulk atomic set
- `reset() -> None` — restore construction defaults
- `name in model` / `len(model)` / `iter(model)` — membership / count / names

**Introspection** (properties)
- `unknowns: list[str]`, `dim: int`, `nnz: int`
- `params: list[str]`, `values: dict[str, float]`
- `node_names: list[str]`
- `unknown_index(ref) -> int`, `unknown_name(ref) -> str`
- `profile: list[(stage, ms)]`, `transforms`, `eliminated`
- `symbolic_context: Context`, `core` *(raw `_core.Model`)*

## `Model` — symbolic graph & transforms

The model shares the symbolic `Context`, so its equations come back as `Expr`.

**Symbolic access**
- `residuals: list[Expr]` — the equations `F(x, x', t)`, one per row
- `jacobian_x_symbolic() -> list[list[Expr]]` — `dF/dx`
- `jacobian_xdot_symbolic() -> list[list[Expr]]` — `dF/dx'`
- `system_matrix() -> list[list[Expr]]` — `A(s) = dF/dx + s·dF/dx'`
- `transfer_function(input, output) -> Expr | None` — closed-form `H(s)`
- `transfer_approx(input, output, tol=1e-3, freq=1e3)` / `transfer_approx_at(...)` / `transfer_approx_named_at(...)` — pruned transfer (Analog-Insydes / Sherman-Morrison)
- `term_estimate(cap=None) -> int`, `partition_sizes() -> (int, int) | None`
- `latex() -> str`, `transfer_latex(input, output) -> str`, `to_dot(...) -> str`

**Transforms** (return a new `Model` on the same context)
- `linearize(canonical=False)` — small-signal mass-matrix DAE `G·dx + C·dx' = 0`
- `reduce(rel_tol=1e-3, freqs=None, values=None, x0=None)` — OP-guided branch pruning; sets `.transforms`
- `eliminate(keep=None)` — exact resistive-node elimination; sets `.eliminated`
- `fold(*paths)` — bind parameters/groups to their current value (constant-fold, drop from the sensitivity set)

## Result objects

All are labeled; `__getitem__(ref)` resolves a node/unknown/source name.

**`OperatingPoint`** — `vector: ndarray`, `unknowns`, `node_names`;
`op[ref] -> float`, `get(ref, default=None)`, `to_dict()`;
`sensitivity(output) -> Sensitivity`, `hessian(output, wrt) -> ndarray`.

**`Trajectory`** — `t: ndarray`, `matrix: ndarray (T,n)`, `unknowns`, `node_names`;
`traj[ref] -> ndarray`, `to_dict()`, `plot(*refs, ax=None, show=False)`;
`sensitivity(output, t=None, wrt=None, rtol=1e-4, atol=1e-7) -> Sensitivity`.

**`Sensitivity`** — `params`, `gradient: ndarray`, `output`;
`s[name] -> float`, `to_dict()`, `relative() -> ndarray`;
`ranked(relative=True, threshold=0.0) -> list[(param, value)]`;
`rollup(relative=True) -> list[(component, value)]` (L2-aggregated to device level).

**`SmallSignal`** — `G, C, B`, `unknowns`, `input_name`, `output`;
`poles() -> ndarray` [rad/s], `zeros() -> ndarray`,
`response(freqs_hz) -> ndarray` (complex `H`), `bode(freqs_hz) -> (mag_db, phase_deg)`,
`pole_sensitivity()`, `zero_sensitivity()` → `list[(value, Sensitivity)]`.

**`AcResponse`** — `freqs`, `value: ndarray` (complex `H`);
`sensitivity(f, metric="mag") -> Sensitivity` (metrics `"mag"`/`"phase"`/`"real"`/`"imag"`),
`hessian(f, wrt, metric="mag") -> ndarray`.

**`NoiseSpectrum`** — `freqs`, `noise: ndarray` [V/√Hz], `output`;
`sensitivity(f) -> Sensitivity` (gradient of the PSD), `integrated_rms() -> float` [V].

**`StateSpace`** — `states`, `E, A: (n,n)`, `B, C: (n,)`, `D: float`, `input`, `output`.
Model: `E·x' = A·x + B·u`, `y = C·x + D·u`.

**`TempSweep`** — `temps_c: ndarray`, `values: ndarray`, `output`.

**`Optimization`** — `tuned: dict`, `results: list[(output, target, achieved)]`,
`trace: ndarray` (RMS per iteration), `converged: bool`.

**`ReducedModel`** — `freqs`, `full_db`, `reduced_db`, `poles`, `zeros`,
`max_err_db: float`.

**`HarmonicBalance`** — `spectra: (n, K+1)`, `f0`, `harmonics`, `freqs`,
`converged`, `iters`, `residual_norm`, `setup_ms`, `solve_ms`;
`hb[ref] -> ndarray` (complex spectrum), `harmonic(ref, k) -> complex`,
`dc(ref) -> float`, `magnitude(ref)`, `phase(ref, deg=True)`, `thd(ref) -> float`,
`plot(*refs, db=False)`; `sensitivity(ref, k, metric="coeff") -> Sensitivity`,
`hessian(ref, k, wrt, metric="coeff") -> ndarray`.

## Symbolic engine

The hash-consed core is exposed directly. `symbols("x y")` returns `Expr` that
support the Python operators and:
`.exp() .ln() .sqrt() .sin() .cos() .sinh() .cosh() .tanh() .floor()`,
`.diff(sym)`, `.grad([syms])`, `.hess([syms])`, `.simplify()`,
`.eval(**vals) -> float`, `.eval_complex(**vals) -> complex`, `.free_symbols`,
`.context`.

Derivatives are symbolic expressions in the same graph, so they compose to any
order and with respect to any leaves: `.diff` is forward-mode (one symbol),
`.grad` is a single reverse-mode (adjoint) sweep over all requested leaves,
`.hess` is forward-over-reverse.

```python
x, y = sane.symbols("x y")
f = 2 * x + (y ** 2).exp()
f.diff(x)                       # 2
f.grad([x, y])                  # [2, 2*y*exp(y^2)]
f.grad([y])[0].diff(y)          # second derivative, any order by repetition
f.eval(x=1.0, y=0.5)            # 2 + exp(0.25)
((x + 1) / (x + 1)).simplify()  # 1
sane.compile_tape([x * y], [x, y]).eval([3.0, 4.0])   # [12.0]
```

---

## Verilog-A and PDK compact models

A Verilog-A module is loaded with `.veriloga "file.va"` (or inline) and
instantiated with an `N` element; its analog block is lowered onto the DAG, so
the device runs through every analysis like a built-in. The lowering supports
the analog subset (contributions, `ddt`/`idt`, `if`/`case`/`for`, analog
functions, `white_noise`/`flicker_noise`, `laplace_nd`); constructs that cannot
be lowered exactly (`transition`/`slew`, `laplace_zp`, ...) error at parse time.

**System tasks and diagnostics.** The dollar-diagnostic tasks are honored, not
dropped: `$error`/`$fatal` reached unconditionally (or under a parameter guard
that folds true for the instance) is a hard load failure carrying the message,
module and line; `$warning` and an unresolved `` `include `` become catchable
`SaneConvergenceWarning`s at parse/extract; a `$error` under a *runtime* guard
(which SANE's single analysis-agnostic model cannot enforce) becomes a load-time
warning that the assertion is unenforced rather than a silent no-op; a
non-finite compile-time constant baked into the residual always warns. Purely
textual tasks (`$strobe`, `$display`, `$write`, `$monitor`, `$debug`, `$fopen`/
`$fclose`/`$fwrite`, and `$finish` under a runtime guard) are ignored no-ops.

**Analog events.** `@(cross(expr, dir))` and `@(above(expr))` declare a
*switching surface* `expr = 0` (direction `0` either way, `+1` rising, `-1`
falling; `above` is rising). The transient integrator evaluates every declared
surface once per candidate step, locates a sign change on the step's dense
output and retakes the step to the crossing, so a hard mode change written as
an `if` or `?:` on the same comparison flips at a step boundary instead of
being discovered through rejected steps. The body of such an event must be
passive (empty, `$discontinuity`, `$bound_step`): an event body executes only
at the event instant, which needs a discrete state the one continuous model
does not carry, and a body with statements is rejected at load. `@(timer(...))`
and `@(final_step)` are rejected for the same reason. `@(initial_step)` /
`@(initial_model)` bodies run **unconditionally** as part of the assembled
residual (not gated to the first time step). This is exact for the common
precompute idiom — bias-independent constants derived from parameters — but a
stateful `@(initial_step)` block would not be gated to t=0. The events of the
most recent transient are reported by `Model.transient_events()` as
`(instance#k, t, direction)`.

**`idt` and guarded division.** `idt(u, ic)` mints a differential state with the
correct DC semantics, but the `ic` argument is **not** applied: SANE lowers one
DC/transient-shared DAE with no per-state initial-condition override, so the
transient always seeds from the DC operating point. A non-zero `ic` raises a
catchable `SaneConvergenceWarning` at load rather than being silently dropped.
Division inside a conditional arm is lowered with a sign-preserving magnitude
floor on the divisor, so an eagerly-evaluated not-taken branch (e.g. the `1/x`
in `if (x != 0) y = 1/x;` at `x = 0`) keeps both the residual and the Jacobian
finite; the taken arm is exact.

```python
ckt = sane.Circuit.parse("""
    .veriloga
    module diode(a, c);
      inout a, c; electrical a, c;
      parameter real Is = 1e-14;
      parameter real N  = 1.0;
      parameter real Vt = 0.025852;
      analog I(a, c) <+ Is * (limexp(V(a,c)/(N*Vt)) - 1.0);
    endmodule
    .endveriloga
    V1 1 0 0.7
    R1 1 2 1k
    N1 2 0 diode Is=2e-14
""")
op = ckt.extract().operating_point()   # solved by the native engine
```

**PDK level idiom.** Foundry PDKs ship devices as Verilog-A bound by a SPICE
`level` number. SANE has no built-in compiled compact models — the `.va` *is* the
model — so the level is bound explicitly:

```spice
.veriloga "pdk/bsim4.va"          ; module bsim4va
.model_alias level=54 bsim4va     ; bind the legacy level
.include "pdk/corners/tt.spice"   ; binned level=54 .model cards
M1 d g s b nch L=0.15u W=1u       ; routed to bsim4va; nmos/pmos -> type +/-1
```

The card's `nmos`/`pmos` token sets the module `type`, `.option scale` converts
drawn microns to meters, model binning picks the card by `L`/`W`, and `M`/`mult`/`nf`
set multiplicity. Device subcircuits (`X... sky130_fd_pr__nfet_01v8 l='...' w='...'`)
over binned cards are ingested natively, so a raw foundry netlist + corner parses
with no external flattener.

## Rust embedding

The analysis stack is pure Rust with the same labeled API the Python binding
wraps.

```rust
use sane_analysis::Model;

let model = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end")?;
let op = model.operating_point(&[])?;            // DC bias
let s  = op.sensitivity("out")?;                 // exact dy/dp over all parameters
model.set("R2", 3e3)?;
let traj = model.transient(&[], &t_eval, 1e-4, 1e-7)?;

let x = op.vector().to_vec();
let p = model.pvec(&[]);
let poles  = model.poles(x.clone(), p.clone())?;
let dpoles = model.pole_gradient(x, p)?;         // exact dpole/dp, all params
```

See `crates/analysis/tests/embed.rs` for a worked acceptance suite.

## Configuration and environment variables

Every run-time switch of the engine is a field of `sane_core::Config`, documented there. The
configuration is read from the environment once, on first use, and a host sets it directly
with `sane_core::set_config` or `update_config` (the browser build, for example, selects the
interpreter-friendly tape schedule). The variables below override the defaults for A/B
benchmarking and debugging; the log level (`SANE_LOG`) and the test-corpus locations
(`SANE_VA_CORPUS`, `SANE_OPENVAF_BIN`) are read where they are used.

| Variable | Effect |
|---|---|
| `SANE_THREADS` | worker threads of the parallel sweeps, default 4 (see also `set_parallelism`); the linear solves themselves are sequential |
| `SANE_JIT` | `0` disables the native compilation of hot tapes (default on; build must have the `jit` feature) |
| `SANE_GRAPH_SOLVE` | `0` solves every Newton system with the sparse LU library instead of rsdag's graph solve (the A/B reference; default on) |
| `SANE_TAPE_SPEC` | `0` disables choice specialization of the circuit-level tapes |
| `SANE_NO_DEVBUNDLE` | set to inline every Verilog-A instance as its own graph clone instead of calls into one shared function body |
| `SANE_TRAN_FIXED` | force a fixed transient step instead of adaptive |
| `SANE_EVENTS` | `0` disables locating the declared switching surfaces (the controller then finds every mode change through rejects; the A/B reference) |
| `SANE_LOG` | log level (`debug`/`info`/`warning`/`error`); the sparse solver's own records (rslab: ordering picked, fill, threads, factor time) appear at `debug` with an `rslab:` prefix, its warnings at `warning`, and rsdag's compile stages likewise with an `rsdag:` prefix |
| `SANE_DC_TRACE` | per-iteration DC Newton residual/step trace |
| `SANE_TRAN_TRACE` | per-candidate transient step trace: time, size, error, verdict, located events, stage Newton iterations |
| `SANE_VA_CORPUS` | path to a Verilog-A model corpus (for the corpus tests) |
| `SANE_VA_TRACE_NAN` | report non-finite compile-time constants during VA lowering |
| `SANE_NO_TEMPLATE` | lower every Verilog-A instance from scratch instead of once per module structure |

## Crates

| Crate | Purpose |
|---|---|
| `sane-core` | SANE's constants, configuration, logging, profiling and the lowering of named math calls; the graph itself is rsdag |
| `vendor/rsdag` | the expression graph, differentiation, tape, native backend and the sparse solve programs (vendored, see `vendor/rsdag/VENDOR.md`) |
| `sane-mna` | MNA stamps, symbolic determinant, Cramer → `H(s)` |
| `sane-netlist` | SPICE parser (preprocessor, expressions, subckt flattening) |
| `sane-veriloga` | native Verilog-A frontend (parse → elaborate → lower to DAG); ships the built-in device models as Verilog-A source (`builtin/*.va`) |
| `sane-device` | the device contract (`lower_behavioral` → DAE fragment) and lowering support types |
| `sane-dae` | DAE assembly + small-signal matrix |
| `sane-solve` | native sparse Newton DC + homotopy + ESDIRK32 transient |
| `sane-analysis` | high-level analyses (OP/transient/AC/sweeps/PZ/noise/MOR/opt) |
| `sane-export` | export to LaTeX and Python/NumPy |
| `sane-py` | PyO3 binding |

## Validation

The numeric path is cross-validated against **ngspice** and **Xyce**:

- DC vs ngspice: 29/32 decks.
- BSIM4 DC + AC vs ngspice + OSDI: 12/12 SKY130 AnalogGym op-amps, relative error 1e-6..1e-4.
- Harmonic balance vs Xyce: ~6e-6 %.
- Exact sensitivity (DC/AC/pole/transient, first and second order) vs finite-difference oracles (`crates/py/tests/`).
- Devices: the built-in models (diode, MOSFET, BJT, JFET, MESFET, switches, transformer, transmission line) are themselves Verilog-A (`crates/veriloga/builtin/`), lowered through the same pipeline as user compact models; closed-form physics checks in `crates/veriloga/tests/builtin_devices.rs`.
- Verilog-A corpus (OpenVAF `integration_tests`, via `SANE_VA_CORPUS`): 19/20 models — BSIM3/4/6/BULK/CMG/IMG/SOI, PSP102/103, MEXTRAM, HICUM, EKV, ASMHEMT, HiSIM family — parse, elaborate and lower; the one exception (HiSIM2) uses an unbounded `while` the analog subset rejects. PSP102/PSP103 and EKV solve DC/AC/pole-zero/transient end-to-end (`crates/py/tests/validate_psp103_ring.py`).

```
cargo test --workspace                      # Rust tests
maturin develop -m crates/py/Cargo.toml     # Python module `sane`
python crates/py/tests/validate_ngspice.py  # local; needs ngspice + scipy
```

The native backend (residual, Jacobian and device bodies emitted as machine
code by rsdag, see `vendor/rsdag`) is the `jit` build feature, on by default;
numeric results are identical to the interpreter. `SANE_JIT=0` disables it at
runtime.

The Newton systems of every analysis (operating point, transient stages,
sensitivities and adjoints) are solved by rsdag's graph solve: the static LU
of the Jacobian's pattern as one program over the entry values, with the
factorization as its parameter-pure prolog and the substitution as its main
pass, pivot rows guarded and repivoted on the values when a guard fails. The
sparse LU library (KLU, vendored rslab) keeps what the program does not cover:
the complex small-signal systems of AC, noise and their adjoints, the
block-dense harmonic-balance Jacobian, and any pattern beyond the program's
range by the cost predictor. `SANE_GRAPH_SOLVE=0` selects the library
everywhere for comparison.

## License

SANE is source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE) —
free for noncommercial use: research, evaluation, education, and academia.
Commercial use requires a separate license, see [COMMERCIAL.md](COMMERCIAL.md)
or contact info@milanrother.com.
