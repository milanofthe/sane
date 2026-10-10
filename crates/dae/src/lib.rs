//! Time-domain nonlinear DAE assembly and extraction.
//!
//! Builds the system `I(x, t) + d/dt Q(x) = 0` for a circuit, where
//! `x = [node voltages; branch currents; device states]`: every row is a
//! current `I` and the time derivative of a charge `Q`. A capacitor stores
//! the charge `C*(v_a - v_b)` on its nodes, an inductor adds a branch current
//! with the current `v_a - v_b` and the charge (flux) `-L*i` on its row.
//! Nonlinear devices contribute their large-signal terminal currents and
//! charges. Nothing is solved here; the output is the expression graph the
//! analyses compile.
//!
//! Naming convention for the minted symbols: node voltages `v{n}`, branch
//! currents `i_{name}`, and time `t`.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use rsdag::{sparse_jacobian, Crossing, ExprId, Node, SymbolId};
use sane_circuit::SourceFn;
use sane_core::Graph;

// Assembly of the symbolic DAE from a parsed circuit.
mod assemble;
// Subcircuit instances: placed bodies, their renaming and topology.
pub mod linearize;
mod observers;
mod rewrite;
// Sensitivity machinery: directional derivatives, augmentation, Hessian blocks.
mod sens;
// Graph transformations (shorts / opens / exact node elimination).
#[cfg(test)]
mod tests;
mod transform;

pub use assemble::{assemble, assemble_at};
pub use observers::Observers;
pub use sens::{
    ac_param_derivatives, augment_with_scaled_sensitivities, augment_with_sensitivities, frozen,
    lagrangian_hessian, param_column, Frozen, HessianSym,
};
pub use transform::{eliminate_nodes, reduce_graph};

pub use sane_device::{
    Assertion, DeviceInstance, DeviceModel, FragmentLimit as Limit, LimitKind, NoiseSource,
    UnknownKind,
};

impl Dae {
    /// Keep the limits that hold at the parameter values `value_of` gives
    /// (by symbol; an unvalued symbol keeps its limit): the Newton aids of
    /// the model built at those values. Limits only shape the Newton path,
    /// so this decides no result.
    pub fn keep_limits_at(&mut self, ctx: &Graph, value_of: impl Fn(SymbolId) -> Option<f64>) {
        let whens: Vec<ExprId> = self.limits.iter().filter_map(|l| l.when).collect();
        if whens.is_empty() {
            return;
        }
        let env: HashMap<SymbolId, f64> = ctx
            .free_symbols_in(&whens)
            .into_iter()
            .filter_map(|s| value_of(s).map(|v| (s, v)))
            .collect();
        let on = rsdag::eval(ctx, &whens, &env);
        let mut on = on.into_iter();
        self.limits.retain(|l| match l.when {
            None => true,
            Some(_) => on.next().is_some_and(|v| v != 0.0 || v.is_nan()),
        });
    }
}

/// One transport delay: the engine integrates the SOURCE unknown `src`
/// (an auxiliary algebraic signal, e.g. the Branin wave `v + Z0*i` or an
/// `absdelay` operand), and the OUTPUT unknown `out` is pinned to the
/// interpolated history `src(t - tau)` through the residual
/// `x[out] - hist = 0`, where `hist` is the history input symbol the
/// integrator fills per stage evaluation. `tau` is a parameter expression
/// (constant per run).
#[derive(Clone, Debug)]
pub struct DelaySpec {
    /// The signal the line delays, over the unknowns (an unknown as
    /// assembled; what a transform makes of it after).
    pub src: ExprId,
    /// The history input: `src` a delay `tau` earlier.
    pub hist: SymbolId,
    /// Delay time as an expression over parameters.
    pub tau: ExprId,
}

/// A switching surface `g(x, t) = 0` a device declares (Verilog-A
/// `@(cross ...)`, a built-in switch threshold): the transient integrator
/// lands a step on every zero crossing of `g` in direction `dir` (`0` either
/// way, `+1` rising, `-1` falling), so a hard mode change in the residuals
/// happens at a step boundary. `name` is `instance#k`.
#[derive(Clone, Debug)]
pub struct EventSpec {
    pub g: ExprId,
    /// Which sign change of `g` is an event (rsdag's role vocabulary; the
    /// guard is registered with this direction on the DAE function).
    pub dir: Crossing,
    pub name: String,
}

/// An extracted differential-algebraic system `I(x, t) + d/dt Q(x) = 0`.
pub struct Dae {
    /// Every row `i` as `I_i(x, t) + d/dt Q_i(x) = 0`: the current `I_i` and
    /// the charge `Q_i` (zero for an algebraic row), one per unknown. A
    /// device body's charge model is not in its current and its current
    /// model not in its charge.
    pub currents: Vec<ExprId>,
    pub charges: Vec<ExprId>,
    /// Names of the unknowns `x`, aligned with `x`.
    pub unknowns: Vec<String>,
    /// What each unknown physically is, parallel to `unknowns` (see
    /// [`UnknownKind`]); the solver's tolerances and shunts are taken per kind.
    pub kinds: Vec<UnknownKind>,
    /// Unknown symbols `x`.
    pub x: Vec<SymbolId>,
    /// The time symbol.
    pub t: SymbolId,
    /// Number of leading node-voltage unknowns (their KCL rows lead).
    pub n_nodes: usize,
    /// Module default of every device parameter symbol the residuals
    /// reference, by symbol: the value an unstated parameter takes.
    pub param_defaults: rustc_hash::FxHashMap<SymbolId, f64>,
    /// Switching surfaces (see [`EventSpec`]); empty for circuits without
    /// declared events. Graph transforms drop them like the delays.
    pub events: Vec<EventSpec>,
    /// Transport delays (`absdelay`, ideal transmission lines); empty for
    /// delay-free circuits. Transforms that cannot preserve the unknown
    /// indexing drop these (the analysis layer guards misuse).
    pub delays: Vec<DelaySpec>,
    /// Companion conductance network (node-KCL row, node-voltage col, value) for
    /// homotopy continuation: each device's linear `lambda = 0` form (see
    /// [`sane_device::DeviceModel::companion`]), by the node voltage of the
    /// row and of the column.
    pub companion: Vec<(SymbolId, SymbolId, f64)>,
    /// The small-signal noise sources (current noise generators with a PSD
    /// expression) and the operating-point variables (`(* desc *)`
    /// annotations, named expressions for OP reporting) the devices export:
    /// read-only observers, no residual references them. See
    /// [`Observers::flatten`].
    pub observers: Observers,
    /// DC Newton seeds by unknown name (`idt(u, ic)` states with constant ic):
    /// `.nodeset`-style starting values, not constraints. Keyed by name so the
    /// registry survives unknown reordering; entries whose unknown no longer
    /// exists are ignored at lookup.
    pub dc_seeds: Vec<(String, f64)>,
    /// Per-device controlling-voltage limits (SPICE `pnjlim`/`fetlim`) over
    /// the unknowns' symbols (`None`: ground), for the Newton loops'
    /// `device_limiting`.
    pub limits: Vec<Limit>,
    /// Independent-source stimulus shapes in the circuit, by element name. Carries
    /// the structural facts the lowered graph cannot recover -- transient
    /// breakpoints (waveform corners the integrator must land on) and the HB
    /// fundamental. Empty for transformed DAEs.
    pub sources: Vec<(String, SourceFn)>,
    /// Names of *every* independent V/I source element (shaped or plain DC,
    /// top-level or subcircuit-scoped like `Xop.I0`). Each name is also the
    /// symbol of the element's DC value, so the DC source-stepping continuation
    /// ramps exactly these -- structurally, instead of guessing from parameter
    /// names (a dot-free V/I prefix misses every subcircuit bias source and a
    /// name test cannot tell `X1.I0`, a source, from `Q1.Is`, a saturation
    /// current). Empty for transformed or hand-built DAEs, where the solver
    /// falls back to the name heuristic.
    pub source_names: Vec<String>,
    /// What must hold of the parameters for the residuals to be the circuit's
    /// (see [`sane_device::Assertion`]), over the circuit's parameter symbols:
    /// the devices' own and what their structure rests on (see
    /// `BehavioralFragment::structural`).
    pub assertions: Vec<sane_device::Assertion>,
    /// What the devices' structure rests on (see
    /// `BehavioralFragment::structural`): the integer modes and the
    /// conditions on parameters that decided their topology at the values
    /// the DAE was assembled at. A binding where one fails is a circuit of
    /// another structure, assembled anew.
    pub structure: Vec<sane_device::Assertion>,
    /// The internal nodes the structure collapsed, by name, with the
    /// unknown whose voltage each takes (`None`: ground).
    pub aliases: Vec<(String, Option<String>)>,
    /// Display names of nodes the residuals reach, for views of the graph:
    /// a call by the instance that made it (`X1`, `M1`; inside a subcircuit
    /// body the body's own name, `__inv__.M1`), a subcircuit body's formal
    /// net by the net's name in the body.
    pub labels: rustc_hash::FxHashMap<ExprId, String>,
    /// Where each noise generator enters, once asked for (see
    /// [`noise_injection`](Self::noise_injection)).
    pub injection: OnceLock<Arc<Vec<Vec<(usize, ExprId)>>>>,
    /// The currents and charges at rest, once asked for (see
    /// [`at_rest`](Self::at_rest)).
    pub rest: OnceLock<Arc<(Vec<ExprId>, Vec<ExprId>)>>,
}

/// A noise source's level at an operating point (see [`Dae::noise_levels`]).
#[derive(Clone, Debug, PartialEq)]
pub enum NoiseLevel {
    /// White or flicker: `psd / f^fexp`.
    Spectral { psd: f64, fexp: f64 },
    /// Tabular: `(frequency, psd)` points, interpolated linearly.
    Table(Vec<(f64, f64)>),
}

impl Dae {
    /// The currents and charges with every noise generator at zero, where
    /// they are in every evaluation, the calls that pass one specialized to
    /// it: what the solver's programs and the Jacobians are built from, so
    /// the generators cost no evaluation.
    pub fn at_rest(&self, ctx: &mut Graph) -> Arc<(Vec<ExprId>, Vec<ExprId>)> {
        if let Some(r) = self.rest.get() {
            return r.clone();
        }
        let generators = self.observers.generators(ctx);
        let rest = if generators.is_empty() {
            (self.currents.clone(), self.charges.clone())
        } else {
            let zero = ctx.zero();
            let quiet: rustc_hash::FxHashMap<SymbolId, ExprId> =
                generators.iter().map(|&g| (g, zero)).collect();
            let roots: Vec<ExprId> = self.currents.iter().chain(&self.charges).copied().collect();
            let roots = rsdag::substitute(ctx, &roots, &quiet);
            let mut roots = ctx.specialize_calls(&roots);
            let charges = roots.split_off(self.currents.len());
            (roots, charges)
        };
        self.rest.get_or_init(|| Arc::new(rest)).clone()
    }

    /// Where each noise generator (in [`Observers::noise`] order) enters
    /// the rows: `(row, d current / d generator)` per source. A generator a
    /// device puts between two nodes enters as `+1` and `-1`; through a
    /// transform (a node eliminated, merged) wherever the rows now carry it.
    pub fn noise_injection(&self, ctx: &mut Graph) -> Arc<Vec<Vec<(usize, ExprId)>>> {
        if let Some(inj) = self.injection.get() {
            return inj.clone();
        }
        let inputs = self.observers.generators(ctx);
        let mut by_source = vec![Vec::new(); inputs.len()];
        for (i, row) in sparse_jacobian(ctx, &self.currents, &inputs)
            .into_iter()
            .enumerate()
        {
            for (q, e) in row {
                by_source[q].push((i, e));
            }
        }
        self.injection.get_or_init(|| Arc::new(by_source)).clone()
    }

    /// Every noise source's level over `env`, in one arena sweep: the sources
    /// share the device subexpressions and calls they read.
    pub fn noise_levels(&self, ctx: &mut Graph, env: &HashMap<SymbolId, f64>) -> Vec<NoiseLevel> {
        let noise = self.observers.noise(ctx);
        let roots: Vec<ExprId> = noise.iter().flat_map(|ns| ns.exprs()).collect();
        let mut vals = rsdag::eval(ctx, &roots, env).into_iter();
        let mut next = || vals.next().expect("one per expression");
        noise
            .iter()
            .map(|ns| {
                let (psd, fexp) = (next(), next());
                match ns.table.len() {
                    0 => NoiseLevel::Spectral { psd, fexp },
                    n => NoiseLevel::Table((0..n).map(|_| (next(), next())).collect()),
                }
            })
            .collect()
    }

    /// Register the system as an rsdag function carrying its roles: the
    /// unknowns as `State`, time as `Time`, every parameter as `Param`, the
    /// currents as `Residual` outputs, the charges a row stores as `Charge`
    /// outputs of the same row, every switching surface as a `Guard`
    /// output with its crossing direction, the delays' signals and times,
    /// and the noise sources' levels. An op-var is the `Observer` output of
    /// the function it is computed in (a device's, a subcircuit body's),
    /// stated there once rather than per system.
    ///
    /// This is how the system layer states what it is: a consumer (SANE's own
    /// solver, an exporter, another backend) reads the structure off the graph
    /// instead of off SANE-side metadata.
    pub fn register_function(&self, ctx: &mut Graph, name: &str) -> rsdag::FuncId {
        let mut params: Vec<SymbolId> = Vec::with_capacity(self.x.len() * 2 + 1);
        let mut roles: Vec<rsdag::ParamRole> = Vec::with_capacity(params.capacity());
        for (i, &s) in self.x.iter().enumerate() {
            params.push(s);
            roles.push(rsdag::ParamRole::State { id: i as u32 });
        }
        // The parameter vector's order (`params`), then time and the delay
        // histories: the signature the solver's programs take their inputs
        // in (`rsdag::Signature`).
        for s in self.params(ctx) {
            params.push(s);
            roles.push(rsdag::ParamRole::Param);
        }
        params.push(self.t);
        roles.push(rsdag::ParamRole::Time);
        for (k, dl) in self.delays.iter().enumerate() {
            params.push(dl.hist);
            roles.push(rsdag::ParamRole::History { id: k as u32 });
        }
        let noise = self.observers.noise(ctx);
        for (k, n) in noise.iter().enumerate() {
            params.push(n.input);
            roles.push(rsdag::ParamRole::Noise { id: k as u32 });
        }

        let mut outputs: Vec<ExprId> = self.currents.clone();
        let mut out_roles: Vec<rsdag::OutputRole> = (0..self.currents.len())
            .map(|i| rsdag::OutputRole::Residual { id: i as u32 })
            .collect();
        for (i, &q) in self.charges.iter().enumerate() {
            if !ctx.is_zero(q) {
                outputs.push(q);
                out_roles.push(rsdag::OutputRole::Charge { id: i as u32 });
            }
        }
        for (i, ev) in self.events.iter().enumerate() {
            outputs.push(ev.g);
            out_roles.push(rsdag::OutputRole::Guard {
                id: i as u32,
                dir: ev.dir,
            });
        }
        for (k, dl) in self.delays.iter().enumerate() {
            outputs.extend([dl.src, dl.tau]);
            out_roles.push(rsdag::OutputRole::DelaySource { id: k as u32 });
            out_roles.push(rsdag::OutputRole::DelayTime { id: k as u32 });
        }
        for (k, n) in noise.iter().enumerate() {
            for (elem, e) in n.exprs().into_iter().enumerate() {
                outputs.push(e);
                out_roles.push(rsdag::OutputRole::NoiseLevel {
                    id: k as u32,
                    elem: elem as u32,
                });
            }
        }

        // Idempotent per system, not per name: a DAE compiled twice (an AC
        // run after a DC one) states its signature once, while a derived
        // DAE in the same graph (folded parameters, a linearization) is
        // another system and gets its own function.
        let same = |f: &rsdag::Function| {
            f.name() == name
                && f.params() == params.as_slice()
                && f.param_roles() == roles.as_slice()
                && f.output_roles() == out_roles.as_slice()
                && f.outputs().len() == outputs.len()
                && f.outputs()
                    .iter()
                    .zip(&outputs)
                    .all(|(o, &e)| matches!(*o, rsdag::Output::Expr(x) if x == e))
        };
        if let Some(f) = (0..ctx.n_funcs())
            .map(|i| rsdag::FuncId(i as u32))
            .find(|&f| same(ctx.func(f)))
        {
            return f;
        }
        let f = ctx.define_func(name, params, outputs);
        for (i, r) in roles.into_iter().enumerate() {
            ctx.set_param_role(f, i as u32, r);
        }
        for (i, r) in out_roles.into_iter().enumerate() {
            ctx.set_output_role(f, i as u32, r);
        }
        f
    }

    /// Classify every unknown (see [`UnknownKind`]). Derived from the mint
    /// order rather than stored, so transforms that filter `unknowns` stay
    /// consistent for free. A device state that happens to be named `i_...`
    /// reads as a branch current -- it then merely escapes the step clamp,
    /// which is a degradation in damping, never in correctness.
    pub fn unknown_kinds(&self) -> Vec<UnknownKind> {
        self.kinds.clone()
    }

    pub fn dim(&self) -> usize {
        self.currents.len()
    }

    /// Classify the currents' and charges' nonlinearity in the unknowns `x`
    /// -- the graph fact a harmonic-balance solve reads to pick its harmonic
    /// count and time sampling (degree, transcendental / piecewise / opaque
    /// flags; see [`rsdag::nonlinearity_of`]). Differentiating a charge in
    /// time only scales each harmonic by `jkw0`, so the charges' degree is
    /// the harmonic-generating order of their rates.
    pub fn nonlinearity(&self, ctx: &Graph) -> rsdag::Nonlinearity {
        let vars: std::collections::BTreeSet<SymbolId> = self.x.iter().copied().collect();
        let roots: Vec<ExprId> = self.currents.iter().chain(&self.charges).copied().collect();
        rsdag::nonlinearity_of(ctx, &roots, &vars)
    }

    /// `G = dI/dx` and `C = dQ/dx` as sparse `(rows, cols, exprs)`, from one
    /// sparse Jacobian over the currents and charges together.
    pub fn jacobian_iq_coo(&self, ctx: &mut Graph) -> (Coo, Coo) {
        self.split_coo(ctx, &self.x_cols())
    }

    /// `dI/dt` and `dQ/dt` as sparse `(rows, 0, exprs)`: the explicit time
    /// dependence of the sources, which a Rosenbrock stage reads.
    pub fn jacobian_t_iq_coo(&self, ctx: &mut Graph) -> (Coo, Coo) {
        self.split_coo(ctx, &[(0, self.t)])
    }

    /// `dI/dhist` and `dQ/dhist` as sparse `(rows, cols, exprs)`, columns
    /// indexed like `delays`: how the rows move with the delayed signals.
    pub fn jacobian_hist_iq_coo(&self, ctx: &mut Graph) -> (Coo, Coo) {
        let cols: Vec<(usize, SymbolId)> = (self.delays.iter().enumerate())
            .map(|(c, d)| (c, d.hist))
            .collect();
        self.split_coo(ctx, &cols)
    }

    /// `dI/dp` and `dQ/dp` as sparse `(rows, cols, exprs)`, columns indexed
    /// like `params`.
    pub fn jacobian_p_iq_coo(&self, ctx: &mut Graph, params: &[SymbolId]) -> (Coo, Coo) {
        self.split_coo(ctx, &params.iter().copied().enumerate().collect::<Vec<_>>())
    }

    /// The sparse Jacobian of the currents and charges together over `cols`,
    /// split into the currents' rows and the charges'.
    fn split_coo(&self, ctx: &mut Graph, cols: &[(usize, SymbolId)]) -> (Coo, Coo) {
        let n = self.currents.len();
        let rest = self.at_rest(ctx);
        let roots: Vec<ExprId> = rest.0.iter().chain(&rest.1).copied().collect();
        let wrt: Vec<SymbolId> = cols.iter().map(|&(_, s)| s).collect();
        let (mut i, mut q) = (Coo::default(), Coo::default());
        for (r, row) in sparse_jacobian(ctx, &roots, &wrt).into_iter().enumerate() {
            for (j, e) in row {
                let (block, row) = if r < n { (&mut i, r) } else { (&mut q, r - n) };
                block.0.push(row);
                block.1.push(cols[j].0);
                block.2.push(e);
            }
        }
        (i, q)
    }

    /// The state columns: `(index in x, unknown)`.
    fn x_cols(&self) -> Vec<(usize, SymbolId)> {
        self.x.iter().copied().enumerate().collect()
    }

    /// Sparse `∂I/∂hist` as `(rows, cols, exprs)`, columns indexed like
    /// `delays`. In the frequency domain a delayed source contributes
    /// `Hist_k = e^{-jωτ_k} X_{src_k}`, so these entries move to column
    /// `delays[k].src` scaled by `e^{-jωτ_k}` (see the AC path).
    pub fn jacobian_hist_coo(&self, ctx: &mut Graph) -> Coo {
        let cols: Vec<(usize, SymbolId)> = self
            .delays
            .iter()
            .enumerate()
            .map(|(c, d)| (c, d.hist))
            .collect();
        let rest = self.at_rest(ctx);
        coo(ctx, &rest.0, &cols)
    }

    /// Parameter symbols: free symbols in the currents, charges, assertions
    /// and observers (noise sources, op-vars) that are neither unknowns nor
    /// time, sorted by id.
    pub fn params(&self, ctx: &mut Graph) -> Vec<SymbolId> {
        // The currents and charges, the delay times (parameters even though they appear
        // only in the delay registry), the switching surfaces (which may
        // reference a parameter nothing else does) and what the observers read
        // (a noise-only coefficient, the temperature of thermal noise), in one
        // walk.
        let observed = self.observers.reads(ctx);
        let roots: Vec<ExprId> = self
            .currents
            .iter()
            .chain(&self.charges)
            .copied()
            .chain(self.delays.iter().flat_map(|dl| [dl.src, dl.tau]))
            .chain(self.events.iter().map(|ev| ev.g))
            .chain(
                self.assertions
                    .iter()
                    .chain(&self.structure)
                    .map(|a| a.holds),
            )
            .chain(observed)
            .collect();
        let mut all = ctx.free_symbols_in(&roots);
        for s in &self.x {
            all.remove(s);
        }
        all.remove(&self.t);
        // history inputs are integrator-provided, noise generators zero in
        // every evaluation: neither is a parameter
        for dl in &self.delays {
            all.remove(&dl.hist);
        }
        for g in self.observers.generators(ctx) {
            all.remove(&g);
        }
        all.into_iter().collect()
    }

    /// Derive a new DAE with `fold` (parameter symbol -> constant) substituted into
    /// every current, charge, stamp and noise PSD. Because substitution rebuilds through
    /// the smart constructors, the now-constant subexpressions collapse; the folded
    /// symbols are no longer free, so they drop out of [`params`](Self::params).
    /// The unknown structure (`x`/`t`/`unknowns`) and the numeric companion
    /// and limit tables are unchanged -- folding a coefficient preserves the
    /// topology and sparsity. This is the graph-transform core of parameter fold,
    /// in the same family as `linearize` / `eliminate_nodes`.
    pub fn fold_params(&self, ctx: &mut Graph, fold: &[(SymbolId, f64)]) -> Dae {
        let subst = fold.iter().map(|&(s, v)| (s, ctx.konst_f64(v))).collect();
        self.rewrite(
            ctx,
            rewrite::Rewrite {
                rows: None,
                keep: (0..self.dim()).collect(),
                n_nodes: self.n_nodes,
                subst,
            },
        )
    }
}

/// A sparse block as `(rows, cols, exprs)`.
pub type Coo = (Vec<usize>, Vec<usize>, Vec<ExprId>);

/// `rsdag::sparse_jacobian` of `rows` over the symbols of `cols`, each
/// column labelled with its paired index.
pub(crate) fn coo(ctx: &mut Graph, rows: &[ExprId], cols: &[(usize, SymbolId)]) -> Coo {
    let wrt: Vec<SymbolId> = cols.iter().map(|&(_, s)| s).collect();
    let mut out = Coo::default();
    for (i, row) in sparse_jacobian(ctx, rows, &wrt).into_iter().enumerate() {
        for (j, e) in row {
            out.0.push(i);
            out.1.push(cols[j].0);
            out.2.push(e);
        }
    }
    out
}

pub(crate) fn sym2(ctx: &mut Graph, name: &str) -> (ExprId, SymbolId) {
    let e = ctx.sym(name);
    let s = match ctx.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!("sym() yields a Symbol"),
    };
    (e, s)
}
