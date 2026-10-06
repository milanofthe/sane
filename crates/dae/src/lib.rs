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

use rsdag::{sparse_jacobian, Crossing, ExprId, Node, SymbolId};
use sane_core::Graph;
use sane_mna::SourceFn;

// Assembly of the symbolic DAE from a parsed circuit.
mod assemble;
// Subcircuit instances: placed bodies, their renaming and topology.
mod hierarchy;
pub mod linearize;
mod observers;
// Sensitivity machinery: directional derivatives, augmentation, Hessian blocks.
mod sens;
// Graph transformations (shorts / opens / exact node elimination).
#[cfg(test)]
mod tests;
mod transform;

pub use assemble::{assemble, assemble_dae};
pub use hierarchy::{topology, Body, Instance};
pub use observers::{Flat, Observers};
pub use sens::{
    ac_param_derivatives, augment_with_scaled_sensitivities, augment_with_sensitivities,
    lagrangian_hessian, HessianSym,
};
pub use transform::{eliminate_nodes, reduce_graph};

pub use sane_device::{DeviceInstance, DeviceModel, LimitKind, NoiseSource, UnknownKind};

/// A device controlling-voltage limit (SPICE `pnjlim` / `fetlim`) mapped to the
/// global unknown vector: `v(hi) - v(lo)`, where `None` denotes ground. The DC
/// solver curve-limits this difference between Newton iterates (see
/// [`sane_device::DeviceModel::limits`]).
#[derive(Clone, Copy, Debug)]
pub struct Limit {
    pub hi: Option<usize>,
    pub lo: Option<usize>,
    pub kind: LimitKind,
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
    /// Unknown index of the delayed source signal.
    pub src: usize,
    /// Unknown index of the delay output.
    pub out: usize,
    /// The history input symbol appearing in `out`'s residual.
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
    /// [`sane_device::DeviceModel::companion`]). Empty for transformed DAEs.
    pub companion: Vec<(usize, usize, f64)>,
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
    /// Per-device controlling-voltage limits (SPICE `pnjlim`/`fetlim`) mapped to
    /// global unknown indices, for the DC solver's `device_limiting` trick. Empty
    /// for transformed DAEs (which carry no device limits).
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
    /// Display names of nodes the residuals reach, for views of the graph:
    /// a call by the instance that made it (`X1`, `M1`; inside a subcircuit
    /// body the body's own name, `__inv__.M1`), a subcircuit body's formal
    /// net by the net's name in the body.
    pub labels: rustc_hash::FxHashMap<ExprId, String>,
}

impl Dae {
    /// Every white/flicker noise source's `(psd, flicker exponent)` over
    /// `env`, in one arena sweep: the sources share the device subexpressions
    /// and calls they read. `None` for a tabular source.
    pub fn noise_levels(
        &self,
        ctx: &mut Graph,
        env: &HashMap<SymbolId, f64>,
    ) -> Vec<Option<(f64, f64)>> {
        let flat = self.observers.flatten(ctx);
        let roots: Vec<ExprId> = flat
            .noise
            .iter()
            .filter(|ns| ns.table.is_empty())
            .flat_map(|ns| [ns.psd, ns.flicker_exp])
            .collect();
        let vals = rsdag::eval(ctx, &roots, env);
        let mut pairs = vals.as_chunks::<2>().0.iter();
        flat.noise
            .iter()
            .map(|ns| {
                let &[psd, fexp] = ns.table.is_empty().then(|| pairs.next())??;
                Some((psd, fexp))
            })
            .collect()
    }

    /// Register the system as an rsdag function carrying its roles: the
    /// unknowns as `State`, time as `Time`, every parameter as `Param`, the
    /// currents as `Residual` outputs, the charges a row stores as `Charge`
    /// outputs of the same row, and every switching surface as a `Guard`
    /// output with its crossing direction.
    ///
    /// Nothing calls this function; it is how the system layer states what it
    /// is, so a consumer (SANE's own solver, an exporter, another backend)
    /// reads the structure off the graph instead of off SANE-side metadata.
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

    /// `dI/dp` and `dQ/dp` as sparse `(rows, cols, exprs)`, columns indexed
    /// like `params`.
    pub fn jacobian_p_iq_coo(&self, ctx: &mut Graph, params: &[SymbolId]) -> (Coo, Coo) {
        self.split_coo(ctx, &params.iter().copied().enumerate().collect::<Vec<_>>())
    }

    /// The sparse Jacobian of the currents and charges together over `cols`,
    /// split into the currents' rows and the charges'.
    fn split_coo(&self, ctx: &mut Graph, cols: &[(usize, SymbolId)]) -> (Coo, Coo) {
        let n = self.currents.len();
        let roots: Vec<ExprId> = self.currents.iter().chain(&self.charges).copied().collect();
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
        coo(ctx, &self.currents, &cols)
    }

    /// Parameter symbols: free symbols in the currents and charges that are
    /// neither unknowns nor time, sorted by id.
    pub fn params(&self, ctx: &Graph) -> Vec<SymbolId> {
        // The currents and charges, the delay times (parameters even though they appear
        // only in the delay registry) and the switching surfaces (which may
        // reference a parameter nothing else does), in one walk.
        let roots: Vec<ExprId> = self
            .currents
            .iter()
            .chain(&self.charges)
            .copied()
            .chain(self.delays.iter().map(|dl| dl.tau))
            .chain(self.events.iter().map(|ev| ev.g))
            .collect();
        let mut all = ctx.free_symbols_in(&roots);
        for s in &self.x {
            all.remove(s);
        }
        all.remove(&self.t);
        // history inputs are integrator-provided, not parameters
        for dl in &self.delays {
            all.remove(&dl.hist);
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
        // Build the symbol -> constant substitution. An empty `fold` is a no-op
        // (every node re-interns to itself), so the same path also serves the
        // "derive an independent copy" case.
        let map: rustc_hash::FxHashMap<SymbolId, ExprId> =
            fold.iter().map(|&(s, v)| (s, ctx.konst_f64(v))).collect();
        let fold = &map;
        let currents = rsdag::substitute(ctx, &self.currents, fold);
        let charges = rsdag::substitute(ctx, &self.charges, fold);
        let flat = self.observers.flatten(ctx);
        let noise = flat
            .noise
            .iter()
            .map(|n| NoiseSource {
                hi: n.hi,
                lo: n.lo,
                psd: rsdag::substitute(ctx, &[n.psd], fold)[0],
                flicker_exp: rsdag::substitute(ctx, &[n.flicker_exp], fold)[0],
                table: n.table.clone(),
            })
            .collect();
        let op_vars = flat
            .op_vars
            .iter()
            .map(|v| sane_device::OpVar {
                value: rsdag::substitute(ctx, &[v.value], fold)[0],
                ..v.clone()
            })
            .collect();
        Dae {
            currents,
            charges,
            n_nodes: self.n_nodes,
            param_defaults: self.param_defaults.clone(),
            events: self
                .events
                .iter()
                .map(|ev| EventSpec {
                    g: rsdag::substitute(ctx, &[ev.g], fold)[0],
                    ..ev.clone()
                })
                .collect(),
            delays: self
                .delays
                .iter()
                .map(|dl| DelaySpec {
                    tau: rsdag::substitute(ctx, &[dl.tau], fold)[0],
                    ..dl.clone()
                })
                .collect(),
            unknowns: self.unknowns.clone(),
            kinds: self.kinds.clone(),
            x: self.x.clone(),
            t: self.t,
            companion: self.companion.clone(),
            observers: Observers::flat(noise, op_vars),
            dc_seeds: self.dc_seeds.clone(),
            limits: self.limits.clone(),
            sources: self.sources.clone(),
            source_names: self.source_names.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// A sparse block as `(rows, cols, exprs)`.
pub type Coo = (Vec<usize>, Vec<usize>, Vec<ExprId>);

/// `rsdag::sparse_jacobian` of `rows` over the symbols of `cols`, each
/// column labelled with its paired index.
pub(crate) fn coo(ctx: &mut Graph, rows: &[ExprId], cols: &[(usize, SymbolId)]) -> Coo {
    let wrt: Vec<SymbolId> = cols.iter().map(|&(_, s)| s).collect();
    let mut out = Coo::default();
    for (i, row) in sparse_jacobian(ctx, rows, &wrt)
        .into_iter()
        .enumerate()
    {
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
