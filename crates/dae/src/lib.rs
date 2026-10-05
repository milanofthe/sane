//! Time-domain nonlinear DAE assembly and extraction.
//!
//! Builds the implicit residual system `F(x, x', t) = 0` for a circuit, where
//! `x = [node voltages; branch currents]` and `x'` are companion time-derivative
//! symbols. Dynamics enter through derivative symbols: a capacitor contributes
//! `C*(v'_a - v'_b)`, an inductor adds a branch current with the constraint
//! `v_a - v_b - L*i' = 0`. Nonlinear devices contribute their large-signal
//! terminal currents. Nothing is solved here; the output is the symbolic DAE
//! plus its analytic Jacobians, ready for export.
//!
//! Naming convention for the minted symbols: node voltages `v{n}`, their
//! derivatives `vdot{n}`, branch currents `i_{name}`, inductor current
//! derivatives `idot_{name}`, and time `t`.

use std::collections::HashMap;

use rsdag::determinant;
use rsdag::{differentiate, sparse_jacobian, Crossing, ExprId, Node, SymbolId};
use sane_core::Graph;
use sane_mna::SourceFn;

// Assembly of the symbolic DAE from a parsed circuit.
mod assemble;
// Subcircuit instances: placed bodies, their renaming and topology.
mod hierarchy;
pub mod linearize;
// Sensitivity machinery: directional derivatives, augmentation, Hessian blocks.
mod sens;
// Graph transformations (shorts / opens / exact node elimination).
#[cfg(test)]
mod tests;
mod transform;

pub use assemble::{assemble, assemble_dae};
pub use hierarchy::{topology, Instance};
pub use sens::{
    ac_param_derivatives, augment_with_scaled_sensitivities, augment_with_sensitivities, hessian,
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

/// An extracted differential-algebraic system `F(x, x', t) = 0`.
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

pub struct Dae {
    /// Residual expressions, one per unknown (`F_i = 0`).
    pub residuals: Vec<ExprId>,
    /// Names of the unknowns `x`, aligned with `x` / `xdot`.
    pub unknowns: Vec<String>,
    /// What each unknown physically is, parallel to `unknowns` (see
    /// [`UnknownKind`]); the solver's tolerances and shunts are taken per kind.
    pub kinds: Vec<UnknownKind>,
    /// Unknown symbols `x`.
    pub x: Vec<SymbolId>,
    /// Derivative symbol per unknown (`None` for purely algebraic unknowns).
    pub xdot: Vec<Option<SymbolId>>,
    /// The time symbol.
    pub t: SymbolId,
    /// Number of leading node-voltage unknowns (their KCL rows lead `residuals`).
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
    /// Small-signal noise sources emitted by behavioral / Verilog-A devices
    /// (current noise generators with a PSD expression). Empty otherwise.
    pub noise_sources: Vec<NoiseSource>,
    /// Operating-point variables exported by behavioral devices (`(* desc *)`
    /// annotations): named expressions evaluated at a solved point for OP
    /// reporting. They are read-only observers -- no residual references them.
    pub op_vars: Vec<sane_device::OpVar>,
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
        ctx: &Graph,
        env: &HashMap<SymbolId, f64>,
    ) -> Vec<Option<(f64, f64)>> {
        let roots: Vec<ExprId> = self
            .noise_sources
            .iter()
            .filter(|ns| ns.table.is_empty())
            .flat_map(|ns| [ns.psd, ns.flicker_exp])
            .collect();
        let vals = rsdag::eval(ctx, &roots, env);
        let mut pairs = vals.as_chunks::<2>().0.iter();
        self.noise_sources
            .iter()
            .map(|ns| {
                let &[psd, fexp] = ns.table.is_empty().then(|| pairs.next())??;
                Some((psd, fexp))
            })
            .collect()
    }

    /// Register the system as an rsdag function carrying its roles: the
    /// unknowns as `State`, their derivatives as `StateDot`, time as `Time`,
    /// every parameter as `Param`, the residuals as `Residual` outputs and
    /// every switching surface as a `Guard` output with its crossing
    /// direction.
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
        for (i, sd) in self.xdot.iter().enumerate() {
            if let Some(s) = *sd {
                params.push(s);
                roles.push(rsdag::ParamRole::StateDot { id: i as u32 });
            }
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

        let mut outputs: Vec<ExprId> = self.residuals.clone();
        let mut out_roles: Vec<rsdag::OutputRole> = (0..self.residuals.len())
            .map(|i| rsdag::OutputRole::Residual { id: i as u32 })
            .collect();
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
        self.residuals.len()
    }

    /// Classify the residual system's nonlinearity in the unknowns `x` and their
    /// derivatives `xdot` -- the graph fact a harmonic-balance solve reads to
    /// pick its harmonic count and time sampling (degree, transcendental /
    /// piecewise / opaque flags; see [`rsdag::nonlinearity_of`]). `x` and
    /// `xdot` share the same bandwidth `K` (differentiation only scales each
    /// harmonic by `jkw0`), so a term like `x*xdot` is a degree-2 nonlinearity
    /// over the combined variable set, exactly the harmonic-generating order.
    pub fn nonlinearity(&self, ctx: &Graph) -> rsdag::Nonlinearity {
        let vars: std::collections::BTreeSet<SymbolId> = self
            .x
            .iter()
            .copied()
            .chain(self.xdot.iter().flatten().copied())
            .collect();
        rsdag::nonlinearity_of(ctx, &self.residuals, &vars)
    }

    /// Jacobian `∂F/∂x`, dense (zero where a residual does not depend on
    /// an unknown).
    pub fn jacobian_x(&self, ctx: &mut Graph) -> Vec<Vec<ExprId>> {
        let rows = sparse_jacobian(ctx, &self.residuals, &self.x);
        let zero = ctx.zero();
        rows.into_iter()
            .map(|row| {
                let mut dense = vec![zero; self.x.len()];
                for (j, e) in row {
                    dense[j] = e;
                }
                dense
            })
            .collect()
    }

    /// Jacobian `∂F/∂x'`, dense (zero columns for algebraic unknowns).
    pub fn jacobian_xdot(&self, ctx: &mut Graph) -> Vec<Vec<ExprId>> {
        let (rows, cols, exprs) = self.jacobian_xdot_coo(ctx);
        let zero = ctx.zero();
        let mut dense = vec![vec![zero; self.xdot.len()]; self.residuals.len()];
        for ((r, c), e) in rows.into_iter().zip(cols).zip(exprs) {
            dense[r][c] = e;
        }
        dense
    }

    /// The columns of `∂F/∂x`: `(index in x, unknown)`.
    fn x_cols(&self) -> Vec<(usize, SymbolId)> {
        self.x.iter().copied().enumerate().collect()
    }

    /// The columns of `∂F/∂x'`: `(index in x, derivative)`, the
    /// differential unknowns only.
    fn xdot_cols(&self) -> Vec<(usize, SymbolId)> {
        self.xdot
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.map(|s| (i, s)))
            .collect()
    }

    /// Sparse `∂F/∂x` as `(rows, cols, exprs)`, columns indexed like `x`.
    pub fn jacobian_x_coo(&self, ctx: &mut Graph) -> Coo {
        coo(ctx, &self.residuals, &self.x_cols())
    }

    /// Sparse `∂F/∂x'` as `(rows, cols, exprs)`, columns indexed like `x`
    /// (only differential unknowns contribute).
    pub fn jacobian_xdot_coo(&self, ctx: &mut Graph) -> Coo {
        coo(ctx, &self.residuals, &self.xdot_cols())
    }

    /// [`jacobian_x_coo`](Self::jacobian_x_coo) and
    /// [`jacobian_xdot_coo`](Self::jacobian_xdot_coo) from one sparse Jacobian
    /// over the unknowns and their derivatives together.
    pub fn jacobian_x_xdot_coo(&self, ctx: &mut Graph) -> (Coo, Coo) {
        let mut cols = self.x_cols();
        let nx = cols.len();
        cols.extend(self.xdot_cols());
        let wrt: Vec<SymbolId> = cols.iter().map(|&(_, s)| s).collect();
        let (mut x, mut xd) = (Coo::default(), Coo::default());
        for (i, row) in sparse_jacobian(ctx, &self.residuals, &wrt)
            .into_iter()
            .enumerate()
        {
            for (j, e) in row {
                let block = if j < nx { &mut x } else { &mut xd };
                block.0.push(i);
                block.1.push(cols[j].0);
                block.2.push(e);
            }
        }
        (x, xd)
    }

    /// Sparse `∂F/∂hist` as `(rows, cols, exprs)`, columns indexed like
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
        coo(ctx, &self.residuals, &cols)
    }

    /// Sparse `∂F/∂p` as `(rows, cols, exprs)`, columns indexed like
    /// `params` (a subset of the parameters builds only those columns). The
    /// basis for exact (adjoint) component sensitivity analysis.
    pub fn jacobian_p_coo(&self, ctx: &mut Graph, params: &[SymbolId]) -> Coo {
        coo(
            ctx,
            &self.residuals,
            &params.iter().copied().enumerate().collect::<Vec<_>>(),
        )
    }

    /// Parameter symbols: free symbols in the residuals that are neither
    /// unknowns nor derivatives nor time, sorted by id.
    pub fn params(&self, ctx: &Graph) -> Vec<SymbolId> {
        // The residuals, the delay times (parameters even though they appear
        // only in the delay registry) and the switching surfaces (which may
        // reference a parameter nothing else does), in one walk.
        let roots: Vec<ExprId> = self
            .residuals
            .iter()
            .copied()
            .chain(self.delays.iter().map(|dl| dl.tau))
            .chain(self.events.iter().map(|ev| ev.g))
            .collect();
        let mut all = ctx.free_symbols_in(&roots);
        for s in &self.x {
            all.remove(s);
        }
        for s in self.xdot.iter().flatten() {
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
    /// every residual, stamp and noise PSD. Because substitution rebuilds through
    /// the smart constructors, the now-constant subexpressions collapse; the folded
    /// symbols are no longer free, so they drop out of [`params`](Self::params).
    /// The unknown structure (`x`/`xdot`/`t`/`unknowns`) and the numeric companion
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
        let residuals = self
            .residuals
            .iter()
            .map(|&r| rsdag::substitute(ctx, &[r], fold)[0])
            .collect();
        let noise_sources = self
            .noise_sources
            .iter()
            .map(|n| NoiseSource {
                hi: n.hi,
                lo: n.lo,
                psd: rsdag::substitute(ctx, &[n.psd], fold)[0],
                flicker_exp: rsdag::substitute(ctx, &[n.flicker_exp], fold)[0],
                table: n.table.clone(),
            })
            .collect();
        let op_vars = self
            .op_vars
            .iter()
            .map(|v| sane_device::OpVar {
                value: rsdag::substitute(ctx, &[v.value], fold)[0],
                ..v.clone()
            })
            .collect();
        Dae {
            residuals,
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
            xdot: self.xdot.clone(),
            t: self.t,
            companion: self.companion.clone(),
            noise_sources,
            op_vars,
            dc_seeds: self.dc_seeds.clone(),
            limits: self.limits.clone(),
            sources: self.sources.clone(),
            source_names: self.source_names.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// Small-signal AC system matrix `A(s) = dF/dx + s * dF/dx'` -- the Laplace-domain
/// linearisation of the DAE at its operating point. For a linear circuit this is
/// the usual `G + sC` MNA matrix; for a nonlinear one the entries are the
/// symbolic small-signal stamps (gm, gds, junction caps) in terms of the
/// operating-point unknowns and device parameters.
pub fn small_signal_matrix(ctx: &mut Graph, dae: &Dae) -> Vec<Vec<ExprId>> {
    let s = ctx.sym("s");
    let jx = dae.jacobian_x(ctx);
    let jxd = dae.jacobian_xdot(ctx);
    let n = dae.dim();
    let zero = ctx.zero();
    let mut a = vec![vec![zero; n]; n];
    for i in 0..n {
        for j in 0..n {
            let s_jxd = ctx.mul(s, jxd[i][j]);
            a[i][j] = ctx.add(jx[i][j], s_jxd);
        }
    }
    a
}

/// Small-signal transfer function `H(s) = X(output) / In`, where `input` is the
/// name of an independent-source parameter (the AC excitation) and `output` is
/// an unknown name (e.g. `"v2"`). Solved by Cramer's rule on `A(s)`. Returns
/// `None` if the output unknown is not found.
pub fn small_signal_transfer(
    ctx: &mut Graph,
    dae: &Dae,
    input: &str,
    output: &str,
) -> Option<ExprId> {
    let (num, den) = small_signal_transfer_nd(ctx, dae, input, output)?;
    Some(ctx.div(num, den))
}

/// Like [`small_signal_transfer`] but returns the numerator and denominator
/// determinants `(N(s), D(s))` separately (both polynomials in `s` with symbolic
/// coefficients), for symbolic term-pruning model reduction.
pub fn small_signal_transfer_nd(
    ctx: &mut Graph,
    dae: &Dae,
    input: &str,
    output: &str,
) -> Option<(ExprId, ExprId)> {
    let col = dae.unknowns.iter().position(|u| u == output)?;
    let (_, input_sym) = sym2(ctx, input);

    // Excitation vector b = -dF/d(input): the source moved to the RHS.
    let mut b = Vec::with_capacity(dae.dim());
    for &r in &dae.residuals {
        let d = differentiate(ctx, r, input_sym);
        b.push(ctx.neg(d));
    }

    let a = small_signal_matrix(ctx, dae);
    let det_a = determinant(ctx, &a);
    let mut a_b = a.clone();
    for (row, b_row) in b.iter().enumerate() {
        a_b[row][col] = *b_row;
    }
    let det_b = determinant(ctx, &a_b);
    Some((det_b, det_a))
}

/// A sparse block as `(rows, cols, exprs)`.
pub type Coo = (Vec<usize>, Vec<usize>, Vec<ExprId>);

/// `rsdag::sparse_jacobian` of `residuals` over the symbols of `cols`, each
/// column labelled with its paired index.
pub(crate) fn coo(ctx: &mut Graph, residuals: &[ExprId], cols: &[(usize, SymbolId)]) -> Coo {
    let wrt: Vec<SymbolId> = cols.iter().map(|&(_, s)| s).collect();
    let mut out = Coo::default();
    for (i, row) in sparse_jacobian(ctx, residuals, &wrt)
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
