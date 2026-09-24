//! Exact symbolic sensitivity machinery on a [`Dae`]: directional derivatives,
//! forward-sensitivity augmentation, second-order (Hessian) blocks via the
//! Lagrangian, and the total small-signal matrix derivatives -- all by pure
//! symbolic differentiation, no finite differences anywhere.

use std::collections::HashMap;

use rsdag::{differentiate, ExprId, ReduceOp, SymbolId};
use sane_core::Graph;

use crate::{coo, stamp, sym2, Dae, UnknownKind};

/// Directional derivative `D_d e = sum_l d_x[l]*de/dx_l + de/dp` along the
/// combined `(x, p)` direction `d = (d_x, e_p)` (the unknowns move along the
/// numeric vector `d_x`, plus a unit step in the parameter `p_sym`). Numeric
/// directions are baked in as exact rational constants, so nesting this gives
/// exact second (and higher) directional derivatives -- pure AD, no differences.
fn directional(
    ctx: &mut Graph,
    e: ExprId,
    d_x: &[f64],
    x_syms: &[SymbolId],
    p_sym: SymbolId,
) -> ExprId {
    let fs = ctx.free_symbols(e);
    let mut acc = ctx.zero();
    for (l, &xs) in x_syms.iter().enumerate() {
        if d_x[l] != 0.0 && fs.contains(&xs) {
            let d = differentiate(ctx, e, xs);
            if !ctx.is_zero(d) {
                let c = ctx.konst_f64(d_x[l]);
                let term = ctx.mul(c, d);
                acc = ctx.add(acc, term);
            }
        }
    }
    if fs.contains(&p_sym) {
        let dp = differentiate(ctx, e, p_sym);
        acc = ctx.add(acc, dp);
    }
    acc
}

/// Exact second-order sensitivity (Hessian) of the metric `y = c^T x` w.r.t. a
/// subset of parameters, by the second-order adjoint identity
///
///   H_ij = -lambda^T B_ij,   B_ij = D_i D_j F,
///
/// where `D_k` is the directional derivative of the residual along
/// `d_k = (s_k, e_{p_k})` (`s_k = dx/dp_k` the state sensitivity), and
/// `(dF/dx)^T lambda = c`. Both `B_ij` (via nested [`directional`]) and the
/// final values are computed by **exact symbolic differentiation** -- no finite
/// differences anywhere. `subset` is `(param_symbol, s_k)`; `env` evaluates all
/// symbols at the operating point. Returns the dense `|subset| x |subset|`
/// Hessian (symmetric).
pub fn hessian(
    ctx: &mut Graph,
    dae: &Dae,
    lambda: &[f64],
    subset: &[(SymbolId, Vec<f64>)],
    env: &HashMap<SymbolId, f64>,
) -> Vec<Vec<f64>> {
    let k = subset.len();
    // G[i][r] = D_i F_r  (first directional derivative along d_i).
    let g: Vec<Vec<ExprId>> = subset
        .iter()
        .map(|(p_sym, s)| {
            dae.residuals
                .iter()
                .map(|&r| directional(ctx, r, s, &dae.x, *p_sym))
                .collect()
        })
        .collect();

    let mut h = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in i..k {
            let (p_sym_j, s_j) = &subset[j];
            // B_ij[r] = D_j G_i[r]; H_ij = -sum_r lambda_r * B_ij[r](op).
            let b: Vec<ExprId> = g[i]
                .iter()
                .map(|&gi| directional(ctx, gi, s_j, &dae.x, *p_sym_j))
                .collect();
            let vals = rsdag::eval(ctx, &b, env);
            let hij: f64 = -lambda.iter().zip(&vals).map(|(l, v)| l * v).sum::<f64>();
            h[i][j] = hij;
            h[j][i] = hij;
        }
    }
    h
}

/// Augment the DAE with the forward-sensitivity systems for a set of
/// parameters. For each parameter `p`, appends `n` unknowns `s = dx/dp` whose
/// residual is the (exact, symbolic) sensitivity equation
///
///   sum_j (dF_i/dx_j) s_j + sum_j (dF_i/dx'_j) s'_j + dF_i/dp = 0,
///
/// i.e. `G s + C s' + dF/dp = 0` along the trajectory. Because `dF/dp` and the
/// Jacobian entries are kept symbolic (in the original `x`, `x'`, params), this
/// is correct for ALL parameters including reactive ones (a capacitor's
/// `dF/dC = (x'_a - x'_b)` couples the state derivative automatically via the
/// existing `vdot` symbols). Integrating the augmented system yields `x(t)` and
/// every `dx/dp(t)` together. The new unknowns carry derivative symbols exactly
/// where the corresponding state unknown does.
pub fn augment_with_sensitivities(ctx: &mut Graph, dae: &Dae, params: &[SymbolId]) -> Dae {
    let pairs: Vec<(SymbolId, f64)> = params.iter().map(|&p| (p, 1.0)).collect();
    augment_with_scaled_sensitivities(ctx, dae, &pairs)
}

/// [`augment_with_sensitivities`] with a per-parameter state scaling: appends
/// `u = k · dx/dp` (residual `G u + C u' + k·dF/dp = 0`). With `k = |p0|` the
/// appended states are log-parameter sensitivities `~ dx/d ln p`, which live on
/// the same magnitude scale as the circuit states -- so a scalar
/// `atol`/`rtol` error control over the augmented system stays meaningful.
/// (Raw `dx/dp` states can be ~1/p0 times larger than the circuit; a scalar
/// `atol` then crushes the step size at every zero crossing.) Callers divide
/// the resulting trajectories by `k` to recover `dx/dp`.
pub fn augment_with_scaled_sensitivities(
    ctx: &mut Graph,
    dae: &Dae,
    params: &[(SymbolId, f64)],
) -> Dae {
    let n = dae.dim();
    let (jr, jc, je) = dae.jacobian_x_coo(ctx);
    let (xr, xc, xe) = dae.jacobian_xdot_coo(ctx);

    let mut unknowns = dae.unknowns.clone();

    let mut kinds = dae.kinds.clone();
    let mut x = dae.x.clone();
    let mut xdot = dae.xdot.clone();
    let mut residuals = dae.residuals.clone();

    for &(p, scale) in params {
        let pname = ctx.symbol_name(p).to_string();
        // Mint the sensitivity unknowns s_j (and derivatives where x_j is
        // differential).
        let mut s_e = Vec::with_capacity(n);
        let mut sdot_e = vec![ctx.zero(); n];
        let zero = ctx.zero();
        let mut new_x = Vec::with_capacity(n);
        let mut new_xdot = Vec::with_capacity(n);
        let mut new_names = Vec::with_capacity(n);
        for j in 0..n {
            let (e, sy) = sym2(ctx, &format!("S[{pname}]{j}"));
            s_e.push(e);
            new_x.push(sy);
            new_names.push(format!("d({})/d({pname})", dae.unknowns[j]));
            if dae.xdot[j].is_some() {
                let (de, dsy) = sym2(ctx, &format!("Sdot[{pname}]{j}"));
                sdot_e[j] = de;
                new_xdot.push(Some(dsy));
            } else {
                new_xdot.push(None);
            }
        }
        // Sensitivity residual: G u + C u' + scale·dF/dp.
        let scale_e = ctx.konst_f64(scale);
        let mut res = vec![zero; n];
        for k in 0..je.len() {
            let term = ctx.mul(je[k], s_e[jc[k]]);
            res[jr[k]] = ctx.add(res[jr[k]], term);
        }
        for k in 0..xe.len() {
            let term = ctx.mul(xe[k], sdot_e[xc[k]]);
            res[xr[k]] = ctx.add(res[xr[k]], term);
        }
        for i in 0..n {
            let dfp = differentiate(ctx, dae.residuals[i], p);
            let sdfp = ctx.mul(scale_e, dfp);
            res[i] = ctx.add(res[i], sdfp);
        }
        kinds.extend(std::iter::repeat_n(UnknownKind::DeviceState, new_x.len()));
        unknowns.extend(new_names);
        x.extend(new_x);
        xdot.extend(new_xdot);
        residuals.extend(res);
    }

    let stamps = stamp::stamps_from_residuals(ctx, &residuals);
    Dae {
        n_nodes: dae.n_nodes,
        param_defaults: dae.param_defaults.clone(),
        residuals,
        unknowns,
        kinds,
        x,
        xdot,
        t: dae.t,
        events: dae.events.clone(),
        delays: dae.delays.clone(),
        stamps,
        companion: Vec::new(),
        noise_sources: Vec::new(),
        op_vars: dae.op_vars.clone(),
        dc_seeds: dae.dc_seeds.clone(),
        limits: Vec::new(),
        sources: dae.sources.clone(),
        source_names: dae.source_names.clone(),
    }
}

/// Exact total derivatives of the small-signal matrices w.r.t. a parameter,
/// **including the operating-point shift**: for `G = dF/dx`, `C = dF/dx'`, and
/// the input coupling `B = -dF/d(input)`,
///
///   dG/dp = ∂G/∂p + (∂G/∂x)·s,   dC/dp = ∂C/∂p + (∂C/∂x)·s,   dB/dp likewise,
///
/// where `s = dx/dp` is the state sensitivity. Each entry is the AD directional
/// derivative ([`directional`]) of the corresponding symbolic Jacobian entry
/// along `d = (s, e_p)`, evaluated at the operating point `env`. No finite
/// differences. Returns dense `(dG, dC, dB)`.
pub fn ac_param_derivatives(
    ctx: &mut Graph,
    dae: &Dae,
    input_sym: SymbolId,
    p_sym: SymbolId,
    s: &[f64],
    env: &HashMap<SymbolId, f64>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let n = dae.dim();
    let mut dg = vec![vec![0.0; n]; n];
    let mut dc = vec![vec![0.0; n]; n];

    // Build the directional-derivative expressions up front (mutating the
    // arena), then evaluate each block in a single arena sweep rather than one
    // full sweep per nonzero.
    let (gr, gc, ge) = dae.jacobian_x_coo(ctx);
    let dgs: Vec<ExprId> = ge
        .iter()
        .map(|&e| directional(ctx, e, s, &dae.x, p_sym))
        .collect();
    let dgv = rsdag::eval(ctx, &dgs, env);
    for k in 0..ge.len() {
        dg[gr[k]][gc[k]] = dgv[k];
    }
    let (cr, cc, ce) = dae.jacobian_xdot_coo(ctx);
    let dcs: Vec<ExprId> = ce
        .iter()
        .map(|&e| directional(ctx, e, s, &dae.x, p_sym))
        .collect();
    let dcv = rsdag::eval(ctx, &dcs, env);
    for k in 0..ce.len() {
        dc[cr[k]][cc[k]] = dcv[k];
    }

    // B = -dF/d(input);  dB/dp = -directional(dF/d(input)).
    let mut db = vec![0.0; n];
    let mut db_rows = Vec::new();
    let mut db_exprs = Vec::new();
    for (r, &res) in dae.residuals.iter().enumerate() {
        let bexpr = differentiate(ctx, res, input_sym);
        if !ctx.is_zero(bexpr) {
            db_rows.push(r);
            db_exprs.push(directional(ctx, bexpr, s, &dae.x, p_sym));
        }
    }
    let dbv = rsdag::eval(ctx, &db_exprs, env);
    for (j, &r) in db_rows.iter().enumerate() {
        db[r] = -dbv[j];
    }
    (dg, dc, db)
}

/// The symbolic second-derivative blocks of the Lagrangian `L = Σ_r λ_r F_r`,
/// with the multipliers `λ_r` minted as fresh input symbols. These blocks depend
/// only on the DAE structure (not on the operating point or the multipliers'
/// numeric values), so they are built once and compiled into a reusable tape;
/// the exact second-order-adjoint Hessian is then
///
///   H_ij = -( s_iᵀ L_xx s_j + s_iᵀ L_{x,p_j} + s_jᵀ L_{x,p_i} + L_{p_i,p_j} ),
///
/// a numeric contraction of the evaluated blocks with the state sensitivities.
pub struct HessianSym {
    /// Multiplier input symbols `λ_r`, one per residual (aligned with `x`).
    pub lambda: Vec<SymbolId>,
    /// `L_xx` (state-state) as sparse `(rows, cols, exprs)`.
    pub xx: (Vec<usize>, Vec<usize>, Vec<ExprId>),
    /// `L_xp` (state-parameter), columns indexed like the `params` argument.
    pub xp: (Vec<usize>, Vec<usize>, Vec<ExprId>),
    /// `L_pp` (parameter-parameter).
    pub pp: (Vec<usize>, Vec<usize>, Vec<ExprId>),
    /// `L_{x'x'}` (rate-rate); rows and cols indexed like `x` (state index).
    /// Empty entries where a state has no `x'` symbol. Needed for the exact
    /// second-order sensitivity of any analysis whose unknowns drive `x'`
    /// (e.g. harmonic balance with nonlinear charge storage); the DC Hessian,
    /// where `x' = 0` and the state sensitivity has no rate component, never
    /// touches these blocks.
    pub xdxd: (Vec<usize>, Vec<usize>, Vec<ExprId>),
    /// `L_{x x'}` (state-rate): rows indexed like `x`, cols like `x'` (state index).
    pub xxd: (Vec<usize>, Vec<usize>, Vec<ExprId>),
    /// `L_{x' p}` (rate-parameter): rows like `x'` (state index), cols like `params`.
    pub xdp: (Vec<usize>, Vec<usize>, Vec<ExprId>),
}

/// Build the symbolic Lagrangian-Hessian blocks (see [`HessianSym`]) for the
/// given parameter set. One-time symbolic work; the result compiles to a tape
/// that turns every later Hessian into a leaf-value re-evaluation.
pub fn lagrangian_hessian(ctx: &mut Graph, dae: &Dae, params: &[SymbolId]) -> HessianSym {
    let n = dae.dim();

    // λ_r input symbols and the Lagrangian L = Σ_r λ_r F_r.
    let mut lambda = Vec::with_capacity(n);
    let mut terms = Vec::with_capacity(n);
    for (r, &f) in dae.residuals.iter().enumerate() {
        let (le, ls) = sym2(ctx, &format!("__lam{r}"));
        lambda.push(ls);
        terms.push(ctx.mul(le, f));
    }
    let l = ctx.reduce(ReduceOp::Sum, terms);

    // Gradients ∂L/∂x, ∂L/∂x' and ∂L/∂p (each linear in λ). The rate gradient is
    // kept aligned with the state index: a zero where a state has no `x'` symbol.
    let x_syms = dae.x.clone();
    let zero = ctx.zero();
    let gx: Vec<ExprId> = x_syms.iter().map(|&xm| differentiate(ctx, l, xm)).collect();
    let gxd: Vec<ExprId> = dae
        .xdot
        .iter()
        .map(|o| match o {
            Some(s) => differentiate(ctx, l, *s),
            None => zero,
        })
        .collect();
    let gp: Vec<ExprId> = params.iter().map(|&pi| differentiate(ctx, l, pi)).collect();

    // Second-derivative blocks (sparse): differentiate the gradients again.
    let x_col: HashMap<SymbolId, usize> = x_syms.iter().enumerate().map(|(c, &s)| (s, c)).collect();
    // Rate columns, also indexed by state index (so the contraction can share
    // the same flattening as `x`).
    let xd_col: HashMap<SymbolId, usize> = dae
        .xdot
        .iter()
        .enumerate()
        .filter_map(|(i, o)| o.map(|s| (s, i)))
        .collect();
    let p_col: HashMap<SymbolId, usize> = params.iter().enumerate().map(|(c, &s)| (s, c)).collect();
    let xx = coo(ctx, &gx, &x_col);
    let xp = coo(ctx, &gx, &p_col);
    let pp = coo(ctx, &gp, &p_col);
    let xdxd = coo(ctx, &gxd, &xd_col);
    let xxd = coo(ctx, &gx, &xd_col);
    let xdp = coo(ctx, &gxd, &p_col);

    HessianSym {
        lambda,
        xx,
        xp,
        pp,
        xdxd,
        xxd,
        xdp,
    }
}
