//! Exact symbolic sensitivity machinery on a [`Dae`]: directional derivatives,
//! forward-sensitivity augmentation, second-order (Hessian) blocks via the
//! Lagrangian, and the total small-signal matrix derivatives -- all by pure
//! symbolic differentiation, no finite differences anywhere.

use std::collections::HashMap;

use rsdag::{differentiate, ExprId, ReduceOp, SymbolId};
use sane_core::Graph;

use crate::{coo, sym2, Dae, UnknownKind};

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
    // what the derivative can be nonzero in, through calls exactly
    let fs = ctx.support_in(&[e]);
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

/// Augment the DAE with the forward-sensitivity systems for a set of
/// parameters. For each parameter `p`, appends `n` unknowns `s = dx/dp` whose
/// row is the (exact, symbolic) sensitivity equation, the derivative of
/// `I + d/dt Q` in `p`:
///
///   G s + dI/dp + d/dt (C s + dQ/dp) = 0,   G = dI/dx,   C = dQ/dx,
///
/// a current `G s + dI/dp` and a charge `C s + dQ/dp` along the trajectory.
/// Because the Jacobian entries and the parameter derivatives are kept
/// symbolic (in the original `x` and params), this is correct for ALL
/// parameters including reactive ones (a capacitor's `dQ/dC = v_a - v_b`).
/// Integrating the augmented system yields `x(t)` and every `dx/dp(t)`
/// together.
pub fn augment_with_sensitivities(ctx: &mut Graph, dae: &Dae, params: &[SymbolId]) -> Dae {
    let pairs: Vec<(SymbolId, f64)> = params.iter().map(|&p| (p, 1.0)).collect();
    augment_with_scaled_sensitivities(ctx, dae, &pairs)
}

/// [`augment_with_sensitivities`] with a per-parameter state scaling: appends
/// `u = k · dx/dp` (current `G u + k·dI/dp`, charge `C u + k·dQ/dp`). With
/// `k = |p0|` the appended states are log-parameter sensitivities
/// `~ dx/d ln p`, which live on the same magnitude scale as the circuit states
/// -- so a scalar `atol`/`rtol` error control over the augmented system stays
/// meaningful. (Raw `dx/dp` states can be ~1/p0 times larger than the circuit;
/// a scalar `atol` then crushes the step size at every zero crossing.) Callers
/// divide the resulting trajectories by `k` to recover `dx/dp`.
pub fn augment_with_scaled_sensitivities(
    ctx: &mut Graph,
    dae: &Dae,
    params: &[(SymbolId, f64)],
) -> Dae {
    let n = dae.dim();
    let (g, c) = dae.jacobian_iq_coo(ctx);
    let syms: Vec<SymbolId> = params.iter().map(|&(p, _)| p).collect();
    let (gp, cp) = dae.jacobian_p_iq_coo(ctx, &syms);

    let mut unknowns = dae.unknowns.clone();
    let mut kinds = dae.kinds.clone();
    let mut x = dae.x.clone();
    let mut currents = dae.currents.clone();
    let mut charges = dae.charges.clone();

    for (k, &(p, scale)) in params.iter().enumerate() {
        let pname = ctx.symbol_name(p).to_string();
        // Mint the sensitivity unknowns u_j.
        let (u, new_x): (Vec<ExprId>, Vec<SymbolId>) = (0..n)
            .map(|j| sym2(ctx, &format!("S[{pname}]{j}")))
            .unzip();
        unknowns.extend((dae.unknowns.iter()).map(|name| format!("d({name})/d({pname})")));
        // The current `G u + k dI/dp` and the charge `C u + k dQ/dp`.
        let scale = ctx.konst_f64(scale);
        let [i_u, q_u] = [(&g, &gp), (&c, &cp)].map(|((rows, cols, exprs), (prows, pcols, pexprs))| {
            let mut terms: Vec<Vec<ExprId>> = vec![Vec::new(); n];
            for ((&r, &col), &e) in rows.iter().zip(cols).zip(exprs) {
                terms[r].push(ctx.mul(e, u[col]));
            }
            for ((&r, &col), &e) in prows.iter().zip(pcols).zip(pexprs) {
                if col == k {
                    terms[r].push(ctx.mul(scale, e));
                }
            }
            (terms.into_iter())
                .map(|t| ctx.reduce(ReduceOp::Sum, t))
                .collect::<Vec<_>>()
        });
        kinds.extend(std::iter::repeat_n(UnknownKind::DeviceState, n));
        x.extend(new_x);
        currents.extend(i_u);
        charges.extend(q_u);
    }

    Dae {
        n_nodes: dae.n_nodes,
        param_defaults: dae.param_defaults.clone(),
        currents,
        charges,
        assertions: dae.assertions.clone(),
        structure: dae.structure.clone(),
        aliases: dae.aliases.clone(),
        unknowns,
        kinds,
        x,
        t: dae.t,
        events: dae.events.clone(),
        delays: dae.delays.clone(),
        companion: Vec::new(),
        observers: dae.observers.clone(),
        dc_seeds: dae.dc_seeds.clone(),
        limits: Vec::new(),
        sources: dae.sources.clone(),
        source_names: dae.source_names.clone(),
        labels: dae.labels.clone(),
    }
}

/// Exact total derivatives of the small-signal matrices w.r.t. a parameter,
/// **including the operating-point shift**: for `G = dI/dx`, `C = dQ/dx`, and
/// the input coupling `B = -dI/d(input)`,
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
    let ((gr, gc, ge), (cr, cc, ce)) = dae.jacobian_iq_coo(ctx);
    let dgs: Vec<ExprId> = ge
        .iter()
        .map(|&e| directional(ctx, e, s, &dae.x, p_sym))
        .collect();
    let dgv = rsdag::eval(ctx, &dgs, env);
    for k in 0..ge.len() {
        dg[gr[k]][gc[k]] = dgv[k];
    }
    let dcs: Vec<ExprId> = ce
        .iter()
        .map(|&e| directional(ctx, e, s, &dae.x, p_sym))
        .collect();
    let dcv = rsdag::eval(ctx, &dcs, env);
    for k in 0..ce.len() {
        dc[cr[k]][cc[k]] = dcv[k];
    }

    // B = -dI/d(input);  dB/dp = -directional(dI/d(input)).
    let mut db = vec![0.0; n];
    let mut db_rows = Vec::new();
    let mut db_exprs = Vec::new();
    for (r, &res) in dae.currents.iter().enumerate() {
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

/// The symbolic second-derivative blocks of the Lagrangian
/// `L = Σ_r λ_r I_r + μ_r Q_r` over the states and the parameters, with the
/// multipliers `λ_r` and `μ_r` minted as fresh input symbols. These blocks
/// depend only on the DAE structure (not on the operating point or the
/// multipliers' numeric values), so they are built once and compiled into a
/// reusable tape. At an operating point (`μ = 0`) the exact second-order-
/// adjoint Hessian is then
///
///   H_ij = -( s_iᵀ L_xx s_j + s_iᵀ L_{x,p_j} + s_jᵀ L_{x,p_i} + L_{p_i,p_j} ),
///
/// a numeric contraction of the evaluated blocks with the state
/// sensitivities; a periodic analysis weights the charges with the
/// adjoint's time derivative through `μ`.
pub struct HessianSym {
    /// Multiplier input symbols `λ_r` of the currents and `μ_r` of the
    /// charges, one each per row.
    pub lambda: Vec<SymbolId>,
    pub mu: Vec<SymbolId>,
    /// `L_xx` (state-state) as sparse `(rows, cols, exprs)`.
    pub xx: (Vec<usize>, Vec<usize>, Vec<ExprId>),
    /// `L_xp` (state-parameter), columns indexed like the `params` argument.
    pub xp: (Vec<usize>, Vec<usize>, Vec<ExprId>),
    /// `L_pp` (parameter-parameter).
    pub pp: (Vec<usize>, Vec<usize>, Vec<ExprId>),
}

/// Build the symbolic Lagrangian-Hessian blocks (see [`HessianSym`]) for the
/// given parameter set. One-time symbolic work; the result compiles to a tape
/// that turns every later Hessian into a leaf-value re-evaluation.
pub fn lagrangian_hessian(ctx: &mut Graph, dae: &Dae, params: &[SymbolId]) -> HessianSym {
    let n = dae.dim();

    // λ_r, μ_r input symbols and the Lagrangian L = Σ_r λ_r I_r + μ_r Q_r.
    let (mut lambda, mut mu) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut terms = Vec::with_capacity(2 * n);
    for (r, (&i, &q)) in dae.currents.iter().zip(&dae.charges).enumerate() {
        let (le, ls) = sym2(ctx, &format!("__lam{r}"));
        let (me, ms) = sym2(ctx, &format!("__mu{r}"));
        lambda.push(ls);
        mu.push(ms);
        terms.push(ctx.mul(le, i));
        if !ctx.is_zero(q) {
            terms.push(ctx.mul(me, q));
        }
    }
    let l = ctx.reduce(ReduceOp::Sum, terms);

    // The gradient of L over x and p in one reverse sweep (each entry linear
    // in the multipliers), then its Jacobians.
    let wrt: Vec<SymbolId> = dae.x.iter().chain(params).copied().collect();
    let g = rsdag::gradient(ctx, l, &wrt);
    let (gx, gp) = g.split_at(n);
    let x_col: Vec<(usize, SymbolId)> = dae.x.iter().copied().enumerate().collect();
    let p_col: Vec<(usize, SymbolId)> = params.iter().copied().enumerate().collect();
    let xx = coo(ctx, gx, &x_col);
    let xp = coo(ctx, gx, &p_col);
    let pp = coo(ctx, gp, &p_col);

    HessianSym {
        lambda,
        mu,
        xx,
        xp,
        pp,
    }
}
