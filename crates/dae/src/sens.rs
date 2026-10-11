//! Exact symbolic sensitivity machinery on a [`Dae`]: the second-order
//! (Hessian) blocks of the Lagrangian, the frozen Jacobians a linearly
//! implicit step differentiates, and parameter columns -- all by symbolic
//! differentiation, no finite differences.

use rsdag::{ExprId, ReduceOp, SymbolId};
use sane_core::Graph;

use crate::{coo, sym2, Dae};

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

/// The Jacobians and time rates a linearly implicit method freezes at its
/// step start, as entries: `dI/dx` and `dQ/dx` (the charges' rows offset by
/// `n`), then `dI/dt` and `dQ/dt` (in column `n`). A stage takes them along
/// its direction `w = (v, τ)`, `E_r = Σ entry_k w[col_k]` over the entries of
/// row `r`; how that moves with the start is `dE/dx = Σ d(entry_k)/dx
/// w[col_k]`, from the entries' gradients by the states, with a parameter
/// from their [`param_column`]. The gradients are the step's, whatever the
/// stage: one evaluation per step, the directions contracted after.
pub struct Frozen {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub entries: Vec<ExprId>,
    /// The entries' gradients by the states: `(entry, column, expression)`.
    pub x: (Vec<usize>, Vec<usize>, Vec<ExprId>),
}

/// Build [`Frozen`].
pub fn frozen(ctx: &mut Graph, dae: &Dae) -> Frozen {
    let n = dae.dim();
    let ((ir, ic, ie), (qr, qc, qe)) = dae.jacobian_iq_coo(ctx);
    let ((tir, _, tie), (tqr, _, tqe)) = dae.jacobian_t_iq_coo(ctx);
    let mut rows: Vec<usize> = ir;
    rows.extend(qr.iter().map(|&r| n + r));
    rows.extend_from_slice(&tir);
    rows.extend(tqr.iter().map(|&r| n + r));
    let mut cols: Vec<usize> = ic;
    cols.extend_from_slice(&qc);
    cols.extend(std::iter::repeat_n(n, tie.len() + tqe.len()));
    let entries: Vec<ExprId> = ie.into_iter().chain(qe).chain(tie).chain(tqe).collect();
    let x_cols: Vec<(usize, SymbolId)> = dae.x.iter().copied().enumerate().collect();
    let x = coo(ctx, &entries, &x_cols);
    Frozen {
        rows,
        cols,
        entries,
        x,
    }
}

/// The derivative of `rows` by the parameter `p`: the rows it moves, and
/// how. A parameter enters the rows of its own elements, so the column is
/// as sparse as they are.
pub fn param_column(ctx: &mut Graph, rows: &[ExprId], p: SymbolId) -> (Vec<usize>, Vec<ExprId>) {
    let (mut r, mut e) = (Vec::new(), Vec::new());
    for (i, row) in rsdag::sparse_jacobian(ctx, rows, &[p])
        .into_iter()
        .enumerate()
    {
        for (_, d) in row {
            r.push(i);
            e.push(d);
        }
    }
    (r, e)
}
