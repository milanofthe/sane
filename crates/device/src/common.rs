//! Shared symbolic building blocks: the overflow-safe junction exponential
//! (also the lowering of Verilog-A `limexp`) and the smooth switch
//! conductance. Pure symbolic constructions over the core DAG.

use rsdag::{CmpOp, ExprId};
use sane_core::constants::EXP_VCRIT;
use sane_core::Graph;

/// Mint an instance-scoped parameter symbol, e.g. `W1.Ron`.
pub(crate) fn param(ctx: &mut Graph, instance: &str, name: &str) -> ExprId {
    ctx.sym(&format!("{instance}.{name}"))
}

/// `clamp(x, 0, 1)` built from two `Select`s.
pub(crate) fn clamp01(ctx: &mut Graph, x: ExprId) -> ExprId {
    let zero = ctx.zero();
    let one = ctx.one();
    let above0 = ctx.cmp(CmpOp::Gt, x, zero);
    let lo = ctx.select(above0, x, zero);
    let below1 = ctx.cmp(CmpOp::Lt, lo, one);
    ctx.select(below1, lo, one)
}

/// Switch conductance: the smooth transition of [`smooth_switch_g`] for a
/// positive half-width `vh`, a hard threshold at `center` for `vh = 0` (the
/// transient integrator lands on the threshold, see the device's events).
pub(crate) fn switch_g(
    ctx: &mut Graph,
    ctrl: ExprId,
    center: ExprId,
    vh: ExprId,
    g_on: ExprId,
    g_off: ExprId,
) -> ExprId {
    let zero = ctx.zero();
    let soft = ctx.cmp(CmpOp::Gt, vh, zero);
    let smooth = smooth_switch_g(ctx, ctrl, center, vh, g_on, g_off);
    let above = ctx.cmp(CmpOp::Gt, ctrl, center);
    let hard = ctx.select(above, g_on, g_off);
    ctx.select(soft, smooth, hard)
}

/// Smooth switch conductance: interpolates in log-conductance across the control
/// window `[center - vh, center + vh]` (a Hermite smoothstep), from `g_off`
/// (control below the window) to `g_on` (above). The half-width `vh` (SPICE-style
/// `VON = center + vh`, `VOFF = center - vh`) sets how soft the switch is; a small
/// `vh` approaches a hard threshold but stays differentiable, which Newton and
/// harmonic balance handle far better than a discontinuous step.
pub(crate) fn smooth_switch_g(
    ctx: &mut Graph,
    ctrl: ExprId,
    center: ExprId,
    vh: ExprId,
    g_on: ExprId,
    g_off: ExprId,
) -> ExprId {
    let two = ctx.konst_int(2);
    let lo = ctx.sub(center, vh);
    let num = ctx.sub(ctrl, lo);
    let two_vh = ctx.mul(two, vh);
    let q = ctx.div(num, two_vh);
    let u = clamp01(ctx, q);
    // smoothstep s = u^2 (3 - 2u): zero slope at both ends -> C1 transition.
    let u2 = ctx.pow_i(u, 2);
    let three = ctx.konst_int(3);
    let two_u = ctx.mul(two, u);
    let inner = ctx.sub(three, two_u);
    let s = ctx.mul(u2, inner);
    // log-conductance interpolation: G = exp(ln g_off + s (ln g_on - ln g_off)).
    let ln_on = ctx.ln(g_on);
    let ln_off = ctx.ln(g_off);
    let d = ctx.sub(ln_on, ln_off);
    let sd = ctx.mul(s, d);
    let lng = ctx.add(ln_off, sd);
    ctx.exp(lng)
}

/// Overflow-safe junction exponential. For `arg <= EXP_VCRIT` it is exactly
/// `exp(arg)`; above the knot it continues along the tangent line
/// `exp(vc)*(1 + (arg - vc))`, which is C1-continuous at `vc` and grows only
/// linearly, so the integrator's / Newton's trial steps cannot overflow `exp`.
/// Physical operating points sit well below the knot, so DC results are
/// unchanged. This is the smooth analogue of SPICE junction voltage limiting.
/// Public because the Verilog-A frontend lowers `limexp` to exactly this
/// construction.
pub fn safe_exp(ctx: &mut Graph, arg: ExprId) -> ExprId {
    let vc = ctx.konst_f64(EXP_VCRIT);
    let e_vc = ctx.exp(vc); // constant exp(vcrit)
    let one = ctx.one();
    let d = ctx.sub(arg, vc); // arg - vc
    let one_plus_d = ctx.add(one, d);
    let lin = ctx.mul(e_vc, one_plus_d); // exp(vc)*(1 + arg - vc)
    let e = ctx.exp(arg);
    let over = ctx.cmp(CmpOp::Gt, arg, vc);
    ctx.select(over, lin, e)
}
