//! Shared lowering of elementary math functions onto the symbolic DAG.
//!
//! One canonical construction per function name, used by every expression
//! frontend (the Verilog-A lowering and the netlist behavioral-source
//! translator), so `abs`/`min`/`atan2`/... lower to the *same* graph shape
//! everywhere instead of each frontend keeping its own drifting copy.

use crate::Graph;
use rsdag::{CmpOp, ExprId};

/// Division guarded against a constant-zero denominator (avoids `recip(0)`
/// panicking when an intermediate folds at a domain edge, e.g. `asin(1)`;
/// runtime symbolic denominators are exact).
fn gdiv(ctx: &mut Graph, a: ExprId, b: ExprId) -> ExprId {
    if ctx.is_zero(b) {
        ctx.zero()
    } else {
        ctx.div(a, b)
    }
}

/// Lower a call to an elementary math function over already-lowered arguments.
/// Returns `None` for an unknown name/arity combination (the caller decides how
/// to report that). Functions with a native [`crate::node::UnaryOp`] map to it
/// directly; the rest are built from the primitive ops with well-defined
/// subgradients (`abs`/`min`/`max` via `Cmp` + `Select`), so autodiff stays
/// consistent everywhere.
pub fn lower_math_call(ctx: &mut Graph, name: &str, a: &[ExprId]) -> Option<ExprId> {
    Some(match (name, a.len()) {
        ("exp", 1) => ctx.exp(a[0]),
        ("ln", 1) => ctx.ln(a[0]),
        ("log" | "log10", 1) => {
            let l = ctx.ln(a[0]);
            let ln10 = ctx.konst_f64(std::f64::consts::LN_10);
            ctx.div(l, ln10)
        }
        ("log2", 1) => {
            let l = ctx.ln(a[0]);
            let ln2 = ctx.konst_f64(std::f64::consts::LN_2);
            ctx.div(l, ln2)
        }
        ("sqrt", 1) => ctx.sqrt(a[0]),
        ("abs", 1) => {
            let zero = ctx.zero();
            let neg = ctx.neg(a[0]);
            let ge = ctx.cmp(CmpOp::Ge, a[0], zero);
            ctx.select(ge, a[0], neg)
        }
        ("sin", 1) => ctx.sin(a[0]),
        ("cos", 1) => ctx.cos(a[0]),
        ("tan", 1) => {
            let s = ctx.sin(a[0]);
            let c = ctx.cos(a[0]);
            ctx.div(s, c)
        }
        ("asin", 1) => {
            // atan(x / sqrt(1 - x^2))
            let one = ctx.one();
            let x2 = ctx.mul(a[0], a[0]);
            let d = ctx.sub(one, x2);
            let r = ctx.sqrt(d);
            let q = gdiv(ctx, a[0], r);
            ctx.atan(q)
        }
        ("acos", 1) => {
            // pi/2 - asin(x)
            let one = ctx.one();
            let x2 = ctx.mul(a[0], a[0]);
            let d = ctx.sub(one, x2);
            let r = ctx.sqrt(d);
            let q = gdiv(ctx, a[0], r);
            let asin = ctx.atan(q);
            let half_pi = ctx.konst_f64(std::f64::consts::FRAC_PI_2);
            ctx.sub(half_pi, asin)
        }
        ("atan", 1) => ctx.atan(a[0]),
        ("atan2", 2) => {
            // 2*atan(y/(hypot(x,y)+x)); the x<0,y=0 branch -> pi.
            let (y, x) = (a[0], a[1]);
            let x2 = ctx.mul(x, x);
            let y2 = ctx.mul(y, y);
            let r = {
                let s = ctx.add(x2, y2);
                ctx.sqrt(s)
            };
            let denom = ctx.add(r, x);
            let zero = ctx.zero();
            let nonzero = ctx.cmp(CmpOp::Ne, denom, zero);
            let main = {
                let q = gdiv(ctx, y, denom);
                let at = ctx.atan(q);
                let two = ctx.konst_f64(2.0);
                ctx.mul(two, at)
            };
            let pi = ctx.konst_f64(std::f64::consts::PI);
            ctx.select(nonzero, main, pi)
        }
        ("sinh", 1) => ctx.sinh(a[0]),
        ("cosh", 1) => ctx.cosh(a[0]),
        ("tanh", 1) => ctx.tanh(a[0]),
        ("asinh", 1) => {
            // ln(x + sqrt(x^2 + 1))
            let x2 = ctx.mul(a[0], a[0]);
            let one = ctx.one();
            let s = ctx.add(x2, one);
            let r = ctx.sqrt(s);
            let arg = ctx.add(a[0], r);
            ctx.ln(arg)
        }
        ("acosh", 1) => {
            // ln(x + sqrt(x^2 - 1))
            let one = ctx.one();
            let x2 = ctx.mul(a[0], a[0]);
            let d = ctx.sub(x2, one);
            let r = ctx.sqrt(d);
            let s = ctx.add(a[0], r);
            ctx.ln(s)
        }
        ("atanh", 1) => {
            // 0.5 * ln((1 + x)/(1 - x))
            let one = ctx.one();
            let num = ctx.add(one, a[0]);
            let den = ctx.sub(one, a[0]);
            let q = gdiv(ctx, num, den);
            let l = ctx.ln(q);
            let half = ctx.konst_f64(0.5);
            ctx.mul(half, l)
        }
        ("floor", 1) => ctx.floor(a[0]),
        ("ceil", 1) => {
            let n = ctx.neg(a[0]);
            let f = ctx.floor(n);
            ctx.neg(f)
        }
        ("pow", 2) => {
            let l = ctx.ln(a[0]);
            let bl = ctx.mul(a[1], l);
            ctx.exp(bl)
        }
        ("min", 2) => {
            let le = ctx.cmp(CmpOp::Le, a[0], a[1]);
            ctx.select(le, a[0], a[1])
        }
        ("max", 2) => {
            let ge = ctx.cmp(CmpOp::Ge, a[0], a[1]);
            ctx.select(ge, a[0], a[1])
        }
        ("hypot", 2) => {
            let x2 = ctx.mul(a[0], a[0]);
            let y2 = ctx.mul(a[1], a[1]);
            let s = ctx.add(x2, y2);
            ctx.sqrt(s)
        }
        _ => return None,
    })
}
