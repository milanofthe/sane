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

/// `a ** b`. A constant integer exponent is an integer power (repeated
/// products): exact for a negative or zero base, as the language defines it,
/// and smooth everywhere. Any other exponent goes through the domain-guarded
/// `exp(b ln a)`, whose base must be positive.
pub fn pow(ctx: &mut Graph, a: ExprId, b: ExprId) -> ExprId {
    if let Some(n) = ctx
        .const_f64(b)
        .filter(|n| n.fract() == 0.0 && n.abs() <= i32::MAX as f64)
    {
        return ctx.pow_i(a, n as i64);
    }
    let l = ctx.ln(a);
    let bl = ctx.mul(b, l);
    ctx.exp(bl)
}

/// An elementary function: one per name across every expression frontend
/// (the netlist's parameters and behavioral sources, Verilog-A), with its
/// numeric value and its construction on the symbolic DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MathFn {
    Exp,
    Ln,
    Log10,
    Log2,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Floor,
    Ceil,
    Pow,
    Min,
    Max,
    Hypot,
}

impl MathFn {
    /// Every function, for listing them.
    pub const ALL: [MathFn; 25] = [
        MathFn::Exp,
        MathFn::Ln,
        MathFn::Log10,
        MathFn::Log2,
        MathFn::Sqrt,
        MathFn::Abs,
        MathFn::Sin,
        MathFn::Cos,
        MathFn::Tan,
        MathFn::Asin,
        MathFn::Acos,
        MathFn::Atan,
        MathFn::Atan2,
        MathFn::Sinh,
        MathFn::Cosh,
        MathFn::Tanh,
        MathFn::Asinh,
        MathFn::Acosh,
        MathFn::Atanh,
        MathFn::Floor,
        MathFn::Ceil,
        MathFn::Pow,
        MathFn::Min,
        MathFn::Max,
        MathFn::Hypot,
    ];

    /// The function a name calls, but `log`, whose base the dialect sets
    /// (see [`from_spice`](Self::from_spice),
    /// [`from_veriloga`](Self::from_veriloga)).
    pub fn from_name(name: &str) -> Option<MathFn> {
        Some(match name {
            "exp" => MathFn::Exp,
            "ln" => MathFn::Ln,
            "log10" => MathFn::Log10,
            "log2" => MathFn::Log2,
            "sqrt" => MathFn::Sqrt,
            "abs" => MathFn::Abs,
            "sin" => MathFn::Sin,
            "cos" => MathFn::Cos,
            "tan" => MathFn::Tan,
            "asin" => MathFn::Asin,
            "acos" => MathFn::Acos,
            "atan" => MathFn::Atan,
            "atan2" => MathFn::Atan2,
            "sinh" => MathFn::Sinh,
            "cosh" => MathFn::Cosh,
            "tanh" => MathFn::Tanh,
            "asinh" => MathFn::Asinh,
            "acosh" => MathFn::Acosh,
            "atanh" => MathFn::Atanh,
            "floor" => MathFn::Floor,
            "ceil" => MathFn::Ceil,
            "pow" => MathFn::Pow,
            "min" => MathFn::Min,
            "max" => MathFn::Max,
            "hypot" => MathFn::Hypot,
            _ => return None,
        })
    }

    /// The function a netlist (SPICE) expression calls: `log` is the
    /// natural logarithm, as in ngspice and HSPICE.
    pub fn from_spice(name: &str) -> Option<MathFn> {
        match name {
            "log" => Some(MathFn::Ln),
            _ => MathFn::from_name(name),
        }
    }

    /// The function a Verilog-A expression calls: `log` is the base-10
    /// logarithm, `ln` the natural one.
    pub fn from_veriloga(name: &str) -> Option<MathFn> {
        match name {
            "log" => Some(MathFn::Log10),
            _ => MathFn::from_name(name),
        }
    }

    /// Its name (see [`from_name`](Self::from_name)).
    pub fn name(self) -> &'static str {
        match self {
            MathFn::Exp => "exp",
            MathFn::Ln => "ln",
            MathFn::Log10 => "log10",
            MathFn::Log2 => "log2",
            MathFn::Sqrt => "sqrt",
            MathFn::Abs => "abs",
            MathFn::Sin => "sin",
            MathFn::Cos => "cos",
            MathFn::Tan => "tan",
            MathFn::Asin => "asin",
            MathFn::Acos => "acos",
            MathFn::Atan => "atan",
            MathFn::Atan2 => "atan2",
            MathFn::Sinh => "sinh",
            MathFn::Cosh => "cosh",
            MathFn::Tanh => "tanh",
            MathFn::Asinh => "asinh",
            MathFn::Acosh => "acosh",
            MathFn::Atanh => "atanh",
            MathFn::Floor => "floor",
            MathFn::Ceil => "ceil",
            MathFn::Pow => "pow",
            MathFn::Min => "min",
            MathFn::Max => "max",
            MathFn::Hypot => "hypot",
        }
    }

    /// How many arguments it takes.
    pub fn arity(self) -> usize {
        match self {
            MathFn::Atan2 | MathFn::Pow | MathFn::Min | MathFn::Max | MathFn::Hypot => 2,
            _ => 1,
        }
    }

    /// Its value at `a` (`arity` arguments).
    pub fn eval(self, a: &[f64]) -> f64 {
        let x = a[0];
        let y = || a[1];
        match self {
            MathFn::Exp => x.exp(),
            MathFn::Ln => x.ln(),
            MathFn::Log10 => x.log10(),
            MathFn::Log2 => x.log2(),
            MathFn::Sqrt => x.sqrt(),
            MathFn::Abs => x.abs(),
            MathFn::Sin => x.sin(),
            MathFn::Cos => x.cos(),
            MathFn::Tan => x.tan(),
            MathFn::Asin => x.asin(),
            MathFn::Acos => x.acos(),
            MathFn::Atan => x.atan(),
            MathFn::Atan2 => x.atan2(y()),
            MathFn::Sinh => x.sinh(),
            MathFn::Cosh => x.cosh(),
            MathFn::Tanh => x.tanh(),
            MathFn::Asinh => x.asinh(),
            MathFn::Acosh => x.acosh(),
            MathFn::Atanh => x.atanh(),
            MathFn::Floor => x.floor(),
            MathFn::Ceil => x.ceil(),
            MathFn::Pow => x.powf(y()),
            MathFn::Min => x.min(y()),
            MathFn::Max => x.max(y()),
            MathFn::Hypot => x.hypot(y()),
        }
    }

    /// Its construction over already-lowered arguments (`arity` of them).
    /// Functions with a native unary op map to it directly; the rest are
    /// built from the primitive ops with well-defined subgradients
    /// (`abs`/`min`/`max` via `Cmp` + `Select`), so autodiff stays
    /// consistent everywhere.
    pub fn lower(self, ctx: &mut Graph, a: &[ExprId]) -> ExprId {
        match self {
            MathFn::Exp => ctx.exp(a[0]),
            MathFn::Ln => ctx.ln(a[0]),
            MathFn::Log10 => {
                let l = ctx.ln(a[0]);
                let ln10 = ctx.konst_f64(std::f64::consts::LN_10);
                ctx.div(l, ln10)
            }
            MathFn::Log2 => {
                let l = ctx.ln(a[0]);
                let ln2 = ctx.konst_f64(std::f64::consts::LN_2);
                ctx.div(l, ln2)
            }
            MathFn::Sqrt => ctx.sqrt(a[0]),
            MathFn::Abs => {
                let zero = ctx.zero();
                let neg = ctx.neg(a[0]);
                let ge = ctx.cmp(CmpOp::Ge, a[0], zero);
                ctx.select(ge, a[0], neg)
            }
            MathFn::Sin => ctx.sin(a[0]),
            MathFn::Cos => ctx.cos(a[0]),
            MathFn::Tan => {
                let s = ctx.sin(a[0]);
                let c = ctx.cos(a[0]);
                ctx.div(s, c)
            }
            MathFn::Asin => {
                // atan(x / sqrt(1 - x^2))
                let one = ctx.one();
                let x2 = ctx.mul(a[0], a[0]);
                let d = ctx.sub(one, x2);
                let r = ctx.sqrt(d);
                let q = gdiv(ctx, a[0], r);
                ctx.atan(q)
            }
            MathFn::Acos => {
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
            MathFn::Atan => ctx.atan(a[0]),
            MathFn::Atan2 => {
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
            MathFn::Sinh => ctx.sinh(a[0]),
            MathFn::Cosh => ctx.cosh(a[0]),
            MathFn::Tanh => ctx.tanh(a[0]),
            MathFn::Asinh => {
                // ln(x + sqrt(x^2 + 1))
                let x2 = ctx.mul(a[0], a[0]);
                let one = ctx.one();
                let s = ctx.add(x2, one);
                let r = ctx.sqrt(s);
                let arg = ctx.add(a[0], r);
                ctx.ln(arg)
            }
            MathFn::Acosh => {
                // ln(x + sqrt(x^2 - 1))
                let one = ctx.one();
                let x2 = ctx.mul(a[0], a[0]);
                let d = ctx.sub(x2, one);
                let r = ctx.sqrt(d);
                let s = ctx.add(a[0], r);
                ctx.ln(s)
            }
            MathFn::Atanh => {
                // 0.5 * ln((1 + x)/(1 - x))
                let one = ctx.one();
                let num = ctx.add(one, a[0]);
                let den = ctx.sub(one, a[0]);
                let q = gdiv(ctx, num, den);
                let l = ctx.ln(q);
                let half = ctx.konst_f64(0.5);
                ctx.mul(half, l)
            }
            MathFn::Floor => ctx.floor(a[0]),
            MathFn::Ceil => {
                let n = ctx.neg(a[0]);
                let f = ctx.floor(n);
                ctx.neg(f)
            }
            MathFn::Pow => pow(ctx, a[0], a[1]),
            MathFn::Min => {
                let le = ctx.cmp(CmpOp::Le, a[0], a[1]);
                ctx.select(le, a[0], a[1])
            }
            MathFn::Max => {
                let ge = ctx.cmp(CmpOp::Ge, a[0], a[1]);
                ctx.select(ge, a[0], a[1])
            }
            MathFn::Hypot => {
                let x2 = ctx.mul(a[0], a[0]);
                let y2 = ctx.mul(a[1], a[1]);
                let s = ctx.add(x2, y2);
                ctx.sqrt(s)
            }
        }
    }
}
