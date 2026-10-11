//! Independent-source stimulus shapes as a first-class type.
//!
//! A [`SourceFn`] is the single owner of a source's parameters: how it
//! **binds** its positional netlist parameters to instance-scoped symbols, its
//! **stimulus** (the waveform by its parameters' names, which the analyses read
//! the breakpoints, the fundamental and the small-signal drive off; see
//! [`sane_core::stimulus`]), and how it **lowers** into the expression graph.
//!
//! Numeric parameters stay symbolic: each is an instance-scoped symbol named
//! `"{element}.{suffix}"` (e.g. `V1.sin_w`), bound to a value separately and
//! resolved per-analysis from the parameter vector, so a source parameter can be
//! swept like any other. The suffix convention lives *here* alone.
//!
//! DC and AC need no entry here: the DC operating point evaluates the lowered
//! waveform at `t = 0` (a sine collapses to its offset, a pulse to `v1`, ...), and
//! SANE carries no separate small-signal source spec. If an `AC mag phase` spec is
//! ever added, its stamp belongs alongside `lower` as one more method.

use rsdag::{CmpOp, ExprId};
use sane_core::stimulus::Stimulus;
use sane_core::Graph;

/// What an independent source drives: a DC value or a time-domain
/// waveform. A number is its DC value.
#[derive(Clone, Debug, PartialEq)]
pub enum Waveform {
    Dc(f64),
    /// `offset + amplitude * sin(2 pi freq t)`.
    Sin {
        offset: f64,
        amplitude: f64,
        freq: f64,
    },
    /// `v1` until `delay`, then a trapezoid to `v2` (`rise`, `width`,
    /// `fall`) repeating every `period`; a zero `fall`, `width` or `period`
    /// takes SPICE's default.
    Pulse {
        v1: f64,
        v2: f64,
        delay: f64,
        rise: f64,
        fall: f64,
        width: f64,
        period: f64,
    },
    /// A double exponential from `v1` toward `v2` at `td1` (time constant
    /// `tau1`), back at `td2` (`tau2`).
    Exp {
        v1: f64,
        v2: f64,
        td1: f64,
        tau1: f64,
        td2: f64,
        tau2: f64,
    },
    /// Piecewise linear through the points `(t, value)`.
    Pwl(Vec<(f64, f64)>),
}

impl From<f64> for Waveform {
    fn from(v: f64) -> Waveform {
        Waveform::Dc(v)
    }
}

/// Time-domain stimulus shape of an independent source. `None` on an element (the
/// default) means a constant whose value is the element's own symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceFn {
    /// `offset + amplitude * sin(w * t)`.
    Sin,
    /// Trapezoidal pulse train (`v1 v2 td tr tf pw per`).
    Pulse,
    /// Double-exponential (`v1 v2 td1 tau1 td2 tau2`).
    Exp,
    /// Piecewise-linear over `n` breakpoints `(t_i, v_i)`.
    Pwl(usize),
}

impl SourceFn {
    /// Identify a source shape from its netlist function name and argument count
    /// (`pwl` carries its point count). Returns `None` for an unknown function.
    pub fn from_func(func: &str, nargs: usize) -> Option<Self> {
        match func.to_ascii_lowercase().as_str() {
            "sin" => Some(SourceFn::Sin),
            "pulse" => Some(SourceFn::Pulse),
            "exp" => Some(SourceFn::Exp),
            "pwl" => Some(SourceFn::Pwl(nargs / 2)),
            _ => None,
        }
    }

    /// Bind positional netlist arguments to instance-scoped parameter `(suffix,
    /// value)` pairs (the caller prefixes each suffix with the element name). A
    /// missing trailing argument (`None`) is simply left unbound, so its symbol
    /// stays free. The frequency of a `SIN` is stored as the angular `sin_w =
    /// 2*pi*f`; that transform lives here so the parser and the lowering agree.
    pub fn bind_params(&self, args: &[Option<f64>]) -> Vec<(String, f64)> {
        let at = |i: usize| args.get(i).copied().flatten();
        let mut out = Vec::new();
        let mut push = |suffix: &str, v: f64| out.push((suffix.to_string(), v));
        match self {
            SourceFn::Sin => {
                if let Some(v) = at(0) {
                    push("sin_off", v);
                }
                if let Some(v) = at(1) {
                    push("sin_amp", v);
                }
                if let Some(v) = at(2) {
                    push("sin_w", 2.0 * std::f64::consts::PI * v);
                }
            }
            SourceFn::Pulse => {
                for (i, s) in PULSE_SUFFIX.iter().enumerate() {
                    if let Some(v) = at(i) {
                        push(s, v);
                    }
                }
                // SPICE's defaults: the fall as long as the rise, the pulse
                // held and not repeated (SPICE takes the end of the run)
                if let (None, Some(tr)) = (at(4), at(3)) {
                    push("pulse_tf", tr);
                }
                if at(5).is_none() {
                    push("pulse_pw", PULSE_FOREVER);
                }
                if at(6).is_none() {
                    push("pulse_per", PULSE_FOREVER);
                }
            }
            SourceFn::Exp => {
                for (i, s) in EXP_SUFFIX.iter().enumerate() {
                    if let Some(v) = at(i) {
                        push(s, v);
                    }
                }
            }
            SourceFn::Pwl(npts) => {
                for i in 0..*npts {
                    if let Some(t) = at(2 * i) {
                        push(&format!("pwl_t{i}"), t);
                    }
                    if let Some(v) = at(2 * i + 1) {
                        push(&format!("pwl_v{i}"), v);
                    }
                }
            }
        }
        out
    }

    /// The waveform of the source `element` by its parameters' names,
    /// instance-scoped `"{element}.{suffix}"` (the names
    /// [`bind_params`](Self::bind_params) binds).
    pub fn stimulus(&self, element: &str) -> Stimulus<String> {
        let p = |suffix: &str| format!("{element}.{suffix}");
        match self {
            SourceFn::Sin => Stimulus::Sin {
                offset: p("sin_off"),
                amplitude: p("sin_amp"),
                w: p("sin_w"),
            },
            SourceFn::Pulse => Stimulus::Pulse {
                v1: p("pulse_v1"),
                v2: p("pulse_v2"),
                td: p("pulse_td"),
                tr: p("pulse_tr"),
                tf: p("pulse_tf"),
                pw: p("pulse_pw"),
                per: p("pulse_per"),
            },
            SourceFn::Exp => Stimulus::Exp {
                v1: p("exp_v1"),
                v2: p("exp_v2"),
                td1: p("exp_td1"),
                tau1: p("exp_tau1"),
                td2: p("exp_td2"),
                tau2: p("exp_tau2"),
            },
            SourceFn::Pwl(n) => Stimulus::Pwl(
                (0..*n)
                    .map(|i| (p(&format!("pwl_t{i}")), p(&format!("pwl_v{i}"))))
                    .collect(),
            ),
        }
    }

    /// Lower the waveform into the expression graph as a function of the time node
    /// `t`, over its parameters' symbols (see [`stimulus`](Self::stimulus)).
    /// Region splits use `Select`; periodicity uses `floor`.
    pub fn lower(&self, ctx: &mut Graph, name: &str, t: ExprId) -> ExprId {
        match self.stimulus(name).map(|n| ctx.sym(n)) {
            Stimulus::Sin {
                offset: off,
                amplitude: amp,
                w,
            } => {
                // offset + amplitude * sin(w * t)
                let wt = ctx.mul(w, t);
                let s = ctx.sin(wt);
                let amp_s = ctx.mul(amp, s);
                ctx.add(off, amp_s)
            }
            Stimulus::Pulse {
                v1,
                v2,
                td,
                tr,
                tf,
                pw,
                per,
            } => {
                // v1 until td, then a trapezoid (tr, pw, tf) repeating every per.

                // tp = (t - td) - per*floor((t - td)/per)  (phase within the period)
                let tmtd = ctx.sub(t, td);
                let ratio = ctx.div(tmtd, per);
                let cyc = ctx.floor(ratio);
                let per_cyc = ctx.mul(per, cyc);
                let tp = ctx.sub(tmtd, per_cyc);

                // rise: v1 + (v2-v1)*(tp/tr)
                let dv = ctx.sub(v2, v1);
                let tp_tr = ctx.div(tp, tr);
                let rise_d = ctx.mul(dv, tp_tr);
                let rise = ctx.add(v1, rise_d);
                // fall: v2 + (v1-v2)*((tp-(tr+pw))/tf)
                let dv2 = ctx.sub(v1, v2);
                let tr_pw = ctx.add(tr, pw);
                let tp_off = ctx.sub(tp, tr_pw);
                let tp_off_tf = ctx.div(tp_off, tf);
                let fall_d = ctx.mul(dv2, tp_off_tf);
                let fall = ctx.add(v2, fall_d);

                // body = tp<tr ? rise : (tp<tr+pw ? v2 : (tp<tr+pw+tf ? fall : v1))
                let c_rise = ctx.cmp(CmpOp::Lt, tp, tr);
                let c_high = ctx.cmp(CmpOp::Lt, tp, tr_pw);
                let tr_pw_tf = ctx.add(tr_pw, tf);
                let c_fall = ctx.cmp(CmpOp::Lt, tp, tr_pw_tf);
                let s_fall = ctx.select(c_fall, fall, v1);
                let s_high = ctx.select(c_high, v2, s_fall);
                let body = ctx.select(c_rise, rise, s_high);
                // before td: hold v1
                let c_before = ctx.cmp(CmpOp::Lt, t, td);
                ctx.select(c_before, v1, body)
            }
            Stimulus::Exp {
                v1,
                v2,
                td1,
                tau1,
                td2,
                tau2,
            } => {
                // double-exponential rise/fall
                let dv = ctx.sub(v2, v1);
                let dv2 = ctx.sub(v1, v2);

                // rise = v1 + (v2-v1)*(1 - exp(-(t-td1)/tau1))
                let one = ctx.one();
                let t_td1 = ctx.sub(t, td1);
                let r1 = ctx.div(t_td1, tau1);
                let nr1 = ctx.neg(r1);
                let e1 = ctx.exp(nr1);
                let om1 = ctx.sub(one, e1);
                let rise_d = ctx.mul(dv, om1);
                let rise = ctx.add(v1, rise_d);
                // extra = (v1-v2)*(1 - exp(-(t-td2)/tau2))
                let t_td2 = ctx.sub(t, td2);
                let r2 = ctx.div(t_td2, tau2);
                let nr2 = ctx.neg(r2);
                let e2 = ctx.exp(nr2);
                let om2 = ctx.sub(one, e2);
                let extra = ctx.mul(dv2, om2);
                let after = ctx.add(rise, extra);

                let c1 = ctx.cmp(CmpOp::Lt, t, td1);
                let c2 = ctx.cmp(CmpOp::Lt, t, td2);
                let mid = ctx.select(c2, rise, after);
                ctx.select(c1, v1, mid)
            }
            Stimulus::Pwl(points) => {
                // piecewise-linear over (t_i, v_i); flat-held outside the range.
                let Some(&(_, last)) = points.last() else {
                    return ctx.zero();
                };
                let mut expr = last; // value beyond the last point
                for w in points.windows(2).rev() {
                    let ((ti, vi), (ti1, vi1)) = (w[0], w[1]);
                    // seg = vi + (vi1-vi)*(t-ti)/(ti1-ti)
                    let dvv = ctx.sub(vi1, vi);
                    let tnum = ctx.sub(t, ti);
                    let tden = ctx.sub(ti1, ti);
                    let frac = ctx.div(tnum, tden);
                    let seg_d = ctx.mul(dvv, frac);
                    let seg = ctx.add(vi, seg_d);
                    let cond = ctx.cmp(CmpOp::Lt, t, ti1);
                    expr = ctx.select(cond, seg, expr);
                }
                // before the first point: hold v0
                let (t0, v0) = points[0];
                let c0 = ctx.cmp(CmpOp::Lt, t, t0);
                ctx.select(c0, v0, expr)
            }
        }
    }
}

use sane_core::constants::PULSE_FOREVER;

const PULSE_SUFFIX: [&str; 7] = [
    "pulse_v1",
    "pulse_v2",
    "pulse_td",
    "pulse_tr",
    "pulse_tf",
    "pulse_pw",
    "pulse_per",
];
const EXP_SUFFIX: [&str; 6] = [
    "exp_v1", "exp_v2", "exp_td1", "exp_tau1", "exp_td2", "exp_tau2",
];
