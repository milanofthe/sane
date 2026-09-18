//! Independent-source stimulus shapes as a first-class type.
//!
//! A [`SourceFn`] is the single owner of everything an independent source knows:
//! how to **lower** itself into the expression graph (the constitutive waveform),
//! how to **bind** its positional netlist parameters to instance-scoped symbols,
//! and the structural facts each analysis needs that the lowered graph can no
//! longer recover -- the transient **breakpoints** (waveform discontinuities the
//! integrator must land on exactly) and the harmonic-balance **fundamental**.
//!
//! Numeric parameters stay symbolic: each is an instance-scoped symbol named
//! `"{element}.{suffix}"` (e.g. `V1.sin_w`), bound to a value separately and
//! resolved per-analysis from the parameter vector, so a source parameter can be
//! swept like any other. The suffix convention lives *here*, next to the lowering
//! and breakpoint logic that consume it, rather than being duplicated across the
//! parser and the assembler.
//!
//! DC and AC need no entry here: the DC operating point evaluates the lowered
//! waveform at `t = 0` (a sine collapses to its offset, a pulse to `v1`, ...), and
//! SANE carries no separate small-signal source spec. If an `AC mag phase` spec is
//! ever added, its stamp belongs alongside `lower` as one more method.

use rsdag::{CmpOp, ExprId, Graph};

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

    /// Lower the waveform into the expression graph as a function of the time node
    /// `t`. Numeric parameters become instance-scoped symbols `"{name}.{suffix}"`.
    /// Region splits use `Select`; periodicity uses `floor`.
    pub fn lower(&self, ctx: &mut Graph, name: &str, t: ExprId) -> ExprId {
        let p = |ctx: &mut Graph, suffix: &str| ctx.sym(&format!("{name}.{suffix}"));
        match self {
            SourceFn::Sin => {
                // offset + amplitude * sin(w * t)
                let off = p(ctx, "sin_off");
                let amp = p(ctx, "sin_amp");
                let w = p(ctx, "sin_w");
                let wt = ctx.mul(w, t);
                let s = ctx.sin(wt);
                let amp_s = ctx.mul(amp, s);
                ctx.add(off, amp_s)
            }
            SourceFn::Pulse => {
                // v1 until td, then a trapezoid (tr, pw, tf) repeating every per.
                let (v1, v2) = (p(ctx, "pulse_v1"), p(ctx, "pulse_v2"));
                let td = p(ctx, "pulse_td");
                let tr = p(ctx, "pulse_tr");
                let tf = p(ctx, "pulse_tf");
                let pw = p(ctx, "pulse_pw");
                let per = p(ctx, "pulse_per");

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
            SourceFn::Exp => {
                // double-exponential rise/fall
                let (v1, v2) = (p(ctx, "exp_v1"), p(ctx, "exp_v2"));
                let td1 = p(ctx, "exp_td1");
                let tau1 = p(ctx, "exp_tau1");
                let td2 = p(ctx, "exp_td2");
                let tau2 = p(ctx, "exp_tau2");
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
            SourceFn::Pwl(npts) => {
                // piecewise-linear over (t_i, v_i); flat-held outside the range.
                let tsym = |ctx: &mut Graph, i: usize| ctx.sym(&format!("{name}.pwl_t{i}"));
                let vsym = |ctx: &mut Graph, i: usize| ctx.sym(&format!("{name}.pwl_v{i}"));
                let npts = *npts;
                if npts == 0 {
                    return ctx.zero();
                }
                let mut expr = vsym(ctx, npts - 1); // value beyond the last point
                for i in (0..npts.saturating_sub(1)).rev() {
                    let ti = tsym(ctx, i);
                    let ti1 = tsym(ctx, i + 1);
                    let vi = vsym(ctx, i);
                    let vi1 = vsym(ctx, i + 1);
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
                let t0 = tsym(ctx, 0);
                let v0 = vsym(ctx, 0);
                let c0 = ctx.cmp(CmpOp::Lt, t, t0);
                ctx.select(c0, v0, expr)
            }
        }
    }

    /// Append the waveform's discontinuity times in `(t0, t1]` to `out` -- the
    /// instants the transient integrator must step onto exactly (a `C0` corner
    /// inside a step breaks the smooth-LTE assumption). `resolve("{suffix}")`
    /// returns a parameter's value, or `None` if unbound (then a sensible default
    /// is used). A smooth waveform (sine) contributes nothing.
    pub fn breakpoints(
        &self,
        resolve: impl Fn(&str) -> Option<f64>,
        t0: f64,
        t1: f64,
        out: &mut Vec<f64>,
    ) {
        let emit = |t: f64, out: &mut Vec<f64>| {
            if t > t0 && t <= t1 {
                out.push(t);
            }
        };
        match self {
            SourceFn::Sin => {} // C-infinity: no breakpoints
            SourceFn::Pulse => {
                let td = resolve("pulse_td").unwrap_or(0.0);
                let tr = resolve("pulse_tr").unwrap_or(0.0).max(0.0);
                let tf = resolve("pulse_tf").unwrap_or(0.0).max(0.0);
                let pw = resolve("pulse_pw").unwrap_or(0.0).max(0.0);
                let per = resolve("pulse_per").unwrap_or(f64::INFINITY);
                // Corners within one period, measured from the period start.
                let corners = [0.0, tr, tr + pw, tr + pw + tf];
                if !(per > 0.0) || !per.is_finite() {
                    // Single (non-repeating) pulse.
                    for c in corners {
                        emit(td + c, out);
                    }
                    return;
                }
                // Repeat until past t1; cap the count so a tiny period cannot spin.
                let k0 = if t0 > td {
                    ((t0 - td) / per).floor() as i64
                } else {
                    0
                };
                let mut k = k0.max(0);
                let kmax = k + (((t1 - td) / per).ceil() as i64).max(0) + 2;
                while k <= kmax {
                    let base = td + (k as f64) * per;
                    if base > t1 {
                        break;
                    }
                    for c in corners {
                        emit(base + c, out);
                    }
                    k += 1;
                }
            }
            SourceFn::Exp => {
                // Kinks where each exponential switches on.
                if let Some(td1) = resolve("exp_td1") {
                    emit(td1, out);
                }
                if let Some(td2) = resolve("exp_td2") {
                    emit(td2, out);
                }
            }
            SourceFn::Pwl(npts) => {
                for i in 0..*npts {
                    if let Some(ti) = resolve(&format!("pwl_t{i}")) {
                        emit(ti, out);
                    }
                }
            }
        }
    }

    /// The harmonic-balance fundamental frequency (Hz), if the source is periodic
    /// with a well-defined tone: a sine's frequency, or a pulse train's `1/per`.
    pub fn fundamental(&self, resolve: impl Fn(&str) -> Option<f64>) -> Option<f64> {
        match self {
            SourceFn::Sin => resolve("sin_w").map(|w| w / (2.0 * std::f64::consts::PI)),
            SourceFn::Pulse => match resolve("pulse_per") {
                Some(per) if per > 0.0 && per.is_finite() => Some(1.0 / per),
                _ => None,
            },
            SourceFn::Exp | SourceFn::Pwl(_) => None,
        }
    }
}

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
