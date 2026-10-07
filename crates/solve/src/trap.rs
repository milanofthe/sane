//! Trapezoidal transient integrator (SPICE `trap`): 2nd-order, A-stable, one
//! implicit solve per step -- the iteration-economy method for switching /
//! digital-style transients, next to [`crate::esdirk32`]'s L-stable three-stage
//! workhorse for hard analog stiffness.
//!
//! The corrector is the trapezoidal rule `C·(x_{n+1} − x_n) = h/2·(f_n +
//! f_{n+1})` with `C·x' = f = −F̃(x, t)`, solved as a γ = 1/2 stage through the
//! shared limiting modified Newton ([`CompiledDc::stage_newton`]). The
//! predictor is a polynomial extrapolation of the accepted state history --
//! quadratic once two earlier points exist -- and serves twice: as the Newton
//! start (a good guess is the iteration-count lever) and, Milne-style, in the
//! predictor-corrector local-error estimate `LTE ≈ k·(x_corr − x_pred)` with
//! the variable-step coefficient `k` from the two methods' error constants.
//!
//! The raw difference `x_corr − x_pred` rings on stiff and algebraic unknowns
//! (the trapezoidal rule does not damp the fast modes, and an extrapolated
//! constraint variable measures Newton noise, not truncation error). The
//! estimate is therefore *filtered* through the stage iteration matrix,
//! `e = (G + 2C/h)⁻¹ · (2/h)·C · e_raw`: on slow nodes this is the
//! identity, on stiff nodes the fast modes are attenuated by their own
//! conductance, and on algebraic rows (zero `C` row) the estimate vanishes --
//! constraints carry no independent truncation error. The state history is
//! cleared at source breakpoints (the loop lands on them exactly), so the
//! predictor never extrapolates across a kink -- the other classic ringing
//! source.

use sane_core::constants::*;

use crate::esdirk32::ws_error_norm;
use crate::stage::StageWorkspace;
use crate::CompiledDc;

/// Accepted-state history the trapezoidal predictor extrapolates from, and
/// the predictor itself.
#[derive(Default)]
pub(crate) struct TrapHist {
    /// Up to two earlier accepted points `(t, x)`, oldest first, the first
    /// `len` valid (the current step start `(t_n, x_n)` lives in the
    /// integration loop, not here).
    pts: [(f64, Vec<f64>); 2],
    len: usize,
    /// Step-start slope `f_n = −F̃(x_n)` in residual space, cached from the
    /// previous accepted step's endpoint (a reject keeps `x_n`, so it stays
    /// valid; sources are C0 at breakpoints, so it survives those too).
    /// Empty before the first accepted step.
    f_n: Vec<f64>,
    /// The predicted state of the step, then its raw error estimate.
    pred: Vec<f64>,
}

impl TrapHist {
    /// Record an accepted step `(t_prev, x_prev) -> f_end`: the old step start
    /// joins the history (keeping the last two points), the endpoint slope
    /// becomes the next step's `f_n`.
    pub fn accept(&mut self, t_prev: f64, x_prev: &[f64], f_end: &[f64]) {
        if self.len == 2 {
            self.pts.swap(0, 1);
        } else {
            self.len += 1;
        }
        let (t, x) = &mut self.pts[self.len - 1];
        *t = t_prev;
        x.clear();
        x.extend_from_slice(x_prev);
        self.f_n.clear();
        self.f_n.extend_from_slice(f_end);
    }
    /// A source kink: extrapolating across it would ring. Drop the polynomial
    /// history (the slope is C0-continuous and survives).
    pub fn breakpoint(&mut self) {
        self.len = 0;
    }
}

impl CompiledDc {
    /// One trapezoidal step from `(t, xn)` with size `h`. Writes the endpoint
    /// slopes into `ws.slopes[0]` / `ws.slopes[last]` (the loop's dense-output
    /// contract, shared with ESDIRK32) and `x_{n+1}` into `x`, and returns
    /// the scaled error, or `None` if the corrector Newton diverges.
    pub(crate) fn trap_step(
        &self,
        ws: &mut StageWorkspace<'_>,
        hist: &mut TrapHist,
        xn: &[f64],
        t: f64,
        h: f64,
        x: &mut [f64],
    ) -> Option<f64> {
        let n = self.n;
        let TrapHist {
            pts,
            len,
            f_n,
            pred,
        } = hist;

        // Step-start slope f_n (an explicit residual eval, like ESDIRK stage 0,
        // when the cache is cold).
        if f_n.is_empty() {
            ws.fill_hist(t);
            self.residual_slope(ws, xn, t);
            std::mem::swap(&mut ws.slopes[0], &mut ws.f);
        } else {
            ws.slopes[0].copy_from_slice(f_n);
        }

        // Predictor: extrapolate the accepted history to t + h. Quadratic with
        // two earlier points, linear with one; on a cold start fall back to the
        // Taylor step `xn + h·C⁻¹·f_n` (or `xn` when `C` is singular).
        let te = t + h;
        pred.resize(n, 0.0);
        // `k` maps `x_c − x_p` to the corrector LTE via the two error constants.
        let k_milne = match &pts[..*len] {
            [(t0, x0), (t1, x1)] => {
                let (d10, d21) = (t1 - t0, t - t1);
                for r in 0..n {
                    let d01 = (x1[r] - x0[r]) / d10;
                    let d12 = (xn[r] - x1[r]) / d21;
                    let d012 = (d12 - d01) / (t - t0);
                    pred[r] = xn[r] + (te - t) * (d12 + (te - t1) * d012);
                }
                // LTE_corr = −h³/12·x''', pred rest = h(h+h1)(h+h1+h2)/6·x'''.
                let (hh, hhh) = (h + d21, h + d21 + d10);
                (h * h / 12.0) / (hh * hhh / 6.0 - h * h / 12.0)
            }
            [(t1, x1)] => {
                let d = t - t1;
                for r in 0..n {
                    pred[r] = xn[r] + (te - t) * (xn[r] - x1[r]) / d;
                }
                // First-order predictor: `x_c − x_p` measures x'' rather than
                // x'''; the fixed Milne 1/6 is a conservative start-up bound.
                1.0 / 6.0
            }
            // Cold start: Taylor `xn + h·C⁻¹f_n` when the mass matrix permits,
            // else the constant predictor. Both keep a conservative estimate --
            // a blind first step (err = ok) once swallowed 1% of a waveform
            // when `h_init` was span-scaled.
            _ => {
                if ws.mass.rate_at(self, xn, t, &ws.slopes[0], pred) {
                    for r in 0..n {
                        pred[r] = xn[r] + h * pred[r];
                    }
                } else {
                    pred.copy_from_slice(xn);
                }
                1.0 / 6.0
            }
        };

        // Corrector: the trapezoidal rule as a γ = 1/2 stage, warm-started at
        // the predictor.
        for r in 0..n {
            ws.psi[r] = 0.5 * h * ws.slopes[0][r];
        }
        ws.fill_hist(te);
        x.copy_from_slice(pred);
        if !self.stage_newton(ws, x, te, h, 0.5) {
            return None;
        }
        let last = ws.slopes.len() - 1;
        std::mem::swap(&mut ws.slopes[last], &mut ws.f);

        // Filtered predictor-corrector error estimate (see module docs). The
        // filter needs the stage factorization at the endpoint; a stalled
        // corrector may have left it stale -- rebuild it there.
        if !ws.fac_fresh && !self.refactor_at(ws, x, te, h, 0.5) {
            return None;
        }
        for r in 0..n {
            pred[r] = k_milne * (x[r] - pred[r]);
        }
        ws.mass.matvec(pred, &mut ws.cdx);
        for r in 0..n {
            ws.rhs[r] = ws.cdx[r] * 2.0 / h;
        }
        if !ws.fac.solve_into(&ws.rhs, &mut ws.step) {
            return None;
        }
        let (mut err, _) = ws_error_norm(ws, x, &ws.step, 1.0, &[]);
        // Rejected for truncation error only, not for the corrector's
        // rounding: the estimate moves by `k` times its floor.
        if err > 1.0 && self.rounding_floor(ws, x, 0.5 * h, k_milne.abs()) {
            (err, _) = ws_error_norm(ws, x, &ws.step, 1.0, &ws.floor);
        }
        Some(err.max(IRK_ERR_FLOOR))
    }
}
