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
//! `e = (dF/dx + 2C/h)⁻¹ · (2/h)·C · e_raw`: on slow nodes this is the
//! identity, on stiff nodes the fast modes are attenuated by their own
//! conductance, and on algebraic rows (zero `C` row) the estimate vanishes --
//! constraints carry no independent truncation error. The state history is
//! cleared at source breakpoints (the loop lands on them exactly), so the
//! predictor never extrapolates across a kink -- the other classic ringing
//! source.

use sane_core::constants::*;

use crate::esdirk32::ws_error_norm;
use crate::stage::StageWorkspace;
use crate::{sparse, CompiledDc};

/// Accepted-state history the trapezoidal predictor extrapolates from.
#[derive(Default)]
pub(crate) struct TrapHist {
    /// Up to two earlier accepted points `(t, x)`, oldest first (the current
    /// step start `(t_n, x_n)` lives in the integration loop, not here).
    pub pts: Vec<(f64, Vec<f64>)>,
    /// Step-start slope `f_n = −F̃(x_n)` in residual space, cached from the
    /// previous accepted step's endpoint (a reject keeps `x_n`, so it stays
    /// valid; sources are C0 at breakpoints, so it survives those too).
    pub f_n: Option<Vec<f64>>,
}

impl TrapHist {
    /// Record an accepted step `(t_prev, x_prev) -> f_end`: the old step start
    /// joins the history (keeping the last two points), the endpoint slope
    /// becomes the next step's `f_n`.
    pub fn accept(&mut self, t_prev: f64, x_prev: &[f64], f_end: &[f64]) {
        self.pts.push((t_prev, x_prev.to_vec()));
        if self.pts.len() > 2 {
            self.pts.remove(0);
        }
        self.f_n = Some(f_end.to_vec());
    }
    /// A source kink: extrapolating across it would ring. Drop the polynomial
    /// history (the slope is C0-continuous and survives).
    pub fn breakpoint(&mut self) {
        self.pts.clear();
    }
}

impl CompiledDc {
    /// One trapezoidal step from `(t, xn)` with size `h`. Writes the endpoint
    /// slopes into `ws.slopes[0]` / `ws.slopes[last]` (the loop's dense-output
    /// contract, shared with ESDIRK32) and returns `(x_{n+1}, err)`, or `None`
    /// if the corrector Newton diverges.
    pub(crate) fn trap_step(
        &self,
        ws: &mut StageWorkspace<'_>,
        c_lu: Option<&sparse::TripletLu>,
        hist: &TrapHist,
        xn: &[f64],
        t: f64,
        h: f64,
    ) -> Option<(Vec<f64>, f64)> {
        let n = self.n;

        // Step-start slope f_n (an explicit residual eval, like ESDIRK stage 0,
        // when the cache is cold).
        let f_n: Vec<f64> = match &hist.f_n {
            Some(f) => f.clone(),
            None => {
                ws.fill_hist(t);
                self.residual_slope(ws, xn, t)
            }
        };
        ws.slopes[0].copy_from_slice(&f_n);

        // Predictor: extrapolate the accepted history to t + h. Quadratic with
        // two earlier points, linear with one; on a cold start fall back to the
        // Taylor step `xn + h·C⁻¹·f_n` (or `xn` when `C` is singular).
        let te = t + h;
        // `k` maps `x_c − x_p` to the corrector LTE via the two error constants.
        let (x_p, k_milne): (Vec<f64>, f64) = match hist.pts.as_slice() {
            [(t0, x0), (t1, x1)] => {
                let (d10, d21) = (t1 - t0, t - t1);
                let xp = (0..n)
                    .map(|r| {
                        let d01 = (x1[r] - x0[r]) / d10;
                        let d12 = (xn[r] - x1[r]) / d21;
                        let d012 = (d12 - d01) / (t - t0);
                        xn[r] + (te - t) * (d12 + (te - t1) * d012)
                    })
                    .collect();
                // LTE_corr = −h³/12·x''', pred rest = h(h+h1)(h+h1+h2)/6·x'''.
                let (hh, hhh) = (h + d21, h + d21 + d10);
                let k = (h * h / 12.0) / (hh * hhh / 6.0 - h * h / 12.0);
                (xp, k)
            }
            [(t1, x1)] => {
                let d = t - t1;
                let xp = (0..n)
                    .map(|r| xn[r] + (te - t) * (xn[r] - x1[r]) / d)
                    .collect();
                // First-order predictor: `x_c − x_p` measures x'' rather than
                // x'''; the fixed Milne 1/6 is a conservative start-up bound.
                (xp, 1.0 / 6.0)
            }
            // Cold start: Taylor `xn + h·C⁻¹f_n` when the mass matrix permits,
            // else the constant predictor. Both keep a conservative estimate --
            // a blind first step (err = ok) once swallowed 1% of a waveform
            // when `h_init` was span-scaled.
            [] => match c_lu {
                Some(lu) => {
                    let m = lu.solve(&f_n)?;
                    ((0..n).map(|r| xn[r] + h * m[r]).collect(), 1.0 / 6.0)
                }
                None => (xn.to_vec(), 1.0 / 6.0),
            },
            _ => unreachable!("TrapHist keeps at most two points"),
        };

        // Corrector: the trapezoidal rule as a γ = 1/2 stage, warm-started at
        // the predictor.
        for r in 0..n {
            ws.psi[r] = 0.5 * h * f_n[r];
        }
        ws.fill_hist(te);
        let (x_new, f_end, ok) = self.stage_newton(ws, xn, &x_p, te, h, 0.5);
        if !ok {
            return None;
        }
        let last = ws.slopes.len() - 1;
        ws.slopes[last].copy_from_slice(&f_end);

        // Filtered predictor-corrector error estimate (see module docs). The
        // filter needs the stage factorization at the endpoint; a stalled
        // corrector may have left it stale -- rebuild it there.
        if !ws.fac_fresh && !self.refactor_at(ws, &x_new, te, h, 0.5) {
            return None;
        }
        let e_raw: Vec<f64> = (0..n).map(|r| k_milne * (x_new[r] - x_p[r])).collect();
        ws.mass.matvec(&e_raw, &mut ws.cdx);
        let b: Vec<f64> = (0..n).map(|r| ws.cdx[r] * 2.0 / h).collect();
        let e_filt = ws.fac.solve(&b)?;
        let (err, _) = ws_error_norm(ws, &x_new, &e_filt, 1.0);
        Some((x_new, err.max(IRK_ERR_FLOOR)))
    }
}
