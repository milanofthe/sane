//! Implicit-RK transient integration for the SANE DAE: the shared adaptive outer
//! loop, and the dispatch over the pluggable [`TransientMethod`] (the
//! `C·ẋ = f(x,t)` mass-matrix form, `C = dF/dx'` constant).
//!
//! Each method owns one step ([`esdirk32`](crate::esdirk32)'s
//! `esdirk32_step`, [`trap`](crate::trap)'s `trap_step`) over the shared
//! [`StageWorkspace`](crate::stage::StageWorkspace) and returns `(x_{n+1},
//! err)`; everything else -- the consistent IC, the constant mass matrix, the
//! PI step-size control, the discontinuity schedule (source breakpoints,
//! delayed arrivals, switching surfaces), the consistent restart after a
//! landing and the dense output at `t_eval` -- is shared here.

use sane_core::log_stage;

use crate::events::Discontinuities;
use crate::stage::StageWorkspace;
use crate::{sparse, CompiledDc, TransientMethod};
use sane_core::constants::*;

/// Step controller factor, clamped. With an accepted previous error the PI
/// (Gustafsson) form `β·err^(-kI)·err_prev^(kP)` smooths the step sequence and
/// cuts the reject rate of the pure I-controller near activity onsets; without
/// history (first step, or after a reject) it degrades to the I-controller
/// `β/err^(1/p)`.
fn step_factor(err: f64, err_prev: Option<f64>) -> f64 {
    let f = match err_prev {
        Some(ep) => {
            IRK_SAFETY_BETA
                * err.powf(-IRK_PI_KI / IRK_ERR_ORDER)
                * ep.powf(IRK_PI_KP / IRK_ERR_ORDER)
        }
        None => IRK_SAFETY_BETA / err.powf(1.0 / IRK_ERR_ORDER),
    };
    f.clamp(IRK_SCALE_MIN, IRK_SCALE_MAX)
}

/// The requested output points, emitted as the integration passes them.
struct DenseOutput<'a> {
    t_eval: &'a [f64],
    cursor: usize,
    rows: Vec<Vec<f64>>,
}

impl<'a> DenseOutput<'a> {
    fn new(t_eval: &'a [f64], t0: f64, x0: &[f64]) -> Self {
        let mut d = DenseOutput {
            t_eval,
            cursor: 0,
            rows: Vec::with_capacity(t_eval.len()),
        };
        // Any requested points at or before the start clamp to the initial state.
        while d.cursor < t_eval.len() && t_eval[d.cursor] <= t0 {
            d.rows.push(x0.to_vec());
            d.cursor += 1;
        }
        d
    }

    /// Emit every requested point in `(t_prev, t]` through `interp(te)`.
    fn emit(&mut self, t: f64, mut interp: impl FnMut(f64) -> Vec<f64>) {
        while self.cursor < self.t_eval.len() && self.t_eval[self.cursor] <= t {
            let te = self.t_eval[self.cursor];
            self.rows.push(interp(te));
            self.cursor += 1;
        }
    }

    /// Trailing points beyond the last step (e.g. `t_eval == final_time` hit
    /// by the `h_min` guard) clamp to the final state.
    fn finish(mut self, x: &[f64]) -> Vec<Vec<f64>> {
        while self.cursor < self.t_eval.len() {
            self.rows.push(x.to_vec());
            self.cursor += 1;
        }
        self.rows
    }
}

impl CompiledDc {
    /// Integrate the DAE with `method`. Same interface as
    /// [`solve_transient`](Self::solve_transient): returns the state at each
    /// `t_eval`, interpolated from the internal steps by the cubic Hermite
    /// dense output (linear when the mass matrix is singular).
    pub(crate) fn solve_transient_irk(
        &self,
        method: TransientMethod,
        p: &[f64],
        x0: &[f64],
        t_eval: &[f64],
        rtol: f64,
        atol: f64,
        dt_max: Option<f64>,
    ) -> Result<Vec<Vec<f64>>, String> {
        let _stage = sane_core::log::scope("tran");
        let n = self.n;
        if t_eval.is_empty() {
            return Ok(Vec::new());
        }
        let t0 = t_eval[0];
        let final_time = t_eval[t_eval.len() - 1];

        // Initial condition: caller IC, else a self-computed consistent DC point.
        let mut x = if x0.len() == n {
            x0.to_vec()
        } else {
            log_stage!(
                "tran/dc_ic",
                self.solve_dc(p, &[], TRANSIENT_ATOL, GMIN_STEP_MAX_ITER).0
            )
        };

        // Transport delays: evaluate the taus, seed the history with the
        // initial state, and cap the step at the smallest delay so every
        // stage query lands in ACCEPTED history (t_stage - tau <= t_prev).
        let taus = self.delay_taus(p);
        if let Some(bad) = taus.iter().find(|v| !(**v > 0.0)) {
            return Err(format!(
                "transient: delay td must be positive (got {bad:.3e})"
            ));
        }
        let max_tau = taus.iter().cloned().fold(0.0, f64::max);
        let min_tau = taus.iter().cloned().fold(f64::INFINITY, f64::min);
        let dhist = (!taus.is_empty()).then(|| {
            let mut h = crate::delay::DelayHistory::new(taus.len(), max_tau);
            let vals: Vec<f64> = self
                .delay_src
                .iter()
                .map(|&s| x.get(s).copied().unwrap_or(0.0))
                .collect();
            let ders = vec![0.0; taus.len()];
            h.push(t0, &vals, &ders);
            crate::delay::set_hist_values(&vals);
            h
        });

        // The stage workspace: tapes with their prolog run once (the parameter
        // vector is fixed over the integration), the transient-wide stage
        // factorization (the first factor pays full pivoting, every refresh
        // is a numeric-only refactor), the constant mass matrix, scratch.
        let sym = self
            .stage_symbolic()
            .ok_or("irk: stage symbolic build failed")?;
        let mut ws = StageWorkspace::new(self, &sym, &x, p, t0, taus.clone(), dhist, rtol, atol);
        let trace = ws.trace;

        // Adaptive by default; the fixed-step switch forces steps at the cap or
        // a span division. `dt_max` is a step *cap* (e.g. to resolve a fast
        // source), not the step itself; the smallest delay caps it too.
        let cfg = sane_core::config();
        let fixed = cfg.transient_fixed_step;
        let hcap = match dt_max {
            Some(c) if c.is_finite() && c > 0.0 => Some(c),
            _ => None,
        };
        let hcap = if min_tau.is_finite() {
            Some(hcap.map_or(min_tau, |c| c.min(min_tau)))
        } else {
            hcap
        };
        let span = (final_time - t0).max(f64::MIN_POSITIVE);
        let h_min = span * TRANSIENT_SPAN_EPS_FRAC;
        let h_init = hcap.unwrap_or(span / 100.0).min(span);
        let mut h = h_init;

        // Where the trajectory is not smooth (see `events`): the source
        // waveform kinks the integrator must land on exactly (a C0 corner
        // inside a step breaks the smooth-LTE assumption the embedded estimate
        // relies on), their delayed arrivals, and the declared switching
        // surfaces, located as they are crossed. A circuit with no dynamic
        // elements gives the LTE controller nothing to measure -- every
        // accepted point is an exact algebraic solve and the dense output can
        // only blend linearly between them -- so its output points are landed
        // on too (the emitted rows are solves, not interpolants).
        let kinks = if self.tricks.breakpoints {
            self.transient_breakpoints(p, t0, final_time)
        } else {
            Vec::new() // trick off: let the controller step over discontinuities
        };
        let extra: &[f64] = if ws.mass.is_zero() { t_eval } else { &[] };
        let mut disc = Discontinuities::new(
            self,
            p,
            &x,
            t0,
            final_time,
            h_min,
            &taus,
            kinks,
            extra,
            self.tricks.events && cfg.events && !fixed,
        );

        let mut tracker = sane_core::ProgressTracker::with_details(
            span,
            "TRANSIENT",
            &format!("(method: {}, span: {span:.3e} s, dim: {n})", method.label()),
        );
        tracker.start();

        // Streaming dense output: only the previous and current state are kept,
        // the requested points inside each accepted step are emitted as it
        // lands (peak memory O(n x |t_eval|), independent of the step count).
        // Cubic Hermite through the endpoint states and the state-space rates
        // `x' = C⁻¹ slope` recovers the method order at output points; a
        // singular `C` (a genuine DAE) falls back to the linear blend, exact for
        // the algebraic constraint at the endpoints.
        let mut dense = DenseOutput::new(t_eval, t0, &x);
        let c_lu: Option<sparse::TripletLu> = ws.mass.factor(n);

        // PI controller history: the error of the last ACCEPTED step (cleared on
        // rejects and landings, where the local smoothness assumption breaks).
        let mut err_prev: Option<f64> = None;
        // Trapezoidal predictor history (unused by ESDIRK32).
        let mut trap_hist = crate::trap::TrapHist::default();
        // Stage-factorization carry: the factors stay valid across steps as
        // long as the step size (its `C/(hγ)` shift) is unchanged; the
        // modified-Newton stall guard refreshes them when the circuit moves.
        let mut fac_h = f64::NAN;
        let mut t = t0;
        while t < final_time - h_min {
            let mut htry = (final_time - t).min(h);
            if let Some(c) = hcap {
                htry = htry.min(c);
            }
            htry = disc.clamp(t, htry);
            // Pre-step state, for the dense-output interval [t_prev, t] once accepted.
            let t_prev = t;
            let x_prev = x.clone();
            // The controller's step, before any event retake shortens it.
            let h_before = htry;
            let mut ev_rounds = 0usize;
            // The step size that lands on a located surface; the accepted
            // candidate is a landing only if the controller kept it.
            let mut landing_h = f64::NAN;
            loop {
                if htry != fac_h {
                    ws.fac_fresh = false;
                    fac_h = htry;
                }
                let iters_before = ws.stats.iters;
                let stepped = log_stage!(
                    "tran/irk_step",
                    match method {
                        TransientMethod::Esdirk32 => self.esdirk32_step(&mut ws, &x, t, htry),
                        TransientMethod::Trap => {
                            self.trap_step(&mut ws, c_lu.as_ref(), &trap_hist, &x, t, htry)
                        }
                    }
                );
                if trace {
                    match &stepped {
                        None => eprintln!("tran: t={t:.9e} h={htry:.3e} stage Newton diverged"),
                        Some((_, err)) => eprintln!(
                            "tran: t={t:.9e} h={htry:.3e} err={err:.3e} {}",
                            if *err <= 1.0 { "ok" } else { "reject" }
                        ),
                    }
                }
                let Some((x_new, err)) = stepped else {
                    // A stage Newton diverged: shrink and retry (a too-large step
                    // through a fast region; smaller `h` recovers convergence).
                    ws.stats.rejects += 1;
                    htry *= IRK_STEP_SHRINK;
                    if htry < h_min {
                        tracker.interrupt();
                        tracker.close();
                        return Err(format!(
                            "irk: step underflow at t={t:.3e} (stage Newton diverged)"
                        ));
                    }
                    continue;
                };
                // Convergence-based carry: a step that needed more than the
                // budget per implicit stage ran on a stale Jacobian; start the
                // next one fresh.
                let stages = match method {
                    TransientMethod::Esdirk32 => ESDIRK32_STAGES - 1,
                    TransientMethod::Trap => 1,
                };
                if ws.stats.iters - iters_before > IRK_CARRY_MAX_ITERS_PER_STAGE * stages {
                    ws.fac_fresh = false;
                }
                // Accept the candidate: the trapezoidal history takes the step
                // start and its endpoint slope, the state advances.
                macro_rules! accept {
                    () => {{
                        if matches!(method, TransientMethod::Trap) {
                            let last = ws.slopes.len() - 1;
                            trap_hist.accept(t, &x, &ws.slopes[last]);
                        }
                        x = x_new;
                        t += htry;
                        ws.stats.steps += 1;
                        tracker.update(((t - t0) / span).clamp(0.0, 1.0), true);
                    }};
                }
                if fixed {
                    accept!();
                    break;
                }
                // A switching surface crossed inside the candidate: retake the
                // step to the located crossing (bounded rounds). Checked
                // before the error verdict: the mode change inside the step is
                // what inflates the error, and shrinking blindly would only
                // creep up to it.
                if ev_rounds < EVENT_RESTEP_MAX {
                    let crossing = disc.locate(&x, &x_new, t, htry, c_lu.as_ref(), &ws.slopes);
                    if let Some(te) = crossing.filter(|te| te - t > h_min) {
                        if trace {
                            eprintln!(
                                "tran:   event located at t={te:.9e}, retaking with h={:.3e}",
                                te - t
                            );
                        }
                        ev_rounds += 1;
                        ws.stats.event_steps += 1;
                        htry = te - t;
                        landing_h = htry;
                        continue;
                    }
                }
                if err > 1.0 {
                    // Rejected: shrink and retry without advancing (pure
                    // I-controller; the PI history no longer applies).
                    ws.stats.rejects += 1;
                    err_prev = None;
                    htry *= step_factor(err, None);
                    if htry < h_min {
                        tracker.interrupt();
                        tracker.close();
                        return Err(format!("irk: step underflow at t={t:.3e} (error control)"));
                    }
                    continue;
                }
                let rescale = step_factor(err, err_prev);
                err_prev = Some(err);
                accept!();
                // Landing on a discontinuity -- a scheduled instant, or a
                // surface landed on or crossed inside -- makes the region
                // beyond it a fresh problem: the step heuristic restarts from
                // a fraction of the pre-landing step, and the state is
                // reinitialised on the far side (see `reinit_step`).
                let landed = disc.accept(&x, t, htry == landing_h);
                if trace && landed.fired {
                    eprintln!("tran:   event fired at t={t:.9e}");
                }
                if landed.any() {
                    err_prev = None;
                    trap_hist.breakpoint();
                    let delta = (h_before * EVENT_REINIT_FRAC).max(h_min);
                    match self.reinit_step(&mut ws, &x, t, delta) {
                        Some(xr) => {
                            x = xr;
                            t += delta;
                            if trace {
                                eprintln!("tran:   reinitialised at t={t:.9e}");
                            }
                        }
                        None if trace => {
                            eprintln!("tran:   reinitialisation Newton failed at t={t:.9e}")
                        }
                        None => {}
                    }
                    fac_h = f64::NAN;
                }
                h = if landed.any() {
                    (h_before * EVENT_RESTART_SHRINK).max(h_min)
                } else {
                    htry * rescale
                };
                if let Some(c) = hcap {
                    h = h.min(c);
                }
                break;
            }

            // Accepted step: the dense output over [t_prev, t] and the delay
            // history knot at t.
            self.emit_step(&mut ws, c_lu.as_ref(), &mut dense, &x_prev, &x, t_prev, t);
        }

        let res = dense.finish(&x);
        let fired = disc.into_events();
        ws.stats.events = fired.len();
        for ev in &fired {
            sane_core::log::debug(&format!(
                "event: {} at t={:.6e} ({})",
                self.event_names[ev.index],
                ev.t,
                if ev.direction > 0 {
                    "rising"
                } else {
                    "falling"
                }
            ));
        }
        *self.last_events.lock().unwrap() = fired;
        let stats = &ws.stats;
        tracker.stats.rejected_steps = stats.rejects;
        tracker.close();
        sane_core::log::debug(&format!(
            "IRK: steps={} rejects={} newton_iters={} refactors={} events={} event_steps={} (iters/step={:.1}, refac/step={:.2})",
            stats.steps, stats.rejects, stats.iters, stats.refacs, stats.events, stats.event_steps,
            stats.iters as f64 / stats.steps.max(1) as f64,
            stats.refacs as f64 / stats.steps.max(1) as f64,
        ));
        Ok(res)
    }

    /// After an accepted step `[t_prev, t]`: emit the requested output points
    /// inside it and append the delay-history knot at `t`. With a factorable
    /// mass matrix the endpoint rates `x' = C⁻¹ slope` give the cubic Hermite
    /// dense output and the knot's two one-sided slopes (the step-start rate
    /// is the previous knot's right limit, patched in now); with a singular
    /// mass matrix both are the linear blend.
    #[allow(clippy::too_many_arguments)]
    fn emit_step(
        &self,
        ws: &mut StageWorkspace<'_>,
        c_lu: Option<&sparse::TripletLu>,
        dense: &mut DenseOutput<'_>,
        x_prev: &[f64],
        x: &[f64],
        t_prev: f64,
        t: f64,
    ) {
        let n = self.n;
        let hstep = t - t_prev;
        match c_lu {
            Some(lu) => {
                let m0 = solve_mass(lu, &ws.slopes[0], n);
                let m1 = solve_mass(lu, &ws.slopes[ESDIRK32_STAGES - 1], n);
                if let Some(h) = ws.dhist.as_mut() {
                    let out: Vec<f64> = self.delay_src.iter().map(|&s| m0[s]).collect();
                    h.patch_last_out(&out);
                    let vals: Vec<f64> = self.delay_src.iter().map(|&s| x[s]).collect();
                    let ders: Vec<f64> = self.delay_src.iter().map(|&s| m1[s]).collect();
                    h.push(t, &vals, &ders);
                }
                dense.emit(t, |te| {
                    hermite_point(x_prev, x, &m0, &m1, t_prev, hstep, te, n)
                });
            }
            None => {
                if let Some(h) = ws.dhist.as_mut() {
                    // singular mass: secant slopes on both interval ends, so
                    // the history interpolates linearly -- consistent with the
                    // linear dense output
                    let vals: Vec<f64> = self.delay_src.iter().map(|&s| x[s]).collect();
                    let ders: Vec<f64> = self
                        .delay_src
                        .iter()
                        .map(|&s| (x[s] - x_prev[s]) / hstep.max(f64::MIN_POSITIVE))
                        .collect();
                    h.patch_last_out(&ders);
                    h.push(t, &vals, &ders);
                }
                dense.emit(t, |te| linear_point(x_prev, x, t_prev, t, te, n));
            }
        }
    }
}

/// Solve `C m = rhs` for the state-space rate `m = x'` using the pre-factored
/// (constant) mass matrix. Zeros on a solve failure (cannot happen for a
/// successfully factored `C` of matching dimension).
pub(crate) fn solve_mass(lu: &sparse::TripletLu, rhs: &[f64], n: usize) -> Vec<f64> {
    lu.solve(rhs).unwrap_or_else(|| vec![0.0; n])
}

/// Cubic-Hermite dense output at `te` in the interval `[t_prev, t_prev + h]` from
/// the endpoint states `xa, xb` and endpoint state derivatives `ma, mb` (`= x'`).
/// O(h^4) accurate, so it recovers ESDIRK32's order at output points.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hermite_point(
    xa: &[f64],
    xb: &[f64],
    ma: &[f64],
    mb: &[f64],
    t_prev: f64,
    h: f64,
    te: f64,
    n: usize,
) -> Vec<f64> {
    let th = if h > 0.0 {
        ((te - t_prev) / h).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let t2 = th * th;
    let t3 = t2 * th;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + th;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    (0..n)
        .map(|k| h00 * xa[k] + h10 * h * ma[k] + h01 * xb[k] + h11 * h * mb[k])
        .collect()
}

/// Linear dense output at `te` in `[ta, tb]` (the singular-mass fallback).
pub(crate) fn linear_point(
    xa: &[f64],
    xb: &[f64],
    ta: f64,
    tb: f64,
    te: f64,
    n: usize,
) -> Vec<f64> {
    let w = if tb > ta {
        ((te - ta) / (tb - ta)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0..n).map(|k| xa[k] * (1.0 - w) + xb[k] * w).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Cubic Hermite reproduces any cubic exactly from its two endpoint values and
    // derivatives (its local truncation error is O(h^4)); linear interpolation of
    // the same endpoints carries O(h^2) error.
    #[test]
    fn hermite_is_exact_on_cubics_linear_is_not() {
        let g = |t: f64| 1.0 - 2.0 * t + 0.5 * t * t + 3.0 * t * t * t;
        let gp = |t: f64| -2.0 + t + 9.0 * t * t;
        let (t0, h) = (0.7, 0.4);
        let (ta, tb) = (t0, t0 + h);
        let (xa, xb) = (vec![g(ta)], vec![g(tb)]);
        let (ma, mb) = (vec![gp(ta)], vec![gp(tb)]);

        let mut worst_hermite = 0.0f64;
        let mut worst_linear = 0.0f64;
        for i in 1..10 {
            let te = ta + h * i as f64 / 10.0;
            let exact = g(te);
            let herr = (hermite_point(&xa, &xb, &ma, &mb, t0, h, te, 1)[0] - exact).abs();
            let lerr = (linear_point(&xa, &xb, ta, tb, te, 1)[0] - exact).abs();
            worst_hermite = worst_hermite.max(herr);
            worst_linear = worst_linear.max(lerr);
        }
        assert!(
            worst_hermite < 1e-12,
            "hermite cubic error {worst_hermite:e}"
        );
        assert!(
            worst_linear > 1e-2,
            "linear error unexpectedly small {worst_linear:e}"
        );
        assert!((hermite_point(&xa, &xb, &ma, &mb, t0, h, tb, 1)[0] - g(tb)).abs() < 1e-12);
    }
}
