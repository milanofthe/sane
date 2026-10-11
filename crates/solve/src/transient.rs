//! The transient integration of the SANE DAE: Rodas4 (see
//! [`crate::rosenbrock`]) under an adaptive outer loop, integrating the
//! charges `dQ/dt = −I(x, t)` (see [`crate::stage`]).
//!
//! The loop owns the consistent start, the PI step-size control, the
//! discontinuity schedule (source breakpoints, delayed arrivals, switching
//! surfaces), the consistent restart past a landing, the dense output at
//! `t_eval` and the delay history.

use sane_core::log_stage;

use crate::events::Discontinuities;
use crate::rosenbrock::{Rosenbrock, RODAS4};
use crate::stage::StageWorkspace;
use crate::transient_sens::Sens;
use crate::{CompiledDc, Symbolic, TransientRun};
use sane_core::constants::*;

/// Step controller factor, clamped. With an accepted previous error the PI
/// (Gustafsson) form `β·err^(-kI)·err_prev^(kP)` smooths the step sequence and
/// cuts the reject rate of the pure I-controller near activity onsets; without
/// history (first step, or after a reject) it degrades to the I-controller
/// `β/err^(1/p)`; `p` is the order of the error estimate.
fn step_factor(err: f64, err_prev: Option<f64>, p: f64) -> f64 {
    let f = match err_prev {
        Some(ep) => STEP_SAFETY * err.powf(-STEP_PI_KI / p) * ep.powf(STEP_PI_KP / p),
        None => STEP_SAFETY / err.powf(1.0 / p),
    };
    f.clamp(STEP_SCALE_MIN, STEP_SCALE_MAX)
}

/// Forward sensitivities asked of an integration (see
/// [`CompiledDc::solve_transient_sensitivity`]).
pub(crate) struct SensRequest<'r> {
    pub cols: &'r [usize],
    pub outputs: &'r [usize],
    pub dc_start: bool,
}

/// The requested output points, emitted as the integration passes them into
/// rows allocated before the first step.
struct DenseOutput<'a> {
    t_eval: &'a [f64],
    cursor: usize,
    rows: Vec<Vec<f64>>,
    /// The state rates at both ends of the current segment, for the delay
    /// history's knots.
    rates: [Vec<f64>; 2],
}

impl<'a> DenseOutput<'a> {
    fn new(t_eval: &'a [f64], t0: f64, x0: &[f64]) -> Self {
        let n = x0.len();
        let mut d = DenseOutput {
            t_eval,
            cursor: 0,
            rows: vec![vec![0.0; n]; t_eval.len()],
            rates: [vec![0.0; n], vec![0.0; n]],
        };
        // Any requested points at or before the start clamp to the initial state.
        while d.cursor < t_eval.len() && t_eval[d.cursor] <= t0 {
            d.rows[d.cursor].copy_from_slice(x0);
            d.cursor += 1;
        }
        d
    }

    /// Emit every requested point in the segment `(ta, tb]` from `xa` to
    /// `xb`: on the method's continuous extension when it is the method's
    /// step (`own`), linear over a restart's sliver otherwise.
    fn emit(&mut self, method: &Rosenbrock, own: bool, xa: &[f64], xb: &[f64], ta: f64, tb: f64) {
        while self.cursor < self.t_eval.len() && self.t_eval[self.cursor] <= tb {
            let te = self.t_eval[self.cursor];
            let row = &mut self.rows[self.cursor];
            if own {
                method.interpolate(te, ta, tb, xa, xb, row);
            } else {
                linear_into(row, xa, xb, ta, tb, te);
            }
            self.cursor += 1;
        }
    }

    /// Trailing points beyond the last step (e.g. `t_eval == final_time` hit
    /// by the `h_min` guard) clamp to the final state.
    fn finish(mut self, x: &[f64]) -> Vec<Vec<f64>> {
        while self.cursor < self.t_eval.len() {
            self.rows[self.cursor].copy_from_slice(x);
            self.cursor += 1;
        }
        self.rows
    }
}

/// One integration under way: the state, the step candidate and the step
/// start (three buffers that trade places as steps are accepted), the time
/// and the step, the controller's history, and what the integration writes
/// and watches as it goes.
struct Run<'a> {
    cdc: &'a CompiledDc,
    p: &'a [f64],
    ws: StageWorkspace<'a>,
    sens: Option<Sens<'a>>,
    stepper: Rosenbrock,
    dense: DenseOutput<'a>,
    disc: Discontinuities<'a>,
    tracker: sane_core::ProgressTracker,
    x: Vec<f64>,
    x_new: Vec<f64>,
    x_prev: Vec<f64>,
    t: f64,
    t0: f64,
    final_time: f64,
    span: f64,
    h: f64,
    h_min: f64,
    /// The step's cap: `dt_max`, and the smallest delay.
    hcap: Option<f64>,
    /// The error of the last ACCEPTED step, the PI controller's history
    /// (cleared on rejects and landings, where the local smoothness
    /// assumption breaks).
    err_prev: Option<f64>,
    fixed: bool,
    trace: bool,
}

impl<'a> Run<'a> {
    /// An integration of `cdc` at `p` from `x0` (cold where empty: the
    /// operating point) over `t_eval`, its stage factorizations on `syms`,
    /// up to the consistent restart into its first step.
    #[allow(clippy::too_many_arguments)]
    fn start(
        cdc: &'a CompiledDc,
        syms: (&'a Symbolic, Option<&'a Symbolic>),
        p: &'a [f64],
        x0: &[f64],
        t_eval: &'a [f64],
        (rtol, atol): (f64, f64),
        dt_max: Option<f64>,
        sens: Option<&SensRequest<'_>>,
    ) -> Result<Run<'a>, String> {
        let n = cdc.n;
        let t0 = t_eval[0];
        let final_time = t_eval[t_eval.len() - 1];

        // Initial condition: caller IC, else a self-computed consistent DC
        // point, solved to the integration's tolerances where they are
        // tighter than the operating point's: a start less consistent than
        // the steps resolve enters the first step's error as a jump.
        let mut x = if x0.len() == n {
            x0.to_vec()
        } else {
            let conv = crate::Convergence {
                reltol: DC_RELTOL.min(rtol),
                abstol: TRANSIENT_ATOL.min(atol),
                vntol: TRANSIENT_ATOL.min(atol),
            };
            log_stage!(
                "tran/dc_ic",
                cdc.solve_dc_conv_with(p, &[], conv, DC_OP_MAXIT, cdc.tricks)
                    .0
            )
        };

        // Transport delays: evaluate the taus, seed the history with the
        // initial state, and cap the step at the smallest delay so every
        // stage query lands in ACCEPTED history (t_stage - tau <= t_prev).
        let taus = cdc.delay_taus(p);
        if let Some(bad) = taus.iter().find(|v| !(**v > 0.0)) {
            return Err(format!(
                "transient: delay td must be positive (got {bad:.3e})"
            ));
        }
        let max_tau = taus.iter().cloned().fold(0.0, f64::max);
        let min_tau = taus.iter().cloned().fold(f64::INFINITY, f64::min);
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
        let dhist = (!taus.is_empty()).then(|| {
            // twice the knots the cap leaves in the horizon, with room for
            // the short steps after each restart: no growth as it goes
            let knots = hcap.map_or(0.0, |c| 2.0 * max_tau / c) as usize + DELAY_KNOT_SLACK;
            let mut h = crate::delay::DelayHistory::with_capacity(taus.len(), max_tau, knots);
            let mut vals = Vec::new();
            cdc.delay_values(&x, p, t0, &mut vals);
            let ders = vec![0.0; taus.len()];
            h.push(t0, &vals, &ders);
            crate::delay::set_hist_values(&vals);
            h
        });

        // The stage workspace: tapes with their prolog run once (the parameter
        // vector is fixed over the integration), the transient-wide stage
        // factorization (the first factor pays full pivoting, every refresh
        // is a numeric-only refactor), the mass matrix, scratch.
        let (sym, mass_sym) = syms;
        let mut ws = StageWorkspace::new(
            cdc,
            sym,
            mass_sym,
            &x,
            p,
            t0,
            taus.clone(),
            dhist,
            rtol,
            atol,
        );
        let trace = ws.trace;
        // the sensitivities, from the start before its consistent stage
        let mut sens = match sens {
            Some(r) => Some(Sens::new(
                cdc,
                sym,
                p,
                &x,
                r.dc_start || x0.len() != n,
                r.cols,
                r.outputs,
                t_eval,
            )?),
            None => None,
        };
        let span = (final_time - t0).max(f64::MIN_POSITIVE);
        let h_first = dt_max
            .filter(|c| c.is_finite() && *c > 0.0)
            .unwrap_or(span * TRANSIENT_H0_FRAC)
            .min(span);
        // A caller's initial state need not be consistent (SPICE's `uic`:
        // every node at zero, the sources not yet applied). One implicit
        // Euler step of a sliver of the first step, the time held, makes it
        // so: the algebraic unknowns take their values, the charges barely
        // move.
        let mut x_new = x.clone();
        if x0.len() == n {
            let h_ic = h_first * TRANSIENT_IC_STEP_FRAC;
            ws.fac_fresh = false;
            x_new.copy_from_slice(&x);
            let ok = cdc.stage_newton(&mut ws, &mut x_new, t0, h_ic);
            ws.fac_fresh = false;
            if !ok {
                return Err("transient: no consistent state from the initial one".into());
            }
            cdc.residual_slope(&mut ws, &x_new, t0);
            std::mem::swap(&mut x, &mut x_new);
            ws.advance();
            if let Some(s) = sens.as_mut() {
                if !s.euler(&x, t0, h_ic) {
                    return Err("transient sensitivity: singular at the start".into());
                }
            }
        }
        let x_prev = x.clone();

        let h_min = span * TRANSIENT_SPAN_EPS_FRAC;
        let h_init = hcap.unwrap_or(span * TRANSIENT_H0_FRAC).min(span);

        // Where the trajectory is not smooth (see `events`): the source
        // waveform kinks the integrator must land on exactly (a C0 corner
        // inside a step breaks the smooth-LTE assumption the embedded estimate
        // relies on), their delayed arrivals, and the declared switching
        // surfaces, located as they are crossed. A circuit with no dynamic
        // elements gives the LTE controller nothing to measure -- every
        // accepted point is an exact algebraic solve -- so its output points
        // are landed on too (the emitted rows are solves, not interpolants).
        let kinks = if cdc.tricks.breakpoints {
            cdc.transient_breakpoints(p, t0, final_time)
        } else {
            Vec::new() // trick off: let the controller step over discontinuities
        };
        let extra: &[f64] = if ws.mass.is_zero() { t_eval } else { &[] };
        let disc = Discontinuities::new(
            cdc,
            p,
            &x,
            t0,
            final_time,
            h_min,
            &taus,
            kinks,
            extra,
            cdc.tricks.events && cfg.events && !fixed,
        );

        let mut tracker = sane_core::ProgressTracker::with_details(
            span,
            "TRANSIENT",
            &format!("(span: {span:.3e} s, dim: {n})"),
        );
        tracker.start();

        // Streaming dense output: only the previous and current state are kept,
        // the requested points inside each accepted step are emitted as it
        // lands (peak memory O(n x |t_eval|), independent of the step count).
        let dense = DenseOutput::new(t_eval, t0, &x);
        if let Some(s) = sens.as_mut() {
            s.emit_start(t0);
        }

        let mut stepper = Rosenbrock::new(&RODAS4);
        if sens.is_some() {
            stepper.keep_stage_jacobians();
        }
        let mut run = Run {
            cdc,
            p,
            ws,
            sens,
            stepper,
            dense,
            disc,
            tracker,
            x,
            x_new,
            x_prev,
            t: t0,
            t0,
            final_time,
            span,
            h: h_init,
            h_min,
            hcap,
            err_prev: None,
            fixed,
            trace,
        };
        // The start is a discontinuity like a landing: the initial state
        // holds the sources at their values at `t0`, at rest, and whatever
        // moves from `t0` on -- a ramp, a sine -- moves the unknowns that are
        // rates of others (a capacitor's current across a source). One
        // consistent restart into the region beyond, as after a landing.
        if !fixed {
            run.restart((h_init * EVENT_REINIT_FRAC).max(h_min))?;
        }
        Ok(run)
    }

    /// One consistent restart of the size `delta` from `(t, x)`, into the
    /// segment past it: the state moves there and the segment is emitted.
    /// `false` when its Newton fails.
    fn restart(&mut self, delta: f64) -> Result<bool, String> {
        let t = self.t;
        let ok = (self.cdc).reinit_step(&mut self.ws, &self.x, t, delta, &mut self.x_new);
        if ok {
            std::mem::swap(&mut self.x, &mut self.x_new);
            self.ws.advance();
            if let Some(s) = self.sens.as_mut() {
                if !s.euler(&self.x, t + delta, delta) {
                    return Err(format!(
                        "transient sensitivity: singular at the restart at t={t:.3e}"
                    ));
                }
                s.emit(false, t, t + delta);
            }
            self.cdc.emit_step(
                &mut self.ws,
                &mut self.dense,
                &self.stepper,
                false,
                self.p,
                &self.x_new,
                &self.x,
                t,
                t + delta,
            );
            self.t += delta;
        }
        Ok(ok)
    }

    /// One step of the controller: tried, retaken to a located switching
    /// surface, shrunk on rejection until accepted; past a landing on a
    /// discontinuity, a consistent restart into the region beyond.
    fn step(&mut self) -> Result<(), String> {
        let (cdc, trace, h_min) = (self.cdc, self.trace, self.h_min);
        let mut htry = (self.final_time - self.t).min(self.h);
        if let Some(c) = self.hcap {
            htry = htry.min(c);
        }
        htry = self.disc.clamp(self.t, htry);
        // Pre-step time, for the dense-output interval [t_prev, t] once
        // accepted (the state moves to `x_prev` then).
        let t_prev = self.t;
        // The controller's step, before any event retake shortens it.
        let h_before = htry;
        let mut ev_rounds = 0usize;
        // The step size that lands on a located surface; the accepted
        // candidate is a landing only if the controller kept it.
        let mut landing_h = f64::NAN;
        let landed = loop {
            let t = self.t;
            let stepped = log_stage!(
                "tran/step",
                (self.stepper).step(cdc, &mut self.ws, &self.x, t, htry, &mut self.x_new)
            );
            if trace {
                match &stepped {
                    None => eprintln!("tran: t={t:.9e} h={htry:.3e} step failed"),
                    Some(err) => eprintln!(
                        "tran: t={t:.9e} h={htry:.3e} err={err:.3e} {}",
                        if *err <= 1.0 { "ok" } else { "reject" }
                    ),
                }
            }
            let Some(err) = stepped else {
                // A stage left the finite numbers or a stage matrix was
                // singular: shrink and retry.
                self.ws.stats.rejects += 1;
                htry *= STEP_SHRINK;
                if htry < h_min {
                    self.tracker.interrupt();
                    self.tracker.close();
                    return Err(format!(
                        "transient: step underflow at t={t:.3e} (step failed)"
                    ));
                }
                continue;
            };
            // A switching surface crossed inside the candidate: retake the
            // step to the located crossing (bounded rounds). Checked
            // before the error verdict: the mode change inside the step is
            // what inflates the error, and shrinking blindly would only
            // creep up to it.
            if !self.fixed && ev_rounds < EVENT_RESTEP_MAX {
                let (ws, x, x_new) = (&mut self.ws, &self.x, &self.x_new);
                let mut rates = || cdc.end_rates(ws, x, x_new, t, t + htry);
                let crossing = self.disc.locate(x, x_new, t, htry, &mut rates);
                if let Some(te) = crossing.filter(|te| te - t > h_min) {
                    if trace {
                        eprintln!(
                            "tran:   event located at t={te:.9e}, retaking with h={:.3e}",
                            te - t
                        );
                    }
                    ev_rounds += 1;
                    self.ws.stats.event_steps += 1;
                    htry = te - t;
                    landing_h = htry;
                    continue;
                }
            }
            if !self.fixed && err > 1.0 {
                // Rejected: shrink and retry without advancing (pure
                // I-controller; the PI history no longer applies).
                self.ws.stats.rejects += 1;
                self.err_prev = None;
                htry *= step_factor(err, None, self.stepper.err_order());
                if htry < h_min {
                    self.tracker.interrupt();
                    self.tracker.close();
                    return Err(format!(
                        "transient: step underflow at t={t:.3e} (error control)"
                    ));
                }
                continue;
            }
            // Accepted: the state advances.
            let rescale = step_factor(err, self.err_prev, self.stepper.err_order());
            self.err_prev = Some(err);
            std::mem::swap(&mut self.x_prev, &mut self.x);
            std::mem::swap(&mut self.x, &mut self.x_new);
            self.ws.advance();
            if let Some(s) = self.sens.as_mut() {
                if !s.rodas(&mut self.ws.matrix, &self.stepper, &self.x_prev, t, htry) {
                    return Err(format!(
                        "transient sensitivity: the step from t={t:.3e} failed"
                    ));
                }
                s.emit(true, t, t + htry);
            }
            self.t += htry;
            self.ws.stats.steps += 1;
            let progress = ((self.t - self.t0) / self.span).clamp(0.0, 1.0);
            self.tracker.update(progress, true);
            cdc.emit_step(
                &mut self.ws,
                &mut self.dense,
                &self.stepper,
                true,
                self.p,
                &self.x_prev,
                &self.x,
                t_prev,
                self.t,
            );
            if self.fixed {
                break false;
            }
            // Landing on a discontinuity -- a scheduled instant, or a
            // surface landed on or crossed inside -- makes the region
            // beyond it a fresh problem: the step heuristic restarts from
            // a fraction of the pre-landing step, and the state is
            // reinitialised on the far side (see `reinit_step`).
            let landed = self.disc.accept(&self.x, self.t, htry == landing_h);
            if trace && landed.fired {
                eprintln!("tran:   event fired at t={:.9e}", self.t);
            }
            self.h = if landed.any() {
                (h_before * EVENT_RESTART_SHRINK).max(h_min)
            } else {
                htry * rescale
            };
            break landed.any();
        };
        if landed {
            self.err_prev = None;
            self.stepper.restart();
            if self.restart((h_before * EVENT_REINIT_FRAC).max(h_min))? {
                if trace {
                    eprintln!("tran:   reinitialised at t={:.9e}", self.t);
                }
            } else if trace {
                eprintln!("tran:   reinitialisation Newton failed at t={:.9e}", self.t)
            }
        }
        if let Some(c) = self.hcap {
            self.h = self.h.min(c);
        }
        Ok(())
    }

    /// The run's result: the rows at the requested points, the events
    /// fired, the sensitivities; the counters logged.
    fn finish(mut self) -> (TransientRun, Option<Vec<Vec<f64>>>) {
        let rows = self.dense.finish(&self.x);
        let fired = self.disc.into_events();
        self.ws.stats.events = fired.len();
        for ev in &fired {
            sane_core::log::debug(&format!(
                "event: {} at t={:.6e} ({})",
                self.cdc.event_names[ev.index],
                ev.t,
                if ev.direction > 0 {
                    "rising"
                } else {
                    "falling"
                }
            ));
        }
        let stats = &self.ws.stats;
        self.tracker.stats.rejected_steps = stats.rejects;
        self.tracker.close();
        sane_core::log::debug(&format!(
            "transient: steps={} rejects={} evaluations={} factorizations={} events={} event_steps={} (evaluations/step={:.1}, factorizations/step={:.2})",
            stats.steps, stats.rejects, stats.iters, stats.refacs, stats.events, stats.event_steps,
            stats.iters as f64 / stats.steps.max(1) as f64,
            stats.refacs as f64 / stats.steps.max(1) as f64,
        ));
        let run = TransientRun {
            rows,
            events: fired,
        };
        (run, self.sens.map(Sens::finish))
    }
}

impl CompiledDc {
    /// Integrate the DAE (see [`solve_transient`](Self::solve_transient)),
    /// with the forward sensitivities `sens` asks for.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn integrate(
        &self,
        p: &[f64],
        x0: &[f64],
        t_eval: &[f64],
        rtol: f64,
        atol: f64,
        dt_max: Option<f64>,
        sens: Option<&SensRequest<'_>>,
    ) -> Result<(TransientRun, Option<Vec<Vec<f64>>>), String> {
        let _stage = sane_core::log::scope("tran");
        if t_eval.is_empty() {
            return Ok((TransientRun::default(), None));
        }
        // The symbolic analyses the stage factorization and the mass matrix
        // refactor on, for the whole integration.
        let sym = self
            .stage_symbolic()
            .ok_or("transient: stage symbolic build failed")?;
        let mass_sym = Self::build_symbolic(self.n, &self.jxd_rows, &self.jxd_cols);
        let mut run = Run::start(
            self,
            (&sym, mass_sym.as_ref()),
            p,
            x0,
            t_eval,
            (rtol, atol),
            dt_max,
            sens,
        )?;
        while run.t < run.final_time - run.h_min {
            run.step()?;
        }
        Ok(run.finish())
    }

    /// After a segment `[t_prev, t]` -- the method's step (`own`), or the
    /// restart past a discontinuity at `t_prev` -- the requested output
    /// points inside it and the delay-history knot at `t`. The knots' rates
    /// are the continuous extension's, the secant over a restart's sliver.
    /// A knot has one rate on both sides, the end rate of the step that
    /// reached it: the history is smooth through it, as a delayed signal
    /// driving a capacitor directly needs (the capacitor's current is its
    /// rate). On a discontinuity the restart's rate is the knot's right
    /// limit.
    #[allow(clippy::too_many_arguments)]
    fn emit_step(
        &self,
        ws: &mut StageWorkspace<'_>,
        dense: &mut DenseOutput<'_>,
        method: &Rosenbrock,
        own: bool,
        p: &[f64],
        x_prev: &[f64],
        x: &[f64],
        t_prev: f64,
        t: f64,
    ) {
        if let Some(h) = ws.program.history_mut() {
            let [m0, m1] = &mut dense.rates;
            if own {
                method.rates(t_prev, t, x_prev, x, m0, m1);
            } else {
                let hstep = (t - t_prev).max(f64::MIN_POSITIVE);
                for k in 0..self.n {
                    m0[k] = (x[k] - x_prev[k]) / hstep;
                }
                m1.copy_from_slice(m0);
            }
            let (rate, vals) = (&mut ws.delay_rate, &mut ws.delay_vals);
            if !own {
                self.delay_rates(x_prev, m0, p, t_prev, rate);
                h.patch_last_out(&rate[..]);
            }
            self.delay_values(x, p, t, vals);
            self.delay_rates(x, m1, p, t, rate);
            h.push(t, &vals[..], &rate[..]);
        }
        dense.emit(method, own, x_prev, x, t_prev, t);
    }

    /// The state rates at both ends of the candidate step `[ta, tb]` from
    /// `xa` to `xb` (`None` when the mass matrix is singular), for locating
    /// a crossing on the step's Hermite interpolant.
    fn end_rates(
        &self,
        ws: &mut StageWorkspace<'_>,
        xa: &[f64],
        xb: &[f64],
        ta: f64,
        tb: f64,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        let n = self.n;
        let (mut ma, mut mb) = (vec![0.0; n], vec![0.0; n]);
        let ok = ws
            .mass
            .rate_at(&mut ws.program, xa, ta, &ws.slopes[0], &mut ma)
            && ws
                .mass
                .rate_at(&mut ws.program, xb, tb, &ws.slopes[1], &mut mb);
        ok.then_some((ma, mb))
    }
}

/// Cubic-Hermite interpolation at `te` in the interval `[t_prev, t_prev + h]`
/// from the endpoint states `xa, xb` and endpoint state derivatives `ma, mb`
/// (`= x'`), O(h^4) accurate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hermite_into(
    out: &mut [f64],
    xa: &[f64],
    xb: &[f64],
    ma: &[f64],
    mb: &[f64],
    t_prev: f64,
    h: f64,
    te: f64,
) {
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
    for (k, o) in out.iter_mut().enumerate() {
        *o = h00 * xa[k] + h10 * h * ma[k] + h01 * xb[k] + h11 * h * mb[k];
    }
}

/// Where `te` lies in `[ta, tb]`, as a fraction (`0` on an empty interval).
pub(crate) fn fraction(te: f64, ta: f64, tb: f64) -> f64 {
    if tb > ta {
        ((te - ta) / (tb - ta)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Linear interpolation at `te` in `[ta, tb]`.
pub(crate) fn linear_into(out: &mut [f64], xa: &[f64], xb: &[f64], ta: f64, tb: f64, te: f64) {
    let w = fraction(te, ta, tb);
    for (k, o) in out.iter_mut().enumerate() {
        *o = xa[k] * (1.0 - w) + xb[k] * w;
    }
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
            let (mut hv, mut lv) = ([0.0], [0.0]);
            hermite_into(&mut hv, &xa, &xb, &ma, &mb, t0, h, te);
            linear_into(&mut lv, &xa, &xb, ta, tb, te);
            let (herr, lerr) = ((hv[0] - exact).abs(), (lv[0] - exact).abs());
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
        let mut hv = [0.0];
        hermite_into(&mut hv, &xa, &xb, &ma, &mb, t0, h, tb);
        assert!((hv[0] - g(tb)).abs() < 1e-12);
    }
}
