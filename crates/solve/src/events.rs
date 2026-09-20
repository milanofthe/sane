//! Where a transient trajectory is not smooth: the instants known in advance
//! and the switching surfaces located as they are crossed.
//!
//! [`Discontinuities`] is the one schedule the integrator consults. It holds
//! the *scheduled instants* -- source waveform kinks (PWL corners, PULSE
//! edges, EXP knees), their arrivals through every transport delay (a kink
//! re-arrives after each delay, and its arrival re-arrives again: the bounce
//! cascade of a mismatched line, scheduled incrementally as each instant is
//! landed on), and the output points of a purely algebraic circuit -- and the
//! *switching surfaces* of [`EventTracker`]. The driver clamps each step to
//! the next instant, retakes a candidate that crossed a surface, and restarts
//! its step heuristic on either kind of landing; only the surfaces are
//! reported as events.
//!
//! A device declares a switching surface `g(x, t) = 0` -- a Verilog-A
//! `@(cross(expr, dir))`, the threshold of a built-in switch -- and the
//! integrator lands a step on every crossing the way it lands on a source
//! breakpoint, so a hard `Select` in the residual flips at a step boundary and
//! never inside a step. The surfaces are the DAE's event expressions compiled
//! as one tape over the step inputs; they cost one evaluation per candidate
//! step.
//!
//! Detection is a sign change of `g_k` across an accepted-quality candidate
//! step, in the declared direction. Location is a root search on the
//! candidate's dense output (the cubic Hermite, or the linear blend of a
//! singular mass matrix, that the output points use), which needs no extra
//! solves; the driver then retakes the step to the located time, checks the
//! retaken candidate the same way (the interpolant is a model of the
//! trajectory, not the trajectory), and accepts when no crossing remains
//! inside. The landing is on the near side of the surface: the accepted step
//! is wholly in the old mode, the next step starts on the surface and flips.
//!
//! A landed-on surface is disarmed until the trajectory has left it (crossed,
//! or moved off by a fraction of the surface's scale), so the step that
//! actually crosses is not located again, and a grazing approach that turns
//! back re-arms cleanly.

use rsdag::{Crossing, Tape};
use sane_core::constants::*;

use crate::transient::{hermite_point, linear_point, solve_mass};
use crate::{sparse, CompiledDc, TransientEvent};

/// What an accepted step landed on.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Landing {
    /// A scheduled instant (a source kink, a delayed arrival, an output point).
    pub instant: bool,
    /// A switching surface fired (landed on, or crossed inside the step).
    pub fired: bool,
}

impl Landing {
    pub fn any(self) -> bool {
        self.instant || self.fired
    }
}

pub(crate) struct Discontinuities<'a> {
    /// Scheduled instants, sorted; `cursor` is the first not yet passed.
    times: Vec<f64>,
    cursor: usize,
    taus: Vec<f64>,
    t0: f64,
    final_time: f64,
    /// Instants closer than this coincide (and a step this short is a sliver).
    h_min: f64,
    surfaces: Option<EventTracker<'a>>,
}

impl<'a> Discontinuities<'a> {
    /// `instants`: the source-waveform kinks in `[t0, final_time]`, sorted.
    /// `extra`: further instants to land on (the output points of a circuit
    /// without dynamics). `surfaces`: locate the declared switching surfaces
    /// (off for fixed-step integration, which steps across).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cdc: &'a CompiledDc,
        p: &[f64],
        x0: &[f64],
        t0: f64,
        final_time: f64,
        h_min: f64,
        taus: &[f64],
        instants: Vec<f64>,
        extra: &[f64],
        surfaces: bool,
    ) -> Self {
        let mut d = Discontinuities {
            times: instants,
            cursor: 0,
            taus: taus.to_vec(),
            t0,
            final_time,
            h_min,
            surfaces: None,
        };
        // Delayed arrivals of every kink and of the start itself: the first
        // generation; deeper generations are scheduled as instants are landed on.
        if !d.taus.is_empty() {
            let seeds: Vec<f64> = std::iter::once(t0).chain(d.times.iter().copied()).collect();
            for b in seeds {
                d.schedule_arrivals(b);
            }
        }
        for &te in extra {
            d.schedule(te);
        }
        if surfaces {
            if let Some(tape) = &cdc.tape_event {
                d.surfaces = Some(EventTracker::new(cdc, tape, &cdc.event_dirs, x0, p, t0));
            }
        }
        d
    }

    /// Add an instant inside the span (kept sorted; one within `h_min` of an
    /// existing instant coincides with it).
    fn schedule(&mut self, tb: f64) {
        if tb <= self.t0 + self.h_min || tb >= self.final_time - self.h_min {
            return;
        }
        let pos = self.times.partition_point(|x| *x < tb);
        if pos < self.cursor {
            return;
        }
        if self
            .times
            .get(pos)
            .is_none_or(|x| (x - tb).abs() > self.h_min)
        {
            self.times.insert(pos, tb);
        }
    }

    /// A kink at `t` re-arrives after every transport delay.
    fn schedule_arrivals(&mut self, t: f64) {
        for k in 0..self.taus.len() {
            let tb = t + self.taus[k];
            self.schedule(tb);
        }
    }

    /// Clamp a proposed step from `t` so it lands exactly on the next scheduled
    /// instant instead of stepping over it. The `1.9x` guard halves a step that
    /// would leave a tiny sliver short of the instant (ngspice's rule).
    pub fn clamp(&mut self, t: f64, htry: f64) -> f64 {
        while self.cursor < self.times.len() && self.times[self.cursor] <= t + self.h_min {
            self.cursor += 1;
        }
        let Some(&next) = self.times.get(self.cursor) else {
            return htry;
        };
        let dt = next - t;
        if htry >= dt {
            dt
        } else if htry * 1.9 > dt {
            dt * 0.5
        } else {
            htry
        }
    }

    /// The earliest switching-surface crossing inside the candidate step (see
    /// [`EventTracker::locate`]); `None` without surfaces or crossing.
    pub fn locate(
        &mut self,
        x_prev: &[f64],
        x_new: &[f64],
        t: f64,
        h: f64,
        c_lu: Option<&sparse::TripletLu>,
        slopes: &[Vec<f64>],
    ) -> Option<f64> {
        self.surfaces
            .as_mut()
            .and_then(|s| s.locate(x_prev, x_new, t, h, c_lu, slopes))
    }

    /// Bookkeeping at an accepted step ending at `(x, t)`: a landed-on instant
    /// schedules its delayed arrivals, the surfaces fire and re-arm (see
    /// [`EventTracker::accept`]; `landed_surface` says the step was retaken to
    /// a located crossing).
    pub fn accept(&mut self, x: &[f64], t: f64, landed_surface: bool) -> Landing {
        let instant =
            self.cursor < self.times.len() && (self.times[self.cursor] - t).abs() <= self.h_min;
        if instant {
            self.schedule_arrivals(t);
        }
        let fired = self
            .surfaces
            .as_mut()
            .is_some_and(|s| s.accept(x, t, landed_surface));
        Landing { instant, fired }
    }

    /// The switching events fired over the integration, in time order.
    pub fn into_events(self) -> Vec<TransientEvent> {
        self.surfaces.map(|s| s.fired).unwrap_or_default()
    }
}

/// Did `g` cross zero from `g0` to `g1` in direction `dir` (`0`: either way)?
/// The switching surfaces of one integration: detection, location, arming.
pub(crate) struct EventTracker<'a> {
    cdc: &'a CompiledDc,
    tape: &'a Tape,
    dirs: &'a [Crossing],
    inputs: Vec<f64>,
    work: Vec<f64>,
    out: Vec<f64>,
    zc: Vec<f64>,
    /// `g` at the last accepted point (the start of the current step).
    g_prev: Vec<f64>,
    /// `g` at the candidate step end.
    g_new: Vec<f64>,
    /// Largest `|g_k|` seen: the scale of the re-arm threshold.
    scale: Vec<f64>,
    /// A sign change of an armed surface is a crossing to locate; a surface is
    /// disarmed by landing on it.
    armed: Vec<bool>,
    /// Sign of `g_k` when its event fired (the re-arm rule reads it).
    fire_sign: Vec<f64>,
    /// Surfaces whose located time the current candidate step ends at, with
    /// the direction the candidate step crossed them in: at the landing the
    /// surface value is on the near side, too close to zero to read the
    /// direction off again.
    landing: Vec<(usize, i8)>,
    /// Per-surface located crossing time of the last `locate` (scratch).
    roots: Vec<f64>,
    pub fired: Vec<TransientEvent>,
}

impl<'a> EventTracker<'a> {
    pub fn new(
        cdc: &'a CompiledDc,
        tape: &'a Tape,
        dirs: &'a [Crossing],
        x0: &[f64],
        p: &[f64],
        t0: f64,
    ) -> Self {
        let m = dirs.len();
        let mut s = EventTracker {
            cdc,
            tape,
            dirs,
            inputs: Vec::new(),
            work: Vec::new(),
            out: Vec::new(),
            zc: vec![0.0; x0.len()],
            g_prev: Vec::new(),
            g_new: Vec::new(),
            scale: vec![0.0; m],
            armed: vec![true; m],
            fire_sign: vec![0.0; m],
            landing: Vec::new(),
            roots: vec![0.0; m],
            fired: Vec::new(),
        };
        s.cdc.fill_inputs(x0, &s.zc, p, t0, &mut s.inputs);
        s.tape.eval(&s.inputs, &mut s.work, &mut s.out);
        s.g_prev = s.out.clone();
        for k in 0..m {
            s.scale[k] = s.g_prev[k].abs();
        }
        s
    }

    fn eval(&mut self, x: &[f64], t: f64) {
        self.cdc.patch_inputs(x, &self.zc, t, &mut self.inputs);
        self.tape.eval(&self.inputs, &mut self.work, &mut self.out);
    }

    /// The earliest crossing inside the candidate step `[t, t + h]` ending at
    /// `x_new`, on the step's dense output; `None` when no armed surface
    /// changes sign across the step. The returned time is on the near side of
    /// the surface, within `EVENT_TIME_TOL * h` of the crossing.
    pub fn locate(
        &mut self,
        x_prev: &[f64],
        x_new: &[f64],
        t: f64,
        h: f64,
        c_lu: Option<&sparse::TripletLu>,
        slopes: &[Vec<f64>],
    ) -> Option<f64> {
        self.eval(x_new, t + h);
        self.g_new.clear();
        self.g_new.extend_from_slice(&self.out);
        let flagged: Vec<usize> = (0..self.dirs.len())
            .filter(|&k| self.armed[k] && self.dirs[k].crosses(self.g_prev[k], self.g_new[k]))
            .collect();
        if flagged.is_empty() {
            return None;
        }
        let n = x_prev.len();
        let herm = c_lu.map(|lu| {
            (
                solve_mass(lu, &slopes[0], n),
                solve_mass(lu, &slopes[slopes.len() - 1], n),
            )
        });
        let tol = EVENT_TIME_TOL * h;
        let mut best = t + h;
        for &k in &flagged {
            // Illinois regula falsi on g_k along the interpolant; `a` stays on
            // the old side of the surface and is the landing time.
            let (mut a, mut b) = (t, t + h);
            let (mut fa, mut fb) = (self.g_prev[k], self.g_new[k]);
            let mut side = 0i8;
            for _ in 0..EVENT_LOCATE_MAX_ITER {
                if b - a <= tol {
                    break;
                }
                let mut c = b - fb * (b - a) / (fb - fa);
                if !(c > a && c < b) {
                    c = 0.5 * (a + b);
                }
                let xc = match &herm {
                    Some((m0, m1)) => hermite_point(x_prev, x_new, m0, m1, t, h, c, n),
                    None => linear_point(x_prev, x_new, t, t + h, c, n),
                };
                self.eval(&xc, c);
                let fc = self.out[k];
                if fc == 0.0 || (fc < 0.0) != (fa < 0.0) {
                    b = c;
                    fb = fc;
                    if side == 1 {
                        fa *= 0.5;
                    }
                    side = 1;
                } else {
                    a = c;
                    fa = fc;
                    if side == -1 {
                        fb *= 0.5;
                    }
                    side = -1;
                }
            }
            self.roots[k] = a;
            best = best.min(a);
        }
        self.landing.clear();
        self.landing.extend(
            flagged
                .iter()
                .copied()
                .filter(|&k| self.roots[k] <= best + tol)
                .map(|k| {
                    let dir = if Crossing::Rising.crosses(self.g_prev[k], self.g_new[k]) {
                        1
                    } else {
                        -1
                    };
                    (k, dir)
                }),
        );
        Some(best)
    }

    /// Bookkeeping at an accepted step ending at `(x, t)`: fire the surfaces
    /// the step landed on (`landed`) or crossed inside, re-arm the surfaces the
    /// trajectory has left. Returns whether an event fired.
    pub fn accept(&mut self, x: &[f64], t: f64, landed: bool) -> bool {
        self.eval(x, t);
        let mut any = false;
        for k in 0..self.dirs.len() {
            let (g0, g1) = (self.g_prev[k], self.out[k]);
            self.scale[k] = self.scale[k].max(g1.abs());
            if self.armed[k] {
                let landed_on = landed
                    .then(|| self.landing.iter().find(|&&(i, _)| i == k))
                    .flatten()
                    .copied();
                let on_surface = landed_on.is_some();
                if on_surface || self.dirs[k].crosses(g0, g1) {
                    let direction = match landed_on {
                        // The direction the candidate step crossed in, kept
                        // from the detection: at the landing itself the value
                        // is on the near side and says nothing.
                        Some((_, dir)) => dir,
                        None if Crossing::Rising.crosses(g0, g1) => 1,
                        None => -1,
                    };
                    self.fired.push(TransientEvent {
                        t,
                        index: k,
                        direction,
                    });
                    self.armed[k] = false;
                    self.fire_sign[k] = if g1 != 0.0 { g1.signum() } else { g0.signum() };
                    any = true;
                }
            } else {
                let crossed_since = g1 != 0.0 && g1.signum() != self.fire_sign[k];
                let moved_off = g1.abs() > EVENT_REARM_FRAC * self.scale[k];
                if crossed_since || moved_off {
                    self.armed[k] = true;
                }
            }
            self.g_prev[k] = g1;
        }
        self.landing.clear();
        any
    }
}
