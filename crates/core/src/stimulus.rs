//! An independent source's waveform by its parameters: what the analyses
//! read off a source that its lowered expression no longer shows. The
//! transient lands its steps on the waveform's corners (a `C0` corner inside
//! a step breaks the smooth-error assumption), harmonic balance takes its
//! tone, the small-signal analyses drive the source through the parameters
//! that shift its level, and the harmonic-balance continuation ramps a
//! sine's amplitude.
//!
//! A [`Stimulus`] is generic over what names a parameter: the circuit names
//! them (`V1.pulse_td`), the DAE holds their symbols, the solver their
//! columns in the parameter vector. The waveform's semantics live here once.

use crate::constants::BREAKPOINT_MERGE_FRAC;

/// A waveform by its parameters, each a `P`.
#[derive(Clone, Debug, PartialEq)]
pub enum Stimulus<P> {
    /// `offset + amplitude * sin(w t)`, `w` angular.
    Sin { offset: P, amplitude: P, w: P },
    /// `v1` until `td`, then a trapezoid to `v2` (`tr`, `pw`, `tf`)
    /// repeating every `per`.
    Pulse {
        v1: P,
        v2: P,
        td: P,
        tr: P,
        tf: P,
        pw: P,
        per: P,
    },
    /// From `v1` toward `v2` at `td1` (time constant `tau1`), back at `td2`
    /// (`tau2`).
    Exp {
        v1: P,
        v2: P,
        td1: P,
        tau1: P,
        td2: P,
        tau2: P,
    },
    /// Piecewise linear through the points `(t, v)`.
    Pwl(Vec<(P, P)>),
}

impl<P> Stimulus<P> {
    /// The same waveform, each parameter named by `f` of its name here.
    pub fn map<Q>(&self, mut f: impl FnMut(&P) -> Q) -> Stimulus<Q> {
        match self {
            Stimulus::Sin {
                offset,
                amplitude,
                w,
            } => Stimulus::Sin {
                offset: f(offset),
                amplitude: f(amplitude),
                w: f(w),
            },
            Stimulus::Pulse {
                v1,
                v2,
                td,
                tr,
                tf,
                pw,
                per,
            } => Stimulus::Pulse {
                v1: f(v1),
                v2: f(v2),
                td: f(td),
                tr: f(tr),
                tf: f(tf),
                pw: f(pw),
                per: f(per),
            },
            Stimulus::Exp {
                v1,
                v2,
                td1,
                tau1,
                td2,
                tau2,
            } => Stimulus::Exp {
                v1: f(v1),
                v2: f(v2),
                td1: f(td1),
                tau1: f(tau1),
                td2: f(td2),
                tau2: f(tau2),
            },
            Stimulus::Pwl(points) => {
                Stimulus::Pwl(points.iter().map(|(t, v)| (f(t), f(v))).collect())
            }
        }
    }

    /// The parameters that shift the waveform's level as a whole: moved
    /// together, they add their move to the waveform at every instant, the
    /// way a small-signal input drives the source.
    pub fn levels(&self) -> Vec<&P> {
        match self {
            Stimulus::Sin { offset, .. } => vec![offset],
            Stimulus::Pulse { v1, v2, .. } | Stimulus::Exp { v1, v2, .. } => vec![v1, v2],
            Stimulus::Pwl(points) => points.iter().map(|(_, v)| v).collect(),
        }
    }

    /// A sine's amplitude.
    pub fn amplitude(&self) -> Option<&P> {
        match self {
            Stimulus::Sin { amplitude, .. } => Some(amplitude),
            _ => None,
        }
    }

    /// The waveform's corners in `(t0, t1]`, appended to `out`, its
    /// parameters' values by `value` (`None`, an unbound one, takes the
    /// default). A sine has none.
    pub fn breakpoints(
        &self,
        value: impl Fn(&P) -> Option<f64>,
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
            Stimulus::Sin { .. } => {}
            Stimulus::Pulse {
                td,
                tr,
                tf,
                pw,
                per,
                ..
            } => {
                let td = value(td).unwrap_or(0.0);
                let tr = value(tr).unwrap_or(0.0).max(0.0);
                let tf = value(tf).unwrap_or(0.0).max(0.0);
                let pw = value(pw).unwrap_or(0.0).max(0.0);
                let per = value(per).unwrap_or(f64::INFINITY);
                // corners within one period, from the period's start
                let corners = [0.0, tr, tr + pw, tr + pw + tf];
                if !(per > 0.0) || !per.is_finite() {
                    for c in corners {
                        emit(td + c, out);
                    }
                    return;
                }
                // repeat until past t1, the count capped so a tiny period
                // cannot spin
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
            Stimulus::Exp { td1, td2, .. } => {
                // where each exponential switches on
                for td in [td1, td2] {
                    if let Some(td) = value(td) {
                        emit(td, out);
                    }
                }
            }
            Stimulus::Pwl(points) => {
                for (t, _) in points {
                    if let Some(t) = value(t) {
                        emit(t, out);
                    }
                }
            }
        }
    }

    /// The waveform's tone (Hz) where it is periodic with one: a sine's
    /// frequency, a pulse train's `1/per`.
    pub fn fundamental(&self, value: impl Fn(&P) -> Option<f64>) -> Option<f64> {
        match self {
            Stimulus::Sin { w, .. } => value(w).map(|w| w / (2.0 * std::f64::consts::PI)),
            Stimulus::Pulse { per, .. } => match value(per) {
                Some(per) if per > 0.0 && per.is_finite() => Some(1.0 / per),
                _ => None,
            },
            Stimulus::Exp { .. } | Stimulus::Pwl(_) => None,
        }
    }
}

/// Every corner in `(t0, t1]` of the waveforms `stimuli`, sorted, corners
/// within a hair of the span merged (a zero-width rise puts two on one
/// instant).
pub fn breakpoints<'a, P: 'a>(
    stimuli: impl IntoIterator<Item = &'a Stimulus<P>>,
    value: impl Fn(&P) -> Option<f64>,
    t0: f64,
    t1: f64,
) -> Vec<f64> {
    let mut bps = Vec::new();
    for s in stimuli {
        s.breakpoints(&value, t0, t1, &mut bps);
    }
    bps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let merge = (t1 - t0).abs() * BREAKPOINT_MERGE_FRAC;
    bps.dedup_by(|a, b| (*a - *b).abs() <= merge);
    bps
}
