//! Curve-aware Newton-step limiting of device controlling voltages: the classic
//! SPICE convergence aids `pnjlim` (PN junctions) and `fetlim` (FET channels).
//!
//! Limiting is applied *between* Newton iterates: the proposed controlling
//! voltage is pulled back along its own curve so the device exponential /
//! square-law cannot overshoot and send the next residual to infinity. It shapes
//! only the iteration path -- once the step is small the limiting is inactive, so
//! the converged fixed point is identical with or without it. The numeric
//! thresholds live in [`sane_core::constants`] (see the rationale there); the device
//! declares only which voltages to limit and their kind (see [`sane_dae::Limit`]).

use sane_core::constants::{LIMIT_VCRIT, LIMIT_VT, LIMIT_VTO};
use sane_dae::{Limit, LimitKind};

/// SPICE `pnjlim`: limit a forward PN-junction voltage update so the junction
/// exponential grows at most one decade per step. Returns the limited `vnew`.
/// Inactive below the critical voltage or for small steps (then `vnew` passes
/// through unchanged), so it never perturbs a nearly-converged junction.
pub fn pnjlim(vnew: f64, vold: f64) -> f64 {
    let (vt, vcrit) = (LIMIT_VT, LIMIT_VCRIT);
    if vnew > vcrit && (vnew - vold).abs() > 2.0 * vt {
        if vold > 0.0 {
            // Step onto the logarithmic curve relative to the old voltage.
            let arg = 1.0 + (vnew - vold) / vt;
            if arg > 0.0 {
                vold + vt * arg.ln()
            } else {
                vcrit
            }
        } else {
            // Coming up from reverse bias: snap onto the log curve (vnew > vcrit > 0).
            vt * (vnew / vt).ln()
        }
    } else {
        vnew
    }
}

/// SPICE `fetlim`: bound a FET gate-control voltage update around the threshold
/// so the square-law channel current cannot overshoot. Mirrors SPICE3
/// `DEVfetlim` with the threshold from [`LIMIT_VTO`].
pub fn fetlim(vnew: f64, vold: f64) -> f64 {
    let vto = LIMIT_VTO;
    let vtsthi = (2.0 * (vold - vto)).abs() + 2.0;
    let vtstlo = 0.5 * vtsthi + 2.0;
    let vtox = vto + 3.5;
    let delv = vnew - vold;
    let mut v = vnew;
    if vold >= vto {
        if vold >= vtox {
            if delv <= 0.0 {
                if vnew >= vtox {
                    if -delv > vtstlo {
                        v = vold - vtstlo;
                    }
                } else {
                    v = vnew.max(vto + 2.0);
                }
            } else if delv > vtsthi {
                v = vold + vtsthi;
            }
        } else if delv <= 0.0 {
            if vnew < vto + 0.5 {
                v = vnew.max(vto - 0.5);
            }
        } else {
            v = vnew.min(vto + 4.0);
        }
    } else if delv <= 0.0 {
        if -delv > vtsthi {
            v = vold - vtsthi;
        }
    } else if vnew > vto + 0.5 {
        v = vto + 0.5;
    }
    v
}

#[inline]
fn volt(x: &[f64], idx: Option<usize>) -> f64 {
    idx.map_or(0.0, |i| x[i])
}

/// Apply per-device voltage limiting to the proposed iterate `x_new`, given the
/// current iterate `x_old`. For each controlling voltage the limited difference
/// is computed and the correction distributed back onto its (non-ground)
/// endpoints, so the limited difference is realized while preserving the common
/// mode. Returns the limited iterate (a no-op when `limits` is empty).
pub fn apply(limits: &[Limit], x_old: &[f64], x_new: &[f64]) -> Vec<f64> {
    let alpha = fraction(limits, x_old, x_new);
    if alpha >= 1.0 {
        return x_new.to_vec();
    }
    x_old
        .iter()
        .zip(x_new)
        .map(|(o, n)| o + alpha * (n - o))
        .collect()
}

/// The fraction of the move `x_old -> x_new` every limited junction allows
/// (`1.0` when nothing limits).
pub fn fraction(limits: &[Limit], x_old: &[f64], x_new: &[f64]) -> f64 {
    // The Newton step is shortened as a whole, by the largest fraction that
    // keeps every limited junction within its own curve-aware bound. Moving
    // the endpoints of each junction separately (the SPICE-style node
    // correction) is not consistent between iterates here: two junctions on a
    // shared node re-limited each other, and the half-split of a correction
    // threw the junction's far node off by the whole overshoot. A shortened
    // Newton step keeps the direction, the limited junction lands exactly on
    // its bound, and the rest of the circuit moves in proportion; the next
    // iteration relinearizes there.
    let mut alpha = 1.0f64;
    for lim in limits {
        let v_old = volt(x_old, lim.hi) - volt(x_old, lim.lo);
        let v_new = volt(x_new, lim.hi) - volt(x_new, lim.lo);
        let v_lim = match lim.kind {
            LimitKind::PnJunction => pnjlim(v_new, v_old),
            LimitKind::Fet => fetlim(v_new, v_old),
        };
        let d_newton = v_new - v_old;
        let d_lim = v_lim - v_old;
        if v_lim != v_new && d_newton != 0.0 {
            let a = (d_lim / d_newton).clamp(0.0, 1.0);
            alpha = alpha.min(a);
        }
    }
    alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pnjlim_passes_small_steps_limits_large() {
        // Below the critical voltage, or for a small step, the proposed voltage
        // passes through unchanged (the limiting is inactive near convergence).
        assert_eq!(pnjlim(0.5, 0.4), 0.5);
        assert_eq!(pnjlim(0.65, 0.649), 0.65);
        // A large forward step above vcrit is logarithmically pulled back, so the
        // limited voltage is strictly less than the raw step but still advances.
        let vold = 0.65;
        let vnew = 5.0;
        let vlim = pnjlim(vnew, vold);
        assert!(
            vlim > vold && vlim < vnew,
            "expected vold < vlim < vnew, got {vlim}"
        );
        // Idempotent at the limited point: re-limiting a now-small step is a no-op.
        assert_eq!(pnjlim(vlim, vlim), vlim);
    }

    #[test]
    fn fetlim_bounds_large_gate_steps() {
        // A large positive gate-voltage step above threshold is bounded.
        let (vold, vnew) = (0.6_f64, 50.0_f64);
        let vlim = fetlim(vnew, vold);
        assert!(
            vlim > vold && vlim < vnew,
            "fetlim should bound the step: {vlim}"
        );
        // Small steps pass through.
        assert_eq!(fetlim(0.7, 0.69), 0.7);
    }
}
