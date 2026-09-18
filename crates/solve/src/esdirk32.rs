//! ESDIRK32 step for the SANE DAE `F(x, x', t) = 0` in mass-matrix form.
//!
//! Kvaerno's four-stage, explicit-first-stage, singly diagonally implicit
//! Runge-Kutta method (`ESDIRK32_*` in the constants module): order 3 with
//! a stiffly accurate embedded order-2 solution, L-stable, every abscissa in
//! `[0, 1]`. Each implicit stage is the shared stage solve of
//! [`crate::stage`] against the factorization frozen across the step.

use sane_core::constants::*;

use crate::stage::StageWorkspace;
use crate::CompiledDc;

/// Butcher `A` of Kvaerno ESDIRK 3/2 in closed form over γ (see
/// `ESDIRK32_GAMMA`):
///   a21 = γ
///   a31 = (−4γ² + 6γ − 1)/(4γ),   a32 = (1 − 2γ)/(4γ)
///   a41 = (6γ − 1)/(12γ),  a42 = −1/((24γ − 12)γ),  a43 = (−6γ² + 6γ − 1)/(6γ − 3)
/// Row 3 is the embedded order-2 solution, row 4 the order-3 solution.
pub(crate) fn a_ij(i: usize, j: usize) -> f64 {
    const G: f64 = ESDIRK32_GAMMA;
    match (i, j) {
        (1, 0) => G,
        (1, 1) => G,
        (2, 0) => (-4.0 * G * G + 6.0 * G - 1.0) / (4.0 * G),
        (2, 1) => (1.0 - 2.0 * G) / (4.0 * G),
        (2, 2) => G,
        (3, 0) => (6.0 * G - 1.0) / (12.0 * G),
        (3, 1) => -1.0 / ((24.0 * G - 12.0) * G),
        (3, 2) => (-6.0 * G * G + 6.0 * G - 1.0) / (6.0 * G - 3.0),
        (3, 3) => G,
        _ => 0.0,
    }
}

impl CompiledDc {
    /// One ESDIRK32 step from `(t, xn)` with size `h`: runs all stages, writes
    /// the stage slopes into `ws.slopes`, and returns the new state
    /// `x_{n+1} = X_s` (stiffly accurate) with the scaled embedded error, or
    /// `None` if any stage Newton diverges.
    pub(crate) fn esdirk32_step(
        &self,
        ws: &mut StageWorkspace<'_>,
        xn: &[f64],
        t: f64,
        h: f64,
    ) -> Option<(Vec<f64>, f64)> {
        let n = self.n;
        let mut x_new = xn.to_vec();
        // Warm start each stage from the previous stage value (close, fewer iters).
        let mut guess = xn.to_vec();
        for i in 0..ESDIRK32_STAGES {
            let ti = t + ESDIRK32_C[i] * h;
            ws.fill_hist(ti);
            if i == 0 {
                // Explicit first stage: X_0 = xn, slope f_0 = -F̃(xn, 0, t).
                let f0 = self.residual_slope(ws, xn, ti);
                ws.slopes[0].copy_from_slice(&f0);
                continue;
            }
            for r in 0..n {
                ws.psi[r] = 0.0;
            }
            for j in 0..i {
                let aij = a_ij(i, j);
                if aij != 0.0 {
                    for r in 0..n {
                        ws.psi[r] += h * aij * ws.slopes[j][r];
                    }
                }
            }
            let (xi, fi, ok) = self.stage_newton(ws, xn, &guess, ti, h, ESDIRK32_GAMMA);
            if !ok {
                return None;
            }
            ws.slopes[i].copy_from_slice(&fi);
            guess = xi.clone();
            if i == ESDIRK32_STAGES - 1 {
                x_new = xi; // stiffly accurate: x_{n+1} = X_s
            }
        }

        // Stabilized (Hairer-Wanner) embedded error estimate. The raw embedded
        // error `h·Σ_i tr_i·f_i` is in residual (current) units; mapping it through
        // the stage iteration matrix `A = dF/dx + C/(hγ) + gmin` (the frozen
        // factorization) gives a solution-space estimate that DAMPS the stiff
        // fast modes -- a tiny-capacitance node's huge raw error is attenuated
        // by its large conductance -- where the naive `1/C[j,j]` scaling instead
        // exploded and stalled the step to underflow on stiff BSIM4 circuits. The
        // `A⁻¹` factor also vanishes with `h`, so a small step is never spuriously
        // rejected (the IC-inconsistency transient at t=0 no longer dead-locks).
        let mut err_raw = vec![0.0; n];
        for r in 0..n {
            let mut s = 0.0;
            for i in 0..ESDIRK32_STAGES {
                s += ESDIRK32_TR[i] * ws.slopes[i][r];
            }
            err_raw[r] = h * s;
        }
        // A factorization is needed to filter the error; a stalled final stage may
        // have invalidated it, so rebuild it at the step endpoint when stale.
        if !ws.fac_fresh && !self.refactor_at(ws, &x_new, t + h, h, ESDIRK32_GAMMA) {
            return None; // singular at the endpoint -> treat as a failed step
        }
        // `factorize_stage` builds `A/(hγ)`, so the solve returns `hγ·A⁻¹·b`;
        // divide by `hγ` to recover the solution-space error `A⁻¹·err_raw ≈ C⁻¹·err_raw`
        // (the voltage LTE) on slow nodes, while on a stiff node `A ≈ hγ·G` makes the
        // `h` cancel, so the estimate stays bounded as the step shrinks.
        let hg = h * ESDIRK32_GAMMA;
        let e_filt = ws.fac.solve(&err_raw)?;
        // WRMS norm (Hairer II.4 / SUNDIALS): `err = sqrt(mean((e_r/sc_r)^2))`.
        // A max norm let a single switching node dictate the step for the whole
        // system, systematically over-restricting circuits whose error is
        // concentrated in a few active nodes (a ring oscillator: 2 of 47).
        let (err, worst) = ws_error_norm(ws, &x_new, &e_filt, hg);
        if ws.trace && err > 1.0 {
            eprintln!(
                "tran:     error worst=x[{worst}] e={:.3e} raw={:.3e} x={:.6e} slopes={:?}",
                e_filt[worst] / hg,
                err_raw[worst],
                x_new[worst],
                (0..ESDIRK32_STAGES)
                    .map(|i| ws.slopes[i][worst])
                    .collect::<Vec<_>>()
            );
        }
        Some((x_new, err.max(IRK_ERR_FLOOR)))
    }
}

/// The WRMS norm of a filtered error estimate against the integration
/// tolerances, `(norm, index of the worst component)`.
pub(crate) fn ws_error_norm(
    ws: &StageWorkspace<'_>,
    x_new: &[f64],
    e_filt: &[f64],
    scale: f64,
) -> (f64, usize) {
    let n = x_new.len();
    let mut acc = 0.0;
    let (mut worst, mut worst_e) = (0usize, 0.0f64);
    for r in 0..n {
        let sc = (ws.atol + ws.rtol * x_new[r].abs()).max(f64::MIN_POSITIVE);
        let e = (e_filt[r] / scale) / sc;
        acc += e * e;
        if e.abs() > worst_e {
            worst_e = e.abs();
            worst = r;
        }
    }
    ((acc / n as f64).sqrt(), worst)
}
