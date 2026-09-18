//! The stage machinery every transient method shares: the workspace one
//! integration carries from step to step, and the implicit stage solve.
//!
//! An implicit Runge-Kutta stage (ESDIRK32's γ-stages, the trapezoidal
//! corrector as a γ = 1/2 stage, the consistent restart after a
//! discontinuity as a γ = 1 stage) is the same problem,
//!
//! `r(X) = C·(X − xₙ) − ψ + h·γ·F̃(X, tᵢ) = 0`,   `F̃ = F + gmin·x`,
//!
//! solved by the modified Newton of [`CompiledDc::stage_newton`] against a
//! factorization frozen across the step and refreshed on a stall. What the
//! stages need beyond `(xₙ, ψ, tᵢ, h, γ)` -- the two prolog-split tapes, the
//! stage factorization and whether it is fresh, the constant mass matrix, the
//! scratch buffers, the iteration statistics, the transport-delay history and
//! the integration tolerances -- lives in one [`StageWorkspace`] instead of
//! two dozen parameters per call.

use sane_core::constants::*;

use crate::{newton, sparse, CompiledDc, Convergence, PrologToken, Stats, Symbolic};

/// The constant mass matrix `C = dF/dx'` in triplet form.
pub(crate) struct MassMatrix {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f64>,
}

impl MassMatrix {
    /// `out = C v`.
    pub fn matvec(&self, v: &[f64], out: &mut [f64]) {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for k in 0..self.vals.len() {
            out[self.rows[k]] += self.vals[k] * v[self.cols[k]];
        }
    }

    /// No dynamic element at all: every accepted point is an algebraic solve.
    pub fn is_zero(&self) -> bool {
        self.vals.iter().all(|v| *v == 0.0)
    }

    /// A factorization of `C` for the state-space rate `x' = C⁻¹ slope`,
    /// `None` when `C` is singular (a genuine DAE with algebraic unknowns).
    pub fn factor(&self, n: usize) -> Option<sparse::TripletLu> {
        sparse::factor_triplets_both(n, &self.rows, &self.cols, &self.vals)
    }
}

/// Everything one transient integration carries into every stage solve.
pub(crate) struct StageWorkspace<'a> {
    /// Prolog tokens of the step (residual + Jacobian) and residual-only
    /// tapes: the parameter-pure prefix ran once, every stage evaluation runs
    /// the main phase over the persistent `work` / `res_work` buffers.
    pub step_tok: PrologToken,
    pub res_tok: PrologToken,
    /// The transient-wide stage factorization (numeric-only refactors after
    /// the first), and whether its factors belong to the current iterate and
    /// step size.
    pub fac: sparse::Refactorable<'a>,
    pub fac_fresh: bool,
    pub mass: MassMatrix,
    /// Zero derivative vector: stage residuals are evaluated at `xdot = 0`,
    /// the derivative entering through `C`.
    pub zc: Vec<f64>,
    pub inputs: Vec<f64>,
    pub work: Vec<f64>,
    /// Own buffer for `tape_res` (its layout differs; sharing `work` would
    /// clobber `tape_step`'s prolog slots).
    pub res_work: Vec<f64>,
    pub out: Vec<f64>,
    pub valbuf: Vec<f64>,
    pub cdx: Vec<f64>,
    /// Stage slopes `fᵢ = −F̃(Xᵢ)`; `slopes[0]` is the rate at the step start
    /// and the last entry the rate at the step end (the dense-output contract
    /// every method honours).
    pub slopes: Vec<Vec<f64>>,
    pub psi: Vec<f64>,
    pub stats: Stats,
    /// Transport-delay history and the delays, `None` / empty without delays.
    pub dhist: Option<crate::delay::DelayHistory>,
    pub taus: Vec<f64>,
    pub rtol: f64,
    pub atol: f64,
    pub trace: bool,
}

impl<'a> StageWorkspace<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cdc: &CompiledDc,
        sym: &'a Symbolic,
        x0: &[f64],
        p: &[f64],
        t0: f64,
        taus: Vec<f64>,
        dhist: Option<crate::delay::DelayHistory>,
        rtol: f64,
        atol: f64,
    ) -> Self {
        let n = cdc.n;
        let zc = vec![0.0; n];
        let (rows, cols, vals) = cdc.jacobian_xdot_sparse(x0, &zc, p, t0);
        let mut inputs = Vec::new();
        let (mut work, mut res_work) = (Vec::new(), Vec::new());
        cdc.fill_inputs(x0, &zc, p, t0, &mut inputs);
        let step_tok = cdc.tape_step.eval_prolog(&inputs, &mut work);
        let res_tok = cdc.tape_res.eval_prolog(&inputs, &mut res_work);
        StageWorkspace {
            step_tok,
            res_tok,
            fac: sym.pattern.factorizer(),
            fac_fresh: false,
            mass: MassMatrix { rows, cols, vals },
            zc,
            inputs,
            work,
            res_work,
            out: Vec::new(),
            valbuf: Vec::new(),
            cdx: vec![0.0; n],
            slopes: vec![vec![0.0; n]; ESDIRK32_STAGES],
            psi: vec![0.0; n],
            stats: Stats::default(),
            dhist,
            taus,
            rtol,
            atol,
            trace: sane_core::config().tran_trace,
        }
    }

    /// The integration tolerances as the shared convergence contract (one
    /// floor for every kind: a transient carries `atol` alone).
    pub fn tolerances(&self) -> Convergence {
        Convergence {
            reltol: self.rtol,
            abstol: self.atol,
            vntol: self.atol,
        }
    }

    /// Publish the interpolated delay-history values at stage time `ti` for
    /// every residual / Jacobian evaluation at that time.
    pub fn fill_hist(&mut self, ti: f64) {
        if let Some(h) = self.dhist.as_mut() {
            let vals: Vec<f64> = self
                .taus
                .iter()
                .enumerate()
                .map(|(k, tau)| h.eval(k, ti - tau))
                .collect();
            crate::delay::set_hist_values(&vals);
        }
    }
}

impl CompiledDc {
    /// The shunted residual slope `f = −F̃(x, 0, t)` at `(x, t)`, through the
    /// residual-only tape (an explicit stage needs no Jacobian).
    pub(crate) fn residual_slope(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &[f64],
        t: f64,
    ) -> Vec<f64> {
        self.patch_inputs(x, &ws.zc, t, &mut ws.inputs);
        self.tape_res
            .eval_main(&mut ws.res_tok, &ws.inputs, &mut ws.res_work, &mut ws.out);
        (0..self.n).map(|k| -(ws.out[k] + GMIN_DC * x[k])).collect()
    }

    /// Solve one implicit stage `r(X) = C·(X − xₙ) − ψ + h·γ·F̃(X) = 0` by the
    /// limiting *modified* Newton: reuse the frozen factorization (evaluating
    /// only `F` via the cheap residual tape), refactorizing -- and
    /// re-evaluating `dF/dx` -- only when a stale Jacobian converges too
    /// slowly. `ws.psi` is the stage's history term. Returns
    /// `(X, f = −F̃(X, 0, tᵢ), converged)`.
    pub(crate) fn stage_newton(
        &self,
        ws: &mut StageWorkspace<'_>,
        xn: &[f64],
        guess: &[f64],
        ti: f64,
        h: f64,
        gamma: f64,
    ) -> (Vec<f64>, Vec<f64>, bool) {
        let n = self.n;
        let mut x = guess.to_vec();
        let alpha = 1.0 / (h * gamma);
        let hg = h * gamma;
        let mut step = vec![0.0; n];
        let mut r = vec![0.0; n];
        let mut prev_wn = f64::INFINITY;
        let crit = self.criterion(&ws.tolerances());
        let slope = |out: &[f64], x: &[f64]| -> Vec<f64> {
            (0..n).map(|k| -(out[k] + GMIN_DC * x[k])).collect()
        };

        for it in 0..IRK_STAGE_MAX_ITER {
            ws.stats.iters += 1;
            // Building/refreshing the factorization needs dF/dx (full `tape_step`);
            // a reused factorization needs only F (the cheaper `tape_res`).
            let refresh = !ws.fac_fresh;
            self.patch_inputs(&x, &ws.zc, ti, &mut ws.inputs);
            if refresh {
                ws.stats.refacs += 1;
                self.tape_step
                    .eval_main(&mut ws.step_tok, &ws.inputs, &mut ws.work, &mut ws.out);
                let (jac, valbuf) = (&ws.out[n..], &mut ws.valbuf);
                if !self.factorize_stage(&mut ws.fac, jac, &ws.mass.vals, alpha, GMIN_DC, valbuf) {
                    return (x.clone(), slope(&ws.out, &x), false);
                }
                ws.fac_fresh = true;
            } else {
                self.tape_res
                    .eval_main(&mut ws.res_tok, &ws.inputs, &mut ws.res_work, &mut ws.out);
            }
            // r = C(x - xn) - psi + hγ·F̃,  F̃ = F + gmin·x  (F = out[..n]).
            let dxn: Vec<f64> = (0..n).map(|k| x[k] - xn[k]).collect();
            ws.mass.matvec(&dxn, &mut ws.cdx);
            for k in 0..n {
                r[k] = ws.cdx[k] - ws.psi[k] + hg * (ws.out[k] + GMIN_DC * x[k]);
            }
            let rhs: Vec<f64> = (0..n).map(|k| r[k] / hg).collect();
            let delta = match ws.fac.solve(&rhs) {
                Some(d) => d,
                None => return (x.clone(), slope(&ws.out, &x), false),
            };

            // Scaled update norm (convergence + stall detection).
            let (wn, worst) = crit.update_norm(&delta, &x);
            if ws.trace {
                eprintln!(
                    "tran:     stage t={ti:.9e} it={it} wn={wn:.3e} worst=x[{worst}] delta={:.3e} x={:.6e}{}",
                    delta[worst],
                    x[worst],
                    if refresh { " (refactored)" } else { "" }
                );
            }

            // No step limiting inside a time step: the globalization here is
            // the step size. A stage Newton that diverges rejects the step and
            // `h` shrinks, which shortens the move without bending the Newton
            // direction. Device limiting still applies, on the device's own
            // scale.
            step[..n].copy_from_slice(&delta[..n]);
            if !self.limits.is_empty() {
                let x_new: Vec<f64> = (0..n).map(|k| x[k] - step[k]).collect();
                newton::limit_step(&self.limits, &x, &mut step[..n]);
                if ws.trace {
                    let x_lim: Vec<f64> = (0..n).map(|k| x[k] - step[k]).collect();
                    for (k, lim) in self.limits.iter().enumerate() {
                        let v = |xx: &[f64]| {
                            lim.hi.map_or(0.0, |i| xx[i]) - lim.lo.map_or(0.0, |i| xx[i])
                        };
                        if (v(&x_new) - v(&x_lim)).abs() > 0.0 {
                            eprintln!(
                                "tran:       limit[{k}] {:?}: v_old={:.4e} v_newton={:.4e} v_limited={:.4e}",
                                lim.kind,
                                v(&x),
                                v(&x_new),
                                v(&x_lim)
                            );
                        }
                    }
                }
            }
            for k in 0..n {
                x[k] -= step[k];
            }

            if wn < IRK_STAGE_TOL {
                let f = self.residual_slope(ws, &x, ti);
                return (x, f, true);
            }
            // A reused (stale) Jacobian that is not contracting fast enough: mark it
            // stale so the next iterate re-evaluates dF/dx and refactorizes here.
            // Two tests: the contraction ratio itself, and (Hairer-Wanner IV.8)
            // whether the iteration at this ratio would still reach the tolerance
            // within the remaining budget -- a junction commutating from reverse
            // to forward bias contracts at a steady 0.8 on a frozen exponential
            // slope, which the ratio test alone lets run out the budget.
            if !refresh {
                let theta = wn / prev_wn;
                let remaining = (IRK_STAGE_MAX_ITER - it - 1) as f64;
                let predicted = if theta < 1.0 {
                    theta.powf(remaining) * wn / (1.0 - theta)
                } else {
                    f64::INFINITY
                };
                if theta > IRK_STALL_THETA || predicted > IRK_STAGE_TOL {
                    ws.fac_fresh = false;
                }
            }
            prev_wn = wn;
        }
        let f = slope(&ws.out, &x);
        (x, f, false)
    }

    /// Refresh the stage factorization at `(x, t)` for step size `h` and
    /// stage coefficient `gamma` (the error filters need it at the step end;
    /// a stalled final stage may have left it stale). `false` when singular.
    pub(crate) fn refactor_at(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &[f64],
        t: f64,
        h: f64,
        gamma: f64,
    ) -> bool {
        let n = self.n;
        ws.fill_hist(t);
        self.patch_inputs(x, &ws.zc, t, &mut ws.inputs);
        self.tape_step
            .eval_main(&mut ws.step_tok, &ws.inputs, &mut ws.work, &mut ws.out);
        let (jac, valbuf) = (&ws.out[n..], &mut ws.valbuf);
        let ok = self.factorize_stage(
            &mut ws.fac,
            jac,
            &ws.mass.vals,
            1.0 / (h * gamma),
            GMIN_DC,
            valbuf,
        );
        ws.fac_fresh = ok;
        ok
    }

    /// Consistent restart after a discontinuity at `(t, x)`: one implicit-Euler
    /// step of the vanishing size `delta` into the region beyond it. The
    /// algebraic part of the state jumps there (a switch closes, a source kinks)
    /// while the differential states are pinned by `C/delta`; the returned
    /// state is the right-limit state the next step's explicit first stage
    /// needs -- evaluated at the landing itself it would carry the old
    /// region's slope and degrade the embedded estimate to first order.
    /// `None` when the Newton fails (the caller then restarts unreinitialised).
    pub(crate) fn reinit_step(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &[f64],
        t: f64,
        delta: f64,
    ) -> Option<Vec<f64>> {
        for v in ws.psi.iter_mut() {
            *v = 0.0;
        }
        ws.fac_fresh = false;
        let (xr, _f, ok) = self.stage_newton(ws, x, x, t + delta, delta, 1.0);
        // the factorization now belongs to `delta`, not to any step size
        ws.fac_fresh = false;
        ok.then_some(xr)
    }
}
