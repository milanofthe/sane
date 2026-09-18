//! Discrete transient adjoint of a fixed-grid ESDIRK32 transient.
//!
//! Forward pass: one ESDIRK32 step per `t_eval` interval (no substepping, no
//! error control), storing every implicit stage state. Writing the stage
//! equations with the regularized residual `Φ(x, x', t) = F(x, x', p, t) +
//! gmin·x`,
//!
//!   Gᵢ = h·γ·Φ(Xᵢ, (Xᵢ − xₙ)/(hγ), tᵢ) − h·Σ_{j<i} aᵢⱼ·fⱼ = 0,  fⱼ = −Φ(Xⱼ, 0, tⱼ),
//!
//! with `xₙ₊₁ = X_{S−1}` (stiffly accurate) and `X₀ = xₙ` (the explicit first
//! stage). The slopes `f` live in residual space, the only form that survives
//! the singular mass matrix of a genuine DAE. Every quantity is evaluated at
//! its own stage point, so a state-dependent `dF/dx'` stays exact -- unlike
//! [`crate::transient`], which freezes the mass matrix at the operating point.
//!
//! Backward pass: the Lagrangian `L + Σₙ Σᵢ λᵢᵀ Gᵢ` is stationary, which runs
//! the stages in REVERSE order -- a DIRK transposed is again triangular, so no
//! coupled stage system appears. With `λ̃ᵢ = hγ·λᵢ`, `Mᵢ = dΦ/dx|(Xᵢ,wᵢ) +
//! C(Xᵢ)/(hγ)` the stage matrix the forward Newton already factors, `Jⱼ =
//! dΦ/dx|(Xⱼ,0)` and `σ̃ⱼ = Σ_{m>j} a_{mj}·λ̃ₘ`:
//!
//!   Mᵢᵀ λ̃ᵢ = −[gₙ₊₁ + carry]·δ_{i,S−1} − Jᵢᵀ σ̃ᵢ / γ,        i = S−1 … 1
//!   carry  = −Σᵢ C(Xᵢ)ᵀ λ̃ᵢ / (hγ) + J₀ᵀ σ̃₀ / γ              (→ step n−1)
//!   dL/dp += Σᵢ (dΦ/dp)|(Xᵢ,wᵢ)ᵀ λ̃ᵢ + Σⱼ (dΦ/dp)|(Xⱼ,0)ᵀ σ̃ⱼ / γ
//!
//! plus the operating-point shift through one DC transpose solve. For a single
//! implicit stage (`γ = 1`, `S = 2`) this reduces term by term to the
//! backward-Euler adjoint it replaces. The gradient is EXACT for this discrete
//! trajectory -- the one
//! [`solve_transient_grid`](CompiledDc::solve_transient_grid) returns -- which
//! is what a gradient-based optimizer differentiates.
//!
//! Cost: `S−1` transposed stage solves per step, independent of the parameter
//! count -- the complement of the forward sensitivity path
//! (`augment_with_sensitivities`), whose cost is linear in the parameters.
//! Memory: the stored trajectory plus its stage states, O(S · len(t_eval) · n).

use sane_core::constants::*;

use crate::{esdirk32::a_ij, limiting, sparse, CompiledDc};

/// The implicit stage states `X₁ … X_{S−1}` of one step (`X₀` is the previous
/// grid state, `X_{S−1}` the next one).
type StepStages = Vec<Vec<f64>>;

impl CompiledDc {
    /// Fixed-grid ESDIRK32 transient on `t_eval`: one implicit step per
    /// interval, no substepping and no error control -- the caller's grid IS
    /// the step sequence, which is what makes the trajectory exactly
    /// differentiable. `x0` of matching dimension seeds the initial state;
    /// empty runs the internal DC operating point. Returns the state at every
    /// grid point (including `t_eval[0]`).
    ///
    /// This is the forward pass [`transient_adjoint`](Self::transient_adjoint)
    /// differentiates; it is public so callers can evaluate the SAME discrete
    /// objective the adjoint gradient belongs to (e.g. finite-difference
    /// validation, line searches in an optimizer).
    pub fn solve_transient_grid(
        &self,
        p: &[f64],
        x0: &[f64],
        t_eval: &[f64],
        dc_guess: &[f64],
    ) -> Result<Vec<Vec<f64>>, String> {
        self.esdirk32_grid(p, x0, t_eval, dc_guess)
            .map(|(states, _)| states)
    }

    /// [`solve_transient_grid`](Self::solve_transient_grid) plus the stage
    /// states the backward sweep re-linearizes at.
    fn esdirk32_grid(
        &self,
        p: &[f64],
        x0: &[f64],
        t_eval: &[f64],
        dc_guess: &[f64],
    ) -> Result<(Vec<Vec<f64>>, Vec<StepStages>), String> {
        let _stage = sane_core::log::scope("tran_grid");
        if self.has_delays() {
            return Err("solve_transient_grid: transport delays (tline/absdelay) are not supported on the adjoint path yet".into());
        }
        let n = self.n;
        if t_eval.len() < 2 {
            return Err("solve_transient_grid: need at least two time points".into());
        }
        let x_init = if x0.len() == n {
            x0.to_vec()
        } else {
            // a warm DC guess (e.g. the previous optimizer iterate's operating
            // point) only seeds Newton -- the solved point is the same fixed point
            let (x, conv, _) = self.solve_dc(p, dc_guess, TRANSIENT_ATOL, GMIN_STEP_MAX_ITER);
            if !conv {
                return Err("solve_transient_grid: DC operating point did not converge".into());
            }
            x
        };
        let zeros = vec![0.0; n];
        let mut states = Vec::with_capacity(t_eval.len());
        let mut traces = Vec::with_capacity(t_eval.len() - 1);
        states.push(x_init);
        for k in 1..t_eval.len() {
            let h = t_eval[k] - t_eval[k - 1];
            if !(h > 0.0) {
                return Err("solve_transient_grid: t_eval must be strictly increasing".into());
            }
            let tn = t_eval[k - 1];
            let hg = h * ESDIRK32_GAMMA;
            let xn = states[k - 1].clone();
            let mut slopes = vec![vec![0.0; n]; ESDIRK32_STAGES];
            slopes[0] = self.neg_phi(&xn, &zeros, p, tn);
            let mut stages: StepStages = Vec::with_capacity(ESDIRK32_STAGES - 1);
            let mut guess = xn.clone();
            for i in 1..ESDIRK32_STAGES {
                let ti = tn + ESDIRK32_C[i] * h;
                // ψᵢ = h·Σ_{j<i} aᵢⱼ·fⱼ -- only earlier stages, the explicit coupling
                let mut psi = vec![0.0; n];
                for j in 0..i {
                    let a = a_ij(i, j);
                    if a != 0.0 {
                        for r in 0..n {
                            psi[r] += h * a * slopes[j][r];
                        }
                    }
                }
                let xi = self.grid_stage_newton(p, &xn, &guess, &psi, ti, hg)?;
                slopes[i] = self.neg_phi(&xi, &zeros, p, ti);
                guess = xi.clone();
                stages.push(xi);
            }
            states.push(stages[ESDIRK32_STAGES - 2].clone()); // stiffly accurate
            traces.push(stages);
        }
        Ok((states, traces))
    }

    /// One implicit stage: `Φ(X, (X − xₙ)/(hγ), tᵢ) − ψ/(hγ) = 0` by full
    /// Newton (a fresh factorization per iterate).
    ///
    /// The tolerance is far tighter than the production stage tolerance: the
    /// adjoint assumes the discrete stage residuals vanish EXACTLY, and
    /// whatever is left over leaks first-order error into the gradient.
    /// Globalization is device limiting on the device's own scale -- the same
    /// `pnjlim`/`fetlim` the DC and transient solves use. There is no step size
    /// to fall back on here (the grid IS the step sequence), so a stage that
    /// will not converge is reported rather than papered over.
    fn grid_stage_newton(
        &self,
        p: &[f64],
        xn: &[f64],
        guess: &[f64],
        psi: &[f64],
        ti: f64,
        hg: f64,
    ) -> Result<Vec<f64>, String> {
        let n = self.n;
        let mut x = guess.to_vec();
        for _ in 0..IRK_STAGE_MAX_ITER {
            let w: Vec<f64> = (0..n).map(|i| (x[i] - xn[i]) / hg).collect();
            let mut r = self.phi(&x, &w, p, ti);
            for i in 0..n {
                r[i] -= psi[i] / hg;
            }
            let lu = self
                .factor_stage(&x, &w, p, ti, hg)
                .ok_or_else(|| format!("solve_transient_grid: singular system at t={ti:.3e}"))?;
            let delta = lu
                .solve(&r)
                .ok_or_else(|| format!("solve_transient_grid: solve failed at t={ti:.3e}"))?;
            let mut wn: f64 = 0.0;
            for i in 0..n {
                let sc = (TRANSIENT_ATOL + TRANSIENT_RTOL * x[i].abs()).max(f64::MIN_POSITIVE);
                wn = wn.max((delta[i] / sc).abs());
            }
            let x_new: Vec<f64> = (0..n).map(|i| x[i] - delta[i]).collect();
            x = if self.limits.is_empty() {
                x_new
            } else {
                limiting::apply(&self.limits, &x, &x_new)
            };
            if wn < ADJOINT_STAGE_TOL {
                return Ok(x);
            }
        }
        Err(format!(
            "solve_transient_grid: stage Newton did not converge at t={ti:.3e}"
        ))
    }

    /// Discrete adjoint of [`solve_transient_grid`](Self::solve_transient_grid):
    /// given the cotangents `dL/dx_k` at every grid state (one length-`n`
    /// vector per `t_eval` entry), return `dL/dp` for EVERY parameter, in
    /// `param_names` order.
    ///
    /// With `x0` empty the initial state is the DC operating point and its
    /// parameter dependence is included through one DC transpose solve; a
    /// caller-supplied `x0` is treated as parameter-independent.
    ///
    /// Requires the parameter Jacobian tape (`ensure_param_jac`), like the
    /// other sensitivity entry points.
    pub fn transient_adjoint(
        &self,
        p: &[f64],
        x0: &[f64],
        t_eval: &[f64],
        cotangent: &[Vec<f64>],
        dc_guess: &[f64],
    ) -> Result<Vec<f64>, String> {
        let n = self.n;
        let s = ESDIRK32_STAGES;
        if cotangent.len() != t_eval.len() {
            return Err("transient_adjoint: need one cotangent per time point".into());
        }
        if cotangent.iter().any(|c| c.len() != n) {
            return Err("transient_adjoint: cotangent dimension mismatch".into());
        }
        let user_ic = x0.len() == n;
        let (states, traces) = self.esdirk32_grid(p, x0, t_eval, dc_guess)?;

        let _stage = sane_core::log::scope("backward");
        let zeros = vec![0.0; n];
        let mut dldp = vec![0.0; self.param_syms.len()];
        // dL/dxₙ contributed by the step that follows; zero entering the last one
        let mut carry = vec![0.0; n];
        for k in (1..t_eval.len()).rev() {
            let h = t_eval[k] - t_eval[k - 1];
            let hg = h * ESDIRK32_GAMMA;
            let tn = t_eval[k - 1];
            let xn = &states[k - 1];
            let stages = &traces[k - 1];
            let stage_x = |i: usize| -> &[f64] {
                if i == 0 {
                    xn
                } else {
                    &stages[i - 1]
                }
            };
            let stage_t = |i: usize| tn + ESDIRK32_C[i] * h;
            // wᵢ = (Xᵢ − xₙ)/(hγ), the derivative argument the stage was solved at
            let stage_w = |i: usize| -> Vec<f64> {
                let xi = stage_x(i);
                (0..n).map(|r| (xi[r] - xn[r]) / hg).collect()
            };

            let mut lam = vec![vec![0.0; n]; s];
            let mut sig = vec![vec![0.0; n]; s];
            for i in (1..s).rev() {
                // σ̃ᵢ = Σ_{m>i} a_{mi}·λ̃ₘ -- the explicit coupling, transposed
                for m in (i + 1)..s {
                    let a = a_ij(m, i);
                    if a != 0.0 {
                        for r in 0..n {
                            sig[i][r] += a * lam[m][r];
                        }
                    }
                }
                // only the last stage is a grid state, so only it carries a cotangent
                let mut rhs = vec![0.0; n];
                if i == s - 1 {
                    for r in 0..n {
                        rhs[r] = -(cotangent[k][r] + carry[r]);
                    }
                }
                if sig[i].iter().any(|v| *v != 0.0) {
                    let jt = self.phi_x_transpose_mul(stage_x(i), &zeros, p, stage_t(i), &sig[i]);
                    for r in 0..n {
                        rhs[r] -= jt[r] / ESDIRK32_GAMMA;
                    }
                }
                let ti = stage_t(i);
                let lu = self
                    .factor_stage(stage_x(i), &stage_w(i), p, ti, hg)
                    .ok_or_else(|| format!("transient_adjoint: singular system at t={ti:.3e}"))?;
                lam[i] = lu.solve_transpose(&rhs).ok_or_else(|| {
                    format!("transient_adjoint: transpose solve failed at t={ti:.3e}")
                })?;
            }
            for m in 1..s {
                let a = a_ij(m, 0);
                if a != 0.0 {
                    for r in 0..n {
                        sig[0][r] += a * lam[m][r];
                    }
                }
            }

            // dL/dp: the implicit stage term (evaluated with the stage's own x',
            // so the tape carries the reactive parameters) plus the explicit
            // coupling term through the slopes.
            for i in 1..s {
                self.accumulate_phi_p(
                    &mut dldp,
                    stage_x(i),
                    &stage_w(i),
                    p,
                    stage_t(i),
                    &lam[i],
                    1.0,
                );
            }
            for j in 0..(s - 1) {
                if sig[j].iter().any(|v| *v != 0.0) {
                    let sc = 1.0 / ESDIRK32_GAMMA;
                    self.accumulate_phi_p(
                        &mut dldp,
                        stage_x(j),
                        &zeros,
                        p,
                        stage_t(j),
                        &sig[j],
                        sc,
                    );
                }
            }

            // carry to step k−1: every stage depends on xₙ both through the mass
            // term and through the explicit first stage f₀.
            for c in carry.iter_mut() {
                *c = 0.0;
            }
            for m in 1..s {
                let (cr, cc, cv) =
                    self.jacobian_xdot_sparse(stage_x(m), &stage_w(m), p, stage_t(m));
                for i in 0..cv.len() {
                    carry[cc[i]] -= cv[i] * lam[m][cr[i]] / hg;
                }
            }
            if sig[0].iter().any(|v| *v != 0.0) {
                let jt = self.phi_x_transpose_mul(xn, &zeros, p, tn, &sig[0]);
                for r in 0..n {
                    carry[r] += jt[r] / ESDIRK32_GAMMA;
                }
            }
        }

        // Operating-point shift: what is left on dL/dx₀ lands on dx₀/dp; with
        // G_dcᵀ μ = −g₀ the contribution is μᵀ (dΦ/dp)|(x₀, x'=0, t₀).
        if !user_ic {
            let x_init = &states[0];
            let t0 = t_eval[0];
            let g0: Vec<f64> = (0..n).map(|i| cotangent[0][i] + carry[i]).collect();
            if g0.iter().any(|v| *v != 0.0) {
                let (mut jr, mut jc, mut jv) = self.jacobian_x_sparse(x_init, &zeros, p, t0);
                for i in 0..n {
                    jr.push(i);
                    jc.push(i);
                    jv.push(GMIN_DC);
                }
                let lu = sparse::factor_triplets_both(n, &jr, &jc, &jv)
                    .ok_or("transient_adjoint: singular DC Jacobian for the IC term")?;
                let rhs: Vec<f64> = g0.iter().map(|v| -v).collect();
                let mu = lu
                    .solve_transpose(&rhs)
                    .ok_or_else(|| "transient_adjoint: DC transpose solve failed".to_string())?;
                self.accumulate_phi_p(&mut dldp, x_init, &zeros, p, t0, &mu, 1.0);
            }
        }
        Ok(dldp)
    }

    /// Regularized residual `Φ = F(x, x', p, t) + gmin·x`, the quantity every
    /// equation on this path is written in.
    fn phi(&self, x: &[f64], xdot: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let mut r = self.residual(x, xdot, p, t);
        for i in 0..self.n {
            r[i] += GMIN_DC * x[i];
        }
        r
    }

    /// `−Φ`: a stage slope in residual space.
    fn neg_phi(&self, x: &[f64], xdot: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let mut r = self.phi(x, xdot, p, t);
        for v in r.iter_mut() {
            *v = -*v;
        }
        r
    }

    /// `(dΦ/dx)ᵀ·u` at the point, without forming the matrix.
    fn phi_x_transpose_mul(
        &self,
        x: &[f64],
        xdot: &[f64],
        p: &[f64],
        t: f64,
        u: &[f64],
    ) -> Vec<f64> {
        let (jr, jc, jv) = self.jacobian_x_sparse(x, xdot, p, t);
        let mut out = vec![0.0; self.n];
        for i in 0..jv.len() {
            out[jc[i]] += jv[i] * u[jr[i]];
        }
        for i in 0..self.n {
            out[i] += GMIN_DC * u[i];
        }
        out
    }

    /// `dldp += scale · (dΦ/dp)ᵀ·u` at the point (the `gmin·x` shunt carries no
    /// parameter dependence, so `dΦ/dp = dF/dp`).
    fn accumulate_phi_p(
        &self,
        dldp: &mut [f64],
        x: &[f64],
        xdot: &[f64],
        p: &[f64],
        t: f64,
        u: &[f64],
        scale: f64,
    ) {
        let (pr, pc, pv) = self.jacobian_p_sparse(x, xdot, p, t);
        for i in 0..pv.len() {
            dldp[pc[i]] += scale * pv[i] * u[pr[i]];
        }
    }

    /// Factor the stage matrix `dΦ/dx + dF/dx'/(hγ)` at the point, through the
    /// KLU backend (the adjoint needs its transpose solve).
    fn factor_stage(
        &self,
        x: &[f64],
        xdot: &[f64],
        p: &[f64],
        t: f64,
        hg: f64,
    ) -> Option<sparse::TripletLu> {
        let n = self.n;
        let (mut jr, mut jc, mut jv) = self.jacobian_x_sparse(x, xdot, p, t);
        let (cr, cc, cv) = self.jacobian_xdot_sparse(x, xdot, p, t);
        jr.extend_from_slice(&cr);
        jc.extend_from_slice(&cc);
        jv.extend(cv.iter().map(|v| v / hg));
        for i in 0..n {
            jr.push(i);
            jc.push(i);
            jv.push(GMIN_DC);
        }
        sparse::factor_triplets_both(n, &jr, &jc, &jv)
    }
}
