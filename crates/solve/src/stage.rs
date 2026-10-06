//! The stage machinery every transient method shares: the workspace one
//! integration carries from step to step, and the implicit stage solve.
//!
//! Every row of the DAE reads `I(x, t) + d/dt Q(x)`: `I` the residual at
//! rest, `Q` the charge it stores. The methods integrate the charges,
//! `dQ/dt = −Ĩ` with `Ĩ = I + gmin·x`, and an implicit Runge-Kutta stage
//! (ESDIRK32's γ-stages, the trapezoidal corrector as a γ = 1/2 stage, the
//! consistent start and the restart after a discontinuity as γ = 1 stages)
//! is the same problem,
//!
//! `r(X) = Q(X) − Q(xₙ) − ψ + h·γ·Ĩ(X, tᵢ) = 0`,
//!
//! with the slopes `f = −Ĩ` in charge space. A nonlinear capacitance is
//! exact at every stage, and the integration conserves charge. The stage is
//! solved by the modified Newton of [`CompiledDc::stage_newton`] against the
//! stage matrix `C(X)/(hγ) + G(X) + gmin`, `G = dI/dx` and `C = dQ/dx`,
//! factorized once and refreshed on a stall. What the stages need beyond
//! `(ψ, tᵢ, h, γ)` -- the two prolog-split tapes, the stage factorization
//! and whether it is fresh, the charge at the step start, the mass matrix,
//! the scratch buffers, the iteration statistics, the transport-delay
//! history and the integration tolerances -- lives in one
//! [`StageWorkspace`] instead of two dozen parameters per call.

use sane_core::constants::*;

use crate::{newton, sparse, CompiledDc, Convergence, PrologToken, Stats, Symbolic};

/// The mass matrix `C = dQ/dx` in triplet form, in the jacobian-x' pattern: its
/// values where the stage matrix was last factorized, and, when `C` is
/// regular, its factorization at any state for the state rates
/// `x' = C⁻¹ f`.
pub(crate) struct MassMatrix<'a> {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    /// `C` where the stage matrix was last factorized.
    pub vals: Vec<f64>,
    /// `None` when `C` is singular (a genuine DAE with algebraic unknowns).
    lu: Option<sparse::Refactorable<'a>>,
    n: usize,
    valbuf: Vec<f64>,
    /// The inputs, work and values of `C` evaluated for a rate.
    inputs: Vec<f64>,
    work: Vec<f64>,
    at: Vec<f64>,
}

impl<'a> MassMatrix<'a> {
    /// `C` at the state `inputs` holds, factorized on `sym` when regular.
    fn new(cdc: &CompiledDc, sym: Option<&'a Symbolic>, inputs: &[f64]) -> Self {
        let mut m = MassMatrix {
            rows: cdc.jxd_rows.clone(),
            cols: cdc.jxd_cols.clone(),
            vals: Vec::new(),
            lu: sym.map(|s| s.pattern.factorizer()),
            n: cdc.n,
            valbuf: Vec::new(),
            inputs: inputs.to_vec(),
            work: Vec::new(),
            at: Vec::new(),
        };
        cdc.tape_c.eval(&m.inputs, &mut m.work, &mut m.at);
        m.vals = m.at.clone();
        if !m.factor() {
            m.lu = None;
        }
        m
    }

    /// Factorize `C` at the values `at` holds.
    fn factor(&mut self) -> bool {
        let Some(lu) = self.lu.as_mut() else {
            return false;
        };
        self.valbuf.clear();
        self.valbuf.extend_from_slice(&self.at);
        self.valbuf.resize(self.at.len() + self.n, 0.0);
        lu.factor(&self.valbuf, false)
    }

    /// `out = C v`.
    pub fn matvec(&self, v: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for k in 0..self.vals.len() {
            out[self.rows[k]] += self.vals[k] * v[self.cols[k]];
        }
    }

    /// No dynamic element at all: every accepted point is an algebraic solve.
    pub fn is_zero(&self) -> bool {
        self.vals.iter().all(|v| *v == 0.0)
    }

    /// The state rate `x' = C(x)⁻¹ slope` into `out`, `C` evaluated at
    /// `(x, t)`; `false` when `C` is singular.
    pub fn rate_at(
        &mut self,
        cdc: &CompiledDc,
        x: &[f64],
        t: f64,
        slope: &[f64],
        out: &mut [f64],
    ) -> bool {
        if self.lu.is_none() {
            return false;
        }
        cdc.patch_inputs(x, t, &mut self.inputs);
        cdc.tape_c.eval(&self.inputs, &mut self.work, &mut self.at);
        self.factor() && self.lu.as_mut().is_some_and(|lu| lu.solve_into(slope, out))
    }
}

/// Everything one transient integration carries into every stage solve.
pub(crate) struct StageWorkspace<'a> {
    /// Prolog tokens of the step (`I`, `Q`, `G`, `C`) and residual-only
    /// (`I`, `Q`) tapes: the parameter-pure prefix ran once, every stage
    /// evaluation runs the main phase over the persistent `work` /
    /// `res_work` buffers.
    pub step_tok: PrologToken,
    pub res_tok: PrologToken,
    /// The transient-wide stage factorization (numeric-only refactors after
    /// the first), and whether its factors belong to the current iterate and
    /// step size.
    pub fac: sparse::Refactorable<'a>,
    pub fac_fresh: bool,
    pub mass: MassMatrix<'a>,
    pub inputs: Vec<f64>,
    pub work: Vec<f64>,
    /// Own buffer for the residual tape (its layout differs; sharing `work`
    /// would clobber the step tape's prolog slots).
    pub res_work: Vec<f64>,
    pub out: Vec<f64>,
    pub valbuf: Vec<f64>,
    pub cdx: Vec<f64>,
    /// The charge at the step start, and at the latest evaluation.
    pub qn: Vec<f64>,
    pub q: Vec<f64>,
    /// `|G|` then `|C|` at the latest stage factorization, and per row the
    /// magnitude `|G| |x| + |C| |x| / hγ` its residual is computed from (the
    /// residual's rounding floor).
    pub jmag: Vec<f64>,
    pub rowmag: Vec<f64>,
    /// The stage Newton's scratch: the scaled right-hand side and the
    /// Newton step. The error estimates reuse both once the stages are
    /// solved.
    pub rhs: Vec<f64>,
    pub step: Vec<f64>,
    /// The slope `f = −Ĩ(x)` of the last stage solve or residual slope; a
    /// method swaps it into its `slopes` entry.
    pub f: Vec<f64>,
    /// Stage slopes `fᵢ = −Ĩ(Xᵢ)`; `slopes[0]` is the rate at the step start
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
    /// The workspace of an integration from `(t0, x0)`: the stage
    /// factorization on `sym`, the mass matrix's on `mass_sym`, and the
    /// charge at `x0` as the first step's start.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cdc: &CompiledDc,
        sym: &'a Symbolic,
        mass_sym: Option<&'a Symbolic>,
        x0: &[f64],
        p: &[f64],
        t0: f64,
        taus: Vec<f64>,
        dhist: Option<crate::delay::DelayHistory>,
        rtol: f64,
        atol: f64,
    ) -> Self {
        let n = cdc.n;
        let mut inputs = Vec::new();
        let (mut work, mut res_work) = (Vec::new(), Vec::new());
        cdc.fill_inputs(x0, p, t0, &mut inputs);
        let step_tok = cdc.tape_tran_step.eval_prolog(&inputs, &mut work);
        let res_tok = cdc.tape_tran_res.eval_prolog(&inputs, &mut res_work);
        let mut ws = StageWorkspace {
            step_tok,
            res_tok,
            fac: sym.pattern.factorizer(),
            fac_fresh: false,
            mass: MassMatrix::new(cdc, mass_sym, &inputs),
            inputs,
            work,
            res_work,
            out: Vec::new(),
            valbuf: Vec::new(),
            cdx: vec![0.0; n],
            qn: vec![0.0; n],
            q: vec![0.0; n],
            jmag: Vec::new(),
            rowmag: vec![0.0; n],
            rhs: vec![0.0; n],
            step: vec![0.0; n],
            f: vec![0.0; n],
            slopes: vec![vec![0.0; n]; ESDIRK32_STAGES],
            psi: vec![0.0; n],
            stats: Stats::default(),
            dhist,
            taus,
            rtol,
            atol,
            trace: sane_core::config().tran_trace,
        };
        ws.fill_hist(t0);
        cdc.residual_slope(&mut ws, x0, t0);
        ws.advance();
        ws
    }

    /// The state moved to where the latest evaluation was: its charge is
    /// the next step's start.
    pub fn advance(&mut self) {
        self.qn.copy_from_slice(&self.q);
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
            let taus = self.taus.iter().enumerate();
            crate::delay::set_hist_values(taus.map(|(k, tau)| h.eval(k, ti - tau)));
        }
    }
}

impl CompiledDc {
    /// The shunted slope `f = −Ĩ(x, t)` at `(x, t)` into `ws.f` and the
    /// charge `Q(x)` into `ws.q`, through the residual-only tape (an
    /// explicit stage needs no Jacobian).
    pub(crate) fn residual_slope(&self, ws: &mut StageWorkspace<'_>, x: &[f64], t: f64) {
        let n = self.n;
        self.patch_inputs(x, t, &mut ws.inputs);
        self.tape_tran_res
            .eval_main(&mut ws.res_tok, &ws.inputs, &mut ws.res_work, &mut ws.out);
        for k in 0..n {
            ws.f[k] = -(ws.out[k] + GMIN_DC * x[k]);
        }
        ws.q.copy_from_slice(&ws.out[n..2 * n]);
    }

    /// Evaluate the step tape at the state `ws.inputs` holds and factorize
    /// the stage matrix `C/(hγ) + G + gmin` there; `C` is kept as the mass
    /// matrix's values. `false` when singular.
    fn refactor_stage(&self, ws: &mut StageWorkspace<'_>, hg: f64) -> bool {
        let n = self.n;
        self.tape_tran_step
            .eval_main(&mut ws.step_tok, &ws.inputs, &mut ws.work, &mut ws.out);
        let (g, c) = ws.out[2 * n..].split_at(self.nnz_x);
        ws.mass.vals.copy_from_slice(c);
        ws.jmag.clear();
        ws.jmag.extend(g.iter().chain(c).map(|v| v.abs()));
        ws.q.copy_from_slice(&ws.out[n..2 * n]);
        let ok = self.factorize_stage(&mut ws.fac, g, c, 1.0 / hg, GMIN_DC, &mut ws.valbuf);
        ws.fac_fresh = ok;
        ok
    }

    /// Per row, `|G| |x| + |C| |x| / hγ` over the latest factorization's
    /// Jacobians: the size of the terms the row's residual sums.
    fn row_magnitudes(&self, ws: &mut StageWorkspace<'_>, x: &[f64], hg: f64) {
        ws.rowmag.fill(0.0);
        let (g, c) = ws.jmag.split_at(self.nnz_x.min(ws.jmag.len()));
        for (e, &v) in g.iter().enumerate() {
            ws.rowmag[self.jx_rows[e]] += v * x[self.jx_cols[e]].abs();
        }
        for (e, &v) in c.iter().enumerate() {
            ws.rowmag[self.jxd_rows[e]] += v * x[self.jxd_cols[e]].abs() / hg;
        }
    }

    /// Solve one implicit stage `r(X) = Q(X) − Q(xₙ) − ψ + h·γ·Ĩ(X) = 0`
    /// (`Q(xₙ)` is `ws.qn`, `ψ` is `ws.psi`) by the limiting *modified*
    /// Newton: reuse the frozen factorization (evaluating only `I` and `Q`
    /// via the cheap residual tape), refactorizing -- and re-evaluating the
    /// Jacobians -- only when a stale one converges too slowly. `x` is the
    /// guess on entry and the stage value `X` on return. Converged: `true`,
    /// with `f = −Ĩ(X, tᵢ)` in `ws.f` and `Q(X)` in `ws.q`.
    pub(crate) fn stage_newton(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &mut [f64],
        ti: f64,
        h: f64,
        gamma: f64,
    ) -> bool {
        let n = self.n;
        let hg = h * gamma;
        let mut prev_wn = f64::INFINITY;
        let crit = self.criterion(&ws.tolerances());
        let tol = (10.0 * f64::EPSILON / ws.rtol).max(IRK_STAGE_TOL_MAX.min(ws.rtol.sqrt()));

        for it in 0..IRK_STAGE_MAX_ITER {
            ws.stats.iters += 1;
            // Building/refreshing the factorization needs the Jacobians (the
            // step tape); a reused factorization needs only `I` and `Q`.
            let refresh = !ws.fac_fresh;
            self.patch_inputs(x, ti, &mut ws.inputs);
            if refresh {
                ws.stats.refacs += 1;
                if !self.refactor_stage(ws, hg) {
                    return false;
                }
            } else {
                self.tape_tran_res.eval_main(
                    &mut ws.res_tok,
                    &ws.inputs,
                    &mut ws.res_work,
                    &mut ws.out,
                );
            }
            // r = Q(x) - Q(xn) - psi + hγ·Ĩ,  Ĩ = I + gmin·x  (I = out[..n],
            // Q = out[n..2n]), solved for the Newton step against r / hγ;
            // within the rounding of what it is computed from in every row,
            // it is solved.
            self.row_magnitudes(ws, x, hg);
            let mut rounded = true;
            for k in 0..n {
                let (i, q) = (ws.out[k] + GMIN_DC * x[k], ws.out[n + k]);
                let r = (q - ws.qn[k] - ws.psi[k]) / hg + i;
                ws.rhs[k] = r;
                let terms = (q.abs() + ws.qn[k].abs() + ws.psi[k].abs()) / hg + ws.rowmag[k];
                rounded &= r.abs() <= IRK_STAGE_ROUNDOFF * f64::EPSILON * terms;
            }
            if rounded {
                self.residual_slope(ws, x, ti);
                return true;
            }
            if !ws.fac.solve_into(&ws.rhs, &mut ws.step) {
                return false;
            }
            let step = &mut ws.step;

            // Scaled update norm (convergence + stall detection).
            let (wn, worst) = crit.update_norm(step, x);
            if ws.trace {
                eprintln!(
                    "tran:     stage t={ti:.9e} it={it} wn={wn:.3e} worst=x[{worst}] delta={:.3e} x={:.6e}{}",
                    step[worst],
                    x[worst],
                    if refresh { " (refactored)" } else { "" }
                );
            }

            // No step limiting inside a time step: the globalization here is
            // the step size. A stage Newton that diverges rejects the step and
            // `h` shrinks, which shortens the move without bending the Newton
            // direction. Device limiting still applies, on the device's own
            // scale.
            if !self.limits.is_empty() {
                let x_new: Vec<f64> = if ws.trace {
                    (0..n).map(|k| x[k] - step[k]).collect()
                } else {
                    Vec::new()
                };
                newton::limit_step(&self.limits, x, &mut step[..n]);
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
                                v(x),
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

            if wn < tol {
                self.residual_slope(ws, x, ti);
                return true;
            }
            // A reused (stale) Jacobian that is not contracting fast enough: mark it
            // stale so the next iterate re-evaluates G and refactorizes here.
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
                if theta > IRK_STALL_THETA || predicted > tol {
                    ws.fac_fresh = false;
                }
            }
            prev_wn = wn;
        }
        false
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
        ws.fill_hist(t);
        self.patch_inputs(x, t, &mut ws.inputs);
        self.refactor_stage(ws, h * gamma)
    }

    /// Consistent restart after a discontinuity at `(t, x)`: one implicit-Euler
    /// step of the vanishing size `delta` into the region beyond it. The
    /// algebraic part of the state jumps there (a switch closes, a source kinks)
    /// while the differential states are pinned by `C/delta`; the returned
    /// state is the right-limit state the next step's explicit first stage
    /// needs -- evaluated at the landing itself it would carry the old
    /// region's slope and degrade the embedded estimate to first order.
    /// `ws.qn` is the charge at `x`. The state goes to `out`; `false` when
    /// the Newton fails (the caller then restarts unreinitialised).
    pub(crate) fn reinit_step(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &[f64],
        t: f64,
        delta: f64,
        out: &mut [f64],
    ) -> bool {
        ws.psi.fill(0.0);
        ws.fac_fresh = false;
        out.copy_from_slice(x);
        let ok = self.stage_newton(ws, out, t + delta, delta, 1.0);
        // the factorization now belongs to `delta`, not to any step size
        ws.fac_fresh = false;
        ok
    }
}
