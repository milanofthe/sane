//! Noise analysis on [`Model`]: the output-referred spectrum and its exact
//! all-parameter gradient.

use std::collections::HashMap;
use std::f64::consts::PI;

use num_complex::Complex64;
use rsdag::{differentiate, ExprId, Node, SymbolId};
use sane_core::log;

use crate::model::{Model, ModelError};
use crate::{noise_on_dae, solve_complex};

impl Model {
    /// Exact analytic output-noise sensitivity `dN/dp` at one frequency w.r.t.
    /// **every** parameter (op-point shift included). The output noise PSD is
    /// `N = sum_q S_q |T_q|^2` with transimpedance `T_q = lambda . u_q`,
    /// `lambda = A^{-T} e_out`, injection `u_q` (+/-1 at the source's nodes), and
    /// PSD `S_q`. With `r = sum_q 2 conj(T_q) S_q u_q` and `xi = A^{-1} r`, `dN/dp`
    /// is the total derivative of the real functional
    /// `Phi = Re(-lambda^T A xi) + sum_q |T_q|^2 fac_q psd_q`: explicit `dPhi/dp`
    /// plus one DC adjoint for the bias shift (same trick as the AC gradient; no
    /// finite differences). White/flicker sources only. Returns `(N, [(name, dN/dp)])`.
    pub fn noise_gradient(
        &self,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freq: f64,
    ) -> Result<(f64, Vec<(String, f64)>), ModelError> {
        let _g = log::scope("sens/noise_gradient");
        let n = self.dae().dim();
        let z = vec![0.0; n];
        let w = 2.0 * PI * freq;
        let g = self.cdc().system_matrix_dc(&x, &z, &p, 0.0);
        let cm = self.cdc().jacobian_xdot(&x, &z, &p, 0.0);
        let a: Vec<Vec<Complex64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| Complex64::new(g[i][j], w * cm[i][j]))
                    .collect()
            })
            .collect();
        let mut at = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            for j in 0..n {
                at[j][i] = a[i][j];
            }
        }
        let mut e_out = vec![Complex64::new(0.0, 0.0); n];
        e_out[out_idx] = Complex64::new(1.0, 0.0);
        let lam = solve_complex(at, e_out)
            .ok_or_else(|| ModelError::Numeric("noise_gradient: singular A^T".to_string()))?;

        self.cdc()
            .ensure_param_jac(&mut self.context_arc().lock().unwrap(), self.dae());
        let pnames = self.cdc().param_names(&self.context_arc().lock().unwrap());
        let (prr, prc, prv) = self.cdc().jacobian_p_sparse(&x, &z, &p, 0.0);

        let arc = self.context_arc();
        let mut cg = arc.lock().unwrap();
        let c = &mut *cg;
        let (gr, gc, ge) = self.dae().jacobian_x_coo(c);
        let (cr, cc, ce) = self.dae().jacobian_xdot_coo(c);
        // Operating-point environment.
        let mut env: HashMap<SymbolId, f64> = HashMap::new();
        for (i, &s) in self.dae().x.iter().enumerate() {
            env.insert(s, x.get(i).copied().unwrap_or(0.0));
        }
        for s in self.dae().xdot.iter().flatten() {
            env.insert(*s, 0.0);
        }
        let mut psyms = Vec::with_capacity(pnames.len());
        for (j, name) in pnames.iter().enumerate() {
            let e = c.sym(name);
            if let Node::Symbol(s) = c.node(e) {
                env.insert(*s, p.get(j).copied().unwrap_or(0.0));
                psyms.push(*s);
            } else {
                psyms.push(self.dae().t);
            }
        }
        env.insert(self.dae().t, 0.0);
        let idx_of = |sid: SymbolId| self.dae().x.iter().position(|&s| s == sid);

        // Per source: injection u_q (as node indices), numeric S_q = fac*sp, the
        // transimpedance T_q = lambda.u_q, and (white/flicker) the PSD expr.
        let mut nval = 0.0;
        let mut r = vec![Complex64::new(0.0, 0.0); n]; // r = sum_q 2 conj(T_q) S_q u_q
        let mut psd_terms: Vec<(f64, ExprId)> = Vec::new(); // (|T_q|^2 fac, psd_expr)
        for ns in &self.dae().noise_sources {
            if !ns.table.is_empty() {
                continue; // tabular sources: not yet in the gradient
            }
            let v = rsdag::eval(c, &[ns.psd, ns.flicker_exp], &env);
            let (sp, fexp) = (v[0], v[1]);
            if !sp.is_finite() || sp <= 0.0 || !fexp.is_finite() {
                continue;
            }
            let fac = if fexp == 0.0 { 1.0 } else { freq.powf(-fexp) };
            let sq = sp * fac;
            // injection: +1 at hi index, -1 at lo index
            let mut tq = Complex64::new(0.0, 0.0);
            let mut idxs: Vec<(usize, f64)> = Vec::new();
            if let Some(i) = ns.hi.and_then(idx_of) {
                idxs.push((i, 1.0));
            }
            if let Some(i) = ns.lo.and_then(idx_of) {
                idxs.push((i, -1.0));
            }
            for &(i, sgn) in &idxs {
                tq += lam[i] * sgn;
            }
            nval += sq * tq.norm_sqr();
            let wgt = 2.0 * sq * tq.conj();
            for &(i, sgn) in &idxs {
                r[i] += wgt * sgn;
            }
            psd_terms.push((tq.norm_sqr() * fac, ns.psd));
        }
        // xi = A^{-1} r.
        let xi = solve_complex(a, r)
            .ok_or_else(|| ModelError::Numeric("noise_gradient: singular A".to_string()))?;

        // Phi_re = Re(-lambda^T A xi) + sum_q (|T_q|^2 fac) psd_q.
        // For -lambda_i A_ij xi_j: c_ij = -lambda_i xi_j; Re(c A) on G = Re(c_ij),
        // on C = -w Im(c_ij).
        let mut phi = c.zero();
        for k in 0..ge.len() {
            let (i, j) = (gr[k], gc[k]);
            let cij = -(lam[i] * xi[j]);
            let coef = c.konst_f64(cij.re);
            let t = c.mul(coef, ge[k]);
            phi = c.add(phi, t);
        }
        for k in 0..ce.len() {
            let (i, j) = (cr[k], cc[k]);
            let cij = -(lam[i] * xi[j]);
            let coef = c.konst_f64(-w * cij.im);
            let t = c.mul(coef, ce[k]);
            phi = c.add(phi, t);
        }
        for (wgt, psd) in &psd_terms {
            let coef = c.konst_f64(*wgt);
            let t = c.mul(coef, *psd);
            phi = c.add(phi, t);
        }

        // dN/dp = dPhi/dp (explicit) + grad_x Phi . s_k (one real DC adjoint).
        let dpre: Vec<ExprId> = psyms.iter().map(|&ps| differentiate(c, phi, ps)).collect();
        let expl = rsdag::eval(c, &dpre, &env);
        let dxe: Vec<ExprId> = self
            .dae()
            .x
            .iter()
            .map(|&xm| differentiate(c, phi, xm))
            .collect();
        let u = rsdag::eval(c, &dxe, &env);
        drop(cg);
        // G_dc^T nu = u (real).
        let mut gt = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                gt[j][i] = g[i][j];
            }
        }
        let nu = crate::solve_real(gt, u).ok_or_else(|| {
            ModelError::Numeric("noise_gradient: singular DC Jacobian".to_string())
        })?;
        let mut grad = expl;
        for t in 0..prv.len() {
            let (i, k) = (prr[t], prc[t]);
            grad[k] -= nu[i] * prv[t];
        }
        Ok((nval, pnames.into_iter().zip(grad).collect()))
    }

    /// Output-referred noise PSD from the unknown at `out_idx`, at `(x, p)`.
    pub fn noise_raw(
        &self,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        fstart: f64,
        fstop: f64,
        points: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
        self.ensure_no_delays("noise")?;
        let arc = self.context_arc();
        let mut c = arc.lock().unwrap();
        let temp_k = self.temp_k();
        noise_on_dae(
            &mut c,
            self.dae(),
            self.cdc(),
            out_idx,
            &x,
            &p,
            fstart,
            fstop,
            points,
            temp_k,
        )
        .map_err(ModelError::Numeric)
    }
}
