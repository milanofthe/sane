//! Pole/zero analyses on [`Model`]: pencil roots, their exact parameter
//! gradients (eigenvalue perturbation over the symbolic stamps) and
//! per-root sensitivities.

use std::collections::HashMap;

use num_complex::Complex64;
use rsdag::{differentiate, ExprId, Node, SymbolId};
use sane_core::log;

use crate::model::{Model, ModelError};
use crate::{finite_pencil_roots, pencil_eigvectors, solve_complex};

impl Model {
    /// Small-signal poles at the operating point `x` (parameters `p`): the finite
    /// generalized eigenvalues of the pencil `(G, C)` with `G = dF/dx`,
    /// `C = dF/dx'`, computed natively (the engine's standard-reduction
    /// eigensolver -- no Python-side linear algebra, so no drift). `(re, im)` in
    /// rad/s.
    pub fn poles(&self, x: Vec<f64>, p: Vec<f64>) -> Result<Vec<(f64, f64)>, ModelError> {
        let n = self.dae().dim();
        let z = vec![0.0; n];
        let g = self.cdc().system_matrix_dc(&x, &z, &p, 0.0);
        let c = self.cdc().jacobian_xdot(&x, &z, &p, 0.0);
        finite_pencil_roots(&g, &c)
            .map(|v| v.into_iter().map(|r| (r[0], r[1])).collect())
            .map_err(ModelError::Numeric)
    }

    /// Transmission zeros from source `input` to the unknown at `out_idx`, at the
    /// operating point `x`: the finite generalized eigenvalues of the Rosenbrock
    /// system-matrix pencil, computed natively (same eigensolver as `poles`).
    pub fn zeros(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> Result<Vec<(f64, f64)>, ModelError> {
        let n = self.dae().dim();
        let z = vec![0.0; n];
        let g = self.cdc().system_matrix_dc(&x, &z, &p, 0.0);
        let c = self.cdc().jacobian_xdot(&x, &z, &p, 0.0);
        let b = self.input_jacobian(input, x.clone(), z.clone(), p.clone(), 0.0)?;
        // M = [[G, b], [e_out^T, 0]],  N = [[C, 0], [0, 0]].
        let mut m = vec![vec![0.0; n + 1]; n + 1];
        let mut nn = vec![vec![0.0; n + 1]; n + 1];
        for i in 0..n {
            for j in 0..n {
                m[i][j] = g[i][j];
                nn[i][j] = c[i][j];
            }
            m[i][n] = b[i];
            m[n][i] = if i == out_idx { 1.0 } else { 0.0 };
        }
        finite_pencil_roots(&m, &nn)
            .map(|v| v.into_iter().map(|r| (r[0], r[1])).collect())
            .map_err(ModelError::Numeric)
    }

    /// Exact analytic pole sensitivity `ds/dp` for every finite pole w.r.t.
    /// **every** parameter at once, including the operating-point shift. For a
    /// simple pole `s=-1/mu` (eigenvalue `mu` of `A=G^{-1}C`, right/left
    /// eigenvectors `v`/`w`), the perturbation `d(mu)/dp_k = w_hat^T(dC/dp_k -
    /// mu dG/dp_k)v/(w^T v)` (`w_hat = G^{-T} w`) collapses over all `k` via the
    /// scalar functional `Phi = w_hat^T(C - mu G)v`: explicit `dPhi/dp` plus one
    /// DC adjoint for the shift (the same all-parameter trick as the AC gradient),
    /// then `ds = d(mu)/mu^2`. Returns one `(s, [(name, ds_re, ds_im)])` per pole.
    #[allow(clippy::type_complexity)]
    pub fn pole_gradient(
        &self,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> Result<Vec<((f64, f64), Vec<(String, f64, f64)>)>, ModelError> {
        let _g = log::scope("sens/pole_gradient");
        let n = self.dae().dim();
        let z = vec![0.0; n];
        let g = self.cdc().system_matrix_dc(&x, &z, &p, 0.0);
        let cmat = self.cdc().jacobian_xdot(&x, &z, &p, 0.0);
        let eigs = pencil_eigvectors(&g, &cmat).map_err(ModelError::Numeric)?;
        self.cdc()
            .ensure_param_jac(&mut self.context_arc().lock().unwrap(), self.dae());
        let pnames = self.cdc().param_names(&self.context_arc().lock().unwrap());
        let (prr, prc, prv) = self.cdc().jacobian_p_sparse(&x, &z, &p, 0.0);
        // G^T as a complex matrix for the w_hat and DC-adjoint solves.
        let gt: Vec<Vec<Complex64>> = (0..n)
            .map(|i| (0..n).map(|j| Complex64::new(g[j][i], 0.0)).collect())
            .collect();

        let arc = self.context_arc();
        let mut cg = arc.lock().unwrap();
        let c = &mut *cg;
        let (gr, gc, ge) = self.dae().jacobian_x_coo(c);
        let (cr, cc, ce) = self.dae().jacobian_xdot_coo(c);
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

        let mut out = Vec::new();
        for (s_arr, mu_arr, v_arr, w_arr) in &eigs {
            let mu = Complex64::new(mu_arr[0], mu_arr[1]);
            let v: Vec<Complex64> = v_arr.iter().map(|a| Complex64::new(a[0], a[1])).collect();
            let wv: Vec<Complex64> = w_arr.iter().map(|a| Complex64::new(a[0], a[1])).collect();
            let mut den = Complex64::new(0.0, 0.0);
            for k in 0..n {
                den += wv[k] * v[k];
            }
            if den.norm() < 1e-300 {
                continue;
            }
            let what = match solve_complex(gt.clone(), wv.clone()) {
                Some(x) => x,
                None => continue,
            };
            // Phi = sum_ij (what_i v_j)(C_ij - mu G_ij).
            let mut phi_re = c.zero();
            let mut phi_im = c.zero();
            for k in 0..ge.len() {
                let (i, j) = (gr[k], gc[k]);
                let coef = -mu * (what[i] * v[j]); // -mu * what_i v_j  (on G)
                let cre = c.konst_f64(coef.re);
                let tr = c.mul(cre, ge[k]);
                phi_re = c.add(phi_re, tr);
                let cim = c.konst_f64(coef.im);
                let ti = c.mul(cim, ge[k]);
                phi_im = c.add(phi_im, ti);
            }
            for k in 0..ce.len() {
                let (i, j) = (cr[k], cc[k]);
                let coef = what[i] * v[j]; // what_i v_j  (on C)
                let cre = c.konst_f64(coef.re);
                let tr = c.mul(cre, ce[k]);
                phi_re = c.add(phi_re, tr);
                let cim = c.konst_f64(coef.im);
                let ti = c.mul(cim, ce[k]);
                phi_im = c.add(phi_im, ti);
            }
            let dpre: Vec<ExprId> = psyms
                .iter()
                .map(|&ps| differentiate(c, phi_re, ps))
                .collect();
            let dpim: Vec<ExprId> = psyms
                .iter()
                .map(|&ps| differentiate(c, phi_im, ps))
                .collect();
            let er = rsdag::eval(c, &dpre, &env);
            let ei = rsdag::eval(c, &dpim, &env);
            let dxre: Vec<ExprId> = self
                .dae()
                .x
                .iter()
                .map(|&xm| differentiate(c, phi_re, xm))
                .collect();
            let dxim: Vec<ExprId> = self
                .dae()
                .x
                .iter()
                .map(|&xm| differentiate(c, phi_im, xm))
                .collect();
            let ure = rsdag::eval(c, &dxre, &env);
            let uim = rsdag::eval(c, &dxim, &env);
            let urhs: Vec<Complex64> = (0..n).map(|m| Complex64::new(ure[m], uim[m])).collect();
            let nu = match solve_complex(gt.clone(), urhs) {
                Some(x) => x,
                None => continue,
            };
            let mut num_re = er;
            let mut num_im = ei;
            for t in 0..prv.len() {
                let (i, k) = (prr[t], prc[t]);
                num_re[k] -= nu[i].re * prv[t];
                num_im[k] -= nu[i].im * prv[t];
            }
            let mu2 = mu * mu;
            let items: Vec<(String, f64, f64)> = pnames
                .iter()
                .enumerate()
                .map(|(k, name)| {
                    let num = Complex64::new(num_re[k], num_im[k]);
                    let ds = (num / den) / mu2;
                    (name.clone(), ds.re, ds.im)
                })
                .collect();
            out.push(((s_arr[0], s_arr[1]), items));
        }
        Ok(out)
    }

    /// Exact analytic transmission-zero sensitivity `ds/dp` for every finite zero
    /// w.r.t. **every** parameter at once, including the operating-point shift.
    /// Same eigenvalue-perturbation + adjoint-shift machinery as
    /// [`pole_gradient`], but on the Rosenbrock pencil
    /// `M=[[G,b],[e_out^T,0]]`, `N=[[C,0],[0,0]]` (`b=dF/d(input)`); the functional
    /// is `Phi = w_hat^T(N - mu M)v` over the augmented system, whose only extra
    /// term is the `b`-column `-mu v_{n} sum_i w_hat_i b_i`. Returns one
    /// `(s, [(name, ds_re, ds_im)])` per zero.
    #[allow(clippy::type_complexity)]
    pub fn zero_gradient(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> Result<Vec<((f64, f64), Vec<(String, f64, f64)>)>, ModelError> {
        let _g = log::scope("sens/zero_gradient");
        let n = self.dae().dim();
        let z = vec![0.0; n];
        let g = self.cdc().system_matrix_dc(&x, &z, &p, 0.0);
        let cmat = self.cdc().jacobian_xdot(&x, &z, &p, 0.0);
        let bnum = self.input_jacobian(input, x.clone(), z.clone(), p.clone(), 0.0)?;
        // Augmented Rosenbrock pencil (dimension n+1).
        let na = n + 1;
        let mut m = vec![vec![0.0; na]; na];
        let mut nn = vec![vec![0.0; na]; na];
        for i in 0..n {
            for j in 0..n {
                m[i][j] = g[i][j];
                nn[i][j] = cmat[i][j];
            }
            m[i][n] = bnum[i];
            m[n][i] = if i == out_idx { 1.0 } else { 0.0 };
        }
        let eigs = pencil_eigvectors(&m, &nn).map_err(ModelError::Numeric)?;
        self.cdc()
            .ensure_param_jac(&mut self.context_arc().lock().unwrap(), self.dae());
        let pnames = self.cdc().param_names(&self.context_arc().lock().unwrap());
        let (prr, prc, prv) = self.cdc().jacobian_p_sparse(&x, &z, &p, 0.0);
        // M^T (augmented) as complex for the w_hat solve; G^T (n) for the shift.
        let mt: Vec<Vec<Complex64>> = (0..na)
            .map(|i| (0..na).map(|j| Complex64::new(m[j][i], 0.0)).collect())
            .collect();
        let gt: Vec<Vec<Complex64>> = (0..n)
            .map(|i| (0..n).map(|j| Complex64::new(g[j][i], 0.0)).collect())
            .collect();

        let arc = self.context_arc();
        let mut cg = arc.lock().unwrap();
        let c = &mut *cg;
        let (gr, gc, ge) = self.dae().jacobian_x_coo(c);
        let (cr, cc, ce) = self.dae().jacobian_xdot_coo(c);
        let isym = {
            let e = c.sym(input);
            match c.node(e) {
                Node::Symbol(s) => *s,
                _ => return Err(ModelError::Numeric(format!("'{input}' is not a symbol"))),
            }
        };
        let bsym: Vec<ExprId> = self
            .dae()
            .residuals
            .iter()
            .map(|&r| differentiate(c, r, isym))
            .collect();
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

        let mut out = Vec::new();
        for (s_arr, mu_arr, v_arr, w_arr) in &eigs {
            let mu = Complex64::new(mu_arr[0], mu_arr[1]);
            let v: Vec<Complex64> = v_arr.iter().map(|a| Complex64::new(a[0], a[1])).collect();
            let wv: Vec<Complex64> = w_arr.iter().map(|a| Complex64::new(a[0], a[1])).collect();
            let mut den = Complex64::new(0.0, 0.0);
            for k in 0..na {
                den += wv[k] * v[k];
            }
            if den.norm() < 1e-300 {
                continue;
            }
            let what = match solve_complex(mt.clone(), wv.clone()) {
                Some(x) => x,
                None => continue,
            };
            let vn = v[n]; // augmented component multiplying the b-column
            let mut phi_re = c.zero();
            let mut phi_im = c.zero();
            // N block (C): + what_i v_j C_ij
            for k in 0..ce.len() {
                let (i, j) = (cr[k], cc[k]);
                let coef = what[i] * v[j];
                let cre = c.konst_f64(coef.re);
                let tr = c.mul(cre, ce[k]);
                phi_re = c.add(phi_re, tr);
                let cim = c.konst_f64(coef.im);
                let ti = c.mul(cim, ce[k]);
                phi_im = c.add(phi_im, ti);
            }
            // M block (G): - mu what_i v_j G_ij
            for k in 0..ge.len() {
                let (i, j) = (gr[k], gc[k]);
                let coef = -mu * (what[i] * v[j]);
                let cre = c.konst_f64(coef.re);
                let tr = c.mul(cre, ge[k]);
                phi_re = c.add(phi_re, tr);
                let cim = c.konst_f64(coef.im);
                let ti = c.mul(cim, ge[k]);
                phi_im = c.add(phi_im, ti);
            }
            // M b-column: - mu v_n what_i b_i
            for i in 0..n {
                let coef = -mu * vn * what[i];
                let cre = c.konst_f64(coef.re);
                let tr = c.mul(cre, bsym[i]);
                phi_re = c.add(phi_re, tr);
                let cim = c.konst_f64(coef.im);
                let ti = c.mul(cim, bsym[i]);
                phi_im = c.add(phi_im, ti);
            }
            let dpre: Vec<ExprId> = psyms
                .iter()
                .map(|&ps| differentiate(c, phi_re, ps))
                .collect();
            let dpim: Vec<ExprId> = psyms
                .iter()
                .map(|&ps| differentiate(c, phi_im, ps))
                .collect();
            let er = rsdag::eval(c, &dpre, &env);
            let ei = rsdag::eval(c, &dpim, &env);
            let dxre: Vec<ExprId> = self
                .dae()
                .x
                .iter()
                .map(|&xm| differentiate(c, phi_re, xm))
                .collect();
            let dxim: Vec<ExprId> = self
                .dae()
                .x
                .iter()
                .map(|&xm| differentiate(c, phi_im, xm))
                .collect();
            let ure = rsdag::eval(c, &dxre, &env);
            let uim = rsdag::eval(c, &dxim, &env);
            let urhs: Vec<Complex64> = (0..n).map(|mm| Complex64::new(ure[mm], uim[mm])).collect();
            let nu = match solve_complex(gt.clone(), urhs) {
                Some(x) => x,
                None => continue,
            };
            let mut num_re = er;
            let mut num_im = ei;
            for t in 0..prv.len() {
                let (i, k) = (prr[t], prc[t]);
                num_re[k] -= nu[i].re * prv[t];
                num_im[k] -= nu[i].im * prv[t];
            }
            let mu2 = mu * mu;
            let items: Vec<(String, f64, f64)> = pnames
                .iter()
                .enumerate()
                .map(|(k, name)| {
                    let num = Complex64::new(num_re[k], num_im[k]);
                    let ds = (num / den) / mu2;
                    (name.clone(), ds.re, ds.im)
                })
                .collect();
            out.push(((s_arr[0], s_arr[1]), items));
        }
        Ok(out)
    }

    /// Exact pole sensitivity `(pole, dpole/dp)` for every finite pole w.r.t.
    /// `param`, including the operating-point shift, computed natively (the same
    /// pencil eigensolver as `poles`, so the poles are consistent). `(re, im)`.
    pub fn pole_sensitivity(
        &self,
        input: &str,
        param: &str,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> Result<Vec<((f64, f64), (f64, f64))>, ModelError> {
        let n = self.dae().dim();
        let z = vec![0.0; n];
        let g = self.cdc().system_matrix_dc(&x, &z, &p, 0.0);
        let c = self.cdc().jacobian_xdot(&x, &z, &p, 0.0);
        let (dg, dc, _db) = self.ac_derivatives(input, param, x.clone(), p.clone(), 0.0)?;
        crate::pencil_root_sensitivity(&g, &c, &dg, &dc)
            .map(|v| {
                v.into_iter()
                    .map(|(s, ds)| ((s[0], s[1]), (ds[0], ds[1])))
                    .collect()
            })
            .map_err(ModelError::Numeric)
    }

    /// Exact transmission-zero sensitivity `(zero, dzero/dp)` for every finite
    /// zero of the `input -> out_idx` transfer w.r.t. `param`, native (same
    /// Rosenbrock pencil and eigensolver as `zeros`). `(re, im)`.
    pub fn zero_sensitivity(
        &self,
        input: &str,
        out_idx: usize,
        param: &str,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> Result<Vec<((f64, f64), (f64, f64))>, ModelError> {
        let n = self.dae().dim();
        let z = vec![0.0; n];
        let g = self.cdc().system_matrix_dc(&x, &z, &p, 0.0);
        let c = self.cdc().jacobian_xdot(&x, &z, &p, 0.0);
        let bin = self.input_jacobian(input, x.clone(), z.clone(), p.clone(), 0.0)?; // dF/d(input)
        let (dg, dc, db) = self.ac_derivatives(input, param, x.clone(), p.clone(), 0.0)?;
        // Rosenbrock M = [[G, dF/din], [e_out, 0]], N = [[C, 0], [0, 0]] (the same
        // convention as `zeros`); its parameter derivatives. d(dF/din)/dp = -dB.
        let m = n + 1;
        let mut mm = vec![vec![0.0; m]; m];
        let mut nn = vec![vec![0.0; m]; m];
        let mut dmm = vec![vec![0.0; m]; m];
        let mut dnn = vec![vec![0.0; m]; m];
        for i in 0..n {
            for j in 0..n {
                mm[i][j] = g[i][j];
                nn[i][j] = c[i][j];
                dmm[i][j] = dg[i][j];
                dnn[i][j] = dc[i][j];
            }
            mm[i][n] = bin[i];
            dmm[i][n] = -db[i];
            mm[n][i] = if i == out_idx { 1.0 } else { 0.0 };
        }
        crate::pencil_root_sensitivity(&mm, &nn, &dmm, &dnn)
            .map(|v| {
                v.into_iter()
                    .map(|(s, ds)| ((s[0], s[1]), (ds[0], ds[1])))
                    .collect()
            })
            .map_err(ModelError::Numeric)
    }
}
