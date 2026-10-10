//! Noise analysis on [`Model`]: the output-referred spectrum and its exact
//! all-parameter gradient.

use std::collections::HashMap;
use std::f64::consts::PI;

use num_complex::Complex64;
use rsdag::{differentiate, ExprId, Node, SymbolId};
use sane_core::log;

use ndarray::{Array2, Array3, ArrayView1};

use crate::model::{Model, ModelError, Point};
use crate::{noise_on_dae, solve_complex};

/// Output-referred noise (see [`Point::noise`]): `psd[[i, k]]` is the power
/// spectral density of `outputs[i]` at `freqs[k]` (V^2/Hz for a voltage,
/// A^2/Hz for a current), linearized at the operating point.
pub struct NoiseSpectrum {
    at: Point,
    x: Vec<f64>,
    outs: Vec<usize>,
    pub freqs: Vec<f64>,
    pub outputs: Vec<String>,
    pub psd: Array2<f64>,
}

/// Sensitivities of noise spectra: `grad[[i, k, j]]` is
/// `d psd[[i, k]] / d params[j]`.
#[derive(Clone, Debug)]
pub struct NoiseSensitivity {
    pub freqs: Vec<f64>,
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    pub param_values: Vec<f64>,
    pub psd: Array2<f64>,
    pub grad: Array3<f64>,
}

impl Point {
    /// The noise of `outputs` (nodes, unknowns, branch currents) at `freqs`
    /// (Hz, each positive): every noise source of the circuit (resistor
    /// thermal noise, the devices' white, flicker and tabular sources)
    /// carried to each output, linearized at the operating point.
    pub fn noise(&self, outputs: &[&str], freqs: &[f64]) -> Result<NoiseSpectrum, ModelError> {
        if let Some((there, _)) = &self.other {
            return there.noise(outputs, freqs);
        }
        let model = &self.model;
        let x = self.operating_point()?.x;
        // a transport delay couples its signal at e^{-jw tau}, as in AC
        model.ensure_hist_jac_ready();
        let (dr, dc, dv, dtau) = model.delay_ac_entries(&x, &self.p);
        let outs = model.inner.outputs(outputs)?;
        let rows = {
            let arc = model.context_arc();
            let mut c = arc.lock().unwrap();
            noise_on_dae(
                &mut c,
                model.dae(),
                model.cdc(),
                &outs,
                &x,
                &self.p,
                freqs,
                (&dr, &dc, &dv, &dtau),
            )
            .map_err(ModelError::Numeric)?
        };
        let bad: Vec<bool> = rows.iter().map(|r| r.iter().any(|v| v.is_nan())).collect();
        super::warn_singular("noise", freqs, &bad);
        Ok(NoiseSpectrum {
            at: self.clone(),
            x,
            psd: Array2::from_shape_fn((outs.len(), freqs.len()), |(i, k)| rows[k][i]),
            outs,
            freqs: freqs.to_vec(),
            outputs: outputs.iter().map(|o| o.to_string()).collect(),
        })
    }
}

impl NoiseSpectrum {
    /// The power spectral density of `output` at every frequency.
    pub fn of(&self, output: &str) -> Option<ArrayView1<'_, f64>> {
        let i = self.outputs.iter().position(|o| o == output)?;
        Some(self.psd.row(i))
    }

    /// The spectral density of `output`, the square root of its PSD (V or
    /// A per square-root hertz).
    pub fn density(&self, output: &str) -> Option<Vec<f64>> {
        Some(self.of(output)?.iter().map(|v| v.sqrt()).collect())
    }

    /// The RMS noise of `output` over the band of `freqs`: the square root
    /// of the PSD integrated by the trapezoidal rule.
    pub fn rms(&self, output: &str) -> Option<f64> {
        let psd = self.of(output)?;
        let area: f64 = (1..self.freqs.len())
            .map(|k| 0.5 * (psd[k - 1] + psd[k]) * (self.freqs[k] - self.freqs[k - 1]))
            .sum();
        Some(area.sqrt())
    }

    /// The derivatives of every PSD by the parameters under `wrt` (all for
    /// none) at every frequency, exact (the operating point's shift
    /// included). White and flicker sources only.
    pub fn sensitivity(&self, wrt: &[&str]) -> Result<NoiseSensitivity, ModelError> {
        let model = &self.at.model;
        model.ensure_no_delays("noise sensitivity")?;
        let cols = model.inner.columns(wrt)?;
        let names: Vec<String> = cols
            .iter()
            .map(|&c| model.inner.store.pnames[c].clone())
            .collect();
        let mut grad = Array3::zeros((self.outs.len(), self.freqs.len(), names.len()));
        for (i, &o) in self.outs.iter().enumerate() {
            for (k, &f) in self.freqs.iter().enumerate() {
                let (_, g) = model.noise_gradient(o, self.x.clone(), self.at.p.clone(), f)?;
                for (j, name) in names.iter().enumerate() {
                    grad[[i, k, j]] = g.iter().find(|(n, _)| n == name).map_or(0.0, |&(_, d)| d);
                }
            }
        }
        Ok(NoiseSensitivity {
            freqs: self.freqs.clone(),
            outputs: self.outputs.clone(),
            param_values: cols.iter().map(|&c| self.at.p[c]).collect(),
            params: names,
            psd: self.psd.clone(),
            grad,
        })
    }
}

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
    pub(crate) fn noise_gradient(
        &self,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freq: f64,
    ) -> Result<(f64, Vec<(String, f64)>), ModelError> {
        self.inner.bound(&p)?;
        let _g = log::scope("sens/noise_gradient");
        let n = self.dae().dim();
        let w = 2.0 * PI * freq;
        let g = self.cdc().system_matrix_dc(&x, &p, 0.0);
        let cm = self.cdc().jacobian_q_x(&x, &p, 0.0);
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
        let ((prr, prc, prv), _) = self.cdc().jacobian_p_sparse(&x, &p, 0.0);

        let arc = self.context_arc();
        let mut cg = arc.lock().unwrap();
        let c = &mut *cg;
        let ((gr, gc, ge), (cr, cc, ce)) = self.dae().jacobian_iq_coo(c);
        // Operating-point environment.
        let mut env: HashMap<SymbolId, f64> = HashMap::new();
        for (i, &s) in self.dae().x.iter().enumerate() {
            env.insert(s, x.get(i).copied().unwrap_or(0.0));
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

        // Per source: injection u_q (as node indices), numeric S_q = fac*sp, the
        // transimpedance T_q = lambda.u_q, and (white/flicker) the PSD expr.
        let mut nval = 0.0;
        let mut r = vec![Complex64::new(0.0, 0.0); n]; // r = sum_q 2 conj(T_q) S_q u_q
        let mut psd_terms: Vec<(f64, ExprId)> = Vec::new(); // (|T_q|^2 fac, psd_expr)
        let at = self.cdc().noise_at(c, self.dae(), &x, &p);
        let noise = self.dae().observers.noise(c);
        let injection = self.dae().noise_injection(c);
        // the injections' own dependence: sum over sources and rows of
        // Re(2 S_q conj(T_q) lambda_i) times d current_i / d generator_q
        let mut inj_terms: Vec<(f64, ExprId)> = Vec::new();
        for ((ns, src), inj) in noise.iter().zip(at).zip(injection.iter()) {
            // Tabular sources: not yet in the gradient.
            let sane_dae::NoiseLevel::Spectral { psd: sp, fexp } = src.level else {
                continue;
            };
            if !sp.is_finite() || sp <= 0.0 || !fexp.is_finite() {
                continue;
            }
            let fac = if fexp == 0.0 { 1.0 } else { freq.powf(-fexp) };
            let sq = sp * fac;
            // the injection where the generator enters, at the point
            let idxs = &src.injection;
            let mut tq = Complex64::new(0.0, 0.0);
            for &(i, u) in idxs {
                tq += lam[i] * u;
            }
            nval += sq * tq.norm_sqr();
            let wgt = 2.0 * sq * tq.conj();
            for &(i, u) in idxs {
                r[i] += wgt * u;
            }
            for &(i, e) in inj.iter() {
                inj_terms.push(((wgt * lam[i]).re, e));
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
        for (wgt, e) in psd_terms.iter().chain(&inj_terms) {
            let coef = c.konst_f64(*wgt);
            let t = c.mul(coef, *e);
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
}
