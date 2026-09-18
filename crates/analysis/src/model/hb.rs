//! Harmonic balance on [`Model`]: the periodic steady-state solve and the
//! exact coefficient gradient / Hessian against the Toeplitz HB Jacobian.

use std::f64::consts::PI;

use num_complex::Complex64;
use sane_core::constants::{DC_OP_MAXIT, DC_OP_TOL};
use sane_core::log;
use sane_solve::hb::{hb_samples, CompiledHb};

use crate::model::{Model, ModelError};

impl Model {
    pub fn solve_hb(
        &self,
        p: Vec<f64>,
        f0: f64,
        harmonics: usize,
        x0: Option<Vec<f64>>,
        oversample: usize,
        tol: f64,
        max_iter: usize,
        samples: Option<usize>,
        continuation: Option<bool>,
    ) -> Result<(Vec<Vec<(f64, f64)>>, bool, usize, f64, f64, f64, f64), ModelError> {
        self.ensure_no_delays("harmonic balance")?;
        // Infer the fundamental from a periodic source when the caller passes
        // f0 <= 0 (a SIN's frequency, a PULSE train's 1/period).
        let f0 = if f0 > 0.0 {
            f0
        } else {
            self.cdc().source_fundamental(&p).ok_or_else(|| {
                ModelError::Numeric(
                    "harmonic balance needs f0 > 0 (no periodic source to infer it from)"
                        .to_string(),
                )
            })?
        };
        let w0 = 2.0 * PI * f0;
        let m = match samples {
            Some(s) => s,
            None => {
                let arc = self.context_arc();
                let nl = self.dae().nonlinearity(&arc.lock().unwrap());
                hb_samples(&nl, harmonics, oversample)
            }
        };
        let mut task = sane_core::log::task(
            "HB",
            "hb",
            &format!(
                "(f0: {f0:.4e}, harmonics: {harmonics}, samples: {m}, dim: {})",
                self.dim()
            ),
        );
        let t_setup = sane_core::time::Instant::now();
        let hb = CompiledHb::new(self.cdc(), harmonics, m).ok_or_else(|| {
            ModelError::Numeric("harmonic balance setup failed (need samples >= 2*K)".to_string())
        })?;
        let setup_ms = t_setup.elapsed().as_secs_f64() * 1e3;
        let x_dc = match x0 {
            Some(v) if v.len() == self.dae().dim() => v,
            _ => {
                let (x, conv, iters) = self.cdc().solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
                if !conv {
                    return Err(ModelError::Numeric(format!(
                        "DC operating point did not converge in {iters} iterations"
                    )));
                }
                x
            }
        };
        let ramp: Vec<usize> = {
            let arc = self.context_arc();
            let pnames = self.cdc().param_names(&arc.lock().unwrap());
            pnames
                .iter()
                .enumerate()
                .filter(|(_, n)| n.ends_with(".sin_amp"))
                .map(|(i, _)| i)
                .collect()
        };
        let t_solve = sane_core::time::Instant::now();
        let res = match continuation {
            Some(true) if !ramp.is_empty() => {
                hb.solve_continuation(&p, &x_dc, w0, &ramp, tol, max_iter)
            }
            Some(false) => hb.solve(&p, &x_dc, w0, tol, max_iter),
            _ => {
                let direct = hb.solve(&p, &x_dc, w0, tol, max_iter);
                if !direct.converged && !ramp.is_empty() {
                    hb.solve_continuation(&p, &x_dc, w0, &ramp, tol, max_iter)
                } else {
                    direct
                }
            }
        };
        let solve_ms = t_solve.elapsed().as_secs_f64() * 1e3;
        let spectra = res
            .spectra
            .iter()
            .map(|row| row.iter().map(|c| (c.re, c.im)).collect())
            .collect();
        task.finish(format!(
            "converged: {}, iters: {}, residual: {:.2e}",
            res.converged, res.iters, res.residual_norm
        ));
        // Return the actual fundamental used (resolved from a source when the
        // caller passed f0 <= 0) so the caller can label the spectra correctly.
        Ok((
            spectra,
            res.converged,
            res.iters,
            res.residual_norm,
            setup_ms,
            solve_ms,
            f0,
        ))
    }

    /// All-parameter sensitivity of the harmonic-balance steady-state
    /// coefficients at unknown `out_idx`, evaluated at the converged `spectra`.
    /// Returns, per harmonic `k = 0..=K`, the complex gradient
    /// `dX_{out,k}/dp_j` over every parameter as `(name, re, im)` rows.
    ///
    /// Exact autodiff: the implicit-function adjoint on the two-sided
    /// harmonic-balance Jacobian, with `dF/dp` routed through the AFT (see
    /// [`CompiledHb::coeff_gradient`]). `f0`, `harmonics`, `oversample`,
    /// `samples` must match the solve that produced `spectra` so the AFT grid
    /// is reconstructed identically.
    #[allow(clippy::too_many_arguments)]
    pub fn hb_gradient(
        &self,
        out_idx: usize,
        spectra: Vec<Vec<(f64, f64)>>,
        p: Vec<f64>,
        f0: f64,
        harmonics: usize,
        oversample: usize,
        samples: Option<usize>,
    ) -> Result<Vec<Vec<(String, f64, f64)>>, ModelError> {
        self.ensure_no_delays("hb_gradient")?;
        let mut task = log::task("SENS-HB", "sens_hb", &format!("(harmonics: {harmonics})"));
        if !(f0 > 0.0) {
            return Err(ModelError::Numeric("hb_gradient needs f0 > 0".to_string()));
        }
        let w0 = 2.0 * PI * f0;
        let m = match samples {
            Some(s) => s,
            None => {
                let nl = self.dae().nonlinearity(&self.context_arc().lock().unwrap());
                hb_samples(&nl, harmonics, oversample)
            }
        };
        // The parameter Jacobian dF/dp must be available for the AFT path.
        self.cdc()
            .ensure_param_jac(&mut self.context_arc().lock().unwrap(), self.dae());
        let pnames = self.cdc().param_names(&self.context_arc().lock().unwrap());
        let hb = CompiledHb::new(self.cdc(), harmonics, m).ok_or_else(|| {
            ModelError::Numeric("hb_gradient setup failed (need samples >= 2*K)".to_string())
        })?;
        let spec: Vec<Vec<Complex64>> = spectra
            .iter()
            .map(|row| row.iter().map(|&(re, im)| Complex64::new(re, im)).collect())
            .collect();
        let grad = hb
            .coeff_gradient(&spec, &p, w0, out_idx, pnames.len())
            .ok_or_else(|| {
                ModelError::Numeric(
                    "hb_gradient: AFT grid too coarse for the Jacobian (need m/2 >= 2K) \
                     or singular harmonic-balance Jacobian"
                        .to_string(),
                )
            })?;
        task.finish(format!("params: {}", pnames.len()));
        Ok(grad
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .zip(&pnames)
                    .map(|(c, name)| (name.clone(), c.re, c.im))
                    .collect()
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hb_hessian(
        &self,
        out_idx: usize,
        k_metric: usize,
        subset: Vec<String>,
        spectra: Vec<Vec<(f64, f64)>>,
        p: Vec<f64>,
        f0: f64,
        harmonics: usize,
        oversample: usize,
        samples: Option<usize>,
    ) -> Result<Vec<Vec<(f64, f64)>>, ModelError> {
        let _g = log::scope("sens/hb_hessian");
        if !(f0 > 0.0) {
            return Err(ModelError::Numeric("hb_hessian needs f0 > 0".to_string()));
        }
        let w0 = 2.0 * PI * f0;
        let m = match samples {
            Some(s) => s,
            None => {
                let nl = self.dae().nonlinearity(&self.context_arc().lock().unwrap());
                hb_samples(&nl, harmonics, oversample)
            }
        };
        // Map parameter names to columns; build the Lagrangian-Hessian tape (and
        // the parameter Jacobian it needs) on first use.
        let pnames = self.cdc().param_names(&self.context_arc().lock().unwrap());
        let mut cols = Vec::with_capacity(subset.len());
        for name in &subset {
            let col = pnames
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| ModelError::Numeric(format!("unknown parameter '{name}'")))?;
            cols.push(col);
        }
        self.cdc()
            .ensure_hessian(&mut self.context_arc().lock().unwrap(), self.dae());
        let hb = CompiledHb::new(self.cdc(), harmonics, m).ok_or_else(|| {
            ModelError::Numeric("hb_hessian setup failed (need samples >= 2*K)".to_string())
        })?;
        let spec: Vec<Vec<Complex64>> = spectra
            .iter()
            .map(|row| row.iter().map(|&(re, im)| Complex64::new(re, im)).collect())
            .collect();
        let h = hb
            .coeff_hessian(&spec, &p, w0, out_idx, k_metric, &cols)
            .ok_or_else(|| {
                ModelError::Numeric(
                    "hb_hessian: AFT grid too coarse for the Jacobian (need m/2 >= 2K) \
                     or singular harmonic-balance Jacobian"
                        .to_string(),
                )
            })?;
        Ok(h.into_iter()
            .map(|row| row.into_iter().map(|c| (c.re, c.im)).collect())
            .collect())
    }
}
