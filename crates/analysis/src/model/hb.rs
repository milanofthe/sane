//! Harmonic balance at a [`Point`]: the periodic steady state under the
//! circuit's periodic drive, and the exact derivatives of its spectra (the
//! implicit-function adjoint on the two-sided HB Jacobian, the Hessian over
//! the AFT).

use std::f64::consts::PI;

use ndarray::{Array2, Array3, ArrayView1, Axis};
use num_complex::Complex64;

use sane_core::{log, log_stage};
use sane_solve::hb::{hb_samples, CompiledHb, HbResult};

use crate::model::{Model, ModelError, Point};

/// The tunables of [`Point::harmonic_balance`].
#[derive(Clone, Debug)]
pub struct HbOptions {
    /// The fundamental (Hz); `None` takes the circuit's own periodic source
    /// (a SIN's frequency, a PULSE train's 1/period).
    pub f0: Option<f64>,
    /// Harmonics solved for, beyond DC.
    pub harmonics: usize,
    /// AFT oversampling over the alias-free minimum.
    pub oversample: usize,
    /// Time samples per period, instead of the oversampled alias-free count.
    pub samples: Option<usize>,
    /// Newton tolerance on the infinity norm of the harmonic residual.
    pub tol: f64,
    pub max_iter: usize,
    /// Source-stepping continuation (ramping the SIN amplitudes from zero):
    /// `None` falls back to it where the direct Newton fails, `Some(true)`
    /// always takes it, `Some(false)` never.
    pub continuation: Option<bool>,
}

impl Default for HbOptions {
    fn default() -> Self {
        HbOptions {
            f0: None,
            harmonics: 8,
            oversample: 16,
            samples: None,
            tol: 1e-10,
            max_iter: 60,
            continuation: None,
        }
    }
}

/// The periodic steady state (see [`Point::harmonic_balance`]): per unknown,
/// the complex Fourier coefficients `X_k`, `k = 0..=harmonics`, of the
/// two-sided series `x(t) = sum_{|k| <= K} X_k e^{j k w0 t}` (`X_{-k} =
/// conj X_k`), so `x(t) = X_0 + 2 sum_{k >= 1} |X_k| cos(k w0 t + arg X_k)`.
pub struct HarmonicBalance {
    at: Point,
    /// The fundamental solved at (Hz), the inferred one where none was given.
    pub f0: f64,
    pub harmonics: usize,
    /// Time samples per period on the AFT grid.
    pub samples: usize,
    pub iters: usize,
    /// Infinity norm of the final harmonic residual.
    pub residual: f64,
    /// `spectra[[i, k]]` is `X_k` of unknown `i`.
    pub spectra: Array2<Complex64>,
}

/// Sensitivities of harmonic-balance spectra: `grad[[i, k, j]]` is
/// `d X_k(outputs[i]) / d params[j]`.
#[derive(Clone, Debug)]
pub struct HbSensitivity {
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    pub param_values: Vec<f64>,
    /// `spectra[[i, k]]` is `X_k` of `outputs[i]`.
    pub spectra: Array2<Complex64>,
    pub grad: Array3<Complex64>,
}

/// Second derivatives of one coefficient `X_harmonic(output)`: `h[[a, b]]`
/// by `params[a]` and `params[b]`.
#[derive(Clone, Debug)]
pub struct HbHessian {
    pub output: String,
    pub harmonic: usize,
    pub params: Vec<String>,
    pub param_values: Vec<f64>,
    pub h: Array2<Complex64>,
}

impl Point {
    /// The periodic steady state under the circuit's periodic drive, from
    /// the operating point (all harmonics zero). Not converging within
    /// `max_iter` is an error.
    pub fn harmonic_balance(&self, opts: &HbOptions) -> Result<HarmonicBalance, ModelError> {
        if let Some((there, _)) = &self.other {
            return there.harmonic_balance(opts);
        }
        let model = &self.model;
        model.ensure_no_delays("harmonic balance")?;
        let f0 = model.hb_f0(opts.f0, &self.p)?;
        let m = model.hb_grid(opts.harmonics, opts.oversample, opts.samples);
        let x_dc = self.operating_point()?.x;
        let (res, _) = model.hb_run(
            &self.p,
            &x_dc,
            f0,
            opts.harmonics,
            m,
            opts.tol,
            opts.max_iter,
            opts.continuation,
        )?;
        if !res.converged {
            return Err(ModelError::Numeric(format!(
                "harmonic balance did not converge in {} iterations (residual {:.2e})",
                res.iters, res.residual_norm
            )));
        }
        Ok(HarmonicBalance {
            at: self.clone(),
            f0,
            harmonics: opts.harmonics,
            samples: m,
            iters: res.iters,
            residual: res.residual_norm,
            spectra: super::stack(&res.spectra, opts.harmonics + 1),
        })
    }
}

impl HarmonicBalance {
    /// The coefficients of `output` (node, unknown, branch current).
    pub fn spectrum(&self, output: &str) -> Option<ArrayView1<'_, Complex64>> {
        let i = self.at.model.resolve(output)?;
        Some(self.spectra.row(i))
    }

    /// The spectra as the HB core takes them, one row per unknown.
    fn rows(&self) -> Vec<Vec<Complex64>> {
        self.spectra.outer_iter().map(|r| r.to_vec()).collect()
    }

    /// The coefficient magnitudes `|X_k|` of `output`.
    pub fn magnitude(&self, output: &str) -> Option<Vec<f64>> {
        Some(self.spectrum(output)?.iter().map(|c| c.norm()).collect())
    }

    /// The amplitude of every harmonic of `output`: `|X_0|` for DC, the
    /// peak `2 |X_k|` above it.
    pub fn amplitude(&self, output: &str) -> Option<Vec<f64>> {
        let x = self.spectrum(output)?;
        Some(
            (x.iter().enumerate())
                .map(|(k, c)| if k == 0 { c.norm() } else { 2.0 * c.norm() })
                .collect(),
        )
    }

    /// The phase of `X_k` of `output` (degrees).
    pub fn phase_deg(&self, output: &str) -> Option<Vec<f64>> {
        Some(
            self.spectrum(output)?
                .iter()
                .map(|c| c.arg().to_degrees())
                .collect(),
        )
    }

    /// The total harmonic distortion of `output`: the RMS of the harmonics
    /// above the fundamental over the fundamental (amplitude ratio).
    pub fn thd(&self, output: &str) -> Option<f64> {
        let x = self.spectrum(output)?;
        let fund = x.get(1)?.norm();
        let rest: f64 = x.iter().skip(2).map(|c| c.norm_sqr()).sum();
        Some(rest.sqrt() / fund)
    }

    /// The derivatives of every coefficient of `outputs` by the parameters
    /// under `wrt` (all for none), exact: one adjoint per output harmonic on
    /// the two-sided HB Jacobian.
    pub fn sensitivity(&self, outputs: &[&str], wrt: &[&str]) -> Result<HbSensitivity, ModelError> {
        let model = &self.at.model;
        let outs = model.inner.outputs(outputs)?;
        let cols = model.inner.columns(wrt)?;
        let hb = model.hb_compiled(self.harmonics, self.samples)?;
        model.ensure_param_jac();
        let w0 = 2.0 * PI * self.f0;
        let mut task = log::task("SENS-HB", "sens_hb", &format!("(outputs: {})", outs.len()));
        let spectra = self.rows();
        let mut grad = Array3::zeros((outs.len(), self.harmonics + 1, cols.len()));
        for (i, &o) in outs.iter().enumerate() {
            let g = hb
                .coeff_gradient(&spectra, &self.at.p, w0, o, model.inner.store.pnames.len())
                .ok_or_else(singular)?;
            for (k, row) in g.iter().enumerate() {
                for (j, &c) in cols.iter().enumerate() {
                    grad[[i, k, j]] = row[c];
                }
            }
        }
        task.finish(format!("params: {}", cols.len()));
        Ok(HbSensitivity {
            outputs: outputs.iter().map(|o| o.to_string()).collect(),
            params: cols
                .iter()
                .map(|&c| model.inner.store.pnames[c].clone())
                .collect(),
            param_values: cols.iter().map(|&c| self.at.p[c]).collect(),
            spectra: self.spectra.select(Axis(0), &outs),
            grad,
        })
    }

    /// The second derivatives of `X_harmonic(output)` by the parameters
    /// under `wrt` (all for none), exact, nonlinear charge storage included.
    pub fn hessian(
        &self,
        output: &str,
        harmonic: usize,
        wrt: &[&str],
    ) -> Result<HbHessian, ModelError> {
        if harmonic > self.harmonics {
            return Err(ModelError::Numeric(format!(
                "harmonic {harmonic} beyond the {} solved for",
                self.harmonics
            )));
        }
        let model = &self.at.model;
        let o = model.inner.outputs(&[output])?[0];
        let cols = model.inner.columns(wrt)?;
        let _g = log::scope("sens/hb_hessian");
        model.ensure_hessian();
        let hb = model.hb_compiled(self.harmonics, self.samples)?;
        let h = hb
            .coeff_hessian(
                &self.rows(),
                &self.at.p,
                2.0 * PI * self.f0,
                o,
                harmonic,
                &cols,
            )
            .ok_or_else(singular)?;
        let h = super::stack(&h, cols.len());
        Ok(HbHessian {
            output: output.to_string(),
            harmonic,
            params: cols
                .iter()
                .map(|&c| model.inner.store.pnames[c].clone())
                .collect(),
            param_values: cols.iter().map(|&c| self.at.p[c]).collect(),
            h,
        })
    }
}

fn singular() -> ModelError {
    ModelError::Numeric(
        "harmonic balance: AFT grid too coarse for the Jacobian (need samples/2 >= 2K) \
         or singular harmonic-balance Jacobian"
            .into(),
    )
}

impl Model {
    /// The fundamental: `f0` where given, else the circuit's periodic source's.
    fn hb_f0(&self, f0: Option<f64>, p: &[f64]) -> Result<f64, ModelError> {
        match f0 {
            Some(f) if f > 0.0 => Ok(f),
            Some(f) => Err(ModelError::Numeric(format!(
                "harmonic balance needs f0 > 0, got {f}"
            ))),
            None => self.cdc().source_fundamental(p).ok_or_else(|| {
                ModelError::Numeric(
                    "harmonic balance needs f0 (no periodic source to infer it from)".into(),
                )
            }),
        }
    }

    /// Time samples per period: `samples` where given, else the alias-free
    /// count of the nonlinearity, oversampled.
    fn hb_grid(&self, harmonics: usize, oversample: usize, samples: Option<usize>) -> usize {
        samples.unwrap_or_else(|| {
            let arc = self.context_arc();
            let nl = self.dae().nonlinearity(&arc.lock().unwrap());
            hb_samples(&nl, harmonics, oversample)
        })
    }

    fn hb_compiled(&self, harmonics: usize, samples: usize) -> Result<CompiledHb<'_>, ModelError> {
        CompiledHb::new(self.cdc(), harmonics, samples).ok_or_else(|| {
            ModelError::Numeric("harmonic balance setup failed (need samples >= 2K)".into())
        })
    }

    fn ensure_param_jac(&self) {
        let arc = self.context_arc();
        self.cdc()
            .ensure_param_jac(&mut arc.lock().unwrap(), self.dae());
    }

    fn ensure_hessian(&self) {
        let arc = self.context_arc();
        self.cdc()
            .ensure_hessian(&mut arc.lock().unwrap(), self.dae());
    }

    /// The HB Newton from the DC point `x_dc`, with source-stepping
    /// continuation as `continuation` asks; and the setup's time (ms).
    #[allow(clippy::too_many_arguments)]
    fn hb_run(
        &self,
        p: &[f64],
        x_dc: &[f64],
        f0: f64,
        harmonics: usize,
        samples: usize,
        tol: f64,
        max_iter: usize,
        continuation: Option<bool>,
    ) -> Result<(HbResult, f64), ModelError> {
        let mut task = log::task(
            "HB",
            "hb",
            &format!(
                "(f0: {f0:.4e}, harmonics: {harmonics}, samples: {samples}, dim: {})",
                self.dim()
            ),
        );
        let t = sane_core::time::Instant::now();
        let hb = log_stage!("hb/setup", self.hb_compiled(harmonics, samples))?;
        let setup_ms = t.elapsed().as_secs_f64() * 1e3;
        let w0 = 2.0 * PI * f0;
        let ramp: Vec<usize> = (self.inner.store.pnames.iter().enumerate())
            .filter(|(_, n)| n.ends_with(".sin_amp"))
            .map(|(i, _)| i)
            .collect();
        let res = log_stage!(
            "hb/solve",
            match continuation {
                Some(true) if !ramp.is_empty() => {
                    hb.solve_continuation(p, x_dc, w0, &ramp, tol, max_iter)
                }
                Some(true) | Some(false) => hb.solve(p, x_dc, w0, tol, max_iter),
                None => {
                    let direct = hb.solve(p, x_dc, w0, tol, max_iter);
                    if !direct.converged && !ramp.is_empty() {
                        hb.solve_continuation(p, x_dc, w0, &ramp, tol, max_iter)
                    } else {
                        direct
                    }
                }
            }
        );
        task.finish(format!(
            "converged: {}, iters: {}, residual: {:.2e}",
            res.converged, res.iters, res.residual_norm
        ));
        Ok((res, setup_ms))
    }
}
