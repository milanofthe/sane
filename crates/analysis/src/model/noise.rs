//! Noise analysis on [`Model`]: the output-referred spectrum and its exact
//! all-parameter gradient.

use rayon::prelude::*;
use sane_core::{log_stage, ProgressTracker};
use sane_solve::CompiledDc;
use std::f64::consts::PI;

use num_complex::Complex64;
use rsdag::ExprId;
use sane_core::log;

use ndarray::{Array2, Array3, ArrayView1};

use crate::model::{Model, ModelError, Point};

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
        let (dr, dc, dv, dtau) = model.delay_ac_entries(&x, &self.p);
        let outs = model.inner.outputs(outputs)?;
        let rows = noise_on_dae(
            model.cdc(),
            &outs,
            &x,
            &self.p,
            freqs,
            (&dr, &dc, &dv, &dtau),
        )
        .map_err(ModelError::Numeric)?;
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
    /// `lambda = A^{-T} e_out`, injection `u_q` (where the generator enters),
    /// and PSD `S_q`. With `r = sum_q 2 conj(T_q) S_q u_q` and `xi = A^{-1} r`,
    /// `dN/dp` is the total derivative of the real functional
    /// `Phi = Re(-lambda^T A xi) + sum_q |T_q|^2 fac_q psd_q + sum_q Re(2 S_q
    /// conj(T_q) lambda^T u_q)`, the last two over the sources' level and
    /// injection expressions: one pull (see [`super::adjoint`]). White and
    /// flicker sources. Returns `(N, [(name, dN/dp)])`.
    pub(crate) fn noise_gradient(
        &self,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freq: f64,
    ) -> Result<(f64, Vec<(String, f64)>), ModelError> {
        let _g = log::scope("sens/noise_gradient");
        let at = self.adjoint_point(&x, &p)?;
        let n = at.n;
        let w = 2.0 * PI * freq;
        let g = self.cdc().system_triplets_dc(&x, &p);
        let cq = self.cdc().jacobian_q_x_sparse(&x, &p, 0.0);
        let singular = || ModelError::Numeric("noise_gradient: singular A".to_string());
        let sys = crate::sparse_ac::SymbolicAc::new(
            n,
            (&g.rows, &g.cols, &g.vals),
            (&cq.rows, &cq.cols, &cq.vals),
            (&[], &[], &[], &[]),
            false,
            w,
        )
        .ok_or_else(singular)?;
        let mut fac = sys.solver();
        let mut e_out = vec![Complex64::new(0.0, 0.0); n];
        e_out[out_idx] = Complex64::new(1.0, 0.0);
        let lam = fac
            .solve_transpose(w, &e_out)
            .ok_or_else(singular)?
            .to_vec();

        // The sources at the point, and the pull's own entries: every
        // source's level, then every injection entry, in source order.
        let sources = self.cdc().noise_at(&x, &p);
        let inj_rows: Vec<Vec<usize>> = (self.dae().noise_injection(&mut self.cdc().graph()))
            .iter()
            .map(|row| row.iter().map(|&(i, _)| i).collect())
            .collect();
        let dae = self.dae();
        let make = |c: &mut sane_core::Graph| -> Vec<ExprId> {
            let psd = dae
                .observers
                .noise(c)
                .iter()
                .map(|ns| ns.psd)
                .collect::<Vec<_>>();
            let inj = dae.noise_injection(c);
            psd.into_iter()
                .chain(inj.iter().flat_map(|row| row.iter().map(|&(_, e)| e)))
                .collect()
        };
        let pull = self.pencil_pull(None, Some(("noise", &make)))?;
        let nsrc = sources.len();
        let n_inj: usize = inj_rows.iter().map(Vec::len).sum();
        let mut extra = vec![Complex64::new(0.0, 0.0); nsrc + n_inj];
        let mut nval = 0.0;
        let mut r = vec![Complex64::new(0.0, 0.0); n]; // r = sum_q 2 conj(T_q) S_q u_q
        let mut at_inj = nsrc;
        for (q, src) in sources.iter().enumerate() {
            let base = at_inj;
            at_inj += inj_rows[q].len();
            // Tabular sources: not yet in the gradient.
            let sane_dae::NoiseLevel::Spectral { psd: sp, fexp } = src.level else {
                continue;
            };
            if !sp.is_finite() || sp <= 0.0 || !fexp.is_finite() {
                continue;
            }
            let fac_q = if fexp == 0.0 { 1.0 } else { freq.powf(-fexp) };
            let sq = sp * fac_q;
            let tq: Complex64 = src.injection.iter().map(|&(i, u)| lam[i] * u).sum();
            nval += sq * tq.norm_sqr();
            let wgt = 2.0 * sq * tq.conj();
            for &(i, u) in &src.injection {
                r[i] += wgt * u;
            }
            extra[q] = Complex64::new(tq.norm_sqr() * fac_q, 0.0);
            for (k, &i) in inj_rows[q].iter().enumerate() {
                extra[base + k] = Complex64::new((wgt * lam[i]).re, 0.0);
            }
        }
        let xi = fac.solve(w, &r).ok_or_else(singular)?.to_vec();
        let phi = super::adjoint::Bilinear {
            l: &lam,
            r: &xi,
            a_g: Complex64::new(-1.0, 0.0),
            a_c: Complex64::new(0.0, -w),
            a_b: Complex64::new(0.0, 0.0),
            extra: &extra,
        };
        let (dx, dp) = self.pull(&at, &pull, &phi)?;
        let grad = self.total(&at, &dx, dp)?;
        Ok((
            nval,
            at.pnames
                .into_iter()
                .zip(grad.iter().map(|g| g.re))
                .collect(),
        ))
    }
}

/// Output-referred noise vs frequency from resistor thermal noise. Each resistor
/// R is a current-noise source of PSD `4kT/R` between its nodes; the contribution
/// to the output is its transimpedance `Z = e_out^T (G + jwC)^{-1} (e_a - e_b)`,
/// got cheaply for all resistors at once via one adjoint solve `w = A^{-T} e_out`
/// per frequency. Output PSD = sum_R (4kT/R) |w . (e_a - e_b)|^2.
/// (Semiconductor shot/flicker noise is a later extension.)
/// A tabular noise source: its injection vector and a `(frequency, psd)` table.
type NoiseTab = (Vec<f64>, Vec<(f64, f64)>);

/// Linear interpolation of a `(frequency, psd)` table (sorted by frequency),
/// clamped to the end values outside the range.
fn interp_table(table: &[(f64, f64)], f: f64) -> f64 {
    if table.is_empty() {
        return 0.0;
    }
    if f <= table[0].0 {
        return table[0].1;
    }
    let last = table[table.len() - 1];
    if f >= last.0 {
        return last.1;
    }
    for w in table.windows(2) {
        let ((f0, p0), (f1, p1)) = (w[0], w[1]);
        if f >= f0 && f <= f1 {
            if (f1 - f0).abs() < f64::MIN_POSITIVE {
                return p0;
            }
            let t = (f - f0) / (f1 - f0);
            return p0 + t * (p1 - p0);
        }
    }
    last.1
}

/// Output-referred noise power spectral densities at the frequencies `freqs`
/// (Hz, each positive), from the already-assembled DAE at the operating point
/// `x` (parameters `p`). Consumes the DAE's noise registry uniformly (resistor
/// thermal noise plus any Verilog-A white / flicker / tabular sources): each
/// source's transimpedance to an output comes from one adjoint solve
/// `A^T w = e_out` per frequency (`A = G + jwC`, one factorization for every
/// output), and the output PSD is `sum |w . (e_a - e_b)|^2 * PSD`. Returns
/// `psd[k][i]` for `outs[i]` at `freqs[k]` (V^2/Hz or A^2/Hz; `NaN` where `A`
/// is singular).
pub(crate) fn noise_on_dae(
    cdc: &CompiledDc,
    outs: &[usize],
    x: &[f64],
    p: &[f64],
    freqs: &[f64],
    delays: (&[usize], &[usize], &[f64], &[f64]),
) -> Result<Vec<Vec<f64>>, String> {
    if freqs.iter().any(|&f| !(f > 0.0)) {
        return Err("noise needs positive frequencies".into());
    }
    let n = cdc.dim();
    // Sparse G (+ gmin) and C, fetched once; the adjoint system A^T w = e_out is
    // solved per frequency on the sparse path (matrix-free transpose).
    let sane_solve::Triplets {
        rows: g_r,
        cols: g_c,
        vals: g_v,
    } = log_stage!("noise/assemble_g", cdc.system_triplets_dc(x, p));
    let sane_solve::Triplets {
        rows: c_r,
        cols: c_c,
        vals: c_v,
    } = log_stage!("noise/assemble_c", cdc.jacobian_q_x_sparse(x, p, 0.0));

    // White/flicker sources: (injection vector, PSD, flicker exponent). Tabular
    // sources carry a (frequency, psd) table interpolated per frequency. A
    // source's injection is where its generator enters the rows, at the
    // operating point.
    let mut srcs: Vec<(Vec<f64>, f64, f64)> = Vec::new();
    let mut tab_srcs: Vec<NoiseTab> = Vec::new();
    for src in cdc.noise_at(x, p) {
        let mut u = vec![0.0; n];
        for &(i, g) in &src.injection {
            u[i] += g;
        }
        match src.level {
            sane_dae::NoiseLevel::Table(table) => tab_srcs.push((u, table)),
            sane_dae::NoiseLevel::Spectral { psd, fexp } => {
                if psd.is_finite() && psd > 0.0 && fexp.is_finite() {
                    srcs.push((u, psd, fexp));
                }
            }
        }
    }

    if srcs.is_empty() && tab_srcs.is_empty() {
        return Err("no noise sources (resistors or Verilog-A) to contribute".into());
    }

    let seeds: Vec<Vec<Complex64>> = (outs.iter())
        .map(|&o| {
            let mut e = vec![Complex64::new(0.0, 0.0); n];
            e[o] = Complex64::new(1.0, 0.0);
            e
        })
        .collect();

    // Adjoint system A^T w = e_out with A = G + jwC: the pattern is fixed across
    // the sweep, so factor A^T symbolically once and reuse it per frequency. The
    // sweep is embarrassingly parallel (sources read-only) -- run it on the worker
    // pool with faer pinned sequential; `collect` preserves frequency order.
    let sys = log_stage!(
        "noise/symbolic",
        crate::sparse_ac::SymbolicAc::new(
            n,
            (&g_r, &g_c, &g_v),
            (&c_r, &c_c, &c_v),
            delays,
            true,
            2.0 * PI * freqs.first().copied().unwrap_or(1.0),
        )
    );
    let mut tracker = ProgressTracker::with_details(
        freqs.len() as f64,
        "NOISE",
        &format!(
            "(points: {}, outputs: {}, sources: {} white/flicker + {} tabular)",
            freqs.len(),
            outs.len(),
            srcs.len(),
            tab_srcs.len()
        ),
    );
    tracker.start();
    let rows: Vec<Vec<f64>> = log_stage!(
        "noise/sweep",
        sane_solve::parallel::install(|| {
            freqs
                .par_iter()
                // Per-worker sweep state (KLU refactor across frequencies).
                .map_init(
                    || sys.as_ref().map(|s| s.solver()),
                    |fac, &fk| {
                    let w = 2.0 * PI * fk;
                    seeds.iter().map(|e_out| {
                    let adj = match fac.as_mut().and_then(|f| f.solve(w, e_out)) {
                        Some(v) => v,
                        // Singular A^T at this frequency: no adjoint, hence no
                        // noise transfer. NaN, not a silent 0.0 that reads as
                        // a noiseless point (same policy as AC, issue #39).
                        None => return f64::NAN,
                    };
                    let mut psd = 0.0;
                    for (u, sp, exp) in &srcs {
                        let mut z = Complex64::new(0.0, 0.0);
                        for i in 0..n {
                            if u[i] != 0.0 {
                                z += adj[i] * u[i];
                            }
                        }
                        // White: flat PSD. Flicker: PSD scales as 1/f^exp.
                        let fac = if *exp == 0.0 { 1.0 } else { fk.powf(-*exp) };
                        psd += sp * fac * z.norm_sqr();
                    }
                    // Tabular sources: PSD linearly interpolated at this frequency.
                    for (u, table) in &tab_srcs {
                        let sp = interp_table(table, fk);
                        if sp <= 0.0 {
                            continue;
                        }
                        let mut z = Complex64::new(0.0, 0.0);
                        for i in 0..n {
                            if u[i] != 0.0 {
                                z += adj[i] * u[i];
                            }
                        }
                        psd += sp * z.norm_sqr();
                    }
                    psd.max(0.0)
                    }).collect()
                })
                .collect()
        })
    );
    tracker.close();
    Ok(rows)
}
