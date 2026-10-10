//! Output-referred noise analysis: resistor thermal noise plus Verilog-A
//! white / flicker / tabular sources, via one adjoint solve per frequency.

use num_complex::Complex64;
use rayon::prelude::*;
use sane_core::Graph;
use sane_core::{log_stage, ProgressTracker};
use sane_solve::CompiledDc;
use std::f64::consts::PI;

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
#[allow(clippy::too_many_arguments)]
pub fn noise_on_dae(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
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
    let n = dae.dim();
    // Sparse G (+ gmin) and C, fetched once; the adjoint system A^T w = e_out is
    // solved per frequency on the sparse path (matrix-free transpose).
    let (g_r, g_c, g_v) = log_stage!("noise/assemble_g", cdc.system_triplets_dc(x, p));
    let (c_r, c_c, c_v) = log_stage!("noise/assemble_c", cdc.jacobian_q_x_sparse(x, p, 0.0));

    // White/flicker sources: (injection vector, PSD, flicker exponent). Tabular
    // sources carry a (frequency, psd) table interpolated per frequency. A
    // source's injection is where its generator enters the rows, at the
    // operating point.
    let mut srcs: Vec<(Vec<f64>, f64, f64)> = Vec::new();
    let mut tab_srcs: Vec<NoiseTab> = Vec::new();
    for src in cdc.noise_at(ctx, dae, x, p) {
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
