//! Output-referred noise analysis: resistor thermal noise plus Verilog-A
//! white / flicker / tabular sources, via one adjoint solve per frequency.

use num_complex::Complex64;
use rayon::prelude::*;
use rsdag::{Node, SymbolId};
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

/// Output-referred noise voltage spectral density over a log frequency band,
/// from the already-assembled DAE at the operating point `x` (parameters `p`).
/// Consumes the DAE's noise registry uniformly (resistor thermal noise plus any
/// Verilog-A white / flicker / tabular sources): each source's transimpedance to
/// the output `out_idx` comes from one adjoint solve `A^T w = e_out` per
/// frequency (`A = G + jwC`), and the output PSD is `sum |w . (e_a - e_b)|^2 * PSD`.
/// Returns `(freqs_hz, noise_v)` with `noise_v = sqrt(PSD)` in V/sqrt(Hz).
pub fn noise_on_dae(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    cdc: &CompiledDc,
    out_idx: usize,
    x: &[f64],
    p: &[f64],
    fstart: f64,
    fstop: f64,
    points: usize,
    temp_k: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if !(fstart > 0.0) || !(fstop > fstart) || points < 2 {
        return Err("noise needs 0 < fstart < fstop and points >= 2".into());
    }
    let n = dae.dim();
    let xdot0 = vec![0.0; n];
    // Sparse G (+ gmin) and C, fetched once; the adjoint system A^T w = e_out is
    // solved per frequency on the sparse path (matrix-free transpose).
    let (g_r, g_c, g_v) = log_stage!("noise/assemble_g", cdc.system_triplets_dc(x, p));
    let (c_r, c_c, c_v) = log_stage!(
        "noise/assemble_c",
        cdc.jacobian_xdot_sparse(x, &xdot0, p, 0.0)
    );

    // Evaluation environment: unknowns -> x, parameters -> p (the PSD expressions
    // reference both).
    let pnames = cdc.param_names(ctx);
    let mut env: std::collections::HashMap<SymbolId, f64> = std::collections::HashMap::new();
    for (i, &xs) in dae.x.iter().enumerate() {
        env.insert(xs, x.get(i).copied().unwrap_or(0.0));
    }
    for (j, nm) in pnames.iter().enumerate() {
        let s = ctx.sym(nm);
        if let Node::Symbol(sid) = ctx.node(s) {
            env.insert(*sid, p.get(j).copied().unwrap_or(0.0));
        }
    }
    // The global circuit temperature `$temp` drives resistor thermal noise even
    // in a resistor-only deck, where no device makes it a DC parameter (so it is
    // absent from `pnames`/`p`). Bind it from the deck temperature, without
    // overriding a value the parameter vector already supplies (device circuits,
    // temperature sweeps, `$temp` overrides).
    {
        let ts = ctx.sym(sane_core::constants::TEMP_SYMBOL);
        if let Node::Symbol(sid) = ctx.node(ts) {
            env.entry(*sid).or_insert(temp_k);
        }
    }
    let idx_of = |sid: SymbolId| dae.x.iter().position(|&s| s == sid);

    // White/flicker sources: (injection vector, PSD, flicker exponent). Tabular
    // sources carry a (frequency, psd) table interpolated per frequency.
    let mut srcs: Vec<(Vec<f64>, f64, f64)> = Vec::new();
    let mut tab_srcs: Vec<NoiseTab> = Vec::new();
    for ns in &dae.noise_sources {
        let mut u = vec![0.0; n];
        if let Some(s) = ns.hi {
            if let Some(i) = idx_of(s) {
                u[i] += 1.0;
            }
        }
        if let Some(s) = ns.lo {
            if let Some(i) = idx_of(s) {
                u[i] -= 1.0;
            }
        }
        if !ns.table.is_empty() {
            tab_srcs.push((u, ns.table.clone()));
        } else {
            let v = rsdag::eval(ctx, &[ns.psd, ns.flicker_exp], &env);
            let (psd, fexp) = (v[0], v[1]);
            if psd.is_finite() && psd > 0.0 && fexp.is_finite() {
                srcs.push((u, psd, fexp));
            }
        }
    }

    if srcs.is_empty() && tab_srcs.is_empty() {
        return Err("no noise sources (resistors or Verilog-A) to contribute".into());
    }

    let mut e_out = vec![Complex64::new(0.0, 0.0); n];
    e_out[out_idx] = Complex64::new(1.0, 0.0);

    // Adjoint system A^T w = e_out with A = G + jwC: the pattern is fixed across
    // the sweep, so factor A^T symbolically once and reuse it per frequency. The
    // sweep is embarrassingly parallel (sources read-only) -- run it on the worker
    // pool with faer pinned sequential; `collect` preserves frequency order.
    let sys = log_stage!(
        "noise/symbolic",
        crate::sparse_ac::SymbolicAc::new(n, (&g_r, &g_c, &g_v), (&c_r, &c_c, &c_v), true)
    );
    let (l0, l1) = (fstart.log10(), fstop.log10());
    let mut tracker = ProgressTracker::with_details(
        points as f64,
        "NOISE",
        &format!(
            "(points: {points}, sources: {} white/flicker + {} tabular)",
            srcs.len(),
            tab_srcs.len()
        ),
    );
    tracker.start();
    let rows: Vec<(f64, f64)> = log_stage!(
        "noise/sweep",
        sane_solve::parallel::install(|| {
            (0..points)
                .into_par_iter()
                // Per-worker sweep state (KLU refactor across frequencies).
                .map_init(
                    || sys.as_ref().map(|s| s.solver()),
                    |fac, k| {
                    let fk = 10f64.powf(l0 + (l1 - l0) * k as f64 / (points - 1) as f64);
                    let w = 2.0 * PI * fk;
                    let adj = match fac.as_mut().and_then(|f| f.solve(w, &e_out)) {
                        Some(v) => v,
                        // Singular A^T at this frequency: no adjoint, hence no
                        // noise transfer. NaN, not a silent 0.0 that reads as
                        // a noiseless point (same policy as AC, issue #39).
                        None => return (fk, f64::NAN),
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
                    (fk, psd.max(0.0).sqrt())
                })
                .collect()
        })
    );
    tracker.close();
    let (mut f, mut noise_v) = (Vec::with_capacity(points), Vec::with_capacity(points));
    for (fk, nv) in rows {
        f.push(fk);
        noise_v.push(nv);
    }
    Ok((f, noise_v))
}
