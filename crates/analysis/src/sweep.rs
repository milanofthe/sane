//! Temperature sweep of an output over the DC operating point, driven by the
//! global `$temp` symbol.

use sane_core::ProgressTracker;
use sane_solve::CompiledDc;

/// Temperature sweep of the output `out_idx`, from the already-assembled DAE.
/// Junction parameters follow the standard SPICE temperature model:
/// `Vt(T) = Vt_nom * T/Tnom` and
/// `Is(T) = Is_nom * (T/Tnom)^XTI * exp(-(Eg/k)(1/T - 1/Tnom))`
/// (Tnom = 300 K, Eg = 1.11 eV, XTI = 3), applied to every `.Vt`/`.Is`
/// parameter. `p0` is the nominal parameter vector in `pnames` order. Returns
/// the converged `(temps_C, output_value)` points.
pub fn temp_sweep_on_dae(
    cdc: &CompiledDc,
    pnames: &[String],
    out_idx: usize,
    p0: &[f64],
    tstart: f64,
    tstop: f64,
    points: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if !(2..=100_000).contains(&points) {
        return Err("temp_sweep needs points >= 2".into());
    }
    let mut x_warm: Vec<f64> = Vec::new();
    let (mut temps_c, mut values) = (Vec::with_capacity(points), Vec::with_capacity(points));
    let mut tracker = ProgressTracker::with_details(
        points as f64,
        "TEMP-SWEEP",
        &format!("(points: {points}, T: {tstart}..{tstop} C)"),
    );
    tracker.start();
    for k in 0..points {
        let tc = tstart + (tstop - tstart) * k as f64 / (points - 1) as f64;
        let t = tc + sane_core::constants::ZERO_CELSIUS_K;
        // The whole temperature model now lives in the device equations (thermal
        // voltage k*T/q, Is(T) with Eg/XTI, mobility (T/Tnom)^-1.5), driven by the
        // single global `$temp` symbol -- both native and Verilog-A devices read
        // it. So the sweep just sets `$temp` [K]; rescaling individual `.Is`/`.Vt`
        // params here would double-count the dependence.
        let p: Vec<f64> = pnames
            .iter()
            .zip(p0)
            .map(|(name, &v)| {
                if name == sane_core::constants::TEMP_SYMBOL {
                    t
                } else {
                    v
                }
            })
            .collect();
        let (x, conv, _) = cdc.solve_dc(&p, &x_warm, 1e-10, 100);
        tracker.update((k + 1) as f64 / points as f64, conv);
        if conv {
            x_warm = x.clone();
            temps_c.push(tc);
            values.push(x.get(out_idx).copied().unwrap_or(0.0));
        }
    }
    tracker.close();
    if values.is_empty() {
        return Err("no temperature point converged".into());
    }
    Ok((temps_c, values))
}
