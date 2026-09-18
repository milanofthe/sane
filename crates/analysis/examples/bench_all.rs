//! Stage-resolved benchmark of every analysis over the big circuits, using the
//! global `Profile` sink (no log parsing).
//!
//! For each circuit it builds the model and runs OP / AC / transient / noise /
//! pole-zero / DC-sensitivity / pole-gradient, capturing the per-stage timing of
//! each via `sane_core::profile::{collect_begin, collect_take}`. Prints, per
//! circuit and analysis, the wall time and the dominant stages.
//!
//! ```text
//! cargo run -q --release --example bench_all
//! cargo run -q --release --example bench_all -- path1.cir:Vin:vout path2.cir:Vd:22
//! ```
//!
//! A circuit spec is `path:ac_input:output`. With no args it uses the bundled
//! AnalogGym SANE netlists (emit them first with `emit_sane_amps.py`) plus µA741.

use std::time::Instant;

use sane_analysis::Model;
use sane_core::profile::{self, Profile};

/// Run `f` with profile collection active; return (wall ms, captured profile).
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64, Profile) {
    profile::collect_begin();
    let t0 = Instant::now();
    let r = f();
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    (r, ms, profile::collect_take())
}

/// Print one analysis row: wall time + the top stages by aggregated time.
fn report(name: &str, ok: bool, ms: f64, prof: &Profile) {
    let status = if ok { "" } else { "  [FAILED]" };
    let mut stages = prof.aggregated_millis();
    stages.retain(|(_, t)| *t >= 0.05);
    stages.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top: Vec<String> = stages
        .iter()
        .take(4)
        .map(|(n, t)| format!("{n} {t:.1}"))
        .collect();
    println!("    {name:<14} {ms:>9.2} ms{status}   {}", top.join(" | "));
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

fn analyggym_specs() -> Vec<(String, String, String)> {
    // The 12 amps that converge in SANE's DC, plus µA741.
    let amps = [
        "Alfio_RAFFC_Pin_3",
        "Fan_SMC_Pin_3",
        "HoiLee_AFFC_Pin_3",
        "Leung_DFCFC1_Pin_3",
        "Leung_DFCFC2_Pin_3",
        "Leung_NMCF_Pin_3",
        "Leung_NMCNR_Pin_3",
        "Peng_ACBC_Pin_3",
        "Qu2017_AZC_Pin_3",
        "Ramos_PFC_Pin_3",
        "Sau_CFCC_Pin_3",
        "Yan_AZ_Pin_3",
    ];
    let mut v: Vec<(String, String, String)> = amps
        .iter()
        .map(|a| {
            (
                format!("paper/figures/data/_sane_{a}.cir"),
                "Vin".to_string(),
                "vout".to_string(),
            )
        })
        .collect();
    v.push((
        "crates/netlist/tests/fixtures/ua741.cir".to_string(),
        "Vd".to_string(),
        "22".to_string(),
    ));
    v
}

fn run_circuit(path: &str, input: &str, output: &str) {
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path);
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => {
            println!("\n=== {name}  (skipped: file not found) ===");
            return;
        }
    };

    let (model, ms, prof) = timed(|| Model::from_netlist(&src));
    let model = match model {
        Ok(m) => m,
        Err(e) => {
            println!("\n=== {name}  (build failed: {e:?}) ===");
            return;
        }
    };
    println!("\n=== {name}  (dim {}) ===", model.dim());
    report("build", true, ms, &prof);

    // OP first: reused (x, p) for the point-wise sensitivities.
    let (op, ms, prof) = timed(|| model.operating_point(&[]));
    let op = match op {
        Ok(o) => {
            report("op", true, ms, &prof);
            o
        }
        Err(e) => {
            report("op", false, ms, &prof);
            println!("    (operating point failed: {e:?}; skipping rest)");
            return;
        }
    };
    let x = op.vector().to_vec();
    let p = model.pvec(&[]);

    // AC sweep.
    let (r, ms, prof) = timed(|| model.ac(&[], input, output, 1.0, 1e8, 2000));
    report("ac", r.is_ok(), ms, &prof);

    // Noise sweep.
    let (r, ms, prof) = timed(|| model.noise(&[], output, 1.0, 1e8, 2000));
    report("noise", r.is_ok(), ms, &prof);

    // Transient over a short window (exercises the BDF path + DC IC).
    let t_eval: Vec<f64> = (0..=200).map(|k| k as f64 * 1e-6 / 200.0).collect();
    let (r, ms, prof) = timed(|| {
        model.transient(
            sane_solve::TransientMethod::Esdirk32,
            &[],
            &t_eval,
            1e-4,
            1e-7,
        )
    });
    report("transient", r.is_ok(), ms, &prof);

    // Pole/zero.
    let (r, ms, prof) = timed(|| model.poles_zeros(&[], input, output));
    report("poles_zeros", r.is_ok(), ms, &prof);

    // DC sensitivity (adjoint) at the solved point -- no re-solve.
    let (r, ms, prof) = timed(|| op.sensitivity(output));
    report("sensitivity", r.is_ok(), ms, &prof);
    let _ = (&x, &p);

    // NOTE: `pole_gradient` is intentionally not run here -- on a BSIM4-sized
    // system (dim > ~300) it overflows the main thread's stack (deep recursion),
    // which aborts the whole process. Tracked as a scaling bug to fix separately;
    // run it explicitly on a small circuit if needed.
}

fn main() {
    sane_analysis::logging::init_from_env();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let specs: Vec<(String, String, String)> = if args.is_empty() {
        analyggym_specs()
    } else {
        args.iter()
            .filter_map(|a| {
                let parts: Vec<&str> = a.split(':').collect();
                match parts.as_slice() {
                    [path, inp, out] => Some((path.to_string(), inp.to_string(), out.to_string())),
                    _ => {
                        eprintln!("bad spec '{a}' (want path:input:output)");
                        None
                    }
                }
            })
            .collect()
    };

    println!(
        "threads={}  (SANE_THREADS)",
        sane_solve::parallel::threads()
    );
    for (path, inp, out) in &specs {
        run_circuit(path, inp, out);
    }
}
