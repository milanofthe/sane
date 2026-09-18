//! Transient parallelism / staging benchmark on an RC ladder driven by SIN.
//!
//! Exercises the full pipeline (parse -> DAE -> compile -> DC IC -> integrate)
//! so `SANE_LOG=debug` shows every stage. Args: `[sections] [tstop] [tstep]`.
//!
//! ```text
//! SANE_LOG=debug cargo run -q --release --example tran_bench -- 40 1e-3 1e-6
//! ```

use std::time::Instant;

use sane_analysis::Model;
use sane_solve::TransientMethod;

fn ladder(n: usize) -> String {
    let mut s = String::from("Vin 1 0 SIN(0 1 1000)\n");
    for k in 1..=n {
        s.push_str(&format!("R{k} {k} {} 1k\n", k + 1));
        s.push_str(&format!("C{k} {} 0 1n\n", k + 1));
    }
    s.push_str(".end\n");
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(40);
    let tstop: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(2e-3);
    let tstep: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(1e-6);

    sane_analysis::logging::init_from_env();
    let model = Model::from_netlist(&ladder(n)).expect("build model");
    // Uniform time grid matching the old facade's fixed-step transient.
    let steps = (tstop / tstep).round().max(1.0) as usize;
    let t: Vec<f64> = (0..=steps).map(|k| k as f64 * tstep).collect();

    let t0 = Instant::now();
    let _r = model.transient(TransientMethod::Esdirk32, &[], &t, tstep, 1e-7);
    let dt = t0.elapsed();
    // Result detail goes to the UI struct; here the stage log (SANE_LOG=debug)
    // and the wall time are what we want.
    println!(
        "sections={n} tstop={tstop:.1e} tstep={tstep:.1e} elapsed={:.3} ms",
        dt.as_secs_f64() * 1e3
    );
}
