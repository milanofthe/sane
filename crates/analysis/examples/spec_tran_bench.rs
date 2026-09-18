//! End-to-end A/B for tape specialization: a transient on a MOSFET
//! common-source chain (region-structured devices, the case choice pinning
//! targets). Run twice to compare:
//!
//!   cargo run -q --release --example spec_tran_bench                  # specialized
//!   SANE_TAPE_SPEC=0 cargo run -q --release --example spec_tran_bench # full tape
//!
//! Args: `[stages] [periods]` (default 50 stages, 5 periods of the 10 kHz input).

use std::time::Instant;

use sane_analysis::Model;
use sane_solve::TransientMethod;

fn main() {
    sane_core::log::init_from_env();
    let mut args = std::env::args().skip(1);
    let stages: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(50);
    let periods: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(5);

    // Gate driven around the saturation bias; the swing crosses regions a few
    // times per period, so the guards see realistic flips, not a static pin.
    let mut d = String::from("VDD 1 0 5\nVIN 2 0 SIN(2 0.8 10k)\n");
    for i in 0..stages {
        let n = 3 + i;
        d.push_str(&format!(
            "RD{i} 1 {n} 1k\nCL{i} {n} 0 1n\nM{i} {n} 2 0 0 nm W=1 L=1\n"
        ));
    }
    d.push_str(".model nm NMOS(Vto=1 Kp=2e-3 lambda=0.01 theta=0.05)\n.end\n");

    let model = Model::from_netlist(&d).expect("build model");
    let tstop = periods as f64 * 1e-4;
    let t: Vec<f64> = (0..=1000).map(|k| k as f64 * tstop / 1000.0).collect();

    // Warm-up (JIT-free engine, but page in tapes/factorizations).
    let _ = model.transient(TransientMethod::Esdirk32, &[], &t, 1e-6, 1e-9);

    let reps = 5;
    let t0 = Instant::now();
    let mut check = 0.0;
    for _ in 0..reps {
        let tr = model
            .transient(TransientMethod::Esdirk32, &[], &t, 1e-6, 1e-9)
            .expect("transient");
        // Trajectory checksum: identical across the A/B modes iff the
        // specialized evaluations are bit-exact along the whole solve.
        check = tr
            .signal("3")
            .map(|row| row.iter().sum::<f64>())
            .unwrap_or(f64::NAN);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let mode = if std::env::var_os("SANE_TAPE_SPEC").is_some_and(|v| v == "0") {
        "full tape"
    } else {
        "specialized"
    };
    println!(
        "{mode:12} stages={stages} periods={periods}: {ms:8.1} ms/transient  checksum {check:.17e}"
    );
}
