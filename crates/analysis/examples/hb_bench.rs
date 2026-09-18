//! Harmonic-balance parallelism benchmark.
//!
//! Builds a SIN-driven diode rectifier chain of `N` stages and solves the
//! periodic steady state for `K` harmonics, timing the solve. The AFT device
//! sampling (residual + Jacobian tapes over the period) runs on the worker pool,
//! so run the same binary under different `SANE_THREADS` to A/B:
//!
//! ```text
//! SANE_THREADS=1 cargo run -q --release --example hb_bench -- 40 12
//! SANE_THREADS=4 cargo run -q --release --example hb_bench -- 40 12
//! ```
//!
//! Args: `[stages] [harmonics] [f0]` (defaults 40, 12, 1000). Prints threads,
//! elapsed, convergence and a spectrum checksum (identical across thread counts).

use std::time::Instant;

use sane_analysis::Model;

fn rectifier_chain(n: usize) -> String {
    // SIN drive -> (R, diode-to-ground) chain: n nonlinear devices, n+1 nodes.
    let mut s = String::from("Vin 1 0 SIN(0.6 0.15 1000)\n");
    for k in 1..=n {
        s.push_str(&format!("R{k} {k} {} 1k\n", k + 1));
        s.push_str(&format!("D{k} {} 0 DMOD\n", k + 1));
    }
    s.push_str(".model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end\n");
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(40);
    let harmonics: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(12);
    let f0: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(1000.0);

    sane_analysis::logging::init_from_env(); // SANE_LOG=debug surfaces stage timings
    let model = Model::from_netlist(&rectifier_chain(n)).expect("build model");

    // Warm up: DC solve + first pool build, so timing is the HB solve itself.
    let _ = model
        .harmonic_balance(&[], f0, harmonics, None)
        .expect("warmup");

    let threads = sane_solve::parallel::threads();
    let t0 = Instant::now();
    let hb = model
        .harmonic_balance(&[], f0, harmonics, None)
        .expect("hb solve");
    let dt = t0.elapsed();

    // Checksum over every chain node's harmonic magnitudes (public accessor).
    let sum: f64 = (2..=n + 1)
        .filter_map(|node| {
            hb.magnitude(&node.to_string())
                .map(|m| m.iter().sum::<f64>())
        })
        .sum();
    println!(
        "stages={n} harmonics={harmonics} threads={threads} converged={} \
         elapsed={:>8.3} ms  spectrum[checksum]={sum:.6}",
        hb.converged,
        dt.as_secs_f64() * 1e3,
    );
}
