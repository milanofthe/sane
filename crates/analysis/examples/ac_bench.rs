//! AC-sweep parallelism benchmark.
//!
//! Builds an RC ladder of `N` sections and sweeps its AC response over `P`
//! frequency points, timing the sweep. The worker-thread count is taken from
//! `SANE_THREADS` (default 4), so run the same binary under different settings
//! to A/B the speedup:
//!
//! ```text
//! SANE_THREADS=1 cargo run -q --release --example ac_bench -- 120 6000
//! SANE_THREADS=4 cargo run -q --release --example ac_bench -- 120 6000
//! ```
//!
//! Args: `[sections] [points]` (defaults 120, 6000). Prints threads, elapsed,
//! and a checksum of the magnitude response so different thread counts can be
//! confirmed bit-identical.

use std::time::Instant;

use sane_analysis::Model;

fn ladder(n: usize) -> String {
    // VS -> node 1, then R_k from node k to k+1, C_k from k+1 to ground.
    let mut s = String::from("VS 1 0 AC 1\n");
    for k in 1..=n {
        s.push_str(&format!("R{k} {k} {} 1k\n", k + 1));
        s.push_str(&format!("C{k} {} 0 1n\n", k + 1));
    }
    s.push_str(".end\n");
    s
}

fn main() {
    sane_core::log::init_from_env();
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(120);
    let points: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(6000);

    let netlist = ladder(n);
    let model = Model::from_netlist(&netlist).expect("build model");
    let output = format!("{}", n + 1); // last ladder node

    // Warm up (DC solve + first pool build) so timing is the sweep itself.
    let _ = model.ac(&[], "VS", &output, 1.0, 1e8, 16).expect("warmup");

    let threads = sane_solve::parallel::threads();
    let t0 = Instant::now();
    let ac = model
        .ac(&[], "VS", &output, 1.0, 1e8, points)
        .expect("ac sweep");
    let dt = t0.elapsed();

    // Checksum: identical across thread counts iff the math is unchanged.
    let sum: f64 = ac.mag_db.iter().sum();
    println!(
        "sections={n} nodes~={n} points={points} threads={threads} \
         elapsed={:.3} ms  mag_db[checksum]={sum:.6}",
        dt.as_secs_f64() * 1e3
    );
}
