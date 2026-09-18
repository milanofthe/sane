//! Harmonic-balance solve time on a driven diode ladder -- AFT evaluation bound
//! (many time samples x many junction `exp`), for A/B of the tape batch eval.
//!
//!   cargo run -q --release --example hb_timing [k_diodes] [harmonics]

use std::time::Instant;

use sane_analysis::Model;

fn deck(k: usize, caps: bool) -> String {
    let mut s = String::from("V1 1 0 SIN(1.2 0.4 1000)\n");
    for i in 1..=k {
        s.push_str(&format!("R{i} {i} {} 1k\n", i + 1));
        s.push_str(&format!("D{i} {} 0 dm\n", i + 1));
        if caps {
            s.push_str(&format!("C{i} {} 0 10n\n", i + 1));
        }
    }
    s.push_str(".model dm D(Is=1e-14 N=1 Vt=0.025852)\n.end\n");
    s
}

fn main() {
    let k: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let kh: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let caps = std::env::args().nth(3).is_some_and(|s| s == "caps");
    let model = Model::from_netlist(&deck(k, caps)).expect("build");
    // warm
    let hb = model.harmonic_balance(&[], 1000.0, kh, None).expect("hb");
    assert!(hb.converged, "HB must converge");

    let reps = 50;
    sane_core::profile::collect_begin();
    let t = Instant::now();
    for _ in 0..reps {
        let _ = model.harmonic_balance(&[], 1000.0, kh, None).expect("hb");
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let prof = sane_core::profile::collect_take();
    println!(
        "k={k} diodes, K={kh} harmonics, dim={}: HB solve {ms:.2} ms/solve",
        model.dim()
    );

    // Per-solve breakdown of the hb/* Newton stages, aggregated over all reps.
    let iters = prof
        .millis()
        .iter()
        .filter(|(n, _)| n == "hb/synth")
        .count()
        / reps;
    println!("  ({iters} Newton iterations/solve)");
    let mut covered = 0.0;
    for (name, total) in prof.aggregated_millis() {
        if name == "hb/solve" {
            continue; // the outer whole-solve span; the lines below are its parts
        }
        if let Some(short) = name.strip_prefix("hb/") {
            let per = total / reps as f64;
            covered += per;
            println!(
                "  {short:12} {per:8.3} ms/solve  ({:5.1} %)",
                per / ms * 100.0
            );
        }
    }
    println!(
        "  {:12} {:8.3} ms/solve  ({:5.1} %)",
        "other",
        ms - covered,
        (ms - covered) / ms * 100.0
    );
}
