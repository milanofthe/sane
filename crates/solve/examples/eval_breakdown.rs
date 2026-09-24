//! How much of a device-heavy DC solve is tape evaluation (residual + Jacobian,
//! dominated by the junction `exp`) versus the rest (sparse LU + Newton
//! orchestration + homotopy)? This sizes the win available from a faster `exp`.
//!
//!   cargo run -q --release --example eval_breakdown [k]
//!
//! `k` is the number of diodes (each a junction `exp` per residual eval); default 80.

use std::time::Instant;

use sane_core::Graph;
use sane_dae::{assemble_dae, Dae, DeviceInstance};
use sane_mna::Circuit;
use sane_solve::CompiledDc;

/// A diode ladder: node 1 driven by V1, a series resistor between consecutive
/// nodes, and a diode-to-ground at each node. `k` diodes => `k` exp() per
/// residual eval, so device evaluation dominates the solve.
fn build_ladder(k: usize) -> (Graph, Dae) {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0);
    let mut devs = Vec::new();
    for i in 1..=k {
        c.resistor(&format!("R{i}"), i, i + 1);
        devs.push(DeviceInstance::new(
            Box::new(sane_veriloga::builtin_device(
                "sane_diode",
                format!("D{i}"),
                &[],
            )),
            vec![i + 1, 0],
        ));
    }
    let dae = assemble_dae(&mut ctx, &c, &devs);
    (ctx, dae)
}

fn params(ctx: &Graph, dae: &Dae) -> Vec<f64> {
    let tnom = sane_core::constants::TEMP_NOMINAL_K;
    dae.params(ctx)
        .iter()
        .map(|&s| {
            let n = ctx.symbol_name(s);
            if n == "V1" {
                3.0
            } else if n.starts_with('R') && !n.contains('.') {
                1000.0
            } else if n.ends_with(".Is") {
                1e-12
            } else if n.ends_with(".N") {
                1.0
            } else if n.ends_with(".Vt") {
                0.025852
            } else if n == sane_core::constants::TEMP_SYMBOL || n.ends_with(".Tnom") {
                tnom
            } else if n.ends_with(".Eg") {
                1.11
            } else if n.ends_with(".XTI") {
                3.0
            } else {
                0.0
            }
        })
        .collect()
}

fn main() {
    let k: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);
    let (mut ctx, dae) = build_ladder(k);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params(&ctx, &dae);
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 200);
    assert!(conv, "DC must converge");
    let n = cdc.dim();
    let xdot = vec![0.0; n];

    // Isolated eval: residual + Jacobian-x values at the solved point, no LU, no
    // Newton -- exactly the work a faster `exp` speeds up.
    let reps = 5000;
    let t = Instant::now();
    for _ in 0..reps {
        let _ = cdc.jacobian_x_sparse(&x, &xdot, &p, 0.0);
        let _ = cdc.residual(&x, &xdot, &p, 0.0);
    }
    let eval_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

    // Full DC solve (eval + LU + Newton + homotopy).
    let reps = 500;
    let t = Instant::now();
    for _ in 0..reps {
        let _ = cdc.solve_dc(&p, &[], 1e-12, 200);
    }
    let solve_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

    println!(
        "k={k} diodes, dim={n}: isolated eval {eval_us:8.2} us | full DC solve {solve_us:9.1} us \
         (eval is ~{:.0} us of one Newton step)",
        eval_us
    );
}
