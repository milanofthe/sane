//! Timing probe for the exact transient-sensitivity path (augmented DAE with
//! parameter-scaled sensitivity states).
//!
//!     cargo run --release -p sane-analysis --example tran_sens_probe

use std::time::Instant;

use sane_analysis::Model;

fn main() {
    sane_core::log::init_from_env();
    let deck = "* rc\nV1 in 0 0 SIN(0 1 1k)\nR1 in out 1k\nC1 out 0 100n\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    let t_eval: Vec<f64> = (0..120).map(|i| 3e-3 * i as f64 / 119.0).collect();
    let knobs: Vec<String> = model
        .params()
        .iter()
        .filter(|q| !q.contains('.'))
        .cloned()
        .collect();
    println!("knobs: {knobs:?}");
    let t0 = Instant::now();
    let (n, traj) = model
        .transient_sensitivity(knobs.clone(), t_eval.clone(), 1e-4, 1e-7, None)
        .expect("sens");
    println!(
        "transient_sensitivity: {:.1} ms (n={n}, rows={})",
        t0.elapsed().as_secs_f64() * 1e3,
        traj.len()
    );
    // consistency: R * dOut/dR == C * dOut/dC for the RC (both act through tau)
    let out_idx = model.resolve("out").expect("out");
    let mid = traj.len() / 2;
    let d_r = traj[mid][n + out_idx];
    let d_c = traj[mid][2 * n + out_idx];
    println!(
        "mid: R*dOut/dR = {:.6e}, C*dOut/dC = {:.6e}",
        1e3 * d_r,
        1e-7 * d_c
    );
}
