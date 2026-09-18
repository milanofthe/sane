//! Timing probe: discrete transient adjoint vs forward sensitivity on a
//! many-parameter circuit (RC ladder, ~2 parameters per stage). The adjoint
//! cost must stay flat in the parameter count.
//!
//!     cargo run --release -p sane-analysis --example adjoint_probe [N]

use std::time::Instant;

use sane_analysis::Model;

fn rc_ladder(n: usize) -> String {
    let mut s = String::from("* rc ladder\nV1 n0 0 0 SIN(0 1 1k)\n");
    for i in 1..=n {
        s.push_str(&format!("R{i} n{} n{i} 1k\n", i - 1));
        s.push_str(&format!("C{i} n{i} 0 100n\n"));
    }
    s.push_str(".end\n");
    s
}

fn main() {
    sane_core::log::init_from_env();
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100);
    let model = Model::from_netlist(&rc_ladder(n)).expect("model");
    let dim = model.dim();
    let out_idx = model.resolve(&format!("n{n}")).expect("out");
    let npts = 400usize;
    let t_eval: Vec<f64> = (0..npts)
        .map(|k| 2e-3 * k as f64 / (npts - 1) as f64)
        .collect();
    let w: Vec<f64> = (0..npts).map(|k| 0.5 + (k as f64 * 0.7).sin()).collect();
    let cot: Vec<Vec<f64>> = w
        .iter()
        .map(|wk| {
            let mut c = vec![0.0; dim];
            c[out_idx] = *wk;
            c
        })
        .collect();

    let t0 = Instant::now();
    let (pnames, grad) = model
        .transient_adjoint(t_eval.clone(), cot, None, None)
        .expect("adjoint");
    let t_adj = t0.elapsed().as_secs_f64();
    println!(
        "N={n} dim={dim} params={} | adjoint (ALL params): {:.3} s | dJ/dR1={:.4e} dJ/dC1={:.4e}",
        pnames.len(),
        t_adj,
        grad[pnames.iter().position(|q| q == "R1").unwrap()],
        grad[pnames.iter().position(|q| q == "C1").unwrap()],
    );

    // forward path over the same top-level knobs, same grid, for scale
    let knobs: Vec<String> = pnames
        .iter()
        .filter(|q| !q.contains('.'))
        .cloned()
        .collect();
    let t0 = Instant::now();
    let (nn, traj) = model
        .transient_sensitivity(knobs.clone(), t_eval.clone(), 1e-4, 1e-7, None)
        .expect("forward");
    let t_fwd = t0.elapsed().as_secs_f64();
    let ki = knobs.iter().position(|q| q == "R1").unwrap();
    let fwd_r1: f64 = traj
        .iter()
        .zip(&w)
        .map(|(row, wk)| wk * row[nn * (ki + 1) + out_idx])
        .sum();
    println!(
        "forward ({} knobs): {:.3} s | dJ/dR1={:.4e} | ratio fwd/adj = {:.1}x",
        knobs.len(),
        t_fwd,
        fwd_r1,
        t_fwd / t_adj
    );
}
