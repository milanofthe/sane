//! Discrete transient adjoint vs. central finite differences of the SAME
//! fixed-grid ESDIRK32 objective: the adjoint gradient must match FD to
//! truncation accuracy on every parameter, linear and nonlinear circuits,
//! including reactive parameters (C) and the operating-point IC term.

use sane_analysis::Model;

/// J(p) = sum_k w_k * x_out(t_k) over the fixed-grid trajectory on `t_eval`, with a
/// named parameter override (immune to parameter-vector ordering).
fn objective(
    model: &Model,
    over: &[(&str, f64)],
    t_eval: &[f64],
    out_idx: usize,
    w: &[f64],
) -> f64 {
    let p = model.pvec(over);
    let traj = model
        .cdc()
        .solve_transient_grid(&p, &[], t_eval, &[])
        .expect("BE forward");
    traj.iter().zip(w).map(|(row, wk)| wk * row[out_idx]).sum()
}

fn check_adjoint_vs_fd(deck: &str, out: &str, t_end: f64, npts: usize) {
    let model = Model::from_netlist(deck).expect("model");
    let n = model.dim();
    let out_idx = model.resolve(out).expect("out");
    let t_eval: Vec<f64> = (0..npts)
        .map(|k| t_end * k as f64 / (npts - 1) as f64)
        .collect();
    // deterministic non-uniform weights exercise every time step
    let w: Vec<f64> = (0..npts).map(|k| 0.5 + (k as f64 * 0.7).sin()).collect();
    let cot: Vec<Vec<f64>> = w
        .iter()
        .map(|wk| {
            let mut c = vec![0.0; n];
            c[out_idx] = *wk;
            c
        })
        .collect();

    let (pnames, grad) = model
        .transient_adjoint(t_eval.clone(), cot, None, None)
        .expect("adjoint");

    let bound = model.values();
    for (j, name) in pnames.iter().enumerate() {
        let base = bound.get(name).copied().unwrap_or(0.0);
        // relative step for live parameters, O(1) absolute step for parked
        // zeros (offsets, phases) -- big enough to rise above solver noise
        let h = if base != 0.0 { base.abs() * 1e-5 } else { 1e-6 };
        let fd = (objective(&model, &[(name, base + h)], &t_eval, out_idx, &w)
            - objective(&model, &[(name, base - h)], &t_eval, out_idx, &w))
            / (2.0 * h);
        let scale = fd.abs().max(grad[j].abs()).max(1e-9);
        assert!(
            (grad[j] - fd).abs() <= 2e-4 * scale + 1e-12,
            "{name}: adjoint {} vs FD {} (rel {})",
            grad[j],
            fd,
            (grad[j] - fd).abs() / scale
        );
    }
}

#[test]
fn rc_lowpass_matches_fd() {
    let deck = "* rc\nV1 in 0 0 SIN(0 1 1k)\nR1 in out 1k\nC1 out 0 100n\n.end\n";
    check_adjoint_vs_fd(deck, "out", 2e-3, 80);
}

#[test]
fn diode_clipper_matches_fd() {
    // nonlinear + reactive: series R into a diode clamp with a smoothing cap,
    // DC offset so the operating point (and its adjoint IC term) is nontrivial
    let deck = "* clipper\n.model Dmod D(Is=1e-12 N=1)\nV1 in 0 0.3 SIN(0.3 1 1k)\nR1 in out 1k\nD1 out 0 Dmod\nC1 out 0 200n\n.end\n";
    check_adjoint_vs_fd(deck, "out", 2e-3, 80);
}

#[test]
fn adjoint_matches_forward_sensitivity_on_fine_grid() {
    // cross-method: the discrete BE gradient approaches the exact
    // (ESDIRK/forward-AD) gradient of the same functional as the grid refines
    let deck = "* rc\nV1 in 0 0 SIN(0 1 1k)\nR1 in out 1k\nC1 out 0 100n\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    let n = model.dim();
    let out_idx = model.resolve("out").expect("out");
    let npts = 2000;
    let t_eval: Vec<f64> = (0..npts)
        .map(|k| 2e-3 * k as f64 / (npts - 1) as f64)
        .collect();
    let w: Vec<f64> = (0..npts).map(|k| 0.5 + (k as f64 * 0.7).sin()).collect();
    let cot: Vec<Vec<f64>> = w
        .iter()
        .map(|wk| {
            let mut c = vec![0.0; n];
            c[out_idx] = *wk;
            c
        })
        .collect();
    let (pnames, grad) = model
        .transient_adjoint(t_eval.clone(), cot, None, None)
        .expect("adjoint");

    let knobs: Vec<String> = pnames
        .iter()
        .filter(|q| !q.contains('.'))
        .cloned()
        .collect();
    let (nn, traj) = model
        .transient_sensitivity(knobs.clone(), t_eval.clone(), 1e-7, 1e-10, None)
        .expect("forward");
    for (ki, name) in knobs.iter().enumerate() {
        let fwd: f64 = traj
            .iter()
            .zip(&w)
            .map(|(row, wk)| wk * row[nn * (ki + 1) + out_idx])
            .sum();
        let j = pnames.iter().position(|q| q == name).unwrap();
        let scale = fwd.abs().max(grad[j].abs()).max(1e-9);
        // different discretizations of the same continuous gradient: expect
        // first-order (BE) agreement on a fine grid
        assert!(
            (grad[j] - fwd).abs() <= 2e-2 * scale,
            "{name}: adjoint {} vs forward {} (rel {})",
            grad[j],
            fwd,
            (grad[j] - fwd).abs() / scale
        );
    }
}
