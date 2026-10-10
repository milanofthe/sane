//! The gradient of a transient objective (`Trajectory::vjp`, the cotangents
//! contracted with the forward sensitivities) against central finite
//! differences of the SAME objective: at fixed steps the trajectory's step
//! sequence does not move with the parameters, and the gradient is the
//! discrete trajectory's exactly -- on linear and nonlinear circuits,
//! nonlinear charges, the sources' time dependence, reactive parameters and
//! the operating-point start. Its own test binary: it switches the process
//! to fixed steps.

use sane_analysis::{Gradient, Model, TransientOptions};

fn options(t_end: f64) -> TransientOptions {
    TransientOptions {
        dt_max: Some(t_end / 400.0),
        ..Default::default()
    }
}

/// The gradient of `L` with `dL/dout(t_eval[k]) = w[k]`, by every parameter.
fn gradient(model: &Model, out: &str, t_eval: &[f64], w: &[f64]) -> Gradient {
    let t_end = t_eval[t_eval.len() - 1];
    model
        .at(&[])
        .and_then(|pt| pt.transient(t_eval, &options(t_end)))
        .and_then(|tr| {
            let c = ndarray::Array2::from_shape_fn((1, t_eval.len()), |(_, k)| w[k]);
            tr.vjp(&[out], c.view(), &[])
        })
        .expect("gradient")
}

/// `L(p) = sum_k w_k out(t_k)`, with a named parameter override.
fn objective(model: &Model, over: &[(&str, f64)], out: usize, t_eval: &[f64], w: &[f64]) -> f64 {
    let t_end = t_eval[t_eval.len() - 1];
    let traj = model
        .at(over)
        .and_then(|pt| pt.transient(t_eval, &options(t_end)))
        .expect("transient");
    traj.x
        .outer_iter()
        .zip(w)
        .map(|(row, wk)| wk * row[out])
        .sum()
}

fn check_against_fd(deck: &str, out: &str, t_end: f64, npts: usize) {
    sane_core::update_config(|c| c.transient_fixed_step = true);
    let model = Model::from_netlist(deck).expect("model");
    let out_idx = model.resolve(out).expect("out");
    let t_eval: Vec<f64> = (0..npts)
        .map(|k| t_end * k as f64 / (npts - 1) as f64)
        .collect();
    // deterministic non-uniform weights exercise every output point
    let w: Vec<f64> = (0..npts).map(|k| 0.5 + (k as f64 * 0.7).sin()).collect();
    let Gradient {
        params: pnames,
        grad,
    } = gradient(&model, out, &t_eval, &w);

    let bound = model.values();
    let l0 = objective(&model, &[], out_idx, &t_eval, &w);
    for (j, name) in pnames.iter().enumerate() {
        let base = bound.get(name).copied().unwrap_or(0.0);
        // relative step for live parameters, O(1) absolute step for parked
        // zeros (offsets, phases) -- big enough to rise above solver noise
        let h = if base != 0.0 { base.abs() * 1e-5 } else { 1e-6 };
        let (up, down) = (
            objective(&model, &[(name, base + h)], out_idx, &t_eval, &w),
            objective(&model, &[(name, base - h)], out_idx, &t_eval, &w),
        );
        // a parameter that moves the objective by less than its rounding
        // (a reverse-biased junction's saturation current) leaves FD
        // measuring noise: the gradient must stay below that floor too
        if (up - down).abs() < 1e-9 * l0.abs().max(1e-12) {
            assert!(
                (grad[j] * 2.0 * h).abs() < 1e-8 * l0.abs().max(1e-12),
                "{name}: gradient {} where FD sees nothing",
                grad[j]
            );
            continue;
        }
        let fd = (up - down) / (2.0 * h);
        let scale = fd.abs().max(grad[j].abs()).max(1e-9);
        assert!(
            (grad[j] - fd).abs() <= 2e-6 * scale + 1e-12,
            "{name}: gradient {} vs FD {} (rel {})",
            grad[j],
            fd,
            (grad[j] - fd).abs() / scale
        );
    }
}

#[test]
fn rc_lowpass_matches_fd() {
    let deck = "* rc\nV1 in 0 0 SIN(0 1 1k)\nR1 in out 1k\nC1 out 0 100n\n.end\n";
    check_against_fd(deck, "out", 2e-3, 80);
}

#[test]
fn diode_clipper_matches_fd() {
    // nonlinear + reactive: series R into a diode clamp with a smoothing cap,
    // DC offset so the operating point (and its sensitivity) is nontrivial
    let deck = "* clipper\n.model Dmod D(Is=1e-12 N=1)\nV1 in 0 0.3 SIN(0.3 1 1k)\nR1 in out 1k\nD1 out 0 Dmod\nC1 out 0 200n\n.end\n";
    check_against_fd(deck, "out", 2e-3, 80);
}

#[test]
fn varactor_matches_fd() {
    // a junction capacitance that moves with the voltage: the charge's
    // second derivatives enter the sensitivities' stage Jacobians
    let deck = "* varactor\n.model Dv D(Is=1e-14 Cjo=50n Vj=0.7 M=0.5)\nV1 in 0 -1 SIN(-1 0.8 2k)\nR1 in out 1k\nD1 out 0 Dv\n.end\n";
    check_against_fd(deck, "out", 1e-3, 60);
}
