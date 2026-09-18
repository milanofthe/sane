//! Trapezoidal integrator (`TransientMethod::Trap`): accuracy against the
//! analytic RC solution, cross-method parity with ESDIRK32 on a nonlinear
//! circuit, and the transport-delay path.

use sane_analysis::Model;
use sane_solve::TransientMethod;

fn run(deck: &str, method: TransientMethod, t_eval: &[f64]) -> Vec<Vec<f64>> {
    let m = Model::from_netlist(deck).expect("model");
    let p = m.pvec(&[]);
    m.solve_transient(method, p, t_eval.to_vec(), None, 1e-5, 1e-9, None)
        .expect("transient")
}

/// RC step response against the analytic `V0·(1 − e^{−t/RC})` (source ramps in
/// 1 ns, so compare from well past the corner using the exact shifted form).
#[test]
fn trap_matches_analytic_rc() {
    let deck = "* rc\nV1 in 0 pulse 0 1 0 1n 1n 1 2\nR1 in out 1k\nC1 out 0 1u\n.end\n";
    let m = Model::from_netlist(deck).expect("model");
    let p = m.pvec(&[]);
    let t_eval: Vec<f64> = (0..=200).map(|i| i as f64 * 5e-6).collect();
    let rows = m
        .solve_transient(
            TransientMethod::Trap,
            p,
            t_eval.clone(),
            None,
            1e-5,
            1e-9,
            None,
        )
        .expect("transient");
    let iout = m.resolve("out").expect("out");
    let rc = 1e3 * 1e-6;
    let mut max_err = 0.0f64;
    for (k, &t) in t_eval.iter().enumerate() {
        if t < 10e-9 {
            continue; // inside the source ramp
        }
        // For t >> the 1 ns ramp the response is a step to within tr/(2·RC)
        // ~ 5e-7, far below the assertion tolerance.
        let step_exact = 1.0 - (-t / rc).exp();
        max_err = max_err.max((rows[k][iout] - step_exact).abs());
    }
    assert!(max_err < 2e-4, "trap vs analytic RC: max err {max_err}");
}

/// Nonlinear accuracy: a diode clipper driven by a sine, trap at rtol 1e-5
/// against a tightly-converged ESDIRK32 reference (rtol 1e-9). The deviation
/// must sit in the requested tolerance envelope and shrink with rtol.
#[test]
fn trap_tracks_reference_on_diode_clipper() {
    let deck = "* clipper\nV1 in 0 sin(0 2 1k)\nR1 in out 1k\nD1 out 0 dclip\n.model dclip d is=1e-14\nC1 out 0 10n\n.end\n";
    let m = Model::from_netlist(deck).expect("model");
    let iout = m.resolve("out").expect("out");
    let t_eval: Vec<f64> = (0..=400).map(|i| i as f64 * 5e-6).collect();
    let tight = m
        .solve_transient(
            TransientMethod::Esdirk32,
            m.pvec(&[]),
            t_eval.clone(),
            None,
            1e-9,
            1e-12,
            None,
        )
        .expect("reference");
    let dev_at = |rtol: f64, atol: f64| {
        let rows = m
            .solve_transient(
                TransientMethod::Trap,
                m.pvec(&[]),
                t_eval.clone(),
                None,
                rtol,
                atol,
                None,
            )
            .expect("trap");
        (0..t_eval.len())
            .map(|k| (rows[k][iout] - tight[k][iout]).abs())
            .fold(0.0f64, f64::max)
    };
    let d5 = dev_at(1e-5, 1e-9);
    let d6 = dev_at(1e-6, 1e-10);
    assert!(d5 < 1e-2, "trap rtol 1e-5 vs reference: {d5}");
    assert!(d6 < 2e-3, "trap rtol 1e-6 vs reference: {d6}");
    assert!(d6 < d5, "error must shrink with rtol: {d6} vs {d5}");
}

/// Transport delay (`absdelay` history path) under trap: a delayed echo through
/// a behavioral source must arrive intact (parity with ESDIRK32).
#[test]
fn trap_handles_transport_delay() {
    let deck = "* delay line\n\
.veriloga\n\
module vdel(a, b);\n\
  inout a, b; electrical a, b;\n\
  analog V(b) <+ absdelay(V(a), 2u);\n\
endmodule\n\
.endveriloga\n\
V1 in 0 pulse 0 1 1u 10n 10n 5u 20u\n\
R1 in 0 1k\n\
N1 in out vdel\n\
R2 out 0 1k\n\
.end\n";
    let t_eval: Vec<f64> = (0..=300).map(|i| i as f64 * 5e-8).collect();
    let a = run(deck, TransientMethod::Trap, &t_eval);
    let b = run(deck, TransientMethod::Esdirk32, &t_eval);
    let m = Model::from_netlist(deck).expect("model");
    let iout = m.resolve("out").expect("out");
    let mut max_dev = 0.0f64;
    for k in 0..t_eval.len() {
        max_dev = max_dev.max((a[k][iout] - b[k][iout]).abs());
    }
    assert!(max_dev < 5e-3, "trap vs esdirk absdelay: max dev {max_dev}");
    // the delayed edge actually arrives: out rises past 0.5 only after 3 us
    let rise = t_eval
        .iter()
        .zip(a.iter())
        .find(|(_, r)| r[iout] > 0.5)
        .map(|(t, _)| *t);
    assert!(rise.is_some_and(|t| t > 2.9e-6), "delayed edge at {rise:?}");
}
