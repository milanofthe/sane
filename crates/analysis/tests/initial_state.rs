//! A transient from a given state: a state that is not consistent (SPICE's
//! `uic`, every node at zero with the supplies already on) is made so before
//! the first step, and a pulse that leaves its width and period unsaid holds
//! its level, as in SPICE.

use sane_analysis::{Model, TransientOptions};

/// A supply that is on at once, over an RC: from all zeros, the transient
/// integrates, the supply at its value from the start and the output on its
/// way to it.
#[test]
fn a_start_from_zero_is_made_consistent() {
    let deck = "* uic\nV1 vdd 0 1\nR1 vdd out 1k\nC1 out 0 1n\nC2 vdd out 1p\n.end\n";
    let m = Model::from_netlist(deck).expect("model");
    let (vdd, out) = (
        m.resolve("vdd").expect("vdd"),
        m.resolve("out").expect("out"),
    );
    let t_eval: Vec<f64> = (0..=50).map(|k| k as f64 * 1e-7).collect();
    {
        let opts = TransientOptions {
            rtol: 1e-6,
            atol: 1e-9,
            x0: Some(vec![0.0; m.dim()]),
            ..Default::default()
        };
        let traj = m
            .at(&[])
            .and_then(|pt| pt.transient(&t_eval, &opts))
            .expect("transient from zeros");
        for row in traj.x.outer_iter() {
            assert!((row[vdd] - 1.0).abs() < 1e-9, "the supply is on");
        }
        // the RC charge of the output, from what C2 couples in at the step
        let (tau, t_end) = (1e3 * (1e-9 + 1e-12), t_eval[t_eval.len() - 1]);
        let v0 = 1e-12 / (1e-9 + 1e-12);
        let want = 1.0 - (1.0 - v0) * (-t_end / tau).exp();
        let got = traj.x[[t_eval.len() - 1, out]];
        assert!((got - want).abs() < 1e-3, "{got} vs {want}");
    }
}

/// `pulse(v1 v2 td tr)`: the fall as long as the rise, and the pulse held,
/// not repeated.
#[test]
fn a_pulse_without_width_holds_its_level() {
    let deck = "* pulse\nV1 in 0 pulse 0 1 1n 1n\nR1 in 0 1k\n.end\n";
    let m = Model::from_netlist(deck).expect("model");
    let i = m.resolve("in").expect("in");
    let t_eval = vec![0.0, 0.5e-9, 1.5e-9, 2.5e-9, 10e-9, 1e-6];
    let opts = TransientOptions {
        rtol: 1e-6,
        atol: 1e-9,
        ..Default::default()
    };
    let traj = m
        .at(&[])
        .and_then(|pt| pt.transient(&t_eval, &opts))
        .expect("transient");
    let v: Vec<f64> = traj.x.column(i).to_vec();
    let want = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
    for (k, (&a, &b)) in v.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-6, "point {k}: {a} vs {b}");
    }
}
