//! Rodas4 at fixed steps: its order of convergence on an RC driven by a sine
//! (the sources' time rate in every stage, the source's current an algebraic
//! unknown) and on a reverse-biased junction (a nonlinear charge). Its own
//! test binary: it switches the process to fixed steps.

use sane_analysis::{Model, TransientOptions};

const DECKS: [&str; 2] = [
    "* rc\nV1 in 0 sin(0 2 1k)\nR1 in out 1k\nC1 out 0 100n\n.end\n",
    "* varactor\nV1 in 0 sin(-2 1 1k)\nR1 in out 1k\nD1 out 0 dv\n\
     .model dv d is=1e-14 cjo=100n vj=0.7 m=0.5\n.end\n",
];
const SPAN: f64 = 4e-4;

/// `out` at the end of the span after `steps` fixed steps.
fn end_value(m: &Model, steps: usize) -> f64 {
    let opts = TransientOptions {
        dt_max: Some(SPAN / steps as f64),
        ..Default::default()
    };
    let traj = m
        .at(&[])
        .and_then(|pt| pt.transient(&[0.0, SPAN], &opts))
        .expect("transient");
    traj.x.outer_iter().last().unwrap()[m.resolve("out").expect("out")]
}

#[test]
fn rodas4_converges_at_its_order() {
    sane_core::update_config(|c| c.transient_fixed_step = true);
    for deck in DECKS {
        let m = Model::from_netlist(deck).expect("model");
        let reference = end_value(&m, 12_800);
        let err = |steps| (end_value(&m, steps) - reference).abs();
        let (e1, e2, e3) = (err(25), err(50), err(100));
        let rates = [(e1 / e2).log2(), (e2 / e3).log2()];
        assert!(
            rates.iter().all(|r| *r > 4.0 - 0.3),
            "errors {e1:.2e} {e2:.2e} {e3:.2e}, observed orders {rates:?}"
        );
    }
}
