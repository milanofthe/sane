//! Transients through index-2 loops: a source directly across a capacitor
//! carries the current `C dV/dt`, which jumps at every kink of the source.
//! The integration controls the capacitor's voltage, not that current (the
//! voltage's rate), and passes the kinks.

use sane_analysis::{Model, TransientOptions};

#[test]
fn a_kinked_source_across_a_capacitor_passes() {
    // ramps of 1 V/us up, flat, down: the capacitor current steps between
    // -1 mA, 0 and +1 mA (the source's current flows out of its + node)
    let deck = "* index 2\nV1 a 0 pwl(0 0 1u 1 2u 1 3u 0)\nC1 a 0 1n\nR1 a 0 1meg\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    let a = model.resolve("a").expect("a");
    let i = (model.unknowns().iter())
        .position(|u| u.contains("V1"))
        .expect("the source's current");
    let t: Vec<f64> = (0..=80).map(|k| 4e-6 * k as f64 / 80.0).collect();
    {
        let opts = TransientOptions::default();
        let run = model.at(&[]).and_then(|pt| pt.transient(&t, &opts));
        let x = run.unwrap_or_else(|e| panic!("{e}")).x;
        for (k, &tk) in t.iter().enumerate() {
            let (v, slope) = match tk {
                t if t <= 1e-6 => (t * 1e6, 1e6),
                t if t <= 2e-6 => (1.0, 0.0),
                t if t <= 3e-6 => (3.0 - t * 1e6, -1e6),
                _ => (0.0, 0.0),
            };
            assert!((x[[k, a]] - v).abs() < 1e-6, "t={tk:.3e}: v={}", x[[k, a]]);
            // away from the kinks, the current is the capacitor's plus the
            // resistor's
            if [0.0, 1e-6, 2e-6, 3e-6]
                .iter()
                .all(|&tb| (tk - tb).abs() > 1e-8)
            {
                let expect = -(1e-9 * slope + v / 1e6);
                assert!(
                    (x[[k, i]] - expect).abs() < 1e-6,
                    "t={tk:.3e}: i={} expect {expect}",
                    x[[k, i]]
                );
            }
        }
    }
}
