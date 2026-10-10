//! Ideal transmission line (method of characteristics): DC transparency,
//! matched single-step propagation, the classic mismatch reflection
//! staircase against the analytic bounce diagram, and the delay guards.

use sane_analysis::{Model, TransientOptions};

fn tran(model: &Model, tstop: f64, npts: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let t: Vec<f64> = (0..npts)
        .map(|k| tstop * k as f64 / (npts - 1) as f64)
        .collect();
    let opts = TransientOptions {
        rtol: 1e-6,
        atol: 1e-9,
        ..Default::default()
    };
    let rows = model
        .at(&[])
        .and_then(|pt| pt.transient(&t, &opts))
        .expect("transient")
        .x
        .outer_iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    (t, rows)
}

/// At DC the lossless line is transparent: v1 = v2, i1 = -i2.
#[test]
fn dc_is_transparent() {
    let deck = "* dc\nV1 in 0 5\nT1 in 0 out 0 Z0=50 TD=1n\nR1 out 0 100\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    let op = model
        .at(&[])
        .and_then(|pt| pt.operating_point())
        .expect("dc");
    let out = model.resolve("out").expect("out");
    let inn = model.resolve("in").expect("in");
    assert!(
        (op.vector()[out] - 5.0).abs() < 1e-6,
        "out = {}",
        op.vector()[out]
    );
    assert!((op.vector()[inn] - 5.0).abs() < 1e-9);
    // Through-current: 5 V over 100 ohm. The port branch currents are the
    // Verilog-A module's probed flows, named in canonical node order.
    let i1 = model.resolve("T1.flow_n1_p1").expect("i1");
    let i2 = model.resolve("T1.flow_n2_p2").expect("i2");
    assert!(
        (op.vector()[i1].abs() - 0.05).abs() < 1e-6,
        "|i1| = {}",
        op.vector()[i1].abs()
    );
    assert!(
        (op.vector()[i1] + op.vector()[i2]).abs() < 1e-6,
        "i1 + i2 = {}",
        op.vector()[i1] + op.vector()[i2]
    );
}

/// Matched source, open end: the incident half-step doubles at the open end
/// (arrives at TD), the reflection is absorbed at the matched source -- one
/// clean step, no ringing.
#[test]
fn matched_source_open_end_single_step() {
    let deck =
        "* step\nV1 s 0 PULSE(0 1 0 50p 50p 1 2)\nR1 s in 50\nT1 in 0 out 0 Z0=50 TD=5n\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    let out = model.resolve("out").expect("out");
    let inn = model.resolve("in").expect("in");
    let (t, rows) = tran(&model, 40e-9, 400);
    for (tk, row) in t.iter().zip(&rows) {
        let vo = row[out];
        let expect = if *tk < 5e-9 { 0.0 } else { 1.0 };
        if (*tk - 5e-9).abs() > 0.3e-9 {
            assert!(
                (vo - expect).abs() < 2e-3,
                "t={tk:.3e}: out={vo} expect={expect}"
            );
        }
        // the launch side steps to 0.5 immediately and stays (reflection
        // returns at 2 TD and is absorbed without changing the voltage)
        if *tk > 0.3e-9 && (*tk - 10e-9).abs() > 0.3e-9 {
            let vi = row[inn];
            let expect_in = if *tk < 10e-9 { 0.5 } else { 1.0 };
            assert!(
                (vi - expect_in).abs() < 2e-3,
                "t={tk:.3e}: in={vi} expect={expect_in}"
            );
        }
    }
}

/// Mismatched source (Rs = 3 Z0), open end: the bounce diagram gives the
/// staircase 0.5, 0.75, 0.875, ... at the open end (arrivals at TD, 3 TD,
/// 5 TD), converging to the DC value 1.
#[test]
fn mismatch_staircase_matches_bounce_diagram() {
    let deck =
        "* stair\nV1 s 0 PULSE(0 1 0 50p 50p 1 2)\nR1 s in 150\nT1 in 0 out 0 Z0=50 TD=5n\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    let out = model.resolve("out").expect("out");
    let (t, rows) = tran(&model, 50e-9, 500);
    // plateau midpoints between arrivals at (2k+1) TD
    let plateaus = [
        (7.5e-9, 0.5),
        (12.5e-9, 0.5),
        (17.5e-9, 0.75),
        (22.5e-9, 0.75),
        (27.5e-9, 0.875),
        (37.5e-9, 0.9375),
        (47.5e-9, 0.96875),
    ];
    for (tm, expect) in plateaus {
        let idx = t.iter().position(|tk| *tk >= tm).unwrap();
        let vo = rows[idx][out];
        assert!(
            (vo - expect).abs() < 5e-3,
            "t={tm:.2e}: out={vo} expect={expect}"
        );
    }
}

/// Analyses without delay support must reject the circuit loudly (AC is
/// supported exactly; PZ stays out because a delay makes the characteristic
/// equation transcendental -- infinitely many roots, not an eigenproblem).
#[test]
fn unsupported_analyses_error_cleanly() {
    let deck = "* g\nV1 in 0 0 SIN(0 1 1k)\nT1 in 0 out 0 Z0=50 TD=1n\nR1 out 0 50\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    let op = model
        .at(&[])
        .and_then(|pt| pt.operating_point())
        .expect("dc works");

    let hb = sane_analysis::HbOptions {
        f0: Some(1e3),
        harmonics: 4,
        ..Default::default()
    };
    assert!(model.at(&[]).unwrap().harmonic_balance(&hb).is_err());
    assert!(model.at(&[]).unwrap().poles().is_err());
    // AC and noise ARE supported (exact e^{-j w tau}); see tests/ac_delay.rs
    // and `noise_through_a_matched_line`
    assert!(model.at(&[]).unwrap().ac("V1", &["out"], &[1e3]).is_ok());
    assert!(model.at(&[]).unwrap().noise(&["out"], &[1e3]).is_ok());
    assert!(op.sensitivity(&["out"], &[]).is_err());
    assert!(model
        .at(&[])
        .and_then(|pt| pt.transient(&[0.0, 1e-9], &TransientOptions::default()))
        .and_then(|tr| tr.sensitivity(&["out"], &["R1"]))
        .is_err());
    assert!(model
        .at(&[])
        .and_then(|pt| pt.transient(&[0.0, 1e-9], &TransientOptions::default()))
        .and_then(|tr| tr.vjp(&["out"], ndarray::Array2::zeros((1, 2)).view(), &["R1"]))
        .is_err());
}

/// Noise travels a matched lossless line unchanged in power: the source
/// resistor's and the load's noise at the far end is what it is at a direct
/// connection, at every frequency (the line only adds phase).
#[test]
fn noise_through_a_matched_line() {
    let psd = |deck: &str| {
        let model = Model::from_netlist(deck).expect("model");
        let freqs = [1e6, 1.3e8, 7.7e8, 2e9];
        let n = model
            .at(&[])
            .unwrap()
            .noise(&["out"], &freqs)
            .expect("noise");
        n.psd.row(0).to_vec()
    };
    let line = psd("V1 s 0 0\nR1 s in 50\nT1 in 0 out 0 Z0=50 TD=1n\nR2 out 0 50\n");
    let direct = psd("V1 s 0 0\nR1 s out 50\nR2 out 0 50\n");
    for (a, b) in line.iter().zip(&direct) {
        assert!((a - b).abs() <= 1e-9 * b, "{a} vs {b}");
    }
}
