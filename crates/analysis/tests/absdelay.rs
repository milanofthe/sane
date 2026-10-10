//! Verilog-A `absdelay`: a delayed pulse edge, a delayed sine against the
//! analytic shift, and the template-clone path (two instances of the same
//! module must each get their own history slot).

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

const DLINE: &str = "\
.veriloga
module dline(in, out);
  inout in, out; electrical in, out;
  parameter real td = 1e-9;
  analog V(out) <+ absdelay(V(in), td);
endmodule
.endveriloga
";

/// A pulse edge at 2 ns re-appears at 2 ns + td, undistorted.
#[test]
fn delayed_pulse_edge() {
    let deck = format!(
        "* absdelay pulse\n{DLINE}V1 in 0 PULSE(0 1 2n 100p 100p 1 2)\nN1 in out dline td=5n\nR1 out 0 1k\n.end\n"
    );
    let model = Model::from_netlist(&deck).expect("model");
    let out = model.resolve("out").expect("out");
    let (t, rows) = tran(&model, 15e-9, 300);
    for (tk, row) in t.iter().zip(&rows) {
        let vo = row[out];
        // edge midpoints excluded (finite 100 ps ramp)
        let expect = if *tk < 7.0e-9 { 0.0 } else { 1.0 };
        if (*tk - 7.05e-9).abs() > 0.2e-9 {
            assert!(
                (vo - expect).abs() < 2e-3,
                "t={tk:.3e}: out={vo} expect={expect}"
            );
        }
    }
}

/// A delayed sine matches sin(2 pi f (t - td)) once the wave has arrived.
#[test]
fn delayed_sine_matches_shift() {
    let deck = format!(
        "* absdelay sine\n{DLINE}V1 in 0 SIN(0 1 100Meg)\nN1 in out dline td=3n\nR1 out 0 1k\n.end\n"
    );
    let model = Model::from_netlist(&deck).expect("model");
    let out = model.resolve("out").expect("out");
    let (t, rows) = tran(&model, 30e-9, 300);
    let f = 100e6;
    let td = 3e-9;
    for (tk, row) in t.iter().zip(&rows) {
        let expect = if *tk < td {
            0.0
        } else {
            (2.0 * std::f64::consts::PI * f * (tk - td)).sin()
        };
        assert!(
            (row[out] - expect).abs() < 5e-3,
            "t={tk:.3e}: out={} expect={expect}",
            row[out]
        );
    }
}

/// Two instances with identical parameters share a lowering template; the clone
/// must re-mint its own history input (cascade of two td=2.5n = one td=5n).
#[test]
fn template_clone_cascade() {
    let deck = format!(
        "* absdelay cascade\n{DLINE}V1 in 0 PULSE(0 1 0 100p 100p 1 2)\nN1 in mid dline td=2.5n\nN2 mid out dline td=2.5n\nR1 out 0 1k\n.end\n"
    );
    let model = Model::from_netlist(&deck).expect("model");
    let out = model.resolve("out").expect("out");
    let mid = model.resolve("mid").expect("mid");
    let (t, rows) = tran(&model, 12e-9, 240);
    for (tk, row) in t.iter().zip(&rows) {
        let e_mid = if *tk < 2.55e-9 { 0.0 } else { 1.0 };
        let e_out = if *tk < 5.05e-9 { 0.0 } else { 1.0 };
        if (*tk - 2.6e-9).abs() > 0.2e-9 {
            assert!(
                (row[mid] - e_mid).abs() < 2e-3,
                "t={tk:.3e}: mid={}",
                row[mid]
            );
        }
        if (*tk - 5.1e-9).abs() > 0.2e-9 {
            assert!(
                (row[out] - e_out).abs() < 2e-3,
                "t={tk:.3e}: out={}",
                row[out]
            );
        }
    }
}

/// DC treats the delay as transparent: out follows in exactly.
#[test]
fn dc_is_transparent() {
    let deck = format!("* dc\n{DLINE}V1 in 0 2.5\nN1 in out dline td=1n\nR1 out 0 1k\n.end\n");
    let model = Model::from_netlist(&deck).expect("model");
    let op = model
        .at(&[])
        .and_then(|pt| pt.operating_point())
        .expect("dc");
    let out = model.resolve("out").expect("out");
    assert!(
        (op.vector()[out] - 2.5).abs() < 1e-6,
        "out = {}",
        op.vector()[out]
    );
}

/// A delayed sine straight onto a capacitor: the delay's output branch and
/// the capacitor close an index-2 loop, whose current `C dV/dt` jumps where
/// the wave arrives. The integration passes the kink and carries the
/// current, the delayed signal's rate off its history, within a few per
/// mille.
#[test]
fn delayed_sine_onto_a_capacitor() {
    let deck = format!(
        "* absdelay index 2\n{DLINE}V1 in 0 SIN(0 1 100k)\nR1 in 0 1k\nN1 in out dline td=2u\n\
         R2 out 0 1k\nC2 out 0 1n\n.end\n"
    );
    let model = Model::from_netlist(&deck).expect("model");
    let out = model.resolve("out").expect("out");
    let branch = (model.unknowns().iter())
        .position(|u| u.starts_with("N1.flow"))
        .expect("the delay's output branch");
    let (w, td, c, r) = (2.0 * std::f64::consts::PI * 1e5, 2e-6, 1e-9, 1e3);
    let t: Vec<f64> = (0..=120).map(|k| 6e-6 * k as f64 / 120.0).collect();
    let (tol_v, tol_i) = (1e-4, 5e-3);
    {
        let opts = TransientOptions {
            dt_max: Some(1e-7),
            ..Default::default()
        };
        let run = model.at(&[]).and_then(|pt| pt.transient(&t, &opts));
        let x = run.unwrap_or_else(|e| panic!("{e}")).x;
        for (k, &tk) in t.iter().enumerate() {
            let s = (tk - td).max(0.0);
            let v = (w * s).sin();
            assert!(
                (x[[k, out]] - v).abs() < tol_v,
                "t={tk:.3e}: out={}",
                x[[k, out]]
            );
            // the branch current, past the arrival's transition
            if tk > td + 1e-7 {
                let i = -(c * w * (w * s).cos() + v / r);
                let e = (x[[k, branch]] - i).abs() / (c * w);
                assert!(e < tol_i, "t={tk:.3e}: i={} expect {i}", x[[k, branch]]);
            }
        }
    }
}
