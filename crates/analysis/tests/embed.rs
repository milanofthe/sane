//! Embeddability acceptance: drive SANE entirely from Rust via `Model`, with no
//! Python in the loop. If this passes, the analysis API is usable as a library.

use sane_analysis::{Model, TransientOptions};

#[test]
fn operating_point_read_set_override() {
    let sim = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").expect("build sim");

    // Solve and read a node by name.
    let op = sim
        .at(&[])
        .and_then(|pt| pt.operating_point())
        .expect("solve");
    assert!(
        (op.get("out").unwrap() - 2.5).abs() < 1e-6,
        "out = {:?}",
        op.get("out")
    );

    // Persistent mutation of a named parameter is picked up by the next solve.
    sim.set("R2", 3e3).unwrap();
    let op = sim
        .at(&[])
        .and_then(|pt| pt.operating_point())
        .expect("solve");
    assert!(
        (op.get("out").unwrap() - 5.0 * 3e3 / 4e3).abs() < 1e-6,
        "out = {:?}",
        op.get("out")
    );

    // Per-call override is non-destructive (store still holds R2 = 3k).
    let op = sim
        .at(&[("R2", 1e3)])
        .and_then(|pt| pt.operating_point())
        .expect("solve");
    assert!((op.get("out").unwrap() - 2.5).abs() < 1e-6);
    assert!((sim.get("R2").unwrap() - 3e3).abs() < 1e-9);

    // reset() restores the netlist default.
    sim.reset();
    assert!((sim.get("R2").unwrap() - 1e3).abs() < 1e-9);
}

#[test]
fn sensitivity_off_the_solved_point() {
    let sim = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").unwrap();
    let op = sim.at(&[]).unwrap().operating_point().unwrap();
    let s = op.sensitivity(&["out"], &[]).unwrap();
    assert_eq!(s.grad.dim(), (1, s.params.len()));
    assert!((s.values[0] - 2.5).abs() < 1e-6);
    // d v(out)/d R2 > 0 (raising R2 raises the divider output).
    let d = s.get("out", "R2").expect("R2 in params");
    assert!(d > 0.0, "dV/dR2 = {d}");
    // `wrt` picks the columns.
    let only = op.sensitivity(&["out"], &["R2"]).unwrap();
    assert_eq!(only.params, vec!["R2".to_string()]);
    assert_eq!(only.grad[[0, 0]], d);
}

#[test]
fn hessian_off_the_solved_point() {
    let sim = Model::from_netlist(
        "V1 in 0 0.72\nR1 in out 1k\nD1 out 0 dm\n.model dm D(Is=1e-14 N=1 Vt=0.025852)\n.end",
    )
    .unwrap();
    let op = sim.at(&[]).unwrap().operating_point().unwrap();
    let h = op.hessian(&["out"], &["R1"]).unwrap().h;
    assert_eq!(h.dim(), (1, 1, 1));
    assert!(h[[0, 0, 0]].is_finite());
}

/// A Hessian over several parameters solves its sensitivities as one
/// multi-right-hand-side block; its diagonal matches the single-parameter
/// Hessians.
#[test]
fn hessian_over_several_parameters() {
    let sim = Model::from_netlist(
        "V1 in 0 0.72
R1 in mid 1k
R2 mid out 500
D1 out 0 dm
.model dm D(Is=1e-14 N=1 Vt=0.025852)
.end",
    )
    .unwrap();
    let op = sim.at(&[]).unwrap().operating_point().unwrap();
    let h = op.hessian(&["out"], &["R1", "R2"]).unwrap().h;
    assert_eq!(h.dim(), (1, 2, 2));
    for (k, name) in ["R1", "R2"].iter().enumerate() {
        let single = op.hessian(&["out"], &[name]).unwrap().h[[0, 0, 0]];
        assert!(
            (h[[0, k, k]] - single).abs() <= 1e-9 * single.abs().max(1e-30),
            "{name}: {} vs {single}",
            h[[0, k, k]]
        );
    }
    assert!((h[[0, 0, 1]] - h[[0, 1, 0]]).abs() <= 1e-9 * h[[0, 0, 1]].abs().max(1e-30));
}

#[test]
fn transient_and_noise() {
    let sim = Model::from_netlist("V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.end").unwrap();
    let t: Vec<f64> = (0..=20).map(|k| k as f64 * 5e-3 / 20.0).collect();
    let traj = sim
        .at(&[])
        .unwrap()
        .transient(&t, &TransientOptions::default())
        .unwrap();
    let out = traj.signal("out").unwrap();
    assert_eq!(out.len(), t.len());
    // RC charge toward 1 V.
    assert!(
        out[out.len() - 1] > 0.9,
        "v(out) end = {}",
        out[out.len() - 1]
    );

    let freqs = sane_analysis::log_grid(1.0, 1e6, 10);
    let ns = sim.at(&[]).unwrap().noise(&["out"], &freqs).unwrap();
    assert_eq!(ns.psd.dim(), (1, freqs.len()));
    assert!(ns.psd.iter().all(|&v| v >= 0.0));
}

#[test]
fn ac_poles_state_space() {
    let sim = Model::from_netlist("V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end").unwrap();
    // First-order RC: -3 dB near f = 1/(2*pi*RC) ~ 159 Hz, rolling off above.
    let ac = (sim.at(&[]).unwrap())
        .ac("V1", &["out"], &sane_analysis::log_grid(1.0, 1e5, 50))
        .unwrap();
    assert_eq!(ac.freqs.len(), 50);
    let mag_db = ac.mag_db("out").unwrap();
    assert!(mag_db[0] > mag_db[49], "should roll off");

    // One real pole at s = -1/(RC).
    let pt = sim.at(&[]).unwrap();
    let poles = pt.poles().unwrap();
    assert!(!poles.poles.is_empty());
    let p = poles.poles[0];
    assert!(
        (p.re + 1.0 / (1e3 * 1e-6)).abs() < 1.0,
        "pole re = {}",
        p.re
    );
    // d s / d R1 = 1/(R1^2 C1) for s = -1/(R1 C1).
    let g = poles.sensitivity(&["R1"]).unwrap();
    assert_eq!(g.params, ["R1"]);
    let want = 1.0 / (1e3 * 1e3 * 1e-6);
    assert!(
        (g.grad[[0, 0]].re - want).abs() < 1e-6 * want,
        "ds/dR1 = {}",
        g.grad[[0, 0]]
    );

    let ss = pt.state_space(&["V1"], &["out"]).unwrap();
    assert_eq!(ss.e.dim(), (sim.dim(), sim.dim()));
}

#[test]
fn gradient_family_pure_rust() {
    // Every sensitivity / gradient analysis is reachable from Rust, with no
    // Python in the loop: one point, its operating point solved once and
    // shared by every analysis at it.
    let sim = Model::from_netlist("V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end").unwrap();
    let pt = sim.at(&[]).unwrap();

    // Poles: one real RC pole at -1/(RC).
    let poles = pt.poles().unwrap();
    assert!(!poles.poles.is_empty());
    let s = poles.poles[0];
    assert!((s.re + 1.0 / (1e3 * 1e-6)).abs() < 1.0, "pole = {s}");

    // All-parameter pole gradient (the adjoint, exact AD) -- one row per pole.
    let pg = poles.sensitivity(&[]).unwrap();
    assert_eq!(pg.grad.nrows(), poles.poles.len());
    assert!(pg.params.iter().any(|n| n == "R1"));

    // AC response at the corner, at the same operating point.
    let ac = pt.ac("V1", &["out"], &[159.0]).unwrap();
    assert_eq!(ac.h.dim(), (1, 1));
}

#[test]
fn temp_sweep_runs() {
    let sim = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").unwrap();
    let kelvin = [273.15, 298.15, 323.15, 348.15, 373.15];
    let ts = sim.at(&[]).unwrap().dc_sweep("$temp", &kelvin).unwrap();
    assert_eq!(ts.signal("out").unwrap().len(), kelvin.len());
}

#[test]
fn harmonic_balance_diode() {
    let sim = Model::from_netlist(
        "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 dm\n.model dm D(Is=1e-14 N=1 Vt=0.025852)\n.end",
    )
    .unwrap();
    let opts = sane_analysis::HbOptions {
        f0: Some(1000.0),
        harmonics: 5,
        ..Default::default()
    };
    let hb = sim.at(&[]).unwrap().harmonic_balance(&opts).unwrap();
    let mag = hb.magnitude("mid").expect("mid spectrum");
    assert_eq!(mag.len(), 6); // DC + 5 harmonics
                              // Rectifying nonlinearity puts energy at the fundamental.
    assert!(mag[1] > 0.0);
}

#[test]
fn hierarchical_param_navigation() {
    let sim = Model::from_netlist(
        ".subckt rcdiv a b\nR1 a m 1k\nR2 m b 2k\n.ends\nV1 in 0 5\nX1 in 0 rcdiv\n.end",
    )
    .unwrap();
    assert!(sim.is_group("X1"));
    assert!(sim.is_param("X1.R1"));
    assert!((sim.get("X1.R2").unwrap() - 2e3).abs() < 1e-9);
    assert!(sim.children("X1").contains(&"R1".to_string()));
}

#[test]
fn fold_single_param_keeps_result_drops_param() {
    let m = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").unwrap();
    let v0 = m
        .at(&[])
        .and_then(|pt| pt.operating_point())
        .unwrap()
        .get("out")
        .unwrap();
    let n0 = m.params().len();
    let folded = m.fold(&["R2"]).unwrap();
    let v1 = folded
        .at(&[])
        .and_then(|pt| pt.operating_point())
        .unwrap()
        .get("out")
        .unwrap();
    assert!((v0 - v1).abs() < 1e-9, "fold changed OP: {v0} vs {v1}");
    assert!(
        !folded.params().iter().any(|p| p == "R2"),
        "R2 still a param"
    );
    assert_eq!(folded.params().len(), n0 - 1);
    assert!(
        m.params().iter().any(|p| p == "R2"),
        "master must be unchanged"
    );
}

#[test]
fn fold_group_folds_all_children() {
    let m = Model::from_netlist(
        ".subckt rcdiv a b\nR1 a m 1k\nR2 m b 2k\n.ends\nV1 in 0 5\nX1 in 0 rcdiv\n.end",
    )
    .unwrap();
    let folded = m.fold(&["X1"]).unwrap();
    assert!(
        !folded.params().iter().any(|p| p.starts_with("X1.")),
        "X1.* should be folded away"
    );
    assert!(m.is_param("X1.R1"), "master must be unchanged");
}

#[test]
fn get_set_accept_dotted_paths() {
    // Functional, path-based mutation must accept dotted hierarchical names,
    // consistent with `fold("X1.R2")`.
    let m = Model::from_netlist(
        ".subckt rcdiv a b\nR1 a m 1k\nR2 m b 2k\n.ends\nV1 in 0 5\nX1 in 0 rcdiv\n.end",
    )
    .unwrap();
    assert!((m.get("X1.R2").unwrap() - 2e3).abs() < 1e-9, "dotted get");
    m.set("X1.R2", 5e3).unwrap();
    assert!(
        (m.get("X1.R2").unwrap() - 5e3).abs() < 1e-9,
        "dotted set round-trips"
    );
    assert!(m.set("X1", 1.0).is_err(), "a group is not a settable leaf");
}

/// A point is the model at one binding: a name that is no parameter is an
/// error, the binding overrides the bound values without changing them, and
/// one sensitivity covers several outputs.
#[test]
fn a_point_is_the_model_at_one_binding() {
    let sim = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").unwrap();
    assert!(matches!(
        sim.at(&[("R9", 1.0)]),
        Err(sane_analysis::ModelError::UnknownParam(_))
    ));
    let pt = sim.at(&[("R2", 3e3)]).unwrap();
    assert_eq!(pt.values()["R2"], 3e3);
    assert_eq!(sim.get("R2"), Some(1e3));
    let op = pt.operating_point().unwrap();
    let out = op.get("out").unwrap();
    assert!((out - 3.75).abs() < 1e-6, "out = {out}");
    let s = op.sensitivity(&["out", "in"], &["R1", "R2"]).unwrap();
    assert_eq!(s.outputs, vec!["out".to_string(), "in".to_string()]);
    assert_eq!(s.params, vec!["R1".to_string(), "R2".to_string()]);
    // the source pins `in`: no parameter moves it
    assert!(s.of("in").unwrap().iter().all(|g| g.abs() < 1e-12));
    assert!(s.get("out", "R1").unwrap() < 0.0);
}

/// One AC analysis answers several outputs; its sensitivities are the
/// analytic ones, and its adjoint agrees with them.
#[test]
fn ac_of_several_outputs_with_its_derivatives() {
    use num_complex::Complex64;
    let sim = Model::from_netlist("V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end").unwrap();
    let freqs = [10.0, 159.0, 1e3];
    let ac = sim
        .at(&[])
        .unwrap()
        .ac("V1", &["out", "in"], &freqs)
        .unwrap();
    let (r, c) = (1e3, 1e-6);
    for (k, &f) in freqs.iter().enumerate() {
        let jwc = Complex64::new(0.0, 2.0 * std::f64::consts::PI * f * c);
        let want = 1.0 / (1.0 + jwc * r);
        assert!((ac.of("out").unwrap()[k] - want).norm() < 1e-9);
        assert!((ac.of("in").unwrap()[k] - 1.0).norm() < 1e-12);
    }
    let s = ac.sensitivity(&["R1"]).unwrap();
    assert_eq!(s.params, vec!["R1".to_string()]);
    for (k, &f) in freqs.iter().enumerate() {
        let jwc = Complex64::new(0.0, 2.0 * std::f64::consts::PI * f * c);
        let want = -jwc / ((1.0 + jwc * r) * (1.0 + jwc * r));
        let got = s.grad[[0, k, 0]];
        assert!((got - want).norm() < 1e-6 * want.norm(), "{got} vs {want}");
    }
    // L = Re h_out(f1): the adjoint's d/dR1 is Re(dH/dR1) there.
    let mut cot = ndarray::Array2::zeros((2, freqs.len()));
    cot[[0, 1]] = Complex64::new(1.0, 0.0);
    let g = ac.vjp(cot.view()).unwrap();
    let j = g.params.iter().position(|p| p == "R1").unwrap();
    let want = s.grad[[0, 1, 0]].re;
    assert!((g.grad[j] - want).abs() < 1e-9 * want.abs());
}

/// The S-parameters of a matched through connection: no reflection, full
/// transmission.
#[test]
fn s_parameters_of_a_matched_through() {
    let sim = Model::from_netlist("P1 a 0 Z0=50\nP2 b 0 Z0=50\nR1 a b 1m\n.end").unwrap();
    let sp = sim.at(&[]).unwrap().s_parameters(&[1e6]).unwrap();
    assert_eq!(sp.port_names.len(), 2);
    let s = sp.s.index_axis(ndarray::Axis(0), 0);
    assert!(s[[0, 0]].norm() < 1e-4 && s[[1, 1]].norm() < 1e-4, "{s:?}");
    assert!((s[[1, 0]].norm() - 1.0).abs() < 1e-4 && (s[[0, 1]].norm() - 1.0).abs() < 1e-4);
    let g = sp.sensitivity(&["R1"]).unwrap();
    assert_eq!(g.params, vec!["R1".to_string()]);
}
