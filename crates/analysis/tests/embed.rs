//! Embeddability acceptance: drive SANE entirely from Rust via `Model`, with no
//! Python in the loop. If this passes, the analysis API is usable as a library.

use sane_analysis::Model;

#[test]
fn operating_point_read_set_override() {
    let sim = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").expect("build sim");

    // Solve and read a node by name.
    let op = sim.operating_point(&[]).expect("solve");
    assert!(
        (op.get("out").unwrap() - 2.5).abs() < 1e-6,
        "out = {:?}",
        op.get("out")
    );

    // Persistent mutation of a named parameter is picked up by the next solve.
    sim.set("R2", 3e3).unwrap();
    let op = sim.operating_point(&[]).expect("solve");
    assert!(
        (op.get("out").unwrap() - 5.0 * 3e3 / 4e3).abs() < 1e-6,
        "out = {:?}",
        op.get("out")
    );

    // Per-call override is non-destructive (store still holds R2 = 3k).
    let op = sim.operating_point(&[("R2", 1e3)]).expect("solve");
    assert!((op.get("out").unwrap() - 2.5).abs() < 1e-6);
    assert!((sim.get("R2").unwrap() - 3e3).abs() < 1e-9);

    // reset() restores the netlist default.
    sim.reset();
    assert!((sim.get("R2").unwrap() - 1e3).abs() < 1e-9);
}

#[test]
fn sensitivity_off_the_solved_point() {
    let sim = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").unwrap();
    let op = sim.operating_point(&[]).unwrap();
    let s = op.sensitivity("out").unwrap();
    assert_eq!(s.names.len(), s.grad.len());
    assert!((s.value - 2.5).abs() < 1e-6);
    // d v(out)/d R2 > 0 (raising R2 raises the divider output).
    let i = s
        .names
        .iter()
        .position(|n| n == "R2")
        .expect("R2 in params");
    assert!(s.grad[i] > 0.0, "dV/dR2 = {}", s.grad[i]);
}

#[test]
fn hessian_off_the_solved_point() {
    let sim = Model::from_netlist(
        "V1 in 0 0.72\nR1 in out 1k\nD1 out 0 dm\n.model dm D(Is=1e-14 N=1 Vt=0.025852)\n.end",
    )
    .unwrap();
    let op = sim.operating_point(&[]).unwrap();
    let h = op.hessian("out", &["R1"]).unwrap();
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].len(), 1);
    assert!(h[0][0].is_finite());
}

#[test]
fn transient_and_noise() {
    let sim = Model::from_netlist("V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.end").unwrap();
    let t: Vec<f64> = (0..=20).map(|k| k as f64 * 5e-3 / 20.0).collect();
    let traj = sim
        .transient(sane_solve::TransientMethod::Esdirk32, &[], &t, 1e-4, 1e-7)
        .unwrap();
    let out = traj.signal("out").unwrap();
    assert_eq!(out.len(), t.len());
    // RC charge toward 1 V.
    assert!(
        out[out.len() - 1] > 0.9,
        "v(out) end = {}",
        out[out.len() - 1]
    );

    let ns = sim.noise(&[], "out", 1.0, 1e6, 10).unwrap();
    assert_eq!(ns.freqs.len(), ns.psd.len());
    assert!(ns.psd.iter().all(|&v| v >= 0.0));
}

#[test]
fn ac_poles_state_space() {
    let sim = Model::from_netlist("V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end").unwrap();
    // First-order RC: -3 dB near f = 1/(2*pi*RC) ~ 159 Hz, rolling off above.
    let ac = sim.ac(&[], "V1", "out", 1.0, 1e5, 50).unwrap();
    assert_eq!(ac.freqs.len(), 50);
    assert!(ac.mag_db[0] > ac.mag_db[49], "should roll off");

    // One real pole at s = -1/(RC).
    let pz = sim.poles_zeros(&[], "V1", "out").unwrap();
    assert!(!pz.poles.is_empty());
    let p = pz.poles[0];
    assert!(
        (p[0] + 1.0 / (1e3 * 1e-6)).abs() < 1.0,
        "pole re = {}",
        p[0]
    );

    let ss = sim.state_space(&[], "V1", "out").unwrap();
    assert_eq!(ss.e.len(), sim.dim());
}

#[test]
fn gradient_family_pure_rust() {
    // Every sensitivity / gradient analysis is reachable from Rust, with no
    // Python in the loop: solve the OP once, reuse (x, p) across analyses.
    let sim = Model::from_netlist("V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end").unwrap();
    let op = sim.operating_point(&[]).unwrap();
    let x = op.vector().to_vec();
    let p = sim.pvec(&[]);

    // Poles: one real RC pole at -1/(RC).
    let poles = sim.poles(x.clone(), p.clone()).unwrap();
    assert!(!poles.is_empty());
    assert!(
        (poles[0].0 + 1.0 / (1e3 * 1e-6)).abs() < 1.0,
        "pole = {:?}",
        poles[0]
    );

    // All-parameter pole gradient (the adjoint, exact AD) -- one entry per pole.
    let pg = sim.pole_gradient(x.clone(), p.clone()).unwrap();
    assert_eq!(pg.len(), poles.len());
    assert!(pg[0].1.iter().any(|(n, _, _)| n == "R1"));

    // AC response at the corner reuses the same operating point.
    let out_idx = sim.resolve("out").unwrap();
    let h = sim
        .ac_response("V1", out_idx, x.clone(), p.clone(), vec![159.0])
        .unwrap();
    assert_eq!(h.len(), 1);
}

#[test]
fn temp_sweep_runs() {
    let sim = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").unwrap();
    let ts = sim.temp_sweep(&[], "out", 0.0, 100.0, 5).unwrap();
    assert_eq!(ts.temps.len(), ts.values.len());
}

#[test]
fn harmonic_balance_diode() {
    let sim = Model::from_netlist(
        "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 dm\n.model dm D(Is=1e-14 N=1 Vt=0.025852)\n.end",
    )
    .unwrap();
    let hb = sim.harmonic_balance(&[], 1000.0, 5, None).unwrap();
    assert!(hb.converged, "HB should converge");
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
    let v0 = m.operating_point(&[]).unwrap().get("out").unwrap();
    let n0 = m.params().len();
    let folded = m.fold(&["R2"]).unwrap();
    let v1 = folded.operating_point(&[]).unwrap().get("out").unwrap();
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
