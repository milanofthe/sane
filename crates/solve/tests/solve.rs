//! The solver end to end on small systems: DC, transient, breakpoints,
//! continuation, the Hessian, partitioning, limiting.

use sane_circuit::SourceFn;
use sane_core::constants::{TRANSIENT_ATOL, TRANSIENT_RTOL};
use sane_core::Graph;
use sane_solve::*;
use std::collections::HashMap;

use sane_circuit::Elements;
use sane_dae::{Dae, DeviceInstance};

fn cdc_unknown_index(dae: &Dae, name: &str) -> usize {
    dae.find(name).expect("unknown")
}

fn params_vec(ctx: &mut Graph, dae: &Dae, vals: &[(&str, f64)]) -> Vec<f64> {
    let map: HashMap<&str, f64> = vals.iter().copied().collect();
    // Native devices read the global `$temp` and per-instance Tnom/Eg/XTI;
    // default them to nominal (so temperature scalings are identities) unless
    // the test overrides them, instead of the bare 0 that would zero V_T.
    let tnom = sane_core::constants::TEMP_NOMINAL_K;
    dae.params(ctx)
        .iter()
        .map(|&s| {
            let name = ctx.symbol_name(s);
            if let Some(v) = map.get(name) {
                return *v;
            }
            if name == sane_core::constants::TEMP_SYMBOL {
                tnom
            } else if name.ends_with(".Tnom") {
                tnom
            } else if name.ends_with(".Eg") {
                1.11
            } else if name.ends_with(".XTI") {
                3.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Build a one-source test circuit (`V1` with the given shape, loaded by a 1k
/// resistor) and return its compiled DAE plus the parameter vector.
fn one_source(src: SourceFn, params: &[(&str, f64)]) -> (CompiledDc, Vec<f64>) {
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0).set_source(src);
    c.resistor("R1", 1, 0);
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &[])).unwrap();
    let p = params_vec(&mut ctx, &dae, params);
    (CompiledDc::new(ctx, dae), p)
}

#[test]
fn divider_dc() {
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .resistor("R2", 2, 0);
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &[])).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[("V1", 10.0), ("R1", 1000.0), ("R2", 1000.0)],
    );
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 50);
    assert!(conv);
    // Tolerance accommodates the baseline GMIN_DC node-to-ground shunt
    // (ngspice keeps the same ~1e-12 gmin, shifting v2 by ~2.5e-9).
    assert!(
        (x[0] - 10.0).abs() < 1e-6 && (x[1] - 5.0).abs() < 1e-6,
        "{x:?}"
    );
}

#[test]
fn transient_rc_step() {
    // RC step response: the transient must track the analytic
    // 1 - e^{-t/RC} on the assembled DAE.
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .capacitor("C1", 2, 0);
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &[])).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let n = cdc.dim();
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[("V1", 1.0), ("R1", 1000.0), ("C1", 1e-6)],
    ); // RC = 1ms
    let v1 = cdc_unknown_index(cdc.dae(), "v1");
    let v2 = cdc_unknown_index(cdc.dae(), "v2");

    // Step start: source applied (v1 = 1), cap uncharged (v2 = 0).
    let mut x0 = vec![0.0; n];
    x0[v1] = 1.0;
    let t_eval: Vec<f64> = (0..=50).map(|k| k as f64 * 1e-4).collect(); // 0..5 ms
    let dtm = Some(5e-5);
    let y_irk = cdc
        .solve_transient(&p, &x0, &t_eval, 1e-6, 1e-9, dtm)
        .expect("transient")
        .rows;

    let analytic = 1.0 - (-5.0f64).exp();
    assert!(
        (y_irk[t_eval.len() - 1][v2] - analytic).abs() < 1e-2,
        "final v2 = {} (analytic {analytic})",
        y_irk[t_eval.len() - 1][v2]
    );
}

#[test]
fn ideal_transformer_steps_voltage() {
    // 2:1 step-down: V1 = 10 V on the primary, ratio Np/Ns = 2, so the
    // secondary (loaded by RL) sits at V_p/2 = 5 V. Power is conserved by the
    // ampere-turn balance, so the primary draws 2x the secondary current.
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0).resistor("RL", 2, 0);
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_transformer", "X1", &[])),
        vec![1, 0, 2, 0], // [p+, p-, s+, s-]
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[("V1", 10.0), ("RL", 1000.0), ("X1.ratio", 2.0)],
    );
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 50);
    assert!(conv, "transformer DC did not converge");
    let v2 = x[cdc_unknown_index(cdc.dae(), "v2")];
    assert!(
        (v2 - 5.0).abs() < 1e-6,
        "secondary voltage {v2}, expected 5 V"
    );
    // Primary current i_p (its own unknown) times the secondary load current:
    // i_s = V_s/RL = 5 mA, ampere-turns Np*i_p + Ns*i_s = 0 -> |i_p| = 2.5 mA.
    let ip = x[cdc_unknown_index(cdc.dae(), "X1.flow_pn_pp")];
    assert!(
        (ip.abs() - 2.5e-3).abs() < 1e-6,
        "primary current {ip}, expected |i_p| = 2.5 mA"
    );
}

#[test]
fn hessian_matches_finite_differences() {
    // Diode (exp nonlinearity -> real curvature) in series with a resistor.
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    let cdc = CompiledDc::new(ctx, dae);

    let pnames = cdc.param_names().to_vec();
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[
            ("V1", 1.0),
            ("R1", 1000.0),
            ("D1.Is", 1e-12),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let out = cdc_unknown_index(cdc.dae(), "v2");
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    assert!(conv);

    let col = |nm: &str| pnames.iter().position(|n| n == nm).unwrap();
    let sub = vec![col("R1"), col("V1")];
    let h = cdc.hessian(out, &sub, &x, &p, 0.0);

    // Central-difference Hessian oracle on y = solve_dc(p)[out].
    let metric = |ci: usize, cj: usize, di: f64, dj: f64| -> f64 {
        let mut q = p.clone();
        q[ci] += di;
        q[cj] += dj;
        cdc.solve_dc(&q, &[], 1e-12, 100).0[out]
    };
    for a in 0..2 {
        for b in 0..2 {
            let (ci, cj) = (sub[a], sub[b]);
            let (hi, hj) = (1e-4 * p[ci].abs(), 1e-4 * p[cj].abs());
            let fd = (metric(ci, cj, hi, hj) - metric(ci, cj, hi, -hj) - metric(ci, cj, -hi, hj)
                + metric(ci, cj, -hi, -hj))
                / (4.0 * hi * hj);
            let rel = (h[a][b] - fd).abs() / (fd.abs() + 1e-9);
            assert!(rel < 5e-3, "H[{a}][{b}] ad={} fd={fd} rel={rel}", h[a][b]);
        }
    }
}

#[test]
fn partition_isolates_nonlinear_block() {
    // A long RC ladder (linear) with a single diode at the far end: only the
    // diode-touched nodes should land in the nonlinear block V; the rest of
    // the ladder is the linear block L.
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0);
    let stages = 80;
    for k in 1..=stages {
        c.resistor(&format!("R{k}"), k, k + 1);
        c.capacitor(&format!("C{k}"), k + 1, 0);
    }
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![stages + 1, 0],
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let (nl, nv) = cdc.partition_sizes().expect("should partition");
    // Only the diode anode node varies; everything else is linear.
    assert!(nv <= 2, "nonlinear block too large: {nv}");
    assert!(nl + nv == cdc.dim());
    assert!(nl > stages, "linear block should hold the ladder: {nl}");
}

#[test]
fn partitioned_solve_matches_residual() {
    // A long RC ladder (linear block) with a forward diode at the end so the
    // nonlinear block is exercised; the partition (Schur) path is active
    // (dim >> 64, |V| small). The solution must satisfy F(x) ~= 0 -- a
    // method-independent check that the cached-factorization Schur solve is
    // numerically equivalent to a full LU.
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0);
    let stages = 80;
    for k in 1..=stages {
        c.resistor(&format!("R{k}"), k, k + 1);
        c.capacitor(&format!("C{k}"), k + 1, 0);
    }
    // Diode from the last node to ground (forward-biased by the source).
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![stages + 1, 0],
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    assert!(cdc.partition_sizes().is_some(), "should be partitioned");
    let mut vals = vec![
        ("V1", 5.0),
        ("D1.Is", 1e-14),
        ("D1.N", 1.0),
        ("D1.Vt", 0.025852),
    ];
    for k in 1..=stages {
        vals.push((Box::leak(format!("R{k}").into_boxed_str()) as &str, 1000.0));
        vals.push((Box::leak(format!("C{k}").into_boxed_str()) as &str, 1e-9));
    }
    let p = params_vec(&mut cdc.graph(), cdc.dae(), &vals);
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-10, 100);
    assert!(conv, "partitioned DC did not converge");
    // The currents at the solution must be ~zero (DC).
    let r = cdc.currents(&x, &p, 0.0);
    let rnorm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        rnorm < 1e-8,
        "residual at partitioned solution too large: {rnorm}"
    );
}

#[test]
fn diode_rectifier_dc() {
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[
            ("V1", 0.72),
            ("R1", 1000.0),
            ("D1.Is", 1e-14),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    assert!(conv, "diode Newton did not converge: {x:?}");
    assert!((x[1] - 0.6).abs() < 0.05, "v_out={}", x[1]);
}

#[test]
fn transient_tricks_toggle_is_correctness_neutral() {
    // Row equilibration and breakpoints aid conditioning / accuracy, not the
    // solution: toggling them off via the modular trick set must still integrate
    // the RC step to the analytic 1 - e^{-t/RC}.
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .capacitor("C1", 2, 0);
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &[])).unwrap();
    let mut cdc = CompiledDc::new(ctx, dae);
    let (r, cap) = (1000.0, 1e-6);
    let tau = r * cap;
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[("V1", 1.0), ("R1", r), ("C1", cap)],
    );
    let v2 = cdc_unknown_index(cdc.dae(), "v2");
    let mut x0 = vec![0.0; cdc.dim()];
    x0[cdc_unknown_index(cdc.dae(), "v1")] = 1.0;
    let times: Vec<f64> = (0..=5).map(|k| k as f64 * tau).collect();
    cdc.set_tricks(SolverTricks {
        row_equilibration: false,
        breakpoints: false,
        ..SolverTricks::default()
    });
    let traj = cdc
        .solve_transient(&p, &x0, &times, TRANSIENT_RTOL, TRANSIENT_ATOL, None)
        .expect("transient")
        .rows;
    let want = 1.0 - (-1.0f64).exp(); // at t = tau
    assert!(
        (traj[1][v2] - want).abs() < 5e-3,
        "tricks-off v_C={}",
        traj[1][v2]
    );
}

#[test]
fn pulse_breakpoints_match_edges() {
    // PULSE(0 5 td=1n tr=1n tf=1n pw=5n per=10n): per-period corners at
    // td + {0, tr, tr+pw, tr+pw+tf} = {1,2,7,8} ns, repeating every 10 ns.
    let (cdc, p) = one_source(
        SourceFn::Pulse,
        &[
            ("R1", 1e3),
            ("V1.pulse_v1", 0.0),
            ("V1.pulse_v2", 5.0),
            ("V1.pulse_td", 1e-9),
            ("V1.pulse_tr", 1e-9),
            ("V1.pulse_tf", 1e-9),
            ("V1.pulse_pw", 5e-9),
            ("V1.pulse_per", 10e-9),
        ],
    );
    let bps = cdc.transient_breakpoints(&p, 0.0, 25e-9);
    let want: Vec<f64> = [1, 2, 7, 8, 11, 12, 17, 18, 21, 22]
        .iter()
        .map(|&k| k as f64 * 1e-9)
        .collect();
    assert_eq!(bps.len(), want.len(), "got {bps:?}");
    for (g, w) in bps.iter().zip(&want) {
        assert!((g - w).abs() < 1e-15, "breakpoint {g:e} != {w:e}");
    }
}

#[test]
fn pwl_breakpoints_at_knots() {
    // PWL knots at 1ms, 2ms (t=0 is excluded as the integration start).
    let (cdc, p) = one_source(
        SourceFn::Pwl(3),
        &[
            ("R1", 1e3),
            ("V1.pwl_t0", 0.0),
            ("V1.pwl_v0", 0.0),
            ("V1.pwl_t1", 1e-3),
            ("V1.pwl_v1", 1.0),
            ("V1.pwl_t2", 2e-3),
            ("V1.pwl_v2", 0.0),
        ],
    );
    let bps = cdc.transient_breakpoints(&p, 0.0, 5e-3);
    assert_eq!(bps.len(), 2);
    assert!(
        (bps[0] - 1e-3).abs() < 1e-15 && (bps[1] - 2e-3).abs() < 1e-15,
        "got {bps:?}"
    );
}

#[test]
fn fundamental_reads_sin_and_pulse() {
    let (cdc, p) = one_source(
        SourceFn::Sin,
        &[("R1", 1e3), ("V1.sin_w", 2.0 * std::f64::consts::PI * 1e6)],
    );
    assert!((cdc.source_fundamental(&p).unwrap() - 1e6).abs() < 1e-3);

    let (cdc2, p2) = one_source(SourceFn::Pulse, &[("R1", 1e3), ("V1.pulse_per", 4e-9)]);
    assert!((cdc2.source_fundamental(&p2).unwrap() - 0.25e9).abs() < 1.0);
}

/// RC low-pass step response: with V1 = 1 V applied and the cap initially
/// discharged, v_C(t) = 1 - exp(-t/RC). Validates the transient solve against
/// the analytic solution.
#[test]
fn rc_step_transient() {
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .capacitor("C1", 2, 0);
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &[])).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let (r, cap) = (1000.0, 1e-6);
    let tau = r * cap; // 1e-3 s
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[("V1", 1.0), ("R1", r), ("C1", cap)],
    );

    // Consistent IC: v1 = 1 (source), v2 = 0 (discharged), branch current
    // i_V1 set by the assembler's sign convention is recovered by diffsol's
    // algebraic constraints; seed the differential state v2 = 0.
    let v2_idx = cdc_unknown_index(cdc.dae(), "v2");
    let mut x0 = vec![0.0; cdc.dim()];
    x0[cdc_unknown_index(cdc.dae(), "v1")] = 1.0;
    x0[v2_idx] = 0.0;

    let times: Vec<f64> = (0..=5).map(|k| k as f64 * tau).collect();
    let traj = cdc
        .solve_transient(&p, &x0, &times, TRANSIENT_RTOL, TRANSIENT_ATOL, None)
        .expect("transient")
        .rows;
    for (k, t) in times.iter().enumerate() {
        let want = 1.0 - (-t / tau).exp();
        let got = traj[k][v2_idx];
        assert!((got - want).abs() < 5e-3, "t={t}: v_C={got}, want {want}");
    }
}

/// A common-source MOSFET stage: needs gmin homotopy to converge from x=0.
#[test]
fn mosfet_cs_gmin_homotopy() {
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    // VDD=5 at node 1, RD from 1->drain(2), gate tied to a 2V source node 3.
    c.voltage_source("VDD", 1, 0)
        .resistor("RD", 1, 2)
        .voltage_source("VG", 3, 0);
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_mos", "M1", &[])),
        vec![2, 3, 0, 0], // d g s b (body at ground)
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[
            ("VDD", 5.0),
            ("RD", 1000.0),
            ("VG", 2.0),
            ("M1.Vth", 1.0),
            ("M1.Kp", 2e-3),
            ("M1.W", 1.0),
            ("M1.L", 1.0),
            ("M1.lambda", 0.0),
            ("M1.gamma", 0.0),
            ("M1.phi", 0.6),
            ("M1.theta", 0.0),
            ("M1.eta", 0.0),
        ],
    );
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-10, 100);
    assert!(conv, "MOSFET CS did not converge: {x:?}");
    // Drain voltage must sit between 0 and VDD.
    let vd = x[1];
    assert!(vd > 0.0 && vd < 5.0, "v_drain out of range: {vd}");
}

/// Device limiting is *path-only*: the `device_limiting` trick must reach a
/// bit-close operating point to the default solve, never a shifted one. It
/// only reshapes the Newton iteration. Checked on a MOSFET CS stage (fetlim)
/// and a diode rectifier (pnjlim).
#[test]
fn device_limiting_preserves_operating_point() {
    // MOSFET common-source (channel control -> fetlim).
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("VDD", 1, 0)
        .resistor("RD", 1, 2)
        .voltage_source("VG", 3, 0);
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_mos", "M1", &[])),
        vec![2, 3, 0, 0],
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[
            ("VDD", 5.0),
            ("RD", 1000.0),
            ("VG", 2.0),
            ("M1.Vth", 1.0),
            ("M1.Kp", 2e-3),
            ("M1.W", 1.0),
            ("M1.L", 1.0),
            ("M1.lambda", 0.0),
            ("M1.gamma", 0.0),
            ("M1.phi", 0.6),
            ("M1.theta", 0.0),
            ("M1.eta", 0.0),
        ],
    );
    let on = SolverTricks {
        device_limiting: true,
        ..SolverTricks::default()
    };
    let (x_off, c_off, _) = cdc.solve_dc(&p, &[], 1e-10, 100);
    let (x_on, c_on, _) = cdc.solve_dc_conv_with(&p, &[], Convergence::from_tol(1e-10), 100, on);
    assert!(
        c_off && c_on,
        "MOSFET CS did not converge (off={c_off}, on={c_on})"
    );
    for i in 0..x_off.len() {
        assert!(
            (x_off[i] - x_on[i]).abs() < 1e-7,
            "fetlim shifted unknown {i}: {} vs {}",
            x_off[i],
            x_on[i]
        );
    }

    // Diode rectifier (forward junction -> pnjlim).
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let p = params_vec(
        &mut cdc.graph(),
        cdc.dae(),
        &[
            ("V1", 0.72),
            ("R1", 1000.0),
            ("D1.Is", 1e-14),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let (x_off, c_off, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    let (x_on, c_on, _) = cdc.solve_dc_conv_with(&p, &[], Convergence::from_tol(1e-12), 100, on);
    assert!(
        c_off && c_on,
        "diode did not converge (off={c_off}, on={c_on})"
    );
    for i in 0..x_off.len() {
        assert!(
            (x_off[i] - x_on[i]).abs() < 1e-9,
            "pnjlim shifted unknown {i}: {} vs {}",
            x_off[i],
            x_on[i]
        );
    }
}
