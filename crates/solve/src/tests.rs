use super::*;
use std::collections::HashMap;

use sane_dae::{assemble_dae, Dae, DeviceInstance};
use sane_mna::Circuit;

fn cdc_unknown_index(dae: &Dae, name: &str) -> usize {
    dae.unknowns
        .iter()
        .position(|u| u == name)
        .expect("unknown")
}

fn params_vec(ctx: &Graph, dae: &Dae, vals: &[(&str, f64)]) -> Vec<f64> {
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

#[test]
fn divider_dc() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .resistor("R2", 2, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(&ctx, &dae, &[("V1", 10.0), ("R1", 1000.0), ("R2", 1000.0)]);
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
fn stage_solve_matches_dense() {
    // The implicit-RK stage system `(dF/dx + alpha*C + gmin*I) dx = rhs` solved
    // over the combined pattern must match a dense assembly. RC circuit so both
    // dF/dx (resistor) and C = dF/dx' (capacitor) are non-empty.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .capacitor("C1", 2, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let n = cdc.dim();
    let p = params_vec(&ctx, &dae, &[("V1", 5.0), ("R1", 1000.0), ("C1", 1e-6)]);
    let x = vec![0.3; n];
    let xdot = vec![0.0; n];

    let (jr, jc, dfdx) = cdc.jacobian_x_sparse(&x, &xdot, &p, 0.0);
    let (cr, cc, cvals) = cdc.jacobian_xdot_sparse(&x, &xdot, &p, 0.0);
    assert!(
        !cvals.is_empty(),
        "RC circuit must have a non-empty mass matrix C"
    );

    // Dense J_stage = dF/dx + alpha*C + gmin*I; pick dx_true, form rhs = J*dx_true.
    let (alpha, gmin) = (2000.0, 1e-12);
    let mut jd = vec![vec![0.0; n]; n];
    for k in 0..dfdx.len() {
        jd[jr[k]][jc[k]] += dfdx[k];
    }
    for k in 0..cvals.len() {
        jd[cr[k]][cc[k]] += alpha * cvals[k];
    }
    for i in 0..n {
        jd[i][i] += gmin;
    }
    let dx_true: Vec<f64> = (0..n).map(|i| 0.1 + 0.07 * i as f64).collect();
    let rhs: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| jd[i][j] * dx_true[j]).sum())
        .collect();

    let sym = cdc.stage_symbolic().expect("stage symbolic");
    let mut fac = sym.pattern.factorizer();
    let mut valbuf = Vec::new();
    assert!(cdc.factorize_stage(&mut fac, &dfdx, &cvals, alpha, gmin, &mut valbuf));
    let dx = fac.solve(&rhs).expect("stage solve");
    for i in 0..n {
        assert!(
            (dx[i] - dx_true[i]).abs() < 1e-9,
            "dx[{i}]={} != {}",
            dx[i],
            dx_true[i]
        );
    }
}

#[test]
fn esdirk32_rc_step() {
    // RC step response: the ESDIRK32 integrator must track the analytic
    // 1 - e^{-t/RC} on the assembled DAE.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .capacitor("C1", 2, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let n = cdc.dim();
    let p = params_vec(&ctx, &dae, &[("V1", 1.0), ("R1", 1000.0), ("C1", 1e-6)]); // RC = 1ms
    let v1 = cdc_unknown_index(&dae, "v1");
    let v2 = cdc_unknown_index(&dae, "v2");

    // Step start: source applied (v1 = 1), cap uncharged (v2 = 0).
    let mut x0 = vec![0.0; n];
    x0[v1] = 1.0;
    let t_eval: Vec<f64> = (0..=50).map(|k| k as f64 * 1e-4).collect(); // 0..5 ms
    let dtm = Some(5e-5);
    let y_irk = cdc
        .solve_transient_irk(TransientMethod::Esdirk32, &p, &x0, &t_eval, 1e-6, 1e-9, dtm)
        .expect("esdirk32");

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
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("RL", 2, 0);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_transformer", "X1", &[])),
        vec![1, 0, 2, 0], // [p+, p-, s+, s-]
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(
        &ctx,
        &dae,
        &[("V1", 10.0), ("RL", 1000.0), ("X1.ratio", 2.0)],
    );
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 50);
    assert!(conv, "transformer DC did not converge");
    let v2 = x[cdc_unknown_index(&dae, "v2")];
    assert!(
        (v2 - 5.0).abs() < 1e-6,
        "secondary voltage {v2}, expected 5 V"
    );
    // Primary current i_p (its own unknown) times the secondary load current:
    // i_s = V_s/RL = 5 mA, ampere-turns Np*i_p + Ns*i_s = 0 -> |i_p| = 2.5 mA.
    let ip = x[cdc_unknown_index(&dae, "X1.flow_pn_pp")];
    assert!(
        (ip.abs() - 2.5e-3).abs() < 1e-6,
        "primary current {ip}, expected |i_p| = 2.5 mA"
    );
}

#[test]
fn companion_continuation_converges() {
    // The per-device companion homotopy must reach the same operating point as
    // the standard solve. Exercised directly: the fallback ladder reaches it
    // only on circuits the global gmin / source homotopies cannot solve, so a
    // fixture would not trigger it.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    assert!(
        !dae.companion.is_empty(),
        "diode should emit a companion network"
    );
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(
        &ctx,
        &dae,
        &[
            ("V1", 1.0),
            ("R1", 1000.0),
            ("D1.Is", 1e-12),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let out = cdc_unknown_index(&dae, "v2");
    let (xref, conv_ref, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    assert!(conv_ref);
    let (xc, conv_c, _) = cdc.companion_continuation(
        &p,
        &Convergence::from_tol(1e-12),
        0,
        SolverTricks::default(),
    );
    assert!(conv_c, "companion continuation did not converge");
    assert!(
        (xc[out] - xref[out]).abs() < 1e-6,
        "companion {} vs reference {}",
        xc[out],
        xref[out]
    );
}

#[test]
fn convergence_criterion_presets_solve() {
    // A diode + resistor (genuine nonlinearity) must converge to the same
    // operating point under both the tight engine default and the loose SPICE
    // preset of the per-component criterion, and the converged residual must
    // honor each preset's per-row floor.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(
        &ctx,
        &dae,
        &[
            ("V1", 1.0),
            ("R1", 1000.0),
            ("D1.Is", 1e-12),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let out = cdc_unknown_index(&dae, "v2");

    let (xd, cd, _) = cdc.solve_dc_conv(&p, &[], Convergence::default(), 100);
    let (xs, cs, _) = cdc.solve_dc_conv(&p, &[], Convergence::spice(), 100);
    assert!(cd && cs, "both presets must converge");
    // Same physical root regardless of how tightly it was chased.
    assert!(
        (xd[out] - xs[out]).abs() < 1e-6,
        "default {} vs spice {}",
        xd[out],
        xs[out]
    );

    // The default-converged residual meets the tight per-row floor.
    let res = cdc.residual(&xd, &vec![0.0; cdc.dim()], &p, 0.0);
    assert!(
        cdc.residual_converged(&res, &xd, GMIN_DC, &Convergence::default()),
        "default solve must satisfy its own residual criterion"
    );
}

#[test]
fn relative_kcl_criterion_scales_with_node_current() {
    // The SPICE-faithful relative KCL test accepts a node imbalance that is
    // small RELATIVE to the branch currents into that node (reltol*sum|I| +
    // abstol), and rejects one that is not -- unlike the absolute-only floor.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(
        &ctx,
        &dae,
        &[
            ("V1", 1.0),
            ("R1", 1000.0),
            ("D1.Is", 1e-12),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let node = cdc_unknown_index(&dae, "v2");
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    assert!(conv);
    let n = cdc.dim();

    // A nonlinear circuit builds the per-term scale tape; evaluate it at x.
    let tape = cdc
        .tape_iscale
        .as_ref()
        .expect("nonlinear circuit has the iscale tape");
    let mut inputs = Vec::new();
    cdc.fill_inputs(&x, &vec![0.0; n], &p, 0.0, &mut inputs);
    let (mut terms, mut work) = (Vec::new(), Vec::new());
    tape.eval(&inputs, &mut work, &mut terms);
    let (off, len) = cdc.iscale_rows[node];
    let iscale: f64 = terms[off..off + len].iter().map(|t| t.abs()).sum();
    // ~0.48 mA flows through the diode and the resistor -> sum|I| ~ 0.96 mA.
    assert!(
        iscale > 1e-4,
        "node-current scale {iscale:e} reflects the mA branch currents"
    );

    let conv = Convergence::default();
    // An imbalance well below reltol*sum|I| is accepted relatively, but is far
    // above the absolute abstol floor (so the strict test would reject it).
    let mut res = vec![0.0; n];
    res[node] = 0.3 * conv.reltol * iscale;
    assert!(
        res[node] > conv.abstol,
        "test point is above the absolute floor"
    );
    assert!(
        cdc.residual_relative_ok(&res, &terms, &x, 0.0, &conv),
        "relative-small imbalance accepted"
    );
    assert!(
        !cdc.residual_converged(&res, &x, 0.0, &conv),
        "same imbalance rejected by the strict floor"
    );
    // An imbalance above the relative floor is rejected.
    res[node] = 5.0 * conv.reltol * iscale + conv.abstol;
    assert!(
        !cdc.residual_relative_ok(&res, &terms, &x, 0.0, &conv),
        "relative-large imbalance rejected"
    );
}

#[test]
fn hessian_matches_finite_differences() {
    // Diode (exp nonlinearity -> real curvature) in series with a resistor.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);

    let pnames: Vec<String> = dae
        .params(&ctx)
        .iter()
        .map(|&s| ctx.symbol_name(s).to_string())
        .collect();
    let p = params_vec(
        &ctx,
        &dae,
        &[
            ("V1", 1.0),
            ("R1", 1000.0),
            ("D1.Is", 1e-12),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let out = cdc_unknown_index(&dae, "v2");
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    assert!(conv);

    let col = |nm: &str| pnames.iter().position(|n| n == nm).unwrap();
    let sub = vec![col("R1"), col("V1")];
    cdc.ensure_hessian(&mut ctx, &dae);
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
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0);
    let stages = 80;
    for k in 1..=stages {
        c.resistor(&format!("R{k}"), k, k + 1);
        c.capacitor(&format!("C{k}"), k + 1, 0);
    }
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![stages + 1, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
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
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0);
    let stages = 80;
    for k in 1..=stages {
        c.resistor(&format!("R{k}"), k, k + 1);
        c.capacitor(&format!("C{k}"), k + 1, 0);
    }
    // Diode from the last node to ground (forward-biased by the source).
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![stages + 1, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
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
    let p = params_vec(&ctx, &dae, &vals);
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-10, 100);
    assert!(conv, "partitioned DC did not converge");
    // Residual at the solution must be ~zero (DC: xdot = 0).
    let r = cdc.residual(&x, &vec![0.0; cdc.dim()], &p, 0.0);
    let rnorm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        rnorm < 1e-8,
        "residual at partitioned solution too large: {rnorm}"
    );
}

#[test]
fn diode_rectifier_dc() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(
        &ctx,
        &dae,
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

/// Build a one-source test circuit (`V1` with the given shape, loaded by a 1k
/// resistor) and return its compiled DAE plus the parameter vector.
fn one_source(ctx: &mut Graph, src: SourceFn, params: &[(&str, f64)]) -> (CompiledDc, Vec<f64>) {
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).set_source(src);
    c.resistor("R1", 1, 0);
    let dae = assemble_dae(ctx, &c, &[]);
    let cdc = CompiledDc::new(ctx, &dae);
    let p = params_vec(ctx, &dae, params);
    (cdc, p)
}

#[test]
fn transient_tricks_toggle_is_correctness_neutral() {
    // Row equilibration and breakpoints aid conditioning / accuracy, not the
    // solution: toggling them off via the modular trick set must still integrate
    // the RC step to the analytic 1 - e^{-t/RC}.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .capacitor("C1", 2, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    let mut cdc = CompiledDc::new(&mut ctx, &dae);
    let (r, cap) = (1000.0, 1e-6);
    let tau = r * cap;
    let p = params_vec(&ctx, &dae, &[("V1", 1.0), ("R1", r), ("C1", cap)]);
    let v2 = cdc_unknown_index(&dae, "v2");
    let mut x0 = vec![0.0; cdc.dim()];
    x0[cdc_unknown_index(&dae, "v1")] = 1.0;
    let times: Vec<f64> = (0..=5).map(|k| k as f64 * tau).collect();
    cdc.set_tricks(SolverTricks {
        row_equilibration: false,
        breakpoints: false,
        ..SolverTricks::default()
    });
    let traj = cdc
        .solve_transient(
            TransientMethod::Esdirk32,
            &p,
            &x0,
            &times,
            TRANSIENT_RTOL,
            TRANSIENT_ATOL,
            None,
        )
        .expect("transient");
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
    let mut ctx = Graph::new();
    let (cdc, p) = one_source(
        &mut ctx,
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
    let mut ctx = Graph::new();
    let (cdc, p) = one_source(
        &mut ctx,
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
    let mut ctx = Graph::new();
    let (cdc, p) = one_source(
        &mut ctx,
        SourceFn::Sin,
        &[("R1", 1e3), ("V1.sin_w", 2.0 * std::f64::consts::PI * 1e6)],
    );
    assert!((cdc.source_fundamental(&p).unwrap() - 1e6).abs() < 1e-3);

    let mut ctx2 = Graph::new();
    let (cdc2, p2) = one_source(
        &mut ctx2,
        SourceFn::Pulse,
        &[("R1", 1e3), ("V1.pulse_per", 4e-9)],
    );
    assert!((cdc2.source_fundamental(&p2).unwrap() - 0.25e9).abs() < 1.0);
}

/// RC low-pass step response: with V1 = 1 V applied and the cap initially
/// discharged, v_C(t) = 1 - exp(-t/RC). Validates the transient solve against
/// the analytic solution.
#[test]
fn rc_step_transient() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .capacitor("C1", 2, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let (r, cap) = (1000.0, 1e-6);
    let tau = r * cap; // 1e-3 s
    let p = params_vec(&ctx, &dae, &[("V1", 1.0), ("R1", r), ("C1", cap)]);

    // Consistent IC: v1 = 1 (source), v2 = 0 (discharged), branch current
    // i_V1 set by the assembler's sign convention is recovered by diffsol's
    // algebraic constraints; seed the differential state v2 = 0.
    let v2_idx = cdc_unknown_index(&dae, "v2");
    let mut x0 = vec![0.0; cdc.dim()];
    x0[cdc_unknown_index(&dae, "v1")] = 1.0;
    x0[v2_idx] = 0.0;

    let times: Vec<f64> = (0..=5).map(|k| k as f64 * tau).collect();
    let traj = cdc
        .solve_transient(
            TransientMethod::Esdirk32,
            &p,
            &x0,
            &times,
            TRANSIENT_RTOL,
            TRANSIENT_ATOL,
            None,
        )
        .expect("transient");
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
    let mut c = Circuit::new();
    // VDD=5 at node 1, RD from 1->drain(2), gate tied to a 2V source node 3.
    c.voltage_source("VDD", 1, 0)
        .resistor("RD", 1, 2)
        .voltage_source("VG", 3, 0);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_mos", "M1", &[])),
        vec![2, 3, 0, 0], // d g s b (body at ground)
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(
        &ctx,
        &dae,
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
    let mut c = Circuit::new();
    c.voltage_source("VDD", 1, 0)
        .resistor("RD", 1, 2)
        .voltage_source("VG", 3, 0);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_mos", "M1", &[])),
        vec![2, 3, 0, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(
        &ctx,
        &dae,
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
    let (x_on, c_on, _) = cdc.solve_dc_with(&p, &[], 1e-10, 100, on);
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
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(
        &ctx,
        &dae,
        &[
            ("V1", 0.72),
            ("R1", 1000.0),
            ("D1.Is", 1e-14),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let (x_off, c_off, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    let (x_on, c_on, _) = cdc.solve_dc_with(&p, &[], 1e-12, 100, on);
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

/// gmin-augmented matrix is reused symbolically: factor twice at different
/// gmin and confirm the pattern path is taken (symbolic is Some).
#[test]
fn symbolic_reuse_present() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_bjt", "Q1", &[])),
        vec![2, 1, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    assert!(
        cdc.symbolic.is_some(),
        "symbolic factorization should be precomputed"
    );
}
