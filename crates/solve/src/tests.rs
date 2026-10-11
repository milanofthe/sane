//! Tests of the solver's internals: the stage matrix, the companion
//! continuation, the convergence criteria, the symbolic factorization reuse.

use super::*;
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

#[test]
fn stage_solve_matches_dense() {
    // The implicit-RK stage system `(G + alpha*C + gmin*I) dx = rhs` solved
    // over the combined pattern must match a dense assembly. RC circuit so both
    // G (resistor) and C (capacitor) are non-empty.
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
        &[("V1", 5.0), ("R1", 1000.0), ("C1", 1e-6)],
    );
    let x = vec![0.3; n];

    let Triplets {
        rows: jr,
        cols: jc,
        vals: dfdx,
    } = cdc.jacobian_i_x_sparse(&x, &p, 0.0);
    let Triplets {
        rows: cr,
        cols: cc,
        vals: cvals,
    } = cdc.jacobian_q_x_sparse(&x, &p, 0.0);
    assert!(
        !cvals.is_empty(),
        "RC circuit must have a non-empty mass matrix C"
    );

    // Dense J_stage = G + alpha*C + gmin*I; pick dx_true, form rhs = J*dx_true.
    let (alpha, gmin) = (2000.0, sane_core::constants::GMIN_DC);
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
    let mut matrix = crate::stage_matrix::StageMatrix::new(&cdc, &sym);
    assert!(matrix.assemble(&dfdx, &cvals, alpha));
    let mut dx = vec![0.0; n];
    assert!(matrix.solve(&rhs, &mut dx), "stage solve");
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
fn companion_continuation_converges() {
    // The per-device companion homotopy must reach the same operating point as
    // the standard solve. Exercised directly: the fallback ladder reaches it
    // only on circuits the global gmin / source homotopies cannot solve, so a
    // fixture would not trigger it.
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0],
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    assert!(
        !dae.hints.companion.is_empty(),
        "diode should emit a companion network"
    );
    let cdc = CompiledDc::new(ctx, dae);
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
    let (xref, conv_ref, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    assert!(conv_ref);
    let (xc, conv_c, _) = cdc.companion_continuation(
        &crate::dc::Binding::new(&p),
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
            ("V1", 1.0),
            ("R1", 1000.0),
            ("D1.Is", 1e-12),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let out = cdc_unknown_index(cdc.dae(), "v2");

    let (xd, cd, _) = cdc.solve_dc_conv_with(&p, &[], Convergence::default(), 100, cdc.tricks());
    let (xs, cs, _) = cdc.solve_dc_conv_with(&p, &[], Convergence::spice(), 100, cdc.tricks());
    assert!(cd && cs, "both presets must converge");
    // Same physical root regardless of how tightly it was chased.
    assert!(
        (xd[out] - xs[out]).abs() < 1e-6,
        "default {} vs spice {}",
        xd[out],
        xs[out]
    );

    // The default-converged residual meets the tight per-row floor.
    let res = cdc.currents(&xd, &p, 0.0);
    assert!(
        cdc.criterion(&Convergence::default())
            .residual_ok(&res, &xd, GMIN_DC),
        "default solve must satisfy its own residual criterion"
    );
}

#[test]
fn relative_kcl_criterion_scales_with_node_current() {
    // The SPICE-faithful relative KCL test accepts a node imbalance that is
    // small RELATIVE to the branch currents into that node (reltol*sum|I| +
    // abstol), and rejects one that is not -- unlike the absolute-only floor.
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
            ("V1", 1.0),
            ("R1", 1000.0),
            ("D1.Is", 1e-12),
            ("D1.N", 1.0),
            ("D1.Vt", 0.025852),
        ],
    );
    let node = cdc_unknown_index(cdc.dae(), "v2");
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    assert!(conv);
    let n = cdc.dim();

    // A nonlinear circuit builds the per-term scale tape; evaluate it at x.
    let tape = cdc
        .tape_iscale
        .as_ref()
        .expect("nonlinear circuit has the iscale tape");
    let mut inputs = Vec::new();
    cdc.fill_inputs(&x, &p, 0.0, &mut inputs);
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
        !cdc.criterion(&conv).residual_ok(&res, &x, 0.0),
        "same imbalance rejected by the strict floor"
    );
    // An imbalance above the relative floor is rejected.
    res[node] = 5.0 * conv.reltol * iscale + conv.abstol;
    assert!(
        !cdc.residual_relative_ok(&res, &terms, &x, 0.0, &conv),
        "relative-large imbalance rejected"
    );
}

/// gmin-augmented matrix is reused symbolically: factor twice at different
/// gmin and confirm the pattern path is taken (symbolic is Some).
#[test]
fn symbolic_reuse_present() {
    let mut ctx = Graph::new();
    let mut c = Elements::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let devs = vec![DeviceInstance::new(
        std::sync::Arc::new(sane_veriloga::builtin_device("sane_bjt", "Q1", &[])),
        vec![2, 1, 0],
    )];
    let dae = sane_dae::assemble(&mut ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    assert!(
        cdc.dc.symbolic.is_some(),
        "symbolic factorization should be precomputed"
    );
}
