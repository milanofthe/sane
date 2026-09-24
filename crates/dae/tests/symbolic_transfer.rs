//! Closed-form small-signal transfer `H(s)` over the symbolic DAE, verified
//! numerically against analytic references. This is the live path that
//! superseded the old standalone MNA assembler: a circuit is assembled into the
//! DAE `F(x, x', t) = 0`, the small-signal matrix `A(s) = dF/dx + s*dF/dx'` is
//! formed by autodiff, and `H(s)` comes out by Cramer's rule
//! ([`small_signal_transfer`]).

use std::collections::HashMap;

use num_complex::Complex64;
use rsdag::{eval, Node, SymbolId};
use sane_core::Graph;
use sane_dae::{assemble_dae, small_signal_transfer};
use sane_mna::Circuit;

/// Intern `name` and return its [`SymbolId`] (panics if it is not a symbol).
fn sym_id(ctx: &mut Graph, name: &str) -> SymbolId {
    let id = ctx.sym(name);
    match ctx.node(id) {
        Node::Symbol(s) => *s,
        _ => unreachable!("'{name}' is not a symbol"),
    }
}

/// RC lowpass `Vin -[R]- out -[C]- gnd`: `H(s) = 1 / (1 + s*R*C)`.
#[test]
fn rc_lowpass_matches_analytic() {
    let mut ctx = Graph::new();
    // Nodes: 1 = in, 2 = out, 0 = ground.
    let mut cct = Circuit::new();
    cct.voltage_source("Vin", 1, 0)
        .resistor("R", 1, 2)
        .capacitor("C", 2, 0);
    let dae = assemble_dae(&mut ctx, &cct, &[]);

    // Output node 2 -> unknown "v2"; input is the source parameter "Vin".
    let h = small_signal_transfer(&mut ctx, &dae, "Vin", "v2").expect("transfer exists");

    let (r, c) = (1000.0, 1e-6);
    let r_id = sym_id(&mut ctx, "R");
    let c_id = sym_id(&mut ctx, "C");
    let s_id = sym_id(&mut ctx, "s");

    let h_at = |ctx: &Graph, f: f64| -> Complex64 {
        let s = Complex64::new(0.0, 2.0 * std::f64::consts::PI * f);
        let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
        env.insert(r_id, Complex64::new(r, 0.0));
        env.insert(c_id, Complex64::new(c, 0.0));
        env.insert(s_id, s);
        eval(ctx, &[h], &env)[0]
    };

    for &f in &[10.0, 100.0, 159.155, 1000.0, 10_000.0] {
        let omega = 2.0 * std::f64::consts::PI * f;
        let s = Complex64::new(0.0, omega);
        let got = h_at(&ctx, f);
        let expected = Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + s * r * c);
        assert!(
            (got - expected).norm() < 1e-9,
            "f={f} Hz: got {got}, expected {expected}"
        );
    }

    // At the corner frequency the magnitude is 1/sqrt(2).
    let fc = 1.0 / (2.0 * std::f64::consts::PI * r * c);
    let mag = h_at(&ctx, fc).norm();
    assert!(
        (mag - 1.0 / 2.0_f64.sqrt()).abs() < 1e-6,
        "corner mag {mag}"
    );
}

/// VCCS into a load: `V1 in`, `G1: I(out->0) = gm*V(in,0)`, `RL: out->0`.
/// KCL at out: `V_out/RL + gm*V_in = 0  =>  H = -gm*RL` (frequency-independent).
#[test]
fn transconductance_into_load() {
    let mut ctx = Graph::new();
    let mut cct = Circuit::new();
    cct.voltage_source("V1", 1, 0)
        .vccs("G1", 2, 0, 1, 0)
        .resistor("RL", 2, 0);
    let dae = assemble_dae(&mut ctx, &cct, &[]);

    let h = small_signal_transfer(&mut ctx, &dae, "V1", "v2").expect("transfer exists");

    let g_id = sym_id(&mut ctx, "G1");
    let rl_id = sym_id(&mut ctx, "RL");
    let s_id = sym_id(&mut ctx, "s");
    let (gm, rl) = (0.01, 2000.0);

    let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
    env.insert(g_id, Complex64::new(gm, 0.0));
    env.insert(rl_id, Complex64::new(rl, 0.0));
    env.insert(s_id, Complex64::new(0.0, 1.0)); // no s-dependence expected

    let got = eval(&ctx, &[h], &env)[0];
    assert!(
        (got - Complex64::new(-gm * rl, 0.0)).norm() < 1e-9,
        "got {got}, expected {}",
        -gm * rl
    );
}
