//! Physics checks of the built-in Verilog-A device models, ported from the
//! retired native Rust models' test suite: closed-form currents per region,
//! breakdown, polarity folding, switch conductance, BJT KCL and Early effect.

use num_complex::Complex64;
use rsdag::{eval, ExprId, Graph, Node, SymbolId};
use sane_core::constants::{K_OVER_Q, TEMP_NOMINAL_K, TEMP_SYMBOL};
use sane_device::{DeviceModel, Lowerer};
use sane_veriloga::builtin_device;
use std::collections::HashMap;

/// Lower a builtin device over named terminal symbols and return its terminal
/// current expressions (the fragment's KCL contributions per terminal).
fn currents(
    ctx: &mut Graph,
    module: &str,
    inst: &str,
    fold_params: &[(&str, f64)],
    terms: &[&str],
) -> Vec<ExprId> {
    let dev = builtin_device(module, inst, fold_params);
    let term_v: Vec<ExprId> = terms.iter().map(|n| ctx.sym(n)).collect();
    let term_vdot: Vec<ExprId> = terms.iter().map(|n| ctx.sym(&format!("{n}_dot"))).collect();
    let mut lo = Lowerer::new(ctx);
    dev.lower_behavioral(&mut lo, &term_v, &term_vdot, &[])
        .terminal_currents
}

fn env_of(ctx: &mut Graph, vals: &[(&str, f64)]) -> HashMap<SymbolId, Complex64> {
    // The models read the global `$temp` and per-instance Tnom/Eg/XTI; default
    // them to nominal (temperature scalings become identities) for every
    // instance referenced, unless the test overrides them.
    let tnom = TEMP_NOMINAL_K;
    let mut all: Vec<(String, f64)> = vec![(TEMP_SYMBOL.to_string(), tnom)];
    let mut insts = std::collections::HashSet::new();
    for (name, _) in vals {
        if let Some((inst, _)) = name.split_once('.') {
            if insts.insert(inst.to_string()) {
                all.push((format!("{inst}.Tnom"), tnom));
                all.push((format!("{inst}.Eg"), 1.11));
                all.push((format!("{inst}.XTI"), 3.0));
            }
        }
    }
    all.extend(vals.iter().map(|(n, x)| (n.to_string(), *x)));
    let mut env = HashMap::new();
    for (name, x) in &all {
        let id = ctx.sym(name);
        if let Node::Symbol(s) = ctx.node(id) {
            env.insert(*s, Complex64::new(*x, 0.0));
        }
    }
    env
}

#[test]
fn diode_current_matches_formula() {
    let mut ctx = Graph::new();
    let i = currents(&mut ctx, "sane_diode", "D1", &[], &["va", "vk"]);
    let env = env_of(
        &mut ctx,
        &[("va", 0.7), ("vk", 0.0), ("D1.Is", 1e-14), ("D1.N", 1.0)],
    );
    let got = eval(&ctx, &[i[0]], &env)[0].re;
    let vt = K_OVER_Q * TEMP_NOMINAL_K;
    let want = 1e-14 * ((0.7_f64 / (1.0 * vt)).exp() - 1.0);
    assert!(
        (got - want).abs() <= want.abs() * 1e-6,
        "got {got} want {want}"
    );
    // KCL: anode + cathode current sum to zero.
    assert!((eval(&ctx, &[i[0]], &env)[0].re + eval(&ctx, &[i[1]], &env)[0].re).abs() < 1e-12);
}

#[test]
fn diode_breakdown_clamps_in_reverse() {
    // With BV given, the reverse current is dominated by the breakdown term
    // `-IBV*exp(-(vd+BV)/(N*Vt))`: ~ -IBV at vd = -BV, and far larger below.
    let mut ctx = Graph::new();
    let i = currents(
        &mut ctx,
        "sane_diode",
        "D1",
        &[("BV", 5.6), ("IBV", 1e-3)],
        &["va", "vk"],
    );
    let params = [
        ("D1.Is", 1e-14),
        ("D1.N", 1.0),
        ("D1.BV", 5.6),
        ("D1.IBV", 1e-3),
    ];
    let id_at = |ctx: &mut Graph, vd: f64| {
        let mut vals = vec![("va", 0.0), ("vk", -vd)]; // va - vk = vd
        vals.extend_from_slice(&params);
        let env = env_of(ctx, &vals);
        eval(ctx, &[i[0]], &env)[0].re
    };
    let i_bv = id_at(&mut ctx, -5.6);
    assert!(
        (i_bv + 1e-3).abs() < 1e-4,
        "I(-BV) should be ~ -IBV, got {i_bv}"
    );
    let i_deep = id_at(&mut ctx, -6.0);
    assert!(
        i_deep < i_bv * 5.0,
        "breakdown current should grow below -BV"
    );
    let i_fwd = id_at(&mut ctx, 0.6);
    assert!(i_fwd > 0.0, "forward current stays positive, got {i_fwd}");
}

#[test]
fn mosfet_regions() {
    let mut ctx = Graph::new();
    let i = currents(&mut ctx, "sane_mos", "M1", &[], &["vd", "vg", "vs", "vb"]);

    // Body tied to source (vbs = 0) and gamma = 0: the plain square law holds.
    let base = |vd_: f64, vg_: f64| {
        [
            ("vd", vd_),
            ("vg", vg_),
            ("vs", 0.0),
            ("vb", 0.0),
            ("M1.Kp", 2e-4),
            ("M1.W", 10.0),
            ("M1.L", 1.0),
            ("M1.Vto", 0.4),
            ("M1.lambda", 0.0),
            ("M1.gamma", 0.0),
            ("M1.phi", 0.6),
            ("M1.theta", 0.0),
            ("M1.eta", 0.0),
            ("M1.nsub", 1.0),
        ]
    };
    let k = 2e-4 * 10.0; // Kp*(W/L)
    let vov = 0.8; // vgs=1.2, Vth=0.4 (>> von, so the square law holds exactly)

    // Saturation: vds=1.8 >= vov.
    let env = env_of(&mut ctx, &base(1.8, 1.2));
    let want_sat = 0.5 * k * vov * vov;
    assert!((eval(&ctx, &[i[0]], &env)[0].re - want_sat).abs() <= want_sat * 1e-9);
    assert!(
        eval(&ctx, &[i[1]], &env)[0].re.abs() < 1e-15,
        "gate current"
    );

    // Triode: vds=0.1 < vov.
    let env = env_of(&mut ctx, &base(0.1, 1.2));
    let want_tri = k * (vov * 0.1 - 0.5 * 0.1 * 0.1);
    assert!((eval(&ctx, &[i[0]], &env)[0].re - want_tri).abs() <= want_tri * 1e-9);

    // Subthreshold (weak inversion): 0.2 V below Vth is a small exponential
    // leakage, decaying ~exp(vov/(n*Vt)).
    let vt = K_OVER_Q * TEMP_NOMINAL_K;
    let e02 = env_of(&mut ctx, &base(1.8, 0.2));
    let i_02 = eval(&ctx, &[i[0]], &e02)[0].re;
    let e03 = env_of(&mut ctx, &base(1.8, 0.3));
    let i_03 = eval(&ctx, &[i[0]], &e03)[0].re;
    assert!(i_02 > 0.0 && i_02 < 1e-6, "subthreshold leakage {i_02}");
    let ratio = i_03 / i_02; // 0.1 V step in vgs (n = 1)
    let expect = (0.1_f64 / vt).exp();
    assert!(
        (ratio / expect - 1.0).abs() < 0.05,
        "subthreshold slope {ratio} vs {expect}"
    );
    // Deep cutoff: 1.4 V below Vth -> negligible.
    let edeep = env_of(&mut ctx, &base(1.8, -1.0));
    let i_deep = eval(&ctx, &[i[0]], &edeep)[0].re;
    assert!(i_deep < 1e-12, "deep cutoff {i_deep}");
}

#[test]
fn pmos_regions() {
    // P-channel via the `type` fold: source at the high rail, conducts for
    // vsg > |Vth|; the n-equivalent law predicts the (negative) drain current.
    let mut ctx = Graph::new();
    let i = currents(
        &mut ctx,
        "sane_mos",
        "M1",
        &[("type", -1.0)],
        &["vd", "vg", "vs", "vb"],
    );
    let base = |vd_: f64, vg_: f64| {
        [
            ("vd", vd_),
            ("vg", vg_),
            ("vs", 5.0),
            ("vb", 5.0),
            ("M1.Kp", 2e-4),
            ("M1.W", 10.0),
            ("M1.L", 1.0),
            ("M1.Vto", -0.4),
            ("M1.lambda", 0.0),
            ("M1.gamma", 0.0),
            ("M1.phi", 0.6),
            ("M1.theta", 0.0),
            ("M1.eta", 0.0),
            ("M1.nsub", 1.0),
        ]
    };
    let k = 2e-4 * 10.0;
    // vsg = 5 - 3.8 = 1.2, vov = vsg - |Vth| = 0.8.
    let vov = 0.8;

    // Saturation: vsd = 1.8 >= vov -> Id = -0.5*k*vov^2 (negative for P).
    let env = env_of(&mut ctx, &base(3.2, 3.8));
    let want_sat = -0.5 * k * vov * vov;
    assert!((eval(&ctx, &[i[0]], &env)[0].re - want_sat).abs() <= want_sat.abs() * 1e-9);
    assert!(
        eval(&ctx, &[i[1]], &env)[0].re.abs() < 1e-15,
        "gate current"
    );

    // Subthreshold: vsg = 0.2 -> small negative exponential leakage.
    let ecut = env_of(&mut ctx, &base(3.2, 4.8));
    let i_cut = eval(&ctx, &[i[0]], &ecut)[0].re;
    assert!(
        i_cut < 0.0 && i_cut.abs() < 1e-6,
        "subthreshold leakage {i_cut}"
    );
    // Deep cutoff: vsg = -0.6 -> negligible.
    let edeep = env_of(&mut ctx, &base(3.2, 5.6));
    let i_deep = eval(&ctx, &[i[0]], &edeep)[0].re;
    assert!(i_deep.abs() < 1e-12, "deep cutoff {i_deep}");

    // KCL closes: drain + source currents sum to zero (gate is 0).
    let env = env_of(&mut ctx, &base(3.2, 3.8));
    let sum = eval(&ctx, &[i[0]], &env)[0].re + eval(&ctx, &[i[2]], &env)[0].re;
    assert!(sum.abs() < 1e-15, "KCL: {sum}");
}

#[test]
fn vswitch_on_and_off() {
    let mut ctx = Graph::new();
    let i = currents(&mut ctx, "sane_vswitch", "S1", &[], &["a", "b", "cp", "cm"]);
    let pars = [
        ("a", 2.0),
        ("b", 0.0),
        ("S1.Ron", 10.0),
        ("S1.Roff", 1e6),
        ("S1.Vt", 0.5),
        ("S1.Vh", 1e-3),
    ];

    // On: control 1.0 >> Vt+Vh -> fully on, i = (Va-Vb)/Ron = 0.2.
    let mut on = pars.to_vec();
    on.extend([("cp", 1.0), ("cm", 0.0)]);
    let env = env_of(&mut ctx, &on);
    assert!((eval(&ctx, &[i[0]], &env)[0].re - 0.2).abs() < 1e-9);

    // Off: control 0.0 << Vt-Vh -> fully off, i = (Va-Vb)/Roff = 2e-6.
    let mut off = pars.to_vec();
    off.extend([("cp", 0.0), ("cm", 0.0)]);
    let env = env_of(&mut ctx, &off);
    assert!((eval(&ctx, &[i[0]], &env)[0].re - 2e-6).abs() < 1e-12);

    // Mid-window (control exactly at Vt, wide Vh): smoothstep = 0.5, so the
    // log-interpolated conductance is the geometric mean sqrt(Gon*Goff).
    let mut mid = vec![
        ("a", 2.0),
        ("b", 0.0),
        ("S1.Ron", 10.0),
        ("S1.Roff", 1e6),
        ("S1.Vt", 0.5),
        ("S1.Vh", 0.2),
    ];
    mid.extend([("cp", 0.5), ("cm", 0.0)]);
    let env = env_of(&mut ctx, &mid);
    let g_mid = (0.1_f64 * 1e-6).sqrt(); // sqrt(Gon*Goff)
    assert!(
        (eval(&ctx, &[i[0]], &env)[0].re - 2.0 * g_mid).abs() < 1e-9 * (2.0 * g_mid),
        "mid-window conductance should be the geometric mean"
    );
}

#[test]
fn bjt_kcl_and_beta() {
    let mut ctx = Graph::new();
    let i = currents(&mut ctx, "sane_bjt", "Q1", &[], &["vc", "vb", "ve"]);
    let inf = 1e12;
    let env = env_of(
        &mut ctx,
        &[
            ("vc", 3.0),
            ("vb", 0.7),
            ("ve", 0.0),
            ("Q1.Is", 1e-15),
            ("Q1.betaF", 150.0),
            ("Q1.betaR", 2.0),
            ("Q1.VAf", inf),
            ("Q1.VAr", inf),
            ("Q1.IKF", inf),
            ("Q1.IKR", inf),
            ("Q1.ISE", 0.0),
            ("Q1.NE", 2.0),
            ("Q1.ISC", 0.0),
            ("Q1.NC", 2.0),
        ],
    );
    let ic = eval(&ctx, &[i[0]], &env)[0].re;
    let ib = eval(&ctx, &[i[1]], &env)[0].re;
    let ie = eval(&ctx, &[i[2]], &env)[0].re;
    assert!(
        (ic + ib + ie).abs() <= 1e-12 * (1.0 + ic.abs()),
        "KCL: {ic}+{ib}+{ie}"
    );
    // Forward-active sanity: Ic ~ betaF * Ib (Early negligible at VAf=inf).
    assert!((ic - 150.0 * ib).abs() <= ic.abs() * 1e-6, "beta relation");
}

#[test]
fn bjt_early_effect_finite_output_resistance() {
    // With a finite VAf the collector current rises with Vce: the Early slope
    // gce = dIc/dVce should be ~ Ic/VAf (output resistance ro = VAf/Ic).
    let mut ctx = Graph::new();
    let i = currents(&mut ctx, "sane_bjt", "Q1", &[], &["vc", "vb", "ve"]);
    let (vaf, inf) = (100.0, 1e12);
    let env_at = |ctx: &mut Graph, vce: f64| {
        env_of(
            ctx,
            &[
                ("vc", vce),
                ("vb", 0.7),
                ("ve", 0.0),
                ("Q1.Is", 1e-15),
                ("Q1.betaF", 150.0),
                ("Q1.betaR", 2.0),
                ("Q1.VAf", vaf),
                ("Q1.VAr", inf),
                ("Q1.IKF", inf),
                ("Q1.IKR", inf),
                ("Q1.ISE", 0.0),
                ("Q1.NE", 2.0),
                ("Q1.ISC", 0.0),
                ("Q1.NC", 2.0),
            ],
        )
    };
    let e1 = env_at(&mut ctx, 5.0);
    let e2 = env_at(&mut ctx, 6.0);
    let ic1 = eval(&ctx, &[i[0]], &e1)[0].re;
    let ic2 = eval(&ctx, &[i[0]], &e2)[0].re;
    let gce = (ic2 - ic1) / 1.0;
    let want = ic1 / vaf;
    assert!((gce - want).abs() <= want * 0.05, "gce={gce}, want~{want}");
    assert!(ic2 > ic1, "Ic must rise with Vce (Early)");
}
