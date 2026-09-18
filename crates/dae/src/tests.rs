use super::*;
use num_complex::Complex64;
use rsdag::eval;
use sane_device::Lowerer;
use sane_mna::{Circuit, SourceFn};
use std::collections::{HashMap, HashSet};

fn env_of(ctx: &mut Graph, vals: &[(&str, f64)]) -> HashMap<SymbolId, Complex64> {
    // Native devices read the global `$temp` and per-instance Tnom/Eg/XTI;
    // default them to nominal (temperature scalings become identities) for
    // every instance referenced, unless the test overrides them.
    let tnom = sane_core::constants::TEMP_NOMINAL_K;
    let mut all: Vec<(String, f64)> = vec![(sane_core::constants::TEMP_SYMBOL.to_string(), tnom)];
    let mut insts = HashSet::new();
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
        let e = ctx.sym(name);
        if let Node::Symbol(s) = ctx.node(e) {
            env.insert(*s, Complex64::new(*x, 0.0));
        }
    }
    env
}

fn max_resid(ctx: &Graph, dae: &Dae, env: &HashMap<SymbolId, Complex64>) -> f64 {
    dae.residuals
        .iter()
        .map(|&r| eval(ctx, &[r], env)[0].norm())
        .fold(0.0, f64::max)
}

/// A capacitor expressed through the behavioral lowering path:
/// `i_a = C * d/dt(v_a - v_b)`, no extra unknowns. Used to prove the
/// `lower_behavioral` plumbing reproduces a native circuit element exactly.
struct BehavioralCap {
    name: String,
}
impl sane_device::DeviceModel for BehavioralCap {
    fn n_terminals(&self) -> usize {
        2
    }
    fn lower_behavioral(
        &self,
        lo: &mut Lowerer,
        _term_v: &[ExprId],
        term_vdot: &[ExprId],
        _control_i: &[ExprId],
    ) -> sane_device::BehavioralFragment {
        let ctx = lo.ctx();
        let c = ctx.sym(&self.name);
        let dvd = ctx.sub(term_vdot[0], term_vdot[1]);
        let i = ctx.mul(c, dvd);
        let ni = ctx.neg(i);
        sane_device::BehavioralFragment {
            param_syms: Vec::new(),
            events: Vec::new(),
            terminal_currents: vec![i, ni],
            residuals: Vec::new(),
            noise: Vec::new(),
            op_vars: Vec::new(),
            limits: Vec::new(),
        }
    }
}

#[test]
fn behavioral_capacitor_matches_native() {
    // Same V1+R network, capacitor once as a native element, once as a
    // behavioral lowering. Same unknowns; residuals identical at any point.
    let mut ctx = Graph::new();
    let mut cn = Circuit::new();
    cn.voltage_source("V1", 1, 0)
        .resistor("R", 1, 2)
        .capacitor("C", 2, 0);
    let native = assemble_dae(&mut ctx, &cn, &[]);

    let mut cb = Circuit::new();
    cb.voltage_source("V1", 1, 0).resistor("R", 1, 2);
    let devs = vec![DeviceInstance::new(
        Box::new(BehavioralCap { name: "C".into() }),
        vec![2, 0],
    )];
    let behav = assemble_dae(&mut ctx, &cb, &devs);

    assert_eq!(native.unknowns, behav.unknowns, "same unknown layout");
    assert_eq!(behav.dim(), 3, "behavioral cap mints no extra unknown");

    // Arbitrary (inconsistent) point: residual VALUES must still agree.
    let env = env_of(
        &mut ctx,
        &[
            ("V1", 1.0),
            ("R", 1000.0),
            ("C", 1e-6),
            ("v1", 1.3),
            ("v2", 0.4),
            ("vdot1", 2.0),
            ("vdot2", 7.0),
            ("i_V1", -0.6),
            ("t", 0.0),
        ],
    );
    for (i, (rn, rb)) in native.residuals.iter().zip(&behav.residuals).enumerate() {
        let d = (eval(&ctx, &[*rn], &env)[0] - eval(&ctx, &[*rb], &env)[0]).norm();
        assert!(d < 1e-9, "residual {i} differs by {d}");
    }
}

#[test]
fn mfactor_scales_terminal_current() {
    // The behavioral cap from node 2 to ground contributes C*vdot2 to the
    // node-2 KCL. With mfactor = m, that contribution scales to m*C*vdot2,
    // while every other residual is untouched (m parallel devices).
    let mut ctx = Graph::new();
    let build = |ctx: &mut Graph, m: f64| {
        let mut cb = Circuit::new();
        cb.voltage_source("V1", 1, 0).resistor("R", 1, 2);
        let devs =
            vec![
                DeviceInstance::new(Box::new(BehavioralCap { name: "C".into() }), vec![2, 0])
                    .with_mfactor(m),
            ];
        assemble_dae(ctx, &cb, &devs)
    };
    let d1 = build(&mut ctx, 1.0);
    let d2 = build(&mut ctx, 2.0);
    assert_eq!(d1.unknowns, d2.unknowns);
    assert_eq!(d1.unknowns, vec!["v1", "v2", "i_V1"]);

    let (cap, vdot2) = (1e-6, 7.0);
    let env = env_of(
        &mut ctx,
        &[
            ("V1", 1.0),
            ("R", 1000.0),
            ("C", cap),
            ("v1", 1.3),
            ("v2", 0.4),
            ("vdot1", 2.0),
            ("vdot2", vdot2),
            ("i_V1", -0.6),
            ("t", 0.0),
        ],
    );
    // Row 1 is the node-2 KCL: its m=2 residual exceeds m=1 by exactly C*vdot2.
    let r1 = eval(&ctx, &[d1.residuals[1]], &env)[0];
    let r2 = eval(&ctx, &[d2.residuals[1]], &env)[0];
    assert!(
        ((r2 - r1).re - cap * vdot2).abs() < 1e-12,
        "delta={}",
        (r2 - r1).re
    );
    // The other rows are identical (multiplicity touches only this device).
    for i in [0usize, 2] {
        let d = (eval(&ctx, &[d1.residuals[i]], &env)[0] - eval(&ctx, &[d2.residuals[i]], &env)[0])
            .norm();
        assert!(d < 1e-12, "residual {i} changed by {d}");
    }
}

#[test]
fn rc_dae_consistent_point() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R", 1, 2)
        .capacitor("C", 2, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    assert_eq!(dae.dim(), 3);
    assert_eq!(dae.unknowns, vec!["v1", "v2", "i_V1"]);

    // A consistent transient point: v1 forced, v2 free, vdot2 = (v1-v2)/(R*C).
    let (r, cap, v1, v2) = (1000.0, 1e-6, 1.0, 0.5);
    let vdot2 = (v1 - v2) / (r * cap);
    let iv = -(v1 - v2) / r;
    let env = env_of(
        &mut ctx,
        &[
            ("V1", v1),
            ("R", r),
            ("C", cap),
            ("v1", v1),
            ("v2", v2),
            ("vdot1", 0.0),
            ("vdot2", vdot2),
            ("i_V1", iv),
            ("t", 0.0),
        ],
    );
    assert!(max_resid(&ctx, &dae, &env) < 1e-9);
}

#[test]
fn eliminate_resistive_node_preserves_point() {
    // A resistive divider chain: V1 - R1 - (node 2) - R2 - (node 3) - Rload.
    // Node 2 is internal and purely resistive: eliminating it must fold R1,R2
    // into a direct branch and preserve the divider's consistent point.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 2)
        .resistor("R2", 2, 3)
        .resistor("Rload", 3, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    assert_eq!(dae.unknowns, vec!["v1", "v2", "v3", "i_V1"]);

    let mut keep = HashSet::new();
    keep.insert("v3".to_string()); // protect the probe; node 2 is eliminated
    let (red, gone) = eliminate_nodes(&mut ctx, &dae, &keep);
    assert_eq!(gone, vec!["v2".to_string()]);
    assert_eq!(red.unknowns, vec!["v1", "v3", "i_V1"]);

    // Divider: v3 = 2 * Rload/(R1+R2+Rload) = 2/3, i_V1 = -(v1-v3)/(R1+R2).
    let env = env_of(
        &mut ctx,
        &[
            ("V1", 2.0),
            ("R1", 1.0),
            ("R2", 1.0),
            ("Rload", 1.0),
            ("v1", 2.0),
            ("v3", 2.0 / 3.0),
            ("vdot1", 0.0),
            ("vdot3", 0.0),
            ("i_V1", -(2.0 - 2.0 / 3.0) / 2.0),
            ("t", 0.0),
        ],
    );
    assert!(max_resid(&ctx, &red, &env) < 1e-9);
}

#[test]
fn diode_rectifier_consistent_point() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R", 1, 2);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
        vec![2, 0], // anode = node 2, cathode = ground
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    assert_eq!(dae.dim(), 3);

    // Pick a diode voltage, derive the rest so F = 0. The diode's thermal
    // voltage is k*T/q at the nominal $temp (env_of binds it), so use the
    // same value here rather than a hard-coded constant.
    let vt = sane_core::constants::K_OVER_Q * sane_core::constants::TEMP_NOMINAL_K;
    let (is, nn, r) = (1e-14_f64, 1.0_f64, 1000.0_f64);
    let v2 = 0.6_f64;
    let id = is * ((v2 / (nn * vt)).exp() - 1.0);
    let v1 = v2 + r * id;
    let iv = -id;
    let env = env_of(
        &mut ctx,
        &[
            ("V1", v1),
            ("R", r),
            ("D1.Is", is),
            ("D1.N", nn),
            ("v1", v1),
            ("v2", v2),
            ("i_V1", iv),
            ("vdot1", 0.0),
            ("vdot2", 0.0),
            ("t", 0.0),
        ],
    );
    let m = max_resid(&ctx, &dae, &env);
    assert!(m < 1e-9 * (1.0 + id.abs()), "residual {m}");
}

#[test]
fn ccvs_consistent_point() {
    // V1 1 0; R1 1 0; H1: V(2,0) = H1*I(V1); RL 2 0.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R1", 1, 0)
        .ccvs("H1", 2, 0, "V1")
        .resistor("RL", 2, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    assert_eq!(dae.dim(), 4); // v1, v2, i_V1, i_H1

    let (vin, r1, gain, rl) = (1.0, 1000.0, 5.0, 2000.0);
    let i_v1 = -vin / r1;
    let v2 = gain * i_v1;
    let i_h1 = -v2 / rl;
    let env = env_of(
        &mut ctx,
        &[
            ("V1", vin),
            ("R1", r1),
            ("H1", gain),
            ("RL", rl),
            ("v1", vin),
            ("v2", v2),
            ("i_V1", i_v1),
            ("i_H1", i_h1),
            ("t", 0.0),
        ],
    );
    assert!(max_resid(&ctx, &dae, &env) < 1e-9);
}

#[test]
fn sin_source_residual_is_time_dependent() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).set_source(SourceFn::Sin);
    c.resistor("R1", 1, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    let i_v1 = dae.unknowns.iter().position(|u| u == "i_V1").unwrap();
    let s = rsdag::to_string(&ctx, dae.residuals[i_v1]);
    assert!(
        s.contains("sin("),
        "V constraint should be a sine of t: {s}"
    );
}

#[test]
fn current_switch_uses_control_current() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("Vc", 1, 0).resistor("R1", 1, 0);
    let devs = vec![DeviceInstance::new(
        Box::new(sane_device::CSwitch::new("W1", "Vc")),
        vec![2, 0],
    )];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    // node 2 KCL carries the switch conductance (a Select on I(Vc)).
    let v2 = dae.unknowns.iter().position(|u| u == "v2").unwrap();
    let s = rsdag::to_string(&ctx, dae.residuals[v2]);
    assert!(s.contains("select("), "switch should use Select: {s}");
    assert!(
        s.contains("i_Vc"),
        "switch should reference control current: {s}"
    );
}

#[test]
fn mutual_inductance_couples_constraints() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .inductor("L1", 1, 0)
        .inductor("L2", 2, 0)
        .resistor("RL", 2, 0)
        .mutual("K1", "L1", "L2");
    let dae = assemble_dae(&mut ctx, &c, &[]);
    let i_l1 = dae.unknowns.iter().position(|u| u == "i_L1").unwrap();
    let s = rsdag::to_string(&ctx, dae.residuals[i_l1]);
    assert!(s.contains("K1"), "L1 constraint missing mutual coeff: {s}");
    assert!(s.contains("idot_L2"), "L1 constraint missing idot_L2: {s}");
}

#[test]
fn small_signal_rc_matches_analytic() {
    // The small-signal AC transfer of the RC equals 1/(1 + s R C), derived
    // purely from the DAE Jacobians dF/dx + s dF/dx'.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R", 1, 2)
        .capacitor("C", 2, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    let h = small_signal_transfer(&mut ctx, &dae, "V1", "v2").unwrap();

    let s_e = ctx.sym("s");
    let sid = match ctx.node(s_e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let (r, cap) = (1000.0, 1e-6);
    for &f in &[10.0, 159.155, 1000.0, 10_000.0] {
        let w = 2.0 * std::f64::consts::PI * f;
        let mut env = env_of(&mut ctx, &[("R", r), ("C", cap)]);
        env.insert(sid, Complex64::new(0.0, w));
        let got = eval(&ctx, &[h], &env)[0];
        let sval = Complex64::new(0.0, w);
        let expected = Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + sval * r * cap);
        assert!((got - expected).norm() < 1e-9, "f={f}: {got} vs {expected}");
    }
}

#[test]
fn jacobian_dims_and_sparsity() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0)
        .resistor("R", 1, 2)
        .capacitor("C", 2, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);

    let jx = dae.jacobian_x(&mut ctx);
    assert_eq!(jx.len(), 3);
    assert_eq!(jx[0].len(), 3);

    let jxd = dae.jacobian_xdot(&mut ctx);
    let sp: Vec<Vec<bool>> = jxd
        .iter()
        .map(|row| row.iter().map(|&e| !ctx.is_zero(e)).collect())
        .collect();
    // Columns are [vdot1, vdot2, (branch -> None)]. Only the capacitor on
    // node 2 makes vdot2 appear, and only in node 2's KCL (row index 1).
    assert!(sp[1][1], "vdot2 should appear in node-2 residual");
    assert!(
        sp.iter().all(|row| !row[0]),
        "vdot1 has no cap -> zero column"
    );
    assert!(
        sp.iter().all(|row| !row[2]),
        "algebraic branch -> zero column"
    );
}
