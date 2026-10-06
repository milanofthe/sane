use super::*;
use rsdag::eval;
use sane_device::Lowerer;
use sane_mna::{Circuit, SourceFn};
use std::collections::{HashMap, HashSet};

fn env_of(ctx: &mut Graph, vals: &[(&str, f64)]) -> HashMap<SymbolId, f64> {
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
            env.insert(*s, *x);
        }
    }
    env
}

/// The largest `|I + C x'|` at `env`, the state moving at `rates` (one per
/// unknown, empty for rest).
fn max_resid(ctx: &mut Graph, dae: &Dae, env: &HashMap<SymbolId, f64>, rates: &[f64]) -> f64 {
    let (_, (rows, cols, c)) = dae.jacobian_iq_coo(ctx);
    let mut f: Vec<f64> = dae
        .currents
        .iter()
        .map(|&r| eval(ctx, &[r], env)[0])
        .collect();
    for ((&r, &col), &e) in rows.iter().zip(&cols).zip(&c) {
        if let Some(&v) = rates.get(col) {
            f[r] += eval(ctx, &[e], env)[0] * v;
        }
    }
    f.iter().map(|z| z.abs()).fold(0.0, f64::max)
}

/// A capacitor expressed through the behavioral lowering path: the charge
/// `q_a = C * (v_a - v_b)`, no current, no extra unknowns. Used to prove the
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
        term_v: &[ExprId],
        _control_i: &[ExprId],
    ) -> sane_device::BehavioralFragment {
        let ctx = lo.ctx();
        let c = ctx.sym(&self.name);
        let zero = ctx.zero();
        let dv = ctx.sub(term_v[0], term_v[1]);
        let q = ctx.mul(c, dv);
        let nq = ctx.neg(q);
        sane_device::BehavioralFragment {
            param_syms: Vec::new(),
            events: Vec::new(),
            terminal_currents: vec![zero, zero],
            currents: Vec::new(),
            terminal_charges: vec![q, nq],
            charges: Vec::new(),
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
            ("i_V1", -0.6),
            ("t", 0.0),
        ],
    );
    let rows = |d: &Dae| {
        d.currents
            .iter()
            .chain(&d.charges)
            .copied()
            .collect::<Vec<_>>()
    };
    for (i, (rn, rb)) in rows(&native).iter().zip(&rows(&behav)).enumerate() {
        let d = (eval(&ctx, &[*rn], &env)[0] - eval(&ctx, &[*rb], &env)[0]).abs();
        assert!(d < 1e-9, "row {i} differs by {d}");
    }
}

#[test]
fn mfactor_scales_terminal_current() {
    // The behavioral cap from node 2 to ground contributes the charge C*v2 to
    // the node-2 KCL. With mfactor = m, that contribution scales to m*C*v2,
    // while every other row is untouched (m parallel devices).
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

    let cap = 1e-6;
    let env = env_of(
        &mut ctx,
        &[
            ("V1", 1.0),
            ("R", 1000.0),
            ("C", cap),
            ("v1", 1.3),
            ("v2", 0.4),
            ("i_V1", -0.6),
            ("t", 0.0),
        ],
    );
    // Row 1 is the node-2 KCL: its m=2 charge exceeds m=1 by exactly C*v2.
    let q1 = eval(&ctx, &[d1.charges[1]], &env)[0];
    let q2 = eval(&ctx, &[d2.charges[1]], &env)[0];
    assert!(((q2 - q1) - cap * 0.4).abs() < 1e-12, "delta={}", q2 - q1);
    // Every current and the other charges are identical (multiplicity
    // touches only this device's charge).
    let rows = |d: &Dae| {
        let mut r = d.currents.clone();
        r.extend([d.charges[0], d.charges[2]]);
        r
    };
    for (i, (a, b)) in rows(&d1).iter().zip(&rows(&d2)).enumerate() {
        let d = (eval(&ctx, &[*a], &env)[0] - eval(&ctx, &[*b], &env)[0]).abs();
        assert!(d < 1e-12, "row {i} changed by {d}");
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
            ("i_V1", iv),
            ("t", 0.0),
        ],
    );
    assert!(max_resid(&mut ctx, &dae, &env, &[0.0, vdot2, 0.0]) < 1e-9);
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
            ("i_V1", -(2.0 - 2.0 / 3.0) / 2.0),
            ("t", 0.0),
        ],
    );
    assert!(max_resid(&mut ctx, &red, &env, &[]) < 1e-9);
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
            ("t", 0.0),
        ],
    );
    let m = max_resid(&mut ctx, &dae, &env, &[]);
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
    assert!(max_resid(&mut ctx, &dae, &env, &[]) < 1e-9);
}

#[test]
fn sin_source_residual_is_time_dependent() {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).set_source(SourceFn::Sin);
    c.resistor("R1", 1, 0);
    let dae = assemble_dae(&mut ctx, &c, &[]);
    let i_v1 = dae.unknowns.iter().position(|u| u == "i_V1").unwrap();
    let s = rsdag::to_string(&ctx, dae.currents[i_v1]);
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
    let s = rsdag::to_string(&ctx, dae.currents[v2]);
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
    // the coupling is in the L1 row's flux
    let s = rsdag::to_string(&ctx, dae.charges[i_l1]);
    assert!(s.contains("K1"), "L1 flux missing mutual coeff: {s}");
    assert!(s.contains("i_L2"), "L1 flux missing i_L2: {s}");
}
