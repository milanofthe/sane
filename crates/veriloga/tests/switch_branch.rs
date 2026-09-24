use num_complex::Complex64;
use rsdag::{eval, Node, SymbolId};
use sane_core::Graph;
use sane_dae::{assemble_dae, DeviceInstance};
use sane_mna::Circuit;
use sane_veriloga::{device::VerilogADevice, elaborate, parse_modules};
use std::collections::HashMap;
use std::sync::Arc;

fn elab(src: &str) -> Arc<sane_veriloga::ElaboratedModule> {
    let ms = parse_modules(src, "t.va", &[]).expect("parse");
    Arc::new(elaborate(&ms[0]).expect("elab"))
}

fn env_of(ctx: &mut Graph, vals: &[(&str, f64)]) -> HashMap<SymbolId, Complex64> {
    let mut env = HashMap::new();
    for (n, x) in vals {
        let e = ctx.sym(n);
        if let Node::Symbol(s) = ctx.node(e) {
            env.insert(*s, Complex64::new(*x, 0.0));
        }
    }
    env
}

#[test]
fn switch_branch_single_arm_is_constrained() {
    // A switch branch contributed in only ONE arm (no `else`), gated at runtime.
    // The not-taken path must constrain the branch current to 0 (open branch),
    // not leave it free -- otherwise the DAE is structurally singular. We assert
    // that some residual equals the branch current `i` when the gate is FALSE and
    // 0 when it is TRUE (the `select(cond, V(p)-V(n), i)` constraint); the buggy
    // version had `select(cond, ..., 0)`, for which no residual satisfies this.
    let src = "module sw1(p,n,c); inout p,n,c; electrical p,n,c; \
               analog begin if (V(c) > 0.5) V(p,n) <+ 0.0; end endmodule";
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("VP", 1, 0).voltage_source("VC", 2, 0);
    let dev = VerilogADevice::new("X1", elab(src));
    let devs = vec![DeviceInstance::new(Box::new(dev), vec![1, 0, 2])];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    let sw = dae
        .unknowns
        .iter()
        .find(|u| u.contains(".sw_"))
        .expect("a switch-branch current unknown")
        .clone();

    // i = 1, everything else 0. Gate FALSE (v2 = 0) vs TRUE (v2 = 1).
    let base = |gate: f64| -> Vec<(&str, f64)> {
        vec![
            ("VP", 0.0),
            ("VC", gate),
            ("v1", 0.0),
            ("v2", gate),
            ("i_VP", 0.0),
            ("i_VC", 0.0),
            ("t", 0.0),
            (Box::leak(sw.clone().into_boxed_str()), 1.0),
        ]
    };
    let env_false = env_of(&mut ctx, &base(0.0));
    let env_true = env_of(&mut ctx, &base(1.0));
    let found = dae.residuals.iter().any(|r| {
        (eval(&ctx, &[*r], &env_false)[0].re - 1.0).abs() < 1e-9
            && eval(&ctx, &[*r], &env_true)[0].re.abs() < 1e-9
    });
    assert!(
        found,
        "no residual constrains the switch current on the open path"
    );
}

#[test]
fn switchable_resistor_branch() {
    // `if (on) I(p,n) <+ g*V(p,n); else V(p,n) <+ 0;` -- a switchable parasitic.
    let src = "module sw(p,n); inout p,n; electrical p,n; \
               parameter real g = 1e-3; parameter integer on = 1; \
               analog begin if (on) I(p,n) <+ g*V(p,n); else V(p,n) <+ 0.0; end endmodule";
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0);
    let dev = VerilogADevice::with_instance(
        "X1",
        elab(src),
        rustc_hash::FxHashMap::from_iter([("g".into(), 1e-3), ("on".into(), 1.0)]),
        Default::default(),
    );
    let devs = vec![DeviceInstance::new(Box::new(dev), vec![1, 0])];
    let dae = assemble_dae(&mut ctx, &c, &devs);
    println!("UNKNOWNS = {:?}", dae.unknowns);

    // Consistent operating point: v1 = 2 V, branch current i = g*v1 = 2 mA,
    // source current i_V1 = -i. Every residual must vanish there.
    let g = 1e-3;
    let v1 = 2.0;
    let i = g * v1;
    let mut vals = vec![
        ("V1", v1),
        ("X1.g", g),
        ("X1.on", 1.0),
        ("v1", v1),
        ("vdot1", 0.0),
        ("i_V1", -i),
        ("t", 0.0),
    ];
    // The switch-branch current unknown (name discovered from dae.unknowns). Its
    // canonical orientation is (n,p), so the branch current is -g*V(p,n) = -i.
    for u in &dae.unknowns {
        if u.contains(".sw_") {
            vals.push((Box::leak(u.clone().into_boxed_str()), -i));
        }
    }
    let env = env_of(&mut ctx, &vals);
    for (k, r) in dae.residuals.iter().enumerate() {
        let val = eval(&ctx, &[*r], &env)[0].norm();
        assert!(
            val < 1e-9,
            "residual {k} ({}) = {val}",
            dae.unknowns.get(k).map(|s| s.as_str()).unwrap_or("?")
        );
    }
}
