use super::*;
use num_complex::Complex64;
use rsdag::{eval, Node, SymbolId};
use sane_core::Graph;
use std::collections::HashMap;

fn env_of(ctx: &mut Graph, vals: &[(&str, f64)]) -> HashMap<SymbolId, Complex64> {
    let mut env = HashMap::new();
    for (name, x) in vals {
        let id = ctx.sym(name);
        if let Node::Symbol(s) = ctx.node(id) {
            env.insert(*s, Complex64::new(*x, 0.0));
        }
    }
    env
}

#[test]
fn cswitch_on_and_off() {
    // The current-controlled switch lowers through `lower_behavioral` like
    // every device; its conductance interpolates in log space over the control
    // current window.
    let mut ctx = Graph::new();
    let (a, b) = (ctx.sym("a"), ctx.sym("b"));
    let (ad, bd) = (ctx.sym("ad"), ctx.sym("bd"));
    let ic = ctx.sym("i_V1");
    let mut lo = Lowerer::new(&mut ctx);
    let frag = CSwitch::new("W1", "V1").lower_behavioral(&mut lo, &[a, b], &[ad, bd], &[ic]);
    assert!(frag.residuals.is_empty(), "no extra unknowns");
    let i = frag.terminal_currents.clone();
    drop(lo);

    let pars = [
        ("a", 2.0),
        ("b", 0.0),
        ("W1.Ron", 10.0),
        ("W1.Roff", 1e6),
        ("W1.It", 0.5e-3),
        ("W1.Ih", 1e-6),
    ];
    // On: control current 1 mA >> It+Ih -> fully on, i = (Va-Vb)/Ron = 0.2.
    let mut on = pars.to_vec();
    on.push(("i_V1", 1e-3));
    let env = env_of(&mut ctx, &on);
    assert!((eval(&ctx, &[i[0]], &env)[0].re - 0.2).abs() < 1e-9);
    // Off: zero control current -> fully off, i = (Va-Vb)/Roff = 2e-6.
    let mut off = pars.to_vec();
    off.push(("i_V1", 0.0));
    let env = env_of(&mut ctx, &off);
    assert!((eval(&ctx, &[i[0]], &env)[0].re - 2e-6).abs() < 1e-12);
    // KCL closes.
    let s = eval(&ctx, &[i[0]], &env)[0].re + eval(&ctx, &[i[1]], &env)[0].re;
    assert!(s.abs() < 1e-15);
}

#[test]
fn safe_exp_is_exact_below_and_linear_above_the_knot() {
    let vc = sane_core::constants::EXP_VCRIT;
    let mut ctx = Graph::new();
    let x = ctx.sym("x");
    let e = safe_exp(&mut ctx, x);
    let at = |ctx: &mut Graph, xv: f64| {
        let env = env_of(ctx, &[("x", xv)]);
        eval(ctx, &[e], &env)[0].re
    };
    // Exact below the knot.
    assert!((at(&mut ctx, 1.5) - 1.5_f64.exp()).abs() < 1e-12 * 1.5_f64.exp());
    // Linear (finite) above it: exp(vc)*(1 + (x - vc)).
    let want = vc.exp() * (1.0 + 10.0);
    assert!((at(&mut ctx, vc + 10.0) - want).abs() < 1e-9 * want);
}
