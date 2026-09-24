//! Full pipeline from a netlist string: parse -> DAE assembly -> small-signal
//! transfer by Cramer's rule, verified numerically against the analytic RC
//! lowpass response. (This replaced the standalone MNA assembler with the live
//! `dae` small-signal path.)

use std::collections::HashMap;

use num_complex::Complex64;
use rsdag::{eval, Node, SymbolId};
use sane_core::Graph;
use sane_dae::{assemble_dae, small_signal_transfer};
use sane_netlist::parse;

#[test]
fn rc_lowpass_from_netlist() {
    let src = "\
* RC lowpass
V1 in 0 1
R1 in out 1k
C1 out 0 1u
.end";

    let parsed = parse(src).expect("parse");
    let out = parsed.node("out").expect("out node");

    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
    // The output node id maps to the unknown `v{id}`.
    let out_u = format!("v{out}");
    let h = small_signal_transfer(&mut ctx, &dae, "V1", &out_u).expect("transfer exists");

    // Bind every free symbol: component values from the netlist, plus s = jw.
    let r = parsed.values["R1"];
    let c = parsed.values["C1"];

    let bind = |ctx: &mut Graph, name: &str| -> SymbolId {
        let id = ctx.sym(name);
        match ctx.node(id) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        }
    };
    let rid = bind(&mut ctx, "R1");
    let cid = bind(&mut ctx, "C1");
    let sid = bind(&mut ctx, "s");

    for &f in &[10.0, 159.155, 1000.0, 10_000.0] {
        let omega = 2.0 * std::f64::consts::PI * f;
        let sval = Complex64::new(0.0, omega);
        let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
        env.insert(rid, Complex64::new(r, 0.0));
        env.insert(cid, Complex64::new(c, 0.0));
        env.insert(sid, sval);

        let got = eval(&ctx, &[h], &env)[0];
        let expected = Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + sval * r * c);
        assert!((got - expected).norm() < 1e-9, "f={f}: {got} vs {expected}");
    }
}
