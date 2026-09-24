//! `.nodeset` symmetry breaking. A pair of cross-coupled behavioural inverters
//! `a = g(b)`, `b = g(a)` with a steep decreasing sigmoid `g` is bistable: the
//! symmetric root `a = b = 0.5` is an unstable saddle, flanked by two stable
//! asymmetric roots. A cold Newton from the symmetric start `x = 0` stays on the
//! `a = b` line and converges to the saddle (classic latch metastability); a
//! node-set pins one node, breaks the symmetry, and the solve falls into a
//! stable branch.

use sane_core::Graph;
use sane_dae::{assemble_dae, Dae};
use sane_netlist::{parse, ParsedCircuit};
use sane_solve::{CompiledDc, Convergence};

// Cross-coupled inverters with *high-impedance* outputs (a unit load resistor on
// each node, driven by a behavioural current source from the opposite node), so
// KCL gives V(a) = g(V(b)), V(b) = g(V(a)) with
//   g(x) = 0.5 - (1/pi)*atan(K*(x-0.5)),  decreasing, g(0.5) = 0.5.
// With K=10 the composed map g(g(.)) has slope (K/pi)^2 > 1 at 0.5, so the
// symmetric root is an unstable saddle. The finite node impedance is what lets a
// stiff node-set pin override the node (an ideal voltage source could not be).
const LATCH: &str = "\
R1 a 0 1
R2 b 0 1
B1 0 a I=0.5 - 0.3183098862*atan(10*(V(b)-0.5))
B2 0 b I=0.5 - 0.3183098862*atan(10*(V(a)-0.5))
.end";

fn build() -> (Graph, ParsedCircuit, Dae, CompiledDc, Vec<f64>) {
    let parsed = parse(LATCH).expect("parse");
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p: Vec<f64> = parsed.pvec(&cdc.param_names(&ctx));
    (ctx, parsed, dae, cdc, p)
}

fn node_idx(parsed: &ParsedCircuit, dae: &Dae, node: &str) -> usize {
    let k = parsed.node(node).expect("node");
    dae.unknowns
        .iter()
        .position(|u| *u == format!("v{k}"))
        .expect("node unknown")
}

#[test]
fn cold_solve_lands_on_the_unstable_symmetric_root() {
    let (_ctx, parsed, dae, cdc, p) = build();
    let (ia, ib) = (node_idx(&parsed, &dae, "a"), node_idx(&parsed, &dae, "b"));
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-10, 100);
    assert!(conv, "cold DC should converge");
    // Symmetric start -> symmetric (metastable) root a = b = 0.5.
    assert!((x[ia] - 0.5).abs() < 1e-6, "a = {} (expected ~0.5)", x[ia]);
    assert!((x[ib] - 0.5).abs() < 1e-6, "b = {} (expected ~0.5)", x[ib]);
}

#[test]
fn nodeset_breaks_symmetry_onto_a_stable_branch() {
    let (_ctx, parsed, dae, cdc, p) = build();
    let (ia, ib) = (node_idx(&parsed, &dae, "a"), node_idx(&parsed, &dae, "b"));

    // Pin a high: the solve must land on the a-high / b-low stable branch.
    let (xh, ch, _) = cdc.solve_dc_nodeset(&p, &[(ia, 1.0)], Convergence::from_tol(1e-10), 100);
    assert!(ch, "node-set DC should converge");
    assert!(xh[ia] > 0.6, "a = {} (expected high branch)", xh[ia]);
    assert!(xh[ib] < 0.4, "b = {} (expected low branch)", xh[ib]);

    // Pin a low: the mirror branch (a low, b high).
    let (xl, cl, _) = cdc.solve_dc_nodeset(&p, &[(ia, 0.0)], Convergence::from_tol(1e-10), 100);
    assert!(cl, "node-set DC should converge");
    assert!(xl[ia] < 0.4, "a = {} (expected low branch)", xl[ia]);
    assert!(xl[ib] > 0.6, "b = {} (expected high branch)", xl[ib]);

    // The two branches must be distinct and mirror-symmetric.
    assert!((xh[ia] - xl[ib]).abs() < 1e-3, "branches should mirror");
}
