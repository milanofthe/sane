//! Subcircuit instances are calls of one function per body structure: the
//! instances of a subcircuit share it whatever their parameter values, and
//! solve, differentiate and name like the elements they contain.

use sane_analysis::Model;

const DECK: &str = "\
.subckt half a b
R1 a mid {r}
R2 mid b {r}
.ends
.subckt quarter p q
X1 p m half r=1k
X2 m q half r=1k
.ends
V1 in 0 4
X1 in out half r=1k
X2 out 0 half r=3k
X3 in 0 quarter
.end";

#[test]
fn instances_share_one_function_per_body() {
    let parsed = sane_netlist::parse(DECK).unwrap();
    let mut ctx = sane_core::Graph::new();
    let dae = parsed.assemble(&mut ctx);
    // `half` (all four instances, two nested in `quarter`) and `quarter`,
    // each its residuals and its observers (the resistors' noise)
    assert_eq!(ctx.n_funcs(), 4);
    // every node of the hierarchy is a node unknown, internal nodes included
    for node in ["out", "X1.mid", "X2.mid", "X3.m", "X3.X1.mid", "X3.X2.mid"] {
        assert!(parsed.node(node).is_some(), "{node}");
    }
    assert_eq!(dae.n_nodes, parsed.node_names.len() - 1);
}

#[test]
fn instances_solve_and_differentiate_by_their_names() {
    let m = Model::from_netlist(DECK).unwrap();
    let op = m.operating_point(&[]).unwrap();
    // 6k under 2k from 4 V (to the solver's gmin shunts)
    let v = |n: &str| op.get(n).unwrap();
    assert!((v("out") - 3.0).abs() < 1e-6);
    assert!((v("X1.mid") - 3.5).abs() < 1e-6);
    assert!((v("X3.m") - 2.0).abs() < 1e-6);
    assert!((v("X3.X2.mid") - 1.0).abs() < 1e-6);
    // each instance keeps its own parameters
    assert_eq!(m.get("X2.R1"), Some(3e3));
    assert_eq!(m.get("X3.X1.R2"), Some(1e3));
    let s = op.sensitivity("out").unwrap();
    let grad = |p: &str| s.grad[s.names.iter().position(|n| n == p).unwrap()];
    // out = 4 * R_X2 / (R_X1 + R_X2), R_X2 = X2.R1 + X2.R2 = 6k, R_X1 = 2k
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6 * b.abs();
    assert!(close(grad("X2.R1"), 4.0 * 2e3 / 8e3_f64.powi(2)));
    assert!(close(grad("X1.R2"), -4.0 * 6e3 / 8e3_f64.powi(2)));
    assert!(grad("X3.X1.R1").abs() < 1e-15);
}
