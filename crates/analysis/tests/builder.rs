//! A circuit built by name is the circuit the netlist states: the same
//! model, the same results, by the same names.

use std::sync::Arc;

use sane_analysis::{Model, TransientOptions};
use sane_circuit::{Circuit, Waveform};

fn op(model: &Model, binding: &[(&str, f64)], node: &str) -> f64 {
    let op = (model.at(binding))
        .and_then(|pt| pt.operating_point())
        .expect("operating point");
    op.get(node).unwrap_or_else(|| panic!("no {node}"))
}

fn close(got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol * (1.0 + want.abs()),
        "got {got}, want {want}"
    );
}

#[test]
fn a_built_divider_is_the_parsed_one() {
    let mut c = Circuit::new();
    c.voltage_source("V1", "in", "0", 5.0)
        .resistor("R1", "in", "out", 1e3)
        .resistor("R2", "out", "gnd", 3e3);
    let built = Model::new(c).unwrap();
    let parsed = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 3k\n").unwrap();
    assert_eq!(built.node_names(), parsed.node_names());
    close(op(&built, &[], "out"), 3.75, 1e-8);
    close(op(&built, &[("R1", 3e3)], "out"), 2.5, 1e-8);
    close(op(&built, &[], "out"), op(&parsed, &[], "out"), 1e-8);
}

#[test]
fn a_built_diode_is_the_parsed_one() {
    let mut c = Circuit::new();
    c.voltage_source("V1", "in", "0", 1.0)
        .resistor("R1", "in", "a", 1e3);
    c.diode("D1", "a", "0", &[("is", 2e-14), ("N", 1.1)])
        .unwrap();
    let built = Model::new(c).unwrap();
    let parsed =
        Model::from_netlist("V1 in 0 1\nR1 in a 1k\nD1 a 0 dm\n.model dm D(Is=2e-14 N=1.1)\n")
            .unwrap();
    close(op(&built, &[], "a"), op(&parsed, &[], "a"), 1e-9);
    assert!(Circuit::new()
        .diode("D1", "a", "0", &[("bogus", 1.0)])
        .is_err());
    assert!(Circuit::new()
        .device("M1", "sane_mos", &["d", "g"], &[])
        .is_err());
}

#[test]
fn a_built_subcircuit_is_the_parsed_one() {
    let mut half = Circuit::subckt("half", &["in", "out"]);
    half.resistor("R1", "in", "mid", 1e3)
        .resistor("R2", "mid", "out", 1e3)
        .resistor("R3", "out", "0", 2e3);
    let half = Arc::new(half);
    let mut c = Circuit::new();
    c.voltage_source("V1", "a", "0", 8.0);
    c.instance("X1", &half, &["a", "b"]).unwrap();
    c.instance("X2", &half, &["b", "c"]).unwrap();
    let built = Model::new(c.clone()).unwrap();
    let parsed = Model::from_netlist(concat!(
        ".subckt half in out\n",
        "R1 in mid 1k\nR2 mid out 1k\nR3 out 0 2k\n",
        ".ends\n",
        "V1 a 0 8\nX1 a b half\nX2 b c half\n",
    ))
    .unwrap();
    // b sees X1's R3 (2k) parallel to X2's 4k
    for (node, want) in [("b", 3.2), ("c", 1.6), ("X1.mid", 5.6), ("X2.mid", 2.4)] {
        close(op(&built, &[], node), want, 1e-8);
        close(op(&parsed, &[], node), want, 1e-8);
    }
    // an instance's value by its name: X2's R3 at 1k makes it 3k
    close(op(&built, &[("X2.R3", 1e3)], "b"), 3.0, 1e-8);
    c.set("X2.R3", 1e3);
    close(op(&Model::new(c).unwrap(), &[], "c"), 1.0, 1e-8);
    // the pins and the nodes they are wired to agree in number
    assert!(Circuit::new().instance("X1", &half, &["a"]).is_err());
}

#[test]
fn a_built_waveform_is_the_parsed_one() {
    let mut c = Circuit::new();
    let sin = Waveform::Sin {
        offset: 0.5,
        amplitude: 1.0,
        freq: 1e3,
    };
    c.voltage_source("V1", "in", "0", sin)
        .resistor("R1", "in", "out", 1e3)
        .capacitor("C1", "out", "0", 1e-7);
    let built = Model::new(c).unwrap();
    let parsed =
        Model::from_netlist("V1 in 0 SIN(0.5 1 1k)\nR1 in out 1k\nC1 out 0 100n\n").unwrap();
    let t: Vec<f64> = (0..=40).map(|k| k as f64 * 5e-5).collect();
    let opts = TransientOptions::default();
    let run = |m: &Model| {
        m.at(&[])
            .unwrap()
            .transient(&t, &opts)
            .unwrap()
            .signal("out")
            .unwrap()
    };
    for (a, b) in run(&built).iter().zip(run(&parsed)) {
        close(*a, b, 1e-9);
    }
}

const COLLAPSIBLE: &str = "
module sw(a, b);
  inout a, b; electrical a, b, ai;
  parameter real R = 1.0;
  analog begin
    if (R > 0)
      I(a, ai) <+ V(a, ai) / R;
    else
      V(a, ai) <+ 0.0;
    I(ai, b) <+ V(ai, b);
  end
endmodule
";

/// A built circuit sets itself up anew where a binding crosses a device's
/// structure, as a parsed one does.
#[test]
fn a_built_circuit_restructures() {
    let mut c = Circuit::new();
    c.module(COLLAPSIBLE).unwrap();
    c.voltage_source("V1", "a", "0", 1.0);
    c.device("N1", "sw", &["a", "0"], &[("R", 0.0)]).unwrap();
    let model = Model::new(c).unwrap();
    let current = |b: &[(&str, f64)]| {
        let op = model.at(b).and_then(|pt| pt.operating_point()).unwrap();
        -op.vector()[model.resolve("V1").unwrap()]
    };
    close(current(&[]), 1.0, 1e-8);
    close(current(&[("N1.R", 1.0)]), 0.5, 1e-8);
    close(current(&[]), 1.0, 1e-8);
}

/// The model hands its circuit back to build on.
#[test]
fn a_model_circuit_builds_on() {
    let mut c = Circuit::new();
    c.voltage_source("V1", "in", "0", 2.0)
        .resistor("R1", "in", "out", 1e3);
    let first = Model::new(c).unwrap();
    let mut more = (**first.circuit().unwrap()).clone();
    more.resistor("R2", "out", "0", 1e3).nodeset("out", 1.0);
    let second = Model::new(more).unwrap();
    close(op(&first, &[], "out"), 2.0, 1e-8);
    close(op(&second, &[], "out"), 1.0, 1e-8);
}
