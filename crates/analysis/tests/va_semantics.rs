//! Verilog-A constructs that lowered to a different value than the language
//! gives them, each against the value it must have.

use sane_analysis::Model;

/// The current a one-port module draws at 1 V, its analog block in `body`.
fn current(decls: &str, body: &str) -> f64 {
    let deck = format!(
        ".veriloga\nmodule m(a, b);\n  inout a, b; electrical a, b;\n{decls}\n  \
         analog begin\n{body}\n  end\nendmodule\n.endveriloga\nV1 a 0 1\nN1 a 0 m\n.end\n"
    );
    let model = Model::from_netlist(&deck).expect("model");
    let op = model
        .at(&[])
        .and_then(|pt| pt.operating_point())
        .expect("dc");
    -op.vector()[model.resolve("V1").unwrap()]
}

/// Equal up to the solver's gmin conductance.
fn close(got: f64, want: f64) {
    assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
}

#[test]
fn pow_of_a_negative_base_with_an_integer_exponent() {
    let decls = "  parameter real c = -2.0;";
    close(current(decls, "I(a, b) <+ pow(c, 3) * V(a, b);"), -8.0);
    close(current(decls, "I(a, b) <+ c ** 2 * V(a, b);"), 4.0);
    close(current(decls, "I(a, b) <+ pow(c, -1) * V(a, b);"), -0.5);
}

#[test]
fn pow_of_a_probe_with_an_integer_exponent() {
    // (V - 2)^3 at V = 1 is -1, not exp(3 ln(-1)).
    let i = current("", "I(a, b) <+ 1e-3 * pow(V(a, b) - 2.0, 3) + V(a, b);");
    close(i, 1.0 - 1e-3);
}

#[test]
fn declaration_initializers() {
    let decls = "  parameter real g = 2.0;\n  real x = 0.5;\n  real y = g * 3.0;";
    close(current(decls, "I(a, b) <+ (x + y) * V(a, b);"), 6.5);
    // A block-local initializer runs where it is declared.
    let body = "begin : blk\n real z = 4.0;\n I(a, b) <+ z * V(a, b);\nend";
    close(current("", body), 4.0);
}

#[test]
fn initializer_in_an_analog_function() {
    let decls = "  analog function real f;\n    input u; real u;\n    real k = 3.0;\n    \
                 f = k * u;\n  endfunction";
    close(current(decls, "I(a, b) <+ f(V(a, b));"), 3.0);
}

#[test]
fn array_element_assignment_is_refused() {
    let deck = ".veriloga\nmodule m(a, b);\n  inout a, b; electrical a, b;\n  real x[0:1];\n  \
                analog begin\n    x[1] = 2.0;\n    I(a, b) <+ V(a, b);\n  end\nendmodule\n\
                .endveriloga\nV1 a 0 1\nN1 a 0 m\n.end\n";
    let err = Model::from_netlist(deck).err().expect("refused");
    assert!(err.to_string().contains("array"), "{err}");
}
