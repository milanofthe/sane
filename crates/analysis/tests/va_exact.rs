//! The Verilog-A lowering is exact in the parameters: a model built at one
//! value of a parameter its module branches on gives, set or overridden to
//! another, what a deck written with that value gives. Loops over a
//! parameter and `$error`s on parameter values hold as assertions.

use sane_analysis::{Model, ModelError};

const SWITCHED: &str = "\
.veriloga
module sw(a, b);
  inout a, b; electrical a, b;
  parameter real k = 1.0;
  analog begin
    if (k > 0)
      I(a, b) <+ k * V(a, b);
    else
      I(a, b) <+ 2.0 * V(a, b);
  end
endmodule
.endveriloga
";

/// The current module `body` draws at 1 V, its parameters `params`.
fn deck(module: &str, params: &str) -> String {
    format!("{module}V1 a 0 1\nN1 a 0 sw {params}\n.end\n")
}

fn current(model: &Model, overrides: &[(&str, f64)]) -> Result<f64, ModelError> {
    let op = model.operating_point(overrides)?;
    Ok(-op.vector()[model.resolve("V1").unwrap()])
}

fn close(got: f64, want: f64) {
    assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
}

#[test]
fn override_across_a_branch() {
    let model = Model::from_netlist(&deck(SWITCHED, "k=1")).expect("model");
    close(current(&model, &[]).unwrap(), 1.0);
    close(current(&model, &[("N1.k", -1.0)]).unwrap(), 2.0);
    close(current(&model, &[("N1.k", 3.0)]).unwrap(), 3.0);
}

#[test]
fn set_across_a_branch() {
    let model = Model::from_netlist(&deck(SWITCHED, "k=-1")).expect("model");
    close(current(&model, &[]).unwrap(), 2.0);
    model.set("N1.k", 0.5).expect("k is a parameter whatever its branch");
    close(current(&model, &[]).unwrap(), 0.5);
}

#[test]
fn subcircuit_instances_across_a_branch() {
    let d = format!(
        "{SWITCHED}.subckt cell p q k=1\nN1 p q sw k={{k}}\n.ends\n\
         V1 a 0 1\nX1 a 0 cell k=1\nV2 b 0 1\nX2 b 0 cell k=-1\n.end\n"
    );
    let model = Model::from_netlist(&d).expect("model");
    let op = model.operating_point(&[]).expect("dc");
    close(-op.vector()[model.resolve("V1").unwrap()], 1.0);
    close(-op.vector()[model.resolve("V2").unwrap()], 2.0);
}

const LOOP: &str = "\
.veriloga
module sw(a, b);
  inout a, b; electrical a, b;
  parameter real n = 3;
  integer i;
  real g;
  analog begin
    g = 0.0;
    for (i = 0; i < n; i = i + 1)
      g = g + 1.0;
    I(a, b) <+ g * V(a, b);
  end
endmodule
.endveriloga
";

#[test]
fn a_loop_over_a_parameter_is_exact_or_refused() {
    let model = Model::from_netlist(&deck(LOOP, "n=3")).expect("model");
    close(current(&model, &[]).unwrap(), 3.0);
    close(current(&model, &[("N1.n", 10.0)]).unwrap(), 10.0);
    close(current(&model, &[("N1.n", 0.0)]).unwrap(), 0.0);
    let past = current(&model, &[("N1.n", 100.0)]);
    assert!(
        matches!(&past, Err(ModelError::Invalid(m)) if m.contains("loop")),
        "{past:?}"
    );
}

const GUARDED: &str = "\
.veriloga
module sw(a, b);
  inout a, b; electrical a, b;
  parameter real R = 1.0;
  analog begin
    if (R <= 0)
      $error(\"R must be positive\");
    I(a, b) <+ V(a, b) / R;
  end
endmodule
.endveriloga
";

#[test]
fn an_error_on_parameter_values_rejects_them() {
    let model = Model::from_netlist(&deck(GUARDED, "R=0.5")).expect("model");
    close(current(&model, &[]).unwrap(), 2.0);
    let bad = current(&model, &[("N1.R", -1.0)]);
    assert!(
        matches!(&bad, Err(ModelError::Invalid(m)) if m.contains("R must be positive")),
        "{bad:?}"
    );
    let built_bad = Model::from_netlist(&deck(GUARDED, "R=-1")).expect("model");
    assert!(matches!(current(&built_bad, &[]), Err(ModelError::Invalid(_))));
}

const COLLAPSIBLE: &str = "\
.veriloga
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
.endveriloga
";

/// A condition on parameters that shorts a branch or not is the topology:
/// decided at setup (a shorted branch costs no unknown), and set up anew
/// where a binding crosses it, overridden or set, the results in the
/// model's own layout.
#[test]
fn a_binding_across_a_topology_sets_it_up_anew() {
    let shorted = Model::from_netlist(&deck(COLLAPSIBLE, "R=0")).expect("model");
    let open = Model::from_netlist(&deck(COLLAPSIBLE, "R=1")).expect("model");
    assert_eq!(shorted.dim() + 1, open.dim(), "the short merges ai into a");
    close(current(&shorted, &[]).unwrap(), 1.0);
    close(current(&open, &[]).unwrap(), 0.5);
    // Across, both ways, and back.
    close(current(&shorted, &[("N1.R", 1.0)]).unwrap(), 0.5);
    close(current(&shorted, &[("N1.R", 3.0)]).unwrap(), 0.25);
    close(current(&open, &[("N1.R", 0.0)]).unwrap(), 1.0);
    close(current(&shorted, &[]).unwrap(), 1.0);
    // A node only the other structure has is its own there, by name; the
    // layout stays the model's.
    let op = shorted.operating_point(&[("N1.R", 1.0)]).unwrap();
    close(op.get("N1.ai").unwrap(), 0.5);
    assert_eq!(op.vector().len(), shorted.dim());
    assert_eq!(
        shorted.resolve("N1.ai"),
        None,
        "collapsed: no index of its own"
    );
    let op = open.operating_point(&[("N1.R", 0.0)]).unwrap();
    close(op.get("N1.ai").unwrap(), 1.0);
    close(op.vector()[open.resolve("N1.ai").unwrap()], 1.0);
    // Set across, and back.
    shorted.set("N1.R", 2.0).expect("a value like any");
    close(current(&shorted, &[]).unwrap(), 1.0 / 3.0);
    shorted.set("N1.R", 0.0).unwrap();
    close(current(&shorted, &[]).unwrap(), 1.0);
}

/// A sweep and a transient across the topology: each point, each run, in
/// the structure its binding has.
#[test]
fn sweeps_and_transients_cross_a_topology() {
    let model = Model::from_netlist(&deck(COLLAPSIBLE, "R=0")).expect("model");
    let sweep = model.dc_sweep("N1.R", 0.0, 2.0, 1.0).expect("sweep");
    let i = sweep.signal("V1").expect("the source current");
    let want = [1.0, 0.5, 1.0 / 3.0];
    for (got, want) in i.iter().zip(want) {
        close(-got, want);
    }
    let ai = sweep.signal("N1.ai").expect("ai, by name");
    close(ai[0], 1.0); // collapsed onto a
    close(ai[1], 0.5);
    let tr = model
        .transient(
            sane_solve::TransientMethod::default(),
            &[("N1.R", 1.0)],
            &[0.0, 1e-6],
            1e-6,
            1e-9,
        )
        .expect("transient");
    close(-tr.signal("V1").unwrap()[1], 0.5);
    close(tr.signal("N1.ai").unwrap()[1], 0.5);
    assert_eq!(tr.rows()[1].len(), model.dim());
}

const MODE: &str = "\
.veriloga
module sw(a, b);
  inout a, b; electrical a, b;
  parameter integer mode = 0 from [0:1];
  analog begin
    if (mode == 0)
      I(a, b) <+ V(a, b);
    else
      I(a, b) <+ 3.0 * V(a, b);
  end
endmodule
.endveriloga
";

/// An integer parameter is a mode of the model, its structure: another
/// mode, set or overridden, is set up anew. A raw evaluation at a state of
/// this model refuses it, the state not describing the other mode's circuit.
#[test]
fn an_integer_mode_is_structure() {
    let model = Model::from_netlist(&deck(MODE, "mode=0")).expect("model");
    close(current(&model, &[]).unwrap(), 1.0);
    close(current(&model, &[("N1.mode", 1.0)]).unwrap(), 3.0);
    let x = model.operating_point(&[]).unwrap().vector().to_vec();
    let p = model.pvec(&[("N1.mode", 1.0)]);
    let raw = model.currents(x, p, 0.0);
    assert!(
        matches!(&raw, Err(ModelError::Invalid(m)) if m.contains("structural")),
        "{raw:?}"
    );
    model.set("N1.mode", 1.0).unwrap();
    close(current(&model, &[]).unwrap(), 3.0);
}
