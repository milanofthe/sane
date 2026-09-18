//! Operating-point variables: `(* desc *)`-annotated Verilog-A variables are
//! exported per instance and evaluated at the solved point via
//! `OperatingPoint::op_vars` (the compact-model OPP idiom: gm, vth, ids, ...).

use sane_analysis::Model;

const DECK_HEAD: &str = "\
.veriloga
module vares(a, b);
  inout a, b; electrical a, b;
  (* type=\"instance\", units=\"Ohm\", desc=\"resistance\" *) parameter real R = 1000.0;
  (* desc=\"branch current\", units=\"A\" *) real ival;
  (* desc=\"dissipated power\" *) real pwr;
  real scratch;
  analog begin
    ival = V(a,b) / R;
    pwr = V(a,b) * ival;
    scratch = 2.0 * pwr;
    I(a,b) <+ ival;
  end
endmodule
.endveriloga
";

/// Single instance: both op-vars appear instance-qualified with desc/units and
/// evaluate to the analytic values (2 V over 1 kOhm: 2 mA, 4 mW).
#[test]
fn opvars_evaluate_at_dc_point() {
    let deck = format!("* opvar readout\n{DECK_HEAD}V1 in 0 2.0\nN1 in 0 vares\n.end\n");
    let model = Model::from_netlist(&deck).expect("model");
    let op = model.operating_point(&[]).expect("dc");
    let vars = op.op_vars();
    assert_eq!(vars.len(), 2, "ival and pwr exported, scratch not");
    let ival = vars.iter().find(|v| v.name == "N1.ival").expect("N1.ival");
    assert_eq!(ival.desc, "branch current");
    assert_eq!(ival.units.as_deref(), Some("A"));
    assert!((ival.value - 2e-3).abs() < 1e-12, "ival = {}", ival.value);
    let pwr = vars.iter().find(|v| v.name == "N1.pwr").expect("N1.pwr");
    assert!(pwr.units.is_none());
    assert!((pwr.value - 4e-3).abs() < 1e-12, "pwr = {}", pwr.value);
}

/// Template clones: the second instance goes through the template-instantiate
/// path (substituted expressions), so its op-vars must bind its own terminals
/// and its own instance parameter value.
#[test]
fn opvars_per_instance_through_template_clones() {
    let deck = format!(
        "* opvar clones\n{DECK_HEAD}\
         V1 in 0 2.0\nN1 in 0 vares\nN2 in 0 vares R=500\n.end\n"
    );
    let model = Model::from_netlist(&deck).expect("model");
    let op = model.operating_point(&[]).expect("dc");
    let vars = op.op_vars();
    assert_eq!(vars.len(), 4, "two op-vars per instance");
    let get = |n: &str| vars.iter().find(|v| v.name == n).map(|v| v.value).expect(n);
    assert!((get("N1.ival") - 2e-3).abs() < 1e-12);
    assert!(
        (get("N2.ival") - 4e-3).abs() < 1e-12,
        "R=500 doubles the current"
    );
    assert!((get("N1.pwr") - 4e-3).abs() < 1e-12);
    assert!((get("N2.pwr") - 8e-3).abs() < 1e-12);
}
