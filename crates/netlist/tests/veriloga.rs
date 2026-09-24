//! Inline Verilog-A models in a deck: a `.veriloga ... .endveriloga` block
//! defines modules, and `N` instances place them. They must parse, bind their
//! parameters (instance overrides + module defaults), and assemble into a DAE.

use sane_core::Graph;
use sane_dae::assemble_dae;
use sane_netlist::parse;

const DECK: &str = r#"
* Verilog-A diode in a deck
.veriloga
module diode(a, c);
  inout a, c;
  electrical a, c;
  parameter real Is = 1e-14;
  parameter real N  = 1.0;
  parameter real Vt = 0.025852;
  analog I(a, c) <+ Is * (limexp(V(a,c)/(N*Vt)) - 1.0);
endmodule
.endveriloga
V1 1 0 0.7
R1 1 2 1k
N1 2 0 diode Is=2e-14
.end
"#;

#[test]
fn inline_veriloga_diode_parses_and_binds() {
    let p = parse(DECK).expect("deck parses");
    assert_eq!(p.devices.len(), 1, "one VA device");
    // Instance override and module default both bound under the instance name.
    assert_eq!(p.param_value("N1.Is"), Some(2e-14), "instance override");
    assert_eq!(p.param_value("N1.N"), Some(1.0), "module default");
    assert_eq!(p.param_value("N1.Vt"), Some(0.025852), "module default");
}

#[test]
fn inline_veriloga_diode_assembles_dae() {
    let p = parse(DECK).expect("deck parses");
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &p.circuit, &p.devices);
    // nodes v1, v2 + branch i_V1.
    assert!(dae.dim() >= 3, "dim {}", dae.dim());
    assert!(dae.unknowns.iter().any(|u| u == "v2"));
}

#[test]
fn unknown_veriloga_model_errors() {
    let deck = "N1 1 0 nosuchmodel\n.end\n";
    let e = parse(deck).unwrap_err();
    assert!(e.msg.contains("unknown veriloga model"), "msg: {}", e.msg);
}

#[test]
fn port_count_mismatch_errors() {
    let deck = "\
.veriloga
module r2(a,b); inout a,b; electrical a,b; parameter real R=1k; analog I(a,b)<+V(a,b)/R; endmodule
.endveriloga
N1 1 0 2 r2
.end
";
    let e = parse(deck).unwrap_err();
    assert!(e.msg.contains("ports"), "msg: {}", e.msg);
}

// WP6: a switch branch (voltage- vs current-defined depending on a parameter)
// must lower by const-folding the gating parameter at the instance.
const SWITCH: &str = r#"
.veriloga
module sw_r(a, b);
  inout a, b; electrical a, b;
  parameter real R = 1k;
  parameter real shorted = 0;
  analog begin
    if (shorted > 0) V(a,b) <+ 0.0;
    else I(a,b) <+ V(a,b)/R;
  end
endmodule
.endveriloga
V1 1 0 1
N1 1 2 sw_r shorted=1
N2 2 0 sw_r shorted=0
.end
"#;

#[test]
fn switch_branch_lowers_per_instance() {
    let p = parse(SWITCH).expect("parse");
    assert_eq!(p.devices.len(), 2);
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &p.circuit, &p.devices);
    // The shorted instance mints a branch-current unknown (voltage source);
    // the resistor instance mints none -> at least one extra branch unknown.
    assert!(dae.dim() >= 3, "dim {}", dae.dim());
}

// WP6: multiple modules in one .veriloga block, each instantiable.
const MULTI: &str = r#"
.veriloga
module ra(a, b); inout a,b; electrical a,b; parameter real R = 1k; analog I(a,b) <+ V(a,b)/R; endmodule
module rb(a, b); inout a,b; electrical a,b; parameter real G = 1m; analog I(a,b) <+ V(a,b)*G; endmodule
.endveriloga
V1 1 0 1
N1 1 2 ra
N2 2 0 rb
.end
"#;

#[test]
fn multiple_modules_per_block() {
    let p = parse(MULTI).expect("parse");
    assert_eq!(p.devices.len(), 2);
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &p.circuit, &p.devices);
    assert!(dae.unknowns.iter().any(|u| u == "v2"));
}

// Correctness: a model using an unsupported construct must fail at PARSE time
// (via the device's trial-lowering), not silently produce wrong results.
#[test]
fn unsupported_construct_errors_at_parse() {
    let deck = "\
.veriloga
module bad(a, b); inout a,b; electrical a,b;
analog I(a,b) <+ slew(V(a,b));
endmodule
.endveriloga
N1 1 0 bad
.end
";
    let e = parse(deck).unwrap_err();
    assert!(
        e.msg.contains("not supported"),
        "expected unsupported error, got: {}",
        e.msg
    );
}

#[test]
fn current_controlled_flow_probe_lowers() {
    // A current-controlled source: the probed input-branch current drives the
    // output. The probe promotes that branch to an explicit current unknown.
    let deck = "\
.veriloga
module cccs(in, out); inout in,out; electrical in,out;
parameter real G = 2.0; parameter real R = 1k;
analog begin I(in) <+ V(in)/R; I(out) <+ G * I(in); end
endmodule
.endveriloga
V1 in 0 1
N1 in out cccs
R2 out 0 1k
.end
";
    let p = parse(deck).expect("current-controlled model parses + lowers");
    assert_eq!(p.devices.len(), 1);
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &p.circuit, &p.devices);
    assert!(
        dae.unknowns.iter().any(|u| u.contains("flow_")),
        "promoted current unknown: {:?}",
        dae.unknowns
    );
}

// File-include form of `.veriloga`: load a real ECL-2.0 compact model straight
// from the corpus (its own directory becomes the `include` search path, so the
// model's companion header resolves). Environment-specific path -> ignored;
// run with `--ignored`.
#[test]
#[ignore]
fn veriloga_file_include_loads_real_ekv() {
    let deck = r#"
.veriloga "C:\Repositories\TEMP\VA-Models\code\ekv\vacode\ekv26.va"
* diode-connected EKV NMOS as a DC load
V1 d 0 1.0
N1 d d 0 0 ekv26_va W=10u L=1u
.end
"#;
    let p = parse(deck).expect("EKV file loads, instance binds + lowers");
    assert_eq!(p.devices.len(), 1, "one EKV device");
    assert!(
        (p.values["N1.W"] - 10e-6).abs() < 1e-15,
        "instance W override: {}",
        p.values["N1.W"]
    );
    assert!(
        (p.values["N1.L"] - 1e-6).abs() < 1e-15,
        "instance L override: {}",
        p.values["N1.L"]
    );
    assert_eq!(p.param_value("N1.VTO"), Some(0.5), "model default VTO");
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &p.circuit, &p.devices);
    // node d (index 1 -> "v1") + branch i_V1.
    assert!(dae.dim() >= 2, "dim {}", dae.dim());
    assert!(
        dae.unknowns.iter().any(|u| u == "v1"),
        "node present: {:?}",
        dae.unknowns
    );
}

// Compact-model import: a legacy `.model ... <type> level=<n>` card instantiated
// with `M` routes to a Verilog-A module bound via `.model_alias`, the standard
// PDK idiom (cards carry only a `level`, the physics lives in a `.va`). A tiny
// 4-port MOS-like module stands in for BSIM4 so the test is fast + deterministic.
fn compact_deck(extra: &str, inst: &str) -> String {
    format!(
        r#"
.veriloga
module mostest(d, g, s, b);
  inout d, g, s, b;
  electrical d, g, s, b;
  parameter real type = 1;
  parameter real L = 1;
  parameter real W = 1;
  parameter real gain = 1e-3;
  analog I(d, s) <+ type * gain * (W / L) * V(d, s);
endmodule
.endveriloga
.model_alias level=54 mostest
{extra}
V1 d 0 1
{inst}
.end
"#
    )
}

#[test]
fn compact_level_card_routes_to_va_via_model_alias() {
    let deck = compact_deck(
        ".model nch nmos level=54 version=4.5 gain=2e-3",
        "M1 d g s 0 nch L=2 W=8",
    );
    let p = parse(&deck).expect("compact M routes to VA module");
    assert_eq!(p.devices.len(), 1, "one routed VA device");
    // nmos type token -> module `type` = +1; geometry + card param bound.
    assert_eq!(p.param_value("M1.type"), Some(1.0), "nmos polarity");
    assert_eq!(p.param_value("M1.L"), Some(2.0));
    assert_eq!(p.param_value("M1.W"), Some(8.0));
    assert_eq!(p.param_value("M1.gain"), Some(2e-3), "card param bound");
    // It actually assembles into a DAE (the device lowered).
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &p.circuit, &p.devices);
    assert!(
        dae.unknowns.iter().any(|u| u == "v1"),
        "node d present: {:?}",
        dae.unknowns
    );
}

#[test]
fn compact_pmos_card_sets_negative_type() {
    let deck = compact_deck(".model pch pmos level=54", "M1 d g s 0 pch L=1 W=4");
    let p = parse(&deck).expect("pmos routes");
    assert_eq!(
        p.param_value("M1.type"),
        Some(-1.0),
        "pmos polarity -> type=-1"
    );
}

#[test]
fn compact_routing_applies_option_scale_to_geometry() {
    // `.option scale=2`: drawn L/W are scaled before the model sees them.
    let deck = compact_deck(
        ".option scale=2\n.model nch nmos level=54",
        "M1 d g s 0 nch L=2 W=8",
    );
    let p = parse(&deck).expect("scale routes");
    assert_eq!(p.param_value("M1.L"), Some(4.0), "L scaled by 2");
    assert_eq!(p.param_value("M1.W"), Some(16.0), "W scaled by 2");
}

#[test]
fn compact_m_routing_matches_n_element() {
    // The `M`-routed instance and an equivalent `N` instance of the same module
    // must bind identical parameters (M is just SPICE-idiomatic sugar over N).
    let m = compact_deck(
        ".model nch nmos level=54 gain=3e-3",
        "M1 d g s 0 nch L=2 W=8",
    );
    let n = compact_deck("", "N1 d g s 0 mostest type=1 gain=3e-3 L=2 W=8");
    let pm = parse(&m).expect("M");
    let pn = parse(&n).expect("N");
    for key in ["type", "L", "W", "gain"] {
        assert_eq!(
            pm.param_value(&format!("M1.{key}")),
            pn.param_value(&format!("N1.{key}")),
            "param {key} parity"
        );
    }
}

// PDK idiom end-to-end: a device-subckt wrapper (`sky130_fd_pr__nfet_01v8`)
// references a BINNED `level=54` card by base name; the instance carries
// HSPICE single-quoted geometry (`l='...'`) and `.option scale=1u`. The bin must
// be selected on the SCALED L/W (meters), and the quotes must resolve. This is
// the exact shape the AnalogGym SKY130 decks ingest natively.
#[test]
fn compact_device_subckt_binning_with_scale_and_quotes() {
    let deck = r#"
.veriloga
module mostest(d, g, s, b);
  inout d, g, s, b; electrical d, g, s, b;
  parameter real type = 1;
  parameter real L = 1; parameter real W = 1; parameter real gain = 1e-3;
  analog I(d, s) <+ type * gain * (W / L) * V(d, s);
endmodule
.endveriloga
.model_alias level=54 mostest
.option scale=1u
.model nch nmos level=54 lmin=1e-7 lmax=5e-7 wmin=1e-7 wmax=1e-4 gain=10e-3
.model nch nmos level=54 lmin=5e-7 lmax=1e-4 wmin=1e-7 wmax=1e-4 gain=20e-3
.subckt sky130_nfet d g s b l=1 w=1 m=1
M0 d g s b nch l={l} w={w} m={m}
.ends
.param WDES=4
V1 d 0 1
X1 d g 0 0 sky130_nfet l='2' w='WDES*1' m='1'
.end
"#;
    let p = parse(deck).expect("device-subckt + binning + scale + quotes parses");
    assert_eq!(p.devices.len(), 1, "one routed VA device");
    // scale=1u: drawn l=2 -> 2e-6 m, w=4 -> 4e-6 m (bound on the device).
    assert!(
        (p.values["X1.M0.L"] - 2e-6).abs() < 1e-18,
        "scaled L: {}",
        p.values["X1.M0.L"]
    );
    assert!(
        (p.values["X1.M0.W"] - 4e-6).abs() < 1e-18,
        "scaled W: {}",
        p.values["X1.M0.W"]
    );
    // L=2e-6 lands in the SECOND bin [5e-7,1e-4) -> gain=20e-3 (not the first
    // bin's 10e-3). Without scaling the bin select, raw L=2 matches no window.
    assert_eq!(
        p.param_value("X1.M0.gain"),
        Some(20e-3),
        "scaled-L bin selected"
    );
}

// Regression for the SKY130 corner card form: a model parameter given as a
// statistical (AGAUSS) mismatch expression must resolve to its nominal value
// when the mismatch switch is off and the slope symbols are defined to zero --
// otherwise the whole card param silently drops to the module default.
#[test]
fn compact_card_agauss_expression_resolves_to_nominal() {
    let deck = r#"
.veriloga
module mostest(d, g, s, b);
  inout d, g, s, b; electrical d, g, s, b;
  parameter real type = 1;
  parameter real L = 1; parameter real W = 1;
  parameter real toxe = 3e-9;
  analog I(d, s) <+ type * (W / L) * toxe * V(d, s);
endmodule
.endveriloga
.model_alias level=54 mostest
.param mc_mm_switch=0
.param l=1 w=1 mult=1
.param my_toxe_slope=0
.model nch nmos level=54 toxe={4.148e-09+MC_MM_SWITCH*AGAUSS(0,1.0,1)*(4.148e-09*1.0*(my_toxe_slope/sqrt(l*w*mult)))}
V1 d 0 1
M1 d g s 0 nch L=2 W=8
.end
"#;
    let p = parse(deck).expect("card with AGAUSS expression parses");
    let toxe = p.param_value("M1.toxe");
    assert_eq!(
        toxe,
        Some(4.148e-9),
        "toxe resolves to nominal (not module default 3e-9): {toxe:?}"
    );
}

#[test]
fn compact_level_without_alias_errors_with_hint() {
    // No `.model_alias` and the type token is not a module -> the clear error
    // must name both the `.model_alias` and `N`-element escape routes.
    let deck = "M1 d g s 0 nch L=1 W=1\n.model nch nmos level=54\nV1 d 0 1\n.end\n";
    let e = parse(deck).unwrap_err();
    assert!(
        e.msg.contains("model_alias"),
        "hint names .model_alias: {}",
        e.msg
    );
    assert!(e.msg.contains("compact model"), "msg: {}", e.msg);
}

#[test]
fn aliasparam_binds_target_parameter() {
    // `aliasparam res = R`: a deck setting `res=` must land on parameter `R`,
    // both from the instance line and from a `.model` card.
    let deck = "\
.veriloga
module vres(a,b);
  inout a,b; electrical a,b;
  parameter real R = 1k;
  aliasparam res = R;
  analog I(a,b) <+ V(a,b)/R;
endmodule
.endveriloga
V1 1 0 1
N1 1 0 vres res=2k
.end
";
    let p = parse(deck).expect("deck parses");
    assert_eq!(
        p.param_value("N1.R"),
        Some(2000.0),
        "alias binds the target"
    );
    assert!(
        !p.values.contains_key("N1.res"),
        "alias itself is not a parameter"
    );
    assert!(
        !p.report.unknown_params.iter().any(|u| u.contains("res")),
        "alias key must not be reported as unknown: {:?}",
        p.report.unknown_params
    );
}

#[test]
fn aliasparam_mfactor_sets_multiplicity() {
    // `aliasparam mult2 = $mfactor`: setting the alias scales the device like
    // the built-in `m=` multiplicity (node-1 KCL carries 2x the current).
    let module = "\
.veriloga
module vres(a,b);
  inout a,b; electrical a,b;
  parameter real R = 1k;
  aliasparam mult2 = $mfactor;
  analog I(a,b) <+ V(a,b)/R;
endmodule
.endveriloga
V1 1 0 1
";
    let base = format!("{module}N1 1 0 vres\n.end\n");
    let doubled = format!("{module}N1 1 0 vres mult2=2\n.end\n");
    let mut ctx = Graph::new();
    let p1 = parse(&base).expect("base parses");
    let d1 = assemble_dae(&mut ctx, &p1.circuit, &p1.devices);
    let p2 = parse(&doubled).expect("doubled parses");
    let d2 = assemble_dae(&mut ctx, &p2.circuit, &p2.devices);
    // Evaluate the node-1 KCL residual at the same point: the m=2 device
    // contributes exactly twice the branch current.
    let mut env = std::collections::HashMap::new();
    for (name, v) in [
        ("v1", 1.0),
        ("vdot1", 0.0),
        ("i_V1", 0.0),
        ("t", 0.0),
        ("N1.R", 1000.0),
    ] {
        let e = ctx.sym(name);
        if let rsdag::Node::Symbol(s) = ctx.node(e) {
            env.insert(*s, v);
        }
    }
    let r1: f64 = rsdag::eval(&ctx, &[d1.residuals[0]], &env)[0];
    let r2: f64 = rsdag::eval(&ctx, &[d2.residuals[0]], &env)[0];
    assert!(
        (r2 - 2.0 * r1).abs() < 1e-12,
        "m=2 doubles the flow: {r1} vs {r2}"
    );
}
