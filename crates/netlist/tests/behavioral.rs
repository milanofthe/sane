//! Behavioral (`B`) source robustness: unknown functions and wrong arity are
//! hard parse errors (never a silent zero), and the comparison operators + the
//! `if(cond, then, else)` conditional lower correctly through to the DC solve.

use sane_core::Graph;
use sane_dae::assemble_dae;
use sane_netlist::parse;
use sane_solve::CompiledDc;

/// Solve the DC operating point of a deck and return the value at node `node`.
fn dc_node(src: &str, node: &str) -> f64 {
    let parsed = parse(src).expect("parse");
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let pnames = cdc.param_names(&ctx);
    let p: Vec<f64> = parsed.pvec(&pnames);
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-10, 100);
    assert!(conv, "DC did not converge for:\n{src}");
    let k = parsed.node(node).expect("node");
    let idx = dae
        .unknowns
        .iter()
        .position(|u| *u == format!("v{k}"))
        .expect("node unknown");
    x[idx]
}

#[test]
fn unknown_function_is_a_hard_error() {
    let err = parse("V1 in 0 1\nB1 out 0 V=bogus(V(in))\n.end").unwrap_err();
    assert!(
        format!("{err}").contains("unknown function"),
        "expected an 'unknown function' error, got: {err}"
    );
}

#[test]
fn wrong_arity_is_a_hard_error() {
    let err = parse("V1 in 0 1\nB1 out 0 V=sin(1,2)\n.end").unwrap_err();
    assert!(
        format!("{err}").contains("argument"),
        "expected an arity error, got: {err}"
    );
}

#[test]
fn supported_functions_parse() {
    // sin/atan/floor (1-arg), pow/max (2-arg), if (3-arg), comparisons.
    for body in [
        "B1 out 0 V=sin(V(in)) + atan(V(in)) + floor(V(in))",
        "B1 out 0 V=pow(V(in), 2) + max(V(in), 1)",
        "B1 out 0 V=if(V(in) > 1.0, 2.0, 3.0)",
        "B1 out 0 I=if(V(in) >= 1.0, 1m, 0)",
    ] {
        let src = format!("V1 in 0 1\n{body}\n.end");
        assert!(parse(&src).is_ok(), "should parse: {body}");
    }
}

#[test]
fn conditional_selects_the_right_branch() {
    // V(in) = 2 > 1 -> the `then` branch (5 V); V(in) = 0 -> the `else` (7 V).
    let hi = dc_node(
        "V1 in 0 2\nB1 out 0 V=if(V(in) > 1.0, 5.0, 7.0)\n.end",
        "out",
    );
    assert!((hi - 5.0).abs() < 1e-9, "got {hi}, expected 5");
    let lo = dc_node(
        "V1 in 0 0\nB1 out 0 V=if(V(in) > 1.0, 5.0, 7.0)\n.end",
        "out",
    );
    assert!((lo - 7.0).abs() < 1e-9, "got {lo}, expected 7");
}

#[test]
fn diode_series_resistance_drops_voltage() {
    // A diode with Rs mints an internal anode node `D1.a`; the drop from the
    // external anode to it must equal I_diode * Rs at the operating point.
    let rs = 100.0_f64;
    let src = format!("V1 in 0 1\nR1 in a 1k\nD1 a 0 dm\n.model dm D(Is=1e-14 Rs={rs})\n.end");
    let parsed = parse(&src).expect("parse");
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
    // The internal node adds one unknown.
    assert!(
        dae.unknowns.iter().any(|u| u == "D1.ai"),
        "internal node missing: {:?}",
        dae.unknowns
    );
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let pnames = cdc.param_names(&ctx);
    let p: Vec<f64> = parsed.pvec(&pnames);
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-10, 100);
    assert!(conv, "DC did not converge");
    let idx = |name: &str| dae.unknowns.iter().position(|u| u == name).unwrap();
    let a = parsed.node("a").unwrap();
    let v_a = x[idx(&format!("v{a}"))];
    let v_int = x[idx("D1.ai")];
    // Diode current = current through R1 = (v_in - v_a)/R1.
    let v_in = x[idx(&format!("v{}", parsed.node("in").unwrap()))];
    let i = (v_in - v_a) / 1000.0;
    let drop = v_a - v_int;
    assert!(
        (drop - i * rs).abs() < 1e-6 * (1.0 + (i * rs).abs()),
        "Rs drop {drop} vs I*Rs {}",
        i * rs
    );
    assert!(i > 1e-6, "diode should conduct, I={i}");
}

#[test]
fn lossless_line_is_dc_transparent() {
    // A lossless (LC) transmission line: series inductances are shorts and shunt
    // capacitances draw no current at DC, so the line passes the source straight
    // through to a load. (The ladder approximation must reproduce this exactly.)
    let v = dc_node(
        "V1 in 0 1\nT1 in 0 out 0 Z0=50 TD=1n N=8\nRL out 0 1k\n.end",
        "out",
    );
    assert!(
        (v - 1.0).abs() < 1e-6,
        "lossless line DC out = {v}, expected 1.0"
    );
}

#[test]
fn rc_line_drops_dc_through_series_r() {
    // A distributed-RC (URC) line has real series resistance: into a 1 Mohm load
    // the 1 kohm of line resistance forms a divider, out = RL/(RL+R) ~= 0.999.
    let v = dc_node(
        "V1 in 0 1\nU1 in 0 out 0 R=1k C=1p N=8\nRL out 0 1Meg\n.end",
        "out",
    );
    let expected = 1.0e6 / (1.0e6 + 1.0e3);
    assert!(
        (v - expected).abs() < 1e-4,
        "RC line DC out = {v}, expected {expected}"
    );
}

#[test]
fn comparison_yields_indicator() {
    // A bare comparison is 1.0 (true) / 0.0 (false).
    let t = dc_node("V1 in 0 3\nB1 out 0 V=V(in) > 2.0\n.end", "out");
    assert!((t - 1.0).abs() < 1e-9, "got {t}, expected 1");
    let f = dc_node("V1 in 0 1\nB1 out 0 V=V(in) > 2.0\n.end", "out");
    assert!(f.abs() < 1e-9, "got {f}, expected 0");
}
