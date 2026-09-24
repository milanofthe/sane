use super::*;
use sane_mna::SourceFn;

#[test]
fn parse_error_carries_column_and_renders_caret() {
    // An indented unknown element: the caret must land under the element
    // name, not at column 1, and the rendered snippet must echo the line.
    let deck = "* title\n   Z1 a b\n.end";
    let e = parse(deck).unwrap_err();
    assert_eq!(e.line, 2);
    assert_eq!(e.col, 4); // three spaces of indent -> column 4
    let r = e.render(deck);
    assert!(r.contains("netlist:2:4: error:"), "header: {r}");
    assert!(r.contains("   Z1 a b"), "source line: {r}");
    assert!(r.contains("   ^"), "caret: {r}");
}

#[test]
fn inline_veriloga_error_maps_span_into_deck() {
    // A syntax error inside an inline `.veriloga` block must surface with a
    // deck line/column (structurally mapped from the VA span), not a scraped
    // line-1 message. The broken `endmodule` (missing) trips the VA parser.
    let deck = "V1 in 0 1\n.veriloga\nmodule m(a); electrical a; analog I(a) <+ @;\n.endveriloga\nN1 in m\n.end";
    let e = parse(deck).unwrap_err();
    // The `.veriloga` directive is deck line 2; the module is deck line 3.
    assert_eq!(e.line, 3, "error should map onto the module line: {e}");
    assert!(e.col > 0, "column should be preserved: {e:?}");
    assert!(e.msg.starts_with("veriloga"), "phase kept: {e}");
}

#[test]
fn parses_engineering_values() {
    let close = |s: &str, want: f64| {
        let got = parse_value(s).unwrap();
        assert!(
            (got - want).abs() <= want.abs() * 1e-12,
            "{s}: {got} != {want}"
        );
    };
    close("1k", 1e3);
    close("2.2u", 2.2e-6);
    close("1Meg", 1e6);
    close("1e3", 1e3);
    close("4.7nF", 4.7e-9);
    close("100", 100.0);
}

#[test]
fn maps_nodes_with_ground_aliases() {
    let p = parse("R1 in 0 1k\nC1 in gnd 1u\n.end").unwrap();
    assert_eq!(p.node("0"), Some(0));
    assert_eq!(p.node("gnd"), Some(0));
    assert_eq!(p.node("in"), Some(1));
    assert_eq!(p.values["R1"], 1e3);
}

#[test]
fn rejects_unknown_element() {
    let e = parse("Z1 a b").unwrap_err();
    assert_eq!(e.line, 1);
}

#[test]
fn binds_model_and_inline_params() {
    let p =
        parse("D1 a 0 dmod\nD2 b 0 dmod Is=2e-14\n.model dmod D(Is=1e-14 N=1.0 Vt=0.02585)\n.end")
            .unwrap();
    assert!((p.values["D1.Is"] - 1e-14).abs() < 1e-26);
    assert!((p.values["D1.N"] - 1.0).abs() < 1e-12);
    assert!((p.values["D1.Vt"] - 0.02585).abs() < 1e-9);
    assert!((p.values["D2.Is"] - 2e-14).abs() < 1e-26); // inline overrides model
}

#[test]
fn mixed_case_inline_comment_and_continuation() {
    let p = parse("R1 IN OUT 1k ; load resistor\n.model DM D\n+ (Is=1e-14 N=1)\nD1 out 0 dm\n.END")
        .unwrap();
    // Case-insensitive node identity.
    assert_eq!(p.node("in"), p.node("IN"));
    assert_eq!(p.node("OUT"), p.node("out"));
    assert_eq!(p.values["R1"], 1e3);
    // .model continued across a `+` line still binds.
    assert!((p.values["D1.Is"] - 1e-14).abs() < 1e-26);
}

#[test]
fn params_and_expressions() {
    let p = parse(
        ".param R0=1k fc=1k\n\
         .param C0={1/(2*pi*R0*fc)}\n\
         R1 in out {R0}\n\
         C1 out 0 {C0}\n\
         .end",
    )
    .unwrap();
    assert!((p.values["R1"] - 1000.0).abs() < 1e-9);
    let want = 1.0 / (2.0 * std::f64::consts::PI * 1000.0 * 1000.0);
    assert!(
        (p.values["C1"] - want).abs() <= want * 1e-9,
        "C1={}",
        p.values["C1"]
    );
}

#[test]
fn leading_continuation_line_does_not_panic() {
    // A deck whose first non-blank line is a `+` continuation (no previous
    // logical line) used to push an empty-token Line and panic downstream.
    let p = parse("+ orphan tokens\nV1 a 0 1\nR1 a 0 1k\n.end").expect("parse");
    assert!(p.values.contains_key("R1"));
    // A bare `+` / `+ ; comment` must also be harmless.
    parse("+\nV1 a 0 1\n.end").expect("parse bare +");
}

#[test]
fn compat_report_flags_ignored_directives_and_unknown_params() {
    let p = parse(
        ".model dmod D Is=1e-15 bogusparam=3\n\
         V1 a 0 1\n\
         D1 a 0 dmod foo=2 m=2\n\
         .print dc v(a)\n\
         .save all\n\
         .end",
    )
    .unwrap();
    // Analysis/output directives this parser does not act on are listed.
    assert!(p.report.ignored_directives.contains(".print"));
    assert!(p.report.ignored_directives.contains(".save"));
    // Unknown model-card and instance params are surfaced, not dropped silently.
    assert!(p
        .report
        .unknown_params
        .iter()
        .any(|s| s.contains("bogusparam")));
    assert!(p.report.unknown_params.iter().any(|s| s.contains("foo")));
    // The `m=` instance modifier is NOT flagged as an unknown parameter.
    assert!(!p.report.unknown_params.iter().any(|s| s.ends_with(": m")));
    assert!(!p.report.is_clean());

    // BJT parasitic resistances on the card activate the series-resistance
    // topology and must bind as known parameters, not report as unknown.
    let b = parse(
        ".model qmod NPN(Is=1e-14 Rb=500 Rc=10 Re=2 Cje=1p)\n\
         V1 a 0 1\n\
         Q1 a a 0 qmod\n\
         .end",
    )
    .unwrap();
    assert!(!b.report.unknown_params.iter().any(|s| s.contains("Rb")));
    assert!(!b.report.unknown_params.iter().any(|s| s.contains("Rc")));
    assert!(!b.report.unknown_params.iter().any(|s| s.contains("Re")));
    assert_eq!(b.param_value("Q1.Rb"), Some(500.0));

    // A plain, fully-supported deck reports clean.
    let q = parse("V1 a 0 1\nR1 a 0 1k\n.end").unwrap();
    assert!(q.report.is_clean(), "summary: {}", q.report.summary());
}

/// Evaluate the first terminal current of parsed device `di` at a fixed
/// terminal bias, binding parameters from the parse's value map. Verilog-A
/// devices fold their parallel multiplicity `m` into the lowered graph, so the
/// modifier is observable only through the current.
fn device_terminal_current(p: &crate::ParsedCircuit, di: usize, bias: &[f64]) -> f64 {
    use num_complex::Complex64;
    use std::collections::HashMap as Map;
    let mut ctx = sane_core::Graph::new();
    let n = p.devices[di].model.n_terminals();
    let term_v: Vec<_> = (0..n).map(|k| ctx.sym(&format!("tv{k}"))).collect();
    let term_vdot: Vec<_> = (0..n).map(|k| ctx.sym(&format!("tvd{k}"))).collect();
    let mut lo = sane_device::Lowerer::new(&mut ctx);
    let frag = p.devices[di]
        .model
        .lower_behavioral(&mut lo, &term_v, &term_vdot, &[]);
    let i = frag.terminal_currents[0];
    let mut env: Map<rsdag::SymbolId, Complex64> = Map::new();
    for s in ctx.free_symbols(i) {
        let name = ctx.symbol_name(s).to_string();
        let v = if let Some(k) = name.strip_prefix("tv") {
            if let Some(k) = k.strip_prefix('d') {
                let _: usize = k.parse().unwrap();
                0.0
            } else {
                bias[k.parse::<usize>().unwrap()]
            }
        } else if name == sane_core::constants::TEMP_SYMBOL {
            sane_core::constants::TEMP_NOMINAL_K
        } else {
            p.param_value(&name).unwrap_or(0.0)
        };
        env.insert(s, Complex64::new(v, 0.0));
    }
    rsdag::eval(&ctx, &[i], &env)[0].re
}

#[test]
fn instance_modifiers_set_mfactor_and_scale() {
    let deck = |mods: &str| {
        format!(
            ".option scale=0.5\n\
             .model dmod D Is=1e-15\n\
             .model nch NMOS Vto=0.5 Kp=1e-4\n\
             V1 a 0 1\n\
             D1 a 0 dmod {mods}\n\
             M1 a g 0 0 nch L=2u W=10u m=3 nf=2\n\
             .end"
        )
    };
    let p1 = parse(&deck("")).unwrap();
    let p4 = parse(&deck("m=4")).unwrap();
    // Devices are pushed in source order: D1 first, then M1.
    assert_eq!(p4.devices.len(), 2);
    // Diode m=4: the lowered terminal current scales fourfold.
    let i1 = device_terminal_current(&p1, 0, &[0.6, 0.0]);
    let i4 = device_terminal_current(&p4, 0, &[0.6, 0.0]);
    assert!(
        (i4 / i1 - 4.0).abs() < 1e-9,
        "diode m=4 scales the current: {i1} vs {i4}"
    );
    // MOSFET multiplicity = m * nf = 3 * 2 = 6.
    let im = device_terminal_current(&p4, 1, &[1.0, 1.0, 0.0, 0.0]);
    let im1 = {
        let single = parse(
            ".option scale=0.5\n\
             .model nch NMOS Vto=0.5 Kp=1e-4\n\
             V1 a 0 1\n\
             M1 a g 0 0 nch L=2u W=10u\n\
             .end",
        )
        .unwrap();
        device_terminal_current(&single, 0, &[1.0, 1.0, 0.0, 0.0])
    };
    assert!(
        (im / im1 - 6.0).abs() < 1e-9,
        "mosfet m*nf=6 scales the current: {im1} vs {im}"
    );
    // `.option scale=0.5` halves the drawn MOSFET geometry.
    assert!(
        (p4.values["M1.L"] - 1e-6).abs() < 1e-15,
        "L={}",
        p4.values["M1.L"]
    );
    assert!(
        (p4.values["M1.W"] - 5e-6).abs() < 1e-15,
        "W={}",
        p4.values["M1.W"]
    );
}

#[test]
fn model_binning_selects_by_geometry() {
    // Two bins of the same base model `nch`, split at L = 1u. Each instance
    // is served by the bin whose L/W window contains its geometry.
    let p = parse(
        ".model nch.1 NMOS(lmin=0 lmax=1u wmin=0 wmax=1m Vto=0.4 Kp=100u)\n\
         .model nch.2 NMOS(lmin=1u lmax=10u wmin=0 wmax=1m Vto=0.7 Kp=50u)\n\
         M1 d g s b nch L=0.5u W=2u\n\
         M2 d g s b nch L=2u   W=2u\n\
         V1 d 0 1\n\
         .end",
    )
    .unwrap();
    // M1 (L=0.5u) -> bin .1; M2 (L=2u) -> bin .2.
    assert_eq!(p.param_value("M1.Vto"), Some(0.4));
    assert_eq!(p.param_value("M2.Vto"), Some(0.7));
    // Binning bounds are not bound as device parameters.
    assert!(!p.values.contains_key("M1.lmin"));
}

#[test]
fn temp_directive_sets_global_temperature() {
    // `.temp 85` -> $temp = 85 + 273.15 K, and the `.param` `temp` constant
    // reflects the same value (in Celsius).
    let p = parse(
        ".temp 85\n\
         .param Rt={temp}\n\
         V1 a 0 1\n\
         R1 a 0 {Rt}\n\
         .end",
    )
    .unwrap();
    let t = p.values[sane_core::constants::TEMP_SYMBOL];
    assert!((t - (85.0 + 273.15)).abs() < 1e-9, "$temp={t}");
    assert!((p.values["R1"] - 85.0).abs() < 1e-9, "temp const in .param");

    // Default (no directive) stays nominal 27 degC.
    let d = parse("V1 a 0 1\nR1 a 0 1k\n.end").unwrap();
    let td = d.values[sane_core::constants::TEMP_SYMBOL];
    assert!(
        (td - sane_core::constants::TEMP_NOMINAL_K).abs() < 1e-9,
        "$temp={td}"
    );
}

#[test]
fn subckt_flatten_and_param_override() {
    let p = parse(
        ".subckt rcpair a b\n\
         R1 a mid {Rval}\n\
         R2 mid b {Rval}\n\
         .ends\n\
         .param Rval=1k\n\
         X1 in out rcpair\n\
         X2 out 0 rcpair Rval=2k\n\
         .end",
    )
    .unwrap();
    // Internal node got the instance prefix; ports mapped to caller nodes.
    assert!(p.node("X1.mid").is_some());
    assert!(p.node("X2.mid").is_some());
    assert_eq!(p.values["X1.R1"], 1000.0);
    assert_eq!(p.values["X2.R1"], 2000.0); // instance param override
}

#[test]
fn behavioral_elements_inside_subckt() {
    // A `B` source and a voltage-dependent (behavioral) resistor inside a
    // subcircuit used to be rejected; their node references inside `V(...)`
    // are now port-/prefix-remapped, and the `r={...}` keyword form with a
    // node-voltage value lowers to a behavioral element. This is what lets a
    // foundry PDK's behavioral passives (e.g. sky130 `res_high_po`) be read
    // natively rather than pre-flattened by an external tool.
    let p = parse(
        ".subckt blk a b\n\
         B1 a mid I=V(a,mid)/1000\n\
         R1 mid b r={2000*(1+0.1*abs(V(mid,b)))}\n\
         .ends\n\
         V1 in 0 1\n\
         X1 in 0 blk\n\
         .end",
    )
    .expect("behavioral B-source and resistor inside a subckt should parse");
    // The subckt-internal node carries the instance prefix (so the in-expr
    // references were remapped through the flattener, not left dangling).
    assert!(
        p.node("X1.mid").is_some(),
        "internal node X1.mid should exist"
    );
}

#[test]
fn behavioral_resistor_keyword_and_value_forms() {
    // The `r=`/`{...}` keyword value form and a node-voltage-dependent value
    // both parse at the top level (the latter as a behavioral element).
    parse("V1 in 0 1\nR1 in out r={1000}\nR2 out 0 1k\n.end").expect("r={const} keyword form");
    parse("V1 in 0 1\nR1 in out r={1000*(1+0.1*abs(v(in,out)))}\nR2 out 0 1k\n.end")
        .expect("voltage-dependent behavioral resistor");
}

#[test]
fn source_dc_keyword_value() {
    // The `DC=<value>` keyword form (HSPICE/ngspice testbenches, e.g. a bias
    // `Ib vdd n DC=current_0_bias`) binds the source value, like a bare value
    // or the `DC <value>` token form.
    let p = parse(".param ib=5e-6\nI1 0 out DC=ib\nR1 out 0 1k\n.end").unwrap();
    assert!((p.values["I1"] - 5e-6).abs() < 1e-12, "I DC=<param>");
    let v = parse("V1 a 0 DC=1.8\nR1 a 0 1k\n.end").unwrap();
    assert!((v.values["V1"] - 1.8).abs() < 1e-12, "V DC=<num>");
}

#[test]
fn subckt_prefixes_controller_names() {
    // A CCVS controlled by an internal source: both must get the X1. prefix.
    let p = parse(
        ".subckt amp in out\n\
         Vs in mid 0\n\
         H1 out 0 Vs 5\n\
         .ends\n\
         X1 a b amp\n\
         .end",
    )
    .unwrap();
    let h = p
        .circuit
        .elements()
        .iter()
        .find(|e| e.name == "X1.H1")
        .unwrap();
    assert_eq!(h.ctrl_elem.as_deref(), Some("X1.Vs"));
}

#[test]
fn nested_subckt() {
    let p = parse(
        ".subckt inner a b\nR1 a b 1k\n.ends\n\
         .subckt outer p q\nX1 p q inner\n.ends\n\
         X9 in 0 outer\n.end",
    )
    .unwrap();
    // R inside the doubly-nested instance carries the full path.
    assert!(p.values.contains_key("X9.X1.R1"));
}

#[test]
fn spice_param_aliasing() {
    let p = parse("Q1 c b e qm\n.model qm NPN(Bf=120 IS=2e-15)\n.end").unwrap();
    assert_eq!(p.values["Q1.betaF"], 120.0); // Bf -> betaF
    assert!((p.values["Q1.Is"] - 2e-15).abs() < 1e-27); // IS -> Is
}

#[test]
fn compact_mosfet_level_is_rejected_with_va_hint() {
    // A compact model (level > 3) has no native form: the `M` element is
    // rejected with a clear pointer to the Verilog-A path, rather than being
    // silently mis-modelled by a black box.
    let err = parse("M1 d g s b nch\n.model nch NMOS(level=54 vth0=0.4)\n.end")
        .expect_err("compact-level M must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("compact model") && msg.contains("Verilog-A"),
        "got: {msg}"
    );

    // Levels 1-3 stay the native square-law MOSFET (d g s b).
    let l1 = parse("M2 d g s b m1\n.model m1 NMOS(level=1 Kp=1m Vth=0.5)\n.end").unwrap();
    assert_eq!(l1.devices[0].terminals.len(), 4);
}

#[test]
fn sin_source_binds_and_marks() {
    let p = parse("V1 1 0 SIN(0 1 1k)\nR1 1 0 1k\n.end").unwrap();
    assert_eq!(p.values["V1.sin_off"], 0.0);
    assert_eq!(p.values["V1.sin_amp"], 1.0);
    let w = 2.0 * std::f64::consts::PI * 1000.0;
    assert!((p.values["V1.sin_w"] - w).abs() < 1e-6);
    let v1 = p
        .circuit
        .elements()
        .iter()
        .find(|e| e.name == "V1")
        .unwrap();
    assert_eq!(v1.source, Some(SourceFn::Sin));
}

/// The bare ngspice source form (`dc 0 sin 0 1 1k`, no parentheses) binds
/// the same waveform as `SIN(0 1 1k)`.
#[test]
fn bare_source_form_binds_waveform() {
    let p = parse("V1 1 0 dc 0 sin 0 1 1k\nR1 1 0 1k\n.end").unwrap();
    assert_eq!(p.values["V1.sin_amp"], 1.0);
    let v1 = p
        .circuit
        .elements()
        .iter()
        .find(|e| e.name == "V1")
        .unwrap();
    assert_eq!(v1.source, Some(SourceFn::Sin));
    // pulse likewise, with trailing key=value tokens left alone
    let p = parse("I1 0 2 pulse 0 1 1u 1u 1u 1m 2m m=2\nR1 2 0 1k\n.end").unwrap();
    assert_eq!(p.values["I1.pulse_v2"], 1.0);
    assert_eq!(p.values["I1.pulse_per"], 2e-3);
}

/// `.global` rails keep their identity inside subckt instances.
#[test]
fn global_nodes_pass_through_subckts() {
    let p = parse(
        ".global vdd\nVd vdd 0 5\n.subckt stage out\nR1 vdd out 1k\n.ends\nX1 a stage\nR2 a 0 1k\n.end",
    )
    .unwrap();
    // X1's internal R1 must connect to the TOP vdd (voltage divider a = 2.5)
    let names: Vec<_> = p
        .circuit
        .elements()
        .iter()
        .map(|e| e.name.clone())
        .collect();
    assert!(names.iter().any(|n| n.contains("R1")), "{names:?}");
    let mut ctx = sane_core::Graph::new();
    let dae = sane_dae::assemble_dae(&mut ctx, &p.circuit, &p.devices);
    assert!(
        dae.unknowns.iter().any(|u| u == "vvdd" || u == "v1"),
        "{:?}",
        dae.unknowns
    );
}

/// SPICE-canonical `CJO` (letter O) activates the diode junction charge
/// exactly like the model-native `Cj0` spelling.
#[test]
fn diode_cjo_spelling_activates_charge() {
    let p = parse("D1 a 0 dm\n.model dm d is=1e-14 cjo=1p m=0.4 vj=0.8\n.end").unwrap();
    assert_eq!(p.values["D1.Cj0"], 1e-12);
    assert_eq!(p.values["D1.M"], 0.4);
    assert_eq!(p.values["D1.Vj"], 0.8);
}

#[test]
fn parses_voltage_switch() {
    let p = parse("S1 1 0 c 0 sw\n.model sw SW(Ron=1 Roff=1Meg Vt=0.5)\n.end").unwrap();
    assert_eq!(p.devices.len(), 1);
    assert_eq!(p.devices[0].terminals.len(), 4);
    assert!((p.values["S1.Ron"] - 1.0).abs() < 1e-12);
    assert!((p.values["S1.Roff"] - 1e6).abs() < 1.0);
}

#[test]
fn pulse_pwl_exp_sources() {
    let p = parse("V1 1 0 PULSE(0 5 1n 1n 1n 5n 10n)\n.end").unwrap();
    assert_eq!(p.circuit.elements()[0].source, Some(SourceFn::Pulse));
    assert_eq!(p.values["V1.pulse_v2"], 5.0);

    let p = parse("V2 1 0 PWL(0 0 1m 1 2m 0)\n.end").unwrap();
    assert_eq!(p.circuit.elements()[0].source, Some(SourceFn::Pwl(3)));
    assert_eq!(p.values["V2.pwl_v1"], 1.0);

    let p = parse("V3 1 0 EXP(0 1 0 1m 5m 2m)\n.end").unwrap();
    assert_eq!(p.circuit.elements()[0].source, Some(SourceFn::Exp));
    assert_eq!(p.values["V3.exp_v2"], 1.0);
}

#[test]
fn parses_current_switch() {
    let p = parse("Vc 2 0 0\nW1 3 0 Vc sw\n.model sw CSW(Ron=1 Roff=1Meg It=0.5)\n.end").unwrap();
    assert_eq!(p.devices.len(), 1);
    assert_eq!(p.devices[0].terminals.len(), 2);
    assert!((p.values["W1.Ron"] - 1.0).abs() < 1e-12);
}

#[test]
fn parses_current_controlled_sources() {
    use sane_mna::Kind;
    let p = parse("V1 1 0 1\nR1 1 0 1k\nF1 2 0 V1 10\nH1 3 0 V1 5\nRL 2 0 1k\n.end").unwrap();
    let kinds: Vec<Kind> = p.circuit.elements().iter().map(|e| e.kind).collect();
    assert!(kinds.contains(&Kind::Cccs));
    assert!(kinds.contains(&Kind::Ccvs));
}

#[test]
fn parses_devices() {
    let p = parse("V1 in 0 5\nR1 in d 1k\nD1 d 0 dmod\nQ1 c b e\n.model dmod D\n.end").unwrap();
    assert_eq!(p.devices.len(), 2); // diode + bjt
    assert_eq!(p.devices[0].terminals, vec![p.node("d").unwrap(), 0]);
    // R1 is the only linear non-source... plus V1; devices are separate.
    assert!(p.node("c").is_some() && p.node("b").is_some() && p.node("e").is_some());
}
