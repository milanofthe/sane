//! `idt(u, ic)` DC semantics: DC enforces the steady state `u = 0` (matching
//! OpenVAF/OSDI), and a constant ic is routed as the state's DC Newton seed
//! (`.nodeset`-style) -- it picks the branch on multi-stable integrands
//! instead of being silently dropped.

use sane_analysis::Model;

/// A bistable integrand: `u = s (1 - s^2)` has DC roots s in {0, +1, -1}. The
/// ic seed must select the branch the cold solve converges to.
fn bistable_deck(ic: f64) -> String {
    format!(
        ".veriloga\n\
         module bistab(out, ref);\n\
           inout out, ref; electrical out, ref;\n\
           real s;\n\
           analog begin\n\
             s = idt(V(out,ref)*(1.0 - V(out,ref)*V(out,ref)), {ic});\n\
             V(out,ref) <+ s;\n\
           end\n\
         endmodule\n\
         .endveriloga\n\
         N1 out 0 bistab\nR1 out 0 1k\n.end\n"
    )
}

/// The constant ic lands in the DAE's seed registry under the state's name.
#[test]
fn idt_ic_registers_a_dc_seed() {
    let model = Model::from_netlist(&bistable_deck(0.9)).expect("model");
    let dae = model.dae();
    assert_eq!(dae.dc_seeds.len(), 1);
    assert_eq!(dae.dc_seeds[0].0, "N1.idt0");
    assert!((dae.dc_seeds[0].1 - 0.9).abs() < 1e-15);
}

/// The seed steers the cold DC solve onto the matching stable root; DC still
/// enforces u = 0 exactly (V is a root of s (1 - s^2)).
#[test]
fn idt_ic_selects_the_dc_branch() {
    for (ic, root) in [(0.9, 1.0), (-0.9, -1.0)] {
        let model = Model::from_netlist(&bistable_deck(ic)).expect("model");
        let op = model.operating_point(&[]).expect("dc");
        let v = op.get("out").expect("out");
        assert!((v - root).abs() < 1e-9, "ic={ic}: expected {root}, got {v}");
    }
}

/// Template clones carry the seed per instance.
#[test]
fn idt_ic_seeds_survive_template_clones() {
    let deck = "\
.veriloga
module intg(a, b);
  inout a, b; electrical a, b;
  analog V(a,b) <+ idt(1.0 - V(a,b), 0.25);
endmodule
.endveriloga
N1 x 0 intg
N2 y 0 intg
R1 x 0 1k
R2 y 0 1k
.end
";
    let model = Model::from_netlist(deck).expect("model");
    let mut names: Vec<&str> = model
        .dae()
        .dc_seeds
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    names.sort();
    assert_eq!(names, ["N1.idt0", "N2.idt0"]);
    assert!(model
        .dae()
        .dc_seeds
        .iter()
        .all(|(_, c)| (*c - 0.25).abs() < 1e-15));
}

/// A non-constant ic cannot seed a numeric solve: it warns (captured) and the
/// DC solve still lands on the steady state.
#[test]
fn idt_nonconstant_ic_warns() {
    let deck = "\
.veriloga
module drift(a, b);
  inout a, b; electrical a, b;
  analog V(a,b) <+ idt(1.0 - V(a,b), V(a,b));
endmodule
.endveriloga
N1 x 0 drift
R1 x 0 1k
.end
";
    sane_core::log::drain_captured();
    let model = Model::from_netlist(deck).expect("model");
    let warns = sane_core::log::drain_captured();
    assert!(
        warns.iter().any(|w| w.contains("not a constant")),
        "expected a non-constant-ic warning, got {warns:?}"
    );
    assert!(model.dae().dc_seeds.is_empty());
    let op = model.operating_point(&[]).expect("dc");
    assert!(
        (op.get("x").unwrap() - 1.0).abs() < 1e-9,
        "steady state u = 0"
    );
}
