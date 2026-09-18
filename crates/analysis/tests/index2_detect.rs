// Index-2 topologies must be detected on the PARSED deck, not just on a
// hand-built circuit: the netlist front end is what users actually feed in.
use sane_mna::index2;

fn report(src: &str) -> index2::Index2Report {
    let parsed = sane_netlist::parse(src).expect("parse");
    // the nonlinear devices conduct too, so they break cutsets
    let terminals: Vec<Vec<usize>> = parsed.devices.iter().map(|d| d.terminals.clone()).collect();
    index2::detect(&parsed.circuit, &terminals)
}

#[test]
fn cv_loop_from_a_deck() {
    let r = report("V1 n1 0 SIN(0 1 15.9155)\nR1 n1 0 1\nR2 n2 0 1\nC1 n1 n2 1\nC2 n2 0 1\n.end\n");
    assert!(r.is_index2(), "expected index 2, got {r:?}");
    assert_eq!(r.cv_loops.len(), 1);
    assert_eq!(r.cv_loops[0].source, "V1");
}

#[test]
fn capacitor_across_a_source_from_a_deck() {
    let r = report("V1 n1 0 SIN(0 1 15.9155)\nR1 n1 0 1\nC1 n1 0 1\n.end\n");
    assert!(r.is_index2(), "expected index 2, got {r:?}");
}

#[test]
fn rc_lowpass_from_a_deck_is_index_1() {
    let r = report("V1 n1 0 SIN(0 1 15.9155)\nR1 n1 n2 1\nC1 n2 0 1\n.end\n");
    assert!(!r.is_index2(), "expected index 1, got {r:?}");
}
