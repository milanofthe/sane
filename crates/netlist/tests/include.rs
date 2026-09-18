//! End-to-end `.include` / `.lib` resolution: a deck that references external
//! model/corner files parses as if their contents were inlined.

use std::fs;
use std::path::PathBuf;

use sane_netlist::parse_with_base;

/// A throwaway directory under the system temp dir, unique to this test process.
fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sane_netlist_inc_{}_{tag}", std::process::id()));
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[test]
fn truncated_lib_section_does_not_panic() {
    // A `.lib` file whose requested section header is the last line (no `.endl`)
    // used to slice `raw[i+1 .. end-1]` with start > end and panic. It must now
    // resolve to an empty section instead.
    let dir = scratch_dir("trunc");
    fs::write(dir.join("corners.lib"), ".lib tt").unwrap(); // header only, no body / .endl
    let deck = "* corner\nV1 a 0 1\n.lib \"corners.lib\" tt\n.end";
    // Must not panic; parses (the section is simply empty).
    let parsed = parse_with_base(deck, Some(&dir)).expect("parse with truncated .lib");
    assert!(parsed.values.contains_key("V1") || parsed.node_index.contains_key("a"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn include_pulls_in_model_and_subckt() {
    let dir = scratch_dir("inc");
    fs::write(dir.join("models.inc"), ".model dmod D Is=1e-15 N=1.2\n").unwrap();

    let deck = "\
* top deck
V1 a 0 1
D1 a 0 dmod
.include \"models.inc\"
.end";

    let parsed = parse_with_base(deck, Some(&dir)).expect("parse with include");
    // The diode model was found via the included card -> its params are bound.
    assert_eq!(parsed.param_value("D1.Is"), Some(1e-15));
    assert_eq!(parsed.param_value("D1.N"), Some(1.2));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn lib_selects_one_corner_section() {
    let dir = scratch_dir("lib");
    // A corner library with two sections; only `tt` should be pulled in.
    fs::write(
        dir.join("corners.lib"),
        ".lib tt\n.model dmod D Is=1e-15\n.endl\n\
         .lib ff\n.model dmod D Is=5e-15\n.endl\n",
    )
    .unwrap();

    let deck = "\
* corner selection
V1 a 0 1
D1 a 0 dmod
.lib \"corners.lib\" tt
.end";

    let parsed = parse_with_base(deck, Some(&dir)).expect("parse with .lib");
    // `tt` Is, not `ff` Is.
    assert_eq!(parsed.param_value("D1.Is"), Some(1e-15));

    fs::remove_dir_all(&dir).ok();
}
