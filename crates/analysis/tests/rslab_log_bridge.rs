//! rslab's log records reach SANE's logger: its factorization summaries land
//! at SANE's `Debug` level with an `rslab:` prefix, and SANE's threshold is
//! mirrored into rslab so nothing is formatted that SANE would not show.

use std::sync::Mutex;

use sane_analysis::Model;
use sane_core::log::{self, LogLevel};

static LINES: Mutex<Vec<(LogLevel, String)>> = Mutex::new(Vec::new());

fn capture(level: LogLevel, msg: &str) {
    LINES.lock().unwrap().push((level, msg.to_string()));
}

#[test]
fn rslab_records_are_bridged_at_debug() {
    let deck = "\
* bridge
V1 in 0 1
R1 in mid 1k
D1 mid 0 DMOD
.model DMOD D(Is=1e-14)
.end
";
    // The graph solve carries the Newton systems; the library runs only on
    // request, which is what this test is about.
    sane_core::update_config(|c| c.graph_solve = false);
    log::set_sink(Some(capture));
    log::set_level(LogLevel::Debug);
    let model = Model::from_netlist(deck).expect("model");
    let _ = model.operating_point(&[]).expect("op");
    log::set_level(LogLevel::Disabled);
    log::set_sink(None);
    sane_core::update_config(|c| c.graph_solve = true);
    let lines = LINES.lock().unwrap().clone();
    let rslab: Vec<&(LogLevel, String)> =
        lines.iter().filter(|(_, m)| m.contains("rslab:")).collect();
    assert!(
        !rslab.is_empty(),
        "no rslab records among {} lines",
        lines.len()
    );
    assert!(
        rslab.iter().all(|(l, _)| *l == LogLevel::Debug),
        "rslab summaries land at Debug: {rslab:?}"
    );
    assert!(
        rslab.iter().any(|(_, m)| m.contains("factor:")),
        "{rslab:?}"
    );
}
