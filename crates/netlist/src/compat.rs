//! Compatibility / diagnostics report for a parsed netlist.
//!
//! The parser surfaces hard errors for things it cannot model correctly
//! (compact MOSFETs, unknown elements, unsupported Verilog-A constructs). This
//! report covers the quieter cases that would otherwise pass unnoticed: deck
//! directives the netlist parser does not act on, and model/instance parameters
//! that no device model consumed (so they were dropped). It lets a user see
//! exactly what of a real PDK deck was and was not honoured.

use std::collections::BTreeSet;

/// What the parser did with the parts of a deck it did not fully model.
#[derive(Clone, Default, Debug)]
pub struct CompatReport {
    /// Directive heads the netlist parser ignored (e.g. `.print`, `.ic`),
    /// deduplicated. Analyses (`.dc`/`.ac`/`.tran`) are handled elsewhere, not
    /// here, so they appear too — this lists what *this* parser skipped.
    pub ignored_directives: BTreeSet<String>,
    /// Parameters a deck set that no device model consumed, as `"<inst>: <key>"`.
    /// These were dropped silently before; listing them catches typos and
    /// unsupported model parameters.
    pub unknown_params: BTreeSet<String>,
    /// Free-form notes (e.g. a global option that was recorded but not applied).
    pub notes: Vec<String>,
}

impl CompatReport {
    /// Whether everything in the deck was fully honoured.
    pub fn is_clean(&self) -> bool {
        self.ignored_directives.is_empty()
            && self.unknown_params.is_empty()
            && self.notes.is_empty()
    }

    /// Record an unknown parameter for instance `inst`.
    pub(crate) fn unknown_param(&mut self, inst: &str, key: &str) {
        self.unknown_params.insert(format!("{inst}: {key}"));
    }

    /// A human-readable multi-line summary (empty string when clean).
    pub fn summary(&self) -> String {
        if self.is_clean() {
            return String::new();
        }
        let mut s = String::new();
        if !self.ignored_directives.is_empty() {
            let list: Vec<&str> = self.ignored_directives.iter().map(String::as_str).collect();
            s.push_str(&format!("ignored directives: {}\n", list.join(", ")));
        }
        if !self.unknown_params.is_empty() {
            let list: Vec<&str> = self.unknown_params.iter().map(String::as_str).collect();
            s.push_str(&format!(
                "unknown parameters (dropped): {}\n",
                list.join(", ")
            ));
        }
        for n in &self.notes {
            s.push_str(n);
            s.push('\n');
        }
        s
    }
}
