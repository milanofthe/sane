//! The circuit: the one representation of a circuit in SANE.
//!
//! A [`Circuit`] is what every frontend builds -- the netlist parser, a
//! program through the builder methods -- and what a model is set up from:
//! its elements ([`Elements`]), devices, subcircuit instances, node names,
//! bound values, power ports and DC settings. A subcircuit body is a circuit
//! too, with pins and a namespace (see [`Instance`]).
//!
//! Node `0` is ground throughout. The analyses build their systems from a
//! circuit in the `dae` crate (`I(x, t) + d/dt Q(x) = 0`).

/// Topological index-2 detection (CV loops / LI cutsets).
pub mod index2;

mod build;
mod circuit;
mod elements;
mod hierarchy;
mod source;

pub use circuit::{Circuit, DcSettings, PortDef};
pub use elements::{BExpr, BKind, BehavioralSource, Coupling, Element, Elements, Kind};
pub use hierarchy::{rename, topology, Instance};
pub use source::{SourceFn, Waveform};

/// Symbol name for an element's value parameter, kept out of the namespace
/// SANE reserves for the solver's own unknowns.
///
/// An element's name doubles as the symbol of its defining value (`R1` is the
/// resistance, `V1` the source voltage, ...). Node voltages, however, are minted
/// internally as `v{k}`, branch currents as `i_{elem}`; time is `t`. Symbols
/// are interned by name, so
/// an element whose name lands in that reserved space would be hash-consed onto
/// an unknown rather than staying a free parameter. The classic case is a power
/// grid with a voltage source literally named `v91`: it would share the symbol
/// of node 91's voltage, so its DC value silently vanishes from the parameter
/// set and the source stops constraining its node.
///
/// This maps any reserved-looking element name to a private, collision-free
/// spelling (a leading `_`, which no SPICE element name can have, since element
/// names start with their type letter); all ordinary names pass through
/// unchanged. It MUST be applied consistently wherever an element value symbol
/// is created and wherever its numeric value is bound by name, so the parameter
/// symbol and its value key always agree.
pub fn value_symbol_name(name: &str) -> String {
    if is_reserved_unknown_name(name) {
        format!("_{name}")
    } else {
        name.to_string()
    }
}

/// Whether `name` collides with SANE's internally generated unknown / time
/// symbol namespace (see [`value_symbol_name`]).
fn is_reserved_unknown_name(name: &str) -> bool {
    if name == "t" {
        return true;
    }
    // `v{digits}` (node voltage).
    let node_voltage = name
        .strip_prefix('v')
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()));
    // ... or a prefixed branch-current unknown.
    node_voltage || name.starts_with("i_")
}
