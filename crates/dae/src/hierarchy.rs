//! Subcircuit instances: a body placed once in its own names, and where it
//! sits in its parent.
//!
//! A subcircuit body is assembled like the top level, over its own nodes and
//! names, and closed into one function of the graph (see `assemble`): its
//! node voltages, the unknowns it mints and its parameters in, the current it
//! draws from each of its nodes and the residuals of its unknowns out. Every
//! instance is a call. Bodies that lower to the same expressions are one
//! function, whatever instance placed them.
//!
//! The body's names carry its namespace (`__inv__.R1`, `i___inv__.V1`); an
//! instance renames a body name into its parent's frame by replacing that
//! prefix with its own name (`X1.R1`). A name without the prefix (a model
//! card's parameter, `$temp`, a `.global` node) is shared by every instance.

use sane_device::DeviceInstance;
use sane_mna::{Circuit, Element};

/// A subcircuit instance: its placed body and how it connects to its parent.
pub struct Instance {
    /// The instance's name in its parent's frame (`X1`, `__inv__.Xa`).
    pub name: String,
    /// The prefix every instance-owned name in the body carries (`__inv__.`).
    pub ns: String,
    /// The parent's node of each body node: body node `k` is parent node
    /// `nodes[k - 1]` (`0` is ground). Ports and internal nodes alike.
    pub nodes: Vec<usize>,
    /// The body's name of each body node (`mid`, a port's name), for views.
    pub node_names: Vec<String>,
    pub circuit: Circuit,
    pub devices: Vec<DeviceInstance>,
    pub instances: Vec<Instance>,
}

impl Instance {
    /// A body name in the parent's frame: the namespace replaced by the
    /// instance's name (`i___inv__.V1` -> `i_X1.V1`); a shared name as is.
    pub fn rename(&self, name: &str) -> String {
        match name.find(&self.ns) {
            Some(i) => format!(
                "{}{}.{}",
                &name[..i],
                self.name,
                &name[i + self.ns.len()..]
            ),
            None => name.to_string(),
        }
    }

    /// The parent node of body node `k` (`0` stays ground).
    pub fn node(&self, k: usize) -> usize {
        if k == 0 {
            0
        } else {
            self.nodes[k - 1]
        }
    }
}

/// The elements and device terminals of a hierarchy in the top frame: every
/// instance's elements renamed and rewired onto the top-level nodes. For the
/// topological checks that look at the whole circuit at once (index-2 loops
/// and cutsets); the DAE never flattens.
pub fn topology(
    circuit: &Circuit,
    devices: &[DeviceInstance],
    instances: &[Instance],
) -> (Vec<Element>, Vec<Vec<usize>>) {
    let mut elements = circuit.elements().to_vec();
    let mut terminals: Vec<Vec<usize>> = devices.iter().map(|d| d.terminals.clone()).collect();
    for inst in instances {
        let (es, ts) = topology(&inst.circuit, &inst.devices, &inst.instances);
        elements.extend(es.into_iter().map(|mut e| {
            e.name = inst.rename(&e.name);
            e.a = inst.node(e.a);
            e.b = inst.node(e.b);
            e.ctrl = e.ctrl.map(|(p, q)| (inst.node(p), inst.node(q)));
            e.ctrl_elem = e.ctrl_elem.map(|c| inst.rename(&c));
            e
        }));
        terminals.extend(
            ts.into_iter()
                .map(|t| t.into_iter().map(|k| inst.node(k)).collect()),
        );
    }
    (elements, terminals)
}
