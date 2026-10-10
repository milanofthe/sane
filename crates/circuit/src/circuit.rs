//! The circuit: elements, devices and subcircuit instances over named nodes,
//! with the values bound to their parameters.

use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;
use sane_device::{DeviceInstance, ParamDefaults};

use crate::{Elements, Instance};

/// A circuit (see the crate docs). The top level has no namespace and no
/// pins; a subcircuit body has both.
#[derive(Clone, Default)]
pub struct Circuit {
    /// The prefix every instance-owned name of a subcircuit body carries
    /// (`__inv__.`); empty at the top level.
    pub ns: String,
    /// A subcircuit body's interface nodes, in order; empty at the top level.
    pub pins: Vec<String>,
    /// The linear and controlled elements, couplings and behavioral sources.
    pub elements: Elements,
    /// The devices (diodes, transistors, Verilog-A and OSDI modules).
    pub devices: Vec<DeviceInstance>,
    /// The subcircuit instances, each its body in the body's own names.
    pub instances: Vec<Instance>,
    nodes: Nodes,
    /// Parameter symbol name -> value, for every value the circuit states
    /// (an element's own, a device's, an instance's renamed into this frame).
    pub values: HashMap<String, f64>,
    /// Power ports (`P` elements), in placement order, for S-parameters.
    pub ports: Vec<PortDef>,
    /// Operating-point seeds (`.nodeset`): node name, volts.
    pub nodeset: Vec<(String, f64)>,
    /// The DC convergence settings the circuit states.
    pub dc: DcSettings,
    /// The Verilog-A modules registered to place devices of by name (see
    /// [`Circuit::module`]), by lowercased module name.
    pub modules: HashMap<String, Arc<sane_veriloga::ElaboratedModule>>,
}

/// The circuit's named nodes: index 0 is ground (`0`, `gnd`, `ground`);
/// names are case-insensitive, the first spelling is kept.
#[derive(Clone)]
struct Nodes {
    index: HashMap<String, usize>,
    names: Vec<String>,
}

impl Default for Nodes {
    fn default() -> Nodes {
        let index = ["0", "gnd", "ground"]
            .into_iter()
            .map(|g| (g.to_string(), 0))
            .collect();
        Nodes {
            index,
            names: vec!["0".to_string()],
        }
    }
}

/// DC convergence settings a circuit states (`.option reltol= abstol=
/// vntol= itl1=`); `None` where it states none.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DcSettings {
    pub reltol: Option<f64>,
    pub abstol: Option<f64>,
    pub vntol: Option<f64>,
    /// Newton iterations (`itl1`).
    pub max_iter: Option<usize>,
}

/// A power port (`P<name> n+ n- [Z0=..]`): the Thevenin form the S-parameter
/// extraction assumes. The element lowers to an ideal source (named like the
/// port, so it doubles as the AC drive) behind a `Z0` series resistor onto
/// `node`; the source value symbol is the port name, the resistor's is
/// `<name>.z0`.
#[derive(Debug, Clone)]
pub struct PortDef {
    /// Element name (`P1`, ...), also the drive-source name for AC/SP.
    pub name: String,
    /// Network-side terminal (the `n+` node, original spelling).
    pub node: String,
    /// Reference impedance in ohms.
    pub z0: f64,
}

impl std::fmt::Debug for Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Circuit")
            .field("ns", &self.ns)
            .field("elements", &self.elements)
            .field("devices", &self.devices.len())
            .field("instances", &self.instances.len())
            .field("nodes", &self.nodes.names)
            .field("values", &self.values)
            .finish()
    }
}

impl Circuit {
    /// An empty circuit, ground alone.
    pub fn new() -> Circuit {
        Circuit::default()
    }

    /// A flat circuit of an element graph and devices over node indices
    /// (no names, no subcircuits).
    pub fn flat(elements: &Elements, devices: &[DeviceInstance]) -> Circuit {
        Circuit {
            elements: elements.clone(),
            devices: devices.to_vec(),
            ..Circuit::default()
        }
    }

    /// The index of node `name`, added if new (case-insensitive; `0`,
    /// `gnd` and `ground` are ground, index 0).
    pub fn node(&mut self, name: &str) -> usize {
        let key = name.to_ascii_lowercase();
        if let Some(&k) = self.nodes.index.get(&key) {
            return k;
        }
        let k = self.nodes.names.len();
        self.nodes.names.push(name.to_string());
        self.nodes.index.insert(key, k);
        k
    }

    /// The index of node `name`, if the circuit has it (case-insensitive).
    pub fn find_node(&self, name: &str) -> Option<usize> {
        self.nodes.index.get(&name.to_ascii_lowercase()).copied()
    }

    /// The number of non-ground nodes: every node named, and every node an
    /// element, device or instance touches.
    pub fn node_count(&self) -> usize {
        let dev = self
            .devices
            .iter()
            .flat_map(|d| d.terminals.iter().copied());
        let inst = self.instances.iter().flat_map(|i| i.nodes.iter().copied());
        dev.chain(inst)
            .max()
            .unwrap_or(0)
            .max(self.elements.node_count())
            .max(self.nodes.names.len() - 1)
    }

    /// The name of every node by index (`"0"` for ground); a node placed by
    /// index without a name is named by its index.
    pub fn node_names(&self) -> Vec<String> {
        let mut names = self.nodes.names.clone();
        let n = self.node_count();
        while names.len() <= n {
            names.push(names.len().to_string());
        }
        names
    }

    /// Every placed device, subcircuit bodies' included, under its name in
    /// this frame (see [`ParamDefaults`]).
    fn defaults(&self) -> ParamDefaults<'_> {
        fn add<'a>(
            d: &mut ParamDefaults<'a>,
            instances: &'a [Instance],
            outer: &dyn Fn(&str) -> String,
        ) {
            for inst in instances {
                let rename = |n: &str| outer(&inst.rename(n));
                d.add(&inst.body.devices, &rename);
                add(d, &inst.body.instances, &rename);
            }
        }
        let mut d = ParamDefaults::new(&self.devices);
        add(&mut d, &self.instances, &|n| n.to_string());
        d
    }

    /// Parameter value by symbol name (`R1`, `M1.W`): the circuit's bound
    /// value, else the placed device's module default. `None` for a symbol
    /// neither the circuit nor a device gives a value.
    pub fn param_value(&self, name: &str) -> Option<f64> {
        self.values
            .get(name)
            .copied()
            .or_else(|| self.defaults().get(name))
    }

    /// The name of the parameter symbol device instance `inst` reads its
    /// module parameter `param` from: the card's (`nmos.vth0`, shared by
    /// the card's instances) or its own (`M1.w`, `X1.M1.w`). `None` when no
    /// placed device has that instance name.
    pub fn param_symbol(&self, inst: &str, param: &str) -> Option<String> {
        fn find(
            c: &Circuit,
            rename: &dyn Fn(&str) -> String,
            inst: &str,
            param: &str,
        ) -> Option<String> {
            let hit = (c.devices.iter())
                .find(|d| d.model.instance_name().is_some_and(|n| rename(n) == inst));
            if let Some(d) = hit {
                return d.model.param_symbol(param).map(|s| rename(&s));
            }
            c.instances.iter().find_map(|i| {
                let r = |n: &str| rename(&i.rename(n));
                find(&i.body, &r, inst, param)
            })
        }
        find(self, &|n| n.to_string(), inst, param)
    }

    /// The parameter vector for `names` (the engine's column order): bound
    /// values, device defaults for unstated parameters, `0.0` for anything
    /// still unbound.
    pub fn pvec(&self, names: &[String]) -> Vec<f64> {
        let defaults = self.defaults();
        (names.iter())
            .map(|n| {
                (self.values.get(n).copied())
                    .or_else(|| defaults.get(n))
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Place an instance `name` of the subcircuit `body`, its pins on the
    /// nodes `conn` (by name, in pin order): its internal nodes, bound values
    /// and power ports come in under the instance's names (`X1.mid`,
    /// `X1.R1`). A body is placed once for all its instances; share it
    /// through the one `Arc`.
    pub fn instance(
        &mut self,
        name: &str,
        body: &Arc<Circuit>,
        conn: &[&str],
    ) -> Result<&mut Instance, String> {
        if conn.len() != body.pins.len() {
            return Err(format!(
                "instance {name}: {} nodes for the {} pins of its subcircuit",
                conn.len(),
                body.pins.len()
            ));
        }
        let mut inst = Instance {
            name: name.to_string(),
            nodes: Vec::new(),
            body: body.clone(),
        };
        inst.nodes = (body.node_names().iter().skip(1))
            .map(
                |node| match body.pins.iter().position(|p| p.eq_ignore_ascii_case(node)) {
                    Some(i) => self.node(conn[i]),
                    None => self.node(&inst.rename(node)),
                },
            )
            .collect();
        for (k, v) in &body.values {
            self.values.insert(inst.rename(k), *v);
        }
        self.ports.extend(body.ports.iter().map(|p| PortDef {
            name: inst.rename(&p.name),
            node: inst.rename(&p.node),
            z0: p.z0,
        }));
        self.instances.push(inst);
        Ok(self.instances.last_mut().unwrap())
    }
}
