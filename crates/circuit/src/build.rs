//! Building a circuit by name: elements, devices, subcircuits and their
//! values, the way a netlist states them.
//!
//! Nodes and elements are named; a value given is bound to the element's
//! parameter symbol. Inside a subcircuit body (see [`Circuit::subckt`]) every
//! name the body owns -- elements, devices, internal nodes, its parameters
//! -- is placed in the body's namespace, so that each instance renames it
//! into its own (`X1.R1`, `X1.mid`); pins and ground stay as they are.

use std::sync::Arc;

use sane_device::{CSwitch, DeviceInstance, DeviceModel};
use sane_veriloga::device::VerilogADevice;
use sane_veriloga::{builtin_module, ElaboratedModule};

use crate::{value_symbol_name, BExpr, BKind, Circuit, PortDef, SourceFn, Waveform};

impl Circuit {
    /// An empty subcircuit body named `name` with the interface nodes `pins`,
    /// in order; place it with [`instance`](Self::instance).
    pub fn subckt(name: &str, pins: &[&str]) -> Circuit {
        let mut c = Circuit::new();
        c.ns = format!("__{name}__.");
        c.pins = pins.iter().map(|p| p.to_string()).collect();
        for p in pins {
            c.node(p);
        }
        c
    }

    /// `name` as this circuit owns it: in a subcircuit body's namespace.
    fn own(&self, name: &str) -> String {
        if self.ns.is_empty() || name.starts_with(&self.ns) {
            name.to_string()
        } else {
            format!("{}{name}", self.ns)
        }
    }

    /// The index of node `name` as this circuit owns it: a pin or ground as
    /// it is, an internal node of a body in its namespace.
    fn at(&mut self, name: &str) -> usize {
        let pin = self.pins.iter().any(|p| p.eq_ignore_ascii_case(name));
        if pin || self.find_node(name) == Some(0) {
            return self.node(name);
        }
        let own = self.own(name);
        self.node(&own)
    }

    /// Bind `value` to the value symbol of element `name`.
    fn bind(&mut self, name: &str, value: f64) {
        self.values.insert(value_symbol_name(name), value);
    }

    /// Bind the parameter `name` (`R1`, `D1.is`, a card's `nmos.vth0`) to
    /// `value`; in a subcircuit body, the body's own. A `$`-name (`$temp`)
    /// is global.
    pub fn set(&mut self, name: &str, value: f64) -> &mut Self {
        let name = if name.starts_with('$') {
            name.to_string()
        } else {
            self.own(name)
        };
        self.values.insert(name, value);
        self
    }

    /// A resistor `name` between `a` and `b`, `r` ohms.
    pub fn resistor(&mut self, name: &str, a: &str, b: &str, r: f64) -> &mut Self {
        let (name, a, b) = (self.own(name), self.at(a), self.at(b));
        self.elements.resistor(&name, a, b);
        self.bind(&name, r);
        self
    }

    /// A capacitor `name` between `a` and `b`, `c` farads.
    pub fn capacitor(&mut self, name: &str, a: &str, b: &str, c: f64) -> &mut Self {
        let (name, a, b) = (self.own(name), self.at(a), self.at(b));
        self.elements.capacitor(&name, a, b);
        self.bind(&name, c);
        self
    }

    /// An inductor `name` between `a` and `b`, `l` henries.
    pub fn inductor(&mut self, name: &str, a: &str, b: &str, l: f64) -> &mut Self {
        let (name, a, b) = (self.own(name), self.at(a), self.at(b));
        self.elements.inductor(&name, a, b);
        self.bind(&name, l);
        self
    }

    /// A voltage source `name` from `p` (+) to `n` (-): a DC value or a
    /// waveform (see [`Waveform`]).
    pub fn voltage_source(
        &mut self,
        name: &str,
        p: &str,
        n: &str,
        v: impl Into<Waveform>,
    ) -> &mut Self {
        let (name, a, b) = (self.own(name), self.at(p), self.at(n));
        self.elements.voltage_source(&name, a, b);
        self.drive(&name, v.into());
        self
    }

    /// A current source `name` driving current from `p` through itself to
    /// `n`: a DC value or a waveform (see [`Waveform`]).
    pub fn current_source(
        &mut self,
        name: &str,
        p: &str,
        n: &str,
        i: impl Into<Waveform>,
    ) -> &mut Self {
        let (name, a, b) = (self.own(name), self.at(p), self.at(n));
        self.elements.current_source(&name, a, b);
        self.drive(&name, i.into());
        self
    }

    /// The waveform of the source just placed as `name`.
    fn drive(&mut self, name: &str, w: Waveform) {
        let (shape, args) = w.shape();
        match shape {
            None => self.bind(name, args[0].unwrap_or(0.0)),
            Some(shape) => {
                self.elements.set_source(shape);
                for (suffix, v) in shape.bind_params(&args) {
                    self.values.insert(format!("{name}.{suffix}"), v);
                }
            }
        }
    }

    /// A voltage-controlled voltage source `name`: `V(p, n) = gain * V(cp, cn)`.
    pub fn vcvs(
        &mut self,
        name: &str,
        p: &str,
        n: &str,
        cp: &str,
        cn: &str,
        gain: f64,
    ) -> &mut Self {
        let name = self.own(name);
        let (a, b, c, d) = (self.at(p), self.at(n), self.at(cp), self.at(cn));
        self.elements.vcvs(&name, a, b, c, d);
        self.bind(&name, gain);
        self
    }

    /// A voltage-controlled current source `name`: `I(p -> n) = gm * V(cp, cn)`.
    pub fn vccs(&mut self, name: &str, p: &str, n: &str, cp: &str, cn: &str, gm: f64) -> &mut Self {
        let name = self.own(name);
        let (a, b, c, d) = (self.at(p), self.at(n), self.at(cp), self.at(cn));
        self.elements.vccs(&name, a, b, c, d);
        self.bind(&name, gm);
        self
    }

    /// A current-controlled current source `name`: `I(p -> n) = gain *
    /// I(ctrl)`, `ctrl` a voltage-defined element (a voltage source, an
    /// inductor, ...).
    pub fn cccs(&mut self, name: &str, p: &str, n: &str, ctrl: &str, gain: f64) -> &mut Self {
        let (name, ctrl) = (self.own(name), self.own(ctrl));
        let (a, b) = (self.at(p), self.at(n));
        self.elements.cccs(&name, a, b, &ctrl);
        self.bind(&name, gain);
        self
    }

    /// A current-controlled voltage source `name`: `V(p, n) = r * I(ctrl)`.
    pub fn ccvs(&mut self, name: &str, p: &str, n: &str, ctrl: &str, r: f64) -> &mut Self {
        let (name, ctrl) = (self.own(name), self.own(ctrl));
        let (a, b) = (self.at(p), self.at(n));
        self.elements.ccvs(&name, a, b, &ctrl);
        self.bind(&name, r);
        self
    }

    /// The mutual inductance `name` between inductors `l1` and `l2`,
    /// coupling coefficient `k`.
    pub fn mutual(&mut self, name: &str, l1: &str, l2: &str, k: f64) -> &mut Self {
        let (name, l1, l2) = (self.own(name), self.own(l1), self.own(l2));
        self.elements.mutual(&name, &l1, &l2);
        self.bind(&name, k);
        self
    }

    /// A behavioral source `name` between `p` and `n`: a voltage (`V(p, n)
    /// = expr`) or a current (`p -> n`), `expr` over node indices of this
    /// circuit (see [`node`](Self::node)).
    pub fn behavioral(
        &mut self,
        name: &str,
        p: &str,
        n: &str,
        kind: BKind,
        expr: BExpr,
    ) -> &mut Self {
        let (name, a, b) = (self.own(name), self.at(p), self.at(n));
        self.elements.behavioral_source(&name, a, b, kind, expr);
        self
    }

    /// Register the Verilog-A modules `source` defines, for
    /// [`device`](Self::device) to place by module name.
    pub fn module(&mut self, source: &str) -> Result<&mut Self, String> {
        let modules =
            sane_veriloga::parse_modules(source, "module", &[]).map_err(|d| d.render(source))?;
        for m in &modules {
            let em = sane_veriloga::elaborate(m).map_err(|d| d.render(source))?;
            self.modules
                .insert(m.name.to_ascii_lowercase(), Arc::new(em));
        }
        Ok(self)
    }

    /// A device `name` of the Verilog-A module `module` -- one registered
    /// with [`module`](Self::module), or a built-in (`sane_diode`,
    /// `sane_mos`, `sane_bjt`, `sane_jfet`, `sane_vswitch`, ...) -- its
    /// terminals on `nodes` in the module's port order, its parameters
    /// `params` set (by name, case-insensitive); the others take the
    /// module's defaults.
    pub fn device(
        &mut self,
        name: &str,
        module: &str,
        nodes: &[&str],
        params: &[(&str, f64)],
    ) -> Result<&mut Self, String> {
        let em: Arc<ElaboratedModule> = (self.modules.get(&module.to_ascii_lowercase()).cloned())
            .or_else(|| builtin_module(module))
            .ok_or_else(|| format!("device {name}: no Verilog-A module '{module}'"))?;
        if nodes.len() != em.ports.len() {
            return Err(format!(
                "device {name}: {} nodes for the {} ports of module '{module}'",
                nodes.len(),
                em.ports.len()
            ));
        }
        let name = self.own(name);
        let probe = VerilogADevice::new(&name, em.clone());
        let mut set = Vec::with_capacity(params.len());
        for &(key, v) in params {
            let canon = (probe.canonical_param(key)).ok_or_else(|| {
                format!("device {name}: module '{module}' has no parameter '{key}'")
            })?;
            set.push((canon.to_string(), v));
        }
        let dev = VerilogADevice::with_instance(
            &name,
            em,
            set.iter().cloned().collect(),
            set.iter().map(|(k, _)| k.clone()).collect(),
        );
        for (k, v) in &set {
            self.values.insert(dev.param_symbol(k), *v);
        }
        let terminals = nodes.iter().map(|n| self.at(n)).collect();
        self.devices
            .push(DeviceInstance::new(Arc::new(dev), terminals));
        Ok(self)
    }

    /// A junction diode `name` from `anode` to `cathode` (`sane_diode`).
    pub fn diode(
        &mut self,
        name: &str,
        anode: &str,
        cathode: &str,
        params: &[(&str, f64)],
    ) -> Result<&mut Self, String> {
        self.device(name, "sane_diode", &[anode, cathode], params)
    }

    /// A MOSFET `name` (`sane_mos`): drain, gate, source, bulk.
    pub fn mosfet(
        &mut self,
        name: &str,
        d: &str,
        g: &str,
        s: &str,
        b: &str,
        params: &[(&str, f64)],
    ) -> Result<&mut Self, String> {
        self.device(name, "sane_mos", &[d, g, s, b], params)
    }

    /// A bipolar transistor `name` (`sane_bjt`): collector, base, emitter.
    pub fn bjt(
        &mut self,
        name: &str,
        c: &str,
        b: &str,
        e: &str,
        params: &[(&str, f64)],
    ) -> Result<&mut Self, String> {
        self.device(name, "sane_bjt", &[c, b, e], params)
    }

    /// A voltage-controlled switch `name` (`sane_vswitch`) between `a` and
    /// `b`, controlled by `V(cp, cn)`.
    pub fn vswitch(
        &mut self,
        name: &str,
        a: &str,
        b: &str,
        cp: &str,
        cn: &str,
        params: &[(&str, f64)],
    ) -> Result<&mut Self, String> {
        self.device(name, "sane_vswitch", &[a, b, cp, cn], params)
    }

    /// A current-controlled switch `name` between `a` and `b`, controlled by
    /// the current of the voltage-defined element `ctrl`, its parameters
    /// (`it`, `ih`, `ron`, `roff`) set.
    pub fn cswitch(
        &mut self,
        name: &str,
        a: &str,
        b: &str,
        ctrl: &str,
        params: &[(&str, f64)],
    ) -> Result<&mut Self, String> {
        let (name, ctrl) = (self.own(name), self.own(ctrl));
        let dev = CSwitch::new(&name, &ctrl);
        for &(key, v) in params {
            let canon = (dev.canonical_param(key))
                .ok_or_else(|| format!("switch {name}: no parameter '{key}'"))?;
            let sym = dev.param_symbol(canon).expect("a named switch");
            self.values.insert(sym, v);
        }
        let terminals = vec![self.at(a), self.at(b)];
        self.devices
            .push(DeviceInstance::new(Arc::new(dev), terminals));
        Ok(self)
    }

    /// Any device model `model`, its terminals on `nodes` (its parameters
    /// bound with [`set`](Self::set) by the symbols it names).
    pub fn place(
        &mut self,
        model: Arc<dyn DeviceModel>,
        nodes: &[&str],
    ) -> Result<&mut Self, String> {
        if nodes.len() != model.n_terminals() {
            return Err(format!(
                "a device of {} terminals placed on {} nodes",
                model.n_terminals(),
                nodes.len()
            ));
        }
        let terminals = nodes.iter().map(|n| self.at(n)).collect();
        self.devices.push(DeviceInstance::new(model, terminals));
        Ok(self)
    }

    /// A power port `name` (`P`): the Thevenin form the S-parameter
    /// extraction assumes, an ideal source named like the port (it doubles
    /// as the AC drive) behind a `z0` series resistor `<name>.z0` onto `p`;
    /// `n` is the reference.
    pub fn port(&mut self, name: &str, p: &str, n: &str, z0: f64) -> Result<&mut Self, String> {
        if !(z0 > 0.0) {
            return Err(format!("port {name}: Z0 must be positive"));
        }
        let name = self.own(name);
        let (a, b) = (self.at(p), self.at(n));
        let t = self.node(&format!("{name}.t"));
        self.elements.voltage_source(&name, t, b);
        self.bind(&name, 0.0);
        let rn = format!("{name}.z0");
        self.elements.resistor(&rn, t, a);
        self.bind(&rn, z0);
        self.ports.push(PortDef {
            name,
            node: p.to_string(),
            z0,
        });
        Ok(self)
    }

    /// Seed the operating point with `V(node) = volts` (`.nodeset`).
    pub fn nodeset(&mut self, node: &str, volts: f64) -> &mut Self {
        self.nodeset.push((node.to_string(), volts));
        self
    }
}

impl Waveform {
    /// The source shape and its positional arguments; `None` for DC (its
    /// value the one argument).
    fn shape(&self) -> (Option<SourceFn>, Vec<Option<f64>>) {
        match self {
            Waveform::Dc(v) => (None, vec![Some(*v)]),
            Waveform::Sin {
                offset,
                amplitude,
                freq,
            } => (
                Some(SourceFn::Sin),
                vec![Some(*offset), Some(*amplitude), Some(*freq)],
            ),
            // a zero fall, width or period takes SPICE's default (the fall
            // as long as the rise, the pulse held, not repeated)
            Waveform::Pulse {
                v1,
                v2,
                delay,
                rise,
                fall,
                width,
                period,
            } => (
                Some(SourceFn::Pulse),
                [v1, v2, delay, rise, fall, width, period]
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i < 4 || **v != 0.0).then_some(**v))
                    .collect(),
            ),
            Waveform::Exp {
                v1,
                v2,
                td1,
                tau1,
                td2,
                tau2,
            } => (
                Some(SourceFn::Exp),
                [v1, v2, td1, tau1, td2, tau2]
                    .iter()
                    .map(|v| Some(**v))
                    .collect(),
            ),
            Waveform::Pwl(points) => (
                Some(SourceFn::Pwl(points.len())),
                points
                    .iter()
                    .flat_map(|&(t, v)| [Some(t), Some(v)])
                    .collect(),
            ),
        }
    }
}
