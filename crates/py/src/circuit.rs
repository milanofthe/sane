//! The circuit: Python's handle on `sane_circuit::Circuit`, built by name
//! or parsed from a netlist, and the model is set up from it.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sane_circuit::{Circuit, Kind, Waveform};

use crate::model::PyModel;
use crate::surface;

/// A circuit: elements, devices and subcircuit instances over named nodes,
/// with the values bound to their parameters. Built element by element (each
/// method returns the circuit, so calls chain) or parsed from a SPICE-like
/// netlist; ``sane.Model(circuit)`` sets up its model.
///
/// >>> c = sane.Circuit()
/// >>> c.voltage_source("V1", "in", "0", 5.0).resistor("R1", "in", "out", 1e3)
/// >>> c.resistor("R2", "out", "0", 1e3)
/// >>> sane.Model(c).at().operating_point()["out"]
/// 2.5
#[pyclass(name = "Circuit", module = "sane")]
#[derive(Clone)]
pub struct PyCircuit {
    /// Shared with the models and the instances that place it; an edit
    /// copies it first where it is shared.
    pub inner: Arc<Circuit>,
}

/// An independent source's waveform (see ``Circuit.voltage_source``).
#[pyclass(name = "Waveform", module = "sane", frozen)]
#[derive(Clone)]
pub struct PyWaveform {
    inner: Waveform,
}

fn invalid(m: String) -> PyErr {
    PyValueError::new_err(m)
}

/// Keyword parameters as `(name, value)`: `lambda_` is the Python spelling
/// of `lambda`.
fn params(kw: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, f64)>> {
    let mut out = Vec::new();
    if let Some(kw) = kw {
        for (k, v) in kw.iter() {
            let k: String = k.extract()?;
            let k = if k == "lambda_" {
                "lambda".to_string()
            } else {
                k
            };
            out.push((k, v.extract::<f64>()?));
        }
    }
    Ok(out)
}

fn pairs(p: &[(String, f64)]) -> Vec<(&str, f64)> {
    p.iter().map(|(k, v)| (k.as_str(), *v)).collect()
}

/// A source's value: a number (DC) or a `Waveform`.
fn waveform(v: &Bound<'_, PyAny>) -> PyResult<Waveform> {
    if let Ok(w) = v.downcast::<PyWaveform>() {
        return Ok(w.get().inner.clone());
    }
    Ok(Waveform::Dc(v.extract::<f64>()?))
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Resistor => "resistor",
        Kind::Capacitor => "capacitor",
        Kind::Inductor => "inductor",
        Kind::VoltageSource => "voltage_source",
        Kind::CurrentSource => "current_source",
        Kind::Vcvs => "vcvs",
        Kind::Vccs => "vccs",
        Kind::Cccs => "cccs",
        Kind::Ccvs => "ccvs",
    }
}

impl PyCircuit {
    fn edit(&mut self) -> &mut Circuit {
        Arc::make_mut(&mut self.inner)
    }
}

#[pymethods]
impl PyCircuit {
    #[new]
    fn new() -> PyCircuit {
        PyCircuit {
            inner: Arc::new(Circuit::new()),
        }
    }

    /// The circuit of a SPICE-like netlist: ``.model`` cards, ``.param``,
    /// ``.subckt``, ``.include``, Verilog-A blocks (``.veriloga``), every
    /// device level.
    #[staticmethod]
    fn parse(py: Python<'_>, netlist: &str) -> PyResult<PyCircuit> {
        let parsed = py.allow_threads(|| sane_netlist::parse(netlist));
        surface(py)?;
        let inner = parsed.map_err(|e| invalid(e.render(netlist)))?;
        Ok(PyCircuit {
            inner: Arc::new(inner),
        })
    }

    /// An empty subcircuit body named ``name`` with the interface nodes
    /// ``pins``, in order; place it with ``instance``. What it owns (its
    /// elements, devices, internal nodes) each instance renames into its own
    /// names (``X1.R1``, ``X1.mid``).
    #[staticmethod]
    fn subckt(name: &str, pins: Vec<String>) -> PyCircuit {
        let pins: Vec<&str> = pins.iter().map(String::as_str).collect();
        PyCircuit {
            inner: Arc::new(Circuit::subckt(name, &pins)),
        }
    }

    /// A resistor ``name`` between ``a`` and ``b``, ``r`` ohms.
    fn resistor<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        a: &str,
        b: &str,
        r: f64,
    ) -> PyRefMut<'p, Self> {
        slf.edit().resistor(name, a, b, r);
        slf
    }

    /// A capacitor ``name`` between ``a`` and ``b``, ``c`` farads.
    fn capacitor<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        a: &str,
        b: &str,
        c: f64,
    ) -> PyRefMut<'p, Self> {
        slf.edit().capacitor(name, a, b, c);
        slf
    }

    /// An inductor ``name`` between ``a`` and ``b``, ``l`` henries.
    fn inductor<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        a: &str,
        b: &str,
        l: f64,
    ) -> PyRefMut<'p, Self> {
        slf.edit().inductor(name, a, b, l);
        slf
    }

    /// A voltage source ``name`` from ``p`` (+) to ``n`` (-): a DC value or
    /// a ``Waveform``.
    fn voltage_source<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        p: &str,
        n: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'p, Self>> {
        let w = waveform(value)?;
        slf.edit().voltage_source(name, p, n, w);
        Ok(slf)
    }

    /// A current source ``name`` driving current from ``p`` through itself
    /// to ``n``: a DC value or a ``Waveform``.
    fn current_source<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        p: &str,
        n: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'p, Self>> {
        let w = waveform(value)?;
        slf.edit().current_source(name, p, n, w);
        Ok(slf)
    }

    /// A voltage-controlled voltage source: ``V(p, n) = gain * V(cp, cn)``.
    fn vcvs<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        p: &str,
        n: &str,
        cp: &str,
        cn: &str,
        gain: f64,
    ) -> PyRefMut<'p, Self> {
        slf.edit().vcvs(name, p, n, cp, cn, gain);
        slf
    }

    /// A voltage-controlled current source: ``I(p -> n) = gm * V(cp, cn)``.
    fn vccs<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        p: &str,
        n: &str,
        cp: &str,
        cn: &str,
        gm: f64,
    ) -> PyRefMut<'p, Self> {
        slf.edit().vccs(name, p, n, cp, cn, gm);
        slf
    }

    /// A current-controlled current source: ``I(p -> n) = gain * I(ctrl)``,
    /// ``ctrl`` a voltage-defined element.
    fn cccs<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        p: &str,
        n: &str,
        ctrl: &str,
        gain: f64,
    ) -> PyRefMut<'p, Self> {
        slf.edit().cccs(name, p, n, ctrl, gain);
        slf
    }

    /// A current-controlled voltage source: ``V(p, n) = r * I(ctrl)``.
    fn ccvs<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        p: &str,
        n: &str,
        ctrl: &str,
        r: f64,
    ) -> PyRefMut<'p, Self> {
        slf.edit().ccvs(name, p, n, ctrl, r);
        slf
    }

    /// The mutual inductance ``name`` between inductors ``l1`` and ``l2``,
    /// coupling coefficient ``k``.
    fn mutual<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        l1: &str,
        l2: &str,
        k: f64,
    ) -> PyRefMut<'p, Self> {
        slf.edit().mutual(name, l1, l2, k);
        slf
    }

    /// Register the Verilog-A modules ``source`` defines, for ``device``.
    fn module<'p>(mut slf: PyRefMut<'p, Self>, source: &str) -> PyResult<PyRefMut<'p, Self>> {
        slf.edit().module(source).map_err(invalid)?;
        Ok(slf)
    }

    /// A device ``name`` of the Verilog-A module ``module`` (one registered
    /// with ``module``, or a built-in: ``sane_diode``, ``sane_mos``,
    /// ``sane_bjt``, ``sane_jfet``, ``sane_vswitch``, ...), its terminals on
    /// ``nodes`` in the module's port order, its parameters set by keyword;
    /// the others take the module's defaults.
    #[pyo3(signature = (name, module, nodes, **params))]
    fn device<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        module: &str,
        nodes: Vec<String>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'p, Self>> {
        let p = self::params(params)?;
        let nodes: Vec<&str> = nodes.iter().map(String::as_str).collect();
        slf.edit()
            .device(name, module, &nodes, &pairs(&p))
            .map_err(invalid)?;
        Ok(slf)
    }

    /// A junction diode ``name`` from ``anode`` to ``cathode``.
    #[pyo3(signature = (name, anode, cathode, **params))]
    fn diode<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        anode: &str,
        cathode: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'p, Self>> {
        let p = self::params(params)?;
        slf.edit()
            .diode(name, anode, cathode, &pairs(&p))
            .map_err(invalid)?;
        Ok(slf)
    }

    /// A MOSFET ``name``: drain, gate, source and bulk (the source when
    /// not given).
    #[pyo3(signature = (name, d, g, s, b=None, **params))]
    #[allow(clippy::too_many_arguments)]
    fn mosfet<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        d: &str,
        g: &str,
        s: &str,
        b: Option<&str>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'p, Self>> {
        let p = self::params(params)?;
        slf.edit()
            .mosfet(name, d, g, s, b.unwrap_or(s), &pairs(&p))
            .map_err(invalid)?;
        Ok(slf)
    }

    /// A bipolar transistor ``name``: collector, base, emitter.
    #[pyo3(signature = (name, c, b, e, **params))]
    fn bjt<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        c: &str,
        b: &str,
        e: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'p, Self>> {
        let p = self::params(params)?;
        slf.edit().bjt(name, c, b, e, &pairs(&p)).map_err(invalid)?;
        Ok(slf)
    }

    /// A voltage-controlled switch ``name`` between ``a`` and ``b``,
    /// controlled by ``V(cp, cn)``.
    #[pyo3(signature = (name, a, b, cp, cn, **params))]
    #[allow(clippy::too_many_arguments)]
    fn vswitch<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        a: &str,
        b: &str,
        cp: &str,
        cn: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'p, Self>> {
        let p = self::params(params)?;
        slf.edit()
            .vswitch(name, a, b, cp, cn, &pairs(&p))
            .map_err(invalid)?;
        Ok(slf)
    }

    /// A current-controlled switch ``name`` between ``a`` and ``b``,
    /// controlled by the current of the voltage-defined element ``ctrl``.
    #[pyo3(signature = (name, a, b, ctrl, **params))]
    fn cswitch<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        a: &str,
        b: &str,
        ctrl: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'p, Self>> {
        let p = self::params(params)?;
        slf.edit()
            .cswitch(name, a, b, ctrl, &pairs(&p))
            .map_err(invalid)?;
        Ok(slf)
    }

    /// A power port ``name``: an ideal source behind a ``z0`` series
    /// resistor onto ``p``, ``n`` the reference; the drive of AC and the
    /// ports of S-parameters.
    #[pyo3(signature = (name, p, n, z0=50.0))]
    fn port<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        p: &str,
        n: &str,
        z0: f64,
    ) -> PyResult<PyRefMut<'p, Self>> {
        slf.edit().port(name, p, n, z0).map_err(invalid)?;
        Ok(slf)
    }

    /// An instance ``name`` of the subcircuit ``body`` (see ``subckt``), its
    /// pins on ``nodes``, in pin order.
    fn instance<'p>(
        mut slf: PyRefMut<'p, Self>,
        name: &str,
        body: &PyCircuit,
        nodes: Vec<String>,
    ) -> PyResult<PyRefMut<'p, Self>> {
        let nodes: Vec<&str> = nodes.iter().map(String::as_str).collect();
        let body = body.inner.clone();
        slf.edit().instance(name, &body, &nodes).map_err(invalid)?;
        Ok(slf)
    }

    /// Bind the parameter ``name`` (``R1``, ``D1.is``, ``X1.R1``,
    /// ``$temp``) to ``value``.
    fn set<'p>(mut slf: PyRefMut<'p, Self>, name: &str, value: f64) -> PyRefMut<'p, Self> {
        slf.edit().set(name, value);
        slf
    }

    /// Seed the operating point with ``V(node) = volts`` (``.nodeset``).
    fn nodeset<'p>(mut slf: PyRefMut<'p, Self>, node: &str, volts: f64) -> PyRefMut<'p, Self> {
        slf.edit().nodeset(node, volts);
        slf
    }

    /// The node names by index; index 0 is ground.
    #[getter]
    fn node_names(&self) -> Vec<String> {
        self.inner.node_names()
    }

    /// The values bound to parameters, by name.
    #[getter]
    fn values(&self) -> std::collections::HashMap<String, f64> {
        self.inner
            .values
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// A subcircuit body's interface nodes (empty at the top level).
    #[getter]
    fn pins(&self) -> Vec<String> {
        self.inner.pins.clone()
    }

    /// The linear and controlled elements of this level, each as
    /// ``{"name", "kind", "nodes", "control"}`` (``control`` the sensed
    /// element of a current-controlled source).
    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let names = self.inner.node_names();
        (self.inner.elements.elements().iter())
            .map(|e| {
                let d = PyDict::new_bound(py);
                d.set_item("name", &e.name)?;
                d.set_item("kind", kind_name(e.kind))?;
                d.set_item("nodes", (&names[e.a], &names[e.b]))?;
                d.set_item("control", &e.ctrl_elem)?;
                Ok(d)
            })
            .collect()
    }

    /// The inductor couplings of this level: ``(name, l1, l2)``.
    #[getter]
    fn couplings(&self) -> Vec<(String, String, String)> {
        (self.inner.elements.couplings().iter())
            .map(|k| (k.name.clone(), k.l1.clone(), k.l2.clone()))
            .collect()
    }

    /// The number of devices of this level.
    #[getter]
    fn device_count(&self) -> usize {
        self.inner.devices.len()
    }

    /// The power ports: ``(name, node, z0)``.
    #[getter]
    fn ports(&self) -> Vec<(String, String, f64)> {
        (self.inner.ports.iter())
            .map(|p| (p.name.clone(), p.node.clone(), p.z0))
            .collect()
    }

    fn __repr__(&self) -> String {
        let c = &*self.inner;
        format!(
            "<sane.Circuit: {} nodes, {} elements, {} devices, {} instances>",
            c.node_count(),
            c.elements.elements().len(),
            c.devices.len(),
            c.instances.len()
        )
    }
}

#[pymethods]
impl PyWaveform {
    /// A DC value.
    #[staticmethod]
    fn dc(v: f64) -> PyWaveform {
        PyWaveform {
            inner: Waveform::Dc(v),
        }
    }

    /// ``offset + amplitude * sin(2 pi freq t)``.
    #[staticmethod]
    fn sin(offset: f64, amplitude: f64, freq: f64) -> PyWaveform {
        PyWaveform {
            inner: Waveform::Sin {
                offset,
                amplitude,
                freq,
            },
        }
    }

    /// ``v1`` until ``delay``, then a trapezoid to ``v2`` (``rise``,
    /// ``width``, ``fall``) every ``period``; a zero ``fall``, ``width`` or
    /// ``period`` takes SPICE's default.
    #[staticmethod]
    #[pyo3(signature = (v1, v2, delay=0.0, rise=0.0, fall=0.0, width=0.0, period=0.0))]
    fn pulse(
        v1: f64,
        v2: f64,
        delay: f64,
        rise: f64,
        fall: f64,
        width: f64,
        period: f64,
    ) -> PyWaveform {
        PyWaveform {
            inner: Waveform::Pulse {
                v1,
                v2,
                delay,
                rise,
                fall,
                width,
                period,
            },
        }
    }

    /// A double exponential from ``v1`` toward ``v2`` at ``td1`` (time
    /// constant ``tau1``), back at ``td2`` (``tau2``).
    #[staticmethod]
    #[pyo3(signature = (v1, v2, td1=0.0, tau1=0.0, td2=0.0, tau2=0.0))]
    fn exp(v1: f64, v2: f64, td1: f64, tau1: f64, td2: f64, tau2: f64) -> PyWaveform {
        PyWaveform {
            inner: Waveform::Exp {
                v1,
                v2,
                td1,
                tau1,
                td2,
                tau2,
            },
        }
    }

    /// Piecewise linear through the points ``(t, value)``.
    #[staticmethod]
    fn pwl(points: Vec<(f64, f64)>) -> PyWaveform {
        PyWaveform {
            inner: Waveform::Pwl(points),
        }
    }

    fn __repr__(&self) -> String {
        format!("<sane.Waveform {:?}>", self.inner)
    }
}

impl PyModel {
    /// The model of a circuit (see ``sane.Model``).
    pub fn of(py: Python<'_>, circuit: &PyCircuit) -> PyResult<PyModel> {
        let c = circuit.inner.clone();
        crate::run(py, || sane_analysis::Model::new(c)).map(|inner| PyModel { inner })
    }
}
