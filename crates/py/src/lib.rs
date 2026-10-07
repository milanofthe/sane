//! Python bindings for SANE.
//!
//! SANE does not generate code. All orchestration, numeric analysis and the
//! sensitivity/gradient machinery live in `sane_analysis::Model` (pure Rust);
//! these bindings are thin marshalling wrappers over it (`PyModel` delegates to
//! `Model`): a numeric interface (`Model.currents`, `Model.jacobian_i_x`,
//! `Model.ac_response`, ... evaluated in Rust).

// The `#[pymethods]` macro expands `?`/return paths into `PyErr: From<PyErr>`
// conversions clippy flags as useless; they are macro-generated, not our code.
#![allow(clippy::useless_conversion)]

// Extraction (parse + symbolic build + AD + tape compile) is dominated by many
// small allocations; the OS allocator's global lock is the binding constraint
// and contends badly under the worker pool. mimalloc removes that wall.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sane_dae::DeviceInstance;
use sane_device::CSwitch;
use sane_mna::{Kind, SourceFn};
use sane_netlist::ParsedCircuit;

/// A sparse matrix as `(rows, cols, values)`.
type Coo = (Vec<usize>, Vec<usize>, Vec<f64>);

/// A short lowercase name for an element kind, for introspection.
fn kind_name(k: &Kind) -> &'static str {
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

/// A circuit builder. Nodes are integers; 0 is ground.
#[pyclass]
struct Circuit {
    /// The circuit, parsed or built, as every model of it is set up from
    /// (see `sane_analysis::Model::from_parsed`).
    parsed: ParsedCircuit,
    /// The netlist it was parsed from, while no builder call edited it.
    source: Option<String>,
}

#[pymethods]
impl Circuit {
    #[new]
    fn new() -> Self {
        Circuit {
            parsed: ParsedCircuit::new(),
            source: None,
        }
    }

    /// Node names indexed by internal node id (`node_names()[k]` is the node
    /// behind DAE unknown `v{k}`). Index 0 is ground.
    fn node_names(&self) -> Vec<String> {
        self.parsed.node_names.clone()
    }

    /// Power ports (`P` elements) in deck order: `(name, node, z0)` per port.
    /// The port name doubles as the AC/SP drive-source name.
    fn ports(&self) -> Vec<(String, String, f64)> {
        (self.parsed.ports.iter())
            .map(|p| (p.name.clone(), p.node.clone(), p.z0))
            .collect()
    }

    /// Bound element / parameter values by symbol name (e.g. `R1`, `D1.Is`).
    /// Populated by `parse`; empty for a programmatically built circuit (whose
    /// values are tracked on the Python wrapper until extraction).
    fn values(&self) -> HashMap<String, f64> {
        self.parsed
            .values
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    fn resistor(&mut self, name: &str, a: usize, b: usize) {
        self.edit().circuit.resistor(name, a, b);
    }
    fn capacitor(&mut self, name: &str, a: usize, b: usize) {
        self.edit().circuit.capacitor(name, a, b);
    }
    fn inductor(&mut self, name: &str, a: usize, b: usize) {
        self.edit().circuit.inductor(name, a, b);
    }
    fn voltage_source(&mut self, name: &str, a: usize, b: usize) {
        self.edit().circuit.voltage_source(name, a, b);
    }
    fn current_source(&mut self, name: &str, a: usize, b: usize) {
        self.edit().circuit.current_source(name, a, b);
    }
    fn vccs(&mut self, name: &str, np: usize, nm: usize, cp: usize, cm: usize) {
        self.edit().circuit.vccs(name, np, nm, cp, cm);
    }
    fn vcvs(&mut self, name: &str, np: usize, nm: usize, cp: usize, cm: usize) {
        self.edit().circuit.vcvs(name, np, nm, cp, cm);
    }
    fn cccs(&mut self, name: &str, np: usize, nm: usize, ctrl: &str) {
        self.edit().circuit.cccs(name, np, nm, ctrl);
    }
    fn ccvs(&mut self, name: &str, np: usize, nm: usize, ctrl: &str) {
        self.edit().circuit.ccvs(name, np, nm, ctrl);
    }
    fn mutual(&mut self, name: &str, l1: &str, l2: &str) {
        self.edit().circuit.mutual(name, l1, l2);
    }

    fn diode(&mut self, name: &str, anode: usize, cathode: usize) {
        let dev = sane_veriloga::builtin_device("sane_diode", name, &[]);
        self.push_device(Box::new(dev), vec![anode, cathode]);
    }
    fn mosfet(&mut self, name: &str, d: usize, g: usize, s: usize, body: usize) {
        let dev = sane_veriloga::builtin_device("sane_mos", name, &[]);
        self.push_device(Box::new(dev), vec![d, g, s, body]);
    }
    fn bjt(&mut self, name: &str, c: usize, b: usize, e: usize) {
        let dev = sane_veriloga::builtin_device("sane_bjt", name, &[]);
        self.push_device(Box::new(dev), vec![c, b, e]);
    }
    fn vswitch(&mut self, name: &str, a: usize, b: usize, cp: usize, cm: usize) {
        let dev = sane_veriloga::builtin_device("sane_vswitch", name, &[]);
        self.push_device(Box::new(dev), vec![a, b, cp, cm]);
    }
    fn cswitch(&mut self, name: &str, a: usize, b: usize, ctrl: &str) {
        self.push_device(Box::new(CSwitch::new(name, ctrl)), vec![a, b]);
    }

    /// Attach a time-domain source shape to the most recently added element.
    /// Parameter values (e.g. `V1.sin_w`) are supplied numerically at eval time.
    fn source_sin(&mut self) {
        self.edit().circuit.set_source(SourceFn::Sin);
    }
    fn source_pulse(&mut self) {
        self.edit().circuit.set_source(SourceFn::Pulse);
    }
    fn source_exp(&mut self) {
        self.edit().circuit.set_source(SourceFn::Exp);
    }
    fn source_pwl(&mut self, n: usize) {
        self.edit().circuit.set_source(SourceFn::Pwl(n));
    }

    /// The top level's linear/controlled elements as tuples `(name, kind,
    /// node_a, node_b, control_element)`. Nonlinear devices (D/M/Q/switches)
    /// and the elements inside subcircuit instances are not included.
    fn elements(&self) -> Vec<(String, String, usize, usize, Option<String>)> {
        self.parsed
            .circuit
            .elements()
            .iter()
            .map(|e| {
                (
                    e.name.clone(),
                    kind_name(&e.kind).to_string(),
                    e.a,
                    e.b,
                    e.ctrl_elem.clone(),
                )
            })
            .collect()
    }

    /// Inductive couplings as tuples `(name, inductor_1, inductor_2)`.
    fn couplings(&self) -> Vec<(String, String, String)> {
        self.parsed
            .circuit
            .couplings()
            .iter()
            .map(|k| (k.name.clone(), k.l1.clone(), k.l2.clone()))
            .collect()
    }

    /// Number of the top level's nonlinear device instances (D/M/Q/switches);
    /// devices inside subcircuit instances are not counted.
    fn device_count(&self) -> usize {
        self.parsed.devices.len()
    }

    /// Extract the DAE `I(x, t) + d/dt Q(x) = 0` (with its analytic Jacobians).
    fn extract_dae(&self) -> PyResult<PyModel> {
        sane_analysis::Model::from_parsed(&self.parsed, self.source.as_deref())
            .map(|inner| PyModel { inner })
            .map_err(model_err)
    }
}

impl Circuit {
    /// The circuit to edit: no longer the netlist it was parsed from.
    fn edit(&mut self) -> &mut ParsedCircuit {
        self.source = None;
        &mut self.parsed
    }

    fn push_device(&mut self, model: Box<dyn sane_device::DeviceModel>, terminals: Vec<usize>) {
        self.edit()
            .devices
            .push(DeviceInstance::new(model, terminals));
    }
}

/// Parse a SPICE-like netlist into a [`Circuit`].
#[pyfunction]
fn parse(netlist: &str) -> PyResult<Circuit> {
    let parsed = sane_netlist::parse(netlist).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Circuit {
        parsed,
        source: Some(netlist.to_string()),
    })
}

/// Map an element name to the symbol name under which its value parameter is
/// bound, keeping names out of the reserved unknown namespace (`v{k}`,
/// `i_*`, `t`, ...). Programmatic builders bind values by this name so the key
/// matches the parameter the netlist front-end and DAE assembly produce.
#[pyfunction]
fn value_symbol_name(name: &str) -> String {
    sane_mna::value_symbol_name(name)
}

/// The thread count for the parallel work (the sweeps, and in every solve
/// the device instances of each evaluation): `0` = the default, `n` = `n`
/// threads. Takes effect before the pools are first used.
#[pyfunction]
fn set_parallelism(threads: usize) {
    sane_solve::set_parallelism(threads);
}

/// Set the native log level by name ("debug"/"info"/"warning"/"error"/"off").
/// Enables the engine's fastsim-style progress logging (DC homotopy fallbacks,
/// transient progress bar, sweep progress) on stdout/stderr. Unknown
/// names default to INFO.
#[pyfunction]
fn set_log_level(level: &str) {
    sane_core::log::set_level(
        sane_core::LogLevel::parse(level).unwrap_or(sane_core::LogLevel::Info),
    );
}

/// Drain and return the engine's captured correctness warnings (gmin
/// regularization, out-of-range device parameters) accumulated since the last
/// drain, clearing the buffer. The Python layer calls this at analysis
/// boundaries and re-raises each as a catchable `SaneConvergenceWarning`,
/// independent of the log level (issue #54).
#[pyfunction]
fn drain_warnings() -> Vec<String> {
    sane_core::log::drain_captured()
}

/// Fit tabulated p-port admittance data `Y(f)` (flattened p^2 entries per
/// frequency, split into real/imag parts) with relaxed vector fitting and
/// return the deterministic Verilog-A macromodel source targeting SANE's
/// AD-friendly grammar subset. Returns `(va_source, fit_error, n_poles)`.
#[pyfunction]
#[pyo3(signature = (freqs, y_re, y_im, n_ports, name, port_names=None, z0=50.0,
                    tol=1e-4, max_poles=40, enforce_passivity=true,
                    threshold=1e-2, force=false, date=String::new()))]
#[allow(clippy::too_many_arguments)]
fn vectfit_verilog_a(
    freqs: Vec<f64>,
    y_re: Vec<Vec<f64>>,
    y_im: Vec<Vec<f64>>,
    n_ports: usize,
    name: String,
    port_names: Option<Vec<String>>,
    z0: f64,
    tol: f64,
    max_poles: usize,
    enforce_passivity: bool,
    threshold: f64,
    force: bool,
    date: String,
) -> PyResult<(String, f64, usize)> {
    let nf = freqs.len();
    let pp = n_ports * n_ports;
    if y_re.len() != nf || y_im.len() != nf {
        return Err(PyValueError::new_err(
            "y_re/y_im must have one row per frequency",
        ));
    }
    let data: Vec<Vec<num_complex::Complex64>> = y_re
        .iter()
        .zip(&y_im)
        .map(|(re, im)| {
            if re.len() != pp || im.len() != pp {
                return Err(PyValueError::new_err(format!(
                    "each row must carry {pp} flattened entries"
                )));
            }
            Ok(re
                .iter()
                .zip(im)
                .map(|(&a, &b)| num_complex::Complex64::new(a, b))
                .collect())
        })
        .collect::<PyResult<_>>()?;
    let ports = port_names.unwrap_or_else(|| (1..=n_ports).map(|i| format!("p{i}")).collect());
    let mut model = vectfit::ratmodel::fit(
        &freqs,
        &data,
        n_ports,
        max_poles,
        tol,
        ports,
        vec![z0; n_ports],
    );
    if enforce_passivity {
        model.enforce_passivity(20);
    }
    let va = model
        .to_verilog_a(&name, &date, force, threshold)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((va, model.fit_error, model.n_poles()))
}

/// Begin collecting the engine's internal per-stage timings into the global
/// profiling sink (clearing any prior run). Every instrumented stage that flows
/// through the native logger -- `log_stage!` / `time_stage!` / `log::scope`,
/// e.g. `dc/newton`, `ac/eval_b`, `tran/irk_step`, the extract/compile phases --
/// is recorded, independent of the log level. Pair with [`profile_take`].
#[pyfunction]
fn profile_begin() {
    sane_core::profile::collect_begin();
}

/// Stop collecting and return the `(stage, total_ms, count)` breakdown per
/// distinct stage name, in first-seen order. `count` is how many times the stage
/// fired (e.g. `tran/irk_step` once per time step, `ac/eval_b` once per frequency
/// point, `dc/newton` once per continuation attempt), so a per-step cost is
/// `total_ms / count`. Empty if collection was not active.
#[pyfunction]
fn profile_take() -> Vec<(String, f64, u32)> {
    let prof = sane_core::profile::collect_take();
    let mut order: Vec<String> = Vec::new();
    let mut idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut tot: Vec<f64> = Vec::new();
    let mut cnt: Vec<u32> = Vec::new();
    for (n, ms) in prof.millis() {
        match idx.get(&n) {
            Some(&i) => {
                tot[i] += ms;
                cnt[i] += 1;
            }
            None => {
                idx.insert(n.clone(), tot.len());
                order.push(n);
                tot.push(ms);
                cnt.push(1);
            }
        }
    }
    order
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, tot[i], cnt[i]))
        .collect()
}

// ---------------------------------------------------------------------------
// Embeddable analysis facade: thin pyclass wrappers over `sane_analysis::Model`.
// The orchestration, parameter store, name resolution and solved state all live
// in Rust; these types only marshal arguments and results across the boundary.
// ---------------------------------------------------------------------------

fn model_err(e: sane_analysis::ModelError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn overrides_vec(ov: &HashMap<String, f64>) -> Vec<(&str, f64)> {
    ov.iter().map(|(k, v)| (k.as_str(), *v)).collect()
}

/// Embeddable analysis object: build it from a netlist, then call any analysis,
/// which returns a result handle. The orchestration, named parameter store and
/// solved state all live in Rust (``sane_analysis::Model``); this is a thin handle.
/// The identical API is available natively in Rust for embedding without Python.
///
/// >>> sim = sane.Model.from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end")
/// >>> sim.operating_point()["out"]
/// 2.5
/// >>> sim.set("R2", 3e3); sim.operating_point()["out"]   # persistent mutation
/// >>> sim.operating_point({"R2": 1e3})["out"]            # non-destructive override
#[pyclass(name = "Model")]
struct PyModel {
    inner: sane_analysis::Model,
}

#[pymethods]
impl PyModel {
    fn params(&self) -> Vec<String> {
        self.inner.params().to_vec()
    }
    fn unknowns(&self) -> Vec<String> {
        self.inner.unknowns().to_vec()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn values(&self) -> HashMap<String, f64> {
        self.inner.values()
    }

    /// Fold parameters to their current values: each becomes a constant in a
    /// derived `Model` (sharing this context), shrinking the graph and removing it
    /// from the parameter set (no dF/dp column, no sensitivity). A path is a single
    /// parameter (`"X1.R1"`) or a group prefix (`"X1"` -> all `X1.*`). Returns the
    /// folded model; this one is unchanged.
    fn fold(&self, paths: Vec<String>) -> PyResult<PyModel> {
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.fold(&refs).map_err(model_err)?;
        Ok(PyModel { inner })
    }
    /// Fold every parameter except the ones under `paths` (the inverse of
    /// `fold`). Returns the folded model; this one is unchanged.
    fn keep(&self, paths: Vec<String>) -> PyResult<PyModel> {
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.keep(&refs).map_err(model_err)?;
        Ok(PyModel { inner })
    }

    // --- parameter store (low-level; the hierarchical Python API builds on it) ---

    /// Build the parameter vector in column order from the bound store plus
    /// optional per-call `overrides` (keyed by user name).
    #[pyo3(signature = (overrides=None))]
    fn param_vector(&self, overrides: Option<HashMap<String, f64>>) -> Vec<f64> {
        match overrides {
            None => self.inner.pvec(&[]),
            Some(ov) => self.inner.pvec(&overrides_vec(&ov)),
        }
    }

    /// Parameters referenced in the residual but not bound to a value (numeric
    /// analyses use 0 for them; symbolic analyses keep them free).
    fn unbound_params(&self) -> Vec<String> {
        self.inner.unbound_params()
    }

    /// Set one parameter (reserved-namespace resolved); raises if not a parameter.
    fn set_param(&self, name: &str, value: f64) -> PyResult<()> {
        self.inner.set(name, value).map_err(model_err)
    }

    /// Read one parameter's bound value; raises if `name` is not a parameter.
    fn get_param(&self, name: &str) -> PyResult<f64> {
        self.inner
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("'{name}' is not a parameter")))
    }

    /// Bulk set (each key resolved + validated, atomic on a bad key).
    fn set_params(&self, vals: HashMap<String, f64>) -> PyResult<()> {
        for k in vals.keys() {
            if !self.inner.is_param(k) {
                return Err(PyValueError::new_err(format!("'{k}' is not a parameter")));
            }
        }
        let vals: Vec<(&str, f64)> = vals.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        self.inner.set_many(&vals).map_err(model_err)
    }

    // --- graph transforms (return a new Model sharing this context) --------

    /// Operating-point-guided graph reduction: drop branch contributions that
    /// are negligible at `(x, p)`. Returns a new reduced `Model` sharing this
    /// context, plus the pruned `(branch, node)` pairs.
    fn prune_graph(
        &self,
        rel_tol: f64,
        x: Vec<f64>,
        p: Vec<f64>,
        omegas: Vec<f64>,
    ) -> (PyModel, Vec<(String, String)>) {
        let (model, pruned) = self.inner.prune_graph(rel_tol, &x, &p, &omegas);
        (PyModel { inner: model }, pruned)
    }

    /// Exactly eliminate internal resistive nodes (Schur/series reduction).
    /// `keep` protects node-unknown names. Returns the reduced `Model` and the
    /// eliminated node names.
    fn eliminate_nodes(&self, keep: Vec<String>) -> (PyModel, Vec<String>) {
        let (model, gone) = self.inner.eliminate_nodes(&keep);
        (PyModel { inner: model }, gone)
    }

    /// Linearise about the operating point into the small-signal mass-matrix DAE
    /// `G dx + d/dt (C dx) = 0`, sharing this context. Returns the linearised
    /// `Model`.
    fn linearize(&self) -> PyModel {
        let model = self.inner.linearize();
        PyModel { inner: model }
    }

    // --- low-level whole-circuit analyses (explicit index/state forms) -----
    // The rich Python result objects orchestrate the OP solve themselves and
    // call these with explicit `(x, p)`; the name-based high-level forms above
    // are the ergonomic entry points.

    /// Output-referred noise PSD from the unknown at `out_idx`, at `(x, p)`.
    fn noise_raw(
        &self,
        py: Python<'_>,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        fstart: f64,
        fstop: f64,
        points: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        py.allow_threads(|| self.inner.noise_raw(out_idx, x, p, fstart, fstop, points))
            .map_err(model_err)
    }

    /// Linearised descriptor state-space `(states, E, A, B, C, D)` at `(x, p)`.
    #[allow(clippy::type_complexity)]
    fn state_space_raw(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> PyResult<(
        Vec<String>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        f64,
    )> {
        self.inner
            .state_space_raw(input, out_idx, x, p)
            .map_err(model_err)
    }

    /// Temperature sweep of the unknown at `out_idx` over `[tstart, tstop]` degC.
    fn temp_sweep_raw(
        &self,
        py: Python<'_>,
        out_idx: usize,
        p0: Vec<f64>,
        tstart: f64,
        tstop: f64,
        points: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        py.allow_threads(|| {
            self.inner
                .temp_sweep_raw(out_idx, p0, tstart, tstop, points)
        })
        .map_err(model_err)
    }

    /// Dominant-pole model-order reduction of `input -> out_idx` to `order` poles,
    /// at `(x, p)`. Returns `(freqs, full_db, reduced_db, poles, zeros, max_err_db)`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn model_reduce_raw(
        &self,
        py: Python<'_>,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        order: usize,
        fstart: f64,
        fstop: f64,
        points: usize,
    ) -> PyResult<(
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<(f64, f64)>,
        Vec<(f64, f64)>,
        f64,
    )> {
        py.allow_threads(|| {
            self.inner
                .model_reduce_raw(input, out_idx, x, p, order, fstart, fstop, points)
        })
        .map_err(model_err)
    }

    // --- low-level numeric interface (evaluated in Rust) ------------------

    /// The currents `I(x, p, t)`. Every row reads `I(x, t) + d/dt Q(x)`.
    fn currents(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
        self.inner.currents(x, p, t).map_err(model_err)
    }
    /// The charges `Q(x, p)`.
    fn charges(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
        self.inner.charges(x, p, t).map_err(model_err)
    }
    /// The residual `F = I(x, t) + C(x) x'` at the state `x` moving at the
    /// rate `xdot`.
    fn residual(&self, x: Vec<f64>, xdot: Vec<f64>, p: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
        self.inner.residual(x, xdot, p, t).map_err(model_err)
    }
    /// `G = dI/dx` as a dense matrix.
    fn jacobian_i_x(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> PyResult<Vec<Vec<f64>>> {
        self.inner.jacobian_i_x(x, p, t).map_err(model_err)
    }
    /// `C = dQ/dx` as a dense matrix.
    fn jacobian_q_x(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> PyResult<Vec<Vec<f64>>> {
        self.inner.jacobian_q_x(x, p, t).map_err(model_err)
    }
    /// Sparse `G = dI/dx` as `(rows, cols, values)` (COO).
    fn jacobian_i_x_sparse(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> PyResult<Coo> {
        self.inner.jacobian_i_x_sparse(x, p, t).map_err(model_err)
    }
    /// Sparse `C = dQ/dx` as `(rows, cols, values)` (COO).
    fn jacobian_q_x_sparse(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> PyResult<Coo> {
        self.inner.jacobian_q_x_sparse(x, p, t).map_err(model_err)
    }
    /// Sparse `dI/dp` as `(rows, cols, values)` (COO).
    fn jacobian_i_p_sparse(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> PyResult<Coo> {
        self.inner.jacobian_i_p_sparse(x, p, t).map_err(model_err)
    }
    /// Sparse `dQ/dp` as `(rows, cols, values)` (COO).
    fn jacobian_q_p_sparse(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> PyResult<Coo> {
        self.inner.jacobian_q_p_sparse(x, p, t).map_err(model_err)
    }
    /// Number of structural nonzeros in the sparse `G` pattern.
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }
    /// Schur partition sizes `(linear_block, nonlinear_block)`, or `None`.
    fn partition_sizes(&self) -> Option<(usize, usize)> {
        self.inner.partition_sizes()
    }

    /// Exact input-coupling vector `dI/d(input)` for the named source, by
    /// symbolic differentiation (autodiff), evaluated at `(x, p, t)`.
    fn jacobian_i_input(
        &self,
        input: &str,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> PyResult<Vec<f64>> {
        self.inner
            .jacobian_i_input(input, x, p, t)
            .map_err(model_err)
    }

    /// Exact first-order sensitivity `dy/dp` of `y = output` (an unknown name)
    /// w.r.t. every parameter at the point `x`, via the adjoint. Returns
    /// `(param_names, dy/dp)`.
    fn sensitivity(
        &self,
        py: Python<'_>,
        output: &str,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> PyResult<(Vec<String>, Vec<f64>)> {
        py.allow_threads(|| self.inner.sensitivity(output, x, p, t))
            .map_err(model_err)
    }

    /// Exact second-order sensitivity (Hessian) of `y = output` w.r.t. a `subset`
    /// of parameters, by the second-order adjoint (exact AD directional
    /// derivatives). Returns the dense symmetric `len(subset) x len(subset)` matrix.
    fn hessian(
        &self,
        py: Python<'_>,
        output: &str,
        subset: Vec<String>,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> PyResult<Vec<Vec<f64>>> {
        py.allow_threads(|| self.inner.hessian(output, subset, x, p, t))
            .map_err(model_err)
    }

    /// Exact total derivatives of the small-signal matrices w.r.t. `param`,
    /// including the operating-point shift, via AD: returns `(dG, dC, dB)`.
    fn ac_derivatives(
        &self,
        py: Python<'_>,
        input: &str,
        param: &str,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>)> {
        py.allow_threads(|| self.inner.ac_derivatives(input, param, x, p, t))
            .map_err(model_err)
    }

    /// Exact forward transient sensitivity: integrate the circuit and the
    /// sensitivity systems `dx(t)/dp` for each parameter in `subset` together.
    /// Returns `(n, traj)` (see the analysis layer for the slice layout).
    #[pyo3(signature = (subset, t_eval, rtol=1e-4, atol=1e-7, values=None))]
    fn transient_sensitivity(
        &self,
        py: Python<'_>,
        subset: Vec<String>,
        t_eval: Vec<f64>,
        rtol: f64,
        atol: f64,
        values: Option<HashMap<String, f64>>,
    ) -> PyResult<(usize, Vec<Vec<f64>>)> {
        py.allow_threads(|| {
            self.inner
                .transient_sensitivity(subset, t_eval, rtol, atol, values)
        })
        .map_err(model_err)
    }

    /// Fixed-grid ESDIRK32 transient on the exact grid `t_eval` (one implicit
    /// step per interval): the forward pass the discrete adjoint differentiates.
    /// `p` in `params()` order; `x0=None` starts from the DC operating point.
    #[pyo3(signature = (p, t_eval, x0=None, dc_guess=None))]
    fn solve_transient_grid(
        &self,
        py: Python<'_>,
        p: Vec<f64>,
        t_eval: Vec<f64>,
        x0: Option<Vec<f64>>,
        dc_guess: Option<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        py.allow_threads(|| self.inner.solve_transient_grid(p, x0, t_eval, dc_guess))
            .map_err(model_err)
    }

    /// Discrete transient adjoint (VJP): cotangents `dL/dx_k` over the
    /// fixed-grid ESDIRK32 trajectory on `t_eval` in, `(param_names, dL/dp)` out --
    /// every parameter from one backward sweep, cost independent of the
    /// parameter count.
    #[pyo3(signature = (t_eval, cotangent, values=None, dc_guess=None))]
    fn transient_adjoint(
        &self,
        py: Python<'_>,
        t_eval: Vec<f64>,
        cotangent: Vec<Vec<f64>>,
        values: Option<HashMap<String, f64>>,
        dc_guess: Option<Vec<f64>>,
    ) -> PyResult<(Vec<String>, Vec<f64>)> {
        py.allow_threads(|| {
            self.inner
                .transient_adjoint(t_eval, cotangent, values, dc_guess)
        })
        .map_err(model_err)
    }

    /// Solve the DC operating point in Rust. `p` is the parameter vector in
    /// `params()` order. The convergence aids and the per-component criterion are
    /// individually toggleable; `nodeset` runs the stiff-pin pre-phase.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (p, x0=None, tol=1e-10, max_iter=100, device_limiting=None,
        line_search=None, gmin_continuation=None,
        source_continuation=None, companion_continuation=None, node_adaptive=None,
        pseudo_transient=None, partition=None, nodeset=None, reltol=None,
        abstol=None, vntol=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve_dc(
        &self,
        py: Python<'_>,
        p: Vec<f64>,
        x0: Option<Vec<f64>>,
        tol: f64,
        max_iter: usize,
        device_limiting: Option<bool>,
        line_search: Option<bool>,
        gmin_continuation: Option<bool>,
        source_continuation: Option<bool>,
        companion_continuation: Option<bool>,
        node_adaptive: Option<bool>,
        pseudo_transient: Option<bool>,
        partition: Option<bool>,
        nodeset: Option<Vec<(usize, f64)>>,
        reltol: Option<f64>,
        abstol: Option<f64>,
        vntol: Option<f64>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.inner.solve_dc(
                p,
                x0,
                tol,
                max_iter,
                device_limiting,
                line_search,
                gmin_continuation,
                source_continuation,
                companion_continuation,
                node_adaptive,
                pseudo_transient,
                partition,
                nodeset,
                reltol,
                abstol,
                vntol,
            )
        })
        .map_err(model_err)
    }

    /// The gmin at which the most recent DC solve held, or `None` if it reached
    /// the true `GMIN_DC` floor. `Some(g)` marks a converged-but-regularized
    /// (physically suspect) operating point; query right after `solve_dc` (#54).
    fn regularized_at_gmin(&self) -> Option<f64> {
        self.inner.regularized_at_gmin()
    }

    /// `(unknown index, relative first-order shift)` of the worst node voltage
    /// the gmin shunt sets in the most recent DC solve, else `None`: the detail
    /// behind `regularized_at_gmin`. Query right after `solve_dc` (#54).
    fn gmin_dominance(&self) -> Option<(usize, f64)> {
        self.inner.gmin_dominance()
    }

    /// Transient solve over `t_eval` on the Rust DAE. `x0` defaults to the DC
    /// operating point; `method` is `"esdirk32"` (default, L-stable analog
    /// workhorse) or `"trap"` (trapezoidal, one implicit solve per step -- the
    /// fast choice for switching / digital-style transients). Returns one state
    /// vector per time point.
    #[pyo3(signature = (p, t_eval, x0=None, rtol=1e-4, atol=1e-7, dt_max=None, method=None))]
    fn solve_transient(
        &self,
        py: Python<'_>,
        p: Vec<f64>,
        t_eval: Vec<f64>,
        x0: Option<Vec<f64>>,
        rtol: f64,
        atol: f64,
        dt_max: Option<f64>,
        method: Option<&str>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let m = sane_solve::TransientMethod::from_name(method.unwrap_or(""))
            .ok_or_else(|| PyValueError::new_err("method must be 'esdirk32' or 'trap'"))?;
        py.allow_threads(|| {
            self.inner
                .solve_transient(m, p, t_eval, x0, rtol, atol, dt_max)
        })
        .map_err(model_err)
    }

    /// Switching events of the most recent transient: `(name, t, direction)`.
    fn transient_events(&self) -> Vec<(String, f64, i8)> {
        self.inner.transient_events()
    }

    /// Single-tone harmonic balance: periodic steady state at fundamental `f0`
    /// (Hz). Returns `(spectra, converged, iters, residual_norm, setup_ms, solve_ms)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (p, f0=0.0, harmonics=8, x0=None, oversample=16, tol=1e-10,
        max_iter=60, samples=None, continuation=None))]
    fn solve_hb(
        &self,
        py: Python<'_>,
        p: Vec<f64>,
        f0: f64,
        harmonics: usize,
        x0: Option<Vec<f64>>,
        oversample: usize,
        tol: f64,
        max_iter: usize,
        samples: Option<usize>,
        continuation: Option<bool>,
    ) -> PyResult<(Vec<Vec<(f64, f64)>>, bool, usize, f64, f64, f64, f64)> {
        py.allow_threads(|| {
            self.inner.solve_hb(
                p,
                f0,
                harmonics,
                x0,
                oversample,
                tol,
                max_iter,
                samples,
                continuation,
            )
        })
        .map_err(model_err)
    }

    /// All-parameter sensitivity of the harmonic-balance steady-state
    /// coefficients at unknown `out_idx`, evaluated at the converged `spectra`.
    /// Returns, per harmonic `k = 0..=K`, the complex gradient
    /// `dX_{out,k}/dp_j` over every parameter as `(name, re, im)` rows.
    ///
    /// Exact autodiff: the implicit-function adjoint on the two-sided
    /// harmonic-balance Jacobian, with `dF/dp` routed through the AFT (see
    /// [`CompiledHb::coeff_gradient`]). `f0`, `harmonics`, `oversample`,
    /// `samples` must match the solve that produced `spectra` so the AFT grid
    /// is reconstructed identically.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (out_idx, spectra, p, f0, harmonics=8, oversample=16, samples=None))]
    fn hb_gradient(
        &self,
        py: Python<'_>,
        out_idx: usize,
        spectra: Vec<Vec<(f64, f64)>>,
        p: Vec<f64>,
        f0: f64,
        harmonics: usize,
        oversample: usize,
        samples: Option<usize>,
    ) -> PyResult<Vec<Vec<(String, f64, f64)>>> {
        py.allow_threads(|| {
            self.inner
                .hb_gradient(out_idx, spectra, p, f0, harmonics, oversample, samples)
        })
        .map_err(model_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (out_idx, k_metric, subset, spectra, p, f0, harmonics=8, oversample=16, samples=None))]
    fn hb_hessian(
        &self,
        py: Python<'_>,
        out_idx: usize,
        k_metric: usize,
        subset: Vec<String>,
        spectra: Vec<Vec<(f64, f64)>>,
        p: Vec<f64>,
        f0: f64,
        harmonics: usize,
        oversample: usize,
        samples: Option<usize>,
    ) -> PyResult<Vec<Vec<(f64, f64)>>> {
        py.allow_threads(|| {
            self.inner.hb_hessian(
                out_idx, k_metric, subset, spectra, p, f0, harmonics, oversample, samples,
            )
        })
        .map_err(model_err)
    }

    /// Small-signal poles at the operating point `x` (parameters `p`): the finite
    /// generalized eigenvalues of the pencil `(G, C)` with `G = dF/dx`,
    /// `C = dQ/dx`, computed natively (the engine's standard-reduction
    /// eigensolver -- no Python-side linear algebra, so no drift). `(re, im)` in
    /// rad/s.
    fn poles(&self, py: Python<'_>, x: Vec<f64>, p: Vec<f64>) -> PyResult<Vec<(f64, f64)>> {
        py.allow_threads(|| self.inner.poles(x, p))
            .map_err(model_err)
    }

    /// Transmission zeros from source `input` to the unknown at `out_idx`, at the
    /// operating point `x`: the finite generalized eigenvalues of the Rosenbrock
    /// system-matrix pencil, computed natively (same eigensolver as `poles`).
    fn zeros(
        &self,
        py: Python<'_>,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> PyResult<Vec<(f64, f64)>> {
        py.allow_threads(|| self.inner.zeros(input, out_idx, x, p))
            .map_err(model_err)
    }

    /// Small-signal AC response `H(j2*pi*f)` from source `input` to the unknown at
    /// `out_idx`, at the operating point `x`, solved numerically per frequency in
    /// Rust: `e_out^T (G + jwC)^{-1} (-dF/d(input))`. Returns `(re, im)` per
    /// frequency.
    fn ac_response(
        &self,
        py: Python<'_>,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freqs_hz: Vec<f64>,
    ) -> PyResult<Vec<(f64, f64)>> {
        py.allow_threads(|| self.inner.ac_response(input, out_idx, x, p, freqs_hz))
            .map_err(model_err)
    }

    /// Exact AC-transfer sensitivity `dH/dp(jw)` from `input` w.r.t. `param` at
    /// the operating point `x`, computed natively (the complex solves run in
    /// Rust). Returns `(re, im)` per frequency.
    fn ac_sensitivity(
        &self,
        py: Python<'_>,
        input: &str,
        param: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freqs_hz: Vec<f64>,
    ) -> PyResult<Vec<(f64, f64)>> {
        py.allow_threads(|| {
            self.inner
                .ac_sensitivity(input, param, out_idx, x, p, freqs_hz)
        })
        .map_err(model_err)
    }

    /// Exact AC sensitivity `dH/dp` of the transfer `H(jw) = v[out_idx]` w.r.t.
    /// **every** parameter at one frequency, by the adjoint. With the forward
    /// state `A v = b` and adjoint `A^T lambda = e_out` (`A = G + jwC`), the
    /// scalar functional `Psi(x,p) = lambda^T b - lambda^T A v` (lambda, v frozen)
    /// has total derivative `dH/dp_k = dPsi/dp_k - mu^T dF/dp_k`, where
    /// `G_dc^T mu = grad_x Psi` is one DC adjoint (the operating-point shift). All
    /// pieces are exact first-order autodiff of one scalar -- no finite
    /// differences, two complex solves plus one real solve total, all parameters
    /// at once. Returns `(name, dHre/dp, dHim/dp)`.
    fn ac_gradient(
        &self,
        py: Python<'_>,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freq: f64,
    ) -> PyResult<Vec<(String, f64, f64)>> {
        py.allow_threads(|| self.inner.ac_gradient(input, out_idx, x, p, freq))
            .map_err(model_err)
    }

    /// S-parameter sweep at an explicit operating point: `ports` are
    /// `(drive source, out_idx, z0)` in port order; returns per frequency the
    /// flattened row-major n x n scattering matrix as (re, im) pairs.
    fn sp_response(
        &self,
        py: Python<'_>,
        ports: Vec<(String, usize, f64)>,
        x: Vec<f64>,
        p: Vec<f64>,
        freqs: Vec<f64>,
    ) -> PyResult<Vec<Vec<(f64, f64)>>> {
        py.allow_threads(|| self.inner.sp_response(&ports, x, p, freqs))
            .map_err(model_err)
    }

    /// Exact VJP of `sp_response`: complex cotangent (dL/dRe, dL/dIm) per
    /// flattened S entry and frequency, pulled back to (param, dL/dp) pairs.
    fn sp_vjp(
        &self,
        py: Python<'_>,
        ports: Vec<(String, usize, f64)>,
        x: Vec<f64>,
        p: Vec<f64>,
        freqs: Vec<f64>,
        cot: Vec<Vec<(f64, f64)>>,
    ) -> PyResult<Vec<(String, f64)>> {
        py.allow_threads(|| self.inner.sp_vjp(&ports, x, p, freqs, &cot))
            .map_err(model_err)
    }

    /// Exact analytic AC Hessian via the **second-order adjoint** -- no finite
    /// differences. The trick: AC analysis at one frequency is itself an algebraic
    /// system, so build the combined system `[F(x,0,p)=0; Re(Av-b)=0; Im(Av-b)=0]`
    /// in the unknowns `(x, v_re, v_im)`, then run the *existing* DC second-order
    /// adjoint (`hessian`) on it. The operating-point shift falls out for free
    /// because `x` is part of the combined solution. Returns, for the output node,
    /// `(v_re, v_im, grad_re[subset], grad_im[subset], H_re[subset^2], H_im[subset^2])`;
    /// the caller projects these onto the desired metric (mag/phase/real/imag).
    #[allow(clippy::type_complexity)]
    fn ac_hessian(
        &self,
        py: Python<'_>,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freq: f64,
        subset: Vec<String>,
    ) -> PyResult<(f64, f64, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        py.allow_threads(|| self.inner.ac_hessian(input, out_idx, x, p, freq, subset))
            .map_err(model_err)
    }

    /// Exact analytic pole sensitivity `ds/dp` for every finite pole w.r.t.
    /// **every** parameter at once, including the operating-point shift. For a
    /// simple pole `s=-1/mu` (eigenvalue `mu` of `A=G^{-1}C`, right/left
    /// eigenvectors `v`/`w`), the perturbation `d(mu)/dp_k = w_hat^T(dC/dp_k -
    /// mu dG/dp_k)v/(w^T v)` (`w_hat = G^{-T} w`) collapses over all `k` via the
    /// scalar functional `Phi = w_hat^T(C - mu G)v`: explicit `dPhi/dp` plus one
    /// DC adjoint for the shift (the same all-parameter trick as the AC gradient),
    /// then `ds = d(mu)/mu^2`. Returns one `(s, [(name, ds_re, ds_im)])` per pole.
    #[allow(clippy::type_complexity)]
    fn pole_gradient(
        &self,
        py: Python<'_>,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> PyResult<Vec<((f64, f64), Vec<(String, f64, f64)>)>> {
        py.allow_threads(|| self.inner.pole_gradient(x, p))
            .map_err(model_err)
    }

    /// Exact analytic transmission-zero sensitivity `ds/dp` for every finite zero
    /// w.r.t. **every** parameter at once, including the operating-point shift.
    /// Same eigenvalue-perturbation + adjoint-shift machinery as
    /// [`pole_gradient`], but on the Rosenbrock pencil
    /// `M=[[G,b],[e_out^T,0]]`, `N=[[C,0],[0,0]]` (`b=dF/d(input)`); the functional
    /// is `Phi = w_hat^T(N - mu M)v` over the augmented system, whose only extra
    /// term is the `b`-column `-mu v_{n} sum_i w_hat_i b_i`. Returns one
    /// `(s, [(name, ds_re, ds_im)])` per zero.
    #[allow(clippy::type_complexity)]
    fn zero_gradient(
        &self,
        py: Python<'_>,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> PyResult<Vec<((f64, f64), Vec<(String, f64, f64)>)>> {
        py.allow_threads(|| self.inner.zero_gradient(input, out_idx, x, p))
            .map_err(model_err)
    }

    /// Exact analytic output-noise sensitivity `dN/dp` at one frequency w.r.t.
    /// **every** parameter (op-point shift included). The output noise PSD is
    /// `N = sum_q S_q |T_q|^2` with transimpedance `T_q = lambda . u_q`,
    /// `lambda = A^{-T} e_out`, injection `u_q` (+/-1 at the source's nodes), and
    /// PSD `S_q`. With `r = sum_q 2 conj(T_q) S_q u_q` and `xi = A^{-1} r`, `dN/dp`
    /// is the total derivative of the real functional
    /// `Phi = Re(-lambda^T A xi) + sum_q |T_q|^2 fac_q psd_q`: explicit `dPhi/dp`
    /// plus one DC adjoint for the bias shift (same trick as the AC gradient; no
    /// finite differences). White/flicker sources only. Returns `(N, [(name, dN/dp)])`.
    fn noise_gradient(
        &self,
        py: Python<'_>,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freq: f64,
    ) -> PyResult<(f64, Vec<(String, f64)>)> {
        py.allow_threads(|| self.inner.noise_gradient(out_idx, x, p, freq))
            .map_err(model_err)
    }

    /// Exact pole sensitivity `(pole, dpole/dp)` for every finite pole w.r.t.
    /// `param`, including the operating-point shift, computed natively (the same
    /// pencil eigensolver as `poles`, so the poles are consistent). `(re, im)`.
    fn pole_sensitivity(
        &self,
        py: Python<'_>,
        input: &str,
        param: &str,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> PyResult<Vec<((f64, f64), (f64, f64))>> {
        py.allow_threads(|| self.inner.pole_sensitivity(input, param, x, p))
            .map_err(model_err)
    }

    /// Exact transmission-zero sensitivity `(zero, dzero/dp)` for every finite
    /// zero of the `input -> out_idx` transfer w.r.t. `param`, native (same
    /// Rosenbrock pencil and eigensolver as `zeros`). `(re, im)`.
    fn zero_sensitivity(
        &self,
        py: Python<'_>,
        input: &str,
        out_idx: usize,
        param: &str,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> PyResult<Vec<((f64, f64), (f64, f64))>> {
        py.allow_threads(|| self.inner.zero_sensitivity(input, out_idx, param, x, p))
            .map_err(model_err)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // No thread setting here: the linear solves are sequential by construction
    // (rslab's KLU) and the parallelism lives in
    // the outer sweeps, on the worker pool with its own default. `1` used to
    // mean "faer sequential" and would now shrink that pool to one thread.
    m.add_class::<Circuit>()?;
    m.add_class::<PyModel>()?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(value_symbol_name, m)?)?;
    m.add_function(wrap_pyfunction!(set_parallelism, m)?)?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(drain_warnings, m)?)?;
    m.add_function(wrap_pyfunction!(vectfit_verilog_a, m)?)?;
    m.add_function(wrap_pyfunction!(profile_begin, m)?)?;
    m.add_function(wrap_pyfunction!(profile_take, m)?)?;
    Ok(())
}
