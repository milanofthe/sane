//! The model and the point: Python's handles on `sane_analysis::Model` and
//! `sane_analysis::Point`, every analysis a method of the point.

use std::collections::HashMap;

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayMethods};
use pyo3::exceptions::{PyAttributeError, PyKeyError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use sane_analysis::{DcOptions, HbOptions, TransientOptions};

use crate::arrays::{floats, vec1, Floats, Names};
use crate::results::*;
use crate::{err, run};

/// A dense matrix handed to numpy.
fn matrix(py: Python<'_>, rows: Vec<Vec<f64>>) -> Bound<'_, PyArray2<f64>> {
    let (r, c) = (rows.len(), rows.first().map_or(0, Vec::len));
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    vec1(py, flat)
        .reshape([r, c])
        .expect("rows of equal length")
}

/// A sparse matrix `(rows, cols, values)` handed to numpy.
type Coo<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);

fn coo(py: Python<'_>, (r, c, v): (Vec<usize>, Vec<usize>, Vec<f64>)) -> Coo<'_> {
    (vec1(py, r), vec1(py, c), vec1(py, v))
}

/// A binding: a dict of name -> value and keyword values, merged.
fn binding(
    values: Option<&Bound<'_, PyDict>>,
    kw: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(String, f64)>> {
    let mut out = Vec::new();
    for d in [values, kw].into_iter().flatten() {
        for (k, v) in d.iter() {
            out.push((k.extract::<String>()?, v.extract::<f64>()?));
        }
    }
    Ok(out)
}

fn pairs(b: &[(String, f64)]) -> Vec<(&str, f64)> {
    b.iter().map(|(k, v)| (k.as_str(), *v)).collect()
}

/// A circuit as an analyzable model: its unknowns, its parameters and the
/// values bound to them. `at(...)` takes it to a binding of the parameters,
/// the point every analysis runs at.
///
/// >>> m = sane.Model.from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k")
/// >>> m.at(R2=3e3).operating_point()["out"]
/// 3.75
#[pyclass(name = "Model", module = "sane", frozen)]
pub struct PyModel {
    pub inner: sane_analysis::Model,
}

#[pymethods]
impl PyModel {
    /// The model of a circuit (``sane.Circuit``): it keeps the circuit, to
    /// set it up anew where a binding crosses a device's structure, and
    /// hands it back (``circuit``) to build on.
    #[new]
    fn new(py: Python<'_>, circuit: PyRef<'_, crate::circuit::PyCircuit>) -> PyResult<PyModel> {
        PyModel::of(py, &circuit)
    }

    /// The circuit the model was set up from; ``None`` for a model a
    /// transform derived.
    #[getter]
    fn circuit(&self) -> Option<crate::circuit::PyCircuit> {
        (self.inner.circuit()).map(|c| crate::circuit::PyCircuit { inner: c.clone() })
    }

    /// The model of a SPICE-like netlist.
    #[staticmethod]
    fn from_netlist(py: Python<'_>, netlist: &str) -> PyResult<PyModel> {
        run(py, || sane_analysis::Model::from_netlist(netlist)).map(|inner| PyModel { inner })
    }

    /// The parameters, in the order of a parameter vector.
    #[getter]
    fn params(&self) -> Vec<String> {
        self.inner.params().to_vec()
    }
    /// The unknowns, in the order of a state.
    #[getter]
    fn unknowns(&self) -> Vec<String> {
        self.inner.unknowns().to_vec()
    }
    /// The node names by node id (`0` ground).
    #[getter]
    fn node_names(&self) -> Vec<String> {
        self.inner.node_names().to_vec()
    }
    /// The ports (deck `P` elements): `(name, node, z0)` each.
    #[getter]
    fn ports(&self) -> Vec<(String, String, f64)> {
        self.inner.ports().to_vec()
    }
    /// The number of unknowns.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    /// The index of a node, unknown or branch current in a state.
    fn resolve(&self, reference: &str) -> Option<usize> {
        self.inner.resolve(reference)
    }

    // --- the bound values --------------------------------------------------

    /// Every parameter's bound value.
    fn values(&self) -> HashMap<String, f64> {
        self.inner.values()
    }
    /// A parameter's bound value.
    fn get(&self, name: &str) -> PyResult<f64> {
        self.inner
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }
    /// Binds `value` to the parameter `name`.
    fn set(&self, name: &str, value: f64) -> PyResult<()> {
        self.inner.set(name, value).map_err(err)
    }
    /// Binds every value of `values` and the keywords (all or none).
    #[pyo3(signature = (values=None, **kw))]
    fn update(
        &self,
        values: Option<&Bound<'_, PyDict>>,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let b = binding(values, kw)?;
        if let Some((k, _)) = b.iter().find(|(k, _)| !self.inner.is_param(k)) {
            return Err(PyKeyError::new_err(format!("'{k}' is not a parameter")));
        }
        self.inner.set_many(&pairs(&b)).map_err(err)
    }
    /// Restores the values bound at construction.
    fn reset(&self) {
        self.inner.reset()
    }
    fn is_param(&self, name: &str) -> bool {
        self.inner.is_param(name)
    }
    fn is_group(&self, name: &str) -> bool {
        self.inner.is_group(name)
    }
    /// The children of a parameter group (`""` for the top level).
    #[pyo3(signature = (prefix=""))]
    fn children(&self, prefix: &str) -> Vec<String> {
        self.inner.children(prefix)
    }
    /// Parameters the equations read but no value binds (a numeric
    /// analysis takes them as zero).
    fn unbound_params(&self) -> Vec<String> {
        self.inner.unbound_params()
    }
    fn __getitem__(&self, name: &str) -> PyResult<f64> {
        self.get(name)
    }
    fn __setitem__(&self, name: &str, value: f64) -> PyResult<()> {
        self.set(name, value)
    }
    fn __contains__(&self, name: &str) -> bool {
        self.inner.is_param(name)
    }
    fn __len__(&self) -> usize {
        self.inner.params().len()
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let names = pyo3::types::PyList::new_bound(py, self.inner.params());
        Ok(names.call_method0("__iter__")?.unbind())
    }
    /// The top-level parameters and groups, with the methods.
    fn __dir__(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
        let mut out: Vec<String> = slf.get_type().dir()?.extract()?;
        out.extend(slf.get().inner.children(""));
        Ok(out)
    }
    /// `model.R1` reads a parameter, `model.X1` is a group of them.
    fn __getattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<PyObject> {
        let py = slf.py();
        let m = &slf.get().inner;
        if let Some(v) = m.get(name) {
            return Ok(v.into_py(py));
        }
        if m.is_group(name) {
            let g = ParamGroup {
                model: m.clone(),
                prefix: name.to_string(),
            };
            return Ok(g.into_py(py));
        }
        Err(PyAttributeError::new_err(name.to_string()))
    }
    /// `model.R1 = value` binds a parameter.
    fn __setattr__(&self, name: &str, value: f64) -> PyResult<()> {
        if !self.inner.is_param(name) {
            return Err(PyAttributeError::new_err(format!(
                "'{name}' is not a parameter"
            )));
        }
        self.set(name, value)
    }

    /// How operating points are solved: `reltol`, `abstol` (A), `vntol` (V),
    /// `max_iter` (the deck's `.option`s over the defaults, unless set).
    #[getter]
    fn dc_options(&self) -> HashMap<String, f64> {
        dc_dict(&self.inner.dc_options())
    }
    /// Solve the operating points of the points taken from now on with
    /// these settings (the others keep theirs).
    #[pyo3(signature = (*, reltol=None, abstol=None, vntol=None, max_iter=None))]
    fn set_dc_options(
        &self,
        reltol: Option<f64>,
        abstol: Option<f64>,
        vntol: Option<f64>,
        max_iter: Option<usize>,
    ) {
        let o = self.inner.dc_options();
        self.inner.set_dc_options(DcOptions {
            reltol: reltol.unwrap_or(o.reltol),
            abstol: abstol.unwrap_or(o.abstol),
            vntol: vntol.unwrap_or(o.vntol),
            max_iter: max_iter.unwrap_or(o.max_iter),
        });
    }

    // --- the point ---------------------------------------------------------

    /// The model at the binding `values` (and keywords) over the bound
    /// values: the point every analysis runs at.
    #[pyo3(signature = (values=None, **kw))]
    fn at(
        &self,
        py: Python<'_>,
        values: Option<&Bound<'_, PyDict>>,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Point> {
        let b = binding(values, kw)?;
        run(py, || self.inner.at(&pairs(&b))).map(|inner| Point { inner })
    }

    // --- transforms --------------------------------------------------------

    /// The model with the parameters under `paths` folded to their values.
    #[pyo3(signature = (*paths))]
    fn fold(&self, py: Python<'_>, paths: &Bound<'_, PyTuple>) -> PyResult<PyModel> {
        let p = flat_paths(paths)?;
        let r: Vec<&str> = p.iter().map(|s| s.as_str()).collect();
        run(py, || self.inner.fold(&r)).map(|inner| PyModel { inner })
    }
    /// The model with every parameter folded but those under `paths`.
    #[pyo3(signature = (*paths))]
    fn keep(&self, py: Python<'_>, paths: &Bound<'_, PyTuple>) -> PyResult<PyModel> {
        let p = flat_paths(paths)?;
        let r: Vec<&str> = p.iter().map(|s| s.as_str()).collect();
        run(py, || self.inner.keep(&r)).map(|inner| PyModel { inner })
    }
    /// The small-signal model about the operating point.
    fn linearize(&self, py: Python<'_>) -> PyModel {
        PyModel {
            inner: py.allow_threads(|| self.inner.linearize()),
        }
    }
    /// The model with its internal resistive nodes eliminated exactly, but
    /// the node unknowns under `keep`; and the eliminated ones.
    #[pyo3(signature = (keep=None))]
    fn eliminate(&self, py: Python<'_>, keep: Option<Names>) -> (PyModel, Vec<String>) {
        let keep = keep.map_or_else(Vec::new, |k| k.0);
        let (inner, gone) = py.allow_threads(|| self.inner.eliminate_nodes(&keep));
        (PyModel { inner }, gone)
    }

    // --- evaluations at a given state ----------------------------------------

    /// The parameter vector of the bound values and `values` (and keywords).
    #[pyo3(signature = (values=None, **kw))]
    fn param_vector<'py>(
        &self,
        py: Python<'py>,
        values: Option<&Bound<'_, PyDict>>,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let b = binding(values, kw)?;
        Ok(vec1(py, self.inner.pvec(&pairs(&b))))
    }
    /// `I(x, t)` at the parameter vector `p`.
    #[pyo3(signature = (x, p, t=0.0))]
    fn currents<'py>(
        &self,
        py: Python<'py>,
        x: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self
            .inner
            .currents(floats(&x), floats(&p), t)
            .map_err(err)?;
        Ok(vec1(py, v))
    }
    /// `Q(x, t)` at the parameter vector `p`.
    #[pyo3(signature = (x, p, t=0.0))]
    fn charges<'py>(
        &self,
        py: Python<'py>,
        x: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self.inner.charges(floats(&x), floats(&p), t).map_err(err)?;
        Ok(vec1(py, v))
    }
    /// `I(x, t) + dQ/dx xdot` at the parameter vector `p`.
    #[pyo3(signature = (x, xdot, p, t=0.0))]
    fn residual<'py>(
        &self,
        py: Python<'py>,
        x: Floats<'_>,
        xdot: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = (self.inner)
            .residual(floats(&x), floats(&xdot), floats(&p), t)
            .map_err(err)?;
        Ok(vec1(py, v))
    }
    /// `dI/dx`, dense.
    #[pyo3(signature = (x, p, t=0.0))]
    fn jacobian_i_x<'py>(
        &self,
        py: Python<'py>,
        x: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self
            .inner
            .jacobian_i_x(floats(&x), floats(&p), t)
            .map_err(err)?;
        Ok(matrix(py, m))
    }
    /// `dQ/dx`, dense.
    #[pyo3(signature = (x, p, t=0.0))]
    fn jacobian_q_x<'py>(
        &self,
        py: Python<'py>,
        x: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self
            .inner
            .jacobian_q_x(floats(&x), floats(&p), t)
            .map_err(err)?;
        Ok(matrix(py, m))
    }
    /// `dI/dx` as `(rows, cols, values)`.
    #[pyo3(signature = (x, p, t=0.0))]
    fn jacobian_i_x_sparse<'py>(
        &self,
        py: Python<'py>,
        x: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Coo<'py>> {
        let m = self
            .inner
            .jacobian_i_x_sparse(floats(&x), floats(&p), t)
            .map_err(err)?;
        Ok(coo(py, m))
    }
    /// `dQ/dx` as `(rows, cols, values)`.
    #[pyo3(signature = (x, p, t=0.0))]
    fn jacobian_q_x_sparse<'py>(
        &self,
        py: Python<'py>,
        x: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Coo<'py>> {
        let m = self
            .inner
            .jacobian_q_x_sparse(floats(&x), floats(&p), t)
            .map_err(err)?;
        Ok(coo(py, m))
    }
    /// `dI/dp` as `(rows, cols, values)`.
    #[pyo3(signature = (x, p, t=0.0))]
    fn jacobian_i_p_sparse<'py>(
        &self,
        py: Python<'py>,
        x: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Coo<'py>> {
        let m = self
            .inner
            .jacobian_i_p_sparse(floats(&x), floats(&p), t)
            .map_err(err)?;
        Ok(coo(py, m))
    }
    /// `dQ/dp` as `(rows, cols, values)`.
    #[pyo3(signature = (x, p, t=0.0))]
    fn jacobian_q_p_sparse<'py>(
        &self,
        py: Python<'py>,
        x: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Coo<'py>> {
        let m = self
            .inner
            .jacobian_q_p_sparse(floats(&x), floats(&p), t)
            .map_err(err)?;
        Ok(coo(py, m))
    }
    /// `dI/d input` for a source parameter `input`.
    #[pyo3(signature = (input, x, p, t=0.0))]
    fn jacobian_i_input<'py>(
        &self,
        py: Python<'py>,
        input: &str,
        x: Floats<'_>,
        p: Floats<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = (self.inner)
            .jacobian_i_input(input, floats(&x), floats(&p), t)
            .map_err(err)?;
        Ok(vec1(py, v))
    }
    /// The nonzeros of the system Jacobian.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }
    /// The sizes of the system's linear and nonlinear partitions.
    #[getter]
    fn partition_sizes(&self) -> Option<(usize, usize)> {
        self.inner.partition_sizes()
    }
    fn __repr__(&self) -> String {
        format!(
            "<Model of {} unknowns, {} parameters>",
            self.inner.dim(),
            self.inner.params().len()
        )
    }
}

fn dc_dict(o: &DcOptions) -> HashMap<String, f64> {
    HashMap::from([
        ("reltol".to_string(), o.reltol),
        ("abstol".to_string(), o.abstol),
        ("vntol".to_string(), o.vntol),
        ("max_iter".to_string(), o.max_iter as f64),
    ])
}

fn flat_paths(paths: &Bound<'_, PyTuple>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for p in paths.iter() {
        out.extend(p.extract::<Names>()?.0);
    }
    Ok(out)
}

/// A group of a model's parameters (a subcircuit instance, a device):
/// `group.R2` reads a member, `group.R2 = v` binds it.
#[pyclass(module = "sane", frozen)]
pub struct ParamGroup {
    model: sane_analysis::Model,
    prefix: String,
}

#[pymethods]
impl ParamGroup {
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let full = format!("{}.{name}", self.prefix);
        if let Some(v) = self.model.get(&full) {
            return Ok(v.into_py(py));
        }
        if self.model.is_group(&full) {
            let g = ParamGroup {
                model: self.model.clone(),
                prefix: full,
            };
            return Ok(g.into_py(py));
        }
        Err(PyAttributeError::new_err(full))
    }
    fn __getitem__(&self, name: &str) -> PyResult<f64> {
        let full = format!("{}.{name}", self.prefix);
        self.model
            .get(&full)
            .ok_or_else(|| PyKeyError::new_err(full))
    }
    fn __setitem__(&self, name: &str, value: f64) -> PyResult<()> {
        let full = format!("{}.{name}", self.prefix);
        if !self.model.is_param(&full) {
            return Err(PyKeyError::new_err(full));
        }
        self.model.set(&full, value).map_err(err)
    }
    fn __setattr__(&self, name: &str, value: f64) -> PyResult<()> {
        let full = format!("{}.{name}", self.prefix);
        if !self.model.is_param(&full) {
            return Err(PyAttributeError::new_err(format!(
                "'{full}' is not a parameter"
            )));
        }
        self.model.set(&full, value).map_err(err)
    }
    fn __dir__(&self) -> Vec<String> {
        self.model.children(&self.prefix)
    }
    fn __repr__(&self) -> String {
        format!(
            "<ParamGroup {:?}: {:?}>",
            self.prefix,
            self.model.children(&self.prefix)
        )
    }
}

/// A model at one binding of its parameters: every analysis runs at it, and
/// its operating point is solved once and kept.
#[pyclass(module = "sane", frozen)]
pub struct Point {
    pub inner: sane_analysis::Point,
}

#[pymethods]
impl Point {
    /// The model this point is of.
    #[getter]
    fn model(&self) -> PyModel {
        PyModel {
            inner: self.inner.model().clone(),
        }
    }
    /// The binding, every parameter by name.
    fn values(&self) -> HashMap<String, f64> {
        self.inner.values()
    }
    /// How its operating point is solved.
    #[getter]
    fn dc_options(&self) -> HashMap<String, f64> {
        dc_dict(&self.inner.dc_options())
    }
    /// This point, its operating point solved from `other`'s rather than
    /// cold (cold where that does not converge, or where the two are of
    /// different structures): for runs of nearby bindings.
    fn near(&self, py: Python<'_>, other: &Point) -> PyResult<Point> {
        run(py, || self.inner.near(&other.inner)).map(|inner| Point { inner })
    }
    /// The DC operating point.
    fn operating_point(&self, py: Python<'_>) -> PyResult<OperatingPoint> {
        run(py, || self.inner.operating_point()).map(OperatingPoint)
    }
    /// The operating point at each of `values` of the parameter `param`.
    fn dc_sweep(&self, py: Python<'_>, param: &str, values: Floats<'_>) -> PyResult<DcSweep> {
        let v = floats(&values);
        run(py, || self.inner.dc_sweep(param, &v)).map(DcSweep)
    }
    /// The transient response at the times `t` (the first the start), by
    /// adaptive Rodas4 to the tolerances `rtol`, `atol`, its steps capped at
    /// `dt_max`; `x0` a state to start from instead of the operating point.
    #[pyo3(signature = (t, *, rtol=1e-4, atol=1e-7, dt_max=None, x0=None))]
    fn transient(
        &self,
        py: Python<'_>,
        t: Floats<'_>,
        rtol: f64,
        atol: f64,
        dt_max: Option<f64>,
        x0: Option<PyArrayLike1<'_, f64, AllowTypeChange>>,
    ) -> PyResult<Trajectory> {
        let opts = TransientOptions {
            rtol,
            atol,
            dt_max,
            x0: x0.map(|x| x.as_array().to_vec()),
        };
        let t = floats(&t);
        run(py, || self.inner.transient(&t, &opts)).map(Trajectory)
    }
    /// The small-signal responses of `outputs` to the source `input` at
    /// `freqs` (Hz).
    fn ac(
        &self,
        py: Python<'_>,
        input: &str,
        outputs: Names,
        freqs: Floats<'_>,
    ) -> PyResult<AcResponse> {
        let f = floats(&freqs);
        run(py, || self.inner.ac(input, &outputs.refs(), &f)).map(AcResponse)
    }
    /// The scattering parameters over the circuit's ports at `freqs` (Hz).
    fn s_parameters(&self, py: Python<'_>, freqs: Floats<'_>) -> PyResult<SParameters> {
        let f = floats(&freqs);
        run(py, || self.inner.s_parameters(&f)).map(SParameters)
    }
    /// The output noise of `outputs` at `freqs` (Hz).
    fn noise(&self, py: Python<'_>, outputs: Names, freqs: Floats<'_>) -> PyResult<NoiseSpectrum> {
        let f = floats(&freqs);
        run(py, || self.inner.noise(&outputs.refs(), &f)).map(NoiseSpectrum)
    }
    /// The finite poles of the small-signal system.
    fn poles(&self, py: Python<'_>) -> PyResult<Poles> {
        run(py, || self.inner.poles()).map(Poles)
    }
    /// The finite transmission zeros from `input` to each of `outputs`.
    fn zeros(&self, py: Python<'_>, input: &str, outputs: Names) -> PyResult<Zeros> {
        run(py, || self.inner.zeros(input, &outputs.refs())).map(Zeros)
    }
    /// The descriptor state space from the sources `inputs` to `outputs`.
    fn state_space(&self, py: Python<'_>, inputs: Names, outputs: Names) -> PyResult<StateSpace> {
        run(py, || {
            self.inner.state_space(&inputs.refs(), &outputs.refs())
        })
        .map(StateSpace)
    }
    /// The `order` most dominant poles of the transfer from `input` to
    /// `output`, against the full transfer at `freqs` (Hz).
    fn reduce(
        &self,
        py: Python<'_>,
        input: &str,
        output: &str,
        order: usize,
        freqs: Floats<'_>,
    ) -> PyResult<ReducedModel> {
        let f = floats(&freqs);
        run(py, || self.inner.reduce(input, output, order, &f)).map(ReducedModel)
    }
    /// The model reduced at the operating point: branches negligible at DC
    /// and at `freqs` (Hz; 1 Hz to 1 GHz for `None`) opened, dominating ones
    /// shorted; and what was done, `(element, "open" | "short")` each.
    #[pyo3(signature = (rel_tol=1e-3, freqs=None))]
    fn prune(
        &self,
        py: Python<'_>,
        rel_tol: f64,
        freqs: Option<Floats<'_>>,
    ) -> PyResult<(PyModel, Vec<(String, String)>)> {
        let f = freqs.map_or_else(|| sane_analysis::log_grid(1.0, 1e9, 6), |f| floats(&f));
        run(py, || self.inner.prune(rel_tol, &f)).map(|(inner, gone)| (PyModel { inner }, gone))
    }
    /// The periodic steady state under the circuit's periodic drive.
    /// `f0` the fundamental (Hz; the periodic source's for `None`),
    /// `continuation` source stepping (`None`: where the direct Newton
    /// fails).
    #[pyo3(signature = (*, f0=None, harmonics=8, oversample=16, samples=None, tol=1e-10, max_iter=60, continuation=None))]
    #[allow(clippy::too_many_arguments)]
    fn harmonic_balance(
        &self,
        py: Python<'_>,
        f0: Option<f64>,
        harmonics: usize,
        oversample: usize,
        samples: Option<usize>,
        tol: f64,
        max_iter: usize,
        continuation: Option<bool>,
    ) -> PyResult<HarmonicBalance> {
        let opts = HbOptions {
            f0,
            harmonics,
            oversample,
            samples,
            tol,
            max_iter,
            continuation,
        };
        run(py, || self.inner.harmonic_balance(&opts)).map(HarmonicBalance)
    }
    fn __repr__(&self) -> String {
        "<Point>".to_string()
    }
}
