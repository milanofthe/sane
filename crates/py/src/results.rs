//! The results of the analyses at a point, as Python sees them: the Rust
//! results themselves, their arrays numpy views of their own memory.

use std::collections::HashMap;

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayLike2, PyArrayLike3};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;

use crate::arrays::{vec1, view, view1, Names, C64};
use crate::run;

fn missing(name: &str) -> PyErr {
    PyKeyError::new_err(name.to_string())
}

fn index(names: &[String], name: &str) -> PyResult<usize> {
    names
        .iter()
        .position(|n| n == name)
        .ok_or_else(|| missing(name))
}

// --- operating point -------------------------------------------------------

/// How the gmin regularization holds an operating point: converged, but
/// depending on the shunt rather than on the circuit alone.
#[pyclass(module = "sane", frozen)]
pub struct Regularization(pub sane_analysis::Regularization);

#[pymethods]
impl Regularization {
    /// The shunt to ground (S) the point held at.
    #[getter]
    fn gmin(&self) -> f64 {
        self.0.gmin
    }
    /// The unknown the shunt sets, and the relative shift removing it would
    /// cause; `None` where the solve did not reach the floor.
    #[getter]
    fn dominant(&self) -> Option<(String, f64)> {
        self.0.dominant.clone()
    }
    fn __str__(&self) -> String {
        self.0.to_string()
    }
    fn __repr__(&self) -> String {
        format!("<Regularization gmin={:.1e}>", self.0.gmin)
    }
}

/// The DC operating point at a binding.
#[pyclass(module = "sane", frozen)]
pub struct OperatingPoint(pub sane_analysis::OperatingPoint);

#[pymethods]
impl OperatingPoint {
    /// The state, over the model's unknowns.
    #[getter]
    fn x<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), slf.get().0.vector())
    }
    /// How the gmin regularization holds the point, or `None` for a true DC
    /// solution.
    #[getter]
    fn regularization(&self) -> Option<Regularization> {
        self.0.regularization.clone().map(Regularization)
    }
    /// The value of a node, unknown or branch current.
    fn __getitem__(&self, reference: &str) -> PyResult<f64> {
        self.0.get(reference).ok_or_else(|| missing(reference))
    }
    /// The value of a node, unknown or branch current, or `default`.
    #[pyo3(signature = (reference, default=None))]
    fn get(&self, reference: &str, default: Option<f64>) -> Option<f64> {
        self.0.get(reference).or(default)
    }
    /// Every unknown's value by name.
    fn to_dict(&self) -> HashMap<String, f64> {
        self.0.to_map()
    }
    /// The operating-point variables the devices export (gm, vth, ...):
    /// `(name, value, units, description)` each.
    fn op_vars(&self) -> Vec<(String, f64, Option<String>, String)> {
        (self.0.op_vars().into_iter())
            .map(|v| (v.name, v.value, v.units, v.desc))
            .collect()
    }
    /// First-order sensitivities of `outputs` by the parameters under `wrt`
    /// (all for `None`).
    #[pyo3(signature = (outputs, wrt=None))]
    fn sensitivity(
        &self,
        py: Python<'_>,
        outputs: Names,
        wrt: Option<Names>,
    ) -> PyResult<Sensitivity> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.sensitivity(&outputs.refs(), &w)).map(Sensitivity)
    }
    /// Second-order sensitivities (Hessians) of `outputs` by the parameters
    /// under `wrt` (all for `None`).
    #[pyo3(signature = (outputs, wrt=None))]
    fn hessian(&self, py: Python<'_>, outputs: Names, wrt: Option<Names>) -> PyResult<Hessian> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.hessian(&outputs.refs(), &w)).map(Hessian)
    }
    fn __repr__(&self) -> String {
        format!("<OperatingPoint of {} unknowns>", self.0.vector().len())
    }
}

/// First-order sensitivities: `grad[i, j]` is `d outputs[i] / d params[j]`.
#[pyclass(module = "sane", frozen)]
pub struct Sensitivity(pub sane_analysis::Sensitivity);

#[pymethods]
impl Sensitivity {
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    /// The outputs' values.
    #[getter]
    fn values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.values)
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn grad<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.grad)
    }
    /// The derivatives of `output` by every parameter.
    fn of<'py>(slf: &Bound<'py, Self>, output: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let s = &slf.get().0;
        let i = index(&s.outputs, output)?;
        Ok(view(slf.as_any(), &s.grad.row(i)))
    }
    /// `d output / d param`.
    fn get(&self, output: &str, param: &str) -> PyResult<f64> {
        self.0
            .get(output, param)
            .ok_or_else(|| missing(&format!("{output}/{param}")))
    }
    /// `d ln output / d ln param` (zero where either is zero).
    fn relative<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        numpy::PyArray2::from_owned_array_bound(py, self.0.relative())
    }
    /// The parameters by the magnitude of `output`'s sensitivity to them,
    /// largest first: `(param, sensitivity)` pairs.
    #[pyo3(signature = (output, relative=true))]
    fn ranked(&self, output: &str, relative: bool) -> PyResult<Vec<(String, f64)>> {
        self.0
            .ranked(output, relative)
            .ok_or_else(|| missing(output))
    }
    /// `output`'s sensitivities rolled up to the devices and subcircuits.
    #[pyo3(signature = (output, relative=true))]
    fn rollup(&self, output: &str, relative: bool) -> PyResult<Vec<(String, f64)>> {
        self.0
            .rollup(output, relative)
            .ok_or_else(|| missing(output))
    }
    fn __repr__(&self) -> String {
        format!(
            "<Sensitivity of {:?} by {} parameters>",
            self.0.outputs,
            self.0.params.len()
        )
    }
}

/// Hessians: `h[i, a, b]` is `d^2 outputs[i] / d params[a] d params[b]`.
#[pyclass(module = "sane", frozen)]
pub struct Hessian(pub sane_analysis::Hessian);

#[pymethods]
impl Hessian {
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.values)
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn h<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray3<f64>> {
        view(slf.as_any(), &slf.get().0.h)
    }
}

/// The gradient of a scalar by every parameter.
#[pyclass(module = "sane", frozen)]
pub struct Gradient(pub sane_analysis::Gradient);

#[pymethods]
impl Gradient {
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn grad<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.grad)
    }
    fn __getitem__(&self, param: &str) -> PyResult<f64> {
        Ok(self.0.grad[index(&self.0.params, param)?])
    }
    fn to_dict(&self) -> HashMap<String, f64> {
        (self.0.params.iter().cloned())
            .zip(self.0.grad.iter().copied())
            .collect()
    }
}

// --- DC sweep, transient ---------------------------------------------------

/// A DC sweep: `x[k, i]` is unknown `i` at `values[k]` (`NaN` where that
/// point did not converge).
#[pyclass(module = "sane", frozen)]
pub struct DcSweep(pub sane_analysis::DcSweep);

#[pymethods]
impl DcSweep {
    #[getter]
    fn param(&self) -> String {
        self.0.param.clone()
    }
    #[getter]
    fn values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.values)
    }
    #[getter]
    fn converged<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<bool>> {
        view1(slf.as_any(), &slf.get().0.converged)
    }
    #[getter]
    fn x<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.x)
    }
    /// The swept series of a node, unknown or branch current.
    fn signal<'py>(&self, py: Python<'py>, reference: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(
            py,
            self.0.signal(reference).ok_or_else(|| missing(reference))?,
        ))
    }
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        reference: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.signal(py, reference)
    }
}

/// A transient solution: `x[k, i]` is unknown `i` at `t[k]`, and the
/// switching events crossed.
#[pyclass(module = "sane", frozen)]
pub struct Trajectory(pub sane_analysis::Trajectory);

#[pymethods]
impl Trajectory {
    #[getter]
    fn t<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.t)
    }
    #[getter]
    fn x<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.x)
    }
    /// The switching events crossed: `(name, t, direction)` each.
    #[getter]
    fn events(&self) -> Vec<(String, f64, i8)> {
        (self.0.events.iter())
            .map(|e| (e.name.clone(), e.t, e.direction))
            .collect()
    }
    /// The waveform of a node, unknown or branch current.
    fn signal<'py>(&self, py: Python<'py>, reference: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(
            py,
            self.0.signal(reference).ok_or_else(|| missing(reference))?,
        ))
    }
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        reference: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.signal(py, reference)
    }
    /// Forward sensitivities of `outputs` by the parameters under `wrt` (all
    /// for `None`) over the trajectory's times.
    #[pyo3(signature = (outputs, wrt=None))]
    fn sensitivity(
        &self,
        py: Python<'_>,
        outputs: Names,
        wrt: Option<Names>,
    ) -> PyResult<TrajectorySensitivity> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.sensitivity(&outputs.refs(), &w)).map(TrajectorySensitivity)
    }
    /// The gradient of a scalar `L` of the waveforms of `outputs` by the
    /// parameters under `wrt` (all for `None`), given
    /// `cotangent[i, k] = dL / d outputs[i](t[k])`: the cotangents contracted
    /// with the forward sensitivities.
    #[pyo3(signature = (outputs, cotangent, wrt=None))]
    fn vjp(
        &self,
        py: Python<'_>,
        outputs: Names,
        cotangent: PyArrayLike2<'_, f64, AllowTypeChange>,
        wrt: Option<Names>,
    ) -> PyResult<Gradient> {
        let c = cotangent.as_array();
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.vjp(&outputs.refs(), c, &w)).map(Gradient)
    }
    fn __repr__(&self) -> String {
        format!("<Trajectory of {} times>", self.0.t.len())
    }
}

/// Forward sensitivities over time: `grad[i, k, j]` is
/// `d outputs[i] / d params[j]` at `t[k]`.
#[pyclass(module = "sane", frozen)]
pub struct TrajectorySensitivity(pub sane_analysis::TrajectorySensitivity);

#[pymethods]
impl TrajectorySensitivity {
    #[getter]
    fn t<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.t)
    }
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.values)
    }
    #[getter]
    fn grad<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray3<f64>> {
        view(slf.as_any(), &slf.get().0.grad)
    }
}

// --- small signal ----------------------------------------------------------

/// Small-signal responses: `h[i, k]` is `outputs[i]` per unit `input` at
/// `freqs[k]`.
#[pyclass(module = "sane", frozen)]
pub struct AcResponse(pub sane_analysis::AcResponse);

#[pymethods]
impl AcResponse {
    #[getter]
    fn input(&self) -> String {
        self.0.input.clone()
    }
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn freqs<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.freqs)
    }
    #[getter]
    fn h<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<C64>> {
        view(slf.as_any(), &slf.get().0.h)
    }
    /// The response of `output` at every frequency.
    fn of<'py>(slf: &Bound<'py, Self>, output: &str) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let r = &slf.get().0;
        let i = index(&r.outputs, output)?;
        Ok(view(slf.as_any(), &r.h.row(i)))
    }
    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        output: &str,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        Self::of(slf, output)
    }
    /// The magnitude of `output`'s response (dB).
    fn mag_db<'py>(&self, py: Python<'py>, output: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(
            py,
            self.0.mag_db(output).ok_or_else(|| missing(output))?,
        ))
    }
    /// The phase of `output`'s response (degrees).
    fn phase_deg<'py>(&self, py: Python<'py>, output: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(
            py,
            self.0.phase_deg(output).ok_or_else(|| missing(output))?,
        ))
    }
    /// The derivatives of every response by the parameters under `wrt` (all
    /// for `None`) at every frequency.
    #[pyo3(signature = (wrt=None))]
    fn sensitivity(&self, py: Python<'_>, wrt: Option<Names>) -> PyResult<AcSensitivity> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.sensitivity(&w)).map(AcSensitivity)
    }
    /// The gradient of a real scalar `L` of the responses by every
    /// parameter, given `cotangent[i, k] = dL/dRe h + j dL/dIm h`.
    fn vjp(
        &self,
        py: Python<'_>,
        cotangent: PyArrayLike2<'_, C64, AllowTypeChange>,
    ) -> PyResult<Gradient> {
        let c = cotangent.as_array();
        run(py, || self.0.vjp(c)).map(Gradient)
    }
    /// The Hessians of every response by the parameters under `wrt` (all
    /// for `None`) at every frequency.
    #[pyo3(signature = (wrt=None))]
    fn hessian(&self, py: Python<'_>, wrt: Option<Names>) -> PyResult<AcHessian> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.hessian(&w)).map(AcHessian)
    }
    fn __repr__(&self) -> String {
        format!(
            "<AcResponse {:?} -> {:?} at {} frequencies>",
            self.0.input,
            self.0.outputs,
            self.0.freqs.len()
        )
    }
}

/// `grad[i, k, j]` is `d h[i, k] / d params[j]`.
#[pyclass(module = "sane", frozen)]
pub struct AcSensitivity(pub sane_analysis::AcSensitivity);

#[pymethods]
impl AcSensitivity {
    #[getter]
    fn freqs<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.freqs)
    }
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn h<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<C64>> {
        view(slf.as_any(), &slf.get().0.h)
    }
    #[getter]
    fn grad<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray3<C64>> {
        view(slf.as_any(), &slf.get().0.grad)
    }
}

/// `h[i, k, a, b]` is `d^2 h[i, k] / d params[a] d params[b]`.
#[pyclass(module = "sane", frozen)]
pub struct AcHessian(pub sane_analysis::AcHessian);

#[pymethods]
impl AcHessian {
    #[getter]
    fn freqs<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.freqs)
    }
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn h<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray4<C64>> {
        view(slf.as_any(), &slf.get().0.h)
    }
}

/// Scattering parameters: `s[k, i, j]` at `freqs[k]`, port `j` driven.
#[pyclass(module = "sane", frozen)]
pub struct SParameters(pub sane_analysis::SParameters);

#[pymethods]
impl SParameters {
    #[getter]
    fn port_names(&self) -> Vec<String> {
        self.0.port_names.clone()
    }
    #[getter]
    fn z0<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.z0)
    }
    #[getter]
    fn freqs<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.freqs)
    }
    #[getter]
    fn s<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray3<C64>> {
        view(slf.as_any(), &slf.get().0.s)
    }
    /// The derivatives of every `S_ij` by the parameters under `wrt` (all
    /// for `None`).
    #[pyo3(signature = (wrt=None))]
    fn sensitivity(&self, py: Python<'_>, wrt: Option<Names>) -> PyResult<SpSensitivity> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.sensitivity(&w)).map(SpSensitivity)
    }
    /// The gradient of a real scalar `L` of the S-parameters by every
    /// parameter, given `cotangent[k, i, j] = dL/dRe S + j dL/dIm S`.
    fn vjp(
        &self,
        py: Python<'_>,
        cotangent: PyArrayLike3<'_, C64, AllowTypeChange>,
    ) -> PyResult<Gradient> {
        let c = cotangent.as_array();
        run(py, || self.0.vjp(c)).map(Gradient)
    }
}

/// `grad[k, i, j, q]` is `d S_ij / d params[q]` at `freqs[k]`.
#[pyclass(module = "sane", frozen)]
pub struct SpSensitivity(pub sane_analysis::SpSensitivity);

#[pymethods]
impl SpSensitivity {
    #[getter]
    fn freqs<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.freqs)
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn grad<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray4<C64>> {
        view(slf.as_any(), &slf.get().0.grad)
    }
}

/// Output noise: `psd[i, k]` the power spectral density of `outputs[i]` at
/// `freqs[k]` (V^2/Hz or A^2/Hz).
#[pyclass(module = "sane", frozen)]
pub struct NoiseSpectrum(pub sane_analysis::NoiseSpectrum);

#[pymethods]
impl NoiseSpectrum {
    #[getter]
    fn freqs<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.freqs)
    }
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn psd<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.psd)
    }
    /// The PSD of `output`.
    fn of<'py>(slf: &Bound<'py, Self>, output: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let r = &slf.get().0;
        let i = index(&r.outputs, output)?;
        Ok(view(slf.as_any(), &r.psd.row(i)))
    }
    /// The spectral density of `output` (V or A per square-root hertz).
    fn density<'py>(&self, py: Python<'py>, output: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(
            py,
            self.0.density(output).ok_or_else(|| missing(output))?,
        ))
    }
    /// The RMS noise of `output` over the band of `freqs`.
    fn rms(&self, output: &str) -> PyResult<f64> {
        self.0.rms(output).ok_or_else(|| missing(output))
    }
    /// The derivatives of every PSD by the parameters under `wrt` (all for
    /// `None`).
    #[pyo3(signature = (wrt=None))]
    fn sensitivity(&self, py: Python<'_>, wrt: Option<Names>) -> PyResult<NoiseSensitivity> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.sensitivity(&w)).map(NoiseSensitivity)
    }
}

/// `grad[i, k, j]` is `d psd[i, k] / d params[j]`.
#[pyclass(module = "sane", frozen)]
pub struct NoiseSensitivity(pub sane_analysis::NoiseSensitivity);

#[pymethods]
impl NoiseSensitivity {
    #[getter]
    fn freqs<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.freqs)
    }
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn psd<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.psd)
    }
    #[getter]
    fn grad<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray3<f64>> {
        view(slf.as_any(), &slf.get().0.grad)
    }
}

/// The finite poles (rad/s).
#[pyclass(module = "sane", frozen)]
pub struct Poles(pub sane_analysis::Poles);

#[pymethods]
impl Poles {
    #[getter]
    fn poles<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<C64>> {
        view1(slf.as_any(), &slf.get().0.poles)
    }
    /// The derivatives of every pole by the parameters under `wrt` (all for
    /// `None`).
    #[pyo3(signature = (wrt=None))]
    fn sensitivity(&self, py: Python<'_>, wrt: Option<Names>) -> PyResult<RootSensitivity> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.sensitivity(&w)).map(RootSensitivity)
    }
    fn __len__(&self) -> usize {
        self.0.poles.len()
    }
}

/// The finite transmission zeros (rad/s) from `input` to each output.
#[pyclass(module = "sane", frozen)]
pub struct Zeros(pub sane_analysis::Zeros);

#[pymethods]
impl Zeros {
    #[getter]
    fn input(&self) -> String {
        self.0.input.clone()
    }
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    /// The zeros of the transfer to `output`.
    fn of<'py>(slf: &Bound<'py, Self>, output: &str) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let z = &slf.get().0;
        let i = index(&z.outputs, output)?;
        Ok(view1(slf.as_any(), &z.zeros[i]))
    }
    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        output: &str,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        Self::of(slf, output)
    }
    /// The derivatives of every zero by the parameters under `wrt` (all for
    /// `None`), one per output.
    #[pyo3(signature = (wrt=None))]
    fn sensitivity(&self, py: Python<'_>, wrt: Option<Names>) -> PyResult<Vec<RootSensitivity>> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.sensitivity(&w)).map(|v| v.into_iter().map(RootSensitivity).collect())
    }
}

/// `grad[r, j]` is `d roots[r] / d params[j]`.
#[pyclass(module = "sane", frozen)]
pub struct RootSensitivity(pub sane_analysis::RootSensitivity);

#[pymethods]
impl RootSensitivity {
    #[getter]
    fn roots<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<C64>> {
        view1(slf.as_any(), &slf.get().0.roots)
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn grad<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<C64>> {
        view(slf.as_any(), &slf.get().0.grad)
    }
}

/// The descriptor state space `E x' = A x + B u`, `y = C x + D u`.
#[pyclass(module = "sane", frozen)]
pub struct StateSpace(pub sane_analysis::StateSpace);

#[pymethods]
impl StateSpace {
    #[getter]
    fn states(&self) -> Vec<String> {
        self.0.states.clone()
    }
    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.0.inputs.clone()
    }
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn e<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.e)
    }
    #[getter]
    fn a<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.a)
    }
    #[getter]
    fn b<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.b)
    }
    #[getter]
    fn c<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.c)
    }
    #[getter]
    fn d<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        view(slf.as_any(), &slf.get().0.d)
    }
}

/// A dominant-pole reduced model of one transfer against the full one.
#[pyclass(module = "sane", frozen)]
pub struct ReducedModel(pub sane_analysis::ReducedModel);

#[pymethods]
impl ReducedModel {
    #[getter]
    fn input(&self) -> String {
        self.0.input.clone()
    }
    #[getter]
    fn output(&self) -> String {
        self.0.output.clone()
    }
    #[getter]
    fn poles<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<C64>> {
        view1(slf.as_any(), &slf.get().0.poles)
    }
    #[getter]
    fn zeros<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<C64>> {
        view1(slf.as_any(), &slf.get().0.zeros)
    }
    #[getter]
    fn gain(&self) -> C64 {
        self.0.gain
    }
    #[getter]
    fn freqs<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.freqs)
    }
    #[getter]
    fn full<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<C64>> {
        view1(slf.as_any(), &slf.get().0.full)
    }
    #[getter]
    fn reduced<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<C64>> {
        view1(slf.as_any(), &slf.get().0.reduced)
    }
    fn full_db<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.full_db())
    }
    fn reduced_db<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.reduced_db())
    }
    fn max_error_db(&self) -> f64 {
        self.0.max_error_db()
    }
}

// --- harmonic balance ------------------------------------------------------

/// The periodic steady state: `spectra[i, k]` the coefficient `X_k` of
/// unknown `i` in `x(t) = X_0 + 2 sum_k |X_k| cos(k w0 t + arg X_k)`.
#[pyclass(module = "sane", frozen)]
pub struct HarmonicBalance(pub sane_analysis::HarmonicBalance);

#[pymethods]
impl HarmonicBalance {
    #[getter]
    fn f0(&self) -> f64 {
        self.0.f0
    }
    #[getter]
    fn harmonics(&self) -> usize {
        self.0.harmonics
    }
    #[getter]
    fn samples(&self) -> usize {
        self.0.samples
    }
    #[getter]
    fn iters(&self) -> usize {
        self.0.iters
    }
    #[getter]
    fn residual(&self) -> f64 {
        self.0.residual
    }
    #[getter]
    fn spectra<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<C64>> {
        view(slf.as_any(), &slf.get().0.spectra)
    }
    /// The coefficients of a node, unknown or branch current.
    fn spectrum<'py>(slf: &Bound<'py, Self>, output: &str) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let s = slf
            .get()
            .0
            .spectrum(output)
            .ok_or_else(|| missing(output))?;
        Ok(view(slf.as_any(), &s))
    }
    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        output: &str,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        Self::spectrum(slf, output)
    }
    /// The coefficient magnitudes `|X_k|`.
    fn magnitude<'py>(&self, py: Python<'py>, output: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(
            py,
            self.0.magnitude(output).ok_or_else(|| missing(output))?,
        ))
    }
    /// The amplitude of every harmonic: `|X_0|`, then `2 |X_k|`.
    fn amplitude<'py>(&self, py: Python<'py>, output: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(
            py,
            self.0.amplitude(output).ok_or_else(|| missing(output))?,
        ))
    }
    /// The phase of every coefficient (degrees).
    fn phase_deg<'py>(&self, py: Python<'py>, output: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(
            py,
            self.0.phase_deg(output).ok_or_else(|| missing(output))?,
        ))
    }
    /// The total harmonic distortion of `output`.
    fn thd(&self, output: &str) -> PyResult<f64> {
        self.0.thd(output).ok_or_else(|| missing(output))
    }
    /// The derivatives of every coefficient of `outputs` by the parameters
    /// under `wrt` (all for `None`).
    #[pyo3(signature = (outputs, wrt=None))]
    fn sensitivity(
        &self,
        py: Python<'_>,
        outputs: Names,
        wrt: Option<Names>,
    ) -> PyResult<HbSensitivity> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.sensitivity(&outputs.refs(), &w)).map(HbSensitivity)
    }
    /// The second derivatives of `X_harmonic(output)` by the parameters
    /// under `wrt` (all for `None`).
    #[pyo3(signature = (output, harmonic, wrt=None))]
    fn hessian(
        &self,
        py: Python<'_>,
        output: &str,
        harmonic: usize,
        wrt: Option<Names>,
    ) -> PyResult<HbHessian> {
        let w = crate::arrays::wrt(&wrt);
        run(py, || self.0.hessian(output, harmonic, &w)).map(HbHessian)
    }
    fn __repr__(&self) -> String {
        format!(
            "<HarmonicBalance f0={} Hz, {} harmonics, {} iterations>",
            self.0.f0, self.0.harmonics, self.0.iters
        )
    }
}

/// `grad[i, k, j]` is `d X_k(outputs[i]) / d params[j]`.
#[pyclass(module = "sane", frozen)]
pub struct HbSensitivity(pub sane_analysis::HbSensitivity);

#[pymethods]
impl HbSensitivity {
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.0.outputs.clone()
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn spectra<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<C64>> {
        view(slf.as_any(), &slf.get().0.spectra)
    }
    #[getter]
    fn grad<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray3<C64>> {
        view(slf.as_any(), &slf.get().0.grad)
    }
}

/// `h[a, b]` is `d^2 X_harmonic(output) / d params[a] d params[b]`.
#[pyclass(module = "sane", frozen)]
pub struct HbHessian(pub sane_analysis::HbHessian);

#[pymethods]
impl HbHessian {
    #[getter]
    fn output(&self) -> String {
        self.0.output.clone()
    }
    #[getter]
    fn harmonic(&self) -> usize {
        self.0.harmonic
    }
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }
    #[getter]
    fn param_values<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().0.param_values)
    }
    #[getter]
    fn h<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray2<C64>> {
        view(slf.as_any(), &slf.get().0.h)
    }
}

/// Registers the result types.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Regularization>()?;
    m.add_class::<OperatingPoint>()?;
    m.add_class::<Sensitivity>()?;
    m.add_class::<Hessian>()?;
    m.add_class::<Gradient>()?;
    m.add_class::<DcSweep>()?;
    m.add_class::<Trajectory>()?;
    m.add_class::<TrajectorySensitivity>()?;
    m.add_class::<AcResponse>()?;
    m.add_class::<AcSensitivity>()?;
    m.add_class::<AcHessian>()?;
    m.add_class::<SParameters>()?;
    m.add_class::<SpSensitivity>()?;
    m.add_class::<NoiseSpectrum>()?;
    m.add_class::<NoiseSensitivity>()?;
    m.add_class::<Poles>()?;
    m.add_class::<Zeros>()?;
    m.add_class::<RootSensitivity>()?;
    m.add_class::<StateSpace>()?;
    m.add_class::<ReducedModel>()?;
    m.add_class::<HarmonicBalance>()?;
    m.add_class::<HbSensitivity>()?;
    m.add_class::<HbHessian>()?;
    Ok(())
}
