//! Python bindings for SANE: the Rust API as Python sees it. A model, the
//! point of a binding, every analysis a method of the point and every result
//! the Rust result, its arrays handed to numpy without a copy. Everything
//! runs in Rust; these are the handles.

// The `#[pymethods]` macro expands `?`/return paths into `PyErr: From<PyErr>`
// conversions clippy flags as useless; they are macro-generated, not our code.
#![allow(clippy::useless_conversion)]

// Extraction (parse + symbolic build + AD + tape compile) is dominated by many
// small allocations; the OS allocator's global lock is the binding constraint
// and contends badly under the worker pool. mimalloc removes that wall.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

mod arrays;
mod circuit;
mod model;
mod results;

use model::PyModel;

/// A model error as a Python exception.
pub(crate) fn err(e: sane_analysis::ModelError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Runs `f` without the GIL, then raises the warnings it captured (as
/// `sane.warnings` categories) before its result.
pub(crate) fn run<T: Send>(
    py: Python<'_>,
    f: impl FnOnce() -> Result<T, sane_analysis::ModelError> + Send,
) -> PyResult<T> {
    let r = py.allow_threads(f);
    surface(py)?;
    r.map_err(err)
}

/// Raises every warning the engine captured since the last call, each as
/// the `sane.warnings` category of its concern, unconditionally (whatever
/// the log level).
pub(crate) fn surface(py: Python<'_>) -> PyResult<()> {
    let caught = sane_core::log::drain_captured();
    if caught.is_empty() {
        return Ok(());
    }
    let module = py.import_bound("sane.warnings").ok();
    for (concern, msg) in caught {
        let name = match concern {
            sane_core::log::Concern::Convergence => "SaneConvergenceWarning",
            sane_core::log::Concern::Numerical => "SaneNumericalWarning",
        };
        let category = match module.as_ref().and_then(|m| m.getattr(name).ok()) {
            Some(c) => c,
            None => py
                .get_type_bound::<pyo3::exceptions::PyUserWarning>()
                .into_any(),
        };
        PyErr::warn_bound(py, &category, &msg, 1)?;
    }
    Ok(())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // No thread setting here: the linear solves are sequential by construction
    // (rslab's KLU) and the parallelism lives in
    // the outer sweeps, on the worker pool with its own default. `1` used to
    // mean "faer sequential" and would now shrink that pool to one thread.
    m.add_class::<circuit::PyCircuit>()?;
    m.add_class::<circuit::PyWaveform>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<model::Point>()?;
    m.add_class::<model::ParamGroup>()?;
    results::register(m)?;
    m.add_function(wrap_pyfunction!(set_parallelism, m)?)?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(vectfit_verilog_a, m)?)?;
    m.add_function(wrap_pyfunction!(profile_begin, m)?)?;
    m.add_function(wrap_pyfunction!(profile_take, m)?)?;
    Ok(())
}
