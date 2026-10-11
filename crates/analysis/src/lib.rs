//! High-level circuit analyses for SANE over the engine crates (OP, transient,
//! AC, sweeps, pole-zero, sensitivity, noise, model reduction).
//!
//! **Allocator note (embedders).** Extraction -- parse, symbolic build, AD, tape
//! compilation -- is dominated by many small allocations. On large circuits (e.g.
//! the IBM power grids) the system allocator's global lock is the binding
//! constraint; a fast multithread-friendly allocator cuts extraction ~25%. The
//! SANE Python module ships `mimalloc` as its global allocator; a Rust host
//! embedding this crate should set one too (`#[global_allocator]`), since a
//! library cannot choose it for you.
//!
//! Two distinct surfaces live here, and they should not be conflated:
//!
//! 1. **[`Model`]** -- the embeddable analysis object (netlist in, result
//!    handles out), the one orchestration layer. Its analysis methods are
//!    grouped by domain in the `model/` submodules (`num`, `ac`, `tran`, `hb`,
//!    `pz`, `noise`); every construction path enters the engine through the
//!    same [`prepare`] setup. The Python bindings and the wasm app are thin
//!    wrappers over it.
//!
//! 2. **Reusable numeric kernels** -- the `*_on_dae` analysis cores, the
//!    pencil kernels and the `linalg` / `sparse_ac` modules. These take
//!    already-compiled engine objects (no netlist parsing) and are what the
//!    `Model` methods dispatch to.
//!
//! UI-free throughout: the analysis stack is a plain Rust library.

use rsdag::{differentiate, ExprId, ReduceOp, SymbolId};
use sane_core::Graph;
use sane_core::{log, log_stage};

/// Re-export the engine logger so hosts (the Python API, embedding apps) can
/// configure logging without depending on `sane-core` directly.
pub use sane_core::log as logging;

/// Set the global log level by name ("debug"/"info"/"warning"/"error"/"off").
/// Unknown names enable INFO. The single entry point hosts call to turn native
/// progress logging on or off for the whole tool.
pub fn set_log_level(level: &str) {
    log::set_level(sane_core::LogLevel::parse(level).unwrap_or(sane_core::LogLevel::Info));
}
use sane_solve::CompiledDc;

mod model;
pub use model::{
    AcHessian, AcResponse, AcSensitivity, DcOptions, DcSweep, Event, Gradient, HarmonicBalance,
    HbHessian, HbOptions, HbSensitivity, Hessian, Model, ModelError, NoiseSensitivity,
    NoiseSpectrum, OpVarValue, OperatingPoint, Point, Poles, ReducedModel, Regularization,
    RootSensitivity, SParameters, Sensitivity, SpSensitivity, StateSpace, Trajectory,
    TrajectorySensitivity, TransientOptions, Zeros,
};
/// The arrays results come in, re-exported so a caller takes the same
/// version.
pub use ndarray;
/// The complex scalar of every small-signal result.
pub use num_complex::Complex64;

/// `points` frequencies spaced logarithmically from `fstart` to `fstop`,
/// both included: the usual AC grid.
pub fn log_grid(fstart: f64, fstop: f64, points: usize) -> Vec<f64> {
    let (l0, l1) = (fstart.log10(), fstop.log10());
    let last = points.saturating_sub(1).max(1) as f64;
    (0..points)
        .map(|k| 10f64.powf(l0 + (l1 - l0) * k as f64 / last))
        .collect()
}

mod sparse_ac;

// --- shared analysis front end ----------------------------------------------

/// Assemble a circuit's symbolic DAE and compile it: the one
/// setup every model of a circuit goes through (see
/// [`Model::new`](crate::Model::new)). The devices decide their structure at
/// `values` (by parameter name) where those set a parameter, else at the
/// circuit's own; the circuit's node-set seeds the operating point. An
/// error names what does not assemble.
fn prepare(
    circuit: &sane_circuit::Circuit,
    values: &dyn Fn(&str) -> Option<f64>,
) -> Result<CompiledDc, String> {
    let mut ctx = Graph::new();
    let mut dae = log_stage!(
        "dae/assemble",
        sane_dae::assemble_at(&mut ctx, circuit, values)
    )?;
    // The devices' Newton limits under the parameters the circuit is at.
    dae.keep_limits_at(&ctx, |s| {
        let name = ctx.symbol_name(s);
        values(name).or_else(|| circuit.param_value(name))
    });
    // `.nodeset` symmetry breaking: device-emitted DC seeds first (`idt(u, ic)`
    // states), then explicit `.nodeset` directives on
    // top (an explicit directive overrides a device seed on the same unknown),
    // so every cold DC solve (any analysis) runs the stiff-pin phase.
    let mut nodeset: Vec<(usize, f64)> = dae
        .hints
        .dc_seeds
        .iter()
        .filter_map(|&(s, val)| dae.unknown_of(s).map(|i| (i, val)))
        .collect();
    for &(ref node, val) in &circuit.nodeset {
        let Some(i) = circuit.find_node(node).and_then(|k| dae.node(k)) else {
            continue;
        };
        match nodeset.iter_mut().find(|(j, _)| *j == i) {
            Some(entry) => entry.1 = val,
            None => nodeset.push((i, val)),
        }
    }
    let mut cdc = log_stage!("compile", CompiledDc::new(ctx, dae));
    if !nodeset.is_empty() {
        cdc.set_nodeset(nodeset);
    }
    Ok(cdc)
}

/// The derivative of `e` along the drive `syms` (see `ModelInner::drive`).
pub(crate) fn d_drive(ctx: &mut Graph, e: ExprId, syms: &[SymbolId]) -> ExprId {
    match syms {
        [s] => differentiate(ctx, e, *s),
        _ => {
            let terms: Vec<ExprId> = syms.iter().map(|&s| differentiate(ctx, e, s)).collect();
            ctx.reduce(ReduceOp::Sum, terms)
        }
    }
}
