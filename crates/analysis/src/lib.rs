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
//! 2. **Reusable numeric kernels** -- the `*_on_dae` analysis cores plus
//!    `ac_response_sensitivity`, `pencil_root_sensitivity`, `finite_pencil_roots`,
//!    and the `linalg` / `sparse_ac` modules. These take already-compiled engine objects (no netlist parsing)
//!    and are what the `Model` methods dispatch to.
//!
//! UI-free throughout: the analysis stack is a plain Rust library.

use rsdag::{differentiate, ExprId, Node, ReduceOp, SymbolId};
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
#[cfg(test)]
use sane_core::constants::{DC_OP_MAXIT, DC_OP_TOL};
use sane_solve::CompiledDc;
use std::collections::HashMap;
#[cfg(test)]
use std::f64::consts::PI;

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

mod linalg;
pub use linalg::{solve_complex, solve_real};

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

mod ac;
mod noise;
mod pz;
mod reduce;

pub use ac::ac_response_sensitivity;
pub use noise::noise_on_dae;
pub use pz::{dominant_subset, finite_pencil_roots, pencil_eigvectors, pencil_root_sensitivity};
pub use reduce::{model_reduce_on_dae, Reduction};

// --- shared analysis front end ----------------------------------------------

/// The compiled front end of a circuit: the symbolic context, the assembled
/// DAE, and the compiled DC solver. Returned by [`prepare`] as an owned
/// bundle (no field borrows another), so each caller destructures the pieces
/// it needs.
struct Prepared {
    ctx: Graph,
    dae: sane_dae::Dae,
    cdc: CompiledDc,
}

/// Assemble a circuit's symbolic DAE and compile the DC solver: the one
/// setup every model of a circuit goes through (see
/// [`Model::new`](crate::Model::new)). The devices decide their structure at
/// `values` (by parameter name) where those set a parameter, else at the
/// circuit's own; the circuit's node-set seeds the operating point. An
/// error names what does not assemble.
fn prepare(
    circuit: &sane_circuit::Circuit,
    values: &dyn Fn(&str) -> Option<f64>,
) -> Result<Prepared, String> {
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
    let mut cdc = log_stage!("compile", CompiledDc::new(&mut ctx, &dae));
    // `.nodeset` symmetry breaking: device-emitted DC seeds first (`idt(u, ic)`
    // states, keyed by unknown name), then explicit `.nodeset` directives on
    // top (an explicit directive overrides a device seed on the same unknown),
    // so every cold DC solve (any analysis) runs the stiff-pin phase.
    let mut nodeset: Vec<(usize, f64)> = dae
        .dc_seeds
        .iter()
        .filter_map(|(name, val)| {
            dae.unknowns
                .iter()
                .position(|u| u == name)
                .map(|i| (i, *val))
        })
        .collect();
    for &(ref node, val) in &circuit.nodeset {
        let Some(i) = circuit
            .find_node(node)
            .and_then(|k| dae.unknowns.iter().position(|u| *u == format!("v{k}")))
        else {
            continue;
        };
        match nodeset.iter_mut().find(|(j, _)| *j == i) {
            Some(entry) => entry.1 = val,
            None => nodeset.push((i, val)),
        }
    }
    if !nodeset.is_empty() {
        cdc.set_nodeset(nodeset);
    }
    Ok(Prepared { ctx, dae, cdc })
}

/// Parse an engineering-notation number like `10u`, `1meg`, `2.2k`.
pub fn eng(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let lower = s.to_ascii_lowercase();
    // `meg` before `m`/`g`; longest suffixes first.
    for (suf, mult) in [
        ("meg", 1e6),
        ("t", 1e12),
        ("g", 1e9),
        ("k", 1e3),
        ("m", 1e-3),
        ("u", 1e-6),
        ("µ", 1e-6),
        ("n", 1e-9),
        ("p", 1e-12),
        ("f", 1e-15),
    ] {
        if let Some(num) = lower.strip_suffix(suf) {
            if let Ok(v) = num.trim().parse::<f64>() {
                return Some(v * mult);
            }
        }
    }
    None
}

/// Build the operating-point evaluation environment: bind each state unknown
/// `dae.x[i]` to `x[i]`, each state-derivative symbol to `xdot[i]` (or 0 where
/// `xdot` is shorter -- a DC point passes `&[]`), each parameter name to its
/// value, and the time symbol to `t`. The single source of this map, which was
/// otherwise hand-rebuilt identically across DC / AC / transient / sensitivity.
pub(crate) fn op_env(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    pnames: &[String],
    x: &[f64],
    p: &[f64],
    t: f64,
) -> HashMap<SymbolId, f64> {
    let mut env: HashMap<SymbolId, f64> = HashMap::new();
    for (i, &s) in dae.x.iter().enumerate() {
        env.insert(s, x.get(i).copied().unwrap_or(0.0));
    }
    for (k, name) in pnames.iter().enumerate() {
        let e = ctx.sym(name);
        if let Node::Symbol(s) = ctx.node(e) {
            env.insert(*s, p.get(k).copied().unwrap_or(0.0));
        }
    }
    env.insert(dae.t, t);
    // noise generators are zero in every evaluation
    for g in dae.observers.generators(ctx) {
        env.insert(g, 0.0);
    }
    env
}

/// What drives the small-signal response from the source `input`: the
/// parameters among `pnames` a unit input moves together. A source with a
/// value is driven through it; one driven by a waveform through the
/// parameters that shift its level as a whole (see
/// [`sane_circuit::SourceFn::is_level`]), as an AC source on top of its
/// transient waveform would. Empty when it has neither (a folded source).
pub(crate) fn drive_params(pnames: &[String], input: &str) -> Vec<String> {
    let own = sane_circuit::value_symbol_name(input);
    if pnames.contains(&own) {
        return vec![own];
    }
    let prefix = format!("{input}.");
    (pnames.iter())
        .filter(|p| (p.strip_prefix(&prefix)).is_some_and(sane_circuit::SourceFn::is_level))
        .cloned()
        .collect()
}

/// The symbols of [`drive_params`]; `None` for none.
pub(crate) fn drive_syms(ctx: &mut Graph, pnames: &[String], input: &str) -> Option<Vec<SymbolId>> {
    let syms: Vec<SymbolId> = (drive_params(pnames, input).iter())
        .filter_map(|n| {
            let e = ctx.sym(n);
            match ctx.node(e) {
                Node::Symbol(s) => Some(*s),
                _ => None,
            }
        })
        .collect();
    (!syms.is_empty()).then_some(syms)
}

/// The derivative of `e` along the drive `syms` (see [`drive_params`]).
pub(crate) fn d_drive(ctx: &mut Graph, e: ExprId, syms: &[SymbolId]) -> ExprId {
    match syms {
        [s] => differentiate(ctx, e, *s),
        _ => {
            let terms: Vec<ExprId> = syms.iter().map(|&s| differentiate(ctx, e, s)).collect();
            ctx.reduce(ReduceOp::Sum, terms)
        }
    }
}

/// The input-coupling vector dI/d(input) at the operating point.
pub fn input_vector(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    pnames: &[String],
    p: &[f64],
    x: &[f64],
    input: &str,
) -> Option<Vec<f64>> {
    let syms = drive_syms(ctx, pnames, input)?;
    let db: Vec<_> = (dae.at_rest(ctx).0.iter())
        .map(|&r| d_drive(ctx, r, &syms))
        .collect();
    let env = op_env(ctx, dae, pnames, x, p, 0.0);
    Some(rsdag::eval(ctx, &db, &env))
}

#[cfg(test)]
mod tests;
