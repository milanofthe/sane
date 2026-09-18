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
//!    `symbolic_transfer_approx*`, and the `linalg` / `sparse_ac` / `symbolic_poly`
//!    modules. These take already-compiled engine objects (no netlist parsing)
//!    and are what the `Model` methods dispatch to.
//!
//! UI-free throughout: the analysis stack is a plain Rust library.

use rsdag::{differentiate, Graph, Node, SymbolId};
use sane_core::{log, log_stage};
use sane_dae::assemble_dae;

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
use sane_netlist::parse;
use sane_solve::CompiledDc;
use std::collections::HashMap;
#[cfg(test)]
use std::f64::consts::PI;

mod model;
pub use model::{
    AcResponse as ModelAcResponse, DcSweep, HarmonicBalance as ModelHarmonicBalance, Model,
    ModelError, NoiseSpectrum, OpVarValue, OperatingPoint, PoleZero, ReducedModel, Sensitivity,
    StateSpace as ModelStateSpace, TempSweep, Trajectory,
};

mod linalg;
pub use linalg::{solve_complex, solve_real};

mod sparse_ac;

mod symbolic_poly;

mod ac;
mod noise;
mod pz;
mod reduce;
mod sweep;

pub use ac::{ac_h, ac_on_dae, ac_response_sensitivity, state_space_on_dae};
pub use noise::noise_on_dae;
pub use pz::{dominant_subset, finite_pencil_roots, pencil_eigvectors, pencil_root_sensitivity};
pub use reduce::{
    model_reduce_on_dae, symbolic_transfer_approx, symbolic_transfer_approx_at,
    symbolic_transfer_approx_named_at,
};
pub use sweep::temp_sweep_on_dae;

/// Label a DAE unknown for the UI: `v{k}` -> (node name, "voltage"), else
/// (stripped name, "current").
pub fn label_unknown(u: &str, node_names: &[String]) -> (String, &'static str) {
    if let Some(rest) = u.strip_prefix('v') {
        if let Ok(k) = rest.parse::<usize>() {
            let name = node_names.get(k).cloned().unwrap_or_else(|| u.to_string());
            return (name, "voltage");
        }
    }
    (
        u.strip_prefix("i_")
            .map(str::to_string)
            .unwrap_or_else(|| u.to_string()),
        "current",
    )
}

// --- shared analysis front end ----------------------------------------------

/// The compiled front end of a netlist-string analysis: the symbolic context,
/// the parsed deck, the assembled DAE, and the compiled DC solver. Returned by
/// [`prepare`] as an owned bundle (no field borrows another), so each caller
/// destructures the pieces it needs.
struct Prepared {
    ctx: Graph,
    parsed: sane_netlist::ParsedCircuit,
    dae: sane_dae::Dae,
    cdc: CompiledDc,
}

/// Parse a netlist, assemble the symbolic DAE, and compile the DC solver -- the
/// identical setup [`Model::from_netlist`](crate::Model::from_netlist) begins
/// with. Returns a human-readable error string on a parse failure (callers map
/// it to their own result type). This is the single place the analysis front
/// end lives, so every analysis enters the engine the same way.
fn prepare(netlist: &str) -> Result<Prepared, String> {
    let parsed = log_stage!("parse", parse(netlist)).map_err(|e| format!("parse error: {e}"))?;
    let mut ctx = Graph::new();
    let dae = log_stage!(
        "dae/assemble",
        assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices)
    );
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
    for (node, val) in parse_nodeset(netlist) {
        let Some(i) = parsed
            .node(&node)
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
    Ok(Prepared {
        ctx,
        parsed,
        dae,
        cdc,
    })
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

/// An analysis requested by a SPICE directive in the netlist.
pub enum Analysis {
    Op,
    Tran { tstep: f64, tstop: f64 },
    Hb { f0: f64, harmonics: usize },
}

/// Scan the netlist for analysis directives (`.op`, `.tran tstep tstop`,
/// `.hb f0 [nharmonics]`).
pub fn parse_analyses(netlist: &str) -> Vec<Analysis> {
    let mut out = Vec::new();
    for raw in netlist.lines() {
        let line = raw.trim();
        let lower = line.to_ascii_lowercase();
        if lower.starts_with(".op") {
            out.push(Analysis::Op);
        } else if lower.starts_with(".tran") {
            let toks: Vec<&str> = line.split_whitespace().skip(1).collect();
            if let (Some(ts), Some(tp)) = (
                toks.first().and_then(|s| eng(s)),
                toks.get(1).and_then(|s| eng(s)),
            ) {
                out.push(Analysis::Tran {
                    tstep: ts,
                    tstop: tp,
                });
            }
        } else if lower.starts_with(".hb") {
            // `.hb [f0] [nharmonics]`: a missing/zero f0 is inferred from the
            // circuit's periodic source (a SIN's frequency, a PULSE's 1/period).
            let toks: Vec<&str> = line.split_whitespace().skip(1).collect();
            let f0 = toks.first().and_then(|s| eng(s)).unwrap_or(0.0);
            let harmonics = toks
                .get(1)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(8);
            out.push(Analysis::Hb { f0, harmonics });
        }
    }
    out
}

/// A user initial condition: a node voltage `V(net)` or a branch current
/// `I(element)` (e.g. an inductor) to start the transient from.
pub enum IcTarget {
    V(String),
    I(String),
}

/// Parse `.ic V(net)=value I(L1)=value ...` directives.
pub fn parse_ic(netlist: &str) -> Vec<(IcTarget, f64)> {
    let mut out = Vec::new();
    for raw in netlist.lines() {
        let line = raw.trim();
        if !line.to_ascii_lowercase().starts_with(".ic") {
            continue;
        }
        for tok in line.split_whitespace().skip(1) {
            let Some(eq) = tok.find('=') else { continue };
            let lhs = tok[..eq].trim();
            let Some(val) = eng(tok[eq + 1..].trim()) else {
                continue;
            };
            let low = lhs.to_ascii_lowercase();
            if low.starts_with("v(") && lhs.ends_with(')') {
                out.push((IcTarget::V(lhs[2..lhs.len() - 1].to_string()), val));
            } else if low.starts_with("i(") && lhs.ends_with(')') {
                out.push((IcTarget::I(lhs[2..lhs.len() - 1].to_string()), val));
            }
        }
    }
    out
}

/// Parse `.nodeset V(net)=value ...` directives. Unlike `.ic` (a hard transient
/// initial condition) a node-set is a *soft* DC convergence aid: it pins the
/// nodes in a first solve phase to break the symmetry of a bistable circuit,
/// then releases them. Node voltages only (SPICE allows no branch currents in a
/// node-set). Returns `(node_name, value)` pairs.
pub fn parse_nodeset(netlist: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for raw in netlist.lines() {
        let line = raw.trim();
        if !line.to_ascii_lowercase().starts_with(".nodeset") {
            continue;
        }
        for tok in line.split_whitespace().skip(1) {
            let Some(eq) = tok.find('=') else { continue };
            let lhs = tok[..eq].trim();
            let Some(val) = eng(tok[eq + 1..].trim()) else {
                continue;
            };
            if lhs.to_ascii_lowercase().starts_with("v(") && lhs.ends_with(')') {
                out.push((lhs[2..lhs.len() - 1].to_string(), val));
            }
        }
    }
    out
}

/// Complex AC response H(jw) = e_out^T (G + jwC)^{-1} B at the operating point
/// for parameter vector `p`. Used by the AC sensitivity finite differences.
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
    xdot: &[f64],
    p: &[f64],
    t: f64,
) -> HashMap<SymbolId, f64> {
    let mut env: HashMap<SymbolId, f64> = HashMap::new();
    for (i, &s) in dae.x.iter().enumerate() {
        env.insert(s, x.get(i).copied().unwrap_or(0.0));
    }
    for (i, opt) in dae.xdot.iter().enumerate() {
        if let Some(s) = opt {
            env.insert(*s, xdot.get(i).copied().unwrap_or(0.0));
        }
    }
    for (k, name) in pnames.iter().enumerate() {
        let e = ctx.sym(name);
        if let Node::Symbol(s) = ctx.node(e) {
            env.insert(*s, p.get(k).copied().unwrap_or(0.0));
        }
    }
    env.insert(dae.t, t);
    env
}

pub fn resolve_out_idx(
    parsed: &sane_netlist::ParsedCircuit,
    dae: &sane_dae::Dae,
    output: &str,
) -> Option<usize> {
    let k = parsed.node(output)?;
    if k == 0 {
        return None;
    }
    let target = format!("v{k}");
    dae.unknowns.iter().position(|u| *u == target)
}

/// The input-coupling vector dF/d(input) at the operating point (xdot = 0).
pub fn input_vector(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    pnames: &[String],
    p: &[f64],
    x: &[f64],
    input: &str,
) -> Option<Vec<f64>> {
    let ie = ctx.sym(input);
    let input_sym = match ctx.node(ie) {
        Node::Symbol(s) => *s,
        _ => return None,
    };
    let db: Vec<_> = dae
        .residuals
        .iter()
        .map(|&r| differentiate(ctx, r, input_sym))
        .collect();
    let env = op_env(ctx, dae, pnames, &x, &[], p, 0.0);
    Some(rsdag::eval(ctx, &db, &env))
}

#[cfg(test)]
mod tests;
