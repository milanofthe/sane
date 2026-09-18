//! `Model` -- the embeddable, Rust-first analysis object.
//!
//! A `Model` owns a prepared circuit (symbolic context, DAE, compiled solver) plus
//! a named parameter store, and exposes every analysis as a single call that
//! takes value overrides and returns a result *handle*. The handle keeps the
//! solved state in Rust, so derived analyses (`op.sensitivity(..)`) run without a
//! re-solve and nothing crosses a language boundary. This is the one
//! orchestration layer; the Python bindings and the netlist front end are thin
//! wrappers over it.
//!
//! ```no_run
//! use sane_analysis::Model;
//! let sim = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").unwrap();
//! let op = sim.operating_point(&[]).unwrap();
//! assert!((op.get("out").unwrap() - 2.5).abs() < 1e-9);
//! sim.set("R2", 3e3).unwrap();                 // mutate a named parameter
//! let op = sim.operating_point(&[]).unwrap();  // re-solve picks it up
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;

use rsdag::{Graph, Node, SymbolId};
use sane_core::constants::{DC_OP_MAXIT, DC_OP_TOL};
use sane_core::log_stage;
use sane_dae::linearize::{linearize_with, Granularity};
use sane_dae::{eliminate_nodes as dae_eliminate_nodes, reduce_graph};
use sane_solve::{CompiledDc, Convergence, SolverTricks, TransientMethod};

use crate::{prepare, Prepared};

/// Error from building or analysing a [`Model`].
#[derive(Debug, Clone)]
pub enum ModelError {
    /// Netlist parse / assembly failure.
    Parse(String),
    /// The DC operating point did not converge.
    NoConverge,
    /// A reference is not a known node / unknown / branch current.
    UnknownRef(String),
    /// A name is not a parameter.
    UnknownParam(String),
    /// A numeric step failed (singular matrix, AFT grid too coarse, ...).
    Numeric(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Parse(m) => write!(f, "parse error: {m}"),
            ModelError::NoConverge => write!(f, "DC operating point did not converge"),
            ModelError::UnknownRef(r) => {
                write!(f, "'{r}' is not a known node, unknown or branch current")
            }
            ModelError::UnknownParam(p) => write!(f, "'{p}' is not a parameter"),
            ModelError::Numeric(m) => write!(f, "{m}"),
        }
    }
}
impl std::error::Error for ModelError {}

/// Resolve a reference to an index into the unknown vector: an exact unknown name
/// (`v3`, `i_V1`), a node name (`out` -> its `v{id}`), or an element whose branch
/// current is an unknown (`V1` -> `i_V1`). Mirrors the Python `resolve_unknown`.
fn resolve_ref(unknowns: &[String], node_names: &[String], reference: &str) -> Option<usize> {
    if let Some(i) = unknowns.iter().position(|u| u == reference) {
        return Some(i);
    }
    if let Some(k) = node_names.iter().position(|n| n == reference) {
        let u = format!("v{k}");
        if let Some(i) = unknowns.iter().position(|x| *x == u) {
            return Some(i);
        }
    }
    let ic = format!("i_{reference}");
    unknowns.iter().position(|u| *u == ic)
}

/// Named parameter store: bound values, defaults, and the hierarchical name
/// resolution / `p`-vector assembly that used to live in the Python wrapper.
struct ParamStore {
    /// Parameter names in the engine's column order.
    pnames: Vec<String>,
    /// Fast membership for column names.
    pset: HashSet<String>,
    /// Intermediate group prefixes ("X1", "X1.D1" for "X1.D1.Is").
    prefixes: HashSet<String>,
    /// Current bound values (canonical name -> value), behind a lock so `set`
    /// works through `&self` and `Model` stays `Send + Sync`.
    values: Mutex<HashMap<String, f64>>,
    /// Construction-time snapshot for `reset`.
    defaults: HashMap<String, f64>,
}

/// The module default of every parameter column, by symbol (see
/// [`sane_dae::Dae::param_defaults`]): the value an unstated parameter takes.
fn column_defaults(cdc: &CompiledDc, dae: &sane_dae::Dae) -> Vec<Option<f64>> {
    cdc.param_syms()
        .iter()
        .map(|s| dae.param_defaults.get(s).copied())
        .collect()
}

impl ParamStore {
    fn new(pnames: Vec<String>, bound: HashMap<String, f64>, defaults: Vec<Option<f64>>) -> Self {
        let pset: HashSet<String> = pnames.iter().cloned().collect();
        let mut prefixes = HashSet::new();
        for p in &pnames {
            let parts: Vec<&str> = p.split('.').collect();
            for i in 1..parts.len() {
                prefixes.insert(parts[..i].join("."));
            }
        }
        // Keep only bound values that are actually parameters (canonicalised).
        let mut known = HashMap::new();
        for (k, v) in bound {
            if let Some(c) = Self::resolve_in(&pset, &k) {
                known.insert(c, v);
            }
        }
        // Parameters the deck left unspecified take the device's default.
        for (p, d) in pnames.iter().zip(defaults) {
            if let (false, Some(v)) = (known.contains_key(p), d) {
                known.insert(p.clone(), v);
            }
        }
        ParamStore {
            pnames,
            pset,
            prefixes,
            defaults: known.clone(),
            values: Mutex::new(known),
        }
    }

    fn resolve_in(pset: &HashSet<String>, name: &str) -> Option<String> {
        if pset.contains(name) {
            return Some(name.to_string());
        }
        let r = sane_mna::value_symbol_name(name);
        if pset.contains(&r) {
            Some(r)
        } else {
            None
        }
    }

    fn resolve(&self, name: &str) -> Option<String> {
        Self::resolve_in(&self.pset, name)
    }

    fn get(&self, name: &str) -> Option<f64> {
        let c = self.resolve(name)?;
        Some(self.values.lock().unwrap().get(&c).copied().unwrap_or(0.0))
    }

    fn set(&self, name: &str, value: f64) -> Result<(), ModelError> {
        let c = self
            .resolve(name)
            .ok_or_else(|| ModelError::UnknownParam(name.to_string()))?;
        self.values.lock().unwrap().insert(c, value);
        Ok(())
    }

    fn reset(&self) {
        *self.values.lock().unwrap() = self.defaults.clone();
    }

    fn values_map(&self) -> HashMap<String, f64> {
        self.values.lock().unwrap().clone()
    }

    /// Parameter symbols that appear in the residual but have no bound value, so a
    /// numeric analysis silently uses `0` for them (a symbolic analysis keeps them
    /// free). Typically a foundry `.param`/global the deck never defines (e.g.
    /// `mc_mm_switch`), whose `0` default then propagates through the model card.
    /// Sorted for a stable message.
    fn unbound(&self) -> Vec<String> {
        let vals = self.values.lock().unwrap();
        let mut u: Vec<String> = self
            .pnames
            .iter()
            .filter(|n| !vals.contains_key(*n))
            .cloned()
            .collect();
        u.sort();
        u
    }

    /// The parameter vector in column order, from the bound values plus per-call
    /// overrides (canonicalised; unknown override keys are ignored).
    fn pvec(&self, overrides: &[(&str, f64)]) -> Vec<f64> {
        let vals = self.values.lock().unwrap();
        if overrides.is_empty() {
            return self
                .pnames
                .iter()
                .map(|n| vals.get(n).copied().unwrap_or(0.0))
                .collect();
        }
        let mut ov = HashMap::new();
        for (k, v) in overrides {
            if let Some(c) = self.resolve(k) {
                ov.insert(c, *v);
            }
        }
        self.pnames
            .iter()
            .map(|n| ov.get(n).or_else(|| vals.get(n)).copied().unwrap_or(0.0))
            .collect()
    }
}

/// Shared inner state; result handles hold an `Arc<ModelInner>` so they are owned,
/// `'static`, and need no lifetime threading (and are pyo3-friendly).
struct ModelInner {
    /// Symbolic context (interior mutability: lazy param-Jacobian / Hessian tapes
    /// need `&mut Graph`). The lock keeps `Model` `Send + Sync` for embedding.
    ctx: Arc<Mutex<Graph>>,
    dae: sane_dae::Dae,
    cdc: CompiledDc,
    /// Index-2 topologies of the deck (capacitor/source loops, inductor/
    /// source cutsets), detected once at build time. Transient consults it:
    /// such a circuit's hidden constraint carries no truncation error, so the
    /// step controller has nothing to control and would otherwise stride past
    /// the requested resolution. See `sane_mna::index2`.
    index2: sane_mna::index2::Index2Report,
    store: ParamStore,
    unknowns: Vec<String>,
    node_names: Vec<String>,
    /// Power ports from deck `P` elements, in deck order: `(drive source,
    /// network-side node, z0)`. Empty when the deck has none (or the model
    /// was built from parts rather than a netlist).
    ports: Vec<(String, String, f64)>,
    /// Cached weighted-adjoint tapes for AC/SP VJPs, keyed by the input
    /// source name (see `gradients::AcVjpTape`): the frequency-independent
    /// symbolic scaffolding built once per input, so the per-frequency work
    /// is pure evaluation (no context growth).
    ac_vjp_tapes: Mutex<HashMap<String, ac::AcVjpTape>>,
    /// Last converged DC operating point `(p, x)`, so a run of analyses at the
    /// same parameter point (op, then ac, noise, pole-zero, ...) solves the DC
    /// once instead of per analysis. Keyed on the full parameter vector, so any
    /// parameter change (override or `set`) is an automatic cache miss; a hit
    /// returns exactly what a cold re-solve would, so results are unchanged.
    op_cache: Mutex<Option<(Vec<f64>, Vec<f64>, Option<f64>)>>,
}

/// The embeddable analysis object. Cheap to clone (shares one inner via `Arc`).
/// Detect the deck's index-2 topologies and report them once.
///
/// A capacitor/voltage-source loop (or inductor/current-source cutset) pins its
/// branch algebraically: the hidden constraint carries no local truncation
/// error, so the integrator's embedded estimate stays near zero and the
/// adaptive step may stride past the resolution the caller asked for. The
/// result is not unstable, it is quietly inaccurate -- so the engine names the
/// elements responsible and points at `dt_max`, which already exists for
/// bounding the step. It does not bound it on the caller's behalf.
fn detect_index2(
    circuit: &sane_mna::Circuit,
    devices: &[sane_dae::DeviceInstance],
) -> sane_mna::index2::Index2Report {
    let terminals: Vec<Vec<usize>> = devices.iter().map(|d| d.terminals.clone()).collect();
    let rep = sane_mna::index2::detect(circuit, &terminals);
    if rep.is_index2() {
        sane_core::log::warn_captured(&format!(
            "index-2 topology ({}): its constraint carries no truncation error, so the adaptive transient step may stride past the resolution you asked for -- set dt_max if the trace looks coarse",
            rep.summary()
        ));
    }
    rep
}

#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

impl Model {
    /// Build a `Model` from a SPICE-like netlist string.
    pub fn from_netlist(src: &str) -> Result<Model, ModelError> {
        let mut task = sane_core::log::task("EXTRACT", "extract", "");
        let Prepared {
            ctx,
            parsed,
            dae,
            cdc,
        } = prepare(src).map_err(ModelError::Parse)?;
        let pnames = cdc.param_names(&ctx);
        task.finish(format!("dim: {}, params: {}", dae.dim(), pnames.len()));
        let index2 = detect_index2(&parsed.circuit, &parsed.devices);
        let store = ParamStore::new(
            pnames,
            parsed.values.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            column_defaults(&cdc, &dae),
        );
        let unknowns = dae.unknowns.clone();
        let node_names = parsed.node_names.clone();
        let ports = parsed
            .ports
            .iter()
            .map(|pt| (pt.name.clone(), pt.node.clone(), pt.z0))
            .collect();
        Ok(Model {
            inner: Arc::new(ModelInner {
                ctx: Arc::new(Mutex::new(ctx)),
                dae,
                cdc,
                index2,
                store,
                unknowns,
                node_names,
                ports,
                ac_vjp_tapes: Mutex::new(HashMap::new()),
                op_cache: Mutex::new(None),
            }),
        })
    }

    /// Build a `Model` from already-compiled parts that **share** an existing
    /// symbolic context (e.g. a graph transform: prune / eliminate / linearize
    /// produces a new DAE on the same arena). The shared `Arc<Mutex<Graph>>`
    /// keeps symbol ids consistent, so expression handles interoperate.
    /// The deck's index-2 topologies: capacitor/voltage-source loops and
    /// inductor/current-source cutsets, by element name. Empty for the
    /// index-1 circuits that make up almost everything. Callers that care
    /// about transient accuracy can consult this and set `dt_max`.
    pub fn index2(&self) -> &sane_mna::index2::Index2Report {
        &self.inner.index2
    }

    pub fn from_parts(
        ctx: Arc<Mutex<Graph>>,
        dae: sane_dae::Dae,
        cdc: CompiledDc,
        values: HashMap<String, f64>,
        node_names: Vec<String>,
        // element graph the DAE came from, for topological index detection;
        // None for models assembled without one
        circuit: Option<&sane_mna::Circuit>,
        // the nonlinear devices of that graph (they conduct, so they break
        // cutsets); empty when there are none
        devices: &[sane_dae::DeviceInstance],
    ) -> Model {
        let pnames = {
            let c = ctx.lock().unwrap();
            cdc.param_names(&c)
        };
        let store = ParamStore::new(pnames, values, column_defaults(&cdc, &dae));
        let unknowns = dae.unknowns.clone();
        Model {
            inner: Arc::new(ModelInner {
                ctx,
                dae,
                cdc,
                index2: circuit
                    .map(|c| detect_index2(c, devices))
                    .unwrap_or_default(),
                store,
                unknowns,
                node_names,
                ports: Vec::new(),
                ac_vjp_tapes: Mutex::new(HashMap::new()),
                op_cache: Mutex::new(None),
            }),
        }
    }

    // --- introspection ---

    pub fn params(&self) -> &[String] {
        &self.inner.store.pnames
    }
    pub fn unknowns(&self) -> &[String] {
        &self.inner.unknowns
    }
    pub fn node_names(&self) -> &[String] {
        &self.inner.node_names
    }
    pub fn dim(&self) -> usize {
        self.inner.unknowns.len()
    }

    /// The shared symbolic context (`Arc<Mutex>`), so a host (e.g. the Python
    /// binding) can hand out expression handles that operate on the *same* graph
    /// this model was built from.
    pub fn context_arc(&self) -> Arc<Mutex<Graph>> {
        self.inner.ctx.clone()
    }

    /// The underlying symbolic DAE (residual expression ids, state symbols, ...),
    /// for the symbolic interface.
    pub fn dae(&self) -> &sane_dae::Dae {
        &self.inner.dae
    }

    /// The compiled DC solver (residual / Jacobian tapes), for the low-level
    /// numeric and gradient interfaces.
    /// The deck's power ports (`P` elements) in deck order:
    /// `(drive source, network-side node, z0)` per port.
    pub fn ports(&self) -> &[(String, String, f64)] {
        &self.inner.ports
    }

    /// The per-input weighted-adjoint tape cache (see `gradients::AcVjpTape`).
    pub(crate) fn ac_vjp_tape_cache(&self) -> &Mutex<HashMap<String, ac::AcVjpTape>> {
        &self.inner.ac_vjp_tapes
    }

    pub fn cdc(&self) -> &CompiledDc {
        &self.inner.cdc
    }

    /// The parameter vector in engine column order from the bound store plus
    /// optional per-call overrides (unknown keys ignored). The single place the
    /// `p`-vector is assembled, shared by every analysis.
    pub fn pvec(&self, overrides: &[(&str, f64)]) -> Vec<f64> {
        self.inner.store.pvec(overrides)
    }

    /// Parameters referenced in the residual but not bound to a value (a numeric
    /// analysis uses `0`; a symbolic one keeps them free). Use this to surface a
    /// missing foundry `.param`/global before it silently biases a numeric result.
    pub fn unbound_params(&self) -> Vec<String> {
        self.inner.store.unbound()
    }

    /// Bound value of a parameter by canonical column name (or `0.0`), used by
    /// the gradient interfaces that need fallback values for combined systems.
    pub fn bound_value(&self, name: &str) -> f64 {
        self.inner
            .store
            .values
            .lock()
            .unwrap()
            .get(name)
            .copied()
            .unwrap_or(0.0)
    }

    /// The circuit temperature `$temp` [K] (deck `.temp`, defaulting to nominal).
    /// Drives resistor thermal noise even when no device makes it a DC parameter.
    pub(crate) fn temp_k(&self) -> f64 {
        self.inner
            .store
            .values
            .lock()
            .unwrap()
            .get(sane_core::constants::TEMP_SYMBOL)
            .copied()
            .unwrap_or(sane_core::constants::TEMP_NOMINAL_K)
    }

    // --- parameter store ---

    /// Read a parameter's bound value (or `0.0` if unset); `None` if not a parameter.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.inner.store.get(name)
    }
    /// Set a parameter; errors if `name` is not a parameter.
    pub fn set(&self, name: &str, value: f64) -> Result<(), ModelError> {
        self.inner.store.set(name, value)
    }
    /// Restore all parameters to their construction defaults.
    pub fn reset(&self) {
        self.inner.store.reset()
    }
    /// All bound parameter values as a map.
    pub fn values(&self) -> HashMap<String, f64> {
        self.inner.store.values_map()
    }
    /// Is `name` a parameter (after reserved-namespace resolution)?
    pub fn is_param(&self, name: &str) -> bool {
        self.inner.store.resolve(name).is_some()
    }
    /// Is `name` an intermediate parameter group (e.g. `X1`, `X1.D1`)?
    pub fn is_group(&self, name: &str) -> bool {
        self.inner.store.prefixes.contains(name)
    }
    /// Direct children (next path segment) of a parameter group, sorted.
    pub fn children(&self, prefix: &str) -> Vec<String> {
        let pre = format!("{prefix}.");
        let mut kids = HashSet::new();
        for p in &self.inner.store.pnames {
            if let Some(rest) = p.strip_prefix(&pre) {
                kids.insert(rest.split('.').next().unwrap_or(rest).to_string());
            }
        }
        for g in &self.inner.store.prefixes {
            if let Some(rest) = g.strip_prefix(&pre) {
                kids.insert(rest.split('.').next().unwrap_or(rest).to_string());
            }
        }
        let mut out: Vec<String> = kids.into_iter().collect();
        out.sort();
        out
    }
    /// Resolve a node / unknown / branch-current reference to its state index.
    pub fn resolve(&self, reference: &str) -> Option<usize> {
        resolve_ref(&self.inner.unknowns, &self.inner.node_names, reference)
    }

    // --- analyses ---

    /// Solve the DC operating point with optional per-call value overrides.
    pub fn operating_point(&self, overrides: &[(&str, f64)]) -> Result<OperatingPoint, ModelError> {
        let mut task = sane_core::log::task("DC", "dc", &format!("(dim: {})", self.dim()));
        let (x, p, regularized_at_gmin) = self.inner.solve_dc_reg(overrides)?;
        task.finish(match regularized_at_gmin {
            Some(g) => format!("converged: True, gmin-regularized: {g:.1e}"),
            None => "converged: True".to_string(),
        });
        Ok(OperatingPoint {
            sim: self.inner.clone(),
            x,
            p,
            regularized_at_gmin,
        })
    }

    /// Transient response over `t_eval`, labeled. The ergonomic Rust entry: a thin
    /// convenience over the single [`Model::solve_transient`] engine path --
    /// resolve `overrides` to the parameter vector, integrate with `method` from
    /// the consistent DC start, and label the rows.
    pub fn transient(
        &self,
        method: TransientMethod,
        overrides: &[(&str, f64)],
        t_eval: &[f64],
        rtol: f64,
        atol: f64,
    ) -> Result<Trajectory, ModelError> {
        let p = self.inner.store.pvec(overrides);
        let rows = self.solve_transient(method, p, t_eval.to_vec(), None, rtol, atol, None)?;
        Ok(Trajectory {
            sim: self.inner.clone(),
            t: t_eval.to_vec(),
            rows,
        })
    }

    /// The switching events of the most recent transient (any entry path):
    /// `(surface name, time, direction)`, direction `+1` for the surface
    /// expression rising through zero, `-1` falling. Surfaces are declared by
    /// the devices (Verilog-A `@(cross ...)`, switch thresholds).
    pub fn transient_events(&self) -> Vec<(String, f64, i8)> {
        let names = self.inner.cdc.event_names();
        self.inner
            .cdc
            .last_transient_events()
            .into_iter()
            .map(|e| (names[e.index].clone(), e.t, e.direction))
            .collect()
    }

    /// Small-signal noise spectrum at `output` over `[fstart, fstop]`, linearised
    /// at the operating point (returns frequencies and total output noise PSD).
    pub fn noise(
        &self,
        overrides: &[(&str, f64)],
        output: &str,
        fstart: f64,
        fstop: f64,
        points: usize,
    ) -> Result<NoiseSpectrum, ModelError> {
        let (x, p) = self.inner.solve_dc(overrides)?;
        let out_idx = self
            .resolve(output)
            .ok_or_else(|| ModelError::UnknownRef(output.to_string()))?;
        let temp_k = self.temp_k();
        let mut ctx = self.inner.ctx.lock().unwrap();
        let (freqs, psd) = crate::noise_on_dae(
            &mut ctx,
            &self.inner.dae,
            &self.inner.cdc,
            out_idx,
            &x,
            &p,
            fstart,
            fstop,
            points,
            temp_k,
        )
        .map_err(ModelError::Numeric)?;
        Ok(NoiseSpectrum { freqs, psd })
    }
    /// Linearised descriptor state-space `(E, A, B, C, D)` at the operating point:
    /// `E x' = A x + B u`, `y = C x + D u`.
    pub fn state_space(
        &self,
        overrides: &[(&str, f64)],
        input: &str,
        output: &str,
    ) -> Result<StateSpace, ModelError> {
        let (x, p) = self.inner.solve_dc(overrides)?;
        let out_idx = self
            .resolve(output)
            .ok_or_else(|| ModelError::UnknownRef(output.to_string()))?;
        let mut ctx = self.inner.ctx.lock().unwrap();
        let (e, a, b, c, d) = crate::state_space_on_dae(
            &mut ctx,
            &self.inner.dae,
            &self.inner.cdc,
            input,
            out_idx,
            &x,
            &p,
        );
        Ok(StateSpace {
            e,
            a,
            b,
            c,
            d,
            input: input.to_string(),
            output: output.to_string(),
        })
    }

    /// Sweep `output` over temperature `[t0, t1]` (deg C), re-solving the
    /// operating point at each step. Returns only converged points.
    pub fn temp_sweep(
        &self,
        overrides: &[(&str, f64)],
        output: &str,
        t0: f64,
        t1: f64,
        points: usize,
    ) -> Result<TempSweep, ModelError> {
        let p0 = self.inner.store.pvec(overrides);
        let out_idx = self
            .resolve(output)
            .ok_or_else(|| ModelError::UnknownRef(output.to_string()))?;
        let (temps, values) = crate::temp_sweep_on_dae(
            &self.inner.cdc,
            &self.inner.store.pnames,
            out_idx,
            &p0,
            t0,
            t1,
            points,
        )
        .map_err(ModelError::Numeric)?;
        Ok(TempSweep {
            temps,
            values,
            output: output.to_string(),
        })
    }

    /// Balanced/pole-residue reduced model of `output/input` to `order`, fit over
    /// `[fstart, fstop]`. Returns the magnitude response of full vs reduced and
    /// the kept poles/zeros.
    pub fn model_reduce(
        &self,
        overrides: &[(&str, f64)],
        input: &str,
        output: &str,
        order: usize,
        fstart: f64,
        fstop: f64,
        points: usize,
    ) -> Result<ReducedModel, ModelError> {
        let (x, p) = self.inner.solve_dc(overrides)?;
        let out_idx = self
            .resolve(output)
            .ok_or_else(|| ModelError::UnknownRef(output.to_string()))?;
        let mut ctx = self.inner.ctx.lock().unwrap();
        let (freqs, full_db, red_db, poles, zeros, max_err_db) = crate::model_reduce_on_dae(
            &mut ctx,
            &self.inner.dae,
            &self.inner.cdc,
            input,
            out_idx,
            &x,
            &p,
            order,
            fstart,
            fstop,
            points,
        )
        .map_err(ModelError::Numeric)?;
        Ok(ReducedModel {
            freqs,
            full_db,
            red_db,
            poles,
            zeros,
            max_err_db,
        })
    }

    /// Sweep a source (or any parameter) linearly over `[start, stop]` in `step`
    /// increments, re-solving the DC operating point at each value. Warm-starts
    /// each point from the previous solution (falling back to a cold solve on
    /// non-convergence) so a nonlinear sweep converges in a few iterations per
    /// point while matching the cold result. Non-converging points are dropped;
    /// the returned [`DcSweep`] carries only the converged sweep values and the
    /// labeled state at each.
    pub fn dc_sweep(
        &self,
        source: &str,
        start: f64,
        stop: f64,
        step: f64,
    ) -> Result<DcSweep, ModelError> {
        if step == 0.0 || (stop - start).signum() != step.signum() {
            return Err(ModelError::Numeric(
                "DC sweep needs a non-zero step in the start->stop direction".into(),
            ));
        }
        let col = self
            .param_col(source)
            .ok_or_else(|| ModelError::UnknownParam(source.to_string()))?;
        let base = self.inner.store.pvec(&[]);
        let npts = (((stop - start) / step).abs().round() as usize).min(100_000) + 1;
        let mut sweep = Vec::new();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        // Warm start from the previous converged point; a warm point that fails
        // retries cold before being dropped, so results match a cold sweep.
        let mut warm: Vec<f64> = Vec::new();
        for k in 0..npts {
            let v = start + step * k as f64;
            let mut p = base.clone();
            p[col] = v;
            let (mut x, mut conv, _) = self.inner.cdc.solve_dc(&p, &warm, DC_OP_TOL, DC_OP_MAXIT);
            if !conv && !warm.is_empty() {
                let (xc, cc, _) = self.inner.cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
                x = xc;
                conv = cc;
            }
            if !conv {
                warm.clear();
                continue;
            }
            warm = x.clone();
            sweep.push(v);
            rows.push(x);
        }
        if sweep.is_empty() {
            return Err(ModelError::Numeric(
                "DC sweep did not converge at any point".into(),
            ));
        }
        Ok(DcSweep {
            sim: self.inner.clone(),
            sweep,
            rows,
        })
    }

    /// Small-signal AC magnitude/phase response of `output` to source `input`,
    /// swept logarithmically over `[fstart, fstop]`, linearised at the OP.
    pub fn ac(
        &self,
        overrides: &[(&str, f64)],
        input: &str,
        output: &str,
        fstart: f64,
        fstop: f64,
        points: usize,
    ) -> Result<AcResponse, ModelError> {
        let (x, p) = self.inner.solve_dc(overrides)?;
        let out_idx = self
            .resolve(output)
            .ok_or_else(|| ModelError::UnknownRef(output.to_string()))?;
        let mut ctx = self.inner.ctx.lock().unwrap();
        let (freqs, mag_db, phase_deg) = crate::ac_on_dae(
            &mut ctx,
            &self.inner.dae,
            &self.inner.cdc,
            &self.inner.store.pnames,
            input,
            out_idx,
            &x,
            &p,
            fstart,
            fstop,
            points,
        )
        .map_err(ModelError::Numeric)?;
        Ok(AcResponse {
            freqs,
            mag_db,
            phase_deg,
            input: input.to_string(),
            output: output.to_string(),
        })
    }

    /// Scattering parameters over the deck's `P` power ports (deck order).
    ///
    /// Each port is the Thevenin form the `P` element lowers to: an ideal
    /// drive source behind its `z0` series resistor, port node on the network
    /// side. Then `S_ij = 2*sqrt(z0_j/z0_i) * V_i - delta_ij` under unit drive
    /// of source j, so each column is one complex AC sweep with every other
    /// port source dead. Returns the sweep as `(freqs, s, port_names, z0)`
    /// with `s[k]` the flattened row-major `n x n` matrix at `freqs[k]`.
    pub fn sp_sweep(
        &self,
        overrides: &[(&str, f64)],
        freqs_hz: Vec<f64>,
    ) -> Result<SpSweep, ModelError> {
        let ports = self.inner.ports.clone();
        if ports.is_empty() {
            return Err(ModelError::Numeric(
                "sp: the deck has no P port elements (e.g. `P1 in 0 Z0=50`)".into(),
            ));
        }
        let (x, p) = self.inner.solve_dc(overrides)?;
        let mut spec = Vec::with_capacity(ports.len());
        for (src, node, z0) in &ports {
            let idx = self
                .resolve(node)
                .ok_or_else(|| ModelError::UnknownRef(node.clone()))?;
            spec.push((src.clone(), idx, *z0));
        }
        let s = self.sp_response(&spec, x, p, freqs_hz.clone())?;
        Ok(SpSweep {
            freqs: freqs_hz,
            s,
            port_names: ports.iter().map(|p| p.0.clone()).collect(),
            z0: ports.iter().map(|p| p.2).collect(),
        })
    }

    /// Poles (natural frequencies of the small-signal pencil `G + sC`) and, if
    /// `input`/`output` are given, the transmission zeros from input to output.
    pub fn poles_zeros(
        &self,
        overrides: &[(&str, f64)],
        input: &str,
        output: &str,
    ) -> Result<PoleZero, ModelError> {
        if self.cdc().has_delays() {
            return Err(ModelError::Numeric(
                "poles_zeros: transport delays (tline/absdelay) are not supported yet".into(),
            ));
        }
        let mut task = sane_core::log::task("PZ", "pz", &format!("(dim: {})", self.dim()));
        let (x, p) = self.inner.solve_dc(overrides)?;
        let n = self.dim();
        let xdot0 = vec![0.0; n];
        let mut ctx = self.inner.ctx.lock().unwrap();
        let g = self.inner.cdc.system_matrix_dc(&x, &xdot0, &p, 0.0);
        let c = self.inner.cdc.jacobian_xdot(&x, &xdot0, &p, 0.0);
        let poles = crate::finite_pencil_roots(&g, &c).map_err(ModelError::Numeric)?;
        let mut zeros = Vec::new();
        if !input.trim().is_empty() && !output.trim().is_empty() {
            if let (Some(out_idx), Some(b_real)) = (
                resolve_ref(&self.inner.unknowns, &self.inner.node_names, output),
                crate::input_vector(
                    &mut ctx,
                    &self.inner.dae,
                    &self.inner.store.pnames,
                    &p,
                    &x,
                    input,
                ),
            ) {
                let mut m = vec![vec![0.0; n + 1]; n + 1];
                let mut nn = vec![vec![0.0; n + 1]; n + 1];
                for i in 0..n {
                    for j in 0..n {
                        m[i][j] = g[i][j];
                        nn[i][j] = c[i][j];
                    }
                    m[i][n] = b_real[i];
                    m[n][i] = if i == out_idx { 1.0 } else { 0.0 };
                }
                if let Ok(zs) = crate::finite_pencil_roots(&m, &nn) {
                    zeros = zs;
                }
            }
        }
        task.finish(format!("poles: {}, zeros: {}", poles.len(), zeros.len()));
        Ok(PoleZero { poles, zeros })
    }

    /// Single-tone harmonic balance at fundamental `f0` (Hz) with `harmonics`
    /// harmonics (AFT). The periodic drive comes from a `SIN` source in the deck;
    /// passing `f0 <= 0` infers the fundamental from that source.
    pub fn harmonic_balance(
        &self,
        overrides: &[(&str, f64)],
        f0: f64,
        harmonics: usize,
        x0: Option<&[f64]>,
    ) -> Result<HarmonicBalance, ModelError> {
        if self.inner.cdc.has_delays() {
            return Err(ModelError::Numeric(
                "harmonic_balance: transport delays (tline/absdelay) are not supported yet".into(),
            ));
        }
        // A caller-supplied operating point (e.g. a node-set basin for a
        // multi-solution circuit such as an auto-zeroing chopper) seeds the HB DC
        // and its first Newton iterate, instead of a fresh cold DC solve that may
        // land in the wrong basin.
        let (x_dc, p) = match x0 {
            Some(v) if v.len() == self.inner.dae.dim() => {
                (v.to_vec(), self.inner.store.pvec(overrides))
            }
            _ => self.inner.solve_dc(overrides)?,
        };
        // Default the fundamental to the circuit's own periodic source (a SIN's
        // frequency, a PULSE train's 1/period) when the caller does not pin one.
        let f0 = if f0 > 0.0 {
            f0
        } else {
            self.inner.cdc.source_fundamental(&p).ok_or_else(|| {
                ModelError::Numeric(
                    "harmonic balance needs f0 > 0 (no periodic source to infer it from)".into(),
                )
            })?
        };
        let mut task = sane_core::log::task(
            "HB",
            "hb",
            &format!(
                "(f0: {f0:.4e}, harmonics: {harmonics}, dim: {})",
                self.dim()
            ),
        );
        let ctx = self.inner.ctx.lock().unwrap();
        let nl = self.inner.dae.nonlinearity(&ctx);
        let m = sane_solve::hb::hb_samples(&nl, harmonics, 16);
        let hb = log_stage!(
            "hb/setup",
            sane_solve::hb::CompiledHb::new(&self.inner.cdc, harmonics, m).ok_or_else(|| {
                ModelError::Numeric("harmonic balance setup failed (need samples >= 2*K)".into())
            })
        )?;
        let res = log_stage!(
            "hb/solve",
            hb.solve(&p, &x_dc, 2.0 * std::f64::consts::PI * f0, 1e-10, 60)
        );
        // Per-unknown magnitude/phase spectra (index 0 = DC, 1 = fundamental, ...).
        let mag: Vec<Vec<f64>> = res
            .spectra
            .iter()
            .map(|r| r.iter().map(|c| c.norm()).collect())
            .collect();
        let phase: Vec<Vec<f64>> = res
            .spectra
            .iter()
            .map(|r| r.iter().map(|c| c.arg().to_degrees()).collect())
            .collect();
        task.finish(format!(
            "converged: {}, iters: {}, residual: {:.2e}",
            res.converged, res.iters, res.residual_norm
        ));
        Ok(HarmonicBalance {
            sim: self.inner.clone(),
            mag,
            phase,
            converged: res.converged,
        })
    }

    /// Column index of a parameter in the engine's order, or `None`.
    fn param_col(&self, name: &str) -> Option<usize> {
        let canon = self.inner.store.resolve(name)?;
        self.inner.store.pnames.iter().position(|n| *n == canon)
    }

    // --- graph transforms (return a new Model sharing this context) --------

    /// Operating-point-guided graph reduction at `(x, p)`: drop branch
    /// contributions negligible (conductance/capacitance) at the given angular
    /// frequencies. Returns a reduced `Model` sharing this context and the pruned
    /// `(branch, node)` pairs.
    pub fn prune_graph(
        &self,
        rel_tol: f64,
        x: &[f64],
        p: &[f64],
        omegas: &[f64],
    ) -> (Model, Vec<(String, String)>) {
        let mut c = self.inner.ctx.lock().unwrap();
        let pnames = self.inner.cdc.param_names(&c);
        let mut p_pairs = Vec::with_capacity(pnames.len());
        for (j, name) in pnames.iter().enumerate() {
            let e = c.sym(name);
            if let Node::Symbol(s) = c.node(e) {
                p_pairs.push((*s, p.get(j).copied().unwrap_or(0.0)));
            }
        }
        let (reduced, pruned) = reduce_graph(&mut c, &self.inner.dae, x, &p_pairs, omegas, rel_tol);
        let cdc = CompiledDc::new(&mut c, &reduced);
        drop(c);
        let model = Model::from_parts(
            self.inner.ctx.clone(),
            reduced,
            cdc,
            self.values(),
            self.inner.node_names.clone(),
            // a transformed DAE has no element graph of its own
            None,
            &[],
        );
        (model, pruned)
    }

    /// Exactly eliminate internal resistive nodes (Schur/series reduction).
    /// `keep` protects node-unknown names. Returns the reduced `Model` and the
    /// eliminated node names, in order.
    pub fn eliminate_nodes(&self, keep: &[String]) -> (Model, Vec<String>) {
        let keep_set: HashSet<String> = keep.iter().cloned().collect();
        let mut c = self.inner.ctx.lock().unwrap();
        let (reduced, gone) = dae_eliminate_nodes(&mut c, &self.inner.dae, &keep_set);
        let cdc = CompiledDc::new(&mut c, &reduced);
        drop(c);
        let model = Model::from_parts(
            self.inner.ctx.clone(),
            reduced,
            cdc,
            self.values(),
            self.inner.node_names.clone(),
            // a transformed DAE has no element graph of its own
            None,
            &[],
        );
        (model, gone)
    }

    /// Linearise about the operating point into the small-signal mass-matrix DAE
    /// `G dx + C dx' = 0`, sharing this context. `canonical` emits a single
    /// canonical small-signal element per stamp. Returns the linearised `Model`.
    pub fn linearize(&self, canonical: bool) -> Model {
        let gran = if canonical {
            Granularity::Canonical
        } else {
            Granularity::PerElement
        };
        let mut c = self.inner.ctx.lock().unwrap();
        let lin = linearize_with(&mut c, &self.inner.dae, gran);
        let cdc = CompiledDc::new(&mut c, &lin);
        drop(c);
        Model::from_parts(
            self.inner.ctx.clone(),
            lin,
            cdc,
            self.values(),
            self.inner.node_names.clone(),
            None,
            &[],
        )
    }

    /// Fold a set of parameters to their current values: each becomes a constant
    /// in a derived `Model` (sharing this context). Its now-constant subexpressions
    /// collapse (smaller graph / faster eval) and it leaves the parameter set
    /// (`params()` shrinks -> no `dF/dp` column, no sensitivity). A `path` is either
    /// a single parameter (`X1.R1`, `N1.vth0`) or a group prefix (`X1` -> all
    /// `X1.*`, recursively). Same transform family as `linearize` /
    /// `eliminate_nodes`; the master `Model` is unchanged.
    pub fn fold(&self, paths: &[&str]) -> Result<Model, ModelError> {
        // Resolve each path to canonical parameter names: a leaf parameter, or
        // every parameter under a group prefix.
        let mut names: HashSet<String> = HashSet::new();
        for &path in paths {
            if let Some(canon) = self.inner.store.resolve(path) {
                names.insert(canon);
            } else if self.is_group(path) {
                let pre = format!("{path}.");
                for n in &self.inner.store.pnames {
                    if n.starts_with(&pre) {
                        names.insert(n.clone());
                    }
                }
            } else {
                return Err(ModelError::UnknownParam(path.to_string()));
            }
        }

        let mut c = self.inner.ctx.lock().unwrap();
        // (param symbol, frozen current value) pairs for the substitution.
        let mut fold: Vec<(SymbolId, f64)> = Vec::with_capacity(names.len());
        for name in &names {
            let val = self.inner.store.get(name).unwrap_or(0.0);
            let e = c.sym(name);
            if let Node::Symbol(s) = c.node(e) {
                fold.push((*s, val));
            }
        }
        let folded = self.inner.dae.fold_params(&mut c, &fold);
        let cdc = CompiledDc::new(&mut c, &folded);
        drop(c);

        // The folded parameters are constants now: drop them from the value store.
        let mut values = self.values();
        for name in &names {
            values.remove(name);
        }
        Ok(Model::from_parts(
            self.inner.ctx.clone(),
            folded,
            cdc,
            values,
            self.inner.node_names.clone(),
            // folding parameters keeps the topology, but not the graph object
            None,
            &[],
        ))
    }
}

/// Small-signal AC response: magnitude (dB) and phase (deg) over frequency.
/// An S-parameter sweep over the deck's `P` ports: `s[k]` is the flattened
/// row-major `n x n` scattering matrix at `freqs[k]`, entries as `(re, im)`.
pub struct SpSweep {
    pub freqs: Vec<f64>,
    pub s: Vec<Vec<(f64, f64)>>,
    pub port_names: Vec<String>,
    pub z0: Vec<f64>,
}

pub struct AcResponse {
    pub freqs: Vec<f64>,
    pub mag_db: Vec<f64>,
    pub phase_deg: Vec<f64>,
    pub input: String,
    pub output: String,
}

/// Poles and (optionally) transmission zeros, each `[re, im]` in rad/s.
pub struct PoleZero {
    pub poles: Vec<[f64; 2]>,
    pub zeros: Vec<[f64; 2]>,
}

/// Periodic steady-state spectra (per unknown, index 0 = DC, 1 = fundamental).
pub struct HarmonicBalance {
    sim: Arc<ModelInner>,
    mag: Vec<Vec<f64>>,
    phase: Vec<Vec<f64>>,
    pub converged: bool,
}

impl HarmonicBalance {
    /// Magnitude spectrum of a node / unknown / branch-current reference.
    pub fn magnitude(&self, reference: &str) -> Option<&[f64]> {
        let i = resolve_ref(&self.sim.unknowns, &self.sim.node_names, reference)?;
        self.mag.get(i).map(|v| v.as_slice())
    }
    /// Phase spectrum (degrees) of a reference.
    pub fn phase(&self, reference: &str) -> Option<&[f64]> {
        let i = resolve_ref(&self.sim.unknowns, &self.sim.node_names, reference)?;
        self.phase.get(i).map(|v| v.as_slice())
    }
}

/// Descriptor state-space realisation `E x' = A x + B u`, `y = C x + D u`.
pub struct StateSpace {
    pub e: Vec<Vec<f64>>,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: f64,
    pub input: String,
    pub output: String,
}

/// Output value versus temperature (deg C).
pub struct TempSweep {
    pub temps: Vec<f64>,
    pub values: Vec<f64>,
    pub output: String,
}

/// Reduced-order model: magnitude (dB) of full vs reduced over frequency, plus
/// the kept poles/zeros `[re, im]` and the worst-case fit error (dB).
pub struct ReducedModel {
    pub freqs: Vec<f64>,
    pub full_db: Vec<f64>,
    pub red_db: Vec<f64>,
    pub poles: Vec<[f64; 2]>,
    pub zeros: Vec<[f64; 2]>,
    pub max_err_db: f64,
}

impl ModelInner {
    /// Build `p` from the store + overrides and solve the DC operating point,
    /// reusing the cached operating point when the parameter vector is unchanged.
    fn solve_dc(&self, overrides: &[(&str, f64)]) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
        let (x, p, _reg) = self.solve_dc_reg(overrides)?;
        Ok((x, p))
    }

    /// As [`solve_dc`](Self::solve_dc), but also returns the gmin-regularization
    /// flag for the operating point: `Some(g)` when it converged only under a
    /// raised shunt (a physically suspect, regularized solution), else `None`.
    /// Cached alongside the point so a cache hit reports the same status (#54).
    fn solve_dc_reg(
        &self,
        overrides: &[(&str, f64)],
    ) -> Result<(Vec<f64>, Vec<f64>, Option<f64>), ModelError> {
        let p = self.store.pvec(overrides);
        if let Some((cp, cx, creg)) = self.op_cache.lock().unwrap().as_ref() {
            if cp == &p {
                return Ok((cx.clone(), p, *creg));
            }
        }
        // Cold solve on a miss (identical to no-cache behaviour); cache the result.
        let (mut x, conv, _it) = self.cdc.solve_dc_conv_with(
            &p,
            &[],
            Convergence::default(),
            100,
            SolverTricks::default(),
        );
        if !conv {
            return Err(ModelError::NoConverge);
        }
        let mut reg = self.cdc.last_regularized_gmin();
        if reg.is_some() {
            // Settle fallback: a cold-start Newton on a high-gain feedback loop
            // can converge onto a gmin-held rail solution while the true floor
            // point lives in another basin (seen on the LM741 internals). The
            // real dynamics escape that basin through the circuit's own
            // capacitances -- integrate briefly from the regularized point and
            // re-solve Newton from the settled state; adopt the result only if
            // it reaches the true DC floor.
            let _g = sane_core::log::scope("dc/settle");
            sane_core::log::info(
                "dc: operating point is gmin-held; settling the dynamics (short internal transient) and re-solving",
            );
            // empty x0: let the integrator build its own consistent start (the
            // raw regularized vector is not differential-consistent and makes
            // the first step underflow)
            if let Ok(rows) = self.cdc.solve_transient(
                sane_solve::TransientMethod::default(),
                &p,
                &[],
                &[0.0, 0.06],
                1e-4,
                1e-7,
                None,
            ) {
                if let Some(xs) = rows.last() {
                    let (x2, conv2, _) = self.cdc.solve_dc_conv_with(
                        &p,
                        xs,
                        Convergence::default(),
                        100,
                        SolverTricks::default(),
                    );
                    if conv2 && self.cdc.last_regularized_gmin().is_none() {
                        x = x2;
                        reg = None;
                    }
                }
            }
        }
        // Two ways a point can depend on the regularization, one question for
        // the caller. The settle fallback above keys off the hold alone -- it
        // exists to escape a basin, and dominance is not one -- but what gets
        // reported covers both.
        let reg = reg.or_else(|| {
            self.cdc
                .last_gmin_dominance()
                .map(|_| sane_core::constants::GMIN_DC)
        });
        *self.op_cache.lock().unwrap() = Some((p.clone(), x.clone(), reg));
        Ok((x, p, reg))
    }
}

/// A solved DC operating point. Holds the state and parameters in Rust, so node
/// values and derived sensitivities are read off it without a re-solve.
pub struct OperatingPoint {
    sim: Arc<ModelInner>,
    x: Vec<f64>,
    p: Vec<f64>,
    /// `Some(g)` if this operating point is gmin-regularized (held only at shunt
    /// `g`, never reaching the `GMIN_DC` floor) -- converged=true but physically
    /// suspect; `None` for a true DC solution (issue #54).
    regularized_at_gmin: Option<f64>,
}

impl OperatingPoint {
    /// The raw state vector in unknown/column order.
    pub fn vector(&self) -> &[f64] {
        &self.x
    }
    /// The gmin at which this operating point held, if it is a gmin-regularized
    /// (physically suspect) solution rather than a true DC floor point (#54).
    pub fn regularized_at_gmin(&self) -> Option<f64> {
        self.regularized_at_gmin
    }
    /// The value at a node / unknown / branch-current reference.
    pub fn get(&self, reference: &str) -> Option<f64> {
        resolve_ref(&self.sim.unknowns, &self.sim.node_names, reference).map(|i| self.x[i])
    }
    /// The operating point as a `{unknown: value}` map (built on demand).
    pub fn to_map(&self) -> HashMap<String, f64> {
        self.sim
            .unknowns
            .iter()
            .cloned()
            .zip(self.x.iter().copied())
            .collect()
    }

    /// First-order sensitivity `dy/dp` of `output` over every parameter, by one
    /// adjoint solve at this point (no re-solve).
    pub fn sensitivity(&self, output: &str) -> Result<Sensitivity, ModelError> {
        let metric = resolve_ref(&self.sim.unknowns, &self.sim.node_names, output)
            .ok_or_else(|| ModelError::UnknownRef(output.to_string()))?;
        let mut ctx = self.sim.ctx.lock().unwrap();
        self.sim.cdc.ensure_param_jac(&mut ctx, &self.sim.dae);
        let grad = self.sim.cdc.sensitivity(metric, &self.x, &self.p, 0.0);
        let names = self.sim.cdc.param_names(&ctx);
        Ok(Sensitivity {
            names,
            grad,
            output: output.to_string(),
            value: self.x[metric],
        })
    }

    /// Every operating-point variable the circuit's behavioral devices export
    /// (`(* desc *)`-annotated Verilog-A variables: gm, vth, ids, ...), evaluated
    /// at this solved point (xdot = 0). Empty when no device declares any.
    pub fn op_vars(&self) -> Vec<OpVarValue> {
        let dae = &self.sim.dae;
        if dae.op_vars.is_empty() {
            return Vec::new();
        }
        let mut ctx = self.sim.ctx.lock().unwrap();
        let pnames = self.sim.cdc.param_names(&ctx);
        let env = crate::op_env(&mut ctx, dae, &pnames, &self.x, &[], &self.p, 0.0);
        // One arena sweep: an op-var is a call into its device's template
        // function (evaluated once per instance) or a plain expression.
        let roots: Vec<_> = dae.op_vars.iter().map(|v| v.value).collect();
        let vals = rsdag::eval(&ctx, &roots, &env);
        dae.op_vars
            .iter()
            .zip(vals)
            .map(|(v, value)| OpVarValue {
                name: v.name.clone(),
                desc: v.desc.clone(),
                units: v.units.clone(),
                value,
            })
            .collect()
    }

    /// Sparse second-order sensitivity (Hessian) of `output` w.r.t. the parameter
    /// subset `wrt`, by the second-order adjoint at this point (no re-solve).
    /// Returns the dense symmetric `len(wrt) x len(wrt)` matrix.
    pub fn hessian(&self, output: &str, wrt: &[&str]) -> Result<Vec<Vec<f64>>, ModelError> {
        let metric = resolve_ref(&self.sim.unknowns, &self.sim.node_names, output)
            .ok_or_else(|| ModelError::UnknownRef(output.to_string()))?;
        let mut ctx = self.sim.ctx.lock().unwrap();
        let pnames = self.sim.cdc.param_names(&ctx);
        let mut cols = Vec::with_capacity(wrt.len());
        for name in wrt {
            let canon = self
                .sim
                .store
                .resolve(name)
                .ok_or_else(|| ModelError::UnknownParam(name.to_string()))?;
            let col = pnames
                .iter()
                .position(|n| *n == canon)
                .ok_or_else(|| ModelError::UnknownParam(name.to_string()))?;
            cols.push(col);
        }
        self.sim.cdc.ensure_hessian(&mut ctx, &self.sim.dae);
        let h = self.sim.cdc.hessian(metric, &cols, &self.x, &self.p, 0.0);
        if h.is_empty() {
            return Err(ModelError::NoConverge);
        }
        Ok(h)
    }
}

/// An operating-point variable evaluated at a solved point (see
/// [`OperatingPoint::op_vars`]).
#[derive(Clone, Debug)]
pub struct OpVarValue {
    /// Instance-qualified name (`M1.gm`).
    pub name: String,
    pub desc: String,
    pub units: Option<String>,
    pub value: f64,
}

/// A transient solution: state versus time, with labeled access (the ergonomic
/// Rust return for [`Model::transient`]; the Python layer labels rows itself).
pub struct Trajectory {
    sim: Arc<ModelInner>,
    pub t: Vec<f64>,
    /// `rows[k]` is the state at `t[k]`, in unknown/column order.
    rows: Vec<Vec<f64>>,
}

impl Trajectory {
    /// The time series of a node / unknown / branch-current reference.
    pub fn signal(&self, reference: &str) -> Option<Vec<f64>> {
        let i = resolve_ref(&self.sim.unknowns, &self.sim.node_names, reference)?;
        Some(
            self.rows
                .iter()
                .map(|r| r.get(i).copied().unwrap_or(0.0))
                .collect(),
        )
    }
    /// All rows (state per time point).
    pub fn rows(&self) -> &[Vec<f64>] {
        &self.rows
    }
}

/// A DC sweep solution: the converged sweep values and the state at each, with
/// labeled access (the ergonomic Rust return for [`Model::dc_sweep`]).
pub struct DcSweep {
    sim: Arc<ModelInner>,
    /// The source values that converged, in sweep order.
    pub sweep: Vec<f64>,
    /// `rows[k]` is the operating point at `sweep[k]`, in unknown/column order.
    rows: Vec<Vec<f64>>,
}

impl DcSweep {
    /// The swept series of a node / unknown / branch-current reference.
    pub fn signal(&self, reference: &str) -> Option<Vec<f64>> {
        let i = resolve_ref(&self.sim.unknowns, &self.sim.node_names, reference)?;
        Some(
            self.rows
                .iter()
                .map(|r| r.get(i).copied().unwrap_or(0.0))
                .collect(),
        )
    }
    /// All rows (state per swept point).
    pub fn rows(&self) -> &[Vec<f64>] {
        &self.rows
    }
}

/// Output-referred noise spectrum: PSD over frequency.
pub struct NoiseSpectrum {
    pub freqs: Vec<f64>,
    pub psd: Vec<f64>,
}

/// First-order sensitivity result: `grad[i] = d(output)/d(names[i])`.
pub struct Sensitivity {
    pub names: Vec<String>,
    pub grad: Vec<f64>,
    pub output: String,
    pub value: f64,
}

// Analysis methods by domain, one module each; shared guards live below.
mod ac;
mod hb;
mod noise;
mod num;
mod pz;
mod tran;

impl Model {
    /// Reject analyses that have no transport-delay support yet: silent
    /// mis-results (a missing e^{-j w tau}, a delay-blind adjoint) are worse
    /// than an error.
    fn ensure_no_delays(&self, what: &str) -> Result<(), ModelError> {
        if self.cdc().has_delays() {
            return Err(ModelError::Numeric(format!(
                "{what}: transport delays (tline/absdelay) are not supported in this analysis yet"
            )));
        }
        Ok(())
    }

    /// Compile the history Jacobian if the circuit has delays (idempotent).
    fn ensure_hist_jac_ready(&self) {
        if !self.cdc().has_delays() {
            return;
        }
        let arc = self.context_arc();
        self.cdc()
            .ensure_hist_jac(&mut arc.lock().unwrap(), self.dae());
    }

    /// Frequency-domain delay coupling as `(rows, cols, values, taus)`:
    /// `∂F/∂hist_k` moved onto the delayed SOURCE unknown, to be scaled by
    /// `e^{-jωτ_k}` per frequency.
    fn delay_ac_entries(
        &self,
        x: &[f64],
        p: &[f64],
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
        if !self.cdc().has_delays() {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }
        let taus = self.cdc().delay_taus(p);
        let src = self.cdc().delay_sources();
        let (rows, dcols, vals) = self.cdc().hist_jac_sparse(x, p);
        let cols: Vec<usize> = dcols.iter().map(|&k| src[k]).collect();
        let tau: Vec<f64> = dcols.iter().map(|&k| taus[k]).collect();
        (rows, cols, vals, tau)
    }
}
