//! `Model` -- the embeddable, Rust-first analysis object.
//!
//! A `Model` owns a prepared circuit (symbolic context, DAE, compiled solver) plus
//! a named parameter store. [`Model::at`] binds its parameters to a [`Point`],
//! and every analysis is a method of the point that returns a result *handle*.
//! The handle keeps the solved state in Rust, so derived analyses
//! (`op.sensitivity(..)`) run without a re-solve and nothing crosses a language
//! boundary. This is the one orchestration layer; the Python bindings and the
//! netlist front end are thin wrappers over it.
//!
//! ```no_run
//! use sane_analysis::Model;
//! let sim = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").unwrap();
//! let op = sim.at(&[]).unwrap().operating_point().unwrap();
//! assert!((op.get("out").unwrap() - 2.5).abs() < 1e-6);
//! sim.set("R2", 3e3).unwrap();                              // mutate a named parameter
//! let op = sim.at(&[]).unwrap().operating_point().unwrap(); // the binding picks it up
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;

use ndarray::{Array2, Array3, ArrayView1};
use sane_circuit::Circuit;
use sane_core::constants::DC_OP_MAXIT;
use sane_core::log_stage;
use sane_core::Graph;
use sane_solve::{CompiledDc, Convergence, SolverTricks};

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
    /// The parameters leave the model invalid: an assertion of a device does
    /// not hold (a loop unrolled only so far, an `$error` on these values).
    Invalid(String),
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
            ModelError::Invalid(m) => write!(f, "invalid parameters: {m}"),
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
        let r = sane_circuit::value_symbol_name(name);
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
    /// the requested resolution. See `sane_circuit::index2`.
    index2: sane_circuit::index2::Index2Report,
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
    /// How its operating points are solved (see [`Model::dc_options`]).
    dc: Mutex<DcOptions>,
    /// The devices' assertions (see [`sane_dae::Dae::assertions`]) compiled
    /// over the parameter vector, with their messages; `None` without any.
    assertions: Option<(rsdag::Tape, Vec<String>)>,
    /// What the devices' structure rests on (see
    /// [`sane_dae::Dae::structure`]), likewise.
    structure: Option<(rsdag::Tape, Vec<String>)>,
    /// For a model built from a netlist, what it sets its other structures
    /// up from (see [`restructure`]).
    restructure: Option<restructure::Restructure>,
    /// The circuit the model was set up from, or derives from.
    circuit: Option<Arc<Circuit>>,
    /// The transforms the model was derived by from the circuit's (see
    /// [`transform`]); empty for the circuit's own.
    lineage: Vec<transform::Transform>,
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
    elements: &[sane_circuit::Element],
    terminals: &[Vec<usize>],
) -> sane_circuit::index2::Index2Report {
    let rep = sane_circuit::index2::detect(elements, terminals);
    if rep.is_index2() {
        sane_core::log::warn_captured(&format!(
            "index-2 topology ({}): its constraint carries no truncation error, so the adaptive transient step may stride past the resolution you asked for -- set dt_max if the trace looks coarse",
            rep.summary()
        ));
    }
    rep
}

/// `list` compiled over `cdc`'s parameter vector, with the messages;
/// `None` for an empty one.
fn assertion_tape(
    ctx: &Graph,
    list: &[sane_dae::Assertion],
    cdc: &CompiledDc,
) -> Option<(rsdag::Tape, Vec<String>)> {
    if list.is_empty() {
        return None;
    }
    let holds: Vec<rsdag::ExprId> = list.iter().map(|a| a.holds).collect();
    let tape = rsdag::Tape::compile(ctx, &holds, cdc.param_syms());
    let messages = list.iter().map(|a| a.message.clone()).collect();
    Some((tape, messages))
}

#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

impl Model {
    /// Build a `Model` from a SPICE-like netlist string: the parsed
    /// circuit's model (see [`new`](Self::new)).
    pub fn from_netlist(src: &str) -> Result<Model, ModelError> {
        let circuit = log_stage!("parse", sane_netlist::parse(src))
            .map_err(|e| ModelError::Parse(format!("parse error: {e}")))?;
        Model::new(circuit)
    }

    /// The model of a circuit, its devices' structure at the circuit's own
    /// values. The model keeps the circuit: a binding across a device's
    /// structure decision sets the circuit up anew at that binding (see
    /// [`restructure`]), and [`circuit`](Self::circuit) hands it back to
    /// build on.
    pub fn new(circuit: impl Into<Arc<Circuit>>) -> Result<Model, ModelError> {
        Model::set_up(circuit.into(), &|_| None, true)
    }

    /// The circuit this model was set up from; a transform's (see
    /// [`fold`](Self::fold), [`linearize`](Self::linearize), ...) the one
    /// it derives from.
    pub fn circuit(&self) -> Option<&Arc<Circuit>> {
        self.inner.circuit.as_ref()
    }

    /// The model of `circuit` with its devices' structure decided at
    /// `values` (by parameter name) where those set a parameter; a `root`
    /// sets its other structures up from the circuit (see [`restructure`]).
    pub(super) fn set_up(
        circuit: Arc<Circuit>,
        values: &dyn Fn(&str) -> Option<f64>,
        root: bool,
    ) -> Result<Model, ModelError> {
        let mut task = sane_core::log::task("EXTRACT", "extract", "");
        let Prepared { ctx, dae, cdc } = prepare(&circuit, values).map_err(ModelError::Parse)?;
        let pnames = cdc.param_names(&ctx);
        task.finish(format!("dim: {}, params: {}", dae.dim(), pnames.len()));
        let (elements, terminals) = sane_circuit::topology(&circuit);
        let index2 = detect_index2(&elements, &terminals);
        let store = ParamStore::new(
            pnames,
            circuit
                .values
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            column_defaults(&cdc, &dae),
        );
        let unknowns = dae.unknowns.clone();
        let node_names = circuit.node_names();
        let ports = (circuit.ports.iter())
            .map(|pt| (pt.name.clone(), pt.node.clone(), pt.z0))
            .collect();
        let assertions = assertion_tape(&ctx, &dae.assertions, &cdc);
        let structure = assertion_tape(&ctx, &dae.structure, &cdc);
        let restructure = (root && structure.is_some()).then(restructure::Restructure::default);
        let dc = Mutex::new(DcOptions::from_circuit(&circuit.dc));
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
                assertions,
                structure,
                restructure,
                circuit: Some(circuit),
                lineage: Vec::new(),
                ac_vjp_tapes: Mutex::new(HashMap::new()),
                dc,
            }),
        })
    }

    /// How the operating points of this model are solved: its deck's
    /// `.option reltol/abstol/vntol/itl1` over the defaults, unless set.
    pub fn dc_options(&self) -> DcOptions {
        *self.inner.dc.lock().unwrap()
    }

    /// Solve the operating points of the points taken from now on as `opts`
    /// says (a point keeps the options it was taken with).
    pub fn set_dc_options(&self, opts: DcOptions) {
        *self.inner.dc.lock().unwrap() = opts;
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

    // --- parameter store ---

    /// Read a parameter's bound value (or `0.0` if unset); `None` if not a parameter.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.inner.store.get(name)
    }
    /// Set a parameter; errors if `name` is not a parameter or the value
    /// fails one of the devices' assertions, which leaves it as it was. A
    /// value that changes the circuit's structure is a binding like any:
    /// the analyses run on the circuit of that structure (see
    /// [`restructure`]).
    pub fn set(&self, name: &str, value: f64) -> Result<(), ModelError> {
        self.set_many(&[(name, value)])
    }

    /// Set several parameters at once: the binding they make is checked
    /// once, as a whole (see [`set`](Self::set)), so values valid only
    /// together set fine; on an error none is set.
    pub fn set_many(&self, values: &[(&str, f64)]) -> Result<(), ModelError> {
        if self.inner.assertions.is_some() {
            self.inner.valued_pvec(values)?;
        }
        for (name, _) in values {
            if self.inner.store.resolve(name).is_none() {
                return Err(ModelError::UnknownParam(name.to_string()));
            }
        }
        for &(name, value) in values {
            self.inner.store.set(name, value)?;
        }
        Ok(())
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
    /// Direct children (next path segment) of a parameter group (`""` for
    /// the top level), sorted.
    pub fn children(&self, prefix: &str) -> Vec<String> {
        let pre = match prefix {
            "" => String::new(),
            p => format!("{p}."),
        };
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
    /// Resolve a node / unknown / branch-current reference to its state
    /// index. A node the structure collapsed has none (a result of another
    /// structure may hold it apart): its value is the result's, by name.
    pub fn resolve(&self, reference: &str) -> Option<usize> {
        resolve_ref(&self.inner.unknowns, &self.inner.node_names, reference)
    }

    // --- analyses ---
}

impl ModelInner {
    /// The symbols the source `input` drives the small-signal response
    /// through (see [`crate::drive_params`]).
    pub(crate) fn drive(
        &self,
        c: &mut Graph,
        input: &str,
    ) -> Result<Vec<rsdag::SymbolId>, ModelError> {
        let own = self.store.resolve(input);
        let names = match &own {
            Some(own) => std::slice::from_ref(own),
            None => &self.store.pnames[..],
        };
        crate::drive_syms(c, names, input).ok_or_else(|| {
            ModelError::Invalid(format!(
                "'{input}' drives nothing: neither its value nor its waveform's level is a parameter of the model (a folded source drives nothing)"
            ))
        })
    }

    /// The column of the parameter `name` (or its reserved-namespace form).
    pub(crate) fn column(&self, name: &str) -> Option<usize> {
        let canon = self.store.resolve(name)?;
        self.store.pnames.iter().position(|n| *n == canon)
    }

    /// The canonical parameter names under `paths`: a leaf parameter, or
    /// every parameter under a group prefix.
    pub(crate) fn paths(&self, paths: &[&str]) -> Result<HashSet<String>, ModelError> {
        let mut names: HashSet<String> = HashSet::new();
        for &path in paths {
            if let Some(canon) = self.store.resolve(path) {
                names.insert(canon);
            } else if self.store.prefixes.contains(path) {
                let pre = format!("{path}.");
                for n in &self.store.pnames {
                    if n.starts_with(&pre) {
                        names.insert(n.clone());
                    }
                }
            } else {
                return Err(ModelError::UnknownParam(path.to_string()));
            }
        }
        Ok(names)
    }

    /// The columns derivatives are taken by: the parameters under `wrt`
    /// (names or groups, see [`paths`](Self::paths)), all of them for none,
    /// in column order.
    pub(crate) fn columns(&self, wrt: &[&str]) -> Result<Vec<usize>, ModelError> {
        if wrt.is_empty() {
            return Ok((0..self.store.pnames.len()).collect());
        }
        let names = self.paths(wrt)?;
        Ok((0..self.store.pnames.len())
            .filter(|&k| names.contains(&self.store.pnames[k]))
            .collect())
    }

    /// The outputs `outputs` (nodes, unknowns, branch currents) as indices.
    pub(crate) fn outputs(&self, outputs: &[&str]) -> Result<Vec<usize>, ModelError> {
        (outputs.iter())
            .map(|o| {
                self.resolve(o)
                    .ok_or_else(|| ModelError::UnknownRef(o.to_string()))
            })
            .collect()
    }

    /// The parameter vector of `overrides` over the bound values, checked
    /// against the devices' assertions on values only: a binding of another
    /// structure passes.
    fn valued_pvec(&self, overrides: &[(&str, f64)]) -> Result<Vec<f64>, ModelError> {
        let p = self.store.pvec(overrides);
        self.values_hold(&p)?;
        Ok(p)
    }

    /// Whether the devices' assertions on values hold at `p`.
    fn values_hold(&self, p: &[f64]) -> Result<(), ModelError> {
        if let Some((tape, messages)) = &self.assertions {
            let (mut w, mut out) = (Vec::new(), Vec::new());
            tape.eval(p, &mut w, &mut out);
            if let Some(k) = out.iter().position(|&h| h == 0.0) {
                return Err(ModelError::Invalid(messages[k].clone()));
            }
        }
        Ok(())
    }

    /// Whether `p` is a binding of this model as built: the devices'
    /// assertions hold and the structure is this one, so a state of its
    /// unknowns describes the circuit there. The raw evaluations at a given
    /// state ask this; an analysis sets another structure up instead (see
    /// [`restructure`]).
    pub(crate) fn bound(&self, p: &[f64]) -> Result<(), ModelError> {
        self.values_hold(p)?;
        self.structure_at(p).map_err(|m| {
            ModelError::Invalid(format!(
                "{m} (a state of this model does not describe the circuit at this binding; \
                 the analyses set its structure up)"
            ))
        })
    }

    /// A reference (node, unknown, branch current) in this model; a node
    /// the structure collapsed resolves to the one it collapsed onto.
    fn resolve(&self, reference: &str) -> Option<usize> {
        resolve_ref(&self.unknowns, &self.node_names, reference)
            .or_else(|| self.resolve_unknown(reference).flatten())
    }

    /// One DC solve at `p` as `dc` says, from `x0` (cold where empty): the
    /// state, and whether it converged.
    fn solve_from(&self, p: &[f64], dc: &DcOptions, x0: &[f64]) -> (Vec<f64>, bool) {
        let (x, conv, _) = (self.cdc).solve_dc_conv_with(
            p,
            x0,
            dc.convergence(),
            dc.max_iter,
            SolverTricks::default(),
        );
        (x, conv)
    }

    /// The DC operating point at the parameter vector `p`, solved as `dc`
    /// says, from `seed` where one is given (a cold start where it is empty
    /// or does not converge); and how the gmin regularization holds it where
    /// it does (converged only under a raised shunt, or with the floor setting
    /// a node: physically suspect, #54).
    fn solve_op(
        &self,
        p: &[f64],
        dc: &DcOptions,
        seed: &[f64],
    ) -> Result<(Vec<f64>, Option<Regularization>), ModelError> {
        let solve = |x0: &[f64]| self.solve_from(p, dc, x0);
        let (mut x, mut conv) = solve(seed);
        if !conv && !seed.is_empty() {
            (x, conv) = solve(&[]);
        }
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
            if let Ok(run) = self
                .cdc
                .solve_transient(&p, &[], &[0.0, 0.06], 1e-4, 1e-7, None)
            {
                if let Some(xs) = run.rows.last() {
                    let (x2, conv2) = solve(xs);
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
        let dominant = (self.cdc.last_gmin_dominance()).map(|(j, shift)| {
            let u = &self.unknowns[j];
            let node = (u.strip_prefix('v').and_then(|k| k.parse::<usize>().ok()))
                .and_then(|k| self.node_names.get(k));
            (node.unwrap_or(u).clone(), shift)
        });
        let reg = match (reg, dominant) {
            (None, None) => None,
            (gmin, dominant) => Some(Regularization {
                gmin: gmin.unwrap_or(sane_core::constants::GMIN_DC),
                dominant,
            }),
        };
        Ok((x, reg))
    }
}

/// A solved DC operating point. Holds the state and parameters in Rust, so node
/// values and derived sensitivities are read off it without a re-solve.
pub struct OperatingPoint {
    sim: Arc<ModelInner>,
    x: Vec<f64>,
    p: Vec<f64>,
    /// How the gmin regularization holds the point, where it does: converged,
    /// but physically suspect (issue #54); `None` for a true DC solution.
    pub regularization: Option<Regularization>,
    /// Where the point was solved in a model of another structure (see
    /// [`restructure`]): the asking model and the state in its layout.
    shown: Option<(Arc<ModelInner>, Vec<f64>)>,
}

impl OperatingPoint {
    /// The raw state vector in unknown/column order (of the model the
    /// analysis was asked of).
    pub fn vector(&self) -> &[f64] {
        match &self.shown {
            Some((_, x)) => x,
            None => &self.x,
        }
    }

    /// Rejects transport delays, which the DC adjoints do not carry yet.
    fn no_delays(&self, what: &str) -> Result<(), ModelError> {
        match self.sim.cdc.has_delays() {
            true => Err(ModelError::Numeric(format!(
                "{what}: transport delays (tline/absdelay) are not supported in this analysis yet"
            ))),
            false => Ok(()),
        }
    }

    /// The model the state is shown in, and the state there.
    fn view(&self) -> (&ModelInner, &[f64]) {
        match &self.shown {
            Some((m, x)) => (m, x),
            None => (&self.sim, &self.x),
        }
    }

    /// The value at a node / unknown / branch-current reference.
    pub fn get(&self, reference: &str) -> Option<f64> {
        self.sim.resolve(reference).map(|i| self.x[i])
    }
    /// The operating point as a `{unknown: value}` map (built on demand).
    pub fn to_map(&self) -> HashMap<String, f64> {
        let (m, x) = self.view();
        m.unknowns.iter().cloned().zip(x.iter().copied()).collect()
    }

    /// First-order sensitivities `d output / d param` of `outputs` by the
    /// parameters under `wrt` (all for none), one adjoint solve per output
    /// at this point.
    pub fn sensitivity(&self, outputs: &[&str], wrt: &[&str]) -> Result<Sensitivity, ModelError> {
        self.no_delays("sensitivity")?;
        let rows = self.sim.outputs(outputs)?;
        let cols = self.sim.columns(wrt)?;
        let mut ctx = self.sim.ctx.lock().unwrap();
        self.sim.cdc.ensure_param_jac(&mut ctx, &self.sim.dae);
        let mut grad = Array2::zeros((rows.len(), cols.len()));
        for (i, &r) in rows.iter().enumerate() {
            let g = self.sim.cdc.sensitivity(r, &self.x, &self.p, 0.0);
            if g.is_empty() {
                return Err(ModelError::Numeric(
                    "the operating point's Jacobian is singular".into(),
                ));
            }
            for (j, &c) in cols.iter().enumerate() {
                grad[[i, j]] = g[c];
            }
        }
        Ok(Sensitivity {
            outputs: outputs.iter().map(|o| o.to_string()).collect(),
            params: cols
                .iter()
                .map(|&c| self.sim.store.pnames[c].clone())
                .collect(),
            values: rows.iter().map(|&r| self.x[r]).collect(),
            param_values: cols.iter().map(|&c| self.p[c]).collect(),
            grad,
        })
    }

    /// Every operating-point variable the circuit's behavioral devices export
    /// (`(* desc *)`-annotated Verilog-A variables: gm, vth, ids, ...), evaluated
    /// at this solved point. Empty when no device declares any.
    pub fn op_vars(&self) -> Vec<OpVarValue> {
        let dae = &self.sim.dae;
        if dae.observers.is_empty() {
            return Vec::new();
        }
        let mut ctx = self.sim.ctx.lock().unwrap();
        let op_vars = dae.observers.op_vars(&mut ctx);
        let pnames = self.sim.cdc.param_names(&ctx);
        let env = crate::op_env(&mut ctx, dae, &pnames, &self.x, &self.p, 0.0);
        // One arena sweep: an op-var is a call into its device's template
        // function (evaluated once per instance) or a plain expression.
        let roots: Vec<_> = op_vars.iter().map(|v| v.value).collect();
        let vals = rsdag::eval(&ctx, &roots, &env);
        op_vars
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

    /// Second-order sensitivities (Hessians) of `outputs` by the parameters
    /// under `wrt` (all for none), by the second-order adjoint at this
    /// point.
    pub fn hessian(&self, outputs: &[&str], wrt: &[&str]) -> Result<Hessian, ModelError> {
        self.no_delays("hessian")?;
        let rows = self.sim.outputs(outputs)?;
        let cols = self.sim.columns(wrt)?;
        let mut ctx = self.sim.ctx.lock().unwrap();
        self.sim.cdc.ensure_hessian(&mut ctx, &self.sim.dae);
        let mut h = Array3::zeros((rows.len(), cols.len(), cols.len()));
        for (i, &r) in rows.iter().enumerate() {
            let hr = self.sim.cdc.hessian(r, &cols, &self.x, &self.p, 0.0);
            if hr.is_empty() {
                return Err(ModelError::Numeric(
                    "the operating point's Jacobian is singular".into(),
                ));
            }
            for (a, row) in hr.iter().enumerate() {
                for (b, &v) in row.iter().enumerate() {
                    h[[i, a, b]] = v;
                }
            }
        }
        Ok(Hessian {
            outputs: outputs.iter().map(|o| o.to_string()).collect(),
            params: cols
                .iter()
                .map(|&c| self.sim.store.pnames[c].clone())
                .collect(),
            values: rows.iter().map(|&r| self.x[r]).collect(),
            param_values: cols.iter().map(|&c| self.p[c]).collect(),
            h,
        })
    }
}

/// How operating points are solved (see [`Model::dc_options`]): the Newton
/// convergence criterion per unknown (relative, and absolute for currents and
/// voltages) and the iteration budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DcOptions {
    pub reltol: f64,
    /// The absolute floor of a current (A).
    pub abstol: f64,
    /// The absolute floor of a voltage (V).
    pub vntol: f64,
    pub max_iter: usize,
}

impl Default for DcOptions {
    fn default() -> Self {
        let c = Convergence::default();
        DcOptions {
            reltol: c.reltol,
            abstol: c.abstol,
            vntol: c.vntol,
            max_iter: DC_OP_MAXIT,
        }
    }
}

impl DcOptions {
    /// The defaults with what a circuit states (a deck's `.option`).
    fn from_circuit(s: &sane_circuit::DcSettings) -> Self {
        let d = DcOptions::default();
        DcOptions {
            reltol: s.reltol.unwrap_or(d.reltol),
            abstol: s.abstol.unwrap_or(d.abstol),
            vntol: s.vntol.unwrap_or(d.vntol),
            max_iter: s.max_iter.unwrap_or(d.max_iter),
        }
    }

    pub(crate) fn convergence(&self) -> Convergence {
        Convergence {
            reltol: self.reltol,
            abstol: self.abstol,
            vntol: self.vntol,
        }
    }
}

/// How the gmin regularization holds an operating point (issue #54): the
/// point converged, but depends on the shunt rather than on the circuit alone.
#[derive(Clone, Debug, PartialEq)]
pub struct Regularization {
    /// The shunt to ground (S) the point held at.
    pub gmin: f64,
    /// Where the solve reached the floor and the floor still sets a node:
    /// that node (or unknown), and the relative first-order shift removing
    /// the shunt would cause.
    pub dominant: Option<(String, f64)>,
}

impl std::fmt::Display for Regularization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.dominant {
            Some((unknown, shift)) => write!(
                f,
                "removing the gmin={:.1e} shunt to ground would shift '{unknown}' by {:.1}% \
                 (first order), so its voltage is the regularization's answer rather than \
                 the circuit's (that node needs a real path to ground)",
                self.gmin,
                100.0 * shift
            ),
            None => write!(
                f,
                "it held only at gmin={:.1e} and never reached the true DC floor \
                 (a high-impedance node is unstable at the floor)",
                self.gmin
            ),
        }
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

/// A DC sweep (see [`Point::dc_sweep`]): the state at every swept value.
pub struct DcSweep {
    sim: Arc<ModelInner>,
    /// The swept parameter.
    pub param: String,
    /// The swept values.
    pub values: Vec<f64>,
    /// Whether the point at `values[k]` converged.
    pub converged: Vec<bool>,
    /// `x[[k, i]]` is unknown `i` at `values[k]` (`NaN` where that point
    /// did not converge).
    pub x: Array2<f64>,
    /// Per point solved in a model of another structure (see
    /// [`restructure`]), that model and the point there.
    solved: Vec<Option<(Arc<ModelInner>, Vec<f64>)>>,
}

impl DcSweep {
    /// The swept series of a node / unknown / branch-current reference, each
    /// point's in the structure it was solved in (`NaN` where that has no
    /// such unknown); `None` where none has.
    pub fn signal(&self, reference: &str) -> Option<Vec<f64>> {
        let at = |k: usize| -> Option<f64> {
            match &self.solved[k] {
                None => self.sim.resolve(reference).map(|i| self.x[[k, i]]),
                Some((m, x)) => m.resolve(reference).map(|i| x[i]),
            }
        };
        let vals: Vec<Option<f64>> = (0..self.values.len()).map(at).collect();
        vals.iter()
            .any(Option::is_some)
            .then(|| vals.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
    }
}

/// First-order sensitivities of outputs by parameters: `grad[[i, j]]` is
/// `d outputs[i] / d params[j]`.
#[derive(Clone, Debug)]
pub struct Sensitivity {
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    /// The outputs' values.
    pub values: Vec<f64>,
    /// The parameters' values.
    pub param_values: Vec<f64>,
    pub grad: Array2<f64>,
}

impl Sensitivity {
    /// The derivatives of `output` by every parameter.
    pub fn of(&self, output: &str) -> Option<ArrayView1<'_, f64>> {
        let i = self.outputs.iter().position(|o| o == output)?;
        Some(self.grad.row(i))
    }

    /// `d output / d param`.
    pub fn get(&self, output: &str, param: &str) -> Option<f64> {
        let j = self.params.iter().position(|p| p == param)?;
        self.of(output).map(|g| g[j])
    }

    /// The dimensionless sensitivities `d ln output / d ln param =
    /// grad * param / output` (zero where the parameter or the output is).
    pub fn relative(&self) -> Array2<f64> {
        Array2::from_shape_fn(self.grad.dim(), |(i, j)| {
            let (p, y) = (self.param_values[j], self.values[i]);
            match p != 0.0 && y != 0.0 {
                true => self.grad[[i, j]] * p / y,
                false => 0.0,
            }
        })
    }

    /// The parameters by the magnitude of `output`'s sensitivity to them
    /// (relative, or raw), largest first.
    pub fn ranked(&self, output: &str, relative: bool) -> Option<Vec<(String, f64)>> {
        let i = self.outputs.iter().position(|o| o == output)?;
        let vals = if relative {
            self.relative()
        } else {
            self.grad.clone()
        };
        let mut out: Vec<(String, f64)> = (self.params.iter().cloned())
            .zip(vals.row(i).iter().copied())
            .collect();
        out.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        Some(out)
    }

    /// `output`'s sensitivities rolled up to the devices and subcircuits
    /// they belong to (the dotted prefix of a parameter, `X1.Q3.bf` ->
    /// `X1.Q3`): the 2-norm of each one's, largest first.
    pub fn rollup(&self, output: &str, relative: bool) -> Option<Vec<(String, f64)>> {
        let ranked = self.ranked(output, relative)?;
        let mut groups: Vec<(String, f64)> = Vec::new();
        for (name, v) in ranked {
            let comp = name
                .rsplit_once('.')
                .map_or(name.as_str(), |(c, _)| c)
                .to_string();
            match groups.iter_mut().find(|(c, _)| *c == comp) {
                Some((_, s)) => *s += v * v,
                None => groups.push((comp, v * v)),
            }
        }
        for g in &mut groups {
            g.1 = g.1.sqrt();
        }
        groups.sort_by(|a, b| b.1.total_cmp(&a.1));
        Some(groups)
    }
}

/// Warns (captured, see [`sane_core::log::warn_numerical`]) where the
/// small-signal system `G + jwC` of `what` was singular: at the frequencies
/// `bad` marks, its result is `NaN`.
pub(crate) fn warn_singular(what: &str, freqs: &[f64], bad: &[bool]) {
    let fs: Vec<f64> = (freqs.iter().zip(bad))
        .filter(|(_, &b)| b)
        .map(|(&f, _)| f)
        .collect();
    if fs.is_empty() {
        return;
    }
    let msg = if fs.len() == freqs.len() {
        format!(
            "{what}: G + jwC is singular at all {} frequencies, the result is NaN.              This is a structural singularity: check for floating nodes, an ideal              VCVS/inductor loop, or a bad DC operating point.",
            fs.len()
        )
    } else {
        let shown: Vec<String> = fs.iter().take(5).map(|f| format!("{f}")).collect();
        let more = if fs.len() > 5 { " ..." } else { "" };
        format!(
            "{what}: G + jwC is singular at {} of {} frequencies [{}{more}] Hz, the result is NaN there.",
            fs.len(),
            freqs.len(),
            shown.join(", ")
        )
    };
    sane_core::log::warn_numerical(&msg);
}

/// `rows` (each of `cols` entries) as one array.
pub(crate) fn stack<T: Clone>(rows: &[Vec<T>], cols: usize) -> ndarray::Array2<T> {
    let flat: Vec<T> = rows.iter().flat_map(|r| r.iter().cloned()).collect();
    ndarray::Array2::from_shape_vec((rows.len(), cols), flat).expect("rows of equal length")
}

/// The gradient of a scalar by every parameter.
#[derive(Clone, Debug)]
pub struct Gradient {
    pub params: Vec<String>,
    pub grad: Vec<f64>,
}

impl Gradient {
    /// From `(parameter, derivative)` pairs.
    pub(crate) fn from_pairs(pairs: Vec<(String, f64)>) -> Gradient {
        let (params, grad) = pairs.into_iter().unzip();
        Gradient { params, grad }
    }
}

/// Second-order sensitivities of outputs by parameters: `h[[i, a, b]]` is
/// `d^2 outputs[i] / d params[a] d params[b]`.
#[derive(Clone, Debug)]
pub struct Hessian {
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    pub values: Vec<f64>,
    pub param_values: Vec<f64>,
    pub h: Array3<f64>,
}

// Analysis methods by domain, one module each; shared guards live below.
mod ac;
mod hb;
mod lti;
mod noise;
mod num;
mod point;
mod pz;

pub use ac::{AcHessian, AcResponse, AcSensitivity, SParameters, SpSensitivity};
pub use hb::{HarmonicBalance, HbHessian, HbOptions, HbSensitivity};
pub use lti::{Poles, ReducedModel, RootSensitivity, StateSpace, Zeros};
pub use noise::{NoiseSensitivity, NoiseSpectrum};
pub use point::Point;
pub use tran::{Event, Trajectory, TrajectorySensitivity, TransientOptions};
mod restructure;
mod tran;
mod transform;

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
        // dI/dhist_k times the delayed signal's gradient: the coupling of the
        // signal's unknowns, delayed by tau_k
        let taus = self.cdc().delay_taus(p);
        let src = self.cdc().delay_source_jac(x, p, 0.0);
        let (hrows, hcols, hvals) = self.cdc().hist_jac_sparse(x, p);
        let (mut rows, mut cols, mut vals, mut tau) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for ((&i, &k), &v) in hrows.iter().zip(&hcols).zip(&hvals) {
            for &(_, j, w) in src.iter().filter(|e| e.0 == k) {
                rows.push(i);
                cols.push(j);
                vals.push(v * w);
                tau.push(taus[k]);
            }
        }
        (rows, cols, vals, tau)
    }
}
