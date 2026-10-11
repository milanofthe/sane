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

use sane_solve::Triplets;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;

use ndarray::{Array2, Array3, ArrayView1};
use sane_circuit::Circuit;
use sane_core::constants::DC_OP_MAXIT;
use sane_core::log_stage;
use sane_core::Graph;
use sane_solve::{CompiledDc, Convergence, SolverTricks};

use crate::prepare;

mod error;
mod op;
mod params;
pub use error::ModelError;
pub use op::{
    DcOptions, DcSweep, Gradient, Hessian, OpVarValue, OperatingPoint, Regularization, Sensitivity,
};
use params::{column_defaults, ParamStore};

/// The unknown a reference names: an unknown by its own name (`v3`,
/// `i_V1`), a node by its name (`out`), or an element's branch current
/// (`V1`, `I(V1)`).
fn resolve_ref(dae: &sane_dae::Dae, node_names: &[String], reference: &str) -> Option<usize> {
    if let Some(i) = dae.find(reference) {
        return Some(i);
    }
    if let Some(i) = (node_names.iter().position(|n| n == reference)).and_then(|k| dae.node(k)) {
        return Some(i);
    }
    let element = (reference
        .strip_prefix("I(")
        .and_then(|r| r.strip_suffix(')')))
    .unwrap_or(reference);
    dae.branch(element)
}

/// Shared inner state; result handles hold an `Arc<ModelInner>` so they are owned,
/// `'static`, and need no lifetime threading (and are pyo3-friendly).
struct ModelInner {
    /// The compiled system, with the graph and the DAE it compiles.
    cdc: CompiledDc,
    /// The unknowns' names (see [`sane_dae::UnknownId::name`]).
    names: Vec<String>,
    /// Index-2 topologies of the deck (capacitor/source loops, inductor/
    /// source cutsets), detected once at build time. Transient consults it:
    /// such a circuit's hidden constraint carries no truncation error, so the
    /// step controller has nothing to control and would otherwise stride past
    /// the requested resolution. See `sane_circuit::index2`.
    index2: sane_circuit::index2::Index2Report,
    store: ParamStore,
    node_names: Vec<String>,
    /// Power ports from deck `P` elements, in deck order: `(drive source,
    /// network-side node, z0)`. Empty when the deck has none (or the model
    /// was built from parts rather than a netlist).
    ports: Vec<(String, String, f64)>,
    /// The layouts of the small-signal pulls built so far, by key (see
    /// [`adjoint`]).
    pull_layouts: Mutex<HashMap<String, Arc<adjoint::PullLayout>>>,
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

/// The assertions `list` picks of `cdc`'s DAE compiled over its parameter
/// vector, with the messages; `None` for none.
fn assertion_tape(
    cdc: &CompiledDc,
    list: impl Fn(&sane_dae::Dae) -> &[sane_dae::Assertion],
) -> Option<(rsdag::Tape, Vec<String>)> {
    let list = list(cdc.dae());
    if list.is_empty() {
        return None;
    }
    let holds: Vec<rsdag::ExprId> = list.iter().map(|a| a.holds).collect();
    let tape = rsdag::Tape::compile(&cdc.graph(), &holds, cdc.param_syms());
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
        let circuit = log_stage!("parse", sane_netlist::parse(src)).map_err(ModelError::Parse)?;
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
        let cdc = prepare(&circuit, values).map_err(ModelError::Build)?;
        let pnames = cdc.param_names().to_vec();
        task.finish(format!("dim: {}, params: {}", cdc.dim(), pnames.len()));
        let (elements, terminals) = sane_circuit::topology(&circuit);
        let index2 = detect_index2(&elements, &terminals);
        let store = ParamStore::new(
            pnames,
            circuit
                .values
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            column_defaults(&cdc),
        );
        let node_names = circuit.node_names();
        let ports = (circuit.ports.iter())
            .map(|pt| (pt.name.clone(), pt.node.clone(), pt.z0))
            .collect();
        let assertions = assertion_tape(&cdc, |d| &d.assertions);
        let structure = assertion_tape(&cdc, |d| &d.structure);
        let restructure = (root && structure.is_some()).then(restructure::Restructure::default);
        let dc = Mutex::new(DcOptions::from_circuit(&circuit.dc));
        Ok(Model {
            inner: Arc::new(ModelInner {
                names: cdc.dae().unknown_names(),
                cdc,
                index2,
                store,
                node_names,
                ports,
                assertions,
                structure,
                restructure,
                circuit: Some(circuit),
                lineage: Vec::new(),
                pull_layouts: Mutex::new(HashMap::new()),
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
        self.inner.unknowns()
    }
    pub fn node_names(&self) -> &[String] {
        &self.inner.node_names
    }
    /// How unknown `i` reads to a user: a node by its name, a branch
    /// current as `I(element)`, an internal unknown by its own name. Every
    /// label resolves back to its unknown (see [`resolve`](Self::resolve)).
    pub fn label(&self, i: usize) -> String {
        self.inner.label(i)
    }
    pub fn dim(&self) -> usize {
        self.inner.cdc.dim()
    }

    /// The shared symbolic context (`Arc<Mutex>`), so a host (e.g. the Python
    /// binding) can hand out expression handles that operate on the *same* graph
    /// this model was built from.
    pub fn context_arc(&self) -> Arc<Mutex<Graph>> {
        self.inner.cdc.context().clone()
    }

    /// The underlying symbolic DAE (residual expression ids, state symbols, ...),
    /// for the symbolic interface.
    pub fn dae(&self) -> &sane_dae::Dae {
        self.inner.cdc.dae()
    }

    /// The compiled DC solver (residual / Jacobian tapes), for the low-level
    /// numeric and gradient interfaces.
    /// The deck's power ports (`P` elements) in deck order:
    /// `(drive source, network-side node, z0)` per port.
    pub fn ports(&self) -> &[(String, String, f64)] {
        &self.inner.ports
    }

    /// The small-signal pulls' layouts (see [`adjoint`]).
    pub(crate) fn pull_layouts(&self) -> &Mutex<HashMap<String, Arc<adjoint::PullLayout>>> {
        &self.inner.pull_layouts
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
        resolve_ref(self.dae(), &self.inner.node_names, reference)
    }

    // --- analyses ---
}

impl ModelInner {
    /// The DAE the model compiles.
    fn dae(&self) -> &sane_dae::Dae {
        self.cdc.dae()
    }

    /// The unknowns' names.
    fn unknowns(&self) -> &[String] {
        &self.names
    }

    /// How unknown `i` reads to a user: a node by its name, a branch
    /// current as `I(element)`, an internal unknown by its own name.
    fn label(&self, i: usize) -> String {
        use sane_dae::UnknownId;
        match &self.dae().unknowns[i] {
            UnknownId::Node(k) => match self.node_names.get(*k) {
                Some(name) if !name.is_empty() => name.clone(),
                _ => self.names[i].clone(),
            },
            UnknownId::Branch(e) => format!("I({e})"),
            UnknownId::Internal(s) => s.clone(),
        }
    }

    /// The symbols the source `input` drives the small-signal response
    /// through: the parameter `input` names, or the source `input`'s DC
    /// value or waveform level (see [`sane_dae::Source::drive`]).
    pub(crate) fn drive(&self, input: &str) -> Result<Vec<rsdag::SymbolId>, ModelError> {
        let syms = self.cdc.param_syms();
        let drive: Vec<rsdag::SymbolId> = match self.column(input) {
            Some(j) => vec![syms[j]],
            None => (self
                .dae()
                .source(input)
                .map(|s| s.drive())
                .unwrap_or_default())
            .into_iter()
            .filter(|s| syms.contains(s))
            .collect(),
        };
        if drive.is_empty() {
            return Err(ModelError::Invalid(format!(
                "'{input}' drives nothing: neither its value nor its waveform's level is a parameter of the model (a folded source drives nothing)"
            )));
        }
        Ok(drive)
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
        resolve_ref(self.dae(), &self.node_names, reference)
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
            if let Ok(run) = self.cdc.solve_transient(
                &p,
                &[],
                &[0.0, sane_core::constants::GMIN_SETTLE_SPAN],
                sane_core::constants::TRANSIENT_RTOL,
                sane_core::constants::TRANSIENT_ATOL,
                None,
            ) {
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
        let dominant = (self.cdc.last_gmin_dominance()).map(|(j, shift)| (self.label(j), shift));
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

// Analysis methods by domain, one module each; shared guards live below.
mod ac;
mod adjoint;
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
        let Triplets {
            rows: hrows,
            cols: hcols,
            vals: hvals,
        } = self.cdc().hist_jac_sparse(x, p);
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
