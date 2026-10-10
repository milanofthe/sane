//! The transforms of a model: the same circuit, another DAE.
//!
//! Folding parameters to constants, linearizing about the operating point,
//! eliminating internal resistive nodes and pruning negligible branches each
//! rewrite the model's DAE on its graph and keep everything that belongs to
//! the circuit: the circuit itself, its ports, its node names, how its
//! operating points are solved, and -- where the topology stays, as a fold
//! keeps it -- its index-2 report. A derived model remembers the transforms
//! it was derived by, so a binding that crosses a device's structure sets
//! the circuit up anew there and derives the same model of it (see
//! [`restructure`](super::restructure)); a pruned model, whose pruning rests
//! on one structure's operating point, keeps its structure.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use rsdag::Node;
use sane_dae::linearize::linearize;
use sane_dae::{eliminate_nodes, reduce_graph, Dae};
use sane_solve::CompiledDc;

use super::{
    assertion_tape, column_defaults, restructure, Model, ModelError, ModelInner, ParamStore,
};

/// A transform a model is derived by.
#[derive(Clone, Debug)]
pub(crate) enum Transform {
    /// The parameters folded to these values.
    Fold(Vec<(String, f64)>),
    /// Linearized about the operating point.
    Linearize,
    /// The internal resistive nodes eliminated, these node unknowns kept.
    Eliminate(Vec<String>),
    /// Negligible branches opened and dominant ones shorted at one operating
    /// point: not derived again in another structure.
    Prune,
}

impl Transform {
    /// The DAE of this transform of `model`'s, with the names of the nodes
    /// it eliminated (not for [`Prune`](Transform::Prune), whose DAE its
    /// caller computes).
    fn dae(&self, model: &Model) -> (Dae, Vec<String>) {
        let inner = &model.inner;
        let mut c = inner.ctx.lock().unwrap();
        match self {
            Transform::Fold(values) => {
                let mut fold = Vec::with_capacity(values.len());
                for (name, v) in values {
                    let e = c.sym(name);
                    if let Node::Symbol(s) = c.node(e) {
                        fold.push((*s, *v));
                    }
                }
                (inner.dae.fold_params(&mut c, &fold), Vec::new())
            }
            Transform::Linearize => (linearize(&mut c, &inner.dae), Vec::new()),
            Transform::Eliminate(keep) => {
                let keep: HashSet<String> = keep.iter().cloned().collect();
                eliminate_nodes(&mut c, &inner.dae, &keep)
            }
            Transform::Prune => unreachable!("a pruned DAE is its caller's"),
        }
    }
}

impl Model {
    /// The model of `dae`, the transform `t` of this one's: the circuit, its
    /// ports, node names, DC options and (for a fold) index-2 report this
    /// one's, the parameters the DAE's, valued as here.
    fn derive(&self, t: Transform, dae: Dae) -> Model {
        let inner = &self.inner;
        let mut c = inner.ctx.lock().unwrap();
        let cdc = CompiledDc::new(&mut c, &dae);
        let pnames = cdc.param_names(&c);
        let mut values = self.values();
        if let Transform::Fold(folded) = &t {
            for (name, _) in folded {
                values.remove(name);
            }
        }
        let store = ParamStore::new(pnames, values, column_defaults(&cdc, &dae));
        let assertions = assertion_tape(&c, &dae.assertions, &cdc);
        let structure = assertion_tape(&c, &dae.structure, &cdc);
        drop(c);
        let index2 = match t {
            Transform::Fold(_) => inner.index2.clone(),
            _ => Default::default(),
        };
        let replayable = !matches!(t, Transform::Prune);
        let restructure = (inner.restructure.is_some() && replayable && structure.is_some())
            .then(restructure::Restructure::default);
        let mut lineage = inner.lineage.clone();
        lineage.push(t);
        Model {
            inner: Arc::new(ModelInner {
                ctx: inner.ctx.clone(),
                unknowns: dae.unknowns.clone(),
                dae,
                cdc,
                index2,
                store,
                node_names: inner.node_names.clone(),
                ports: inner.ports.clone(),
                ac_vjp_tapes: Mutex::new(HashMap::new()),
                dc: Mutex::new(self.dc_options()),
                assertions,
                structure,
                restructure,
                circuit: inner.circuit.clone(),
                lineage,
            }),
        }
    }

    /// This model derived by `t`, with the nodes it eliminated.
    pub(super) fn apply(&self, t: Transform) -> (Model, Vec<String>) {
        let (dae, gone) = t.dae(self);
        (self.derive(t, dae), gone)
    }

    /// Operating-point-guided graph reduction at `(x, p)`: drop branch
    /// contributions negligible (conductance/capacitance) at the given angular
    /// frequencies. Returns the reduced model and the pruned `(branch, node)`
    /// pairs.
    pub(crate) fn prune_graph(
        &self,
        rel_tol: f64,
        x: &[f64],
        p: &[f64],
        omegas: &[f64],
    ) -> (Model, Vec<(String, String)>) {
        let (reduced, pruned) = {
            let mut c = self.inner.ctx.lock().unwrap();
            let pnames = self.inner.cdc.param_names(&c);
            let mut p_pairs = Vec::with_capacity(pnames.len());
            for (j, name) in pnames.iter().enumerate() {
                let e = c.sym(name);
                if let Node::Symbol(s) = c.node(e) {
                    p_pairs.push((*s, p.get(j).copied().unwrap_or(0.0)));
                }
            }
            reduce_graph(&mut c, &self.inner.dae, x, &p_pairs, omegas, rel_tol)
        };
        (self.derive(Transform::Prune, reduced), pruned)
    }

    /// Exactly eliminate internal resistive nodes (Schur/series reduction).
    /// `keep` protects node-unknown names. Returns the reduced `Model` and the
    /// eliminated node names, in order.
    pub fn eliminate_nodes(&self, keep: &[String]) -> (Model, Vec<String>) {
        self.apply(Transform::Eliminate(keep.to_vec()))
    }

    /// Linearise about the operating point into the small-signal mass-matrix DAE
    /// `G dx + d/dt (C dx) = 0`. Returns the linearised `Model`.
    pub fn linearize(&self) -> Model {
        self.apply(Transform::Linearize).0
    }

    /// Fold a set of parameters to their current values: each becomes a
    /// constant in a derived `Model`. Its now-constant subexpressions collapse
    /// (smaller graph, faster evaluation) and it leaves the parameter set
    /// (`params()` shrinks: no `dF/dp` column, no sensitivity). A `path` is
    /// either a single parameter (`X1.R1`, `nmos.vth0`) or a group prefix
    /// (`X1` -> all `X1.*`, recursively). This model is unchanged.
    pub fn fold(&self, paths: &[&str]) -> Result<Model, ModelError> {
        let names = self.inner.paths(paths)?;
        Ok(self.fold_names(&names))
    }

    /// [`fold`](Self::fold) every parameter except those under `paths`: the
    /// named ones stay symbolic, the rest become constants. A source driven
    /// as an AC input has to be kept.
    pub fn keep(&self, paths: &[&str]) -> Result<Model, ModelError> {
        let kept = self.inner.paths(paths)?;
        let names: HashSet<String> = (self.inner.store.pnames.iter())
            .filter(|n| !kept.contains(*n))
            .cloned()
            .collect();
        Ok(self.fold_names(&names))
    }

    /// The model with the parameters `names` folded to their values.
    fn fold_names(&self, names: &HashSet<String>) -> Model {
        let mut values: Vec<(String, f64)> = (names.iter())
            .map(|n| (n.clone(), self.inner.store.get(n).unwrap_or(0.0)))
            .collect();
        values.sort_by(|a, b| a.0.cmp(&b.0));
        self.apply(Transform::Fold(values)).0
    }
}
