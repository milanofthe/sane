//! A model across the structures its parameters decide.
//!
//! A device decides its topology from parameter values at setup: a
//! resistance shorted for `R = 0` collapses its internal node, an integer
//! mode selects its equations (see `sane_dae::Dae::structure`). A binding
//! that crosses such a decision is a circuit of another structure. The
//! model sets its circuit up anew at the binding, as a
//! simulator re-runs a device's setup, keeps it for every later binding of
//! that structure, and runs the analysis there. What the analysis returns
//! in unknowns is reported in this model's layout ([`Layout`]): an unknown
//! of both by name, a node the other structure collapsed by the node it
//! collapsed onto, one the other structure lacks as `NaN`.

use std::sync::{Arc, Mutex};

use super::transform::Transform;
use super::{Model, ModelError, ModelInner};

/// The models of a circuit's other structures (see the module docs), each
/// derived as this model is (see [`super::transform`]).
#[derive(Default)]
pub(super) struct Restructure {
    /// The models of the other structures set up so far, each with its
    /// layout in this one's.
    others: Mutex<Vec<(Model, Arc<Layout>)>>,
    /// Where the latest solve ran, when in another structure.
    last: Mutex<Option<(Model, Arc<Layout>)>>,
}

/// Where each unknown of a model is in a model of another structure.
pub(crate) struct Layout {
    at: Vec<Source>,
}

#[derive(Clone, Copy)]
enum Source {
    /// That model's unknown.
    At(usize),
    /// Collapsed onto ground.
    Ground,
    /// Not in that structure.
    Missing,
}

impl Layout {
    /// The layout of `of` in `from`'s.
    fn between(from: &ModelInner, of: &ModelInner) -> Layout {
        let at = (from.unknowns.iter())
            .map(|u| match of.resolve_unknown(u) {
                Some(Some(j)) => Source::At(j),
                Some(None) => Source::Ground,
                None => Source::Missing,
            })
            .collect();
        Layout { at }
    }

    /// `v`, over the other model's unknowns, in this model's.
    pub(crate) fn map(&self, v: &[f64]) -> Vec<f64> {
        self.pick(v, 0.0, f64::NAN)
    }

    /// Per unknown of this model, its entry of `v` (one per unknown of the
    /// other model): `ground` for a node collapsed onto ground, `missing`
    /// for one the other structure lacks.
    pub(crate) fn pick<T: Clone>(&self, v: &[T], ground: T, missing: T) -> Vec<T> {
        (self.at.iter())
            .map(|s| match *s {
                Source::At(j) => v[j].clone(),
                Source::Ground => ground.clone(),
                Source::Missing => missing.clone(),
            })
            .collect()
    }

    /// `v`, over this model's unknowns, in the other model's: what this
    /// layout says of them, `fill` where it says nothing.
    pub(crate) fn unmap(&self, v: &[f64], n: usize, fill: f64) -> Vec<f64> {
        let mut out = vec![fill; n];
        for (k, s) in self.at.iter().enumerate() {
            if let Source::At(j) = *s {
                out[j] = v[k];
            }
        }
        out
    }
}

/// An analysis's binding in a model of another structure: the model, the
/// binding as its overrides, and its layout in the asking model's.
pub(crate) struct Other {
    pub(crate) model: Model,
    overrides: Vec<(String, f64)>,
    pub(crate) layout: Arc<Layout>,
}

impl Other {
    /// The binding, as overrides for [`model`](Self::model)'s analyses.
    pub(crate) fn overrides(&self) -> Vec<(&str, f64)> {
        self.overrides
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect()
    }

    /// The binding as the other model's parameter vector.
    pub(crate) fn p(&self) -> Vec<f64> {
        self.model.inner.store.pvec(&self.overrides())
    }
}

impl ModelInner {
    /// Note where the latest solve ran: in this model (`None`) or in the
    /// model of another structure, for the queries about it.
    pub(crate) fn solved_in(&self, other: Option<&Other>) {
        if let Some(re) = &self.restructure {
            *re.last.lock().unwrap() = other.map(|o| (o.model.clone(), o.layout.clone()));
        }
    }

    /// Whether the binding `p` is of this model's structure; else the
    /// message of the first decision it crosses.
    pub(super) fn structure_at(&self, p: &[f64]) -> Result<(), String> {
        let Some((tape, messages)) = &self.structure else {
            return Ok(());
        };
        let (mut w, mut out) = (Vec::new(), Vec::new());
        tape.eval(p, &mut w, &mut out);
        match out.iter().position(|&h| h == 0.0) {
            Some(k) => Err(messages[k].clone()),
            None => Ok(()),
        }
    }

    /// An unknown of this model by name: its index, or where the structure
    /// collapsed it, the index of the unknown it collapsed onto (`None`
    /// inside: ground); `None` when the model has no such unknown.
    pub(super) fn resolve_unknown(&self, name: &str) -> Option<Option<usize>> {
        if let Some(k) = self.unknowns.iter().position(|u| u == name) {
            return Some(Some(k));
        }
        let (_, onto) = self.dae.aliases.iter().find(|(n, _)| n == name)?;
        match onto {
            None => Some(None),
            Some(o) => self.unknowns.iter().position(|u| u == o).map(Some),
        }
    }
}

impl Model {
    /// Where the parameter vector `p` crosses the structure this model is
    /// built at: the model of its structure, set up from the netlist on first
    /// use (see the module docs); `None` where it does not. Rejects a binding
    /// that fails a device's assertion, and one of another structure for a
    /// model without a netlist to set it up from.
    pub(crate) fn restructured_at(&self, p: &[f64]) -> Result<Option<Other>, ModelError> {
        let inner = &self.inner;
        inner.values_hold(p)?;
        let Err(crossed) = inner.structure_at(p) else {
            inner.solved_in(None);
            return Ok(None);
        };
        let (Some(re), Some(circuit)) = (&inner.restructure, &inner.circuit) else {
            if inner.lineage.iter().any(|t| matches!(t, Transform::Prune)) {
                return Err(ModelError::Invalid(format!(
                    "{crossed} (a pruned model keeps the structure it was pruned in)"
                )));
            }
            return Err(ModelError::Invalid(crossed));
        };
        let binding: Vec<(String, f64)> = inner
            .store
            .pnames
            .iter()
            .cloned()
            .zip(p.iter().copied())
            .collect();
        let of = |m: &Model| -> Result<(), String> {
            let ov: Vec<(&str, f64)> = binding.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            m.inner.structure_at(&m.inner.store.pvec(&ov))
        };
        let mut others = re.others.lock().unwrap();
        let found = others.iter().find(|(m, _)| of(m).is_ok()).cloned();
        let (model, layout) = match found {
            Some(o) => o,
            None => {
                // the binding, and what the derivation folded at its values
                let folded = (inner.lineage.iter()).flat_map(|t| match t {
                    Transform::Fold(values) => values.as_slice(),
                    _ => &[],
                });
                let all: Vec<&(String, f64)> = binding.iter().chain(folded).collect();
                let value = |name: &str| all.iter().find(|(k, _)| k == name).map(|&&(_, v)| v);
                let mut model = Model::set_up(circuit.clone(), &value, false)?;
                for t in &inner.lineage {
                    model = model.apply(t.clone()).0;
                }
                if let Err(m) = of(&model) {
                    return Err(ModelError::Invalid(format!(
                        "the structure at this binding does not settle: {m}"
                    )));
                }
                let layout = Arc::new(Layout::between(inner, &model.inner));
                others.push((model.clone(), layout.clone()));
                (model, layout)
            }
        };
        let other = Other {
            model,
            overrides: binding,
            layout,
        };
        inner.solved_in(Some(&other));
        Ok(Some(other))
    }
}
