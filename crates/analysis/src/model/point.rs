//! A model at one binding of its parameters (see [`Model::at`]): the binding
//! checked once, the model of its structure chosen once, and every analysis a
//! method of it.
//!
//! ```no_run
//! use sane_analysis::Model;
//! let m = Model::from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end").unwrap();
//! let op = m.at(&[("R2", 3e3)]).unwrap().operating_point().unwrap();
//! assert!((op.get("out").unwrap() - 3.75).abs() < 1e-6);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::restructure::Layout;
use ndarray::Array2;

use super::{DcOptions, DcSweep, Model, ModelError, ModelInner, OperatingPoint, Regularization};

/// A [`Model`] at one binding of its parameters: what every analysis runs
/// at. Cheap to clone; results hold what they need of it.
#[derive(Clone)]
pub struct Point {
    pub(super) model: Model,
    /// The binding, as the model's parameter vector.
    pub(super) p: Vec<f64>,
    /// Where the binding is of another structure (see [`super::restructure`]):
    /// the point in the model of that structure, and its layout in this one's.
    pub(super) other: Option<(Box<Point>, Arc<Layout>)>,
    /// How its operating point is solved, the model's options when the
    /// point was taken.
    dc: DcOptions,
    /// Where its operating point's solve starts (see [`Point::near`]).
    seed: Option<Arc<Vec<f64>>>,
    /// The operating point, solved on first use and shared by every analysis
    /// at the point (and every clone of it).
    op: Arc<Mutex<Option<(Vec<f64>, Option<Regularization>)>>>,
}

impl Model {
    /// The model at the binding `values` (parameter or group name, value)
    /// over its bound values: the point every analysis runs at. Rejects a
    /// name that is no parameter, a binding that fails a device's
    /// assertion, and one of another structure for a model with no netlist
    /// to set that structure up from.
    pub fn at(&self, values: &[(&str, f64)]) -> Result<Point, ModelError> {
        if let Some((name, _)) = values.iter().find(|(n, _)| !self.is_param(n)) {
            return Err(ModelError::UnknownParam(name.to_string()));
        }
        self.at_p(self.inner.store.pvec(values))
    }

    /// [`at`](Self::at) for the parameter vector `p`.
    pub(crate) fn at_p(&self, p: Vec<f64>) -> Result<Point, ModelError> {
        self.at_with(p, self.dc_options())
    }

    /// [`at_p`](Self::at_p), its operating point solved as `dc` says.
    fn at_with(&self, p: Vec<f64>, dc: DcOptions) -> Result<Point, ModelError> {
        let other = match self.restructured_at(&p)? {
            None => None,
            Some(o) => Some((Box::new(o.model.at_with(o.p(), dc)?), o.layout.clone())),
        };
        Ok(Point {
            model: self.clone(),
            p,
            other,
            dc,
            seed: None,
            op: Arc::default(),
        })
    }
}

impl Point {
    /// The model this point is of.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The binding, every parameter by name.
    pub fn values(&self) -> HashMap<String, f64> {
        (self.model.inner.store.pnames.iter().cloned())
            .zip(self.p.iter().copied())
            .collect()
    }

    /// This point, its operating point solved from `other`'s rather than
    /// cold: for a run of nearby bindings (an optimizer's, a sweep's), each
    /// solve a few Newton steps from the last. Where that does not converge
    /// the solve starts cold, and where the two are of different structures
    /// it starts cold too. Which operating point a multi-stable circuit lands
    /// on can depend on the start, so the start is explicit.
    pub fn near(&self, other: &Point) -> Result<Point, ModelError> {
        let (mine, theirs) = (self.solver(), other.solver());
        let seed = match Arc::ptr_eq(&mine.model.inner, &theirs.model.inner) {
            true => Some(Arc::new(theirs.op()?.0)),
            false => None,
        };
        let fresh = |pt: &Point, seed: Option<Arc<Vec<f64>>>| Point {
            seed,
            op: Arc::default(),
            ..pt.clone()
        };
        Ok(match &self.other {
            None => fresh(self, seed),
            Some((there, layout)) => Point {
                other: Some((Box::new(fresh(there, seed)), layout.clone())),
                ..fresh(self, None)
            },
        })
    }

    /// How its operating point is solved.
    pub fn dc_options(&self) -> DcOptions {
        self.dc
    }

    /// The point that solves: this one, or the one in the model of its
    /// structure.
    fn solver(&self) -> &Point {
        match &self.other {
            Some((there, _)) => there,
            None => self,
        }
    }

    /// The DC operating point, solved once per binding and kept for every
    /// analysis at it.
    pub fn operating_point(&self) -> Result<OperatingPoint, ModelError> {
        if let Some((there, layout)) = &self.other {
            let op = there.operating_point()?;
            let shown = layout.map(&op.x);
            return Ok(OperatingPoint {
                shown: Some((self.model.inner.clone(), shown)),
                ..op
            });
        }
        let inner = &self.model.inner;
        let (x, regularization) = self.op()?;
        if let Some(r) = &regularization {
            sane_core::log::warn_captured(&format!(
                "DC operating point is gmin-regularized: {r}. It is reported as converged \
                 but is physically suspect."
            ));
        }
        Ok(OperatingPoint {
            sim: inner.clone(),
            x,
            p: self.p.clone(),
            regularization,
            shown: None,
        })
    }

    /// The operating point of this point's own model, solved on first use.
    fn op(&self) -> Result<(Vec<f64>, Option<Regularization>), ModelError> {
        let mut op = self.op.lock().unwrap();
        if let Some(done) = op.as_ref() {
            return Ok(done.clone());
        }
        let inner = &self.model.inner;
        let mut task = sane_core::log::task("DC", "dc", &format!("(dim: {})", inner.dae.dim()));
        let seed = self.seed.as_deref().map_or(&[][..], |s| s.as_slice());
        let solved = inner.solve_op(&self.p, &self.dc, seed)?;
        task.finish(match &solved.1 {
            Some(r) => format!("converged: True, gmin-regularized: {:.1e}", r.gmin),
            None => "converged: True".to_string(),
        });
        *op = Some(solved.clone());
        Ok(solved)
    }

    /// Where a transient at this point starts: its operating point, when
    /// that is the circuit's own (not held by the regularization, which is
    /// no consistent state to integrate from); `None` leaves the integrator
    /// to find its start.
    pub(super) fn transient_start(&self) -> Option<Vec<f64>> {
        match self.op() {
            Ok((x, None)) => Some(x),
            _ => None,
        }
    }

    /// The operating point at each of `values` of the parameter `param`,
    /// the rest of the binding held: a DC sweep. Each point starts from the
    /// one before (and cold where that fails, so a point is the one a cold
    /// solve finds); a point where neither converges is marked so, its
    /// state `NaN`. A value
    /// may cross a device topology: that point is solved in the model of
    /// its structure.
    pub fn dc_sweep(&self, param: &str, values: &[f64]) -> Result<DcSweep, ModelError> {
        let inner = &self.model.inner;
        let col = inner
            .column(param)
            .ok_or_else(|| ModelError::UnknownParam(param.to_string()))?;
        let n = inner.dae.dim();
        let mut x_all = Array2::from_elem((values.len(), n), f64::NAN);
        let (mut converged, mut solved) = (Vec::new(), Vec::new());
        let mut warm: Vec<f64> = Vec::new();
        for (k, &v) in values.iter().enumerate() {
            let mut p = self.p.clone();
            p[col] = v;
            let at = self.model.at_p(p)?;
            // The point in the model of its structure: the state there, and
            // here.
            let solve = |warm: &[f64]| -> (Vec<f64>, Option<(Arc<ModelInner>, Vec<f64>)>, bool) {
                match &at.other {
                    None => {
                        let (x, conv) = inner.solve_from(&at.p, &self.dc, warm);
                        (x, None, conv)
                    }
                    Some((there, layout)) => {
                        let m = &there.model.inner;
                        let warm = if warm.len() == n {
                            layout.unmap(warm, m.dae.dim(), 0.0)
                        } else {
                            Vec::new()
                        };
                        let (x, conv) = m.solve_from(&there.p, &self.dc, &warm);
                        (layout.map(&x), Some((m.clone(), x)), conv)
                    }
                }
            };
            let (mut x, mut there, mut conv) = solve(&warm);
            if !conv && !warm.is_empty() {
                (x, there, conv) = solve(&[]);
            }
            converged.push(conv);
            if !conv {
                warm.clear();
                solved.push(None);
                continue;
            }
            x_all.row_mut(k).assign(&ndarray::ArrayView1::from(&x));
            warm = x;
            solved.push(there);
        }
        if !converged.iter().any(|&c| c) {
            return Err(ModelError::Numeric(
                "DC sweep did not converge at any point".into(),
            ));
        }
        Ok(DcSweep {
            sim: inner.clone(),
            param: param.to_string(),
            values: values.to_vec(),
            converged,
            x: x_all,
            solved,
        })
    }

    /// The model reduced at the operating point on its graph: a branch
    /// negligible (below `rel_tol` of its node's admittance) at DC and all of
    /// `freqs` (Hz) opened, one that dominates its node shorted; and what
    /// was done, `(element, "open" | "short")` once per element (a short
    /// over the open of its then-zero term). The reduced model shares this
    /// one's graph.
    pub fn prune(
        &self,
        rel_tol: f64,
        freqs: &[f64],
    ) -> Result<(Model, Vec<(String, String)>), ModelError> {
        if let Some((there, _)) = &self.other {
            return there.prune(rel_tol, freqs);
        }
        let x = self.operating_point()?.x;
        let omegas: Vec<f64> = std::iter::once(0.0)
            .chain(freqs.iter().map(|f| 2.0 * std::f64::consts::PI * f))
            .collect();
        let (model, applied) = self.model.prune_graph(rel_tol, &x, &self.p, &omegas);
        let mut seen = std::collections::HashSet::new();
        let applied = (applied.into_iter())
            .filter(|(e, _)| seen.insert(e.clone()))
            .collect();
        Ok((model, applied))
    }
}
