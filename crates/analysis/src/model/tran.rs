//! Transient analysis at a [`Point`]: the trajectory and its forward
//! sensitivities (see `sane_solve::transient_sens`).

use std::sync::Arc;

use ndarray::{Array2, Array3, ArrayView2};
use sane_core::log;

use super::restructure::Layout;
use super::{Gradient, Model, ModelError, Point};

/// How a transient integrates (see [`Point::transient`]).
#[derive(Clone, Debug)]
pub struct TransientOptions {
    /// The local error tolerances, relative and absolute.
    pub rtol: f64,
    pub atol: f64,
    /// The largest step the error control may take; `None` leaves the step
    /// to it.
    pub dt_max: Option<f64>,
    /// The initial state, in the model's layout (a consistent one is found
    /// from it); `None` starts from the operating point.
    pub x0: Option<Vec<f64>>,
}

impl Default for TransientOptions {
    fn default() -> Self {
        TransientOptions {
            rtol: 1e-4,
            atol: 1e-7,
            dt_max: None,
            x0: None,
        }
    }
}

/// A switching surface a transient crossed: its name (`instance#k`), the
/// time, and `+1` when the surface expression rose through zero, `-1` when
/// it fell.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub name: String,
    pub t: f64,
    pub direction: i8,
}

/// A transient solution: the state at each requested time, labeled, and the
/// switching events crossed. Its derivatives by the parameters are methods.
pub struct Trajectory {
    /// The point it was integrated at (of the binding's structure) and how.
    at: Point,
    opts: TransientOptions,
    pub t: Vec<f64>,
    /// `x[[k, i]]` is unknown `i` (of the model asked) at `t[k]`.
    pub x: Array2<f64>,
    /// Integrated in a model of another structure than the asking one (see
    /// [`super::restructure`]): the states there, and its layout in the
    /// asking model.
    there: Option<(Array2<f64>, Arc<Layout>)>,
    pub events: Vec<Event>,
}

/// Forward sensitivities of a trajectory's outputs over time:
/// `grad[[i, k, j]]` is `d outputs[i] / d params[j]` at `t[k]`.
#[derive(Clone, Debug)]
pub struct TrajectorySensitivity {
    pub t: Vec<f64>,
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    pub param_values: Vec<f64>,
    /// `values[[i, k]]` is `outputs[i]` at `t[k]`.
    pub values: Array2<f64>,
    pub grad: Array3<f64>,
}

impl Point {
    /// The transient response at the time points `t` (the first is the
    /// start), integrated as `opts` says.
    pub fn transient(&self, t: &[f64], opts: &TransientOptions) -> Result<Trajectory, ModelError> {
        if let Some((there, layout)) = &self.other {
            let x0 = (opts.x0.as_ref()).map(|x| layout.unmap(x, there.model.dim(), 0.0));
            let mut tr = there.transient(t, &TransientOptions { x0, ..opts.clone() })?;
            let rows: Vec<Vec<f64>> = (tr.x.outer_iter())
                .map(|r| layout.map(&r.to_vec()))
                .collect();
            let shown = super::stack(&rows, self.model.dim());
            tr.there = Some((std::mem::replace(&mut tr.x, shown), layout.clone()));
            return Ok(tr);
        }
        let inner = &self.model.inner;
        let cdc = &inner.cdc;
        let x0 = (opts.x0.clone())
            .or_else(|| self.transient_start())
            .unwrap_or_default();
        let run = cdc
            .solve_transient(&self.p, &x0, t, opts.rtol, opts.atol, opts.dt_max)
            .map_err(|e| self.model.failure(e))?;
        let names = cdc.event_names();
        let events = (run.events.iter())
            .map(|e| Event {
                name: names[e.index].clone(),
                t: e.t,
                direction: e.direction,
            })
            .collect();
        let rows = run.rows;
        Ok(Trajectory {
            at: self.clone(),
            opts: opts.clone(),
            t: t.to_vec(),
            x: super::stack(&rows, inner.dae.dim()),
            there: None,
            events,
        })
    }
}

impl Trajectory {
    /// The time series of a node, unknown or branch current.
    pub fn signal(&self, reference: &str) -> Option<Vec<f64>> {
        let i = self.at.model.inner.resolve(reference)?;
        let x = self.there.as_ref().map_or(&self.x, |(x, _)| x);
        Some(x.column(i).to_vec())
    }

    /// Forward sensitivities of `outputs` by the parameters under `wrt`
    /// (all for none) over the trajectory's times: the derivative of every
    /// step the integration took (see `sane_solve::transient_sens`), the
    /// trajectory integrated again with them. From the operating point the
    /// sensitivities start at the point's; from a given state, at zero.
    pub fn sensitivity(
        &self,
        outputs: &[&str],
        wrt: &[&str],
    ) -> Result<TrajectorySensitivity, ModelError> {
        let model = &self.at.model;
        model.ensure_no_delays("transient sensitivity")?;
        let inner = &model.inner;
        let outs = inner.outputs(outputs)?;
        let cols = inner.columns(wrt)?;
        let mut task = log::task(
            "SENS-TRANSIENT",
            "sens_tran",
            &format!("(params: {}, points: {})", cols.len(), self.t.len()),
        );
        {
            let arc = model.context_arc();
            let mut c = arc.lock().unwrap();
            inner
                .cdc
                .ensure_transient_sensitivity(&mut c, &inner.dae, &cols);
        }
        let o = &self.opts;
        let x0 = (o.x0.clone())
            .or_else(|| self.at.transient_start())
            .unwrap_or_default();
        let (run, sens) = (inner.cdc)
            .solve_transient_sensitivity(
                &self.at.p,
                &x0,
                o.x0.is_none(),
                &self.t,
                o.rtol,
                o.atol,
                o.dt_max,
                &cols,
                &outs,
            )
            .map_err(|e| model.failure(e))?;
        task.finish(format!("outputs: {}", outs.len()));
        let (nt, np) = (self.t.len(), cols.len());
        Ok(TrajectorySensitivity {
            t: self.t.clone(),
            outputs: outputs.iter().map(|o| o.to_string()).collect(),
            params: cols
                .iter()
                .map(|&c| inner.store.pnames[c].clone())
                .collect(),
            param_values: cols.iter().map(|&c| self.at.p[c]).collect(),
            values: Array2::from_shape_fn((outs.len(), nt), |(i, k)| run.rows[k][outs[i]]),
            grad: Array3::from_shape_fn((outs.len(), nt, np), |(i, k, j)| sens[k][i * np + j]),
        })
    }

    /// The gradient of a scalar `L` of the waveforms of `outputs` by the
    /// parameters under `wrt` (all for none), given its cotangents
    /// `cotangent[[i, k]] = dL/d outputs[i](t[k])`: the cotangents
    /// contracted with the forward sensitivities.
    pub fn vjp(
        &self,
        outputs: &[&str],
        cotangent: ArrayView2<'_, f64>,
        wrt: &[&str],
    ) -> Result<Gradient, ModelError> {
        if cotangent.dim() != (outputs.len(), self.t.len()) {
            return Err(ModelError::Numeric(format!(
                "a cotangent per output and time: {:?} for {:?}",
                cotangent.dim(),
                (outputs.len(), self.t.len())
            )));
        }
        let s = self.sensitivity(outputs, wrt)?;
        let grad = (0..s.params.len())
            .map(|j| {
                let mut g = 0.0;
                for i in 0..outputs.len() {
                    for k in 0..self.t.len() {
                        g += cotangent[[i, k]] * s.grad[[i, k, j]];
                    }
                }
                g
            })
            .collect();
        Ok(Gradient {
            params: s.params,
            grad,
        })
    }
}

impl Model {
    /// A transient failure as an error, with the index-2 loops and cutsets
    /// of the deck named where there are any: their unknowns are rates of
    /// the others, which a failure there often traces back to.
    fn failure(&self, err: String) -> ModelError {
        let rep = &self.inner.index2;
        if !rep.is_index2() {
            return ModelError::Numeric(err);
        }
        ModelError::Numeric(format!(
            "{err} -- this deck is index 2 ({}): a series resistance in the loop, a shunt across the cutset, the parasitic that exists in reality, makes it index 1.",
            rep.summary()
        ))
    }
}
