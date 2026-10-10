//! The linear time-invariant views of a circuit at a [`Point`]: its poles,
//! the zeros of its transfers, a descriptor state space and reduced-order
//! models, all of the small-signal pencil `G + sC` at the operating point
//! (`G` with the gmin of the AC analysis, so every view agrees with it).

use std::collections::HashMap;

use ndarray::Array2;
use num_complex::Complex64;

use super::{Model, ModelError, Point};

/// The finite poles of the small-signal pencil (rad/s), see [`Point::poles`].
pub struct Poles {
    at: Point,
    x: Vec<f64>,
    pub poles: Vec<Complex64>,
}

/// The finite transmission zeros (rad/s) of the transfers from `input` to
/// each of `outputs`, see [`Point::zeros`]: `zeros[i]` those of `outputs[i]`.
pub struct Zeros {
    at: Point,
    x: Vec<f64>,
    outs: Vec<usize>,
    pub input: String,
    pub outputs: Vec<String>,
    pub zeros: Vec<Vec<Complex64>>,
}

/// Sensitivities of roots (poles or zeros): `grad[[r, j]]` is
/// `d roots[r] / d params[j]` (rad/s per unit), the operating point's shift
/// included.
#[derive(Clone, Debug)]
pub struct RootSensitivity {
    pub roots: Vec<Complex64>,
    pub params: Vec<String>,
    pub param_values: Vec<f64>,
    pub grad: Array2<Complex64>,
}

/// The descriptor state space `E x' = A x + B u`, `y = C x + D u` at the
/// operating point, see [`Point::state_space`]: `b[[k, m]]` the coupling of
/// `inputs[m]` into equation `k`, row `c[[i, ..]]` the selector of
/// `outputs[i]`.
pub struct StateSpace {
    pub states: Vec<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub e: Array2<f64>,
    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub c: Array2<f64>,
    pub d: Array2<f64>,
}

/// A dominant-pole reduced model of one transfer, see [`Point::reduce`]:
/// `H_r(s) = gain * prod(s - zeros) / prod(s - poles)`, against the full
/// transfer over `freqs`.
pub struct ReducedModel {
    pub input: String,
    pub output: String,
    pub poles: Vec<Complex64>,
    pub zeros: Vec<Complex64>,
    pub gain: Complex64,
    pub freqs: Vec<f64>,
    pub full: Vec<Complex64>,
    pub reduced: Vec<Complex64>,
}

impl Point {
    /// The finite poles of the small-signal pencil `G + sC` at the
    /// operating point (rad/s): the natural frequencies of the circuit.
    pub fn poles(&self) -> Result<Poles, ModelError> {
        if let Some((there, _)) = &self.other {
            return there.poles();
        }
        let model = &self.model;
        model.ensure_no_delays("poles")?;
        let x = self.operating_point()?.x;
        let _task = sane_core::log::task("PZ", "poles", &format!("(dim: {})", model.dim()));
        let poles = (model.poles(x.clone(), self.p.clone())?.into_iter())
            .map(|(re, im)| Complex64::new(re, im))
            .collect();
        Ok(Poles {
            at: self.clone(),
            x,
            poles,
        })
    }

    /// The finite transmission zeros (rad/s) of the transfers from the
    /// source parameter `input` to each of `outputs`: the roots of the
    /// Rosenbrock pencil of each.
    pub fn zeros(&self, input: &str, outputs: &[&str]) -> Result<Zeros, ModelError> {
        if let Some((there, _)) = &self.other {
            return there.zeros(input, outputs);
        }
        let model = &self.model;
        model.ensure_no_delays("zeros")?;
        input_param(model, input)?;
        let x = self.operating_point()?.x;
        let outs = model.inner.outputs(outputs)?;
        let zeros = (outs.iter())
            .map(|&o| {
                let z = model.zeros(input, o, x.clone(), self.p.clone())?;
                Ok(z.into_iter()
                    .map(|(re, im)| Complex64::new(re, im))
                    .collect())
            })
            .collect::<Result<_, ModelError>>()?;
        Ok(Zeros {
            at: self.clone(),
            x,
            outs,
            input: input.to_string(),
            outputs: outputs.iter().map(|o| o.to_string()).collect(),
            zeros,
        })
    }

    /// The descriptor state space at the operating point from the source
    /// parameters `inputs` to `outputs` (nodes, unknowns, branch currents):
    /// `E = dQ/dx`, `A = -(dI/dx + gmin)`, `B = -dI/du`, `C` the output
    /// selectors, `D = 0`.
    pub fn state_space(&self, inputs: &[&str], outputs: &[&str]) -> Result<StateSpace, ModelError> {
        if let Some((there, _)) = &self.other {
            return there.state_space(inputs, outputs);
        }
        let model = &self.model;
        model.ensure_no_delays("state_space")?;
        for input in inputs {
            input_param(model, input)?;
        }
        let x = self.operating_point()?.x;
        let outs = model.inner.outputs(outputs)?;
        let n = model.dim();
        let cdc = model.cdc();
        let a = -super::stack(&cdc.system_matrix_dc(&x, &self.p, 0.0), n);
        let e = super::stack(&cdc.jacobian_q_x(&x, &self.p, 0.0), n);
        let mut b = Array2::zeros((n, inputs.len()));
        for (m, input) in inputs.iter().enumerate() {
            let col = model.jacobian_i_input(input, x.clone(), self.p.clone(), 0.0)?;
            for (k, v) in col.into_iter().enumerate() {
                b[[k, m]] = -v;
            }
        }
        let c = Array2::from_shape_fn(
            (outs.len(), n),
            |(i, j)| if j == outs[i] { 1.0 } else { 0.0 },
        );
        Ok(StateSpace {
            states: model.unknowns().to_vec(),
            inputs: inputs.iter().map(|i| i.to_string()).collect(),
            outputs: outputs.iter().map(|o| o.to_string()).collect(),
            e,
            a,
            b,
            c,
            d: Array2::zeros((outs.len(), inputs.len())),
        })
    }

    /// The `order` most dominant poles (and as many matching zeros) of the
    /// transfer from `input` to `output`, the gain fitted to its DC value,
    /// against the full transfer at `freqs` (Hz).
    pub fn reduce(
        &self,
        input: &str,
        output: &str,
        order: usize,
        freqs: &[f64],
    ) -> Result<ReducedModel, ModelError> {
        if let Some((there, _)) = &self.other {
            return there.reduce(input, output, order, freqs);
        }
        let model = &self.model;
        model.ensure_no_delays("reduce")?;
        input_param(model, input)?;
        let x = self.operating_point()?.x;
        let out = model.inner.outputs(&[output])?[0];
        let r = {
            let arc = model.context_arc();
            let mut c = arc.lock().unwrap();
            let (dae, cdc) = (model.dae(), model.cdc());
            crate::model_reduce_on_dae(&mut c, dae, cdc, input, out, &x, &self.p, order, freqs)
                .map_err(ModelError::Numeric)?
        };
        let roots = |v: Vec<[f64; 2]>| v.into_iter().map(|r| Complex64::new(r[0], r[1])).collect();
        Ok(ReducedModel {
            input: input.to_string(),
            output: output.to_string(),
            poles: roots(r.poles),
            zeros: roots(r.zeros),
            gain: r.gain,
            freqs: freqs.to_vec(),
            full: r.full,
            reduced: r.reduced,
        })
    }
}

/// Rejects an input that is no source parameter of the model.
fn input_param(model: &Model, input: &str) -> Result<(), ModelError> {
    match model.is_param(input) {
        true => Ok(()),
        false => Err(ModelError::UnknownParam(input.to_string())),
    }
}

/// The root gradients `grads` (as the eigenvalue perturbation returns them,
/// every parameter by name) restricted to `wrt` and aligned with `roots`, each
/// root taking the gradient of the nearest perturbed root.
#[allow(clippy::type_complexity)]
fn align(
    at: &Point,
    roots: &[Complex64],
    grads: &[((f64, f64), Vec<(String, f64, f64)>)],
    wrt: &[&str],
) -> Result<RootSensitivity, ModelError> {
    let inner = &at.model.inner;
    let cols = inner.columns(wrt)?;
    let params: Vec<String> = cols
        .iter()
        .map(|&c| inner.store.pnames[c].clone())
        .collect();
    let mut grad = Array2::zeros((roots.len(), params.len()));
    for (r, root) in roots.iter().enumerate() {
        let near = (grads.iter()).min_by(|a, b| {
            let d = |s: (f64, f64)| (Complex64::new(s.0, s.1) - root).norm();
            d(a.0).total_cmp(&d(b.0))
        });
        let by_name: HashMap<&str, Complex64> = (near.iter())
            .flat_map(|(_, g)| {
                g.iter()
                    .map(|(n, re, im)| (n.as_str(), Complex64::new(*re, *im)))
            })
            .collect();
        for (j, n) in params.iter().enumerate() {
            grad[[r, j]] = by_name.get(n.as_str()).copied().unwrap_or_default();
        }
    }
    Ok(RootSensitivity {
        roots: roots.to_vec(),
        param_values: cols.iter().map(|&c| at.p[c]).collect(),
        params,
        grad,
    })
}

impl Poles {
    /// The derivatives of every pole by the parameters under `wrt` (all for
    /// none), exact: eigenvalue perturbation plus one adjoint for the
    /// operating point's shift.
    pub fn sensitivity(&self, wrt: &[&str]) -> Result<RootSensitivity, ModelError> {
        let g = (self.at.model).pole_gradient(self.x.clone(), self.at.p.clone())?;
        align(&self.at, &self.poles, &g, wrt)
    }
}

impl Zeros {
    /// The zeros of the transfer to `output`.
    pub fn of(&self, output: &str) -> Option<&[Complex64]> {
        let i = self.outputs.iter().position(|o| o == output)?;
        Some(&self.zeros[i])
    }

    /// The derivatives of every zero by the parameters under `wrt` (all
    /// for none), one per output, exact as for the poles.
    pub fn sensitivity(&self, wrt: &[&str]) -> Result<Vec<RootSensitivity>, ModelError> {
        let model = &self.at.model;
        (self.outs.iter().zip(&self.zeros))
            .map(|(&o, zeros)| {
                let g = model.zero_gradient(&self.input, o, self.x.clone(), self.at.p.clone())?;
                align(&self.at, zeros, &g, wrt)
            })
            .collect()
    }
}

impl ReducedModel {
    /// The magnitude of the full transfer (dB) at every frequency.
    pub fn full_db(&self) -> Vec<f64> {
        db(&self.full)
    }

    /// The magnitude of the reduced transfer (dB) at every frequency.
    pub fn reduced_db(&self) -> Vec<f64> {
        db(&self.reduced)
    }

    /// The largest magnitude error of the reduced transfer (dB).
    pub fn max_error_db(&self) -> f64 {
        (self.full_db().iter().zip(self.reduced_db()))
            .map(|(f, r)| (f - r).abs())
            .fold(0.0, f64::max)
    }
}

fn db(h: &[Complex64]) -> Vec<f64> {
    h.iter()
        .map(|v| 20.0 * v.norm().max(1e-30).log10())
        .collect()
}
