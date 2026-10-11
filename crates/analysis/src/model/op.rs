//! The operating point and what is read off it: its solve options, its
//! regularization report, the operating-point variables, the
//! sensitivities and Hessians, the DC sweep.

use super::*;

/// A solved DC operating point. Holds the state and parameters in Rust, so node
/// values and derived sensitivities are read off it without a re-solve.
pub struct OperatingPoint {
    pub(super) sim: Arc<ModelInner>,
    pub(super) x: Vec<f64>,
    pub(super) p: Vec<f64>,
    /// How the gmin regularization holds the point, where it does: converged,
    /// but physically suspect (issue #54); `None` for a true DC solution.
    pub regularization: Option<Regularization>,
    /// Where the point was solved in a model of another structure (see
    /// [`restructure`]): the asking model and the state in its layout.
    pub(super) shown: Option<(Arc<ModelInner>, Vec<f64>)>,
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
    pub(super) fn no_delays(&self, what: &str) -> Result<(), ModelError> {
        match self.sim.cdc.has_delays() {
            true => Err(ModelError::Numeric(format!(
                "{what}: transport delays (tline/absdelay) are not supported in this analysis yet"
            ))),
            false => Ok(()),
        }
    }

    /// The model the state is shown in, and the state there.
    pub(super) fn view(&self) -> (&ModelInner, &[f64]) {
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
        m.unknowns()
            .iter()
            .cloned()
            .zip(x.iter().copied())
            .collect()
    }

    /// First-order sensitivities `d output / d param` of `outputs` by the
    /// parameters under `wrt` (all for none), one adjoint solve per output
    /// at this point.
    pub fn sensitivity(&self, outputs: &[&str], wrt: &[&str]) -> Result<Sensitivity, ModelError> {
        self.no_delays("sensitivity")?;
        let rows = self.sim.outputs(outputs)?;
        let cols = self.sim.columns(wrt)?;
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
        let dae = self.sim.cdc.dae();
        if dae.observers.is_empty() {
            return Vec::new();
        }
        let mut ctx = self.sim.cdc.graph();
        let op_vars = dae.observers.op_vars(&mut ctx);
        let syms = self.sim.cdc.param_syms().to_vec();
        let p = syms.into_iter().zip(self.p.iter().copied());
        let env = dae.point_env(&mut ctx, &self.x, p, 0.0);
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
    pub(super) fn from_circuit(s: &sane_circuit::DcSettings) -> Self {
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
    pub(super) sim: Arc<ModelInner>,
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
    pub(super) solved: Vec<Option<(Arc<ModelInner>, Vec<f64>)>>,
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
