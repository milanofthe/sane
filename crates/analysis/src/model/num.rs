//! Numeric passthroughs and DC-level entry points on [`Model`]: residual /
//! Jacobian evaluation, DC solve, exact first/second-order DC sensitivities,
//! and the raw state-space / temperature-sweep / model-reduction kernels.

use rsdag::{differentiate, Node};
use sane_core::log_stage;

use crate::model::{Model, ModelError};
use crate::{model_reduce_on_dae, op_env, state_space_on_dae, temp_sweep_on_dae};

impl Model {
    /// Residual `F(x, x', t)` as a numeric vector.
    pub fn residual(&self, x: Vec<f64>, xdot: Vec<f64>, p: Vec<f64>, t: f64) -> Vec<f64> {
        self.cdc().residual(&x, &xdot, &p, t)
    }

    /// Jacobian `dF/dx` as a dense matrix.
    pub fn jacobian_x(&self, x: Vec<f64>, xdot: Vec<f64>, p: Vec<f64>, t: f64) -> Vec<Vec<f64>> {
        self.cdc().jacobian_x(&x, &xdot, &p, t)
    }

    /// Jacobian `dF/dx'` as a dense matrix.
    pub fn jacobian_xdot(&self, x: Vec<f64>, xdot: Vec<f64>, p: Vec<f64>, t: f64) -> Vec<Vec<f64>> {
        self.cdc().jacobian_xdot(&x, &xdot, &p, t)
    }

    /// Sparse `dF/dx` as `(rows, cols, values)` (COO).
    pub fn jacobian_x_sparse(
        &self,
        x: Vec<f64>,
        xdot: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        self.cdc().jacobian_x_sparse(&x, &xdot, &p, t)
    }

    /// Sparse `dF/dx'` as `(rows, cols, values)` (COO).
    pub fn jacobian_xdot_sparse(
        &self,
        x: Vec<f64>,
        xdot: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        self.cdc().jacobian_xdot_sparse(&x, &xdot, &p, t)
    }

    /// Sparse `dF/dp` as `(rows, cols, values)` (COO).
    pub fn jacobian_p_sparse(
        &self,
        x: Vec<f64>,
        xdot: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let arc = self.context_arc();
        self.cdc()
            .ensure_param_jac(&mut arc.lock().unwrap(), self.dae());
        self.cdc().jacobian_p_sparse(&x, &xdot, &p, t)
    }

    /// Number of structural nonzeros in the sparse `dF/dx` pattern.
    pub fn nnz(&self) -> usize {
        self.cdc().nnz()
    }

    /// Schur partition sizes `(linear_block, nonlinear_block)`, or `None`.
    pub fn partition_sizes(&self) -> Option<(usize, usize)> {
        self.cdc().partition_sizes()
    }

    /// Exact input-coupling vector `B = dF/d(input)` for the named source, by
    /// symbolic differentiation (autodiff), evaluated at `(x, xdot, p, t)`.
    pub fn input_jacobian(
        &self,
        input: &str,
        x: Vec<f64>,
        xdot: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> Result<Vec<f64>, ModelError> {
        let arc = self.context_arc();
        let mut c = arc.lock().unwrap();
        let dae = self.dae();
        let ie = c.sym(input);
        let isym = match c.node(ie) {
            Node::Symbol(s) => *s,
            _ => return Err(ModelError::Numeric(format!("'{input}' is not a symbol"))),
        };
        let residuals = dae.residuals.clone();
        let db: Vec<_> = residuals
            .iter()
            .map(|&r| differentiate(&mut c, r, isym))
            .collect();
        let pnames = self.cdc().param_names(&c);
        let env = op_env(&mut c, dae, &pnames, &x, &xdot, &p, t);
        Ok(rsdag::eval(&c, &db, &env))
    }

    /// Exact first-order sensitivity `dy/dp` of `y = output` (an unknown name)
    /// w.r.t. every parameter at the point `x`, via the adjoint. Returns
    /// `(param_names, dy/dp)`.
    pub fn sensitivity(
        &self,
        output: &str,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> Result<(Vec<String>, Vec<f64>), ModelError> {
        self.ensure_no_delays("sensitivity")?;
        let metric = self
            .dae()
            .unknowns
            .iter()
            .position(|u| u == output)
            .ok_or_else(|| ModelError::Numeric(format!("unknown output '{output}'")))?;
        let arc = self.context_arc();
        self.cdc()
            .ensure_param_jac(&mut arc.lock().unwrap(), self.dae());
        let s = log_stage!("sens/dc_adjoint", self.cdc().sensitivity(metric, &x, &p, t));
        if s.is_empty() {
            return Err(ModelError::Numeric(
                "sensitivity: singular Jacobian".to_string(),
            ));
        }
        let pnames = self.cdc().param_names(&arc.lock().unwrap());
        Ok((pnames, s))
    }

    /// Exact second-order sensitivity (Hessian) of `y = output` w.r.t. a `subset`
    /// of parameters, by the second-order adjoint (exact AD directional
    /// derivatives). Returns the dense symmetric `len(subset) x len(subset)` matrix.
    pub fn hessian(
        &self,
        output: &str,
        subset: Vec<String>,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> Result<Vec<Vec<f64>>, ModelError> {
        let metric = self
            .dae()
            .unknowns
            .iter()
            .position(|u| u == output)
            .ok_or_else(|| ModelError::Numeric(format!("unknown output '{output}'")))?;
        let arc = self.context_arc();
        let pnames = self.cdc().param_names(&arc.lock().unwrap());
        let mut cols = Vec::with_capacity(subset.len());
        for name in &subset {
            let col = pnames
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| ModelError::Numeric(format!("unknown parameter '{name}'")))?;
            cols.push(col);
        }
        self.cdc()
            .ensure_hessian(&mut arc.lock().unwrap(), self.dae());
        let h = log_stage!(
            "sens/hessian_adjoint",
            self.cdc().hessian(metric, &cols, &x, &p, t)
        );
        if h.is_empty() {
            return Err(ModelError::Numeric(
                "hessian: singular Jacobian".to_string(),
            ));
        }
        Ok(h)
    }

    pub fn solve_dc(
        &self,
        p: Vec<f64>,
        x0: Option<Vec<f64>>,
        tol: f64,
        max_iter: usize,
        device_limiting: Option<bool>,
        line_search: Option<bool>,
        gmin_continuation: Option<bool>,
        source_continuation: Option<bool>,
        companion_continuation: Option<bool>,
        node_adaptive: Option<bool>,
        pseudo_transient: Option<bool>,
        partition: Option<bool>,
        nodeset: Option<Vec<(usize, f64)>>,
        reltol: Option<f64>,
        abstol: Option<f64>,
        vntol: Option<f64>,
    ) -> Result<Vec<f64>, ModelError> {
        let x0 = x0.unwrap_or_default();
        let mut tricks = sane_solve::SolverTricks::default();
        if let Some(v) = device_limiting {
            tricks.device_limiting = v;
        }
        if let Some(v) = line_search {
            tricks.line_search = v;
        }
        if let Some(v) = gmin_continuation {
            tricks.gmin_continuation = v;
        }
        if let Some(v) = source_continuation {
            tricks.source_continuation = v;
        }
        if let Some(v) = companion_continuation {
            tricks.companion_continuation = v;
        }
        if let Some(v) = node_adaptive {
            tricks.node_adaptive = v;
        }
        if let Some(v) = pseudo_transient {
            tricks.pseudo_transient = v;
        }
        if let Some(v) = partition {
            tricks.partition = v;
        }
        let mut conv = sane_solve::Convergence::from_tol(tol);
        if let Some(v) = reltol {
            conv.reltol = v;
        }
        if let Some(v) = abstol {
            conv.abstol = v;
        }
        if let Some(v) = vntol {
            conv.vntol = v;
        }
        let (x, converged, iters) = match nodeset {
            Some(ns) if !ns.is_empty() => self
                .cdc()
                .solve_dc_nodeset_with(&p, &ns, conv, max_iter, tricks),
            _ => self
                .cdc()
                .solve_dc_conv_with(&p, &x0, conv, max_iter, tricks),
        };
        if converged {
            Ok(x)
        } else {
            Err(ModelError::Numeric(format!(
                "DC Newton did not converge in {iters} iterations"
            )))
        }
    }

    /// The gmin at which the most recent DC solve held, or `None` if it reached
    /// the true `GMIN_DC` floor. A `Some(g)` means the returned (converged=true)
    /// operating point is gmin-regularized and physically suspect (issue #54).
    /// Query right after a DC solve (`solve_dc` / any analysis that solves the OP).
    pub fn regularized_at_gmin(&self) -> Option<f64> {
        // Either cause -- held above the floor, or the floor sets an unknown's
        // value -- makes the point depend on the regularization; one flag for
        // callers who only ask that.
        self.cdc().last_regularized_gmin().or_else(|| {
            self.cdc()
                .last_gmin_dominance()
                .map(|_| sane_core::constants::GMIN_DC)
        })
    }

    /// `(unknown index, relative first-order shift)` of the worst node voltage
    /// the gmin shunt sets in the most recent DC solve, or `None` when the
    /// circuit sets them all. The other half of `regularized_at_gmin`: that one
    /// says the answer depends on the regularization, this one says which node
    /// and by how much removing the shunt would move it.
    pub fn gmin_dominance(&self) -> Option<(usize, f64)> {
        self.cdc().last_gmin_dominance()
    }

    /// Linearised descriptor state-space `(states, E, A, B, C, D)` at `(x, p)`.
    #[allow(clippy::type_complexity)]
    pub fn state_space_raw(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> (
        Vec<String>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        f64,
    ) {
        let arc = self.context_arc();
        let mut c = arc.lock().unwrap();
        let (e, a, b, cc, d) =
            state_space_on_dae(&mut c, self.dae(), self.cdc(), input, out_idx, &x, &p);
        (self.dae().unknowns.clone(), e, a, b, cc, d)
    }

    /// Temperature sweep of the unknown at `out_idx` over `[tstart, tstop]` degC.
    pub fn temp_sweep_raw(
        &self,
        out_idx: usize,
        p0: Vec<f64>,
        tstart: f64,
        tstop: f64,
        points: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
        self.ensure_no_delays("temp_sweep")?;
        let arc = self.context_arc();
        let pnames = self.cdc().param_names(&arc.lock().unwrap());
        temp_sweep_on_dae(self.cdc(), &pnames, out_idx, &p0, tstart, tstop, points)
            .map_err(ModelError::Numeric)
    }

    /// Dominant-pole model-order reduction of `input -> out_idx` to `order` poles,
    /// at `(x, p)`. Returns `(freqs, full_db, reduced_db, poles, zeros, max_err_db)`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn model_reduce_raw(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        order: usize,
        fstart: f64,
        fstop: f64,
        points: usize,
    ) -> Result<
        (
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<(f64, f64)>,
            Vec<(f64, f64)>,
            f64,
        ),
        ModelError,
    > {
        let arc = self.context_arc();
        let mut c = arc.lock().unwrap();
        let (f, full, red, kp, kz, err) = model_reduce_on_dae(
            &mut c,
            self.dae(),
            self.cdc(),
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
        Ok((
            f,
            full,
            red,
            kp.into_iter().map(|r| (r[0], r[1])).collect(),
            kz.into_iter().map(|r| (r[0], r[1])).collect(),
            err,
        ))
    }
}
