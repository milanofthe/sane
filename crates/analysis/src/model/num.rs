//! Numeric passthroughs and DC-level entry points on [`Model`]: residual /
//! Jacobian evaluation, DC solve, exact first/second-order DC sensitivities,
//! and the raw state-space / temperature-sweep / model-reduction kernels.

use crate::model::{Model, ModelError};

/// A sparse matrix as `(rows, cols, values)`.
type Coo = (Vec<usize>, Vec<usize>, Vec<f64>);

impl Model {
    /// The currents `I(x, p, t)`. Every row reads `I(x, t) + d/dt Q(x)`.
    pub fn currents(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> Result<Vec<f64>, ModelError> {
        self.inner.bound(&p)?;
        Ok(self.cdc().currents(&x, &p, t))
    }

    /// The charges `Q(x, p)`.
    pub fn charges(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> Result<Vec<f64>, ModelError> {
        self.inner.bound(&p)?;
        Ok(self.cdc().charges(&x, &p, t))
    }

    /// The residual `F = I(x, t) + C(x) x'` at the state `x` moving at the
    /// rate `xdot`.
    pub fn residual(
        &self,
        x: Vec<f64>,
        xdot: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> Result<Vec<f64>, ModelError> {
        self.inner.bound(&p)?;
        Ok(self.cdc().residual(&x, &xdot, &p, t))
    }

    /// `G = dI/dx` as a dense matrix.
    pub fn jacobian_i_x(
        &self,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> Result<Vec<Vec<f64>>, ModelError> {
        self.inner.bound(&p)?;
        Ok(self.cdc().jacobian_i_x(&x, &p, t))
    }

    /// `C = dQ/dx` as a dense matrix.
    pub fn jacobian_q_x(
        &self,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> Result<Vec<Vec<f64>>, ModelError> {
        self.inner.bound(&p)?;
        Ok(self.cdc().jacobian_q_x(&x, &p, t))
    }

    /// Sparse `G = dI/dx` as `(rows, cols, values)` (COO).
    pub fn jacobian_i_x_sparse(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> Result<Coo, ModelError> {
        self.inner.bound(&p)?;
        Ok(self.cdc().jacobian_i_x_sparse(&x, &p, t))
    }

    /// Sparse `C = dQ/dx` as `(rows, cols, values)` (COO).
    pub fn jacobian_q_x_sparse(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> Result<Coo, ModelError> {
        self.inner.bound(&p)?;
        Ok(self.cdc().jacobian_q_x_sparse(&x, &p, t))
    }

    /// Sparse `dI/dp` as `(rows, cols, values)` (COO).
    pub fn jacobian_i_p_sparse(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> Result<Coo, ModelError> {
        self.inner.bound(&p)?;
        Ok(self.jacobian_p_sparse(&x, &p, t).0)
    }

    /// Sparse `dQ/dp` as `(rows, cols, values)` (COO).
    pub fn jacobian_q_p_sparse(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> Result<Coo, ModelError> {
        self.inner.bound(&p)?;
        Ok(self.jacobian_p_sparse(&x, &p, t).1)
    }

    /// `(dI/dp, dQ/dp)`, the parameter Jacobian built on first use.
    pub(crate) fn jacobian_p_sparse(&self, x: &[f64], p: &[f64], t: f64) -> (Coo, Coo) {
        let arc = self.context_arc();
        self.cdc()
            .ensure_param_jac(&mut arc.lock().unwrap(), self.dae());
        self.cdc().jacobian_p_sparse(x, p, t)
    }

    /// Number of structural nonzeros in the sparse `G` pattern.
    pub fn nnz(&self) -> usize {
        self.cdc().nnz()
    }

    /// Schur partition sizes `(linear_block, nonlinear_block)`, or `None`.
    pub fn partition_sizes(&self) -> Option<(usize, usize)> {
        self.cdc().partition_sizes()
    }

    /// Exact input-coupling vector `dI/d(input)` for the named source, by
    /// symbolic differentiation (autodiff), evaluated at `(x, p, t)`.
    pub fn jacobian_i_input(
        &self,
        input: &str,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> Result<Vec<f64>, ModelError> {
        self.inner.bound(&p)?;
        let arc = self.context_arc();
        let mut c = arc.lock().unwrap();
        let dae = self.dae();
        let mut b = vec![0.0; dae.dim()];
        for s in self.inner.drive(&mut c, input)? {
            let col = self.cdc().jacobian_i_input(&mut c, dae, s, &x, &p, t);
            b.iter_mut().zip(col).for_each(|(b, v)| *b += v);
        }
        Ok(b)
    }
}
