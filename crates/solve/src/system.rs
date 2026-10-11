//! The system at a point: its currents and charges, the residual, the
//! Jacobians `G = dI/dx` and `C = dQ/dx`, and the small-signal conductance
//! the small-signal analyses linearize. Each is one evaluation of a program
//! (see [`Bound`](crate::Bound)).

use sane_core::constants::GMIN_DC;

use crate::{CompiledDc, Triplets};

impl CompiledDc {
    /// The currents `I(x, p, t)`: the residual at rest.
    pub fn currents(&self, x: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        self.eval_once(&self.tape_res_dc, x, p, t, &[])
    }

    /// The charges `Q(x, p)`.
    pub fn charges(&self, x: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        self.currents_charges(x, p, t).1
    }

    /// The currents and the charges in one evaluation.
    pub(crate) fn currents_charges(&self, x: &[f64], p: &[f64], t: f64) -> (Vec<f64>, Vec<f64>) {
        let mut out = self.eval_once(&self.tape_tran_res, x, p, t, &[]);
        let q = out.split_off(self.n);
        (out, q)
    }

    /// The residual `F = I(x, t) + C(x) x'` at the state `x` moving at the
    /// rate `xdot` (every row is affine in the rate).
    pub fn residual(&self, x: &[f64], xdot: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let mut f = self.currents(x, p, t);
        for (r, c, v) in self.jacobian_q_x_sparse(x, p, t).iter() {
            f[r] += v * xdot.get(c).copied().unwrap_or(0.0);
        }
        f
    }

    /// Sparse `G = dI/dx`.
    pub fn jacobian_i_x_sparse(&self, x: &[f64], p: &[f64], t: f64) -> Triplets {
        Triplets {
            rows: self.jx_rows.clone(),
            cols: self.jx_cols.clone(),
            vals: self
                .eval_once(&self.tape_step_dc, x, p, t, &[])
                .split_off(self.n),
        }
    }

    /// Sparse `C = dQ/dx`.
    pub fn jacobian_q_x_sparse(&self, x: &[f64], p: &[f64], t: f64) -> Triplets {
        Triplets {
            rows: self.jxd_rows.clone(),
            cols: self.jxd_cols.clone(),
            vals: self.eval_once(&self.tape_c, x, p, t, &[]),
        }
    }

    /// Sparse regularized small-signal conductance `G + GMIN_DC*I` at the
    /// operating point `(x, p)` (`t = 0`), the gmin shunt appended on every
    /// diagonal: the same regularization the DC solve carried, so every
    /// small-signal analysis (AC, noise, poles and zeros, reduction)
    /// linearizes the system that was actually solved. It also keeps the
    /// pencil `G + sC` regular where an unknown is structurally decoupled at
    /// the bias (the branch current of a grounded thermal node, say).
    pub fn system_triplets_dc(&self, x: &[f64], p: &[f64]) -> Triplets {
        let mut g = self.jacobian_i_x_sparse(x, p, 0.0);
        g.rows.extend(0..self.n);
        g.cols.extend(0..self.n);
        g.vals.extend(std::iter::repeat_n(GMIN_DC, self.n));
        g
    }
}
