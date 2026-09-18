//! Transient analyses on [`Model`]: the adaptive solve, exact forward
//! (augmented-system) sensitivities, and the discrete transient adjoint.

use std::collections::HashMap;

use rsdag::Node;
use sane_core::constants::{DC_OP_MAXIT, DC_OP_TOL};
use sane_core::log;
use sane_dae::augment_with_scaled_sensitivities;
use sane_solve::CompiledDc;

use crate::model::{Model, ModelError};

impl Model {
    /// Exact forward transient sensitivity: integrate the circuit and the
    /// sensitivity systems `dx(t)/dp` for each parameter in `subset` together.
    /// Returns `(n, traj)` (see the analysis layer for the slice layout).
    pub fn transient_sensitivity(
        &self,
        subset: Vec<String>,
        t_eval: Vec<f64>,
        rtol: f64,
        atol: f64,
        values: Option<HashMap<String, f64>>,
    ) -> Result<(usize, Vec<Vec<f64>>), ModelError> {
        self.ensure_no_delays("transient_sensitivity")?;
        let mut task = log::task(
            "SENS-TRANSIENT",
            "sens_tran",
            &format!("(params: {}, points: {})", subset.len(), t_eval.len()),
        );
        let arc = self.context_arc();
        let mut c = arc.lock().unwrap();
        let dae = self.dae();
        let n = dae.dim();
        // dF/dp must exist before the per-parameter state-sensitivity solves below
        // (issue #38); otherwise dx/dp collapses silently to zero.
        self.cdc().ensure_param_jac(&mut c, dae);
        let pnames = self.cdc().param_names(&c);
        let bound = self.values();
        let val_of = |nm: &str| -> f64 {
            values
                .as_ref()
                .and_then(|v| v.get(nm).copied())
                .or_else(|| bound.get(nm).copied())
                .unwrap_or(0.0)
        };
        let p_base: Vec<f64> = pnames.iter().map(|nm| val_of(nm)).collect();
        let (op, conv, _) = self.cdc().solve_dc(&p_base, &[], DC_OP_TOL, DC_OP_MAXIT);
        if !conv {
            return Err(ModelError::Numeric(
                "transient_sensitivity: DC operating point did not converge".to_string(),
            ));
        }
        let mut x0_aug = op.clone();
        // Integrate log-parameter-scaled sensitivity states u = |p0|·dx/dp:
        // they live on circuit magnitudes, so the scalar atol/rtol error
        // control keeps sane step sizes (raw dx/dp states can be ~1/p0 times
        // larger and crush the step at every zero crossing). Unscaled dx/dp
        // is restored below, so the returned layout is unchanged.
        let mut subset_syms = Vec::with_capacity(subset.len());
        let mut scales = Vec::with_capacity(subset.len());
        for name in &subset {
            let col = pnames
                .iter()
                .position(|nm| nm == name)
                .ok_or_else(|| ModelError::Numeric(format!("unknown parameter '{name}'")))?;
            let s0 = self.cdc().state_sensitivity(col, &op, &p_base, 0.0);
            if s0.is_empty() {
                return Err(ModelError::Numeric(
                    "transient_sensitivity: singular Jacobian".to_string(),
                ));
            }
            let p0 = val_of(name);
            let k = if p0 != 0.0 { p0.abs() } else { 1.0 };
            scales.push(k);
            x0_aug.extend(s0.iter().map(|v| v * k));
            let e = c.sym(name);
            match c.node(e) {
                Node::Symbol(sy) => subset_syms.push((*sy, k)),
                _ => return Err(ModelError::Numeric(format!("'{name}' is not a symbol"))),
            }
        }
        let aug = augment_with_scaled_sensitivities(&mut c, dae, &subset_syms);
        let aug_cdc = CompiledDc::new(&mut c, &aug);
        let p_aug: Vec<f64> = aug_cdc
            .param_names(&c)
            .iter()
            .map(|nm| val_of(nm))
            .collect();
        drop(c);
        let mut traj = aug_cdc
            .solve_transient(
                sane_solve::TransientMethod::Esdirk32,
                &p_aug,
                &x0_aug,
                &t_eval,
                rtol,
                atol,
                None,
            )
            .map_err(ModelError::Numeric)?;
        for row in &mut traj {
            for (ki, k) in scales.iter().enumerate() {
                for v in &mut row[n * (ki + 1)..n * (ki + 2)] {
                    *v /= k;
                }
            }
        }
        task.finish(format!("augmented dim: {}", aug.dim()));
        Ok((n, traj))
    }

    /// Fixed-grid ESDIRK32 transient on the exact grid `t_eval` (the forward
    /// pass [`transient_adjoint`](Self::transient_adjoint) differentiates). `p` is
    /// the parameter vector in `params()` order; an empty `x0` starts from the
    /// DC operating point. Returns the state at every grid point.
    pub fn solve_transient_grid(
        &self,
        p: Vec<f64>,
        x0: Option<Vec<f64>>,
        t_eval: Vec<f64>,
        dc_guess: Option<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, ModelError> {
        self.cdc()
            .solve_transient_grid(
                &p,
                &x0.unwrap_or_default(),
                &t_eval,
                &dc_guess.unwrap_or_default(),
            )
            .map_err(ModelError::Numeric)
    }

    /// Discrete transient adjoint (VJP): the gradient of a scalar objective
    /// `L(x_0..x_N)` over the fixed-grid ESDIRK32 trajectory on `t_eval`,
    /// w.r.t. EVERY parameter -- from one forward transient plus `S-1` backward
    /// transposed stage solves per step. Cost is independent of the parameter count,
    /// the complement of [`transient_sensitivity`](Self::transient_sensitivity)
    /// (whose cost is linear in the parameters but yields whole trajectories).
    ///
    /// `cotangent[k]` is `dL/dx_k` (length `dim`); the initial state is the DC
    /// operating point and its parameter dependence is included. The forward
    /// trajectory this gradient belongs to is exactly
    /// `CompiledDc::solve_transient_grid` on the same grid. Returns
    /// `(param_names, dL/dp)`.
    pub fn transient_adjoint(
        &self,
        t_eval: Vec<f64>,
        cotangent: Vec<Vec<f64>>,
        values: Option<HashMap<String, f64>>,
        dc_guess: Option<Vec<f64>>,
    ) -> Result<(Vec<String>, Vec<f64>), ModelError> {
        self.ensure_no_delays("transient_adjoint")?;
        let mut task = log::task(
            "TRANSIENT-ADJOINT",
            "adjoint",
            &format!("(points: {})", t_eval.len()),
        );
        let arc = self.context_arc();
        let mut c = arc.lock().unwrap();
        self.cdc().ensure_param_jac(&mut c, self.dae());
        let pnames = self.cdc().param_names(&c);
        drop(c);
        let bound = self.values();
        let val_of = |nm: &str| {
            values
                .as_ref()
                .and_then(|v| v.get(nm).copied())
                .or_else(|| bound.get(nm).copied())
                .unwrap_or(0.0)
        };
        let p: Vec<f64> = pnames.iter().map(|nm| val_of(nm)).collect();
        let g = self
            .cdc()
            .transient_adjoint(&p, &[], &t_eval, &cotangent, &dc_guess.unwrap_or_default())
            .map_err(ModelError::Numeric)?;
        task.finish(format!("params: {}", pnames.len()));
        Ok((pnames, g))
    }

    /// Append the topological cause to a transient failure, when there is one.
    /// An index-2 deck fails in the step-size machinery (the constraint has no
    /// truncation error to control), which reads as a bare underflow unless the
    /// engine names the loop or cutset behind it.
    fn explain_failure(&self, err: String) -> String {
        let rep = &self.inner.index2;
        if !rep.is_index2() {
            return err;
        }
        format!(
            "{err} -- this deck is index 2 ({}); its constraint is differentiated, which the integrator cannot resolve. Break it with the parasitic that exists in reality: a series resistance in the loop, a shunt across the cutset.",
            rep.summary()
        )
    }
    /// Transient solve over `t_eval` (ESDIRK32 on the Rust DAE). `x0` defaults
    /// to the DC operating point. Returns one state vector per time point.
    /// The single transient entry point: integrate with `method` over `t_eval`
    /// from `x0` (the consistent DC start if `None`), returning the raw state
    /// rows. The labeled [`Model::transient`] convenience delegates here.
    pub fn solve_transient(
        &self,
        method: sane_solve::TransientMethod,
        p: Vec<f64>,
        t_eval: Vec<f64>,
        x0: Option<Vec<f64>>,
        rtol: f64,
        atol: f64,
        dt_max: Option<f64>,
    ) -> Result<Vec<Vec<f64>>, ModelError> {
        let x0 = x0.unwrap_or_default();
        self.cdc()
            .solve_transient(method, &p, &x0, &t_eval, rtol, atol, dt_max)
            // a step-size failure on an index-2 deck has a topological cause;
            // say so rather than leaving the caller with a bare underflow
            .map_err(|e| ModelError::Numeric(self.explain_failure(e)))
    }
}
