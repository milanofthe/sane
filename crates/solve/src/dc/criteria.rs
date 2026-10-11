//! When a DC Newton has converged: the per-component criterion, the
//! SPICE-relative KCL test, the residual's own rounding, and what the
//! gmin shunt sets at the solution.

use super::*;

impl CompiledDc {
    /// The shared convergence contract over this system's unknown kinds (see
    /// [`newton::Criterion`]).
    pub(crate) fn criterion<'a>(&'a self, c: &Convergence) -> newton::Criterion<'a> {
        newton::Criterion::new(&self.kinds, *c)
    }

    /// SPICE-faithful *relative* residual test for the KCL node rows: a node is
    /// converged when its current imbalance `|F_i + gmin*x_i|` is within
    /// `reltol * sum|I_branch| + abstol`, where `sum|I_branch|` is the magnitude
    /// of the branch currents into the node (from the per-term `tape_iscale`
    /// output `terms`). Branch / constraint rows keep the absolute `vntol` floor.
    /// This matches SPICE, which scales the per-node current tolerance by the
    /// currents actually flowing, instead of demanding a fixed absolute floor on
    /// every node regardless of its operating current.
    pub(crate) fn residual_relative_ok(
        &self,
        res: &[f64],
        terms: &[f64],
        x: &[f64],
        gmin: f64,
        c: &Convergence,
    ) -> bool {
        for i in 0..self.n {
            let r = res[i] + gmin * x[i];
            if !r.is_finite() {
                return false;
            }
            let floor = if i < self.n_nodes {
                let (off, len) = self.iscale_rows[i];
                let iscale: f64 = terms[off..off + len].iter().map(|t| t.abs()).sum();
                c.reltol * iscale + c.abstol
            } else {
                self.criterion(c).residual_floor(i)
            };
            if r.abs() > floor {
                return false;
            }
        }
        true
    }

    #[inline]
    pub(super) fn is_node(&self, i: usize) -> bool {
        self.kinds[i] == sane_dae::UnknownKind::NodeVoltage
    }

    /// Damped Newton at a fixed `gmin` (at rest, `t = 0`), warm-started from
    /// `x_init`. Returns `(x, converged, iterations)`. `tricks` gates the per-step
    /// shaping (uniform clamp, device limiting, line search, partitioning).
    /// The rounding the shunted residual `res + gmin x` at `x` is computed
    /// with: per row (into `terms`) the size of the terms it sums, `|G| |x|`
    /// over the Jacobian values `g` plus the shunt's, `NEWTON_ROUNDOFF` ulps
    /// of them; returned, whether every row is within it (solved to machine
    /// precision) and that floor of the residual's 2-norm.
    pub(super) fn rounding(
        &self,
        res: &[f64],
        g: &[f64],
        x: &[f64],
        gmin: f64,
        terms: &mut Vec<f64>,
    ) -> (bool, f64) {
        terms.clear();
        terms.extend(
            x.iter()
                .map(|v| NEWTON_ROUNDOFF * f64::EPSILON * gmin * v.abs()),
        );
        for (e, v) in g.iter().enumerate().take(self.nnz_x) {
            terms[self.jx_rows[e]] +=
                NEWTON_ROUNDOFF * f64::EPSILON * v.abs() * x[self.jx_cols[e]].abs();
        }
        let rounded =
            (res.iter().zip(x).zip(terms.iter())).all(|((r, xi), t)| (r + gmin * xi).abs() <= *t);
        (rounded, norm2(terms))
    }

    /// Record whether the converged point is set by the gmin shunt rather than
    /// by the circuit, and say so.
    ///
    /// The regularization puts a `GMIN_DC` conductance from every unknown to
    /// ground. Against the currents a circuit actually carries that is nothing;
    /// where it is not, the reported value is the shunt's answer, not the
    /// circuit's. `I1 0 1 1u / R1 1 0 1e15` is the plain case: gmin outweighs
    /// R1 by three decades, so the solve converges cleanly at the floor and
    /// reports 999 kV where the circuit says 1 GV. Nothing else catches that --
    /// the hold flag only fires when the solve could not REACH the floor, which
    /// here it does on the first Newton.
    pub(super) fn record_gmin_dominance(&self, x: &[f64], p: &[f64], converged: bool, vntol: f64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.dc.last_gmin_row.store(usize::MAX, Relaxed);
        if !converged {
            return;
        }
        let Some((row, shift)) = self.gmin_dominance(x, p, vntol) else {
            return;
        };
        self.dc.last_gmin_row.store(row, Relaxed);
        self.dc.last_gmin_share.store(shift.to_bits(), Relaxed);
        // Deliberately NOT folded into `last_gmin_hold`: that flag drives the
        // settle fallback in the analysis layer, which integrates the circuit's
        // own dynamics to escape a gmin-held basin. Dominance is not a basin
        // problem -- no amount of settling moves a node whose only path to
        // ground is the shunt -- and firing that fallback on it cost a
        // transport-delay DC solve four minutes. Callers that want one answer
        // to "does this point depend on gmin?" combine the two themselves.
        sane_core::log::warning(&format!(
            "DC: node unknown #{row} is set by the gmin={GMIN_DC:.0e} shunt (removing it would              shift the value by {:.0}% to first order); the reported voltage is the              regularization's answer, not the circuit's",
            shift * 100.0
        ));
    }

    /// The node voltage the gmin shunt sets rather than the circuit, if any:
    /// `(unknown index, relative first-order shift)` for the worst node.
    ///
    /// The converged point solves `F(x) + g*x = 0`, so its dependence on the
    /// shunt is exact: `(J + g*I) dx/dg = -x`, one solve with the same
    /// regularized Jacobian Newton just converged on. `g*|dx_i/dg|` is the
    /// first-order shift of unknown `i` were the shunt removed -- the error
    /// gmin imprints on the answer -- and a node whose relative shift exceeds
    /// [`GMIN_DOMINANCE_FRAC`] is flagged. Asking about the VALUE is what a
    /// current-share heuristic got wrong: at a node held by a source and
    /// probed by nothing, the shunt carries 100% of the (pico-ampere) row
    /// current yet shifts the pinned voltage by exactly zero.
    ///
    /// Only the leading node-voltage block is examined. A branch current can
    /// always be "set" by the shunt -- the leakage `g*v` IS the shunt's own
    /// current, picoamperes by construction -- while a dominated node voltage
    /// is `i/g`, unboundedly wrong; the asymmetry is structural, so the check
    /// follows it. Shifts below `vntol` are ignored: a node resting at zero
    /// behind a capacitor is an open circuit, not a wrong answer.
    pub(super) fn gmin_dominance(&self, x: &[f64], p: &[f64], vntol: f64) -> Option<(usize, f64)> {
        let n = self.n;
        // Through the compiled step path (the reused symbolic pattern and
        // whichever backend it picked, the symmetric LDLT on a power grid):
        // the one-shot KLU this used to build cost 50x the Newton's own
        // factorization on a 45k-node grid (580 ms against 12 ms), on every
        // converged operating point.
        let jv = &self.jacobian_i_x_sparse(x, p, 0.0).vals;
        let mut valbuf = Vec::new();
        let mut fac: Option<sparse::Refactorable> = None;
        // (J + g*I) s = x  =>  dx/dg = -s
        let mut s = vec![0.0; n];
        if !self.solve_step(jv, GMIN_DC, x, &mut s, &mut valbuf, &mut fac) {
            return None;
        }
        let mut worst: Option<(usize, f64)> = None;
        for i in (0..n).filter(|&i| self.is_node(i)) {
            let dv = GMIN_DC * s[i].abs();
            if dv <= vntol {
                continue;
            }
            let shift = dv / x[i].abs().max(vntol);
            if shift > GMIN_DOMINANCE_FRAC && worst.is_none_or(|(_, w)| shift > w) {
                worst = Some((i, shift));
            }
        }
        worst
    }
}
