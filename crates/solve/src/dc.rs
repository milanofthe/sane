//! The DC operating-point solve: damped Newton with per-component convergence,
//! and the robust continuation cascade behind it (gmin stepping, source
//! stepping, per-device companion homotopy, node-adaptive damping, and
//! pseudo-transient relaxation), each stage a toggleable
//! [`SolverTricks`](crate::SolverTricks) trick.

use sane_core::constants::*;
use sane_core::log_stage;

use crate::{limiting, newton, sparse, CompiledDc, Convergence, LinCache, SolverTricks, Symbolic};

/// A `.nodeset` pin: stiff springs `g*(x_i - target_i)` on selected unknown rows
/// for the first phase of a node-set solve. Forcing the pinned unknowns toward
/// their targets breaks the symmetry of a bistable circuit (a latch, an
/// identical differential pair) that otherwise sits on the unstable symmetric
/// root; phase 2 removes the pin and re-solves freely, so the final operating
/// point is exact -- the pin shapes only *which* root is found.
struct Pin {
    idx: Vec<usize>,
    target: Vec<f64>,
    g: f64,
}

impl CompiledDc {
    /// Factor `J + gmin*I` for the Jacobian values in `jac` (the `out[n..]`
    /// slice of a step evaluation) and solve `(J + gmin*I) dx = rhs`. Returns
    /// the solution vector, or `None` if the factorization is singular.
    ///
    /// `fac` is the caller loop's factorization cache: the first call factors
    /// with full pivoting, subsequent calls replay the frozen pivot sequence
    /// (KLU numeric-only refactor, no DFS / pivot search) on the fresh values --
    /// the dominant per-iteration saving of the KLU backend. Pass a fresh
    /// `None`-initialized slot per Newton loop; it must not be shared across
    /// loops with different augmentation patterns.
    fn solve_step<'a>(
        &'a self,
        jac: &[f64],
        gmin: f64,
        rhs: &[f64],
        valbuf: &mut Vec<f64>,
        fac: &mut Option<sparse::Refactorable<'a>>,
    ) -> Option<Vec<f64>> {
        let diag = vec![gmin; self.n];
        self.solve_with_diag(jac, &diag, rhs, valbuf, fac)
    }

    /// Solve `(J + diag(d)) dx = rhs` with the Jacobian nonzeros `jac` (in the
    /// compiled order) and a per-row diagonal shunt `d`: the reused symbolic
    /// pattern with a numeric-only refactor, or, for a degenerate pattern the
    /// symbolic analysis rejected, a one-shot triplet factorization.
    fn solve_with_diag<'a>(
        &'a self,
        jac: &[f64],
        diag: &[f64],
        rhs: &[f64],
        valbuf: &mut Vec<f64>,
        fac: &mut Option<sparse::Refactorable<'a>>,
    ) -> Option<Vec<f64>> {
        match &self.symbolic {
            Some(sym) => {
                // values: jacobian nonzeros, then the full diagonal.
                valbuf.clear();
                valbuf.extend_from_slice(jac);
                valbuf.extend_from_slice(diag);
                let f = fac.get_or_insert_with(|| sym.pattern.factorizer());
                if !f.factor(valbuf, self.tricks.row_equilibration) {
                    return None;
                }
                f.solve(rhs)
            }
            None => self.solve_triplets(jac, &[], diag, rhs),
        }
    }

    /// One-shot triplet factorization of `J + extra + diag(d)` (the fallback
    /// for a pattern the symbolic analysis rejected as structurally singular).
    fn solve_triplets(
        &self,
        jac: &[f64],
        extra: &[(usize, usize, f64)],
        diag: &[f64],
        rhs: &[f64],
    ) -> Option<Vec<f64>> {
        let n = self.n;
        let mut rows = self.jx_rows.clone();
        let mut cols = self.jx_cols.clone();
        let mut vals = jac.to_vec();
        for &(r, c, v) in extra {
            rows.push(r);
            cols.push(c);
            vals.push(v);
        }
        for (i, &d) in diag.iter().enumerate() {
            rows.push(i);
            cols.push(i);
            vals.push(d);
        }
        sparse::factor_triplets_both(n, &rows, &cols, &vals)?.solve(rhs)
    }

    /// Residual half of the convergence test: every row's residual `F_i + gmin*x_i`
    /// is within its absolute floor -- `abstol` (current) on a KCL node row,
    /// `vntol` (voltage) on a KVL branch row. `res` is the raw residual `F(x)`.
    pub(crate) fn residual_converged(
        &self,
        res: &[f64],
        x: &[f64],
        gmin: f64,
        c: &Convergence,
    ) -> bool {
        self.criterion(c).residual_ok(res, x, gmin)
    }

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
    fn is_node(&self, i: usize) -> bool {
        self.kinds[i] == sane_dae::UnknownKind::NodeVoltage
    }

    /// Update half of the convergence test: every unknown's Newton step is within
    /// `reltol*|x_i| + floor`, the floor being `vntol` (voltage) for a node-voltage
    /// unknown and `abstol` (current) for a branch-current unknown.
    fn update_converged(&self, dx: &[f64], x: &[f64], c: &Convergence) -> bool {
        self.criterion(c).update_ok(dx, x)
    }

    /// Damped Newton at a fixed `gmin` (`xdot = 0`, `t = 0`), warm-started from
    /// `x_init`. Returns `(x, converged, iterations)`. `tricks` gates the per-step
    /// shaping (uniform clamp, device limiting, line search, partitioning).
    fn newton(
        &self,
        p: &[f64],
        x_init: &[f64],
        gmin: f64,
        conv: &Convergence,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        let n = self.n;
        let xdot = vec![0.0; n];
        let mut x = if x_init.len() == n {
            x_init.to_vec()
        } else {
            vec![0.0; n]
        };

        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        let (mut inb, mut wb, mut ob) = (Vec::new(), Vec::new(), Vec::new()); // line search
        let mut valbuf = Vec::new();
        // Per-iteration scratch reused across the whole solve (no realloc per step).
        let mut rhs = vec![0.0; n];
        let mut step = vec![0.0; n];
        // KLU factorization cache: full pivoting on the first iteration, frozen
        // pivot replay (numeric-only refactor) on the rest of this Newton loop.
        let mut fac: Option<sparse::Refactorable> = None;

        // Cached linear-block factorization for this (fixed) gmin, built lazily
        // on the first iteration once we have the Jacobian values.
        let mut lin_cache: Option<LinCache> = None;
        let mut use_partition = self.partition.is_some() && tricks.partition;

        // Diagnostic: per-iteration residual / largest-step trace, for
        // classifying non-convergence (overshoot vs oscillation vs stall).
        let trace = sane_core::config().dc_trace;

        // Prolog split: the parameter vector is fixed across this Newton loop,
        // so the tapes' parameter-pure prefixes are evaluated once here and
        // only the main phase runs per iteration (the prolog reads no state,
        // so the initial `x` in `inputs` is irrelevant to it). The tokens pin
        // each buffer's backend for the episode.
        self.fill_inputs(&x, &xdot, p, 0.0, &mut inputs);
        let mut step_tok = self.tape_step.eval_prolog(&inputs, &mut work);
        let mut res_tok = self.tape_res.eval_prolog(&inputs, &mut wb);
        let mut stall = newton::StallGuard::new();

        for it in 0..max_iter {
            self.fill_inputs(&x, &xdot, p, 0.0, &mut inputs);
            self.tape_step
                .eval_main(&mut step_tok, &inputs, &mut work, &mut out);
            // Residual of the *homotopy* system F(x) + gmin*x: the diagonal
            // gmin term must enter the norm so the line search measures the
            // problem we are actually solving.
            let fnorm = shunted_norm(&out[..n], &x, gmin);
            // Per-component residual test (the update half is checked once the
            // step is known, below). `fnorm` is kept for the line search and
            // the stall guard.
            let res_ok = self.residual_converged(&out[..n], &x, gmin, conv);
            if !res_ok && stall.stalled(fnorm) {
                if trace {
                    sane_core::log::debug(&format!("DCTRACE stalled at it={it} fnorm={fnorm:.3e}"));
                }
                return (x, false, it);
            }

            // Newton step: (J + gmin*I) dx = F + gmin*x. With a large constant
            // linear block, solve via the cached-factorization Schur complement
            // (build the cache once for this gmin); otherwise a plain sparse LU.
            for i in 0..n {
                rhs[i] = out[i] + gmin * x[i];
            }
            // Early acceptance without a fresh factorization: once the residual
            // half passes, probe the update half with the PREVIOUS iteration's
            // factors -- one cheap substitution, no refactor. For a linear
            // system the probe step is the exact Newton step; near a nonlinear
            // solution the Jacobian differs by O(|dx|) from the fresh one, the
            // same staleness SPICE's classic last-step update test accepts.
            // This removes the redundant end-of-solve factorization every
            // convergent Newton run otherwise pays (and, together with the
            // identity fast path in `sparse::Refactorable`, halves purely
            // linear solves). The residual test above always judges the true
            // tape residual, so acceptance quality is unchanged.
            if res_ok {
                if let Some(dxp) = fac.as_mut().and_then(|f| f.solve(&rhs)) {
                    if self.update_converged(&dxp, &x, conv) {
                        return (x, true, it);
                    }
                }
            }
            let dx: Vec<f64> = if use_partition {
                let part = self.partition.as_ref().unwrap();
                if lin_cache.is_none() {
                    lin_cache = self.build_lin_cache(part, &out[n..], gmin);
                    if lin_cache.is_none() {
                        use_partition = false; // degenerate: fall back to full LU
                    }
                }
                match lin_cache
                    .as_ref()
                    .and_then(|cache| self.solve_partitioned(cache, part, &out[n..], &rhs))
                {
                    Some(d) => d,
                    None => match self.solve_step(&out[n..], gmin, &rhs, &mut valbuf, &mut fac) {
                        Some(d) => d,
                        None => return (x, res_ok, it),
                    },
                }
            } else {
                match self.solve_step(&out[n..], gmin, &rhs, &mut valbuf, &mut fac) {
                    Some(d) => d,
                    None => return (x, res_ok, it),
                }
            };

            // Converged when both the residual and the proposed update are small.
            // Near the solution the limiting and line search below are inactive, so
            // the raw step `dx` is the applied update -- using it here is exact.
            // The fast path keeps the STRICT (absolute) residual floor: routine
            // solves must converge to a bit-reproducible point so finite-difference
            // derivatives stay smooth. The SPICE-faithful *relative* acceptance is a
            // last-resort, applied only in the node-adaptive fallback once the strict
            // cascade has stalled (see `adaptive_newton`).
            if res_ok && self.update_converged(&dx, &x, conv) {
                return (x, true, it);
            }

            if trace {
                let (idx, mx) =
                    dx.iter()
                        .enumerate()
                        .fold((0usize, 0.0f64), |(bi, bm), (i, &v)| {
                            if v.abs() > bm {
                                (i, v.abs())
                            } else {
                                (bi, bm)
                            }
                        });
                sane_core::log::debug(&format!(
                    "DCTRACE gmin={gmin:.1e} it={it} fnorm={fnorm:.3e} maxdx={mx:.3e}@{idx} x@={:.4}",
                    x.get(idx).copied().unwrap_or(0.0)
                ));
            }

            // No solver-side step limiting: an absolute bound in volts is
            // meaningless for a 400 V converter, for a current, or for a
            // Verilog-A state, and clamping components one by one bends the
            // Newton direction rather than shortening the step. Globalization
            // is the homotopy ladder below; limiting, where a device needs it,
            // belongs in the model (`$limit`, on the device's own scale) --
            // the division VACASK draws.
            step[..n].copy_from_slice(&dx[..n]);

            // Curve-aware per-device limiting (`pnjlim`/`fetlim`): pull the
            // proposed full step back along each controlling voltage's own curve
            // so a junction / channel current cannot overshoot. Path-only -- it
            // redefines the step direction; the line search and convergence test
            // below are unchanged, so the converged fixed point is identical.
            if tricks.device_limiting && !self.limits.is_empty() {
                newton::limit_step(&self.limits, &x, &mut step[..n]);
            }

            // Backtracking line search on |F + gmin*x|: accept the first step
            // that reduces the residual norm. With line search off, take the full
            // (limited) step directly.
            let tries = if tricks.line_search {
                LINE_SEARCH_TRIES
            } else {
                1
            };
            let alpha = newton::backtrack(&mut x, &step, fnorm, tries, |trial| {
                self.fill_inputs(trial, &xdot, p, 0.0, &mut inb);
                self.tape_res
                    .eval_main(&mut res_tok, &inb, &mut wb, &mut ob);
                shunted_norm(&ob, trial, gmin)
            });
            // Composite (Traub) step: a full Newton step was taken, so the
            // iterate is in the contracting regime; one chord step on the
            // factorization just built, from the residual at the new iterate,
            // makes the pair third-order. Residual-only tape, no Jacobian.
            if tricks.composite_step && alpha == 1.0 {
                if let Some(f) = fac.as_mut() {
                    let res_norm = |ob: &[f64], xx: &[f64]| shunted_norm(ob, xx, gmin);
                    self.fill_inputs(&x, &xdot, p, 0.0, &mut inb);
                    self.tape_res
                        .eval_main(&mut res_tok, &inb, &mut wb, &mut ob);
                    let f1 = res_norm(&ob, &x);
                    for i in 0..n {
                        rhs[i] = ob[i] + gmin * x[i];
                    }
                    // Taken whenever it contracts the residual; gating it on
                    // the contraction regime was measured and declined (see
                    // the constants module).
                    if let Some(d2) = f.solve(&rhs) {
                        let mut trial: Vec<f64> = (0..n).map(|i| x[i] - d2[i]).collect();
                        if tricks.device_limiting && !self.limits.is_empty() {
                            trial = limiting::apply(&self.limits, &x, &trial);
                        }
                        self.fill_inputs(&trial, &xdot, p, 0.0, &mut inb);
                        self.tape_res
                            .eval_main(&mut res_tok, &inb, &mut wb, &mut ob);
                        let f2 = res_norm(&ob, &trial);
                        if trace {
                            sane_core::log::debug(&format!(
                                "DCTRACE composite it={it} fnorm={fnorm:.3e} f1={f1:.3e} f2={f2:.3e} {}",
                                if f2 < f1 { "taken" } else { "rejected" }
                            ));
                        }
                        if f2 < f1 {
                            x = trial;
                        }
                    }
                }
            }
        }
        (x, false, max_iter)
    }

    /// Damped Newton with **node-adaptive diagonal loading**: each iteration the
    /// weakly-coupled unknowns (tiny `|J_ii|`, e.g. high-impedance BSIM4 internal
    /// nodes) get their diagonal loaded up to `ADAPT_DIAG_FRAC * max|J_ii|`, so
    /// their otherwise-huge oscillating Newton step is damped while well-coupled
    /// unknowns keep full Newton. The loading is matrix-only (the residual still
    /// uses just `GMIN_DC*x`), so the converged fixed point `F + GMIN_DC*x = 0` is
    /// unshifted -- the damping shapes only the iteration path. Cold-started.
    fn adaptive_newton(
        &self,
        p: &[f64],
        conv: &Convergence,
        mut iters: usize,
    ) -> (Vec<f64>, bool, usize) {
        let n = self.n;
        // Diagonal position of each node within the Jacobian value array, fixed
        // by the sparsity pattern and precomputed once in `CompiledDc`.
        let diag_idx = &self.diag_idx;
        let xdot = vec![0.0; n];
        let mut x = vec![0.0; n];
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        let (mut inb, mut wb, mut ob) = (Vec::new(), Vec::new(), Vec::new());
        let mut valbuf = Vec::new();
        let mut jacbuf: Vec<f64> = Vec::new();
        // Per-iteration scratch reused across the whole solve (no realloc per step).
        let mut rhs = vec![0.0; n];
        let mut step = vec![0.0; n];
        let mut fac: Option<sparse::Refactorable> = None;
        let (mut iterms, mut iwork) = (Vec::new(), Vec::new());
        let trace = sane_core::config().dc_trace;
        // Keep the lowest-residual iterate: once damping has decayed the last step
        // may drift slightly back up, so the best point (not the last) is returned.
        let (mut best_x, mut best_fnorm) = (x.clone(), f64::INFINITY);

        for _ in 0..ADAPT_MAX_ITER {
            iters += 1;
            self.fill_inputs(&x, &xdot, p, 0.0, &mut inputs);
            self.tape_step.eval(&inputs, &mut work, &mut out);
            let fnorm = shunted_norm(&out[..n], &x, GMIN_DC);
            if fnorm < best_fnorm {
                best_fnorm = fnorm;
                best_x = x.clone();
            }
            if trace {
                sane_core::log::debug(&format!("DCADAPT it={iters} fnorm={fnorm:.3e}"));
            }
            // Residual half now; the update half is checked on the (damped) step.
            let res_ok = self.residual_converged(&out[..n], &x, GMIN_DC, conv);
            // Load weak diagonals (matrix only) up to a fraction of the strongest.
            let mut maxd = 0.0f64;
            for &di in diag_idx.iter().flatten() {
                maxd = maxd.max(out[n + di].abs());
            }
            // Decay the damping with the residual: strong far from the solution
            // (stabilizes the weak nodes into the right basin), vanishing linearly
            // as `fnorm -> 0` so the last mile recovers full Newton and closes
            // (a slower `sqrt` decay keeps too much damping to close on these amps).
            let floor = ADAPT_DIAG_FRAC * maxd * fnorm.min(1.0);
            jacbuf.clear();
            jacbuf.extend_from_slice(&out[n..]);
            for &di in diag_idx.iter().flatten() {
                let jii = jacbuf[di].abs();
                if jii < floor {
                    jacbuf[di] += floor - jii; // positive diagonal loading (damping)
                }
            }
            for i in 0..n {
                rhs[i] = out[i] + GMIN_DC * x[i];
            }
            // Early acceptance with the previous iteration's factors (see the
            // fast-path Newton): probe the update half before refactoring.
            if res_ok {
                if let Some(dxp) = fac.as_mut().and_then(|f| f.solve(&rhs)) {
                    if self.update_converged(&dxp, &x, conv) {
                        return (x, true, iters);
                    }
                }
            }
            let dx: Vec<f64> = match self.solve_step(&jacbuf, GMIN_DC, &rhs, &mut valbuf, &mut fac)
            {
                Some(d) => d,
                None => return (x, res_ok, iters),
            };
            if self.update_converged(&dx, &x, conv) {
                if res_ok {
                    return (x, true, iters);
                }
                // Last-resort SPICE-faithful acceptance: the strict cascade has
                // stalled, so accept if the KCL imbalance is small relative to the
                // node branch-current scale (not just the absolute floor).
                if let Some(tape) = &self.tape_iscale {
                    tape.eval(&inputs, &mut iwork, &mut iterms);
                    if self.residual_relative_ok(&out[..n], &iterms, &x, GMIN_DC, conv) {
                        return (x, true, iters);
                    }
                }
            }
            step[..n].copy_from_slice(&dx[..n]);
            // Curve-aware per-device limiting (`pnjlim`/`fetlim`), on the
            // device's own scale. This is the engine's only limiting now that
            // the solver-side clamp is gone, so it applies here too -- in the
            // first Newton, where a junction runaway actually happens. VACASK
            // draws the same line: the model limits on every evaluation, the
            // solver never does.
            if !self.limits.is_empty() {
                newton::limit_step(&self.limits, &x, &mut step[..n]);
            }
            // Backtracking line search on the (true) residual norm.
            newton::backtrack(&mut x, &step, fnorm, LINE_SEARCH_TRIES, |trial| {
                self.fill_inputs(trial, &xdot, p, 0.0, &mut inb);
                self.tape_res.eval(&inb, &mut wb, &mut ob);
                shunted_norm(&ob, trial, GMIN_DC)
            });
        }
        // Not converged to tol: return the best (lowest-residual) iterate, which a
        // late drift after the damping decayed would otherwise have spoiled.
        (best_x, false, iters)
    }

    /// Solve the DC operating point (`xdot = 0`, `t = 0`) with sparse LU and the
    /// default set of convergence aids. Tries a plain damped-Newton solve first;
    /// if it stalls, falls back through the continuation tricks. Returns
    /// `(x, converged, iterations)`. The scalar `tol` is the absolute residual
    /// floor of the per-component criterion ([`Convergence::from_tol`]); use
    /// [`solve_dc_conv`](Self::solve_dc_conv) for full `reltol`/`abstol`/`vntol`
    /// control, or [`solve_dc_with`](Self::solve_dc_with) for explicit
    /// [`SolverTricks`].
    pub fn solve_dc(
        &self,
        p: &[f64],
        x0: &[f64],
        tol: f64,
        max_iter: usize,
    ) -> (Vec<f64>, bool, usize) {
        self.solve_dc_conv_with(p, x0, Convergence::from_tol(tol), max_iter, self.tricks)
    }

    /// DC operating point with an explicit set of [`SolverTricks`] (scalar `tol`,
    /// mapped via [`Convergence::from_tol`]).
    pub fn solve_dc_with(
        &self,
        p: &[f64],
        x0: &[f64],
        tol: f64,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        self.solve_dc_conv_with(p, x0, Convergence::from_tol(tol), max_iter, tricks)
    }

    /// DC operating point with an explicit per-component [`Convergence`] criterion
    /// and the default convergence aids.
    pub fn solve_dc_conv(
        &self,
        p: &[f64],
        x0: &[f64],
        conv: Convergence,
        max_iter: usize,
    ) -> (Vec<f64>, bool, usize) {
        self.solve_dc_conv_with(p, x0, conv, max_iter, self.tricks)
    }

    /// DC operating point with an explicit [`Convergence`] criterion and
    /// [`SolverTricks`]. The plain damped Newton (step shaping / line search gated
    /// by `tricks`) runs first; each enabled continuation is then tried in turn as
    /// a fallback. Disabling a trick removes exactly that stage, so a caller can
    /// A/B a single aid or enable device limiting for hard FET-dense circuits.
    pub fn solve_dc_conv_with(
        &self,
        p: &[f64],
        x0: &[f64],
        conv: Convergence,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        let _stage = sane_core::log::scope("dc");
        if self.delay_src.is_empty() {
            let (x, ok, its) = self.dc_cascade(p, x0, conv, max_iter, tricks, &self.nodeset);
            self.record_gmin_dominance(&x, p, ok, conv.vntol);
            return (x, ok, its);
        }
        // Transport delays at DC: the delayed output equals its source
        // (`d = y`). The history inputs are relaxed onto the source values
        // around the Newton cascade (damped fixed point, exact at the fixed
        // point); e.g. an ideal line's fixed point is the transparent
        // connection v1 = v2, i1 = -i2.
        let m = self.delay_src.len();
        let mut hist = vec![0.0; m];
        if x0.len() == self.n {
            for (h, &src) in hist.iter_mut().zip(&self.delay_src) {
                *h = x0[src];
            }
        }
        let mut x = x0.to_vec();
        let (mut cc, mut it_total) = (false, 0usize);
        for round in 0..DELAY_DC_MAX_ROUNDS {
            crate::delay::set_hist_values(&hist);
            let (xr, c, it) = self.dc_cascade(p, &x, conv, max_iter, tricks, &self.nodeset);
            x = xr;
            cc = c;
            it_total += it;
            if !c {
                break;
            }
            let mut delta: f64 = 0.0;
            for (k, &src) in self.delay_src.iter().enumerate() {
                let target = x[src];
                delta = delta.max((target - hist[k]).abs() / (1.0 + target.abs()));
                // damped relaxation: stable through |reflection| = 1 corners
                hist[k] += DELAY_DC_DAMPING * (target - hist[k]);
            }
            if delta < conv.reltol.max(1e-12) {
                break;
            }
            if round + 1 == DELAY_DC_MAX_ROUNDS {
                sane_core::log::warn_captured(
                    "DC: delay-history relaxation did not settle; operating point may be inconsistent",
                );
            }
        }
        self.record_gmin_dominance(&x, p, cc, conv.vntol);
        (x, cc, it_total)
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
    fn record_gmin_dominance(&self, x: &[f64], p: &[f64], converged: bool, vntol: f64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.last_gmin_row.store(usize::MAX, Relaxed);
        if !converged {
            return;
        }
        let Some((row, shift)) = self.gmin_dominance(x, p, vntol) else {
            return;
        };
        self.last_gmin_row.store(row, Relaxed);
        self.last_gmin_share.store(shift.to_bits(), Relaxed);
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
    fn gmin_dominance(&self, x: &[f64], p: &[f64], vntol: f64) -> Option<(usize, f64)> {
        let n = self.n;
        let zeros = vec![0.0; n];
        // Through the compiled step path (the reused symbolic pattern and
        // whichever backend it picked, the symmetric LDLT on a power grid):
        // the one-shot KLU this used to build cost 50x the Newton's own
        // factorization on a 45k-node grid (580 ms against 12 ms), on every
        // converged operating point.
        let (_, _, jv) = self.jacobian_x_sparse(x, &zeros, p, 0.0);
        let mut valbuf = Vec::new();
        let mut fac: Option<sparse::Refactorable> = None;
        let s = self.solve_step(&jv, GMIN_DC, x, &mut valbuf, &mut fac)?; // (J + g*I) s = x  =>  dx/dg = -s
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

    /// The full DC cascade with an explicit node-set (the registered one for the
    /// public entries, the per-call one for [`Self::solve_dc_nodeset`]), so basin
    /// selection reaches every stage rather than only the cold-start seed.
    fn dc_cascade(
        &self,
        p: &[f64],
        x0: &[f64],
        conv: Convergence,
        max_iter: usize,
        tricks: SolverTricks,
        ns: &[(usize, f64)],
    ) -> (Vec<f64>, bool, usize) {
        use sane_core::log;
        log::debug(&format!(
            "DC operating point: dim {}, nnz {}, device limits {}",
            self.n,
            self.nnz(),
            self.limits.len()
        ));
        // Fresh solve: clear any gmin-hold flag from a previous solve. Only the
        // gmin-stepping branch of `source_continuation` below sets it (issue #54).
        self.last_gmin_hold
            .store(0, std::sync::atomic::Ordering::Relaxed);
        // The fast path limits whenever the devices declare junction limits: a
        // Newton step shortened to the junction bound (`limiting::apply`) is
        // what lets an exponential device converge plainly from a cold start
        // instead of oscillating into the cascade (fixture corpus: 1851 -> 172
        // Newton iterations, ua741 816 -> 61). The trick flag gates only the
        // continuation correctors.
        let fast = SolverTricks {
            device_limiting: !self.limits.is_empty(),
            ..tricks
        };
        // A `.nodeset` seeds every *cold* solve: phase 1 stiff-pins the
        // node-set unknowns onto their targets (symmetry breaking), and the cascade
        // below warm-starts from that point. A warm `x0` supersedes the node-set.
        let mut it0 = 0usize;
        let seed: Vec<f64> = if tricks.nodeset && x0.is_empty() && !ns.is_empty() {
            let pin = self.build_pin(ns);
            log::info(&format!(
                "DC: node-set phase 1 (pinning {} node(s))",
                pin.idx.len()
            ));
            let (x1, c1, itp) = log_stage!(
                "dc/nodeset",
                self.newton_pinned(p, &self.nodeset_x0(ns), &pin, &conv, NODESET_MAX_ITER)
            );
            log::debug(&format!(
                "DC: node-set pin phase converged={c1} ({itp} iterations)"
            ));
            it0 = itp;
            x1
        } else {
            x0.to_vec()
        };
        // Like ngspice, the operating-point solve always keeps a tiny baseline
        // shunt `GMIN_DC` on every node (never solved at exactly gmin = 0).
        let (x, cc, it) = log_stage!(
            "dc/newton",
            self.newton(p, &seed, GMIN_DC, &conv, max_iter, fast)
        );
        let it = it + it0;
        if cc {
            // Routine path runs once per sweep/optimize point, so keep it at
            // DEBUG; the high-level analysis loop owns the INFO progress bar.
            log::debug(&format!("DC: converged (damped Newton, {it} iterations)"));
            return (x, true, it);
        }
        let (mut xbest, mut itbest) = (x, it);
        // Ladder order follows VACASK's `op_homotopy` (gdev -> gshunt -> src):
        // the device-conductance homotopy first, because a circuit that plain
        // Newton cannot start is usually device-nonlinearity dominated, and a
        // global shunt or source ramp then costs a full failed ladder before it
        // gets its turn (measured on ua741: 130 ms of fruitless gmin levels
        // ahead of a 10 ms companion solve).
        // A gmin-held (regularized) candidate from the source stage: remembered
        // here, accepted only at the very end -- the later stages get their shot
        // at the true floor first.
        let mut held: Option<Vec<f64>> = None;
        // Fallback 1: per-device companion continuation (the device models' native
        // linear lambda=0 form). Suits device-nonlinearity-dominated circuits,
        // where the others' global shunt / source ramp do not help.
        if tricks.companion_continuation && !self.companion.is_empty() {
            log::info("DC: plain Newton stalled, engaging companion continuation");
            let (xc, convc, itc) = log_stage!(
                "dc/companion_cont",
                self.companion_continuation(p, &conv, itbest, tricks)
            );
            if convc {
                log::info(&format!(
                    "DC: converged (companion continuation, {itc} iterations)"
                ));
                self.last_gmin_hold
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                return (xc, true, itc);
            }
            (xbest, itbest) = (xc, itc);
        }
        // Fallback 2: classic gmin stepping at *full* excitation (ngspice's first
        // fallback). Under a strong shunt every node is dominated by its local
        // conductance while the supplies and the bias sources are fully on, so
        // the relaxed solve lands in the powered basin of a self-biased circuit;
        // the adaptive step-down then tracks that basin to the floor. (An earlier
        // gmin-relaxation stage was removed as redundant with the source ramp --
        // but that was measured against mis-binned BSIM4 devices; on correctly
        // binned amplifiers the two select different basins, and this one
        // matches the references.)
        if tricks.gmin_continuation {
            log::info("DC: companion continuation stalled, engaging gmin stepping");
            let (xg, convg, itg) = log_stage!(
                "dc/gmin_step",
                self.gmin_stepping(p, &seed, &conv, itbest, tricks)
            );
            if convg {
                log::info(&format!("DC: converged (gmin stepping, {itg} iterations)"));
                return (xg, true, itg);
            }
            (xbest, itbest) = (xg, itg);
        }
        // Fallback 3: source-stepping continuation (ramp the supplies 0 -> full,
        // under a fixed baseline shunt). Converges the high-impedance-node and
        // high-gain-feedback circuits gmin stepping does not.
        if tricks.source_continuation {
            log::info("DC: gmin stepping stalled, engaging source-stepping continuation");
            let (xs, convs, its) = log_stage!(
                "dc/source_cont",
                self.source_continuation(p, &conv, itbest, tricks, ns)
            );
            if convs {
                log::info(&format!(
                    "DC: converged (source continuation, {its} iterations)"
                ));
                return (xs, true, its);
            }
            if self.last_regularized_gmin().is_some() {
                held = Some(xs.clone());
            }
            (xbest, itbest) = (xs, its);
        }
        // Fallback 4: node-adaptive damped Newton -- loads the weak diagonals of
        // under-determined internal nodes (high-impedance BSIM4 gi/di) so their
        // oscillating step that derails the other homotopies onto a non-physical
        // root is damped, while well-conditioned nodes keep full Newton.
        if tricks.node_adaptive {
            log::info("DC: engaging node-adaptive damped Newton");
            let (xa, conva, ita) =
                log_stage!("dc/node_adaptive", self.adaptive_newton(p, &conv, itbest));
            if conva {
                log::info(&format!(
                    "DC: converged (node-adaptive Newton, {ita} iterations)"
                ));
                self.last_gmin_hold
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                return (xa, true, ita);
            }
            (xbest, itbest) = (xa, ita);
        }
        // Fallback 5: pseudo-transient relaxation. A static continuation can die
        // at a fold -- the branch it is tracking (typically a self-biased
        // circuit's off-state) simply ceases to exist below some gmin, and no
        // step refinement crosses that. The physical circuit crosses it by
        // *moving*: integrate the circuit's own dynamics at full excitation from
        // the tightest held point (or the best any stage produced), over
        // geometrically growing horizons, and let the off-state relax into the
        // remaining (powered) basin; then polish with Newton at the floor.
        // Sources follow their waveforms during the relaxation (a drive rides on
        // top of the settling bias), which the final DC polish at `t = 0`
        // removes again.
        if tricks.pseudo_transient {
            log::info("DC: engaging pseudo-transient relaxation");
            let seed_ptc: &[f64] = held.as_deref().unwrap_or(&xbest);
            let (xr, convr, itr) = log_stage!(
                "dc/pseudo_tran",
                self.pseudo_transient_rescue(p, seed_ptc, &conv, itbest, tricks)
            );
            if convr {
                log::info(&format!(
                    "DC: converged (pseudo-transient relaxation, {itr} iterations)"
                ));
                self.last_gmin_hold
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                return (xr, true, itr);
            }
            (xbest, itbest) = (xr, itr);
        }
        // Last resort: accept the tightest gmin-held point as the final answer
        // (converged=true, machine-readably flagged as regularized, issue #54)
        // rather than failing outright.
        if let (Some(xh), Some(g)) = (held, self.last_regularized_gmin()) {
            log::warning(&format!(
                "DC: operating point holds only at gmin={g:.0e} (a high-impedance node \
                 is unstable at the {GMIN_DC:.0e} floor); reporting the regularized solution"
            ));
            return (xh, true, itbest);
        }
        log::warning(&format!(
            "DC: did not converge after {itbest} iterations (all homotopies exhausted)"
        ));
        (xbest, false, itbest)
    }

    /// Pseudo-transient continuation (PTC) rescue. Backward Euler on the
    /// *augmented* flow `dx/dtau + F(x) = 0` -- every row, the algebraic ones
    /// included, which is exactly what the physical transient cannot offer (a
    /// massless runaway node is unstable at every instant of real time). One BE
    /// step from anchor `x_k` with pseudo-step `h` solves
    ///
    ///   `F(x) + (x - x_k)/h = 0`,
    ///
    /// i.e. the stiff-pinned system with the pin on *all* rows, target `x_k`
    /// and conductance `1/h` -- a shunt to the *moving anchor* instead of the
    /// gmin shunt to ground, which is why PTC walks through folds that stall
    /// gmin stepping. `h` grows on success (the anchor follows the trajectory
    /// into the surviving basin) and shrinks on a failed step; once an accepted
    /// step barely moves, the flow is stationary and a plain Newton polish at
    /// the floor finishes the job.
    fn pseudo_transient_rescue(
        &self,
        p: &[f64],
        x_from: &[f64],
        conv: &Convergence,
        mut iters: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        let n = self.n;
        let mut xk = x_from.to_vec();
        if xk.len() != n {
            xk = vec![0.0; n];
        }
        let mut h = PTC_H0;
        let all: Vec<usize> = (0..n).collect();
        for _ in 0..PTC_MAX_STEPS {
            let pin = Pin {
                idx: all.clone(),
                target: xk.clone(),
                g: 1.0 / h,
            };
            // Inexact BE steps: the pinned residual carries the anchor term
            // `g*(x - target)`, so the meaningful per-step tolerance is "every
            // unknown within vntol of its BE solution" on the pinned scale --
            // `g * vntol` per row -- not the raw-abstol floor (unreachable for
            // large `g` and pointless here; the final floor polish is exact).
            let (xn, cn, it) = self.newton_pinned(p, &xk, &pin, conv, SOURCE_STEP_MAX_ITER);
            iters += it;
            if !cn {
                h *= PTC_H_SHRINK;
                if h < PTC_H_MIN {
                    sane_core::log::debug("PTC: pseudo-step underflow, giving up");
                    break;
                }
                continue;
            }
            let moved = xn
                .iter()
                .zip(&xk)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            xk = xn;
            sane_core::log::debug(&format!("PTC: step h={h:.1e} accepted, moved {moved:.2e}"));
            if moved < conv.vntol || h >= PTC_H_MAX {
                // Stationary pseudo-flow (or anchor effectively released): the
                // trajectory has settled into a basin; polish at the floor.
                let (xf, cf, itf) =
                    self.newton(p, &xk, GMIN_DC, conv, SOURCE_STEP_MAX_ITER, tricks);
                iters += itf;
                if cf {
                    return (xf, true, iters);
                }
                if h >= PTC_H_MAX {
                    break; // settled but the floor still rejects: genuinely stuck
                }
            }
            h *= PTC_H_GROW;
        }
        (xk, false, iters)
    }

    /// Build the stiff pin from a node-set list (dropping out-of-range indices).
    fn build_pin(&self, nodeset: &[(usize, f64)]) -> Pin {
        Pin {
            idx: nodeset
                .iter()
                .map(|&(i, _)| i)
                .filter(|&i| i < self.n)
                .collect(),
            target: nodeset
                .iter()
                .filter(|&&(i, _)| i < self.n)
                .map(|&(_, v)| v)
                .collect(),
            g: GSET,
        }
    }

    /// A cold start with the node-set targets written in (the rest zero).
    fn nodeset_x0(&self, nodeset: &[(usize, f64)]) -> Vec<f64> {
        let mut x = vec![0.0; self.n];
        for &(i, v) in nodeset {
            if i < self.n {
                x[i] = v;
            }
        }
        x
    }

    /// DC operating point with explicit `.nodeset` symmetry breaking. `nodeset` is
    /// a list of `(unknown_index, target_value)`. Phase 1 solves the *stiff-pinned*
    /// system so the solution lands near the targets (breaking the symmetry of a
    /// bistable circuit onto one branch); phase 2 removes the pin and runs the
    /// full robust cascade ([`solve_dc_with`](Self::solve_dc_with)) warm-started
    /// from phase 1, so the converged operating point is exact. With no node-sets
    /// it is exactly a plain [`solve_dc`](Self::solve_dc). (Analyses normally use
    /// [`set_nodeset`](Self::set_nodeset) instead, which routes every solve through
    /// the same phase-1 automatically.)
    pub fn solve_dc_nodeset(
        &self,
        p: &[f64],
        nodeset: &[(usize, f64)],
        conv: Convergence,
        max_iter: usize,
    ) -> (Vec<f64>, bool, usize) {
        self.solve_dc_nodeset_with(p, nodeset, conv, max_iter, self.tricks)
    }

    /// [`Self::solve_dc_nodeset`] with an explicit trick set. The per-call
    /// node-set flows into every cascade stage (the basin re-pin of the source
    /// continuation included), exactly like a registered one.
    pub fn solve_dc_nodeset_with(
        &self,
        p: &[f64],
        nodeset: &[(usize, f64)],
        conv: Convergence,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        use sane_core::log;
        if nodeset.is_empty() {
            return self.dc_cascade(p, &[], conv, max_iter, tricks, &self.nodeset);
        }
        let pin = self.build_pin(nodeset);
        log::info(&format!(
            "DC: node-set phase 1 (pinning {} node(s))",
            pin.idx.len()
        ));
        let (x1, _c1, it1) =
            self.newton_pinned(p, &self.nodeset_x0(nodeset), &pin, &conv, NODESET_MAX_ITER);
        log::info("DC: node-set phase 2 (pin released, free solve)");
        let (x2, cc, it2) = self.dc_cascade(p, &x1, conv, max_iter, tricks, nodeset);
        (x2, cc, it1 + it2)
    }

    /// Stiff-pin damped Newton for `.nodeset` phase 1. Solves the pinned system
    ///
    ///   `F(x) + GMIN_DC*x + g_set * S * (x - target) = 0`
    ///
    /// where `S` selects the node-set rows. Self-contained (its own triplet LU,
    /// like [`companion_solve`](Self::companion_solve)) so the fast-path
    /// [`newton`](Self::newton), `solve_step`, and the Schur partition stay
    /// untouched -- the pin is a one-off symmetry-breaking phase, not a hot path.
    fn newton_pinned(
        &self,
        p: &[f64],
        x_init: &[f64],
        pin: &Pin,
        conv: &Convergence,
        max_iter: usize,
    ) -> (Vec<f64>, bool, usize) {
        let n = self.n;
        let xdot = vec![0.0; n];
        let mut x = if x_init.len() == n {
            x_init.to_vec()
        } else {
            vec![0.0; n]
        };
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        let (mut inb, mut wb, mut ob) = (Vec::new(), Vec::new(), Vec::new());
        // The pinned matrix is `dF/dx + diag(GMIN_DC + g on the pinned rows)` -- the
        // pin entries land on the augmented diagonal, so its pattern is exactly the
        // precomputed `self.symbolic` (dF/dx nonzeros + full diagonal). Reuse that
        // symbolic every iteration (refill values, one numeric LU) instead of a fresh
        // AMD ordering per step (#52). Precompute the constant diagonal shunt once.
        // Per-row spring stiffness: `pin.g` is sized for MNA node rows; a pinned
        // row of a different physical scale (an `idt` state residual, say) can
        // dwarf it, leaving the pin soft and the pinned system ill-conditioned.
        // Rescale each pinned row's spring to dominate that row's own couplings
        // at the start point (one extra tape evaluation).
        let gpin: Vec<f64> = {
            self.fill_inputs(&x, &xdot, p, 0.0, &mut inputs);
            self.tape_step.eval(&inputs, &mut work, &mut out);
            let jac = &out[n..];
            pin.idx
                .iter()
                .map(|&i| {
                    let rownorm = self
                        .jx_rows
                        .iter()
                        .zip(jac)
                        .filter(|(&r, _)| r == i)
                        .map(|(_, v)| v.abs())
                        .fold(0.0f64, f64::max);
                    pin.g.max(GSET_ROW_SCALE * rownorm)
                })
                .collect()
        };
        let mut diag = vec![GMIN_DC; n];
        let mut row_floor = vec![0.0; n];
        for (k, &i) in pin.idx.iter().enumerate() {
            diag[i] += gpin[k];
            row_floor[i] = gpin[k] * conv.vntol;
        }
        let mut valbuf: Vec<f64> = Vec::new();
        // KLU factorization cache: full pivoting on the first iteration, frozen
        // pivot replay (numeric-only refactor) afterwards.
        let mut fac: Option<sparse::Refactorable> = None;
        // Pinned residual `F + GMIN_DC*x + g*(x - target)` on the pinned rows.
        let pin_res = |x: &[f64], res: &[f64]| -> Vec<f64> {
            let mut h: Vec<f64> = (0..n).map(|i| res[i] + GMIN_DC * x[i]).collect();
            for (k, &i) in pin.idx.iter().enumerate() {
                h[i] += gpin[k] * (x[i] - pin.target[k]);
            }
            h
        };

        let mut stall = newton::StallGuard::new();
        for it in 0..max_iter {
            self.fill_inputs(&x, &xdot, p, 0.0, &mut inputs);
            self.tape_step.eval(&inputs, &mut work, &mut out); // residual ++ jac-x
            let h = pin_res(&x, &out[..n]);
            let fnorm = norm2(&h);
            if stall.stalled(fnorm) {
                return (x, false, it);
            }
            // The shared contract, with the anchor term's own floor on every
            // pinned row: `g * vntol` is "within vntol of the pin", the
            // meaningful residual scale there (unreachable at the raw floor
            // for a large `g`, and pointless: the released solve polishes).
            let crit = self.criterion(conv).with_row_floor(&row_floor);
            let res_ok = crit.residual_ok(&h, &x, 0.0);
            // J + GMIN_DC*I + g_set on the pinned diagonals (duplicates summed).
            let dx = match self.solve_with_diag(&out[n..], &diag, &h, &mut valbuf, &mut fac) {
                Some(d) => d,
                None => return (x, false, it),
            };
            if res_ok && crit.update_ok(&dx, &x) {
                return (x, true, it);
            }
            let step: Vec<f64> = dx.clone();
            // Backtracking line search on the pinned residual norm.
            newton::backtrack(&mut x, &step, fnorm, LINE_SEARCH_TRIES, |trial| {
                self.fill_inputs(trial, &xdot, p, 0.0, &mut inb);
                self.tape_res.eval(&inb, &mut wb, &mut ob);
                norm2(&pin_res(trial, &ob))
            });
        }
        (x, false, max_iter)
    }

    /// Source-stepping continuation (predictor-corrector). The homotopy ramps the
    /// independent sources `0 -> full` via `lambda`, with a modest `SOURCE_GMIN`
    /// shunt during the ramp. The exact path tangent uses that independent sources
    /// enter the residual *linearly*, so `dF/dlambda = F(x; full) - F(x; off)` is a
    /// constant vector `b_src`; the tangent solves `[J + gmin*I] (dx/dlambda) =
    /// -b_src` with the corrector's own factorization. This is the robust path for
    /// high-gain feedback (op-amps), where the gmin shunt alone breaks the loop.
    fn source_continuation(
        &self,
        p: &[f64],
        conv: &Convergence,
        mut iters: usize,
        tricks: SolverTricks,
        ns: &[(usize, f64)],
    ) -> (Vec<f64>, bool, usize) {
        let n = self.n;
        let scaled = |lambda: f64| -> Vec<f64> {
            p.iter()
                .zip(&self.source_mask)
                .map(|(&v, &is_src)| if is_src { v * lambda } else { v })
                .collect::<Vec<_>>()
        };
        let p_full = scaled(1.0);
        let p_zero = scaled(0.0);

        // lambda = 0: the unexcited circuit (its devices may have small offsets,
        // so solve rather than assume all-zero).
        let (mut x, c0, it0) = self.newton(
            &p_zero,
            &[],
            SOURCE_GMIN,
            conv,
            SOURCE_STEP_MAX_ITER,
            tricks,
        );
        iters += it0;
        if !c0 {
            return (x, false, iters);
        }

        let xdot = vec![0.0; n];
        let (mut inputs, mut work, mut out, mut valbuf) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // Tangent-solve factorization cache (fixed SOURCE_GMIN along the ramp).
        let mut fac: Option<sparse::Refactorable> = None;
        let mut lambda = 0.0_f64;
        let mut dlam = HOMOTOPY_DLAM0;
        let mut steps = 0usize;
        while lambda < 1.0 {
            steps += 1;
            if steps > HOMOTOPY_MAX_STEPS {
                return (x, false, iters);
            }
            let target = (lambda + dlam).min(1.0);

            // dF/dlambda = b_src = residual(x; full sources) - residual(x; off).
            let r_full = self.residual(&x, &xdot, &p_full, 0.0);
            let r_zero = self.residual(&x, &xdot, &p_zero, 0.0);
            let neg_b: Vec<f64> = (0..n).map(|i| r_zero[i] - r_full[i]).collect();
            // J = dF/dx at x (independent of source values).
            self.fill_inputs(&x, &xdot, p, 0.0, &mut inputs);
            self.tape_step.eval(&inputs, &mut work, &mut out);
            let x_pred: Vec<f64> =
                match self.solve_step(&out[n..], SOURCE_GMIN, &neg_b, &mut valbuf, &mut fac) {
                    Some(t) => (0..n).map(|i| x[i] + t[i] * (target - lambda)).collect(),
                    None => x.clone(),
                };

            let ps = scaled(target);
            let (xc, cc, it) = self.newton(
                &ps,
                &x_pred,
                SOURCE_GMIN,
                conv,
                SOURCE_STEP_MAX_ITER,
                tricks,
            );
            iters += it;
            if cc {
                x = xc;
                lambda = target;
                dlam = (dlam * HOMOTOPY_GROW).min(HOMOTOPY_DLAM_MAX);
            } else {
                dlam *= HOMOTOPY_SHRINK;
                if dlam < HOMOTOPY_DLAM_MIN {
                    return (x, false, iters);
                }
            }
        }

        // Basin re-selection under full excitation. The lambda = 0 solve above is
        // cold, so a registered node-set's symmetry breaking (phase 1 of the cold
        // cascade) is lost by the time the ramp completes -- a self-biased circuit
        // can ride the ramp into its degenerate off-state root, and no gmin
        // reduction recovers from the wrong basin. Re-pin the node-set targets
        // warm from the ramp point, then release at the ramp shunt, so the
        // step-down below starts inside the selected basin.
        if tricks.nodeset && !ns.is_empty() {
            let pin = self.build_pin(ns);
            let (xp, cp, itp) = self.newton_pinned(p, &x, &pin, conv, NODESET_MAX_ITER);
            iters += itp;
            if cp {
                let (xr, cr, itr) =
                    self.newton(p, &xp, SOURCE_GMIN, conv, SOURCE_STEP_MAX_ITER, tricks);
                iters += itr;
                if cr {
                    x = xr;
                }
            }
        }
        // Final polish at full excitation and the baseline shunt. The common case:
        // the source-ramped point converges directly at the `GMIN_DC` floor.
        let (xf, cc, it) = self.newton(p, &x, GMIN_DC, conv, SOURCE_STEP_MAX_ITER, tricks);
        iters += it;
        if cc {
            return (xf, true, iters);
        }
        // The single jump to the `GMIN_DC` floor destabilised a high-impedance node
        // (a high-gain output that the shunt was holding in basin). Step the shunt
        // down adaptively from the source level; reach the floor if possible,
        // otherwise report the tightest regularized operating point that holds,
        // rather than failing outright.
        let (best, reached, its, best_gmin) =
            self.gmin_step_down(p, x, SOURCE_GMIN, conv, iters, tricks);
        iters = its;
        if reached {
            return (best, true, iters); // reached the true DC floor by stepping
        }
        // Tightest regularized point: record the hold (machine-readable via
        // `last_regularized_gmin`, issue #54) but report *not converged* -- the
        // remaining cascade stages (pseudo-transient relaxation in particular)
        // get their shot at the true floor first, and only the cascade's end
        // accepts a held point as the final, physically-suspect answer.
        self.last_gmin_hold
            .store(best_gmin.to_bits(), std::sync::atomic::Ordering::Relaxed);
        (best, false, iters)
    }

    /// Classic gmin stepping: solve at a strong shunt under *full* excitation
    /// (every node dominated by its local conductance -- the powered basin of a
    /// self-biased circuit), then relax the shunt to the floor with the adaptive
    /// step-down ladder. `seed` carries the node-set phase-1 point when one is
    /// registered, so the strong-shunt solve starts inside the selected basin.
    /// Returns non-converged unless the true `GMIN_DC` floor is reached -- a
    /// held (regularized) point is left to the later stages to improve on.
    fn gmin_stepping(
        &self,
        p: &[f64],
        seed: &[f64],
        conv: &Convergence,
        mut iters: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        let (x, c, it) = self.newton(p, seed, GMIN_STEP_START, conv, SOURCE_STEP_MAX_ITER, tricks);
        iters += it;
        if !c {
            sane_core::log::debug("gmin stepping: strong-shunt solve did not converge");
            return (x, false, iters);
        }
        let (best, reached, its, held) =
            self.gmin_step_down(p, x, GMIN_STEP_START, conv, iters, tricks);
        if !reached {
            sane_core::log::debug(&format!("gmin stepping: ladder held at gmin={held:.1e}"));
        }
        (best, reached, its)
    }

    /// Adaptive gmin step-down ladder shared by [`Self::gmin_stepping`] and the
    /// source-continuation tail: relax the shunt from `from` toward the
    /// `GMIN_DC` floor, warm-starting each level. A failed level is retried at
    /// half the log-step from the last held point (ngspice's "dynamic gmin");
    /// a success re-accelerates the ratio toward full decades. Returns the
    /// tightest held point, whether the floor was reached, the iteration total
    /// and the gmin it held at.
    fn gmin_step_down(
        &self,
        p: &[f64],
        x: Vec<f64>,
        from: f64,
        conv: &Convergence,
        mut iters: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize, f64) {
        let mut best = x;
        let mut best_gmin = from;
        let mut ratio = GMIN_RAMP_DOWN;
        let mut levels = 0usize;
        while best_gmin > GMIN_DC && levels < GMIN_STEP_MAX_LEVELS {
            let g = (best_gmin * ratio).max(GMIN_DC);
            let (xg, cg, it) = self.newton(p, &best, g, conv, SOURCE_STEP_MAX_ITER, tricks);
            iters += it;
            levels += 1;
            sane_core::log::debug(&format!(
                "gmin step-down: level {levels} gmin={g:.2e} {} ({it} iters)",
                if cg { "held" } else { "failed" }
            ));
            if cg {
                best = xg;
                best_gmin = g;
                ratio = (ratio * ratio).max(GMIN_RAMP_DOWN);
            } else {
                ratio = ratio.sqrt();
                if ratio > GMIN_STEP_RATIO_FLOOR {
                    break; // level thinner than ~1% of a decade: genuinely stuck
                }
            }
        }
        let reached = best_gmin <= GMIN_DC;
        (best, reached, iters, best_gmin)
    }

    /// The gmin at which the most recent DC solve held, if the solve converged
    /// only under a raised shunt (gmin stepping could not relax to the `GMIN_DC`
    /// floor) -- a converged-but-regularized, physically suspect operating point.
    /// `None` when the true floor was reached. Valid until the next DC solve on
    /// this `CompiledDc` (issue #54).
    pub fn last_regularized_gmin(&self) -> Option<f64> {
        let bits = self
            .last_gmin_hold
            .load(std::sync::atomic::Ordering::Relaxed);
        (bits != 0).then(|| f64::from_bits(bits))
    }

    /// `(row, current share)` of the worst unknown the gmin shunt sets in the
    /// most recent DC solve, or `None` when the circuit sets them all. See
    /// [`Self::gmin_dominance`].
    pub fn last_gmin_dominance(&self) -> Option<(usize, f64)> {
        use std::sync::atomic::Ordering::Relaxed;
        let row = self.last_gmin_row.load(Relaxed);
        (row != usize::MAX).then(|| (row, f64::from_bits(self.last_gmin_share.load(Relaxed))))
    }

    /// Companion residual `H = F(x) + (1 - lambda)*G_comp*x` at `(x, lambda)`.
    fn companion_residual(
        &self,
        x: &[f64],
        p: &[f64],
        lambda: f64,
        work: &mut Vec<f64>,
    ) -> Vec<f64> {
        let n = self.n;
        let mut inputs = Vec::new();
        let xdot = vec![0.0; n];
        self.fill_inputs(x, &xdot, p, 0.0, &mut inputs);
        let mut out = Vec::new();
        self.tape_res.eval(&inputs, work, &mut out);
        let mut h = out;
        let s = 1.0 - lambda;
        for &(r, c, g) in &self.companion {
            h[r] += s * g * x[c];
        }
        // Ground baseline on node rows so the lambda = 0 network is regular even
        // when a device cluster has no companion path to ground (the star links
        // terminals, not ground); fades with lambda. Branch/internal rows are
        // constraints, not KCL -- a shunt there would corrupt them.
        for i in (0..n).filter(|&i| self.is_node(i)) {
            h[i] += s * GMIN_START * x[i];
        }
        h
    }

    /// Solve `[J(x) + (1-lambda)*G_comp + GMIN_DC*I] dx = rhs` (the augmented
    /// homotopy Jacobian), evaluating `J` from the step tape at `x`. `fac` is
    /// the caller loop's factorization cache over the companion pattern (KLU
    /// numeric-only refactor after the first iteration).
    fn companion_solve<'a>(
        &'a self,
        x: &[f64],
        p: &[f64],
        lambda: f64,
        rhs: &[f64],
        fac: &mut Option<sparse::Refactorable<'a>>,
    ) -> Option<Vec<f64>> {
        let n = self.n;
        let xdot = vec![0.0; n];
        let mut inputs = Vec::new();
        self.fill_inputs(x, &xdot, p, 0.0, &mut inputs);
        let (mut work, mut out) = (Vec::new(), Vec::new());
        self.tape_step.eval(&inputs, &mut work, &mut out);
        let jac = &out[n..];
        let s = 1.0 - lambda;
        // Reuse a precomputed symbolic (#52): the pattern (dF/dx nonzeros + companion
        // positions + full diagonal) is fixed across the continuation, so only the
        // values change with `lambda`. This refills that pattern and runs one numeric
        // LU per iteration rather than a fresh ordering + symbolic analysis.
        // Value order must match `companion_symbolic`: jac ++ companion ++ diagonal.
        if let Some(sym) = self.companion_symbolic() {
            let mut valbuf: Vec<f64> = Vec::with_capacity(jac.len() + self.companion.len() + n);
            valbuf.extend_from_slice(jac);
            for &(_, _, g) in &self.companion {
                valbuf.push(s * g);
            }
            for i in 0..n {
                let base = if self.is_node(i) { s * GMIN_START } else { 0.0 };
                valbuf.push(base + GMIN_DC);
            }
            let f = fac.get_or_insert_with(|| sym.pattern.factorizer());
            if !f.factor(&valbuf, self.tricks.row_equilibration) {
                return None;
            }
            return f.solve(rhs);
        }
        // Fallback (degenerate pattern): one-shot triplet factorization.
        let extra: Vec<(usize, usize, f64)> = self
            .companion
            .iter()
            .map(|&(r, c, g)| (r, c, s * g))
            .collect();
        let diag: Vec<f64> = (0..n)
            .map(|i| if self.is_node(i) { s * GMIN_START } else { 0.0 } + GMIN_DC)
            .collect();
        self.solve_triplets(jac, &extra, &diag, rhs)
    }

    /// Lazily build and cache the reused symbolic for the companion-augmented
    /// homotopy matrix (see [`companion_symbolic`](Self.companion_symbolic)). The
    /// value order the caller must supply is `dF/dx nonzeros ++ companion values ++
    /// full diagonal`, matching how the pattern's `(row, col)` pairs are appended.
    fn companion_symbolic(&self) -> Option<&Symbolic> {
        self.companion_symbolic
            .get_or_init(|| {
                let mut rows = self.jx_rows.clone();
                let mut cols = self.jx_cols.clone();
                for &(r, c, _) in &self.companion {
                    rows.push(r);
                    cols.push(c);
                }
                Self::build_symbolic(self.n, &rows, &cols)
            })
            .as_ref()
    }

    /// Damped Newton on the companion-augmented system at a fixed `lambda`.
    /// `tricks` gates the per-step shaping (uniform clamp, device limiting, line
    /// search), exactly as in [`newton`](Self::newton).
    fn companion_newton(
        &self,
        p: &[f64],
        x_init: &[f64],
        lambda: f64,
        conv: &Convergence,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        let n = self.n;
        let mut x = if x_init.len() == n {
            x_init.to_vec()
        } else {
            vec![0.0; n]
        };
        let mut work = Vec::new();
        // Companion-pattern factorization cache for this lambda's corrector.
        let mut fac: Option<sparse::Refactorable> = None;
        let mut stall = newton::StallGuard::new();
        for it in 0..max_iter {
            let h = self.companion_residual(&x, p, lambda, &mut work);
            let fnorm = norm2(&h);
            if stall.stalled(fnorm) {
                return (x, false, it);
            }
            // `h` already folds in the companion and gmin terms, so test it
            // directly (per component) with no extra shunt. The intermediate
            // continuation points need only the residual half; the final polish
            // in `newton` enforces the full residual+update criterion.
            // The residual half of the shared contract only: an intermediate
            // continuation point is a waypoint, and demanding the update half
            // there changed which point the polish started from (measured as
            // termination noise in a finite-difference check); the final
            // polish in `newton` enforces both halves.
            if self.residual_converged(&h, &x, 0.0, conv) {
                return (x, true, it);
            }
            let dx = match self.companion_solve(&x, p, lambda, &h, &mut fac) {
                Some(d) => d,
                None => return (x, false, it),
            };
            let mut step: Vec<f64> = dx.clone();
            // Curve-aware per-device limiting (path-only; see `newton`).
            if tricks.device_limiting && !self.limits.is_empty() {
                newton::limit_step(&self.limits, &x, &mut step[..n]);
            }
            // Backtracking line search on |H|.
            let tries = if tricks.line_search {
                LINE_SEARCH_TRIES
            } else {
                1
            };
            newton::backtrack(&mut x, &step, fnorm, tries, |trial| {
                norm2(&self.companion_residual(trial, p, lambda, &mut work))
            });
        }
        (x, false, max_iter)
    }

    /// Per-device companion homotopy continuation (predictor-corrector). Deforms
    /// `H(x, lambda) = F(x) + (1-lambda)*G_comp*x` from the device models' linear
    /// `lambda = 0` network (trivially solvable) to the real circuit, using the
    /// exact path tangent `[dH/dx] (dx/dlambda) = G_comp*x`. This is the native,
    /// template-sourced alternative to the global gmin / source homotopies -- the
    /// `lambda = 0` start keeps the devices' terminals connected (vs. the gmin
    /// shunt to ground), which suits strongly nonlinear blocks.
    pub(crate) fn companion_continuation(
        &self,
        p: &[f64],
        conv: &Convergence,
        mut iters: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        if self.companion.is_empty() {
            return (vec![0.0; self.n], false, iters);
        }
        let n = self.n;
        // lambda = 0: the companion-dominated linear network.
        let (mut x, c0, it0) = self.companion_newton(p, &[], 0.0, conv, GMIN_STEP_MAX_ITER, tricks);
        iters += it0;
        if !c0 {
            return (x, false, iters);
        }
        let mut lambda = 0.0_f64;
        let mut dlam = HOMOTOPY_DLAM0;
        let mut steps = 0usize;
        // Tangent-solve factorization cache (pattern fixed along the ramp).
        let mut fac_t: Option<sparse::Refactorable> = None;
        while lambda < 1.0 {
            steps += 1;
            if steps > HOMOTOPY_MAX_STEPS {
                return (x, false, iters);
            }
            let target = (lambda + dlam).min(1.0);
            // Exact tangent: [dH/dx] t = (G_comp + GMIN_START*I_nodes) x.
            let mut rhs_t = vec![0.0; n];
            for i in (0..n).filter(|&i| self.is_node(i)) {
                rhs_t[i] = GMIN_START * x[i];
            }
            for &(r, c, g) in &self.companion {
                rhs_t[r] += g * x[c];
            }
            let dl = target - lambda;
            let x_pred: Vec<f64> = match self.companion_solve(&x, p, lambda, &rhs_t, &mut fac_t) {
                Some(t) => (0..n).map(|i| x[i] + t[i] * dl).collect(),
                None => x.clone(),
            };
            let (xc, cc, it) =
                self.companion_newton(p, &x_pred, target, conv, GMIN_STEP_MAX_ITER, tricks);
            iters += it;
            sane_core::log::debug(&format!(
                "companion: lambda {lambda:.4} -> {target:.4} {} ({it} iters)",
                if cc { "held" } else { "failed" }
            ));
            if cc {
                x = xc;
                lambda = target;
                dlam = (dlam * HOMOTOPY_GROW).min(HOMOTOPY_DLAM_MAX);
            } else {
                dlam *= HOMOTOPY_SHRINK;
                if dlam < HOMOTOPY_DLAM_MIN {
                    return (x, false, iters);
                }
            }
        }
        // lambda = 1: companion gone -> final polish at the baseline shunt.
        let (xf, cc, it) = self.newton(p, &x, GMIN_DC, conv, GMIN_STEP_MAX_ITER, tricks);
        (xf, cc, iters + it)
    }
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The 2-norm of the shunted residual `F + gmin * x` (`gmin = 0` is the raw
/// residual): the scalar the line searches and the stall guard read.
fn shunted_norm(res: &[f64], x: &[f64], gmin: f64) -> f64 {
    if gmin == 0.0 {
        return norm2(res);
    }
    res.iter()
        .zip(x)
        .map(|(r, xi)| {
            let v = r + gmin * xi;
            v * v
        })
        .sum::<f64>()
        .sqrt()
}
