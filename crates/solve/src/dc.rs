//! The DC operating-point solve: damped Newton with per-component convergence,
//! and the robust continuation cascade behind it (gmin stepping, source
//! stepping, per-device companion homotopy, node-adaptive damping, and
//! pseudo-transient relaxation), each stage a toggleable
//! [`SolverTricks`](crate::SolverTricks) trick.

use sane_core::constants::*;
use sane_core::log_stage;

use crate::{
    newton, sparse, CompiledDc, Convergence, LinCache, PrologToken, SolverTricks, StepEval,
    Symbolic,
};

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
        dx: &mut [f64],
        valbuf: &mut Vec<f64>,
        fac: &mut Option<sparse::Refactorable<'a>>,
    ) -> bool {
        let diag = std::iter::repeat_n(gmin, self.n);
        self.solve_with_diag(jac, diag, rhs, dx, valbuf, fac)
    }

    /// Solve `(J + diag(d)) dx = rhs` into `dx` with the Jacobian nonzeros
    /// `jac` (in the compiled order) and a per-row diagonal shunt `d`: the
    /// reused symbolic pattern refactored in place, or, for a degenerate
    /// pattern the symbolic analysis rejected, a one-shot triplet
    /// factorization. `false` when singular.
    fn solve_with_diag<'a>(
        &'a self,
        jac: &[f64],
        diag: impl IntoIterator<Item = f64>,
        rhs: &[f64],
        dx: &mut [f64],
        valbuf: &mut Vec<f64>,
        fac: &mut Option<sparse::Refactorable<'a>>,
    ) -> bool {
        match &self.symbolic {
            Some(sym) => {
                // values: jacobian nonzeros, then the full diagonal.
                valbuf.clear();
                valbuf.extend_from_slice(jac);
                valbuf.extend(diag);
                let f = fac.get_or_insert_with(|| sym.pattern.factorizer());
                f.factor(valbuf, self.tricks.row_equilibration) && f.solve_into(rhs, dx)
            }
            None => {
                let diag: Vec<f64> = diag.into_iter().collect();
                self.solve_triplets(jac, &[], &diag, rhs)
                    .map(|x| dx.copy_from_slice(&x))
                    .is_some()
            }
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
        sparse::factor_triplets(n, &rows, &cols, &vals)?.solve(rhs)
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

    /// Damped Newton at a fixed `gmin` (at rest, `t = 0`), warm-started from
    /// `x_init`. Returns `(x, converged, iterations)`. `tricks` gates the per-step
    /// shaping (uniform clamp, device limiting, line search, partitioning).
    /// The rounding the shunted residual `res + gmin x` at `x` is computed
    /// with: per row (into `terms`) the size of the terms it sums, `|G| |x|`
    /// over the Jacobian values `g` plus the shunt's, `NEWTON_ROUNDOFF` ulps
    /// of them; returned, whether every row is within it (solved to machine
    /// precision) and that floor of the residual's 2-norm.
    fn rounding(
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

    /// The DC Newton under the shunt `gmin` from `x_init` (cold where empty):
    /// the Newton core on the shunted system, with the line search, the
    /// composite step and device limiting as `tricks` say.
    fn newton(
        &self,
        p: &Binding<'_>,
        x_init: &[f64],
        gmin: f64,
        conv: &Convergence,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        let tries = if tricks.line_search {
            LINE_SEARCH_TRIES
        } else {
            1
        };
        let policy = newton::Policy {
            globalization: newton::Globalization::LineSearch { tries },
            limiting: tricks.device_limiting,
            composite: tricks.composite_step,
            ..self.policy(max_iter)
        };
        let sys = Dc::new(self, p, Shunt::Uniform(gmin), tricks);
        self.run(sys, x_init, conv, true, &policy)
    }

    /// Damped Newton with **node-adaptive diagonal loading**: each iteration the
    /// weakly-coupled unknowns (tiny `|J_ii|`, e.g. high-impedance BSIM4 internal
    /// nodes) get their diagonal loaded up to `ADAPT_DIAG_FRAC * max|J_ii|`, so
    /// their otherwise-huge oscillating Newton step is damped while well-coupled
    /// unknowns keep full Newton. The loading is matrix-only (the residual still
    /// uses just `GMIN_DC*x`), so the converged fixed point `F + GMIN_DC*x = 0` is
    /// unshifted -- the damping shapes only the iteration path. Cold-started;
    /// the residual half also holds where the SPICE-relative test does, and
    /// where it does not converge the lowest-residual iterate is returned.
    fn adaptive_newton(
        &self,
        p: &Binding<'_>,
        conv: &Convergence,
        iters: usize,
    ) -> (Vec<f64>, bool, usize) {
        let policy = newton::Policy {
            stall: false,
            composite: false,
            keep_best: true,
            ..self.policy(ADAPT_MAX_ITER)
        };
        let sys = Dc::new(self, p, Shunt::Uniform(GMIN_DC), self.tricks).adaptive(*conv);
        let (x, conv, it) = self.run(sys, &[], conv, true, &policy);
        (x, conv, iters + it)
    }

    /// Solve the DC operating point (at rest, `t = 0`) with sparse LU and the
    /// default set of convergence aids. Tries a plain damped-Newton solve first;
    /// if it stalls, falls back through the continuation tricks. Returns
    /// `(x, converged, iterations)`. The scalar `tol` is the absolute residual
    /// floor of the per-component criterion ([`Convergence::from_tol`]); use
    /// [`solve_dc_conv_with`](Self::solve_dc_conv_with) for full
    /// `reltol`/`abstol`/`vntol` control and explicit [`SolverTricks`].
    pub fn solve_dc(
        &self,
        p: &[f64],
        x0: &[f64],
        tol: f64,
        max_iter: usize,
    ) -> (Vec<f64>, bool, usize) {
        self.solve_dc_conv_with(p, x0, Convergence::from_tol(tol), max_iter, self.tricks)
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
        crate::parallel::solve(|| self.solve_dc_conv_with_here(p, x0, conv, max_iter, tricks))
    }

    /// [`solve_dc_conv_with`](Self::solve_dc_conv_with) on this thread.
    fn solve_dc_conv_with_here(
        &self,
        p: &[f64],
        x0: &[f64],
        conv: Convergence,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        let _stage = sane_core::log::scope("dc");
        let p = &Binding::new(p);
        if !self.has_delays() {
            let (x, ok, its) = self.dc_cascade(p, x0, conv, max_iter, tricks, &self.nodeset);
            self.record_gmin_dominance(&x, p, ok, conv.vntol);
            return (x, ok, its);
        }
        // Transport delays at DC: the delayed output equals its source
        // (`d = y`). The history inputs are relaxed onto the source values
        // around the Newton cascade (damped fixed point, exact at the fixed
        // point); e.g. an ideal line's fixed point is the transparent
        // connection v1 = v2, i1 = -i2.
        let mut hist = vec![0.0; self.delay_count()];
        let mut target = Vec::new();
        if x0.len() == self.n {
            self.delay_values(x0, p, 0.0, &mut hist);
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
            self.delay_values(&x, p, 0.0, &mut target);
            for (h, &target) in hist.iter_mut().zip(&target) {
                delta = delta.max((target - *h).abs() / (1.0 + target.abs()));
                // damped relaxation: stable through |reflection| = 1 corners
                *h += DELAY_DC_DAMPING * (target - *h);
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
        // Through the compiled step path (the reused symbolic pattern and
        // whichever backend it picked, the symmetric LDLT on a power grid):
        // the one-shot KLU this used to build cost 50x the Newton's own
        // factorization on a 45k-node grid (580 ms against 12 ms), on every
        // converged operating point. The Jacobian likewise comes from the
        // DC step program the Newton loops run (warm, native), not from the
        // general one (on c6288 the difference was two seconds per point).
        let jv = &self.jacobian_dc(x, p);
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

    /// The full DC cascade with an explicit node-set (the registered one for the
    /// public entries, the per-call one for [`Self::solve_dc_nodeset_with`]), so basin
    /// selection reaches every stage rather than only the cold-start seed.
    fn dc_cascade(
        &self,
        p: &Binding<'_>,
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
            self.newton(p, &seed, GMIN_DC, &conv, max_iter, tricks)
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
        p: &Binding<'_>,
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
    /// full robust cascade warm-started from phase 1, so the converged operating
    /// point is exact. With no node-sets it is a plain cascade. The per-call
    /// node-set flows into every cascade stage (the basin re-pin of the source
    /// continuation included), exactly like a registered one. (Analyses normally
    /// use [`set_nodeset`](Self::set_nodeset) instead, which routes every solve
    /// through the same phase 1 automatically.)
    pub fn solve_dc_nodeset_with(
        &self,
        p: &[f64],
        nodeset: &[(usize, f64)],
        conv: Convergence,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        crate::parallel::solve(|| {
            self.solve_dc_nodeset_with_here(p, nodeset, conv, max_iter, tricks)
        })
    }

    /// [`solve_dc_nodeset_with`](Self::solve_dc_nodeset_with) on this thread.
    fn solve_dc_nodeset_with_here(
        &self,
        p: &[f64],
        nodeset: &[(usize, f64)],
        conv: Convergence,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        use sane_core::log;
        let p = &Binding::new(p);
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
    /// where `S` selects the node-set rows, each spring stiff enough to
    /// dominate its row's own couplings at the start point. Residual within
    /// `g * vntol` of the pin on a pinned row: within `vntol` of it.
    fn newton_pinned(
        &self,
        p: &Binding<'_>,
        x_init: &[f64],
        pin: &Pin,
        conv: &Convergence,
        max_iter: usize,
    ) -> (Vec<f64>, bool, usize) {
        let n = self.n;
        let x0 = match x_init.len() == n {
            true => x_init.to_vec(),
            false => vec![0.0; n],
        };
        // Per-row spring stiffness: `pin.g` is sized for MNA node rows; a pinned
        // row of a different physical scale (an `idt` state residual, say) can
        // dwarf it, leaving the pin soft and the pinned system ill-conditioned.
        // Rescale each pinned row's spring to dominate that row's own couplings
        // at the start point (one extra tape evaluation).
        let gpin: Vec<f64> = {
            let mut tapes = p.tapes.borrow_mut();
            self.eval_episode(&self.tape_step_dc, &x0, p, &mut tapes.step);
            let jac = &tapes.step.out[n..];
            (pin.idx.iter())
                .map(|&i| {
                    let rownorm = (self.jx_rows.iter().zip(jac))
                        .filter(|(&r, _)| r == i)
                        .map(|(_, v)| v.abs())
                        .fold(0.0f64, f64::max);
                    pin.g.max(GSET_ROW_SCALE * rownorm)
                })
                .collect()
        };
        let (mut diag, mut anchor, mut row_floor) = (vec![GMIN_DC; n], vec![0.0; n], vec![0.0; n]);
        for (k, &i) in pin.idx.iter().enumerate() {
            diag[i] += gpin[k];
            anchor[i] = gpin[k] * pin.target[k];
            row_floor[i] = gpin[k] * conv.vntol;
        }
        let contract = newton::Contract {
            criterion: self.criterion(conv).with_row_floor(&row_floor),
            residual: true,
            update: Some(1.0),
        };
        let policy = newton::Policy {
            limiting: false,
            early_accept: false,
            composite: false,
            ..self.policy(max_iter)
        };
        let mut sys = Dc::new(self, p, Shunt::Pinned { diag, anchor }, self.tricks);
        let mut x = x0;
        let out = newton::solve(
            &mut sys,
            &mut x,
            &contract,
            &policy,
            &mut newton::Scratch::default(),
        );
        (x, out.converged, out.iters)
    }

    /// Source-stepping continuation (predictor-corrector). The homotopy ramps the
    /// independent sources `0 -> full` via `lambda`, with a modest `SOURCE_GMIN`
    /// shunt during the ramp. The exact path tangent uses that independent sources
    /// enter the residual *linearly*, so `dI/dlambda = I(x; full) - I(x; off)` is a
    /// constant vector `b_src`; the tangent solves `[J + gmin*I] (dx/dlambda) =
    /// -b_src` with the corrector's own factorization. This is the robust path for
    /// high-gain feedback (op-amps), where the gmin shunt alone breaks the loop.
    fn source_continuation(
        &self,
        p: &Binding<'_>,
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
            &Binding::new(&p_zero),
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

        let (mut inputs, mut work, mut out, mut valbuf) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut tangent = vec![0.0; n];
        // Tangent-solve factorization cache (fixed SOURCE_GMIN along the ramp).
        let mut fac: Option<sparse::Refactorable> = None;
        let mut ramp = Ramp::new();
        while ramp.lambda < 1.0 {
            let Some(target) = ramp.target() else {
                return (x, false, iters);
            };
            let lambda = ramp.lambda;

            // dI/dlambda = b_src = I(x; full sources) - I(x; off).
            let r_full = self.currents(&x, &p_full, 0.0);
            let r_zero = self.currents(&x, &p_zero, 0.0);
            let neg_b: Vec<f64> = (0..n).map(|i| r_zero[i] - r_full[i]).collect();
            // G = dI/dx at x (independent of source values).
            self.fill_inputs(&x, p, 0.0, &mut inputs);
            self.tape_step_dc.eval(&inputs, &mut work, &mut out);
            let solved = self.solve_step(
                &out[n..],
                SOURCE_GMIN,
                &neg_b,
                &mut tangent,
                &mut valbuf,
                &mut fac,
            );
            let x_pred: Vec<f64> = if solved {
                (0..n)
                    .map(|i| x[i] + tangent[i] * (target - lambda))
                    .collect()
            } else {
                x.clone()
            };

            let ps = scaled(target);
            let (xc, cc, it) = self.newton(
                &Binding::new(&ps),
                &x_pred,
                SOURCE_GMIN,
                conv,
                SOURCE_STEP_MAX_ITER,
                tricks,
            );
            iters += it;
            if cc {
                x = xc;
            }
            if !ramp.record(target, cc) {
                return (x, false, iters);
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
        p: &Binding<'_>,
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
        p: &Binding<'_>,
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

    /// Evaluate `tape` at `(x, p)` into `tb.out`: the prolog on the buffers'
    /// first evaluation, the main phase on every one. `tb` must stay with
    /// `tape` and `p` for its lifetime.
    fn eval_episode(&self, tape: &StepEval, x: &[f64], p: &[f64], tb: &mut TapeBufs) {
        self.fill_inputs(x, p, 0.0, &mut tb.inputs);
        let ep = tb
            .episode
            .get_or_insert_with(|| tape.eval_prolog(&tb.inputs, &mut tb.work));
        tape.eval_main(ep, &tb.inputs, &mut tb.work, &mut tb.out);
    }

    /// Solve `[J(x) + (1-lambda)*G_comp + GMIN_DC*I] dx = rhs` (the augmented
    /// homotopy Jacobian), evaluating `J` from the step tape at `x`. `fac` is
    /// the caller loop's factorization cache over the companion pattern (KLU
    /// numeric-only refactor after the first iteration).
    #[allow(clippy::too_many_arguments)]
    fn companion_solve<'a>(
        &'a self,
        x: &[f64],
        p: &[f64],
        lambda: f64,
        rhs: &[f64],
        dx: &mut [f64],
        tb: &mut TapeBufs,
        fac: &mut Option<sparse::Refactorable<'a>>,
    ) -> bool {
        let n = self.n;
        self.eval_episode(&self.tape_step_dc, x, p, tb);
        let jac = &tb.out[n..];
        let s = 1.0 - lambda;
        // Reuse a precomputed symbolic (#52): the pattern (G nonzeros + companion
        // positions + full diagonal) is fixed across the continuation, so only the
        // values change with `lambda`. This refills that pattern and runs one numeric
        // LU per iteration rather than a fresh ordering + symbolic analysis.
        // Value order must match `companion_symbolic`: jac ++ companion ++ diagonal.
        if let Some(sym) = self.companion_symbolic() {
            let valbuf = &mut tb.vals;
            valbuf.clear();
            valbuf.extend_from_slice(jac);
            for &(_, _, g) in &self.companion {
                valbuf.push(s * g);
            }
            for i in 0..n {
                let base = if self.is_node(i) { s * GMIN_START } else { 0.0 };
                valbuf.push(base + GMIN_DC);
            }
            let f = fac.get_or_insert_with(|| sym.pattern.factorizer());
            return f.factor(valbuf, self.tricks.row_equilibration) && f.solve_into(rhs, dx);
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
            .map(|x| dx.copy_from_slice(&x))
            .is_some()
    }

    /// Lazily build and cache the reused symbolic for the companion-augmented
    /// homotopy matrix (see [`companion_symbolic`](Self.companion_symbolic)). The
    /// value order the caller must supply is `G nonzeros ++ companion values ++
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

    /// Damped Newton on the companion-augmented system at a fixed `lambda`,
    /// to the residual half alone: an intermediate continuation point is a
    /// waypoint, and demanding the update half there changed which point the
    /// polish started from (measured as termination noise in a
    /// finite-difference check); the final polish in `newton` enforces both
    /// halves. `tricks` gates the line search and device limiting as in
    /// [`newton`](Self::newton).
    fn companion_newton(
        &self,
        p: &Binding<'_>,
        x_init: &[f64],
        lambda: f64,
        conv: &Convergence,
        max_iter: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        let tries = if tricks.line_search {
            LINE_SEARCH_TRIES
        } else {
            1
        };
        let policy = newton::Policy {
            globalization: newton::Globalization::LineSearch { tries },
            limiting: tricks.device_limiting,
            early_accept: false,
            composite: false,
            ..self.policy(max_iter)
        };
        let sys = Dc::new(self, p, Shunt::Companion(lambda), tricks);
        self.run(sys, x_init, conv, false, &policy)
    }

    /// The Newton policy of the DC loops: the full Jacobian, the line search
    /// with every probe, device limiting, the stall rule, the early
    /// acceptance and the composite step.
    fn policy(&self, max_iter: usize) -> newton::Policy {
        newton::Policy {
            max_iter,
            jacobian: newton::Jacobian::Full,
            globalization: newton::Globalization::LineSearch {
                tries: LINE_SEARCH_TRIES,
            },
            limiting: self.tricks.device_limiting,
            stall: true,
            early_accept: true,
            composite: self.tricks.composite_step,
            keep_best: false,
            trace: sane_core::config().dc_trace,
        }
    }

    /// The Newton core on `sys` from `x_init` (cold where empty), to the
    /// contract over this system's unknowns: the residual half, and the
    /// update half where `update`.
    fn run(
        &self,
        mut sys: Dc<'_, '_>,
        x_init: &[f64],
        conv: &Convergence,
        update: bool,
        policy: &newton::Policy,
    ) -> (Vec<f64>, bool, usize) {
        let mut x = match x_init.len() == self.n {
            true => x_init.to_vec(),
            false => vec![0.0; self.n],
        };
        let contract = newton::Contract {
            criterion: self.criterion(conv),
            residual: true,
            update: update.then_some(1.0),
        };
        let out = newton::solve(
            &mut sys,
            &mut x,
            &contract,
            policy,
            &mut newton::Scratch::default(),
        );
        (x, out.converged, out.iters)
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
        p: &Binding<'_>,
        conv: &Convergence,
        mut iters: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        if self.companion.is_empty() {
            return (vec![0.0; self.n], false, iters);
        }
        let n = self.n;
        // lambda = 0: the companion-dominated linear network. The binding's
        // episodes serve the whole ramp: its prologs ran once.
        let (mut x, c0, it0) = self.companion_newton(p, &[], 0.0, conv, GMIN_STEP_MAX_ITER, tricks);
        iters += it0;
        if !c0 {
            return (x, false, iters);
        }
        let mut ramp = Ramp::new();
        // Tangent-solve factorization cache (pattern fixed along the ramp).
        let mut fac_t: Option<sparse::Refactorable> = None;
        let mut tangent = vec![0.0; n];
        while ramp.lambda < 1.0 {
            let Some(target) = ramp.target() else {
                return (x, false, iters);
            };
            let lambda = ramp.lambda;
            // Exact tangent: [dH/dx] t = (G_comp + GMIN_START*I_nodes) x.
            let mut rhs_t = vec![0.0; n];
            for i in (0..n).filter(|&i| self.is_node(i)) {
                rhs_t[i] = GMIN_START * x[i];
            }
            for &(r, c, g) in &self.companion {
                rhs_t[r] += g * x[c];
            }
            let dl = target - lambda;
            let solved = {
                let mut tapes = p.tapes.borrow_mut();
                self.companion_solve(
                    &x,
                    p,
                    lambda,
                    &rhs_t,
                    &mut tangent,
                    &mut tapes.step,
                    &mut fac_t,
                )
            };
            let x_pred: Vec<f64> = if solved {
                (0..n).map(|i| x[i] + tangent[i] * dl).collect()
            } else {
                x.clone()
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
            }
            if !ramp.record(target, cc) {
                return (x, false, iters);
            }
        }
        // lambda = 1: companion gone -> final polish at the baseline shunt.
        let (xf, cc, it) = self.newton(p, &x, GMIN_DC, conv, GMIN_STEP_MAX_ITER, tricks);
        (xf, cc, iters + it)
    }
}

/// The step control of a predictor-corrector continuation: `lambda` from 0
/// to 1, the step grown after a held corrector and shrunk after a failed one.
/// The ramp gives up past `HOMOTOPY_MAX_STEPS` points, below
/// `HOMOTOPY_DLAM_MIN`, or after `HOMOTOPY_ENDPOINT_TRIES` failed correctors
/// at `lambda = 1` itself.
struct Ramp {
    lambda: f64,
    dlam: f64,
    steps: usize,
    endpoint_fails: usize,
}

impl Ramp {
    fn new() -> Ramp {
        Ramp {
            lambda: 0.0,
            dlam: HOMOTOPY_DLAM0,
            steps: 0,
            endpoint_fails: 0,
        }
    }

    /// The next corrector's target, `None` past the step cap.
    fn target(&mut self) -> Option<f64> {
        self.steps += 1;
        (self.steps <= HOMOTOPY_MAX_STEPS).then(|| (self.lambda + self.dlam).min(1.0))
    }

    /// Take the corrector's verdict at `target`; `false` gives the ramp up.
    fn record(&mut self, target: f64, held: bool) -> bool {
        if held {
            self.lambda = target;
            self.dlam = (self.dlam * HOMOTOPY_GROW).min(HOMOTOPY_DLAM_MAX);
            return true;
        }
        if target == 1.0 {
            self.endpoint_fails += 1;
        }
        self.dlam *= HOMOTOPY_SHRINK;
        self.dlam >= HOMOTOPY_DLAM_MIN && self.endpoint_fails < HOMOTOPY_ENDPOINT_TRIES
    }
}

/// What a DC Newton adds to `F(x)` to make its system: a shunt to ground,
/// pin springs, or the companion homotopy.
enum Shunt {
    /// `gmin x` on every unknown.
    Uniform(f64),
    /// `diag x - anchor`: `GMIN_DC` on every unknown and a spring
    /// `g_k (x_k - t_k)` on each pinned one (the node-set phase, the
    /// pseudo-transient anchor).
    Pinned { diag: Vec<f64>, anchor: Vec<f64> },
    /// The companion homotopy at `lambda`: `(1 - lambda) (G_comp x +
    /// GMIN_START x)` on the node rows; `GMIN_DC` regularizes the matrix only.
    Companion(f64),
}

/// The DC system at a binding as the Newton core solves it (see
/// [`newton::System`]): `F(x)` and its Jacobian from the binding's episodes
/// (the step tape for both, the residual tape for the probes), what the
/// [`Shunt`] adds, the reused symbolic pattern refactored in place (or the
/// cached Schur partition of a large mostly-linear system, or a one-shot
/// triplet LU on a degenerate pattern), and the rounding the residual rows
/// sum.
struct Dc<'a, 'b> {
    cdc: &'a CompiledDc,
    p: &'a Binding<'b>,
    shunt: Shunt,
    tapes: std::cell::RefMut<'a, Episodes>,
    valbuf: Vec<f64>,
    fac: Option<sparse::Refactorable<'a>>,
    /// The cached linear block of the partition, built at the first factor.
    lin: Option<LinCache>,
    partition: bool,
    terms: Vec<f64>,
    /// Load the weak diagonals of the matrix (the node-adaptive fallback):
    /// each up to `ADAPT_DIAG_FRAC` of the strongest, decaying with the
    /// residual norm.
    load_weak: bool,
    /// Grant the residual half where the SPICE-relative test holds (see
    /// [`CompiledDc::residual_relative_ok`]).
    relative: Option<Convergence>,
    /// The residual norm at the last Jacobian evaluation.
    fnorm: f64,
    iscale: (Vec<f64>, Vec<f64>),
}

impl<'a, 'b> Dc<'a, 'b> {
    fn new(cdc: &'a CompiledDc, p: &'a Binding<'b>, shunt: Shunt, tricks: SolverTricks) -> Self {
        let partition =
            matches!(shunt, Shunt::Uniform(_)) && cdc.partition.is_some() && tricks.partition;
        Dc {
            cdc,
            p,
            shunt,
            tapes: p.tapes.borrow_mut(),
            valbuf: Vec::new(),
            fac: None,
            lin: None,
            partition,
            terms: Vec::new(),
            load_weak: false,
            relative: None,
            fnorm: f64::INFINITY,
            iscale: (Vec::new(), Vec::new()),
        }
    }

    /// The node-adaptive form: weak diagonals loaded, the relative test
    /// granted, no partition.
    fn adaptive(mut self, conv: Convergence) -> Self {
        self.load_weak = true;
        self.relative = Some(conv);
        self.partition = false;
        self
    }

    /// The Jacobian nonzeros of the last Jacobian evaluation.
    fn jac(&self) -> &[f64] {
        &self.tapes.step.out[self.cdc.n..]
    }

    /// The diagonal the shunt puts on the matrix, row `i`.
    fn diag(&self, i: usize) -> f64 {
        match &self.shunt {
            Shunt::Uniform(g) => *g,
            Shunt::Pinned { diag, .. } => diag[i],
            Shunt::Companion(l) => {
                let base = if self.cdc.is_node(i) {
                    (1.0 - l) * GMIN_START
                } else {
                    0.0
                };
                base + GMIN_DC
            }
        }
    }

    /// The companion entries `(row, col, value)` the homotopy puts on the
    /// matrix (none outside it).
    fn extra(&self) -> Vec<(usize, usize, f64)> {
        match self.shunt {
            Shunt::Companion(l) => (self.cdc.companion.iter())
                .map(|&(r, c, g)| (r, c, (1.0 - l) * g))
                .collect(),
            _ => Vec::new(),
        }
    }
}

impl newton::System<f64> for Dc<'_, '_> {
    fn eval(&mut self, x: &[f64], jacobian: bool, res: &mut [f64]) -> bool {
        let (cdc, p) = (self.cdc, self.p);
        let tb = match jacobian {
            true => &mut self.tapes.step,
            false => &mut self.tapes.res,
        };
        let tape = if jacobian {
            &cdc.tape_step_dc
        } else {
            &cdc.tape_res_dc
        };
        cdc.eval_episode(tape, x, p, tb);
        let f = &tb.out;
        match &self.shunt {
            Shunt::Uniform(g) => {
                for (i, r) in res.iter_mut().enumerate() {
                    *r = f[i] + g * x[i];
                }
            }
            Shunt::Pinned { diag, anchor } => {
                for (i, r) in res.iter_mut().enumerate() {
                    *r = f[i] + diag[i] * x[i] - anchor[i];
                }
            }
            Shunt::Companion(l) => {
                let s = 1.0 - l;
                for (i, r) in res.iter_mut().enumerate() {
                    *r = f[i]
                        + if cdc.is_node(i) {
                            s * GMIN_START * x[i]
                        } else {
                            0.0
                        };
                }
                for &(r, c, g) in &cdc.companion {
                    res[r] += s * g * x[c];
                }
            }
        }
        if jacobian {
            self.fnorm = norm2(res);
        }
        true
    }

    fn factor(&mut self) -> bool {
        let cdc = self.cdc;
        let n = cdc.n;
        if self.partition {
            if let Shunt::Uniform(g) = self.shunt {
                if self.lin.is_none() {
                    let part = cdc.partition.as_ref().unwrap();
                    self.lin = cdc.build_lin_cache(part, self.jac(), g);
                }
            }
            if self.lin.is_some() {
                return true;
            }
            self.partition = false; // degenerate: the full LU
        }
        let companion = matches!(self.shunt, Shunt::Companion(_));
        let sym = match companion {
            true => cdc.companion_symbolic(),
            false => cdc.symbolic.as_ref(),
        };
        let Some(sym) = sym else {
            return true; // the triplet fallback factors as it solves
        };
        // values: the Jacobian nonzeros, the companion entries, the diagonal
        let mut vals = std::mem::take(&mut self.valbuf);
        vals.clear();
        vals.extend_from_slice(self.jac());
        if self.load_weak {
            let out = &self.tapes.step.out;
            let maxd = (cdc.diag_idx.iter().flatten())
                .map(|&di| out[n + di].abs())
                .fold(0.0f64, f64::max);
            // strong far from the solution, vanishing as the residual does
            let floor = ADAPT_DIAG_FRAC * maxd * self.fnorm.min(1.0);
            for &di in cdc.diag_idx.iter().flatten() {
                let jii = vals[di].abs();
                if jii < floor {
                    vals[di] += floor - jii;
                }
            }
        }
        vals.extend(self.extra().into_iter().map(|(_, _, v)| v));
        vals.extend((0..n).map(|i| self.diag(i)));
        let f = self.fac.get_or_insert_with(|| sym.pattern.factorizer());
        let ok = f.factor(&vals, cdc.tricks.row_equilibration);
        self.valbuf = vals;
        ok
    }

    fn solve(&mut self, rhs: &[f64], dx: &mut [f64]) -> bool {
        let cdc = self.cdc;
        if self.partition {
            let (part, lin) = (cdc.partition.as_ref().unwrap(), self.lin.as_ref().unwrap());
            return (cdc.solve_partitioned(lin, part, self.jac(), rhs))
                .map(|d| dx.copy_from_slice(&d))
                .is_some();
        }
        if let Some(f) = &mut self.fac {
            return f.solve_into(rhs, dx);
        }
        let diag: Vec<f64> = (0..cdc.n).map(|i| self.diag(i)).collect();
        (cdc.solve_triplets(self.jac(), &self.extra(), &diag, rhs))
            .map(|d| dx.copy_from_slice(&d))
            .is_some()
    }

    fn probe(&mut self, rhs: &[f64], dx: &mut [f64]) -> bool {
        match (&mut self.fac, self.partition) {
            (Some(f), false) => f.solve_into(rhs, dx),
            _ => false,
        }
    }

    fn chord(&mut self, rhs: &[f64], dx: &mut [f64]) -> bool {
        self.probe(rhs, dx)
    }

    fn rounding(&mut self, x: &[f64], _res: &[f64]) -> (bool, f64) {
        let n = self.cdc.n;
        let out = &self.tapes.step.out;
        match self.shunt {
            Shunt::Uniform(g) => self
                .cdc
                .rounding(&out[..n], &out[n..], x, g, &mut self.terms),
            // the rounding of `F + GMIN_DC x`: a floor for the line search,
            // no acceptance (the spring rows sum terms it does not count)
            Shunt::Pinned { .. } => (
                false,
                self.cdc
                    .rounding(&out[..n], &out[n..], x, GMIN_DC, &mut self.terms)
                    .1,
            ),
            // waypoints take the residual half alone, which the rounding
            // never keeps from passing
            Shunt::Companion(_) => (false, 0.0),
        }
    }

    fn accept_residual(&mut self, x: &[f64], _res: &[f64]) -> bool {
        let (Some(conv), Some(tape)) = (self.relative, &self.cdc.tape_iscale) else {
            return false;
        };
        let (work, terms) = &mut self.iscale;
        let step = &self.tapes.step;
        tape.eval(&step.inputs, work, terms);
        let n = self.cdc.n;
        self.cdc
            .residual_relative_ok(&step.out[..n], terms, x, GMIN_DC, &conv)
    }

    fn limit(&self, x: &[f64], step: &mut [f64]) -> f64 {
        newton::limit_step(&self.cdc.limits, x, step)
    }
}

/// The binding a DC solve runs at, and its tapes' episodes there: the
/// parameter vector is fixed across the whole solve, so each tape's
/// parameter-pure prolog runs once and every Newton loop of the cascade (each
/// homotopy level, each continuation step) evaluates only the main phase.
/// Reads as the parameter vector.
pub(crate) struct Binding<'a> {
    p: &'a [f64],
    tapes: std::cell::RefCell<Episodes>,
}

impl<'a> Binding<'a> {
    pub(crate) fn new(p: &'a [f64]) -> Self {
        Binding {
            p,
            tapes: Default::default(),
        }
    }
}

impl std::ops::Deref for Binding<'_> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        self.p
    }
}

/// The step (residual and Jacobian) and the residual tapes' buffers at one
/// binding.
#[derive(Default)]
struct Episodes {
    step: TapeBufs,
    res: TapeBufs,
}

/// One tape evaluation site's buffers, and the factor values built from its
/// outputs: kept by a Newton loop across its iterations. The buffers belong
/// to one tape at one parameter binding: the first evaluation runs the
/// tape's parameter-pure prolog into `work`, every later one only the main
/// phase (see [`CompiledDc::eval_episode`]).
#[derive(Default)]
struct TapeBufs {
    inputs: Vec<f64>,
    work: Vec<f64>,
    out: Vec<f64>,
    vals: Vec<f64>,
    episode: Option<PrologToken>,
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
