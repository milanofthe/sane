//! The DC continuation fallbacks: the node-set pins, source stepping,
//! gmin stepping, the per-device companion homotopy and the
//! pseudo-transient rescue.

use super::*;

/// A `.nodeset` pin: stiff springs `g*(x_i - target_i)` on selected unknown rows
/// for the first phase of a node-set solve. Forcing the pinned unknowns toward
/// their targets breaks the symmetry of a bistable circuit (a latch, an
/// identical differential pair) that otherwise sits on the unstable symmetric
/// root; phase 2 removes the pin and re-solves freely, so the final operating
/// point is exact -- the pin shapes only *which* root is found.
pub(super) struct Pin {
    pub(super) idx: Vec<usize>,
    pub(super) target: Vec<f64>,
    pub(super) g: f64,
}

impl CompiledDc {
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
    pub(super) fn pseudo_transient_rescue(
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
            let (xn, cn, it) = self.newton_pinned(p, &xk, &pin, conv, CONTINUATION_MAX_ITER);
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
                    self.newton(p, &xk, GMIN_DC, conv, CONTINUATION_MAX_ITER, tricks);
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
    pub(super) fn build_pin(&self, nodeset: &[(usize, f64)]) -> Pin {
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
    pub(super) fn nodeset_x0(&self, nodeset: &[(usize, f64)]) -> Vec<f64> {
        let mut x = vec![0.0; self.n];
        for &(i, v) in nodeset {
            if i < self.n {
                x[i] = v;
            }
        }
        x
    }

    /// Stiff-pin damped Newton for `.nodeset` phase 1. Solves the pinned system
    ///
    ///   `F(x) + GMIN_DC*x + g_set * S * (x - target) = 0`
    ///
    /// where `S` selects the node-set rows, each spring stiff enough to
    /// dominate its row's own couplings at the start point. Residual within
    /// `g * vntol` of the pin on a pinned row: within `vntol` of it.
    pub(super) fn newton_pinned(
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
    pub(super) fn source_continuation(
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
            CONTINUATION_MAX_ITER,
            tricks,
        );
        iters += it0;
        if !c0 {
            return (x, false, iters);
        }

        let (mut step, mut valbuf) = (TapeBufs::default(), Vec::new());
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
            self.eval_episode(&self.tape_step_dc, &x, p, &mut step);
            let solved = self.solve_step(
                &step.out[n..],
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
                CONTINUATION_MAX_ITER,
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
            let (xp, cp, itp) = self.newton_pinned(p, &x, &pin, conv, CONTINUATION_MAX_ITER);
            iters += itp;
            if cp {
                let (xr, cr, itr) =
                    self.newton(p, &xp, SOURCE_GMIN, conv, CONTINUATION_MAX_ITER, tricks);
                iters += itr;
                if cr {
                    x = xr;
                }
            }
        }
        // Final polish at full excitation and the baseline shunt. The common case:
        // the source-ramped point converges directly at the `GMIN_DC` floor.
        let (xf, cc, it) = self.newton(p, &x, GMIN_DC, conv, CONTINUATION_MAX_ITER, tricks);
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
        self.dc
            .last_gmin_hold
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
    pub(super) fn gmin_stepping(
        &self,
        p: &Binding<'_>,
        seed: &[f64],
        conv: &Convergence,
        mut iters: usize,
        tricks: SolverTricks,
    ) -> (Vec<f64>, bool, usize) {
        let (x, c, it) = self.newton(
            p,
            seed,
            GMIN_STEP_START,
            conv,
            CONTINUATION_MAX_ITER,
            tricks,
        );
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
    pub(super) fn gmin_step_down(
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
            let (xg, cg, it) = self.newton(p, &best, g, conv, CONTINUATION_MAX_ITER, tricks);
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

    /// Solve `[J(x) + (1-lambda)*G_comp + GMIN_DC*I] dx = rhs` (the augmented
    /// homotopy Jacobian), evaluating `J` from the step tape at `x`. `fac` is
    /// the caller loop's factorization cache over the companion pattern (KLU
    /// numeric-only refactor after the first iteration).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn companion_solve<'a>(
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
            for &(_, _, g) in &self.dc.companion {
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
            .dc
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
    pub(super) fn companion_symbolic(&self) -> Option<&Symbolic> {
        self.dc
            .companion_symbolic
            .get_or_init(|| {
                let mut rows = self.jx_rows.clone();
                let mut cols = self.jx_cols.clone();
                for &(r, c, _) in &self.dc.companion {
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
    pub(super) fn companion_newton(
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
        if self.dc.companion.is_empty() {
            return (vec![0.0; self.n], false, iters);
        }
        let n = self.n;
        // lambda = 0: the companion-dominated linear network. The binding's
        // episodes serve the whole ramp: its prologs ran once.
        let (mut x, c0, it0) =
            self.companion_newton(p, &[], 0.0, conv, CONTINUATION_MAX_ITER, tricks);
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
            for &(r, c, g) in &self.dc.companion {
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
                self.companion_newton(p, &x_pred, target, conv, CONTINUATION_MAX_ITER, tricks);
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
        let (xf, cc, it) = self.newton(p, &x, GMIN_DC, conv, CONTINUATION_MAX_ITER, tricks);
        (xf, cc, iters + it)
    }
}

/// The step control of a predictor-corrector continuation: `lambda` from 0
/// to 1, the step grown after a held corrector and shrunk after a failed one.
/// The ramp gives up past `HOMOTOPY_MAX_STEPS` points, below
/// `HOMOTOPY_DLAM_MIN`, or after `HOMOTOPY_ENDPOINT_TRIES` failed correctors
/// at `lambda = 1` itself.
pub(super) struct Ramp {
    pub(super) lambda: f64,
    pub(super) dlam: f64,
    pub(super) steps: usize,
    pub(super) endpoint_fails: usize,
}

impl Ramp {
    pub(super) fn new() -> Ramp {
        Ramp {
            lambda: 0.0,
            dlam: HOMOTOPY_DLAM0,
            steps: 0,
            endpoint_fails: 0,
        }
    }

    /// The next corrector's target, `None` past the step cap.
    pub(super) fn target(&mut self) -> Option<f64> {
        self.steps += 1;
        (self.steps <= HOMOTOPY_MAX_STEPS).then(|| (self.lambda + self.dlam).min(1.0))
    }

    /// Take the corrector's verdict at `target`; `false` gives the ramp up.
    pub(super) fn record(&mut self, target: f64, held: bool) -> bool {
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
