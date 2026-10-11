//! The DC operating-point solve: damped Newton with per-component convergence,
//! and the robust continuation cascade behind it (gmin stepping, source
//! stepping, per-device companion homotopy, node-adaptive damping, and
//! pseudo-transient relaxation), each stage a toggleable
//! [`SolverTricks`](crate::SolverTricks) trick.

use sane_core::constants::*;
use sane_core::log_stage;

use crate::schur::Partition;
use crate::{newton, sparse, CompiledDc, Convergence, LinCache, SolverTricks, StepEval, Symbolic};

mod continuation;
mod criteria;
mod linear;
mod system;
use system::{Dc, Shunt};

impl CompiledDc {
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
            let (x, ok, its) = self.dc_cascade(p, x0, conv, max_iter, tricks, &self.dc.nodeset);
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
            let (xr, c, it) = self.dc_cascade(p, &x, conv, max_iter, tricks, &self.dc.nodeset);
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
            if delta < conv.reltol.max(DELAY_DC_TOL_FLOOR) {
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
        self.dc
            .last_gmin_hold
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
                self.newton_pinned(p, &self.nodeset_x0(ns), &pin, &conv, CONTINUATION_MAX_ITER)
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
        if tricks.companion_continuation && !self.dc.companion.is_empty() {
            log::info("DC: plain Newton stalled, engaging companion continuation");
            let (xc, convc, itc) = log_stage!(
                "dc/companion_cont",
                self.companion_continuation(p, &conv, itbest, tricks)
            );
            if convc {
                log::info(&format!(
                    "DC: converged (companion continuation, {itc} iterations)"
                ));
                self.dc
                    .last_gmin_hold
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
                self.dc
                    .last_gmin_hold
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
                self.dc
                    .last_gmin_hold
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
            return self.dc_cascade(p, &[], conv, max_iter, tricks, &self.dc.nodeset);
        }
        let pin = self.build_pin(nodeset);
        log::info(&format!(
            "DC: node-set phase 1 (pinning {} node(s))",
            pin.idx.len()
        ));
        let (x1, _c1, it1) = self.newton_pinned(
            p,
            &self.nodeset_x0(nodeset),
            &pin,
            &conv,
            CONTINUATION_MAX_ITER,
        );
        log::info("DC: node-set phase 2 (pin released, free solve)");
        let (x2, cc, it2) = self.dc_cascade(p, &x1, conv, max_iter, tricks, nodeset);
        (x2, cc, it1 + it2)
    }

    /// The gmin at which the most recent DC solve held, if the solve converged
    /// only under a raised shunt (gmin stepping could not relax to the `GMIN_DC`
    /// floor) -- a converged-but-regularized, physically suspect operating point.
    /// `None` when the true floor was reached. Valid until the next DC solve on
    /// this `CompiledDc` (issue #54).
    pub fn last_regularized_gmin(&self) -> Option<f64> {
        let bits = self
            .dc
            .last_gmin_hold
            .load(std::sync::atomic::Ordering::Relaxed);
        (bits != 0).then(|| f64::from_bits(bits))
    }

    /// `(row, current share)` of the worst unknown the gmin shunt sets in the
    /// most recent DC solve, or `None` when the circuit sets them all. See
    /// [`Self::gmin_dominance`].
    pub fn last_gmin_dominance(&self) -> Option<(usize, f64)> {
        use std::sync::atomic::Ordering::Relaxed;
        let row = self.dc.last_gmin_row.load(Relaxed);
        (row != usize::MAX).then(|| (row, f64::from_bits(self.dc.last_gmin_share.load(Relaxed))))
    }

    /// Evaluate `tape` at `(x, p)` into `tb.out`, bound to `p` on the
    /// buffers' first evaluation. `tb` must stay with `tape` and `p` for its
    /// lifetime.
    fn eval_episode(&self, tape: &StepEval, x: &[f64], p: &[f64], tb: &mut TapeBufs) {
        let bound = tb
            .bound
            .get_or_insert_with(|| crate::Bound::new(self, tape, p, &[]));
        bound.eval(self, tape, x, 0.0, &mut tb.out);
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

/// What the DC operating-point solve keeps across its calls: the symbolic
/// LU of its Jacobian, the Schur partition, the homotopy's companion network
/// and its LU, the node-set, and what the latest solve reports about gmin.
pub(crate) struct DcState {
    /// Reused sparse symbolic factorization (`None` only for degenerate
    /// patterns the backend rejects, in which case we rebuild per iteration).
    pub(crate) symbolic: Option<Symbolic>,
    /// Linear/nonlinear partition of the Jacobian for cached-factorization
    /// (Schur-complement) solves. `None` when partitioning is not worthwhile
    /// (the nonlinear part is not a small fraction of the system).
    pub(crate) partition: Option<Partition>,
    /// Companion conductance network `(row, col, value)` from the device models
    /// (their linear `lambda = 0` form) -- the per-device homotopy continuation.
    pub(crate) companion: Vec<(usize, usize, f64)>,
    /// Reused symbolic factorization for the companion-augmented homotopy matrix
    /// `G + (1-lambda)*G_comp + gmin*I` (#52). Its pattern is the `G`
    /// nonzeros plus the companion `(row, col)` positions plus the full diagonal --
    /// fixed across the continuation, only the values change with `lambda`. Built
    /// lazily on first `companion_solve` (the fallback path most circuits never
    /// hit); `Some(None)` marks a degenerate pattern the backend rejected.
    pub(crate) companion_symbolic: std::sync::OnceLock<Option<Symbolic>>,
    /// User `.nodeset` targets `(unknown_index, value)`. Empty by default. When
    /// set, every *cold-started* DC solve runs a stiff-pin phase first
    /// (symmetry breaking; see [`solve_dc_nodeset_with`](Self::solve_dc_nodeset_with)), so
    /// all analyses honor the node-set without per-call-site plumbing. Warm
    /// (non-empty `x0`) solves ignore it -- the warm start already supersedes it.
    pub(crate) nodeset: Vec<(usize, f64)>,
    /// Regularization flag for the most recent DC solve: the IEEE-754 bits of the
    /// held gmin when the operating point converged only under a raised shunt
    /// (gmin stepping could not reach the `GMIN_DC` floor), or `0` when the true
    /// floor was reached. Set by [`source_continuation`], reset at the top of
    /// [`solve_dc_conv_with`]; read via [`last_regularized_gmin`]. Drives the
    /// analysis layer's settle fallback, so nothing else may set it -- the
    /// second way a point can depend on gmin lives in `last_gmin_row` and is
    /// combined only where the two are reported. Interior mutability (an atomic)
    /// so the flag rides through the `&self` solve API without threading it
    /// through every `(x, conv, iters)` return (issue #54).
    pub(crate) last_gmin_hold: std::sync::atomic::AtomicU64,
    /// The worst gmin-dominated unknown of the most recent DC solve, or
    /// `usize::MAX` for none, alongside the IEEE-754 bits of its current share.
    /// Read via [`last_gmin_dominance`]; same interior-mutability reasoning.
    pub(crate) last_gmin_row: std::sync::atomic::AtomicUsize,
    pub(crate) last_gmin_share: std::sync::atomic::AtomicU64,
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
/// to one tape at one parameter binding (see [`CompiledDc::eval_episode`]).
#[derive(Default)]
struct TapeBufs {
    bound: Option<crate::Bound>,
    out: Vec<f64>,
    vals: Vec<f64>,
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
