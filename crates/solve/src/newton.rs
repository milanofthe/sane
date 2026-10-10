//! One step policy for every Newton in the solver.
//!
//! The DC Newton, its continuation correctors, the pinned Newton of the
//! node-set phase, the node-adaptive fallback, the implicit-RK stage Newton and
//! the harmonic-balance Newton all take the same kind of step: the linear
//! solve gives a direction, device limiting shortens it to the largest
//! fraction that keeps every declared junction within its curve-aware bound,
//! and a backtracking search halves it while the residual norm does not
//! decrease. This module holds that policy once; each solver supplies only
//! how it measures the residual of a trial iterate.
//!
//! The backtracking comes in two shapes. The *forward* form ([`backtrack`])
//! probes trial iterates with a residual evaluation the caller
//! provides, which suits the DC solver, whose residual-only tape is far
//! cheaper than its Jacobian. The *retroactive* form ([`Backtrack`]) is for a
//! solver whose Jacobian comes for free with the residual (harmonic balance,
//! where one alias-free transform pass yields both): the full step is taken,
//! the next iteration's residual is the probe, and a step that did not
//! contract is retracted to a fraction before that Jacobian is used.

use crate::limiting::Limit;
use sane_core::constants::{LINE_SEARCH_SHRINK, NEWTON_STALL_FACTOR, NEWTON_STALL_WINDOW};
use sane_dae::UnknownKind;

use crate::{limiting, Convergence};

/// Magnitude of a real or complex scalar, for the convergence tests.
pub(crate) trait Magnitude: Copy {
    fn mag(self) -> f64;
}
impl Magnitude for f64 {
    fn mag(self) -> f64 {
        self.abs()
    }
}
impl Magnitude for num_complex::Complex64 {
    fn mag(self) -> f64 {
        self.norm()
    }
}

/// The convergence contract shared by every Newton: each residual row within
/// the absolute floor of its unknown's kind (a KCL row of a node potential in
/// amperes, `abstol`; a KVL or flow row of a branch current in volts,
/// `vntol`; a state's defining row `vntol`), and each update within
/// `reltol * |x| + floor` with the floor of the unknown's kind (`vntol` for a
/// potential or a state, `abstol` for a current). Real or complex iterates:
/// harmonic balance tests the same contract per harmonic. A solver that
/// pins rows adds a per-row residual floor on top (`row_floor`).
pub(crate) struct Criterion<'a> {
    kinds: &'a [UnknownKind],
    pub conv: Convergence,
    row_floor: Option<&'a [f64]>,
}

impl<'a> Criterion<'a> {
    pub fn new(kinds: &'a [UnknownKind], conv: Convergence) -> Self {
        Criterion {
            kinds,
            conv,
            row_floor: None,
        }
    }

    /// Add a per-row residual floor (its length is the row count; `0` adds
    /// nothing).
    pub fn with_row_floor(mut self, floor: &'a [f64]) -> Self {
        self.row_floor = Some(floor);
        self
    }

    /// Absolute residual floor of row `i`.
    #[inline]
    pub fn residual_floor(&self, i: usize) -> f64 {
        let base = match self.kinds[i] {
            UnknownKind::NodeVoltage => self.conv.abstol,
            _ => self.conv.vntol,
        };
        match self.row_floor {
            Some(f) => base.max(f[i]),
            None => base,
        }
    }

    /// Absolute update floor of unknown `i`.
    #[inline]
    pub fn update_floor(&self, i: usize) -> f64 {
        match self.kinds[i] {
            UnknownKind::BranchCurrent => self.conv.abstol,
            _ => self.conv.vntol,
        }
    }

    /// Every row of `res + gmin * x` finite and within its floor.
    pub fn residual_ok<T>(&self, res: &[T], x: &[T], gmin: f64) -> bool
    where
        T: Magnitude + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T>,
    {
        (0..res.len()).all(|i| {
            let r = (res[i] + x[i] * gmin).mag();
            r.is_finite() && r <= self.residual_floor(i)
        })
    }

    /// The scaled update norm `max_i |dx_i| / (reltol |x_i| + floor_i)` and
    /// the index attaining it; `< 1` is the update half of convergence. A
    /// non-finite update scores infinite. A component within `rounding`, the
    /// move the residual's own rounding makes (where the system knows it),
    /// scores none: no iteration resolves it further.
    pub fn update_norm<T: Magnitude>(
        &self,
        dx: &[T],
        x: &[T],
        rounding: Option<&[f64]>,
    ) -> (f64, usize) {
        let mut worst = (0.0f64, 0usize);
        for i in 0..dx.len() {
            let d = dx[i].mag();
            if rounding.is_some_and(|r| d <= r[i]) {
                continue;
            }
            let sc = (self.conv.reltol * x[i].mag() + self.update_floor(i)).max(f64::MIN_POSITIVE);
            let w = if d.is_finite() { d / sc } else { f64::INFINITY };
            if w > worst.0 {
                worst = (w, i);
            }
        }
        worst
    }
}

/// Shorten `step` (the update `x - x_new`) to the junction bound of every
/// declared limit; returns the fraction kept (`1.0` when nothing limited).
pub(crate) fn limit_step(limits: &[Limit], x: &[f64], step: &mut [f64]) -> f64 {
    if limits.is_empty() {
        return 1.0;
    }
    let alpha = limiting::fraction_by(limits, x, |i| x[i] - step[i]);
    if alpha < 1.0 {
        for s in step.iter_mut() {
            *s *= alpha;
        }
    }
    alpha
}

/// The forward backtracking search: `x -= alpha * step` with the largest
/// `alpha` (from `1`, halving `tries - 1` times) whose trial residual norm is
/// below `fnorm`, or at most `floor`, the rounding the norm is computed with
/// (below it two norms tell nothing apart, and a step refused on that noise
/// would never be taken); when every probe fails the last, smallest fraction is taken
/// anyway (the direction is still a descent direction of the linear model).
/// Real or complex iterates and steps; `trial` is the caller's scratch of
/// `x`'s length. The last probe `eval_norm` sees is the returned iterate,
/// so what it leaves behind (a residual) is the new iterate's.
pub(crate) fn backtrack<T>(
    x: &mut [T],
    step: &[T],
    trial: &mut [T],
    fnorm: f64,
    floor: f64,
    tries: usize,
    mut eval_norm: impl FnMut(&[T]) -> f64,
) -> f64
where
    T: Copy + std::ops::Sub<Output = T> + std::ops::Mul<f64, Output = T>,
{
    let n = x.len();
    let mut bt = Backtrack::new(tries);
    loop {
        let alpha = bt.alpha();
        for i in 0..n {
            trial[i] = x[i] - step[i] * alpha;
        }
        let f = eval_norm(trial);
        if f < fnorm || f <= floor {
            x.copy_from_slice(trial);
            return alpha;
        }
        if bt.shrink().is_none() {
            x.copy_from_slice(trial);
            return alpha;
        }
    }
}

/// Stall guard shared by the Newton loops: the residual norm must fall by
/// `1 - NEWTON_STALL_FACTOR` over any `NEWTON_STALL_WINDOW` iterations. A
/// loop that cannot is stalled (a near-floating node the linearisation cannot
/// pin, a limited step that no longer moves) and returns early instead of
/// running out its budget; the cascade's next stage is the answer to a stall,
/// not more of the same iteration.
#[derive(Default)]
pub(crate) struct StallGuard {
    history: std::collections::VecDeque<f64>,
}

impl StallGuard {
    /// Forget the iterations of an earlier solve, keeping the buffer.
    fn reset(&mut self) {
        self.history.clear();
    }

    /// Record this iteration's residual norm; `true` when the last window
    /// made too little progress.
    pub fn stalled(&mut self, fnorm: f64) -> bool {
        self.history.push_back(fnorm);
        if self.history.len() <= NEWTON_STALL_WINDOW {
            return false;
        }
        let oldest = self.history.pop_front().unwrap_or(f64::INFINITY);
        fnorm.is_finite() && fnorm > NEWTON_STALL_FACTOR * oldest
    }
}

/// Backtracking state shared by the forward and the retroactive search: the
/// fraction of the current probe and the probes left.
pub(crate) struct Backtrack {
    alpha: f64,
    tries_left: usize,
}

impl Backtrack {
    /// A search with `tries` probes in total (the full step is the first).
    pub fn new(tries: usize) -> Self {
        Backtrack {
            alpha: 1.0,
            tries_left: tries.max(1) - 1,
        }
    }

    /// The fraction to probe next after a failed probe; `None` when the
    /// budget is spent.
    pub fn shrink(&mut self) -> Option<f64> {
        if self.tries_left == 0 {
            return None;
        }
        self.tries_left -= 1;
        self.alpha *= LINE_SEARCH_SHRINK;
        Some(self.alpha)
    }

    /// The fraction of the current probe.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

// --- the core -------------------------------------------------------------

/// A scalar a Newton iterates over: real (DC, transient stages) or complex
/// (harmonic balance).
pub(crate) trait Scalar:
    Copy
    + Default
    + Magnitude
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<f64, Output = Self>
{
}

impl<T> Scalar for T where
    T: Copy
        + Default
        + Magnitude
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<f64, Output = T>
{
}

/// What a Newton solves: a residual `r(x)`, its Jacobian and the linear
/// solves on it, every shift the system carries (a gmin shunt, a pin, a
/// homotopy) part of `r` and `J`. The step is `x -= J^-1 r`. The defaults are
/// the plain system: no rounding measure, no limiting, the 2-norm.
pub(crate) trait System<T: Scalar> {
    /// The residual at `x` into `res`, and with `jacobian` the Jacobian
    /// there (evaluated, for [`factor`](Self::factor)). `false` where the
    /// evaluation fails.
    fn eval(&mut self, x: &[T], jacobian: bool, res: &mut [T]) -> bool;

    /// Factor the Jacobian the last evaluation with `jacobian` gave; `false`
    /// where it is singular.
    fn factor(&mut self) -> bool;

    /// `dx = J^-1 rhs` on the current factors; `false` where there are none
    /// or the solve fails.
    fn solve(&mut self, rhs: &[T], dx: &mut [T]) -> bool;

    /// [`solve`](Self::solve) on the factors of an earlier iteration, before
    /// this one's Jacobian is factored: the early acceptance probe (see
    /// [`Policy::early_accept`]). `false` where there are none.
    fn probe(&mut self, _rhs: &[T], _dx: &mut [T]) -> bool {
        false
    }

    /// The chord solve of the composite step (see [`Policy::composite`]) on
    /// the factors just built; `false` where the system offers none.
    fn chord(&mut self, _rhs: &[T], _dx: &mut [T]) -> bool {
        false
    }

    /// Whether `res` at `x` is within the rounding of the terms each row
    /// sums (solved: no step can do better), and the 2-norm of those terms
    /// (below it two residual norms tell nothing apart).
    fn rounding(&mut self, _x: &[T], _res: &[T]) -> (bool, f64) {
        (false, 0.0)
    }

    /// Per unknown, the move the residual's own rounding makes through the
    /// iteration matrix, where the system knows it: an update within it is
    /// the rounding's, not the iteration's.
    fn update_rounding(&self) -> Option<&[f64]> {
        None
    }

    /// An acceptance of the residual half the system grants beyond its
    /// floors (the SPICE-relative test of the node-adaptive fallback).
    fn accept_residual(&mut self, _x: &[T], _res: &[T]) -> bool {
        false
    }

    /// Shape `step` (the update `x - x_new`): shorten it to the device
    /// limits, keep what the iterate keeps (a real DC harmonic); the
    /// fraction kept.
    fn limit(&self, _x: &[T], _step: &mut [T]) -> f64 {
        1.0
    }

    /// The residual norm the globalization and the stall rule compare.
    fn norm(&self, _x: &[T], res: &[T]) -> f64 {
        res.iter().map(|r| r.mag() * r.mag()).sum::<f64>().sqrt()
    }

    /// Whether the factors are stale (a modified Newton re-evaluates the
    /// Jacobian then); a full Newton does every iteration.
    fn stale(&self) -> bool {
        true
    }

    /// Mark the factors stale (a modified Newton that contracts too slowly).
    fn mark_stale(&mut self) {}
}

/// When a Newton has converged: the residual within its floors *and* the
/// update within its floors -- or the residual within its rounding, which no
/// step improves on. A half that is off holds; with both off only the
/// rounding converges.
pub(crate) struct Contract<'a> {
    pub criterion: Criterion<'a>,
    /// The residual half: every row within [`Criterion::residual_floor`].
    pub residual: bool,
    /// The update half: [`Criterion::update_norm`] below this fraction.
    pub update: Option<f64>,
}

/// How the Jacobian is kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Jacobian {
    /// Evaluated and factored every iteration.
    Full,
    /// Kept while it contracts (modified Newton): re-evaluated when the
    /// system says it is stale, marked so when the update norm falls by less
    /// than `theta` per iteration or would not reach the tolerance in the
    /// iterations left (Hairer-Wanner IV.8).
    Modified { theta: f64 },
}

/// How a step is globalized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Globalization {
    /// The full (limited) step.
    None,
    /// The forward backtracking search ([`backtrack`]), `tries` probes.
    LineSearch { tries: usize },
    /// The retroactive search ([`Backtrack`]): the full step is taken, and
    /// one whose residual grew by more than `growth` is retracted to a
    /// fraction (each retraction an iteration), `tries` probes.
    Retract { growth: f64, tries: usize },
}

/// How a Newton iterates (see [`solve`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Policy {
    pub max_iter: usize,
    pub jacobian: Jacobian,
    pub globalization: Globalization,
    /// Device limiting of every step.
    pub limiting: bool,
    /// The stall rule ([`StallGuard`]) while the residual half fails.
    pub stall: bool,
    /// Once the residual half holds, test the update half with the previous
    /// iteration's factors before factoring this one's (SPICE's last-step
    /// update test: near a solution the stale factors differ by `O(|dx|)`).
    pub early_accept: bool,
    /// After a full line-searched step, one chord step on the factors just
    /// built (Traub's composite step), taken where it contracts the residual.
    pub composite: bool,
    /// Where it does not converge, return the lowest-residual iterate rather
    /// than the last (a damped iteration that drifts back up at its end).
    pub keep_best: bool,
    /// Log every iteration (residual norm, update norm, largest step) at
    /// DEBUG, for classifying non-convergence.
    pub trace: bool,
}

/// What a Newton came to: converged or not, and the iterations it took.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Outcome {
    pub converged: bool,
    pub iters: usize,
}

/// The vectors a Newton works in, kept by a caller that solves many times.
#[derive(Default)]
pub(crate) struct Scratch<T> {
    res: Vec<T>,
    dx: Vec<T>,
    trial: Vec<T>,
    prev: Vec<T>,
    best: Vec<T>,
    stall: StallGuard,
}

impl<T: Scalar> Scratch<T> {
    fn fit(&mut self, n: usize) {
        for v in [
            &mut self.res,
            &mut self.dx,
            &mut self.trial,
            &mut self.prev,
            &mut self.best,
        ] {
            v.resize(n, T::default());
        }
    }
}

/// Solve `sys` from `x` (the solution on return, else the last iterate, or
/// the best where [`Policy::keep_best`]) as `policy` says, to `contract`: the
/// one Newton loop of the solver.
pub(crate) fn solve<T, S>(
    sys: &mut S,
    x: &mut [T],
    contract: &Contract<'_>,
    policy: &Policy,
    ws: &mut Scratch<T>,
) -> Outcome
where
    T: Scalar,
    S: System<T>,
{
    ws.fit(x.len());
    let mut best_fnorm = f64::INFINITY;
    let out = iterate(sys, x, contract, policy, ws, &mut best_fnorm);
    if !out.converged && policy.keep_best && best_fnorm.is_finite() {
        x.copy_from_slice(&ws.best);
    }
    out
}

/// [`solve`]'s loop, the best residual norm met in `best_fnorm` (its
/// iterate in `ws.best`) where [`Policy::keep_best`].
fn iterate<T, S>(
    sys: &mut S,
    x: &mut [T],
    contract: &Contract<'_>,
    policy: &Policy,
    ws: &mut Scratch<T>,
    best_fnorm: &mut f64,
) -> Outcome
where
    T: Scalar,
    S: System<T>,
{
    let n = x.len();
    let Scratch {
        res,
        dx,
        trial,
        prev,
        best,
        stall,
    } = ws;
    stall.reset();
    let (mut prev_wn, mut prev_fnorm) = (f64::INFINITY, f64::INFINITY);
    let mut retract: Option<Backtrack> = None;
    let fail = |iters| Outcome {
        converged: false,
        iters,
    };
    let done = |iters| Outcome {
        converged: true,
        iters,
    };
    let update = |dx: &[T], x: &[T], rounding: Option<&[f64]>| -> (bool, f64) {
        match contract.update {
            Some(tol) => {
                let wn = contract.criterion.update_norm(dx, x, rounding).0;
                (wn < tol, wn)
            }
            None => (true, 0.0),
        }
    };
    let halves = contract.residual || contract.update.is_some();
    for it in 0..policy.max_iter {
        let jacobian = match policy.jacobian {
            Jacobian::Full => true,
            Jacobian::Modified { .. } => sys.stale(),
        };
        if !sys.eval(x, jacobian, res) {
            return fail(it);
        }
        let fnorm = sys.norm(x, res);
        // The retroactive search: a step whose residual grew (or broke down)
        // is retracted before its Jacobian is used.
        if let Globalization::Retract { growth, tries } = policy.globalization {
            if prev_fnorm.is_finite() && !(fnorm <= growth * prev_fnorm) {
                let bt = retract.get_or_insert_with(|| Backtrack::new(tries));
                if let Some(alpha) = bt.shrink() {
                    for i in 0..n {
                        x[i] = prev[i] - dx[i] * alpha;
                    }
                    continue;
                }
            }
            retract = None;
        }
        if !fnorm.is_finite() {
            return fail(it);
        }
        if policy.keep_best && fnorm < *best_fnorm {
            *best_fnorm = fnorm;
            best.copy_from_slice(x);
        }
        let res_ok = !contract.residual
            || contract.criterion.residual_ok(res, x, 0.0)
            || sys.accept_residual(x, res);
        let (rounded, floor) = sys.rounding(x, res);
        if rounded {
            return done(it);
        }
        // the residual half alone: solved where it holds, no step to take
        if contract.residual && contract.update.is_none() && res_ok {
            return done(it + 1);
        }
        if policy.stall && !res_ok && stall.stalled(fnorm) {
            return fail(it);
        }
        if halves
            && policy.early_accept
            && res_ok
            && sys.probe(res, dx)
            && update(dx, x, sys.update_rounding()).0
        {
            apply(x, dx);
            return done(it);
        }
        if jacobian && !sys.factor() {
            return fail(it);
        }
        if !sys.solve(res, dx) {
            return fail(it);
        }
        let (update_ok, wn) = update(dx, x, sys.update_rounding());
        if policy.trace {
            let (k, mx) = (dx.iter().enumerate())
                .map(|(i, d)| (i, d.mag()))
                .fold((0, 0.0f64), |a, b| if b.1 > a.1 { b } else { a });
            sane_core::log::debug(&format!(
                "NEWTON it={it} fnorm={fnorm:.3e} res_ok={res_ok} wn={wn:.3e} maxdx={mx:.3e}@{k}"
            ));
        }
        if policy.limiting {
            sys.limit(x, dx);
        }
        if halves && res_ok && update_ok {
            apply(x, dx);
            return done(it + 1);
        }
        let alpha = match policy.globalization {
            Globalization::None => {
                apply(x, dx);
                1.0
            }
            Globalization::LineSearch { tries } => {
                // the last probe leaves its residual: the new iterate's
                backtrack(x, dx, trial, fnorm, floor, tries, |t| {
                    match sys.eval(t, false, res) {
                        true => sys.norm(t, res),
                        false => f64::INFINITY,
                    }
                })
            }
            Globalization::Retract { .. } => {
                prev.copy_from_slice(x);
                apply(x, dx);
                1.0
            }
        };
        if policy.composite && alpha == 1.0 && policy.globalization != Globalization::None {
            composite(sys, x, res, dx, trial, policy.limiting);
        }
        if let Jacobian::Modified { theta } = policy.jacobian {
            if !jacobian {
                let ratio = wn / prev_wn;
                let tol = contract.update.unwrap_or(1.0);
                let left = (policy.max_iter - it - 1) as f64;
                let predicted = match ratio < 1.0 {
                    true => ratio.powf(left) * wn / (1.0 - ratio),
                    false => f64::INFINITY,
                };
                if ratio > theta || predicted > tol {
                    sys.mark_stale();
                }
            }
        }
        prev_wn = wn;
        prev_fnorm = fnorm;
    }
    fail(policy.max_iter)
}

/// Traub's composite step after a full step to `x`: one chord step on the
/// factors just built from the residual `res` there, taken where it
/// contracts the residual.
fn composite<T: Scalar, S: System<T>>(
    sys: &mut S,
    x: &mut [T],
    res: &mut [T],
    dx: &mut [T],
    trial: &mut [T],
    limiting: bool,
) {
    let f1 = sys.norm(x, res);
    if !sys.chord(res, dx) {
        return;
    }
    if limiting {
        sys.limit(x, dx);
    }
    for i in 0..x.len() {
        trial[i] = x[i] - dx[i];
    }
    if sys.eval(trial, false, res) && sys.norm(trial, res) < f1 {
        x.copy_from_slice(trial);
    }
}

/// `x -= dx`.
fn apply<T: Scalar>(x: &mut [T], dx: &[T]) {
    for (xi, &d) in x.iter_mut().zip(dx) {
        *xi = *xi - d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sane_core::constants::LINE_SEARCH_TRIES;

    #[test]
    fn backtrack_halves_until_the_norm_drops() {
        // f(x) = x^3 - 1 at x = 3: the Newton step overshoots; the search
        // must shorten it until |f| decreases.
        let f = |x: f64| x * x * x - 1.0;
        let x0 = 3.0;
        let step = f(x0) / (3.0 * x0 * x0);
        let mut x = [x0];
        let alpha = backtrack(
            &mut x,
            &[step],
            &mut [0.0],
            f(x0).abs(),
            0.0,
            LINE_SEARCH_TRIES,
            |t| f(t[0]).abs(),
        );
        assert!((0.0..=1.0).contains(&alpha));
        assert!(f(x[0]).abs() < f(x0).abs());
    }

    #[test]
    fn stall_guard_trips_on_a_plateau_only() {
        let mut g = StallGuard::default();
        // geometric decrease: never stalled
        for k in 0..30 {
            assert!(
                !g.stalled(0.7f64.powi(k)),
                "converging iteration flagged at {k}"
            );
        }
        let mut g = StallGuard::default();
        // a plateau trips after the window
        let mut tripped = None;
        for k in 0..30 {
            if g.stalled(1e-7 * (1.0 - 0.001 * k as f64)) {
                tripped = Some(k);
                break;
            }
        }
        assert_eq!(tripped, Some(NEWTON_STALL_WINDOW));
    }

    #[test]
    fn backtrack_state_counts_probes() {
        let mut bt = Backtrack::new(3);
        assert_eq!(bt.alpha(), 1.0);
        assert!(bt.shrink().is_some());
        assert!(bt.shrink().is_some());
        assert!(bt.shrink().is_none());
    }
}
