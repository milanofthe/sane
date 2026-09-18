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

use sane_core::constants::{LINE_SEARCH_SHRINK, NEWTON_STALL_FACTOR, NEWTON_STALL_WINDOW};
use sane_dae::{Limit, UnknownKind};

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
    /// non-finite update scores infinite.
    pub fn update_norm<T: Magnitude>(&self, dx: &[T], x: &[T]) -> (f64, usize) {
        let mut worst = (0.0f64, 0usize);
        for i in 0..dx.len() {
            let d = dx[i].mag();
            let sc = (self.conv.reltol * x[i].mag() + self.update_floor(i)).max(f64::MIN_POSITIVE);
            let w = if d.is_finite() { d / sc } else { f64::INFINITY };
            if w > worst.0 {
                worst = (w, i);
            }
        }
        worst
    }

    /// Every update within `reltol * |x| + floor`.
    pub fn update_ok<T: Magnitude>(&self, dx: &[T], x: &[T]) -> bool {
        self.update_norm(dx, x).0 < 1.0
    }
}

/// Shorten `step` (the update `x - x_new`) to the junction bound of every
/// declared limit; returns the fraction kept (`1.0` when nothing limited).
pub(crate) fn limit_step(limits: &[Limit], x: &[f64], step: &mut [f64]) -> f64 {
    if limits.is_empty() {
        return 1.0;
    }
    let x_new: Vec<f64> = x.iter().zip(step.iter()).map(|(xi, s)| xi - s).collect();
    let alpha = limiting::fraction(limits, x, &x_new);
    if alpha < 1.0 {
        for s in step.iter_mut() {
            *s *= alpha;
        }
    }
    alpha
}

/// The forward backtracking search: `x -= alpha * step` with the largest
/// `alpha` (from `1`, halving `tries - 1` times) whose trial residual norm is
/// below `fnorm`; when every probe fails the last, smallest fraction is taken
/// anyway (the direction is still a descent direction of the linear model).
/// Real or complex iterates and steps.
pub(crate) fn backtrack<T>(
    x: &mut [T],
    step: &[T],
    fnorm: f64,
    tries: usize,
    mut eval_norm: impl FnMut(&[T]) -> f64,
) -> f64
where
    T: Copy + std::ops::Sub<Output = T> + std::ops::Mul<f64, Output = T>,
{
    let n = x.len();
    let mut trial: Vec<T> = x.to_vec();
    let mut bt = Backtrack::new(tries);
    loop {
        let alpha = bt.alpha();
        for i in 0..n {
            trial[i] = x[i] - step[i] * alpha;
        }
        if eval_norm(&trial) < fnorm {
            x.copy_from_slice(&trial);
            return alpha;
        }
        if bt.shrink().is_none() {
            x.copy_from_slice(&trial);
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
pub(crate) struct StallGuard {
    history: std::collections::VecDeque<f64>,
}

impl StallGuard {
    pub fn new() -> Self {
        StallGuard {
            history: std::collections::VecDeque::with_capacity(NEWTON_STALL_WINDOW + 1),
        }
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
        let alpha = backtrack(&mut x, &[step], f(x0).abs(), LINE_SEARCH_TRIES, |t| {
            f(t[0]).abs()
        });
        assert!((0.0..=1.0).contains(&alpha));
        assert!(f(x[0]).abs() < f(x0).abs());
    }

    #[test]
    fn stall_guard_trips_on_a_plateau_only() {
        let mut g = StallGuard::new();
        // geometric decrease: never stalled
        for k in 0..30 {
            assert!(
                !g.stalled(0.7f64.powi(k)),
                "converging iteration flagged at {k}"
            );
        }
        let mut g = StallGuard::new();
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
