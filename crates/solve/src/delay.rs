//! Delay history: the hot data structure behind transport delays (`absdelay`,
//! ideal transmission lines). Stores the accepted-step knots `(t, y, y')` of
//! every delayed signal and answers `y(t - tau)` by cubic Hermite
//! interpolation -- the same order as the integrator's dense output, so the
//! delayed excitation does not degrade the method.
//!
//! Built for the inner loop (one query per delay per stage per step):
//!
//! - **Power-of-two ring** with mask indexing: amortized O(1) append with
//!   doubling growth, no modulo divisions, bounded memory via horizon
//!   eviction (knots older than `t_last - horizon` are dropped; one knot
//!   before the horizon is kept so every in-horizon query stays bracketed).
//! - **SoA layout**: per-signal value/derivative rings, so a query touches
//!   two adjacent elements of two arrays -- cache-friendly for the common
//!   per-delay access pattern (each delay has its own target time).
//! - **Cursor hints**: stage query times march monotonically (with small
//!   retreats after rejected steps), so each signal remembers its last
//!   bracket; the fast path is O(1), with a few linear probes and a binary
//!   search fallback for arbitrary access.
//!
//! Queries before the first knot clamp to the initial value (the DC history
//! convention: everything before `t0` is the operating point). Queries beyond
//! the last knot clamp to the newest value; the integrator's `h <= tau` cap
//! guarantees in-range brackets during normal stepping, so the clamps only
//! fire at the boundaries.

/// Growable power-of-two ring of `f64` with head/tail eviction.
#[derive(Debug, Clone)]
struct Ring {
    buf: Vec<f64>,
    mask: usize,
    head: usize,
    len: usize,
}

impl Ring {
    fn with_capacity(cap: usize) -> Ring {
        let cap = cap.next_power_of_two().max(8);
        Ring {
            buf: vec![0.0; cap],
            mask: cap - 1,
            head: 0,
            len: 0,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> f64 {
        debug_assert!(i < self.len);
        // SAFETY-free fast path: mask keeps the index in bounds by construction.
        self.buf[(self.head + i) & self.mask]
    }

    #[inline]
    fn push(&mut self, v: f64) {
        if self.len == self.buf.len() {
            self.grow();
        }
        let idx = (self.head + self.len) & self.mask;
        self.buf[idx] = v;
        self.len += 1;
    }

    /// Drop the `k` oldest entries (front).
    #[inline]
    fn pop_front(&mut self, k: usize) {
        debug_assert!(k <= self.len);
        self.head = (self.head + k) & self.mask;
        self.len -= k;
    }

    #[cold]
    fn grow(&mut self) {
        let old_cap = self.buf.len();
        let mut buf = vec![0.0; old_cap * 2];
        for i in 0..self.len {
            buf[i] = self.buf[(self.head + i) & self.mask];
        }
        self.buf = buf;
        self.mask = self.buf.len() - 1;
        self.head = 0;
    }
}

/// Accepted-step history of every delayed signal, with Hermite evaluation.
#[derive(Debug, Clone)]
pub struct DelayHistory {
    /// Knot times, strictly increasing (shared by all signals).
    times: Ring,
    /// Per-signal knot values and one-sided time-derivatives (SoA). A knot on
    /// a source kink has two distinct slopes: `ders_in` is the left limit
    /// (closing the interval that ends here), `ders_out` the right limit
    /// (opening the next one). `push` seeds both with the same value;
    /// `patch_last_out` refines the right limit once the following step has
    /// been accepted and its start rate is known.
    vals: Vec<Ring>,
    ders_in: Vec<Ring>,
    ders_out: Vec<Ring>,
    /// Eviction horizon: keep knots covering `[t_last - horizon, t_last]`
    /// plus one bracketing knot before the window.
    horizon: f64,
    /// Per-signal cursor: index of the last bracket's left knot.
    cursors: Vec<usize>,
}

impl DelayHistory {
    /// A history for `n_signals` delayed signals with eviction horizon
    /// `horizon` (the maximum delay; pass `f64::INFINITY` to keep everything,
    /// e.g. for a stored forward pass an adjoint sweeps later).
    pub fn new(n_signals: usize, horizon: f64) -> DelayHistory {
        DelayHistory {
            times: Ring::with_capacity(64),
            vals: (0..n_signals).map(|_| Ring::with_capacity(64)).collect(),
            ders_in: (0..n_signals).map(|_| Ring::with_capacity(64)).collect(),
            ders_out: (0..n_signals).map(|_| Ring::with_capacity(64)).collect(),
            horizon,
            cursors: vec![0; n_signals],
        }
    }

    /// Number of stored knots.
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.len == 0
    }

    /// Newest knot time, or `None` when empty.
    #[inline]
    pub fn last_time(&self) -> Option<f64> {
        (self.times.len > 0).then(|| self.times.get(self.times.len - 1))
    }

    /// Append one accepted knot: time `t` (must exceed the newest knot;
    /// equal-time pushes REPLACE the newest knot, which is how breakpoint
    /// re-landings and the initial condition update cleanly), the delayed
    /// signals' values and their time-derivatives (seeding both one-sided
    /// slopes; refine the right limit later via [`Self::patch_last_out`]).
    pub fn push(&mut self, t: f64, vals: &[f64], ders: &[f64]) {
        debug_assert_eq!(vals.len(), self.vals.len());
        debug_assert_eq!(ders.len(), self.ders_in.len());
        let n = self.times.len;
        if n > 0 {
            let t_last = self.times.get(n - 1);
            if t <= t_last {
                // replace the newest knot (same time re-land / IC refresh)
                let idx = (self.times.head + n - 1) & self.times.mask;
                self.times.buf[idx] = t_last.max(t);
                for (s, v) in self.vals.iter_mut().zip(vals) {
                    let i = (s.head + s.len - 1) & s.mask;
                    s.buf[i] = *v;
                }
                for (s, d) in self.ders_in.iter_mut().zip(ders) {
                    let i = (s.head + s.len - 1) & s.mask;
                    s.buf[i] = *d;
                }
                for (s, d) in self.ders_out.iter_mut().zip(ders) {
                    let i = (s.head + s.len - 1) & s.mask;
                    s.buf[i] = *d;
                }
                return;
            }
        }
        self.times.push(t);
        for (s, v) in self.vals.iter_mut().zip(vals) {
            s.push(*v);
        }
        for (s, d) in self.ders_in.iter_mut().zip(ders) {
            s.push(*d);
        }
        for (s, d) in self.ders_out.iter_mut().zip(ders) {
            s.push(*d);
        }
        self.evict(t);
    }

    /// Overwrite the newest knot's OUTGOING (right-limit) derivatives. Called
    /// when the step leaving that knot is accepted and its start rate is
    /// known: at a source kink (or the DC seed knot, whose slope is unknown at
    /// seed time) the two one-sided slopes differ, and the interval that the
    /// knot opens must interpolate with the right limit.
    pub fn patch_last_out(&mut self, ders: &[f64]) {
        debug_assert_eq!(ders.len(), self.ders_out.len());
        if self.times.len == 0 {
            return;
        }
        for (s, d) in self.ders_out.iter_mut().zip(ders) {
            let i = (s.head + s.len - 1) & s.mask;
            s.buf[i] = *d;
        }
    }

    /// Rewind the history to `t` (drop every knot with time > `t`): called
    /// when the integrator restarts from an earlier accepted state.
    pub fn truncate_after(&mut self, t: f64) {
        let mut n = self.times.len;
        while n > 0 && self.times.get(n - 1) > t {
            n -= 1;
        }
        let drop = self.times.len - n;
        if drop > 0 {
            // dropping from the BACK: shrink `len` (head untouched)
            self.times.len -= drop;
            for s in self.vals.iter_mut() {
                s.len -= drop;
            }
            for s in self.ders_in.iter_mut() {
                s.len -= drop;
            }
            for s in self.ders_out.iter_mut() {
                s.len -= drop;
            }
            for c in self.cursors.iter_mut() {
                *c = (*c).min(self.times.len.saturating_sub(1));
            }
        }
    }

    #[inline]
    fn evict(&mut self, t_last: f64) {
        if !self.horizon.is_finite() {
            return;
        }
        let cutoff = t_last - self.horizon;
        // keep one knot at or before the cutoff so queries down to the
        // horizon stay bracketed
        let mut drop = 0;
        while drop + 1 < self.times.len && self.times.get(drop + 1) <= cutoff {
            drop += 1;
        }
        if drop > 0 {
            self.times.pop_front(drop);
            for s in self.vals.iter_mut() {
                s.pop_front(drop);
            }
            for s in self.ders_in.iter_mut() {
                s.pop_front(drop);
            }
            for s in self.ders_out.iter_mut() {
                s.pop_front(drop);
            }
            for c in self.cursors.iter_mut() {
                *c = c.saturating_sub(drop);
            }
        }
    }

    /// Evaluate signal `sig` at time `tq` by cubic Hermite interpolation on
    /// its bracketing knots. Clamps to the first/last knot value outside the
    /// stored range (pre-history = initial value; the integrator's step cap
    /// keeps normal queries interior).
    #[inline]
    pub fn eval(&mut self, sig: usize, tq: f64) -> f64 {
        let n = self.times.len;
        debug_assert!(n > 0, "DelayHistory::eval on empty history");
        if n == 1 || tq <= self.times.get(0) {
            return self.vals[sig].get(0);
        }
        if tq >= self.times.get(n - 1) {
            return self.vals[sig].get(n - 1);
        }
        let i = self.locate(sig, tq);
        let t0 = self.times.get(i);
        let t1 = self.times.get(i + 1);
        let h = t1 - t0;
        let th = (tq - t0) / h;
        let t2 = th * th;
        let t3 = t2 * th;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + th;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        let y0 = self.vals[sig].get(i);
        let y1 = self.vals[sig].get(i + 1);
        // the interval [i, i+1] is opened by knot i (right limit) and closed
        // by knot i+1 (left limit) -- exact for piecewise-smooth histories
        let m0 = self.ders_out[sig].get(i);
        let m1 = self.ders_in[sig].get(i + 1);
        h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    }

    /// Bracket index for `tq` (`times[i] <= tq < times[i+1]`), starting from
    /// the signal's cursor: O(1) for the monotone stage-query pattern, a few
    /// linear probes for small retreats, binary search otherwise.
    #[inline]
    fn locate(&mut self, sig: usize, tq: f64) -> usize {
        let n = self.times.len;
        let mut i = self.cursors[sig].min(n - 2);
        // fast path: current bracket still valid
        if self.times.get(i) <= tq {
            // advance a few steps (queries march forward)
            let mut probes = 0;
            while i + 2 < n && self.times.get(i + 1) <= tq {
                i += 1;
                probes += 1;
                if probes > 4 {
                    i = self.bisect(tq);
                    break;
                }
            }
        } else {
            // small retreat (rejected step re-query), else bisect
            let mut probes = 0;
            loop {
                if i == 0 || self.times.get(i) <= tq {
                    break;
                }
                i -= 1;
                probes += 1;
                if probes > 4 {
                    i = self.bisect(tq);
                    break;
                }
            }
        }
        self.cursors[sig] = i;
        i
    }

    #[cold]
    fn bisect(&self, tq: f64) -> usize {
        let n = self.times.len;
        let (mut lo, mut hi) = (0usize, n - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.times.get(mid) <= tq {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hermite on knots of a cubic reproduces the cubic exactly.
    #[test]
    fn hermite_exact_on_cubics() {
        let g = |t: f64| 1.0 - 2.0 * t + 0.5 * t * t + 3.0 * t * t * t;
        let gp = |t: f64| -2.0 + t + 9.0 * t * t;
        let mut h = DelayHistory::new(1, f64::INFINITY);
        // non-uniform knots (adaptive steps)
        let knots = [0.0, 0.13, 0.4, 0.45, 0.9, 1.7];
        for &t in &knots {
            h.push(t, &[g(t)], &[gp(t)]);
        }
        for k in 0..100 {
            let t = 1.7 * k as f64 / 99.0;
            assert!((h.eval(0, t) - g(t)).abs() < 1e-12, "t={t}");
        }
    }

    #[test]
    fn clamps_outside_range() {
        let mut h = DelayHistory::new(1, f64::INFINITY);
        h.push(1.0, &[5.0], &[100.0]);
        h.push(2.0, &[7.0], &[100.0]);
        assert_eq!(h.eval(0, 0.0), 5.0); // pre-history = first knot (IC)
        assert_eq!(h.eval(0, 3.0), 7.0); // beyond newest = newest
    }

    #[test]
    fn eviction_keeps_bracket_and_wraps() {
        let mut h = DelayHistory::new(2, 1.0);
        // push far more knots than the initial capacity, with eviction active
        for k in 0..10_000 {
            let t = k as f64 * 0.01;
            h.push(t, &[t, -t], &[1.0, -1.0]);
        }
        let t_last = 9_999.0 * 0.01;
        // window plus one bracketing knot: ~102 knots for horizon 1.0 at dt 0.01
        assert!(h.len() < 140, "len={}", h.len());
        // linear signals reproduce exactly across the whole window
        for k in 0..100 {
            let tq = t_last - 1.0 + k as f64 * 0.01;
            assert!((h.eval(0, tq) - tq).abs() < 1e-9);
            assert!((h.eval(1, tq) + tq).abs() < 1e-9);
        }
    }

    /// A C0 kink (|t - 1|-shaped signal): with per-side slopes patched in, both
    /// intervals interpolate their piecewise-linear branch exactly.
    #[test]
    fn one_sided_slopes_resolve_kinks() {
        let mut h = DelayHistory::new(1, f64::INFINITY);
        h.push(0.0, &[1.0], &[0.0]); // seed knot, slope unknown yet
        h.patch_last_out(&[-1.0]); // falling branch opens here
        h.push(1.0, &[0.0], &[-1.0]); // kink: closes falling ...
        h.patch_last_out(&[1.0]); // ... opens rising
        h.push(2.0, &[1.0], &[1.0]);
        for k in 0..=20 {
            let t = 2.0 * k as f64 / 20.0;
            let expect = (t - 1.0).abs();
            assert!((h.eval(0, t) - expect).abs() < 1e-12, "t={t}");
        }
    }

    #[test]
    fn same_time_push_replaces() {
        let mut h = DelayHistory::new(1, f64::INFINITY);
        h.push(0.0, &[1.0], &[0.0]);
        h.push(1.0, &[2.0], &[0.0]);
        h.push(1.0, &[9.0], &[0.0]);
        assert_eq!(h.len(), 2);
        assert_eq!(h.eval(0, 1.0), 9.0);
    }

    #[test]
    fn truncate_after_rewinds() {
        let mut h = DelayHistory::new(1, f64::INFINITY);
        for k in 0..10 {
            h.push(k as f64, &[k as f64], &[1.0]);
        }
        h.truncate_after(4.5);
        assert_eq!(h.len(), 5);
        assert_eq!(h.last_time(), Some(4.0));
        h.push(4.25, &[4.25], &[1.0]);
        assert!((h.eval(0, 4.1) - 4.1).abs() < 1e-12);
    }

    #[test]
    fn cursor_handles_monotone_and_retreat() {
        let mut h = DelayHistory::new(1, f64::INFINITY);
        for k in 0..1000 {
            let t = k as f64 * 0.1;
            h.push(t, &[t.sin()], &[t.cos()]);
        }
        // monotone sweep
        for k in 0..5000 {
            let tq = 99.0 * k as f64 / 4999.0;
            assert!((h.eval(0, tq) - tq.sin()).abs() < 2e-5);
        }
        // retreats and random jumps
        for &tq in &[42.0, 41.9, 3.0, 98.0, 0.05, 60.0] {
            assert!((h.eval(0, tq) - tq.sin()).abs() < 2e-5);
        }
    }
}

// ======================================================================================
// Thread-local history values -- the integrator -> tape handoff.
// ======================================================================================
// `fill_inputs` runs deep inside the tape machinery with a fixed signature;
// the delay values change per stage evaluation. A thread-local slot keeps the
// handoff allocation-free and signature-free: the integrator calls
// `set_hist_values` before evaluating at a stage time, the tape input filler
// reads `hist_value(k)`. Evaluation stays on one thread, so this is exact.

use std::cell::RefCell;

std::thread_local! {
    static HIST_VALUES: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Publish the interpolated delay-history values for subsequent residual /
/// Jacobian evaluations on this thread.
pub(crate) fn set_hist_values(vals: &[f64]) {
    HIST_VALUES.with(|h| {
        let mut h = h.borrow_mut();
        h.clear();
        h.extend_from_slice(vals);
    });
}

/// The current value of delay-history input `k` (0.0 when unset -- delay-free
/// circuits never read these slots).
#[inline]
pub(crate) fn hist_value(k: usize) -> f64 {
    HIST_VALUES.with(|h| h.borrow().get(k).copied().unwrap_or(0.0))
}
