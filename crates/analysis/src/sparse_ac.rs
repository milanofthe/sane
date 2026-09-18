//! Sparse complex small-signal system `A = G + jwC`.
//!
//! Assembled from the engine's sparse real Jacobians (`G = dF/dx`, `C = dF/dx'`)
//! and factorised with rslab's complex KLU (BTF + per-block AMD + Gilbert-
//! Peierls) -- the same backend family as the real Newton solves, lifted to the
//! complex field, never densified. Adjoint solves run as KLU transpose solves
//! on the same factors. The AC, noise, and symbolic-approximation paths build
//! on this so the small-signal structure stays sparse and the symbolic stamps
//! are preserved.

use num_complex::Complex64;
use rslab::{GeneralCsc, KluSettings, KluSolver, KluSymbolic};

/// Sparse complex system `A = G + jwC` in summed-triplet form, with on-demand
/// KLU factorisation of `A` (and transpose solves for the adjoint).
pub struct AcSystem {
    pub n: usize,
    triplets: Vec<(usize, usize, Complex64)>,
}

impl AcSystem {
    /// Assemble `A = G + jw C` from sparse real `G`/`C` given as `(rows, cols,
    /// vals)`; duplicate `(i, j)` contributions are summed.
    pub fn assemble(
        n: usize,
        g: (&[usize], &[usize], &[f64]),
        c: (&[usize], &[usize], &[f64]),
        w: f64,
    ) -> Self {
        use std::collections::HashMap;
        let mut m: HashMap<(usize, usize), Complex64> = HashMap::new();
        for k in 0..g.2.len() {
            *m.entry((g.0[k], g.1[k]))
                .or_insert(Complex64::new(0.0, 0.0)) += Complex64::new(g.2[k], 0.0);
        }
        for k in 0..c.2.len() {
            *m.entry((c.0[k], c.1[k]))
                .or_insert(Complex64::new(0.0, 0.0)) += Complex64::new(0.0, w * c.2[k]);
        }
        let triplets: Vec<(usize, usize, Complex64)> =
            m.into_iter().map(|((i, j), v)| (i, j, v)).collect();
        Self { n, triplets }
    }

    /// Build directly from pre-summed complex triplets (used by the entry
    /// pruning, which masks a subset of the system's entries per candidate).
    pub fn from_triplets(n: usize, triplets: Vec<(usize, usize, Complex64)>) -> Self {
        Self { n, triplets }
    }

    /// Factor once for combined forward + adjoint use (crate-internal): the
    /// returned KLU handle serves `solve` and `solve_transpose` from the same
    /// factors, which the all-parameter gradients rely on.
    pub(crate) fn factored(&self) -> Option<KluSolver<Complex64>> {
        self.factor()
    }

    fn factor(&self) -> Option<KluSolver<Complex64>> {
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for &(i, j, val) in &self.triplets {
            r.push(i);
            c.push(j);
            v.push(val);
        }
        let a = GeneralCsc::from_triplets(self.n, &r, &c, &v).ok()?;
        KluSolver::factor(&a, &KluSettings::default()).ok()
    }

    /// Solve `A x = b`.
    pub fn solve(&self, b: &[Complex64]) -> Option<Vec<Complex64>> {
        self.factor()?.solve(b).ok()
    }

    /// Solve the adjoint system `A^T x = b` (plain transpose, matching the
    /// holomorphic adjoint the sensitivity derivations use), on the same
    /// factorization path as the forward solve.
    pub fn solve_transpose(&self, b: &[Complex64]) -> Option<Vec<Complex64>> {
        self.factor()?.solve_transpose(b).ok()
    }

    /// Matrix-free mat-vec `y = A v`. Test-only for now (verifies the sparse
    /// solve via an `A (A^{-1} b) = b` round-trip); the scaffolding for a future
    /// matrix-free iterative AC path.
    #[cfg(test)]
    pub fn apply(&self, v: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.n];
        for &(i, j, val) in &self.triplets {
            y[i] += val * v[j];
        }
        y
    }

    /// Matrix-free mat-vec `y = A^T v`. Test-only (see [`Self::apply`]).
    #[cfg(test)]
    pub fn apply_transpose(&self, v: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.n];
        for &(i, j, val) in &self.triplets {
            y[j] += val * v[i];
        }
        y
    }

    /// The assembled nonzeros (summed), for pruning and structure inspection.
    pub fn entries(&self) -> &[(usize, usize, Complex64)] {
        &self.triplets
    }
}

/// AC system `A = G + jwC` for a **frequency sweep**, with the symbolic
/// analysis computed once and reused at every frequency.
///
/// Across a sweep only the numeric values change with `w`; the sparsity pattern
/// of `A` (and thus KLU's BTF + fill-reducing ordering) is fixed. So the
/// analysis is paid once, and per frequency only the complex values are
/// refilled and numerically factored. A sweep worker obtained via
/// [`solver`](Self::solver) goes further: its first frequency factors with full
/// pivoting and every following one replays the frozen pivot sequence (KLU
/// numeric-only `refactor`, no DFS, no pivot search) -- values move smoothly in
/// `w`, so the pivots stay valid and the replay transparently falls back to a
/// full factor if one degenerates. Set `transpose` to build `A^T` instead (the
/// adjoint system the noise analysis solves).
pub struct SymbolicAc {
    n: usize,
    /// Fixed CSC pattern of `A` (or `A^T`), deduplicated.
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    /// CSC slot of each `G` entry / each `C` entry (duplicates share slots).
    slot_g: Vec<usize>,
    slot_c: Vec<usize>,
    slot_d: Vec<usize>,
    /// Real `G` and `C` values; the per-frequency value set is `g + jw c`.
    g_v: Vec<f64>,
    c_v: Vec<f64>,
    /// transport-delay coupling: value and delay per entry (see new_with_delays)
    d_v: Vec<f64>,
    d_tau: Vec<f64>,
    sym: KluSymbolic,
}

impl SymbolicAc {
    /// Build the reusable symbolic analysis of `A = G + jwC` (or `A^T` if
    /// `transpose`) from sparse real `G`/`C` as `(rows, cols, vals)`. Duplicate
    /// `(i, j)` contributions are summed, matching [`AcSystem::assemble`].
    /// `None` if the pattern is degenerate (structurally singular).
    pub fn new(
        n: usize,
        g: (&[usize], &[usize], &[f64]),
        c: (&[usize], &[usize], &[f64]),
        transpose: bool,
    ) -> Option<Self> {
        Self::new_with_delays(n, g, c, (&[], &[], &[], &[]), transpose)
    }

    /// As [`new`](Self::new), plus transport-delay coupling: entry `k` adds
    /// `d.2[k] * e^{-jω d.3[k]}` at `(d.0[k], d.1[k])` -- the frequency-domain
    /// image of `hist_k(t) = x_{src}(t - τ_k)` (rows from `∂F/∂hist`, columns
    /// the delayed source unknowns).
    pub fn new_with_delays(
        n: usize,
        g: (&[usize], &[usize], &[f64]),
        c: (&[usize], &[usize], &[f64]),
        d: (&[usize], &[usize], &[f64], &[f64]),
        transpose: bool,
    ) -> Option<Self> {
        let ng = g.2.len();
        let nc = c.2.len();
        let total = ng + nc + d.2.len();
        // Entry list in g ++ c ++ delay order, transposed at build time if requested.
        let entry = |k: usize| -> (usize, usize) {
            let (i, j) = if k < ng {
                (g.0[k], g.1[k])
            } else if k < ng + nc {
                (c.0[k - ng], c.1[k - ng])
            } else {
                (d.0[k - ng - nc], d.1[k - ng - nc])
            };
            if transpose {
                (j, i)
            } else {
                (i, j)
            }
        };
        // Deduplicate into CSC with per-entry slots (same scheme as the real
        // backend's SparsePattern).
        let mut order: Vec<usize> = (0..total).collect();
        order.sort_unstable_by_key(|&k| {
            let (i, j) = entry(k);
            (j, i)
        });
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_idx: Vec<usize> = Vec::with_capacity(total);
        let mut slot = vec![0usize; total];
        let (mut prev_r, mut prev_c) = (usize::MAX, usize::MAX);
        for &k in &order {
            let (r, c_) = entry(k);
            if r >= n || c_ >= n {
                return None;
            }
            if r != prev_r || c_ != prev_c {
                row_idx.push(r);
                col_ptr[c_ + 1] += 1;
                (prev_r, prev_c) = (r, c_);
            }
            slot[k] = row_idx.len() - 1;
        }
        for j in 0..n {
            col_ptr[j + 1] += col_ptr[j];
        }
        let pattern = GeneralCsc {
            n,
            col_ptr: col_ptr.clone(),
            row_idx: row_idx.clone(),
            values: vec![Complex64::new(1.0, 0.0); row_idx.len()],
        };
        let sym = KluSymbolic::analyze(&pattern).ok()?;
        let (slot_g, slot_c, slot_d) = (
            slot[..ng].to_vec(),
            slot[ng..ng + nc].to_vec(),
            slot[ng + nc..].to_vec(),
        );
        Some(Self {
            n,
            col_ptr,
            row_idx,
            slot_g,
            slot_c,
            slot_d,
            g_v: g.2.to_vec(),
            c_v: c.2.to_vec(),
            d_v: d.2.to_vec(),
            d_tau: d.3.to_vec(),
            sym,
        })
    }

    /// Scatter `g + jw c` into a fresh CSC over the fixed pattern.
    fn scatter(&self, w: f64) -> GeneralCsc<Complex64> {
        let mut vals = vec![Complex64::new(0.0, 0.0); self.row_idx.len()];
        for (k, &gv) in self.g_v.iter().enumerate() {
            vals[self.slot_g[k]] += Complex64::new(gv, 0.0);
        }
        for (k, &cv) in self.c_v.iter().enumerate() {
            vals[self.slot_c[k]] += Complex64::new(0.0, w * cv);
        }
        for (k, &dv) in self.d_v.iter().enumerate() {
            vals[self.slot_d[k]] += dv * Complex64::from_polar(1.0, -w * self.d_tau[k]);
        }
        GeneralCsc {
            n: self.n,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values: vals,
        }
    }

    /// A sweep worker holding per-thread factorization state: first call
    /// factors with full pivoting, subsequent calls run KLU's numeric-only
    /// refactor on the refreshed values (transparent full-factor fallback).
    /// Each rayon worker gets its own (e.g. via `map_init`).
    pub fn solver(&self) -> AcSweepSolver<'_> {
        AcSweepSolver {
            sys: self,
            csc: self.scatter(0.0),
            solver: None,
        }
    }
}

/// Per-worker sweep state for [`SymbolicAc`]; see [`SymbolicAc::solver`].
pub struct AcSweepSolver<'a> {
    sys: &'a SymbolicAc,
    /// Scratch CSC (pattern fixed, values rewritten per frequency).
    csc: GeneralCsc<Complex64>,
    solver: Option<KluSolver<Complex64>>,
}

impl AcSweepSolver<'_> {
    /// Solve at angular frequency `w` (refactor fast path; `None` on a
    /// singular system at this frequency).
    pub fn solve(&mut self, w: f64, b: &[Complex64]) -> Option<Vec<Complex64>> {
        let sys = self.sys;
        for v in self.csc.values.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }
        for (k, &gv) in sys.g_v.iter().enumerate() {
            self.csc.values[sys.slot_g[k]] += Complex64::new(gv, 0.0);
        }
        for (k, &cv) in sys.c_v.iter().enumerate() {
            self.csc.values[sys.slot_c[k]] += Complex64::new(0.0, w * cv);
        }
        for (k, &dv) in sys.d_v.iter().enumerate() {
            self.csc.values[sys.slot_d[k]] += dv * Complex64::from_polar(1.0, -w * sys.d_tau[k]);
        }
        if let Some(s) = self.solver.as_mut() {
            if s.refactor(&self.csc).is_ok() {
                return s.solve(b).ok();
            }
        }
        match sys.sym.factor(&self.csc, &KluSettings::default()) {
            Ok(s) => {
                let x = s.solve(b).ok();
                self.solver = Some(s);
                x
            }
            Err(_) => {
                self.solver = None;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 3x3 complex system, sparse assembly + solve checked against a hand solve,
    // plus the mat-vec round-trip A (A^{-1} b) = b.
    #[test]
    fn sparse_complex_solve() {
        // G (real part) and C (imag part scaled by w) as triplets.
        let g = (
            vec![0usize, 1, 2, 0],
            vec![0usize, 1, 2, 1],
            vec![2.0, 3.0, 4.0, 1.0],
        );
        let c = (vec![1usize], vec![1usize], vec![5.0]);
        let w = 2.0;
        let sys = AcSystem::assemble(3, (&g.0, &g.1, &g.2), (&c.0, &c.1, &c.2), w);
        // A = [[2, 1, 0],[0, 3+10j, 0],[0,0,4]]
        let b = vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(0.0, 20.0),
            Complex64::new(8.0, 0.0),
        ];
        let x = sys.solve(&b).expect("solve");
        // round-trip: A x == b
        let ax = sys.apply(&x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).norm() < 1e-9,
                "row {i}: {:?} vs {:?}",
                ax[i],
                b[i]
            );
        }
        // transpose solve round-trip: A^T y == b
        let y = sys.solve_transpose(&b).expect("solveT");
        let aty = sys.apply_transpose(&y);
        for i in 0..3 {
            assert!((aty[i] - b[i]).norm() < 1e-9);
        }
    }

    // The reusable symbolic factorisation must match the per-call assemble+solve
    // path exactly, for both the system and its adjoint, across frequencies --
    // and the refactor-based sweep worker must match both.
    #[test]
    fn symbolic_matches_assemble() {
        // Overlapping G/C entries (so the slot map must sum duplicates), plus an
        // off-diagonal that makes A non-symmetric (so transpose is a real test).
        let g = (
            vec![0usize, 1, 2, 0, 2],
            vec![0usize, 1, 2, 1, 0],
            vec![2.0, 3.0, 4.0, 1.0, 0.5],
        );
        let c = (vec![1usize, 2, 0], vec![1usize, 2, 0], vec![5.0, 1.5, 0.7]);
        let b = vec![
            Complex64::new(4.0, -1.0),
            Complex64::new(0.0, 20.0),
            Complex64::new(8.0, 3.0),
        ];
        let sym = SymbolicAc::new(3, (&g.0, &g.1, &g.2), (&c.0, &c.1, &c.2), false).expect("sym");
        let symt = SymbolicAc::new(3, (&g.0, &g.1, &g.2), (&c.0, &c.1, &c.2), true).expect("symT");
        let mut sweep = sym.solver();
        let mut sweep_t = symt.solver();
        for &w in &[0.0, 1.0, 7.5, 1e3] {
            let ref_sys = AcSystem::assemble(3, (&g.0, &g.1, &g.2), (&c.0, &c.1, &c.2), w);
            let want = ref_sys.solve(&b).expect("ref solve");
            let got_sw = sweep.solve(w, &b).expect("sweep solve");
            let want_t = ref_sys.solve_transpose(&b).expect("ref solveT");
            let got_t_sw = sweep_t.solve(w, &b).expect("sweep solveT");
            for i in 0..3 {
                assert!(
                    (got_sw[i] - want[i]).norm() < 1e-9,
                    "sweep w={w} row {i}: {:?} vs {:?}",
                    got_sw[i],
                    want[i]
                );
                assert!(
                    (got_t_sw[i] - want_t[i]).norm() < 1e-9,
                    "sweep w={w}^T row {i}"
                );
            }
        }
    }
}
