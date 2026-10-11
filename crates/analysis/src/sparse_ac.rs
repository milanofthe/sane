//! Sparse complex small-signal system `A = G + jwC`.
//!
//! Assembled from the engine's sparse real Jacobians (`G = dI/dx`, `C = dQ/dx`)
//! over one fixed layout and factorised by the solver's complex KLU
//! ([`sane_solve::ComplexPattern`]), never densified. Adjoint solves run as
//! transpose solves on the same factors. The AC, noise and S-parameter paths
//! and their derivatives build on this.

use num_complex::Complex64;
use sane_solve::{ComplexFactor, ComplexPattern, Csc};

/// AC system `A = G + jwC` for a **frequency sweep**, with the symbolic
/// analysis computed once and reused at every frequency.
///
/// Across a sweep only the numeric values change with `w`; the sparsity pattern
/// of `A` is fixed. The analysis (BTF, the maximum-product matching, the
/// fill-reducing order) runs once on the values at a reference frequency, the
/// matching being value dependent; per frequency only the values are refilled
/// in place. A sweep worker obtained via [`solver`](Self::solver) factors with
/// pivoting at its first frequency and replays the frozen pivot sequence after
/// (numeric-only refactor, no DFS, no pivot search) -- values move smoothly in
/// `w`, so the pivots stay valid, and the replay falls back to a full factor
/// if one degenerates. Set `transpose` to build `A^T` instead (the adjoint
/// system the noise analysis solves).
pub struct SymbolicAc {
    n: usize,
    pattern: ComplexPattern,
    vals: AcValues,
}

/// The values of `A = G + jwC (+ delays)` over the pattern, per frequency.
struct AcValues {
    /// The frequency-independent values: `G` summed into the pattern.
    base: Vec<Complex64>,
    /// CSC slot of each `C` entry / each delay entry (duplicates share slots).
    slot_c: Vec<usize>,
    slot_d: Vec<usize>,
    c_v: Vec<f64>,
    /// transport-delay coupling: value and delay per entry (see `new`)
    d_v: Vec<f64>,
    d_tau: Vec<f64>,
}

impl AcValues {
    /// The values at `w` into `values`.
    fn fill(&self, w: f64, values: &mut [Complex64]) {
        values.copy_from_slice(&self.base);
        for (&k, &cv) in self.slot_c.iter().zip(&self.c_v) {
            values[k].im += w * cv;
        }
        for (k, &dv) in self.d_v.iter().enumerate() {
            values[self.slot_d[k]] += dv * Complex64::from_polar(1.0, -w * self.d_tau[k]);
        }
    }
}

impl SymbolicAc {
    /// Build the reusable symbolic analysis of `A = G + jwC` (or `A^T` if
    /// `transpose`) from sparse real `G`/`C` as `(rows, cols, vals)`, analyzed
    /// on the values at angular frequency `w_ref` (the sweep's first, say).
    /// Duplicate `(i, j)` contributions are summed. `None` if the pattern is
    /// degenerate (structurally singular). The transport-delay coupling `d`
    /// (empty without delays): entry `k` adds `d.2[k] * e^{-jω d.3[k]}` at
    /// `(d.0[k], d.1[k])` -- the frequency-domain image of
    /// `hist_k(t) = src_k(t - τ_k)` (rows from `∂F/∂hist`, columns the
    /// unknowns of the delayed signal).
    pub fn new(
        n: usize,
        g: (&[usize], &[usize], &[f64]),
        c: (&[usize], &[usize], &[f64]),
        d: (&[usize], &[usize], &[f64], &[f64]),
        transpose: bool,
        w_ref: f64,
    ) -> Option<Self> {
        let (ng, nc) = (g.2.len(), c.2.len());
        // Entries in g ++ c ++ delay order, transposed at build time if asked.
        let rows: Vec<usize> = g.0.iter().chain(c.0).chain(d.0).copied().collect();
        let cols: Vec<usize> = g.1.iter().chain(c.1).chain(d.1).copied().collect();
        let csc = if transpose {
            Csc::new(n, &cols, &rows)?
        } else {
            Csc::new(n, &rows, &cols)?
        };
        let mut base = vec![Complex64::new(0.0, 0.0); csc.nnz()];
        for (k, &gv) in g.2.iter().enumerate() {
            base[csc.slot[k]] += Complex64::new(gv, 0.0);
        }
        let vals = AcValues {
            base,
            slot_c: csc.slot[ng..ng + nc].to_vec(),
            slot_d: csc.slot[ng + nc..].to_vec(),
            c_v: c.2.to_vec(),
            d_v: d.2.to_vec(),
            d_tau: d.3.to_vec(),
        };
        let mut at_ref = vec![Complex64::new(0.0, 0.0); csc.nnz()];
        vals.fill(w_ref, &mut at_ref);
        let pattern = ComplexPattern::new(csc, &at_ref)?;
        Some(Self { n, pattern, vals })
    }

    /// A sweep worker holding per-thread factorization state and buffers:
    /// first call factors with pivoting, subsequent calls run the numeric-only
    /// refactor on the values refilled in place (full-factor fallback). Each
    /// rayon worker gets its own (e.g. via `map_init`).
    pub fn solver(&self) -> AcSweepSolver<'_> {
        AcSweepSolver {
            sys: self,
            fac: self.pattern.factorizer(),
            factored: None,
            x: vec![Complex64::new(0.0, 0.0); self.n],
        }
    }
}

/// Per-worker sweep state for [`SymbolicAc`]; see [`SymbolicAc::solver`].
pub struct AcSweepSolver<'a> {
    sys: &'a SymbolicAc,
    fac: ComplexFactor<'a>,
    /// The frequency `fac` holds the factorization at, when it holds one.
    factored: Option<f64>,
    /// The solution, kept across frequencies.
    x: Vec<Complex64>,
}

impl AcSweepSolver<'_> {
    /// The factorization at `w`: refactored when the last was at another
    /// frequency; `false` on a singular system there.
    fn at(&mut self, w: f64) -> bool {
        if self.factored != Some(w) {
            self.sys.vals.fill(w, self.fac.values_mut());
            self.factored = self.fac.factor().then_some(w);
        }
        self.factored.is_some()
    }

    /// Solve `A x = b` at angular frequency `w` (`None` on a singular system
    /// at this frequency). The solution lives in the worker until the next
    /// solve; another right-hand side at the same `w` reuses the
    /// factorization.
    pub fn solve(&mut self, w: f64, b: &[Complex64]) -> Option<&[Complex64]> {
        (self.at(w) && self.fac.solve_into(b, &mut self.x)).then_some(&self.x[..])
    }

    /// Solve the plain transpose `A^T x = b` at `w`, on the same factors as
    /// [`solve`](Self::solve).
    pub fn solve_transpose(&mut self, w: f64, b: &[Complex64]) -> Option<&[Complex64]> {
        (self.at(w) && self.fac.solve_transpose_into(b, &mut self.x)).then_some(&self.x[..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sane_solve::Triplets;

    type Coo<'a> = (&'a [usize], &'a [usize], &'a [f64]);

    /// The dense `A = G + jwC` of the triplets, duplicates summed.
    fn dense(n: usize, g: Coo<'_>, c: Coo<'_>, w: f64) -> Vec<Vec<Complex64>> {
        let mut a = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for k in 0..g.2.len() {
            a[g.0[k]][g.1[k]] += Complex64::new(g.2[k], 0.0);
        }
        for k in 0..c.2.len() {
            a[c.0[k]][c.1[k]] += Complex64::new(0.0, w * c.2[k]);
        }
        a
    }

    /// The dense solve of `a x = b` (faer's partial-pivoting LU), the
    /// reference the sparse path is held to.
    fn dense_solve(a: &[Vec<Complex64>], b: &[Complex64]) -> Vec<Complex64> {
        use faer::linalg::solvers::Solve;
        let n = a.len();
        let m = faer::Mat::from_fn(n, n, |i, j| a[i][j]);
        let rhs = faer::Mat::from_fn(n, 1, |i, _| b[i]);
        let x = m.partial_piv_lu().solve(&rhs);
        (0..n).map(|i| x[(i, 0)]).collect()
    }

    fn transposed(a: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        (0..a.len())
            .map(|i| (0..a.len()).map(|j| a[j][i]).collect())
            .collect()
    }

    fn sym(g: Coo<'_>, c: Coo<'_>, n: usize, transpose: bool) -> SymbolicAc {
        SymbolicAc::new(n, g, c, (&[], &[], &[], &[]), transpose, 2.0).expect("pattern")
    }

    // The sparse complex AC system (assembled from the sparse Jacobians + gmin)
    // must reproduce the dense `system_matrix_dc + jwC` solve.
    #[test]
    fn sparse_ac_matches_dense() {
        let net = "V1 in 0 1\nR1 in a 100\nL1 a out 1m\nC1 out 0 1u\nR2 out 0 1k\n";
        let parsed = sane_netlist::parse(net).unwrap();
        let mut ctx = sane_core::Graph::new();
        let dae = sane_dae::assemble(&mut ctx, &parsed).unwrap();
        let cdc = sane_solve::CompiledDc::new(ctx, dae);
        let pnames = cdc.param_names();
        let p: Vec<f64> = parsed.pvec(&pnames);
        let n = cdc.dae().dim();
        let (x, conv, _) = cdc.solve_dc(
            &p,
            &[],
            sane_core::constants::DC_OP_TOL,
            sane_core::constants::DC_OP_MAXIT,
        );
        assert!(conv, "DC did not converge");
        let node = parsed.find_node("out").unwrap();
        let out_idx = cdc.dae().node(node).unwrap();
        let v1 = cdc.graph().sym(&sane_circuit::value_symbol_name("V1"));
        let rsdag::Node::Symbol(v1) = *cdc.graph().node(v1) else {
            unreachable!("a value name is a symbol")
        };
        let db = cdc.jacobian_i_input(v1, &x, &p, 0.0);
        let b: Vec<Complex64> = db.iter().map(|v| Complex64::new(-v, 0.0)).collect();
        let w = 2.0 * std::f64::consts::PI * 1e3;

        // dense reference
        let g = cdc.system_triplets_dc(&x, &p).to_dense(cdc.dim());
        let c = cdc.jacobian_q_x_sparse(&x, &p, 0.0).to_dense(cdc.dim());
        let mut a = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = Complex64::new(g[i][j], w * c[i][j]);
            }
        }
        let h_dense = dense_solve(&a, &b)[out_idx];

        // sparse path (gmin as diagonal triplets, summed in assembly)
        let Triplets {
            rows: mut gr,
            cols: mut gc,
            vals: mut gv,
        } = cdc.jacobian_i_x_sparse(&x, &p, 0.0);
        for i in 0..n {
            gr.push(i);
            gc.push(i);
            gv.push(sane_core::constants::GMIN_DC);
        }
        let Triplets {
            rows: cr,
            cols: cc,
            vals: cv,
        } = cdc.jacobian_q_x_sparse(&x, &p, 0.0);
        let sys = sym((&gr, &gc, &gv), (&cr, &cc, &cv), n, false);
        let h_sparse = sys.solver().solve(w, &b).unwrap()[out_idx];

        let rel = (h_dense - h_sparse).norm() / h_dense.norm().max(1e-30);
        assert!(
            rel < 1e-9,
            "sparse {h_sparse:?} vs dense {h_dense:?} (rel {rel:.2e})"
        );
    }

    // The sweep worker against the dense solve of `A` and of `A^T`, across
    // frequencies: a pattern built transposed and the transpose solve on the
    // forward factors must agree.
    #[test]
    fn sweep_matches_dense() {
        // Overlapping G/C entries (so the slot map must sum duplicates), plus an
        // off-diagonal that makes A non-symmetric (so transpose is a real test).
        let g = (
            vec![0usize, 1, 2, 0, 2],
            vec![0usize, 1, 2, 1, 0],
            vec![2.0, 3.0, 4.0, 1.0, 0.5],
        );
        let c = (vec![1usize, 2, 0], vec![1usize, 2, 0], vec![5.0, 1.5, 0.7]);
        let (g, c) = (
            (&g.0[..], &g.1[..], &g.2[..]),
            (&c.0[..], &c.1[..], &c.2[..]),
        );
        let b = vec![
            Complex64::new(4.0, -1.0),
            Complex64::new(0.0, 20.0),
            Complex64::new(8.0, 3.0),
        ];
        let (fwd, tr) = (sym(g, c, 3, false), sym(g, c, 3, true));
        let (mut sweep, mut sweep_t) = (fwd.solver(), tr.solver());
        for &w in &[0.0, 1.0, 7.5, 1e3] {
            let a = dense(3, g, c, w);
            let want = dense_solve(&a, &b);
            let want_t = dense_solve(&transposed(&a), &b);
            let got = sweep.solve(w, &b).expect("sweep").to_vec();
            let got_t = sweep_t.solve(w, &b).expect("sweep of A^T").to_vec();
            let got_tt = sweep
                .solve_transpose(w, &b)
                .expect("transpose solve")
                .to_vec();
            for i in 0..3 {
                assert!((got[i] - want[i]).norm() < 1e-9, "w={w} row {i}");
                assert!((got_t[i] - want_t[i]).norm() < 1e-9, "w={w} A^T row {i}");
                assert!(
                    (got_tt[i] - want_t[i]).norm() < 1e-9,
                    "w={w} transpose row {i}"
                );
            }
        }
    }
}
