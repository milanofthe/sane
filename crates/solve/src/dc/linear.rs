//! The DC Newton's linear solves: the step on the Jacobian with a shunt
//! on its diagonal, through the symbolic LU or the Schur partition.

use super::*;

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
    pub(super) fn solve_step<'a>(
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
    pub(super) fn solve_with_diag<'a>(
        &'a self,
        jac: &[f64],
        diag: impl IntoIterator<Item = f64>,
        rhs: &[f64],
        dx: &mut [f64],
        valbuf: &mut Vec<f64>,
        fac: &mut Option<sparse::Refactorable<'a>>,
    ) -> bool {
        match &self.dc.symbolic {
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
    pub(super) fn solve_triplets(
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
}
