//! Small dense direct solvers used by the AC / pole-zero paths.
//! Hand-rolled LU with partial pivoting; the complex factorization is split from
//! the triangular solves so several right-hand sides can reuse one factorization.

use num_complex::Complex64;

/// In-place dense complex LU with partial pivoting (LAPACK `getrf` style): on
/// success `a` holds the unit-lower `L` below the diagonal and `U` on/above it,
/// and the returned vector is the row-interchange sequence (`swaps[col]` is the
/// row swapped with `col` at step `col`). `None` if singular. Separated from the
/// triangular solves so several right-hand sides can reuse one factorization.
pub(crate) fn lu_factor_complex(a: &mut [Vec<Complex64>]) -> Option<Vec<usize>> {
    let n = a.len();
    let mut swaps = vec![0usize; n];
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col][col].norm();
        for r in (col + 1)..n {
            let m = a[r][col].norm();
            if m > best {
                best = m;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        swaps[col] = piv;
        let d = a[col][col];
        for r in (col + 1)..n {
            let f = a[r][col] / d;
            a[r][col] = f; // store the multiplier (L factor)
            for c in (col + 1)..n {
                let v = a[col][c];
                a[r][c] -= f * v;
            }
        }
    }
    Some(swaps)
}

/// Solve `A x = b` from the factorization produced by [`lu_factor_complex`]:
/// replay the row interchanges on `b`, forward-solve `L y = Pb`, back-solve
/// `U x = y`. The factorization itself is untouched, so it can be reused.
pub(crate) fn lu_solve_complex(
    a: &[Vec<Complex64>],
    swaps: &[usize],
    b: &[Complex64],
) -> Vec<Complex64> {
    let n = a.len();
    let mut y = b.to_vec();
    for col in 0..n {
        y.swap(col, swaps[col]);
    }
    for col in 0..n {
        let yc = y[col];
        for r in (col + 1)..n {
            y[r] -= a[r][col] * yc;
        }
    }
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for col in (0..n).rev() {
        let mut s = y[col];
        for c in (col + 1)..n {
            s -= a[col][c] * x[c];
        }
        x[col] = s / a[col][col];
    }
    x
}

/// Dense complex solve A x = b (LU with partial pivoting). For several RHS
/// against one matrix, factor once with [`lu_factor_complex`] and reuse it.
pub fn solve_complex(mut a: Vec<Vec<Complex64>>, b: Vec<Complex64>) -> Option<Vec<Complex64>> {
    let swaps = lu_factor_complex(&mut a)?;
    Some(lu_solve_complex(&a, &swaps, &b))
}

/// Dense real solve A x = b (Gaussian elimination, partial pivoting).
pub fn solve_real(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in (col + 1)..n {
            let f = a[r][col] / d;
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s -= a[col][c] * x[c];
        }
        x[col] = s / a[col][col];
    }
    Some(x)
}
