//! Pole/zero analyses on [`Model`]: pencil roots, their exact parameter
//! gradients (eigenvalue perturbation over the symbolic stamps) and
//! per-root sensitivities.
//!
//! The roots come from the dense pencil's standard reduction, band by band
//! (see [`pencil_roots`]); their gradients from the sparse adjoint.

use num_complex::Complex64;
use std::collections::HashMap;

use sane_core::constants::{
    BALANCE_GAIN, DIV_FLOOR, PENCIL_BANDS_MAX, PENCIL_INF_TOL, PENCIL_ROOT_MAX, PENCIL_SHIFTS,
};
use sane_core::log;
use sane_solve::Triplets;

use super::adjoint::Bilinear;
use crate::model::{Model, ModelError};

impl Model {
    /// The pencil `(G + gmin, C)` at `(x, p)`, dense.
    fn dense_pencil(&self, x: &[f64], p: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (cdc, n) = (self.cdc(), self.dim());
        let g = cdc.system_triplets_dc(x, p).to_dense(n);
        (g, cdc.jacobian_q_x_sparse(x, p, 0.0).to_dense(n))
    }

    /// Small-signal poles at the operating point `x` (parameters `p`): the finite
    /// generalized eigenvalues of the pencil `(G, C)` with `G = dF/dx`,
    /// `C = dQ/dx`, computed natively (the engine's standard-reduction
    /// eigensolver -- no Python-side linear algebra, so no drift). `(re, im)` in
    /// rad/s.
    pub(crate) fn poles(&self, x: Vec<f64>, p: Vec<f64>) -> Result<Vec<(f64, f64)>, ModelError> {
        self.inner.bound(&p)?;
        let (g, c) = self.dense_pencil(&x, &p);
        let roots = pencil_roots(&g, &c, false).map_err(ModelError::Numeric)?;
        Ok(roots.into_iter().map(|r| (r.s.re, r.s.im)).collect())
    }

    /// Transmission zeros from source `input` to the unknown at `out_idx`, at the
    /// operating point `x`: the finite generalized eigenvalues of the Rosenbrock
    /// system-matrix pencil, computed natively (same eigensolver as `poles`).
    pub(crate) fn zeros(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> Result<Vec<(f64, f64)>, ModelError> {
        self.inner.bound(&p)?;
        let (g, c) = self.dense_pencil(&x, &p);
        let b = self.jacobian_i_input(input, x.clone(), p.clone(), 0.0)?;
        let (m, nn) = rosenbrock_pencil(&g, &c, &b, out_idx);
        let roots = pencil_roots(&m, &nn, false).map_err(ModelError::Numeric)?;
        Ok(roots.into_iter().map(|r| (r.s.re, r.s.im)).collect())
    }

    /// Exact analytic pole sensitivity `ds/dp` for every finite pole w.r.t.
    /// **every** parameter at once, including the operating-point shift. For a
    /// simple pole `s = sigma - 1/mu` (eigenvalue `mu` of
    /// `A = (G + sigma C)^{-1} C` at its band's shift, right/left eigenvectors
    /// `v`/`w`), `d(mu)/dp = w_hat^T ((1 - mu sigma) dC/dp - mu dG/dp) v / (w^T v)`
    /// with `w_hat = (G + sigma C)^{-T} w`: the total derivative of the
    /// functional `Phi = w_hat^T (C - mu (G + sigma C)) v`, one pull (see
    /// [`super::adjoint`]), then `ds = d(mu)/mu^2`. Returns one
    /// `(s, [(name, ds_re, ds_im)])` per pole.
    #[allow(clippy::type_complexity)]
    pub(crate) fn pole_gradient(
        &self,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> Result<Vec<((f64, f64), Vec<(String, f64, f64)>)>, ModelError> {
        let _g = log::scope("sens/pole_gradient");
        let at = self.adjoint_point(&x, &p)?;
        let (g, cmat) = self.dense_pencil(&x, &p);
        let roots = pencil_roots(&g, &cmat, true).map_err(ModelError::Numeric)?;
        let pull = self.pencil_pull(None, None)?;
        self.root_gradients(&at, &pull, &roots, None)
    }

    /// Exact analytic transmission-zero sensitivity `ds/dp` for every finite zero
    /// w.r.t. **every** parameter at once, including the operating-point shift:
    /// [`pole_gradient`](Self::pole_gradient)'s perturbation on the Rosenbrock
    /// pencil `M = [[G, b], [e_out^T, 0]]`, `N = [[C, 0], [0, 0]]`
    /// (`b = dI/d(input)`), whose only extra term is the `b`-column's
    /// `-mu v_n w_hat^T b`. Returns one `(s, [(name, ds_re, ds_im)])` per zero.
    #[allow(clippy::type_complexity)]
    pub(crate) fn zero_gradient(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
    ) -> Result<Vec<((f64, f64), Vec<(String, f64, f64)>)>, ModelError> {
        let _g = log::scope("sens/zero_gradient");
        let at = self.adjoint_point(&x, &p)?;
        let (g, cmat) = self.dense_pencil(&x, &p);
        let b = self.jacobian_i_input(input, x.clone(), p.clone(), 0.0)?;
        let (m, nn) = rosenbrock_pencil(&g, &cmat, &b, out_idx);
        let roots = pencil_roots(&m, &nn, true).map_err(ModelError::Numeric)?;
        let pull = self.pencil_pull(Some(input), None)?;
        self.root_gradients(&at, &pull, &roots, Some((&b, out_idx)))
    }

    /// `M + sigma N` factored, sparse: the pencil `(G + gmin, C)`, bordered
    /// by the input coupling `b` and the output row `out` (the Rosenbrock
    /// pencil's) when given.
    fn shifted_pencil(
        &self,
        x: &[f64],
        p: &[f64],
        sigma: f64,
        border: Option<(&[f64], usize)>,
    ) -> Result<sane_solve::TripletLu<Complex64>, ModelError> {
        let cdc = self.cdc();
        let c = |v: f64| Complex64::new(v, 0.0);
        let g = cdc.system_triplets_dc(x, p);
        let mut t = Triplets {
            rows: g.rows,
            cols: g.cols,
            vals: g.vals.into_iter().map(c).collect(),
        };
        if sigma != 0.0 {
            for (r, k, &v) in cdc.jacobian_q_x_sparse(x, p, 0.0).iter() {
                t.rows.push(r);
                t.cols.push(k);
                t.vals.push(c(sigma * v));
            }
        }
        let mut dim = self.dim();
        if let Some((b, out)) = border {
            let n = dim;
            for (i, &bi) in b.iter().enumerate() {
                t.rows.push(i);
                t.cols.push(n);
                t.vals.push(c(bi));
            }
            t.rows.push(n);
            t.cols.push(out);
            t.vals.push(c(1.0));
            dim += 1;
        }
        sane_solve::factor_triplets(dim, &t.rows, &t.cols, &t.vals)
            .ok_or_else(|| ModelError::Numeric("singular pencil at its shift".into()))
    }

    /// The sensitivities of the roots of a pencil `(M, N)` whose `n x n`
    /// part is `(G, C)`, `border` the input coupling and output row of the
    /// Rosenbrock pencil's when it is one: per root `s = sigma - 1/mu` with
    /// right and left eigenvectors `v`, `w` at its band's shift `sigma`, the
    /// total derivative of `Phi = w_hat^T (N - mu (M + sigma N)) v`
    /// (`(M + sigma N)^T w_hat = w`, one factorization per band) over
    /// `w^T v`, over `mu^2`.
    #[allow(clippy::type_complexity)]
    fn root_gradients(
        &self,
        at: &super::adjoint::AdjointPoint,
        pull: &super::adjoint::PullLayout,
        roots: &[PencilRoot],
        border: Option<(&[f64], usize)>,
    ) -> Result<Vec<((f64, f64), Vec<(String, f64, f64)>)>, ModelError> {
        let one = Complex64::new(1.0, 0.0);
        let mut factors: HashMap<u64, sane_solve::TripletLu<Complex64>> = HashMap::new();
        let mut out = Vec::new();
        for root in roots {
            let (mu, sigma) = (root.mu, root.sigma);
            let (v, w) = root.vectors.as_ref().expect("roots with their vectors");
            let den: Complex64 = w.iter().zip(v).map(|(a, b)| a * b).sum();
            if den.norm() < DIV_FLOOR {
                continue;
            }
            // the DC Jacobian itself at no shift, else this band's pencil
            let what = if sigma == 0.0 && border.is_none() {
                at.solve_dc_transpose(w).ok()
            } else {
                let f = match factors.entry(sigma.to_bits()) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(self.shifted_pencil(&at.x, &at.p, sigma, border)?)
                    }
                };
                f.solve_transpose(w)
            };
            let Some(what) = what else {
                continue;
            };
            let phi = Bilinear {
                l: &what,
                r: v,
                a_g: -mu,
                a_c: one - mu * sigma,
                a_b: border.map_or(Complex64::new(0.0, 0.0), |_| -mu * v[at.n]),
                extra: &[],
            };
            let (dx, dp) = self.pull(at, pull, &phi)?;
            let Ok(num) = self.total(at, &dx, dp) else {
                continue;
            };
            let scale = den * mu * mu;
            let items = (at.pnames.iter().zip(num))
                .map(|(name, d)| {
                    let ds = d / scale;
                    (name.clone(), ds.re, ds.im)
                })
                .collect();
            out.push(((root.s.re, root.s.im), items));
        }
        Ok(out)
    }
}

/// A finite root of a pencil `(M, N)` (`det(M + s N) = 0`) and the reduction
/// that resolved it: `s = sigma - 1/mu`, `mu` an eigenvalue of
/// `A = (M + sigma N)^{-1} N` at its band's shift `sigma`, and the right and
/// left eigenvectors of `A` there when asked for.
pub(crate) struct PencilRoot {
    pub s: Complex64,
    pub sigma: f64,
    pub mu: Complex64,
    pub vectors: Option<(Vec<Complex64>, Vec<Complex64>)>,
}

/// The Rosenbrock system-matrix pencil of the transfer from the input
/// coupling `b` to the unknown `out`: `M = [[G, b], [e_out^T, 0]]`,
/// `N = [[C, 0], [0, 0]]`.
fn rosenbrock_pencil(
    g: &[Vec<f64>],
    c: &[Vec<f64>],
    b: &[f64],
    out: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = g.len();
    let mut m = vec![vec![0.0; n + 1]; n + 1];
    let mut nn = vec![vec![0.0; n + 1]; n + 1];
    for i in 0..n {
        m[i][..n].copy_from_slice(&g[i]);
        nn[i][..n].copy_from_slice(&c[i]);
        m[i][n] = b[i];
    }
    m[n][out] = 1.0;
    (m, nn)
}

/// The finite roots of the pencil `(M, N)`: every `s` with
/// `det(M + s N) = 0` below [`PENCIL_ROOT_MAX`], with the right and left
/// eigenvectors of its reduction when `vectors` asks for them.
///
/// The reactive matrix `N` (`C = dQ/dx`) is heavily rank-deficient (only
/// charge and flux states are dynamic), which makes a dense QZ iteration both
/// slow and ill-conditioned, so the pencil is reduced to the standard
/// eigenproblem `A = (M + sigma N)^{-1} N` (`mu = 1/(sigma - s)`, the infinite
/// roots at `mu = 0`) and, balanced (see [`balance`]), solved by Hessenberg
/// and Francis steps. Its eigenvalues are accurate to `eps |A|`, and `|A|` is
/// set by the root nearest the shift: one reduction resolves the disc around `sigma` out to
/// `1 / (2 PENCIL_INF_TOL max|mu|)` and no further. A circuit whose roots
/// span more decades than that (a bias pole at `1e-12` rad/s next to a
/// device's at `1e9`) is reduced band by band: each next shift on the
/// negative real axis past the radius the bands so far resolve, until the
/// bands cover `PENCIL_ROOT_MAX`, and each root taken from the band that
/// estimates its part of the plane best. A shift the pencil is singular at (a transfer that
/// vanishes at DC has a zero at `s = 0`) moves off it by [`PENCIL_SHIFTS`].
pub(crate) fn pencil_roots(
    m: &[Vec<f64>],
    n_mat: &[Vec<f64>],
    vectors: bool,
) -> Result<Vec<PencilRoot>, String> {
    let dim = m.len();
    let norm = |a: &[Vec<f64>]| a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    let rate = norm(m) / norm(n_mat).max(DIV_FLOOR);
    let c = |z: faer::c64| Complex64::new(z.re, z.im);
    // the radius around the origin the bands so far resolve, the bands
    // (`(sigma, max|mu|)`), and every root a band resolves with its band
    let mut cover = 0.0f64;
    let mut bands: Vec<(f64, f64)> = Vec::new();
    let mut found: Vec<(usize, PencilRoot)> = Vec::new();
    let (mut sigma, mut nudge) = (0.0f64, PENCIL_SHIFTS.iter().cycle());
    for _ in 0..PENCIL_BANDS_MAX {
        let shifted: Vec<Vec<f64>> = (m.iter().zip(n_mat))
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + sigma * y).collect())
            .collect();
        let Some(mut a) = inverse_apply(&shifted, n_mat) else {
            // a root at the shift: move off it
            sigma += nudge.next().expect("cycles") * sigma.abs().max(rate);
            continue;
        };
        // A's entries span the decades its roots do: balanced, its
        // eigenvalues are good to the balanced norm
        let scale = balance(&mut a);
        let (mu, right, left): (Vec<Complex64>, _, _) = if vectors {
            let at = faer::Mat::from_fn(dim, dim, |i, j| a[(j, i)]);
            let eig = a.eigen().map_err(|e| format!("eig failed: {e:?}"))?;
            let eig_t = at.eigen().map_err(|e| format!("eig failed: {e:?}"))?;
            let mu = (0..dim).map(|i| c(eig.S().column_vector()[i])).collect();
            (mu, Some(eig), Some(eig_t))
        } else {
            let mu = a.eigenvalues().map_err(|e| format!("eig failed: {e:?}"))?;
            (mu.into_iter().map(c).collect(), None, None)
        };
        let mumax = mu.iter().map(|m| m.norm()).fold(0.0_f64, f64::max);
        let radius = if mumax > 0.0 {
            1.0 / (2.0 * PENCIL_INF_TOL * mumax)
        } else {
            f64::INFINITY
        };
        for (i, &mui) in mu.iter().enumerate() {
            // `|sigma - s| < radius`: resolved here
            if !(mui.norm() * radius > 1.0) {
                continue;
            }
            let s = sigma - Complex64::new(1.0, 0.0) / mui;
            if !(s.re.is_finite() && s.im.is_finite()) || s.norm() >= PENCIL_ROOT_MAX {
                continue;
            }
            let vectors = match (&right, &left) {
                (Some(eig), Some(eig_t)) => {
                    // back from the balanced matrix: `v = D v'`, `w = D^{-1} w'`
                    let v = (0..dim).map(|k| c(eig.U()[(k, i)]) * scale[k]).collect();
                    // left: the right eigenvector of A^T at the closest eigenvalue
                    let mu_l = eig_t.S().column_vector();
                    let best = (0..dim)
                        .min_by(|&p, &q| {
                            let (dp, dq) = ((c(mu_l[p]) - mui).norm(), (c(mu_l[q]) - mui).norm());
                            dp.partial_cmp(&dq).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(0);
                    let w = (0..dim)
                        .map(|k| c(eig_t.U()[(k, best)]) / scale[k])
                        .collect();
                    Some((v, w))
                }
                _ => None,
            };
            let root = PencilRoot {
                s,
                sigma,
                mu: mui,
                vectors,
            };
            found.push((bands.len(), root));
        }
        bands.push((sigma, mumax));
        cover = cover.max(radius - sigma.abs());
        if cover >= PENCIL_ROOT_MAX {
            break;
        }
        sigma = -2.0 * cover.max(sigma.abs());
    }
    if cover < PENCIL_ROOT_MAX {
        log::warn_captured(&format!(
            "poles/zeros: the roots above |s| = {cover:.3e} rad/s are not resolved"
        ));
    }
    // The plane parted among the bands: a point is the band's whose estimate
    // there is the best (`mu` is good to `eps |A|`, so `s = sigma - 1/mu` to
    // `eps max|mu| |sigma - s|^2`), and a band keeps the roots it finds in
    // its own part. A parting by the shifts alone, not by the estimates:
    // the parts meet where two bands are equally good, far inside both.
    let owner = |s: Complex64| {
        let err = |&(sigma, mumax): &(f64, f64)| (sigma - s).norm_sqr() * mumax;
        (0..bands.len()).min_by(|&a, &b| {
            err(&bands[a])
                .partial_cmp(&err(&bands[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    };
    Ok((found.into_iter())
        .filter(|(band, root)| owner(root.s) == Some(*band))
        .map(|(_, root)| root)
        .collect())
}

/// Pick the `order` lowest-frequency (smallest |s|) roots, completing complex
/// conjugate pairs so the reduced model stays real.
pub(crate) fn dominant_subset(roots: &[[f64; 2]], order: usize) -> Vec<[f64; 2]> {
    let mut idx: Vec<usize> = (0..roots.len()).collect();
    idx.sort_by(|&i, &j| {
        let mi = roots[i][0].hypot(roots[i][1]);
        let mj = roots[j][0].hypot(roots[j][1]);
        mi.partial_cmp(&mj).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept = vec![false; roots.len()];
    let conj_of = |i: usize| -> Option<usize> {
        let (re, im) = (roots[i][0], roots[i][1]);
        if im.abs() < sane_core::constants::PZ_CONJ_TOL * (re.hypot(im) + 1.0) {
            return None;
        }
        let scale = re.hypot(im) + 1.0;
        roots.iter().position(|r| {
            let tol = sane_core::constants::PZ_CONJ_TOL * scale;
            (r[0] - re).abs() < tol && (r[1] + im).abs() < tol
        })
    };
    let mut count = 0;
    for &i in &idx {
        if count >= order {
            break;
        }
        if !kept[i] {
            kept[i] = true;
            count += 1;
            if let Some(j) = conj_of(i) {
                if !kept[j] {
                    kept[j] = true;
                    count += 1;
                }
            }
        }
    }
    (0..roots.len())
        .filter(|&i| kept[i])
        .map(|i| roots[i])
        .collect()
}

/// Balance `a` in place by a diagonal similarity `D^{-1} A D` of powers of
/// two (Parlett and Reinsch, the scaling of LAPACK's `gebal`): each row's
/// and column's off-diagonal norms brought level, so the eigenvalues of a
/// matrix whose entries span many decades are good to the balanced norm,
/// not the raw one. Returns `D`'s diagonal.
fn balance(a: &mut faer::Mat<f64>) -> Vec<f64> {
    let n = a.nrows();
    let mut d = vec![1.0f64; n];
    let (radix, sqrdx) = (2.0f64, 4.0f64);
    let mut done = false;
    while !done {
        done = true;
        for i in 0..n {
            let (mut c, mut r) = (0.0f64, 0.0f64);
            for j in (0..n).filter(|&j| j != i) {
                c += a[(j, i)].abs();
                r += a[(i, j)].abs();
            }
            if c == 0.0 || r == 0.0 {
                continue;
            }
            let s = c + r;
            let mut f = 1.0f64;
            while c < r / radix {
                f *= radix;
                c *= sqrdx;
            }
            while c > r * radix {
                f /= radix;
                c /= sqrdx;
            }
            if (c + r) / f < BALANCE_GAIN * s {
                done = false;
                d[i] *= f;
                for j in 0..n {
                    a[(i, j)] /= f;
                    a[(j, i)] *= f;
                }
            }
        }
    }
    d
}

/// `A = M^{-1} N` as a dense matrix, one factorization of `M` against the
/// columns of `N` through the reference dense solve; `None` when `M` is
/// singular (a non-finite entry).
fn inverse_apply(m: &[Vec<f64>], n_mat: &[Vec<f64>]) -> Option<faer::Mat<f64>> {
    let n = m.len();
    let m_flat: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();
    // The columns of N as right-hand sides, back to back.
    let mut cols = vec![0.0f64; n * n];
    for (i, row) in n_mat.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            cols[j * n + i] = v;
        }
    }
    let mut x = vec![0.0f64; n * n];
    rsdag::semantics::solve_many(&m_flat, &cols, n, n, &mut x);
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(faer::Mat::from_fn(n, n, |i, j| x[j * n + i]))
}
