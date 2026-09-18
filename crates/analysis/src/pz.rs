//! Pole-zero analysis: finite generalized eigenvalues of the small-signal
//! pencil, their parameter sensitivities and eigenvectors, and dominant-root
//! selection.

use num_complex::Complex64;
use sane_core::constants::{PENCIL_INF_TOL, PENCIL_ROOT_MAX};

/// Finite generalized eigenvalues of the pencil (M, N): solve `M v = lambda N v`
/// and return `-lambda` (the pole/zero `s`), dropping the infinite ones that a
/// singular N produces.
///
/// The reactive matrix N (= C = dF/dx') is heavily rank-deficient (only
/// charge/flux states are dynamic), which makes a dense QZ iteration both slow
/// and ill-conditioned (catastrophically so in faer -- minutes on a ~100x100
/// pencil). `M` is invertible at a valid operating point (the small-signal
/// conductance `G`), so we reduce to the **standard** eigenproblem `A = M^{-1} N`
/// and use its mature Hessenberg+Francis solver. The infinite generalized
/// eigenvalues map to `mu = 0` (the null space of N) and are dropped; a finite
/// root is `s = -1/mu`. Errors if `M` is singular (no QZ fallback -- a singular
/// small-signal `G` is a degenerate, biasless circuit).
pub fn finite_pencil_roots(m: &[Vec<f64>], n_mat: &[Vec<f64>]) -> Result<Vec<[f64; 2]>, String> {
    // A = M^{-1} N via one factorization of M against the columns of N.
    let a = inverse_apply(m, n_mat).ok_or("finite_pencil_roots: singular pencil matrix M")?;
    // faer returns its own Complex; lift into num_complex for the arithmetic.
    let mu: Vec<Complex64> = a
        .eigenvalues()
        .map_err(|e| format!("eig failed: {e:?}"))?
        .into_iter()
        .map(|c| Complex64::new(c.re, c.im))
        .collect();
    let mumax = mu
        .iter()
        .map(|m| m.norm())
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    let mut out = Vec::new();
    for m in mu {
        if m.norm() < PENCIL_INF_TOL * mumax {
            continue; // mu ~ 0 -> lambda = inf -> no finite root
        }
        // lambda = 1/mu, s = -lambda = -1/mu.
        let s = -Complex64::new(1.0, 0.0) / m;
        if s.re.is_finite() && s.im.is_finite() && s.norm() < PENCIL_ROOT_MAX {
            out.push([s.re, s.im]);
        }
    }
    Ok(out)
}

/// Exact first-order sensitivity of every finite generalized eigenvalue (root
/// `s = -lambda` of the pencil `M v = lambda N v`) w.r.t. a parameter, given the
/// total derivatives `dM`, `dN`. Returns `(root, d(root)/dp)` pairs in the same
/// convention as [`finite_pencil_roots`].
///
/// Like [`finite_pencil_roots`], this avoids the fragile/slow dense QZ: with `M`
/// invertible at the operating point it reduces to the **standard** eigenproblem
/// `A = M^{-1} N` (eigenvalues `mu = 1/lambda`, right eigenvectors of `A` are the
/// pencil's; left eigenvectors come from `A^T`). The standard eigenvalue
/// perturbation `d(mu) = w^T M^{-1} (dN - mu*dM) v / (w^T v)` (one reuse of the
/// `M` factorization per eigenvalue) then gives `ds = d(mu)/mu^2` since
/// `s = -1/mu`. Returns an error if `M` is singular.
pub fn pencil_root_sensitivity(
    m: &[Vec<f64>],
    n_mat: &[Vec<f64>],
    dm: &[Vec<f64>],
    dn: &[Vec<f64>],
) -> Result<Vec<([f64; 2], [f64; 2])>, String> {
    let n = m.len();
    let a = inverse_apply(m, n_mat).ok_or("pencil_root_sensitivity: singular M")?;
    let m_flat: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();
    let at = faer::Mat::from_fn(n, n, |i, j| a[(j, i)]);
    let eig = a.eigen().map_err(|e| format!("eig failed: {e:?}"))?;
    let eig_t = at.eigen().map_err(|e| format!("eig failed: {e:?}"))?;
    let mu = eig.S().column_vector();
    let mu_l = eig_t.S().column_vector();
    let vr = eig.U();
    let vl = eig_t.U();
    let mumax = (0..n)
        .map(|i| mu[i].re.hypot(mu[i].im))
        .fold(0.0_f64, f64::max)
        .max(1e-300);

    let mut out = Vec::new();
    for i in 0..n {
        let mui = Complex64::new(mu[i].re, mu[i].im);
        if mui.norm() < PENCIL_INF_TOL * mumax {
            continue; // mu ~ 0  ->  lambda = inf  ->  infinite root
        }
        let s = -Complex64::new(1.0, 0.0) / mui; // root s = -lambda = -1/mu
        if !(s.re.is_finite() && s.im.is_finite()) || s.norm() >= PENCIL_ROOT_MAX {
            continue;
        }
        let v: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(vr[(k, i)].re, vr[(k, i)].im))
            .collect();
        // Match the left eigenvector (right eigenvector of A^T) by closest mu.
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for j in 0..n {
            let d = (Complex64::new(mu_l[j].re, mu_l[j].im) - mui).norm();
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        let w: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(vl[(k, best)].re, vl[(k, best)].im))
            .collect();

        // rhs = (dN - mu*dM) v  (complex), then y = M^{-1} rhs: the real and
        // imaginary parts as two right-hand sides of one real solve.
        let mut rhs = vec![0.0f64; 2 * n];
        for k in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for l in 0..n {
                acc += (dn[k][l] - mui * dm[k][l]) * v[l];
            }
            rhs[k] = acc.re;
            rhs[n + k] = acc.im;
        }
        let mut y = vec![0.0f64; 2 * n];
        rsdag::semantics::solve_many(&m_flat, &rhs, n, 2, &mut y);

        // d(mu) = w^T y / (w^T v);  ds = d(mu)/mu^2.
        let (mut num, mut den) = (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
        for k in 0..n {
            num += w[k] * Complex64::new(y[k], y[n + k]);
            den += w[k] * v[k];
        }
        if den.norm() < 1e-300 {
            continue;
        }
        let dmu = num / den;
        let ds = dmu / (mui * mui);
        out.push(([s.re, s.im], [ds.re, ds.im]));
    }
    Ok(out)
}

/// Finite generalized eigenpairs of the pencil `M v = lambda N v` (root
/// `s = -1/mu`, `mu` eigenvalue of `A = M^{-1} N`), each as
/// `(s, mu, v_right, w_left)` with `v`/`w` the right and left eigenvectors of
/// `A` (left = right eigenvector of `A^T`), complex as `[re, im]`. The building
/// block for all-parameter pole/zero sensitivity: with `(mu, v, w)` fixed, every
/// parameter's `d(mu)/dp = w_hat^T (dN - mu dM) v / (w^T v)`,
/// `w_hat = M^{-T} w`, is one sparse contraction (see the engine's `pole_gradient`).
#[allow(clippy::type_complexity)]
pub fn pencil_eigvectors(
    m: &[Vec<f64>],
    n_mat: &[Vec<f64>],
) -> Result<Vec<([f64; 2], [f64; 2], Vec<[f64; 2]>, Vec<[f64; 2]>)>, String> {
    let n = m.len();
    let a = inverse_apply(m, n_mat).ok_or("pencil_eigvectors: singular M")?;
    let at = faer::Mat::from_fn(n, n, |i, j| a[(j, i)]);
    let eig = a.eigen().map_err(|e| format!("eig failed: {e:?}"))?;
    let eig_t = at.eigen().map_err(|e| format!("eig failed: {e:?}"))?;
    let (mu, mu_l, vr, vl) = (
        eig.S().column_vector(),
        eig_t.S().column_vector(),
        eig.U(),
        eig_t.U(),
    );
    let mumax = (0..n)
        .map(|i| mu[i].re.hypot(mu[i].im))
        .fold(0.0_f64, f64::max)
        .max(1e-300);

    let mut out = Vec::new();
    for i in 0..n {
        let mui = Complex64::new(mu[i].re, mu[i].im);
        if mui.norm() < PENCIL_INF_TOL * mumax {
            continue;
        }
        let s = -Complex64::new(1.0, 0.0) / mui;
        if !(s.re.is_finite() && s.im.is_finite()) || s.norm() >= PENCIL_ROOT_MAX {
            continue;
        }
        let v: Vec<[f64; 2]> = (0..n).map(|k| [vr[(k, i)].re, vr[(k, i)].im]).collect();
        // Left eigenvector: right eigenvector of A^T at the closest eigenvalue.
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for j in 0..n {
            let d = (Complex64::new(mu_l[j].re, mu_l[j].im) - mui).norm();
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        let w: Vec<[f64; 2]> = (0..n)
            .map(|k| [vl[(k, best)].re, vl[(k, best)].im])
            .collect();
        out.push(([s.re, s.im], [mui.re, mui.im], v, w));
    }
    Ok(out)
}

/// Pick the `order` lowest-frequency (smallest |s|) roots, completing complex
/// conjugate pairs so the reduced model stays real.
pub fn dominant_subset(roots: &[[f64; 2]], order: usize) -> Vec<[f64; 2]> {
    let mut idx: Vec<usize> = (0..roots.len()).collect();
    idx.sort_by(|&i, &j| {
        let mi = roots[i][0].hypot(roots[i][1]);
        let mj = roots[j][0].hypot(roots[j][1]);
        mi.partial_cmp(&mj).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept = vec![false; roots.len()];
    let conj_of = |i: usize| -> Option<usize> {
        let (re, im) = (roots[i][0], roots[i][1]);
        if im.abs() < 1e-6 * (re.hypot(im) + 1.0) {
            return None;
        }
        let scale = re.hypot(im) + 1.0;
        roots
            .iter()
            .position(|r| (r[0] - re).abs() < 1e-6 * scale && (r[1] + im).abs() < 1e-6 * scale)
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

#[cfg(test)]
use crate::symbolic_poly::{poly_in_s, prune_poly};
#[cfg(test)]
use crate::{op_env, parse_ic, resolve_out_idx, IcTarget, Model};
#[cfg(test)]
use rsdag::{Graph, Node};
#[cfg(test)]
use sane_core::constants::{DC_OP_MAXIT, DC_OP_TOL};
#[cfg(test)]
use sane_dae::assemble_dae;
#[cfg(test)]
use sane_netlist::parse;
#[cfg(test)]
use sane_solve::CompiledDc;
#[cfg(test)]
use std::f64::consts::PI;

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

#[cfg(test)]
mod pz_tests {
    use super::*;
    use num_complex::Complex64;

    fn nearest_real(poles: &[[f64; 2]], target: f64) -> f64 {
        poles
            .iter()
            .map(|p| (p[0] - target).abs() + p[1].abs())
            .fold(f64::INFINITY, f64::min)
    }

    /// Faithful re-implementation of the deleted `symbolic_reduce` façade over the
    /// `Model` surface plus the kept symbolic-pruning kernels: extract `H = N/D`,
    /// rank each polynomial's terms at the band reference frequency, prune below
    /// `tol`, and report `(ok, terms_full, terms_kept, max_err_db)` over the band.
    fn symbolic_reduce(
        net: &str,
        input: &str,
        output: &str,
        tol: f64,
        fstart: f64,
        fstop: f64,
        points: usize,
    ) -> (bool, usize, usize, f64) {
        use crate::symbolic_poly::{eval_coeffs, poly_eval_c};
        let m = Model::from_netlist(net).expect("model");
        let out_idx = m.resolve(output).expect("output");
        let target = m.unknowns()[out_idx].clone();
        let op = m.operating_point(&[]).expect("operating point");
        let x = op.vector().to_vec();
        let p = m.pvec(&[]);
        let arc = m.context_arc();
        let mut ctx = arc.lock().unwrap();
        let dae = m.dae();
        let (n_expr, d_expr) =
            sane_dae::small_signal_transfer_nd(&mut ctx, dae, input, &target).expect("transfer");
        let s_e = ctx.sym("s");
        let s_sym = match ctx.node(s_e) {
            Node::Symbol(x) => *x,
            _ => panic!("s is not a symbol"),
        };
        let n_poly = poly_in_s(&mut ctx, n_expr, s_sym).expect("numerator polynomial");
        let d_poly = poly_in_s(&mut ctx, d_expr, s_sym).expect("denominator polynomial");
        let pnames = m.cdc().param_names(&ctx);
        let env = op_env(&mut ctx, dae, &pnames, &x, &[], &p, 0.0);
        let w0 = 2.0 * PI * (fstart * fstop).sqrt();
        let (n_pruned, n_tot, n_kept) = prune_poly(&mut ctx, &n_poly, &env, w0, tol);
        let (d_pruned, d_tot, d_kept) = prune_poly(&mut ctx, &d_poly, &env, w0, tol);
        let nf = eval_coeffs(&ctx, &env, &n_poly);
        let df = eval_coeffs(&ctx, &env, &d_poly);
        let nr = eval_coeffs(&ctx, &env, &n_pruned);
        let dr = eval_coeffs(&ctx, &env, &d_pruned);
        let (l0, l1) = (fstart.log10(), fstop.log10());
        let mut max_err_db = 0.0_f64;
        for i in 0..points {
            let fi = 10f64.powf(l0 + (l1 - l0) * i as f64 / (points - 1) as f64);
            let jw = Complex64::new(0.0, 2.0 * PI * fi);
            let hf = poly_eval_c(&nf, jw) / poly_eval_c(&df, jw);
            let hr = poly_eval_c(&nr, jw) / poly_eval_c(&dr, jw);
            let fdb = 20.0 * hf.norm().max(1e-30).log10();
            let rdb = 20.0 * hr.norm().max(1e-30).log10();
            max_err_db = max_err_db.max((fdb - rdb).abs());
        }
        (true, n_tot + d_tot, n_kept + d_kept, max_err_db)
    }

    /// Relative magnitude AC sensitivity `d ln|H| / d ln param` at `freq`,
    /// reconstructed from the exact `Model` AC response and per-parameter
    /// `dH/dp` (the same quantity the deleted `ac_sensitivity` façade ranked).
    fn ac_rel_sens(net: &str, input: &str, output: &str, freq: f64, param: &str) -> f64 {
        let m = Model::from_netlist(net).expect("model");
        let op = m.operating_point(&[]).expect("operating point");
        let x = op.vector().to_vec();
        let p = m.pvec(&[]);
        let out_idx = m.resolve(output).expect("output");
        let h = m
            .ac_response(input, out_idx, x.clone(), p.clone(), vec![freq])
            .expect("ac")[0];
        let h0 = Complex64::new(h.0, h.1);
        let dh = m
            .ac_sensitivity(input, param, out_idx, x, p, vec![freq])
            .expect("ac_sens")[0];
        let dhc = Complex64::new(dh.0, dh.1);
        let dmag_dp = (h0.conj() * dhc).re / (h0.norm() * h0.norm());
        m.get(param).expect("param value") * dmag_dp
    }

    /// Relative transient sensitivity `d ln y(tstop) / d ln param`, replicating the
    /// deleted `transient_sensitivity` façade: honour `.ic` UIC (zeroing the
    /// sensitivity IC on pinned unknowns), integrate the single-parameter augmented
    /// forward-sensitivity DAE, and normalise by the baseline output at `tstop`.
    fn transient_sens_rel(net: &str, output: &str, tstop: f64, tstep: f64, param: &str) -> f64 {
        use sane_dae::augment_with_sensitivities;
        let m = Model::from_netlist(net).expect("model");
        let n = m.dim();
        let out_idx = m.resolve(output).expect("output");
        let p0 = m.pvec(&[]);
        let col = m
            .params()
            .iter()
            .position(|nm| nm == param)
            .expect("param column");
        let vals = m.values();
        let arc = m.context_arc();
        let mut ctx = arc.lock().unwrap();
        let dae = m.dae();
        m.cdc().ensure_param_jac(&mut ctx, dae);
        let (op, conv, _) = m.cdc().solve_dc(&p0, &[], DC_OP_TOL, DC_OP_MAXIT);
        assert!(
            conv,
            "transient sensitivity: DC operating point did not converge"
        );
        // `.ic` UIC: pin those unknowns; a pinned initial value is parameter-
        // independent, so its sensitivity initial condition is exactly zero.
        let mut x0 = op.clone();
        let mut uic: Vec<usize> = Vec::new();
        for (tgt, val) in parse_ic(net) {
            let idx = match tgt {
                IcTarget::V(node) => m.resolve(&node),
                IcTarget::I(elem) => m.resolve(&elem),
            };
            if let Some(i) = idx {
                x0[i] = val;
                uic.push(i);
            }
        }
        let mut s0 = m.cdc().state_sensitivity(col, &op, &p0, 0.0);
        assert_eq!(s0.len(), n, "singular Jacobian for the sensitivity IC");
        for &i in &uic {
            s0[i] = 0.0;
        }
        let p_sym = {
            let e = ctx.sym(param);
            match ctx.node(e) {
                Node::Symbol(s) => *s,
                _ => panic!("{param} is not a symbol"),
            }
        };
        let npts = ((tstop / tstep).round() as usize).clamp(1, 100_000);
        let t_eval: Vec<f64> = (0..=npts).map(|k| k as f64 * tstop / npts as f64).collect();
        let dt_max = tstop / npts as f64;
        let base = m
            .cdc()
            .solve_transient(
                sane_solve::TransientMethod::Esdirk32,
                &p0,
                &x0,
                &t_eval,
                1e-4,
                1e-7,
                None,
            )
            .expect("baseline transient")
            .last()
            .and_then(|r| r.get(out_idx).copied())
            .unwrap_or(0.0);
        let ynorm = if base.abs() > 1e-30 { base } else { 1.0 };
        let aug = augment_with_sensitivities(&mut ctx, dae, std::slice::from_ref(&p_sym));
        let aug_cdc = CompiledDc::new(&mut ctx, &aug);
        let p_aug: Vec<f64> = aug_cdc
            .param_names(&ctx)
            .iter()
            .map(|nm| vals.get(nm).copied().unwrap_or(0.0))
            .collect();
        let mut x0_aug = x0.clone();
        x0_aug.extend_from_slice(&s0);
        let traj = aug_cdc
            .solve_transient(
                sane_solve::TransientMethod::Esdirk32,
                &p_aug,
                &x0_aug,
                &t_eval,
                1e-4,
                1e-7,
                Some(dt_max),
            )
            .expect("augmented transient");
        let dydp = traj
            .last()
            .and_then(|r| r.get(n + out_idx).copied())
            .unwrap_or(0.0);
        dydp * p0[col] / ynorm
    }

    #[test]
    fn rc_lowpass_pole() {
        // R=1k, C=1u -> single real pole at s = -1/(RC) = -1000 rad/s.
        let net = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n";
        let m = Model::from_netlist(net).expect("model");
        let pz = m.poles_zeros(&[], "V1", "out").expect("pole_zero");
        let d = nearest_real(&pz.poles, -1000.0);
        assert!(
            d < 1.0,
            "expected a pole near -1000, got {:?} (err {d})",
            pz.poles
        );
    }

    #[test]
    fn symbolic_reduce_tol0_is_exact() {
        // tol = 0 prunes nothing -> reconstructed H must equal the full H exactly.
        let net = "V1 in 0 1\nR1 in a 10\nL1 a out 1m\nC1 out 0 1u\n";
        let (ok, terms_full, terms_kept, max_err_db) =
            symbolic_reduce(net, "V1", "out", 0.0, 1.0, 1e6, 30);
        assert!(ok, "symbolic_reduce failed");
        assert_eq!(terms_kept, terms_full, "tol=0 should keep all terms");
        assert!(
            max_err_db < 1e-6,
            "tol=0 reconstruction err {max_err_db} dB"
        );
    }

    #[test]
    fn symbolic_reduce_drops_negligible_term() {
        // A 1-pF cap in parallel with a 1-uF cap is ~1e6 smaller; its term in the
        // s^1 coefficient must be pruned, leaving the response essentially unchanged.
        let net = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\nC2 out 0 1p\n";
        let (ok, terms_full, terms_kept, max_err_db) =
            symbolic_reduce(net, "V1", "out", 1e-3, 1.0, 1e6, 30);
        assert!(ok, "symbolic_reduce failed");
        assert!(
            terms_kept < terms_full,
            "expected pruning: {terms_kept}/{terms_full}"
        );
        assert!(max_err_db < 1.0, "pruned response drifted {max_err_db} dB");
    }

    #[test]
    fn transient_sensitivity_rc_discharge() {
        // RC discharge from V(out)=1 (UIC) through R to ground: V(t)=exp(-t/RC).
        // The relative sensitivity d ln V / d ln R = t/(RC) (larger R holds charge
        // longer), so at t = RC it is +1.0. (Exact: the augmented forward-
        // sensitivity DAE with a zero sensitivity IC on the pinned node.)
        let net = ".ic V(out)=1\nR1 out 0 1k\nC1 out 0 1u\n";
        let sr = transient_sens_rel(net, "out", 1e-3, 2e-5, "R1");
        assert!(
            (sr - 1.0).abs() < 0.05,
            "d ln V/d ln R = {sr}, expected ~ +1.0 at t=RC"
        );
    }

    // Run `cargo test -p sane-analysis bench_ua741 -- --ignored --nocapture`.
    #[test]
    #[ignore = "long-running benchmark (pole-zero on dim-109 is ~20 min)"]
    fn bench_ua741() {
        let net = include_str!("../../../crates/netlist/tests/fixtures/ua741.cir").to_string();
        let m = Model::from_netlist(&net).expect("model");
        let t = |label: &str, f: &dyn Fn() -> String| {
            let t0 = sane_core::time::Instant::now();
            let info = f();
            eprintln!(
                "  {:<24} {:>9.1} ms   {}",
                label,
                t0.elapsed().as_secs_f64() * 1000.0,
                info
            );
        };
        eprintln!("\n=== uA741 analysis breakdown (24 BJT, transistor level, dim 109) ===");
        t("Operating point", &|| {
            let op = m.operating_point(&[]);
            format!("ok={} dim={}", op.is_ok(), m.dim())
        });
        t("Transient (1us/10ns)", &|| {
            let npts = ((1e-6 / 1e-8_f64).round() as usize).clamp(1, 100_000);
            let te: Vec<f64> = (0..=npts).map(|k| k as f64 * 1e-6 / npts as f64).collect();
            let tr = m.transient(sane_solve::TransientMethod::default(), &[], &te, 1e-4, 1e-7);
            format!("ok={} traces={}", tr.is_ok(), m.unknowns().len())
        });
        t("AC (50 pts, 1Hz-1MHz)", &|| {
            let ac = m.ac(&[], "Vd", "22", 1.0, 1e6, 50);
            format!(
                "ok={} |H0|={:.1}dB",
                ac.is_ok(),
                ac.as_ref()
                    .ok()
                    .and_then(|a| a.mag_db.first().copied())
                    .unwrap_or(0.0)
            )
        });
        t(
            "Pole-Zero (std-reduce, dim 109)",
            &|| match m.poles_zeros(&[], "Vd", "22") {
                Ok(pz) => format!("ok=true poles={} zeros={}", pz.poles.len(), pz.zeros.len()),
                Err(e) => format!("ok=false {e}"),
            },
        );
        t("DC sensitivity (adjoint)", &|| {
            let s = m.operating_point(&[]).and_then(|op| op.sensitivity("22"));
            format!(
                "ok={} params={}",
                s.is_ok(),
                s.as_ref().map(|s| s.names.len()).unwrap_or(0)
            )
        });
        t("Noise (50 pts)", &|| {
            let ns = m.noise(&[], "22", 1.0, 1e6, 50);
            format!("ok={}", ns.is_ok())
        });
        eprintln!("====================================================================\n");
    }

    #[test]
    fn dc_sweep_cmos_inverter_vtc() {
        // The default example: sweeping the input must produce the inverter
        // transfer curve -- Vout high (~5V) at Vin=0, low (~0V) at Vin=5.
        let net = "VDD vdd 0 DC 5\nVIN vin 0 DC 2.5\nM1 vout vin 0 0 NMOS1\nM2 vout vin vdd vdd PMOS1\n.model NMOS1 NMOS(Kp=120u W=2 L=1 Vto=0.7)\n.model PMOS1 PMOS(Kp=40u W=4 L=1 Vto=-0.7)\n";
        let m = Model::from_netlist(net).expect("model");
        let ds = m.dc_sweep("VIN", 0.0, 5.0, 0.25).expect("dc_sweep");
        let vout = ds.signal("vout").expect("vout trace");
        assert!(vout[0] > 4.0, "Vout at Vin=0 = {}", vout[0]);
        assert!(
            *vout.last().unwrap() < 1.0,
            "Vout at Vin=5 = {}",
            vout.last().unwrap()
        );
    }

    #[test]
    fn ac_sallen_key_rolls_off() {
        // The default example, as the studio regenerates it. AC must succeed and
        // the output must roll off (2nd-order low-pass).
        let net = "VS in 0 AC 1\nR1 in x 11.2k\nR2 x y 11.2k\nC1 x vo 2000pF\nC2 y 0 1000pF\nE_OP vo 0 y vo 1Meg\n";
        let m = Model::from_netlist(net).expect("model");
        let ac = m.ac(&[], "VS", "vo", 10.0, 1e6, 50).expect("AC");
        assert!(ac.mag_db.len() == 50);
        assert!(
            ac.mag_db[49] < ac.mag_db[0] - 20.0,
            "no rolloff: {} -> {}",
            ac.mag_db[0],
            ac.mag_db[49]
        );
    }

    #[test]
    fn ac_sensitivity_rc_pole() {
        // RC lowpass at its pole f = 1/(2 pi RC) = 159.155 Hz: d ln|H|/d ln R and
        // d ln|H|/d ln C are both -0.5 there (|H| = 1/sqrt(1+(wRC)^2), wRC=1).
        let net = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n";
        let sr = ac_rel_sens(net, "V1", "out", 159.1549, "R1");
        let sc = ac_rel_sens(net, "V1", "out", 159.1549, "C1");
        assert!((sr + 0.5).abs() < 0.03, "dln|H|/dlnR = {sr}, expected -0.5");
        assert!((sc + 0.5).abs() < 0.03, "dln|H|/dlnC = {sc}, expected -0.5");
    }

    #[test]
    fn model_reduce_exact_when_order_sufficient() {
        // RLC (2 poles): reducing to an order >= the true order must reproduce the
        // full magnitude response to within rounding.
        let net = "V1 in 0 1\nR1 in a 10\nL1 a out 1m\nC1 out 0 1u\n";
        let m = Model::from_netlist(net).expect("model");
        let r = m
            .model_reduce(&[], "V1", "out", 4, 1.0, 1e6, 40)
            .expect("model_reduce");
        assert!(
            r.max_err_db < 0.1,
            "expected exact reconstruction, max_err {} dB",
            r.max_err_db
        );
    }

    #[test]
    fn noise_single_resistor_johnson() {
        // A lone 1k resistor to ground: output noise = sqrt(4kTR) ~= 4.07 nV/rtHz,
        // flat in frequency.
        let net = "R1 out 0 1k\n";
        let m = Model::from_netlist(net).expect("model");
        let r = m.noise(&[], "out", 1.0, 1e6, 3).expect("noise");
        let expected = (4.0 * 1.380649e-23 * 300.15 * 1000.0_f64).sqrt();
        for nv in &r.psd {
            assert!(
                (nv - expected).abs() / expected < 0.02,
                "noise {nv} vs expected {expected}"
            );
        }
    }

    #[test]
    fn noise_diode_shot() {
        // A diode biased at I by an ideal current source, output at the diode node.
        // The node sees only the diode (rd = Vt/I), driven by shot noise 2q*I, so
        // the output PSD is 2q*I*rd^2 = 2q*Vt^2/I; the resistor-free setup isolates
        // the diode's own shot noise (no thermal term to subtract).
        let i_bias = 1e-3_f64;
        let net = format!("I1 0 a {i_bias}\nD1 a 0\n");
        let m = Model::from_netlist(&net).expect("model");
        let r = m.noise(&[], "a", 1.0, 1e6, 3).expect("noise");
        let q = 1.602176634e-19_f64;
        let vt = (1.380649e-23 / 1.602176634e-19) * 300.15;
        let expected = (2.0 * q * vt * vt / i_bias).sqrt(); // sqrt(2q Vt^2 / I)
        for nv in &r.psd {
            assert!(
                (nv - expected).abs() / expected < 0.05,
                "diode shot noise {nv} vs expected {expected}"
            );
        }
    }

    #[test]
    fn temp_sweep_diode_tempco() {
        // Forward-biased diode (~1 mA). Vf falls with temperature at about
        // -2 mV/degC (the classic silicon junction tempco).
        let net = "V1 in 0 5\nR1 in a 4.3k\nD1 a 0\n";
        let m = Model::from_netlist(net).expect("model");
        let r = m.temp_sweep(&[], "a", 27.0, 77.0, 2).expect("temp_sweep"); // tstart, tstop (deg C)
        assert_eq!(r.values.len(), 2);
        let slope_mv = (r.values[1] - r.values[0]) / (r.temps[1] - r.temps[0]) * 1000.0;
        assert!(
            slope_mv > -3.0 && slope_mv < -1.0,
            "expected ~ -2 mV/degC, got {slope_mv}"
        );
    }

    #[test]
    fn temp_sweep_reaches_veriloga_model() {
        // A Verilog-A diode whose thermal voltage is `$vt = k*T/q`: the device
        // temperature must reach the model (via the global `$temp` symbol), so
        // the operating point moves with temperature rather than staying flat.
        let net = "\
.veriloga
module vadio(a,c); inout a,c; electrical a,c;
parameter real Is=1e-14;
analog I(a,c) <+ Is*(exp(V(a,c)/$vt) - 1.0);
endmodule
.endveriloga
V1 in 0 5
R1 in a 4.3k
N1 a 0 vadio
";
        let m = Model::from_netlist(net).expect("model");
        let r = m.temp_sweep(&[], "a", -40.0, 125.0, 4).expect("temp_sweep");
        assert_eq!(r.values.len(), 4);
        let span = r.values.iter().cloned().fold(f64::MIN, f64::max)
            - r.values.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            span > 1e-3,
            "temperature did not reach the VA model (flat: {:?})",
            r.values
        );
    }

    #[test]
    fn dt_max_is_robust() {
        use super::*;
        use sane_solve::CompiledDc;
        // dt_max must never crash and must be honored. ESDIRK32 caps each step at
        // the requested 50 ns natively (no backend fallback). Both the uncapped and
        // capped runs are independent, error-controlled integrations of the same
        // 1 MHz-driven RC; with the solution-space error estimate the uncapped run
        // already resolves the source, so the two agree closely (not bit-identical,
        // as they take different step sequences).
        let parsed = parse("Vin a 0 SIN(0 1 1e6)\nR1 a b 1k\nC1 b 0 1n\n").unwrap();
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        let cdc = CompiledDc::new(&mut ctx, &dae);
        let pnames = cdc.param_names(&ctx);
        let p: Vec<f64> = parsed.pvec(&pnames);
        let t: Vec<f64> = (0..=4).map(|k| k as f64 * 0.25e-6).collect();
        let without = cdc
            .solve_transient(
                sane_solve::TransientMethod::Esdirk32,
                &p,
                &[],
                &t,
                1e-4,
                1e-7,
                None,
            )
            .unwrap();
        let withcap = cdc
            .solve_transient(
                sane_solve::TransientMethod::Esdirk32,
                &p,
                &[],
                &t,
                1e-4,
                1e-7,
                Some(50e-9),
            )
            .unwrap();
        assert_eq!(withcap.len(), t.len());
        // Both are accurate integrations of the same circuit, so they agree to a
        // tight (but not bit-exact) tolerance.
        let b = resolve_out_idx(&parsed, &dae, "b").unwrap();
        for (u, w) in without.iter().zip(&withcap) {
            assert!(
                (u[b] - w[b]).abs() < 1e-4,
                "capped and uncapped disagree: {} vs {}",
                u[b],
                w[b]
            );
        }
    }

    #[test]
    fn state_space_rc_descriptor() {
        // RC: E=C (one reactive state), A=-G, D=0, output = e_out^T.
        let net = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n";
        let m = Model::from_netlist(net).expect("model");
        let ss = m.state_space(&[], "V1", "out").expect("state_space");
        let n = m.unknowns().len();
        assert!(n >= 1 && ss.a.len() == n && ss.e.len() == n && ss.b.len() == n && ss.c.len() == n);
        assert_eq!(ss.d, 0.0);
        // E must have at least one nonzero entry (the capacitor's dynamics).
        assert!(
            ss.e.iter().any(|row| row.iter().any(|&v| v.abs() > 0.0)),
            "E is all zero"
        );
        // exactly one output pick.
        assert!((ss.c.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rlc_series_complex_poles() {
        // R=10, L=1m, C=1u: s = -R/2L +/- j*sqrt(1/LC - (R/2L)^2) = -5000 +/- j31225.
        let net = "V1 in 0 1\nR1 in a 10\nL1 a out 1m\nC1 out 0 1u\n";
        let m = Model::from_netlist(net).expect("model");
        let pz = m.poles_zeros(&[], "V1", "out").expect("pole_zero");
        let hit = pz
            .poles
            .iter()
            .any(|p| (p[0] + 5000.0).abs() < 200.0 && (p[1].abs() - 31225.0).abs() < 500.0);
        assert!(
            hit,
            "expected poles near -5000 +/- j31225, got {:?}",
            pz.poles
        );
    }
}
