//! AC-domain analyses: complex response, log-frequency sweep, response
//! sensitivity, and the descriptor state-space at the operating point.

use crate::linalg::{lu_factor_complex, lu_solve_complex};
use crate::input_vector;
use num_complex::Complex64;
use sane_core::constants::{DC_OP_MAXIT, DC_OP_TOL};
use sane_core::Graph;
use sane_solve::CompiledDc;
use std::f64::consts::PI;

#[cfg(test)]
use crate::solve_complex;
#[cfg(test)]
#[cfg(test)]
use sane_netlist::parse;

pub fn ac_h(
    ctx: &mut Graph,
    cdc: &CompiledDc,
    dae: &sane_dae::Dae,
    pnames: &[String],
    p: &[f64],
    input: &str,
    out_idx: usize,
    w: f64,
) -> Option<Complex64> {
    let n = dae.dim();
    let xdot0 = vec![0.0; n];
    let (x, conv, _) = cdc.solve_dc(p, &[], DC_OP_TOL, DC_OP_MAXIT);
    if !conv {
        return None;
    }
    // Sparse complex system A = G + jwC straight from the sparse Jacobians, with
    // the same gmin diagonal shunt the operating point was solved under.
    let (gr, gc, gv) = cdc.system_triplets_dc(&x, p);
    let (cr, cc, cv) = cdc.jacobian_xdot_sparse(&x, &xdot0, p, 0.0);
    let db = input_vector(ctx, dae, pnames, p, &x, input)?;
    let b: Vec<Complex64> = db.iter().map(|v| Complex64::new(-v, 0.0)).collect();
    let sys = crate::sparse_ac::AcSystem::assemble(n, (&gr, &gc, &gv), (&cr, &cc, &cv), w);
    sys.solve(&b).map(|v| v[out_idx])
}

/// Exact AC-transfer sensitivity `dH/dp(jw)` at each frequency, solved natively:
/// `dH = e_out^T A^{-1} (dB - (dG + jw dC) A^{-1} B)` with `A = G + jw C`. `b` is
/// the input coupling `B = -dF/d(input)` and `db = dB/dp` -- both already in
/// their final sign (as `input_jacobian` negated and `ac_derivatives` give them).
pub fn ac_response_sensitivity(
    g: &[Vec<f64>],
    c: &[Vec<f64>],
    b: &[f64],
    dg: &[Vec<f64>],
    dc: &[Vec<f64>],
    db: &[f64],
    out_idx: usize,
    freqs_hz: &[f64],
) -> Vec<[f64; 2]> {
    let n = g.len();
    let b: Vec<Complex64> = b.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let db: Vec<Complex64> = db.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut out = Vec::with_capacity(freqs_hz.len());
    for &f in freqs_hz {
        let w = 2.0 * PI * f;
        let mut a: Vec<Vec<Complex64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| Complex64::new(g[i][j], w * c[i][j]))
                    .collect()
            })
            .collect();
        // Factor A once; both the state response dx and the sensitivity dh solve
        // against the same matrix.
        let swaps = match lu_factor_complex(&mut a) {
            Some(s) => s,
            // Singular A = G + jwC at this frequency: no small-signal solution.
            // NaN, not a silent 0.0 that reads as a vanishing sensitivity
            // (same policy as the AC response itself, issue #39).
            None => {
                out.push([f64::NAN, f64::NAN]);
                continue;
            }
        };
        let dx = lu_solve_complex(&a, &swaps, &b);
        // rhs = dB - (dG + jw dC) dx.
        let mut rhs = db.clone();
        for i in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..n {
                acc += Complex64::new(dg[i][j], w * dc[i][j]) * dx[j];
            }
            rhs[i] -= acc;
        }
        let dh = lu_solve_complex(&a, &swaps, &rhs)[out_idx];
        out.push([dh.re, dh.im]);
    }
    out
}

/// Descriptor state-space `(E, A, B, C, D)` at the operating point, from the
/// already-assembled DAE: `E = dF/dx'`, `A = -dF/dx`, `B = -dF/d(input)`,
/// `C = e_out^T`, `D = 0`. (`E x' = A x + B u`, `y = C x + D u`.)
#[allow(clippy::type_complexity)]
pub fn state_space_on_dae(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    cdc: &CompiledDc,
    input: &str,
    out_idx: usize,
    x: &[f64],
    p: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>, f64) {
    let n = dae.dim();
    let xdot0 = vec![0.0; n];
    let g = cdc.jacobian_x(x, &xdot0, p, 0.0);
    let e = cdc.jacobian_xdot(x, &xdot0, p, 0.0);
    let a: Vec<Vec<f64>> = g
        .iter()
        .map(|row| row.iter().map(|v| -v).collect())
        .collect();
    let pnames = cdc.param_names(ctx);
    let b = match input_vector(ctx, dae, &pnames, p, x, input) {
        Some(db) => db.iter().map(|v| -v).collect(),
        None => vec![0.0; n],
    };
    let c: Vec<f64> = (0..n)
        .map(|i| if i == out_idx { 1.0 } else { 0.0 })
        .collect();
    (e, a, b, c, 0.0)
}

#[cfg(test)]
mod sparse_ac_equiv {
    use super::*;
    use num_complex::Complex64;

    // The sparse complex AC system (assembled from the sparse Jacobians + gmin)
    // must reproduce the dense `system_matrix_dc + jwC` solve bit-for-bit.
    #[test]
    fn sparse_ac_matches_dense() {
        let net = "V1 in 0 1\nR1 in a 100\nL1 a out 1m\nC1 out 0 1u\nR2 out 0 1k\n";
        let parsed = parse(net).unwrap();
        let mut ctx = Graph::new();
        let dae = parsed.assemble(&mut ctx);
        let cdc = CompiledDc::new(&mut ctx, &dae);
        let pnames = cdc.param_names(&ctx);
        let p: Vec<f64> = parsed.pvec(&pnames);
        let n = dae.dim();
        let xdot0 = vec![0.0; n];
        let (x, conv, _) = cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
        assert!(conv, "DC did not converge");
        let node = parsed.node("out").unwrap();
        let out_idx = dae
            .unknowns
            .iter()
            .position(|u| *u == format!("v{node}"))
            .unwrap();
        let db = input_vector(&mut ctx, &dae, &pnames, &p, &x, "V1").unwrap();
        let b: Vec<Complex64> = db.iter().map(|v| Complex64::new(-v, 0.0)).collect();
        let w = 2.0 * PI * 1e3;

        // dense reference
        let g = cdc.system_matrix_dc(&x, &xdot0, &p, 0.0);
        let c = cdc.jacobian_xdot(&x, &xdot0, &p, 0.0);
        let mut a = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = Complex64::new(g[i][j], w * c[i][j]);
            }
        }
        let h_dense = solve_complex(a, b.clone()).unwrap()[out_idx];

        // sparse path (gmin as diagonal triplets, summed in assembly)
        let (mut gr, mut gc, mut gv) = cdc.jacobian_x_sparse(&x, &xdot0, &p, 0.0);
        for i in 0..n {
            gr.push(i);
            gc.push(i);
            gv.push(sane_core::constants::GMIN_DC);
        }
        let (cr, cc, cv) = cdc.jacobian_xdot_sparse(&x, &xdot0, &p, 0.0);
        let sys = crate::sparse_ac::AcSystem::assemble(n, (&gr, &gc, &gv), (&cr, &cc, &cv), w);
        let h_sparse = sys.solve(&b).unwrap()[out_idx];

        let rel = (h_dense - h_sparse).norm() / h_dense.norm().max(1e-30);
        assert!(
            rel < 1e-9,
            "sparse {h_sparse:?} vs dense {h_dense:?} (rel {rel:.2e})"
        );
    }
}
