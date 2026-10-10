//! Model-order reduction: dominant-pole reduction of the small-signal
//! transfer.

use crate::{dominant_subset, finite_pencil_roots, input_vector, solve_complex};
use num_complex::Complex64;
use sane_core::Graph;
use sane_solve::CompiledDc;
use std::f64::consts::PI;

/// A dominant-pole reduction (see [`model_reduce_on_dae`]): the kept roots
/// `[re, im]` (rad/s), the gain and the full against the reduced transfer.
pub struct Reduction {
    pub poles: Vec<[f64; 2]>,
    pub zeros: Vec<[f64; 2]>,
    pub gain: Complex64,
    pub full: Vec<Complex64>,
    pub reduced: Vec<Complex64>,
}

/// Dominant-pole model-order reduction of the `input -> out_idx` transfer, from
/// the already-assembled DAE at the operating point `x`. Keeps the `order` most
/// dominant poles (and matching zeros) of the small-signal pencil, fits the gain
/// `K` to the full DC gain, and evaluates the full and the reduced transfer at
/// `freqs` (Hz).
#[allow(clippy::too_many_arguments)]
pub fn model_reduce_on_dae(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    cdc: &CompiledDc,
    input: &str,
    out_idx: usize,
    x: &[f64],
    p: &[f64],
    order: usize,
    freqs: &[f64],
) -> Result<Reduction, String> {
    if order == 0 || freqs.iter().any(|&f| !(f >= 0.0)) {
        return Err("model_reduce needs order >= 1 and frequencies >= 0".into());
    }
    let n = dae.dim();
    let g = cdc.system_matrix_dc(x, p, 0.0);
    let c = cdc.jacobian_q_x(x, p, 0.0);
    let pnames = cdc.param_names(ctx);
    let b_real = match input_vector(ctx, dae, &pnames, p, x, input) {
        Some(db) => db,
        None => return Err(format!("input '{input}' is not a source parameter")),
    };
    let b: Vec<Complex64> = b_real.iter().map(|v| Complex64::new(-v, 0.0)).collect();

    let poles = finite_pencil_roots(&g, &c).unwrap_or_default();
    // Rosenbrock zeros.
    let mut m = vec![vec![0.0; n + 1]; n + 1];
    let mut nn = vec![vec![0.0; n + 1]; n + 1];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = g[i][j];
            nn[i][j] = c[i][j];
        }
        m[i][n] = b_real[i];
        m[n][i] = if i == out_idx { 1.0 } else { 0.0 };
    }
    let zeros = finite_pencil_roots(&m, &nn).unwrap_or_default();

    let kept_poles = dominant_subset(&poles, order);
    let kept_zeros = dominant_subset(&zeros, order.min(kept_poles.len()));

    // Helper: full H at a complex s via (G + sC) solve.
    let solve_full = |w: f64| -> Complex64 {
        let mut a = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = Complex64::new(g[i][j], w * c[i][j]);
            }
        }
        solve_complex(a, b.clone())
            .map(|v| v[out_idx])
            .unwrap_or(Complex64::new(0.0, 0.0))
    };
    // Reduced rational at complex s (without K).
    let prod = |roots: &[[f64; 2]], s: Complex64| -> Complex64 {
        roots.iter().fold(Complex64::new(1.0, 0.0), |acc, r| {
            acc * (s - Complex64::new(r[0], r[1]))
        })
    };
    // Fit K to the full DC gain: H(0) = K * prod(-z)/prod(-p).
    let h0 = solve_full(0.0);
    let s0 = Complex64::new(0.0, 0.0);
    let pz = prod(&kept_zeros, s0);
    let pp = prod(&kept_poles, s0);
    let k = if pz.norm() > 0.0 {
        h0 * pp / pz
    } else {
        h0 * pp
    };

    let (mut full, mut reduced) = (
        Vec::with_capacity(freqs.len()),
        Vec::with_capacity(freqs.len()),
    );
    for &fi in freqs {
        let w = 2.0 * PI * fi;
        full.push(solve_full(w));
        reduced.push(
            k * prod(&kept_zeros, Complex64::new(0.0, w))
                / prod(&kept_poles, Complex64::new(0.0, w)),
        );
    }
    Ok(Reduction {
        poles: kept_poles,
        zeros: kept_zeros,
        gain: k,
        full,
        reduced,
    })
}
