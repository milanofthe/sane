//! Model-order reduction and symbolic transfer-function approximation:
//! dominant-pole reduction plus Analog-Insydes-style symbolic term / entry
//! pruning of the small-signal transfer.

use crate::symbolic_poly::{build_poly_expr, poly_in_s, prune_poly};
use crate::{dominant_subset, finite_pencil_roots, input_vector, solve_complex};
use num_complex::Complex64;
use rsdag::{differentiate, Node, SymbolId};
use sane_core::constants::{DC_OP_MAXIT, DC_OP_TOL};
use sane_core::log;
use sane_core::Graph;
use sane_solve::CompiledDc;
use std::f64::consts::PI;

#[cfg(test)]
use sane_dae::assemble_dae;
#[cfg(test)]
use sane_netlist::parse;

/// Dominant-pole model-order reduction of the `input -> out_idx` transfer, from
/// the already-assembled DAE at the operating point `x`. Keeps the `order` most
/// dominant poles (and matching zeros) of the small-signal pencil, fits the gain
/// `K` to the full DC gain, and reports the full vs. reduced magnitude over a log
/// band. Returns `(freqs_hz, full_db, reduced_db, kept_poles, kept_zeros, max_err_db)`.
#[allow(clippy::type_complexity)]
pub fn model_reduce_on_dae(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    cdc: &CompiledDc,
    input: &str,
    out_idx: usize,
    x: &[f64],
    p: &[f64],
    order: usize,
    fstart: f64,
    fstop: f64,
    points: usize,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<[f64; 2]>,
        Vec<[f64; 2]>,
        f64,
    ),
    String,
> {
    if !(fstart > 0.0) || !(fstop > fstart) || points < 2 || order == 0 {
        return Err("model_reduce needs order >= 1, 0 < fstart < fstop, points >= 2".into());
    }
    let n = dae.dim();
    let xdot0 = vec![0.0; n];
    let g = cdc.system_matrix_dc(x, &xdot0, p, 0.0);
    let c = cdc.jacobian_xdot(x, &xdot0, p, 0.0);
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

    let (l0, l1) = (fstart.log10(), fstop.log10());
    let (mut f, mut full_db, mut red_db) = (
        Vec::with_capacity(points),
        Vec::with_capacity(points),
        Vec::with_capacity(points),
    );
    let mut max_err_db = 0.0_f64;
    for i in 0..points {
        let fi = 10f64.powf(l0 + (l1 - l0) * i as f64 / (points - 1) as f64);
        let w = 2.0 * PI * fi;
        let s = Complex64::new(0.0, w);
        let hf = solve_full(w);
        let hr = k * prod(&kept_zeros, s) / prod(&kept_poles, s);
        let fdb = 20.0 * hf.norm().max(1e-30).log10();
        let rdb = 20.0 * hr.norm().max(1e-30).log10();
        max_err_db = max_err_db.max((fdb - rdb).abs());
        f.push(fi);
        full_db.push(fdb);
        red_db.push(rdb);
    }
    Ok((f, full_db, red_db, kept_poles, kept_zeros, max_err_db))
}

// --- Symbolic term-pruning model reduction (Analog-Insydes SBG/SAG idea) ----
// The polynomial-in-`s` machinery lives in `symbolic_poly`.

/// Analog-Insydes-style symbolic approximation of the small-signal transfer
/// function: collect `H(s)=N(s)/D(s)` as polynomials in `s`, rank every monomial
/// of every coefficient by its magnitude at the DC operating point and reference
/// frequency `freq`, drop those below `tol` of the dominant term, and rebuild the
/// pruned `H(s)`. Operates on an already-extracted DAE (the symbolic pruning is
/// exact term selection; the result is a compact symbolic expression). Returns
/// `(H_pruned, terms_total, terms_kept)`. `output` is a resolved unknown name.
pub fn symbolic_transfer_approx(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    cdc: &CompiledDc,
    values: &std::collections::HashMap<String, f64>,
    input: &str,
    output: &str,
    tol: f64,
    freq: f64,
) -> Option<(rsdag::ExprId, usize, usize)> {
    let (n_expr, d_expr) = sane_dae::small_signal_transfer_nd(ctx, dae, input, output)?;
    let s_e = ctx.sym("s");
    let s_sym = match ctx.node(s_e) {
        Node::Symbol(x) => *x,
        _ => return None,
    };
    let n_poly = poly_in_s(ctx, n_expr, s_sym)?;
    let d_poly = poly_in_s(ctx, d_expr, s_sym)?;

    // DC operating-point environment for ranking the symbolic coefficients.
    let pnames = cdc.param_names(ctx);
    let p: Vec<f64> = pnames
        .iter()
        .map(|n| values.get(n).copied().unwrap_or(0.0))
        .collect();
    let (x, conv, _) = cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
    if !conv {
        return None;
    }
    let mut env: std::collections::HashMap<SymbolId, f64> = std::collections::HashMap::new();
    for (i, &sy) in dae.x.iter().enumerate() {
        env.insert(sy, x.get(i).copied().unwrap_or(0.0));
    }
    for sy in dae.xdot.iter().flatten() {
        env.insert(*sy, 0.0);
    }
    for (j, name) in pnames.iter().enumerate() {
        let ee = ctx.sym(name);
        if let Node::Symbol(sy) = ctx.node(ee) {
            let sy = *sy;
            env.insert(sy, p.get(j).copied().unwrap_or(0.0));
        }
    }
    env.insert(dae.t, 0.0);

    let w0 = 2.0 * PI * freq;
    let (n_pr, n_tot, n_kept) = prune_poly(ctx, &n_poly, &env, w0, tol);
    let (d_pr, d_tot, d_kept) = prune_poly(ctx, &d_poly, &env, w0, tol);
    let n_e = build_poly_expr(ctx, &n_pr, s_e);
    let d_e = build_poly_expr(ctx, &d_pr, s_e);
    let h = ctx.div(n_e, d_e);
    Some((h, n_tot + d_tot, n_kept + d_kept))
}

/// Single-frequency symbolic approximation (Analog-Insydes "simplification before
/// generation") with the device entries kept in expanded form. The small-signal
/// tableau is linearized at the operating point and `freq`, the transfer is solved
/// numerically, and each entry is ranked by the exact change its removal causes in
/// `H` (a Sherman-Morrison rank-1 update, free of determinant cancellation). The
/// entries whose removal keeps the cumulative relative error below `tol` are
/// dropped; the surviving sparse matrix yields a compact closed form by Cramer's
/// rule. For a readable expression in named admittance stamps (recommended for
/// transistor circuits, whose expanded entries are large), use
/// [`symbolic_transfer_approx_named_at`]. Returns
/// `(H, kept_entries, total_entries, det_terms, H_re, H_im)`, where `det_terms`
/// is the number of product terms in the reduced denominator determinant.
pub fn symbolic_transfer_approx_at(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    cdc: &CompiledDc,
    values: &std::collections::HashMap<String, f64>,
    input: &str,
    output: &str,
    freq: f64,
    tol: f64,
    cap: usize,
) -> Option<(rsdag::ExprId, usize, usize, usize, f64, f64)> {
    use num_complex::Complex64;
    let _ = cap; // entry pruning is error-bounded, not term-capped
    let n_params = dae.params(ctx).len();
    log::info(&format!(
        "Symbolic transfer (entry pruning): {input} -> {output}, dim {}, {n_params} params, f={freq:.3e} Hz, tol={tol:.1e}",
        dae.dim()
    ));
    let t_start = sane_core::time::Instant::now();
    let col = dae.unknowns.iter().position(|u| u == output)?;
    let input_e = ctx.sym(input);
    let input_sym = match ctx.node(input_e) {
        Node::Symbol(s) => *s,
        _ => return None,
    };
    // Excitation b = -dF/d(input); Cramer matrix a_b is `a` with column `col` -> b.
    let mut b = Vec::with_capacity(dae.residuals.len());
    for &r in &dae.residuals {
        let d = differentiate(ctx, r, input_sym);
        b.push(ctx.neg(d));
    }
    let a = sane_dae::small_signal_matrix(ctx, dae);
    let mut a_b = a.clone();
    for (row, br) in b.iter().enumerate() {
        a_b[row][col] = *br;
    }

    // Complex environment at the operating point and reference frequency.
    let pnames = cdc.param_names(ctx);
    let p: Vec<f64> = pnames
        .iter()
        .map(|n| values.get(n).copied().unwrap_or(0.0))
        .collect();
    let (x, conv, _) = cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
    if !conv {
        return None;
    }
    let mut env: std::collections::HashMap<SymbolId, Complex64> = std::collections::HashMap::new();
    for (i, &sy) in dae.x.iter().enumerate() {
        env.insert(sy, Complex64::new(x.get(i).copied().unwrap_or(0.0), 0.0));
    }
    for sy in dae.xdot.iter().flatten() {
        env.insert(*sy, Complex64::new(0.0, 0.0));
    }
    for (j, name) in pnames.iter().enumerate() {
        let ee = ctx.sym(name);
        if let Node::Symbol(sy) = ctx.node(ee) {
            let sy = *sy;
            env.insert(sy, Complex64::new(p.get(j).copied().unwrap_or(0.0), 0.0));
        }
    }
    let s_e = ctx.sym("s");
    if let Node::Symbol(sy) = ctx.node(s_e) {
        let sy = *sy;
        env.insert(sy, Complex64::new(0.0, 2.0 * PI * freq));
    }
    env.insert(dae.t, Complex64::new(0.0, 0.0));

    // Numeric values, structural masks, then Sherman-Morrison entry pruning.
    let n = a.len();
    let va: Vec<Vec<Complex64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| rsdag::eval(ctx, &[a[i][j]], &env)[0])
                .collect()
        })
        .collect();
    let b_val: Vec<Complex64> = (0..n).map(|i| rsdag::eval(ctx, &[b[i]], &env)[0]).collect();
    let a_struct: Vec<Vec<bool>> = (0..n)
        .map(|i| (0..n).map(|j| !ctx.is_zero(a[i][j])).collect())
        .collect();
    let b_struct: Vec<bool> = (0..n).map(|i| !ctx.is_zero(b[i])).collect();
    let t_prune = sane_core::time::Instant::now();
    let (keep_a, keep_b, h0, _hv) = prune_tableau(&va, &b_val, &a_struct, &b_struct, col, tol)?;

    let total_entries =
        a_struct.iter().flatten().filter(|&&s| s).count() + b_struct.iter().filter(|&&s| s).count();
    let kept_entries =
        keep_a.iter().flatten().filter(|&&s| s).count() + keep_b.iter().filter(|&&s| s).count();
    log::info(&format!(
        "  entry pruning: kept {kept_entries}/{total_entries} tableau entries ({:.1} ms)",
        t_prune.elapsed().as_secs_f64() * 1e3
    ));

    // Reduced (expanded) matrix + excitation, then symbolic LU generation.
    let zero = ctx.zero();
    let mut ra = vec![vec![zero; n]; n];
    let mut vra = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut rb_sym = vec![zero; n];
    let mut rb_val = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        if keep_b[i] {
            rb_sym[i] = b[i];
            rb_val[i] = b_val[i];
        }
        for j in 0..n {
            if keep_a[i][j] {
                ra[i][j] = a[i][j];
                vra[i][j] = va[i][j];
            }
        }
    }
    let t_gen = sane_core::time::Instant::now();
    let (h, hv) = generate_compact(ctx, &ra, &vra, &rb_sym, &rb_val, col, h0, tol)?;
    let n_terms = count_nodes(ctx, h);
    let rel = if h0.norm() > 0.0 {
        (hv - h0).norm() / h0.norm()
    } else {
        0.0
    };
    log::info(&format!(
        "  compact generation: {n_terms} expression nodes, rel. error {rel:.2e} ({:.1} ms); total {:.1} ms",
        t_gen.elapsed().as_secs_f64() * 1e3,
        t_start.elapsed().as_secs_f64() * 1e3
    ));
    Some((h, kept_entries, total_entries, n_terms, hv.re, hv.im))
}

/// Named-stamp single-frequency symbolic approximation (SLiCAP / Analog-Insydes
/// "symbolic MNA" form) via numeric tableau-entry pruning. The small-signal
/// matrix is linearized at the operating point and `freq`; the transfer `H` is
/// solved numerically and each tableau entry is ranked by the exact change its
/// removal causes in `H` (a Sherman-Morrison rank-1 update of the system inverse,
/// so cheap and free of determinant cancellation). The entries whose removal
/// keeps the cumulative relative error below `tol` are dropped; the surviving
/// entries become named admittance symbols `y{row}_{col}` (excitation `b{row}`),
/// and the compact symbolic `H` is the ratio of determinants of that sparse
/// reduced matrix. Returns `(H, legend, kept_entries, total_entries, H_re, H_im)`,
/// where `legend` lists `(symbol_name, value_re, value_im)` for the stamps in `H`
/// and `(H_re, H_im)` is the reduced transfer for a numeric self-check.
#[allow(clippy::too_many_arguments)]
pub fn symbolic_transfer_approx_named_at(
    ctx: &mut Graph,
    dae: &sane_dae::Dae,
    cdc: &CompiledDc,
    values: &std::collections::HashMap<String, f64>,
    input: &str,
    output: &str,
    freq: f64,
    tol: f64,
    cap: usize,
) -> Option<(
    rsdag::ExprId,
    Vec<(String, f64, f64)>,
    usize,
    usize,
    usize,
    f64,
    f64,
)> {
    use num_complex::Complex64;
    let n_params = dae.params(ctx).len();
    log::info(&format!(
        "Symbolic transfer (named, entry pruning): {input} -> {output}, dim {}, {n_params} params, f={freq:.3e} Hz, tol={tol:.1e}",
        dae.dim()
    ));
    let t_start = sane_core::time::Instant::now();
    let n = dae.dim();
    let col = dae.unknowns.iter().position(|u| u == output)?;

    // Operating point + parameter vector.
    let pnames = cdc.param_names(ctx);
    let p: Vec<f64> = pnames
        .iter()
        .map(|nm| values.get(nm).copied().unwrap_or(0.0))
        .collect();
    let (x, conv, _) = cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
    if !conv {
        return None;
    }

    // Sparse complex tableau A = G(+gmin) + jwC straight from the compiled tape
    // (no symbolic differentiation of the DAE, no per-entry DAG evaluation), and
    // the numeric excitation b = -dF/d(input). Each nonzero is one tableau entry
    // (one symbolic stamp); the system is never densified.
    let xdot0 = vec![0.0; n];
    let w = 2.0 * PI * freq;
    let (gr, gc, gv) = cdc.system_triplets_dc(&x, &p);
    let (cr, cc, cv) = cdc.jacobian_xdot_sparse(&x, &xdot0, &p, 0.0);
    let dbv = input_vector(ctx, dae, &pnames, &p, &x, input)?;
    let b_val: Vec<Complex64> = (0..n).map(|i| Complex64::new(-dbv[i], 0.0)).collect();
    let triplets: Vec<(usize, usize, Complex64)> =
        crate::sparse_ac::AcSystem::assemble(n, (&gr, &gc, &gv), (&cr, &cc, &cv), w)
            .entries()
            .to_vec();
    let _ = cap; // entry pruning is error-bounded, not term-capped

    // Sparse, matrix-free entry pruning: rank each entry by the first-order effect
    // of its removal (from one solve and one adjoint solve), then greedily drop in
    // increasing-effect order, accepting a removal only if the exactly re-solved
    // reduced transfer stays within `tol`. Operates on the sparse symbolic system
    // (each entry is a stamp), so the DAG / device hierarchy behind each kept entry
    // is preserved; every solve is a sparse faer factorisation, never dense.
    let t_prune = sane_core::time::Instant::now();
    let (keep_t, keep_b, h0, _hv) = prune_tableau_sparse(n, &triplets, &b_val, col, tol)?;

    let total_entries = triplets.len() + b_val.iter().filter(|v| v.norm() > 0.0).count();
    let kept_entries =
        keep_t.iter().filter(|&&s| s).count() + keep_b.iter().filter(|&&s| s).count();
    log::info(&format!(
        "  entry pruning: kept {kept_entries}/{total_entries} tableau entries ({:.1} ms)",
        t_prune.elapsed().as_secs_f64() * 1e3
    ));

    // Name the surviving entries (`y{i}_{j}`, excitation `b{i}`) with their numeric
    // values and build the reduced named matrix for the symbolic generation.
    let mut nenv: std::collections::HashMap<SymbolId, Complex64> = std::collections::HashMap::new();
    let mut named = |ctx: &mut Graph, name: String, v: Complex64| -> rsdag::ExprId {
        let e = ctx.sym(&name);
        if let Node::Symbol(sy) = ctx.node(e) {
            nenv.insert(*sy, v);
        }
        e
    };
    let zero2 = ctx.zero();
    let mut ra = vec![vec![zero2; n]; n];
    let mut vra = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut rb_sym = vec![zero2; n];
    let mut rb_val = vec![Complex64::new(0.0, 0.0); n];
    for (k, &(i, j, v)) in triplets.iter().enumerate() {
        if keep_t[k] {
            ra[i][j] = named(ctx, format!("y{i}_{j}"), v);
            vra[i][j] = v;
        }
    }
    for i in 0..n {
        if keep_b[i] {
            rb_sym[i] = named(ctx, format!("b{i}"), b_val[i]);
            rb_val[i] = b_val[i];
        }
    }

    // Generate the compact symbolic transfer by symbolic Gaussian elimination with
    // numeric pivoting and an adaptive drop tolerance: the determinant comes out in
    // factored (pivot-product) form, not as a permutation sum, so it is compact and
    // free of cancellation.
    let t_gen = sane_core::time::Instant::now();
    let (h, hv) = generate_compact(ctx, &ra, &vra, &rb_sym, &rb_val, col, h0, tol)?;
    let n_terms = count_nodes(ctx, h);
    let rel = if h0.norm() > 0.0 {
        (hv - h0).norm() / h0.norm()
    } else {
        0.0
    };
    log::info(&format!(
        "  compact generation: {n_terms} expression nodes, rel. error {rel:.2e} ({:.1} ms); total {:.1} ms",
        t_gen.elapsed().as_secs_f64() * 1e3,
        t_start.elapsed().as_secs_f64() * 1e3
    ));

    // legend: the named stamps that actually survive in H.
    let mut seen = std::collections::HashSet::new();
    let mut syms = std::collections::HashSet::new();
    collect_symbols(ctx, h, &mut seen, &mut syms);
    let mut legend: Vec<(String, f64, f64)> = syms
        .into_iter()
        .filter_map(|sy| {
            nenv.get(&sy)
                .map(|v| (ctx.symbol_name(sy).to_string(), v.re, v.im))
        })
        .collect();
    legend.sort_by(|a, b| a.0.cmp(&b.0));
    let stamps = legend.len();
    Some((h, legend, stamps, total_entries, n_terms, hv.re, hv.im))
}

/// Reduced transfer `H = (A_r^{-1} b_r)[out]` for the tableau masked by
/// `keep_a`/`keep_b` (removed entries set to zero). `None` if singular.
fn solve_reduced_h(
    va: &[Vec<num_complex::Complex64>],
    b: &[num_complex::Complex64],
    keep_a: &[Vec<bool>],
    keep_b: &[bool],
    out: usize,
) -> Option<num_complex::Complex64> {
    use num_complex::Complex64;
    let n = va.len();
    let mut ar = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            if keep_a[i][j] {
                ar[i][j] = va[i][j];
            }
        }
    }
    let br: Vec<Complex64> = (0..n)
        .map(|i| {
            if keep_b[i] {
                b[i]
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();
    let swaps = crate::linalg::lu_factor_complex(&mut ar)?;
    let x = crate::linalg::lu_solve_complex(&ar, &swaps, &br);
    Some(x[out])
}

/// Numeric entry pruning of the linear tableau `(A, b)` for the transfer
/// `H = (A^{-1} b)[out]` (Analog-Insydes "simplification before generation").
/// Each entry is ranked by the first-order change its removal causes in `H`
/// (`a_ij y_i x_j` for the matrix, `b_i y_i` for the excitation, with the adjoint
/// `y = A^{-T} e_out`); entries are then dropped in increasing-effect order, each
/// accepted only if re-solving the reduced system keeps the relative error in `H`
/// below `tol`. The error is checked by an exact factorization every step (no
/// drifting inverse update), so the kept matrix is always nonsingular and the
/// pruning is robust at any size. Returns the keep-masks, `H0`, and the reduced
/// `H`. `None` if `A` is singular or `H` is ~0.
#[allow(clippy::type_complexity)]
fn prune_tableau(
    va: &[Vec<num_complex::Complex64>],
    b: &[num_complex::Complex64],
    a_struct: &[Vec<bool>],
    b_struct: &[bool],
    out: usize,
    tol: f64,
) -> Option<(
    Vec<Vec<bool>>,
    Vec<bool>,
    num_complex::Complex64,
    num_complex::Complex64,
)> {
    use num_complex::Complex64;
    let n = va.len();
    // x = A^{-1} b and adjoint y = A^{-T} e_out, from one factorization.
    let mut fac = va.to_vec();
    let swaps = crate::linalg::lu_factor_complex(&mut fac)?;
    let x = crate::linalg::lu_solve_complex(&fac, &swaps, b);
    let h0 = x[out];
    if h0.norm() < 1e-300 {
        return None; // no meaningful transfer at this point
    }
    // y solves A^T y = e_out; build A^T and factor (n is modest in the symbolic regime).
    let mut at = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            at[i][j] = va[j][i];
        }
    }
    let mut e_out = vec![Complex64::new(0.0, 0.0); n];
    e_out[out] = Complex64::new(1.0, 0.0);
    let swaps_t = crate::linalg::lu_factor_complex(&mut at)?;
    let y = crate::linalg::lu_solve_complex(&at, &swaps_t, &e_out);

    // rank removable entries by first-order |dH| (ascending: least important first).
    let mut items: Vec<(f64, bool, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if a_struct[i][j] {
                let dh = (va[i][j] * y[i] * x[j]).norm();
                items.push((dh, true, i, j));
            }
        }
    }
    for i in 0..n {
        if b_struct[i] {
            let dh = (b[i] * y[i]).norm();
            items.push((dh, false, i, 0));
        }
    }
    items.sort_by(|p, q| p.0.total_cmp(&q.0));

    // greedily drop in that order; accept a removal only if the exactly re-solved
    // reduced transfer stays within tol of H0.
    let mut keep_a: Vec<Vec<bool>> = a_struct.to_vec();
    let mut keep_b: Vec<bool> = b_struct.to_vec();
    let mut h_cur = h0;
    for (_, is_a, i, j) in items {
        if is_a {
            if !keep_a[i][j] {
                continue;
            }
            keep_a[i][j] = false;
            match solve_reduced_h(va, b, &keep_a, &keep_b, out) {
                Some(h) if (h - h0).norm() / h0.norm() < tol => h_cur = h,
                _ => keep_a[i][j] = true, // revert: removal singularizes or over-errs
            }
        } else {
            if !keep_b[i] {
                continue;
            }
            keep_b[i] = false;
            match solve_reduced_h(va, b, &keep_a, &keep_b, out) {
                Some(h) if (h - h0).norm() / h0.norm() < tol => h_cur = h,
                _ => keep_b[i] = true,
            }
        }
    }

    Some((keep_a, keep_b, h0, h_cur))
}

/// Sparse, matrix-free entry pruning of the small-signal tableau for the transfer
/// `H = (A^{-1} b)[out]`, with `A = G + jwC` given as summed complex triplets
/// (each triplet is one tableau entry, i.e. one symbolic stamp; the system is
/// never densified, so the hash-consed DAG / device hierarchy behind each entry
/// is preserved). Ranks every entry by the first-order effect of its removal
/// (`a_ij x_j y_i` for a matrix entry, `b_i y_i` for an excitation entry, from one
/// solve and one adjoint solve), then drops in increasing-effect order, accepting
/// a removal only if the exactly re-solved reduced transfer stays within `tol` of
/// `H0`. Every solve is a sparse faer factorisation. Returns the keep-mask over
/// `triplets`, the kept-`b` mask, `H0`, and the reduced `H`.
#[allow(clippy::type_complexity)]
fn prune_tableau_sparse(
    n: usize,
    triplets: &[(usize, usize, num_complex::Complex64)],
    b: &[num_complex::Complex64],
    out: usize,
    tol: f64,
) -> Option<(
    Vec<bool>,
    Vec<bool>,
    num_complex::Complex64,
    num_complex::Complex64,
)> {
    use crate::sparse_ac::AcSystem;
    use num_complex::Complex64;

    let masked_solve = |keep_t: &[bool], bcur: &[Complex64]| -> Option<Complex64> {
        let sub: Vec<(usize, usize, Complex64)> = triplets
            .iter()
            .enumerate()
            .filter(|(k, _)| keep_t[*k])
            .map(|(_, &t)| t)
            .collect();
        AcSystem::from_triplets(n, sub).solve(bcur).map(|v| v[out])
    };

    let full = AcSystem::from_triplets(n, triplets.to_vec());
    let x = full.solve(b)?;
    let h0 = x[out];
    if h0.norm() < 1e-300 {
        return None;
    }
    let mut e_out = vec![Complex64::new(0.0, 0.0); n];
    e_out[out] = Complex64::new(1.0, 0.0);
    let y = full.solve_transpose(&e_out)?;

    // rank removable entries by first-order |dH| (ascending: least important first).
    let mut items: Vec<(f64, bool, usize)> = Vec::new();
    for (k, &(i, j, v)) in triplets.iter().enumerate() {
        items.push(((v * x[j] * y[i]).norm(), true, k));
    }
    for (i, &bi) in b.iter().enumerate() {
        if bi.norm() > 0.0 {
            items.push(((bi * y[i]).norm(), false, i));
        }
    }
    items.sort_by(|p, q| p.0.total_cmp(&q.0));

    let mut keep_t = vec![true; triplets.len()];
    let mut keep_b: Vec<bool> = b.iter().map(|v| v.norm() > 0.0).collect();
    let mut bcur = b.to_vec();
    let mut h_cur = h0;

    for (_, is_mat, idx) in items {
        if is_mat {
            if !keep_t[idx] {
                continue;
            }
            keep_t[idx] = false;
            match masked_solve(&keep_t, &bcur) {
                Some(h) if (h - h0).norm() / h0.norm() < tol => h_cur = h,
                _ => keep_t[idx] = true,
            }
        } else {
            if !keep_b[idx] {
                continue;
            }
            keep_b[idx] = false;
            let saved = bcur[idx];
            bcur[idx] = Complex64::new(0.0, 0.0);
            match masked_solve(&keep_t, &bcur) {
                Some(h) if (h - h0).norm() / h0.norm() < tol => h_cur = h,
                _ => {
                    keep_b[idx] = true;
                    bcur[idx] = saved;
                }
            }
        }
    }
    Some((keep_t, keep_b, h0, h_cur))
}

/// Number of distinct DAG nodes in `id` (its hash-consed size): the readability
/// metric for a symbolic expression, since shared subexpressions count once.
fn count_nodes(ctx: &Graph, id: rsdag::ExprId) -> usize {
    let mut seen = std::collections::HashSet::new();
    fn rec(ctx: &Graph, id: rsdag::ExprId, seen: &mut std::collections::HashSet<rsdag::ExprId>) {
        if !seen.insert(id) {
            return;
        }
        match ctx.node(id) {
            Node::Symbol(_) | Node::Const(_) => {}
            Node::Add(a, b) | Node::Mul(a, b) | Node::Cmp(_, a, b) | Node::Binary(_, a, b) => {
                rec(ctx, *a, seen);
                rec(ctx, *b, seen);
            }
            Node::Neg(a) | Node::Pow(a, _) | Node::Unary(_, a) => rec(ctx, *a, seen),
            Node::Select(c, t, e) => {
                rec(ctx, *c, seen);
                rec(ctx, *t, seen);
                rec(ctx, *e, seen);
            }
            Node::Reduce(_, l) | Node::Call(_, l) | Node::Dot(l) | Node::Solve(l, _) => {
                ctx.args(*l).iter().for_each(|&c| rec(ctx, c, seen))
            }
        }
    }
    rec(ctx, id, &mut seen);
    seen.len()
}

/// Symbolic solve of `A x = b` for the single unknown `x[out]` by Gaussian
/// elimination with numeric partial pivoting and a relative drop tolerance. Each
/// matrix entry carries both its symbolic form (`sym`) and its numeric value
/// (`val`); the values drive pivoting and dropping while the symbols accumulate
/// the elimination, so the determinant is generated in factored (pivot-product)
/// form, free of the permutation-expansion blow-up and of catastrophic
/// cancellation (every pivot is the real post-cancellation value). A fill entry
/// whose magnitude falls below `drop_tol` times the current row scale is dropped,
/// which is what keeps the symbolic result compact. Returns the symbolic `x[out]`
/// or `None` if a (numerically) singular pivot is hit.
#[allow(clippy::too_many_arguments)]
fn symbolic_lu_solve(
    ctx: &mut Graph,
    mut sym: Vec<Vec<rsdag::ExprId>>,
    mut val: Vec<Vec<num_complex::Complex64>>,
    mut rhs_sym: Vec<rsdag::ExprId>,
    mut rhs_val: Vec<num_complex::Complex64>,
    out: usize,
    drop_tol: f64,
    max_ops: usize,
) -> Option<(rsdag::ExprId, num_complex::Complex64)> {
    use num_complex::Complex64;
    let n = val.len();
    let zero = ctx.zero();
    // bound the generated expression: count the symbolic fill operations and abort
    // (the closed form would not be compact at this drop tolerance anyway). This is
    // the "check the size while generating" guard that prevents blow-up.
    let mut ops = 0usize;
    for k in 0..n {
        // partial pivot on numeric magnitude.
        let mut piv = k;
        let mut best = val[k][k].norm();
        for r in (k + 1)..n {
            if val[r][k].norm() > best {
                best = val[r][k].norm();
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != k {
            sym.swap(k, piv);
            val.swap(k, piv);
            rhs_sym.swap(k, piv);
            rhs_val.swap(k, piv);
        }
        let pv = val[k][k];
        let psym = sym[k][k];
        for i in (k + 1)..n {
            if val[i][k].norm() == 0.0 {
                continue;
            }
            let f_val = val[i][k] / pv;
            let f_sym = ctx.div(sym[i][k], psym);
            // row scale for the drop test (largest current magnitude in row i).
            let rowmax = (k..n)
                .map(|j| val[i][j].norm())
                .fold(0.0f64, f64::max)
                .max(rhs_val[i].norm());
            for j in (k + 1)..n {
                if val[k][j].norm() == 0.0 {
                    continue;
                }
                let nv = val[i][j] - f_val * val[k][j];
                if nv.norm() < drop_tol * rowmax {
                    // dropped: no symbolic node is built for a negligible fill entry.
                    val[i][j] = Complex64::new(0.0, 0.0);
                    sym[i][j] = zero;
                } else {
                    let prod = ctx.mul(f_sym, sym[k][j]);
                    sym[i][j] = if ctx.is_zero(sym[i][j]) {
                        ctx.neg(prod)
                    } else {
                        ctx.sub(sym[i][j], prod)
                    };
                    val[i][j] = nv;
                    ops += 1;
                    if ops > max_ops {
                        return None; // closed form would not be compact at this tolerance
                    }
                }
            }
            // rhs update (never dropped: it carries the excitation).
            let rhs_val_k = rhs_val[k];
            rhs_val[i] -= f_val * rhs_val_k;
            let prod = ctx.mul(f_sym, rhs_sym[k]);
            rhs_sym[i] = if ctx.is_zero(rhs_sym[i]) {
                ctx.neg(prod)
            } else {
                ctx.sub(rhs_sym[i], prod)
            };
            val[i][k] = Complex64::new(0.0, 0.0);
            sym[i][k] = zero;
        }
    }
    // back substitution for the full solution (we need x[out]).
    let mut x_sym = vec![zero; n];
    let mut x_val = vec![Complex64::new(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut s_sym = rhs_sym[i];
        let mut s_val = rhs_val[i];
        for j in (i + 1)..n {
            if val[i][j].norm() == 0.0 {
                continue;
            }
            let prod = ctx.mul(sym[i][j], x_sym[j]);
            s_sym = ctx.sub(s_sym, prod);
            s_val -= val[i][j] * x_val[j];
        }
        x_sym[i] = ctx.div(s_sym, sym[i][i]);
        x_val[i] = s_val / val[i][i];
    }
    // return the numeric value alongside, so the caller can self-check without
    // re-evaluating the (heavily shared) symbolic DAG.
    Some((x_sym[out], x_val[out]))
}

/// Generate the most compact symbolic transfer within the error budget: run the
/// symbolic LU with a sequence of drop tolerances from aggressive down to exact
/// and keep the first (most aggressive, so most compact) whose reduced `H` stays
/// within `tol` of `h0`. Exact elimination (drop `0`) is the guaranteed fallback,
/// since the entry pruning already holds the reduced transfer within `tol`.
/// `eval_env` evaluates the resulting (named or expanded) expression for the check.
#[allow(clippy::too_many_arguments)]
fn generate_compact(
    ctx: &mut Graph,
    ra: &[Vec<rsdag::ExprId>],
    vra: &[Vec<num_complex::Complex64>],
    rb_sym: &[rsdag::ExprId],
    rb_val: &[num_complex::Complex64],
    col: usize,
    h0: num_complex::Complex64,
    tol: f64,
) -> Option<(rsdag::ExprId, num_complex::Complex64)> {
    // Upper bound on the generated expression size; the symbolic elimination aborts
    // as soon as it would exceed it, so a tolerance that cannot yield a compact
    // closed form is skipped cheaply instead of building a huge expression.
    const OP_BUDGET: usize = 4_000;
    const FACTORS: [f64; 9] = [1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-6, 0.0];
    for &f in &FACTORS {
        let drop = tol * f;
        if let Some((h, hv)) = symbolic_lu_solve(
            ctx,
            ra.to_vec(),
            vra.to_vec(),
            rb_sym.to_vec(),
            rb_val.to_vec(),
            col,
            drop,
            OP_BUDGET,
        ) {
            // self-check uses the numeric value the solve already computed (no DAG eval).
            if hv.norm().is_finite() && (hv - h0).norm() / h0.norm() <= tol {
                return Some((h, hv));
            }
        }
    }
    None
}

/// Collect every free symbol id appearing in the DAG rooted at `id`.
fn collect_symbols(
    ctx: &Graph,
    id: rsdag::ExprId,
    seen: &mut std::collections::HashSet<rsdag::ExprId>,
    out: &mut std::collections::HashSet<SymbolId>,
) {
    if !seen.insert(id) {
        return;
    }
    match ctx.node(id) {
        Node::Symbol(s) => {
            out.insert(*s);
        }
        Node::Const(_) => {}
        Node::Add(a, b) | Node::Mul(a, b) | Node::Cmp(_, a, b) | Node::Binary(_, a, b) => {
            collect_symbols(ctx, *a, seen, out);
            collect_symbols(ctx, *b, seen, out);
        }
        Node::Neg(a) | Node::Pow(a, _) | Node::Unary(_, a) => {
            collect_symbols(ctx, *a, seen, out);
        }
        Node::Select(c, t, e) => {
            collect_symbols(ctx, *c, seen, out);
            collect_symbols(ctx, *t, seen, out);
            collect_symbols(ctx, *e, seen, out);
        }
        Node::Reduce(_, l) | Node::Call(_, l) | Node::Dot(l) | Node::Solve(l, _) => {
            for &c in ctx.args(*l) {
                collect_symbols(ctx, c, seen, out);
            }
        }
    }
}

#[cfg(test)]
mod templating_tests {
    //! Gate for the stamp-template Jacobian: it must match the direct
    //! (differentiate-the-residual) Jacobian numerically on every fixture, at a
    //! generic state and at the operating point.
    use super::*;
    use rsdag::ExprId;
    use std::collections::HashMap;

    fn dense_rect(
        ctx: &Graph,
        env: &HashMap<SymbolId, f64>,
        nrows: usize,
        ncols: usize,
        coo: &(Vec<usize>, Vec<usize>, Vec<ExprId>),
    ) -> Vec<Vec<f64>> {
        let vals = rsdag::eval(ctx, &coo.2, env);
        let mut m = vec![vec![0.0; ncols]; nrows];
        for k in 0..coo.0.len() {
            m[coo.0[k]][coo.1[k]] = vals[k];
        }
        m
    }

    fn sym_id(ctx: &mut Graph, name: &str) -> Option<SymbolId> {
        let e = ctx.sym(name);
        match ctx.node(e) {
            Node::Symbol(s) => Some(*s),
            _ => None,
        }
    }

    fn check_fixture(deck: &str, label: &str) {
        let parsed = parse(deck).unwrap_or_else(|e| panic!("{label}: parse {e}"));
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        let n = dae.dim();
        let cdc = CompiledDc::new(&mut ctx, &dae);
        let pnames: Vec<String> = cdc.param_names(&ctx);
        let psyms: Vec<SymbolId> = pnames
            .iter()
            .filter_map(|nm| sym_id(&mut ctx, nm))
            .collect();

        // dF/dx, dF/dx' and dF/dp -- templates vs. direct differentiation.
        let blocks: [(
            (Vec<usize>, Vec<usize>, Vec<ExprId>),
            (Vec<usize>, Vec<usize>, Vec<ExprId>),
            usize,
            &str,
        ); 3] = [
            (
                dae.jacobian_x_coo(&mut ctx),
                dae.jacobian_x_coo_templated(&mut ctx),
                n,
                "dF/dx",
            ),
            (
                dae.jacobian_xdot_coo(&mut ctx),
                dae.jacobian_xdot_coo_templated(&mut ctx),
                n,
                "dF/dx'",
            ),
            (
                dae.jacobian_p_coo(&mut ctx, &psyms),
                dae.jacobian_p_coo_templated(&mut ctx, &psyms),
                psyms.len(),
                "dF/dp",
            ),
        ];

        // A generic (finite, non-degenerate) state + the netlist parameter values.
        let mut env: HashMap<SymbolId, f64> = HashMap::new();
        for (i, &s) in dae.x.iter().enumerate() {
            env.insert(s, 0.3 * ((i as f64 + 1.0) * 0.7).sin());
        }
        for o in dae.xdot.iter().flatten() {
            env.insert(*o, 0.11);
        }
        for nm in &pnames {
            if let Some(s) = sym_id(&mut ctx, nm) {
                env.insert(s, parsed.param_value(nm).unwrap_or(0.0));
            }
        }
        env.insert(dae.t, 0.0);

        for (direct, templated, ncol, name) in &blocks {
            if *ncol == 0 {
                continue;
            }
            let a = dense_rect(&ctx, &env, n, *ncol, direct);
            let b = dense_rect(&ctx, &env, n, *ncol, templated);
            let mut maxdiff = 0.0f64;
            for i in 0..n {
                for j in 0..*ncol {
                    maxdiff = maxdiff.max((a[i][j] - b[i][j]).abs());
                }
            }
            assert!(
                maxdiff < 1e-9,
                "{label} [{name}]: templated differs from direct by {maxdiff:.3e}"
            );
        }
    }

    #[test]
    fn templated_matches_direct_on_transformed_dae() {
        // A reduced DAE (here via internal-resistive-node elimination) must still
        // carry valid stamps, so its templated Jacobian matches the direct one.
        use sane_dae::eliminate_nodes;
        let deck = "V1 in 0 1\nR1 in mid 1k\nR2 mid out 2k\nR3 out 0 3k\nC1 out 0 1u\n";
        let parsed = parse(deck).unwrap();
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        let keep: std::collections::HashSet<String> =
            ["v1", "v3"].iter().map(|s| s.to_string()).collect();
        let (reduced, eliminated) = eliminate_nodes(&mut ctx, &dae, &keep);
        assert!(
            !eliminated.is_empty(),
            "expected an eliminable internal node"
        );
        assert!(!reduced.stamps.is_empty(), "reduced DAE must carry stamps");
        let n = reduced.dim();
        let direct = reduced.jacobian_x_coo(&mut ctx);
        let templated = reduced.jacobian_x_coo_templated(&mut ctx);
        let mut env: HashMap<SymbolId, f64> = HashMap::new();
        for (i, &s) in reduced.x.iter().enumerate() {
            env.insert(s, 0.2 * ((i as f64 + 1.0) * 0.9).cos());
        }
        for o in reduced.xdot.iter().flatten() {
            env.insert(*o, 0.07);
        }
        for nm in CompiledDc::new(&mut ctx, &reduced).param_names(&ctx) {
            if let Some(s) = sym_id(&mut ctx, &nm) {
                env.insert(s, parsed.param_value(&nm).unwrap_or(0.0));
            }
        }
        env.insert(reduced.t, 0.0);
        let a = dense_rect(&ctx, &env, n, n, &direct);
        let b = dense_rect(&ctx, &env, n, n, &templated);
        let mut maxdiff = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                maxdiff = maxdiff.max((a[i][j] - b[i][j]).abs());
            }
        }
        assert!(
            maxdiff < 1e-9,
            "reduced DAE: templated differs by {maxdiff:.3e}"
        );
    }

    #[test]
    fn templated_jacobian_matches_direct_on_all_fixtures() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../netlist/tests/fixtures");
        let mut n = 0;
        for entry in std::fs::read_dir(dir).expect("fixtures dir") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("cir") {
                continue;
            }
            let deck = std::fs::read_to_string(&path).unwrap();
            let label = path.file_name().unwrap().to_string_lossy().into_owned();
            check_fixture(&deck, &label);
            n += 1;
        }
        assert!(n >= 19, "expected the full fixture suite, ran {n}");
    }
}
