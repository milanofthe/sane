//! Symbolic polynomial-in-`s` machinery for term-pruning model reduction
//! (the Analog-Insydes SBG/SAG idea): collect a transfer expression as a
//! polynomial in `s`, expand coefficients into monomials, and prune terms by
//! their numeric contribution at a reference frequency.

use num_complex::Complex64;
use std::collections::HashMap;

use rsdag::{ExprId, Graph, Node, ReduceOp, SymbolId};

/// Collect an expression as a polynomial in `s` (coefficient ExprId per power),
/// or None if it is not polynomial in `s`. Subtrees free of `s` are degree-0.
pub(crate) fn poly_in_s(ctx: &mut Graph, e: ExprId, s: SymbolId) -> Option<Vec<ExprId>> {
    if !ctx.free_symbols(e).contains(&s) {
        return Some(vec![e]);
    }
    match *ctx.node(e) {
        Node::Symbol(_) => {
            let z = ctx.zero();
            let o = ctx.one();
            Some(vec![z, o])
        }
        Node::Add(a, b) => {
            let pa = poly_in_s(ctx, a, s)?;
            let pb = poly_in_s(ctx, b, s)?;
            Some(poly_add(ctx, pa, pb))
        }
        Node::Neg(a) => {
            let mut p = poly_in_s(ctx, a, s)?;
            for c in p.iter_mut() {
                *c = ctx.neg(*c);
            }
            Some(p)
        }
        Node::Mul(a, b) => {
            let pa = poly_in_s(ctx, a, s)?;
            let pb = poly_in_s(ctx, b, s)?;
            Some(poly_mul(ctx, &pa, &pb))
        }
        Node::Pow(a, n) => {
            if n < 0 {
                return None;
            }
            let pa = poly_in_s(ctx, a, s)?;
            let mut acc = vec![ctx.one()];
            for _ in 0..n {
                acc = poly_mul(ctx, &acc, &pa);
            }
            Some(acc)
        }
        Node::Reduce(ReduceOp::Sum, l) => {
            let list = ctx.args(l).to_vec();
            let mut acc = vec![ctx.zero()];
            for it in list {
                let p = poly_in_s(ctx, it, s)?;
                acc = poly_add(ctx, acc, p);
            }
            Some(acc)
        }
        Node::Reduce(ReduceOp::Product, l) => {
            let list = ctx.args(l).to_vec();
            let mut acc = vec![ctx.one()];
            for it in list {
                let p = poly_in_s(ctx, it, s)?;
                acc = poly_mul(ctx, &acc, &p);
            }
            Some(acc)
        }
        Node::Dot(l) => {
            let (al, bl) = ctx.dot_args(l);
            let (al, bl) = (al.to_vec(), bl.to_vec());
            let mut acc = vec![ctx.zero()];
            for (a, b) in al.iter().zip(bl.iter()) {
                let pa = poly_in_s(ctx, *a, s)?;
                let pb = poly_in_s(ctx, *b, s)?;
                let prod = poly_mul(ctx, &pa, &pb);
                acc = poly_add(ctx, acc, prod);
            }
            Some(acc)
        }
        _ => None, // Unary/Cmp/Select/Opaque/Min/Max containing s -> not polynomial
    }
}

pub(crate) fn poly_add(ctx: &mut Graph, mut a: Vec<ExprId>, b: Vec<ExprId>) -> Vec<ExprId> {
    if b.len() > a.len() {
        let z = ctx.zero();
        a.resize(b.len(), z);
    }
    for (i, &bc) in b.iter().enumerate() {
        a[i] = ctx.add(a[i], bc);
    }
    a
}

pub(crate) fn poly_mul(ctx: &mut Graph, a: &[ExprId], b: &[ExprId]) -> Vec<ExprId> {
    let z = ctx.zero();
    let mut out = vec![z; a.len() + b.len() - 1];
    for (i, &ac) in a.iter().enumerate() {
        for (j, &bc) in b.iter().enumerate() {
            let p = ctx.mul(ac, bc);
            out[i + j] = ctx.add(out[i + j], p);
        }
    }
    out
}

/// Expand a coefficient into its additive monomial terms by distributing
/// products over sums (the engine keeps expressions factored; term-pruning needs
/// the flat sum-of-products form). Each returned ExprId is a single product term.
pub(crate) fn expand_terms(ctx: &mut Graph, e: ExprId) -> Vec<ExprId> {
    match *ctx.node(e) {
        Node::Add(a, b) => {
            let mut t = expand_terms(ctx, a);
            t.extend(expand_terms(ctx, b));
            t
        }
        Node::Reduce(ReduceOp::Sum, l) => ctx
            .args(l)
            .to_vec()
            .into_iter()
            .flat_map(|it| expand_terms(ctx, it))
            .collect(),
        Node::Neg(a) => expand_terms(ctx, a)
            .into_iter()
            .map(|t| ctx.neg(t))
            .collect(),
        Node::Mul(a, b) => {
            let ta = expand_terms(ctx, a);
            let tb = expand_terms(ctx, b);
            let mut out = Vec::with_capacity(ta.len() * tb.len());
            for &x in &ta {
                for &y in &tb {
                    out.push(ctx.mul(x, y));
                }
            }
            out
        }
        Node::Reduce(ReduceOp::Product, l) => {
            let list = ctx.args(l).to_vec();
            let mut acc = vec![ctx.one()];
            for it in list {
                let ti = expand_terms(ctx, it);
                let mut next = Vec::with_capacity(acc.len() * ti.len());
                for &x in &acc {
                    for &y in &ti {
                        next.push(ctx.mul(x, y));
                    }
                }
                acc = next;
            }
            acc
        }
        // Small integer powers expand to products; larger ones stay atomic to
        // avoid blow-up (still correct, just coarser ranking).
        Node::Pow(a, n) if (1..=4).contains(&n) => {
            let base = expand_terms(ctx, a);
            let mut acc = vec![ctx.one()];
            for _ in 0..n {
                let mut next = Vec::with_capacity(acc.len() * base.len());
                for &x in &acc {
                    for &y in &base {
                        next.push(ctx.mul(x, y));
                    }
                }
                acc = next;
            }
            acc
        }
        _ => vec![e],
    }
}

/// Prune a polynomial's terms by their contribution `|value| * w0^k` to the
/// polynomial value at the reference frequency, keeping those >= tol*max.
pub(crate) fn prune_poly(
    ctx: &mut Graph,
    poly: &[ExprId],
    env: &HashMap<SymbolId, f64>,
    w0: f64,
    tol: f64,
) -> (Vec<ExprId>, usize, usize) {
    let mut terms: Vec<(usize, ExprId, f64)> = Vec::new();
    for (k, &coeff) in poly.iter().enumerate() {
        for t in expand_terms(ctx, coeff) {
            let v = rsdag::eval(ctx, &[t], env)[0];
            terms.push((k, t, v.abs() * w0.powi(k as i32)));
        }
    }
    let maxc = terms
        .iter()
        .map(|(_, _, c)| *c)
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    let total = terms.len();
    let z = ctx.zero();
    let mut coeffs = vec![z; poly.len()];
    let mut kept = 0;
    for (k, t, c) in &terms {
        if *c >= tol * maxc {
            coeffs[*k] = ctx.add(coeffs[*k], *t);
            kept += 1;
        }
    }
    (coeffs, total, kept)
}

/// Build the expression `Σ_k coeffs[k] * s^k`.
pub(crate) fn build_poly_expr(ctx: &mut Graph, coeffs: &[ExprId], s_e: ExprId) -> ExprId {
    let mut acc = ctx.zero();
    let mut spow = ctx.one();
    for &c in coeffs {
        let term = ctx.mul(c, spow);
        acc = ctx.add(acc, term);
        spow = ctx.mul(spow, s_e);
    }
    acc
}

// Retained for the in-crate symbolic-reduction test (`pz_tests`), which
// reconstructs the full-vs-pruned band response to check `symbolic_transfer_approx`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn eval_coeffs(ctx: &Graph, env: &HashMap<SymbolId, f64>, poly: &[ExprId]) -> Vec<f64> {
    // One arena sweep over all coefficients rather than one per coefficient.
    rsdag::eval(ctx, poly, env)
}

/// Evaluate the real-coefficient polynomial at `jw`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn poly_eval_c(coeffs: &[f64], jw: Complex64) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    let mut p = Complex64::new(1.0, 0.0);
    for &c in coeffs {
        acc += Complex64::new(c, 0.0) * p;
        p *= jw;
    }
    acc
}
