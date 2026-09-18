//! Polynomials in one symbol over expression coefficients: collection,
//! rational canonical forms, term expansion and numeric pruning (the
//! term-pruning model reduction idea of the symbolic-analysis literature).

use std::collections::HashMap;

use crate::eval::eval;
use crate::field::Field;
use crate::graph::Graph;
use crate::node::{ExprId, Node, ReduceOp, SymbolId};

/// `e` as a polynomial in `s`: coefficients by ascending power, each an
/// expression free of `s`. `None` if `s` appears under a non-polynomial op
/// (a transcendental, a comparison, a call, a negative power).
pub fn collect<K: Field>(g: &mut Graph<K>, e: ExprId, s: SymbolId) -> Option<Vec<ExprId>> {
    if !g.free_symbols(e).contains(&s) {
        return Some(vec![e]);
    }
    match *g.node(e) {
        Node::Symbol(_) => {
            let z = g.zero();
            let o = g.one();
            Some(vec![z, o])
        }
        Node::Add(a, b) => {
            let pa = collect(g, a, s)?;
            let pb = collect(g, b, s)?;
            Some(poly_add(g, pa, pb))
        }
        Node::Neg(a) => {
            let mut p = collect(g, a, s)?;
            for c in p.iter_mut() {
                *c = g.neg(*c);
            }
            Some(p)
        }
        Node::Mul(a, b) => {
            let pa = collect(g, a, s)?;
            let pb = collect(g, b, s)?;
            Some(poly_mul(g, &pa, &pb))
        }
        Node::Pow(a, n) => {
            if n < 0 {
                return None;
            }
            let pa = collect(g, a, s)?;
            let mut acc = vec![g.one()];
            for _ in 0..n {
                acc = poly_mul(g, &acc, &pa);
            }
            Some(acc)
        }
        Node::Reduce(ReduceOp::Sum, l) => {
            let list = g.args(l).to_vec();
            let mut acc = vec![g.zero()];
            for it in list {
                let p = collect(g, it, s)?;
                acc = poly_add(g, acc, p);
            }
            Some(acc)
        }
        Node::Reduce(ReduceOp::Product, l) => {
            let list = g.args(l).to_vec();
            let mut acc = vec![g.one()];
            for it in list {
                let p = collect(g, it, s)?;
                acc = poly_mul(g, &acc, &p);
            }
            Some(acc)
        }
        Node::Dot(l) => {
            let (al, bl) = g.dot_args(l);
            let (al, bl) = (al.to_vec(), bl.to_vec());
            let mut acc = vec![g.zero()];
            for (a, b) in al.iter().zip(bl.iter()) {
                let pa = collect(g, *a, s)?;
                let pb = collect(g, *b, s)?;
                let prod = poly_mul(g, &pa, &pb);
                acc = poly_add(g, acc, prod);
            }
            Some(acc)
        }
        _ => None,
    }
}

/// `e` as a rational function `N(s) / D(s)` with polynomial numerator and
/// denominator: negative powers and products of them are gathered into the
/// denominator. `None` if `s` appears under a non-rational op.
pub fn rational_form<K: Field>(
    g: &mut Graph<K>,
    e: ExprId,
    s: SymbolId,
) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    if !g.free_symbols(e).contains(&s) {
        return Some((vec![e], vec![g.one()]));
    }
    match *g.node(e) {
        Node::Add(a, b) => {
            let (na, da) = rational_form(g, a, s)?;
            let (nb, db) = rational_form(g, b, s)?;
            // na/da + nb/db = (na db + nb da) / (da db)
            let t1 = poly_mul(g, &na, &db);
            let t2 = poly_mul(g, &nb, &da);
            Some((poly_add(g, t1, t2), poly_mul(g, &da, &db)))
        }
        Node::Neg(a) => {
            let (mut n, d) = rational_form(g, a, s)?;
            for c in n.iter_mut() {
                *c = g.neg(*c);
            }
            Some((n, d))
        }
        Node::Mul(a, b) => {
            let (na, da) = rational_form(g, a, s)?;
            let (nb, db) = rational_form(g, b, s)?;
            Some((poly_mul(g, &na, &nb), poly_mul(g, &da, &db)))
        }
        Node::Pow(a, n) => {
            let (na, da) = rational_form(g, a, s)?;
            let (base_n, base_d) = if n < 0 { (da, na) } else { (na, da) };
            let mut num = vec![g.one()];
            let mut den = vec![g.one()];
            for _ in 0..n.unsigned_abs() {
                num = poly_mul(g, &num, &base_n);
                den = poly_mul(g, &den, &base_d);
            }
            Some((num, den))
        }
        Node::Reduce(ReduceOp::Sum, l) => {
            let list = g.args(l).to_vec();
            let mut num = vec![g.zero()];
            let mut den = vec![g.one()];
            for it in list {
                let (ni, di) = rational_form(g, it, s)?;
                let t1 = poly_mul(g, &num, &di);
                let t2 = poly_mul(g, &ni, &den);
                num = poly_add(g, t1, t2);
                den = poly_mul(g, &den, &di);
            }
            Some((num, den))
        }
        Node::Reduce(ReduceOp::Product, l) => {
            let list = g.args(l).to_vec();
            let mut num = vec![g.one()];
            let mut den = vec![g.one()];
            for it in list {
                let (ni, di) = rational_form(g, it, s)?;
                num = poly_mul(g, &num, &ni);
                den = poly_mul(g, &den, &di);
            }
            Some((num, den))
        }
        Node::Dot(l) => {
            let (al, bl) = g.dot_args(l);
            let (al, bl) = (al.to_vec(), bl.to_vec());
            let mut num = vec![g.zero()];
            let mut den = vec![g.one()];
            for (a, b) in al.iter().zip(bl.iter()) {
                let (na, da) = rational_form(g, *a, s)?;
                let (nb, db) = rational_form(g, *b, s)?;
                let ni = poly_mul(g, &na, &nb);
                let di = poly_mul(g, &da, &db);
                let t1 = poly_mul(g, &num, &di);
                let t2 = poly_mul(g, &ni, &den);
                num = poly_add(g, t1, t2);
                den = poly_mul(g, &den, &di);
            }
            Some((num, den))
        }
        Node::Symbol(_) => {
            let z = g.zero();
            let o = g.one();
            Some((vec![z, o], vec![o]))
        }
        _ => None,
    }
}

/// Coefficient-wise sum.
pub fn poly_add<K: Field>(g: &mut Graph<K>, mut a: Vec<ExprId>, b: Vec<ExprId>) -> Vec<ExprId> {
    if b.len() > a.len() {
        let z = g.zero();
        a.resize(b.len(), z);
    }
    for (i, &bc) in b.iter().enumerate() {
        a[i] = g.add(a[i], bc);
    }
    a
}

/// Convolution product.
pub fn poly_mul<K: Field>(g: &mut Graph<K>, a: &[ExprId], b: &[ExprId]) -> Vec<ExprId> {
    let z = g.zero();
    if a.is_empty() || b.is_empty() {
        return vec![z];
    }
    let mut out = vec![z; a.len() + b.len() - 1];
    for (i, &ac) in a.iter().enumerate() {
        for (j, &bc) in b.iter().enumerate() {
            let p = g.mul(ac, bc);
            out[i + j] = g.add(out[i + j], p);
        }
    }
    out
}

/// The polynomial `coeffs` in `s_e` as one expression (Horner-free, so the
/// powers stay visible).
pub fn poly_to_expr<K: Field>(g: &mut Graph<K>, coeffs: &[ExprId], s_e: ExprId) -> ExprId {
    let mut acc = g.zero();
    let mut spow = g.one();
    for &c in coeffs {
        let term = g.mul(c, spow);
        acc = g.add(acc, term);
        spow = g.mul(spow, s_e);
    }
    acc
}

/// Expand a coefficient into its additive terms (products distributed,
/// small powers multiplied out).
pub fn expand_terms<K: Field>(g: &mut Graph<K>, e: ExprId) -> Vec<ExprId> {
    match *g.node(e) {
        Node::Add(a, b) => {
            let mut t = expand_terms(g, a);
            t.extend(expand_terms(g, b));
            t
        }
        Node::Reduce(ReduceOp::Sum, l) => g
            .args(l)
            .to_vec()
            .into_iter()
            .flat_map(|it| expand_terms(g, it))
            .collect(),
        Node::Neg(a) => expand_terms(g, a).into_iter().map(|t| g.neg(t)).collect(),
        Node::Mul(a, b) => {
            let ta = expand_terms(g, a);
            let tb = expand_terms(g, b);
            let mut out = Vec::with_capacity(ta.len() * tb.len());
            for &x in &ta {
                for &y in &tb {
                    out.push(g.mul(x, y));
                }
            }
            out
        }
        Node::Reduce(ReduceOp::Product, l) => {
            let list = g.args(l).to_vec();
            let mut acc = vec![g.one()];
            for it in list {
                let ti = expand_terms(g, it);
                let mut next = Vec::with_capacity(acc.len() * ti.len());
                for &x in &acc {
                    for &y in &ti {
                        next.push(g.mul(x, y));
                    }
                }
                acc = next;
            }
            acc
        }
        Node::Pow(a, n) if (1..=4).contains(&n) => {
            let base = expand_terms(g, a);
            let mut acc = vec![g.one()];
            for _ in 0..n {
                let mut next = Vec::with_capacity(acc.len() * base.len());
                for &x in &acc {
                    for &y in &base {
                        next.push(g.mul(x, y));
                    }
                }
                acc = next;
            }
            acc
        }
        _ => vec![e],
    }
}

/// Drop the terms of a polynomial's coefficients whose numeric contribution
/// at `w0` (`|term| * w0^k`) is below `tol` times the largest. Returns the
/// pruned coefficients and `(total terms, kept terms)`.
pub fn prune_poly<K: Field>(
    g: &mut Graph<K>,
    poly: &[ExprId],
    env: &HashMap<SymbolId, f64>,
    w0: f64,
    tol: f64,
) -> (Vec<ExprId>, usize, usize) {
    let mut terms: Vec<(usize, ExprId, f64)> = Vec::new();
    for (k, &coeff) in poly.iter().enumerate() {
        for t in expand_terms(g, coeff) {
            let v = eval(g, &[t], env)[0];
            terms.push((k, t, v.abs() * w0.powi(k as i32)));
        }
    }
    let maxc = terms
        .iter()
        .map(|(_, _, c)| *c)
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    let total = terms.len();
    let z = g.zero();
    let mut coeffs = vec![z; poly.len()];
    let mut kept = 0;
    for (k, t, c) in &terms {
        if *c >= tol * maxc {
            coeffs[*k] = g.add(coeffs[*k], *t);
            kept += 1;
        }
    }
    (coeffs, total, kept)
}
