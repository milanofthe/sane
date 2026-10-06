//! Polynomials in one symbol over expression coefficients: collection,
//! rational canonical forms, term expansion and numeric pruning (the
//! term-pruning model reduction idea of the symbolic-analysis literature).

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::eval::eval;
use crate::field::Field;
use crate::graph::Graph;
use crate::node::{ExprId, Node, ReduceOp, SymbolId};

/// `e` folded bottom-up over the nodes that depend on `s`: `free(g, c)` is
/// the value of an operand free of `s`, `node(g, n, vals)` the value of a
/// node from its operands' values (in operand order), `None` when the node
/// is outside the algebra. One ascending sweep, each node once however
/// often it is shared.
fn fold_in<K: Field, T: Clone>(
    g: &mut Graph<K>,
    e: ExprId,
    s: SymbolId,
    free: impl Fn(&mut Graph<K>, ExprId) -> T,
    mut node: impl FnMut(&mut Graph<K>, Node, &[T]) -> Option<T>,
) -> Option<T> {
    let cone = g.cone_sorted(&[e], false);
    // The values of the nodes that depend on `s`; `None` marks one outside
    // the algebra, which every node above it inherits.
    let mut vals: FxHashMap<ExprId, Option<T>> = FxHashMap::default();
    let mut args: Vec<T> = Vec::new();
    for &x in &cone {
        let n = *g.node(x);
        let depends = match n {
            Node::Symbol(t) => t == s,
            _ => g.operands(x).iter().any(|c| vals.contains_key(c)),
        };
        if !depends {
            continue;
        }
        let ops = g.operands(x).to_vec();
        args.clear();
        let mut ok = true;
        for c in ops {
            match vals.get(&c) {
                Some(Some(v)) => args.push(v.clone()),
                Some(None) => ok = false,
                None => args.push(free(g, c)),
            }
        }
        let v = if ok { node(g, n, &args) } else { None };
        vals.insert(x, v);
    }
    match vals.remove(&e) {
        Some(v) => v,
        None => Some(free(g, e)),
    }
}

/// `e` as a polynomial in `s`: coefficients by ascending power, each an
/// expression free of `s`. `None` if `s` appears under a non-polynomial op
/// (a transcendental, a comparison, a call, a negative power).
pub fn collect<K: Field>(g: &mut Graph<K>, e: ExprId, s: SymbolId) -> Option<Vec<ExprId>> {
    fold_in(
        g,
        e,
        s,
        |_, c| vec![c],
        |g, n, p| match n {
            Node::Symbol(_) => Some(vec![g.zero(), g.one()]),
            Node::Add(..) => Some(poly_add(g, p[0].clone(), p[1].clone())),
            Node::Neg(_) => Some(p[0].iter().map(|&c| g.neg(c)).collect()),
            Node::Mul(..) => Some(poly_mul(g, &p[0], &p[1])),
            Node::Pow(_, k) if k >= 0 => {
                let mut acc = vec![g.one()];
                for _ in 0..k {
                    acc = poly_mul(g, &acc, &p[0]);
                }
                Some(acc)
            }
            Node::Reduce(ReduceOp::Sum, _) => {
                let z = vec![g.zero()];
                Some(p.iter().fold(z, |acc, q| poly_add(g, acc, q.clone())))
            }
            Node::Reduce(ReduceOp::Product, _) => {
                let o = vec![g.one()];
                Some(p.iter().fold(o, |acc, q| poly_mul(g, &acc, q)))
            }
            Node::Dot(_) => {
                let (a, b) = p.split_at(p.len() / 2);
                let mut acc = vec![g.zero()];
                for (pa, pb) in a.iter().zip(b) {
                    let prod = poly_mul(g, pa, pb);
                    acc = poly_add(g, acc, prod);
                }
                Some(acc)
            }
            _ => None,
        },
    )
}

/// `e` as a rational function `N(s) / D(s)` with polynomial numerator and
/// denominator: negative powers and products of them are gathered into the
/// denominator. `None` if `s` appears under a non-rational op.
pub fn rational_form<K: Field>(
    g: &mut Graph<K>,
    e: ExprId,
    s: SymbolId,
) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    type Ratio = (Vec<ExprId>, Vec<ExprId>);
    // `n1/d1 + n2/d2 = (n1 d2 + n2 d1) / (d1 d2)`.
    fn add<K: Field>(g: &mut Graph<K>, (na, da): &Ratio, (nb, db): &Ratio) -> Ratio {
        let t1 = poly_mul(g, na, db);
        let t2 = poly_mul(g, nb, da);
        (poly_add(g, t1, t2), poly_mul(g, da, db))
    }
    fn mul<K: Field>(g: &mut Graph<K>, (na, da): &Ratio, (nb, db): &Ratio) -> Ratio {
        (poly_mul(g, na, nb), poly_mul(g, da, db))
    }
    let one = |g: &mut Graph<K>| -> Ratio { (vec![g.one()], vec![g.one()]) };
    fold_in(
        g,
        e,
        s,
        |g, c| (vec![c], vec![g.one()]),
        |g, n, p| match n {
            Node::Symbol(_) => Some((vec![g.zero(), g.one()], vec![g.one()])),
            Node::Add(..) => Some(add(g, &p[0], &p[1])),
            Node::Neg(_) => Some((p[0].0.iter().map(|&c| g.neg(c)).collect(), p[0].1.clone())),
            Node::Mul(..) => Some(mul(g, &p[0], &p[1])),
            Node::Pow(_, k) => {
                let base = if k < 0 {
                    (p[0].1.clone(), p[0].0.clone())
                } else {
                    p[0].clone()
                };
                let mut acc = one(g);
                for _ in 0..k.unsigned_abs() {
                    acc = mul(g, &acc, &base);
                }
                Some(acc)
            }
            Node::Reduce(ReduceOp::Sum, _) => {
                let mut acc = (vec![g.zero()], vec![g.one()]);
                for q in p {
                    acc = add(g, &acc, q);
                }
                Some(acc)
            }
            Node::Reduce(ReduceOp::Product, _) => {
                let mut acc = one(g);
                for q in p {
                    acc = mul(g, &acc, q);
                }
                Some(acc)
            }
            Node::Dot(_) => {
                let (a, b) = p.split_at(p.len() / 2);
                let mut acc = (vec![g.zero()], vec![g.one()]);
                for (pa, pb) in a.iter().zip(b) {
                    let prod = mul(g, pa, pb);
                    acc = add(g, &acc, &prod);
                }
                Some(acc)
            }
            _ => None,
        },
    )
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
        terms.extend(expand_terms(g, coeff).into_iter().map(|t| (k, t, 0.0)));
    }
    // Every term's value in one sweep over the graph.
    let roots: Vec<ExprId> = terms.iter().map(|&(_, t, _)| t).collect();
    for ((k, _, c), v) in terms.iter_mut().zip(eval(g, &roots, env)) {
        *c = v.abs() * w0.powi(*k as i32);
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
