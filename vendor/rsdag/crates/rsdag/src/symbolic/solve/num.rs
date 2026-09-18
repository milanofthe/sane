//! The scalar of a solve: a real expression, or a complex one as a pair
//! of real expressions. The eliminations are written over [`Num`], so a
//! complex system lowers to real ops at build time (four real dots per
//! complex dot, a real embedding of twice the size for a dense pivot
//! block) and its program runs wherever a real program runs, with the
//! pivot guard comparing moduli.

use crate::field::Field;
use crate::graph::Graph;
use crate::node::{CmpOp, ExprId, UnaryOp};

use super::PIVOT_TOLERANCE;

/// The arithmetic a solve needs from its scalar.
pub trait Num: Copy + std::fmt::Debug {
    fn zero<K: Field>(g: &mut Graph<K>) -> Self;
    fn add<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self;
    fn sub<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self;
    fn mul<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self;
    fn neg<K: Field>(g: &mut Graph<K>, a: Self) -> Self;
    fn recip<K: Field>(g: &mut Graph<K>, a: Self) -> Self;
    fn div<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self;
    /// The dot of two lists.
    fn dot<K: Field>(g: &mut Graph<K>, a: Vec<Self>, b: Vec<Self>) -> Self;
    /// The size a pivot is compared by: the absolute value of a real, the
    /// squared modulus of a complex expression.
    fn size<K: Field>(g: &mut Graph<K>, a: Self) -> ExprId;
    /// [`PIVOT_TOLERANCE`] in the unit of [`size`](Self::size).
    fn tolerance<K: Field>(g: &mut Graph<K>) -> ExprId;
    /// The pivot guard: `size(pivot) >= tolerance * largest`.
    fn guard<K: Field>(g: &mut Graph<K>, pivot: Self, largest: ExprId) -> ExprId {
        let pa = Self::size(g, pivot);
        let tol = Self::tolerance(g);
        let bound = g.mul(tol, largest);
        g.cmp(CmpOp::Ge, pa, bound)
    }
    /// The columns of the inverse of the dense `n` by `n` matrix `a`
    /// (row-major), through the dense solve kernel.
    fn inverse_columns<K: Field>(g: &mut Graph<K>, a: &[Self], n: usize) -> Vec<Vec<Self>>;
}

impl Num for ExprId {
    fn zero<K: Field>(g: &mut Graph<K>) -> Self {
        g.zero()
    }
    fn add<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self {
        g.add(a, b)
    }
    fn sub<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self {
        g.sub(a, b)
    }
    fn mul<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self {
        g.mul(a, b)
    }
    fn neg<K: Field>(g: &mut Graph<K>, a: Self) -> Self {
        g.neg(a)
    }
    fn recip<K: Field>(g: &mut Graph<K>, a: Self) -> Self {
        g.recip(a)
    }
    fn div<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self {
        g.div(a, b)
    }
    fn dot<K: Field>(g: &mut Graph<K>, a: Vec<Self>, b: Vec<Self>) -> Self {
        g.dot(a, b)
    }
    fn size<K: Field>(g: &mut Graph<K>, a: Self) -> ExprId {
        g.unary(UnaryOp::Abs, a)
    }
    fn tolerance<K: Field>(g: &mut Graph<K>) -> ExprId {
        g.konst_f64(PIVOT_TOLERANCE)
    }
    fn inverse_columns<K: Field>(g: &mut Graph<K>, a: &[Self], n: usize) -> Vec<Vec<Self>> {
        let (zero, one) = (g.zero(), g.one());
        (0..n)
            .map(|c| {
                let e: Vec<ExprId> = (0..n).map(|q| if q == c { one } else { zero }).collect();
                g.solve_dense(a.to_vec(), e)
            })
            .collect()
    }
}

/// A complex expression as its real and imaginary parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cx {
    pub re: ExprId,
    pub im: ExprId,
}

impl Cx {
    pub fn new(re: ExprId, im: ExprId) -> Self {
        Cx { re, im }
    }

    /// A real expression as a complex one.
    pub fn real<K: Field>(g: &mut Graph<K>, re: ExprId) -> Self {
        Cx { re, im: g.zero() }
    }
}

impl Num for Cx {
    fn zero<K: Field>(g: &mut Graph<K>) -> Self {
        let z = g.zero();
        Cx { re: z, im: z }
    }
    fn add<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self {
        Cx {
            re: g.add(a.re, b.re),
            im: g.add(a.im, b.im),
        }
    }
    fn sub<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self {
        Cx {
            re: g.sub(a.re, b.re),
            im: g.sub(a.im, b.im),
        }
    }
    fn mul<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self {
        let rr = g.mul(a.re, b.re);
        let ii = g.mul(a.im, b.im);
        let ri = g.mul(a.re, b.im);
        let ir = g.mul(a.im, b.re);
        Cx {
            re: g.sub(rr, ii),
            im: g.add(ri, ir),
        }
    }
    fn neg<K: Field>(g: &mut Graph<K>, a: Self) -> Self {
        Cx {
            re: g.neg(a.re),
            im: g.neg(a.im),
        }
    }
    fn recip<K: Field>(g: &mut Graph<K>, a: Self) -> Self {
        let d = Self::size(g, a);
        let re = g.div(a.re, d);
        let ni = g.neg(a.im);
        Cx {
            re,
            im: g.div(ni, d),
        }
    }
    fn div<K: Field>(g: &mut Graph<K>, a: Self, b: Self) -> Self {
        // (a conj(b)) / |b|^2
        let d = Self::size(g, b);
        let rr = g.mul(a.re, b.re);
        let ii = g.mul(a.im, b.im);
        let ir = g.mul(a.im, b.re);
        let ri = g.mul(a.re, b.im);
        let nre = g.add(rr, ii);
        let nim = g.sub(ir, ri);
        Cx {
            re: g.div(nre, d),
            im: g.div(nim, d),
        }
    }
    fn dot<K: Field>(g: &mut Graph<K>, a: Vec<Self>, b: Vec<Self>) -> Self {
        let ar: Vec<ExprId> = a.iter().map(|c| c.re).collect();
        let ai: Vec<ExprId> = a.iter().map(|c| c.im).collect();
        let br: Vec<ExprId> = b.iter().map(|c| c.re).collect();
        let bi: Vec<ExprId> = b.iter().map(|c| c.im).collect();
        let rr = g.dot(ar.clone(), br.clone());
        let ii = g.dot(ai.clone(), bi.clone());
        let ri = g.dot(ar, bi);
        let ir = g.dot(ai, br);
        Cx {
            re: g.sub(rr, ii),
            im: g.add(ri, ir),
        }
    }
    fn size<K: Field>(g: &mut Graph<K>, a: Self) -> ExprId {
        let rr = g.mul(a.re, a.re);
        let ii = g.mul(a.im, a.im);
        g.add(rr, ii)
    }
    fn tolerance<K: Field>(g: &mut Graph<K>) -> ExprId {
        g.konst_f64(PIVOT_TOLERANCE * PIVOT_TOLERANCE)
    }
    /// The inverse through the real embedding `[[Re, -Im], [Im, Re]]` of
    /// twice the size: its first `n` columns are the real and imaginary
    /// parts of the inverse's columns.
    fn inverse_columns<K: Field>(g: &mut Graph<K>, a: &[Self], n: usize) -> Vec<Vec<Self>> {
        let m = 2 * n;
        let mut e = vec![g.zero(); m * m];
        for r in 0..n {
            for c in 0..n {
                let v = a[r * n + c];
                e[r * m + c] = v.re;
                e[r * m + n + c] = g.neg(v.im);
                e[(n + r) * m + c] = v.im;
                e[(n + r) * m + n + c] = v.re;
            }
        }
        let (zero, one) = (g.zero(), g.one());
        (0..n)
            .map(|c| {
                let unit: Vec<ExprId> = (0..m).map(|q| if q == c { one } else { zero }).collect();
                let sol = g.solve_dense(e.clone(), unit);
                (0..n).map(|q| Cx::new(sol[q], sol[n + q])).collect()
            })
            .collect()
    }
}
