//! How an expression depends on a chosen set of variables: its exact
//! polynomial degree, and what breaks polynomial structure.
//!
//! This is the graph fact a black-box harmonic-balance engine cannot read. A
//! degree-`d` nonlinearity of a signal with `k` harmonics generates
//! harmonics up to `d * k`, so the alias-free sample count `2 d k + 1` is
//! exact rather than a heuristic. A transcendental, rational, piecewise or
//! opaque part has no finite bandwidth; it is flagged so a solver
//! oversamples and says why.

use std::collections::BTreeSet;

use rustc_hash::FxHashMap as HashMap;

use crate::field::Field;
use crate::graph::Graph;
use crate::node::{ExprId, Node, ReduceOp, SymbolId, UnaryOp};

/// Polynomial degree in the variables.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Degree {
    /// A polynomial of this degree; `Finite(0)` does not depend on the
    /// variables at all.
    Finite(u32),
    /// Not a polynomial in the variables.
    Unbounded,
}

impl Default for Degree {
    fn default() -> Degree {
        Degree::Finite(0)
    }
}

impl Degree {
    fn finite(d: u32) -> Degree {
        Degree::Finite(d)
    }
    fn value(self) -> Option<u32> {
        match self {
            Degree::Finite(d) => Some(d),
            Degree::Unbounded => None,
        }
    }
    /// Degree of a product: degrees add.
    fn times(self, other: Degree) -> Degree {
        match (self.value(), other.value()) {
            (Some(a), Some(b)) => Degree::finite(a + b),
            _ => Degree::Unbounded,
        }
    }
    /// Degree of a sum: the larger one.
    fn plus(self, other: Degree) -> Degree {
        match (self.value(), other.value()) {
            (Some(a), Some(b)) => Degree::finite(a.max(b)),
            _ => Degree::Unbounded,
        }
    }
    fn power(self, n: u32) -> Degree {
        match self.value() {
            Some(d) => Degree::finite(d * n),
            None => Degree::Unbounded,
        }
    }
    /// The degree as a number, `None` when unbounded.
    pub fn finite_value(self) -> Option<u32> {
        self.value()
    }
}

/// The classification of an expression, or of a whole residual system, as
/// far as a harmonic-balance solve cares.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Nonlinearity {
    pub degree: Degree,
    /// Elementary functions applied to a variable-dependent argument (the
    /// `exp` of a diode, the `tanh` of an EKV model). Empty for a polynomial.
    pub transcendental: BTreeSet<UnaryOp>,
    /// A variable appears in a denominator (a negative integer power).
    pub rational: bool,
    /// A comparison, a select or an ordered reduction branches on a
    /// variable-dependent value: the expression is piecewise.
    pub piecewise: bool,
    /// A call depends on a variable; its body is not classified here.
    pub opaque: bool,
}

impl Nonlinearity {
    fn constant() -> Nonlinearity {
        Nonlinearity::default()
    }
    fn is_constant(&self) -> bool {
        *self == Nonlinearity::default()
    }
    fn absorb(&mut self, other: &Nonlinearity) {
        self.transcendental
            .extend(other.transcendental.iter().copied());
        self.rational |= other.rational;
        self.piecewise |= other.piecewise;
        self.opaque |= other.opaque;
    }
    fn unbounded(flag: impl FnOnce(&mut Nonlinearity)) -> Nonlinearity {
        let mut out = Nonlinearity {
            degree: Degree::Unbounded,
            ..Default::default()
        };
        flag(&mut out);
        out
    }

    /// A polynomial in the variables: its harmonics follow from finite
    /// spectral convolution, so time sampling is optional. Every structural
    /// feature that defeats this also makes the degree unbounded.
    pub fn is_polynomial(&self) -> bool {
        self.degree != Degree::Unbounded
    }

    /// Smallest alias-free number of time samples over one period for every
    /// harmonic the expression generates from `k` solved harmonics:
    /// `2 d k + 1` for degree `d`, `None` when unbounded.
    pub fn alias_free_samples(&self, k: u32) -> Option<usize> {
        self.degree.value().map(|d| (2 * d * k + 1) as usize)
    }
}

/// Classify one expression's dependence on `vars`.
pub fn nonlinearity<K: Field>(
    g: &Graph<K>,
    expr: ExprId,
    vars: &BTreeSet<SymbolId>,
) -> Nonlinearity {
    let mut memo = HashMap::default();
    classify(g, expr, vars, &mut memo)
}

/// Classify a residual system: the worst degree over the residuals and the
/// union of their flags. What the solver has to plan for.
pub fn nonlinearity_of<K: Field>(
    g: &Graph<K>,
    exprs: &[ExprId],
    vars: &BTreeSet<SymbolId>,
) -> Nonlinearity {
    let mut memo = HashMap::default();
    let mut acc = Nonlinearity::constant();
    for &e in exprs {
        let c = classify(g, e, vars, &mut memo);
        acc.degree = acc.degree.plus(c.degree);
        acc.absorb(&c);
    }
    acc
}

fn classify<K: Field>(
    g: &Graph<K>,
    expr: ExprId,
    vars: &BTreeSet<SymbolId>,
    memo: &mut HashMap<ExprId, Nonlinearity>,
) -> Nonlinearity {
    if let Some(c) = memo.get(&expr) {
        return c.clone();
    }
    let sub = |e: ExprId, memo: &mut HashMap<ExprId, Nonlinearity>| classify(g, e, vars, memo);
    let out = match *g.node(expr) {
        Node::Const(_) => Nonlinearity::constant(),
        Node::Symbol(s) => {
            if vars.contains(&s) {
                Nonlinearity {
                    degree: Degree::Finite(1),
                    ..Default::default()
                }
            } else {
                Nonlinearity::constant()
            }
        }
        Node::Neg(a) => sub(a, memo),
        Node::Add(a, b) => {
            let (ca, cb) = (sub(a, memo), sub(b, memo));
            let mut out = Nonlinearity {
                degree: ca.degree.plus(cb.degree),
                ..Default::default()
            };
            out.absorb(&ca);
            out.absorb(&cb);
            out
        }
        Node::Mul(a, b) => {
            let (ca, cb) = (sub(a, memo), sub(b, memo));
            let mut out = Nonlinearity {
                degree: ca.degree.times(cb.degree),
                ..Default::default()
            };
            out.absorb(&ca);
            out.absorb(&cb);
            out
        }
        Node::Pow(a, n) => {
            let ca = sub(a, memo);
            if n == 0 || ca.is_constant() {
                Nonlinearity::constant()
            } else if n > 0 {
                Nonlinearity {
                    degree: ca.degree.power(n as u32),
                    ..ca
                }
            } else {
                let mut out = Nonlinearity::unbounded(|o| o.rational = true);
                out.absorb(&ca);
                out
            }
        }
        Node::Unary(op, a) => {
            let ca = sub(a, memo);
            if ca.is_constant() {
                Nonlinearity::constant()
            } else {
                let mut out = Nonlinearity::unbounded(|o| {
                    o.transcendental.insert(op);
                });
                out.absorb(&ca);
                out
            }
        }
        // A binary function beyond the ring is transcendental in the same
        // sense as a unary one when either argument moves; there is no
        // unary op to name it by, so the flag is the piecewise-free
        // "unbounded" alone.
        Node::Binary(_, a, b) => {
            let (ca, cb) = (sub(a, memo), sub(b, memo));
            if ca.is_constant() && cb.is_constant() {
                Nonlinearity::constant()
            } else {
                let mut out = Nonlinearity::unbounded(|_| {});
                out.absorb(&ca);
                out.absorb(&cb);
                out
            }
        }
        Node::Cmp(_, a, b) => {
            let (ca, cb) = (sub(a, memo), sub(b, memo));
            if ca.is_constant() && cb.is_constant() {
                Nonlinearity::constant()
            } else {
                let mut out = Nonlinearity::unbounded(|o| o.piecewise = true);
                out.absorb(&ca);
                out.absorb(&cb);
                out
            }
        }
        Node::Select(c, t, e) => {
            let (cc, ct, ce) = (sub(c, memo), sub(t, memo), sub(e, memo));
            let mut out = if cc.is_constant() {
                // A fixed branch: the degree of whichever arm is taken.
                Nonlinearity {
                    degree: ct.degree.plus(ce.degree),
                    ..Default::default()
                }
            } else {
                Nonlinearity::unbounded(|o| o.piecewise = true)
            };
            out.absorb(&cc);
            out.absorb(&ct);
            out.absorb(&ce);
            out
        }
        Node::Reduce(op, l) => {
            let parts: Vec<Nonlinearity> = g.args(l).iter().map(|&a| sub(a, memo)).collect();
            let mut out = Nonlinearity::constant();
            for c in &parts {
                out.absorb(c);
            }
            out.degree = match op {
                ReduceOp::Sum => parts
                    .iter()
                    .fold(Degree::Finite(0), |d, c| d.plus(c.degree)),
                ReduceOp::Product => parts
                    .iter()
                    .fold(Degree::Finite(0), |d, c| d.times(c.degree)),
                ReduceOp::Min | ReduceOp::Max => {
                    if parts.iter().all(Nonlinearity::is_constant) {
                        Degree::Finite(0)
                    } else {
                        out.piecewise = true;
                        Degree::Unbounded
                    }
                }
            };
            out
        }
        Node::Dot(l) => {
            let (a, b) = g.dot_args(l);
            let (a, b) = (a.to_vec(), b.to_vec());
            let mut out = Nonlinearity::constant();
            for (x, y) in a.iter().zip(&b) {
                let (cx, cy) = (sub(*x, memo), sub(*y, memo));
                out.degree = out.degree.plus(cx.degree.times(cy.degree));
                out.absorb(&cx);
                out.absorb(&cy);
            }
            out
        }
        // Rational in the matrix, linear in the right-hand side: a rational
        // function of whatever the entries are.
        Node::Solve(l, _) => {
            let parts: Vec<Nonlinearity> = g.args(l).iter().map(|&a| sub(a, memo)).collect();
            if parts.iter().all(Nonlinearity::is_constant) {
                Nonlinearity::constant()
            } else {
                let mut out = Nonlinearity::unbounded(|o| o.rational = true);
                for c in &parts {
                    out.absorb(c);
                }
                out
            }
        }
        Node::Call(_, l) => {
            let parts: Vec<Nonlinearity> = g.args(l).iter().map(|&a| sub(a, memo)).collect();
            if parts.iter().all(Nonlinearity::is_constant) {
                Nonlinearity::constant()
            } else {
                let mut out = Nonlinearity::unbounded(|o| o.opaque = true);
                for c in &parts {
                    out.absorb(c);
                }
                out
            }
        }
    };
    memo.insert(expr, out.clone());
    out
}
