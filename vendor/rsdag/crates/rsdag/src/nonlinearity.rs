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
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
            (Some(a), Some(b)) => a.checked_add(b).map_or(Degree::Unbounded, Degree::finite),
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
            Some(d) => d.checked_mul(n).map_or(Degree::Unbounded, Degree::finite),
            None => Degree::Unbounded,
        }
    }
    /// The degree as a number, `None` when unbounded.
    pub fn finite_value(self) -> Option<u32> {
        self.value()
    }
}

/// A set of unary ops, one bit each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct UnarySet(u64);

const _: () = assert!(crate::node::UNARY_OPS.len() <= 64);

impl UnarySet {
    pub fn insert(&mut self, op: UnaryOp) {
        self.0 |= 1 << op.code();
    }
    pub fn contains(&self, op: &UnaryOp) -> bool {
        self.0 & (1 << op.code()) != 0
    }
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
    /// The ops in the set, in the order of the enum.
    pub fn iter(&self) -> impl Iterator<Item = UnaryOp> + '_ {
        crate::node::UNARY_OPS
            .iter()
            .map(|s| s.op)
            .filter(|op| self.contains(op))
    }
}

/// The classification of an expression, or of a whole residual system, as
/// far as a harmonic-balance solve cares.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Nonlinearity {
    pub degree: Degree,
    /// Elementary functions applied to a variable-dependent argument (the
    /// `exp` of a diode, the `tanh` of an EKV model). Empty for a polynomial.
    pub transcendental: UnarySet,
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
        self.transcendental.0 |= other.transcendental.0;
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
    nonlinearity_of(g, &[expr], vars)
}

/// Classify a residual system: the worst degree over the residuals and the
/// union of their flags. What the solver has to plan for.
///
/// A call is classified through its function's body, each parameter taking
/// its argument's class: the same answer as for the inlined expression,
/// computed once per output and argument classes. A call into an extern
/// body is opaque.
pub fn nonlinearity_of<K: Field>(
    g: &Graph<K>,
    exprs: &[ExprId],
    vars: &BTreeSet<SymbolId>,
) -> Nonlinearity {
    let leaf = |s: SymbolId| Nonlinearity {
        degree: Degree::Finite(vars.contains(&s) as u32),
        ..Default::default()
    };
    let classes = classify_cone(g, exprs, &leaf, &mut HashMap::default());
    let mut acc = Nonlinearity::constant();
    for c in classes {
        acc.degree = acc.degree.plus(c.degree);
        acc.absorb(&c);
    }
    acc
}

/// The classes of the calls already classified: by output and argument
/// classes.
type CallMemo = HashMap<(crate::func::OutputId, Vec<Nonlinearity>), Nonlinearity>;

/// The classes of `exprs`, a symbol classified by `leaf`.
fn classify_cone<K: Field>(
    g: &Graph<K>,
    exprs: &[ExprId],
    leaf: &dyn Fn(SymbolId) -> Nonlinearity,
    calls: &mut CallMemo,
) -> Vec<Nonlinearity> {
    // One sweep over the nodes the expressions reach, ascending (a node
    // after its operands), each classified from its operands'.
    let mut cone: Vec<ExprId> = Vec::new();
    let mut seen = rustc_hash::FxHashSet::default();
    let mut stack = exprs.to_vec();
    while let Some(e) = stack.pop() {
        if seen.insert(e) {
            cone.push(e);
            stack.extend_from_slice(&g.operands(e));
        }
    }
    cone.sort_unstable();
    let mut memo: HashMap<ExprId, Nonlinearity> = HashMap::default();
    for &e in &cone {
        let c = match *g.node(e) {
            Node::Symbol(s) => leaf(s),
            Node::Call(o, l) => {
                let args: Vec<Nonlinearity> = g.args(l).iter().map(|a| memo[a]).collect();
                classify_call(g, o, args, calls)
            }
            _ => classify(g, e, |a| memo[&a]),
        };
        memo.insert(e, c);
    }
    exprs.iter().map(|e| memo[e]).collect()
}

/// Output `o` called on arguments of classes `args`: its body classified with
/// each parameter taking its argument's class (opaque for an extern body).
fn classify_call<K: Field>(
    g: &Graph<K>,
    o: crate::func::OutputId,
    args: Vec<Nonlinearity>,
    calls: &mut CallMemo,
) -> Nonlinearity {
    let key = (o, args);
    if let Some(&c) = calls.get(&key) {
        return c;
    }
    let (f, out) = g.output(o);
    let func = g.func(f);
    let args = &key.1;
    let c = match func.outputs()[out as usize] {
        crate::func::Output::Zero => Nonlinearity::constant(),
        crate::func::Output::Slot(_) => {
            if args.iter().all(Nonlinearity::is_constant) {
                Nonlinearity::constant()
            } else {
                let mut out = Nonlinearity::unbounded(|o| o.opaque = true);
                for a in args {
                    out.absorb(a);
                }
                out
            }
        }
        crate::func::Output::Expr(e) => {
            let params: HashMap<SymbolId, Nonlinearity> = func
                .params()
                .iter()
                .copied()
                .zip(args.iter().copied())
                .collect();
            // A symbol of the body that is no parameter (a global the
            // function reads) is constant: the function's parameters are
            // its free symbols.
            let leaf = |s: SymbolId| params.get(&s).copied().unwrap_or_default();
            classify_cone(g, &[e], &leaf, calls)[0]
        }
    };
    calls.insert(key, c);
    c
}

/// Node `expr` classified from its operands' classes, `sub`.
fn classify<K: Field>(
    g: &Graph<K>,
    expr: ExprId,
    sub: impl Fn(ExprId) -> Nonlinearity,
) -> Nonlinearity {
    // Every operand's flags, and whether any moves with the variables.
    let parts = || g.operands(expr).iter().map(|&a| sub(a)).collect::<Vec<_>>();
    let joined = |parts: &[Nonlinearity], mut out: Nonlinearity| {
        for c in parts {
            out.absorb(c);
        }
        out
    };
    match *g.node(expr) {
        Node::Const(_) => Nonlinearity::constant(),
        Node::Symbol(_) | Node::Call(..) => unreachable!("classified by the cone sweep"),
        Node::Neg(a) => sub(a),
        Node::Add(a, b) => joined(
            &[sub(a), sub(b)],
            Nonlinearity {
                degree: sub(a).degree.plus(sub(b).degree),
                ..Default::default()
            },
        ),
        Node::Mul(a, b) => joined(
            &[sub(a), sub(b)],
            Nonlinearity {
                degree: sub(a).degree.times(sub(b).degree),
                ..Default::default()
            },
        ),
        Node::Pow(a, n) => {
            let ca = sub(a);
            if n == 0 || ca.is_constant() {
                Nonlinearity::constant()
            } else if n > 0 {
                Nonlinearity {
                    degree: ca.degree.power(n as u32),
                    ..ca
                }
            } else {
                joined(&[ca], Nonlinearity::unbounded(|o| o.rational = true))
            }
        }
        Node::Unary(op, a) => {
            let ca = sub(a);
            if ca.is_constant() {
                Nonlinearity::constant()
            } else {
                joined(
                    &[ca],
                    Nonlinearity::unbounded(|o| o.transcendental.insert(op)),
                )
            }
        }
        // A binary function beyond the ring is transcendental in the same
        // sense as a unary one when either argument moves; there is no
        // unary op to name it by, so the flag is the piecewise-free
        // "unbounded" alone.
        Node::Binary(..) | Node::Cmp(..) | Node::Solve(..) => {
            let parts = parts();
            if parts.iter().all(Nonlinearity::is_constant) {
                return Nonlinearity::constant();
            }
            let out = Nonlinearity::unbounded(|o| match g.node(expr) {
                Node::Cmp(..) => o.piecewise = true,
                // Rational in the matrix, linear in the right-hand side.
                Node::Solve(..) => o.rational = true,
                _ => {}
            });
            joined(&parts, out)
        }
        Node::Select(c, t, e) => {
            let (cc, ct, ce) = (sub(c), sub(t), sub(e));
            let out = if cc.is_constant() {
                // A fixed branch: the degree of whichever arm is taken.
                Nonlinearity {
                    degree: ct.degree.plus(ce.degree),
                    ..Default::default()
                }
            } else {
                Nonlinearity::unbounded(|o| o.piecewise = true)
            };
            joined(&[cc, ct, ce], out)
        }
        Node::Reduce(op, _) => {
            let parts = parts();
            let mut out = joined(&parts, Nonlinearity::constant());
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
            let mut out = Nonlinearity::constant();
            for (&x, &y) in a.iter().zip(b) {
                let (cx, cy) = (sub(x), sub(y));
                out.degree = out.degree.plus(cx.degree.times(cy.degree));
                out.absorb(&cx);
                out.absorb(&cy);
            }
            out
        }
    }
}
