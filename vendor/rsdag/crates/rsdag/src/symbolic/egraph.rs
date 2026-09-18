//! Equality-saturation simplification for exact graphs.
//!
//! The reachable expression is converted into an egg `RecExpr`, saturated
//! against a set of algebraic rewrites (with rational constant folding as an
//! e-graph analysis), and the smallest equivalent form is extracted back
//! into the graph through the smart constructors. This is the general
//! algebraic simplifier for symbolic work (transfer functions, sensitivities
//! in closed form); the numeric pipeline never needs it, and it is only
//! defined for `Graph<BigRational>`, where every rewrite is exact.
//!
//! Comparisons, selects, calls and min/max reductions are left alone: an
//! expression containing one is returned unchanged.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use egg::{
    define_language, rewrite as rw, AstSize, Extractor, Id, RecExpr, Rewrite, Runner, Symbol,
};
use egg::{merge_option, Analysis, DidMerge, EGraph};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive};

use crate::field::ratio_powi;
use crate::graph::Graph;
use crate::node::{BinOp, ExprId, Node, ReduceOp, UnaryOp};

/// An exact rational literal of the e-graph language.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Rat(pub BigRational);

impl fmt::Display for Rat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.denom().is_one() {
            write!(f, "{}", self.0.numer())
        } else {
            write!(f, "{}/{}", self.0.numer(), self.0.denom())
        }
    }
}

impl FromStr for Rat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parse_int = |t: &str| t.trim().parse::<BigInt>().map_err(|e| e.to_string());
        if let Some((n, d)) = s.split_once('/') {
            Ok(Rat(BigRational::new(parse_int(n)?, parse_int(d)?)))
        } else {
            Ok(Rat(BigRational::from_integer(parse_int(s)?)))
        }
    }
}

define_language! {
    /// The e-graph language: the ring, integer powers, every unary and
    /// binary function by name, rationals and symbols.
    pub enum Lang {
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "neg" = Neg([Id; 1]),
        "pow" = Pow([Id; 2]),
        "exp" = Exp([Id; 1]),
        "ln" = Ln([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "sinh" = Sinh([Id; 1]),
        "cosh" = Cosh([Id; 1]),
        "tanh" = Tanh([Id; 1]),
        "atan" = Atan([Id; 1]),
        "floor" = Floor([Id; 1]),
        "tan" = Tan([Id; 1]),
        "log10" = Log10([Id; 1]),
        "log2" = Log2([Id; 1]),
        "log1p" = Log1p([Id; 1]),
        "expm1" = Expm1([Id; 1]),
        "cbrt" = Cbrt([Id; 1]),
        "abs" = Abs([Id; 1]),
        "sign" = Sign([Id; 1]),
        "ceil" = Ceil([Id; 1]),
        "round" = Round([Id; 1]),
        "trunc" = Trunc([Id; 1]),
        "asin" = Asin([Id; 1]),
        "acos" = Acos([Id; 1]),
        "asinh" = Asinh([Id; 1]),
        "acosh" = Acosh([Id; 1]),
        "atanh" = Atanh([Id; 1]),
        "erf" = Erf([Id; 1]),
        "erfc" = Erfc([Id; 1]),
        "lgamma" = Lgamma([Id; 1]),
        "tgamma" = Tgamma([Id; 1]),
        "digamma" = Digamma([Id; 1]),
        "trigamma" = Trigamma([Id; 1]),
        "rand_uniform" = RandUniform([Id; 1]),
        "powf" = Powf([Id; 2]),
        "mod" = Mod([Id; 2]),
        "atan2" = Atan2([Id; 2]),
        "hypot" = Hypot([Id; 2]),
        Num(Rat),
        Sym(Symbol),
    }
}

/// Rational constant folding as an e-graph analysis.
#[derive(Default)]
pub struct ConstantFold;

impl Analysis<Lang> for ConstantFold {
    type Data = Option<BigRational>;
    fn make(egraph: &EGraph<Lang, Self>, enode: &Lang) -> Self::Data {
        let g = |id: &Id| egraph[*id].data.clone();
        Some(match enode {
            Lang::Num(r) => r.0.clone(),
            Lang::Add([a, b]) => g(a)? + g(b)?,
            Lang::Mul([a, b]) => g(a)? * g(b)?,
            Lang::Neg([a]) => -g(a)?,
            Lang::Pow([a, b]) => {
                let base = g(a)?;
                let exp = g(b)?;
                if !exp.denom().is_one() {
                    return None;
                }
                let n = exp.numer().to_i64()?;
                if base == BigRational::from_integer(BigInt::from(0)) && n < 0 {
                    return None;
                }
                ratio_powi(&base, n)
            }
            _ => return None,
        })
    }
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            debug_assert_eq!(*a, b, "constant fold disagreement");
            DidMerge(false, false)
        })
    }
    fn modify(egraph: &mut EGraph<Lang, Self>, id: Id) {
        if let Some(r) = egraph[id].data.clone() {
            let added = egraph.add(Lang::Num(Rat(r)));
            egraph.union(id, added);
        }
    }
}

fn contains_unsupported(g: &Graph<BigRational>, root: ExprId) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match g.node(id) {
            Node::Cmp(..) | Node::Select(..) | Node::Call(..) | Node::Solve(..) => return true,
            Node::Reduce(ReduceOp::Min | ReduceOp::Max, _) => return true,
            _ => stack.extend_from_slice(&g.operands(id)),
        }
    }
    false
}

/// Simplify `root` by equality saturation; returns the extracted form (or
/// `root` itself when it contains an unsupported node).
pub fn simplify_egraph(g: &mut Graph<BigRational>, root: ExprId) -> ExprId {
    if contains_unsupported(g, root) {
        return root;
    }
    let expr = to_egg(g, root);
    let runner = Runner::default()
        .with_expr(&expr)
        .with_iter_limit(40)
        .with_node_limit(100_000)
        .run(&rules());
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_cost, best) = extractor.find_best(runner.roots[0]);
    from_egg(g, &best)
}

fn rules() -> Vec<Rewrite<Lang, ConstantFold>> {
    vec![
        rw!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
        rw!("comm-mul";  "(* ?a ?b)"        => "(* ?b ?a)"),
        rw!("assoc-add"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("assoc-mul"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        rw!("add-0";     "(+ ?a 0)"         => "?a"),
        rw!("mul-1";     "(* ?a 1)"         => "?a"),
        rw!("mul-0";     "(* ?a 0)"         => "0"),
        rw!("neg-neg";   "(neg (neg ?a))"   => "?a"),
        rw!("neg-to-mul"; "(neg ?a)"        => "(* -1 ?a)"),
        rw!("mul-to-neg"; "(* -1 ?a)"       => "(neg ?a)"),
        rw!("distribute"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        rw!("factor";     "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        rw!("pow-mul";  "(* (pow ?a -1) (pow ?b -1))" => "(pow (* ?a ?b) -1)"),
        rw!("inv-cancel"; "(* ?a (pow ?a -1))" => "1"),
        rw!("pow-neg";  "(pow (neg ?a) -1)" => "(neg (pow ?a -1))"),
        rw!("pow-pow";  "(pow (pow ?a ?m) ?n)" => "(pow ?a (* ?m ?n))"),
        rw!("add-self"; "(+ ?a ?a)"         => "(* 2 ?a)"),
        rw!("sub-self"; "(+ ?a (neg ?a))"   => "0"),
    ]
}

fn unary_lang(op: UnaryOp, x: Id) -> Lang {
    match op {
        UnaryOp::Exp => Lang::Exp([x]),
        UnaryOp::Ln => Lang::Ln([x]),
        UnaryOp::Sqrt => Lang::Sqrt([x]),
        UnaryOp::Sin => Lang::Sin([x]),
        UnaryOp::Cos => Lang::Cos([x]),
        UnaryOp::Sinh => Lang::Sinh([x]),
        UnaryOp::Cosh => Lang::Cosh([x]),
        UnaryOp::Tanh => Lang::Tanh([x]),
        UnaryOp::Atan => Lang::Atan([x]),
        UnaryOp::Floor => Lang::Floor([x]),
        UnaryOp::Tan => Lang::Tan([x]),
        UnaryOp::Log10 => Lang::Log10([x]),
        UnaryOp::Log2 => Lang::Log2([x]),
        UnaryOp::Log1p => Lang::Log1p([x]),
        UnaryOp::Expm1 => Lang::Expm1([x]),
        UnaryOp::Cbrt => Lang::Cbrt([x]),
        UnaryOp::Abs => Lang::Abs([x]),
        UnaryOp::Sign => Lang::Sign([x]),
        UnaryOp::Ceil => Lang::Ceil([x]),
        UnaryOp::Round => Lang::Round([x]),
        UnaryOp::Trunc => Lang::Trunc([x]),
        UnaryOp::Asin => Lang::Asin([x]),
        UnaryOp::Acos => Lang::Acos([x]),
        UnaryOp::Asinh => Lang::Asinh([x]),
        UnaryOp::Acosh => Lang::Acosh([x]),
        UnaryOp::Atanh => Lang::Atanh([x]),
        UnaryOp::Erf => Lang::Erf([x]),
        UnaryOp::Erfc => Lang::Erfc([x]),
        UnaryOp::Lgamma => Lang::Lgamma([x]),
        UnaryOp::Tgamma => Lang::Tgamma([x]),
        UnaryOp::Digamma => Lang::Digamma([x]),
        UnaryOp::Trigamma => Lang::Trigamma([x]),
        UnaryOp::RandUniform => Lang::RandUniform([x]),
    }
}

fn lang_unary(node: &Lang) -> Option<(UnaryOp, Id)> {
    Some(match *node {
        Lang::Exp([a]) => (UnaryOp::Exp, a),
        Lang::Ln([a]) => (UnaryOp::Ln, a),
        Lang::Sqrt([a]) => (UnaryOp::Sqrt, a),
        Lang::Sin([a]) => (UnaryOp::Sin, a),
        Lang::Cos([a]) => (UnaryOp::Cos, a),
        Lang::Sinh([a]) => (UnaryOp::Sinh, a),
        Lang::Cosh([a]) => (UnaryOp::Cosh, a),
        Lang::Tanh([a]) => (UnaryOp::Tanh, a),
        Lang::Atan([a]) => (UnaryOp::Atan, a),
        Lang::Floor([a]) => (UnaryOp::Floor, a),
        Lang::Tan([a]) => (UnaryOp::Tan, a),
        Lang::Log10([a]) => (UnaryOp::Log10, a),
        Lang::Log2([a]) => (UnaryOp::Log2, a),
        Lang::Log1p([a]) => (UnaryOp::Log1p, a),
        Lang::Expm1([a]) => (UnaryOp::Expm1, a),
        Lang::Cbrt([a]) => (UnaryOp::Cbrt, a),
        Lang::Abs([a]) => (UnaryOp::Abs, a),
        Lang::Sign([a]) => (UnaryOp::Sign, a),
        Lang::Ceil([a]) => (UnaryOp::Ceil, a),
        Lang::Round([a]) => (UnaryOp::Round, a),
        Lang::Trunc([a]) => (UnaryOp::Trunc, a),
        Lang::Asin([a]) => (UnaryOp::Asin, a),
        Lang::Acos([a]) => (UnaryOp::Acos, a),
        Lang::Asinh([a]) => (UnaryOp::Asinh, a),
        Lang::Acosh([a]) => (UnaryOp::Acosh, a),
        Lang::Atanh([a]) => (UnaryOp::Atanh, a),
        Lang::Erf([a]) => (UnaryOp::Erf, a),
        Lang::Erfc([a]) => (UnaryOp::Erfc, a),
        Lang::Lgamma([a]) => (UnaryOp::Lgamma, a),
        Lang::Tgamma([a]) => (UnaryOp::Tgamma, a),
        Lang::Digamma([a]) => (UnaryOp::Digamma, a),
        Lang::Trigamma([a]) => (UnaryOp::Trigamma, a),
        Lang::RandUniform([a]) => (UnaryOp::RandUniform, a),
        _ => return None,
    })
}

fn to_egg(g: &Graph<BigRational>, root: ExprId) -> RecExpr<Lang> {
    let mut rec = RecExpr::default();
    let mut memo: HashMap<ExprId, Id> = HashMap::new();
    build(g, root, &mut rec, &mut memo);
    rec
}

fn build(
    g: &Graph<BigRational>,
    id: ExprId,
    rec: &mut RecExpr<Lang>,
    memo: &mut HashMap<ExprId, Id>,
) -> Id {
    if let Some(&x) = memo.get(&id) {
        return x;
    }
    let eid = match *g.node(id) {
        Node::Const(c) => rec.add(Lang::Num(Rat(g.const_val(c).clone()))),
        Node::Symbol(s) => rec.add(Lang::Sym(Symbol::from(g.symbol_name(s)))),
        Node::Add(a, b) => {
            let x = build(g, a, rec, memo);
            let y = build(g, b, rec, memo);
            rec.add(Lang::Add([x, y]))
        }
        Node::Mul(a, b) => {
            let x = build(g, a, rec, memo);
            let y = build(g, b, rec, memo);
            rec.add(Lang::Mul([x, y]))
        }
        Node::Neg(a) => {
            let x = build(g, a, rec, memo);
            rec.add(Lang::Neg([x]))
        }
        Node::Unary(op, a) => {
            let x = build(g, a, rec, memo);
            rec.add(unary_lang(op, x))
        }
        Node::Binary(op, a, b) => {
            let x = build(g, a, rec, memo);
            let y = build(g, b, rec, memo);
            rec.add(match op {
                BinOp::Powf => Lang::Powf([x, y]),
                BinOp::Mod => Lang::Mod([x, y]),
                BinOp::Atan2 => Lang::Atan2([x, y]),
                BinOp::Hypot => Lang::Hypot([x, y]),
            })
        }
        Node::Pow(a, n) => {
            let x = build(g, a, rec, memo);
            let e = rec.add(Lang::Num(Rat(BigRational::from_integer(BigInt::from(n)))));
            rec.add(Lang::Pow([x, e]))
        }
        Node::Reduce(op, l) => {
            let ids: Vec<Id> = g.args(l).iter().map(|&a| build(g, a, rec, memo)).collect();
            let mut it = ids.into_iter();
            let first = it.next().expect("reduce has operands");
            it.fold(first, |acc, x| match op {
                ReduceOp::Sum => rec.add(Lang::Add([acc, x])),
                ReduceOp::Product => rec.add(Lang::Mul([acc, x])),
                ReduceOp::Min | ReduceOp::Max => unreachable!("filtered by contains_unsupported"),
            })
        }
        Node::Dot(l) => {
            let (a, b) = g.dot_args(l);
            let prods: Vec<Id> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let xi = build(g, x, rec, memo);
                    let yi = build(g, y, rec, memo);
                    rec.add(Lang::Mul([xi, yi]))
                })
                .collect();
            let mut it = prods.into_iter();
            let first = it.next().expect("dot has terms");
            it.fold(first, |acc, x| rec.add(Lang::Add([acc, x])))
        }
        Node::Cmp(..) | Node::Select(..) | Node::Call(..) | Node::Solve(..) => {
            unreachable!("filtered by contains_unsupported")
        }
    };
    memo.insert(id, eid);
    eid
}

fn from_egg(g: &mut Graph<BigRational>, rec: &RecExpr<Lang>) -> ExprId {
    let nodes = rec.as_ref();
    let mut map: Vec<ExprId> = Vec::with_capacity(nodes.len());
    let at = |map: &Vec<ExprId>, i: &Id| map[usize::from(*i)];
    for node in nodes {
        let cid = match node {
            Lang::Num(r) => g.konst(r.0.clone()),
            Lang::Sym(s) => g.sym(s.as_str()),
            Lang::Add([a, b]) => g.add(at(&map, a), at(&map, b)),
            Lang::Mul([a, b]) => g.mul(at(&map, a), at(&map, b)),
            Lang::Neg([a]) => g.neg(at(&map, a)),
            Lang::Pow([a, b]) => {
                let base = at(&map, a);
                let exp = match &nodes[usize::from(*b)] {
                    Lang::Num(r) if r.0.denom().is_one() => {
                        r.0.numer().to_i64().expect("integer exponent fits in i64")
                    }
                    _ => panic!("non-integer power exponent after simplification"),
                };
                g.pow_i(base, exp)
            }
            Lang::Powf([a, b]) => g.binary(BinOp::Powf, at(&map, a), at(&map, b)),
            Lang::Mod([a, b]) => g.binary(BinOp::Mod, at(&map, a), at(&map, b)),
            Lang::Atan2([a, b]) => g.binary(BinOp::Atan2, at(&map, a), at(&map, b)),
            Lang::Hypot([a, b]) => g.binary(BinOp::Hypot, at(&map, a), at(&map, b)),
            other => {
                let (op, a) = lang_unary(other).expect("a unary node");
                g.unary(op, at(&map, &a))
            }
        };
        map.push(cid);
    }
    *map.last().expect("non-empty expression")
}
