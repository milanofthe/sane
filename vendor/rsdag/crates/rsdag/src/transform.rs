//! Structural rewrites of the DAG: substituting expressions for symbols.

use crate::field::Field;
use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Graph, Memo};
use crate::node::{ExprId, Node, SymbolId};

/// Replace every symbol in `map` with its expression, in every root, in
/// one pass with one shared memo: a subexpression reachable from several
/// roots is rebuilt once. Rebuilt through the smart constructors, so the
/// result is re-folded and hash-consed. The substitution is simultaneous,
/// so a target expression that itself mentions a mapped symbol is not
/// re-substituted. This is the primitive behind merging two nodes (one
/// voltage symbol for another), eliminating one (its solved expression for
/// its symbol) and instantiating a template (its ports for the instance's).
pub fn substitute<K: Field>(
    ctx: &mut Graph<K>,
    roots: &[ExprId],
    map: &HashMap<SymbolId, ExprId>,
) -> Vec<ExprId> {
    let mut memo = ctx.take_memo();
    let resolve = |s: SymbolId| map.get(&s).copied();
    let out: Vec<ExprId> = roots
        .iter()
        .map(|&e| subst_inner(ctx, e, &resolve, &mut memo))
        .collect();
    ctx.put_memo(memo);
    out
}

/// The shared substitution traverser: rebuild `expr` through the smart
/// constructors, replacing every symbol for which `resolve` returns `Some`,
/// memoised over shared subexpressions. Single-symbol and multi-symbol
/// substitution differ only in their `resolve` closure, so the per-node walk
/// lives here once (a new `Node` variant is then handled in exactly one place).
fn subst_inner<K: Field, F: Fn(SymbolId) -> Option<ExprId>>(
    ctx: &mut Graph<K>,
    expr: ExprId,
    resolve: &F,
    memo: &mut Memo,
) -> ExprId {
    if let Some(r) = memo.get(expr) {
        return r;
    }
    let node = *ctx.node(expr);
    let r = match node {
        Node::Const(_) => expr,
        Node::Symbol(s) => resolve(s).unwrap_or(expr),
        Node::Add(a, b) => {
            let a = subst_inner(ctx, a, resolve, memo);
            let b = subst_inner(ctx, b, resolve, memo);
            ctx.add(a, b)
        }
        Node::Mul(a, b) => {
            let a = subst_inner(ctx, a, resolve, memo);
            let b = subst_inner(ctx, b, resolve, memo);
            ctx.mul(a, b)
        }
        Node::Neg(a) => {
            let a = subst_inner(ctx, a, resolve, memo);
            ctx.neg(a)
        }
        Node::Pow(a, n) => {
            let a = subst_inner(ctx, a, resolve, memo);
            ctx.pow_i(a, n)
        }
        Node::Unary(op, a) => {
            let a = subst_inner(ctx, a, resolve, memo);
            ctx.unary(op, a)
        }
        Node::Cmp(op, a, b) => {
            let a = subst_inner(ctx, a, resolve, memo);
            let b = subst_inner(ctx, b, resolve, memo);
            ctx.cmp(op, a, b)
        }
        Node::Binary(op, a, b) => {
            let a = subst_inner(ctx, a, resolve, memo);
            let b = subst_inner(ctx, b, resolve, memo);
            ctx.binary(op, a, b)
        }
        Node::Select(c, t, e) => {
            let c = subst_inner(ctx, c, resolve, memo);
            let t = subst_inner(ctx, t, resolve, memo);
            let e = subst_inner(ctx, e, resolve, memo);
            ctx.select(c, t, e)
        }
        Node::Reduce(op, l) => {
            let args = ctx.args(l).to_vec();
            let na: Vec<ExprId> = args
                .iter()
                .map(|&a| subst_inner(ctx, a, resolve, memo))
                .collect();
            ctx.reduce(op, na)
        }
        Node::Solve(l, i) => {
            let all: Vec<ExprId> = ctx
                .args(l)
                .to_vec()
                .iter()
                .map(|&e| subst_inner(ctx, e, resolve, memo))
                .collect();
            ctx.solve_component(all, i)
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            let (a, b) = (a.to_vec(), b.to_vec());
            let na: Vec<ExprId> = a
                .iter()
                .map(|&e| subst_inner(ctx, e, resolve, memo))
                .collect();
            let nb: Vec<ExprId> = b
                .iter()
                .map(|&e| subst_inner(ctx, e, resolve, memo))
                .collect();
            ctx.dot(na, nb)
        }
        Node::Call(o, l) => {
            let args = ctx.args(l).to_vec();
            let na: Vec<ExprId> = args
                .iter()
                .map(|&a| subst_inner(ctx, a, resolve, memo))
                .collect();
            ctx.call_output(o, &na)
        }
    };
    memo.set(expr, r);
    r
}
