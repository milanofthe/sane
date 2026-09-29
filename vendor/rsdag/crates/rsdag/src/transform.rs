//! Structural rewrites of the DAG: the bottom-up rewrite every transform is
//! written in, and substituting expressions for symbols.

use crate::field::Field;
use rustc_hash::FxHashMap as HashMap;

use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};

/// Rewrite the DAG under `roots` bottom-up: `rule(g, e, node, ops)` is the
/// new id of node `e`, `ops` its operands already rewritten (in
/// [`Graph::operands`] order); [`Graph::build`] is the identity rule. One
/// shared memo, so a subexpression reachable from several roots is rewritten
/// once, and an explicit stack, so a deep chain does not overflow the call
/// stack. Operands are visited in order, the order a recursive walk takes,
/// so new nodes are created in the same order as by one.
pub(crate) fn rewrite<K: Field>(
    g: &mut Graph<K>,
    roots: &[ExprId],
    mut rule: impl FnMut(&mut Graph<K>, ExprId, Node, &[ExprId]) -> ExprId,
) -> Vec<ExprId> {
    let mut memo = g.take_memo();
    let mut stack: Vec<(ExprId, bool)> = Vec::new();
    let mut ops: Vec<ExprId> = Vec::new();
    for &root in roots {
        stack.push((root, false));
        while let Some((e, expanded)) = stack.pop() {
            if memo.get(e).is_some() {
                continue;
            }
            if !expanded {
                stack.push((e, true));
                let pending = g.operands(e);
                stack.extend(
                    pending
                        .iter()
                        .rev()
                        .filter(|&&c| memo.get(c).is_none())
                        .map(|&c| (c, false)),
                );
                continue;
            }
            ops.clear();
            ops.extend(
                g.operands(e)
                    .iter()
                    .map(|&c| memo.get(c).expect("operand rewritten")),
            );
            let node = *g.node(e);
            let r = rule(g, e, node, &ops);
            memo.set(e, r);
        }
    }
    let out = roots
        .iter()
        .map(|&r| memo.get(r).expect("root rewritten"))
        .collect();
    g.put_memo(memo);
    out
}

/// Replace every symbol in `map` with its expression, in every root, in
/// one pass: a subexpression reachable from several roots is rebuilt once.
/// Rebuilt through the smart constructors, so the result is re-folded and
/// hash-consed. The substitution is simultaneous, so a target expression
/// that itself mentions a mapped symbol is not re-substituted. This is the
/// primitive behind merging two nodes (one voltage symbol for another),
/// eliminating one (its solved expression for its symbol) and instantiating
/// a template (its ports for the instance's).
pub fn substitute<K: Field>(
    ctx: &mut Graph<K>,
    roots: &[ExprId],
    map: &HashMap<SymbolId, ExprId>,
) -> Vec<ExprId> {
    rewrite(ctx, roots, |g, e, node, ops| match node {
        Node::Symbol(s) => map.get(&s).copied().unwrap_or(e),
        _ => g.build(node, ops),
    })
}
