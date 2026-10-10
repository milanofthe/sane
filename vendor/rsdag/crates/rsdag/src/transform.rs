//! Structural rewrites of the DAG: the bottom-up rewrite every transform is
//! written in, and substituting expressions for symbols.

use crate::field::Field;
use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Graph, NO_CONTEXT};
use crate::node::{ArgList, ExprId, Node, SymbolId};

/// A node's operands, rewritten: in [`Graph::operands`] order, and for a
/// call the argument list they make, interned once however many calls of
/// the one instance share it, and its context over its bound expressions
/// rewritten.
pub(crate) struct Ops<'a> {
    pub ops: &'a [ExprId],
    pub list: Option<ArgList>,
    pub ctx: u32,
}

impl<K: Field> Graph<K> {
    /// [`build`](Graph::build) over rewritten operands: a call over its
    /// interned list, without hashing the list again.
    pub(crate) fn rebuild(&mut self, node: Node, ops: &Ops) -> ExprId {
        match (node, ops.list) {
            (Node::Call(o, _), Some(l)) => {
                let (f, out) = self.output(o);
                self.call_list_in(f, out, ops.ctx, l)
            }
            _ => self.build(node, ops.ops),
        }
    }
}

/// Rewrite the DAG under `roots` bottom-up: `rule(g, e, node, ops)` is the
/// new id of node `e`, `ops` its operands already rewritten;
/// [`Graph::rebuild`] is the identity rule. One shared memo, so a
/// subexpression reachable from several roots is rewritten once, and an
/// explicit stack, so a deep chain does not overflow the call stack.
/// Operands are visited in order, the order a recursive walk takes, so new
/// nodes are created in the same order as by one. The calls of one instance
/// share their argument list, and it is rewritten once: an instance of a
/// wide body costs its width once, not once per output.
pub(crate) fn rewrite<K: Field>(
    g: &mut Graph<K>,
    roots: &[ExprId],
    mut rule: impl FnMut(&mut Graph<K>, ExprId, Node, &Ops) -> ExprId,
) -> Vec<ExprId> {
    let mut memo = g.take_memo();
    let mut stack: Vec<(ExprId, bool)> = Vec::new();
    let mut ops: Vec<ExprId> = Vec::new();
    // per call argument list: its operands rewritten, and interned
    let mut lists: HashMap<ArgList, (Vec<ExprId>, ArgList)> = HashMap::default();
    // per context: its bound expressions rewritten
    let mut contexts: HashMap<u32, u32> = HashMap::default();
    for &root in roots {
        stack.push((root, false));
        while let Some((e, expanded)) = stack.pop() {
            if memo.get(e).is_some() {
                continue;
            }
            let node = *g.node(e);
            let call_list = match node {
                Node::Call(_, l) => Some(l),
                _ => None,
            };
            if !expanded {
                stack.push((e, true));
                // a list rewritten before: only the binding, which another
                // call over the list may not share
                match (node, call_list) {
                    (Node::Call(o, _), Some(l)) if lists.contains_key(&l) => {
                        let c = g.context_of(o);
                        if c != NO_CONTEXT && !contexts.contains_key(&c) {
                            let bound = g.args(g.context_list(c));
                            stack.extend(
                                (bound.iter().rev())
                                    .filter(|&&c| memo.get(c).is_none())
                                    .map(|&c| (c, false)),
                            );
                        }
                    }
                    _ => {
                        let pending = g.operands(e);
                        stack.extend(
                            (pending.iter().rev())
                                .filter(|&&c| memo.get(c).is_none())
                                .map(|&c| (c, false)),
                        );
                    }
                }
                continue;
            }
            let r = match call_list {
                Some(l) => {
                    let (new, nl) = &*lists.entry(l).or_insert_with(|| {
                        let new: Vec<ExprId> = g
                            .args(l)
                            .iter()
                            .map(|&c| memo.get(c).expect("operand rewritten"))
                            .collect();
                        let nl = g.intern_args(&new);
                        (new, nl)
                    });
                    let Node::Call(o, _) = node else {
                        unreachable!()
                    };
                    let ctx = match g.context_of(o) {
                        NO_CONTEXT => NO_CONTEXT,
                        c => *contexts.entry(c).or_insert_with(|| {
                            let f = g.output(o).0;
                            let new: Vec<ExprId> = g
                                .args(g.context_list(c))
                                .iter()
                                .map(|&e| memo.get(e).expect("bound expression rewritten"))
                                .collect();
                            g.context_over(c, f, &new)
                        }),
                    };
                    let nl = *nl;
                    let ops = Ops {
                        ops: new,
                        list: Some(nl),
                        ctx,
                    };
                    rule(g, e, node, &ops)
                }
                None => {
                    ops.clear();
                    ops.extend(
                        g.operands(e)
                            .iter()
                            .map(|&c| memo.get(c).expect("operand rewritten")),
                    );
                    let ops = Ops {
                        ops: &ops,
                        list: None,
                        ctx: NO_CONTEXT,
                    };
                    rule(g, e, node, &ops)
                }
            };
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
    // per function, the copy its calls run (see `Graph::rebound`)
    let mut copies: HashMap<crate::func::FuncId, Option<crate::func::FuncId>> = HashMap::default();
    rewrite(ctx, roots, |g, e, node, ops| match (node, ops.list) {
        (Node::Symbol(s), _) => map.get(&s).copied().unwrap_or(e),
        // a call whose body reads a symbol `map` binds runs a copy bound so
        (Node::Call(o, _), Some(l)) => {
            let (f, k) = g.output(o);
            let copy = *copies.entry(f).or_insert_with(|| g.rebound(f, map));
            match copy {
                Some(copy) => {
                    let ctx = g.context_onto(ops.ctx, copy);
                    g.call_list_in(copy, k, ctx, l)
                }
                None => g.rebuild(node, ops),
            }
        }
        _ => g.rebuild(node, ops),
    })
}
