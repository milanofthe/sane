//! Graph simplification.
//!
//! The smart constructors already apply their rules at construction time
//! (identities, constant folding in the field, canonical operand order,
//! fused reductions), so a graph never holds `x + 0` or `1 * x`. All of
//! them keep the value bit for bit except two a graph is defined by: `0 * x`
//! is `0` for any `x`, and `(a^m)^n` is `a^(m n)`. What construction cannot see is the shape a graph has *after*
//! transformations: substitution, differentiation and inlining leave dead
//! nodes behind, and a symbol that became a constant can make identities
//! visible far above it. [`rebuild`] re-runs the constructors over the
//! reachable part of a graph, in topological order, into a fresh graph: one
//! pass that is dead-code elimination, renumbering, CSE and identity
//! propagation at once. Functions are rebuilt with it, so calls keep their
//! meaning.
//!
//! Value-changing rewrites (`ln(exp x) -> x`, `powf(x, 1/2) -> sqrt x`,
//! reassociation of floating sums) are deliberately not here: they would
//! break the bit-exactness contract between the graph and its backends.
//! The e-graph simplifier for symbolic work lives in `symbolic`.

use crate::field::Field;
use crate::func::{FuncId, Function, Output};
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};

/// Rebuild `g` over the nodes reachable from `roots` and from the outputs of
/// every function. Returns the new graph and the roots' new ids. Symbol ids
/// and function ids are preserved (every symbol and function is carried
/// over, in order), so callers keep their `SymbolId`s and `FuncId`s.
pub fn rebuild<K: Field>(g: &Graph<K>, roots: &[ExprId]) -> (Graph<K>, Vec<ExprId>) {
    let mut out: Graph<K> = Graph::new();
    for s in 0..g.n_symbols() {
        out.sym(g.symbol_name(SymbolId(s as u32)));
    }
    let mut map: Vec<Option<ExprId>> = vec![None; g.len()];
    // Functions in order: their outputs are rebuilt before any call to them
    // is rebuilt (a call references a function defined earlier).
    for f in 0..g.n_funcs() {
        let func = g.func(FuncId(f as u32));
        let mut nf = Function::new(
            func.name(),
            func.params().to_vec(),
            func.extern_body().cloned(),
        );
        for (k, &p) in func.param_roles().iter().enumerate() {
            nf.set_param_role(k as u32, p);
        }
        for (o, &role) in func.outputs().iter().zip(func.output_roles()) {
            let o = match *o {
                Output::Expr(e) => Output::Expr(copy(g, &mut out, e, &mut map)),
                other => other,
            };
            nf.push_output(o, role);
        }
        let id = out.push_function(nf);
        debug_assert_eq!(id.0 as usize, f);
    }
    let new_roots = roots
        .iter()
        .map(|&r| copy(g, &mut out, r, &mut map))
        .collect();
    (out, new_roots)
}

/// Copy `e` into `out` through the smart constructors, memoised in `map`.
/// Iterative post-order (a deep chain must not overflow the stack).
fn copy<K: Field>(
    g: &Graph<K>,
    out: &mut Graph<K>,
    e: ExprId,
    map: &mut [Option<ExprId>],
) -> ExprId {
    let mut stack: Vec<(ExprId, bool)> = vec![(e, false)];
    let mut ops: Vec<ExprId> = Vec::new();
    while let Some((cur, expanded)) = stack.pop() {
        if map[cur.0 as usize].is_some() {
            continue;
        }
        if !expanded {
            stack.push((cur, true));
            for &c in g.operands(cur).iter() {
                if map[c.0 as usize].is_none() {
                    stack.push((c, false));
                }
            }
            continue;
        }
        ops.clear();
        ops.extend(
            g.operands(cur)
                .iter()
                .map(|&c| map[c.0 as usize].expect("child copied")),
        );
        let id = match *g.node(cur) {
            Node::Const(c) => out.konst(g.const_val(c).clone()),
            Node::Call(o, _) => {
                let (f, k) = g.output(o);
                out.call(f, k, &ops)
            }
            // Symbol ids are carried over unchanged.
            node => out.build(node, &ops),
        };
        map[cur.0 as usize] = Some(id);
    }
    map[e.0 as usize].expect("root copied")
}
