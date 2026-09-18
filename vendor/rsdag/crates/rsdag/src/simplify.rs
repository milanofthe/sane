//! Graph simplification.
//!
//! The smart constructors already apply every bit-preserving rule at
//! construction time (identities, constant folding in the field, canonical
//! operand order, fused reductions), so a graph never holds `x + 0` or
//! `1 * x`. What construction cannot see is the shape a graph has *after*
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
use crate::func::{FunctionBody, Output};
use crate::graph::Graph;
use crate::node::{ExprId, Node};

/// Rebuild `g` over the nodes reachable from `roots` and from the outputs of
/// every function. Returns the new graph and the roots' new ids. Symbol ids
/// and function ids are preserved (every symbol and function is carried
/// over, in order), so callers keep their `SymbolId`s and `FuncId`s.
pub fn rebuild<K: Field>(g: &Graph<K>, roots: &[ExprId]) -> (Graph<K>, Vec<ExprId>) {
    let mut out: Graph<K> = Graph::new();
    for s in 0..g.n_symbols() {
        let name = g.symbol_name(crate::node::SymbolId(s as u32)).to_string();
        out.sym(&name);
    }
    let mut map: Vec<Option<ExprId>> = vec![None; g.len()];
    // Functions in order: their outputs are rebuilt before any call to them
    // is rebuilt (a call references a function defined earlier).
    for f in 0..g.n_funcs() {
        let func = g.func(crate::func::FuncId(f as u32));
        let mut outputs = Vec::with_capacity(func.outputs.len());
        for o in &func.outputs {
            outputs.push(match *o {
                Output::Expr(e) => Output::Expr(copy(g, &mut out, e, &mut map)),
                other => other,
            });
        }
        let id = match &func.body {
            FunctionBody::Symbolic => {
                let exprs: Vec<ExprId> = outputs
                    .iter()
                    .map(|o| match o {
                        Output::Expr(e) => *e,
                        _ => out.zero(),
                    })
                    .collect();
                let id = out.define_func(&func.name, func.params.clone(), exprs);
                // Zero outputs stay zero outputs.
                let nf = out.func_mut(id);
                for (k, o) in outputs.iter().enumerate() {
                    if matches!(o, Output::Zero) {
                        nf.outputs[k] = Output::Zero;
                    }
                }
                id
            }
            FunctionBody::Extern(b) => out.define_extern_func_with_params(
                &func.name,
                func.params.clone(),
                b.clone(),
                outputs,
            ),
        };
        debug_assert_eq!(id.0 as usize, f);
        let nf = out.func_mut(id);
        nf.param_roles = func.param_roles.clone();
        nf.output_roles = func.output_roles.clone();
        nf.deriv_index = func.deriv_index.clone();
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
    if let Some(id) = map[e.0 as usize] {
        return id;
    }
    let mut stack: Vec<(ExprId, bool)> = vec![(e, false)];
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
        let m = |c: ExprId| map[c.0 as usize].expect("child copied");
        let node = *g.node(cur);
        let id = out.rebuild_node(
            &node,
            |n| match *n {
                Node::Const(c) => g.const_val(c).clone(),
                _ => unreachable!("only a constant asks for its value"),
            },
            |s| s,
            m,
            |l| g.args(l).iter().map(|&a| m(a)).collect(),
            |o| g.output(o),
        );
        map[cur.0 as usize] = Some(id);
    }
    map[e.0 as usize].expect("root copied")
}
