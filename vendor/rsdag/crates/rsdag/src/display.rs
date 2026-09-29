use rustc_hash::FxHashMap as HashMap;

use crate::field::Field;
use crate::graph::Graph;
use crate::node::{ExprId, Node};

/// Expressions whose infix form stays under this many bytes print as one
/// infix string; a larger one prints as a listing, one line per node (a DAG
/// printed as a tree grows exponentially with its sharing).
const INFIX_MAX: usize = 4096;

/// Render an expression (raw, unsimplified): infix while that stays short,
/// else a listing `%k = ...` of the nodes it reaches, in dependency order,
/// the last line the root.
///
/// Good enough for inspection and tests. Pretty/canonical rendering of
/// `H(s)` as a collected rational function is a job for the rewrite layer.
pub fn to_string<K: Field>(ctx: &Graph<K>, id: ExprId) -> String {
    // The nodes under `id`, ascending (a dependency order), and the length
    // of each one's infix form, capped past the limit.
    let mut cone = Vec::new();
    let mut seen = rustc_hash::FxHashSet::default();
    let mut stack = vec![id];
    while let Some(e) = stack.pop() {
        if seen.insert(e) {
            cone.push(e);
            stack.extend_from_slice(&ctx.operands(e));
        }
    }
    cone.sort_unstable();
    let mut len: HashMap<ExprId, usize> = HashMap::default();
    for &e in &cone {
        let own = if ctx.operands(e).is_empty() {
            leaf(ctx, e).len()
        } else {
            8
        };
        let total = ctx
            .operands(e)
            .iter()
            .fold(own, |acc, c| acc.saturating_add(len[c]));
        len.insert(e, total.min(INFIX_MAX + 1));
    }
    let mut out = String::new();
    if len[&id] <= INFIX_MAX {
        infix(ctx, id, &mut out);
        return out;
    }
    let mut operand = |c: ExprId, _: bool, out: &mut String| {
        if ctx.operands(c).is_empty() {
            out.push_str(&leaf(ctx, c));
        } else {
            out.push_str(&format!("%{}", c.0));
        }
    };
    for &e in cone.iter().filter(|&&e| !ctx.operands(e).is_empty()) {
        out.push_str(&format!("%{} = ", e.0));
        write_node(ctx, e, &mut out, &mut operand);
        out.push('\n');
    }
    operand(id, false, &mut out);
    out
}

/// A leaf's text: a constant's value or a symbol's name.
fn leaf<K: Field>(ctx: &Graph<K>, e: ExprId) -> String {
    match *ctx.node(e) {
        Node::Const(c) => ctx.const_val(c).render(),
        Node::Symbol(s) => ctx.symbol_name(s).to_string(),
        _ => unreachable!("a leaf"),
    }
}

/// The infix form of `id`, operands written in place.
fn infix<K: Field>(ctx: &Graph<K>, id: ExprId, out: &mut String) {
    if ctx.operands(id).is_empty() {
        out.push_str(&leaf(ctx, id));
        return;
    }
    write_node(ctx, id, out, &mut |c, factor, out| {
        // A sum or a negation as a factor is wrapped in parens.
        let wrap = factor && matches!(ctx.node(c), Node::Add(..) | Node::Neg(..));
        if wrap {
            out.push('(');
        }
        infix(ctx, c, out);
        if wrap {
            out.push(')');
        }
    });
}

/// Node `id` with each operand written by `operand(c, as_factor, out)`.
fn write_node<K: Field>(
    ctx: &Graph<K>,
    id: ExprId,
    out: &mut String,
    operand: &mut dyn FnMut(ExprId, bool, &mut String),
) {
    let mut list = |name: &str, args: &[ExprId], out: &mut String| {
        out.push_str(name);
        out.push('(');
        for (i, &a) in args.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            operand(a, false, out);
        }
        out.push(')');
    };
    match *ctx.node(id) {
        Node::Const(_) | Node::Symbol(_) => out.push_str(&leaf(ctx, id)),
        Node::Add(a, b) => {
            out.push('(');
            operand(a, false, out);
            out.push_str(" + ");
            operand(b, false, out);
            out.push(')');
        }
        Node::Mul(a, b) => {
            operand(a, true, out);
            out.push('*');
            operand(b, true, out);
        }
        Node::Neg(a) => {
            out.push('-');
            operand(a, true, out);
        }
        Node::Pow(a, n) => {
            operand(a, true, out);
            out.push_str(&format!("^{n}"));
        }
        Node::Unary(op, a) => list(op.name(), &[a], out),
        Node::Binary(op, a, b) => list(op.name(), &[a, b], out),
        Node::Cmp(op, a, b) => {
            out.push('(');
            operand(a, false, out);
            out.push_str(&format!(" {} ", op.symbol()));
            operand(b, false, out);
            out.push(')');
        }
        Node::Select(c, t, e) => list("select", &[c, t, e], out),
        Node::Call(o, l) => {
            let (f, k) = ctx.output(o);
            list(&format!("{}#{k}", ctx.func(f).name()), ctx.args(l), out)
        }
        Node::Reduce(op, l) => list(op.name(), ctx.args(l), out),
        Node::Solve(l, i) => {
            let (n, _, _) = ctx.solve_args(l);
            out.push_str(&format!("solve{n}[{i}]"));
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            out.push_str("dot(");
            for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                if i > 0 {
                    out.push_str(" + ");
                }
                operand(x, true, out);
                out.push('*');
                operand(y, true, out);
            }
            out.push(')');
        }
    }
}
