use crate::field::Field;
use crate::graph::Graph;
use crate::node::{BinOp, CmpOp, ExprId, Node, ReduceOp, UnaryOp};

/// Render an expression to an infix string (raw, unsimplified).
///
/// Good enough for inspection and tests. Pretty/canonical rendering of
/// `H(s)` as a collected rational function is a job for the rewrite layer.
pub fn to_string<K: Field>(ctx: &Graph<K>, id: ExprId) -> String {
    let mut out = String::new();
    write_expr(ctx, id, &mut out);
    out
}

fn write_expr<K: Field>(ctx: &Graph<K>, id: ExprId, out: &mut String) {
    match ctx.node(id) {
        Node::Const(c) => {
            out.push_str(&ctx.const_val(*c).render());
        }
        Node::Symbol(s) => out.push_str(ctx.symbol_name(*s)),
        Node::Add(a, b) => {
            out.push('(');
            write_expr(ctx, *a, out);
            out.push_str(" + ");
            write_expr(ctx, *b, out);
            out.push(')');
        }
        Node::Mul(a, b) => {
            write_factor(ctx, *a, out);
            out.push('*');
            write_factor(ctx, *b, out);
        }
        Node::Neg(a) => {
            out.push('-');
            write_factor(ctx, *a, out);
        }
        Node::Pow(a, n) => {
            write_factor(ctx, *a, out);
            out.push_str(&format!("^{n}"));
        }
        Node::Unary(op, a) => {
            out.push_str(UnaryOp::name(*op));
            out.push('(');
            write_expr(ctx, *a, out);
            out.push(')');
        }
        Node::Binary(op, a, b) => {
            out.push_str(match op {
                BinOp::Powf => "powf",
                BinOp::Mod => "mod",
                BinOp::Atan2 => "atan2",
                BinOp::Hypot => "hypot",
            });
            out.push('(');
            write_expr(ctx, *a, out);
            out.push_str(", ");
            write_expr(ctx, *b, out);
            out.push(')');
        }
        Node::Cmp(op, a, b) => {
            out.push('(');
            write_expr(ctx, *a, out);
            out.push_str(match op {
                CmpOp::Gt => " > ",
                CmpOp::Ge => " >= ",
                CmpOp::Lt => " < ",
                CmpOp::Le => " <= ",
                CmpOp::Eq => " == ",
                CmpOp::Ne => " != ",
            });
            write_expr(ctx, *b, out);
            out.push(')');
        }
        Node::Select(c, t, e) => {
            out.push_str("select(");
            write_expr(ctx, *c, out);
            out.push_str(", ");
            write_expr(ctx, *t, out);
            out.push_str(", ");
            write_expr(ctx, *e, out);
            out.push(')');
        }
        Node::Call(o, l) => {
            let (f, k) = ctx.output(*o);
            out.push_str(&format!("{}#{k}", ctx.func(f).name));
            out.push('(');
            for (i, &a) in ctx.args(*l).iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_expr(ctx, a, out);
            }
            out.push(')');
        }
        Node::Reduce(op, l) => {
            let args = ctx.args(*l);
            out.push_str(match op {
                ReduceOp::Sum => "sum",
                ReduceOp::Product => "prod",
                ReduceOp::Min => "min",
                ReduceOp::Max => "max",
            });
            out.push('(');
            for (i, &a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_expr(ctx, a, out);
            }
            out.push(')');
        }
        Node::Solve(l, i) => {
            let (n, _, _) = ctx.solve_args(*l);
            out.push_str(&format!("solve{n}[{i}]"));
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(*l);
            out.push_str("dot(");
            for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                if i > 0 {
                    out.push_str(" + ");
                }
                write_factor(ctx, x, out);
                out.push('*');
                write_factor(ctx, y, out);
            }
            out.push(')');
        }
    }
}

/// Wrap sums in parens when they appear as a factor.
fn write_factor<K: Field>(ctx: &Graph<K>, id: ExprId, out: &mut String) {
    match ctx.node(id) {
        Node::Add(..) | Node::Neg(..) => {
            out.push('(');
            write_expr(ctx, id, out);
            out.push(')');
        }
        _ => write_expr(ctx, id, out),
    }
}
