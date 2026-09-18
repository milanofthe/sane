//! LaTeX export for SANE DAE systems -- the *symbolic* interface for human
//! reading. Numeric evaluation of the residual and Jacobians happens in Rust
//! (`sane_core::eval_real`, exposed via the Python bindings); SANE does not
//! generate code.

use num_traits::One;
use rsdag::{CmpOp, ExprId, Graph, Node, ReduceOp, UnaryOp};
use sane_dae::Dae;

/// The residual equations `0 = F_i(x, x', t)` as a LaTeX `aligned` block.
pub fn export_latex(ctx: &Graph, dae: &Dae) -> String {
    let mut s = String::new();
    s.push_str("\\begin{aligned}\n");
    for &r in &dae.residuals {
        s.push_str(&format!("0 &= {} \\\\\n", latex_expr(ctx, r)));
    }
    s.push_str("\\end{aligned}\n");
    s
}

/// Render a single expression to LaTeX.
pub fn latex_expr(ctx: &Graph, id: ExprId) -> String {
    match ctx.node(id) {
        Node::Const(c) => {
            let r = ctx.const_val(*c);
            if r.denom().is_one() {
                r.numer().to_string()
            } else {
                format!("\\frac{{{}}}{{{}}}", r.numer(), r.denom())
            }
        }
        Node::Symbol(s) => latex_name(ctx.symbol_name(*s)),
        Node::Add(a, b) => format!("{} + {}", latex_expr(ctx, *a), latex_expr(ctx, *b)),
        Node::Mul(a, b) => format!("{} \\cdot {}", latex_factor(ctx, *a), latex_factor(ctx, *b)),
        Node::Neg(a) => format!("-{}", latex_factor(ctx, *a)),
        Node::Pow(a, n) => {
            if *n == -1 {
                format!("\\frac{{1}}{{{}}}", latex_expr(ctx, *a))
            } else if *n < 0 {
                format!("\\frac{{1}}{{{}^{{{}}}}}", latex_factor(ctx, *a), -n)
            } else {
                format!("{}^{{{}}}", latex_factor(ctx, *a), n)
            }
        }
        Node::Unary(op, a) => {
            let inner = latex_expr(ctx, *a);
            match op {
                UnaryOp::Sqrt => format!("\\sqrt{{{inner}}}"),
                UnaryOp::Floor => format!("\\lfloor {inner} \\rfloor"),
                UnaryOp::Atan => format!("\\arctan\\left({inner}\\right)"),
                UnaryOp::Exp
                | UnaryOp::Ln
                | UnaryOp::Sin
                | UnaryOp::Cos
                | UnaryOp::Sinh
                | UnaryOp::Cosh
                | UnaryOp::Tanh => format!("\\{}\\left({inner}\\right)", op.name()),
                _ => format!("\\operatorname{{{}}}\\left({inner}\\right)", op.name()),
            }
        }
        Node::Cmp(op, a, b) => {
            let sym = match op {
                CmpOp::Gt => ">",
                CmpOp::Ge => "\\ge",
                CmpOp::Lt => "<",
                CmpOp::Le => "\\le",
                CmpOp::Eq => "=",
                CmpOp::Ne => "\\ne",
            };
            format!("[{} {} {}]", latex_expr(ctx, *a), sym, latex_expr(ctx, *b))
        }
        Node::Select(c, t, e) => format!(
            "\\mathrm{{select}}\\left({},\\, {},\\, {}\\right)",
            latex_expr(ctx, *c),
            latex_expr(ctx, *t),
            latex_expr(ctx, *e)
        ),
        Node::Binary(op, a, b) => format!(
            "\\operatorname{{{}}}\\left({}, {}\\right)",
            op.name(),
            latex_expr(ctx, *a),
            latex_expr(ctx, *b)
        ),
        Node::Solve(l, n) => format!(
            "\\operatorname{{solve}}_{{{n}}}\\left({}\\right)",
            ctx.args(*l)
                .iter()
                .map(|&a| latex_expr(ctx, a))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Node::Call(o, l) => {
            let inner: Vec<String> = ctx.args(*l).iter().map(|&a| latex_expr(ctx, a)).collect();
            let (f, k) = ctx.output(*o);
            format!(
                "{}_{{{k}}}\\left({}\\right)",
                latex_name(&ctx.func(f).name),
                inner.join(",\\, ")
            )
        }
        Node::Reduce(op, l) => {
            let inner: Vec<String> = ctx.args(*l).iter().map(|&a| latex_factor(ctx, a)).collect();
            match op {
                ReduceOp::Sum => inner.join(" + "),
                ReduceOp::Product => inner.join(" \\cdot "),
                ReduceOp::Min => format!("\\min\\left({}\\right)", inner.join(",\\, ")),
                ReduceOp::Max => format!("\\max\\left({}\\right)", inner.join(",\\, ")),
            }
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(*l);
            let terms: Vec<String> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| format!("{} \\cdot {}", latex_factor(ctx, x), latex_factor(ctx, y)))
                .collect();
            terms.join(" + ")
        }
    }
}

/// Render one or more expression roots as a Graphviz DOT digraph of the
/// hash-consed DAG. Because the graph is shared, a subexpression reached by
/// several parents appears as a single node with several incoming edges, so
/// common-subexpression sharing is visible directly; passing several roots
/// (e.g. a residual and its symbolic derivative) shows the structure they
/// share. Render with `dot -Tpdf`.
///
/// If `highlight` is non-empty, only the nodes reachable from those seed
/// expressions are drawn at full opacity; every other node (and the edges
/// between dimmed nodes) is faded to alpha ~0.2, so a transform's added nodes
/// stand out against the rest of the graph.
pub fn export_dot(ctx: &Graph, roots: &[(ExprId, String)], highlight: &[ExprId]) -> String {
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut order: Vec<ExprId> = Vec::new();
    fn visit(
        ctx: &Graph,
        id: ExprId,
        seen: &mut std::collections::HashSet<u32>,
        order: &mut Vec<ExprId>,
    ) {
        if !seen.insert(id.0) {
            return;
        }
        for op in operands(ctx, id) {
            visit(ctx, op, seen, order);
        }
        order.push(id);
    }
    for (r, _) in roots {
        visit(ctx, *r, &mut seen, &mut order);
    }

    // nodes reachable from the highlight seeds stay at full opacity
    let mut hi: std::collections::HashSet<u32> = std::collections::HashSet::new();
    if !highlight.is_empty() {
        let mut h_order = Vec::new();
        for &h in highlight {
            visit(ctx, h, &mut hi, &mut h_order);
        }
    }
    let dim = |id: u32| !highlight.is_empty() && !hi.contains(&id);

    let mut s = String::from("digraph G {\n  rankdir=TB;\n  node [fontname=\"Helvetica\"];\n");
    s.push_str("  edge [arrowsize=0.7];\n");
    // every expression node is drawn by its operator/leaf style (the root keeps
    // its operator symbol, e.g. the Reduce sum stays a Sigma); dimmed nodes get
    // an alpha suffix on the fill plus a faded border and font.
    for &id in &order {
        let (label, shape, fill) = dot_node_style(ctx, id);
        let (fill, font, pen) = if dim(id.0) {
            (
                format!("{fill}33"),
                ", fontcolor=\"#00000033\"",
                ", color=\"#00000033\"",
            )
        } else {
            (fill.to_string(), "", "")
        };
        s.push_str(&format!(
            "  n{} [label=\"{}\", shape={}, style=filled, fillcolor=\"{}\"{}{}];\n",
            id.0,
            dot_escape(&label),
            shape,
            fill,
            font,
            pen,
        ));
    }
    // Edges run operand -> parent, so leaves rise to the top and the root
    // expression sinks to the bottom (data flows down toward the residual). An
    // edge is faded if either end is dimmed, so only edges inside the
    // highlighted subgraph stay solid.
    for &id in &order {
        for op in operands(ctx, id) {
            let faded = if dim(id.0) || dim(op.0) {
                " [color=\"#00000033\"]"
            } else {
                ""
            };
            s.push_str(&format!("  n{} -> n{}{};\n", op.0, id.0, faded));
        }
    }
    // The residual itself is a named endpoint: a red box below the root
    // expression, fed by it. ("F = ... = 0".)
    for (i, (r, name)) in roots.iter().enumerate() {
        if !name.is_empty() {
            s.push_str(&format!(
                "  res{} [label=\"{}\", shape=box, style=filled, fillcolor=\"#E89A9A\"];\n  n{} -> res{};\n",
                i, dot_escape(name), r.0, i
            ));
        }
    }
    s.push_str("}\n");
    s
}

/// The child expression ids of a node, in operand order.
fn operands(ctx: &Graph, id: ExprId) -> Vec<ExprId> {
    ctx.operands(id).to_vec()
}

/// (label, graphviz shape, fill colour) for a node, by variant. Colours follow
/// the SANE palette: green leaves are symbols, grey leaves constants, orange a
/// black-box (opaque) node, blue the operators.
fn dot_node_style(ctx: &Graph, id: ExprId) -> (String, &'static str, &'static str) {
    match ctx.node(id) {
        // a constant: white box (a fixed numeric literal), distinct from the
        // grey operators, green parameters, and blue state symbols.
        Node::Const(c) => {
            use num_traits::ToPrimitive;
            let r = ctx.const_val(*c);
            let label = match (r.denom().is_one(), r.numer().to_i64()) {
                (true, Some(n)) => n.to_string(),
                _ => fmt_num(r.to_f64().unwrap_or(f64::NAN)),
            };
            (label, "box", CONST_FILL)
        }
        // a free symbol as its plain name (the descriptor, e.g. R1, D1.Is,
        // $temp, v1): state (node voltage / derivative / branch current) is a
        // light-blue box, a parameter a light-green box.
        Node::Symbol(s) => {
            let name = ctx.symbol_name(*s);
            (
                name.to_string(),
                "box",
                if is_state(name) {
                    STATE_FILL
                } else {
                    PARAM_FILL
                },
            )
        }
        // operators: light-grey nodes. The arity-bearing ones carry a placeholder
        // dot so it is clear where the operand goes, e.g. (.)^-1, exp(.).
        Node::Add(..) => ("+".into(), "circle", OP_FILL),
        Node::Mul(..) => ("\u{00d7}".into(), "circle", OP_FILL), // x
        Node::Neg(..) => ("\u{2212}(\u{00b7})".into(), "ellipse", OP_FILL), // -(.)
        Node::Pow(_, n) => (format!("(\u{00b7})^{n}"), "ellipse", OP_FILL), // (.)^-1
        Node::Unary(op, _) => {
            let l = format!("{}(\u{00b7})", op.name());
            (l, "ellipse", OP_FILL)
        }
        Node::Cmp(op, ..) => {
            let sym = match op {
                CmpOp::Gt => ">",
                CmpOp::Ge => ">=",
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Eq => "=",
                CmpOp::Ne => "!=",
            };
            (format!("(\u{00b7}) {sym} (\u{00b7})"), "ellipse", OP_FILL)
        }
        // a decision (region select): light-yellow ellipse, kept visually distinct.
        Node::Select(..) => (
            "(\u{00b7}) ? (\u{00b7}) : (\u{00b7})".into(),
            "ellipse",
            SELECT_FILL,
        ),
        Node::Reduce(op, _) => {
            let sym = match op {
                ReduceOp::Sum => "\u{03a3}",     // Sigma
                ReduceOp::Product => "\u{03a0}", // Pi
                ReduceOp::Min => "min",
                ReduceOp::Max => "max",
            };
            (sym.to_string(), "circle", OP_FILL)
        }
        Node::Dot(..) => ("<\u{00b7},\u{00b7}>".into(), "ellipse", OP_FILL),
        Node::Binary(op, ..) => (op.name().into(), "ellipse", OP_FILL),
        Node::Solve(_, n) => (format!("solve_{n}"), "ellipse", OP_FILL),
        // black-box (opaque) device value: orange box, kept distinct.
        Node::Call(o, _) => {
            let (f, k) = ctx.output(*o);
            (format!("{}#{k}", ctx.func(f).name), "box", "#E8B06A")
        }
    }
}

// node fill colours (light, by role)
const CONST_FILL: &str = "#FFFFFF"; // constants: white
const STATE_FILL: &str = "#CFE7F0"; // node voltages / derivatives / branch currents: light blue
const PARAM_FILL: &str = "#D7EAC8"; // parameters: light green
const OP_FILL: &str = "#E0E0E0"; //    operators: light grey
const SELECT_FILL: &str = "#EDD9A3"; // decision (Select) nodes: light yellow

fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Whether a symbol is a state variable (drawn light blue) rather than a
/// parameter (light green). State names follow the assembler's convention:
/// `v{k}` node voltage, `vdot{k}` its derivative, `i_{x}` a branch current,
/// `idot_{x}` its derivative, and `t` time.
fn is_state(name: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if let Some(r) = name.strip_prefix("vdot") {
        return digits(r);
    }
    if name.starts_with("idot_") || name.starts_with("i_") || name == "t" {
        return true;
    }
    matches!(name.strip_prefix('v'), Some(r) if digits(r))
}

/// Compact label for a constant: a short decimal for moderate magnitudes,
/// scientific for very small/large. (The arena keeps the exact rational; this is
/// only the picture.)
fn fmt_num(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if v.fract() == 0.0 && v.abs() < 1e6 {
        return format!("{}", v as i64);
    }
    if v.abs() >= 1e-3 && v.abs() < 1e5 {
        let s = format!("{v:.4}");
        return s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    format!("{v:.2e}")
}

/// Wrap sums/negations in parentheses when they appear as a factor or base.
fn latex_factor(ctx: &Graph, id: ExprId) -> String {
    match ctx.node(id) {
        Node::Add(..) | Node::Neg(..) => format!("\\left({}\\right)", latex_expr(ctx, id)),
        _ => latex_expr(ctx, id),
    }
}

fn latex_name(name: &str) -> String {
    format!("\\mathrm{{{}}}", name.replace('_', "\\_"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsdag::Graph;
    use sane_dae::{assemble_dae, DeviceInstance};
    use sane_mna::Circuit;

    #[test]
    fn latex_export_shape() {
        let mut ctx = Graph::new();
        let mut c = Circuit::new();
        c.voltage_source("V1", 1, 0).resistor("R", 1, 2);
        let devs = vec![DeviceInstance::new(
            Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
            vec![2, 0],
        )];
        let dae = assemble_dae(&mut ctx, &c, &devs);
        let tex = export_latex(&ctx, &dae);
        assert!(tex.contains("\\begin{aligned}"));
        assert!(tex.contains("\\exp\\left"));
        assert_eq!(tex.matches("0 &=").count(), dae.dim());
    }
}
