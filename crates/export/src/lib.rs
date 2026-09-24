//! LaTeX export for SANE DAE systems -- the *symbolic* interface for human
//! reading. Numeric evaluation of the residual and Jacobians happens in Rust
//! (`sane_core::eval_real`, exposed via the Python bindings); SANE does not
//! generate code.

use num_traits::One;
use rsdag::{CmpOp, ExprId, Node, ReduceOp, UnaryOp};
use sane_core::Graph;
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
/// hash-consed DAG ([`rsdag::dot::GraphView`] in [`theme`]). Because the
/// graph is shared, a subexpression reached by several parents appears as a
/// single node with several incoming edges, so common-subexpression sharing
/// is visible directly; passing several roots (e.g. a residual and its
/// symbolic derivative) shows the structure they share. A named root gets a
/// residual box below it. Render with `dot -Tpdf`.
///
/// If `highlight` is non-empty, only the nodes reachable from those seed
/// expressions are drawn at full opacity; every other node (and the edges
/// between dimmed nodes) is faded, so a transform's added nodes stand out
/// against the rest of the graph.
pub fn export_dot(ctx: &Graph, roots: &[(ExprId, String)], highlight: &[ExprId]) -> String {
    use rsdag::dot::{reachable, GraphView};
    let all: Vec<ExprId> = roots.iter().map(|(r, _)| *r).collect();
    // Symbols that are not states are parameters.
    let params: Vec<rsdag::SymbolId> = reachable(ctx, &all)
        .into_iter()
        .filter_map(|e| match ctx.node(e) {
            Node::Symbol(s) if !is_state(ctx.symbol_name(*s)) => Some(*s),
            _ => None,
        })
        .collect();
    let mut view = GraphView::new(ctx).theme(theme()).params(&params);
    for (r, name) in roots {
        view = view.root(*r, name);
    }
    if !highlight.is_empty() {
        view = view.focus(reachable(ctx, highlight));
    }
    view.render()
}

/// The SANE palette as a DOT theme: black lines and text, a pale fill per
/// role (light blue states, light green parameters, white constants, grey
/// operators, yellow decisions, orange device calls, red residuals) and
/// mathematical operator labels. The web graph view reads the role from the
/// fill.
pub fn theme() -> rsdag::dot::Theme {
    rsdag::dot::Theme {
        style: rsdag::dot::Style::Filled,
        font: "Helvetica",
        font_size: 11.0,
        text: "#000000",
        line: Some("#000000"),
        fill_alpha: "",
        fade_alpha: "33",
        fade_fill_alpha: "33",
        notation: rsdag::dot::Notation::Math,
        input: "#CFE7F0",
        param: "#D7EAC8",
        constant: "#FFFFFF",
        op: "#E0E0E0",
        choice: "#EDD9A3",
        kernel: "#E0E0E0",
        call: "#E8B06A",
        output: "#E89A9A",
        state: "#000000",
        ..rsdag::dot::Theme::default()
    }
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
    use sane_core::Graph;
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

    /// The web graph view reads a node's role from its fill, six hex
    /// digits: states light blue, parameters light green, residuals red.
    #[test]
    fn dot_export_tags_roles_by_fill() {
        let mut ctx = Graph::new();
        let mut c = Circuit::new();
        c.voltage_source("V1", 1, 0).resistor("R", 1, 2);
        let devs = vec![DeviceInstance::new(
            Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
            vec![2, 0],
        )];
        let dae = assemble_dae(&mut ctx, &c, &devs);
        let roots: Vec<(ExprId, String)> = dae
            .residuals
            .iter()
            .enumerate()
            .map(|(k, &r)| (r, format!("F[{k}]")))
            .collect();
        let dot = export_dot(&ctx, &roots, &[]);
        let fill_of = |label: &str| {
            let line = dot
                .lines()
                .find(|l| l.contains(&format!("label=\"{label}\"")))
                .unwrap_or_else(|| panic!("no node {label}: {dot}"));
            let at = line.find("fillcolor=\"").expect("a fill") + 11;
            line[at..at + 8].to_string()
        };
        assert_eq!(fill_of("v1"), "#CFE7F0\"");
        assert_eq!(fill_of("F[0]"), "#E89A9A\"");
        assert!(dot.contains("fillcolor=\"#D7EAC8\""), "a parameter: {dot}");
        assert_eq!(dot.matches("fillcolor=\"#E89A9A\"").count(), dae.dim());
    }
}
