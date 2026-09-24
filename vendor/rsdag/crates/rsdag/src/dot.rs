//! Graphviz DOT of a graph, of a tape, and of diagrams drawn in the same
//! [`Theme`].
//!
//! [`GraphView`] draws the hash-consed DAG: a shared subexpression is one
//! node with several parents. A focus set keeps its nodes at full strength
//! and fades the rest (the nodes a transform added, the arms a
//! specialization keeps), clusters group nodes (a function body), links add
//! dashed edges between nodes. [`TapeView`] draws a program's dataflow, one
//! node per instruction, the prolog and the main phase as two clusters and
//! the values the prolog leaves in the state as dashed edges.
//!
//! Render with `dot -Tsvg`.

use std::fmt::Write;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::field::Field;
use crate::graph::Graph;
use crate::node::{CmpOp, ExprId, Node, ReduceOp, SymbolId};
use crate::tape::{input_index, Tape};

/// What a node is, which decides its hue and shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A varying input: a state, an unknown.
    Input,
    /// A solve-constant input: a parameter.
    Param,
    Const,
    /// Arithmetic and elementary functions.
    Op,
    /// A comparison or a select: where a program branches.
    Choice,
    /// A fused kernel: a reduction, a dot, a product or a dense solve.
    Kernel,
    /// A call of a function body.
    Call,
    /// A named result.
    Output,
}

/// How a graph's operators are labelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Notation {
    /// `*`, `neg`, `^-1`, `exp`, `>`, `select`, `sum`, `dot`.
    Ascii,
    /// `×`, `−(·)`, `(·)^-1`, `exp(·)`, `(·) > (·)`, `(·) ? (·) : (·)`,
    /// `Σ`, `<·,·>`.
    Math,
}

/// How nodes are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// Lines only, no fill: every node in its hue as line and text, a
    /// constant as plain text, a result with a bold label, clusters as
    /// solid frames with a bold title.
    Outline,
    /// A fill per node kind under one line color (or the hue), clusters as
    /// dashed frames.
    Filled,
}

/// Colors and fonts of every diagram. The default is rsdag's: a
/// transparent page, everything in one grey, the branches, guards and the
/// state in one accent, lines only; it reads on a light and a dark page
/// alike.
#[derive(Clone, Debug)]
pub struct Theme {
    pub style: Style,
    pub font: &'static str,
    pub font_size: f64,
    /// A block's title ([`Blocks`]), bold.
    pub title_size: f64,
    /// Labels, edges and cluster frames.
    pub text: &'static str,
    /// Notes and emphasis ([`Blocks`]).
    pub accent: &'static str,
    /// One line color for every node; `None` draws a node in its hue.
    pub line: Option<&'static str>,
    /// Alpha of a node's fill, two hex digits (empty: the hue as is).
    pub fill_alpha: &'static str,
    /// Alpha of a faded node's line, text and edges, two hex digits.
    pub fade_alpha: &'static str,
    /// Alpha of a faded node's fill, two hex digits.
    pub fade_fill_alpha: &'static str,
    pub notation: Notation,
    pub input: &'static str,
    pub param: &'static str,
    pub constant: &'static str,
    pub op: &'static str,
    pub choice: &'static str,
    pub kernel: &'static str,
    pub call: &'static str,
    pub output: &'static str,
    /// Edges carrying the prolog's state into the main phase.
    pub state: &'static str,
}

const GREY: &str = "#8b8b8b";
const ACCENT: &str = "#c55a11";

impl Default for Theme {
    fn default() -> Self {
        Theme {
            style: Style::Outline,
            font: "Helvetica,Arial,sans-serif",
            font_size: 12.0,
            title_size: 14.0,
            text: GREY,
            accent: ACCENT,
            line: None,
            fill_alpha: "",
            fade_alpha: "40",
            fade_fill_alpha: "",
            notation: Notation::Ascii,
            input: GREY,
            param: GREY,
            constant: GREY,
            op: GREY,
            choice: ACCENT,
            kernel: GREY,
            call: GREY,
            output: GREY,
            state: ACCENT,
        }
    }
}

impl Theme {
    pub fn hue(&self, kind: Kind) -> &'static str {
        match kind {
            Kind::Input => self.input,
            Kind::Param => self.param,
            Kind::Const => self.constant,
            Kind::Op => self.op,
            Kind::Choice => self.choice,
            Kind::Kernel => self.kernel,
            Kind::Call => self.call,
            Kind::Output => self.output,
        }
    }

    /// The opening of a digraph in this theme, `rankdir` `TB` or `LR`.
    pub fn header(&self, rankdir: &str) -> String {
        let (arrow, pen, margin, edge_font) = match self.style {
            Style::Outline => (0.8, 1.2, "0.1,0.04", 1.0),
            Style::Filled => (0.6, 0.9, "0.08,0.03", 2.0),
        };
        format!(
            "digraph G {{\n  bgcolor=\"transparent\";\n  rankdir={rankdir};\n  \
             nodesep=0.25;\n  ranksep=0.35;\n  compound=true;\n  \
             fontname=\"{f}\";\n  fontsize={s};\n  fontcolor=\"{t}\";\n  \
             node [fontname=\"{f}\", fontsize={s}, fontcolor=\"{t}\", penwidth=1.2, \
             margin=\"{margin}\", height=0.3];\n  \
             edge [color=\"{t}\", fontname=\"{f}\", fontsize={e}, fontcolor=\"{t}\", \
             arrowsize={arrow}, penwidth={pen}];\n",
            f = self.font,
            s = self.font_size,
            e = self.font_size - edge_font,
            t = self.text,
        )
    }

    /// The attributes of a node of `kind` labelled `label`, faded or not.
    /// An operator or kernel with a label of at most two characters is a
    /// circle.
    pub fn node(&self, kind: Kind, label: &str, faded: bool) -> String {
        match self.style {
            Style::Outline => self.outline_node(kind, label, faded),
            Style::Filled => self.filled_node(kind, label, faded),
        }
    }

    fn outline_node(&self, kind: Kind, label: &str, faded: bool) -> String {
        let hue = self.hue(kind);
        let a = if faded { self.fade_alpha } else { "" };
        let shape = match kind {
            Kind::Op | Kind::Choice | Kind::Kernel if label.chars().count() <= 2 => {
                "circle, width=0.3, fixedsize=false"
            }
            Kind::Input | Kind::Param | Kind::Output => "box, style=\"rounded\"",
            Kind::Kernel => "box",
            Kind::Call => "component",
            Kind::Const => "plaintext",
            Kind::Op | Kind::Choice => "ellipse",
        };
        let label = if kind == Kind::Output {
            format!("<<B>{}</B>>", html(label))
        } else {
            format!("\"{}\"", escape(label))
        };
        format!("label={label}, shape={shape}, color=\"{hue}{a}\", fontcolor=\"{hue}{a}\"")
    }

    fn filled_node(&self, kind: Kind, label: &str, faded: bool) -> String {
        let hue = self.hue(kind);
        let line = self.line.unwrap_or(hue);
        let shape = match kind {
            Kind::Op | Kind::Choice | Kind::Kernel if label.chars().count() <= 2 => {
                "circle, style=\"filled\", width=0.3, fixedsize=false"
            }
            Kind::Input | Kind::Param | Kind::Output | Kind::Call | Kind::Kernel => {
                "box, style=\"rounded,filled\""
            }
            Kind::Const => "box, style=\"filled\"",
            Kind::Op | Kind::Choice => "ellipse, style=\"filled\"",
        };
        if faded {
            format!(
                "label=\"{}\", shape={shape}, color=\"{line}{a}\", fillcolor=\"{hue}{}\", \
                 fontcolor=\"{}{a}\"",
                escape(label),
                self.fade_fill_alpha,
                self.text,
                a = self.fade_alpha,
            )
        } else {
            format!(
                "label=\"{}\", shape={shape}, color=\"{line}\", fillcolor=\"{hue}{}\"",
                escape(label),
                self.fill_alpha,
            )
        }
    }

    /// The attributes of a cluster labelled `label`.
    pub fn cluster(&self, label: &str) -> String {
        match self.style {
            Style::Outline => format!(
                "label=<<B>{}</B>>; labeljust=l; fontsize={}; style=\"rounded\"; \
                 color=\"{}\"; penwidth=1.2;",
                html(label),
                self.title_size,
                self.text
            ),
            Style::Filled => format!(
                "label=\"{}\"; labeljust=l; style=\"rounded,dashed\"; color=\"{}\"; penwidth=0.8;",
                escape(label),
                self.text
            ),
        }
    }

    fn faded_edge(&self) -> String {
        format!("color=\"{}{}\"", self.text, self.fade_alpha)
    }
}

/// Text for a Graphviz HTML-like label: `&`, `<`, `>` escaped.
pub fn html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// An architecture diagram in a [`Theme`]: blocks with a bold title over
/// lines of text, edges, groups of blocks, notes in the accent, and
/// sparsity patterns as grids.
pub struct Blocks {
    theme: Theme,
    body: String,
}

impl Blocks {
    pub fn new(theme: Theme, rankdir: &str) -> Self {
        let mut body = theme.header(rankdir);
        // Blocks breathe more than expression nodes, and a column of them
        // lines up at one width.
        body.push_str("  graph [ranksep=0.55, nodesep=0.3];\n  node [width=1.5];\n");
        Blocks { theme, body }
    }

    fn label(&self, title: &str, lines: &[&str]) -> String {
        let mut l = format!(
            "<<FONT POINT-SIZE=\"{}\"><B>{}</B></FONT>",
            self.theme.title_size,
            html(title)
        );
        for line in lines {
            l.push_str("<BR/>");
            match line.strip_prefix("**").and_then(|r| r.strip_suffix("**")) {
                Some(bold) => {
                    let _ = write!(l, "<B>{}</B>", html(bold));
                }
                None => l.push_str(&html(line)),
            }
        }
        l.push('>');
        l
    }

    /// A block: `title` in bold over `lines` (a line wrapped in `**` bold).
    pub fn block(mut self, id: &str, title: &str, lines: &[&str]) -> Self {
        let label = self.label(title, lines);
        let _ = writeln!(
            self.body,
            "  {id} [label={label}, shape=box, style=\"rounded\", color=\"{}\", \
             margin=\"0.18,0.1\"];",
            self.theme.text
        );
        self
    }

    /// A note in the accent, dashed, `title` in bold over `lines`.
    pub fn note(mut self, id: &str, title: &str, lines: &[&str]) -> Self {
        let label = self.label(title, lines);
        let a = self.theme.accent;
        let _ = writeln!(
            self.body,
            "  {id} [label={label}, shape=box, style=\"rounded,dashed\", color=\"{a}\", \
             fontcolor=\"{a}\", margin=\"0.18,0.1\"];"
        );
        self
    }

    /// Plain text.
    pub fn text(mut self, id: &str, lines: &[&str]) -> Self {
        let label: Vec<String> = lines.iter().map(|l| html(l)).collect();
        let _ = writeln!(
            self.body,
            "  {id} [label=<{}>, shape=plaintext];",
            label.join("<BR/>")
        );
        self
    }

    /// A sparsity pattern as a grid, a mark per entry (any character but
    /// `.` and space), `caption` below.
    pub fn pattern(mut self, id: &str, rows: &[&str], caption: &str) -> Self {
        let t = self.theme.text;
        let mut l = String::from("<<TABLE BORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\">");
        for row in rows {
            l.push_str("<TR>");
            for c in row.chars() {
                let dot = if c == '.' || c == ' ' {
                    String::new()
                } else {
                    format!(
                        "<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR>\
                         <TD WIDTH=\"6\" HEIGHT=\"6\" FIXEDSIZE=\"TRUE\" BGCOLOR=\"{t}\"></TD>\
                         </TR></TABLE>"
                    )
                };
                let _ = write!(
                    l,
                    "<TD WIDTH=\"14\" HEIGHT=\"14\" FIXEDSIZE=\"TRUE\" BORDER=\"1\" \
                     COLOR=\"{t}\">{dot}</TD>"
                );
            }
            l.push_str("</TR>");
        }
        let _ = write!(
            l,
            "<TR><TD COLSPAN=\"{}\" CELLPADDING=\"4\">{}</TD></TR></TABLE>>",
            rows.first().map_or(1, |r| r.chars().count()),
            html(caption)
        );
        let _ = writeln!(self.body, "  {id} [label={l}, shape=plaintext];");
        self
    }

    /// A frame around `ids`, `title` in bold.
    pub fn group(mut self, title: &str, ids: &[&str]) -> Self {
        let n = self.body.matches("subgraph cluster_").count();
        let _ = writeln!(
            self.body,
            "  subgraph cluster_{n} {{\n    {}\n    {};\n  }}",
            self.theme.cluster(title),
            ids.join("; ")
        );
        self
    }

    /// An edge, labelled when `label` is not empty.
    pub fn edge(mut self, a: &str, b: &str, label: &str) -> Self {
        let _ = writeln!(self.body, "  {a} -> {b} [label=\"{}\"];", escape(label));
        self
    }

    /// A dashed edge in the accent: a note's reference, a feedback.
    pub fn accent_edge(mut self, a: &str, b: &str, label: &str) -> Self {
        let c = self.theme.accent;
        let _ = writeln!(
            self.body,
            "  {a} -> {b} [label=\"{}\", style=dashed, color=\"{c}\", fontcolor=\"{c}\"];",
            escape(label)
        );
        self
    }

    /// A dashed feedback edge in the accent from `a` back to `b`, not
    /// taking part in the ranking.
    pub fn back(mut self, a: &str, b: &str, label: &str) -> Self {
        let c = self.theme.accent;
        let _ = writeln!(
            self.body,
            "  {a} -> {b} [label=\"{}\", style=dashed, color=\"{c}\", fontcolor=\"{c}\", \
             constraint=false];",
            escape(label)
        );
        self
    }

    /// A line of text under the whole diagram.
    pub fn caption(mut self, text: &str) -> Self {
        let _ = writeln!(
            self.body,
            "  label=\"{}\"; labelloc=b; labeljust=c;",
            escape(text)
        );
        self
    }

    /// Any DOT statement.
    pub fn raw(mut self, line: &str) -> Self {
        self.body.push_str("  ");
        self.body.push_str(line);
        self.body.push('\n');
        self
    }

    pub fn render(mut self) -> String {
        self.body.push_str("}\n");
        self.body
    }
}

/// A label with its quotes and backslashes escaped and its line breaks
/// as DOT's.
pub fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// A number as a label: an integer as one, else four significant digits,
/// in exponent form beyond `1e5` or below `1e-3`.
pub fn number(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        return format!("{}", x as i64);
    }
    let a = x.abs();
    if !(1e-3..1e5).contains(&a) {
        return format!("{x:.3e}");
    }
    let s = format!("{x:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// The nodes `roots` reach, themselves included.
pub fn reachable<K: Field>(g: &Graph<K>, roots: &[ExprId]) -> FxHashSet<ExprId> {
    let mut seen = FxHashSet::default();
    let mut stack: Vec<ExprId> = roots.to_vec();
    while let Some(e) = stack.pop() {
        if seen.insert(e) {
            stack.extend(g.operands(e).iter().copied());
        }
    }
    seen
}

/// The DOT of a graph's DAG under some roots (see the module docs).
pub struct GraphView<'g, K: Field> {
    g: &'g Graph<K>,
    theme: Theme,
    rankdir: &'static str,
    roots: Vec<(ExprId, String)>,
    params: FxHashSet<SymbolId>,
    focus: Option<FxHashSet<ExprId>>,
    clusters: Vec<(String, Vec<ExprId>)>,
    links: Vec<(ExprId, ExprId, String)>,
}

impl<'g, K: Field> GraphView<'g, K> {
    pub fn new(g: &'g Graph<K>) -> Self {
        GraphView {
            g,
            theme: Theme::default(),
            rankdir: "TB",
            roots: Vec::new(),
            params: FxHashSet::default(),
            focus: None,
            clusters: Vec::new(),
            links: Vec::new(),
        }
    }

    pub fn theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }

    /// `TB` (the default, operands above) or `LR`.
    pub fn rankdir(mut self, rankdir: &'static str) -> Self {
        self.rankdir = rankdir;
        self
    }

    /// Draw `e` and what it reaches, as a result named `name` (no result
    /// node when `name` is empty).
    pub fn root(mut self, e: ExprId, name: &str) -> Self {
        self.roots.push((e, name.to_string()));
        self
    }

    /// Draw these symbols as parameters, the others as inputs.
    pub fn params(mut self, syms: &[SymbolId]) -> Self {
        self.params.extend(syms.iter().copied());
        self
    }

    /// Keep these nodes at full strength and fade the rest.
    pub fn focus(mut self, nodes: impl IntoIterator<Item = ExprId>) -> Self {
        self.focus = Some(nodes.into_iter().collect());
        self
    }

    /// Group these nodes in a frame labelled `label` (a node joins the
    /// first cluster naming it).
    pub fn cluster(mut self, label: &str, nodes: impl IntoIterator<Item = ExprId>) -> Self {
        self.clusters
            .push((label.to_string(), nodes.into_iter().collect()));
        self
    }

    /// A dashed edge from `a` to `b`, labelled.
    pub fn link(mut self, a: ExprId, b: ExprId, label: &str) -> Self {
        self.links.push((a, b, label.to_string()));
        self
    }

    fn faded(&self, e: ExprId) -> bool {
        self.focus.as_ref().is_some_and(|f| !f.contains(&e))
    }

    /// The label and kind of node `e`.
    fn style(&self, e: ExprId) -> (String, Kind) {
        let g = self.g;
        let math = self.theme.notation == Notation::Math;
        match *g.node(e) {
            Node::Const(c) => {
                let exact = g.const_val(c).render();
                let label = match g.const_f64(e) {
                    Some(x) if exact.len() > 8 || !exact.contains('/') => number(x),
                    _ => exact,
                };
                (label, Kind::Const)
            }
            Node::Symbol(s) => (
                g.symbol_name(s).to_string(),
                if self.params.contains(&s) {
                    Kind::Param
                } else {
                    Kind::Input
                },
            ),
            Node::Add(..) => ("+".into(), Kind::Op),
            Node::Mul(..) => (if math { "\u{00d7}" } else { "*" }.into(), Kind::Op),
            Node::Neg(_) => (
                if math { "\u{2212}(\u{00b7})" } else { "neg" }.into(),
                Kind::Op,
            ),
            Node::Pow(_, n) if math => (format!("(\u{00b7})^{n}"), Kind::Op),
            Node::Pow(_, n) => (format!("^{n}"), Kind::Op),
            Node::Unary(op, _) if math => (format!("{}(\u{00b7})", op.name()), Kind::Op),
            Node::Unary(op, _) => (op.name().into(), Kind::Op),
            Node::Binary(op, ..) => (op.name().into(), Kind::Op),
            Node::Cmp(op, ..) => {
                let c = match op {
                    CmpOp::Gt => ">",
                    CmpOp::Ge => ">=",
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Eq => "==",
                    CmpOp::Ne => "!=",
                };
                let label = if math {
                    format!("(\u{00b7}) {c} (\u{00b7})")
                } else {
                    c.to_string()
                };
                (label, Kind::Choice)
            }
            Node::Select(..) if math => {
                ("(\u{00b7}) ? (\u{00b7}) : (\u{00b7})".into(), Kind::Choice)
            }
            Node::Select(..) => ("select".into(), Kind::Choice),
            Node::Reduce(op, _) => (
                match (op, math) {
                    (ReduceOp::Sum, true) => "\u{03a3}",
                    (ReduceOp::Product, true) => "\u{03a0}",
                    (ReduceOp::Sum, false) => "sum",
                    (ReduceOp::Product, false) => "prod",
                    (ReduceOp::Min, _) => "min",
                    (ReduceOp::Max, _) => "max",
                }
                .into(),
                Kind::Kernel,
            ),
            Node::Dot(_) => (
                if math { "<\u{00b7},\u{00b7}>" } else { "dot" }.into(),
                Kind::Kernel,
            ),
            Node::Solve(_, i) => (format!("solve[{i}]"), Kind::Kernel),
            Node::Call(o, _) => {
                let (f, k) = g.output(o);
                let func = g.func(f);
                let name = if func.outputs.len() > 1 {
                    format!("{}#{k}", func.name)
                } else {
                    func.name.clone()
                };
                (name, Kind::Call)
            }
        }
    }

    pub fn render(&self) -> String {
        let g = self.g;
        let t = &self.theme;
        let mut seeds: Vec<ExprId> = self.roots.iter().map(|r| r.0).collect();
        for (_, ids) in &self.clusters {
            seeds.extend(ids.iter().copied());
        }
        for &(a, b, _) in &self.links {
            seeds.push(a);
            seeds.push(b);
        }
        // Nodes in id order: operands before the nodes that read them.
        let mut nodes: Vec<ExprId> = reachable(g, &seeds).into_iter().collect();
        nodes.sort_by_key(|e| e.0);
        let mut home: FxHashMap<ExprId, usize> = FxHashMap::default();
        for (c, (_, ids)) in self.clusters.iter().enumerate() {
            for &e in ids {
                home.entry(e).or_insert(c);
            }
        }
        let mut s = t.header(self.rankdir);
        let node_line = |e: ExprId| {
            let (label, kind) = self.style(e);
            format!("n{} [{}];\n", e.0, t.node(kind, &label, self.faded(e)))
        };
        for (c, (label, _)) in self.clusters.iter().enumerate() {
            let _ = writeln!(s, "  subgraph cluster_{c} {{\n    {}", t.cluster(label));
            for &e in nodes.iter().filter(|e| home.get(e) == Some(&c)) {
                s.push_str("    ");
                s.push_str(&node_line(e));
            }
            s.push_str("  }\n");
        }
        for &e in nodes.iter().filter(|e| !home.contains_key(e)) {
            s.push_str("  ");
            s.push_str(&node_line(e));
        }
        for &e in &nodes {
            let ops = g.operands(e);
            let select = matches!(g.node(e), Node::Select(..));
            for (k, &a) in ops.iter().enumerate() {
                let mut attrs: Vec<String> = Vec::new();
                if select {
                    attrs.push(format!("label=\"{}\"", ["if", "then", "else"][k]));
                }
                if self.faded(e) || self.faded(a) {
                    attrs.push(t.faded_edge());
                }
                let _ = writeln!(s, "  n{} -> n{} [{}];", a.0, e.0, attrs.join(", "));
            }
        }
        for (i, (e, name)) in self.roots.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            let _ = writeln!(
                s,
                "  out{i} [{}];\n  n{} -> out{i};",
                t.node(Kind::Output, name, self.faded(*e)),
                e.0
            );
        }
        for (a, b, label) in &self.links {
            let _ = writeln!(
                s,
                "  n{} -> n{} [style=dashed, label=\"{}\", constraint=false];",
                a.0,
                b.0,
                escape(label)
            );
        }
        s.push_str("}\n");
        s
    }
}

/// The DOT of a tape's dataflow (see the module docs).
pub struct TapeView<'t> {
    tape: &'t Tape,
    theme: Theme,
    rankdir: &'static str,
    inputs: Vec<String>,
    params: Vec<bool>,
    outputs: Vec<String>,
    bundles: FxHashMap<u32, String>,
}

impl<'t> TapeView<'t> {
    pub fn new(tape: &'t Tape) -> Self {
        TapeView {
            tape,
            theme: Theme::default(),
            rankdir: "TB",
            inputs: Vec::new(),
            params: Vec::new(),
            outputs: Vec::new(),
            bundles: FxHashMap::default(),
        }
    }

    pub fn theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }

    pub fn rankdir(mut self, rankdir: &'static str) -> Self {
        self.rankdir = rankdir;
        self
    }

    /// Names of the inputs, in input order (`i0`, `i1`, ... otherwise).
    pub fn inputs(mut self, names: &[&str]) -> Self {
        self.inputs = names.iter().map(|n| n.to_string()).collect();
        self
    }

    /// Which inputs are parameters (the prolog's).
    pub fn params(mut self, mask: &[bool]) -> Self {
        self.params = mask.to_vec();
        self
    }

    /// Names of the outputs, in output order (`out0`, ... otherwise).
    pub fn outputs(mut self, names: &[&str]) -> Self {
        self.outputs = names.iter().map(|n| n.to_string()).collect();
        self
    }

    /// The name bundle `b` is drawn with (`b0`, ... otherwise).
    pub fn bundle(mut self, b: u32, name: &str) -> Self {
        self.bundles.insert(b, name.to_string());
        self
    }

    pub fn render(&self) -> String {
        let tape = self.tape;
        let t = &self.theme;
        let n_ops = tape.n_ops();
        let prolog = tape.prolog_len();
        let views: Vec<_> = (0..n_ops).map(|i| tape.op_view(i)).collect();
        let mut s = t.header(self.rankdir);
        let input_name = |k: u32| {
            self.inputs
                .get(k as usize)
                .cloned()
                .unwrap_or_else(|| format!("i{k}"))
        };
        let mut used: Vec<u32> = views
            .iter()
            .flat_map(|v| v.reads.iter().copied())
            .chain(tape.outputs().iter().copied())
            .filter_map(input_index)
            .collect();
        used.sort_unstable();
        used.dedup();
        let is_param = |k: u32| self.params.get(k as usize).copied().unwrap_or(false);
        // An input sits with the phase that first reads it.
        let read_in_prolog: FxHashSet<u32> = views[..prolog]
            .iter()
            .flat_map(|v| v.reads.iter().copied())
            .filter_map(input_index)
            .collect();
        let input_line = |k: u32| {
            let kind = if is_param(k) {
                Kind::Param
            } else {
                Kind::Input
            };
            format!("in{k} [{}];", t.node(kind, &input_name(k), false))
        };
        if prolog == 0 {
            for &k in &used {
                let _ = writeln!(s, "  {}", input_line(k));
            }
        }
        let label = |i: usize| {
            let v = &views[i];
            match v.bundle {
                Some(b) => {
                    let name = self
                        .bundles
                        .get(&b)
                        .cloned()
                        .unwrap_or_else(|| format!("b{b}"));
                    format!("{} {name}", v.label)
                }
                None => v.label.clone(),
            }
        };
        let phases: Vec<(&str, std::ops::Range<usize>)> = if prolog > 0 {
            vec![("prolog", 0..prolog), ("main", prolog..n_ops)]
        } else {
            vec![("", 0..n_ops)]
        };
        for (c, (name, range)) in phases.iter().enumerate() {
            let indent = if name.is_empty() {
                "  "
            } else {
                let _ = writeln!(s, "  subgraph cluster_{c} {{\n    {}", t.cluster(name));
                "    "
            };
            if !name.is_empty() {
                for &k in used
                    .iter()
                    .filter(|&&k| read_in_prolog.contains(&k) == (c == 0))
                {
                    let _ = writeln!(s, "{indent}{}", input_line(k));
                }
            }
            for i in range.clone() {
                let _ = writeln!(
                    s,
                    "{indent}o{i} [{}];",
                    t.node(views[i].kind, &label(i), false)
                );
            }
            if !name.is_empty() {
                s.push_str("  }\n");
            }
        }
        // Each slot read is the value its last writer left; a value the
        // prolog wrote and the main phase reads crosses in the state.
        let mut writer: FxHashMap<u32, usize> = FxHashMap::default();
        let mut edges: FxHashSet<(String, usize)> = FxHashSet::default();
        let source = |k: u32, writer: &FxHashMap<u32, usize>| match input_index(k) {
            Some(i) => Some((format!("in{i}"), None)),
            None => writer.get(&k).map(|&w| (format!("o{w}"), Some(w))),
        };
        for (i, v) in views.iter().enumerate() {
            for &k in &v.reads {
                let Some((from, w)) = source(k, &writer) else {
                    continue;
                };
                if !edges.insert((from.clone(), i)) {
                    continue;
                }
                let state = w.is_some_and(|w| w < prolog) && i >= prolog;
                let attrs = if state {
                    format!(" [style=dashed, color=\"{}\"]", t.state)
                } else {
                    String::new()
                };
                let _ = writeln!(s, "  {from} -> o{i}{attrs};");
            }
            let dst = tape.op_dst(i);
            for slot in dst..dst + v.width {
                writer.insert(slot, i);
            }
        }
        for (j, &k) in tape.outputs().iter().enumerate() {
            let name = self
                .outputs
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("out{j}"));
            let _ = writeln!(s, "  r{j} [{}];", t.node(Kind::Output, &name, false));
            if let Some((from, _)) = source(k, &writer) {
                let _ = writeln!(s, "  {from} -> r{j};");
            }
        }
        let results: Vec<String> = (0..tape.outputs().len()).map(|j| format!("r{j}")).collect();
        let _ = writeln!(s, "  {{ rank=sink; {}; }}", results.join("; "));
        s.push_str("}\n");
        s
    }
}
