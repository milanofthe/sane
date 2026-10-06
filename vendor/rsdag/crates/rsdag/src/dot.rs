//! A graph's DAG as data for a consumer that lays it out and draws it.
//!
//! [`GraphView`] views the hash-consed DAG: a shared subexpression is one
//! node with several parents. A focus set keeps its nodes at full strength
//! and fades the rest (the nodes a transform added, the arms a
//! specialization keeps), clusters group nodes (a function body), links add
//! dashed edges between nodes; `bodies` takes in the bodies of the functions
//! the drawn calls reach, each once in its frame, every call linked to it;
//! `inline` takes every call as its body, once per instance, in nested
//! frames: the complete graph.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::field::Field;
use crate::func::{FuncId, Output};
use crate::graph::Graph;
use crate::node::{ArgList, ExprId, Node, ReduceOp, SymbolId};

/// What a node is, which decides how it is drawn.
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

/// A graph's DAG under some roots (see the module docs).
pub struct GraphView<'g, K: Field> {
    g: &'g Graph<K>,
    notation: Notation,
    roots: Vec<(ExprId, String)>,
    params: FxHashSet<SymbolId>,
    focus: Option<FxHashSet<ExprId>>,
    clusters: Vec<(String, Vec<ExprId>)>,
    links: Vec<(ExprId, ExprId, String)>,
    bodies: bool,
    inline: bool,
    labels: FxHashMap<ExprId, String>,
}

/// The graph a [`GraphView`] draws, as data: for a consumer that lays it out
/// and draws it itself. Node ids are the DOT names (`n{expr}`, `out{root}`).
#[derive(Clone, Debug, Default)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    /// `(from, to, label)` by node index, operand to consumer (a select's
    /// operands labelled `if`, `then`, `else`).
    pub edges: Vec<(usize, usize, &'static str)>,
    /// The frames' labels; a node names its frame by index.
    pub clusters: Vec<String>,
    /// The frame each frame sits in (nested instances, see
    /// [`GraphView::inline`]).
    pub parents: Vec<Option<usize>>,
    /// `(from, to, label)` by node index: the dashed links (a call to the
    /// output of the body it calls).
    pub links: Vec<(usize, usize, String)>,
}

/// One node of [`GraphData`].
#[derive(Clone, Debug)]
pub struct GraphNode {
    /// The DOT name.
    pub id: String,
    pub label: String,
    pub kind: Kind,
    /// The frame it sits in.
    pub cluster: Option<usize>,
    /// Outside the focus.
    pub faded: bool,
}

impl<'g, K: Field> GraphView<'g, K> {
    pub fn new(g: &'g Graph<K>) -> Self {
        GraphView {
            g,
            notation: Notation::Ascii,
            roots: Vec::new(),
            params: FxHashSet::default(),
            focus: None,
            clusters: Vec::new(),
            links: Vec::new(),
            bodies: false,
            inline: false,
            labels: FxHashMap::default(),
        }
    }

    /// Draw node `e` labelled `label` instead of its own.
    pub fn label(mut self, e: ExprId, label: &str) -> Self {
        self.labels.insert(e, label.to_string());
        self
    }

    /// How the operators are labelled ([`Notation::Ascii`] by default).
    pub fn notation(mut self, notation: Notation) -> Self {
        self.notation = notation;
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

    /// Draw the bodies of the functions the drawn calls reach, to the
    /// bottom: each function once, in a frame named after it, and every
    /// call linked (dashed) to the output it calls. The whole hierarchy,
    /// a function shared by its instances rather than repeated per call.
    pub fn bodies(mut self) -> Self {
        self.bodies = true;
        self
    }

    /// Draw every call as what it computes: the called function's body,
    /// its parameters bound to the call's arguments, once per instance (the
    /// calls of one function over one argument list), in a frame inside its
    /// caller's frame, named by the call's label or else the function. The
    /// complete graph, its hierarchy as nested frames; instances stay apart
    /// however equal their bodies' expressions are.
    pub fn inline(mut self) -> Self {
        self.inline = true;
        self
    }

    /// The clusters and links of the called functions' bodies (see
    /// [`bodies`](Self::bodies)), after the given ones: a body's frame
    /// holds the nodes it reaches that nothing drawn before reaches.
    /// The bodies' parameters with the `Param` role are drawn as
    /// parameters.
    #[allow(clippy::type_complexity)]
    fn with_bodies(
        &self,
        seeds: &[ExprId],
    ) -> (
        Vec<(String, Vec<ExprId>)>,
        Vec<(ExprId, ExprId, String)>,
        FxHashSet<SymbolId>,
    ) {
        let g = self.g;
        let mut clusters = self.clusters.clone();
        let mut links = self.links.clone();
        let mut params = self.params.clone();
        if !self.bodies {
            return (clusters, links, params);
        }
        let mut drawn = reachable(g, seeds);
        let mut frame: FxHashMap<FuncId, usize> = FxHashMap::default();
        let mut entered: FxHashSet<(FuncId, u32)> = FxHashSet::default();
        let mut todo: Vec<ExprId> = drawn.iter().copied().collect();
        while !todo.is_empty() {
            todo.sort_by_key(|e| e.0);
            let mut next = Vec::new();
            for e in todo {
                let Node::Call(o, _) = *g.node(e) else {
                    continue;
                };
                let (f, k) = g.output(o);
                let Output::Expr(b) = g.func(f).outputs()[k as usize] else {
                    continue;
                };
                links.push((e, b, String::new()));
                if !entered.insert((f, k)) {
                    continue;
                }
                let c = *frame.entry(f).or_insert_with(|| {
                    let func = g.func(f);
                    params.extend(
                        func.params()
                            .iter()
                            .zip(func.param_roles())
                            .filter(|(_, r)| matches!(r, crate::ParamRole::Param))
                            .map(|(&s, _)| s),
                    );
                    clusters.push((func.name().to_string(), Vec::new()));
                    clusters.len() - 1
                });
                for n in reachable(g, &[b]) {
                    if drawn.insert(n) {
                        clusters[c].1.push(n);
                        next.push(n);
                    }
                }
            }
            todo = next;
        }
        (clusters, links, params)
    }

    fn faded(&self, e: ExprId) -> bool {
        self.focus.as_ref().is_some_and(|f| !f.contains(&e))
    }

    /// The label and kind of node `e`.
    fn style(&self, e: ExprId, params: &FxHashSet<SymbolId>) -> (String, Kind) {
        let (label, kind) = self.own_style(e, params);
        match self.labels.get(&e) {
            Some(l) => (l.clone(), kind),
            None => (label, kind),
        }
    }

    /// The node's own label and its kind.
    fn own_style(&self, e: ExprId, params: &FxHashSet<SymbolId>) -> (String, Kind) {
        let g = self.g;
        let math = self.notation == Notation::Math;
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
                if params.contains(&s) {
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
                let c = op.symbol();
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
                    (op, _) => op.name(),
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
                let name = if func.outputs().len() > 1 {
                    format!("{}#{k}", func.name())
                } else {
                    func.name().to_string()
                };
                (name, Kind::Call)
            }
        }
    }

    /// The graph this view draws (see [`GraphData`]).
    pub fn data(&self) -> GraphData {
        if self.inline {
            return Inliner::new(self).data();
        }
        let g = self.g;
        let mut seeds: Vec<ExprId> = self.roots.iter().map(|r| r.0).collect();
        for (_, ids) in &self.clusters {
            seeds.extend(ids.iter().copied());
        }
        for &(a, b, _) in &self.links {
            seeds.push(a);
            seeds.push(b);
        }
        let (clusters, links, params) = self.with_bodies(&seeds);
        for (_, ids) in &clusters {
            seeds.extend(ids.iter().copied());
        }
        // Nodes in id order: operands before the nodes that read them.
        let mut exprs: Vec<ExprId> = reachable(g, &seeds).into_iter().collect();
        exprs.sort_by_key(|e| e.0);
        let mut home: FxHashMap<ExprId, usize> = FxHashMap::default();
        for (c, (_, ids)) in clusters.iter().enumerate() {
            for &e in ids {
                home.entry(e).or_insert(c);
            }
        }
        let mut data = GraphData {
            clusters: clusters.iter().map(|(l, _)| l.clone()).collect(),
            parents: vec![None; clusters.len()],
            ..GraphData::default()
        };
        let mut at: FxHashMap<ExprId, usize> = FxHashMap::default();
        for &e in &exprs {
            let (label, kind) = self.style(e, &params);
            at.insert(e, data.nodes.len());
            data.nodes.push(GraphNode {
                id: format!("n{}", e.0),
                label,
                kind,
                cluster: home.get(&e).copied(),
                faded: self.faded(e),
            });
        }
        for &e in &exprs {
            let select = matches!(g.node(e), Node::Select(..));
            for (k, a) in g.operands(e).iter().enumerate() {
                let label = if select {
                    ["if", "then", "else"][k]
                } else {
                    ""
                };
                data.edges.push((at[a], at[&e], label));
            }
        }
        for (i, (e, name)) in self.roots.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            data.nodes.push(GraphNode {
                id: format!("out{i}"),
                label: name.clone(),
                kind: Kind::Output,
                cluster: None,
                faded: self.faded(*e),
            });
            data.edges.push((at[e], data.nodes.len() - 1, ""));
        }
        for (a, b, label) in &links {
            data.links.push((at[a], at[b], label.clone()));
        }
        data
    }
}

/// [`GraphView::inline`]: the drawn graph with every call replaced by its
/// body, per instance. A node is an expression in a frame (0 the top, an
/// instance's frame the index of its cluster plus one); a body's parameter
/// is the argument in the caller's frame, a call's output the body's output
/// in the instance's frame.
struct Inliner<'v, 'g, K: Field> {
    view: &'v GraphView<'g, K>,
    /// Per frame: the frame it sits in and its parameters' arguments there.
    frames: Vec<(usize, FxHashMap<SymbolId, ExprId>)>,
    instances: FxHashMap<(usize, FuncId, ArgList), usize>,
    at: FxHashMap<(usize, ExprId), usize>,
    data: GraphData,
}

impl<'v, 'g, K: Field> Inliner<'v, 'g, K> {
    fn new(view: &'v GraphView<'g, K>) -> Self {
        Inliner {
            view,
            frames: vec![(0, FxHashMap::default())],
            instances: FxHashMap::default(),
            at: FxHashMap::default(),
            data: GraphData::default(),
        }
    }

    /// The node `(frame, e)` stands for: a bound parameter is its argument
    /// in the caller's frame, a free symbol the top frame's, a call of a
    /// symbolic output that output in the instance's frame.
    fn resolve(&mut self, mut fr: usize, mut e: ExprId) -> (usize, ExprId) {
        let g = self.view.g;
        loop {
            match *g.node(e) {
                Node::Symbol(s) if fr != 0 => match self.frames[fr].1.get(&s) {
                    Some(&a) => (fr, e) = (self.frames[fr].0, a),
                    None => return (0, e),
                },
                Node::Call(o, l) => {
                    let (f, k) = g.output(o);
                    let Output::Expr(b) = g.func(f).outputs()[k as usize] else {
                        return (fr, e);
                    };
                    let c = match self.instances.get(&(fr, f, l)) {
                        Some(&c) => c,
                        None => {
                            let func = g.func(f);
                            let bind = func.params().iter().copied().zip(g.args(l).iter().copied());
                            self.frames.push((fr, bind.collect()));
                            let c = self.frames.len() - 1;
                            self.instances.insert((fr, f, l), c);
                            let label = match self.view.labels.get(&e) {
                                Some(l) => l.clone(),
                                None => func.name().to_string(),
                            };
                            self.data.clusters.push(label);
                            self.data.parents.push(fr.checked_sub(1));
                            c
                        }
                    };
                    (fr, e) = (c, b);
                }
                _ => return (fr, e),
            }
        }
    }

    /// The node index of `(frame, e)`, drawing it and what it reads.
    fn node(&mut self, fr: usize, e: ExprId) -> usize {
        let g = self.view.g;
        let root = self.resolve(fr, e);
        let mut stack = vec![(root, false)];
        while let Some((key, ready)) = stack.pop() {
            if self.at.contains_key(&key) {
                continue;
            }
            let (kf, ke) = key;
            let ops: Vec<(usize, ExprId)> = g
                .operands(ke)
                .iter()
                .map(|&a| self.resolve(kf, a))
                .collect();
            if !ready {
                stack.push((key, true));
                stack.extend(
                    ops.into_iter()
                        .filter(|k| !self.at.contains_key(k))
                        .map(|k| (k, false)),
                );
                continue;
            }
            let (label, kind) = self.view.style(ke, &self.view.params);
            let n = self.data.nodes.len();
            self.data.nodes.push(GraphNode {
                id: if kf == 0 {
                    format!("n{}", ke.0)
                } else {
                    format!("n{}_{kf}", ke.0)
                },
                label,
                kind,
                cluster: kf.checked_sub(1),
                faded: self.view.faded(ke),
            });
            self.at.insert(key, n);
            let select = matches!(g.node(ke), Node::Select(..));
            for (k, op) in ops.iter().enumerate() {
                let label = if select {
                    ["if", "then", "else"][k]
                } else {
                    ""
                };
                self.data.edges.push((self.at[op], n, label));
            }
        }
        self.at[&root]
    }

    fn data(mut self) -> GraphData {
        let view = self.view;
        for (_, ids) in &view.clusters {
            for &e in ids {
                self.node(0, e);
            }
        }
        for (i, (e, name)) in view.roots.iter().enumerate() {
            let n = self.node(0, *e);
            if name.is_empty() {
                continue;
            }
            self.data.nodes.push(GraphNode {
                id: format!("out{i}"),
                label: name.clone(),
                kind: Kind::Output,
                cluster: None,
                faded: view.faded(*e),
            });
            self.data.edges.push((n, self.data.nodes.len() - 1, ""));
        }
        for (a, b, label) in &view.links {
            let (a, b) = (self.node(0, *a), self.node(0, *b));
            self.data.links.push((a, b, label.clone()));
        }
        self.data
    }
}
