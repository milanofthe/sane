//! The interchange format: a module as data.
//!
//! A [`Graph`] carries the mathematics of a set of functions plus the caches
//! that make building them fast (the hash-cons tables, the traversal memo)
//! and, for a function a solver has compiled, a binding to native code. Only
//! the first of those three is the module; the caches rebuild on load and a
//! compiled body is re-bound by the consumer.
//!
//! [`Module`] is that first part, and it is what a consumer serializes to
//! cache an analysis, to hand a model to code generation, or to export it.
//! Round-tripping a module reproduces every value bit for bit, which is the
//! property the tests pin.
//!
//! ```
//! # use rsdag::{Graph, Tape, F64};
//! let mut g: Graph<F64> = Graph::new();
//! let x = g.sym("x");
//! let e = g.sin(x);
//! let f = g.close("f", vec![e]);
//! let module = g.to_module();          // plain data, `serde`-serializable
//! let (mut back, map) = Graph::from_module(&module).unwrap();
//! assert_eq!(map.funcs[f.0 as usize], f);
//! # let _ = &mut back;
//! ```

use rustc_hash::FxHashMap as HashMap;

use crate::field::Field;
use std::sync::Arc;

use crate::extern_fn::ExternBundle;
use crate::func::{FuncId, FunctionBody, Output, OutputId};
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};
use crate::role::{OutputRole, ParamRole};

/// A function as data: no body binding, no derivative memo.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionData {
    pub name: String,
    pub params: Vec<SymbolId>,
    pub param_roles: Vec<ParamRole>,
    pub outputs: Vec<Output>,
    pub output_roles: Vec<OutputRole>,
    /// An extern function names the body it expects; the loader is handed
    /// the bodies by name (see [`Graph::load_module_with`]).
    pub extern_body: Option<String>,
}

/// A module as data: the nodes, the constants, the operand pool, the symbol
/// names and the functions. Everything else in a [`Graph`] is a cache.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(serialize = "K: serde::Serialize")))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "K: serde::de::DeserializeOwned"))
)]
pub struct Module<K> {
    /// Format version, so a stored module can be rejected rather than
    /// misread when the node vocabulary changes.
    pub version: u32,
    pub nodes: Vec<Node>,
    pub consts: Vec<K>,
    pub arg_pool: Vec<ExprId>,
    pub symbols: Vec<String>,
    pub funcs: Vec<FunctionData>,
    /// The `(function, output)` pairs the `Call` nodes name, in id order.
    pub call_outputs: Vec<(FuncId, u32)>,
}

/// The current [`Module::version`]. Bump it when the meaning of an existing
/// node changes; adding a variant at the end of an enum does not need it,
/// because an older reader fails on the unknown discriminant anyway.
pub const MODULE_VERSION: u32 = 1;

/// Why a module cannot be loaded. A module is data from outside (a file, a
/// cache, another process), so the loader checks it whole before it builds
/// anything, and reports rather than panics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModuleError {
    /// Written by a build with another [`MODULE_VERSION`].
    Version { found: u32, expected: u32 },
    /// Node `node` names something that is not there, or names a node that
    /// does not precede it.
    Dangling { node: usize, what: &'static str },
    /// Node `node` has an operand list of a length its kind cannot have.
    Shape { node: usize, what: &'static str },
    /// Function `func` names a symbol, node or role slot that is not there.
    Function { func: usize, what: &'static str },
    /// No body was supplied for the extern function of this name.
    MissingExtern(String),
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::Version { found, expected } => write!(
                f,
                "module format version {found} cannot be read by this build (expects {expected})"
            ),
            ModuleError::Dangling { node, what } => write!(f, "node {node}: dangling {what}"),
            ModuleError::Shape { node, what } => write!(f, "node {node}: {what}"),
            ModuleError::Function { func, what } => write!(f, "function {func}: {what}"),
            ModuleError::MissingExtern(name) => {
                write!(f, "no body supplied for the extern function '{name}'")
            }
        }
    }
}

impl std::error::Error for ModuleError {}

impl<K> Module<K> {
    /// Check that every id the module holds names something that is there
    /// and, for a node's operands, precedes it; and that every operand list
    /// has a length its node can have. A module that passes loads without
    /// a panic.
    pub fn validate(&self) -> Result<(), ModuleError> {
        if self.version != MODULE_VERSION {
            return Err(ModuleError::Version {
                found: self.version,
                expected: MODULE_VERSION,
            });
        }
        let n_nodes = self.nodes.len();
        for (fi, f) in self.funcs.iter().enumerate() {
            let bad = |what| Err(ModuleError::Function { func: fi, what });
            if f.params.iter().any(|s| s.0 as usize >= self.symbols.len()) {
                return bad("parameter symbol out of range");
            }
            let dangling = f.outputs.iter().any(|o| match *o {
                Output::Expr(e) => e.0 as usize >= n_nodes,
                _ => false,
            });
            if dangling {
                return bad("output node out of range");
            }
            if f.param_roles.len() > f.params.len() || f.output_roles.len() > f.outputs.len() {
                return bad("more roles than slots");
            }
        }
        for &(f, k) in &self.call_outputs {
            match self.funcs.get(f.0 as usize) {
                Some(data) if (k as usize) < data.outputs.len() => {}
                _ => {
                    return Err(ModuleError::Function {
                        func: f.0 as usize,
                        what: "a call names an output that is not there",
                    })
                }
            }
        }
        for (i, node) in self.nodes.iter().enumerate() {
            let dangling = |what| Err(ModuleError::Dangling { node: i, what });
            let shape = |what| Err(ModuleError::Shape { node: i, what });
            let before = |e: &ExprId| (e.0 as usize) < i;
            let list = |l: &crate::node::ArgList| {
                let (a, n) = (l.start as usize, l.len as usize);
                self.arg_pool.get(a..a + n)
            };
            match *node {
                Node::Const(c) if c.0 as usize >= self.consts.len() => return dangling("constant"),
                Node::Symbol(s) if s.0 as usize >= self.symbols.len() => return dangling("symbol"),
                Node::Const(_) | Node::Symbol(_) => {}
                Node::Add(a, b) | Node::Mul(a, b) | Node::Cmp(_, a, b) | Node::Binary(_, a, b) => {
                    if !before(&a) || !before(&b) {
                        return dangling("operand");
                    }
                }
                Node::Neg(a) | Node::Pow(a, _) | Node::Unary(_, a) => {
                    if !before(&a) {
                        return dangling("operand");
                    }
                }
                Node::Select(c, t, e) => {
                    if !before(&c) || !before(&t) || !before(&e) {
                        return dangling("operand");
                    }
                }
                Node::Reduce(_, l) | Node::Dot(l) | Node::Solve(l, _) | Node::Call(_, l) => {
                    let Some(args) = list(&l) else {
                        return dangling("operand list");
                    };
                    if !args.iter().all(before) {
                        return dangling("operand");
                    }
                    let n = args.len();
                    match *node {
                        Node::Dot(_) if n % 2 != 0 => return shape("dot of uneven halves"),
                        Node::Solve(_, k) => {
                            // `n*n + n` values for some `n`, component below `n`.
                            let m = ((((4 * n + 1) as f64).sqrt() - 1.0) / 2.0).round() as usize;
                            if m == 0 || m * m + m != n || k as usize >= m {
                                return shape("solve list of no square system");
                            }
                        }
                        Node::Call(o, _) => {
                            let Some(&(f, _)) = self.call_outputs.get(o.0 as usize) else {
                                return dangling("call output");
                            };
                            let data = &self.funcs[f.0 as usize];
                            if data.params.len() != n {
                                return shape("call with another argument count than its function");
                            }
                            let late = data.outputs.iter().any(|o| match o {
                                Output::Expr(e) => !before(e),
                                _ => false,
                            });
                            if late {
                                return dangling("callee output after its call");
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
}

/// How the ids of a loaded module map onto the graph it was loaded into.
/// Loading into an empty graph is the identity, but loading into a graph
/// that already holds nodes is not, so the caller gets the mapping rather
/// than an assumption.
#[derive(Clone, Debug, Default)]
pub struct IdMap {
    pub exprs: Vec<ExprId>,
    pub symbols: Vec<SymbolId>,
    pub funcs: Vec<FuncId>,
}

impl<K: Field> Graph<K> {
    /// This graph as data, ready to serialize.
    ///
    /// A function with an extern body keeps its name and its shape; the body
    /// itself is a binding to compiled code and is not part of the module.
    pub fn to_module(&self) -> Module<K> {
        Module {
            version: MODULE_VERSION,
            nodes: self.nodes_slice().to_vec(),
            consts: self.consts_slice().to_vec(),
            arg_pool: self.arg_pool_slice().to_vec(),
            symbols: (0..self.n_symbols())
                .map(|i| self.symbol_name(SymbolId(i as u32)).to_string())
                .collect(),
            funcs: self
                .funcs_slice()
                .iter()
                .map(|f| FunctionData {
                    name: f.name.clone(),
                    params: f.params.clone(),
                    param_roles: f.param_roles.clone(),
                    outputs: f.outputs.clone(),
                    output_roles: f.output_roles.clone(),
                    extern_body: match &f.body {
                        FunctionBody::Symbolic => None,
                        FunctionBody::Extern(_) => Some(f.name.clone()),
                    },
                })
                .collect(),
            call_outputs: self.call_outputs_slice().to_vec(),
        }
    }

    /// Rebuild a graph from a module, re-interning every node so the
    /// hash-cons tables and the caches are consistent with it.
    ///
    /// Nodes are re-interned in id order, which is a topological order, so a
    /// node's operands are already mapped when it is built. Re-interning
    /// (rather than copying the arena) means a module loaded into a graph
    /// that already holds equal subexpressions shares them, exactly as if it
    /// had been built there.
    pub fn from_module(module: &Module<K>) -> Result<(Graph<K>, IdMap), ModuleError> {
        let mut g = Graph::new();
        let map = g.load_module(module)?;
        Ok((g, map))
    }

    /// Load a module into this graph, returning how its ids map onto it.
    ///
    /// A module with an extern function fails with
    /// [`ModuleError::MissingExtern`]: the body is not part of a module, so
    /// such a module is loaded through [`Graph::load_module_with`], which
    /// is handed the bodies by name.
    pub fn load_module(&mut self, module: &Module<K>) -> Result<IdMap, ModuleError> {
        self.load_module_with(module, |_| None)
    }

    /// As [`Graph::load_module`], with `externs(name)` supplying the body of
    /// each extern function the module declares (`None` when there is none).
    ///
    /// The module is [validated](Module::validate) first; nothing is added
    /// to the graph when it fails.
    ///
    /// Nodes are re-interned in id order, which is a dependency order, so a
    /// node's operands are already mapped when it is built. A symbol is
    /// created when its node is met, and a function is defined the first
    /// time a `Call` to it is met (its outputs are nodes with smaller ids,
    /// so they are already mapped by then) or otherwise at the end. So
    /// symbols, functions and calls interleave exactly as they did when the
    /// module was built, and a module loaded into a fresh graph reproduces
    /// its own id layout: `Graph::from_module(&m).0.to_module() == m`.
    ///
    /// Re-interning rather than copying the arena means a module loaded into
    /// a graph that already holds equal subexpressions shares them, exactly
    /// as if it had been built there.
    pub fn load_module_with(
        &mut self,
        module: &Module<K>,
        mut externs: impl FnMut(&str) -> Option<Arc<dyn ExternBundle>>,
    ) -> Result<IdMap, ModuleError> {
        module.validate()?;
        // Every extern body up front, so a missing one fails before the
        // graph changes.
        let mut bodies: Vec<Option<Arc<dyn ExternBundle>>> = Vec::with_capacity(module.funcs.len());
        for data in &module.funcs {
            bodies.push(match &data.extern_body {
                Some(name) => {
                    Some(externs(name).ok_or_else(|| ModuleError::MissingExtern(name.clone()))?)
                }
                None => None,
            });
        }
        let mut map = LoadMap {
            exprs: Vec::with_capacity(module.nodes.len()),
            // A symbol is created when its node is met in the sweep, not up
            // front: `sym` interns the node at creation, so that is where
            // the original put it, and a fresh graph then reproduces the
            // module's id layout exactly.
            symbols: vec![SymbolId(u32::MAX); module.symbols.len()],
            funcs: vec![None; module.funcs.len()],
        };
        // Output ids are interned on demand by `call`, so the `Call` nodes
        // are remapped through the module's own table.
        let mut out_map: HashMap<OutputId, (FuncId, u32)> = HashMap::default();
        for (i, &(f, k)) in module.call_outputs.iter().enumerate() {
            out_map.insert(OutputId(i as u32), (f, k));
        }
        for (i, node) in module.nodes.iter().enumerate() {
            match node {
                Node::Symbol(s) => {
                    let e = self.sym(&module.symbols[s.0 as usize]);
                    map.symbols[s.0 as usize] = match self.node(e) {
                        Node::Symbol(t) => *t,
                        _ => unreachable!("sym returns a symbol node"),
                    };
                    map.exprs.push(e);
                    continue;
                }
                Node::Call(o, _) => {
                    let (f, _) = out_map[o];
                    self.define_loaded(module, f, &mut map, &bodies);
                }
                _ => {}
            }
            let pool = &module.arg_pool;
            let e = self.intern_node(
                node,
                |n| match *n {
                    Node::Const(c) => module.consts[c.0 as usize].clone(),
                    _ => unreachable!("only a constant asks for its value"),
                },
                |s| map.symbols[s.0 as usize],
                |x| map.exprs[x.0 as usize],
                |l| {
                    pool[l.start as usize..(l.start + l.len) as usize]
                        .iter()
                        .map(|&x| map.exprs[x.0 as usize])
                        .collect()
                },
                |o| {
                    let (f, k) = out_map[&o];
                    (
                        map.funcs[f.0 as usize].expect("callee defined before its call"),
                        k,
                    )
                },
            );
            debug_assert_eq!(map.exprs.len(), i);
            map.exprs.push(e);
        }
        for f in 0..module.funcs.len() {
            self.define_loaded(module, FuncId(f as u32), &mut map, &bodies);
        }
        Ok(IdMap {
            exprs: map.exprs,
            symbols: map.symbols,
            funcs: map
                .funcs
                .into_iter()
                .map(|f| f.expect("every function defined"))
                .collect(),
        })
    }

    /// Define function `f` of a module being loaded, once.
    fn define_loaded(
        &mut self,
        module: &Module<K>,
        f: FuncId,
        map: &mut LoadMap,
        bodies: &[Option<Arc<dyn ExternBundle>>],
    ) {
        if map.funcs[f.0 as usize].is_some() {
            return;
        }
        let data = &module.funcs[f.0 as usize];
        let params: Vec<SymbolId> = data
            .params
            .iter()
            .map(|s| map.symbols[s.0 as usize])
            .collect();
        let outputs: Vec<Output> = data
            .outputs
            .iter()
            .map(|o| match *o {
                Output::Expr(e) => Output::Expr(map.exprs[e.0 as usize]),
                other => other,
            })
            .collect();
        let id = match &bodies[f.0 as usize] {
            Some(body) => {
                self.define_extern_func_with_params(&data.name, params, body.clone(), outputs)
            }
            None => {
                let exprs: Vec<ExprId> = outputs
                    .iter()
                    .map(|o| match o {
                        Output::Expr(e) => *e,
                        _ => self.zero(),
                    })
                    .collect();
                let id = self.define_func(&data.name, params, exprs);
                // A zero output stays a zero output.
                for (k, o) in outputs.iter().enumerate() {
                    if matches!(o, Output::Zero) {
                        self.func_mut(id).outputs[k] = Output::Zero;
                    }
                }
                id
            }
        };
        for (k, r) in data.param_roles.iter().enumerate() {
            self.set_param_role(id, k as u32, *r);
        }
        for (k, r) in data.output_roles.iter().enumerate() {
            self.set_output_role(id, k as u32, *r);
        }
        map.funcs[f.0 as usize] = Some(id);
    }
}

/// The id map while a load is in progress: functions are defined on
/// demand, so their entries are optional until the end.
struct LoadMap {
    exprs: Vec<ExprId>,
    symbols: Vec<SymbolId>,
    funcs: Vec<Option<FuncId>>,
}
