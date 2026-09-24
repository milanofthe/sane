//! Building a function incrementally.
//!
//! A [`Scope`] is an open function: it hands out parameters by name as the
//! caller needs them, records their order and their roles, and closes into a
//! [`FuncId`]. It is the seam every frontend builds through -- a netlist
//! lowering, a Verilog-A lowering, the Python tracer -- so that the order of
//! a function's parameters is the order they were asked for rather than an
//! accident of symbol ids.
//!
//! A scope derefs to the graph it builds in, so the ordinary constructors
//! work on it directly:
//!
//! ```
//! use rsdag::{Graph, ParamRole, Scope, F64};
//!
//! let mut g: Graph<F64> = Graph::new();
//! let mut s = Scope::new(&mut g, "rc");
//! let v = s.param_with_role("v", ParamRole::State { id: 0 });
//! let r = s.param("r");
//! let i = s.div(v, r);                 // a graph constructor, on the scope
//! let f = s.close(vec![i]);
//! # let _ = f;
//! ```

use std::ops::{Deref, DerefMut};

use crate::field::{Field, F64};
use crate::func::FuncId;
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};
use crate::role::{OutputRole, ParamRole};

/// An open function over a graph. See the module docs.
pub struct Scope<'g, K: Field = F64> {
    graph: &'g mut Graph<K>,
    name: String,
    params: Vec<SymbolId>,
    roles: Vec<ParamRole>,
}

impl<'g, K: Field> Scope<'g, K> {
    pub fn new(graph: &'g mut Graph<K>, name: &str) -> Scope<'g, K> {
        Scope {
            graph,
            name: name.to_string(),
            params: Vec::new(),
            roles: Vec::new(),
        }
    }

    /// A parameter of this function, in call order. Asking twice for the
    /// same name returns the same parameter rather than a second one.
    pub fn param(&mut self, name: &str) -> ExprId {
        self.param_with_role(name, ParamRole::Free)
    }

    /// As [`Scope::param`], with the role the consumer's system layer needs
    /// (a state, an input port element, a mutable parameter).
    pub fn param_with_role(&mut self, name: &str, role: ParamRole) -> ExprId {
        let e = self.graph.sym(name);
        let s = match self.graph.node(e) {
            Node::Symbol(s) => *s,
            _ => unreachable!("sym returns a symbol node"),
        };
        match self.params.iter().position(|&p| p == s) {
            Some(k) => self.roles[k] = role,
            None => {
                self.params.push(s);
                self.roles.push(role);
            }
        }
        e
    }

    /// The parameters asked for so far, in order.
    pub fn params(&self) -> &[SymbolId] {
        &self.params
    }

    /// Close the scope over `outputs`.
    ///
    /// Free symbols the outputs depend on that were never asked for as
    /// parameters become trailing parameters, so closing over an expression
    /// built outside the scope is total rather than silently wrong.
    pub fn close(self, outputs: Vec<ExprId>) -> FuncId {
        self.close_with_roles(
            outputs
                .into_iter()
                .map(|e| (OutputRole::Plain, e))
                .collect(),
        )
    }

    /// As [`Scope::close`], giving each output its role.
    pub fn close_with_roles(mut self, outputs: Vec<(OutputRole, ExprId)>) -> FuncId {
        let exprs: Vec<ExprId> = outputs.iter().map(|&(_, e)| e).collect();
        for s in self.graph.free_symbols_in(&exprs) {
            if !self.params.contains(&s) {
                self.params.push(s);
                self.roles.push(ParamRole::Free);
            }
        }
        let f = self
            .graph
            .define_func(&self.name, self.params.clone(), exprs);
        for (k, role) in self.roles.iter().enumerate() {
            self.graph.set_param_role(f, k as u32, *role);
        }
        for (k, (role, _)) in outputs.iter().enumerate() {
            self.graph.set_output_role(f, k as u32, *role);
        }
        f
    }
}

impl<K: Field> Deref for Scope<'_, K> {
    type Target = Graph<K>;
    fn deref(&self) -> &Graph<K> {
        self.graph
    }
}

impl<K: Field> DerefMut for Scope<'_, K> {
    fn deref_mut(&mut self) -> &mut Graph<K> {
        self.graph
    }
}
