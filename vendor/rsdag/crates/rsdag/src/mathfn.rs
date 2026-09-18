//! Named function calls from a frontend.
//!
//! A textual frontend -- a Verilog-A lowering, a netlist behavioural source,
//! an expression parser -- reaches this graph with a function *name* and
//! already-lowered arguments. [`lower_call`] is the one place that turns the
//! pair into nodes, so every frontend produces the same shape for `abs` or
//! `atan2` instead of each keeping its own drifting copy, and a derivative
//! rule written once applies to all of them.
//!
//! Most names are the op table's own ([`UnaryOp::from_name`],
//! [`BinOp::from_name`]), so an op added there is callable by name without
//! an edit here. What this module adds is the handful of names that are not
//! op names: the aliases a language uses (`log` for the base-10 logarithm in
//! Verilog-A, `limexp` for the guarded exponential, `pow` for the real
//! power) and the two that lower to a reduction rather than to a call
//! (`min`, `max`).

use crate::field::Field;
use crate::graph::Graph;
use crate::node::{BinOp, ExprId, ReduceOp, UnaryOp};

/// Lower `name(args)` onto the graph, or `None` when no such function of
/// that arity exists (the caller decides how to report it).
///
/// Every name maps to the native op where there is one. That matters beyond
/// node count: `log10` built as `ln(x) / ln(10)` is a different value in the
/// last bits from the platform's `log10`, and the derivative of a native op
/// comes from one rule rather than from the shape it was assembled into.
pub fn lower_call<K: Field>(g: &mut Graph<K>, name: &str, args: &[ExprId]) -> Option<ExprId> {
    match (name, args.len()) {
        // Aliases: the same function under the name a language gives it.
        ("log", 1) => Some(g.unary(UnaryOp::Log10, args[0])),
        // Verilog-A's `limexp` is what `exp` already is here: the guarded
        // exponential every backend agrees on (see `node::EXP_LIMIT`).
        ("limexp", 1) => Some(g.unary(UnaryOp::Exp, args[0])),
        ("pow", 2) => Some(g.binary(BinOp::Powf, args[0], args[1])),
        // Ordered pairs are reductions, not calls: one node, and `Min`/`Max`
        // carry the subgradient rule the autodiff already knows.
        ("min", 2) => Some(g.reduce(ReduceOp::Min, vec![args[0], args[1]])),
        ("max", 2) => Some(g.reduce(ReduceOp::Max, vec![args[0], args[1]])),
        // Everything else is an op of the vocabulary, by its own name.
        (n, 1) => UnaryOp::from_name(n).map(|op| g.unary(op, args[0])),
        (n, 2) => BinOp::from_name(n).map(|op| g.binary(op, args[0], args[1])),
        _ => None,
    }
}

/// Whether [`lower_call`] knows `name` at `arity`.
pub fn is_known(name: &str, arity: usize) -> bool {
    match (name, arity) {
        ("log" | "limexp", 1) => true,
        ("pow" | "min" | "max", 2) => true,
        (n, 1) => UnaryOp::from_name(n).is_some(),
        (n, 2) => BinOp::from_name(n).is_some(),
        _ => false,
    }
}
