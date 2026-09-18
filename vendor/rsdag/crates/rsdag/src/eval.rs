//! Numeric evaluation over the arena, in any execution scalar.
//!
//! One forward sweep over the nodes reachable from the roots (ascending
//! `ExprId` is topological, so every shared subexpression is computed once),
//! with the same per-node semantics the tape uses: arithmetic from
//! [`Scalar`], fold orders from [`reduce_slice_t`] and [`crate::semantics::dot_slice_t`], so a
//! value cannot depend on which evaluator computed it. An unbound symbol is
//! `NaN`, as a missing input is for a tape. Calls evaluate their function
//! once per distinct argument list through the function's body, a bundle
//! called in the scalar at hand (see [`Scalar::call_bundle`]).

use std::collections::HashMap;

use crate::field::Field;
use crate::func::{Body, FuncId, Output};
use crate::graph::Graph;
use crate::node::ArgList;
use crate::node::{ExprId, Node, SymbolId};
use crate::scalar::Scalar;
use crate::semantics::reduce_slice_t;

/// The value of one node from its operands.
fn node_value<T: Scalar, K: Field>(
    ctx: &Graph<K>,
    node: &Node,
    mut get: impl FnMut(ExprId) -> T,
    sym: &mut impl FnMut(SymbolId) -> T,
    call: &mut impl FnMut(FuncId, u32, ArgList, &[T]) -> T,
    solve: &mut impl FnMut(ArgList, &[T]) -> Vec<T>,
) -> T {
    match *node {
        Node::Const(c) => T::from_f64(ctx.const_val(c).to_f64()),
        Node::Symbol(s) => sym(s),
        Node::Add(a, b) => get(a).add(get(b)),
        Node::Mul(a, b) => get(a).mul(get(b)),
        Node::Neg(a) => get(a).neg(),
        Node::Pow(a, k) => get(a).powi(k as i32),
        Node::Unary(op, a) => T::unary(op, get(a)),
        Node::Binary(op, a, b) => T::binary(op, get(a), get(b)),
        Node::Cmp(op, a, b) => T::cmp(op, get(a), get(b)),
        Node::Select(c, t, e) => {
            if get(c).is_true() {
                get(t)
            } else {
                get(e)
            }
        }
        Node::Reduce(op, l) => {
            let vals: Vec<T> = ctx.args(l).iter().map(|&a| get(a)).collect();
            reduce_slice_t(op, &vals)
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            let va: Vec<T> = a.iter().map(|&x| get(x)).collect();
            let vb: Vec<T> = b.iter().map(|&y| get(y)).collect();
            T::dot_slice(&va, &vb)
        }
        Node::Call(o, l) => {
            let vals: Vec<T> = ctx.args(l).iter().map(|&a| get(a)).collect();
            let (f, out) = ctx.output(o);
            call(f, out, l, &vals)
        }
        Node::Solve(l, i) => {
            let vals: Vec<T> = ctx.args(l).iter().map(|&a| get(a)).collect();
            solve(l, &vals)[i as usize]
        }
    }
}

/// The values of `roots` under the symbol bindings `env`, in `T`.
///
/// Every node reachable from a root is evaluated once, in arena order; a
/// symbol absent from `env` is `NaN`. To see every node's value (a
/// diagnostic locating the first non-finite node, say), pass every id as a
/// root: the lowest-index non-finite value is the origin, since a node's
/// operands have smaller ids.
pub fn eval<T: Scalar, K: Field>(
    ctx: &Graph<K>,
    roots: &[ExprId],
    env: &HashMap<SymbolId, T>,
) -> Vec<T> {
    let n = ctx.len();
    let mut reach = vec![false; n];
    let mut stack: Vec<ExprId> = roots.to_vec();
    let mut fe = FuncEval::new();
    while let Some(e) = stack.pop() {
        if std::mem::replace(&mut reach[e.0 as usize], true) {
            continue;
        }
        if let Node::Call(o, _) = *ctx.node(e) {
            let (f, out) = ctx.output(o);
            fe.need(f, out);
        }
        stack.extend(ctx.operands(e).iter().copied());
    }
    let mut w = vec![T::nan(); n];
    // A dense system is solved once for all its components.
    let mut solved: HashMap<ArgList, Vec<T>> = HashMap::new();
    for i in 0..n {
        if !reach[i] {
            continue;
        }
        let node = *ctx.node(ExprId(i as u32));
        let mut sym = |s: SymbolId| env.get(&s).copied().unwrap_or(T::nan());
        let mut call =
            |f: FuncId, out: u32, l: ArgList, args: &[T]| fe.output(ctx, f, out, l, args);
        let mut solve = |l: ArgList, vals: &[T]| -> Vec<T> {
            solved
                .entry(l)
                .or_insert_with(|| {
                    let n = Graph::<K>::solve_n(vals.len());
                    let mut out = vec![T::zero(); n];
                    crate::semantics::solve_t(&vals[..n * n], &vals[n * n..], n, &mut out);
                    out
                })
                .clone()
        };
        w[i] = node_value(
            ctx,
            &node,
            |e| w[e.0 as usize],
            &mut sym,
            &mut call,
            &mut solve,
        );
    }
    roots.iter().map(|&r| w[r.0 as usize]).collect()
}

/// [`eval`] with symbols bound by name.
pub fn eval_named<T: Scalar, K: Field>(
    ctx: &mut Graph<K>,
    roots: &[ExprId],
    values: &[(&str, T)],
) -> Vec<T> {
    let env: HashMap<SymbolId, T> = values
        .iter()
        .map(|(name, v)| {
            let e = ctx.sym(name);
            match ctx.node(e) {
                Node::Symbol(sid) => (*sid, *v),
                _ => unreachable!("sym() always yields a Symbol node"),
            }
        })
        .collect();
    eval(ctx, roots, &env)
}

/// Per-sweep function evaluation: a body per function and the outputs of
/// every distinct `(function, argument list)` already evaluated.
pub struct FuncEval<T: Scalar> {
    bodies: rustc_hash::FxHashMap<FuncId, Body>,
    vals: rustc_hash::FxHashMap<(FuncId, ArgList), Vec<T>>,
    /// The outputs the sweep calls per function, declared up front so one
    /// body serves the whole sweep (see [`Function::body_for`]).
    needed: rustc_hash::FxHashMap<FuncId, Vec<u32>>,
}

impl<T: Scalar> FuncEval<T> {
    pub fn new() -> Self {
        Self {
            bodies: Default::default(),
            vals: Default::default(),
            needed: Default::default(),
        }
    }

    /// Declare that the sweep calls output `out` of `f`, before the first
    /// call; a body is chosen for the declared set.
    pub fn need(&mut self, f: FuncId, out: u32) {
        let v = self.needed.entry(f).or_default();
        if !v.contains(&out) {
            v.push(out);
        }
    }

    /// Output `out` of `f` at the argument values `args` (whose interned list
    /// `l` keys the memo).
    pub fn output<K: Field>(
        &mut self,
        ctx: &Graph<K>,
        f: FuncId,
        out: u32,
        l: ArgList,
        args: &[T],
    ) -> T {
        let func = ctx.func(f);
        if matches!(func.outputs[out as usize], Output::Zero) {
            return T::zero();
        }
        let needed = self.needed.get(&f).cloned().unwrap_or_else(|| vec![out]);
        let body = self
            .bodies
            .entry(f)
            .or_insert_with(|| func.body_for(ctx, &needed));
        let Some(slot) = body.slot_of.get(out as usize).copied().flatten() else {
            // The cached body predates this output (a derivative demanded
            // later): rebuild it over every output.
            let fresh = func.body(ctx);
            self.vals.retain(|k, _| k.0 != f);
            *body = fresh;
            let slot = body.slot_of[out as usize].expect("a body carries every output");
            return self.eval_slot(f, l, args, slot);
        };
        self.eval_slot(f, l, args, slot)
    }

    fn eval_slot(&mut self, f: FuncId, l: ArgList, args: &[T], slot: u32) -> T {
        let body = &self.bodies[&f];
        let vals = self.vals.entry((f, l)).or_insert_with(|| {
            let mut out = vec![T::zero(); body.bundle.n_outputs()];
            T::call_bundle(&*body.bundle, args, &mut out);
            out
        });
        vals[slot as usize]
    }
}

impl<T: Scalar> Default for FuncEval<T> {
    fn default() -> Self {
        Self::new()
    }
}
