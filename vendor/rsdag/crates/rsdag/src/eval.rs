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
use crate::func::{Body, FuncId, Output, OutputId};
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
    call: &mut impl FnMut(OutputId, ArgList, &[T]) -> T,
    solve: &mut impl FnMut(ArgList, &[T], u32) -> T,
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
            // the operands in parameter order, then the globals the body reads
            let globals = ctx.globals(ctx.output(o).0);
            let full = ctx.full_args(o, l);
            let vals: Vec<T> = full.iter().chain(&globals[..]).map(|&a| get(a)).collect();
            call(o, l, &vals)
        }
        Node::Solve(l, i) => {
            let vals: Vec<T> = ctx.args(l).iter().map(|&a| get(a)).collect();
            solve(l, &vals, i)
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
    // the nodes the roots reach, ascending: a node after its operands
    let cone = ctx.cone_sorted(roots, true);
    let at = |e: ExprId| cone.binary_search(&e).expect("in the cone");
    let mut fe = FuncEval::new();
    for &e in &cone {
        if let Node::Call(o, _) = *ctx.node(e) {
            let (f, out) = ctx.output(o);
            fe.need(f, out);
        }
    }
    let mut w: Vec<T> = Vec::with_capacity(cone.len());
    // A dense system is solved once for all its components.
    let mut solved: HashMap<ArgList, Vec<T>> = HashMap::new();
    for &e in &cone {
        let node = *ctx.node(e);
        let mut sym = |s: SymbolId| env.get(&s).copied().unwrap_or(T::nan());
        let mut call = |o: OutputId, l: ArgList, args: &[T]| fe.output(ctx, o, l, args);
        let mut solve = |l: ArgList, vals: &[T], i: u32| -> T {
            solved.entry(l).or_insert_with(|| {
                let n = Graph::<K>::solve_n(vals.len());
                let mut out = vec![T::zero(); n];
                crate::semantics::solve_t(&vals[..n * n], &vals[n * n..], n, &mut out);
                out
            })[i as usize]
        };
        let v = node_value(ctx, &node, |e| w[at(e)], &mut sym, &mut call, &mut solve);
        w.push(v);
    }
    roots.iter().map(|&r| w[at(r)]).collect()
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
    /// Per instance (function, context, argument list), its outputs.
    vals: rustc_hash::FxHashMap<(FuncId, u32, ArgList), Vec<T>>,
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

    /// The value of a call of output `o` over the list `l` at the values
    /// `args` of its operands (in its function's parameter order, then its
    /// globals); the instance keys the memo.
    pub fn output<K: Field>(&mut self, ctx: &Graph<K>, o: OutputId, l: ArgList, args: &[T]) -> T {
        let (f, out) = ctx.output(o);
        let site = (f, ctx.context_of(o), l);
        let func = ctx.func(f);
        if matches!(func.outputs()[out as usize], Output::Zero) {
            return T::zero();
        }
        let needed = &self.needed;
        let body = self
            .bodies
            .entry(f)
            .or_insert_with(|| match needed.get(&f) {
                Some(needed) => func.body_for(ctx, needed),
                None => func.body_for(ctx, &[out]),
            });
        let Some(slot) = body.slot_of.get(out as usize).copied().flatten() else {
            // The cached body predates this output (a derivative demanded
            // later): rebuild it over every output.
            let fresh = func.body(ctx);
            self.vals.retain(|k, _| k.0 != f);
            *body = fresh;
            let slot = body.slot_of[out as usize].expect("a body carries every output");
            return self.eval_slot(site, args, slot);
        };
        self.eval_slot(site, args, slot)
    }

    fn eval_slot(&mut self, site: (FuncId, u32, ArgList), args: &[T], slot: u32) -> T {
        let body = &self.bodies[&site.0];
        let vals = self.vals.entry(site).or_insert_with(|| {
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
