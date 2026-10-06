//! Symbolic differentiation over the DAG.
//!
//! Builds the derivative as new nodes in the same context, so common
//! subexpressions are shared (hash-consing) and the result can be exported or
//! differentiated again. Used to produce analytic Jacobians for DAE export.

use crate::field::Field;
use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Graph, Join, Memo, Set, Through};
use crate::node::{BinOp, CmpOp, ExprId, Node, ReduceOp, SymbolId, UnaryOp};

/// Derivative of `expr` with respect to the symbol `wrt`.
pub fn differentiate<K: Field>(ctx: &mut Graph<K>, expr: ExprId, wrt: SymbolId) -> ExprId {
    let mut memo = ctx.take_memo();
    let d = forward(ctx, &[expr], wrt, &mut memo)[0];
    ctx.put_memo(memo);
    d
}

/// Total time derivative `d/dt(e) = Σ_s (∂e/∂s) · deriv_of[s]`, summed over the
/// free symbols of `e` that have a known time-derivative entry in `deriv_of`
/// (mapping a state symbol to its derivative expression, e.g. `v{k}` -> `vdot{k}`).
/// The symbolic analogue of forming a capacitive current `dq/dt` from a charge
/// `q(v)`; used to lower Verilog-A `ddt(...)` in arbitrary residual rows.
pub fn time_derivative<K: Field>(
    ctx: &mut Graph<K>,
    e: ExprId,
    deriv_of: &HashMap<SymbolId, ExprId>,
) -> ExprId {
    let syms: Vec<SymbolId> = ctx.free_symbols(e).into_iter().collect();
    let mut terms: Vec<ExprId> = Vec::new();
    for s in syms {
        if let Some(&sdot) = deriv_of.get(&s) {
            let de = differentiate(ctx, e, s);
            if !ctx.is_zero(de) {
                let term = ctx.mul(de, sdot);
                terms.push(term);
            }
        }
    }
    ctx.reduce(ReduceOp::Sum, terms)
}

/// Local derivative `d(op(a))/da` of a unary op, shared by the forward
/// ([`differentiate`]) and reverse ([`gradient`]) sweeps so both modes apply the
/// identical rule. The rules mirror the domain guards in
/// [`crate::semantics::unary_f64`] exactly, so the Jacobian stays finite wherever the
/// residual does (an out-of-range internal-node guess must not produce an
/// `inf`/`NaN` Jacobian entry that derails Newton).
fn unary_factor<K: Field>(ctx: &mut Graph<K>, op: UnaryOp, a: ExprId) -> ExprId {
    match op {
        UnaryOp::Exp => {
            // d/dx limexp(a) = exp(a) below the threshold, the constant slope
            // exp(EXP_LIMIT) on the linear tail. Written as a `select` over the
            // PRIMAL `exp(a)` node (not `exp(min(a, EXP_LIMIT))`), so
            // hash-consing shares the one exp between residual and Jacobian:
            // one transcendental per junction per evaluation instead of two.
            // Values are identical: for a <= EXP_LIMIT `unary_f64` evaluates
            // the bare `a.exp()`.
            // The condition names the out-of-range side, so a NaN `a`
            // takes the exp arm and stays NaN, as the value does.
            let hi = ctx.konst_f64(crate::semantics::EXP_LIMIT);
            let above = ctx.cmp(CmpOp::Gt, a, hi);
            let ea = ctx.exp(a);
            let slope = ctx.konst_f64(crate::semantics::EXP_LIMIT.exp());
            ctx.select(above, slope, ea)
        }
        UnaryOp::Ln => {
            // 1/a above the floor, 0 at or below it (ln is clamped flat
            // there); a NaN `a` takes the 1/a arm.
            let lo = ctx.konst_f64(crate::semantics::LN_FLOOR);
            let clamped = ctx.cmp(CmpOp::Le, a, lo);
            let inv_a = ctx.recip(a);
            let zero = ctx.zero();
            ctx.select(clamped, zero, inv_a)
        }
        UnaryOp::Sqrt => {
            // 1/(2*sqrt(a)) for a>0, else 0 (matches sqrt clamped to 0);
            // a NaN `a` takes the first arm.
            let s = ctx.sqrt(a);
            let rs = ctx.recip(s);
            let half = ctx.ratio(1, 2);
            let d = ctx.mul(half, rs);
            let zero = ctx.zero();
            let clamped = ctx.cmp(CmpOp::Le, a, zero);
            ctx.select(clamped, zero, d)
        }
        UnaryOp::Sin => ctx.cos(a),
        UnaryOp::Cos => {
            let s = ctx.sin(a);
            ctx.neg(s)
        }
        UnaryOp::Floor => ctx.zero(), // piecewise-constant: 0 a.e. (factor 0)
        UnaryOp::Sinh => ctx.cosh(a),
        UnaryOp::Cosh => ctx.sinh(a),
        UnaryOp::Tanh => {
            // 1 - tanh(a)^2
            let t = ctx.tanh(a);
            let t2 = ctx.mul(t, t);
            let one = ctx.one();
            ctx.sub(one, t2)
        }
        UnaryOp::Atan => {
            // 1 / (1 + a^2)
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let denom = ctx.add(one, a2);
            ctx.recip(denom)
        }
        UnaryOp::Tan => {
            // 1 + tan^2
            let t = ctx.unary(UnaryOp::Tan, a);
            let t2 = ctx.mul(t, t);
            let one = ctx.one();
            ctx.add(one, t2)
        }
        UnaryOp::Log10 => {
            let r = ctx.recip(a);
            let c = ctx.konst_f64(1.0 / std::f64::consts::LN_10);
            ctx.mul(c, r)
        }
        UnaryOp::Log2 => {
            let r = ctx.recip(a);
            let c = ctx.konst_f64(1.0 / std::f64::consts::LN_2);
            ctx.mul(c, r)
        }
        UnaryOp::Log1p => {
            let one = ctx.one();
            let d = ctx.add(one, a);
            ctx.recip(d)
        }
        UnaryOp::Expm1 => ctx.exp(a),
        UnaryOp::Cbrt => {
            // 1 / (3 cbrt(a)^2)
            let c = ctx.unary(UnaryOp::Cbrt, a);
            let c2 = ctx.mul(c, c);
            let three = ctx.konst_int(3);
            let d = ctx.mul(three, c2);
            ctx.recip(d)
        }
        UnaryOp::Abs => ctx.unary(UnaryOp::Sign, a),
        UnaryOp::Sign | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Trunc | UnaryOp::RandUniform => {
            ctx.zero()
        }
        UnaryOp::Asin | UnaryOp::Acos => {
            // +-1 / sqrt(1 - a^2)
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let d = ctx.sub(one, a2);
            let s = ctx.sqrt(d);
            let r = ctx.recip(s);
            if op == UnaryOp::Asin {
                r
            } else {
                ctx.neg(r)
            }
        }
        UnaryOp::Asinh => {
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let d = ctx.add(one, a2);
            let s = ctx.sqrt(d);
            ctx.recip(s)
        }
        UnaryOp::Acosh => {
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let d = ctx.sub(a2, one);
            let s = ctx.sqrt(d);
            ctx.recip(s)
        }
        UnaryOp::Atanh => {
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let d = ctx.sub(one, a2);
            ctx.recip(d)
        }
        UnaryOp::Erf | UnaryOp::Erfc => {
            // +-2/sqrt(pi) exp(-a^2)
            let a2 = ctx.mul(a, a);
            let na2 = ctx.neg(a2);
            let e = ctx.exp(na2);
            let c = ctx.konst_f64(std::f64::consts::FRAC_2_SQRT_PI);
            let d = ctx.mul(c, e);
            if op == UnaryOp::Erf {
                d
            } else {
                ctx.neg(d)
            }
        }
        UnaryOp::Lgamma => ctx.unary(UnaryOp::Digamma, a),
        UnaryOp::Tgamma => {
            let g = ctx.unary(UnaryOp::Tgamma, a);
            let p = ctx.unary(UnaryOp::Digamma, a);
            ctx.mul(g, p)
        }
        UnaryOp::Digamma => ctx.unary(UnaryOp::Trigamma, a),
        UnaryOp::Trigamma => {
            panic!("rsdag: the derivative of trigamma (polygamma of order 2) is not available")
        }
    }
}

/// Partial derivatives `[d/da, d/db]` of a binary function, each built only
/// if `want` asks for it (zero otherwise).
fn binary_partials<K: Field>(
    ctx: &mut Graph<K>,
    op: BinOp,
    a: ExprId,
    b: ExprId,
    want: [bool; 2],
) -> [ExprId; 2] {
    let zero = ctx.zero();
    match op {
        BinOp::Powf => {
            // d/da = b a^(b-1), d/db = a^b ln a
            let da = if want[0] {
                let one = ctx.one();
                let bm1 = ctx.sub(b, one);
                let p = ctx.binary(BinOp::Powf, a, bm1);
                ctx.mul(b, p)
            } else {
                zero
            };
            let db = if want[1] {
                let ab = ctx.binary(BinOp::Powf, a, b);
                let ln = ctx.ln(a);
                ctx.mul(ab, ln)
            } else {
                zero
            };
            [da, db]
        }
        BinOp::Mod => {
            // fmod(a, b) = a - b trunc(a/b): d/da = 1, d/db = -trunc(a/b)
            let db = if want[1] {
                let q = ctx.div(a, b);
                let t = ctx.unary(UnaryOp::Trunc, q);
                ctx.neg(t)
            } else {
                zero
            };
            [ctx.one(), db]
        }
        BinOp::Atan2 => {
            // d/da = b/(a^2+b^2), d/db = -a/(a^2+b^2)
            let a2 = ctx.mul(a, a);
            let b2 = ctx.mul(b, b);
            let s = ctx.add(a2, b2);
            let r = ctx.recip(s);
            let da = if want[0] { ctx.mul(b, r) } else { zero };
            let db = if want[1] {
                let ar = ctx.mul(a, r);
                ctx.neg(ar)
            } else {
                zero
            };
            [da, db]
        }
        BinOp::Hypot => {
            let h = ctx.binary(BinOp::Hypot, a, b);
            let r = ctx.recip(h);
            let da = if want[0] { ctx.mul(a, r) } else { zero };
            let db = if want[1] { ctx.mul(b, r) } else { zero };
            [da, db]
        }
    }
}

/// For each factor of a product `args` that `want` asks for, the product of
/// the others, from prefix and suffix products: linear in the factors.
fn cofactors<K: Field>(
    ctx: &mut Graph<K>,
    args: &[ExprId],
    want: impl Fn(usize) -> bool,
) -> Vec<Option<ExprId>> {
    let mut prefix = Vec::with_capacity(args.len());
    let mut acc = ctx.one();
    for &x in args {
        prefix.push(acc);
        acc = ctx.mul(acc, x);
    }
    let mut out = vec![None; args.len()];
    let mut suffix = ctx.one();
    for i in (0..args.len()).rev() {
        if want(i) {
            out[i] = Some(ctx.mul(prefix[i], suffix));
        }
        suffix = ctx.mul(suffix, args[i]);
    }
    out
}

/// How many leading operands of `node` carry no derivative: both of a
/// comparison's (piecewise constant), a select's condition.
fn inert(node: &Node) -> usize {
    match node {
        Node::Cmp(..) => 2,
        Node::Select(..) => 1,
        _ => 0,
    }
}

/// Whether `test` holds for every operand of `e` that carries a derivative
/// (see [`carrying`]), without collecting them: the per-node test of a
/// forward sweep.
fn all_carrying<K: Field>(ctx: &Graph<K>, e: ExprId, mut test: impl FnMut(ExprId) -> bool) -> bool {
    match *ctx.node(e) {
        Node::Call(o, l) => {
            let (f, k) = ctx.output(o);
            ctx.output_support(f, k)
                .iter()
                .all(|&p| test(ctx.call_operand(o, l, p)))
        }
        ref node => ctx.operands(e)[inert(node)..].iter().all(|&c| test(c)),
    }
}

/// The operands of `e` its derivative reads, in operand order, into `out`:
/// all but a comparison's and a selector's condition, and of a call only the
/// arguments the called output's support names (see
/// [`Graph::output_support`]), so a derivative through a call is as sparse
/// as the body it calls.
pub(crate) fn carrying<K: Field>(ctx: &Graph<K>, e: ExprId, out: &mut Vec<ExprId>) {
    out.clear();
    match *ctx.node(e) {
        Node::Call(o, l) => {
            let (f, k) = ctx.output(o);
            out.extend(
                ctx.output_support(f, k)
                    .iter()
                    .map(|&p| ctx.call_operand(o, l, p)),
            );
        }
        ref node => out.extend_from_slice(&ctx.operands(e)[inert(node)..]),
    }
}

/// The nodes under `root` through operands that carry a derivative, marked
/// in `seen`, in ascending id order: a topological one, a hash-consed node
/// having a larger id than its operands. Through a call only the operands
/// its output's support names, so a row of a wide body's single call walks
/// that row's share of it, not the call's whole argument list.
fn cone<K: Field>(ctx: &Graph<K>, root: ExprId, seen: &mut Memo) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    let mut ops = Vec::new();
    while let Some(e) = stack.pop() {
        if seen.get(e).is_none() {
            seen.set(e, e);
            out.push(e);
            carrying(ctx, e, &mut ops);
            stack.extend_from_slice(&ops);
        }
    }
    out.sort_unstable();
    out
}

/// One forward sweep: `d root / d wrt` for every root. The derivative of
/// every node below is kept in `memo`, so a subexpression shared by the
/// roots (or by the roots of a later sweep over the same `wrt` and memo)
/// is differentiated once. Operands first, in order, on an explicit stack
/// (the order a recursive walk takes), so a deep chain does not overflow
/// the call stack.
fn forward<K: Field>(
    ctx: &mut Graph<K>,
    roots: &[ExprId],
    wrt: SymbolId,
    memo: &mut Memo,
) -> Vec<ExprId> {
    let mut stack: Vec<(ExprId, bool)> = Vec::with_capacity(64);
    stack.extend(roots.iter().rev().map(|&r| (r, false)));
    let mut ops = Vec::new();
    while let Some((e, expanded)) = stack.pop() {
        // Done already: shared, or a solve component set by a sibling.
        if memo.get(e).is_some() {
            continue;
        }
        if expanded {
            let d = tangent(ctx, e, wrt, memo);
            memo.set(e, d);
        } else {
            stack.push((e, true));
            carrying(ctx, e, &mut ops);
            let pending = ops.iter().rev().filter(|&&c| memo.get(c).is_none());
            stack.extend(pending.map(|&c| (c, false)));
        }
    }
    roots
        .iter()
        .map(|&r| memo.get(r).expect("root differentiated"))
        .collect()
}

/// The derivative of node `e` with respect to `wrt`, its operands' already
/// in `memo`. A node none of whose operands moves is zero: an inactive
/// subgraph builds nothing.
fn tangent<K: Field>(ctx: &mut Graph<K>, e: ExprId, wrt: SymbolId, memo: &mut Memo) -> ExprId {
    let zero = ctx.zero();
    let node = *ctx.node(e);
    if let Node::Symbol(s) = node {
        return if s == wrt { ctx.one() } else { zero };
    }
    let d = |c: ExprId| memo.get(c).expect("operand differentiated");
    if all_carrying(ctx, e, |c| d(c) == zero) {
        return zero;
    }
    match node {
        Node::Const(_) | Node::Symbol(_) | Node::Cmp(..) => unreachable!("no moving operand"),
        Node::Add(a, b) => ctx.add(d(a), d(b)),
        Node::Mul(a, b) => {
            // product rule: da*b + a*db
            let t1 = ctx.mul(d(a), b);
            let t2 = ctx.mul(a, d(b));
            ctx.add(t1, t2)
        }
        Node::Neg(a) => ctx.neg(d(a)),
        Node::Pow(a, n) => {
            // power rule (integer exponent): n * a^(n-1) * da
            let coeff = ctx.konst_int(n);
            let p = ctx.pow_i(a, n - 1);
            let cp = ctx.mul(coeff, p);
            ctx.mul(cp, d(a))
        }
        Node::Unary(op, a) => {
            let factor = unary_factor(ctx, op, a);
            ctx.mul(factor, d(a))
        }
        Node::Binary(op, a, b) => {
            let (da, db) = (d(a), d(b));
            let [pa, pb] = binary_partials(ctx, op, a, b, [da != zero, db != zero]);
            let t1 = ctx.mul(pa, da);
            let t2 = ctx.mul(pb, db);
            ctx.add(t1, t2)
        }
        // Subgradient: differentiate through both branches, keep the condition.
        Node::Select(c, t, f) => ctx.select(c, d(t), d(f)),
        Node::Reduce(op, l) => match op {
            // d(Σ aᵢ) = Σ daᵢ
            ReduceOp::Sum => {
                let dargs: Vec<ExprId> = ctx.args(l).iter().map(|&a| d(a)).collect();
                ctx.reduce(ReduceOp::Sum, dargs)
            }
            // d(Π aᵢ) = Σᵢ daᵢ · Πⱼ≠ᵢ aⱼ  (generalized product rule)
            ReduceOp::Product => {
                let args = ctx.args(l).to_vec();
                let others = cofactors(ctx, &args, |i| d(args[i]) != zero);
                let mut terms = Vec::with_capacity(args.len());
                for (&a, other) in args.iter().zip(others) {
                    if let Some(other) = other {
                        terms.push(ctx.mul(d(a), other));
                    }
                }
                ctx.reduce(ReduceOp::Sum, terms)
            }
            // Subgradient: the derivative of the (first) extremal argument.
            ReduceOp::Min | ReduceOp::Max => {
                let args = ctx.args(l).to_vec();
                let cmp = if op == ReduceOp::Max {
                    CmpOp::Gt
                } else {
                    CmpOp::Lt
                };
                let mut m = args[0];
                let mut dm = d(args[0]);
                for &a in &args[1..] {
                    let cond = ctx.cmp(cmp, a, m);
                    dm = ctx.select(cond, d(a), dm);
                    m = ctx.reduce(op, vec![m, a]);
                }
                dm
            }
        },
        // d(Σ aᵢbᵢ) = Σ (daᵢ·bᵢ + aᵢ·dbᵢ)
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            let (a, b) = (a.to_vec(), b.to_vec());
            let mut terms = Vec::with_capacity(a.len());
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let t1 = ctx.mul(d(ai), bi);
                let t2 = ctx.mul(ai, d(bi));
                terms.push(ctx.add(t1, t2));
            }
            ctx.reduce(ReduceOp::Sum, terms)
        }
        // x = A^-1 b: dx = A^-1 (db - dA x), another solve over the same
        // matrix, with the solution's components as they are; built once
        // for the system and memoized for every component.
        Node::Solve(l, i) => {
            let (n, a, b) = ctx.solve_args(l);
            let (a, b) = (a.to_vec(), b.to_vec());
            let x = ctx.solve_dense(a.clone(), b.clone());
            let mut rhs = Vec::with_capacity(n);
            for r in 0..n {
                let da: Vec<ExprId> = a[r * n..(r + 1) * n].iter().map(|&c| d(c)).collect();
                let dax = ctx.dot(da, x.clone());
                rhs.push(ctx.sub(d(b[r]), dax));
            }
            let dx = ctx.solve_dense(a, rhs);
            for (&xk, &dk) in x.iter().zip(&dx) {
                memo.set(xk, dk);
            }
            dx[i as usize]
        }
        // Chain rule through a call: d/dx f_out(a) = Σ_i (∂f_out/∂p_i)(a) · da_i,
        // each partial a call into the function's derivative output.
        Node::Call(o, l) => {
            let (f, out) = ctx.output(o);
            let moving: Vec<(u32, ExprId)> = ctx
                .output_support(f, out)
                .iter()
                .map(|&i| (i, d(ctx.call_operand(o, l, i))))
                .filter(|&(_, da)| da != zero)
                .collect();
            let params: Vec<u32> = moving.iter().map(|&(i, _)| i).collect();
            let ks = ctx.derivative_outputs(f, out, &params);
            let mut acc = zero;
            for (&(_, dai), k) in moving.iter().zip(ks) {
                // over the call's own list: no width to hash again
                let partial = ctx.call_list_in(f, k, ctx.context_of(o), l);
                let term = ctx.mul(partial, dai);
                acc = ctx.add(acc, term);
            }
            acc
        }
    }
}

/// A sparse matrix of expressions: for each row, the `(column, entry)`
/// pairs that are not the structural zero, in ascending column order.
pub type SparseRows = Vec<Vec<(usize, ExprId)>>;

/// Rows of this many touched unknowns and more are differentiated in
/// reverse mode, one adjoint sweep for the whole row; below it, forward
/// sweeps per unknown, shared by every row that touches it, are cheaper.
pub const REVERSE_MIN_TOUCHED: usize = 16;

/// Sparse Jacobian of `residuals` with respect to `wrt`: row `i` lists the
/// nonzero `(column, d residuals[i] / d wrt[column])`, by column.
///
/// Only the symbols in a row's support can have a nonzero derivative (see
/// [`Graph::support_in`]): through a call, the ones the called output reads.
/// A row that touches few unknowns is differentiated forward, one sweep
/// per unknown, and a sweep is shared by every such row that touches that
/// unknown, so a subexpression common to several rows (a device current
/// in two node equations) is differentiated once per unknown, not once per
/// row. A row that touches [`REVERSE_MIN_TOUCHED`] unknowns or more (a
/// scalar output over a deep shared graph, a node many devices meet at)
/// takes one reverse sweep instead: its cost is the row's graph once, not
/// once per unknown.
pub fn sparse_jacobian<K: Field>(
    ctx: &mut Graph<K>,
    residuals: &[ExprId],
    wrt: &[SymbolId],
) -> SparseRows {
    let col: rustc_hash::FxHashMap<SymbolId, u32> = wrt
        .iter()
        .enumerate()
        .map(|(j, &s)| (s, j as u32))
        .collect();
    // every row's columns in one pass over the rows' cone
    let flow = ctx.flow(residuals, Through::Carries, |n| match *n {
        Node::Symbol(s) => col
            .get(&s)
            .map_or(Set::bottom(), |&j| Set::one(j, wrt.len())),
        _ => Set::bottom(),
    });
    let touched: Vec<Vec<(usize, SymbolId)>> = residuals
        .iter()
        .map(|&r| {
            flow.get(r)
                .iter()
                .map(|j| (j as usize, wrt[j as usize]))
                .collect()
        })
        .collect();
    drop(flow);
    let mut rows: SparseRows = vec![Vec::new(); residuals.len()];
    // Forward rows by the columns they touch, rows in order within one.
    let mut by_col: Vec<Vec<usize>> = vec![Vec::new(); wrt.len()];
    for (i, t) in touched.iter().enumerate() {
        if t.len() >= REVERSE_MIN_TOUCHED {
            let syms: Vec<SymbolId> = t.iter().map(|&(_, s)| s).collect();
            let g = gradient(ctx, residuals[i], &syms);
            rows[i] = t.iter().map(|&(j, _)| j).zip(g).collect();
        } else {
            for &(j, _) in t {
                by_col[j].push(i);
            }
        }
    }
    let mut memo = ctx.take_memo();
    for (j, members) in by_col.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        memo.begin(ctx.len());
        let roots: Vec<ExprId> = members.iter().map(|&i| residuals[i]).collect();
        let col = forward(ctx, &roots, wrt[j], &mut memo);
        for (&i, d) in members.iter().zip(col) {
            rows[i].push((j, d));
        }
    }
    ctx.put_memo(memo);
    for row in rows.iter_mut() {
        row.retain(|&(_, e)| !ctx.is_zero(e));
    }
    rows
}

/// Reverse-mode symbolic gradient: `d(f)/d(wrt[j])` for every `j`, built in ONE
/// adjoint sweep over the reachable sub-DAG instead of one forward sweep per
/// symbol -- the right shape for a scalar objective over many leaves (a
/// parameter gradient). The local rules (domain-guard mirroring, `Select` /
/// `Min` / `Max` subgradients) are shared with [`differentiate`], so both modes
/// return the same values everywhere; only the graph shape of the result
/// differs. The result is an ordinary expression in the same context, so it can
/// be differentiated again (see [`hessian`]).
pub fn gradient<K: Field>(ctx: &mut Graph<K>, f: ExprId, wrt: &[SymbolId]) -> Vec<ExprId> {
    // The cone of f, ascending: walked backwards, every node comes after
    // all of its consumers. `pos` is a node's place in it.
    let mut pos = ctx.take_memo();
    let nodes = cone(ctx, f, &mut pos);
    for (k, &e) in nodes.iter().enumerate() {
        pos.set(e, ExprId(k as u32));
    }
    let at = |e: ExprId| pos.get(e).expect("in the cone").0 as usize;
    // Activity: a node is active if it depends on a symbol of `wrt`. Only
    // active operands receive an adjoint, so a subgraph over other symbols
    // (the parameters not asked for, a call's constant arguments) builds
    // nothing.
    let wanted: rustc_hash::FxHashSet<SymbolId> = wrt.iter().copied().collect();
    let mut active = vec![false; nodes.len()];
    let mut ops = Vec::new();
    for (k, &e) in nodes.iter().enumerate() {
        active[k] = match *ctx.node(e) {
            Node::Symbol(s) => wanted.contains(&s),
            _ => {
                carrying(ctx, e, &mut ops);
                ops.iter().any(|&c| active[at(c)])
            }
        };
    }
    let act = |e: ExprId| active[at(e)];

    // Adjoint accumulation: per node a term list, folded into one fused
    // Reduce(Sum) when the node is visited (all consumers seen by then).
    let mut adj: Vec<Vec<ExprId>> = vec![Vec::new(); nodes.len()];
    if act(f) {
        adj[at(f)].push(ctx.one());
    }
    let push = |adj: &mut Vec<Vec<ExprId>>, child: ExprId, term: ExprId| {
        adj[at(child)].push(term);
    };
    let zero = ctx.zero();
    let mut sym_adj: HashMap<SymbolId, ExprId> = HashMap::default();
    for k in (0..nodes.len()).rev() {
        if adj[k].is_empty() {
            continue; // no adjoint: inactive, or no value path from f
        }
        let e = nodes[k];
        let terms = std::mem::take(&mut adj[k]);
        let a_bar = ctx.reduce(ReduceOp::Sum, terms);
        if a_bar == zero {
            continue;
        }
        match *ctx.node(e) {
            // Piecewise-constant or constant: never active.
            Node::Const(_) | Node::Cmp(..) => {}
            Node::Symbol(s) => {
                sym_adj.insert(s, a_bar);
            }
            Node::Add(x, y) => {
                for c in [x, y] {
                    if act(c) {
                        push(&mut adj, c, a_bar);
                    }
                }
            }
            Node::Mul(x, y) => {
                if act(x) {
                    let t = ctx.mul(a_bar, y);
                    push(&mut adj, x, t);
                }
                if act(y) {
                    let t = ctx.mul(a_bar, x);
                    push(&mut adj, y, t);
                }
            }
            Node::Neg(x) => {
                let t = ctx.neg(a_bar);
                push(&mut adj, x, t);
            }
            Node::Pow(x, n) => {
                // d/dx x^n = n * x^(n-1)
                let coeff = ctx.konst_int(n);
                let p = ctx.pow_i(x, n - 1);
                let cp = ctx.mul(coeff, p);
                let t = ctx.mul(a_bar, cp);
                push(&mut adj, x, t);
            }
            Node::Unary(op, x) => {
                let factor = unary_factor(ctx, op, x);
                let t = ctx.mul(a_bar, factor);
                push(&mut adj, x, t);
            }
            Node::Binary(op, x, y) => {
                let p = binary_partials(ctx, op, x, y, [act(x), act(y)]);
                for (c, p) in [(x, p[0]), (y, p[1])] {
                    if act(c) {
                        let t = ctx.mul(a_bar, p);
                        push(&mut adj, c, t);
                    }
                }
            }
            // Subgradient: the adjoint flows into the taken branch only
            // (matching the forward rule d = select(c, dt, de)).
            Node::Select(c, t, e2) => {
                if act(t) {
                    let tt = ctx.select(c, a_bar, zero);
                    push(&mut adj, t, tt);
                }
                if act(e2) {
                    let te = ctx.select(c, zero, a_bar);
                    push(&mut adj, e2, te);
                }
            }
            Node::Reduce(op, l) => match op {
                ReduceOp::Sum => {
                    for &x in ctx.args(l) {
                        if act(x) {
                            push(&mut adj, x, a_bar);
                        }
                    }
                }
                // d(Π aᵢ)/daᵢ = Πⱼ≠ᵢ aⱼ, via prefix/suffix products (O(k) nodes).
                ReduceOp::Product => {
                    let args = ctx.args(l).to_vec();
                    let others = cofactors(ctx, &args, |i| act(args[i]));
                    for (&a, other) in args.iter().zip(others) {
                        if let Some(other) = other {
                            let t = ctx.mul(a_bar, other);
                            push(&mut adj, a, t);
                        }
                    }
                }
                // Subgradient of the (first) extremal argument, exactly the
                // forward rule's select chain written as 0/1 coefficients:
                // coef_i = cond_i * Π_{k>i} (1 - cond_k), cond_0 = 1, with
                // cond_k = (args[k] <op-cmp> running extremum of args[..k]).
                ReduceOp::Min | ReduceOp::Max => {
                    let args = ctx.args(l).to_vec();
                    let cmp = if op == ReduceOp::Max {
                        CmpOp::Gt
                    } else {
                        CmpOp::Lt
                    };
                    let n = args.len();
                    let mut m = args[0];
                    let mut conds = Vec::with_capacity(n.saturating_sub(1));
                    for &x in &args[1..] {
                        let c = ctx.cmp(cmp, x, m);
                        conds.push(c);
                        m = ctx.reduce(op, vec![m, x]);
                    }
                    let one = ctx.one();
                    let mut tail = one;
                    for i in (0..n).rev() {
                        if act(args[i]) {
                            let c_i = if i == 0 { one } else { conds[i - 1] };
                            let coef = ctx.mul(c_i, tail);
                            let t = ctx.mul(a_bar, coef);
                            push(&mut adj, args[i], t);
                        }
                        if i > 0 {
                            let not_c = ctx.sub(one, conds[i - 1]);
                            tail = ctx.mul(tail, not_c);
                        }
                    }
                }
            },
            Node::Dot(l) => {
                let (xs, ys) = ctx.dot_args(l);
                let (xs, ys) = (xs.to_vec(), ys.to_vec());
                for (&x, &y) in xs.iter().zip(ys.iter()) {
                    if act(x) {
                        let t = ctx.mul(a_bar, y);
                        push(&mut adj, x, t);
                    }
                    if act(y) {
                        let t = ctx.mul(a_bar, x);
                        push(&mut adj, y, t);
                    }
                }
            }
            // The whole system at once: the components of one solve have
            // consecutive ids and every consumer a larger one, so at the
            // first component visited every component's adjoint is final.
            // Then b_bar = A^-T x_bar and A_bar = -b_bar x^T, one transposed
            // solve for the system instead of one per component.
            Node::Solve(l, _) => {
                let (n, a, b) = ctx.solve_args(l);
                let (a, b) = (a.to_vec(), b.to_vec());
                let x = ctx.solve_dense(a.clone(), b.clone());
                let mut x_bar = vec![zero; n];
                for (k, &xk) in x.iter().enumerate() {
                    x_bar[k] = if xk == e {
                        a_bar
                    } else {
                        match pos.get(xk) {
                            Some(p) => {
                                let t = std::mem::take(&mut adj[p.0 as usize]);
                                ctx.reduce(ReduceOp::Sum, t)
                            }
                            None => zero,
                        }
                    };
                }
                let a_t: Vec<ExprId> = (0..n * n).map(|k| a[(k % n) * n + k / n]).collect();
                let b_bar = ctx.solve_dense(a_t, x_bar);
                for r in 0..n {
                    if act(b[r]) {
                        push(&mut adj, b[r], b_bar[r]);
                    }
                    for j in 0..n {
                        if act(a[r * n + j]) {
                            let t = ctx.mul(b_bar[r], x[j]);
                            let nt = ctx.neg(t);
                            push(&mut adj, a[r * n + j], nt);
                        }
                    }
                }
            }
            // Chain rule through a call: the same derivative outputs as the
            // forward mode, for the arguments that move.
            Node::Call(o, l) => {
                let (func, out) = ctx.output(o);
                let moving: Vec<(u32, ExprId)> = ctx
                    .output_support(func, out)
                    .iter()
                    .map(|&i| (i, ctx.call_operand(o, l, i)))
                    .filter(|&(_, a)| act(a))
                    .collect();
                let params: Vec<u32> = moving.iter().map(|&(i, _)| i).collect();
                let ks = ctx.derivative_outputs(func, out, &params);
                for (&(_, arg), k) in moving.iter().zip(ks) {
                    let partial = ctx.call_list_in(func, k, ctx.context_of(o), l);
                    let t = ctx.mul(a_bar, partial);
                    push(&mut adj, arg, t);
                }
            }
        }
    }
    ctx.put_memo(pos);
    wrt.iter()
        .map(|s| sym_adj.get(s).copied().unwrap_or(zero))
        .collect()
}

/// Symbolic Hessian `hess[i][j] = d²f / d(wrt[i]) d(wrt[j])`, built
/// forward-over-reverse: one reverse sweep for the gradient, then one forward
/// sweep per column over the gradient entries up to the diagonal, the
/// lower triangle the mirror of the upper, so the result is exactly
/// symmetric. Like every derivative here it is an ordinary expression, so
/// third and higher orders are just repeated application.
pub fn hessian<K: Field>(ctx: &mut Graph<K>, f: ExprId, wrt: &[SymbolId]) -> Vec<Vec<ExprId>> {
    let grad = gradient(ctx, f, wrt);
    let n = wrt.len();
    let mut hess = vec![vec![ctx.zero(); n]; n];
    let mut memo = ctx.take_memo();
    for j in 0..n {
        memo.begin(ctx.len());
        for (i, d) in forward(ctx, &grad[..=j], wrt[j], &mut memo)
            .into_iter()
            .enumerate()
        {
            hess[i][j] = d;
            hess[j][i] = d;
        }
    }
    ctx.put_memo(memo);
    hess
}
