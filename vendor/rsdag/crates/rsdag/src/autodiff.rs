//! Symbolic differentiation over the DAG.
//!
//! Builds the derivative as new nodes in the same context, so common
//! subexpressions are shared (hash-consing) and the result can be exported or
//! differentiated again. Used to produce analytic Jacobians for DAE export.

use crate::field::Field;
use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Graph, Memo};
use crate::node::{BinOp, CmpOp, ExprId, Node, ReduceOp, SymbolId, UnaryOp};

/// Derivative of `expr` with respect to the symbol `wrt`.
pub fn differentiate<K: Field>(ctx: &mut Graph<K>, expr: ExprId, wrt: SymbolId) -> ExprId {
    let mut memo = ctx.take_memo();
    let d = diff(ctx, expr, wrt, &mut memo);
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

/// Partial derivatives `(d/da, d/db)` of a binary function.
fn binary_partials<K: Field>(
    ctx: &mut Graph<K>,
    op: BinOp,
    a: ExprId,
    b: ExprId,
) -> (ExprId, ExprId) {
    match op {
        BinOp::Powf => {
            // d/da = b a^(b-1), d/db = a^b ln a
            let one = ctx.one();
            let bm1 = ctx.sub(b, one);
            let p = ctx.binary(BinOp::Powf, a, bm1);
            let da = ctx.mul(b, p);
            let ab = ctx.binary(BinOp::Powf, a, b);
            let ln = ctx.ln(a);
            let db = ctx.mul(ab, ln);
            (da, db)
        }
        BinOp::Mod => {
            // fmod(a, b) = a - b trunc(a/b): d/da = 1, d/db = -trunc(a/b)
            let q = ctx.div(a, b);
            let t = ctx.unary(UnaryOp::Trunc, q);
            let one = ctx.one();
            (one, ctx.neg(t))
        }
        BinOp::Atan2 => {
            // d/da = b/(a^2+b^2), d/db = -a/(a^2+b^2)
            let a2 = ctx.mul(a, a);
            let b2 = ctx.mul(b, b);
            let s = ctx.add(a2, b2);
            let r = ctx.recip(s);
            let da = ctx.mul(b, r);
            let ar = ctx.mul(a, r);
            (da, ctx.neg(ar))
        }
        BinOp::Hypot => {
            let h = ctx.binary(BinOp::Hypot, a, b);
            let r = ctx.recip(h);
            (ctx.mul(a, r), ctx.mul(b, r))
        }
    }
}

fn diff<K: Field>(ctx: &mut Graph<K>, expr: ExprId, wrt: SymbolId, memo: &mut Memo) -> ExprId {
    if let Some(d) = memo.get(expr) {
        return d;
    }
    // Copy the node (16 bytes) so we can mutate the context while building the
    // derivative; variadic operand lists are copied out of the pool below.
    let node = *ctx.node(expr);
    let d = match node {
        Node::Const(_) => ctx.zero(),
        Node::Symbol(s) => {
            if s == wrt {
                ctx.one()
            } else {
                ctx.zero()
            }
        }
        Node::Add(a, b) => {
            let da = diff(ctx, a, wrt, memo);
            let db = diff(ctx, b, wrt, memo);
            ctx.add(da, db)
        }
        Node::Mul(a, b) => {
            // product rule: da*b + a*db
            let da = diff(ctx, a, wrt, memo);
            let db = diff(ctx, b, wrt, memo);
            let t1 = ctx.mul(da, b);
            let t2 = ctx.mul(a, db);
            ctx.add(t1, t2)
        }
        Node::Neg(a) => {
            let da = diff(ctx, a, wrt, memo);
            ctx.neg(da)
        }
        Node::Pow(a, n) => {
            // power rule (integer exponent): n * a^(n-1) * da
            let da = diff(ctx, a, wrt, memo);
            let coeff = ctx.konst_int(n);
            let p = ctx.pow_i(a, n - 1);
            let cp = ctx.mul(coeff, p);
            ctx.mul(cp, da)
        }
        Node::Unary(op, a) => {
            let da = diff(ctx, a, wrt, memo);
            let factor = unary_factor(ctx, op, a);
            ctx.mul(factor, da)
        }
        // Comparisons are piecewise-constant: derivative is zero a.e.
        Node::Binary(op, a, b) => {
            let da = diff(ctx, a, wrt, memo);
            let db = diff(ctx, b, wrt, memo);
            let (pa, pb) = binary_partials(ctx, op, a, b);
            let t1 = ctx.mul(pa, da);
            let t2 = ctx.mul(pb, db);
            ctx.add(t1, t2)
        }
        Node::Cmp(..) => ctx.zero(),
        // Subgradient: differentiate through both branches, keep the condition.
        Node::Select(c, t, e) => {
            let dt = diff(ctx, t, wrt, memo);
            let de = diff(ctx, e, wrt, memo);
            ctx.select(c, dt, de)
        }
        Node::Reduce(op, l) => match op {
            // d(Σ aᵢ) = Σ daᵢ
            ReduceOp::Sum => {
                let args = ctx.args(l).to_vec();
                let dargs: Vec<ExprId> = args.iter().map(|&a| diff(ctx, a, wrt, memo)).collect();
                ctx.reduce(ReduceOp::Sum, dargs)
            }
            // d(Π aᵢ) = Σᵢ daᵢ · Πⱼ≠ᵢ aⱼ  (generalized product rule)
            ReduceOp::Product => {
                let args = ctx.args(l).to_vec();
                let mut terms = Vec::with_capacity(args.len());
                for i in 0..args.len() {
                    let dai = diff(ctx, args[i], wrt, memo);
                    if ctx.is_zero(dai) {
                        continue;
                    }
                    let others: Vec<ExprId> = args
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, &a)| a)
                        .collect();
                    let prod = ctx.reduce(ReduceOp::Product, others);
                    terms.push(ctx.mul(dai, prod));
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
                let mut dm = diff(ctx, args[0], wrt, memo);
                for &a in &args[1..] {
                    let da = diff(ctx, a, wrt, memo);
                    let cond = ctx.cmp(cmp, a, m);
                    dm = ctx.select(cond, da, dm);
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
                let dai = diff(ctx, ai, wrt, memo);
                let dbi = diff(ctx, bi, wrt, memo);
                let t1 = ctx.mul(dai, bi);
                let t2 = ctx.mul(ai, dbi);
                terms.push(ctx.add(t1, t2));
            }
            ctx.reduce(ReduceOp::Sum, terms)
        }
        // x = A^-1 b: dx = A^-1 (db - dA x), another solve over the same
        // matrix, with the solution's components as they are.
        Node::Solve(l, i) => {
            let (n, a, b) = ctx.solve_args(l);
            let (a, b) = (a.to_vec(), b.to_vec());
            let x = ctx.solve_dense(a.clone(), b.clone());
            let mut rhs = Vec::with_capacity(n);
            for r in 0..n {
                let db = diff(ctx, b[r], wrt, memo);
                let da: Vec<ExprId> = (0..n).map(|j| diff(ctx, a[r * n + j], wrt, memo)).collect();
                let dax = ctx.dot(da, x.clone());
                rhs.push(ctx.sub(db, dax));
            }
            if rhs.iter().all(|&e| ctx.is_zero(e)) {
                ctx.zero()
            } else {
                ctx.solve_dense(a, rhs)[i as usize]
            }
        }
        // Chain rule through a call: d/dx f_out(a) = Σ_i (∂f_out/∂p_i)(a) · da_i,
        // each partial a call into the function's derivative output.
        Node::Call(o, l) => {
            let args = ctx.args(l).to_vec();
            let (f, out) = ctx.output(o);
            let mut acc = ctx.zero();
            for (i, &arg) in args.iter().enumerate() {
                let dai = diff(ctx, arg, wrt, memo);
                if ctx.is_zero(dai) {
                    continue;
                }
                let k = ctx.derivative_output(f, out, i as u32);
                let partial = ctx.call(f, k, &args);
                let term = ctx.mul(partial, dai);
                acc = ctx.add(acc, term);
            }
            acc
        }
    };
    memo.set(expr, d);
    d
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
/// Only the symbols a row actually contains can have a nonzero derivative.
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
    let col: rustc_hash::FxHashMap<SymbolId, usize> =
        wrt.iter().enumerate().map(|(j, &s)| (s, j)).collect();
    let touched: Vec<Vec<(usize, SymbolId)>> = residuals
        .iter()
        .map(|&r| {
            let mut t: Vec<(usize, SymbolId)> = ctx
                .free_symbols(r)
                .into_iter()
                .filter_map(|s| col.get(&s).map(|&j| (j, s)))
                .collect();
            t.sort_unstable_by_key(|&(j, _)| j);
            t
        })
        .collect();
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
        for &i in members {
            let d = diff(ctx, residuals[i], wrt[j], &mut memo);
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
    // Reachable sub-DAG of f. Ascending ExprId is a topological order (a
    // hash-consed node has a larger id than its children), so iterating the
    // sorted set in REVERSE visits every node after all of its parents.
    let mut reach: std::collections::BTreeSet<ExprId> = std::collections::BTreeSet::new();
    let mut stack = vec![f];
    while let Some(e) = stack.pop() {
        if !reach.insert(e) {
            continue;
        }
        stack.extend_from_slice(&ctx.operands(e));
    }

    // Adjoint accumulation: per node a term list, folded into one fused
    // Reduce(Sum) when the node is visited (all parents seen by then).
    let mut adj: HashMap<ExprId, Vec<ExprId>> = HashMap::default();
    let one = ctx.one();
    adj.insert(f, vec![one]);
    let push = |adj: &mut HashMap<ExprId, Vec<ExprId>>, child: ExprId, term: ExprId| {
        adj.entry(child).or_default().push(term);
    };
    let mut sym_adj: HashMap<SymbolId, ExprId> = HashMap::default();
    for &e in reach.iter().rev() {
        let terms = match adj.remove(&e) {
            Some(t) => t,
            None => continue, // unreachable from f's value path (e.g. below a Cmp)
        };
        let a_bar = ctx.reduce(ReduceOp::Sum, terms);
        if ctx.is_zero(a_bar) {
            continue;
        }
        match *ctx.node(e) {
            Node::Const(_) => {}
            Node::Symbol(s) => {
                sym_adj.insert(s, a_bar);
            }
            Node::Add(x, y) => {
                push(&mut adj, x, a_bar);
                push(&mut adj, y, a_bar);
            }
            Node::Mul(x, y) => {
                let tx = ctx.mul(a_bar, y);
                let ty = ctx.mul(a_bar, x);
                push(&mut adj, x, tx);
                push(&mut adj, y, ty);
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
                let (px, py) = binary_partials(ctx, op, x, y);
                let tx = ctx.mul(a_bar, px);
                let ty = ctx.mul(a_bar, py);
                push(&mut adj, x, tx);
                push(&mut adj, y, ty);
            }
            // Piecewise-constant: no value path into the operands.
            Node::Cmp(..) => {}
            // Subgradient: the adjoint flows into the taken branch only
            // (matching the forward rule d = select(c, dt, de)).
            Node::Select(c, t, e2) => {
                let zero = ctx.zero();
                let tt = ctx.select(c, a_bar, zero);
                let te = ctx.select(c, zero, a_bar);
                push(&mut adj, t, tt);
                push(&mut adj, e2, te);
            }
            Node::Reduce(op, l) => match op {
                ReduceOp::Sum => {
                    for &x in ctx.args(l) {
                        push(&mut adj, x, a_bar);
                    }
                }
                // d(Π aᵢ)/daᵢ = Πⱼ≠ᵢ aⱼ, via prefix/suffix products (O(k) nodes).
                ReduceOp::Product => {
                    let args = ctx.args(l).to_vec();
                    let k = args.len();
                    let mut prefix = Vec::with_capacity(k);
                    let mut acc = ctx.one();
                    for &x in &args {
                        prefix.push(acc);
                        acc = ctx.mul(acc, x);
                    }
                    let mut suffix = ctx.one();
                    for i in (0..k).rev() {
                        let others = ctx.mul(prefix[i], suffix);
                        let t = ctx.mul(a_bar, others);
                        push(&mut adj, args[i], t);
                        suffix = ctx.mul(suffix, args[i]);
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
                        let c_i = if i == 0 { one } else { conds[i - 1] };
                        let coef = ctx.mul(c_i, tail);
                        let t = ctx.mul(a_bar, coef);
                        push(&mut adj, args[i], t);
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
                    let tx = ctx.mul(a_bar, y);
                    let ty = ctx.mul(a_bar, x);
                    push(&mut adj, x, tx);
                    push(&mut adj, y, ty);
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
                let mut x_bar = vec![ctx.zero(); n];
                for (k, &xk) in x.iter().enumerate() {
                    x_bar[k] = if xk == e {
                        a_bar
                    } else {
                        match adj.remove(&xk) {
                            Some(t) => ctx.reduce(ReduceOp::Sum, t),
                            None => ctx.zero(),
                        }
                    };
                }
                let at: Vec<ExprId> = (0..n * n).map(|k| a[(k % n) * n + k / n]).collect();
                let b_bar = ctx.solve_dense(at, x_bar);
                for r in 0..n {
                    push(&mut adj, b[r], b_bar[r]);
                    for j in 0..n {
                        let t = ctx.mul(b_bar[r], x[j]);
                        let nt = ctx.neg(t);
                        push(&mut adj, a[r * n + j], nt);
                    }
                }
            }
            // Chain rule through a call: the same derivative outputs as the
            // forward mode.
            Node::Call(o, l) => {
                let args = ctx.args(l).to_vec();
                let (f, out) = ctx.output(o);
                for (i, &arg) in args.iter().enumerate() {
                    let k = ctx.derivative_output(f, out, i as u32);
                    let partial = ctx.call(f, k, &args);
                    let t = ctx.mul(a_bar, partial);
                    push(&mut adj, arg, t);
                }
            }
        }
    }

    wrt.iter()
        .map(|s| sym_adj.get(s).copied().unwrap_or_else(|| ctx.zero()))
        .collect()
}

/// Symbolic Hessian `hess[i][j] = d²f / d(wrt[i]) d(wrt[j])`, built
/// forward-over-reverse: one reverse sweep for the gradient, then one forward
/// sweep per column. Like every derivative here it is an ordinary expression,
/// so third and higher orders are just repeated application.
pub fn hessian<K: Field>(ctx: &mut Graph<K>, f: ExprId, wrt: &[SymbolId]) -> Vec<Vec<ExprId>> {
    let grad = gradient(ctx, f, wrt);
    grad.iter()
        .map(|&g| wrt.iter().map(|&s| differentiate(ctx, g, s)).collect())
        .collect()
}
