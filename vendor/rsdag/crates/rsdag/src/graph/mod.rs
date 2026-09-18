use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet};

use num_rational::BigRational;

use crate::field::Field;

use crate::extern_fn::ExternBundle;
use crate::func::{FuncId, Function, FunctionBody, Output, OutputId};
use crate::node::{
    ArgList, BinOp, CmpOp, ConstId, ExprId, Node, Operands, ReduceOp, SymbolId, UnaryOp,
};
use crate::role::{OutputRole, ParamRole};
use crate::semantics::{binary_f64, unary_f64};

/// Owns the hash-consed symbolic DAG and the symbol table.
///
/// All expressions are built through this context. Smart constructors fold
/// constants and apply a handful of identities so trivially-equal expressions
/// collapse to the same [`ExprId`]. Heavier canonicalisation (factoring,
/// term collection) is left to a later rewrite layer.
///
/// Storage is three dense arenas plus their dedup indices: the 16-byte
/// [`Node`]s, the exact constants they reference by [`ConstId`], and one shared
/// operand pool that every variadic node windows into by [`ArgList`]. A node
/// is interned by hashing 16 bytes; a constant is hashed once when it is first
/// seen; an operand list is interned by content so equal lists share one
/// window (which is what makes `Reduce`/`Dot`/`Opaque` hash-cons structurally).
pub struct Graph<K: Field = BigRational> {
    nodes: Vec<Node>,
    dedup: HashMap<Node, ExprId>,
    consts: Vec<K>,
    const_dedup: HashMap<K, ConstId>,
    /// `f64` bit pattern -> constant node, so a numeric literal that recurs
    /// (model thresholds, `EXP_LIMIT`, ...) skips the rational conversion.
    f64_cache: HashMap<u64, ExprId>,
    arg_pool: Vec<ExprId>,
    arg_dedup: HashMap<Box<[ExprId]>, ArgList>,
    /// The interned constants `0` and `1` (created in `new`), so identity
    /// folding is an id compare and `zero()`/`one()` never hash.
    zero: ExprId,
    one: ExprId,
    symbol_names: Vec<String>,
    symbol_ids: HashMap<String, SymbolId>,
    /// Functions (see [`crate::func`]) and the interned `(function, output)`
    /// pairs the `Call` nodes name.
    funcs: Vec<Function>,
    outputs: Vec<(FuncId, u32)>,
    output_dedup: HashMap<(FuncId, u32), OutputId>,
    /// Reusable per-node memo for the graph traversals (differentiation,
    /// substitution); see [`Memo`].
    memo: Option<Box<Memo>>,
}

/// A per-node memo table over the arena, cleared in O(1) by bumping an epoch:
/// the traversal primitives (`differentiate`, `substitute*`) key their memo by
/// `ExprId`, and on a hash-consed graph a build pass runs thousands of them
/// (one per stamp and port, one per instance root), each visiting a few
/// thousand nodes -- a hash map per call measured as the dominant cost. This
/// is two dense arrays and an epoch instead: a lookup is an index compare.
/// Keys are always nodes of the graph being traversed (ids below the arena
/// length at `begin`), so nodes created during the traversal never alias a
/// key.
#[derive(Default)]
pub struct Memo {
    epoch: Vec<u32>,
    val: Vec<ExprId>,
    cur: u32,
}

impl Memo {
    /// Start a fresh traversal over an arena of `n` nodes.
    pub fn begin(&mut self, n: usize) {
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            // epoch wrapped: invalidate everything explicitly
            self.epoch.iter_mut().for_each(|e| *e = 0);
            self.cur = 1;
        }
        if self.epoch.len() < n {
            self.epoch.resize(n, 0);
            self.val.resize(n, ExprId(0));
        }
    }
    #[inline]
    pub fn get(&self, e: ExprId) -> Option<ExprId> {
        let k = e.0 as usize;
        if k < self.epoch.len() && self.epoch[k] == self.cur {
            Some(self.val[k])
        } else {
            None
        }
    }
    #[inline]
    pub fn set(&mut self, e: ExprId, v: ExprId) {
        let k = e.0 as usize;
        if k < self.epoch.len() {
            self.epoch[k] = self.cur;
            self.val[k] = v;
        }
    }
}

mod calls;

impl<K: Field> Default for Graph<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Field> Graph<K> {
    pub fn new() -> Self {
        let mut ctx = Graph {
            nodes: Vec::new(),
            dedup: HashMap::default(),
            consts: Vec::new(),
            const_dedup: HashMap::default(),
            f64_cache: HashMap::default(),
            arg_pool: Vec::new(),
            arg_dedup: HashMap::default(),
            zero: ExprId(0),
            one: ExprId(0),
            symbol_names: Vec::new(),
            symbol_ids: HashMap::default(),
            funcs: Vec::new(),
            outputs: Vec::new(),
            output_dedup: HashMap::default(),
            memo: None,
        };
        ctx.zero = ctx.konst(K::zero());
        ctx.one = ctx.konst(K::one());
        ctx
    }

    /// Take the reusable traversal memo (a fresh one if it is in use by an
    /// enclosing traversal), already begun over the current arena.
    pub fn take_memo(&mut self) -> Box<Memo> {
        let mut m = self.memo.take().unwrap_or_default();
        m.begin(self.nodes.len());
        m
    }

    /// Return a memo taken with [`take_memo`](Self::take_memo).
    pub fn put_memo(&mut self, m: Box<Memo>) {
        self.memo = Some(m);
    }

    /// Number of distinct nodes currently interned (useful for sharing checks).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Borrow the node behind an id.
    #[inline]
    pub fn node(&self, id: ExprId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Name of a symbol.
    pub fn symbol_name(&self, s: SymbolId) -> &str {
        &self.symbol_names[s.0 as usize]
    }

    /// The exact rational behind a constant id.
    #[inline]
    pub fn const_val(&self, c: ConstId) -> &K {
        &self.consts[c.0 as usize]
    }

    /// The operand slice of an interned argument list.
    #[inline]
    pub fn args(&self, l: ArgList) -> &[ExprId] {
        &self.arg_pool[l.start as usize..(l.start + l.len) as usize]
    }

    /// The two equal-length halves `(a, b)` of a `Dot` node's operand list.
    #[inline]
    pub fn dot_args(&self, l: ArgList) -> (&[ExprId], &[ExprId]) {
        self.args(l).split_at(l.len() / 2)
    }

    /// The operands a node reads, without allocation (leaves have none).
    #[inline]
    pub fn operands(&self, id: ExprId) -> Operands<'_> {
        let z = ExprId(0);
        match *self.node(id) {
            Node::Const(_) | Node::Symbol(_) => Operands::Inline { buf: [z; 3], n: 0 },
            Node::Add(a, b) | Node::Mul(a, b) | Node::Cmp(_, a, b) | Node::Binary(_, a, b) => {
                Operands::Inline {
                    buf: [a, b, z],
                    n: 2,
                }
            }
            Node::Neg(a) | Node::Pow(a, _) | Node::Unary(_, a) => Operands::Inline {
                buf: [a, z, z],
                n: 1,
            },
            Node::Select(c, t, e) => Operands::Inline {
                buf: [c, t, e],
                n: 3,
            },
            Node::Reduce(_, l) | Node::Dot(l) | Node::Call(_, l) | Node::Solve(l, _) => {
                Operands::Slice(self.args(l))
            }
        }
    }

    /// Intern a node, reusing an existing id if structurally identical.
    #[inline]
    fn intern(&mut self, node: Node) -> ExprId {
        if let Some(&id) = self.dedup.get(&node) {
            return id;
        }
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.dedup.insert(node, id);
        id
    }

    /// Intern an operand list by content.
    fn intern_args(&mut self, args: &[ExprId]) -> ArgList {
        if let Some(&l) = self.arg_dedup.get(args) {
            return l;
        }
        let l = ArgList {
            start: self.arg_pool.len() as u32,
            len: args.len() as u32,
        };
        self.arg_pool.extend_from_slice(args);
        self.arg_dedup.insert(args.into(), l);
        l
    }

    /// Borrow the rational value if `id` is a constant.
    #[inline]
    pub fn const_of(&self, id: ExprId) -> Option<&K> {
        match *self.node(id) {
            Node::Const(c) => Some(self.const_val(c)),
            _ => None,
        }
    }

    /// Value of `id` as an `f64` when it is a constant node, else `None`.
    pub fn const_f64(&self, id: ExprId) -> Option<f64> {
        self.const_of(id).map(|r| r.to_f64())
    }

    /// True if `id` is the constant zero.
    #[inline]
    pub fn is_zero(&self, id: ExprId) -> bool {
        id == self.zero
    }

    /// True if `id` is the constant one.
    #[inline]
    pub fn is_one(&self, id: ExprId) -> bool {
        id == self.one
    }

    // --- leaf constructors -------------------------------------------------

    pub fn konst(&mut self, r: K) -> ExprId {
        let c = match self.const_dedup.get(&r) {
            Some(&c) => c,
            None => {
                let c = ConstId(self.consts.len() as u32);
                self.consts.push(r.clone());
                self.const_dedup.insert(r, c);
                c
            }
        };
        self.intern(Node::Const(c))
    }

    pub fn konst_int(&mut self, n: i64) -> ExprId {
        match n {
            0 => self.zero,
            1 => self.one,
            _ => self.konst(K::from_i64(n)),
        }
    }

    pub fn ratio(&mut self, num: i64, den: i64) -> ExprId {
        self.konst(K::from_ratio(num, den))
    }

    /// Exact rational constant from an `f64` (e.g. a model parameter threshold).
    /// Non-finite values fall back to zero.
    pub fn konst_f64(&mut self, x: f64) -> ExprId {
        // Keyed on the bit pattern, so `0.0` and `-0.0` are distinct keys but
        // both intern to the rational zero (as `from_float` yields for either).
        if let Some(&id) = self.f64_cache.get(&x.to_bits()) {
            return id;
        }
        let id = match K::from_f64(x) {
            Some(r) => self.konst(r),
            None => self.zero,
        };
        self.f64_cache.insert(x.to_bits(), id);
        id
    }

    #[inline]
    pub fn zero(&mut self) -> ExprId {
        self.zero
    }

    #[inline]
    pub fn one(&mut self) -> ExprId {
        self.one
    }

    /// The expression node for an existing symbol id.
    pub fn symbol_expr(&mut self, s: SymbolId) -> ExprId {
        self.intern(Node::Symbol(s))
    }

    /// Look up or create a free symbol by name.
    pub fn sym(&mut self, name: &str) -> ExprId {
        let sid = if let Some(&sid) = self.symbol_ids.get(name) {
            sid
        } else {
            let sid = SymbolId(self.symbol_names.len() as u32);
            self.symbol_names.push(name.to_string());
            self.symbol_ids.insert(name.to_string(), sid);
            sid
        };
        self.intern(Node::Symbol(sid))
    }

    // --- algebraic constructors -------------------------------------------

    pub fn add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        if a == self.zero {
            return b;
        }
        if b == self.zero {
            return a;
        }
        if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
            let v = x.add(y);
            return self.konst(v);
        }
        let (a, b) = order(a, b);
        self.intern(Node::Add(a, b))
    }

    pub fn sub(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let nb = self.neg(b);
        self.add(a, nb)
    }

    pub fn mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        if a == self.zero || b == self.zero {
            return self.zero;
        }
        if a == self.one {
            return b;
        }
        if b == self.one {
            return a;
        }
        if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
            let v = x.mul(y);
            return self.konst(v);
        }
        let (a, b) = order(a, b);
        self.intern(Node::Mul(a, b))
    }

    pub fn neg(&mut self, a: ExprId) -> ExprId {
        if a == self.zero {
            return a;
        }
        if let Some(x) = self.const_of(a) {
            let v = x.neg();
            return self.konst(v);
        }
        if let Node::Neg(inner) = *self.node(a) {
            return inner;
        }
        self.intern(Node::Neg(a))
    }

    /// Integer power. Folds constants and collapses nested powers.
    pub fn pow_i(&mut self, a: ExprId, n: i64) -> ExprId {
        if n == 0 {
            return self.one;
        }
        if n == 1 {
            return a;
        }
        if let Some(x) = self.const_of(a) {
            // `0^(negative)` has no exact rational value (it is `inf` numerically).
            // Don't fold it -- keep the `Pow` node, so the tape evaluates it as
            // `0.powi(n)` = `inf` (matching the numeric path) instead of panicking
            // on a division by zero. This makes the constructor total, which the
            // parameter-fold transform relies on (folding a zero-valued parameter
            // that sits in a denominator must not crash).
            if let Some(v) = x.powi(n) {
                return self.konst(v);
            }
        }
        if let Node::Pow(base, m) = *self.node(a) {
            return self.pow_i(base, m * n);
        }
        self.intern(Node::Pow(a, n))
    }

    pub fn recip(&mut self, a: ExprId) -> ExprId {
        self.pow_i(a, -1)
    }

    pub fn div(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let rb = self.recip(b);
        self.mul(a, rb)
    }

    // --- elementary functions ---------------------------------------------

    /// Apply a unary function, with a few exact-at-special-points identities.
    pub fn unary(&mut self, op: UnaryOp, a: ExprId) -> ExprId {
        // A floating field folds through the reference math; an exact field
        // keeps transcendental constants symbolic.
        if !K::is_exact() {
            if let Some(x) = self.const_of(a) {
                let y = unary_f64(op, x.to_f64());
                // Outside the domain the op stays (a NaN payload is the
                // backend's business, not a constant's).
                if !y.is_nan() {
                    if let Some(v) = K::from_f64(y) {
                        return self.konst(v);
                    }
                }
            }
        }
        // `abs` of a magnitude or an `abs` is itself; of a negation, of the
        // operand (exact in IEEE: abs only clears the sign bit).
        if op == UnaryOp::Abs {
            match *self.node(a) {
                Node::Unary(UnaryOp::Abs | UnaryOp::Sqrt, _) => return a,
                Node::Neg(inner) => return self.unary(UnaryOp::Abs, inner),
                _ => {}
            }
        }
        match op {
            UnaryOp::Exp if self.is_zero(a) => self.one, // exp(0) = 1
            UnaryOp::Ln if self.is_one(a) => self.zero,  // ln(1) = 0
            UnaryOp::Sin if self.is_zero(a) => self.zero,
            UnaryOp::Cos if self.is_zero(a) => self.one,
            UnaryOp::Sinh if self.is_zero(a) => self.zero,
            UnaryOp::Cosh if self.is_zero(a) => self.one,
            UnaryOp::Tanh if self.is_zero(a) => self.zero,
            UnaryOp::Sqrt if self.is_zero(a) => self.zero,
            UnaryOp::Sqrt if self.is_one(a) => self.one,
            _ => self.intern(Node::Unary(op, a)),
        }
    }

    /// Binary function node (`Powf`, `Mod`, `Atan2`, `Hypot`): the powers of
    /// zero and one resolve; a floating field folds constants through the
    /// reference math.
    pub fn binary(&mut self, op: BinOp, a: ExprId, b: ExprId) -> ExprId {
        if op == BinOp::Powf {
            if self.is_zero(b) {
                return self.one;
            }
            if self.is_one(b) {
                return a;
            }
            if let Some(n) = self.const_of(b).and_then(small_integer) {
                return self.pow_i(a, n);
            }
        }
        if !K::is_exact() {
            if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
                let z = binary_f64(op, x.to_f64(), y.to_f64());
                if !z.is_nan() {
                    if let Some(v) = K::from_f64(z) {
                        return self.konst(v);
                    }
                }
            }
        }
        self.intern(Node::Binary(op, a, b))
    }

    pub fn exp(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Exp, a)
    }
    pub fn ln(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Ln, a)
    }
    pub fn sqrt(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Sqrt, a)
    }
    pub fn sin(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Sin, a)
    }
    pub fn cos(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Cos, a)
    }
    pub fn floor(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Floor, a)
    }
    pub fn sinh(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Sinh, a)
    }
    pub fn cosh(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Cosh, a)
    }
    pub fn tanh(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Tanh, a)
    }
    pub fn atan(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Atan, a)
    }

    // --- conditions and selection ----------------------------------------

    /// Comparison node (`1.0`/`0.0`); folds when both operands are constant.
    pub fn cmp(&mut self, op: CmpOp, a: ExprId, b: ExprId) -> ExprId {
        if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
            return if cmp_field(op, x, y) {
                self.one
            } else {
                self.zero
            };
        }
        self.intern(Node::Cmp(op, a, b))
    }

    /// `cond != 0 ? then : else_`. Folds a constant condition and collapses
    /// equal branches.
    pub fn select(&mut self, cond: ExprId, then: ExprId, else_: ExprId) -> ExprId {
        if let Some(c) = self.const_of(cond) {
            return if c.is_zero() { else_ } else { then };
        }
        if then == else_ {
            return then;
        }
        self.intern(Node::Select(cond, then, else_))
    }

    // --- fused / variadic operators --------------------------------------

    /// Associative reduction over `args` (one fused node instead of an
    /// Add/Mul-tree). Commutative operands are sorted to maximise sharing; an
    /// empty/singleton list collapses. Constants are kept (exact) and combined
    /// at evaluation time.
    pub fn reduce(&mut self, op: ReduceOp, args: Vec<ExprId>) -> ExprId {
        // Combine the constant operands exactly (keeping sparsity exact and the
        // tape short); a zero factor zeroes a product, matching `mul`.
        let rest = match op {
            ReduceOp::Sum => {
                let mut acc: Option<K> = None;
                let mut rest = Vec::with_capacity(args.len());
                for a in args {
                    if a == self.zero {
                        continue;
                    }
                    match self.const_of(a) {
                        Some(r) => {
                            acc = Some(match acc {
                                Some(s) => s.add(r),
                                None => r.clone(),
                            })
                        }
                        None => rest.push(a),
                    }
                }
                if let Some(acc) = acc {
                    if !acc.is_zero() {
                        let k = self.konst(acc);
                        rest.push(k);
                    }
                }
                rest
            }
            ReduceOp::Product => {
                let mut acc: Option<K> = None;
                let mut rest = Vec::with_capacity(args.len());
                for a in args {
                    if a == self.zero {
                        return self.zero;
                    }
                    if a == self.one {
                        continue;
                    }
                    match self.const_of(a) {
                        Some(r) => {
                            acc = Some(match acc {
                                Some(s) => s.mul(r),
                                None => r.clone(),
                            })
                        }
                        None => rest.push(a),
                    }
                }
                if let Some(acc) = acc {
                    if acc.is_zero() {
                        return self.zero;
                    }
                    if !acc.is_one() {
                        let k = self.konst(acc);
                        rest.push(k);
                    }
                }
                rest
            }
            ReduceOp::Min | ReduceOp::Max => args,
        };
        self.finish_reduce(op, rest)
    }

    fn finish_reduce(&mut self, op: ReduceOp, mut args: Vec<ExprId>) -> ExprId {
        match args.len() {
            0 => match op {
                ReduceOp::Sum => self.zero,
                ReduceOp::Product => self.one,
                // No finite rational identity; empty min/max is not used.
                ReduceOp::Min | ReduceOp::Max => self.zero,
            },
            1 => args[0],
            _ => {
                args.sort_unstable();
                let l = self.intern_args(&args);
                self.intern(Node::Reduce(op, l))
            }
        }
    }

    /// Inner product `Σ_i a[i]*b[i]`. The two lists must be equal length; an
    /// empty product is zero and a single pair is a plain multiply.
    pub fn dot(&mut self, a: Vec<ExprId>, b: Vec<ExprId>) -> ExprId {
        assert_eq!(a.len(), b.len(), "dot: operand lists differ in length");
        match a.len() {
            0 => self.zero,
            1 => self.mul(a[0], b[0]),
            _ => {
                let mut all = a;
                all.extend(b);
                let l = self.intern_args(&all);
                self.intern(Node::Dot(l))
            }
        }
    }

    /// The solution `x` of the dense system `A x = b`, one expression per
    /// component: `a` is `n * n` row-major, `b` of `n`. One unknown folds
    /// to a division; otherwise the components are one kernel (see
    /// [`Node::Solve`]).
    pub fn solve_dense(&mut self, a: Vec<ExprId>, b: Vec<ExprId>) -> Vec<ExprId> {
        let n = b.len();
        assert_eq!(
            a.len(),
            n * n,
            "solve: a square matrix over the right-hand side"
        );
        match n {
            0 => Vec::new(),
            1 => vec![self.div(b[0], a[0])],
            _ => {
                let mut all = a;
                all.extend(b);
                let l = self.intern_args(&all);
                (0..n as u32)
                    .map(|i| self.intern(Node::Solve(l, i)))
                    .collect()
            }
        }
    }

    /// The matrix and the right-hand side of a [`Node::Solve`] list, and `n`.
    pub fn solve_args(&self, l: ArgList) -> (usize, &[ExprId], &[ExprId]) {
        let n = Self::solve_n(l.len());
        let all = self.args(l);
        (n, &all[..n * n], &all[n * n..])
    }

    /// `n` from the length `n * n + n` of a solve list.
    pub fn solve_n(len: usize) -> usize {
        let n = ((len as f64).sqrt()) as usize;
        debug_assert_eq!(n * n + n, len, "a solve list is n*n + n long");
        n
    }

    /// Component `i` of a solve over the list `all`, rebuilt.
    pub(crate) fn solve_component(&mut self, all: Vec<ExprId>, i: u32) -> ExprId {
        let n = Self::solve_n(all.len());
        let (a, b) = all.split_at(n * n);
        self.solve_dense(a.to_vec(), b.to_vec())[i as usize]
    }

    // --- module interchange (see `crate::module`) ------------------------

    pub(crate) fn nodes_slice(&self) -> &[Node] {
        &self.nodes
    }
    pub(crate) fn consts_slice(&self) -> &[K] {
        &self.consts
    }
    pub(crate) fn arg_pool_slice(&self) -> &[ExprId] {
        &self.arg_pool
    }
    pub(crate) fn funcs_slice(&self) -> &[Function] {
        &self.funcs
    }
    pub(crate) fn call_outputs_slice(&self) -> &[(FuncId, u32)] {
        &self.outputs
    }

    /// Build a node of another graph (or of this one, before a transform)
    /// here, through the smart constructors: folding and canonical operand
    /// order run again, so identities that only became visible after a
    /// transform collapse. `konst`, `sym`, `operand`, `args_of` and
    /// `call_of` map the node's parts into this graph.
    pub(crate) fn rebuild_node(
        &mut self,
        node: &Node,
        konst: impl Fn(&Node) -> K,
        sym: impl Fn(SymbolId) -> SymbolId,
        operand: impl Fn(ExprId) -> ExprId,
        args_of: impl Fn(ArgList) -> Vec<ExprId>,
        call_of: impl Fn(OutputId) -> (FuncId, u32),
    ) -> ExprId {
        let e = &operand;
        match *node {
            Node::Const(_) => self.konst(konst(node)),
            Node::Symbol(s) => self.symbol_expr(sym(s)),
            Node::Call(o, l) => {
                let args = args_of(l);
                let (f, k) = call_of(o);
                self.call(f, k, &args)
            }
            Node::Add(a, b) => self.add(e(a), e(b)),
            Node::Mul(a, b) => self.mul(e(a), e(b)),
            Node::Neg(a) => self.neg(e(a)),
            Node::Pow(a, n) => self.pow_i(e(a), n),
            Node::Unary(op, a) => self.unary(op, e(a)),
            Node::Binary(op, a, b) => self.binary(op, e(a), e(b)),
            Node::Cmp(op, a, b) => self.cmp(op, e(a), e(b)),
            Node::Select(c, t, f) => self.select(e(c), e(t), e(f)),
            Node::Reduce(op, l) => {
                let args = args_of(l);
                self.reduce(op, args)
            }
            Node::Dot(l) => {
                let all = args_of(l);
                let (a, b) = all.split_at(all.len() / 2);
                self.dot(a.to_vec(), b.to_vec())
            }
            Node::Solve(l, i) => {
                let all = args_of(l);
                self.solve_component(all, i)
            }
        }
    }

    /// Intern a node of a stored module as it stands, its parts mapped into
    /// this graph: loading reproduces a graph exactly, it does not rebuild
    /// it (see [`rebuild_node`](Self::rebuild_node) for that).
    pub(crate) fn intern_node(
        &mut self,
        node: &Node,
        konst: impl Fn(&Node) -> K,
        sym: impl Fn(SymbolId) -> SymbolId,
        operand: impl Fn(ExprId) -> ExprId,
        args_of: impl Fn(ArgList) -> Vec<ExprId>,
        call_of: impl Fn(OutputId) -> (FuncId, u32),
    ) -> ExprId {
        let e = &operand;
        let n = match *node {
            Node::Const(_) => return self.konst(konst(node)),
            Node::Symbol(s) => Node::Symbol(sym(s)),
            Node::Call(o, l) => {
                let args = args_of(l);
                let (f, k) = call_of(o);
                return self.call(f, k, &args);
            }
            Node::Add(a, b) => Node::Add(e(a), e(b)),
            Node::Mul(a, b) => Node::Mul(e(a), e(b)),
            Node::Neg(a) => Node::Neg(e(a)),
            Node::Pow(a, k) => Node::Pow(e(a), k),
            Node::Unary(op, a) => Node::Unary(op, e(a)),
            Node::Binary(op, a, b) => Node::Binary(op, e(a), e(b)),
            Node::Cmp(op, a, b) => Node::Cmp(op, e(a), e(b)),
            Node::Select(c, t, f) => Node::Select(e(c), e(t), e(f)),
            Node::Reduce(op, l) => {
                let args = args_of(l);
                Node::Reduce(op, self.intern_args(&args))
            }
            Node::Dot(l) => {
                let args = args_of(l);
                Node::Dot(self.intern_args(&args))
            }
            Node::Solve(l, i) => {
                let args = args_of(l);
                Node::Solve(self.intern_args(&args), i)
            }
        };
        self.intern(n)
    }
}

/// Canonical operand order for commutative ops, to maximise hash-consing.
fn order(a: ExprId, b: ExprId) -> (ExprId, ExprId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// A comparison of two constants of the field; an unordered pair (NaN in a
/// floating field) compares false except for `Ne`.
fn cmp_field<K: Field>(op: CmpOp, x: &K, y: &K) -> bool {
    use std::cmp::Ordering::*;
    match (op, x.partial_cmp(y)) {
        (_, None) => op == CmpOp::Ne,
        (CmpOp::Gt, Some(o)) => o == Greater,
        (CmpOp::Ge, Some(o)) => o != Less,
        (CmpOp::Lt, Some(o)) => o == Less,
        (CmpOp::Le, Some(o)) => o != Greater,
        (CmpOp::Eq, Some(o)) => o == Equal,
        (CmpOp::Ne, Some(o)) => o != Equal,
    }
}

/// A constant that is a small integer (the exponent of a `Powf` that is
/// really an integer power).
fn small_integer<K: Field>(k: &K) -> Option<i64> {
    let x = k.to_f64();
    if x.fract() == 0.0 && x.abs() <= 64.0 && K::from_i64(x as i64) == *k {
        Some(x as i64)
    } else {
        None
    }
}
