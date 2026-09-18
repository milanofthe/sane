//! The tape compiler: the reachable forest lowered to instructions, then
//! scheduled, then given slots.
//!
//! Four passes, each reading only what the ones before it produced:
//!
//! 1. [`Forest::analyze`]: reachability, purity for the prolog split, use
//!    counts and the dispatch fusions (`MulAdd`, `Sub`).
//! 2. [`Forest::lower`]: one [`Inst`] per node, or one per *kernel*: the
//!    calls of a function with distinct argument lists at one call depth
//!    are one batched call, the row dots of one vector one matrix-vector
//!    product, the components of one dense system one solve. A kernel is
//!    an instruction with several outputs; the nodes it stands for are its
//!    output values. Leaves are operands, never instructions: an input is
//!    read where it is used, a constant is an instruction of its own.
//! 3. [`Forest::schedule`]: register-pressure list scheduling over the
//!    instructions, the prolog as a strict first phase.
//! 4. [`Forest::emit`]: lifetimes, slots (a kernel's outputs a block of
//!    consecutive slots), and the instruction stream.

use std::collections::BTreeSet;
use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::{Fold, Op, Src, Tape, INPUT};
use crate::extern_fn::ExternBundle;
use crate::field::Field;
use crate::func::{Body, FuncId};
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};

/// Row dots against one vector fuse into a `Gemv` from this many rows on.
const GEMV_MIN_ROWS: usize = 8;

/// Append operands to the pool; their range.
fn pooled(pool: &mut Vec<Ref>, ins: Vec<Ref>) -> (u32, u32) {
    let start = pool.len() as u32;
    pool.extend(ins);
    (start, pool.len() as u32 - start)
}

impl Tape {
    /// Compile a tape computing `roots`, where `inputs[k]` (passed to
    /// [`eval`](Self::eval)) is the value of symbol `input_syms[k]`. Symbols not
    /// listed evaluate to `NaN`.
    pub fn compile<K: Field>(ctx: &Graph<K>, roots: &[ExprId], input_syms: &[SymbolId]) -> Tape {
        Self::compile_inner(ctx, roots, input_syms, None)
    }

    /// [`compile`](Self::compile) with a prolog split: `pure_inputs[k]` marks
    /// input `k` as solve-constant (a parameter), and every op depending only
    /// on such inputs is scheduled into a *prolog prefix* of the instruction
    /// stream. A Newton loop evaluates the prolog once per parameter binding
    /// ([`eval_prolog`](Self::eval_prolog)) and then only the remainder per
    /// iteration ([`eval_main`](Self::eval_main)); values crossing the boundary
    /// are pinned so iteration-loop slot reuse cannot clobber them. Plain
    /// [`eval`](Self::eval) still runs the whole stream, so the split is
    /// invisible to callers that ignore it.
    pub fn compile_split<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: &[bool],
    ) -> Tape {
        Self::compile_inner(ctx, roots, input_syms, Some(pure_inputs))
    }

    fn compile_inner<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> Tape {
        use crate::hooks::timed;
        let forest = timed("tape analyze", || {
            Forest::analyze(ctx, roots, input_syms, pure_inputs)
        });
        let mut program = timed("tape lower", || forest.lower(ctx, roots));
        timed("tape fuse", || program.fuse_accumulators(pure_inputs));
        let order = timed("tape schedule", || program.schedule());
        timed("tape emit", || program.emit(&order, pure_inputs.is_some()))
    }
}

/// The reachable forest of one compilation, in dependency order, with the
/// tables the later passes index by *base position* (the index into
/// [`Forest::base`], not the arena id).
struct Forest {
    base: Vec<ExprId>,
    bpos: Vec<u32>,
    input_of: HashMap<SymbolId, u32>,
    /// Parameter-purity for the prolog split; all false without one.
    pure: Vec<bool>,
    /// `fused_into[p]` is the base position of the `Add` that absorbed the
    /// node at `p` as a superinstruction operand.
    fused_into: Vec<Option<usize>>,
}

/// One instruction of the lowered program, before scheduling.
///
/// `ins` are the operands in the order the op reads them (slots as value
/// references, or tagged inputs, see [`INPUT`]); `n_out` is the number of
/// values it produces, one for anything but a kernel. Value `(inst, k)` is
/// the `k`th output of instruction `inst`.
pub struct Inst {
    pub kind: Kind,
    /// The operands: `pool[start .. start + len]` of the program's pool,
    /// one flat vector for every instruction, so a million instructions
    /// are one allocation and not a million.
    pub ins: (u32, u32),
    pub n_out: u32,
    /// Parameter-pure: schedulable into the prolog.
    pub pure: bool,
}

/// An operand of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ref {
    /// Output `k` of instruction `inst`.
    Value(u32, u32),
    /// Input `k`, read in place.
    Input(u32),
}

/// What an instruction computes; the operand lists live in `Inst::ins`.
#[derive(Clone, Debug)]
pub enum Kind {
    Const(f64),
    Add,
    Mul,
    MulAdd,
    Sub,
    Neg,
    Powi(i32),
    Unary(crate::node::UnaryOp),
    Binary(crate::node::BinOp),
    Cmp(crate::node::CmpOp),
    Select,
    Reduce(crate::node::ReduceOp),
    /// `n` pairs.
    Dot(u32),
    Call {
        bundle: u32,
    },
    CallBatch {
        bundle: u32,
        n_groups: u32,
        n_args: u32,
    },
    /// With `acc`, the fold code per output and an accumulator operand per
    /// output after the factors.
    Gemv {
        m: u32,
        n: u32,
        acc: Option<Vec<u32>>,
    },
    Gemm {
        m: u32,
        k: u32,
        n: u32,
        acc: Option<Vec<u32>>,
    },
    Solve {
        n: u32,
    },
    SolveMany {
        n: u32,
        k: u32,
    },
}

/// The lowered program: instructions in a dependency order, the bundle
/// table, and which value each root is.
struct Program {
    insts: Vec<Inst>,
    /// The operand pool of every instruction (see [`Inst::ins`]).
    pool: Vec<Ref>,
    bundles: Vec<Arc<dyn ExternBundle>>,
    roots: Vec<Ref>,
}

impl Program {
    fn ins(&self, i: usize) -> &[Ref] {
        let (s, l) = self.insts[i].ins;
        &self.pool[s as usize..(s + l) as usize]
    }

    /// Pass 3: the accumulator fusion of the product kernels. An output of
    /// a `Gemv` or `Gemm` whose one consumer folds it, `Sub` with the output
    /// as the subtrahend, `Add` with the output as one operand, or `Neg`,
    /// is computed folded by the kernel and the consumer vanishes; the
    /// accumulator is the consumer's other operand, an earlier value, or
    /// another output of the same kernel (whose product is then read
    /// before its own fold). Outputs with other consumers stay plain. An
    /// accumulator may be defined after the kernel in this list (the
    /// scheduler orders by dependencies) as long as it does not depend on
    /// it, and a pure kernel takes only pure accumulators (the prolog
    /// keeps it).
    fn fuse_accumulators(&mut self, pure_inputs: Option<&[bool]>) {
        let m = self.insts.len();
        let kernels: Vec<usize> = (0..m)
            .filter(|&i| {
                matches!(
                    self.insts[i].kind,
                    Kind::Gemv { acc: None, .. } | Kind::Gemm { acc: None, .. }
                )
            })
            .collect();
        if kernels.is_empty() {
            return;
        }
        // Uses and the one consumer of every kernel output.
        let mut uses: HashMap<(u32, u32), (u32, u32)> = HashMap::default();
        for j in 0..m {
            for r in self.ins(j) {
                if let Ref::Value(i, c) = *r {
                    if matches!(
                        self.insts[i as usize].kind,
                        Kind::Gemv { .. } | Kind::Gemm { .. }
                    ) {
                        let e = uses.entry((i, c)).or_insert((0, j as u32));
                        e.0 += 1;
                        e.1 = j as u32;
                    }
                }
            }
        }
        for r in &self.roots {
            if let Ref::Value(i, c) = *r {
                uses.entry((i, c)).or_insert((0, u32::MAX)).0 += 2;
            }
        }
        let input_pure =
            |k: u32| pure_inputs.is_some_and(|p| p.get(k as usize).copied().unwrap_or(true));
        let mut alias: HashMap<u32, Ref> = HashMap::default();
        let mut dead = vec![false; m];
        // The first kernel that took an accumulator defined after it: an
        // instruction before it reads only instructions before itself, so
        // it reaches no later one, and the dependency walk stops there.
        let mut first_forward = usize::MAX;
        // The placeholder accumulator of a plain, self or negating fold: a
        // NaN constant, never read.
        let mut placeholder_inst: Option<Ref> = None;
        for &k in &kernels {
            let n_out = self.insts[k].n_out;
            let kernel_pure = self.insts[k].pure;
            let mut codes: Vec<u32> = vec![Fold::PLAIN.0; n_out as usize];
            let placeholder = *placeholder_inst.get_or_insert_with(|| {
                self.insts.push(Inst {
                    kind: Kind::Const(f64::NAN),
                    ins: (self.pool.len() as u32, 0),
                    n_out: 1,
                    pure: true,
                });
                dead.push(false);
                Ref::Value(self.insts.len() as u32 - 1, 0)
            });
            let mut accs: Vec<Ref> = vec![placeholder; n_out as usize];
            let mut fused: Vec<(u32, u32)> = Vec::new(); // (output, consumer)
                                                         // Outputs read as another output's accumulator stay plain.
            let mut held = vec![false; n_out as usize];
            for c in 0..n_out {
                let Some(&(count, j)) = uses.get(&(k as u32, c)) else {
                    continue;
                };
                if count != 1 || dead[j as usize] || held[c as usize] {
                    continue;
                }
                let this = Ref::Value(k as u32, c);
                let ins = self.ins(j as usize);
                let (code, acc) = match self.insts[j as usize].kind {
                    Kind::Sub if ins[1] == this => (Fold::SUB.0, ins[0]),
                    Kind::Neg => (Fold::NEG.0, placeholder),
                    Kind::Add => (Fold::ADD.0, if ins[0] == this { ins[1] } else { ins[0] }),
                    _ => continue,
                };
                // An accumulator that is a consumer folded away earlier is
                // that kernel's output.
                let mut acc = acc;
                while let Ref::Value(i, _) = acc {
                    match alias.get(&i) {
                        Some(&to) => acc = to,
                        None => break,
                    }
                }
                let (code, acc) = match acc {
                    Ref::Value(i, c2) if i as usize == k && code != Fold::NEG.0 => {
                        // Another output of this kernel: its product, if
                        // that is not folded itself.
                        if c2 == c || codes[c2 as usize] != Fold::PLAIN.0 {
                            continue;
                        }
                        held[c2 as usize] = true;
                        (code | 4 | (c2 << 3), placeholder)
                    }
                    Ref::Value(i, _) if code != Fold::NEG.0 => {
                        if kernel_pure && !self.insts[i as usize].pure {
                            continue;
                        }
                        // An earlier instruction reaches this kernel only
                        // through a fused kernel's forward accumulator.
                        if (i as usize > k || first_forward < i as usize)
                            && self.depends_on(i as usize, k, first_forward)
                        {
                            continue;
                        }
                        (code, acc)
                    }
                    Ref::Input(i) if code != Fold::NEG.0 => {
                        if kernel_pure && !input_pure(i) {
                            continue;
                        }
                        (code, acc)
                    }
                    _ => (code, placeholder),
                };
                codes[c as usize] = code;
                accs[c as usize] = acc;
                fused.push((c, j));
            }
            if fused.is_empty() {
                continue;
            }
            if accs
                .iter()
                .any(|r| matches!(*r, Ref::Value(i, _) if i as usize > k))
            {
                first_forward = first_forward.min(k);
            }
            let start = self.pool.len() as u32;
            let old: Vec<Ref> = self.ins(k).to_vec();
            let len = old.len() as u32 + n_out;
            self.pool.extend(old);
            self.pool.extend(accs);
            let inst = &mut self.insts[k];
            inst.ins = (start, len);
            match &mut inst.kind {
                Kind::Gemv { acc, .. } | Kind::Gemm { acc, .. } => *acc = Some(codes),
                _ => unreachable!(),
            }
            for (c, j) in fused {
                dead[j as usize] = true;
                alias.insert(j, Ref::Value(k as u32, c));
            }
        }
        if alias.is_empty() {
            return;
        }
        // Redirect the consumers' values, then drop them.
        let resolve = |r: Ref| -> Ref {
            let mut r = r;
            while let Ref::Value(i, _) = r {
                match alias.get(&i) {
                    Some(&to) => r = to,
                    None => break,
                }
            }
            r
        };
        let mut renumber = vec![u32::MAX; self.insts.len()];
        let mut next = 0u32;
        for (i, &d) in dead.iter().enumerate() {
            if !d {
                renumber[i] = next;
                next += 1;
            }
        }
        let map = |r: Ref| -> Ref {
            match resolve(r) {
                Ref::Value(i, c) => Ref::Value(renumber[i as usize], c),
                r => r,
            }
        };
        for r in self.pool.iter_mut() {
            *r = map(*r);
        }
        for r in self.roots.iter_mut() {
            *r = map(*r);
        }
        let old = std::mem::take(&mut self.insts);
        self.insts = old
            .into_iter()
            .zip(&dead)
            .filter(|(_, &d)| !d)
            .map(|(inst, _)| inst)
            .collect();
    }

    /// Whether instruction `i` reads instruction `k` (transitively). The
    /// list is in lowering order, so an instruction reaches a later one
    /// only through a fused kernel's forward accumulator: the walk stops
    /// at instructions before `k` that also precede the first such kernel.
    fn depends_on(&self, i: usize, k: usize, first_forward: usize) -> bool {
        let mut stack = vec![i as u32];
        let mut seen: HashSet<u32> = HashSet::default();
        while let Some(i) = stack.pop() {
            if i as usize == k {
                return true;
            }
            if ((i as usize) < k && (i as usize) < first_forward) || !seen.insert(i) {
                continue;
            }
            for r in self.ins(i as usize) {
                if let Ref::Value(j, _) = *r {
                    stack.push(j);
                }
            }
        }
        false
    }
}

impl Forest {
    #[inline]
    fn pos(&self, e: ExprId) -> usize {
        self.bpos[e.0 as usize] as usize
    }

    /// Pass 1: reachability, purity, use counts and superinstruction fusion.
    fn analyze<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> Forest {
        let mut input_of: HashMap<SymbolId, u32> =
            HashMap::with_capacity_and_hasher(input_syms.len(), Default::default());
        for (k, &s) in input_syms.iter().enumerate() {
            input_of.insert(s, k as u32);
        }
        let n_arena = ctx.len();
        let mut mark = vec![false; n_arena];
        let mut stack = roots.to_vec();
        while let Some(id) = stack.pop() {
            let k = id.0 as usize;
            if mark[k] {
                continue;
            }
            mark[k] = true;
            stack.extend_from_slice(&ctx.operands(id));
        }
        let base: Vec<ExprId> = (0..n_arena)
            .filter(|&k| mark[k])
            .map(|k| ExprId(k as u32))
            .collect();
        let m = base.len();
        let mut bpos = vec![u32::MAX; n_arena];
        for (i, id) in base.iter().enumerate() {
            bpos[id.0 as usize] = i as u32;
        }
        let bp = |e: ExprId| bpos[e.0 as usize] as usize;

        // Purity: a node is parameter-pure when every operand is; an
        // unmapped symbol is a NaN constant and so pure.
        let mut pure = vec![false; m];
        if let Some(mask) = pure_inputs {
            for (i, id) in base.iter().enumerate() {
                pure[i] = match ctx.node(*id) {
                    Node::Const(_) => true,
                    Node::Symbol(s) => match input_of.get(s) {
                        None => true,
                        Some(&k) => mask.get(k as usize).copied().unwrap_or(false),
                    },
                    _ => ctx.operands(*id).iter().all(|a| pure[bp(*a)]),
                };
            }
        }
        let mut is_root = vec![false; m];
        for r in roots {
            is_root[bp(*r)] = true;
        }
        let mut uses = vec![0u32; m];
        for id in &base {
            for a in ctx.operands(*id).iter() {
                uses[bp(*a)] += 1;
            }
        }
        // Dispatch fusion: an `Add` whose operand is a single-use `Mul` is
        // one `MulAdd`, a single-use `Neg` one `Sub`; the fused operand is
        // never materialized. The arithmetic is the same IEEE sequence, so
        // parity with the arena is bit-exact.
        let mut fused_into: Vec<Option<usize>> = vec![None; m];
        for (i, id) in base.iter().enumerate() {
            if let Node::Add(a, b) = ctx.node(*id) {
                let fusable = |x: &ExprId, fused: &[Option<usize>]| {
                    let px = bp(*x);
                    !is_root[px]
                        && uses[px] == 1
                        && fused[px].is_none()
                        && matches!(ctx.node(*x), Node::Mul(..) | Node::Neg(..))
                };
                if fusable(a, &fused_into) {
                    fused_into[bp(*a)] = Some(i);
                } else if fusable(b, &fused_into) {
                    fused_into[bp(*b)] = Some(i);
                }
            }
        }
        Forest {
            base,
            bpos,
            input_of,
            pure,
            fused_into,
        }
    }

    /// Pass 2: the instructions.
    fn lower<K: Field>(&self, ctx: &Graph<K>, roots: &[ExprId]) -> Program {
        let base = &self.base;
        let m = base.len();
        let mut insts: Vec<Inst> = Vec::with_capacity(m);
        let mut pool: Vec<Ref> = Vec::with_capacity(2 * m);
        let mut bundles: Vec<Arc<dyn ExternBundle>> = Vec::new();
        let mut bundle_idx: HashMap<usize, u32> = HashMap::default();
        // The outputs each call (function and argument list) is made for:
        // what a body must cover to serve it. Calls of one function made
        // for different output sets (a residual alone, the residual with
        // its partials) take different bodies, so they are keyed by the set.
        let mut per_call: HashMap<(u32, crate::node::ArgList), Vec<u32>> = HashMap::default();
        for &id in base {
            if let Node::Call(o, l) = *ctx.node(id) {
                let (f, out) = ctx.output(o);
                let v = per_call.entry((f.0, l)).or_default();
                if !v.contains(&out) {
                    v.push(out);
                }
            }
        }
        let mut set_ids: HashMap<(u32, Vec<u32>), u32> = HashMap::default();
        let mut sets: Vec<Vec<u32>> = Vec::new();
        let mut set_of: HashMap<(u32, crate::node::ArgList), u32> = HashMap::default();
        for ((f, l), mut outs) in per_call {
            outs.sort_unstable();
            let id = *set_ids.entry((f, outs.clone())).or_insert_with(|| {
                sets.push(outs);
                sets.len() as u32 - 1
            });
            set_of.insert((f, l), id);
        }
        let needed = |f: u32, l: crate::node::ArgList| -> (u32, &Vec<u32>) {
            let id = set_of[&(f, l)];
            (id, &sets[id as usize])
        };
        let mut bodies: HashMap<(u32, u32), Body> = HashMap::default();
        // The value each base node is, once lowered.
        let mut value: Vec<Option<Ref>> = vec![None; m];

        // --- the kernels' groups -----------------------------------------
        // Calls of one function with distinct argument lists; row dots by
        // their vector; solve components by their list. Members of one
        // group share a call depth: a member's depth is one more than the
        // deepest kernel member among its operands, so equal depth means
        // no member depends on another's output, and the group's operands
        // all precede it.
        let mut lists: HashMap<u32, Vec<Vec<ExprId>>> = HashMap::default();
        let mut seen: BTreeSet<(u32, Vec<ExprId>)> = BTreeSet::new();
        for &id in base {
            if let Node::Call(o, l) = *ctx.node(id) {
                let (f, _) = ctx.output(o);
                let args = ctx.args(l);
                if seen.insert((f.0, args.to_vec())) {
                    lists.entry(f.0).or_default().push(args.to_vec());
                }
            }
        }
        lists.retain(|_, g| g.len() >= 2 && g.iter().all(|a| a.len() == g[0].len()));
        let mut rows_of: HashMap<Vec<ExprId>, Vec<usize>> = HashMap::default();
        for (i, &id) in base.iter().enumerate() {
            if let Node::Dot(l) = *ctx.node(id) {
                let (_, x) = ctx.dot_args(l);
                if !x.is_empty() {
                    rows_of.entry(x.to_vec()).or_default().push(i);
                }
            }
        }
        rows_of.retain(|_, rows| rows.len() >= GEMV_MIN_ROWS);
        let mut is_member = vec![false; m];
        for rows in rows_of.values() {
            for &r in rows {
                is_member[r] = true;
            }
        }
        for (i, &id) in base.iter().enumerate() {
            is_member[i] |= match *ctx.node(id) {
                Node::Call(o, _) => lists.contains_key(&ctx.output(o).0 .0),
                Node::Solve(..) => true,
                _ => false,
            };
        }
        let mut depth = vec![0u32; m];
        for (i, &id) in base.iter().enumerate() {
            let over = ctx
                .operands(id)
                .iter()
                .map(|&a| depth[self.pos(a)])
                .max()
                .unwrap_or(0);
            depth[i] = over + u32::from(is_member[i]);
        }
        // Group keys, by first encounter: (kind, key, depth) -> members.
        #[derive(PartialEq, Eq, Hash, Clone)]
        // Every key carries the members' purity: a kernel of pure and impure
        // members would be impure as a whole and pull the pure work out of
        // the prolog.
        enum GroupKey {
            /// Function, depth, purity, the id of the output set called for.
            Call(u32, u32, bool, u32),
            Gemv(Vec<ExprId>, u32, bool),
            /// The rows (sorted) against every vector: a matrix product.
            Gemm(Vec<Vec<ExprId>>, Vec<Vec<ExprId>>),
            Solve(crate::node::ArgList, bool),
            /// One matrix against several right-hand sides (their lists, in
            /// first-encounter order).
            SolveMany(Vec<ExprId>, Vec<crate::node::ArgList>),
        }
        let mut group_index: HashMap<GroupKey, usize> = HashMap::default();
        let mut groups: Vec<(GroupKey, Vec<usize>)> = Vec::new();
        for (i, &id) in base.iter().enumerate() {
            if !is_member[i] {
                continue;
            }
            let key = match *ctx.node(id) {
                Node::Call(o, l) => {
                    let f = ctx.output(o).0 .0;
                    GroupKey::Call(f, depth[i], self.pure[i], needed(f, l).0)
                }
                Node::Dot(l) => GroupKey::Gemv(ctx.dot_args(l).1.to_vec(), depth[i], self.pure[i]),
                Node::Solve(l, _) => GroupKey::Solve(l, self.pure[i]),
                _ => unreachable!("members are calls, rows or components"),
            };
            let g = *group_index.entry(key.clone()).or_insert_with(|| {
                groups.push((key, Vec::new()));
                groups.len() - 1
            });
            groups[g].1.push(i);
        }
        // Gemv groups over the same rows at one depth are one Gemm: the
        // rows against every vector at once. A merged-away group is left
        // empty; no member points at it.
        let row_of = |mi: usize| -> Vec<ExprId> {
            let Node::Dot(l) = *ctx.node(base[mi]) else {
                unreachable!()
            };
            ctx.dot_args(l).0.to_vec()
        };
        // Row sets are compared by fingerprint (one hash per row, sorted),
        // and the candidate groups' rows verified before they merge, so a
        // group of long rows costs one pass over its entries.
        let row_hash = |mi: usize| -> u64 {
            use std::hash::{Hash, Hasher};
            let Node::Dot(l) = *ctx.node(base[mi]) else {
                unreachable!()
            };
            let mut h = rustc_hash::FxHasher::default();
            ctx.dot_args(l).0.hash(&mut h);
            h.finish()
        };
        let mut by_rows: HashMap<(Vec<u64>, u32, bool), Vec<usize>> = HashMap::default();
        for (g, (key, members)) in groups.iter().enumerate() {
            if let GroupKey::Gemv(_, d, pure) = key {
                if members.len() < GEMV_MIN_ROWS {
                    continue;
                }
                let mut hashes: Vec<u64> = members.iter().map(|&mi| row_hash(mi)).collect();
                hashes.sort_unstable();
                if hashes.windows(2).any(|w| w[0] == w[1]) {
                    continue;
                }
                by_rows.entry((hashes, *d, *pure)).or_default().push(g);
            }
        }
        let mut merged: Vec<Vec<usize>> =
            by_rows.into_values().filter(|gs| gs.len() >= 2).collect();
        merged.sort();
        for gs in merged {
            // The rows of the first group, sorted, and the check that every
            // other group has exactly them.
            let mut rows: Vec<Vec<ExprId>> = groups[gs[0]].1.iter().map(|&mi| row_of(mi)).collect();
            rows.sort();
            let same_rows = gs[1..].iter().all(|&g| {
                let mut r: Vec<Vec<ExprId>> = groups[g].1.iter().map(|&mi| row_of(mi)).collect();
                r.sort();
                r == rows
            });
            if !same_rows {
                continue;
            }
            let xs: Vec<Vec<ExprId>> = gs
                .iter()
                .map(|&g| match &groups[g].0 {
                    GroupKey::Gemv(x, ..) => x.clone(),
                    _ => unreachable!(),
                })
                .collect();
            let members: Vec<usize> = gs
                .iter()
                .flat_map(|&g| groups[g].1.iter().copied())
                .collect();
            for &g in &gs[1..] {
                groups[g].1.clear();
            }
            groups[gs[0]] = (GroupKey::Gemm(rows, xs), members);
        }
        // Solve groups over one matrix at one depth are one solve of several
        // right-hand sides: the factorization once. Equal depth means no
        // right-hand side depends on another group's solution (a
        // derivative's solve reads the primal solution over the same
        // matrix). A merged-away group is left empty.
        let mut by_matrix: HashMap<(Vec<ExprId>, u32, bool), Vec<usize>> = HashMap::default();
        for (g, (key, members)) in groups.iter().enumerate() {
            if let GroupKey::Solve(l, pure) = key {
                let all = ctx.args(*l);
                let n = Graph::<K>::solve_n(all.len());
                by_matrix
                    .entry((all[..n * n].to_vec(), depth[members[0]], *pure))
                    .or_default()
                    .push(g);
            }
        }
        let mut merged_solves: Vec<(Vec<usize>, Vec<ExprId>)> = by_matrix
            .into_iter()
            .filter(|(_, gs)| gs.len() >= 2)
            .map(|((a, _, _), gs)| (gs, a))
            .collect();
        merged_solves.sort();
        for (gs, a) in merged_solves {
            let lists: Vec<crate::node::ArgList> = gs
                .iter()
                .map(|&g| match &groups[g].0 {
                    GroupKey::Solve(l, _) => *l,
                    _ => unreachable!(),
                })
                .collect();
            let members: Vec<usize> = gs
                .iter()
                .flat_map(|&g| groups[g].1.iter().copied())
                .collect();
            for &g in &gs[1..] {
                groups[g].1.clear();
            }
            groups[gs[0]] = (GroupKey::SolveMany(a, lists), members);
        }
        // A gemv group of too few rows at its depth, or a call group of one
        // argument list, is no kernel: its members lower on their own.
        let mut kernel_of: Vec<Option<usize>> = vec![None; m];
        for (g, (key, members)) in groups.iter().enumerate() {
            let is_kernel = match key {
                GroupKey::Gemv(..) => members.len() >= GEMV_MIN_ROWS,
                GroupKey::Gemm(..) => true,
                GroupKey::Call(..) => {
                    let mut distinct: BTreeSet<Vec<ExprId>> = BTreeSet::new();
                    for &i in members {
                        if let Node::Call(_, l) = *ctx.node(base[i]) {
                            distinct.insert(ctx.args(l).to_vec());
                        }
                    }
                    distinct.len() >= 2
                }
                GroupKey::Solve(..) => true,
                GroupKey::SolveMany(..) => true,
            };
            if is_kernel {
                for &i in members {
                    kernel_of[i] = Some(g);
                }
            }
        }
        let mut lowered_group = vec![false; groups.len()];
        // The outputs of one call (same function, same arguments), so a
        // single call is one instruction whichever output is reached first.
        let mut call_sites: HashMap<(u32, crate::node::ArgList), Vec<usize>> = HashMap::default();
        for (i, &id) in base.iter().enumerate() {
            if let Node::Call(o, l) = *ctx.node(id) {
                call_sites
                    .entry((ctx.output(o).0 .0, l))
                    .or_default()
                    .push(i);
            }
        }

        // --- the order: every unit after its operands ----------------------
        // A unit is a node, or the group it belongs to (by its first
        // member); a group's operands are all its members' operands. A
        // fused operand is no unit, its consumer reads its operands. A
        // depth-first walk from the roots, post-order, with an explicit
        // stack.
        let first_member: Vec<usize> = groups
            .iter()
            .map(|(_, ms)| ms.first().copied().unwrap_or(0))
            .collect();
        let unit_of = |i: usize| -> usize {
            match kernel_of[i] {
                Some(g) => first_member[g],
                None => i,
            }
        };
        let deps_of_node = |i: usize, out: &mut Vec<usize>| {
            for &a in ctx.operands(base[i]).iter() {
                let pa = self.pos(a);
                if self.fused_into[pa] == Some(i) {
                    for &x in ctx.operands(a).iter() {
                        out.push(unit_of(self.pos(x)));
                    }
                } else {
                    out.push(unit_of(pa));
                }
            }
        };
        // Distinct dependencies by a stamp per unit, not a sort: a kernel
        // over thousands of rows names its operands hundreds of thousands
        // of times.
        let mut stamp: Vec<usize> = vec![usize::MAX; m];
        let mut deps_of_unit = |u: usize| -> Vec<usize> {
            let mut raw = Vec::new();
            match kernel_of[u] {
                Some(g) => {
                    for &mi in &groups[g].1 {
                        deps_of_node(mi, &mut raw);
                    }
                }
                None => deps_of_node(u, &mut raw),
            }
            let mut out = Vec::with_capacity(raw.len().min(64));
            for d in raw {
                if stamp[d] != u {
                    stamp[d] = u;
                    out.push(d);
                }
            }
            out
        };
        let mut done = vec![false; m];
        let mut order: Vec<usize> = Vec::with_capacity(m);
        let mut stack: Vec<(usize, bool)> = roots
            .iter()
            .map(|r| (unit_of(self.pos(*r)), false))
            .collect();
        while let Some((u, expanded)) = stack.pop() {
            if done[u] {
                continue;
            }
            if expanded {
                done[u] = true;
                order.push(u);
                continue;
            }
            stack.push((u, true));
            for d in deps_of_unit(u) {
                if !done[d] {
                    stack.push((d, false));
                }
            }
        }

        // --- the instructions, in that order --------------------------------
        let leaf = |e: ExprId| -> Option<Ref> {
            match ctx.node(e) {
                Node::Symbol(s) => Some(Ref::Input(
                    self.input_of.get(s).copied().unwrap_or(u32::MAX),
                )),
                _ => None,
            }
        };
        for &i in &order {
            let id = base[i];
            if self.fused_into[i].is_some() {
                continue; // materialised inside its consuming Add
            }
            if value[i].is_some() {
                continue; // an output of a kernel lowered earlier
            }
            let node = *ctx.node(id);
            // A leaf: an input is an operand (a symbol without an input is
            // NaN), a constant an instruction.
            if let Some(r) = leaf(id) {
                value[i] = Some(match r {
                    Ref::Input(u32::MAX) => {
                        insts.push(Inst {
                            kind: Kind::Const(f64::NAN),
                            ins: (pool.len() as u32, 0),
                            n_out: 1,
                            pure: true,
                        });
                        Ref::Value(insts.len() as u32 - 1, 0)
                    }
                    r => r,
                });
                continue;
            }
            if let Node::Const(c) = node {
                insts.push(Inst {
                    kind: Kind::Const(ctx.const_val(c).to_f64()),
                    ins: (pool.len() as u32, 0),
                    n_out: 1,
                    pure: true,
                });
                value[i] = Some(Ref::Value(insts.len() as u32 - 1, 0));
                continue;
            }
            // The operand of a node: its value (every operand precedes its
            // consumer in base order, and a fused operand's own operands
            // are read by the consumer).
            let val = |e: ExprId, value: &[Option<Ref>]| -> Ref {
                value[self.pos(e)].expect("an operand precedes its consumer")
            };
            if let Some(g) = kernel_of[i] {
                if lowered_group[g] {
                    continue;
                }
                lowered_group[g] = true;
                let (key, members) = &groups[g];
                match key {
                    GroupKey::Call(f, _, _, set) => {
                        let (bundle, n_out) = {
                            let body = bodies.entry((*f, *set)).or_insert_with(|| {
                                ctx.func(FuncId(*f)).body_for(ctx, &sets[*set as usize])
                            });
                            let ptr = Arc::as_ptr(&body.bundle) as *const () as usize;
                            let b = *bundle_idx.entry(ptr).or_insert_with(|| {
                                bundles.push(body.bundle.clone());
                                bundles.len() as u32 - 1
                            });
                            (b, body.bundle.n_outputs() as u32)
                        };
                        // Distinct argument lists in first-encounter order,
                        // and each member's (group, slot) within the block.
                        let mut arg_lists: Vec<Vec<ExprId>> = Vec::new();
                        let mut group_of_list: HashMap<Vec<ExprId>, u32> = HashMap::default();
                        let mut ins: Vec<Ref> = Vec::new();
                        for &mi in members {
                            let Node::Call(_, l) = *ctx.node(base[mi]) else {
                                unreachable!()
                            };
                            let args = ctx.args(l).to_vec();
                            if !group_of_list.contains_key(&args) {
                                group_of_list.insert(args.clone(), arg_lists.len() as u32);
                                ins.extend(args.iter().map(|&a| val(a, &value)));
                                arg_lists.push(args);
                            }
                        }
                        let n_args = arg_lists[0].len() as u32;
                        let n_groups = arg_lists.len() as u32;
                        let pure = members.iter().all(|&mi| self.pure[mi]);
                        let inst = insts.len() as u32;
                        insts.push(Inst {
                            kind: Kind::CallBatch {
                                bundle,
                                n_groups,
                                n_args,
                            },
                            ins: pooled(&mut pool, ins),
                            n_out: n_groups * n_out,
                            pure,
                        });
                        let body = &bodies[&(*f, *set)];
                        for &mi in members {
                            let Node::Call(o, l) = *ctx.node(base[mi]) else {
                                unreachable!()
                            };
                            let (_, out) = ctx.output(o);
                            let gi = group_of_list[&ctx.args(l).to_vec()];
                            value[mi] = Some(match body.slot_of[out as usize] {
                                Some(slot) => Ref::Value(inst, gi * n_out + slot),
                                None => {
                                    // A zero output: a derivative the body
                                    // does not carry.
                                    insts.push(Inst {
                                        kind: Kind::Const(0.0),
                                        ins: (pool.len() as u32, 0),
                                        n_out: 1,
                                        pure: true,
                                    });
                                    Ref::Value(insts.len() as u32 - 1, 0)
                                }
                            });
                        }
                    }
                    GroupKey::Gemv(x, ..) => {
                        let mut ins: Vec<Ref> = Vec::new();
                        for &mi in members {
                            let Node::Dot(l) = *ctx.node(base[mi]) else {
                                unreachable!()
                            };
                            ins.extend(ctx.dot_args(l).0.iter().map(|&a| val(a, &value)));
                        }
                        ins.extend(x.iter().map(|&a| val(a, &value)));
                        let pure = members.iter().all(|&mi| self.pure[mi]);
                        let inst = insts.len() as u32;
                        insts.push(Inst {
                            kind: Kind::Gemv {
                                m: members.len() as u32,
                                n: x.len() as u32,
                                acc: None,
                            },
                            ins: pooled(&mut pool, ins),
                            n_out: members.len() as u32,
                            pure,
                        });
                        for (r, &mi) in members.iter().enumerate() {
                            value[mi] = Some(Ref::Value(inst, r as u32));
                        }
                    }
                    GroupKey::Gemm(rows, xs) => {
                        let (rm, k, cn) = (rows.len(), xs[0].len(), xs.len());
                        let mut ins: Vec<Ref> = Vec::with_capacity((rm + cn) * k);
                        for r in rows {
                            ins.extend(r.iter().map(|&a| val(a, &value)));
                        }
                        for x in xs {
                            ins.extend(x.iter().map(|&a| val(a, &value)));
                        }
                        let row_index: HashMap<&[ExprId], usize> = rows
                            .iter()
                            .enumerate()
                            .map(|(i, r)| (r.as_slice(), i))
                            .collect();
                        let col_index: HashMap<&[ExprId], usize> = xs
                            .iter()
                            .enumerate()
                            .map(|(j, x)| (x.as_slice(), j))
                            .collect();
                        let pure = members.iter().all(|&mi| self.pure[mi]);
                        let inst = insts.len() as u32;
                        insts.push(Inst {
                            kind: Kind::Gemm {
                                m: rm as u32,
                                k: k as u32,
                                n: cn as u32,
                                acc: None,
                            },
                            ins: pooled(&mut pool, ins),
                            n_out: (rm * cn) as u32,
                            pure,
                        });
                        for &mi in members {
                            let Node::Dot(l) = *ctx.node(base[mi]) else {
                                unreachable!()
                            };
                            let (a, x) = ctx.dot_args(l);
                            let (r, c) = (row_index[a], col_index[x]);
                            value[mi] = Some(Ref::Value(inst, (r * cn + c) as u32));
                        }
                    }
                    GroupKey::SolveMany(a, lists) => {
                        let n = (a.len() as f64).sqrt() as u32;
                        let kk = lists.len() as u32;
                        let mut ins: Vec<Ref> = a.iter().map(|&e| val(e, &value)).collect();
                        for l in lists {
                            let all = ctx.args(*l);
                            ins.extend(all[(n * n) as usize..].iter().map(|&e| val(e, &value)));
                        }
                        let col_of: HashMap<crate::node::ArgList, u32> = lists
                            .iter()
                            .enumerate()
                            .map(|(c, &l)| (l, c as u32))
                            .collect();
                        let pure = members.iter().all(|&mi| self.pure[mi]);
                        let inst = insts.len() as u32;
                        insts.push(Inst {
                            kind: Kind::SolveMany { n, k: kk },
                            ins: pooled(&mut pool, ins),
                            n_out: n * kk,
                            pure,
                        });
                        for &mi in members {
                            let Node::Solve(l, c) = *ctx.node(base[mi]) else {
                                unreachable!()
                            };
                            value[mi] = Some(Ref::Value(inst, col_of[&l] * n + c));
                        }
                    }
                    GroupKey::Solve(l, _) => {
                        let all = ctx.args(*l);
                        let n = Graph::<K>::solve_n(all.len()) as u32;
                        let ins: Vec<Ref> = all.iter().map(|&a| val(a, &value)).collect();
                        let pure = members.iter().all(|&mi| self.pure[mi]);
                        let inst = insts.len() as u32;
                        insts.push(Inst {
                            kind: Kind::Solve { n },
                            ins: pooled(&mut pool, ins),
                            n_out: n,
                            pure,
                        });
                        for &mi in members {
                            let Node::Solve(_, c) = *ctx.node(base[mi]) else {
                                unreachable!()
                            };
                            value[mi] = Some(Ref::Value(inst, c));
                        }
                    }
                }
                continue;
            }
            // An ordinary node, one instruction; an Add with a fused
            // operand reads through it.
            let (kind, ins): (Kind, Vec<Ref>) = match node {
                Node::Add(a, b) => {
                    let (pa, pb) = (self.pos(a), self.pos(b));
                    if self.fused_into[pa] == Some(i) || self.fused_into[pb] == Some(i) {
                        let (fused, other) = if self.fused_into[pa] == Some(i) {
                            (a, b)
                        } else {
                            (b, a)
                        };
                        match *ctx.node(fused) {
                            Node::Mul(x, y) => (
                                Kind::MulAdd,
                                vec![val(x, &value), val(y, &value), val(other, &value)],
                            ),
                            Node::Neg(x) => (Kind::Sub, vec![val(other, &value), val(x, &value)]),
                            _ => unreachable!("only a Mul or a Neg fuses"),
                        }
                    } else {
                        (Kind::Add, vec![val(a, &value), val(b, &value)])
                    }
                }
                Node::Mul(a, b) => (Kind::Mul, vec![val(a, &value), val(b, &value)]),
                Node::Neg(a) => (Kind::Neg, vec![val(a, &value)]),
                Node::Pow(a, n) => (Kind::Powi(n as i32), vec![val(a, &value)]),
                Node::Unary(op, a) => (Kind::Unary(op), vec![val(a, &value)]),
                Node::Binary(op, a, b) => (Kind::Binary(op), vec![val(a, &value), val(b, &value)]),
                Node::Cmp(op, a, b) => (Kind::Cmp(op), vec![val(a, &value), val(b, &value)]),
                Node::Select(c, t, e) => (
                    Kind::Select,
                    vec![val(c, &value), val(t, &value), val(e, &value)],
                ),
                Node::Reduce(op, l) => (
                    Kind::Reduce(op),
                    ctx.args(l).iter().map(|&a| val(a, &value)).collect(),
                ),
                Node::Dot(l) => {
                    let (a, b) = ctx.dot_args(l);
                    (
                        Kind::Dot(a.len() as u32),
                        a.iter().chain(b).map(|&e| val(e, &value)).collect(),
                    )
                }
                Node::Call(o, l) => {
                    // A single call: its outputs a block, this node one of
                    // them; other outputs of the same call join it.
                    let (f, _) = ctx.output(o);
                    let args = ctx.args(l).to_vec();
                    let (set, outs) = needed(f.0, l);
                    let body = bodies
                        .entry((f.0, set))
                        .or_insert_with(|| ctx.func(f).body_for(ctx, outs));
                    let ptr = Arc::as_ptr(&body.bundle) as *const () as usize;
                    let bundle = *bundle_idx.entry(ptr).or_insert_with(|| {
                        bundles.push(body.bundle.clone());
                        bundles.len() as u32 - 1
                    });
                    let n_out = body.bundle.n_outputs() as u32;
                    let inst = insts.len() as u32;
                    let ins: Vec<Ref> = args.iter().map(|&a| val(a, &value)).collect();
                    insts.push(Inst {
                        kind: Kind::Call { bundle },
                        ins: pooled(&mut pool, ins),
                        n_out,
                        pure: self.pure[i],
                    });
                    // Every reachable output of this very call, this one
                    // included.
                    for &j in &call_sites[&(f.0, l)] {
                        let Node::Call(o2, _) = *ctx.node(base[j]) else {
                            unreachable!()
                        };
                        let (_, out2) = ctx.output(o2);
                        value[j] = Some(match body.slot_of[out2 as usize] {
                            Some(slot) => Ref::Value(inst, slot),
                            None => {
                                insts.push(Inst {
                                    kind: Kind::Const(0.0),
                                    ins: (pool.len() as u32, 0),
                                    n_out: 1,
                                    pure: true,
                                });
                                Ref::Value(insts.len() as u32 - 1, 0)
                            }
                        });
                    }
                    continue;
                }
                Node::Solve(..) => unreachable!("a solve component is a kernel member"),
                Node::Const(_) | Node::Symbol(_) => unreachable!("leaves handled above"),
            };
            insts.push(Inst {
                kind,
                ins: pooled(&mut pool, ins),
                n_out: 1,
                pure: self.pure[i],
            });
            value[i] = Some(Ref::Value(insts.len() as u32 - 1, 0));
        }
        let roots: Vec<Ref> = roots
            .iter()
            .map(|r| value[self.pos(*r)].expect("a root is lowered"))
            .collect();
        Program {
            insts,
            pool,
            bundles,
            roots,
        }
    }
}

impl Program {
    /// Pass 3: the instruction order. Register-pressure list scheduling:
    /// among the ready instructions prefer the one that kills the most live
    /// values, and among equals the most recently enabled one (finish the
    /// computation in flight before starting another), so a device's
    /// derivatives follow its residual and the live set is one device's
    /// worth, not the circuit's. The prolog split is a strict phase: every
    /// pure instruction precedes every impure one. Constants are placed
    /// right before their first consumer, so a multiply-used one is live
    /// from its first use, not from the start.
    fn schedule(&self) -> Vec<u32> {
        let m = self.insts.len();
        let is_const = |i: usize| matches!(self.insts[i].kind, Kind::Const(_));
        // Distinct value dependencies of each instruction, as instruction
        // indices; a constant is placed on demand and counts as no
        // dependency.
        let mut stamp: Vec<u32> = vec![u32::MAX; m];
        let deps: Vec<Vec<u32>> = (0..m)
            .map(|i| {
                let mut d: Vec<u32> = Vec::new();
                for r in self.ins(i) {
                    if let Ref::Value(j, _) = *r {
                        if stamp[j as usize] != i as u32 {
                            stamp[j as usize] = i as u32;
                            d.push(j);
                        }
                    }
                }
                d
            })
            .collect();
        let mut user_count = vec![0u32; m];
        let mut pending = vec![0u32; m];
        for (i, d) in deps.iter().enumerate() {
            pending[i] = d.iter().filter(|&&x| !is_const(x as usize)).count() as u32;
            for &x in d {
                user_count[x as usize] += 1;
            }
        }
        let mut user_start = vec![0u32; m + 1];
        for i in 0..m {
            user_start[i + 1] = user_start[i] + user_count[i];
        }
        let mut users = vec![0u32; user_start[m] as usize];
        let mut fill = user_start.clone();
        for (i, d) in deps.iter().enumerate() {
            for &x in d {
                users[fill[x as usize] as usize] = i as u32;
                fill[x as usize] += 1;
            }
        }
        let mut remaining = user_count.clone();
        let kills_of = |i: usize, remaining: &[u32]| -> u32 {
            deps[i]
                .iter()
                .filter(|&&d| remaining[d as usize] == 1)
                .count() as u32
        };
        let mut heap: std::collections::BinaryHeap<(bool, u32, u64, u32)> =
            std::collections::BinaryHeap::new();
        let mut seq: u64 = 0;
        for i in (0..m).rev() {
            if !is_const(i) && pending[i] == 0 {
                heap.push((self.insts[i].pure, kills_of(i, &remaining), seq, i as u32));
                seq += 1;
            }
        }
        let mut order: Vec<u32> = Vec::with_capacity(m);
        let mut placed = vec![false; m];
        let mut last: Option<usize> = None;
        while let Some(top) = heap.pop() {
            let (pure, kills, _, iu) = top;
            let mut i = iu as usize;
            if placed[i] {
                continue;
            }
            // Interleave two chains when an equally good candidate does not
            // read the instruction just placed, so the CPU overlaps them.
            if let (Some(lo), Some(&(p2, k2, _, iu2))) = (last, heap.peek()) {
                let i2 = iu2 as usize;
                if p2 == pure
                    && k2 == kills
                    && !placed[i2]
                    && deps[i].contains(&(lo as u32))
                    && !deps[i2].contains(&(lo as u32))
                {
                    heap.pop();
                    heap.push(top);
                    i = i2;
                }
            }
            last = Some(i);
            for &d in &deps[i] {
                let d = d as usize;
                if is_const(d) && !placed[d] {
                    placed[d] = true;
                    order.push(d as u32);
                }
            }
            placed[i] = true;
            order.push(i as u32);
            for &d in &deps[i] {
                remaining[d as usize] -= 1;
            }
            for k in user_start[i]..user_start[i + 1] {
                let u = users[k as usize] as usize;
                pending[u] -= 1;
                if pending[u] == 0 {
                    heap.push((self.insts[u].pure, kills_of(u, &remaining), seq, u as u32));
                    seq += 1;
                }
            }
        }
        // Constants nothing consumes (a constant root).
        for i in 0..m {
            if !placed[i] {
                debug_assert!(is_const(i), "the list scheduler placed every instruction");
                order.push(i as u32);
            }
        }
        debug_assert_eq!(order.len(), m);
        order
    }

    /// Pass 4: lifetimes, slots and the instruction stream.
    fn emit(&self, order: &[u32], split: bool) -> Tape {
        let m = self.insts.len();
        let mut pos = vec![0usize; m];
        for (k, &i) in order.iter().enumerate() {
            pos[i as usize] = k;
        }
        // The prolog is the pure prefix of the schedule.
        let prolog_ops = if split {
            order
                .iter()
                .position(|&i| !self.insts[i as usize].pure)
                .unwrap_or(m)
        } else {
            0
        };

        // Last use of each instruction's block (the highest position that
        // reads any of its values); roots and, under a split, prolog values
        // read by the main phase are pinned.
        let mut last = vec![0usize; m];
        let mut pinned = vec![false; m];
        for (k, &i) in order.iter().enumerate() {
            for r in self.ins(i as usize) {
                if let Ref::Value(j, _) = *r {
                    last[j as usize] = last[j as usize].max(k);
                }
            }
        }
        for r in &self.roots {
            if let Ref::Value(j, _) = *r {
                pinned[j as usize] = true;
                last[j as usize] = usize::MAX;
            }
        }
        if split {
            for i in 0..m {
                if pos[i] < prolog_ops && last[i] >= prolog_ops {
                    pinned[i] = true;
                    last[i] = usize::MAX;
                }
            }
        }
        // A kernel reads its dense operands from consecutive slots when
        // they are consecutive, and gathers them into scratch otherwise: the
        // single values a kernel consumes are placed in the order the kernel
        // reads them, a reserved run per operand, so a block row computed
        // entry by entry is read in place. First come first served in
        // schedule order; a value already placed for one kernel keeps its
        // slot. Reserved slots are never reused.
        let mut reserved = vec![u32::MAX; m];
        let mut next: u32 = 0;
        for &i in order {
            let inst = &self.insts[i as usize];
            let runs: Vec<usize> = match inst.kind {
                Kind::Gemv {
                    m: rows,
                    n,
                    ref acc,
                } => {
                    let mut r = vec![(rows * n) as usize, n as usize];
                    if acc.is_some() {
                        r.push(rows as usize);
                    }
                    r
                }
                Kind::Gemm {
                    m: rows,
                    k,
                    n,
                    ref acc,
                } => {
                    let mut r = vec![(rows * k) as usize, (n * k) as usize];
                    if acc.is_some() {
                        r.push((rows * n) as usize);
                    }
                    r
                }
                Kind::Solve { n } => vec![(n * n) as usize, n as usize],
                Kind::SolveMany { n, k } => vec![(n * n) as usize, (n * k) as usize],
                _ => continue,
            };
            let ins = self.ins(i as usize);
            let mut at = 0usize;
            for len in runs {
                let operand = &ins[at..at + len];
                at += len;
                // Maximal stretches of single values not yet placed.
                let mut s = 0usize;
                while s < operand.len() {
                    let placeable = |r: &Ref| match *r {
                        Ref::Value(j, 0) => {
                            self.insts[j as usize].n_out == 1 && reserved[j as usize] == u32::MAX
                        }
                        _ => false,
                    };
                    if !placeable(&operand[s]) {
                        s += 1;
                        continue;
                    }
                    let mut e = s;
                    while e < operand.len() && placeable(&operand[e]) {
                        e += 1;
                    }
                    if e - s >= 4 {
                        for r in &operand[s..e] {
                            if let Ref::Value(j, _) = *r {
                                reserved[j as usize] = next;
                                next += 1;
                            }
                        }
                    }
                    s = e;
                }
            }
        }
        // Slots: a LIFO free list for single values, a fresh block for a
        // kernel; an instruction's dying operands are freed first, so it
        // can reuse one of their slots.
        let mut base = vec![u32::MAX; m];
        let mut free: Vec<u32> = Vec::new();
        let mut ops: Vec<Op> = Vec::with_capacity(m);
        let mut dst: Vec<u32> = Vec::with_capacity(m);
        let mut arg_pool: Vec<u32> = Vec::new();
        let mut max_args = 0usize;
        let mut n_selects = 0usize;
        let slot_of = |r: Ref, base: &[u32]| -> u32 {
            match r {
                Ref::Value(j, k) => base[j as usize] + k,
                Ref::Input(k) => k | INPUT,
            }
        };
        for (k, &i) in order.iter().enumerate() {
            let inst = &self.insts[i as usize];
            // The operands, before any slot of this step is freed.
            let ins = self.ins(i as usize);
            let operands: Vec<u32> = ins.iter().map(|&r| slot_of(r, &base)).collect();
            let mut dying: Vec<u32> = ins
                .iter()
                .filter_map(|r| match *r {
                    Ref::Value(j, _)
                        if last[j as usize] == k
                            && !pinned[j as usize]
                            && reserved[j as usize] == u32::MAX =>
                    {
                        Some(j)
                    }
                    _ => None,
                })
                .collect();
            dying.sort_unstable();
            dying.dedup();
            for j in dying {
                let b = base[j as usize];
                free.extend(b..b + self.insts[j as usize].n_out);
            }
            let d = if inst.n_out == 1 && reserved[i as usize] != u32::MAX {
                reserved[i as usize]
            } else if inst.n_out == 1 {
                free.pop().unwrap_or_else(|| {
                    let s = next;
                    next += 1;
                    s
                })
            } else {
                let s = next;
                next += inst.n_out;
                s
            };
            base[i as usize] = d;
            let gather = |arg_pool: &mut Vec<u32>, max_args: &mut usize, ops: &[u32]| -> u32 {
                let start = arg_pool.len() as u32;
                arg_pool.extend_from_slice(ops);
                *max_args = (*max_args).max(ops.len());
                start
            };
            // A dense operand: a run of consecutive inputs is read in place.
            let dense = |arg_pool: &mut Vec<u32>, max_args: &mut usize, ops: &[u32]| -> Src {
                let run = ops.first().and_then(|&k0| {
                    let k0 = super::input_index(k0)?;
                    ops.iter()
                        .enumerate()
                        .all(|(j, &k)| super::input_index(k) == Some(k0 + j as u32))
                        .then_some(k0)
                });
                match run {
                    Some(k0) => Src::Inputs(k0),
                    None => Src::Pool(gather(arg_pool, max_args, ops)),
                }
            };
            let o = &operands;
            let op = match inst.kind {
                Kind::Const(v) => Op::Const(v),
                Kind::Add => Op::Add(o[0], o[1]),
                Kind::Mul => Op::Mul(o[0], o[1]),
                Kind::MulAdd => Op::MulAdd(o[0], o[1], o[2]),
                Kind::Sub => Op::Sub(o[0], o[1]),
                Kind::Neg => Op::Neg(o[0]),
                Kind::Powi(n) => Op::Powi(o[0], n),
                Kind::Unary(op) => Op::Unary(op, o[0]),
                Kind::Binary(op) => Op::Binary(op, o[0], o[1]),
                Kind::Cmp(op) => Op::Cmp(op, o[0], o[1]),
                Kind::Select => {
                    n_selects += 1;
                    Op::Select(o[0], o[1], o[2])
                }
                Kind::Reduce(op) => {
                    let start = gather(&mut arg_pool, &mut max_args, o);
                    Op::Reduce(op, start, o.len() as u32)
                }
                Kind::Dot(n) => {
                    let start = gather(&mut arg_pool, &mut max_args, o);
                    Op::Dot(start, n)
                }
                Kind::Call { bundle } => {
                    let start = gather(&mut arg_pool, &mut max_args, o);
                    Op::Call {
                        bundle,
                        start,
                        n_args: o.len() as u32,
                        n_out: inst.n_out,
                    }
                }
                Kind::CallBatch {
                    bundle,
                    n_groups,
                    n_args,
                } => {
                    let start = gather(&mut arg_pool, &mut max_args, o);
                    Op::CallBatch {
                        bundle,
                        start,
                        n_groups,
                        n_args,
                        n_out: inst.n_out / n_groups,
                    }
                }
                Kind::Gemv {
                    m: rows,
                    n,
                    ref acc,
                } => {
                    let (a, rest) = o.split_at((rows * n) as usize);
                    let (x, c) = rest.split_at(n as usize);
                    let a = dense(&mut arg_pool, &mut max_args, a);
                    let x = dense(&mut arg_pool, &mut max_args, x);
                    let acc = acc.as_ref().map(|codes| {
                        let reads = codes.iter().any(|&code| Fold(code).reads_operand());
                        let c = reads.then(|| dense(&mut arg_pool, &mut max_args, c));
                        let start = arg_pool.len() as u32;
                        arg_pool.extend_from_slice(codes);
                        super::Accum { c, codes: start }
                    });
                    // The scratch holds every operand: an input run that the
                    // inputs do not reach is gathered as NaN.
                    max_args = max_args.max((rows * n + n + rows) as usize);
                    Op::Gemv {
                        a,
                        x,
                        m: rows,
                        n,
                        acc,
                    }
                }
                Kind::Gemm {
                    m: rows,
                    k,
                    n,
                    ref acc,
                } => {
                    let (a, rest) = o.split_at((rows * k) as usize);
                    let (b, c) = rest.split_at((n * k) as usize);
                    let a = dense(&mut arg_pool, &mut max_args, a);
                    let b = dense(&mut arg_pool, &mut max_args, b);
                    let acc = acc.as_ref().map(|codes| {
                        let reads = codes.iter().any(|&code| Fold(code).reads_operand());
                        let c = reads.then(|| dense(&mut arg_pool, &mut max_args, c));
                        let start = arg_pool.len() as u32;
                        arg_pool.extend_from_slice(codes);
                        super::Accum { c, codes: start }
                    });
                    max_args = max_args.max((rows * k + n * k + rows * n) as usize);
                    Op::Gemm {
                        a,
                        b,
                        m: rows,
                        k,
                        n,
                        acc,
                    }
                }
                Kind::Solve { n } => {
                    let (a, b) = o.split_at((n * n) as usize);
                    let a = dense(&mut arg_pool, &mut max_args, a);
                    let b = dense(&mut arg_pool, &mut max_args, b);
                    max_args = max_args.max((n * n + n) as usize);
                    Op::Solve { a, b, n }
                }
                Kind::SolveMany { n, k } => {
                    let (a, b) = o.split_at((n * n) as usize);
                    let a = dense(&mut arg_pool, &mut max_args, a);
                    let b = dense(&mut arg_pool, &mut max_args, b);
                    max_args = max_args.max((n * n + n * k) as usize);
                    Op::SolveMany { a, b, n, k }
                }
            };
            ops.push(op);
            dst.push(d);
        }
        let outputs: Vec<u32> = self.roots.iter().map(|&r| slot_of(r, &base)).collect();
        Tape {
            ops,
            dst,
            n_selects,
            arg_pool,
            outputs,
            n_work: next as usize,
            max_args,
            bundles: self.bundles.clone(),
            prolog_ops,
        }
    }
}
