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

use rustc_hash::FxHashMap as HashMap;

use super::{Fold, Op, Src, Tape, INPUT};
use crate::extern_fn::ExternBundle;
use crate::field::Field;
use crate::func::{Body, FuncId};
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};

/// Row dots against one vector fuse into a `Gemv` from this many rows on.
const GEMV_MIN_ROWS: usize = 8;

/// A call's operands and the slot of its state block (the last operand of
/// a stateful call), or [`NO_STATE`](super::NO_STATE).
fn split_state(o: &[u32], stateful: bool) -> (&[u32], u32) {
    match (stateful, o.split_last()) {
        (true, Some((&state, args))) => (args, state),
        _ => (o, super::NO_STATE),
    }
}

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
        let mut tape = timed("tape emit", || program.emit(&order, pure_inputs.is_some()));
        tape.n_inputs = input_syms.len();
        tape
    }
}

/// Dense tables over a graph's nodes and symbols, reused by every
/// compilation on a thread: allocated once to the largest graph seen and
/// valid by generation stamp, so compiling a small tape out of a large
/// graph costs the tape, not the graph. A compilation takes the thread's
/// tables and gives them back when its [`Forest`] drops (a nested one, a
/// body compiled while lowering, gets fresh tables).
#[derive(Default)]
struct Scratch {
    generation: u32,
    /// Node `k` is in this compilation's forest when `seen[k]` is the
    /// generation; `pos[k]` is then its base position.
    seen: Vec<u32>,
    pos: Vec<u32>,
    /// Symbol `s` is input `input[s]` when `input_seen[s]` is the generation.
    input_seen: Vec<u32>,
    input: Vec<u32>,
}

std::thread_local! {
    static SCRATCH: std::cell::Cell<Option<Scratch>> = const { std::cell::Cell::new(None) };
}

impl Scratch {
    fn take(n_nodes: usize, n_symbols: usize) -> Scratch {
        let mut t = SCRATCH.with(|c| c.take()).unwrap_or_default();
        t.generation = t.generation.wrapping_add(1);
        if t.generation == 0 {
            // Every stamp could be a stale match after a wrap: start over.
            t.seen.fill(0);
            t.input_seen.fill(0);
            t.generation = 1;
        }
        if t.seen.len() < n_nodes {
            t.seen.resize(n_nodes, 0);
            t.pos.resize(n_nodes, u32::MAX);
        }
        if t.input_seen.len() < n_symbols {
            t.input_seen.resize(n_symbols, 0);
            t.input.resize(n_symbols, u32::MAX);
        }
        t
    }

    /// Mark node `e`; whether it was unmarked.
    #[inline]
    fn mark(&mut self, e: ExprId) -> bool {
        let k = e.0 as usize;
        let fresh = self.seen[k] != self.generation;
        self.seen[k] = self.generation;
        fresh
    }

    /// Input `k` is symbol `s` (a later listing of one symbol wins).
    fn set_input(&mut self, s: SymbolId, k: u32) {
        if let Some(slot) = self.input.get_mut(s.0 as usize) {
            *slot = k;
            self.input_seen[s.0 as usize] = self.generation;
        }
    }

    #[inline]
    fn input(&self, s: SymbolId) -> Option<u32> {
        let k = s.0 as usize;
        (self.input_seen.get(k) == Some(&self.generation)).then(|| self.input[k])
    }
}

impl Drop for Forest {
    fn drop(&mut self) {
        let t = std::mem::take(&mut self.tables);
        SCRATCH.with(|c| c.set(Some(t)));
    }
}

/// The reachable forest of one compilation, in dependency order, with the
/// tables the later passes index by *base position* (the index into
/// [`Forest::base`], not the arena id).
struct Forest {
    base: Vec<ExprId>,
    /// Base positions and input indices, in tables sized to the arena but
    /// valid only for this compilation's nodes and inputs (see [`Scratch`]).
    tables: Scratch,
    /// Parameter-purity for the prolog split; all false without one.
    pure: Vec<bool>,
    /// `fused_into[p]` is the base position of the `Add` that absorbed the
    /// node at `p` as a superinstruction operand.
    fused_into: Vec<Option<usize>>,
    /// Compiled with a prolog split: a call's parameter-pure part can run
    /// in the prolog (see [`Forest::stateful`]).
    split: bool,
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
    /// With `stateful`, the last operand is the instance state block (the
    /// value of a [`Kind::CallProlog`]).
    Call {
        bundle: u32,
        stateful: bool,
    },
    CallBatch {
        bundle: u32,
        n_groups: u32,
        n_args: u32,
        stateful: bool,
    },
    /// The prolog of `n_groups` instances over their pure arguments,
    /// `n_pure` per group; a block of their states.
    CallProlog {
        bundle: u32,
        n_groups: u32,
        n_pure: u32,
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
    /// The accumulator operand of the kernels' plain folds (see
    /// [`Program::fuse_accumulators`]): never read, so never part of the state.
    placeholder: Option<u32>,
}

impl Program {
    fn ins(&self, i: usize) -> &[Ref] {
        let (s, l) = self.insts[i].ins;
        &self.pool[s as usize..(s + l) as usize]
    }

    /// The operand each foldable kernel output would take as its
    /// accumulator, `(operand, kernel)` sorted: [`Topo`] starts with it
    /// before the kernel where it can.
    fn fold_hints(&self, uses: &HashMap<(u32, u32), (u32, u32)>) -> Vec<(u32, u32)> {
        let mut hints: Vec<(u32, u32)> = Vec::new();
        for (&(k, c), &(count, j)) in uses {
            if count != 1 || j == u32::MAX {
                continue;
            }
            let this = Ref::Value(k, c);
            let ins = self.ins(j as usize);
            let other = match self.insts[j as usize].kind {
                Kind::Sub if ins[1] == this => ins[0],
                Kind::Add if ins[0] == this => ins[1],
                Kind::Add => ins[0],
                _ => continue,
            };
            // Another output of the kernel, or a value read off one, is
            // no accumulator to place first.
            if let Ref::Value(x, _) = other {
                let off_k = |r: &Ref| matches!(*r, Ref::Value(y, _) if y == k);
                if x != k && !self.ins(x as usize).iter().any(off_k) {
                    hints.push((x, k));
                }
            }
        }
        hints.sort_unstable();
        hints.dedup();
        hints
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
    /// it, which [`Topo`] answers, and a pure kernel takes only pure
    /// accumulators (the prolog keeps it).
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
        // Built at the first accumulator after its kernel: until then the
        // lowering order is a topological order that every fold kept.
        let mut topo: Option<Topo> = None;
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
                self.placeholder = Some(self.insts.len() as u32 - 1);
                Ref::Value(self.insts.len() as u32 - 1, 0)
            });
            let mut accs: Vec<Ref> = vec![placeholder; n_out as usize];
            let mut fused: Vec<(u32, u32)> = Vec::new(); // (output, consumer)
                                                         // The instructions this kernel reads as accumulators so far.
            let mut taken: Vec<u32> = Vec::new();
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
                        let fits = match topo.as_mut() {
                            Some(t) => t.read(self, k as u32, i),
                            None if (i as usize) < k => true,
                            None => {
                                let hints = self.fold_hints(&uses);
                                topo.insert(Topo::new(self, &hints, k as u32, &taken))
                                    .read(self, k as u32, i)
                            }
                        };
                        if !fits {
                            continue;
                        }
                        taken.push(i);
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
        self.placeholder = self.placeholder.map(|i| renumber[i as usize]);
        let old = std::mem::take(&mut self.insts);
        self.insts = old
            .into_iter()
            .zip(&dead)
            .filter(|(_, &d)| !d)
            .map(|(inst, _)| inst)
            .collect();
    }
}

/// A topological order of a program's instructions that the fold pass
/// keeps valid as kernels take accumulators, by two-way search (Haeupler,
/// Kavitha, Mathew, Sen and Tarjan). Whether accumulator `a` depends on
/// kernel `k` is only a question when `a` comes after `k`; then the
/// readers of `k` searched forward up to `a`, and what `a` reads searched
/// backward down to `k`, step in turn. Either finding the other end is a
/// cycle. The side that runs out first is closed under its direction
/// between the two and moves, the backward side right before `k` or the
/// forward side right after `a`, so a question costs the smaller side.
/// Instructions added after the order was taken (the placeholder) are
/// constants and take part in no edge.
struct Topo {
    order: Order,
    /// Readers of each instruction (`readers[start[i]..start[i + 1]]`),
    /// and the ones the pass added.
    start: Vec<u32>,
    readers: Vec<u32>,
    added: HashMap<u32, Vec<u32>>,
    seen: Vec<u32>,
    stamp: u32,
    stack_f: Vec<(u32, u32)>,
    stack_b: Vec<(u32, u32)>,
    fwd: Vec<u32>,
    back: Vec<u32>,
}

impl Topo {
    /// The order of `p` with kernel `k` also reading `taken`, the
    /// accumulators it took before the order was needed.
    fn new(p: &Program, hints: &[(u32, u32)], k: u32, taken: &[u32]) -> Topo {
        let m = p.insts.len();
        // Every edge `(j, i)`, instruction `i` reading `j`, once.
        let edges = |f: &mut dyn FnMut(u32, u32)| {
            let mut last = vec![u32::MAX; m];
            for i in 0..m {
                for r in p.ins(i) {
                    if let Ref::Value(j, _) = *r {
                        if last[j as usize] != i as u32 {
                            last[j as usize] = i as u32;
                            f(j, i as u32);
                        }
                    }
                }
            }
            for &a in taken {
                if last[a as usize] != k {
                    last[a as usize] = k;
                    f(a, k);
                }
            }
        };
        let mut count = vec![0u32; m + 1];
        edges(&mut |j, _| count[j as usize + 1] += 1);
        for i in 0..m {
            count[i + 1] += count[i];
        }
        let mut fill = count.clone();
        let mut readers = vec![0u32; count[m] as usize];
        edges(&mut |j, i| {
            readers[fill[j as usize] as usize] = i;
            fill[j as usize] += 1;
        });
        // The starting order: a kernel after the accumulators `hints`
        // offers it (`(operand, kernel)`, sorted) where the program allows,
        // and otherwise as late as its readers allow. The lowering places a
        // kernel with its first consumer, before the accumulators of its
        // other outputs and before the kernels those depend on, so most
        // accumulators would otherwise come after their kernel. When
        // nothing else can go, the kernel with the fewest hints open does
        // (a hint that depends on its kernel never closes).
        let kernel = |i: usize| matches!(p.insts[i].kind, Kind::Gemv { .. } | Kind::Gemm { .. });
        let mut pending = vec![0u32; m];
        for &r in &readers {
            pending[r as usize] += 1;
        }
        let mut hinted = vec![0u32; m];
        for &(_, k) in hints {
            hinted[k as usize] += 1;
        }
        let hints_of = |x: u32| {
            let lo = hints.partition_point(|&(y, _)| y < x);
            let hi = hints.partition_point(|&(y, _)| y <= x);
            &hints[lo..hi]
        };
        let mut ready: Vec<u32> = Vec::new();
        let mut ready_kernels: BTreeSet<u32> = BTreeSet::new();
        // Kernels whose operands are there, by their hints still open.
        let mut blocked: BTreeSet<(u32, u32)> = BTreeSet::new();
        let release = |r: u32,
                       hinted: &[u32],
                       ready: &mut Vec<u32>,
                       ready_kernels: &mut BTreeSet<u32>,
                       blocked: &mut BTreeSet<(u32, u32)>| {
            if !kernel(r as usize) {
                ready.push(r);
            } else if hinted[r as usize] == 0 {
                ready_kernels.insert(r);
            } else {
                blocked.insert((hinted[r as usize], r));
            }
        };
        for i in (0..m as u32).rev() {
            if pending[i as usize] == 0 {
                release(i, &hinted, &mut ready, &mut ready_kernels, &mut blocked);
            }
        }
        let mut at: Vec<u32> = Vec::with_capacity(m);
        while let Some(i) = ready
            .pop()
            .or_else(|| ready_kernels.pop_first())
            .or_else(|| blocked.pop_first().map(|(_, k)| k))
        {
            at.push(i);
            for &(_, k) in hints_of(i) {
                let h = hinted[k as usize];
                hinted[k as usize] = h - 1;
                if blocked.remove(&(h, k)) {
                    if h == 1 {
                        ready_kernels.insert(k);
                    } else {
                        blocked.insert((h - 1, k));
                    }
                }
            }
            for &r in &readers[count[i as usize] as usize..count[i as usize + 1] as usize] {
                pending[r as usize] -= 1;
                if pending[r as usize] == 0 {
                    release(r, &hinted, &mut ready, &mut ready_kernels, &mut blocked);
                }
            }
        }
        debug_assert_eq!(at.len(), m, "the lowered program is acyclic");
        Topo {
            order: Order::new(&at),
            start: count,
            readers,
            added: HashMap::default(),
            seen: vec![0; m],
            stamp: 0,
            stack_f: Vec::new(),
            stack_b: Vec::new(),
            fwd: Vec::new(),
            back: Vec::new(),
        }
    }

    /// Lets kernel `k` read instruction `a` unless `a` depends on `k`;
    /// whether it did.
    fn read(&mut self, p: &Program, k: u32, a: u32) -> bool {
        let n = self.seen.len() as u32;
        if a >= n {
            return true;
        }
        let label = &self.order.label;
        let (lb, ub) = (label[k as usize], label[a as usize]);
        if ub < lb {
            self.added.entry(a).or_default().push(k);
            return true;
        }
        // Forward marks are `sf`, backward ones `sb`. The sides take one
        // edge each in turn (a kernel's readers can be thousands), each a
        // stack of nodes with the next edge to look at.
        if self.stamp > u32::MAX - 4 {
            self.seen.fill(0);
            self.stamp = 0;
        }
        self.stamp += 2;
        let (sf, sb) = (self.stamp, self.stamp + 1);
        self.stack_f.clear();
        self.stack_b.clear();
        self.fwd.clear();
        self.back.clear();
        self.stack_f.push((k, 0));
        self.fwd.push(k);
        self.seen[k as usize] = sf;
        self.stack_b.push((a, 0));
        self.back.push(a);
        self.seen[a as usize] = sb;
        let backward = loop {
            // One forward edge.
            let Some(&(w, e)) = self.stack_f.last() else {
                break false;
            };
            let (s0, s1) = (self.start[w as usize], self.start[w as usize + 1]);
            let deg = s1 - s0;
            let r = if e < deg {
                Some(self.readers[(s0 + e) as usize])
            } else {
                self.added
                    .get(&w)
                    .and_then(|v| v.get((e - deg) as usize))
                    .copied()
            };
            match r {
                None => {
                    self.stack_f.pop();
                }
                Some(r) => {
                    self.stack_f.last_mut().unwrap().1 += 1;
                    if r == a {
                        return false;
                    }
                    if self.seen[r as usize] != sf && label[r as usize] < ub {
                        self.seen[r as usize] = sf;
                        self.fwd.push(r);
                        self.stack_f.push((r, 0));
                    }
                }
            }
            // One backward edge.
            let Some(&(w, e)) = self.stack_b.last() else {
                break true;
            };
            match p.ins(w as usize).get(e as usize) {
                None => {
                    self.stack_b.pop();
                }
                Some(&r) => {
                    self.stack_b.last_mut().unwrap().1 += 1;
                    if let Ref::Value(j, _) = r {
                        if j == k {
                            return false;
                        }
                        if j < n && self.seen[j as usize] != sb && label[j as usize] > lb {
                            self.seen[j as usize] = sb;
                            self.back.push(j);
                            self.stack_b.push((j, 0));
                        }
                    }
                }
            }
        };
        let order = &mut self.order;
        let set = if backward {
            &mut self.back
        } else {
            &mut self.fwd
        };
        set.sort_unstable_by_key(|&w| order.label[w as usize]);
        for &w in set.iter() {
            order.unlink(w);
        }
        let mut at = if backward { order.prev[k as usize] } else { a };
        for &w in set.iter() {
            order.insert_after(at, w);
            at = w;
        }
        self.added.entry(a).or_default().push(k);
        true
    }
}

/// A list whose order is compared by labels (the order maintenance of
/// Bender, Cole, Demaine, Farach-Colton and Zito): an element moved in
/// takes the middle of the gap it lands in, and a gap too narrow relabels
/// the smallest aligned label range around it that is sparse enough, its
/// allowance shrinking with its size, amortized O(log n) labels a move.
struct Order {
    label: Vec<u64>,
    prev: Vec<u32>,
    next: Vec<u32>,
}

impl Order {
    /// The elements `0..n` in `order`, between a head (`n`, label 0) and a
    /// tail (`n + 1`, the largest label).
    fn new(order: &[u32]) -> Order {
        let n = order.len();
        let (head, tail) = (n as u32, n as u32 + 1);
        let mut label = vec![0u64; n + 2];
        let mut prev = vec![0u32; n + 2];
        let mut next = vec![0u32; n + 2];
        let step = u64::MAX / (n as u64 + 2);
        let mut last = head;
        for (q, &w) in order.iter().enumerate() {
            label[w as usize] = (q as u64 + 1) * step;
            prev[w as usize] = last;
            next[last as usize] = w;
            last = w;
        }
        next[last as usize] = tail;
        prev[tail as usize] = last;
        label[tail as usize] = u64::MAX;
        Order { label, prev, next }
    }

    fn unlink(&mut self, w: u32) {
        let (p, q) = (self.prev[w as usize], self.next[w as usize]);
        self.next[p as usize] = q;
        self.prev[q as usize] = p;
    }

    /// Links `w` right after `x`.
    fn insert_after(&mut self, x: u32, w: u32) {
        let y = self.next[x as usize];
        self.prev[w as usize] = x;
        self.next[w as usize] = y;
        self.next[x as usize] = w;
        self.prev[y as usize] = w;
        let (lo, hi) = (self.label[x as usize], self.label[y as usize]);
        if hi - lo >= 2 {
            self.label[w as usize] = lo + (hi - lo) / 2;
            return;
        }
        let head = self.label.len() as u32 - 2;
        let tail = head + 1;
        for i in 1..=64u32 {
            let size = 1u128 << i;
            let base = lo as u128 & !(size - 1);
            let end = base + size;
            let mut first = w;
            let mut c = 1u128;
            let mut v = x;
            while v != head && self.label[v as usize] as u128 >= base {
                first = v;
                c += 1;
                v = self.prev[v as usize];
            }
            let mut v = y;
            while v != tail && (self.label[v as usize] as u128) < end {
                c += 1;
                v = self.next[v as usize];
            }
            // Sparse enough: fewer than (2 / 1.5)^i elements in 2^i labels.
            if i == 64 || (c as f64) < (4.0f64 / 3.0).powi(i as i32) {
                let from = base.max(1);
                let to = end.min(u64::MAX as u128);
                let gap = (to - from) / (c + 1);
                let mut v = first;
                for j in 0..c {
                    self.label[v as usize] = (from + gap * (j + 1)) as u64;
                    v = self.next[v as usize];
                }
                return;
            }
        }
    }
}

impl Forest {
    #[inline]
    fn pos(&self, e: ExprId) -> usize {
        self.tables.pos[e.0 as usize] as usize
    }

    /// Whether calls of `b` on `lists` (argument lists of one or more
    /// instances, all impure as a whole) keep their state in the caller:
    /// under a split, when the bundle has a state and every argument it
    /// flags pure is parameter-pure here, so its prolog can run in ours.
    fn stateful(&self, b: &dyn ExternBundle, lists: &[&[ExprId]]) -> bool {
        let mask = b.pure_args();
        self.split
            && b.state_len() > 0
            && mask.iter().any(|&p| p)
            && lists.iter().all(|args| {
                args.len() == mask.len()
                    && args
                        .iter()
                        .zip(mask)
                        .all(|(&a, &p)| !p || self.pure[self.pos(a)])
            })
    }

    /// The input index of symbol `s`, if it is one.
    #[inline]
    fn input(&self, s: SymbolId) -> Option<u32> {
        self.tables.input(s)
    }

    /// Pass 1: reachability, purity, use counts and superinstruction fusion.
    fn analyze<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> Forest {
        let mut t = Scratch::take(ctx.len(), ctx.n_symbols());
        for (k, &s) in input_syms.iter().enumerate() {
            t.set_input(s, k as u32);
        }
        // The reachable nodes, marked in the stamped table; ascending ids
        // are a dependency order. The walk and the sort cost the forest's
        // size, not the graph's.
        let mut base: Vec<ExprId> = Vec::new();
        let mut stack = roots.to_vec();
        while let Some(id) = stack.pop() {
            if !t.mark(id) {
                continue;
            }
            base.push(id);
            stack.extend_from_slice(&ctx.operands(id));
        }
        base.sort_unstable_by_key(|e| e.0);
        let m = base.len();
        for (i, id) in base.iter().enumerate() {
            t.pos[id.0 as usize] = i as u32;
        }
        let bpos = &t.pos;
        let bp = |e: ExprId| bpos[e.0 as usize] as usize;

        // Purity: a node is parameter-pure when every operand is; an
        // unmapped symbol is a NaN constant and so pure.
        let mut pure = vec![false; m];
        if let Some(mask) = pure_inputs {
            for (i, id) in base.iter().enumerate() {
                pure[i] = match ctx.node(*id) {
                    Node::Const(_) => true,
                    Node::Symbol(s) => match t.input(*s) {
                        None => true,
                        Some(k) => mask.get(k as usize).copied().unwrap_or(false),
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
            tables: t,
            pure,
            fused_into,
            split: pure_inputs.is_some(),
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
        // One buffer for every unit's raw operands and one for the distinct
        // ones, reused: a million units are not two million allocations.
        let mut raw: Vec<usize> = Vec::new();
        let mut deps: Vec<usize> = Vec::new();
        let mut deps_of_unit = |u: usize, deps: &mut Vec<usize>| {
            raw.clear();
            match kernel_of[u] {
                Some(g) => {
                    for &mi in &groups[g].1 {
                        deps_of_node(mi, &mut raw);
                    }
                }
                None => deps_of_node(u, &mut raw),
            }
            deps.clear();
            for &d in &raw {
                if stamp[d] != u {
                    stamp[d] = u;
                    deps.push(d);
                }
            }
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
            deps_of_unit(u, &mut deps);
            for &d in &deps {
                if !done[d] {
                    stack.push((d, false));
                }
            }
        }

        // --- the instructions, in that order --------------------------------
        let leaf = |e: ExprId| -> Option<Ref> {
            match ctx.node(e) {
                Node::Symbol(s) => Some(Ref::Input(self.input(*s).unwrap_or(u32::MAX))),
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
                        let b = bundles[bundle as usize].clone();
                        let lists: Vec<&[ExprId]> = arg_lists.iter().map(Vec::as_slice).collect();
                        let stateful = !pure && self.stateful(&*b, &lists);
                        if stateful {
                            let mask = b.pure_args();
                            let ps = pool.len() as u32;
                            for args in &arg_lists {
                                pool.extend(
                                    args.iter()
                                        .zip(mask)
                                        .filter(|&(_, &p)| p)
                                        .map(|(&a, _)| val(a, &value)),
                                );
                            }
                            let n_pure = mask.iter().filter(|&&p| p).count() as u32;
                            insts.push(Inst {
                                kind: Kind::CallProlog {
                                    bundle,
                                    n_groups,
                                    n_pure,
                                },
                                ins: (ps, pool.len() as u32 - ps),
                                n_out: n_groups * b.state_len() as u32,
                                pure: true,
                            });
                            ins.push(Ref::Value(insts.len() as u32 - 1, 0));
                        }
                        let inst = insts.len() as u32;
                        insts.push(Inst {
                            kind: Kind::CallBatch {
                                bundle,
                                n_groups,
                                n_args,
                                stateful,
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
            // The operands go straight into the program's pool.
            let start = pool.len() as u32;
            let kind = match node {
                Node::Add(a, b) => {
                    let (pa, pb) = (self.pos(a), self.pos(b));
                    if self.fused_into[pa] == Some(i) || self.fused_into[pb] == Some(i) {
                        let (fused, other) = if self.fused_into[pa] == Some(i) {
                            (a, b)
                        } else {
                            (b, a)
                        };
                        match *ctx.node(fused) {
                            Node::Mul(x, y) => {
                                pool.extend_from_slice(&[
                                    val(x, &value),
                                    val(y, &value),
                                    val(other, &value),
                                ]);
                                Kind::MulAdd
                            }
                            Node::Neg(x) => {
                                pool.extend_from_slice(&[val(other, &value), val(x, &value)]);
                                Kind::Sub
                            }
                            _ => unreachable!("only a Mul or a Neg fuses"),
                        }
                    } else {
                        {
                            pool.extend_from_slice(&[val(a, &value), val(b, &value)]);
                            Kind::Add
                        }
                    }
                }
                Node::Mul(a, b) => {
                    pool.extend_from_slice(&[val(a, &value), val(b, &value)]);
                    Kind::Mul
                }
                Node::Neg(a) => {
                    pool.extend_from_slice(&[val(a, &value)]);
                    Kind::Neg
                }
                Node::Pow(a, n) => {
                    pool.extend_from_slice(&[val(a, &value)]);
                    Kind::Powi(n as i32)
                }
                Node::Unary(op, a) => {
                    pool.extend_from_slice(&[val(a, &value)]);
                    Kind::Unary(op)
                }
                Node::Binary(op, a, b) => {
                    pool.extend_from_slice(&[val(a, &value), val(b, &value)]);
                    Kind::Binary(op)
                }
                Node::Cmp(op, a, b) => {
                    pool.extend_from_slice(&[val(a, &value), val(b, &value)]);
                    Kind::Cmp(op)
                }
                Node::Select(c, t, e) => {
                    pool.extend_from_slice(&[val(c, &value), val(t, &value), val(e, &value)]);
                    Kind::Select
                }
                Node::Reduce(op, l) => {
                    pool.extend(ctx.args(l).iter().map(|&a| val(a, &value)));
                    Kind::Reduce(op)
                }
                Node::Dot(l) => {
                    let (a, b) = ctx.dot_args(l);
                    pool.extend(a.iter().chain(b).map(|&e| val(e, &value)));
                    Kind::Dot(a.len() as u32)
                }
                Node::Call(o, l) => {
                    // A single call: its outputs a block, this node one of
                    // them; other outputs of the same call join it.
                    let (f, _) = ctx.output(o);
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
                    let args = ctx.args(l);
                    let stateful = !self.pure[i] && self.stateful(&*body.bundle, &[args]);
                    let state = stateful.then(|| {
                        let mask = body.bundle.pure_args();
                        let ps = pool.len() as u32;
                        pool.extend(
                            args.iter()
                                .zip(mask)
                                .filter(|&(_, &p)| p)
                                .map(|(&a, _)| val(a, &value)),
                        );
                        insts.push(Inst {
                            kind: Kind::CallProlog {
                                bundle,
                                n_groups: 1,
                                n_pure: pool.len() as u32 - ps,
                            },
                            ins: (ps, pool.len() as u32 - ps),
                            n_out: body.bundle.state_len() as u32,
                            pure: true,
                        });
                        Ref::Value(insts.len() as u32 - 1, 0)
                    });
                    let start = pool.len() as u32;
                    pool.extend(args.iter().map(|&a| val(a, &value)));
                    pool.extend(state);
                    let inst = insts.len() as u32;
                    insts.push(Inst {
                        kind: Kind::Call { bundle, stateful },
                        ins: (start, pool.len() as u32 - start),
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
                ins: (start, pool.len() as u32 - start),
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
            placeholder: None,
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
        // indices, in one flat list (`dep_list[dep_start[i]..dep_start[i+1]]`);
        // a constant is placed on demand and counts as no dependency.
        let mut stamp: Vec<u32> = vec![u32::MAX; m];
        let mut dep_start: Vec<u32> = Vec::with_capacity(m + 1);
        let mut dep_list: Vec<u32> = Vec::with_capacity(self.pool.len());
        dep_start.push(0);
        for i in 0..m {
            for r in self.ins(i) {
                if let Ref::Value(j, _) = *r {
                    if stamp[j as usize] != i as u32 {
                        stamp[j as usize] = i as u32;
                        dep_list.push(j);
                    }
                }
            }
            dep_start.push(dep_list.len() as u32);
        }
        let deps = |i: usize| &dep_list[dep_start[i] as usize..dep_start[i + 1] as usize];
        let mut user_count = vec![0u32; m];
        let mut pending = vec![0u32; m];
        for i in 0..m {
            pending[i] = deps(i).iter().filter(|&&x| !is_const(x as usize)).count() as u32;
            for &x in deps(i) {
                user_count[x as usize] += 1;
            }
        }
        let mut user_start = vec![0u32; m + 1];
        for i in 0..m {
            user_start[i + 1] = user_start[i] + user_count[i];
        }
        let mut users = vec![0u32; user_start[m] as usize];
        let mut fill = user_start.clone();
        for i in 0..m {
            for &x in deps(i) {
                users[fill[x as usize] as usize] = i as u32;
                fill[x as usize] += 1;
            }
        }
        let mut remaining = user_count.clone();
        let kills_of = |i: usize, remaining: &[u32]| -> u32 {
            deps(i)
                .iter()
                .filter(|&&d| remaining[d as usize] == 1)
                .count() as u32
        };
        let mut ready = Ready::default();
        for i in (0..m).rev() {
            if !is_const(i) && pending[i] == 0 {
                ready.push(self.insts[i].pure, kills_of(i, &remaining), i as u32);
            }
        }
        let mut order: Vec<u32> = Vec::with_capacity(m);
        let mut placed = vec![false; m];
        let mut last: Option<usize> = None;
        while let Some((pure, kills, iu)) = ready.pop() {
            let mut i = iu as usize;
            // Interleave two chains when an equally good candidate does not
            // read the instruction just placed, so the CPU overlaps them.
            if let (Some(lo), Some(iu2)) = (last, ready.peek_same(pure, kills)) {
                let i2 = iu2 as usize;
                if deps(i).contains(&(lo as u32)) && !deps(i2).contains(&(lo as u32)) {
                    ready.swap_top(pure, kills, iu);
                    i = i2;
                }
            }
            last = Some(i);
            for &d in deps(i) {
                let d = d as usize;
                if is_const(d) && !placed[d] {
                    placed[d] = true;
                    order.push(d as u32);
                }
            }
            placed[i] = true;
            order.push(i as u32);
            for &d in deps(i) {
                remaining[d as usize] -= 1;
            }
            for k in user_start[i]..user_start[i + 1] {
                let u = users[k as usize] as usize;
                pending[u] -= 1;
                if pending[u] == 0 {
                    ready.push(self.insts[u].pure, kills_of(u, &remaining), u as u32);
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
        // read by the main phase are pinned (not the kernels' placeholder
        // accumulator, which nothing reads).
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
                if pos[i] < prolog_ops
                    && last[i] >= prolog_ops
                    && self.placeholder != Some(i as u32)
                {
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
        // The state: every value the prolog computes and something after it
        // reads (the main phase, or the outputs), in one block at the start
        // of the work array, so an instance's prolog result is `work[..state]`
        // whatever else the buffer holds, the same in every backend.
        let mut state_base = vec![u32::MAX; m];
        let mut next: u32 = 0;
        for &i in &order[..prolog_ops] {
            if pinned[i as usize] {
                state_base[i as usize] = next;
                next += self.insts[i as usize].n_out;
            }
        }
        let state_len = next as usize;
        let mut reserved = vec![u32::MAX; m];
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
                            self.insts[j as usize].n_out == 1
                                && reserved[j as usize] == u32::MAX
                                && state_base[j as usize] == u32::MAX
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
        // Per-instruction scratch, reused across the stream.
        let mut operands: Vec<u32> = Vec::new();
        let mut dying: Vec<u32> = Vec::new();
        for (k, &i) in order.iter().enumerate() {
            let inst = &self.insts[i as usize];
            // The operands, before any slot of this step is freed.
            let ins = self.ins(i as usize);
            operands.clear();
            operands.extend(ins.iter().map(|&r| slot_of(r, &base)));
            dying.clear();
            dying.extend(ins.iter().filter_map(|r| match *r {
                Ref::Value(j, _)
                    if last[j as usize] == k
                        && !pinned[j as usize]
                        && reserved[j as usize] == u32::MAX =>
                {
                    Some(j)
                }
                _ => None,
            }));
            dying.sort_unstable();
            dying.dedup();
            for &j in &dying {
                let b = base[j as usize];
                free.extend(b..b + self.insts[j as usize].n_out);
            }
            let d = if state_base[i as usize] != u32::MAX {
                state_base[i as usize]
            } else if inst.n_out == 1 && reserved[i as usize] != u32::MAX {
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
                Kind::Call { bundle, stateful } => {
                    // A stateful call's last operand is its state block.
                    let (args, state) = split_state(o, stateful);
                    let start = gather(&mut arg_pool, &mut max_args, args);
                    Op::Call {
                        bundle,
                        start,
                        n_args: args.len() as u32,
                        n_out: inst.n_out,
                        state,
                    }
                }
                Kind::CallBatch {
                    bundle,
                    n_groups,
                    n_args,
                    stateful,
                } => {
                    let (args, state) = split_state(o, stateful);
                    let start = gather(&mut arg_pool, &mut max_args, args);
                    Op::CallBatch {
                        bundle,
                        start,
                        n_groups,
                        n_args,
                        n_out: inst.n_out / n_groups,
                        state,
                    }
                }
                Kind::CallProlog {
                    bundle,
                    n_groups,
                    n_pure,
                } => {
                    let start = gather(&mut arg_pool, &mut max_args, o);
                    Op::CallProlog {
                        bundle,
                        start,
                        n_groups,
                        n_pure,
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
        let bundle_work = self.bundles.iter().map(|b| b.work_len()).max().unwrap_or(0);
        Tape {
            bundle_work,
            ops,
            dst,
            n_selects,
            arg_pool,
            outputs,
            n_work: next as usize,
            max_args,
            bundles: self.bundles.clone(),
            prolog_ops,
            state_len,
            n_inputs: 0,
        }
    }
}

/// The ready instructions of the list scheduler, best first: pure before
/// impure, more kills before fewer, and among equals the most recently
/// enabled. A stack per `(pure, kills)` bucket gives that order without a
/// heap: pushes come in enabling order, so a bucket's top is its most
/// recent entry, and the best bucket is the highest non-empty one.
#[derive(Default)]
struct Ready {
    /// `buckets[pure][kills]`.
    buckets: [Vec<Vec<u32>>; 2],
    /// Per level, no bucket above this one holds anything.
    hi: [usize; 2],
    len: usize,
}

impl Ready {
    fn push(&mut self, pure: bool, kills: u32, i: u32) {
        let (p, k) = (pure as usize, kills as usize);
        let level = &mut self.buckets[p];
        if level.len() <= k {
            level.resize_with(k + 1, Vec::new);
        }
        level[k].push(i);
        self.hi[p] = self.hi[p].max(k);
        self.len += 1;
    }

    /// The best bucket, `(pure, kills)`, if any holds anything.
    fn best(&mut self) -> Option<(usize, usize)> {
        if self.len == 0 {
            return None;
        }
        for p in [1, 0] {
            let level = &self.buckets[p];
            while self.hi[p] > 0 && level.get(self.hi[p]).is_none_or(Vec::is_empty) {
                self.hi[p] -= 1;
            }
            if level.get(self.hi[p]).is_some_and(|b| !b.is_empty()) {
                return Some((p, self.hi[p]));
            }
        }
        None
    }

    fn pop(&mut self) -> Option<(bool, u32, u32)> {
        let (p, k) = self.best()?;
        let i = self.buckets[p][k].pop()?;
        self.len -= 1;
        Some((p == 1, k as u32, i))
    }

    /// The next best entry when it is in the bucket `(pure, kills)`.
    fn peek_same(&self, pure: bool, kills: u32) -> Option<u32> {
        self.buckets[pure as usize]
            .get(kills as usize)?
            .last()
            .copied()
    }

    /// Take the top of bucket `(pure, kills)` in place of `i`, which goes
    /// back on top: it was that bucket's most recent entry and stays so.
    fn swap_top(&mut self, pure: bool, kills: u32, i: u32) {
        let b = &mut self.buckets[pure as usize][kills as usize];
        let top = b.len() - 1;
        b[top] = i;
    }
}
