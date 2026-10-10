//! The tape compiler: the reachable forest lowered to instructions, then
//! scheduled, then given slots.
//!
//! Five passes, each reading only what the ones before it produced:
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
//! 3. [`Program::fuse_accumulators`]: a product kernel's output whose one
//!    consumer adds, subtracts or negates it is folded by the kernel.
//! 4. [`Program::schedule`]: register-pressure list scheduling over the
//!    instructions, the prolog as a strict first phase.
//! 5. [`Program::emit`]: lifetimes, slots (a kernel's outputs a block of
//!    consecutive slots), and the instruction stream.
//!
//! Passes 1 and 2 serve every tape over one set of roots: a [`Lowered`]
//! program is lowered once, and a tape of some of its roots is a view of
//! it, the instructions those roots reach, its calls computing the outputs
//! read ([`Program::narrow_calls`]), then passes 3 to 5 alone.
//!
//! A compiled tape can be lifted back into its program ([`Tape::lift`]),
//! which is how a transform of a tape (a specialization) is compiled.

use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::topo::Topo;
use super::{input_index, Accum, Fold, Op, Src, Tape, INPUT};
use crate::extern_fn::ExternBundle;
use crate::field::Field;
use crate::func::{Body, FuncId, Output};
use crate::graph::Graph;
use crate::graph::NO_CONTEXT;
use crate::node::{ArgList, ExprId, Node, SymbolId};

/// One instance's calls: the context they run in (see `Graph::bind`) and
/// their argument list.
type Site = (u32, ArgList);
use crate::semantics::{SOLVE_BATCH_MAX_K, SOLVE_BATCH_MAX_N};

/// Row dots against one vector fuse into a `Gemv` from this many rows on.
const GEMV_MIN_ROWS: usize = 2;

/// A call's operands and the slot of its state block (the last operand of
/// a stateful call), or [`NO_STATE`](super::NO_STATE).
fn split_state(o: &[u32], stateful: bool) -> (&[u32], u32) {
    match (stateful, o.split_last()) {
        (true, Some((&state, args))) => (args, state),
        _ => (o, super::NO_STATE),
    }
}

impl Tape {
    /// Compile a tape computing `roots`, where `inputs[k]` (passed to
    /// [`eval`](Self::eval)) is the value of symbol `input_syms[k]`. Symbols not
    /// listed evaluate to `NaN`.
    ///
    /// A hierarchy is compiled as it stands: a composite function (one
    /// whose body calls others) is lowered once into a template, its
    /// instructions over its operands, and a call of it appends the
    /// template with its operands in their place; no graph is rewritten
    /// per instance. The calls of a leaf function (a device model's body)
    /// from every instance then run as one batch, as in a flat graph.
    /// Differentiate and specialize on the hierarchy first: that work stays
    /// on its functions.
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
        let program = lower(ctx, roots, input_syms, pure_inputs);
        finish(program, pure_inputs, input_syms.len())
    }
}

/// Passes 1 and 2: `roots` lowered over the inputs `input_syms`.
fn lower<K: Field>(
    ctx: &Graph<K>,
    roots: &[ExprId],
    input_syms: &[SymbolId],
    pure_inputs: Option<&[bool]>,
) -> Program {
    use crate::hooks::timed;
    let forest = timed("tape analyze", || {
        Forest::analyze(ctx, roots, input_syms, pure_inputs)
    });
    let mut templates = Templates::default();
    let mut program = timed("tape lower", || forest.lower(ctx, roots, &mut templates));
    if templates.expanded {
        timed("tape merge", || program.merge_calls());
    }
    program
}

/// Passes 3 to 5: `program` fused, scheduled and given slots.
fn finish(mut program: Program, pure_inputs: Option<&[bool]>, n_inputs: usize) -> Tape {
    use crate::hooks::timed;
    timed("tape fuse", || program.fuse_accumulators(pure_inputs));
    let order = timed("tape schedule", || program.schedule());
    let split = pure_inputs.is_some();
    let mut tape = timed("tape emit", || program.emit(&order, split));
    tape.n_inputs = n_inputs;
    tape
}

/// The program of a set of roots over one signature, lowered once (the
/// reachable forest analyzed, its kernels grouped, its templates expanded
/// and their calls merged): the tapes of a system that compute some of the
/// same roots (its residuals; with its Jacobian; with its charges) are views
/// of one program rather than compilations of their own. A view takes the
/// instructions its roots reach, its calls computing only the outputs it
/// reads, and is scheduled and given slots alone; it computes what
/// [`Tape::compile`] of its roots computes, value for value.
pub struct Lowered {
    program: Program,
    /// The roots, and the place of each among them (its first).
    roots: Vec<ExprId>,
    at: HashMap<ExprId, u32>,
    inputs: Vec<SymbolId>,
    pure_inputs: Option<Vec<bool>>,
}

impl Lowered {
    /// `roots` lowered over the inputs `input_syms`; with `pure_inputs`,
    /// every view split into a prolog as by [`Tape::compile_split`].
    pub fn new<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> Lowered {
        let program = lower(ctx, roots, input_syms, pure_inputs);
        let mut at: HashMap<ExprId, u32> = HashMap::default();
        for (k, &r) in roots.iter().enumerate() {
            at.entry(r).or_insert(k as u32);
        }
        Lowered {
            program,
            roots: roots.to_vec(),
            at,
            inputs: input_syms.to_vec(),
            pure_inputs: pure_inputs.map(<[bool]>::to_vec),
        }
    }

    /// The roots it was lowered for.
    pub fn roots(&self) -> &[ExprId] {
        &self.roots
    }

    /// Whether it was lowered over the inputs `input_syms`, split by
    /// `pure_inputs`.
    pub fn signature(&self, input_syms: &[SymbolId], pure_inputs: Option<&[bool]>) -> bool {
        self.inputs == input_syms && self.pure_inputs.as_deref() == pure_inputs
    }

    /// Whether a tape of `roots` over that signature is a view of it.
    pub fn serves(
        &self,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> bool {
        self.signature(input_syms, pure_inputs) && roots.iter().all(|r| self.at.contains_key(r))
    }

    /// The tape computing `roots`, each one of the lowered roots.
    pub fn tape<K: Field>(&self, ctx: &Graph<K>, roots: &[ExprId]) -> Tape {
        use crate::hooks::timed;
        let mut p = self.program.clone();
        p.roots = (roots.iter())
            .map(|r| self.program.roots[*self.at.get(r).expect("a lowered root") as usize])
            .collect();
        let pure = self.pure_inputs.as_deref();
        timed("tape view", || {
            p.retain_reachable();
            p.narrow_calls(ctx, pure);
        });
        finish(p, pure, self.inputs.len())
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
    /// The operands of the calls of functions with globals: the arguments,
    /// then the globals (see [`Graph::globals`]); a call of a function
    /// without has its argument list.
    ext: HashMap<(u32, Site), Vec<ExprId>>,
}

/// One instruction of the lowered program, before scheduling.
///
/// `ins` are the operands in the order the op reads them (slots as value
/// references, or tagged inputs, see [`INPUT`]); `n_out` is the number of
/// values it produces, one for anything but a kernel. Value `(inst, k)` is
/// the `k`th output of instruction `inst`.
#[derive(Clone)]
pub(super) struct Inst {
    pub(super) kind: Kind,
    /// The operands: `pool[start .. start + len]` of the program's pool,
    /// one flat vector for every instruction, so a million instructions
    /// are one allocation and not a million.
    pub(super) ins: (u32, u32),
    pub(super) n_out: u32,
    /// Parameter-pure: schedulable into the prolog.
    pub(super) pure: bool,
}

/// An operand of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Ref {
    /// Output `k` of instruction `inst`.
    Value(u32, u32),
    /// Input `k`, read in place.
    Input(u32),
}

/// What an instruction computes; the operand lists live in `Inst::ins`.
#[derive(Clone, Debug)]
pub(super) enum Kind {
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
    /// `n_groups` argument lists of `n_args`, group-major; with `stateful`,
    /// the last operand is the instances' state block (the value of a
    /// [`Kind::CallProlog`]). With `reads`, a list holds only the arguments
    /// at those positions (see [`Op::Call`]).
    /// `func` is the function the bundle is a body of ([`NO_FUNC`] for one
    /// lifted from a tape).
    Call {
        bundle: u32,
        func: u32,
        n_groups: u32,
        n_args: u32,
        reads: Option<Arc<[u32]>>,
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
    /// `k` right-hand sides against one matrix, for each of `count`
    /// systems of one shape (matrices first, then right-hand sides, each
    /// system's back to back).
    Solve {
        n: u32,
        k: u32,
        count: u32,
    },
}

/// The function of a call lifted from a tape: unknown.
pub(super) const NO_FUNC: u32 = u32::MAX;

/// The lowered program: instructions in a dependency order, the bundle
/// table, and which value each root is.
#[derive(Clone)]
pub(super) struct Program {
    pub(super) insts: Vec<Inst>,
    /// The operand pool of every instruction (see [`Inst::ins`]).
    pub(super) pool: Vec<Ref>,
    pub(super) bundles: Vec<Arc<dyn ExternBundle>>,
    pub(super) roots: Vec<Ref>,
    /// The accumulator operand of the kernels' plain folds (see
    /// [`Program::fuse_accumulators`]): never read, so never part of the state.
    pub(super) placeholder: Option<u32>,
}

impl Program {
    pub(super) fn ins(&self, i: usize) -> &[Ref] {
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
        self.compact(&dead, |r| {
            let mut r = r;
            while let Ref::Value(i, _) = r {
                match alias.get(&i) {
                    Some(&to) => r = to,
                    None => break,
                }
            }
            r
        });
    }

    /// Drop the instructions `dead` marks, every operand and root first
    /// redirected by `resolve` (to a value that stays), the rest renumbered
    /// in order; the operand pool holds the ones that stay only.
    pub(super) fn compact(&mut self, dead: &[bool], resolve: impl Fn(Ref) -> Ref) {
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
        let old_pool = std::mem::take(&mut self.pool);
        let old = std::mem::take(&mut self.insts);
        self.pool.reserve(old_pool.len());
        for (mut inst, _) in old.into_iter().zip(dead).filter(|(_, &d)| !d) {
            let (s, l) = inst.ins;
            let start = self.pool.len() as u32;
            self.pool.extend(
                old_pool[s as usize..(s + l) as usize]
                    .iter()
                    .map(|&r| map(r)),
            );
            inst.ins = (start, l);
            self.insts.push(inst);
        }
        for r in self.roots.iter_mut() {
            *r = map(*r);
        }
        self.placeholder = self
            .placeholder
            .filter(|&i| !dead[i as usize])
            .map(|i| renumber[i as usize]);
    }

    /// The calls of one bundle the expanded templates left apart, one per
    /// instance's body, merged into one call each (and their prologs into
    /// one prolog): calls of the same bundle, phase, purity and arguments
    /// read, at the same call depth, so no member reads another's output.
    pub(super) fn merge_calls(&mut self) {
        #[derive(PartialEq, Eq, Hash)]
        struct Key {
            bundle: u32,
            n_args: u32,
            reads: Option<Vec<u32>>,
            stateful: bool,
            pure: bool,
            depth: u32,
        }
        let m = self.insts.len();
        // the insts are in a dependency order: operands first
        let mut depth = vec![0u32; m];
        for i in 0..m {
            let d = self
                .ins(i)
                .iter()
                .filter_map(|r| match *r {
                    Ref::Value(j, _) => Some(depth[j as usize]),
                    Ref::Input(_) => None,
                })
                .max()
                .unwrap_or(0);
            depth[i] = d + u32::from(matches!(self.insts[i].kind, Kind::Call { .. }));
        }
        let mut index: HashMap<Key, usize> = HashMap::default();
        let mut groups: Vec<Vec<u32>> = Vec::new();
        for i in 0..m {
            if let Kind::Call {
                bundle,
                n_args,
                ref reads,
                stateful,
                ..
            } = self.insts[i].kind
            {
                let key = Key {
                    bundle,
                    n_args,
                    reads: reads.as_ref().map(|r| r.to_vec()),
                    stateful,
                    pure: self.insts[i].pure,
                    depth: depth[i],
                };
                let g = *index.entry(key).or_insert_with(|| {
                    groups.push(Vec::new());
                    groups.len() - 1
                });
                groups[g].push(i as u32);
            }
        }
        let mut dead = vec![false; m];
        // per merged-away call: the merged call, and per group of it the
        // group it is in the merged one
        let mut moved: HashMap<u32, (u32, Vec<u32>)> = HashMap::default();
        for members in groups.iter().filter(|g| g.len() >= 2) {
            let first = &self.insts[members[0] as usize];
            let Kind::Call {
                bundle,
                func,
                n_args,
                ref reads,
                stateful,
                ..
            } = first.kind
            else {
                unreachable!()
            };
            let (reads, pure) = (reads.clone(), first.pure);
            let n_groups_of = |c: u32| match self.insts[c as usize].kind {
                Kind::Call { n_groups, .. } | Kind::CallProlog { n_groups, .. } => {
                    n_groups as usize
                }
                _ => unreachable!(),
            };
            let prolog_of = |c: u32| match *self.ins(c as usize).last().expect("a state") {
                Ref::Value(p, 0) => p,
                _ => unreachable!("a stateful call's last operand is its state"),
            };
            // The instances, each once: one with the operands of another
            // (and the pure ones of its prolog) computes the same.
            let mut seen: HashMap<Vec<Ref>, u32> = HashMap::default();
            let (mut ins, mut pure_ins): (Vec<Ref>, Vec<Ref>) = (Vec::new(), Vec::new());
            let (mut out_g, mut state_g) = (0u32, 0u32);
            for &c in members {
                let ng = n_groups_of(c);
                let own = self.ins(c as usize);
                let own = &own[..own.len() - usize::from(stateful)];
                let ni = own.len() / ng;
                let pro = stateful.then(|| {
                    let p = prolog_of(c);
                    (
                        self.ins(p as usize),
                        self.insts[p as usize].n_out as usize / ng,
                    )
                });
                out_g = self.insts[c as usize].n_out / ng as u32;
                let mut map = Vec::with_capacity(ng);
                for g in 0..ng {
                    let mut key = own[g * ni..(g + 1) * ni].to_vec();
                    if let Some((p, _)) = pro {
                        let np = p.len() / ng;
                        key.extend_from_slice(&p[g * np..(g + 1) * np]);
                    }
                    let n = seen.len() as u32;
                    let at = *seen.entry(key).or_insert_with(|| {
                        ins.extend_from_slice(&own[g * ni..(g + 1) * ni]);
                        if let Some((p, sl)) = pro {
                            let np = p.len() / ng;
                            pure_ins.extend_from_slice(&p[g * np..(g + 1) * np]);
                            state_g = sl as u32;
                        }
                        n
                    });
                    map.push(at);
                }
                if stateful {
                    dead[prolog_of(c) as usize] = true;
                }
                dead[c as usize] = true;
                moved.insert(c, (0, map));
            }
            let n_groups = seen.len() as u32;
            if stateful {
                let Kind::CallProlog { n_pure, .. } =
                    self.insts[prolog_of(members[0]) as usize].kind
                else {
                    unreachable!()
                };
                let kind = Kind::CallProlog {
                    bundle,
                    n_groups,
                    n_pure,
                };
                let p = self.push(kind, pure_ins, n_groups * state_g, true);
                ins.push(Ref::Value(p, 0));
            }
            let kind = Kind::Call {
                bundle,
                func,
                n_groups,
                n_args,
                reads,
                stateful,
            };
            let at = self.push(kind, ins, n_groups * out_g, pure);
            for &c in members {
                let (to, map) = moved.get_mut(&c).expect("just moved");
                *to = at;
                // the outputs per group, for the resolution below
                map.push(out_g);
            }
        }
        dead.resize(self.insts.len(), false);
        self.compact(&dead, |r| match r {
            Ref::Value(i, o) => match moved.get(&i) {
                Some((at, map)) => {
                    let w = *map.last().expect("the width last");
                    Ref::Value(*at, map[(o / w) as usize] * w + o % w)
                }
                None => r,
            },
            r => r,
        });
    }

    /// Append an instruction; its index.
    fn push(&mut self, kind: Kind, ins: Vec<Ref>, n_out: u32, pure: bool) -> u32 {
        let start = self.pool.len() as u32;
        self.pool.extend(ins);
        self.insts.push(Inst {
            kind,
            ins: (start, self.pool.len() as u32 - start),
            n_out,
            pure,
        });
        self.insts.len() as u32 - 1
    }

    /// Drop what the roots do not reach.
    pub(super) fn retain_reachable(&mut self) {
        let m = self.insts.len();
        let mut dead = vec![true; m];
        let mut stack: Vec<u32> = self
            .roots
            .iter()
            .filter_map(|r| match *r {
                Ref::Value(i, _) => Some(i),
                Ref::Input(_) => None,
            })
            .collect();
        while let Some(i) = stack.pop() {
            if !std::mem::replace(&mut dead[i as usize], false) {
                continue;
            }
            stack.extend(self.ins(i as usize).iter().filter_map(|r| match *r {
                Ref::Value(j, _) if dead[j as usize] => Some(j),
                _ => None,
            }));
        }
        self.compact(&dead, |r| r);
    }

    /// Every call whose outputs not all are read, over a body of the ones
    /// read: a view of a program lowered for more roots (the residuals out
    /// of the residuals with their Jacobian) runs the bodies it needs, as
    /// its own compilation would. A call's instances, its arguments and,
    /// under a split (`pure_inputs`), its prolog are rebuilt for the body.
    pub(super) fn narrow_calls<K: Field>(&mut self, ctx: &Graph<K>, pure_inputs: Option<&[bool]>) {
        // per call of a known function, the outputs of a group it reads
        let mut read: HashMap<u32, Vec<bool>> = HashMap::default();
        for (i, inst) in self.insts.iter().enumerate() {
            if let Kind::Call { func, n_groups, .. } = inst.kind {
                if func != NO_FUNC {
                    read.insert(i as u32, vec![false; (inst.n_out / n_groups) as usize]);
                }
            }
        }
        if read.is_empty() {
            return;
        }
        let m = self.insts.len();
        for i in 0..m {
            for r in self.ins(i) {
                if let Ref::Value(j, o) = *r {
                    if let Some(u) = read.get_mut(&j) {
                        let w = u.len();
                        u[o as usize % w] = true;
                    }
                }
            }
        }
        for r in &self.roots {
            if let Ref::Value(j, o) = *r {
                if let Some(u) = read.get_mut(&j) {
                    let w = u.len();
                    u[o as usize % w] = true;
                }
            }
        }
        let is_pure = |p: &Program, r: Ref| match r {
            Ref::Value(j, _) => p.insts[j as usize].pure,
            Ref::Input(k) => pure_inputs.is_some_and(|m| m.get(k as usize) == Some(&true)),
        };
        let mut calls: Vec<(u32, Vec<bool>)> = read.into_iter().collect();
        calls.sort_unstable_by_key(|c| c.0);
        let mut dead = vec![false; m];
        // per call rebuilt: the new call, the old width, and per old slot
        // the new one
        let mut moved: HashMap<u32, (u32, u32, Vec<u32>)> = HashMap::default();
        for (i, used) in calls {
            if used.iter().all(|&u| u) {
                continue;
            }
            let Kind::Call {
                bundle,
                func,
                n_groups,
                n_args,
                ref reads,
                stateful,
            } = self.insts[i as usize].kind
            else {
                unreachable!()
            };
            let reads = reads.clone();
            let function = ctx.func(FuncId(func));
            let old = self.bundles[bundle as usize].clone();
            let Some(outs) = function.slot_outputs(&old) else {
                continue;
            };
            let needed: Vec<u32> = (used.iter().zip(&outs))
                .filter(|(&u, _)| u)
                .map(|(_, &k)| k)
                .collect();
            let body = function.body_exact(ctx, &needed);
            if Arc::ptr_eq(&body.bundle, &old) {
                continue;
            }
            // the instances' arguments, whole: from the call's list, and
            // what it does not read after its prolog from the prolog's
            let own = self.ins(i as usize).to_vec();
            let own = &own[..own.len() - usize::from(stateful)];
            let mask = old.pure_args().to_vec();
            let prolog = stateful.then(|| match own_last(self, i) {
                Ref::Value(p, 0) => p,
                _ => unreachable!("a stateful call's last operand is its state"),
            });
            let n_in = own.len() / n_groups as usize;
            let n_pure = mask.iter().filter(|&&p| p).count();
            // per argument: where the call's list holds it, else its place
            // among the prolog's
            let mut from: Vec<Result<usize, usize>> = Vec::with_capacity(n_args as usize);
            let mut rank = 0;
            let mut at_read: Vec<Option<usize>> = vec![None; n_args as usize];
            if let Some(r) = &reads {
                for (k, &q) in r.iter().enumerate() {
                    at_read[q as usize] = Some(k);
                }
            }
            for p in 0..n_args as usize {
                from.push(match (&reads, at_read[p]) {
                    (None, _) => Ok(p),
                    (Some(_), Some(k)) => Ok(k),
                    (Some(_), None) => Err(rank),
                });
                rank += usize::from(mask.get(p) == Some(&true));
            }
            let pro_ins: Option<Vec<Ref>> = prolog.map(|p| self.ins(p as usize).to_vec());
            let mut args: Vec<Ref> = Vec::with_capacity(n_groups as usize * n_args as usize);
            for g in 0..n_groups as usize {
                let list = &own[g * n_in..(g + 1) * n_in];
                for f in &from {
                    args.push(match *f {
                        Ok(k) => list[k],
                        Err(k) => {
                            let pro = pro_ins.as_ref().expect("an argument not read is pure");
                            pro[g * n_pure + k]
                        }
                    });
                }
            }
            if let Some(p) = prolog {
                dead[p as usize] = true;
            }
            dead[i as usize] = true;
            let pure = self.insts[i as usize].pure;
            let b = body.bundle.clone();
            let at_bundle = match self.bundles.iter().position(|x| Arc::ptr_eq(x, &b)) {
                Some(k) => k as u32,
                None => {
                    self.bundles.push(b.clone());
                    self.bundles.len() as u32 - 1
                }
            };
            let lists: Vec<&[Ref]> = args.chunks(n_args as usize).collect();
            let mask = b.pure_args();
            let stateful = !pure
                && pure_inputs.is_some()
                && b.state_len() > 0
                && mask.iter().any(|&p| p)
                && lists.iter().all(|a| {
                    a.len() == mask.len()
                        && a.iter().zip(mask).all(|(&r, &p)| !p || is_pure(self, r))
                });
            let reads: Option<Arc<[u32]>> = stateful
                .then(|| b.main_reads())
                .flatten()
                .filter(|r| r.len() < n_args as usize)
                .map(Into::into);
            let mut ins: Vec<Ref> = match &reads {
                None => args.clone(),
                Some(r) => (lists.iter())
                    .flat_map(|a| r.iter().map(move |&p| a[p as usize]))
                    .collect(),
            };
            if stateful {
                let pure_args: Vec<Ref> = (lists.iter())
                    .flat_map(|a| a.iter().zip(mask).filter(|&(_, &p)| p).map(|(&r, _)| r))
                    .collect();
                let kind = Kind::CallProlog {
                    bundle: at_bundle,
                    n_groups,
                    n_pure: mask.iter().filter(|&&p| p).count() as u32,
                };
                let p = self.push(kind, pure_args, n_groups * b.state_len() as u32, true);
                ins.push(Ref::Value(p, 0));
            }
            let width = b.n_outputs() as u32;
            let kind = Kind::Call {
                bundle: at_bundle,
                func,
                n_groups,
                n_args,
                reads,
                stateful,
            };
            let at = self.push(kind, ins, n_groups * width, pure);
            let slots: Vec<u32> = (outs.iter())
                .map(|&k| {
                    body.slot_of
                        .get(k as usize)
                        .copied()
                        .flatten()
                        .unwrap_or(u32::MAX)
                })
                .collect();
            moved.insert(i, (at, outs.len() as u32, slots));
        }
        if moved.is_empty() {
            return;
        }
        dead.resize(self.insts.len(), false);
        let width = |at: u32| -> u32 {
            match self.insts[at as usize].kind {
                Kind::Call { n_groups, .. } => self.insts[at as usize].n_out / n_groups,
                _ => unreachable!(),
            }
        };
        let widths: HashMap<u32, u32> = moved.values().map(|&(at, _, _)| (at, width(at))).collect();
        self.compact(&dead, |r| match r {
            Ref::Value(i, o) => match moved.get(&i) {
                Some((at, w_old, slots)) => {
                    let (g, s) = (o / w_old, o % w_old);
                    let slot = slots[s as usize];
                    debug_assert!(slot != u32::MAX, "a read output the body carries");
                    Ref::Value(*at, g * widths[at] + slot)
                }
                None => r,
            },
            r => r,
        });
    }
}

/// The last operand of instruction `i` of `p`.
fn own_last(p: &Program, i: u32) -> Ref {
    *p.ins(i as usize).last().expect("an operand")
}

impl Tape {
    /// The tape as the program it was emitted from: an instruction per op,
    /// each operand the value it reads (the op whose block held the slot at
    /// that point, and the offset in the block), the prolog's ops pure.
    /// Emitting it again gives this tape back up to slot numbering, so a
    /// transform of a compiled tape ([`Tape::specialize`]) works on its
    /// program and is scheduled and allocated like a fresh compilation.
    pub(super) fn lift(&self) -> Program {
        let m = self.ops.len();
        // The op whose block holds each slot, as the stream runs.
        let mut held = vec![u32::MAX; self.n_work];
        let value = |k: u32, held: &[u32]| -> Ref {
            match input_index(k) {
                Some(j) => Ref::Input(j),
                None => {
                    let j = held[k as usize];
                    Ref::Value(j, k - self.dst[j as usize])
                }
            }
        };
        // A fold that reads no operand reads the placeholder, appended
        // after the stream as instruction `m`.
        let placeholder = Ref::Value(m as u32, 0);
        let mut uses_placeholder = false;
        let mut insts: Vec<Inst> = Vec::with_capacity(m + 1);
        let mut pool: Vec<Ref> = Vec::with_capacity(self.arg_pool.len() + 2 * m);
        for i in 0..m {
            let start = pool.len();
            self.for_each_operand(i, |k| pool.push(value(k, &held)));
            let width = self.width(i);
            let codes = |acc: Option<Accum>| acc.map(|a| self.pool(a.codes, width).to_vec());
            let kind = match self.ops[i] {
                Op::Const(v) => Kind::Const(v),
                Op::Add(..) => Kind::Add,
                Op::Mul(..) => Kind::Mul,
                Op::MulAdd(..) => Kind::MulAdd,
                Op::Sub(..) => Kind::Sub,
                Op::Neg(_) => Kind::Neg,
                Op::Powi(_, n) => Kind::Powi(n),
                Op::Unary(op, _) => Kind::Unary(op),
                Op::Binary(op, ..) => Kind::Binary(op),
                Op::Cmp(op, ..) => Kind::Cmp(op),
                Op::Select(..) => Kind::Select,
                Op::Reduce(op, ..) => Kind::Reduce(op),
                Op::Dot(_, n) => Kind::Dot(n),
                Op::Call {
                    bundle,
                    n_groups,
                    n_args,
                    n_in,
                    reads,
                    state,
                    ..
                } => Kind::Call {
                    bundle,
                    func: NO_FUNC,
                    n_groups,
                    n_args,
                    reads: (reads != super::ALL_ARGS).then(|| self.pool(reads, n_in).into()),
                    stateful: state != super::NO_STATE,
                },
                Op::CallProlog {
                    bundle,
                    n_groups,
                    n_pure,
                    ..
                } => Kind::CallProlog {
                    bundle,
                    n_groups,
                    n_pure,
                },
                Op::Gemv {
                    m: rows, n, acc, ..
                } => Kind::Gemv {
                    m: rows,
                    n,
                    acc: codes(acc),
                },
                Op::Gemm {
                    m: rows, k, n, acc, ..
                } => Kind::Gemm {
                    m: rows,
                    k,
                    n,
                    acc: codes(acc),
                },
                Op::Solve { n, k, count, .. } => Kind::Solve { n, k, count },
            };
            // A kernel's accumulator entries, one per output after the
            // factors: the operand where its fold reads it, else the
            // placeholder.
            if let Op::Gemv { acc: Some(a), .. } | Op::Gemm { acc: Some(a), .. } = self.ops[i] {
                if a.c.is_none() {
                    pool.extend(std::iter::repeat_n(placeholder, width as usize));
                }
                let entries = pool.len() - width as usize;
                for (j, &code) in self.pool(a.codes, width).iter().enumerate() {
                    if !Fold(code).reads_operand() {
                        pool[entries + j] = placeholder;
                        uses_placeholder = true;
                    }
                }
            }
            insts.push(Inst {
                kind,
                ins: (start as u32, (pool.len() - start) as u32),
                n_out: width,
                pure: i < self.prolog_ops,
            });
            let d = self.dst[i];
            held[d as usize..(d + width) as usize].fill(i as u32);
        }
        if uses_placeholder {
            insts.push(Inst {
                kind: Kind::Const(f64::NAN),
                ins: (pool.len() as u32, 0),
                n_out: 1,
                pure: true,
            });
        }
        Program {
            insts,
            pool,
            bundles: self.bundles.clone(),
            roots: self.outputs.iter().map(|&k| value(k, &held)).collect(),
            placeholder: uses_placeholder.then_some(m as u32),
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

    /// The operands of the call of `f` at `site`: its arguments (and the
    /// bound expressions, in parameter order), then the globals of `f`.
    fn call_ops<'a, K: Field>(&'a self, ctx: &'a Graph<K>, f: u32, site: Site) -> &'a [ExprId] {
        match self.ext.get(&(f, site)) {
            Some(ops) => ops,
            None => ctx.args(site.1),
        }
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
        // the calls of one instance share their list: walked once
        let mut walked: HashSet<Site> = HashSet::default();
        // a call reads its function's globals besides its arguments, and
        // a bound call its context's expressions
        let mut ext: HashMap<(u32, Site), Vec<ExprId>> = HashMap::default();
        let mut funcs: HashSet<u32> = HashSet::default();
        while let Some(id) = stack.pop() {
            if !t.mark(id) {
                continue;
            }
            base.push(id);
            if let Node::Call(o, l) = *ctx.node(id) {
                let f = ctx.output(o).0;
                let site = (ctx.context_of(o), l);
                let globals = ctx.globals(f);
                if !globals.is_empty() || site.0 != NO_CONTEXT {
                    ext.entry((f.0, site)).or_insert_with(|| {
                        let mut ops = ctx.full_args(o, l).into_owned();
                        ops.extend_from_slice(&globals);
                        ops
                    });
                    if funcs.insert(f.0) {
                        stack.extend_from_slice(&globals);
                    }
                }
                if !walked.insert(site) {
                    continue;
                }
            }
            stack.extend_from_slice(&ctx.operands(id));
        }
        let call_ops = |f: u32, site: Site| -> &[ExprId] {
            match ext.get(&(f, site)) {
                Some(ops) => ops,
                None => ctx.args(site.1),
            }
        };
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
            let mut pure_list: HashMap<(u32, Site), bool> = HashMap::default();
            for (i, id) in base.iter().enumerate() {
                pure[i] = match *ctx.node(*id) {
                    Node::Const(_) => true,
                    Node::Symbol(s) => match t.input(s) {
                        None => true,
                        Some(k) => mask.get(k as usize).copied().unwrap_or(false),
                    },
                    Node::Call(o, l) => {
                        let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                        *pure_list
                            .entry((f, site))
                            .or_insert_with(|| call_ops(f, site).iter().all(|a| pure[bp(*a)]))
                    }
                    _ => ctx.operands(*id).iter().all(|a| pure[bp(*a)]),
                };
            }
        }
        let mut is_root = vec![false; m];
        for r in roots {
            is_root[bp(*r)] = true;
        }
        let mut uses = vec![0u32; m];
        walked.clear();
        for id in &base {
            if let Node::Call(o, l) = *ctx.node(*id) {
                if !walked.insert((ctx.context_of(o), l)) {
                    continue;
                }
            }
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
            ext,
        }
    }

    /// Pass 2: the instructions, in three steps: the kernel groups
    /// ([`Forest::groups`]), an order of units, each a node or a whole group,
    /// after its operands ([`Forest::unit_order`]), and the instructions in
    /// that order.
    fn lower<K: Field>(
        &self,
        ctx: &Graph<K>,
        roots: &[ExprId],
        templates: &mut Templates,
    ) -> Program {
        let calls = self.call_sets(ctx);
        // the functions whose calls expand a template rather than batch
        let mut composite: HashSet<u32> = HashSet::default();
        for &id in &self.base {
            if let Node::Call(o, _) = *ctx.node(id) {
                let f = ctx.output(o).0;
                if ctx.func(f).composite_in(ctx) {
                    composite.insert(f.0);
                }
            }
        }
        let groups = self.groups(ctx, &calls, &composite);
        let order = self.unit_order(ctx, roots, &groups);
        let m = self.base.len();
        let mut lw = Lowering {
            p: Program {
                insts: Vec::with_capacity(m),
                pool: Vec::with_capacity(2 * m),
                bundles: Vec::new(),
                roots: Vec::new(),
                placeholder: None,
            },
            value: vec![None; m],
            consts: HashMap::default(),
        };
        let mut bodies = Bodies::default();
        // The nodes of one call (same function, same arguments): a single
        // call is one instruction whichever of its outputs is reached first.
        let mut call_sites: HashMap<(u32, Site), Vec<usize>> = HashMap::default();
        for (i, &id) in self.base.iter().enumerate() {
            if let Node::Call(o, l) = *ctx.node(id) {
                call_sites
                    .entry((ctx.output(o).0 .0, (ctx.context_of(o), l)))
                    .or_default()
                    .push(i);
            }
        }
        let mut lowered = vec![false; groups.groups.len()];
        for &i in &order {
            // Materialized inside its consuming Add, or an output of a call
            // or a kernel lowered earlier.
            if self.fused_into[i].is_some() || lw.value[i].is_some() {
                continue;
            }
            match (groups.kernel_of[i], *ctx.node(self.base[i])) {
                (Some(g), _) => {
                    if !std::mem::replace(&mut lowered[g], true) {
                        let (key, members) = &groups.groups[g];
                        self.lower_group(ctx, &mut lw, &mut bodies, &calls, key, members);
                    }
                }
                (None, Node::Call(o, l)) => {
                    let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                    let members = &call_sites[&(f, site)];
                    let set = calls.set_of[&(f, site)];
                    if composite.contains(&f) {
                        let outs = &calls.sets[set as usize];
                        let ops = self.call_ops(ctx, f, site);
                        let args: Vec<Ref> = ops.iter().map(|&a| self.val(&lw, a)).collect();
                        let mask: Option<Vec<bool>> = self
                            .split
                            .then(|| ops.iter().map(|&a| self.pure[self.pos(a)]).collect());
                        let t = templates.get(ctx, f, outs, mask);
                        let values = lw.expand(&mut bodies, &t, &args);
                        for &mi in members {
                            let Node::Call(o, _) = *ctx.node(self.base[mi]) else {
                                unreachable!()
                            };
                            let k = ctx.output(o).1;
                            lw.value[mi] =
                                Some(values[outs.binary_search(&k).expect("in its set")]);
                        }
                        continue;
                    }
                    self.lower_calls(ctx, &mut lw, &mut bodies, &calls, f, set, members);
                }
                (None, _) => self.lower_node(ctx, &mut lw, i),
            }
        }
        lw.p.roots = roots.iter().map(|&r| self.val(&lw, r)).collect();
        lw.p
    }

    /// The value node `e` was lowered to.
    fn val(&self, lw: &Lowering, e: ExprId) -> Ref {
        lw.value[self.pos(e)].expect("an operand precedes its consumer")
    }

    /// The outputs each call (function and argument list) is made for: what
    /// a body must cover to serve it. Calls of one function made for
    /// different output sets (a residual alone, the residual with its
    /// partials) take different bodies, so they are keyed by the set.
    fn call_sets<K: Field>(&self, ctx: &Graph<K>) -> CallSets {
        let mut per_call: HashMap<(u32, Site), Vec<u32>> = HashMap::default();
        for &id in &self.base {
            if let Node::Call(o, l) = *ctx.node(id) {
                let (f, out) = ctx.output(o);
                per_call
                    .entry((f.0, (ctx.context_of(o), l)))
                    .or_default()
                    .push(out);
            }
        }
        let mut ids: HashMap<(u32, Vec<u32>), u32> = HashMap::default();
        let mut calls = CallSets::default();
        for ((f, site), mut outs) in per_call {
            outs.sort_unstable();
            outs.dedup();
            let id = *ids.entry((f, outs.clone())).or_insert_with(|| {
                calls.sets.push(outs);
                calls.sets.len() as u32 - 1
            });
            calls.set_of.insert((f, site), id);
        }
        calls
    }

    /// The kernel groups: calls of one function with distinct argument
    /// lists, row dots by their vector, solve components by their list.
    /// Members of one group share a call depth: a member's depth is one more
    /// than the deepest member among its operands, so equal depth means no
    /// member depends on another's output, and the group's operands all
    /// precede it. Gemv groups over the same rows become one Gemm, solves
    /// over one matrix one solve of several right-hand sides.
    fn groups<K: Field>(
        &self,
        ctx: &Graph<K>,
        calls: &CallSets,
        composite: &HashSet<u32>,
    ) -> Groups {
        let base = &self.base;
        let m = base.len();
        // The functions called with two or more argument lists of one length.
        let mut lists: HashMap<u32, Vec<Site>> = HashMap::default();
        let mut seen: HashSet<(u32, Site)> = HashSet::default();
        for &id in base {
            if let Node::Call(o, l) = *ctx.node(id) {
                let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                if seen.insert((f, site)) {
                    lists.entry(f).or_default().push(site);
                }
            }
        }
        lists.retain(|&f, g| {
            let width = |&s: &Site| self.call_ops(ctx, f, s).len();
            !composite.contains(&f) && g.len() >= 2 && g.iter().all(|a| width(a) == width(&g[0]))
        });
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
        let mut list_depth: HashMap<(u32, Site), u32> = HashMap::default();
        for (i, &id) in base.iter().enumerate() {
            let deepest = |depth: &[u32], ops: &[ExprId]| {
                ops.iter().map(|&a| depth[self.pos(a)]).max().unwrap_or(0)
            };
            let over = match *ctx.node(id) {
                Node::Call(o, l) => {
                    let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                    *list_depth
                        .entry((f, site))
                        .or_insert_with(|| deepest(&depth, self.call_ops(ctx, f, site)))
                }
                _ => deepest(&depth, &ctx.operands(id)),
            };
            depth[i] = over + u32::from(is_member[i]);
        }
        // Groups by first encounter.
        let mut index: HashMap<GroupKey, usize> = HashMap::default();
        let mut groups: Vec<(GroupKey, Vec<usize>)> = Vec::new();
        for (i, &id) in base.iter().enumerate() {
            if !is_member[i] {
                continue;
            }
            let key = match *ctx.node(id) {
                Node::Call(o, l) => {
                    let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                    GroupKey::Call(f, depth[i], self.pure[i], calls.set_of[&(f, site)])
                }
                Node::Dot(l) => GroupKey::Gemv(ctx.dot_args(l).1.to_vec(), depth[i], self.pure[i]),
                Node::Solve(l, _) => GroupKey::Solve(l, self.pure[i]),
                _ => unreachable!("members are calls, rows or components"),
            };
            let g = *index.entry(key.clone()).or_insert_with(|| {
                groups.push((key, Vec::new()));
                groups.len() - 1
            });
            groups[g].1.push(i);
        }
        self.merge_gemm(ctx, &mut groups);
        self.merge_solves(ctx, &mut groups, &depth);
        self.merge_solve_batches(ctx, &mut groups, &depth);
        // A gemv group of too few rows at its depth, or a call group of one
        // argument list, is no kernel: its members lower on their own.
        let mut kernel_of: Vec<Option<usize>> = vec![None; m];
        for (g, (key, members)) in groups.iter().enumerate() {
            let is_kernel = match key {
                GroupKey::Gemv(..) => members.len() >= GEMV_MIN_ROWS,
                GroupKey::Call(..) => {
                    let mut distinct: HashSet<Site> = HashSet::default();
                    for &i in members {
                        if let Node::Call(o, l) = *ctx.node(base[i]) {
                            distinct.insert((ctx.context_of(o), l));
                        }
                    }
                    distinct.len() >= 2
                }
                GroupKey::Gemm(..)
                | GroupKey::Solve(..)
                | GroupKey::SolveMany(..)
                | GroupKey::SolveBatch(..) => true,
            };
            if is_kernel {
                for &i in members {
                    kernel_of[i] = Some(g);
                }
            }
        }
        Groups { groups, kernel_of }
    }

    /// Gemv groups over the same rows at one depth are one Gemm: the rows
    /// against every vector at once. A merged-away group is left empty; no
    /// member points at it. Row sets are compared by fingerprint (one hash
    /// per row, sorted), and the candidate groups' rows verified before they
    /// merge, so a group of long rows costs one pass over its entries.
    fn merge_gemm<K: Field>(&self, ctx: &Graph<K>, groups: &mut [(GroupKey, Vec<usize>)]) {
        let row_of = |mi: usize| -> &[ExprId] {
            let Node::Dot(l) = *ctx.node(self.base[mi]) else {
                unreachable!()
            };
            ctx.dot_args(l).0
        };
        let row_hash = |mi: usize| -> u64 {
            use std::hash::{Hash, Hasher};
            let mut h = rustc_hash::FxHasher::default();
            row_of(mi).hash(&mut h);
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
        let rows = |g: usize, groups: &[(GroupKey, Vec<usize>)]| -> Vec<Vec<ExprId>> {
            let mut r: Vec<Vec<ExprId>> =
                groups[g].1.iter().map(|&mi| row_of(mi).to_vec()).collect();
            r.sort();
            r
        };
        for gs in merged {
            // The rows of the first group, and the check that every other
            // group has exactly them.
            let first = rows(gs[0], groups);
            if gs[1..].iter().any(|&g| rows(g, groups) != first) {
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
                .flat_map(|&g| std::mem::take(&mut groups[g].1))
                .collect();
            groups[gs[0]] = (GroupKey::Gemm(first, xs), members);
        }
    }

    /// Solve groups over one matrix at one depth are one solve of several
    /// right-hand sides: the factorization once. Equal depth means no
    /// right-hand side depends on another group's solution (a derivative's
    /// solve reads the primal solution over the same matrix). A merged-away
    /// group is left empty.
    fn merge_solves<K: Field>(
        &self,
        ctx: &Graph<K>,
        groups: &mut [(GroupKey, Vec<usize>)],
        depth: &[u32],
    ) {
        let mut by_matrix: HashMap<(Vec<ExprId>, u32, bool), Vec<usize>> = HashMap::default();
        for (g, (key, members)) in groups.iter().enumerate() {
            if let GroupKey::Solve(l, pure) = key {
                let (n, all) = (Graph::<K>::solve_n(l.len()), ctx.args(*l));
                by_matrix
                    .entry((all[..n * n].to_vec(), depth[members[0]], *pure))
                    .or_default()
                    .push(g);
            }
        }
        let mut merged: Vec<(Vec<usize>, Vec<ExprId>)> = by_matrix
            .into_iter()
            .filter(|(_, gs)| gs.len() >= 2)
            .map(|((a, _, _), gs)| (gs, a))
            .collect();
        merged.sort();
        for (gs, a) in merged {
            let lists: Vec<ArgList> = gs
                .iter()
                .map(|&g| match &groups[g].0 {
                    GroupKey::Solve(l, _) => *l,
                    _ => unreachable!(),
                })
                .collect();
            let members: Vec<usize> = gs
                .iter()
                .flat_map(|&g| std::mem::take(&mut groups[g].1))
                .collect();
            groups[gs[0]] = (GroupKey::SolveMany(a, lists), members);
        }
    }

    /// Solves of one shape (unknowns and right-hand sides) at one depth and
    /// of one purity, over different matrices, are one batch: their
    /// systems side by side in one kernel, four to a vector; only shapes
    /// up to [`SOLVE_BATCH_MAX_N`] unknowns and [`SOLVE_BATCH_MAX_K`]
    /// right-hand sides batch (a larger system gains nothing from it and
    /// the batch gathers its operands). At one depth none reads another's solution (a
    /// group's depth is above every group it reads). A merged-away group is
    /// left empty.
    fn merge_solve_batches<K: Field>(
        &self,
        ctx: &Graph<K>,
        groups: &mut [(GroupKey, Vec<usize>)],
        depth: &[u32],
    ) {
        let system = |key: &GroupKey| -> Option<System> {
            match key {
                GroupKey::Solve(l, _) => {
                    let n = Graph::<K>::solve_n(l.len());
                    Some((ctx.args(*l)[..n * n].to_vec(), vec![*l]))
                }
                GroupKey::SolveMany(a, lists) => Some((a.clone(), lists.clone())),
                _ => None,
            }
        };
        let mut by_shape: HashMap<(usize, usize, u32, bool), Vec<usize>> = HashMap::default();
        for (g, (key, members)) in groups.iter().enumerate() {
            let (Some((a, lists)), Some(&first)) = (system(key), members.first()) else {
                continue;
            };
            // `a` holds the n by n entries.
            if a.len() > SOLVE_BATCH_MAX_N * SOLVE_BATCH_MAX_N || lists.len() > SOLVE_BATCH_MAX_K {
                continue;
            }
            by_shape
                .entry((a.len(), lists.len(), depth[first], self.pure[first]))
                .or_default()
                .push(g);
        }
        let mut batches: Vec<Vec<usize>> =
            by_shape.into_values().filter(|gs| gs.len() >= 2).collect();
        batches.sort();
        for gs in batches {
            let systems: Vec<System> = gs.iter().filter_map(|&g| system(&groups[g].0)).collect();
            let members: Vec<usize> = gs
                .iter()
                .flat_map(|&g| std::mem::take(&mut groups[g].1))
                .collect();
            groups[gs[0]] = (GroupKey::SolveBatch(systems), members);
        }
    }

    /// Every unit after its operands: a unit is a node, or the group it
    /// belongs to (by its first member), whose operands are all its members'
    /// operands; a fused operand is no unit, its consumer reads its
    /// operands. A depth-first walk from the roots, post-order, on an
    /// explicit stack.
    fn unit_order<K: Field>(
        &self,
        ctx: &Graph<K>,
        roots: &[ExprId],
        groups: &Groups,
    ) -> Vec<usize> {
        let m = self.base.len();
        let (groups, kernel_of) = (&groups.groups, &groups.kernel_of);
        let first_member: Vec<usize> = groups
            .iter()
            .map(|(_, ms)| ms.first().copied().unwrap_or(0))
            .collect();
        // The calls of one instance are one instruction: one unit, by the
        // first of them, over the one list.
        let mut site_first: HashMap<(u32, Site), usize> = HashMap::default();
        let mut site_of: Vec<usize> = (0..m).collect();
        for (i, &id) in self.base.iter().enumerate() {
            if let Node::Call(o, l) = *ctx.node(id) {
                let key = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                site_of[i] = *site_first.entry(key).or_insert(i);
            }
        }
        let unit_of = |i: usize| -> usize {
            match kernel_of[i] {
                Some(g) => first_member[g],
                None => site_of[i],
            }
        };
        let deps_of_node = |i: usize, out: &mut Vec<usize>| {
            if let Node::Call(o, l) = *ctx.node(self.base[i]) {
                let ops = self.call_ops(ctx, ctx.output(o).0 .0, (ctx.context_of(o), l));
                out.extend(ops.iter().map(|&a| unit_of(self.pos(a))));
                return;
            }
            for &a in ctx.operands(self.base[i]).iter() {
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
        // over thousands of rows names its operands hundreds of thousands of
        // times. One buffer for every unit's raw operands and one for the
        // distinct ones, reused: a million units are not two million
        // allocations.
        let mut stamp: Vec<usize> = vec![usize::MAX; m];
        let mut lists: HashSet<Site> = HashSet::default();
        let mut raw: Vec<usize> = Vec::new();
        let mut deps: Vec<usize> = Vec::new();
        let mut deps_of_unit = |u: usize, deps: &mut Vec<usize>| {
            raw.clear();
            match kernel_of[u] {
                Some(g) => {
                    // a member's list once: the instances of the group
                    lists.clear();
                    for &mi in &groups[g].1 {
                        if let Node::Call(o, l) = *ctx.node(self.base[mi]) {
                            if !lists.insert((ctx.context_of(o), l)) {
                                continue;
                            }
                        }
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
        order
    }

    /// The instruction of a kernel group.
    fn lower_group<K: Field>(
        &self,
        ctx: &Graph<K>,
        lw: &mut Lowering,
        bodies: &mut Bodies,
        calls: &CallSets,
        key: &GroupKey,
        members: &[usize],
    ) {
        let base = &self.base;
        let pure = members.iter().all(|&mi| self.pure[mi]);
        let val = |lw: &Lowering, e: ExprId| self.val(lw, e);
        match key {
            GroupKey::Call(f, _, _, set) => {
                self.lower_calls(ctx, lw, bodies, calls, *f, *set, members)
            }
            GroupKey::Gemv(x, ..) => {
                let mut ins: Vec<Ref> = Vec::new();
                for &mi in members {
                    let Node::Dot(l) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    ins.extend(ctx.dot_args(l).0.iter().map(|&a| val(lw, a)));
                }
                ins.extend(x.iter().map(|&a| val(lw, a)));
                let kind = Kind::Gemv {
                    m: members.len() as u32,
                    n: x.len() as u32,
                    acc: None,
                };
                let inst = lw.push(kind, ins, members.len() as u32, pure);
                for (r, &mi) in members.iter().enumerate() {
                    lw.value[mi] = Some(Ref::Value(inst, r as u32));
                }
            }
            GroupKey::Gemm(rows, xs) => {
                let (rm, k, cn) = (rows.len(), xs[0].len(), xs.len());
                let ins: Vec<Ref> = rows
                    .iter()
                    .chain(xs)
                    .flat_map(|r| r.iter().map(|&a| val(lw, a)))
                    .collect();
                fn index(set: &[Vec<ExprId>]) -> HashMap<&[ExprId], usize> {
                    set.iter()
                        .enumerate()
                        .map(|(i, r)| (r.as_slice(), i))
                        .collect()
                }
                let (row_index, col_index) = (index(rows), index(xs));
                let kind = Kind::Gemm {
                    m: rm as u32,
                    k: k as u32,
                    n: cn as u32,
                    acc: None,
                };
                let inst = lw.push(kind, ins, (rm * cn) as u32, pure);
                for &mi in members {
                    let Node::Dot(l) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    let (a, x) = ctx.dot_args(l);
                    let (r, c) = (row_index[a], col_index[x]);
                    lw.value[mi] = Some(Ref::Value(inst, (r * cn + c) as u32));
                }
            }
            GroupKey::SolveMany(a, lists) => {
                let n = (a.len() as f64).sqrt() as u32;
                let k = lists.len() as u32;
                let mut ins: Vec<Ref> = a.iter().map(|&e| val(lw, e)).collect();
                for &l in lists {
                    ins.extend(ctx.args(l)[(n * n) as usize..].iter().map(|&e| val(lw, e)));
                }
                let col_of: HashMap<ArgList, u32> = lists
                    .iter()
                    .enumerate()
                    .map(|(c, &l)| (l, c as u32))
                    .collect();
                let inst = lw.push(Kind::Solve { n, k, count: 1 }, ins, n * k, pure);
                for &mi in members {
                    let Node::Solve(l, c) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    lw.value[mi] = Some(Ref::Value(inst, col_of[&l] * n + c));
                }
            }
            GroupKey::SolveBatch(systems) => {
                let n = Graph::<K>::solve_n(ctx.args(systems[0].1[0]).len());
                let k = systems[0].1.len();
                let mut ins: Vec<Ref> = systems
                    .iter()
                    .flat_map(|(a, _)| a.iter().map(|&e| val(lw, e)))
                    .collect();
                // Each right-hand side's place: system, then its column.
                let mut col_of: HashMap<ArgList, usize> = HashMap::default();
                for (l, &list) in systems.iter().flat_map(|(_, lists)| lists).enumerate() {
                    ins.extend(ctx.args(list)[n * n..].iter().map(|&e| val(lw, e)));
                    col_of.insert(list, l);
                }
                let count = systems.len();
                let kind = Kind::Solve {
                    n: n as u32,
                    k: k as u32,
                    count: count as u32,
                };
                let inst = lw.push(kind, ins, (count * n * k) as u32, pure);
                for &mi in members {
                    let Node::Solve(l, c) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    lw.value[mi] = Some(Ref::Value(inst, (col_of[&l] * n) as u32 + c));
                }
            }
            GroupKey::Solve(l, _) => {
                let all = ctx.args(*l);
                let n = Graph::<K>::solve_n(all.len()) as u32;
                let ins: Vec<Ref> = all.iter().map(|&a| val(lw, a)).collect();
                let inst = lw.push(Kind::Solve { n, k: 1, count: 1 }, ins, n, pure);
                for &mi in members {
                    let Node::Solve(_, c) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    lw.value[mi] = Some(Ref::Value(inst, c));
                }
            }
        }
    }

    /// Calls of `f` made for the output set `set`, the nodes `members`: one
    /// instruction over their distinct argument lists in first-encounter
    /// order, each member a value of the group its list is, or zero where
    /// the body does not carry the output. A call that keeps state gets its
    /// instances' prolog first.
    #[allow(clippy::too_many_arguments)]
    fn lower_calls<K: Field>(
        &self,
        ctx: &Graph<K>,
        lw: &mut Lowering,
        bodies: &mut Bodies,
        calls: &CallSets,
        f: u32,
        set: u32,
        members: &[usize],
    ) {
        let (bundle, body) = bodies.get(ctx, &mut lw.p.bundles, f, set, &calls.sets);
        let b = body.bundle.clone();
        let n_out = b.n_outputs() as u32;
        let mut lists: Vec<Site> = Vec::new();
        let mut group_of: HashMap<Site, u32> = HashMap::default();
        for &mi in members {
            let Node::Call(o, l) = *ctx.node(self.base[mi]) else {
                unreachable!()
            };
            let site = (ctx.context_of(o), l);
            group_of.entry(site).or_insert_with(|| {
                lists.push(site);
                lists.len() as u32 - 1
            });
        }
        let args: Vec<&[ExprId]> = lists.iter().map(|&l| self.call_ops(ctx, f, l)).collect();
        let (n_groups, n_args) = (lists.len() as u32, args[0].len() as u32);
        let pure = members.iter().all(|&mi| self.pure[mi]);
        let stateful = !pure && self.stateful(&*b, &args);
        // After its prolog a body reads what its main phase reads only: the
        // others (a model card bound to every instance) are not gathered
        // per evaluation.
        let reads: Option<Arc<[u32]>> = stateful
            .then(|| b.main_reads())
            .flatten()
            .filter(|r| r.len() < n_args as usize)
            .map(Into::into);
        let mut ins: Vec<Ref> = match &reads {
            None => args
                .iter()
                .flat_map(|a| a.iter())
                .map(|&a| self.val(lw, a))
                .collect(),
            Some(r) => args
                .iter()
                .flat_map(|a| r.iter().map(move |&p| a[p as usize]))
                .map(|a| self.val(lw, a))
                .collect(),
        };
        if stateful {
            let mask = b.pure_args();
            let pure_args: Vec<Ref> = args
                .iter()
                .flat_map(|a| a.iter().zip(mask).filter(|&(_, &p)| p))
                .map(|(&a, _)| self.val(lw, a))
                .collect();
            let n_pure = mask.iter().filter(|&&p| p).count() as u32;
            let kind = Kind::CallProlog {
                bundle,
                n_groups,
                n_pure,
            };
            let prolog = lw.push(kind, pure_args, n_groups * b.state_len() as u32, true);
            ins.push(Ref::Value(prolog, 0));
        }
        let kind = Kind::Call {
            bundle,
            func: f,
            n_groups,
            n_args,
            reads,
            stateful,
        };
        let inst = lw.push(kind, ins, n_groups * n_out, pure);
        for &mi in members {
            let Node::Call(o, l) = *ctx.node(self.base[mi]) else {
                unreachable!()
            };
            let slot = body.slot_of[ctx.output(o).1 as usize];
            lw.value[mi] = Some(match slot {
                Some(slot) => Ref::Value(inst, group_of[&(ctx.context_of(o), l)] * n_out + slot),
                // A zero output: a derivative the body does not carry.
                None => lw.constant(0.0),
            });
        }
    }

    /// The instruction of node `i` that is no kernel member: a leaf (an
    /// input is an operand, a symbol without one NaN, a constant an
    /// instruction), or one instruction, an Add with a fused operand reading
    /// through it.
    fn lower_node<K: Field>(&self, ctx: &Graph<K>, lw: &mut Lowering, i: usize) {
        let val = |lw: &Lowering, e: ExprId| self.val(lw, e);
        let (kind, ins): (Kind, Vec<Ref>) = match *ctx.node(self.base[i]) {
            Node::Symbol(s) => {
                lw.value[i] = Some(match self.input(s) {
                    Some(k) => Ref::Input(k),
                    None => lw.constant(f64::NAN),
                });
                return;
            }
            Node::Const(c) => {
                lw.value[i] = Some(lw.constant(ctx.const_val(c).to_f64()));
                return;
            }
            Node::Add(a, b) => {
                let fused = |x: ExprId| self.fused_into[self.pos(x)] == Some(i);
                match (fused(a), fused(b)) {
                    (false, false) => (Kind::Add, vec![val(lw, a), val(lw, b)]),
                    (fa, _) => {
                        let (f, other) = if fa { (a, b) } else { (b, a) };
                        match *ctx.node(f) {
                            Node::Mul(x, y) => {
                                (Kind::MulAdd, vec![val(lw, x), val(lw, y), val(lw, other)])
                            }
                            Node::Neg(x) => (Kind::Sub, vec![val(lw, other), val(lw, x)]),
                            _ => unreachable!("only a Mul or a Neg fuses"),
                        }
                    }
                }
            }
            Node::Mul(a, b) => (Kind::Mul, vec![val(lw, a), val(lw, b)]),
            Node::Neg(a) => (Kind::Neg, vec![val(lw, a)]),
            Node::Pow(a, n) => (Kind::Powi(n as i32), vec![val(lw, a)]),
            Node::Unary(op, a) => (Kind::Unary(op), vec![val(lw, a)]),
            Node::Binary(op, a, b) => (Kind::Binary(op), vec![val(lw, a), val(lw, b)]),
            Node::Cmp(op, a, b) => (Kind::Cmp(op), vec![val(lw, a), val(lw, b)]),
            Node::Select(c, t, e) => (Kind::Select, vec![val(lw, c), val(lw, t), val(lw, e)]),
            Node::Reduce(op, l) => (
                Kind::Reduce(op),
                ctx.args(l).iter().map(|&a| val(lw, a)).collect(),
            ),
            Node::Dot(l) => (
                Kind::Dot(l.len() as u32 / 2),
                ctx.args(l).iter().map(|&a| val(lw, a)).collect(),
            ),
            Node::Call(..) | Node::Solve(..) => unreachable!("calls and solves lower as groups"),
        };
        let inst = lw.push(kind, ins, 1, self.pure[i]);
        lw.value[i] = Some(Ref::Value(inst, 0));
    }
}

/// The output sets calls are made for (see [`Forest::call_sets`]).
#[derive(Default)]
struct CallSets {
    /// Per function and argument list, the set of outputs called.
    set_of: HashMap<(u32, Site), u32>,
    sets: Vec<Vec<u32>>,
}

/// What makes the members of a group one kernel. Every key carries the
/// members' purity: a kernel of pure and impure members would be impure as
/// a whole and pull the pure work out of the prolog.
#[derive(PartialEq, Eq, Hash, Clone)]
enum GroupKey {
    /// Function, depth, purity, the output set called for.
    Call(u32, u32, bool, u32),
    Gemv(Vec<ExprId>, u32, bool),
    /// The rows (sorted) against every vector: a matrix product.
    Gemm(Vec<Vec<ExprId>>, Vec<Vec<ExprId>>),
    Solve(ArgList, bool),
    /// One matrix against several right-hand sides (their lists, in
    /// first-encounter order).
    SolveMany(Vec<ExprId>, Vec<ArgList>),
    /// Systems of one shape over different matrices, side by side.
    SolveBatch(Vec<System>),
}

/// A system of a solve kernel: its matrix and its right-hand sides' lists.
type System = (Vec<ExprId>, Vec<ArgList>);

/// The kernel groups of a forest (see [`Forest::groups`]): members by base
/// position, and the group, if a kernel, each position belongs to.
struct Groups {
    groups: Vec<(GroupKey, Vec<usize>)>,
    kernel_of: Vec<Option<usize>>,
}

/// A program as [`Forest::lower`] builds it, and the value each base node
/// is once lowered.
struct Lowering {
    p: Program,
    value: Vec<Option<Ref>>,
    /// The constants the expanded templates share, by value.
    consts: HashMap<u64, Ref>,
}

impl Lowering {
    /// Append an instruction over the operands `ins`; its index.
    fn push(
        &mut self,
        kind: Kind,
        ins: impl IntoIterator<Item = Ref>,
        n_out: u32,
        pure: bool,
    ) -> u32 {
        let start = self.p.pool.len() as u32;
        self.p.pool.extend(ins);
        self.p.insts.push(Inst {
            kind,
            ins: (start, self.p.pool.len() as u32 - start),
            n_out,
            pure,
        });
        self.p.insts.len() as u32 - 1
    }

    /// A constant instruction's value.
    fn constant(&mut self, v: f64) -> Ref {
        Ref::Value(self.push(Kind::Const(v), [], 1, true), 0)
    }

    /// Template `t` appended over the operands `args`: its instructions
    /// with its inputs the operands, its bundles interned here. The values
    /// of its roots.
    fn expand(&mut self, bodies: &mut Bodies, t: &Program, args: &[Ref]) -> Vec<Ref> {
        let bundles: Vec<u32> = t
            .bundles
            .iter()
            .map(|b| bodies.intern(&mut self.p.bundles, b))
            .collect();
        // a template's constant is the program's one of that value
        let mut at: Vec<Option<Ref>> = vec![None; t.insts.len()];
        for (i, inst) in t.insts.iter().enumerate() {
            if let Kind::Const(v) = inst.kind {
                at[i] = Some(*self.consts.entry(v.to_bits()).or_insert_with(|| {
                    let start = self.p.pool.len() as u32;
                    self.p.insts.push(Inst {
                        kind: Kind::Const(v),
                        ins: (start, 0),
                        n_out: 1,
                        pure: true,
                    });
                    Ref::Value(self.p.insts.len() as u32 - 1, 0)
                }));
            }
        }
        let base = self.p.insts.len() as u32;
        let mut next = base;
        let mut index = vec![0u32; t.insts.len()];
        for (i, slot) in at.iter().enumerate() {
            if slot.is_none() {
                index[i] = next;
                next += 1;
            }
        }
        let map = |r: Ref| match r {
            Ref::Input(k) => args[k as usize],
            Ref::Value(i, o) => at[i as usize].unwrap_or(Ref::Value(index[i as usize], o)),
        };
        for (i, inst) in t.insts.iter().enumerate() {
            if at[i].is_some() {
                continue;
            }
            let mut kind = inst.kind.clone();
            if let Kind::Call { bundle, .. } | Kind::CallProlog { bundle, .. } = &mut kind {
                *bundle = bundles[*bundle as usize];
            }
            let ins = t.ins(i).iter().map(|&r| map(r));
            self.push(kind, ins, inst.n_out, inst.pure);
        }
        t.roots.iter().map(|&r| map(r)).collect()
    }
}

/// The templates of a compilation: per composite function, output set and
/// purity of its operands (under a split), its body lowered once over its
/// operands (its parameters, then its globals), the composite calls in it
/// expanded in turn.
#[derive(Default)]
struct Templates {
    /// A template was expanded: the leaf calls of its instances are apart.
    expanded: bool,
}

impl Templates {
    /// The template of `f` for the outputs `outs` (ascending), its operands
    /// pure as `mask` says (`None` without a split).
    fn get<K: Field>(
        &mut self,
        ctx: &Graph<K>,
        f: u32,
        outs: &[u32],
        mask: Option<Vec<bool>>,
    ) -> Arc<Program> {
        self.expanded = true;
        let key = (f, outs.to_vec(), mask);
        // kept by the graph, built outside its lock (a template expands others)
        // kept by the graph, built outside its lock (a template expands others)
        let cached = ctx.templates.lock().unwrap().get(&key).cloned();
        if let Some(t) = cached {
            return t.downcast().expect("a template is a program");
        }
        let func = ctx.func(FuncId(f));
        let mut syms: Vec<SymbolId> = func.params().to_vec();
        syms.extend(
            ctx.globals_of(FuncId(f))
                .iter()
                .map(|&e| match *ctx.node(e) {
                    Node::Symbol(s) => s,
                    _ => unreachable!("a global is a symbol"),
                }),
        );
        let exprs: Vec<ExprId> = outs
            .iter()
            .filter_map(|&k| match func.outputs()[k as usize] {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        let forest = Forest::analyze(ctx, &exprs, &syms, key.2.as_deref());
        let mut p = forest.lower(ctx, &exprs, self);
        drop(forest);
        // a zero output is a constant
        let mut roots = std::mem::take(&mut p.roots).into_iter();
        let mut lw = Lowering {
            p,
            value: Vec::new(),
            consts: HashMap::default(),
        };
        let all: Vec<Ref> = outs
            .iter()
            .map(|&k| match func.outputs()[k as usize] {
                Output::Expr(_) => roots.next().expect("one per expression"),
                _ => lw.constant(0.0),
            })
            .collect();
        lw.p.roots = all;
        let t = Arc::new(lw.p);
        ctx.templates.lock().unwrap().insert(key, t.clone());
        t
    }
}

/// The bodies serving the calls, by function and output set, with their
/// bundles' indices in the program.
#[derive(Default)]
struct Bodies {
    of: HashMap<(u32, u32), (u32, Body)>,
    bundle_idx: HashMap<usize, u32>,
}

impl Bodies {
    /// The index of `bundle` in `bundles`, appended if new.
    fn intern(
        &mut self,
        bundles: &mut Vec<Arc<dyn ExternBundle>>,
        bundle: &Arc<dyn ExternBundle>,
    ) -> u32 {
        let ptr = Arc::as_ptr(bundle) as *const () as usize;
        *self.bundle_idx.entry(ptr).or_insert_with(|| {
            bundles.push(bundle.clone());
            bundles.len() as u32 - 1
        })
    }

    /// The body of `f` for the output set `set`, compiled on first demand,
    /// its bundle interned into `bundles`.
    fn get<K: Field>(
        &mut self,
        ctx: &Graph<K>,
        bundles: &mut Vec<Arc<dyn ExternBundle>>,
        f: u32,
        set: u32,
        sets: &[Vec<u32>],
    ) -> (u32, &Body) {
        let bundle_idx = &mut self.bundle_idx;
        let (b, body) = self.of.entry((f, set)).or_insert_with(|| {
            let body = ctx.func(FuncId(f)).body_for(ctx, &sets[set as usize]);
            let ptr = Arc::as_ptr(&body.bundle) as *const () as usize;
            let b = *bundle_idx.entry(ptr).or_insert_with(|| {
                bundles.push(body.bundle.clone());
                bundles.len() as u32 - 1
            });
            (b, body)
        });
        (*b, body)
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
    pub(super) fn schedule(&self) -> Vec<u32> {
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
    pub(super) fn emit(&self, order: &[u32], split: bool) -> Tape {
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
                Kind::Solve { n, k, count } => {
                    vec![(count * n * n) as usize, (count * n * k) as usize]
                }
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
            // A dense operand: a run of consecutive inputs or of consecutive
            // work slots is read in place, anything else gathered.
            let dense = |arg_pool: &mut Vec<u32>, max_args: &mut usize, ops: &[u32]| -> Src {
                let run = |k0: u32| ops.iter().enumerate().all(|(j, &k)| k == k0 + j as u32);
                match ops.first() {
                    Some(&k0) if run(k0) && input_index(k0).is_some() => Src::Inputs(k0 & !INPUT),
                    Some(&k0) if run(k0) => Src::Slots(k0),
                    _ => Src::Pool(gather(arg_pool, max_args, ops)),
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
                Kind::Call {
                    bundle,
                    n_groups,
                    n_args,
                    ref reads,
                    stateful,
                    ..
                } => {
                    // A stateful call's last operand is its state block.
                    let (args, state) = split_state(o, stateful);
                    let start = gather(&mut arg_pool, &mut max_args, args);
                    // its arguments are laid out whole, read or not
                    max_args = max_args.max((n_groups * n_args) as usize);
                    let (n_in, reads) = match reads {
                        None => (n_args, super::ALL_ARGS),
                        Some(r) => {
                            let at = arg_pool.len() as u32;
                            arg_pool.extend_from_slice(r);
                            (r.len() as u32, at)
                        }
                    };
                    Op::Call {
                        bundle,
                        start,
                        n_groups,
                        n_args,
                        n_in,
                        reads,
                        n_out: inst.n_out / n_groups,
                        state,
                        args: 0,
                    }
                }
                Kind::CallProlog {
                    bundle,
                    n_groups,
                    n_pure,
                    ..
                } => {
                    let start = gather(&mut arg_pool, &mut max_args, o);
                    Op::CallProlog {
                        bundle,
                        start,
                        n_groups,
                        n_pure,
                        args: 0,
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
                Kind::Solve { n, k, count } => {
                    let (a, b) = o.split_at((count * n * n) as usize);
                    let a = dense(&mut arg_pool, &mut max_args, a);
                    let b = dense(&mut arg_pool, &mut max_args, b);
                    max_args = max_args.max((count * (n * n + n * k)) as usize);
                    Op::Solve { a, b, n, k, count }
                }
            };
            ops.push(op);
            dst.push(d);
        }
        let outputs: Vec<u32> = self.roots.iter().map(|&r| slot_of(r, &base)).collect();
        // What the tail of the work buffer lends: a bundle's scratch, or a
        // solve's, never both at once.
        let solves = ops.iter().map(|op| match *op {
            Op::Solve { n, k, .. } => crate::semantics::solve_scratch_len(n as usize, k as usize),
            _ => 0,
        });
        let lent = self
            .bundles
            .iter()
            .map(|b| b.work_len())
            .chain(solves)
            .max()
            .unwrap_or(0);
        let stages = super::plan_stages(&ops, &dst, &arg_pool, &self.bundles, prolog_ops);
        // The calls of a stage gather their arguments apart, one after the
        // other in the gather area.
        for st in &stages {
            let mut at = 0u32;
            for op in &mut ops[st.lo as usize..st.hi as usize] {
                let (args, width) = match op {
                    Op::Call {
                        args,
                        n_groups,
                        n_args,
                        ..
                    } => (args, *n_groups * *n_args),
                    Op::CallProlog {
                        args,
                        n_groups,
                        n_pure,
                        ..
                    } => (args, *n_groups * *n_pure),
                    _ => unreachable!("a stage holds calls only"),
                };
                *args = at;
                at += width;
            }
            max_args = max_args.max(at as usize);
        }
        Tape {
            stages,
            lent,
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
