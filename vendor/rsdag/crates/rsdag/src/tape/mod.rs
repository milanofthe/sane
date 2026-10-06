//! Compiled flat evaluator (tape) for fast, repeated numeric evaluation.
//!
//! A [`Tape`] is the reachable sub-DAG of a set of root expressions, lowered
//! to a flat instruction list over compact slots. Symbols become positional
//! inputs, read where they are used (an operand with the input tag names an
//! input directly, no copy into a slot), and the work buffer is caller-owned
//! and reused, so a Newton loop over a large circuit neither re-hashes
//! symbols nor reallocates a scratch each step.
//!
//! An instruction writes one slot, or, for a kernel (a bundle call, a batch
//! of calls, a matrix-vector product, a dense solve), a block of consecutive
//! slots starting at its `dst`. Every value is a slot; there is no second
//! storage.
//!
//! # What a backend must reproduce
//!
//! - **The domain guards.** [`crate::semantics::unary_f64`] is the reference
//!   for every unary op, including the `exp` cap at
//!   [`crate::semantics::EXP_LIMIT`] and the `ln` floor at
//!   [`crate::semantics::LN_FLOOR`]; a backend that calls its own `exp`
//!   diverges from the arena where the guards bite.
//! - **The special functions.** [`crate::semantics::digamma`],
//!   [`crate::semantics::trigamma`] and [`crate::semantics::rand_uniform`]
//!   are defined there, not taken from a platform library.
//! - **The fold orders.** [`crate::semantics::reduce_slice_t`] and
//!   [`crate::semantics::dot_slice_t`] fold with four accumulators merged as
//!   `(a0 + a1) + (a2 + a3)`, then the tail in order; a kernel is those
//!   folds row by row.
//! - **No contraction.** [`Op::MulAdd`] is a fused *dispatch*, not
//!   a fused rounding: it rounds the product and the sum separately.

use std::sync::Arc;

use crate::extern_fn::ExternBundle;
use crate::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use crate::scalar::Scalar;

/// The tag bit of an operand that names an input rather than a slot.
pub const INPUT: u32 = 1 << 31;

/// The `state` of a call that keeps none: the bundle runs whole.
pub const NO_STATE: u32 = u32::MAX;

/// [`Op::Call::reads`] of a call whose operands are its arguments in order.
pub const ALL_ARGS: u32 = u32::MAX;

/// The input index of a tagged operand.
#[inline]
pub fn input_index(k: u32) -> Option<u32> {
    (k & INPUT != 0).then_some(k & !INPUT)
}

/// One instruction. Operands are slots, or inputs when tagged with
/// [`INPUT`]; every instruction writes its `dst` slot, a kernel the block
/// of slots from `dst` on.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    Const(f64),
    Add(u32, u32),
    Mul(u32, u32),
    /// `a*b + c` as one instruction, a fused *dispatch* with two roundings,
    /// so results stay bit-identical to the arena. Emitted by `compile`
    /// when an `Add` consumes a single-use `Mul`.
    MulAdd(u32, u32, u32),
    /// `a - b`, from an `Add` consuming a single-use `Neg`.
    Sub(u32, u32),
    Neg(u32),
    Powi(u32, i32),
    Unary(UnaryOp, u32),
    Binary(BinOp, u32, u32),
    Cmp(CmpOp, u32, u32),
    Select(u32, u32, u32),
    /// Reduction over `arg_pool[start .. start+len]`.
    Reduce(ReduceOp, u32, u32),
    /// Inner product of `arg_pool[start .. start+len]` and the `len` that
    /// follow.
    Dot(u32, u32),
    /// `bundles[b]` on `n_groups` argument groups of `n_args`, group `g`'s
    /// `n_out` outputs to `dst + g*n_out ..`; with a `state` slot (not
    /// [`NO_STATE`]), the bundle's main phase over group `g`'s instance
    /// state at `state + g*state_len`. The operands are `n_in` per group,
    /// group-major at `arg_pool[start ..]`: the arguments in order, or with
    /// `reads` (not [`ALL_ARGS`]) the ones at the positions
    /// `arg_pool[reads .. reads+n_in]`, the only ones a main phase reads
    /// (see [`Tape::main_reads`]); the others are left unset. The
    /// arguments are gathered at `args` in the gather area (see
    /// [`calls`]).
    Call {
        bundle: u32,
        start: u32,
        n_groups: u32,
        n_args: u32,
        n_in: u32,
        reads: u32,
        n_out: u32,
        state: u32,
        args: u32,
    },
    /// The prolog of `bundles[b]` for `n_groups` instances, their pure
    /// arguments group-major at `arg_pool[start ..]`, `n_pure` per group:
    /// instance `g`'s state to `dst + g*state_len ..`.
    CallProlog {
        bundle: u32,
        start: u32,
        n_groups: u32,
        n_pure: u32,
        args: u32,
    },
    /// A matrix-vector product: `m` rows of `n` in `a` against `x`, the rows
    /// to `dst .. dst+m`, each row the fold of `Dot`; with `acc`, each row
    /// folded per [`Fold`] with its accumulator entry.
    Gemv {
        a: Src,
        x: Src,
        m: u32,
        n: u32,
        acc: Option<Accum>,
    },
    /// A matrix-matrix product: `m` rows of `k` in `a` against `n` rows of
    /// `k` in `b` (the right factor by columns), entry `(i, j)` to
    /// `dst + i*n + j`, each entry the fold of `Dot`; with `acc`, each
    /// entry folded per [`Fold`] with its accumulator entry.
    Gemm {
        a: Src,
        b: Src,
        m: u32,
        k: u32,
        n: u32,
        acc: Option<Accum>,
    },
    /// The dense solves of `count` systems of one shape, each `k`
    /// right-hand sides against its `n` by `n` matrix: `a` holds the
    /// matrices back to back, `b` each system's `k` vectors of `n` back to
    /// back, the solutions go to `dst .. dst + count*n*k` the same way (see
    /// [`crate::semantics::solve_batch_into`]).
    Solve {
        a: Src,
        b: Src,
        n: u32,
        k: u32,
        count: u32,
    },
}

/// How a product kernel folds one of its entries `d` after the fold of
/// its dot: kept as is, `c - d`, `c + d` or `-d`, one rounding, as the
/// `Sub`, `Add` or `Neg` it replaces; `c` is the entry's accumulator, an
/// operand of the kernel or, for a self fold, another entry's product. A
/// kernel with `acc` carries one code per entry in the arg pool
/// (`acc.1 ..`) and the accumulator operand at `acc.0` (a plain or self
/// fold's entry there is a placeholder).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fold(pub u32);

impl Fold {
    pub const PLAIN: Fold = Fold(0);
    pub const NEG: Fold = Fold(3);
    /// `c - d` with `c` the accumulator operand.
    pub const SUB: Fold = Fold(1);
    /// `c + d` with `c` the accumulator operand.
    pub const ADD: Fold = Fold(2);
    pub fn is_plain(self) -> bool {
        self.0 & 3 == 0
    }
    pub fn is_self(self) -> bool {
        self.0 & 4 != 0
    }
    /// The entry whose product is the accumulator of a self fold.
    pub fn self_index(self) -> usize {
        (self.0 >> 3) as usize
    }
    /// The product `d` folded against the accumulator `a` (unused by a
    /// plain or negating fold).
    #[inline]
    pub fn fold<T: Scalar>(self, a: T, d: T) -> T {
        match self.0 & 3 {
            0 => d,
            1 => a.sub(d),
            2 => a.add(d),
            _ => d.neg(),
        }
    }
    /// Whether the fold reads the accumulator operand.
    pub fn reads_operand(self) -> bool {
        matches!(self.0 & 3, 1 | 2) && !self.is_self()
    }
}

/// A product kernel's folds: the fold code per entry from `codes` in the
/// arg pool, and the accumulator operand when any code reads one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Accum {
    pub c: Option<Src>,
    pub codes: u32,
}

/// Where a kernel's dense operand lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// Consecutive inputs from index `k`, read in place.
    Inputs(u32),
    /// Operands listed in the arg pool from `start`, gathered.
    Pool(u32),
    /// Consecutive work slots from `s`, read in place.
    Slots(u32),
}

/// A dense operand as a backend sees it.
#[derive(Clone, Copy, Debug)]
pub enum Operand<'a> {
    /// Consecutive inputs from `k`.
    Inputs(u32),
    /// Consecutive work slots from `s`.
    Run(u32),
    /// Slots to gather.
    Slots(&'a [u32]),
}

pub mod calls;
mod compile;
mod specialize;
mod topo;

use calls::Shared;
pub use specialize::SpecializedTape;

/// Consecutive calls of a tape that read nothing another of them writes:
/// every instance of every call in it is a piece of work of its own, run
/// together on the installed pool ([`crate::parallel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    /// The calls are ops `lo..hi`.
    pub lo: u32,
    pub hi: u32,
    /// Instances over all its calls.
    pub instances: u32,
    /// Ops it carries: per call, its instances times its body's ops.
    pub ops: u64,
}

/// Ops a call of a bundle without a body ([`ExternBundle::body`]) counts as,
/// per instance.
const OPAQUE_OPS: u64 = 1000;

/// Ops one instance of `call` runs in its bundle's body: the prolog for a
/// prolog, the main phase for a call over a state, all of it for a call
/// without one. A bundle without a body counts 1000 ops.
pub fn call_ops(bundle: &dyn ExternBundle, call: &Op) -> u64 {
    let Some(t) = bundle.body() else {
        return OPAQUE_OPS;
    };
    let (all, prolog) = (t.n_ops() as u64, t.prolog_len() as u64);
    match *call {
        Op::CallProlog { .. } => prolog.max(1),
        Op::Call { state, .. } if state != NO_STATE => (all - prolog).max(1),
        _ => all.max(1),
    }
}

/// The stages of an op stream (see [`Stage`]): each maximal run of
/// consecutive calls, within the prolog or within the main phase, in which
/// no call reads or overwrites a slot another one writes, and which holds
/// at least two instances.
fn plan_stages(
    ops: &[Op],
    dst: &[u32],
    pool: &[u32],
    bundles: &[Arc<dyn ExternBundle>],
    prolog_ops: usize,
) -> Vec<Stage> {
    // Per call: the slots it reads, the slots it writes, instances, ops.
    let call = |i: usize| -> Option<(Vec<(u32, u32)>, (u32, u32), u32, u64)> {
        let d = dst[i];
        let slots = |start: u32, len: u32| {
            pool[start as usize..(start + len) as usize]
                .iter()
                .filter(|&&k| input_index(k).is_none())
                .map(|&k| (k, k + 1))
                .collect::<Vec<_>>()
        };
        let (bundle, mut reads, width, ng) = match ops[i] {
            Op::Call {
                bundle,
                start,
                n_groups,
                n_in,
                n_out,
                state,
                ..
            } => {
                let mut r = slots(start, n_groups * n_in);
                if state != NO_STATE {
                    let sl = bundles[bundle as usize].state_len() as u32;
                    r.push((state, state + n_groups * sl));
                }
                (bundle, r, n_groups * n_out, n_groups)
            }
            Op::CallProlog {
                bundle,
                start,
                n_groups,
                n_pure,
                ..
            } => {
                let sl = bundles[bundle as usize].state_len() as u32;
                (
                    bundle,
                    slots(start, n_groups * n_pure),
                    n_groups * sl,
                    n_groups,
                )
            }
            _ => return None,
        };
        reads.sort_unstable();
        let per = call_ops(&*bundles[bundle as usize], &ops[i]);
        Some((reads, (d, d + width), ng, ng as u64 * per))
    };
    let overlaps = |a: (u32, u32), b: (u32, u32)| a.0 < b.1 && b.0 < a.1;
    let mut stages = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        let Some(first) = call(i) else {
            i += 1;
            continue;
        };
        let end = if i < prolog_ops {
            prolog_ops
        } else {
            ops.len()
        };
        let (mut reads, mut writes) = (first.0, vec![first.1]);
        let (mut instances, mut work) = (first.2, first.3);
        let mut hi = i + 1;
        while hi < end {
            let Some((r, w, ng, o)) = call(hi) else {
                break;
            };
            let clash = r.iter().any(|&a| writes.iter().any(|&b| overlaps(a, b)))
                || writes.iter().any(|&b| overlaps(w, b))
                || reads.iter().any(|&a| overlaps(a, w));
            if clash {
                break;
            }
            reads.extend(r);
            writes.push(w);
            instances += ng;
            work += o;
            hi += 1;
        }
        if instances >= 2 {
            stages.push(Stage {
                lo: i as u32,
                hi: hi as u32,
                instances,
                ops: work,
            });
        }
        i = hi;
    }
    stages
}

/// A compiled evaluator for a set of expression roots.
pub struct Tape {
    ops: Vec<Op>,
    /// Destination slot of each op (the first of a kernel's block).
    dst: Vec<u32>,
    n_selects: usize,
    arg_pool: Vec<u32>,
    /// Output operands, one per root (a slot, or a tagged input).
    outputs: Vec<u32>,
    n_work: usize,
    /// Widest gather any variadic op or kernel needs.
    max_args: usize,
    /// The widest scratch a called bundle ([`ExternBundle::work_len`]) or a
    /// dense solve ([`crate::semantics::solve_scratch_len`]) works in, lent
    /// from the tail of the work buffer, after the gather.
    lent: usize,
    bundles: Vec<Arc<dyn ExternBundle>>,
    /// Instruction count of the parameter-pure prolog prefix (0 = no split;
    /// see [`compile_split`](Self::compile_split)).
    prolog_ops: usize,
    /// The prolog's results the main phase reads: `work[..state_len]`
    /// (see [`state_len`](Self::state_len)).
    state_len: usize,
    /// The inputs it was compiled over (`input_syms.len()`).
    n_inputs: usize,
    /// Its independent calls, by op range (see [`Stage`]).
    stages: Vec<Stage>,
}

/// A compiled program as a solver drives it, whichever backend runs it:
/// buffers the caller owns and sizes from [`work_len`](Self::work_len) and
/// [`out_len`](Self::out_len), the prolog/main split, and the state layout
/// of [`Tape::state_len`], so a prolog one backend ran serves another's
/// main phase. [`Tape`] implements it, and so does the native code.
pub trait Program: Send + Sync {
    /// Inputs the program reads; `inputs` holds at least this many.
    fn n_inputs(&self) -> usize;
    fn work_len(&self) -> usize;
    fn out_len(&self) -> usize;
    /// The prolog's results are `work[..state_len]`.
    fn state_len(&self) -> usize;
    /// Everything, prolog and main.
    fn eval_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]);
    /// The parameter-pure prolog, into `work`.
    fn eval_prolog_into(&self, inputs: &[f64], work: &mut [f64]);
    /// The main phase over a `work` whose state a prolog left.
    fn eval_main_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]);
    /// Instances of `stride` inputs back to back (`stride` at least
    /// [`n_inputs`](Self::n_inputs)), their outputs back to back into `out`,
    /// one after the other over one `work`. Running parts of a batch in
    /// parallel is the caller's choice: split `inputs` and `out` alike.
    fn eval_many_into(&self, inputs: &[f64], stride: usize, work: &mut [f64], out: &mut [f64]) {
        let n_out = self.out_len();
        if stride == 0 || n_out == 0 {
            return;
        }
        assert_eq!(
            out.len() / n_out,
            inputs.len() / stride,
            "one output vector per input vector"
        );
        for (ins, dst) in inputs.chunks_exact(stride).zip(out.chunks_exact_mut(n_out)) {
            self.eval_into(ins, work, dst);
        }
    }
}

impl Program for Tape {
    fn n_inputs(&self) -> usize {
        self.n_inputs
    }
    fn work_len(&self) -> usize {
        Tape::work_len(self)
    }
    fn out_len(&self) -> usize {
        Tape::out_len(self)
    }
    fn state_len(&self) -> usize {
        Tape::state_len(self)
    }
    fn eval_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]) {
        Tape::eval_into(self, inputs, work, out)
    }
    fn eval_prolog_into(&self, inputs: &[f64], work: &mut [f64]) {
        Tape::eval_prolog_into(self, inputs, work)
    }
    fn eval_main_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]) {
        Tape::eval_main_into(self, inputs, work, out)
    }
}

/// Observer of `Select` decisions during evaluation: [`NoTrace`] costs
/// nothing, a `Vec<u8>` records one byte per `Select` in op order (`1` =
/// then-arm taken), the trace [`Tape::specialize`] takes.
pub trait TraceSink {
    fn select(&mut self, taken: bool);
}

pub struct NoTrace;
impl TraceSink for NoTrace {
    #[inline(always)]
    fn select(&mut self, _taken: bool) {}
}

impl TraceSink for Vec<u8> {
    #[inline(always)]
    fn select(&mut self, taken: bool) {
        self.push(taken as u8);
    }
}

impl Tape {
    /// Number of work slots needed by [`eval`](Self::eval).
    pub fn n_slots(&self) -> usize {
        self.n_work
    }

    /// Number of outputs (= number of roots).
    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Human-readable instruction listing (diagnostics): one line per op
    /// with its destination slot, the prolog boundary marked; an operand
    /// `i7` is input 7.
    pub fn dump(&self) -> String {
        let name = |k: u32| match input_index(k) {
            Some(i) => format!("i{i}"),
            None => format!("s{k}"),
        };
        let list = |start: u32, len: u32| -> String {
            let names: Vec<String> = self.pool(start, len).iter().map(|&k| name(k)).collect();
            names.join(", ")
        };
        let src = |s: Src, len: u32| match s {
            Src::Inputs(k) => format!("i{k}..i{}", k + len),
            Src::Pool(start) => format!("[{}]", list(start, len)),
            Src::Slots(s) => format!("s{s}..s{}", s + len),
        };
        let mut out = String::new();
        for (i, op) in self.ops.iter().enumerate() {
            if i == self.prolog_ops && self.prolog_ops > 0 {
                out.push_str("---- main ----\n");
            }
            let text = match *op {
                Op::Const(v) => format!("Const({v})"),
                Op::Add(a, b) => format!("Add({}, {})", name(a), name(b)),
                Op::Mul(a, b) => format!("Mul({}, {})", name(a), name(b)),
                Op::MulAdd(a, b, c) => format!("MulAdd({}, {}, {})", name(a), name(b), name(c)),
                Op::Sub(a, b) => format!("Sub({}, {})", name(a), name(b)),
                Op::Neg(a) => format!("Neg({})", name(a)),
                Op::Powi(a, n) => format!("Powi({}, {n})", name(a)),
                Op::Unary(op, a) => format!("Unary({op:?}, {})", name(a)),
                Op::Binary(op, a, b) => format!("Binary({op:?}, {}, {})", name(a), name(b)),
                Op::Cmp(op, a, b) => format!("Cmp({op:?}, {}, {})", name(a), name(b)),
                Op::Select(c, t, e) => format!("Select({}, {}, {})", name(c), name(t), name(e)),
                Op::Reduce(op, s, l) => format!("Reduce({op:?}, [{}])", list(s, l)),
                Op::Dot(s, l) => format!("Dot([{}], [{}])", list(s, l), list(s + l, l)),
                Op::Call {
                    bundle,
                    start,
                    n_groups: 1,
                    n_in,
                    n_out,
                    state,
                    ..
                } => format!(
                    "Call(b{bundle}, [{}]{}) -> {n_out}",
                    list(start, n_in),
                    state_text(state)
                ),
                Op::Call {
                    bundle,
                    start,
                    n_groups,
                    n_in,
                    n_out,
                    state,
                    ..
                } => format!(
                    "CallBatch(b{bundle}, {n_groups} x [{}]{}) -> {n_groups} x {n_out}",
                    list(start, n_groups * n_in),
                    state_text(state)
                ),
                Op::CallProlog {
                    bundle,
                    start,
                    n_groups,
                    n_pure,
                    ..
                } => format!(
                    "CallProlog(b{bundle}, {n_groups} x [{}]) -> {n_groups} x {}",
                    list(start, n_groups * n_pure),
                    self.bundles[bundle as usize].state_len()
                ),
                Op::Gemv { a, x, m, n, acc } => {
                    format!(
                        "Gemv({m}x{n} {}, {}){} -> {m}",
                        src(a, m * n),
                        src(x, n),
                        acc_text(&self.arg_pool, acc, m)
                    )
                }
                Op::Gemm { a, b, m, k, n, acc } => {
                    format!(
                        "Gemm({m}x{k} {}, {n}x{k} {}){} -> {}",
                        src(a, m * k),
                        src(b, n * k),
                        acc_text(&self.arg_pool, acc, m * n),
                        m * n
                    )
                }
                Op::Solve {
                    a,
                    b,
                    n,
                    k: 1,
                    count: 1,
                } => {
                    format!("Solve({n}x{n} {}, {}) -> {n}", src(a, n * n), src(b, n))
                }
                Op::Solve { a, b, n, k, count } if count > 1 => {
                    format!(
                        "SolveBatch({count} of {n}x{n} {}, {k} rhs {}) -> {}",
                        src(a, count * n * n),
                        src(b, count * n * k),
                        count * n * k
                    )
                }
                Op::Solve { a, b, n, k, .. } => {
                    format!(
                        "SolveMany({n}x{n} {}, {k} rhs {}) -> {}",
                        src(a, n * n),
                        src(b, n * k),
                        n * k
                    )
                }
            };
            out.push_str(&format!("{i:5}: s{} <- {text}\n", self.dst[i]));
        }
        let outs: Vec<String> = self.outputs.iter().map(|&k| name(k)).collect();
        out.push_str(&format!("outputs [{}]\n", outs.join(", ")));
        out
    }

    /// Number of `Select` ops in the tape (the length of a choice trace).
    pub fn n_selects(&self) -> usize {
        self.n_selects
    }

    /// Work slots an evaluation needs: the slots themselves plus the gather
    /// scratch. A caller that owns its buffers sizes them with this and
    /// [`out_len`](Self::out_len) once, before the loop that uses them.
    pub fn work_len(&self) -> usize {
        self.buffer_len()
    }

    /// Number of outputs, the length [`eval_into`](Self::eval_into) writes.
    pub fn out_len(&self) -> usize {
        self.outputs.len()
    }

    /// Evaluate into buffers the caller owns: nothing is allocated, nothing
    /// is returned, and `out` may point anywhere (a factorization's value
    /// array, a right-hand side, a numpy array). `work` must be at least
    /// [`work_len`](Self::work_len) long, `out` exactly
    /// [`out_len`](Self::out_len).
    pub fn eval_into<T: Scalar>(&self, inputs: &[T], work: &mut [T], out: &mut [T]) {
        assert!(work.len() >= self.work_len(), "work buffer too short");
        assert_eq!(out.len(), self.out_len(), "output buffer of the wrong size");
        self.run_range(inputs, work, 0, self.ops.len(), &mut NoTrace);
        self.write(inputs, work, out);
    }

    /// Evaluate the tape in any execution scalar (`f64`, `f32`,
    /// `Complex<f64>`): constants convert from their `f64` lowering, every
    /// op goes through [`Scalar`]'s reference arithmetic for `T`, so a value
    /// cannot depend on which scalar computed it beyond the scalar itself.
    /// `work` is resized and reused; `out` receives one value per output.
    ///
    /// The `Vec` form is the convenience over
    /// [`eval_into`](Self::eval_into), the one a caller in an inner loop wants.
    pub fn eval<T: Scalar>(&self, inputs: &[T], work: &mut Vec<T>, out: &mut Vec<T>) {
        self.eval_with(inputs, work, out, &mut NoTrace);
    }

    /// [`eval`](Self::eval) with every `Select` decision reported to `sink`;
    /// a `Vec<u8>` sink is the choice trace [`specialize`](Self::specialize)
    /// takes (the caller clears it first).
    pub fn eval_with<T: Scalar, S: TraceSink>(
        &self,
        inputs: &[T],
        work: &mut Vec<T>,
        out: &mut Vec<T>,
        sink: &mut S,
    ) {
        if work.len() < self.buffer_len() {
            work.resize(self.buffer_len(), T::zero());
        }
        self.run_range(inputs, work, 0, self.ops.len(), sink);
        self.collect(inputs, work, out);
    }

    /// The work buffer: the slots, the gather, then what is lent.
    fn buffer_len(&self) -> usize {
        self.n_work + self.max_args + self.lent
    }

    fn collect<T: Scalar>(&self, inputs: &[T], work: &[T], out: &mut Vec<T>) {
        out.clear();
        out.extend(self.outputs.iter().map(|&k| read(inputs, work, k)));
    }

    /// [`collect`](Self::collect) into a slice the caller sized.
    fn write<T: Scalar>(&self, inputs: &[T], work: &[T], out: &mut [T]) {
        for (dst, &k) in out.iter_mut().zip(self.outputs.iter()) {
            *dst = read(inputs, work, k);
        }
    }

    /// The values the prolog leaves for the main phase are `work[..n]`:
    /// everything a later [`eval_main_into`](Self::eval_main_into) needs of a
    /// prolog run, so an instance's prolog result is saved and restored as
    /// this prefix. The layout is the tape's, shared by every backend. `0`
    /// without a split.
    pub fn state_len(&self) -> usize {
        self.state_len
    }

    /// Instruction count of the parameter-pure prolog (0 when compiled without
    /// [`compile_split`](Self::compile_split)).
    pub fn prolog_len(&self) -> usize {
        self.prolog_ops
    }

    /// Evaluate the parameter-pure prolog into `work` (grown here to the
    /// buffer length; nothing is cleared, every slot is written before read).
    /// A Newton loop calls this once per parameter binding, then
    /// [`eval_main`](Self::eval_main) per iteration over the *same* buffer.
    pub fn eval_prolog<T: Scalar>(&self, inputs: &[T], work: &mut Vec<T>) {
        if work.len() < self.buffer_len() {
            work.resize(self.buffer_len(), T::zero());
        }
        self.run_range(inputs, work, 0, self.prolog_ops, &mut NoTrace);
    }

    /// Evaluate the main phase over a `work` buffer prepared by
    /// [`eval_prolog`](Self::eval_prolog) (prolog results are pinned slots, so
    /// repeated main passes may not clear or resize the buffer).
    pub fn eval_main<T: Scalar>(&self, inputs: &[T], work: &mut [T], out: &mut Vec<T>) {
        out.resize(self.out_len(), T::zero());
        self.eval_main_into(inputs, work, out);
    }

    /// [`eval_main`](Self::eval_main) into a slice the caller sized: the form
    /// a Newton loop uses, one prolog per parameter binding and this per
    /// iteration, with no allocation in either.
    pub fn eval_main_into<T: Scalar>(&self, inputs: &[T], work: &mut [T], out: &mut [T]) {
        assert!(
            work.len() >= self.work_len(),
            "eval_main requires a work buffer prepared by eval_prolog"
        );
        assert_eq!(out.len(), self.out_len(), "output buffer of the wrong size");
        self.run_range(inputs, work, self.prolog_ops, self.ops.len(), &mut NoTrace);
        self.write(inputs, work, out);
    }

    /// [`eval_prolog`](Self::eval_prolog) over a buffer the caller sized.
    pub fn eval_prolog_into<T: Scalar>(&self, inputs: &[T], work: &mut [T]) {
        assert!(work.len() >= self.work_len(), "work buffer too short");
        self.run_range(inputs, work, 0, self.prolog_ops, &mut NoTrace);
    }

    /// Execute ops `lo..hi` over a fully-sized work buffer, feeding each
    /// `Select` decision to `sink` (the no-op sink costs nothing).
    fn run_range<T: Scalar, S: TraceSink>(
        &self,
        inputs: &[T],
        work: &mut [T],
        lo: usize,
        hi: usize,
        sink: &mut S,
    ) {
        use crate::semantics::{reduce_slice_t, solve_batch_into};
        // The gather scratch at the tail of `work`, so nothing is allocated
        // per call.
        let (work, scratch) = work.split_at_mut(self.n_work);
        let pool = |start: u32, len: u32| &self.arg_pool[start as usize..(start + len) as usize];
        // The next stage at or after `lo`, and the op a stage run jumps to.
        let mut next = self.stages.partition_point(|st| (st.lo as usize) < lo);
        let mut skip = lo;
        for i in lo..hi {
            if i < skip {
                continue;
            }
            if let Some(st) = self.stages.get(next).filter(|st| st.lo as usize == i) {
                next += 1;
                if st.hi as usize <= hi && crate::parallel::worth(st.ops as usize) {
                    let (slots, gather) = (
                        Shared::new(work),
                        Shared::new(&mut scratch[..self.max_args]),
                    );
                    // SAFETY: `plan_stages` admits a call to a stage only if
                    // it reads no slot another call of the stage writes and
                    // writes no slot another one reads or writes; their
                    // arguments are apart in the gather area.
                    let (lo, n) = (st.lo as usize, (st.hi - st.lo) as usize);
                    let call = |k| self.call(lo + k);
                    unsafe { calls::run_stage(n, call, slots, gather, inputs) };
                    skip = st.hi as usize;
                    continue;
                }
            }
            let g = |k: u32| read(inputs, work, k);
            let d = self.dst[i] as usize;
            let v = match self.ops[i] {
                Op::Const(v) => T::from_f64(v),
                Op::Add(a, b) => g(a).add(g(b)),
                Op::Mul(a, b) => g(a).mul(g(b)),
                // The product and the sum round separately: a fused
                // dispatch, not a fused rounding.
                Op::MulAdd(a, b, c) => g(a).mul(g(b)).add(g(c)),
                Op::Sub(a, b) => g(a).sub(g(b)),
                Op::Neg(a) => g(a).neg(),
                Op::Powi(a, n) => g(a).powi(n),
                Op::Unary(op, a) => T::unary(op, g(a)),
                Op::Binary(op, a, b) => T::binary(op, g(a), g(b)),
                Op::Cmp(op, a, b) => T::cmp(op, g(a), g(b)),
                Op::Select(c, t, e) => {
                    let taken = g(c).is_true();
                    sink.select(taken);
                    if taken {
                        g(t)
                    } else {
                        g(e)
                    }
                }
                Op::Reduce(op, start, len) => {
                    for (j, &k) in pool(start, len).iter().enumerate() {
                        scratch[j] = g(k);
                    }
                    reduce_slice_t(op, &scratch[..len as usize])
                }
                Op::Dot(start, len) => {
                    for (j, &k) in pool(start, 2 * len).iter().enumerate() {
                        scratch[j] = g(k);
                    }
                    T::dot_slice(
                        &scratch[..len as usize],
                        &scratch[len as usize..2 * len as usize],
                    )
                }
                Op::Call { .. } | Op::CallProlog { .. } => {
                    let c = self.call(i);
                    let (gather, lent) = scratch.split_at_mut(self.max_args);
                    let (slots, gather) = (Shared::new(work), Shared::new(gather));
                    // SAFETY: one call at a time, over its own regions.
                    unsafe { calls::run_call(&c, slots, gather, inputs, 0..c.n_groups, lent) };
                    continue;
                }
                Op::Gemv { a, x, m, n, acc } => {
                    let (m, n) = (m as usize, n as usize);
                    let mut at = 0usize;
                    let pool = &self.arg_pool;
                    let ra = place_operand(inputs, work, scratch, pool, a, m * n, &mut at);
                    let rx = place_operand(inputs, work, scratch, pool, x, n, &mut at);
                    let rc = acc.map(|f| {
                        let c =
                            f.c.map(|c| place_operand(inputs, work, scratch, pool, c, m, &mut at));
                        (c, f.codes)
                    });
                    let base = work.as_mut_ptr();
                    let av: &[T] = dense_slice(inputs, base, scratch, ra, m * n);
                    let xv: &[T] = dense_slice(inputs, base, scratch, rx, n);
                    // The kernel's outputs are fresh slots: no operand, the
                    // accumulator included, lives where they go.
                    let reads = [
                        Some((ra, m * n)),
                        Some((rx, n)),
                        rc.and_then(|(c, _)| c.map(|c| (c, m))),
                    ];
                    let out = unsafe { out_block(base, work.len(), d, m, &reads) };
                    match rc {
                        None => T::gemv(av, xv, m, n, out),
                        Some((rc, codes)) => {
                            let cv: Option<&[T]> =
                                rc.map(|rc| dense_slice(inputs, base, scratch, rc, m));
                            let codes = &self.arg_pool[codes as usize..codes as usize + m];
                            T::gemv_fold(av, xv, m, n, cv, codes, out);
                        }
                    }
                    continue;
                }
                Op::Gemm { a, b, m, k, n, acc } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    let mut at = 0usize;
                    let pool = &self.arg_pool;
                    let ra = place_operand(inputs, work, scratch, pool, a, m * k, &mut at);
                    let rb = place_operand(inputs, work, scratch, pool, b, n * k, &mut at);
                    let rc = acc.map(|f| {
                        let c = f
                            .c
                            .map(|c| place_operand(inputs, work, scratch, pool, c, m * n, &mut at));
                        (c, f.codes)
                    });
                    let base = work.as_mut_ptr();
                    let av: &[T] = dense_slice(inputs, base, scratch, ra, m * k);
                    let bv: &[T] = dense_slice(inputs, base, scratch, rb, n * k);
                    let reads = [
                        Some((ra, m * k)),
                        Some((rb, n * k)),
                        rc.and_then(|(c, _)| c.map(|c| (c, m * n))),
                    ];
                    let out = unsafe { out_block(base, work.len(), d, m * n, &reads) };
                    match rc {
                        None => T::gemm(av, bv, m, k, n, out),
                        Some((rc, codes)) => {
                            let cv: Option<&[T]> =
                                rc.map(|rc| dense_slice(inputs, base, scratch, rc, m * n));
                            let codes = &self.arg_pool[codes as usize..codes as usize + m * n];
                            T::gemm_fold(av, bv, m, k, n, cv, codes, out);
                        }
                    }
                    continue;
                }
                Op::Solve { a, b, n, k, count } => {
                    let (n, k, count) = (n as usize, k as usize, count as usize);
                    let (la, lb) = (count * n * n, count * n * k);
                    // The operands gathered at the front of the scratch, the
                    // solve working behind them.
                    let (scratch, lent) = scratch.split_at_mut(self.max_args);
                    let (ra, rb) =
                        dense_operands(inputs, work, scratch, &self.arg_pool, a, la, b, lb);
                    let base = work.as_mut_ptr();
                    let av: &[T] = dense_slice(inputs, base, scratch, ra, la);
                    let bv: &[T] = dense_slice(inputs, base, scratch, rb, lb);
                    let reads = [Some((ra, la)), Some((rb, lb)), None];
                    let out = unsafe { out_block(base, work.len(), d, lb, &reads) };
                    solve_batch_into(av, bv, n, k, count, out, lent);
                    continue;
                }
            };
            work[d] = v;
        }
    }

    /// Op `i`, a call, as [`calls::run_call`] runs it.
    pub fn call(&self, i: usize) -> calls::Call<'_> {
        let op = &self.ops[i];
        let (bundle, start, n_groups, n_args, n_in, places, n_out, state, args, phase) = match *op {
            Op::Call {
                bundle,
                start,
                n_groups,
                n_args,
                n_in,
                reads,
                n_out,
                state,
                args,
            } => {
                let places = (reads != ALL_ARGS).then(|| self.pool(reads, n_in));
                let phase = match state {
                    NO_STATE => calls::Phase::Whole,
                    _ => calls::Phase::Main,
                };
                (
                    bundle, start, n_groups, n_args, n_in, places, n_out, state, args, phase,
                )
            }
            Op::CallProlog {
                bundle,
                start,
                n_groups,
                n_pure,
                args,
            } => {
                let sl = self.bundles[bundle as usize].state_len() as u32;
                (
                    bundle,
                    start,
                    n_groups,
                    n_pure,
                    n_pure,
                    None,
                    sl,
                    0,
                    args,
                    calls::Phase::Prolog,
                )
            }
            _ => panic!("op {i} is not a call"),
        };
        let b = &*self.bundles[bundle as usize];
        calls::Call {
            bundle: b,
            phase,
            n_groups: n_groups as usize,
            n_args: n_args as usize,
            n_out: n_out as usize,
            operands: Some(self.pool(start, n_groups * n_in)),
            places,
            args: args as usize,
            out: self.dst[i] as usize,
            state: state as usize,
            ops: n_groups as u64 * call_ops(b, op),
        }
    }

    /// Output operands, one per root: a slot, or an input when tagged.
    pub fn outputs(&self) -> &[u32] {
        &self.outputs
    }

    /// The instruction stream, for a backend that lowers it (the native
    /// code, a printer, a code generator): op `i` writes from slot
    /// [`dst`](Self::dst)`(i)`, its lists are windows of [`pool`](Self::pool)
    /// and its bundle an index into [`bundles`](Self::bundles). A slot is
    /// never reused while a value it holds is still needed, so the current
    /// occupant of a slot is the value an operand means. The first
    /// [`prolog_len`](Self::prolog_len) ops are the prolog.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// The destination slot of op `i` (the first of a kernel's block).
    pub fn dst(&self, i: usize) -> u32 {
        self.dst[i]
    }

    /// The inputs the main phase reads, ascending: what a call of this
    /// tape as a body needs per evaluation once its prolog ran.
    pub fn main_reads(&self) -> Vec<u32> {
        let mut r: Vec<u32> = Vec::new();
        for i in self.prolog_ops..self.ops.len() {
            self.for_each_operand(i, |k| {
                if let Some(j) = input_index(k) {
                    r.push(j);
                }
            });
        }
        r.extend(self.outputs.iter().filter_map(|&o| input_index(o)));
        r.sort_unstable();
        r.dedup();
        r
    }

    /// The operand list `start .. start + len` of the pool.
    pub fn pool(&self, start: u32, len: u32) -> &[u32] {
        &self.arg_pool[start as usize..(start + len) as usize]
    }

    /// The bundles the calls call, by index.
    pub fn bundles(&self) -> &[Arc<dyn ExternBundle>] {
        &self.bundles
    }

    /// A kernel's dense operand of `len` values.
    pub fn operand(&self, src: Src, len: u32) -> Operand<'_> {
        match src {
            Src::Inputs(k) => Operand::Inputs(k),
            Src::Pool(start) => Operand::Slots(self.pool(start, len)),
            Src::Slots(s) => Operand::Run(s),
        }
    }

    /// Every operand op `i` reads, in read order: slots, or inputs tagged
    /// with [`INPUT`] (a kernel's run of inputs one by one), and last the
    /// state block of a call that keeps one, by its first slot.
    pub fn for_each_operand(&self, i: usize, mut f: impl FnMut(u32)) {
        let f = &mut f as &mut dyn FnMut(u32);
        let dense = |src: Src, len: u32, f: &mut dyn FnMut(u32)| match src {
            Src::Inputs(k) => (k..k + len).for_each(|j| f(j | INPUT)),
            Src::Pool(start) => self.pool(start, len).iter().for_each(|&k| f(k)),
            Src::Slots(s) => (s..s + len).for_each(f),
        };
        let acc = |acc: Option<Accum>, len: u32, f: &mut dyn FnMut(u32)| {
            if let Some(Accum { c: Some(c), .. }) = acc {
                dense(c, len, f);
            }
        };
        match self.ops[i] {
            Op::Const(_) => {}
            Op::Neg(a) | Op::Powi(a, _) | Op::Unary(_, a) => f(a),
            Op::Add(a, b)
            | Op::Mul(a, b)
            | Op::Sub(a, b)
            | Op::Cmp(_, a, b)
            | Op::Binary(_, a, b) => {
                f(a);
                f(b);
            }
            Op::MulAdd(a, b, c) | Op::Select(a, b, c) => {
                f(a);
                f(b);
                f(c);
            }
            Op::Reduce(_, start, len) => dense(Src::Pool(start), len, f),
            Op::Dot(start, len) => dense(Src::Pool(start), 2 * len, f),
            Op::Call {
                start,
                n_groups,
                n_in,
                state,
                ..
            } => {
                dense(Src::Pool(start), n_groups * n_in, f);
                if state != NO_STATE {
                    f(state);
                }
            }
            Op::CallProlog {
                start,
                n_groups,
                n_pure,
                ..
            } => dense(Src::Pool(start), n_groups * n_pure, f),
            Op::Gemv { a, x, m, n, acc: c } => {
                dense(a, m * n, f);
                dense(x, n, f);
                acc(c, m, f);
            }
            Op::Gemm {
                a,
                b,
                m,
                k,
                n,
                acc: c,
            } => {
                dense(a, m * k, f);
                dense(b, n * k, f);
                acc(c, m * n, f);
            }
            Op::Solve { a, b, n, k, count } => {
                dense(a, count * n * n, f);
                dense(b, count * n * k, f);
            }
        }
    }

    /// How many slots op `i` writes from its destination.
    pub fn width(&self, i: usize) -> u32 {
        match self.ops[i] {
            Op::Call {
                n_groups, n_out, ..
            } => n_groups * n_out,
            Op::CallProlog {
                bundle, n_groups, ..
            } => n_groups * self.bundles[bundle as usize].state_len() as u32,
            Op::Gemv { m, .. } => m,
            Op::Gemm { m, n, .. } => m * n,
            Op::Solve { n, k, count, .. } => count * n * k,
            _ => 1,
        }
    }

    /// Number of instructions (diagnostics; compare against a
    /// [`SpecializedTape::n_ops`] for the shrink factor).
    pub fn n_ops(&self) -> usize {
        self.ops.len()
    }

    /// Its stages of independent calls, in op order (see [`Stage`]).
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }
}

/// The value of an operand: a slot, or an input when tagged (a missing
/// input is `NaN`).
#[inline]
fn read<T: Scalar>(inputs: &[T], work: &[T], k: u32) -> T {
    match input_index(k) {
        Some(i) => inputs.get(i as usize).copied().unwrap_or(T::nan()),
        None => work[k as usize],
    }
}

/// A dense operand resolved for a kernel: in place in the inputs, or in
/// the scratch from an element on.
#[derive(Clone, Copy)]
enum Dense {
    Inputs(usize),
    Scratch(usize),
    /// A run of consecutive work slots, read in place.
    Work(usize),
}

/// The slice a resolved dense operand names, `len` values long. `work` is
/// the work array's base pointer, the one the kernel's output block is
/// derived from too (see [`out_block`]): a run read in place is disjoint
/// from that block (the allocator gives a kernel a fresh block), and both
/// come from one pointer, so neither borrow invalidates the other.
fn dense_slice<'a, T: Scalar>(
    inputs: &'a [T],
    work: *mut T,
    scratch: &'a [T],
    d: Dense,
    len: usize,
) -> &'a [T] {
    match d {
        Dense::Inputs(k) => &inputs[k..k + len],
        Dense::Scratch(s) => &scratch[s..s + len],
        // SAFETY: `s .. s + len` lies within the work array (the tape's
        // slots) and no kernel writes it while it is read.
        Dense::Work(s) => unsafe { std::slice::from_raw_parts(work.add(s), len) },
    }
}

/// A kernel's output block `d .. d + len` of the work array at `work`
/// (`work_len` long), derived from the same pointer as its in-place reads
/// `reads`, which it must not overlap.
///
/// # Safety
///
/// `work` points to `work_len` initialized values that nothing else
/// borrows for the returned lifetime except the in-place reads, which lie
/// outside the block.
unsafe fn out_block<'a, T>(
    work: *mut T,
    work_len: usize,
    d: usize,
    len: usize,
    reads: &[Option<(Dense, usize)>],
) -> &'a mut [T] {
    assert!(
        d + len <= work_len,
        "kernel output block past the work array"
    );
    for &(r, rlen) in reads.iter().flatten() {
        if let Dense::Work(s) = r {
            assert!(
                s + rlen <= d || d + len <= s,
                "kernel reads its own output block"
            );
        }
    }
    std::slice::from_raw_parts_mut(work.add(d), len)
}

/// Where a dense operand is read from: in place (a run of inputs the inputs
/// reach, a run of work slots), or gathered into `scratch` from `*at` on,
/// which advances past it.
fn place_operand<T: Scalar>(
    inputs: &[T],
    work: &[T],
    scratch: &mut [T],
    pool: &[u32],
    src: Src,
    len: usize,
    at: &mut usize,
) -> Dense {
    match src {
        Src::Inputs(k) if inputs.len() >= k as usize + len => Dense::Inputs(k as usize),
        Src::Inputs(k) => {
            for j in 0..len {
                scratch[*at + j] = inputs.get(k as usize + j).copied().unwrap_or(T::nan());
            }
            *at += len;
            Dense::Scratch(*at - len)
        }
        Src::Slots(s) => Dense::Work(s as usize),
        Src::Pool(start) => {
            let run = &pool[start as usize..start as usize + len];
            for j in 0..len {
                scratch[*at + j] = read(inputs, work, run[j]);
            }
            *at += len;
            Dense::Scratch(*at - len)
        }
    }
}

/// Resolve a kernel's two dense operands, `a` first, then `b` (see
/// [`place_operand`]).
#[allow(clippy::too_many_arguments)]
fn dense_operands<T: Scalar>(
    inputs: &[T],
    work: &[T],
    scratch: &mut [T],
    pool: &[u32],
    a: Src,
    len_a: usize,
    b: Src,
    len_b: usize,
) -> (Dense, Dense) {
    let mut at = 0usize;
    let ra = place_operand(inputs, work, scratch, pool, a, len_a, &mut at);
    let rb = place_operand(inputs, work, scratch, pool, b, len_b, &mut at);
    (ra, rb)
}

/// The accumulator part of a kernel's dump line: the operand and the
/// fold codes.
fn state_text(state: u32) -> String {
    if state == NO_STATE {
        String::new()
    } else {
        format!(", state @{state}")
    }
}

fn acc_text(pool: &[u32], acc: Option<Accum>, len: u32) -> String {
    match acc {
        None => String::new(),
        Some(Accum { c, codes }) => {
            let codes = &pool[codes as usize..(codes + len) as usize];
            let text: Vec<String> = codes
                .iter()
                .map(|&code| {
                    let f = Fold(code);
                    let op = match code & 3 {
                        0 => "=",
                        1 => "-",
                        2 => "+",
                        _ => "neg",
                    };
                    if f.is_self() {
                        format!("{op}#{}", f.self_index())
                    } else {
                        op.to_string()
                    }
                })
                .collect();
            format!(
                " acc {} [{}]",
                match c {
                    Some(Src::Inputs(k)) => format!("i{k}..i{}", k + len),
                    Some(Src::Pool(start)) => format!("pool{start}[{len}]"),
                    Some(Src::Slots(s)) => format!("s{s}..s{}", s + len),
                    None => "-".to_string(),
                },
                text.join(",")
            )
        }
    }
}
