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
//! - **No contraction.** [`TapeVisitor::mul_add`] is a fused *dispatch*, not
//!   a fused rounding: it rounds the product and the sum separately.

use std::sync::Arc;

use crate::extern_fn::ExternBundle;
use crate::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use crate::scalar::Scalar;

/// The tag bit of an operand that names an input rather than a slot.
pub const INPUT: u32 = 1 << 31;

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
    /// `bundles[b]` on `arg_pool[start .. start+n_args]`, its outputs to
    /// `dst .. dst+n_out`.
    Call {
        bundle: u32,
        start: u32,
        n_args: u32,
        n_out: u32,
    },
    /// `bundles[b]` on `n_groups` argument groups laid group-major in the
    /// pool, group `g`'s outputs to `dst + g*n_out ..`.
    CallBatch {
        bundle: u32,
        start: u32,
        n_groups: u32,
        n_args: u32,
        n_out: u32,
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
    /// The dense solve `A x = b`, `a` `n` by `n` and `b` of `n`, `x` to
    /// `dst .. dst+n` (see [`crate::semantics::solve_t`]).
    Solve {
        a: Src,
        b: Src,
        n: u32,
    },
    /// The dense solve of `k` right-hand sides against one `n` by `n` `a`:
    /// `b` holds `k` vectors of `n` back to back, the `k` solutions go to
    /// `dst .. dst + n*k` the same way (see [`crate::semantics::solve_many_t`]).
    SolveMany {
        a: Src,
        b: Src,
        n: u32,
        k: u32,
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
    /// `c - d` with `c` the product of entry `j` of the same kernel.
    pub fn sub_self(j: u32) -> Fold {
        Fold(1 | 4 | (j << 3))
    }
    /// `c + d` with `c` the product of entry `j` of the same kernel.
    pub fn add_self(j: u32) -> Fold {
        Fold(2 | 4 | (j << 3))
    }
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
}

/// A dense operand as a backend sees it.
#[derive(Clone, Copy, Debug)]
pub enum Operand<'a> {
    Inputs(u32),
    Slots(&'a [u32]),
}

mod compile;
mod specialize;

pub use compile::Inst;
pub use specialize::SpecializedTape;

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
    bundles: Vec<Arc<dyn ExternBundle>>,
    /// Instruction count of the parameter-pure prolog prefix (0 = no split;
    /// see [`compile_split`](Self::compile_split)).
    prolog_ops: usize,
}

/// A backend that lowers a [`Tape`]'s instruction stream: the seam every
/// code generator sits on.
///
/// The tape is the evaluation IR. [`Tape::eval`] is the interpreting
/// backend, [`Tape::lower`] drives any other one: a printer, another
/// evaluator, or a generator emitting C, Verilog or a GPU kernel. Each method
/// receives the destination slot `dst` and the operands the op reads, slots
/// or tagged inputs (see [`INPUT`]); a backend keeps its own slot-to-value
/// map. A slot is never reused while a value it holds is still needed, so
/// reading the current occupant is always the intended value. A program
/// with a parameter-pure prefix (see [`Tape::compile_split`]) exposes it as
/// the first [`Tape::prolog_len`] ops.
pub trait TapeVisitor {
    fn constant(&mut self, dst: u32, v: f64);
    fn add(&mut self, dst: u32, a: u32, b: u32);
    fn mul(&mut self, dst: u32, a: u32, b: u32);
    /// `a*b + c` as one dispatch, two roundings (see `Op::MulAdd`).
    fn mul_add(&mut self, dst: u32, a: u32, b: u32, c: u32);
    fn sub(&mut self, dst: u32, a: u32, b: u32);
    fn neg(&mut self, dst: u32, a: u32);
    fn powi(&mut self, dst: u32, a: u32, n: i32);
    fn unary(&mut self, dst: u32, op: UnaryOp, a: u32);
    fn binary(&mut self, dst: u32, op: BinOp, a: u32, b: u32);
    fn cmp(&mut self, dst: u32, op: CmpOp, a: u32, b: u32);
    fn select(&mut self, dst: u32, c: u32, t: u32, e: u32);
    fn reduce(&mut self, dst: u32, op: ReduceOp, args: &[u32]);
    fn dot(&mut self, dst: u32, a: &[u32], b: &[u32]);
    /// Call `b` on `args`, its `n_out` outputs to `dst ..`.
    fn call(&mut self, dst: u32, b: &Arc<dyn ExternBundle>, args: &[u32], n_out: u32);
    /// Call `b` on `n_groups` argument groups (group-major `args`), group
    /// `g`'s outputs to `dst + g*n_out ..`.
    fn call_batch(
        &mut self,
        dst: u32,
        b: &Arc<dyn ExternBundle>,
        args: &[u32],
        n_groups: u32,
        n_args: u32,
        n_out: u32,
    );
    /// `m` rows of `n` in `a` against `x`, to `dst .. dst+m`, each row the
    /// fold of [`dot`](Self::dot) (see [`crate::semantics::gemv_t`]).
    fn gemv(
        &mut self,
        dst: u32,
        a: Operand<'_>,
        x: Operand<'_>,
        m: u32,
        n: u32,
        acc: Option<(Option<Operand<'_>>, &[u32])>,
    );
    /// The product of `m` rows of `k` in `a` with `n` rows of `k` in `b`,
    /// entry `(i, j)` to `dst + i*n + j` (see [`crate::semantics::gemm_t`]).
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        dst: u32,
        a: Operand<'_>,
        b: Operand<'_>,
        m: u32,
        k: u32,
        n: u32,
        acc: Option<(Option<Operand<'_>>, &[u32])>,
    );
    /// The dense solve of `a` (`n` by `n`) against `b`, to `dst .. dst+n`
    /// (see [`crate::semantics::solve_t`]).
    fn solve(&mut self, dst: u32, a: Operand<'_>, b: Operand<'_>, n: u32);
    /// The dense solve of `k` right-hand sides (`b`: `k` vectors of `n`
    /// back to back) against `a`, the solutions to `dst .. dst + n*k`
    /// (see [`crate::semantics::solve_many_t`]).
    fn solve_many(&mut self, dst: u32, a: Operand<'_>, b: Operand<'_>, n: u32, k: u32);
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
            self.arg_pool[start as usize..(start + len) as usize]
                .iter()
                .map(|&k| name(k))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let src = |s: Src, len: u32| match s {
            Src::Inputs(k) => format!("i{k}..i{}", k + len),
            Src::Pool(start) => format!("[{}]", list(start, len)),
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
                    n_args,
                    n_out,
                } => format!("Call(b{bundle}, [{}]) -> {n_out}", list(start, n_args)),
                Op::CallBatch {
                    bundle,
                    start,
                    n_groups,
                    n_args,
                    n_out,
                } => format!(
                    "CallBatch(b{bundle}, {n_groups} x [{}]) -> {n_groups} x {n_out}",
                    list(start, n_groups * n_args)
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
                Op::Solve { a, b, n } => {
                    format!("Solve({n}x{n} {}, {}) -> {n}", src(a, n * n), src(b, n))
                }
                Op::SolveMany { a, b, n, k } => {
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

    /// A holder for the buffers, for callers that would rather read values
    /// than manage memory: allocates once, lends the outputs out per
    /// evaluation.
    pub fn runner<T: Scalar>(&self) -> Runner<'_, T> {
        Runner {
            tape: self,
            work: vec![T::zero(); self.work_len()],
            out: vec![T::zero(); self.out_len()],
        }
    }

    /// Evaluate the tape in any execution scalar (`f64`, `f32`,
    /// `Complex<f64>`): constants convert from their `f64` lowering, every
    /// op goes through [`Scalar`]'s reference arithmetic for `T`, so a value
    /// cannot depend on which scalar computed it beyond the scalar itself.
    /// `work` is resized and reused; `out` receives one value per output.
    ///
    /// The `Vec` form is the convenience over
    /// [`eval_into`](Self::eval_into); a caller in an inner loop wants that
    /// one or [`runner`](Self::runner).
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

    /// The work buffer: the slots, then the gather scratch.
    fn buffer_len(&self) -> usize {
        self.n_work + self.max_args
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
        use crate::semantics::{reduce_slice_t, solve_many_t, solve_t};
        // The gather scratch at the tail of `work`, so nothing is allocated
        // per call.
        let (work, scratch) = work.split_at_mut(self.n_work);
        let pool = |start: u32, len: u32| &self.arg_pool[start as usize..(start + len) as usize];
        for i in lo..hi {
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
                Op::Call {
                    bundle,
                    start,
                    n_args,
                    n_out,
                } => {
                    for (j, &k) in pool(start, n_args).iter().enumerate() {
                        scratch[j] = g(k);
                    }
                    let b = &*self.bundles[bundle as usize];
                    T::call_bundle(
                        b,
                        &scratch[..n_args as usize],
                        &mut work[d..d + n_out as usize],
                    );
                    continue;
                }
                Op::CallBatch {
                    bundle,
                    start,
                    n_groups,
                    n_args,
                    n_out,
                } => {
                    let flat = (n_groups * n_args) as usize;
                    for (j, &k) in pool(start, n_groups * n_args).iter().enumerate() {
                        scratch[j] = g(k);
                    }
                    let b = &*self.bundles[bundle as usize];
                    T::call_bundle_batch(
                        b,
                        &scratch[..flat],
                        n_groups as usize,
                        n_args as usize,
                        &mut work[d..d + (n_groups * n_out) as usize],
                    );
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
                    let av: &[T] = dense_slice(inputs, work.as_ptr(), scratch, ra, m * n);
                    let xv: &[T] = dense_slice(inputs, work.as_ptr(), scratch, rx, n);
                    match rc {
                        None => T::gemv(av, xv, m, n, &mut work[d..d + m]),
                        Some((rc, codes)) => {
                            // The kernel's outputs are fresh slots: the
                            // accumulator never lives where they go.
                            let cv: Option<&[T]> =
                                rc.map(|rc| dense_slice(inputs, work.as_ptr(), scratch, rc, m));
                            let codes = &self.arg_pool[codes as usize..codes as usize + m];
                            T::gemv_fold(av, xv, m, n, cv, codes, &mut work[d..d + m]);
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
                    let av: &[T] = dense_slice(inputs, work.as_ptr(), scratch, ra, m * k);
                    let bv: &[T] = dense_slice(inputs, work.as_ptr(), scratch, rb, n * k);
                    match rc {
                        None => T::gemm(av, bv, m, k, n, &mut work[d..d + m * n]),
                        Some((rc, codes)) => {
                            let cv: Option<&[T]> =
                                rc.map(|rc| dense_slice(inputs, work.as_ptr(), scratch, rc, m * n));
                            let codes = &self.arg_pool[codes as usize..codes as usize + m * n];
                            T::gemm_fold(av, bv, m, k, n, cv, codes, &mut work[d..d + m * n]);
                        }
                    }
                    continue;
                }
                Op::SolveMany { a, b, n, k } => {
                    let (n, k) = (n as usize, k as usize);
                    let (ra, rb) =
                        dense_operands(inputs, work, scratch, &self.arg_pool, a, n * n, b, n * k);
                    let av: &[T] = dense_slice(inputs, work.as_ptr(), scratch, ra, n * n);
                    let bv: &[T] = dense_slice(inputs, work.as_ptr(), scratch, rb, n * k);
                    solve_many_t(av, bv, n, k, &mut work[d..d + n * k]);
                    continue;
                }
                Op::Solve { a, b, n } => {
                    let n = n as usize;
                    let (ra, rb) =
                        dense_operands(inputs, work, scratch, &self.arg_pool, a, n * n, b, n);
                    let av: &[T] = dense_slice(inputs, work.as_ptr(), scratch, ra, n * n);
                    let bv: &[T] = dense_slice(inputs, work.as_ptr(), scratch, rb, n);
                    solve_t(av, bv, n, &mut work[d..d + n]);
                    continue;
                }
            };
            work[d] = v;
        }
    }

    /// Output operands, one per root: a slot, or an input when tagged.
    pub fn outputs(&self) -> &[u32] {
        &self.outputs
    }

    /// Drive a backend over the instruction stream.
    pub fn lower(&self, v: &mut dyn TapeVisitor) {
        let pool = |start: u32, len: u32| &self.arg_pool[start as usize..(start + len) as usize];
        let operand = |src: Src, len: u32| match src {
            Src::Inputs(k) => Operand::Inputs(k),
            Src::Pool(start) => Operand::Slots(pool(start, len)),
        };
        for i in 0..self.ops.len() {
            let dst = self.dst[i];
            match self.ops[i] {
                Op::Const(c) => v.constant(dst, c),
                Op::Add(a, b) => v.add(dst, a, b),
                Op::Mul(a, b) => v.mul(dst, a, b),
                Op::MulAdd(a, b, c) => v.mul_add(dst, a, b, c),
                Op::Sub(a, b) => v.sub(dst, a, b),
                Op::Neg(a) => v.neg(dst, a),
                Op::Powi(a, n) => v.powi(dst, a, n),
                Op::Unary(op, a) => v.unary(dst, op, a),
                Op::Binary(op, a, b) => v.binary(dst, op, a, b),
                Op::Cmp(op, a, b) => v.cmp(dst, op, a, b),
                Op::Select(c, t, e) => v.select(dst, c, t, e),
                Op::Reduce(op, s, l) => v.reduce(dst, op, pool(s, l)),
                Op::Dot(s, l) => v.dot(dst, pool(s, l), pool(s + l, l)),
                Op::Call {
                    bundle,
                    start,
                    n_args,
                    n_out,
                } => v.call(
                    dst,
                    &self.bundles[bundle as usize],
                    pool(start, n_args),
                    n_out,
                ),
                Op::CallBatch {
                    bundle,
                    start,
                    n_groups,
                    n_args,
                    n_out,
                } => v.call_batch(
                    dst,
                    &self.bundles[bundle as usize],
                    pool(start, n_groups * n_args),
                    n_groups,
                    n_args,
                    n_out,
                ),
                Op::Gemv { a, x, m, n, acc } => v.gemv(
                    dst,
                    operand(a, m * n),
                    operand(x, n),
                    m,
                    n,
                    acc.map(|f| (f.c.map(|c| operand(c, m)), pool(f.codes, m))),
                ),
                Op::Gemm { a, b, m, k, n, acc } => v.gemm(
                    dst,
                    operand(a, m * k),
                    operand(b, n * k),
                    m,
                    k,
                    n,
                    acc.map(|f| (f.c.map(|c| operand(c, m * n)), pool(f.codes, m * n))),
                ),
                Op::Solve { a, b, n } => v.solve(dst, operand(a, n * n), operand(b, n), n),
                Op::SolveMany { a, b, n, k } => {
                    v.solve_many(dst, operand(a, n * n), operand(b, n * k), n, k)
                }
            }
        }
    }

    /// Number of instructions (diagnostics; compare against a
    /// [`SpecializedTape::n_ops`] for the shrink factor).
    pub fn n_ops(&self) -> usize {
        self.ops.len()
    }
}

/// A tape with the buffers to run it, for callers that want values rather
/// than memory management: [`Tape::runner`] allocates once, every
/// [`eval`](Runner::eval) writes into the same buffers and lends the outputs
/// out. One runner per thread, since it owns the work buffer.
pub struct Runner<'t, T: Scalar> {
    tape: &'t Tape,
    work: Vec<T>,
    out: Vec<T>,
}

impl<T: Scalar> Runner<'_, T> {
    /// Evaluate at `inputs` and borrow the outputs. No allocation, no copy;
    /// a `to_vec` on the result is the caller's choice, not the tape's.
    pub fn eval(&mut self, inputs: &[T]) -> &[T] {
        self.tape.eval_into(inputs, &mut self.work, &mut self.out);
        &self.out
    }

    /// The parameter-pure prolog, once per parameter binding
    /// ([`Tape::compile_split`]); [`main`](Runner::main) then runs per
    /// iteration over the same buffer.
    pub fn prolog(&mut self, inputs: &[T]) {
        self.tape.eval_prolog_into(inputs, &mut self.work);
    }

    /// The main phase over the buffer [`prolog`](Runner::prolog) prepared.
    pub fn main(&mut self, inputs: &[T]) -> &[T] {
        self.tape
            .eval_main_into(inputs, &mut self.work, &mut self.out);
        &self.out
    }

    /// The outputs of the last evaluation.
    pub fn outputs(&self) -> &[T] {
        &self.out
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
enum Dense {
    Inputs(usize),
    Scratch(usize),
    /// A run of consecutive work slots, read in place.
    Work(usize),
}

/// The slice a resolved dense operand names, `len` values long. `work` is
/// the work array's base pointer: a run read in place is disjoint from the
/// kernel's output block (the allocator gives a kernel a fresh block), so
/// the read may overlap the `&mut` the kernel holds on its outputs.
fn dense_slice<'a, T: Scalar>(
    inputs: &'a [T],
    work: *const T,
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

/// Resolve a kernel's two dense operands: an input run that the inputs
/// reach and a run of consecutive work slots are read in place; anything
/// else is gathered into the scratch, `a` first, then `b`.
#[allow(clippy::too_many_arguments)]
/// Where a dense operand is read from: in place (inputs, or a consecutive
/// run of work slots), or gathered into `scratch` from `*at` on, which
/// advances past it.
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
        Src::Pool(start) => {
            let run = &pool[start as usize..start as usize + len];
            let consecutive = len > 0
                && input_index(run[0]).is_none()
                && run.iter().enumerate().all(|(j, &s)| s == run[0] + j as u32);
            if consecutive {
                return Dense::Work(run[0] as usize);
            }
            for j in 0..len {
                scratch[*at + j] = read(inputs, work, run[j]);
            }
            *at += len;
            Dense::Scratch(*at - len)
        }
    }
}

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
                    None => "-".to_string(),
                },
                text.join(",")
            )
        }
    }
}
