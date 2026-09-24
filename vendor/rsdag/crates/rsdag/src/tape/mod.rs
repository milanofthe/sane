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

/// The `state` of a call that keeps none: the bundle runs whole.
pub const NO_STATE: u32 = u32::MAX;

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
    /// `dst .. dst+n_out`; with a `state` slot (not [`NO_STATE`]), the
    /// bundle's main phase over the instance state there.
    Call {
        bundle: u32,
        start: u32,
        n_args: u32,
        n_out: u32,
        state: u32,
    },
    /// `bundles[b]` on `n_groups` argument groups laid group-major in the
    /// pool, group `g`'s outputs to `dst + g*n_out ..`; with a `state`
    /// slot, group `g`'s state at `state + g*state_len`.
    CallBatch {
        bundle: u32,
        start: u32,
        n_groups: u32,
        n_args: u32,
        n_out: u32,
        state: u32,
    },
    /// The prolog of `bundles[b]` for `n_groups` instances, their pure
    /// arguments group-major at `arg_pool[start ..]`, `n_pure` per group:
    /// instance `g`'s state to `dst + g*state_len ..`.
    CallProlog {
        bundle: u32,
        start: u32,
        n_groups: u32,
        n_pure: u32,
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

/// One op as a diagram draws it: its label and kind, the operands it
/// reads (slots, or inputs tagged with [`INPUT`]), how many slots it writes
/// from its destination, and the bundle a call calls.
pub(crate) struct OpView {
    pub label: String,
    pub kind: crate::dot::Kind,
    pub reads: Vec<u32>,
    pub width: u32,
    pub bundle: Option<u32>,
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
    /// The widest scratch a called bundle asks for ([`ExternBundle::work_len`]),
    /// lent to it from the tail of the work buffer.
    bundle_work: usize,
    bundles: Vec<Arc<dyn ExternBundle>>,
    /// Instruction count of the parameter-pure prolog prefix (0 = no split;
    /// see [`compile_split`](Self::compile_split)).
    prolog_ops: usize,
    /// The prolog's results the main phase reads: `work[..state_len]`
    /// (see [`state_len`](Self::state_len)).
    state_len: usize,
    /// The inputs it was compiled over (`input_syms.len()`).
    n_inputs: usize,
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
    /// Call `b` on `args`, its `n_out` outputs to `dst ..`; with `state`,
    /// its main phase over the instance state at that slot.
    fn call(
        &mut self,
        dst: u32,
        b: &Arc<dyn ExternBundle>,
        args: &[u32],
        n_out: u32,
        state: Option<u32>,
    );
    /// Call `b` on `n_groups` argument groups (group-major `args`), group
    /// `g`'s outputs to `dst + g*n_out ..`; with `state`, group `g`'s state
    /// at `state + g * b.state_len()`.
    #[allow(clippy::too_many_arguments)]
    fn call_batch(
        &mut self,
        dst: u32,
        b: &Arc<dyn ExternBundle>,
        args: &[u32],
        n_groups: u32,
        n_args: u32,
        n_out: u32,
        state: Option<u32>,
    );
    /// The prolog of `b` for `n_groups` instances (their pure arguments
    /// group-major in `pure`), instance `g`'s state to
    /// `dst + g * b.state_len() ..`.
    fn call_prolog(&mut self, dst: u32, b: &Arc<dyn ExternBundle>, pure: &[u32], n_groups: u32);
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
    /// Op `i` as a diagram draws it (see [`crate::dot`]).
    pub(crate) fn op_view(&self, i: usize) -> OpView {
        use crate::dot::Kind;
        let pool = |start: u32, len: u32| -> Vec<u32> {
            self.arg_pool[start as usize..(start + len) as usize].to_vec()
        };
        let src = |s: Src, len: u32| -> Vec<u32> {
            match s {
                Src::Inputs(k) => (k..k + len).map(|j| j | INPUT).collect(),
                Src::Pool(start) => pool(start, len),
            }
        };
        let acc = |a: Option<Accum>, len: u32| -> Vec<u32> {
            match a {
                Some(Accum { c: Some(c), .. }) => src(c, len),
                _ => Vec::new(),
            }
        };
        let state = |b: u32, at: u32, n: u32| -> Vec<u32> {
            if at == NO_STATE {
                return Vec::new();
            }
            let len = self.bundles[b as usize].state_len() as u32 * n;
            (at..at + len).collect()
        };
        let cmp = |op: CmpOp| match op {
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
        };
        let reduce = |op: ReduceOp| match op {
            ReduceOp::Sum => "sum",
            ReduceOp::Product => "prod",
            ReduceOp::Min => "min",
            ReduceOp::Max => "max",
        };
        let v = |label: String, kind: Kind, reads: Vec<u32>, width: u32| OpView {
            label,
            kind,
            reads,
            width,
            bundle: None,
        };
        match self.ops[i] {
            Op::Const(c) => v(crate::dot::number(c), Kind::Const, Vec::new(), 1),
            Op::Add(a, b) => v("+".into(), Kind::Op, vec![a, b], 1),
            Op::Mul(a, b) => v("*".into(), Kind::Op, vec![a, b], 1),
            Op::MulAdd(a, b, c) => v("*+".into(), Kind::Op, vec![a, b, c], 1),
            Op::Sub(a, b) => v("-".into(), Kind::Op, vec![a, b], 1),
            Op::Neg(a) => v("neg".into(), Kind::Op, vec![a], 1),
            Op::Powi(a, n) => v(format!("^{n}"), Kind::Op, vec![a], 1),
            Op::Unary(op, a) => v(op.name().into(), Kind::Op, vec![a], 1),
            Op::Binary(op, a, b) => v(op.name().into(), Kind::Op, vec![a, b], 1),
            Op::Cmp(op, a, b) => v(cmp(op).into(), Kind::Choice, vec![a, b], 1),
            Op::Select(c, t, e) => v("select".into(), Kind::Choice, vec![c, t, e], 1),
            Op::Reduce(op, s, l) => v(reduce(op).into(), Kind::Kernel, pool(s, l), 1),
            Op::Dot(s, l) => v(format!("dot {l}"), Kind::Kernel, pool(s, 2 * l), 1),
            Op::Call {
                bundle,
                start,
                n_args,
                n_out,
                state: at,
            } => {
                let mut r = pool(start, n_args);
                r.extend(state(bundle, at, 1));
                OpView {
                    bundle: Some(bundle),
                    ..v("call".into(), Kind::Call, r, n_out)
                }
            }
            Op::CallBatch {
                bundle,
                start,
                n_groups,
                n_args,
                n_out,
                state: at,
            } => {
                let mut r = pool(start, n_groups * n_args);
                r.extend(state(bundle, at, n_groups));
                OpView {
                    bundle: Some(bundle),
                    ..v(format!("call x{n_groups}"), Kind::Call, r, n_groups * n_out)
                }
            }
            Op::CallProlog {
                bundle,
                start,
                n_groups,
                n_pure,
            } => {
                let w = self.bundles[bundle as usize].state_len() as u32 * n_groups;
                OpView {
                    bundle: Some(bundle),
                    ..v(
                        format!("prolog x{n_groups}"),
                        Kind::Call,
                        pool(start, n_groups * n_pure),
                        w,
                    )
                }
            }
            Op::Gemv { a, x, m, n, acc: c } => {
                let mut r = src(a, m * n);
                r.extend(src(x, n));
                r.extend(acc(c, m));
                v(format!("gemv {m}x{n}"), Kind::Kernel, r, m)
            }
            Op::Gemm {
                a,
                b,
                m,
                k,
                n,
                acc: c,
            } => {
                let mut r = src(a, m * k);
                r.extend(src(b, n * k));
                r.extend(acc(c, m * n));
                v(format!("gemm {m}x{k}x{n}"), Kind::Kernel, r, m * n)
            }
            Op::Solve { a, b, n } => {
                let mut r = src(a, n * n);
                r.extend(src(b, n));
                v(format!("solve {n}"), Kind::Kernel, r, n)
            }
            Op::SolveMany { a, b, n, k } => {
                let mut r = src(a, n * n);
                r.extend(src(b, n * k));
                v(format!("solve {n}, {k} rhs"), Kind::Kernel, r, n * k)
            }
        }
    }

    /// The instruction count and destinations a diagram walks.
    pub(crate) fn op_dst(&self, i: usize) -> u32 {
        self.dst[i]
    }

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
                    state,
                } => format!(
                    "Call(b{bundle}, [{}]{}) -> {n_out}",
                    list(start, n_args),
                    state_text(state)
                ),
                Op::CallBatch {
                    bundle,
                    start,
                    n_groups,
                    n_args,
                    n_out,
                    state,
                } => format!(
                    "CallBatch(b{bundle}, {n_groups} x [{}]{}) -> {n_groups} x {n_out}",
                    list(start, n_groups * n_args),
                    state_text(state)
                ),
                Op::CallProlog {
                    bundle,
                    start,
                    n_groups,
                    n_pure,
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
        self.n_work + self.max_args + self.bundle_work
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
                    state,
                } => {
                    for (j, &k) in pool(start, n_args).iter().enumerate() {
                        scratch[j] = g(k);
                    }
                    let b = &*self.bundles[bundle as usize];
                    let n_args = n_args as usize;
                    if state == NO_STATE {
                        let (args, bwork) = scratch.split_at_mut(self.max_args);
                        T::call_bundle_whole(
                            b,
                            &args[..n_args],
                            bwork,
                            &mut work[d..d + n_out as usize],
                        );
                    } else {
                        let (args, bwork) = scratch.split_at_mut(self.max_args);
                        let (st, out) =
                            state_and_out(work, state as usize, b.state_len(), d, n_out as usize);
                        T::call_bundle_main(b, &args[..n_args], st, bwork, out);
                    }
                    continue;
                }
                Op::CallBatch {
                    bundle,
                    start,
                    n_groups,
                    n_args,
                    n_out,
                    state,
                } => {
                    let flat = (n_groups * n_args) as usize;
                    for (j, &k) in pool(start, n_groups * n_args).iter().enumerate() {
                        scratch[j] = g(k);
                    }
                    let b = &*self.bundles[bundle as usize];
                    let (n_args, n_out) = (n_args as usize, n_out as usize);
                    if state == NO_STATE {
                        T::call_bundle_batch(
                            b,
                            &scratch[..flat],
                            n_groups as usize,
                            n_args,
                            &mut work[d..d + n_groups as usize * n_out],
                        );
                    } else {
                        let (args, bwork) = scratch.split_at_mut(self.max_args);
                        let sl = b.state_len();
                        for gi in 0..n_groups as usize {
                            let (st, out) = state_and_out(
                                work,
                                state as usize + gi * sl,
                                sl,
                                d + gi * n_out,
                                n_out,
                            );
                            let a = &args[gi * n_args..(gi + 1) * n_args];
                            T::call_bundle_main(b, a, st, bwork, out);
                        }
                    }
                    continue;
                }
                Op::CallProlog {
                    bundle,
                    start,
                    n_groups,
                    n_pure,
                } => {
                    for (j, &k) in pool(start, n_groups * n_pure).iter().enumerate() {
                        scratch[j] = g(k);
                    }
                    let b = &*self.bundles[bundle as usize];
                    let (args, bwork) = scratch.split_at_mut(self.max_args);
                    let (sl, np) = (b.state_len(), n_pure as usize);
                    for gi in 0..n_groups as usize {
                        let st = &mut work[d + gi * sl..d + (gi + 1) * sl];
                        T::call_bundle_prolog(b, &args[gi * np..(gi + 1) * np], bwork, st);
                    }
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
                Op::SolveMany { a, b, n, k } => {
                    let (n, k) = (n as usize, k as usize);
                    let (ra, rb) =
                        dense_operands(inputs, work, scratch, &self.arg_pool, a, n * n, b, n * k);
                    let base = work.as_mut_ptr();
                    let av: &[T] = dense_slice(inputs, base, scratch, ra, n * n);
                    let bv: &[T] = dense_slice(inputs, base, scratch, rb, n * k);
                    let reads = [Some((ra, n * n)), Some((rb, n * k)), None];
                    let out = unsafe { out_block(base, work.len(), d, n * k, &reads) };
                    solve_many_t(av, bv, n, k, out);
                    continue;
                }
                Op::Solve { a, b, n } => {
                    let n = n as usize;
                    let (ra, rb) =
                        dense_operands(inputs, work, scratch, &self.arg_pool, a, n * n, b, n);
                    let base = work.as_mut_ptr();
                    let av: &[T] = dense_slice(inputs, base, scratch, ra, n * n);
                    let bv: &[T] = dense_slice(inputs, base, scratch, rb, n);
                    let reads = [Some((ra, n * n)), Some((rb, n)), None];
                    let out = unsafe { out_block(base, work.len(), d, n, &reads) };
                    solve_t(av, bv, n, out);
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
                    state,
                } => v.call(
                    dst,
                    &self.bundles[bundle as usize],
                    pool(start, n_args),
                    n_out,
                    (state != NO_STATE).then_some(state),
                ),
                Op::CallBatch {
                    bundle,
                    start,
                    n_groups,
                    n_args,
                    n_out,
                    state,
                } => v.call_batch(
                    dst,
                    &self.bundles[bundle as usize],
                    pool(start, n_groups * n_args),
                    n_groups,
                    n_args,
                    n_out,
                    (state != NO_STATE).then_some(state),
                ),
                Op::CallProlog {
                    bundle,
                    start,
                    n_groups,
                    n_pure,
                } => v.call_prolog(
                    dst,
                    &self.bundles[bundle as usize],
                    pool(start, n_groups * n_pure),
                    n_groups,
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
fn state_text(state: u32) -> String {
    if state == NO_STATE {
        String::new()
    } else {
        format!(", state @{state}")
    }
}

/// A call's instance state `work[s .. s+len]` and its output block
/// `work[d .. d+n]`, which the allocator keeps apart.
fn state_and_out<T>(work: &mut [T], s: usize, len: usize, d: usize, n: usize) -> (&[T], &mut [T]) {
    if s + len <= d {
        let (a, b) = work.split_at_mut(d);
        (&a[s..s + len], &mut b[..n])
    } else {
        assert!(d + n <= s, "a call's state overlaps its outputs");
        let (a, b) = work.split_at_mut(s);
        (&b[..len], &mut a[d..d + n])
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
                    None => "-".to_string(),
                },
                text.join(",")
            )
        }
    }
}
