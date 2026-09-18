//! The op stream: the tape lowered through [`TapeVisitor`] into a flat
//! vector the emitter walks. Recording decouples the chunk partitioning and
//! the parallel per-chunk codegen from the visitor callback structure.
//! Bundle bodies are interned by `Arc` identity into a table the compiled
//! code indexes through the pointer every chunk receives. Operands are the
//! tape's: slots, or inputs when tagged (see [`rsdag::tape::INPUT`]).

use rsdag::extern_fn::ExternBundle;
use rsdag::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use rsdag::tape::{input_index, Operand};
use rsdag::TapeVisitor;
use rustc_hash::FxHashMap;
use std::sync::Arc;

pub(crate) enum ROp {
    Const(u32, f64),
    Add(u32, u32, u32),
    Mul(u32, u32, u32),
    MulAdd(u32, u32, u32, u32),
    Sub(u32, u32, u32),
    Neg(u32, u32),
    Powi(u32, u32, i32),
    Unary(u32, UnaryOp, u32),
    Binary(u32, BinOp, u32, u32),
    Cmp(u32, CmpOp, u32, u32),
    Select(u32, u32, u32, u32),
    Reduce(u32, ReduceOp, Vec<u32>),
    Dot(u32, Vec<u32>, Vec<u32>),
    /// `(dst, bundle, args, n_out)`.
    Call(u32, u32, Vec<u32>, u32),
    /// `(dst, bundle, args, n_groups, n_args, n_out)`.
    CallBatch(u32, u32, Vec<u32>, u32, u32, u32),
    Gemv {
        dst: u32,
        a: Dense,
        x: Dense,
        m: u32,
        n: u32,
        /// The accumulator operand, the fold codes and, once compiled,
        /// the address of the codes' table.
        acc: Option<(Option<Dense>, Vec<u32>, usize)>,
    },
    Gemm {
        dst: u32,
        a: Dense,
        b: Dense,
        m: u32,
        k: u32,
        n: u32,
        /// The accumulator operand, the fold codes and, once compiled,
        /// the address of the codes' table.
        acc: Option<(Option<Dense>, Vec<u32>, usize)>,
    },
    Solve {
        dst: u32,
        a: Dense,
        b: Dense,
        n: u32,
    },
    SolveMany {
        dst: u32,
        a: Dense,
        b: Dense,
        n: u32,
        k: u32,
    },
}

/// A dense operand: a run of inputs read in place, or slots gathered.
pub(crate) enum Dense {
    Inputs(u32),
    Slots(Vec<u32>),
}

impl Dense {
    pub(crate) fn slots(&self) -> &[u32] {
        match self {
            Dense::Inputs(_) => &[],
            Dense::Slots(v) => v,
        }
    }
    fn of(o: Operand<'_>) -> Dense {
        match o {
            Operand::Inputs(k) => Dense::Inputs(k),
            Operand::Slots(s) => Dense::Slots(s.to_vec()),
        }
    }
}

impl ROp {
    /// Every operand the op reads, slots and tagged inputs alike.
    pub(crate) fn for_each_operand(&self, mut f: impl FnMut(u32)) {
        match self {
            ROp::Const(..) => {}
            ROp::Neg(_, a) | ROp::Powi(_, a, _) | ROp::Unary(_, _, a) => f(*a),
            ROp::Add(_, a, b)
            | ROp::Mul(_, a, b)
            | ROp::Sub(_, a, b)
            | ROp::Cmp(_, _, a, b)
            | ROp::Binary(_, _, a, b) => {
                f(*a);
                f(*b);
            }
            ROp::MulAdd(_, a, b, c) | ROp::Select(_, a, b, c) => {
                f(*a);
                f(*b);
                f(*c);
            }
            ROp::Reduce(_, _, args) | ROp::Call(_, _, args, _) | ROp::CallBatch(_, _, args, ..) => {
                args.iter().copied().for_each(f)
            }
            ROp::Dot(_, a, b) => a.iter().chain(b).copied().for_each(f),
            ROp::Gemv { a, x, acc, .. } => a
                .slots()
                .iter()
                .chain(x.slots())
                .chain(
                    acc.iter()
                        .flat_map(|(c, _, _)| c.as_ref().map_or(&[][..], |c| c.slots())),
                )
                .copied()
                .for_each(f),
            ROp::Gemm { a, b, acc, .. } => a
                .slots()
                .iter()
                .chain(b.slots())
                .chain(
                    acc.iter()
                        .flat_map(|(c, _, _)| c.as_ref().map_or(&[][..], |c| c.slots())),
                )
                .copied()
                .for_each(f),
            ROp::Solve { a, b, .. } | ROp::SolveMany { a, b, .. } => {
                a.slots().iter().chain(b.slots()).copied().for_each(f)
            }
        }
    }
    /// Every work slot the op reads.
    pub(crate) fn for_each_read(&self, mut f: impl FnMut(u32)) {
        self.for_each_operand(|k| {
            if input_index(k).is_none() {
                f(k)
            }
        });
    }
    /// The host routine the op calls, if any.
    pub(crate) fn host(&self) -> Option<*const ()> {
        Some(match self {
            ROp::Unary(_, op, _) => match op {
                UnaryOp::Sqrt
                | UnaryOp::Floor
                | UnaryOp::Ceil
                | UnaryOp::Trunc
                | UnaryOp::Abs
                | UnaryOp::Sign => return None,
                _ => crate::host::unary_addr(*op).0,
            },
            ROp::Binary(..) => crate::host::h_binary as *const (),
            ROp::Powi(_, _, n) if *n != -1 && *n != 2 => crate::host::h_powi as *const (),
            ROp::Reduce(_, ReduceOp::Min | ReduceOp::Max, _) => crate::host::h_reduce as *const (),
            ROp::Call(..) => crate::host::h_bundle as *const (),
            ROp::CallBatch(..) => crate::host::h_bundle_batch as *const (),
            ROp::Gemv { acc: None, .. } => crate::host::h_gemv as *const (),
            ROp::Gemv { .. } => crate::host::h_gemv_acc as *const (),
            ROp::Gemm { acc: None, .. } => crate::host::h_gemm as *const (),
            ROp::Gemm { .. } => crate::host::h_gemm_acc as *const (),
            ROp::Solve { .. } => crate::host::h_solve as *const (),
            ROp::SolveMany { .. } => crate::host::h_solve_many as *const (),
            _ => return None,
        })
    }
    /// How many values the op hands to a host routine through the gather
    /// area of the work array.
    pub(crate) fn gather_len(&self) -> usize {
        match self {
            ROp::Reduce(_, ReduceOp::Min | ReduceOp::Max, args) => args.len(),
            ROp::Call(_, _, args, _) | ROp::CallBatch(_, _, args, ..) => args.len(),
            ROp::Gemv { a, x, acc, .. } => {
                a.slots().len()
                    + x.slots().len()
                    + acc
                        .as_ref()
                        .map_or(0, |(c, _, _)| c.as_ref().map_or(0, |c| c.slots().len()))
            }
            ROp::Gemm { a, b, acc, .. } => {
                a.slots().len()
                    + b.slots().len()
                    + acc
                        .as_ref()
                        .map_or(0, |(c, _, _)| c.as_ref().map_or(0, |c| c.slots().len()))
            }
            ROp::Solve { a, b, .. } | ROp::SolveMany { a, b, .. } => {
                a.slots().len() + b.slots().len()
            }
            _ => 0,
        }
    }
}

#[derive(Default)]
pub(crate) struct Recorder {
    pub(crate) ops: Vec<ROp>,
    pub(crate) bundles: Vec<Arc<dyn ExternBundle>>,
    bundle_idx: FxHashMap<usize, u32>,
}

impl Recorder {
    fn intern(&mut self, b: &Arc<dyn ExternBundle>) -> u32 {
        let key = Arc::as_ptr(b) as *const () as usize;
        *self.bundle_idx.entry(key).or_insert_with(|| {
            self.bundles.push(b.clone());
            (self.bundles.len() - 1) as u32
        })
    }
}

impl TapeVisitor for Recorder {
    fn constant(&mut self, dst: u32, v: f64) {
        self.ops.push(ROp::Const(dst, v));
    }
    fn add(&mut self, dst: u32, a: u32, b: u32) {
        self.ops.push(ROp::Add(dst, a, b));
    }
    fn mul(&mut self, dst: u32, a: u32, b: u32) {
        self.ops.push(ROp::Mul(dst, a, b));
    }
    fn mul_add(&mut self, dst: u32, a: u32, b: u32, c: u32) {
        self.ops.push(ROp::MulAdd(dst, a, b, c));
    }
    fn sub(&mut self, dst: u32, a: u32, b: u32) {
        self.ops.push(ROp::Sub(dst, a, b));
    }
    fn neg(&mut self, dst: u32, a: u32) {
        self.ops.push(ROp::Neg(dst, a));
    }
    fn powi(&mut self, dst: u32, a: u32, n: i32) {
        self.ops.push(ROp::Powi(dst, a, n));
    }
    fn unary(&mut self, dst: u32, op: UnaryOp, a: u32) {
        self.ops.push(ROp::Unary(dst, op, a));
    }
    fn binary(&mut self, dst: u32, op: BinOp, a: u32, b: u32) {
        self.ops.push(ROp::Binary(dst, op, a, b));
    }
    fn cmp(&mut self, dst: u32, op: CmpOp, a: u32, b: u32) {
        self.ops.push(ROp::Cmp(dst, op, a, b));
    }
    fn select(&mut self, dst: u32, c: u32, t: u32, e: u32) {
        self.ops.push(ROp::Select(dst, c, t, e));
    }
    fn reduce(&mut self, dst: u32, op: ReduceOp, args: &[u32]) {
        self.ops.push(ROp::Reduce(dst, op, args.to_vec()));
    }
    fn dot(&mut self, dst: u32, a: &[u32], b: &[u32]) {
        self.ops.push(ROp::Dot(dst, a.to_vec(), b.to_vec()));
    }
    fn call(&mut self, dst: u32, b: &Arc<dyn ExternBundle>, args: &[u32], n_out: u32) {
        let idx = self.intern(b);
        self.ops.push(ROp::Call(dst, idx, args.to_vec(), n_out));
    }
    fn call_batch(
        &mut self,
        dst: u32,
        b: &Arc<dyn ExternBundle>,
        args: &[u32],
        n_groups: u32,
        n_args: u32,
        n_out: u32,
    ) {
        let idx = self.intern(b);
        self.ops.push(ROp::CallBatch(
            dst,
            idx,
            args.to_vec(),
            n_groups,
            n_args,
            n_out,
        ));
    }
    fn gemv(
        &mut self,
        dst: u32,
        a: Operand<'_>,
        x: Operand<'_>,
        m: u32,
        n: u32,
        acc: Option<(Option<Operand<'_>>, &[u32])>,
    ) {
        self.ops.push(ROp::Gemv {
            dst,
            a: Dense::of(a),
            x: Dense::of(x),
            m,
            n,
            acc: acc.map(|(c, codes)| (c.map(Dense::of), codes.to_vec(), 0)),
        });
    }
    fn gemm(
        &mut self,
        dst: u32,
        a: Operand<'_>,
        b: Operand<'_>,
        m: u32,
        k: u32,
        n: u32,
        acc: Option<(Option<Operand<'_>>, &[u32])>,
    ) {
        self.ops.push(ROp::Gemm {
            dst,
            a: Dense::of(a),
            b: Dense::of(b),
            m,
            k,
            n,
            acc: acc.map(|(c, codes)| (c.map(Dense::of), codes.to_vec(), 0)),
        });
    }
    fn solve_many(&mut self, dst: u32, a: Operand<'_>, b: Operand<'_>, n: u32, k: u32) {
        self.ops.push(ROp::SolveMany {
            dst,
            a: Dense::of(a),
            b: Dense::of(b),
            n,
            k,
        });
    }
    fn solve(&mut self, dst: u32, a: Operand<'_>, b: Operand<'_>, n: u32) {
        self.ops.push(ROp::Solve {
            dst,
            a: Dense::of(a),
            b: Dense::of(b),
            n,
        });
    }
}
