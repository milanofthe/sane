//! The instruction layer an architecture provides to the emitter.
//!
//! The emitter in [`native`](crate::native) is architecture-neutral: it keeps
//! the value cache, walks the op stream and decides what to compute in which
//! register. Everything that is a matter of encoding goes through this trait,
//! so a second architecture is one file of encodings and nothing else.
//!
//! Registers are the architecture's own float register numbers. A result
//! register handed to a method is never one of its operands, except for
//! [`arith`](Isa::arith), whose result may alias the first operand (the
//! folds accumulate in place).

use rsdag::node::{CmpOp, ReduceOp};

/// Which base pointer a memory operand is relative to.
#[derive(Clone, Copy)]
pub(crate) enum Base {
    /// The work array (slots, bundle scratch, gather area).
    Work,
    /// The input vector.
    Inputs,
}

/// An integer argument of a host call.
#[derive(Clone, Copy)]
pub(crate) enum IArg {
    Imm(u64),
    /// The address of a byte offset into the work array.
    WorkAddr(usize),
    /// The address of a byte offset into the input vector.
    InputAddr(usize),
    /// The bundle table pointer the chunk received.
    Bundles,
}

/// One argument of a host call, in the routine's signature order: the
/// conventions differ in whether floats and integers are counted together.
#[derive(Clone, Copy)]
pub(crate) enum Arg {
    F(u8),
    I(IArg),
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy)]
pub(crate) enum Round {
    Floor,
    Ceil,
    Trunc,
}

pub(crate) trait Isa {
    /// The registers the value cache may use, in eviction order. The first
    /// `SAVED` of them survive a host call.
    const CACHE: &'static [u8];
    const SAVED: usize;
    /// Where a host call leaves its f64 result.
    const RESULT: u8;

    /// `hot` lists host routines the chunk calls often, most frequent first;
    /// an architecture keeps as many of them in callee-saved registers as it
    /// has to spare, so those calls need no address immediate.
    fn new(hot: &[*const ()]) -> Self;
    fn finish(self) -> Vec<u8>;

    /// Function entry: `fn(work: *mut f64, inputs: *const f64, bundles: *const _)`
    /// in the platform's C convention, the three pointers kept in callee-saved
    /// registers for the chunk's lifetime.
    fn prologue(&mut self);
    fn epilogue(&mut self);

    fn load(&mut self, r: u8, base: Base, off: usize);
    fn store(&mut self, r: u8, base: Base, off: usize);
    fn fconst(&mut self, r: u8, v: f64);
    fn mov(&mut self, d: u8, a: u8);

    /// `d = a op b`; `d` may alias `a`.
    fn arith(&mut self, op: Arith, d: u8, a: u8, b: u8);
    fn neg(&mut self, d: u8, a: u8);
    fn abs(&mut self, d: u8, a: u8);
    fn sqrt(&mut self, d: u8, a: u8);
    /// `false` when the instruction is not available; the caller then uses
    /// the host routine.
    fn round(&mut self, mode: Round, d: u8, a: u8) -> bool;
    /// Whether [`minmax`](Self::minmax) is available; without it the
    /// caller uses the host routine.
    const MINMAX: bool;
    /// `d = min(a, b)` or `max(a, b)` with the reference's NaN rule (a NaN
    /// operand yields the other), as one instruction.
    fn minmax(&mut self, op: ReduceOp, d: u8, a: u8, b: u8);

    /// `d = (a op b) ? t : e`, every comparison with a NaN false except `Ne`.
    fn cmp_select(&mut self, op: CmpOp, a: u8, b: u8, t: u8, e: u8, d: u8);
    /// `d = (c != 0) ? t : e` (a NaN condition selects `t`).
    fn select_nz(&mut self, c: u8, t: u8, e: u8, d: u8);

    /// Call `addr` with `args` in signature order; an f64 result is in
    /// `RESULT`.
    fn call(&mut self, addr: *const (), args: &[Arg]);
}
