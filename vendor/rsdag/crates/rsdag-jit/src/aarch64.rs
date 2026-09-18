//! AArch64 encodings (AAPCS64, which Windows on ARM shares).
//!
//! `x19` work, `x20` inputs, `x21` bundles; `x9` holds the current 32 KB
//! window of the work array, `x10` is scratch, `x16` the call target, and
//! `x22` to `x27` hold the chunk's frequent host routines; `d0` and `d1`
//! carry call arguments and `d0` the result. The cache is `d8` to `d15`,
//! callee-saved and so kept across host calls, then `d16` to `d31`.
//!
//! The scaled 12-bit offset of `ldr` reaches 32 KB; beyond that the window
//! base in `x9` is moved only when an access leaves the current window, so
//! an access is one instruction almost always instead of four.

use crate::isa::*;
use rsdag::node::{CmpOp, ReduceOp};

const WORK: u32 = 19;
const INPUTS: u32 = 20;
const BUNDLES: u32 = 21;

// Condition codes after `fcmp`. The unordered case sets C and V and clears N
// and Z; these choices make every comparison with a NaN false except `ne`.
const COND_EQ: u32 = 0x0;
const COND_NE: u32 = 0x1;
const COND_MI: u32 = 0x4; // lt
const COND_LS: u32 = 0x9; // le
const COND_GE: u32 = 0xA;
const COND_GT: u32 = 0xC;

pub(crate) struct A64 {
    code: Vec<u32>,
    /// Byte offset of the work-array window `x9` currently points at.
    window: Option<usize>,
    /// Host routines held in `x22` to `x27`.
    hot: Vec<*const ()>,
}

const WINDOW: usize = 4096 * 8;
const PREFETCH_EVERY: usize = 128;
const PREFETCH_AHEAD: usize = 2048;
const HOT_LO: u32 = 22;
const HOT_N: usize = 6;

impl A64 {
    fn w(&mut self, word: u32) {
        self.code.push(word);
        // A large program streams through the front end once, so every
        // 128 bytes a `prfm plil2keep` asks for the code 2 KB ahead; that
        // is a third off a 7M-op program and nothing on a small one.
        if self.code.len().is_multiple_of(PREFETCH_EVERY / 4) {
            let imm19 = (PREFETCH_AHEAD / 4) as u32;
            self.code.push(0xD800_0000 | (imm19 << 5) | 0b01010);
        }
    }
    /// `movz`/`movk` a 64-bit immediate into `Xd`.
    fn mov_imm(&mut self, xd: u32, v: u64) {
        let mut first = true;
        for hw in 0..4u32 {
            let part = ((v >> (16 * hw)) & 0xFFFF) as u32;
            if part == 0 && !(first && hw == 3) {
                continue;
            }
            let op = if first { 0xD280_0000 } else { 0xF280_0000 };
            self.w(op | (hw << 21) | (part << 5) | xd);
            first = false;
        }
        if first {
            self.w(0xD280_0000 | xd);
        }
    }
    fn base(b: Base) -> u32 {
        match b {
            Base::Work => WORK,
            Base::Inputs => INPUTS,
        }
    }
    /// `ldr`/`str Dt, [Xn, #off]` for any offset; for the work array, through
    /// the window in `x9` when the offset is out of immediate reach.
    fn mem(&mut self, opc_imm: u32, opc_reg: u32, dt: u32, xn: u32, off: usize) {
        if off / 8 < 4096 {
            self.w(opc_imm | (((off / 8) as u32) << 10) | (xn << 5) | dt);
        } else if xn == WORK {
            let base = off - off % WINDOW;
            if self.window != Some(base) {
                if base < (1 << 24) {
                    // add x9, x19, #(base >> 12), lsl #12
                    self.w(0x9140_0000 | (((base >> 12) as u32) << 10) | (WORK << 5) | 9);
                } else {
                    self.mov_imm(10, base as u64);
                    self.w(0x8B00_0000 | (10 << 16) | (WORK << 5) | 9);
                }
                self.window = Some(base);
            }
            self.w(opc_imm | ((((off - base) / 8) as u32) << 10) | (9 << 5) | dt);
        } else {
            self.mov_imm(10, off as u64);
            self.w(opc_reg | (10 << 16) | (xn << 5) | dt);
        }
    }
    fn fbin(&mut self, opc: u32, dd: u32, dn: u32, dm: u32) {
        self.w(opc | (dm << 16) | (dn << 5) | dd);
    }
    fn fcmp(&mut self, dn: u32, dm: u32) {
        self.w(0x1E60_2000 | (dm << 16) | (dn << 5));
    }
    fn fcsel(&mut self, dd: u32, dn: u32, dm: u32, cond: u32) {
        self.w(0x1E60_0C00 | (dm << 16) | (cond << 12) | (dn << 5) | dd);
    }
}

impl Isa for A64 {
    const CACHE: &'static [u8] = &[
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31,
    ];
    const SAVED: usize = 8;
    const RESULT: u8 = 0;

    fn new(hot: &[*const ()]) -> A64 {
        A64 {
            code: Vec::with_capacity(4096),
            window: None,
            hot: hot.iter().copied().take(HOT_N).collect(),
        }
    }
    fn finish(self) -> Vec<u8> {
        self.code.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn prologue(&mut self) {
        self.w(0xA9BF_7BFD); // stp x29, x30, [sp, #-16]!
        self.w(0x9100_03FD); // mov x29, sp
        self.w(0xA9BF_53F3); // stp x19, x20, [sp, #-16]!
        self.w(0xA9BF_5BF5); // stp x21, x22, [sp, #-16]!
        self.w(0xA9BF_63F7); // stp x23, x24, [sp, #-16]!
        self.w(0xA9BF_6BF9); // stp x25, x26, [sp, #-16]!
        self.w(0xA9BF_73FB); // stp x27, x28, [sp, #-16]!
        self.w(0x6DBF_27E8); // stp d8, d9, [sp, #-16]!
        self.w(0x6DBF_2FEA); // stp d10, d11, [sp, #-16]!
        self.w(0x6DBF_37EC); // stp d12, d13, [sp, #-16]!
        self.w(0x6DBF_3FEE); // stp d14, d15, [sp, #-16]!
        self.w(0xAA00_03F3); // mov x19, x0
        self.w(0xAA01_03F4); // mov x20, x1
        self.w(0xAA02_03F5); // mov x21, x2
        for k in 0..self.hot.len() {
            let a = self.hot[k] as usize as u64;
            self.mov_imm(HOT_LO + k as u32, a);
        }
    }
    fn epilogue(&mut self) {
        self.w(0x6CC1_3FEE); // ldp d14, d15, [sp], #16
        self.w(0x6CC1_37EC); // ldp d12, d13, [sp], #16
        self.w(0x6CC1_2FEA); // ldp d10, d11, [sp], #16
        self.w(0x6CC1_27E8); // ldp d8, d9, [sp], #16
        self.w(0xA8C1_73FB); // ldp x27, x28, [sp], #16
        self.w(0xA8C1_6BF9); // ldp x25, x26, [sp], #16
        self.w(0xA8C1_63F7); // ldp x23, x24, [sp], #16
        self.w(0xA8C1_5BF5); // ldp x21, x22, [sp], #16
        self.w(0xA8C1_53F3); // ldp x19, x20, [sp], #16
        self.w(0xA8C1_7BFD); // ldp x29, x30, [sp], #16
        self.w(0xD65F_03C0); // ret
    }

    fn load(&mut self, r: u8, base: Base, off: usize) {
        self.mem(0xFD40_0000, 0xFC60_6800, r as u32, Self::base(base), off);
    }
    fn store(&mut self, r: u8, base: Base, off: usize) {
        self.mem(0xFD00_0000, 0xFC20_6800, r as u32, Self::base(base), off);
    }
    fn fconst(&mut self, r: u8, v: f64) {
        if v.to_bits() == 0 {
            self.w(0x9E67_0000 | (31 << 5) | r as u32); // fmov Dr, xzr
        } else {
            self.mov_imm(10, v.to_bits());
            self.w(0x9E67_0000 | (10 << 5) | r as u32); // fmov Dr, x10
        }
    }
    fn mov(&mut self, d: u8, a: u8) {
        if d != a {
            self.w(0x1E60_4000 | ((a as u32) << 5) | d as u32);
        }
    }

    fn arith(&mut self, op: Arith, d: u8, a: u8, b: u8) {
        let opc = match op {
            Arith::Add => 0x1E60_2800,
            Arith::Sub => 0x1E60_3800,
            Arith::Mul => 0x1E60_0800,
            Arith::Div => 0x1E60_1800,
        };
        self.fbin(opc, d as u32, a as u32, b as u32);
    }
    fn neg(&mut self, d: u8, a: u8) {
        self.w(0x1E61_4000 | ((a as u32) << 5) | d as u32);
    }
    const MINMAX: bool = true;
    fn minmax(&mut self, op: ReduceOp, d: u8, a: u8, b: u8) {
        // fminnm / fmaxnm: IEEE minNum / maxNum, what `f64::min` and
        // `f64::max` compile to on this target.
        let opc = match op {
            ReduceOp::Min => 0x1E60_7800,
            _ => 0x1E60_6800,
        };
        self.fbin(opc, d as u32, a as u32, b as u32);
    }
    fn abs(&mut self, d: u8, a: u8) {
        self.w(0x1E60_C000 | ((a as u32) << 5) | d as u32);
    }
    fn sqrt(&mut self, d: u8, a: u8) {
        self.w(0x1E61_C000 | ((a as u32) << 5) | d as u32);
    }
    fn round(&mut self, mode: Round, d: u8, a: u8) -> bool {
        let opc = match mode {
            Round::Floor => 0x1E65_4000, // frintm
            Round::Ceil => 0x1E64_C000,  // frintp
            Round::Trunc => 0x1E65_C000, // frintz
        };
        self.w(opc | ((a as u32) << 5) | d as u32);
        true
    }

    fn cmp_select(&mut self, op: CmpOp, a: u8, b: u8, t: u8, e: u8, d: u8) {
        self.fcmp(a as u32, b as u32);
        let cond = match op {
            CmpOp::Gt => COND_GT,
            CmpOp::Ge => COND_GE,
            CmpOp::Lt => COND_MI,
            CmpOp::Le => COND_LS,
            CmpOp::Eq => COND_EQ,
            CmpOp::Ne => COND_NE,
        };
        self.fcsel(d as u32, t as u32, e as u32, cond);
    }
    fn select_nz(&mut self, c: u8, t: u8, e: u8, d: u8) {
        self.w(0x1E60_2008 | ((c as u32) << 5)); // fcmp Dc, #0.0
        self.fcsel(d as u32, t as u32, e as u32, COND_NE);
    }

    fn call(&mut self, addr: *const (), args: &[Arg]) {
        // Floats and integers are counted apart: d0.. and x0.. in order.
        let (mut nf, mut ni) = (0u8, 0u32);
        for arg in args {
            match *arg {
                Arg::F(r) => {
                    self.mov(nf, r);
                    nf += 1;
                }
                Arg::I(iarg) => {
                    let xk = ni;
                    ni += 1;
                    match iarg {
                        IArg::Imm(v) => self.mov_imm(xk, v),
                        IArg::WorkAddr(off) if off < 4096 => {
                            self.w(0x9100_0000 | ((off as u32) << 10) | (WORK << 5) | xk);
                        }
                        IArg::WorkAddr(off) => {
                            self.mov_imm(10, off as u64);
                            self.w(0x8B00_0000 | (10 << 16) | (WORK << 5) | xk);
                        }
                        IArg::InputAddr(off) if off < 4096 => {
                            self.w(0x9100_0000 | ((off as u32) << 10) | (INPUTS << 5) | xk);
                        }
                        IArg::InputAddr(off) => {
                            self.mov_imm(10, off as u64);
                            self.w(0x8B00_0000 | (10 << 16) | (INPUTS << 5) | xk);
                        }
                        IArg::Bundles => self.w(0xAA00_03E0 | (BUNDLES << 16) | xk),
                    }
                }
            }
        }
        match self.hot.iter().position(|&h| h == addr) {
            Some(k) => self.w(0xD63F_0000 | ((HOT_LO + k as u32) << 5)), // blr xk
            None => {
                self.mov_imm(16, addr as usize as u64);
                self.w(0xD63F_0000 | (16 << 5)); // blr x16
            }
        }
        self.window = None; // x9 is caller-saved
    }
}
