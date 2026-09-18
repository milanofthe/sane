//! x86-64 encodings, System V and Windows x64.
//!
//! `rbx` work, `r13` inputs, `r14` bundles; `rax` scratch; `xmm0` and
//! `xmm1` carry call arguments, the result and the masks of a select. The
//! cache is `xmm2` to `xmm15`. Under System V every vector register is
//! caller-saved, so nothing survives a host call; under Windows `xmm6` to
//! `xmm15` are callee-saved, so the chunk preserves them and they come
//! first in the cache. Baseline SSE2, with `roundsd` when SSE4.1 is present
//!.

use crate::isa::*;
use rsdag::node::{CmpOp, ReduceOp};

const WORK: u8 = 3; // rbx
const INPUTS: u8 = 13; // r13
const BUNDLES: u8 = 14; // r14
const WIN: bool = cfg!(windows);
/// Integer argument registers: rdi rsi rdx rcx r8 r9, or rcx rdx r8 r9.
const INT_ARGS: &[u8] = if WIN {
    &[1, 2, 8, 9]
} else {
    &[7, 6, 2, 1, 8, 9]
};
/// The Windows frame below the six pushes: `xmm6` to `xmm15` and 8 bytes of
/// alignment. Shadow space and stack arguments are not in it -- a call
/// reserves what it needs (see [`X64::call`]), so no frame has to guess how
/// wide the widest host routine is.
const WIN_FRAME: i32 = 168;
/// Windows wants 32 bytes of shadow space below every call's arguments.
const WIN_SHADOW: i32 = 32;

pub(crate) struct X64 {
    code: Vec<u8>,
    sse41: bool,
    /// Host routines held in `r12` and `r15`.
    hot: Vec<*const ()>,
}

const HOT_REGS: [u8; 2] = [12, 15];

impl X64 {
    fn b(&mut self, byte: u8) {
        self.code.push(byte);
    }
    fn bytes(&mut self, bs: &[u8]) {
        self.code.extend_from_slice(bs);
    }
    /// A REX prefix when any of its bits is set.
    fn rex(&mut self, w: bool, reg: u8, rm: u8) {
        let v = 0x40 | ((w as u8) << 3) | ((reg >> 3) << 2) | (rm >> 3);
        if v != 0x40 {
            self.b(v);
        }
    }
    fn modrm_reg(&mut self, reg: u8, rm: u8) {
        self.b(0xC0 | ((reg & 7) << 3) | (rm & 7));
    }
    /// `[base + disp]` for `rbx` and `r13` (neither needs a SIB byte).
    fn modrm_mem(&mut self, reg: u8, base: u8, disp: usize) {
        let disp = i32::try_from(disp).expect("work array offset fits i32");
        if let Ok(d8) = i8::try_from(disp) {
            self.b(0x40 | ((reg & 7) << 3) | (base & 7));
            self.b(d8 as u8);
        } else {
            self.b(0x80 | ((reg & 7) << 3) | (base & 7));
            self.bytes(&disp.to_le_bytes());
        }
    }
    /// A legacy-SSE register-register instruction `prefix 0F op /r`.
    fn sse(&mut self, prefix: u8, op: u8, reg: u8, rm: u8) {
        self.b(prefix);
        self.rex(false, reg, rm);
        self.bytes(&[0x0F, op]);
        self.modrm_reg(reg, rm);
    }
    fn sse_mem(&mut self, prefix: u8, op: u8, reg: u8, base: u8, disp: usize) {
        self.b(prefix);
        self.rex(false, reg, base);
        self.bytes(&[0x0F, op]);
        self.modrm_mem(reg, base, disp);
    }
    /// `mov r64, imm64` (or the shorter zero-extending `mov r32, imm32`).
    fn mov_imm(&mut self, rd: u8, v: u64) {
        if let Ok(v32) = u32::try_from(v) {
            self.rex(false, 0, rd);
            self.b(0xB8 | (rd & 7));
            self.bytes(&v32.to_le_bytes());
        } else {
            self.rex(true, 0, rd);
            self.b(0xB8 | (rd & 7));
            self.bytes(&v.to_le_bytes());
        }
    }
    /// `[rsp + disp]`, which needs a SIB byte.
    fn modrm_rsp(&mut self, reg: u8, disp: i32) {
        if let Ok(d8) = i8::try_from(disp) {
            self.bytes(&[0x40 | ((reg & 7) << 3) | 4, 0x24, d8 as u8]);
        } else {
            self.bytes(&[0x80 | ((reg & 7) << 3) | 4, 0x24]);
            self.bytes(&disp.to_le_bytes());
        }
    }
    /// `movdqu [rsp + disp], xmm` (`store`) or the load.
    fn xmm_rsp(&mut self, store: bool, xmm: u8, disp: i32) {
        self.b(0xF3);
        self.rex(false, xmm, 0);
        self.bytes(&[0x0F, if store { 0x7F } else { 0x6F }]);
        self.modrm_rsp(xmm, disp);
    }
    /// `add rsp, n`, or `sub rsp, -n`. `n` stays a multiple of 16 so that
    /// `rsp` keeps the alignment every call below relies on.
    fn move_rsp(&mut self, n: i32) {
        debug_assert_eq!(n % 16, 0, "rsp moves in 16-byte steps");
        self.bytes(if n >= 0 {
            &[0x48, 0x81, 0xC4]
        } else {
            &[0x48, 0x81, 0xEC]
        });
        self.bytes(&n.abs().to_le_bytes());
    }
    /// `mov [rsp + disp], r64`.
    fn store_rsp(&mut self, r: u8, disp: i32) {
        self.rex(true, r, 0);
        self.b(0x89);
        self.modrm_rsp(r, disp);
    }
    /// An integer argument into `r64`.
    fn int_arg(&mut self, rk: u8, arg: IArg) {
        match arg {
            IArg::Imm(v) => self.mov_imm(rk, v),
            IArg::WorkAddr(off) => {
                self.rex(true, rk, WORK);
                self.b(0x8D); // lea rk, [rbx + off]
                self.modrm_mem(rk, WORK, off);
            }
            IArg::InputAddr(off) => {
                self.rex(true, rk, INPUTS);
                self.b(0x8D); // lea rk, [r13 + off]
                self.modrm_mem(rk, INPUTS, off);
            }
            IArg::Bundles => {
                self.rex(true, BUNDLES, rk);
                self.b(0x89); // mov rk, r14
                self.modrm_reg(BUNDLES, rk);
            }
        }
    }
    /// `movq xmm, rax`.
    fn movq_from_rax(&mut self, xmm: u8) {
        self.b(0x66);
        self.rex(true, xmm, 0);
        self.bytes(&[0x0F, 0x6E]);
        self.modrm_reg(xmm, 0);
    }
    /// A bit mask into `xmm1`.
    fn mask1(&mut self, bits: u64) {
        self.mov_imm(0, bits);
        self.movq_from_rax(1);
    }
    /// `d = mask ? t : e` with the mask in `xmm0`; clobbers `xmm0`.
    fn blend(&mut self, t: u8, e: u8, d: u8) {
        self.mov(d, t);
        self.sse(0x66, 0x54, d, 0); // andpd d, xmm0
        self.sse(0x66, 0x55, 0, e); // andnpd xmm0, e
        self.sse(0x66, 0x56, d, 0); // orpd d, xmm0
    }
    fn base(b: Base) -> u8 {
        match b {
            Base::Work => WORK,
            Base::Inputs => INPUTS,
        }
    }
}

impl Isa for X64 {
    const CACHE: &'static [u8] = if WIN {
        &[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5]
    } else {
        &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    };
    const SAVED: usize = if WIN { 10 } else { 0 };
    const RESULT: u8 = 0;

    fn new(hot: &[*const ()]) -> X64 {
        #[cfg(target_arch = "x86_64")]
        let sse41 = is_x86_feature_detected!("sse4.1");
        #[cfg(not(target_arch = "x86_64"))]
        let sse41 = true;
        X64 {
            code: Vec::with_capacity(8192),
            sse41,
            hot: hot.iter().copied().take(HOT_REGS.len()).collect(),
        }
    }
    fn finish(self) -> Vec<u8> {
        self.code
    }

    fn prologue(&mut self) {
        // Six pushes and the frame: 16-byte aligned at every call below.
        self.bytes(&[0x55, 0x48, 0x89, 0xE5]); // push rbp; mov rbp, rsp
        self.bytes(&[0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57]); // push rbx r12 r13 r14 r15
        if WIN {
            self.bytes(&[0x48, 0x81, 0xEC]); // sub rsp, WIN_FRAME
            self.bytes(&WIN_FRAME.to_le_bytes());
            for x in 6..16u8 {
                self.xmm_rsp(true, x, 16 * (x as i32 - 6));
            }
            self.bytes(&[0x48, 0x89, 0xCB]); // mov rbx, rcx
            self.bytes(&[0x49, 0x89, 0xD5]); // mov r13, rdx
            self.bytes(&[0x4D, 0x89, 0xC6]); // mov r14, r8
        } else {
            self.bytes(&[0x48, 0x83, 0xEC, 0x08]); // sub rsp, 8
            self.bytes(&[0x48, 0x89, 0xFB]); // mov rbx, rdi
            self.bytes(&[0x49, 0x89, 0xF5]); // mov r13, rsi
            self.bytes(&[0x49, 0x89, 0xD6]); // mov r14, rdx
        }
        for k in 0..self.hot.len() {
            let a = self.hot[k] as usize as u64;
            self.mov_imm(HOT_REGS[k], a);
        }
    }
    fn epilogue(&mut self) {
        if WIN {
            for x in 6..16u8 {
                self.xmm_rsp(false, x, 16 * (x as i32 - 6));
            }
            self.bytes(&[0x48, 0x81, 0xC4]); // add rsp, WIN_FRAME
            self.bytes(&WIN_FRAME.to_le_bytes());
        } else {
            self.bytes(&[0x48, 0x83, 0xC4, 0x08]); // add rsp, 8
        }
        self.bytes(&[
            0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0x5D, 0xC3,
        ]);
    }

    fn load(&mut self, r: u8, base: Base, off: usize) {
        self.sse_mem(0xF2, 0x10, r, Self::base(base), off);
    }
    fn store(&mut self, r: u8, base: Base, off: usize) {
        self.sse_mem(0xF2, 0x11, r, Self::base(base), off);
    }
    fn fconst(&mut self, r: u8, v: f64) {
        if v.to_bits() == 0 {
            self.sse(0x66, 0x57, r, r); // xorpd r, r
        } else {
            self.mov_imm(0, v.to_bits());
            self.movq_from_rax(r);
        }
    }
    fn mov(&mut self, d: u8, a: u8) {
        if d != a {
            self.sse(0x66, 0x28, d, a); // movapd
        }
    }

    fn arith(&mut self, op: Arith, d: u8, a: u8, b: u8) {
        let opc = match op {
            Arith::Add => 0x58,
            Arith::Mul => 0x59,
            Arith::Sub => 0x5C,
            Arith::Div => 0x5E,
        };
        if d == b && d != a {
            if matches!(op, Arith::Add | Arith::Mul) {
                self.sse(0xF2, opc, d, a);
            } else {
                self.mov(0, a);
                self.sse(0xF2, opc, 0, b);
                self.mov(d, 0);
            }
        } else {
            self.mov(d, a);
            self.sse(0xF2, opc, d, b);
        }
    }
    const MINMAX: bool = false;
    fn minmax(&mut self, _op: ReduceOp, _d: u8, _a: u8, _b: u8) {
        // minsd / maxsd return the second operand on a NaN or a tie of
        // signed zeros, not the reference's rule; the host routine it is.
        unreachable!("no min/max instruction with the reference's NaN rule")
    }
    fn neg(&mut self, d: u8, a: u8) {
        self.mask1(0x8000_0000_0000_0000);
        self.mov(d, a);
        self.sse(0x66, 0x57, d, 1); // xorpd
    }
    fn abs(&mut self, d: u8, a: u8) {
        self.mask1(0x7FFF_FFFF_FFFF_FFFF);
        self.mov(d, a);
        self.sse(0x66, 0x54, d, 1); // andpd
    }
    fn sqrt(&mut self, d: u8, a: u8) {
        self.sse(0xF2, 0x51, d, a);
    }
    fn round(&mut self, mode: Round, d: u8, a: u8) -> bool {
        if !self.sse41 {
            return false;
        }
        let imm = match mode {
            Round::Floor => 0x9,
            Round::Ceil => 0xA,
            Round::Trunc => 0xB,
        };
        self.b(0x66);
        self.rex(false, d, a);
        self.bytes(&[0x0F, 0x3A, 0x0B]);
        self.modrm_reg(d, a);
        self.b(imm);
        true
    }
    fn cmp_select(&mut self, op: CmpOp, a: u8, b: u8, t: u8, e: u8, d: u8) {
        // cmpsd predicates: 0 eq, 1 lt, 2 le (ordered, so false on NaN),
        // 4 neq (true on NaN). Gt and Ge are Lt and Le with swapped operands.
        let (x, y, pred) = match op {
            CmpOp::Gt => (b, a, 1),
            CmpOp::Ge => (b, a, 2),
            CmpOp::Lt => (a, b, 1),
            CmpOp::Le => (a, b, 2),
            CmpOp::Eq => (a, b, 0),
            CmpOp::Ne => (a, b, 4),
        };
        self.mov(0, x);
        self.sse(0xF2, 0xC2, 0, y);
        self.b(pred);
        self.blend(t, e, d);
    }
    fn select_nz(&mut self, c: u8, t: u8, e: u8, d: u8) {
        self.sse(0x66, 0x57, 1, 1); // xorpd xmm1, xmm1
        self.mov(0, c);
        self.sse(0xF2, 0xC2, 0, 1);
        self.b(4); // cmpneqsd: true when c != 0 or c is NaN
        self.blend(t, e, d);
    }

    fn call(&mut self, addr: *const (), args: &[Arg]) {
        // System V counts floats and integers apart and passes the first six
        // integers in registers; Windows counts them together, passes four,
        // and wants shadow space on top. What is left over goes on the stack,
        // reserved right here and released after the call -- a kernel that
        // grows an argument then costs one more slot, not a corrupted frame.
        let stacked = if WIN {
            args.len()
        } else {
            args.iter().filter(|a| matches!(a, Arg::I(_))).count()
        }
        .saturating_sub(INT_ARGS.len());
        let shadow = if WIN { WIN_SHADOW } else { 0 };
        let frame = (shadow + 8 * stacked as i32 + 15) & !15;
        if frame > 0 {
            self.move_rsp(-frame);
        }
        let (mut nf, mut ni) = (0usize, 0usize);
        for (pos, arg) in args.iter().enumerate() {
            match *arg {
                Arg::F(r) => {
                    let k = if WIN { pos } else { nf };
                    nf += 1;
                    assert!(k < 4, "float arguments fit the registers");
                    self.mov(k as u8, r);
                }
                Arg::I(iarg) => {
                    let k = if WIN { pos } else { ni };
                    ni += 1;
                    // `rax` is the scratch the address goes through below, so
                    // a stacked argument can borrow it here.
                    match k.checked_sub(INT_ARGS.len()) {
                        Some(slot) => {
                            self.int_arg(0, iarg);
                            self.store_rsp(0, shadow + 8 * slot as i32);
                        }
                        None => self.int_arg(INT_ARGS[k], iarg),
                    }
                }
            }
        }
        match self.hot.iter().position(|&h| h == addr) {
            Some(k) => self.bytes(&[0x41, 0xFF, 0xD0 | (HOT_REGS[k] & 7)]), // call r12 / r15
            None => {
                self.mov_imm(0, addr as usize as u64);
                self.bytes(&[0xFF, 0xD0]); // call rax
            }
        }
        if frame > 0 {
            self.move_rsp(frame);
        }
    }
}
