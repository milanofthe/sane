//! x86-64 encodings, System V and Windows x64.
//!
//! `rbx` work, `r13` inputs, `r14` bundles; `rax` scratch; `xmm0` and
//! `xmm1` carry call arguments, the result and the masks of a select. The
//! cache is `xmm2` to `xmm15`. Under System V every vector register is
//! caller-saved, so nothing survives a host call; under Windows `xmm6` to
//! `xmm15` are callee-saved, so the chunk preserves them and they come
//! first in the cache. Baseline SSE2, with `roundsd` where there is SSE4.1
//! and three-operand VEX forms and `vblendvpd` where there is AVX (the same
//! IEEE operations, fewer bytes); `RSDAG_SSE2` set in the environment
//! keeps to the baseline (as it keeps rsdag's dense kernels to SSE2), so
//! both paths can be tested on one machine.
//! Constants are read rip-relative from a pool at the end of the chunk.
//!
//! `X64<2>` is the same code over two instances at once: the packed forms
//! of the same instructions (`addpd` for `addsd`, `cmppd` for `cmpsd`),
//! each lane the IEEE operation of the scalar code, so a lane's result is
//! the scalar result bit for bit. `X64<4>` runs four in the `ymm`
//! registers, every instruction in its VEX.256 form (AVX). The upper
//! halves of the `ymm` registers are not preserved across calls, not even
//! under Windows, so no value survives a host call there, and `vzeroupper`
//! precedes every call and the return.

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

pub(crate) struct X64<const L: usize> {
    code: Vec<u8>,
    sse41: bool,
    /// Three-operand VEX forms and `vblendvpd`.
    avx: bool,
    /// Host routines held in `r12` and `r15`.
    hot: Vec<*const ()>,
    /// The chunk's constants, and where their displacements go: the byte
    /// offset of each `disp32` and the constant it addresses.
    pool: Vec<u64>,
    fixups: Vec<(usize, usize)>,
}

const HOT_REGS: [u8; 2] = [12, 15];

/// The instruction sets the encodings may use.
#[derive(Clone, Copy)]
pub(crate) struct Features {
    pub sse41: bool,
    /// Three-operand VEX forms, `vblendvpd` and the 256-bit forms.
    pub avx: bool,
}

/// What this CPU has, or the SSE2 baseline when `RSDAG_SSE2` is set.
pub(crate) fn features() -> Features {
    static F: std::sync::OnceLock<Features> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let baseline = std::env::var_os("RSDAG_SSE2").is_some();
        Features {
            sse41: !baseline && is_x86_feature_detected!("sse4.1"),
            avx: !baseline && is_x86_feature_detected!("avx"),
        }
    })
}

impl<const L: usize> X64<L> {
    /// The legacy prefix of the arithmetic forms: `F2` (`sd`) or `66` (`pd`).
    const PFX: u8 = if L == 1 { 0xF2 } else { 0x66 };
    /// The same as a VEX `pp` field.
    const PP: u8 = if L == 1 { 3 } else { 1 };
    /// The VEX `L` bit: 256-bit forms for four lanes.
    const VL: u8 = (L == 4) as u8;
    /// Bytes per constant in the pool: every lane holds it.
    const CONST_BYTES: usize = 8 * if L == 4 { 4 } else { 2 };
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
    /// A move between a register and memory: the legacy form, or for four
    /// lanes the VEX form of vector length `l` (the scalar moves `0`).
    fn mov_mem(&mut self, prefix: u8, pp: u8, l: u8, op: u8, reg: u8, base: u8, disp: usize) {
        if L == 4 {
            self.vex_l(pp, 1, l, reg, 0, base);
            self.b(op);
            self.modrm_mem(reg, base, disp);
        } else {
            self.sse_mem(prefix, op, reg, base, disp);
        }
    }
    /// `op d, a`: a register instruction of map `0F` with one source,
    /// legacy, or VEX for four lanes.
    fn unary_rr(&mut self, prefix: u8, pp: u8, op: u8, d: u8, a: u8) {
        if L == 4 {
            self.vex_rr(pp, op, d, 0, a);
        } else {
            self.sse(prefix, op, d, a);
        }
    }
    /// `xorpd r, r`: zero.
    fn zero(&mut self, r: u8) {
        if L == 4 {
            self.vex_rr(1, 0x57, r, r, r);
        } else {
            self.sse(0x66, 0x57, r, r);
        }
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
    /// A VEX prefix for map `map` (1 `0F`, 3 `0F3A`), the legacy prefix
    /// `pp` (0 none, 1 `66`, 2 `F3`, 3 `F2`) and the vector length `l` (0
    /// 128 bits, 1 256): `reg` the ModRM reg field, `v` the extra source,
    /// `rm` the ModRM rm field. The two-byte form where it can encode them.
    fn vex_l(&mut self, pp: u8, map: u8, l: u8, reg: u8, v: u8, rm: u8) {
        let r = (!reg >> 3) & 1;
        let b = (!rm >> 3) & 1;
        let v = !v & 0xF;
        let tail = (v << 3) | (l << 2) | pp;
        if map == 1 && b == 1 {
            self.bytes(&[0xC5, (r << 7) | tail]);
        } else {
            self.bytes(&[0xC4, (r << 7) | (1 << 6) | (b << 5) | map, tail]);
        }
    }
    /// [`vex_l`](Self::vex_l) at the lanes' vector length.
    fn vex(&mut self, pp: u8, map: u8, reg: u8, v: u8, rm: u8) {
        self.vex_l(pp, map, Self::VL, reg, v, rm);
    }
    /// `op d, a, rm`: a VEX register instruction of map `0F`.
    fn vex_rr(&mut self, pp: u8, op: u8, d: u8, a: u8, rm: u8) {
        self.vex(pp, 1, d, a, rm);
        self.b(op);
        self.modrm_reg(d, rm);
    }
    /// ModRM for `[rip + disp32]` addressing the constant `bits` of the
    /// chunk's pool; `finish` writes the displacement.
    fn modrm_const(&mut self, reg: u8, bits: u64) {
        self.b(((reg & 7) << 3) | 5);
        let k = match self.pool.iter().position(|&c| c == bits) {
            Some(k) => k,
            None => {
                self.pool.push(bits);
                self.pool.len() - 1
            }
        };
        self.fixups.push((self.code.len(), k));
        self.bytes(&[0; 4]);
    }
    /// `d = a op [constant]`: VEX where there is AVX, else a copy and the
    /// legacy two-operand form (`prefix 0F op`).
    fn op_const(&mut self, prefix: u8, pp: u8, op: u8, d: u8, a: u8, bits: u64) {
        if self.avx {
            self.vex(pp, 1, d, a, 0);
        } else {
            self.mov(d, a);
            self.b(prefix);
            self.rex(false, d, 0);
            self.b(0x0F);
        }
        self.b(op);
        self.modrm_const(d, bits);
    }
    /// `d = mask ? t : e` with the mask in `xmm0`; clobbers `xmm0`.
    fn blend(&mut self, t: u8, e: u8, d: u8) {
        if self.avx {
            // vblendvpd d, e, t, xmm0
            self.vex(1, 3, d, e, t);
            self.b(0x4B);
            self.modrm_reg(d, t);
            self.b(0x00);
            return;
        }
        self.mov(d, t);
        self.sse(0x66, 0x54, d, 0); // andpd d, xmm0
        self.sse(0x66, 0x55, 0, e); // andnpd xmm0, e
        self.sse(0x66, 0x56, d, 0); // orpd d, xmm0
    }
    /// `xmm0 = (x pred y) ? all ones : 0`, `cmpsd`.
    fn cmp_mask(&mut self, x: u8, y: u8, pred: u8) {
        if self.avx {
            self.vex_rr(Self::PP, 0xC2, 0, x, y);
        } else {
            self.mov(0, x);
            self.sse(Self::PFX, 0xC2, 0, y);
        }
        self.b(pred);
    }
    /// The scalar opcode (`F2 0F op`) of an arithmetic op.
    fn arith_op(op: Arith) -> u8 {
        match op {
            Arith::Add => 0x58,
            Arith::Mul => 0x59,
            Arith::Sub => 0x5C,
            Arith::Div => 0x5E,
        }
    }
    fn base(b: Base) -> u8 {
        match b {
            Base::Work => WORK,
            Base::Inputs => INPUTS,
        }
    }
}

impl<const L: usize> Isa for X64<L> {
    const LANES: usize = L;
    const CACHE: &'static [u8] = if WIN {
        &[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5]
    } else {
        &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    };
    const SAVED: usize = if WIN && L < 4 { 10 } else { 0 };
    const RESULT: u8 = 0;

    fn new(hot: &[*const ()]) -> X64<L> {
        assert!(matches!(L, 1 | 2 | 4), "one, two or four lanes");
        let Features { sse41, avx } = features();
        assert!(L < 4 || avx, "four lanes need AVX");
        X64 {
            code: Vec::with_capacity(8192),
            sse41,
            avx,
            hot: hot.iter().copied().take(HOT_REGS.len()).collect(),
            pool: Vec::new(),
            fixups: Vec::new(),
        }
    }
    fn finish(mut self) -> Vec<u8> {
        // The constant pool after the code, each use rip-relative. An entry
        // is 16 bytes (32 for four lanes), 16-byte aligned (a chunk is placed
        // so): a legacy `andpd` or `xorpd` reads 16 aligned bytes from
        // memory. Every lane holds the constant, so packed code finds it in
        // each.
        if !self.pool.is_empty() {
            self.code.resize(self.code.len().next_multiple_of(16), 0xCC);
            let base = self.code.len();
            for c in &self.pool {
                for _ in 0..Self::CONST_BYTES / 8 {
                    self.code.extend_from_slice(&c.to_le_bytes());
                }
            }
            for &(at, k) in &self.fixups {
                let disp = (base + Self::CONST_BYTES * k) as i64 - (at + 4) as i64;
                let disp = i32::try_from(disp).expect("a chunk is smaller than 2 GB");
                self.code[at..at + 4].copy_from_slice(&disp.to_le_bytes());
            }
        }
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
        if L == 4 {
            self.bytes(&[0xC5, 0xF8, 0x77]); // vzeroupper
        }
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
        // movsd, or movupd (the work array is 8-byte aligned)
        let b = Self::base(base);
        self.mov_mem(Self::PFX, Self::PP, Self::VL, 0x10, r, b, off);
    }
    fn store(&mut self, r: u8, base: Base, off: usize) {
        let b = Self::base(base);
        self.mov_mem(Self::PFX, Self::PP, Self::VL, 0x11, r, b, off);
    }
    fn load_lane(&mut self, r: u8, base: Base, off: usize) {
        self.mov_mem(0xF2, 3, 0, 0x10, r, Self::base(base), off); // movsd
    }
    fn store_lane(&mut self, r: u8, base: Base, off: usize) {
        self.mov_mem(0xF2, 3, 0, 0x11, r, Self::base(base), off);
    }
    fn fconst(&mut self, r: u8, v: f64) {
        if v.to_bits() == 0 {
            self.zero(r);
        } else if L == 4 {
            // vmovupd r, [rip + constant] (the pool entry holds it four times)
            self.vex(1, 1, r, 0, 0);
            self.b(0x10);
            self.modrm_const(r, v.to_bits());
        } else {
            // movsd r, [rip + constant], or movapd for both lanes (the pool
            // entry is 16 aligned bytes holding the constant twice)
            self.b(if L == 1 { 0xF2 } else { 0x66 });
            self.rex(false, r, 0);
            self.bytes(&[0x0F, if L == 1 { 0x10 } else { 0x28 }]);
            self.modrm_const(r, v.to_bits());
        }
    }
    fn mov(&mut self, d: u8, a: u8) {
        if d != a {
            self.unary_rr(0x66, 1, 0x28, d, a); // movapd
        }
    }

    fn arith(&mut self, op: Arith, d: u8, a: u8, b: u8) {
        let opc = Self::arith_op(op);
        if self.avx {
            self.vex_rr(Self::PP, opc, d, a, b);
        } else if d == b && d != a {
            if matches!(op, Arith::Add | Arith::Mul) {
                self.sse(Self::PFX, opc, d, a);
            } else {
                self.mov(0, a);
                self.sse(Self::PFX, opc, 0, b);
                self.mov(d, 0);
            }
        } else {
            self.mov(d, a);
            self.sse(Self::PFX, opc, d, b);
        }
    }
    // A legacy packed memory operand must be 16-byte aligned; the work
    // array is 8-byte aligned, so two lanes load into a register. Four
    // lanes are VEX, whose memory operands need no alignment.
    const MEM_OPERANDS: bool = L != 2;
    fn arith_mem(&mut self, op: Arith, d: u8, a: u8, base: Base, off: usize) {
        let base = Self::base(base);
        if self.avx {
            self.vex(Self::PP, 1, d, a, base);
        } else {
            self.mov(d, a);
            self.b(Self::PFX);
            self.rex(false, d, base);
            self.b(0x0F);
        }
        self.b(Self::arith_op(op));
        self.modrm_mem(d, base, off);
    }
    const MINMAX: bool = false;
    fn minmax(&mut self, _op: ReduceOp, _d: u8, _a: u8, _b: u8) {
        // minsd / maxsd return the second operand on a NaN or a tie of
        // signed zeros, not the reference's rule; the host routine it is.
        unreachable!("no min/max instruction with the reference's NaN rule")
    }
    fn neg(&mut self, d: u8, a: u8) {
        self.op_const(0x66, 1, 0x57, d, a, 0x8000_0000_0000_0000); // xorpd
    }
    fn abs(&mut self, d: u8, a: u8) {
        self.op_const(0x66, 1, 0x54, d, a, 0x7FFF_FFFF_FFFF_FFFF); // andpd
    }
    fn sqrt(&mut self, d: u8, a: u8) {
        self.unary_rr(Self::PFX, Self::PP, 0x51, d, a);
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
        let op = if L == 1 { 0x0B } else { 0x09 }; // roundsd, or roundpd
        if L == 4 {
            self.vex(1, 3, d, 0, a);
            self.b(op);
        } else {
            self.b(0x66);
            self.rex(false, d, a);
            self.bytes(&[0x0F, 0x3A, op]);
        }
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
        self.cmp_mask(x, y, pred);
        self.blend(t, e, d);
    }
    fn select_nz(&mut self, c: u8, t: u8, e: u8, d: u8) {
        self.zero(1);
        self.cmp_mask(c, 1, 4); // cmpneqsd: true when c != 0 or c is NaN
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
        if L == 4 {
            self.bytes(&[0xC5, 0xF8, 0x77]); // vzeroupper
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
