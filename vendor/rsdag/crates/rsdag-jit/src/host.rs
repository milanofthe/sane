//! Host routines: what the emitted code calls for everything that is not
//! an instruction. Every transcendental goes through the same
//! [`unary_f64`](rsdag::semantics::unary_f64) as the interpreter, so the domain
//! guards hold identically and the backends agree to the bit.

use rsdag::extern_fn::ExternBundle;
use rsdag::node::{BinOp, ReduceOp, UnaryOp};
use rsdag::semantics::{binary_f64, reduce_slice, unary_f64};
use rsdag::tape::calls::{run_call, run_stage, Call, Phase, Shared};
use std::sync::Arc;

pub(crate) type Bundles = Vec<Arc<dyn ExternBundle>>;

macro_rules! unary_trampolines {
    ($($name:ident = $op:ident),* $(,)?) => {
        $(pub(crate) extern "C" fn $name(x: f64) -> f64 {
            unary_f64(UnaryOp::$op, x)
        })*
        /// The dedicated routine of a unary op, or the coded one.
        pub(crate) fn unary_addr(op: UnaryOp) -> (*const (), Option<u32>) {
            match op {
                $(UnaryOp::$op => ($name as *const (), None),)*
                _ => (h_unary_ext as *const (), Some(op.code())),
            }
        }
    };
}
unary_trampolines! {
    h_exp = Exp, h_ln = Ln, h_sqrt = Sqrt, h_sin = Sin, h_cos = Cos,
    h_sinh = Sinh, h_cosh = Cosh, h_tanh = Tanh, h_atan = Atan, h_floor = Floor,
}

pub(crate) extern "C" fn h_unary_ext(op: u32, x: f64) -> f64 {
    unary_f64(UnaryOp::from_code(op), x)
}
pub(crate) extern "C" fn h_binary(op: u32, x: f64, y: f64) -> f64 {
    binary_f64(BinOp::from_code(op), x, y)
}
pub(crate) extern "C" fn h_powi(x: f64, n: i64) -> f64 {
    rsdag::semantics::powi_f64(x, n as i32)
}

pub(crate) fn reduce_code(op: ReduceOp) -> u64 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Product => 1,
        ReduceOp::Min => 2,
        ReduceOp::Max => 3,
    }
}
pub(crate) extern "C" fn h_reduce(op: u64, ptr: *const f64, len: usize) -> f64 {
    let xs = unsafe { std::slice::from_raw_parts(ptr, len) };
    let op = match op {
        0 => ReduceOp::Sum,
        1 => ReduceOp::Product,
        2 => ReduceOp::Min,
        _ => ReduceOp::Max,
    };
    reduce_slice(op, xs)
}

// A panic out of a host call, held until the emitted code has returned:
// unwinding cannot cross the emitted frames (they carry no unwind
// tables, and an `extern "C"` boundary aborts), so a trampoline catches
// it, the chunk runs to its end on whatever the failed call left in its
// outputs, and `resume_panic` raises it again on the caller's side. The
// first panic of an evaluation is the one kept.
std::thread_local! {
    static PANIC: std::cell::Cell<Option<Box<dyn std::any::Any + Send>>> =
        const { std::cell::Cell::new(None) };
}

/// Run `f`, holding a panic out of it for [`resume_panic`].
fn guarded(f: impl FnOnce()) {
    if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        PANIC.with(|c| {
            let first = c.take().unwrap_or(p);
            c.set(Some(first));
        });
    }
}

/// Raise a panic a host call held while the emitted code ran.
pub(crate) fn resume_panic() {
    if let Some(p) = PANIC.with(|c| c.take()) {
        std::panic::resume_unwind(p);
    }
}

/// A call site as the emitted code hands it to [`h_call`]: every size
/// fixed when the tape was compiled, every place an index into the work
/// array's regions: its slots `0..gather`, the gather area
/// `gather..scratch` and the bundles' scratch from `scratch`. The
/// descriptors live as long as the code that points at them.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct CallDesc {
    pub(crate) bundle: u64,
    /// 0 whole, 1 the main phase, 2 the prolog.
    pub(crate) kind: u64,
    pub(crate) n_groups: u64,
    pub(crate) n_args: u64,
    pub(crate) n_in: u64,
    pub(crate) n_out: u64,
    /// The arguments in the gather area; the outputs (the states for a
    /// prolog) and the states (a main phase) among the slots.
    pub(crate) args: u64,
    pub(crate) out: u64,
    pub(crate) state: u64,
    pub(crate) gather: u64,
    pub(crate) scratch: u64,
    pub(crate) scratch_len: u64,
    /// Ops the call carries (its instances times its body's), for
    /// [`h_stage`]'s choice of running in parallel.
    pub(crate) ops: u64,
    /// The operands the arguments are gathered from (a slot, or an input
    /// tagged as in the tape), for a call too wide to gather in its code;
    /// `0` when the code gathered them.
    pub(crate) table: u64,
    /// The argument each of an instance's `n_in` operands is, for a call
    /// that gathers only some; `0` for all, in order.
    pub(crate) places: u64,
    /// The inputs the operands read reach this far.
    pub(crate) n_inputs: u64,
}

/// `d` as the tape's call runner takes it.
///
/// # Safety
/// `d` describes a call of the code `bundles` belong to.
unsafe fn call_of<'a>(bundles: &'a Bundles, d: &'a CallDesc) -> Call<'a> {
    let (ng, ni) = (d.n_groups as usize, d.n_in as usize);
    let table = |p: u64, n: usize| unsafe { std::slice::from_raw_parts(p as *const u32, n) };
    Call {
        bundle: &*bundles[d.bundle as usize],
        phase: match d.kind {
            0 => Phase::Whole,
            1 => Phase::Main,
            _ => Phase::Prolog,
        },
        n_groups: ng,
        n_args: d.n_args as usize,
        n_out: d.n_out as usize,
        operands: (d.table != 0).then(|| table(d.table, ng * ni)),
        places: (d.places != 0).then(|| table(d.places, ni)),
        args: d.args as usize,
        out: d.out as usize,
        state: d.state as usize,
        ops: d.ops,
    }
}

/// The regions of the work array `d` names, and the inputs its operands
/// read.
///
/// # Safety
/// `work` and `inputs` are the program's buffers.
unsafe fn regions<'a>(
    d: &CallDesc,
    work: *mut f64,
    inputs: *const f64,
) -> (Shared<f64>, Shared<f64>, &'a [f64]) {
    let (g, s) = (d.gather as usize, d.scratch as usize);
    unsafe {
        (
            Shared::from_raw(work, g),
            Shared::from_raw(work.add(g), s - g),
            std::slice::from_raw_parts(inputs, d.n_inputs as usize),
        )
    }
}

/// Every bundle call: whole, main phase over its states, or prolog.
pub(crate) extern "C" fn h_call(
    bundles: *const Bundles,
    d: *const CallDesc,
    work: *mut f64,
    inputs: *const f64,
) {
    let d = unsafe { &*d };
    let b = unsafe { &*bundles };
    // SAFETY: the call's regions are disjoint by the layout.
    guarded(|| unsafe {
        let c = call_of(b, d);
        let (slots, gather, inputs) = regions(d, work, inputs);
        let scratch =
            std::slice::from_raw_parts_mut(work.add(d.scratch as usize), d.scratch_len as usize);
        run_call(&c, slots, gather, inputs, 0..c.n_groups, scratch)
    });
}

/// The calls of a stage ([`rsdag::Stage`]): `n` descriptors from `d` on,
/// run by the tape's [`run_stage`].
pub(crate) extern "C" fn h_stage(
    bundles: *const Bundles,
    d: *const CallDesc,
    n: u64,
    work: *mut f64,
    inputs: *const f64,
) {
    let descs = unsafe { std::slice::from_raw_parts(d, n as usize) };
    let b = unsafe { &*bundles };
    guarded(|| unsafe {
        let n_inputs = descs.iter().map(|d| d.n_inputs).max().unwrap_or(0);
        let wide = CallDesc {
            n_inputs,
            ..descs[0]
        };
        let (slots, gather, inputs) = regions(&wide, work, inputs);
        // SAFETY: the calls of a stage read nothing another one writes
        // (`rsdag`'s stage planning), their arguments apart.
        let call = |k: usize| call_of(b, &descs[k]);
        run_stage(descs.len(), call, slots, gather, inputs)
    });
}

/// Where a kernel operand is: a byte offset into the work array or into
/// the inputs, or nowhere (an accumulator no fold reads).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Place {
    pub(crate) base: u64,
    pub(crate) off: u64,
}

impl Place {
    pub(crate) const WORK: u64 = 0;
    pub(crate) const INPUTS: u64 = 1;
    pub(crate) const NONE: Place = Place { base: 2, off: 0 };
}

/// A dense kernel as the code hands it over: its kind (0 `Gemv`, 1 `Gemm`,
/// 2 the solves of `k` right-hand sides, `m` systems of them), its dimensions, where its
/// operands are (`a`, then `x` or `b`, then the accumulator), the address
/// of its fold codes (0 without), the byte offset of its outputs, and
/// the scratch a solve works in (the layout's, apart from slots and gather).
#[repr(C)]
pub(crate) struct KernelDesc {
    pub(crate) kind: u64,
    pub(crate) m: u64,
    pub(crate) k: u64,
    pub(crate) n: u64,
    pub(crate) operands: [Place; 3],
    pub(crate) codes: u64,
    pub(crate) out: u64,
    /// The byte offset and the length of the scratch a solve works in.
    pub(crate) scratch: u64,
    pub(crate) scratch_len: u64,
}

/// Run a dense kernel through the reference kernels of `rsdag::semantics`.
pub(crate) extern "C" fn h_kernel(work: *mut f64, inputs: *const f64, d: *const KernelDesc) {
    use rsdag::semantics as s;
    let d = unsafe { &*d };
    let (m, k, n) = (d.m as usize, d.k as usize, d.n as usize);
    // SAFETY: the code placed every operand, in the work array or the
    // inputs, and the output block apart from all of them (a kernel's
    // outputs are fresh slots); the lengths are the kernel's.
    let operand = |i: usize, len: usize| -> Option<&[f64]> {
        let p = d.operands[i];
        let base = match p.base {
            Place::WORK => work as *const f64,
            Place::INPUTS => inputs,
            _ => return None,
        };
        Some(unsafe { std::slice::from_raw_parts(base.byte_add(p.off as usize), len) })
    };
    let len_out = match d.kind {
        0 => m,
        1 => m * n,
        _ => m * n * k,
    };
    let out = unsafe { std::slice::from_raw_parts_mut(work.byte_add(d.out as usize), len_out) };
    let codes = (d.codes != 0)
        .then(|| unsafe { std::slice::from_raw_parts(d.codes as *const u32, len_out) });
    let (a, b) = match d.kind {
        0 => (operand(0, m * n), operand(1, n)),
        1 => (operand(0, m * k), operand(1, n * k)),
        _ => (operand(0, m * n * n), operand(1, m * n * k)),
    };
    let (a, b) = (a.expect("a kernel's first operand"), b.expect("its second"));
    match (d.kind, codes) {
        (0, None) => s::gemv(a, b, m, n, out),
        (0, Some(codes)) => s::gemv_fold(a, b, m, n, operand(2, m), codes, out),
        (1, None) => s::gemm(a, b, m, k, n, out),
        (1, Some(codes)) => s::gemm_fold(a, b, m, k, n, operand(2, m * n), codes, out),
        _ => {
            let scratch = unsafe {
                std::slice::from_raw_parts_mut(
                    work.byte_add(d.scratch as usize),
                    d.scratch_len as usize,
                )
            };
            s::solve_batch_into(a, b, n, k, m, out, scratch)
        }
    }
}
