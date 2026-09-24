//! Host routines: what the emitted code calls for everything that is not
//! an instruction. Every transcendental goes through the same
//! [`unary_f64`](rsdag::semantics::unary_f64) as the interpreter, so the domain
//! guards hold identically and the backends agree to the bit.

use rsdag::extern_fn::ExternBundle;
use rsdag::node::{BinOp, ReduceOp, UnaryOp};
use rsdag::semantics::{binary_f64, reduce_slice, unary_f64};
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
    x.powi(n as i32)
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
/// fixed when the tape was compiled, every place a byte offset into the
/// work array. The descriptors live as long as the code that points at them.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct CallDesc {
    pub(crate) bundle: u64,
    pub(crate) kind: u64,
    pub(crate) batch: u64,
    pub(crate) n_groups: u64,
    pub(crate) n_args: u64,
    pub(crate) n_out: u64,
    pub(crate) state_len: u64,
    /// Byte offsets: the gathered arguments, the outputs (the states for a
    /// prolog), the states (a main phase), the bundle's scratch.
    pub(crate) args: u64,
    pub(crate) out: u64,
    pub(crate) state: u64,
    pub(crate) scratch: u64,
    pub(crate) scratch_len: u64,
}

/// Every bundle call: whole, main phase over its states, or prolog.
pub(crate) extern "C" fn h_call(bundles: *const Bundles, d: *const CallDesc, work: *mut f64) {
    let d = unsafe { &*d };
    let b = unsafe { &(&*bundles)[d.bundle as usize] };
    let (ng, na, no, sl) = (
        d.n_groups as usize,
        d.n_args as usize,
        d.n_out as usize,
        d.state_len as usize,
    );
    // The regions are disjoint by the layout: the gather area, the output
    // block, the state block, the scratch.
    let at = |off: u64| unsafe { work.add(off as usize / 8) };
    let args = unsafe { std::slice::from_raw_parts(at(d.args), ng * na) };
    let out = unsafe { std::slice::from_raw_parts_mut(at(d.out), ng * no) };
    let scratch = unsafe { std::slice::from_raw_parts_mut(at(d.scratch), d.scratch_len as usize) };
    guarded(|| match d.kind {
        0 => {
            assert_eq!(
                b.n_outputs(),
                no,
                "bundle output count changed since compile"
            );
            if d.batch != 0 {
                b.call_batch(args, ng, na, out);
            } else {
                b.call_into(args, scratch, out);
            }
        }
        1 => {
            assert_eq!(
                b.n_outputs(),
                no,
                "bundle output count changed since compile"
            );
            assert_eq!(
                b.state_len(),
                sl,
                "bundle state length changed since compile"
            );
            let states = unsafe { std::slice::from_raw_parts(at(d.state), ng * sl) };
            for g in 0..ng {
                let (a, o) = (&args[g * na..(g + 1) * na], &mut out[g * no..(g + 1) * no]);
                b.main_into(a, &states[g * sl..(g + 1) * sl], scratch, o);
            }
        }
        _ => {
            assert_eq!(
                b.state_len(),
                sl,
                "bundle state length changed since compile"
            );
            for g in 0..ng {
                let (a, st) = (&args[g * na..(g + 1) * na], &mut out[g * sl..(g + 1) * sl]);
                b.prolog_into(a, scratch, st);
            }
        }
    });
}

pub(crate) extern "C" fn h_solve(a: *const f64, b: *const f64, n: usize, out: *mut f64) {
    let a = unsafe { std::slice::from_raw_parts(a, n * n) };
    let b = unsafe { std::slice::from_raw_parts(b, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    rsdag::semantics::solve(a, b, n, out);
}

pub(crate) extern "C" fn h_solve_many(
    a: *const f64,
    b: *const f64,
    n: usize,
    k: usize,
    out: *mut f64,
) {
    let a = unsafe { std::slice::from_raw_parts(a, n * n) };
    let b = unsafe { std::slice::from_raw_parts(b, n * k) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n * k) };
    rsdag::semantics::solve_many(a, b, n, k, out);
}

pub(crate) extern "C" fn h_gemm(
    a: *const f64,
    b: *const f64,
    m: usize,
    k: usize,
    n: usize,
    out: *mut f64,
) {
    let a = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b = unsafe { std::slice::from_raw_parts(b, n * k) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, m * n) };
    rsdag::semantics::gemm(a, b, m, k, n, out);
}

/// `A x` folded per the codes at `codes` (see `rsdag::tape::Fold`)
/// against `c` (null when no code reads it).
#[allow(clippy::too_many_arguments)]
pub(crate) extern "C" fn h_gemv_acc(
    a: *const f64,
    x: *const f64,
    c: *const f64,
    codes: *const u32,
    m: usize,
    n: usize,
    out: *mut f64,
) {
    let a = unsafe { std::slice::from_raw_parts(a, m * n) };
    let x = unsafe { std::slice::from_raw_parts(x, n) };
    let c = (!c.is_null()).then(|| unsafe { std::slice::from_raw_parts(c, m) });
    let codes = unsafe { std::slice::from_raw_parts(codes, m) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, m) };
    rsdag::semantics::gemv_fold(a, x, m, n, c, codes, out);
}

/// `A B` folded per the codes, as [`h_gemv_acc`].
#[allow(clippy::too_many_arguments)]
pub(crate) extern "C" fn h_gemm_acc(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    codes: *const u32,
    m: usize,
    k: usize,
    n: usize,
    out: *mut f64,
) {
    let a = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b = unsafe { std::slice::from_raw_parts(b, n * k) };
    let c = (!c.is_null()).then(|| unsafe { std::slice::from_raw_parts(c, m * n) });
    let codes = unsafe { std::slice::from_raw_parts(codes, m * n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, m * n) };
    rsdag::semantics::gemm_fold(a, b, m, k, n, c, codes, out);
}

pub(crate) extern "C" fn h_gemv(a: *const f64, x: *const f64, m: usize, n: usize, out: *mut f64) {
    let a = unsafe { std::slice::from_raw_parts(a, m * n) };
    let x = unsafe { std::slice::from_raw_parts(x, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, m) };
    rsdag::semantics::gemv(a, x, m, n, out);
}
