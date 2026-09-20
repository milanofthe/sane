//! The execution scalar: what a program computes in.
//!
//! A [`Tape`](crate::tape::Tape) is lowered from a graph once; it can then be
//! evaluated in any `Scalar`: `f64` (the reference, and the only type the JIT
//! and the C backend emit), `f32`, or `Complex<f64>` for frequency-domain
//! work. Every type provides the reference implementations of the unary and
//! binary functions and of the comparisons, so a tape means the same thing
//! for every backend of a given `T`.
//!
//! Comparisons and selects need real predicates: for a complex `T` they act
//! on the real part, ordered ops (`Floor`, `Sign`, `Ceil`, `Round`, `Trunc`,
//! `Min`, `Max`) and the special real functions act on the real part and
//! return a real value.

use num_complex::Complex64;

use crate::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use crate::semantics::{binary_f64, cmp_bool, unary_f64};

pub trait Scalar: Copy + Send + Sync + std::fmt::Debug + 'static {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(x: f64) -> Self;
    /// Not-a-number, for a missing input.
    fn nan() -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn neg(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn unary(op: UnaryOp, x: Self) -> Self;
    fn binary(op: BinOp, x: Self, y: Self) -> Self;
    /// `1` or `0` from a comparison of the real parts.
    fn cmp(op: CmpOp, x: Self, y: Self) -> Self;
    /// The select predicate: nonzero real part.
    fn is_true(self) -> bool;
    fn min(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    /// `self / o`, one rounding: what a dense kernel divides with.
    fn div(self, o: Self) -> Self;
    /// The size a pivot is chosen by: the absolute value, or the modulus.
    fn magnitude(self) -> f64;
    /// The value as the real number a bundle takes; a bundle is a real
    /// function, so a scalar without a real value refuses.
    fn to_f64(self) -> f64;
    /// The dot of two slices in the reference fold
    /// ([`crate::semantics::dot_slice_t`]); `f64` runs its vector twin.
    fn dot_slice(a: &[Self], b: &[Self]) -> Self {
        crate::semantics::dot_slice_t(a, b)
    }
    /// The matrix-vector product in the reference fold
    /// ([`crate::semantics::gemv_t`]); `f64` runs its vector twin.
    fn gemv(a: &[Self], x: &[Self], m: usize, n: usize, out: &mut [Self]) {
        crate::semantics::gemv_t(a, x, m, n, out)
    }
    /// The matrix-matrix product in the reference fold
    /// ([`crate::semantics::gemm_t`]); `f64` runs its vector twin.
    fn gemm(a: &[Self], b: &[Self], m: usize, k: usize, n: usize, out: &mut [Self]) {
        crate::semantics::gemm_t(a, b, m, k, n, out)
    }
    /// The product with its entries folded ([`crate::semantics::gemm_fold_t`]).
    #[allow(clippy::too_many_arguments)]
    fn gemm_fold(
        a: &[Self],
        b: &[Self],
        m: usize,
        k: usize,
        n: usize,
        c: Option<&[Self]>,
        codes: &[u32],
        out: &mut [Self],
    ) {
        crate::semantics::gemm_fold_t(a, b, m, k, n, c, codes, out)
    }
    /// The matrix-vector product with its rows folded
    /// ([`crate::semantics::gemv_fold_t`]).
    fn gemv_fold(
        a: &[Self],
        x: &[Self],
        m: usize,
        n: usize,
        c: Option<&[Self]>,
        codes: &[u32],
        out: &mut [Self],
    ) {
        crate::semantics::gemv_fold_t(a, x, m, n, c, codes, out)
    }
    /// The dense solve of `k` right-hand sides
    /// ([`crate::semantics::solve_many_t`]); `f64` runs its vector twin.
    fn solve_many(a: &[Self], b: &[Self], n: usize, k: usize, out: &mut [Self]) {
        crate::semantics::solve_many_generic(a, b, n, k, out)
    }
    /// Call a bundle on arguments in `Self`, its outputs back in `Self`.
    /// A bundle computes in `f64`, so a scalar that is not `f64` converts
    /// both ways; `f64` itself calls straight through. The conversion
    /// buffers are borrowed from a thread-local stack and returned, so a
    /// call in a loop does not allocate.
    fn call_bundle(b: &dyn crate::extern_fn::ExternBundle, args: &[Self], out: &mut [Self]) {
        with_f64_scratch(args.len(), out.len(), |a, o| {
            for (dst, &x) in a.iter_mut().zip(args) {
                *dst = x.to_f64();
            }
            b.call(a, o);
            for (dst, &v) in out.iter_mut().zip(o.iter()) {
                *dst = Self::from_f64(v);
            }
        })
    }
    /// [`call_bundle`](Self::call_bundle) for `n_groups` argument groups.
    fn call_bundle_batch(
        b: &dyn crate::extern_fn::ExternBundle,
        args: &[Self],
        n_groups: usize,
        n_args: usize,
        out: &mut [Self],
    ) {
        with_f64_scratch(args.len(), out.len(), |a, o| {
            for (dst, &x) in a.iter_mut().zip(args) {
                *dst = x.to_f64();
            }
            b.call_batch(a, n_groups, n_args, o);
            for (dst, &v) in out.iter_mut().zip(o.iter()) {
                *dst = Self::from_f64(v);
            }
        })
    }
}

/// Two `f64` buffers of the requested lengths, borrowed from a thread-local
/// stack (a bundle that calls a bundle nests) and returned afterwards.
fn with_f64_scratch<R>(
    n_args: usize,
    n_out: usize,
    f: impl FnOnce(&mut [f64], &mut [f64]) -> R,
) -> R {
    thread_local! {
        static STACK: std::cell::RefCell<Vec<(Vec<f64>, Vec<f64>)>> = const {
            std::cell::RefCell::new(Vec::new())
        };
    }
    let (mut a, mut o) = STACK
        .with(|s| s.borrow_mut().pop())
        .unwrap_or_else(|| (Vec::new(), Vec::new()));
    if a.len() < n_args {
        a.resize(n_args, 0.0);
    }
    if o.len() < n_out {
        o.resize(n_out, 0.0);
    }
    let r = f(&mut a[..n_args], &mut o[..n_out]);
    STACK.with(|s| s.borrow_mut().push((a, o)));
    r
}

impl Scalar for f64 {
    fn div(self, o: Self) -> Self {
        self / o
    }
    fn dot_slice(a: &[Self], b: &[Self]) -> Self {
        crate::simd::dot(a, b)
    }
    fn gemv(a: &[Self], x: &[Self], m: usize, n: usize, out: &mut [Self]) {
        crate::simd::gemv(a, x, m, n, out)
    }
    fn gemm(a: &[Self], b: &[Self], m: usize, k: usize, n: usize, out: &mut [Self]) {
        crate::simd::gemm(a, b, m, k, n, out)
    }
    fn gemm_fold(
        a: &[Self],
        b: &[Self],
        m: usize,
        k: usize,
        n: usize,
        c: Option<&[Self]>,
        codes: &[u32],
        out: &mut [Self],
    ) {
        crate::simd::gemm_fold(a, b, m, k, n, c, codes, out)
    }
    fn gemv_fold(
        a: &[Self],
        x: &[Self],
        m: usize,
        n: usize,
        c: Option<&[Self]>,
        codes: &[u32],
        out: &mut [Self],
    ) {
        crate::simd::gemv(a, x, m, n, out);
        crate::semantics::fold_in_place(codes, c, out);
    }
    fn solve_many(a: &[Self], b: &[Self], n: usize, k: usize, out: &mut [Self]) {
        crate::simd::solve_many(a, b, n, k, out)
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn call_bundle(b: &dyn crate::extern_fn::ExternBundle, args: &[f64], out: &mut [f64]) {
        b.call(args, out);
    }
    fn call_bundle_batch(
        b: &dyn crate::extern_fn::ExternBundle,
        args: &[f64],
        n_groups: usize,
        n_args: usize,
        out: &mut [f64],
    ) {
        b.call_batch(args, n_groups, n_args, out);
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(x: f64) -> Self {
        x
    }
    fn nan() -> Self {
        f64::NAN
    }
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn sub(self, o: Self) -> Self {
        self - o
    }
    fn mul(self, o: Self) -> Self {
        self * o
    }
    fn neg(self) -> Self {
        -self
    }
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    fn unary(op: UnaryOp, x: Self) -> Self {
        unary_f64(op, x)
    }
    fn binary(op: BinOp, x: Self, y: Self) -> Self {
        binary_f64(op, x, y)
    }
    fn cmp(op: CmpOp, x: Self, y: Self) -> Self {
        if cmp_bool(op, x, y) {
            1.0
        } else {
            0.0
        }
    }
    fn is_true(self) -> bool {
        self != 0.0
    }
    fn min(self, o: Self) -> Self {
        ReduceOp::Min.combine(self, o)
    }
    fn max(self, o: Self) -> Self {
        ReduceOp::Max.combine(self, o)
    }
}

impl Scalar for f32 {
    fn div(self, o: Self) -> Self {
        self / o
    }
    fn magnitude(self) -> f64 {
        self.abs() as f64
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn nan() -> Self {
        f32::NAN
    }
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn sub(self, o: Self) -> Self {
        self - o
    }
    fn mul(self, o: Self) -> Self {
        self * o
    }
    fn neg(self) -> Self {
        -self
    }
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }
    /// Single precision goes through the double reference and rounds once:
    /// the same guards, one rounding, no second set of algorithms.
    fn unary(op: UnaryOp, x: Self) -> Self {
        unary_f64(op, x as f64) as f32
    }
    fn binary(op: BinOp, x: Self, y: Self) -> Self {
        binary_f64(op, x as f64, y as f64) as f32
    }
    fn cmp(op: CmpOp, x: Self, y: Self) -> Self {
        if cmp_bool(op, x, y) {
            1.0
        } else {
            0.0
        }
    }
    fn is_true(self) -> bool {
        self != 0.0
    }
    fn min(self, o: Self) -> Self {
        ReduceOp::Min.combine(self as f64, o as f64) as f32
    }
    fn max(self, o: Self) -> Self {
        ReduceOp::Max.combine(self as f64, o as f64) as f32
    }
}

impl Scalar for Complex64 {
    fn div(self, o: Self) -> Self {
        self / o
    }
    fn magnitude(self) -> f64 {
        self.norm()
    }
    fn to_f64(self) -> f64 {
        assert!(self.im == 0.0, "a bundle call takes real arguments");
        self.re
    }
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
    fn from_f64(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    fn nan() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn sub(self, o: Self) -> Self {
        self - o
    }
    fn mul(self, o: Self) -> Self {
        self * o
    }
    fn neg(self) -> Self {
        -self
    }
    fn powi(self, n: i32) -> Self {
        Complex64::powi(&self, n)
    }
    fn unary(op: UnaryOp, x: Self) -> Self {
        match op {
            UnaryOp::Exp => x.exp(),
            UnaryOp::Ln => x.ln(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
            UnaryOp::Sinh => x.sinh(),
            UnaryOp::Cosh => x.cosh(),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Atan => x.atan(),
            UnaryOp::Tan => x.tan(),
            UnaryOp::Log10 => x.ln() / std::f64::consts::LN_10,
            UnaryOp::Log2 => x.ln() / std::f64::consts::LN_2,
            UnaryOp::Log1p => (x + 1.0).ln(),
            UnaryOp::Expm1 => x.exp() - 1.0,
            UnaryOp::Cbrt => x.powf(1.0 / 3.0),
            UnaryOp::Abs => Complex64::new(x.norm(), 0.0),
            UnaryOp::Asin => x.asin(),
            UnaryOp::Acos => x.acos(),
            UnaryOp::Asinh => x.asinh(),
            UnaryOp::Acosh => x.acosh(),
            UnaryOp::Atanh => x.atanh(),
            UnaryOp::Floor
            | UnaryOp::Sign
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::Trunc
            | UnaryOp::Erf
            | UnaryOp::Erfc
            | UnaryOp::Lgamma
            | UnaryOp::Tgamma
            | UnaryOp::Digamma
            | UnaryOp::Trigamma
            | UnaryOp::RandUniform => Complex64::new(unary_f64(op, x.re), 0.0),
        }
    }
    fn binary(op: BinOp, x: Self, y: Self) -> Self {
        match op {
            BinOp::Powf => x.powc(y),
            BinOp::Mod | BinOp::Atan2 | BinOp::Hypot => {
                Complex64::new(binary_f64(op, x.re, y.re), 0.0)
            }
        }
    }
    fn cmp(op: CmpOp, x: Self, y: Self) -> Self {
        if cmp_bool(op, x.re, y.re) {
            Self::one()
        } else {
            Self::zero()
        }
    }
    fn is_true(self) -> bool {
        self.re != 0.0
    }
    fn min(self, o: Self) -> Self {
        if o.re < self.re {
            o
        } else {
            self
        }
    }
    fn max(self, o: Self) -> Self {
        if o.re > self.re {
            o
        } else {
            self
        }
    }
}
