//! The execution scalar: what a program computes in.
//!
//! A [`Tape`](crate::tape::Tape) is lowered from a graph once and evaluated
//! in a `Scalar`. `f64` provides the reference implementations of the unary
//! and binary functions and of the comparisons, so a tape means the same
//! thing for every backend.

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
    /// The dense solve of `k` right-hand sides over `scratch` (see
    /// [`crate::semantics::solve_many_into`]); `f64` runs its vector twin.
    fn solve_many(
        a: &[Self],
        b: &[Self],
        n: usize,
        k: usize,
        out: &mut [Self],
        scratch: &mut [Self],
    ) {
        crate::semantics::solve_generic_into(a, b, n, k, out, scratch)
    }
    /// `count` solves of one shape (see
    /// [`crate::semantics::solve_batch_into`]): one after the other, or in
    /// `f64` side by side in the lanes of a vector.
    #[allow(clippy::too_many_arguments)]
    fn solve_batch(
        a: &[Self],
        b: &[Self],
        n: usize,
        k: usize,
        count: usize,
        out: &mut [Self],
        scratch: &mut [Self],
    ) {
        let (sa, sb) = (n * n, n * k);
        for c in 0..count {
            Self::solve_many(
                &a[c * sa..(c + 1) * sa],
                &b[c * sb..(c + 1) * sb],
                n,
                k,
                &mut out[c * sb..(c + 1) * sb],
                scratch,
            );
        }
    }
    /// Call a bundle on arguments in `Self`, its outputs back in `Self`.
    /// A bundle computes in `f64`, so a scalar that is not `f64` converts
    /// both ways; `f64` itself calls straight through. The conversion
    /// buffers are borrowed from a thread-local stack and returned, so a
    /// call in a loop does not allocate.
    fn call_bundle(b: &dyn crate::extern_fn::ExternBundle, args: &[Self], out: &mut [Self]) {
        // A body that is a tape runs in this scalar: a complex small-signal
        // evaluation through a call stays complex. An opaque body runs in f64.
        if let Some(tape) = b.body() {
            crate::scratch::with(|work: &mut Vec<Self>| {
                crate::scratch::with(|res: &mut Vec<Self>| {
                    tape.eval(args, work, res);
                    out.copy_from_slice(&res[..out.len()]);
                })
            });
            return;
        }
        f64_buffers(args.len(), out.len(), |a, o| {
            for (dst, &x) in a.iter_mut().zip(args) {
                *dst = x.to_f64();
            }
            b.call(a, o);
            for (dst, &v) in out.iter_mut().zip(o.iter()) {
                *dst = Self::from_f64(v);
            }
        })
    }
    /// A call run whole over scratch the tape lends (`work`, the bundle's
    /// [`work_len`](crate::extern_fn::ExternBundle::work_len)).
    fn call_bundle_whole(
        b: &dyn crate::extern_fn::ExternBundle,
        args: &[Self],
        _work: &mut [Self],
        out: &mut [Self],
    ) {
        Self::call_bundle(b, args, out);
    }
    /// The prolog of `n_groups` stateful calls: their states from their
    /// pure arguments (`n_pure` each). Only `f64` evaluates bundles in
    /// phases; another scalar runs every call whole
    /// ([`call_bundle_main_batch`](Self::call_bundle_main_batch)), so its
    /// states are never read.
    fn call_bundle_prolog_batch(
        _b: &dyn crate::extern_fn::ExternBundle,
        _pure: &[Self],
        _n_groups: usize,
        _n_pure: usize,
        _work: &mut [Self],
        states: &mut [Self],
    ) {
        states.fill(Self::nan());
    }
    /// The main phase of `n_groups` stateful calls over their states; see
    /// [`call_bundle_prolog_batch`](Self::call_bundle_prolog_batch).
    fn call_bundle_main_batch(
        b: &dyn crate::extern_fn::ExternBundle,
        args: &[Self],
        _states: &[Self],
        n_groups: usize,
        n_args: usize,
        _work: &mut [Self],
        out: &mut [Self],
    ) {
        Self::call_bundle_batch(b, args, n_groups, n_args, out);
    }
    /// [`call_bundle`](Self::call_bundle) for `n_groups` argument groups.
    fn call_bundle_batch(
        b: &dyn crate::extern_fn::ExternBundle,
        args: &[Self],
        n_groups: usize,
        n_args: usize,
        out: &mut [Self],
    ) {
        if b.body().is_some() {
            let n_out = b.n_outputs();
            for g in 0..n_groups {
                Self::call_bundle(
                    b,
                    &args[g * n_args..(g + 1) * n_args],
                    &mut out[g * n_out..(g + 1) * n_out],
                );
            }
            return;
        }
        f64_buffers(args.len(), out.len(), |a, o| {
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

/// Two `f64` buffers of this thread's, of the given lengths.
fn f64_buffers<R>(n_args: usize, n_out: usize, f: impl FnOnce(&mut [f64], &mut [f64]) -> R) -> R {
    crate::scratch::with_len(n_args, 0.0, |a| {
        crate::scratch::with_len(n_out, 0.0, |o| f(a, o))
    })
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
    fn solve_many(
        a: &[Self],
        b: &[Self],
        n: usize,
        k: usize,
        out: &mut [Self],
        scratch: &mut [Self],
    ) {
        crate::simd::solve_many(a, b, n, k, out, scratch)
    }
    fn solve_batch(
        a: &[Self],
        b: &[Self],
        n: usize,
        k: usize,
        count: usize,
        out: &mut [Self],
        scratch: &mut [Self],
    ) {
        crate::simd::solve_batch(a, b, n, k, count, out, scratch)
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
    fn call_bundle_whole(
        b: &dyn crate::extern_fn::ExternBundle,
        args: &[f64],
        work: &mut [f64],
        out: &mut [f64],
    ) {
        b.call_into(args, work, out);
    }
    fn call_bundle_prolog_batch(
        b: &dyn crate::extern_fn::ExternBundle,
        pure: &[f64],
        n_groups: usize,
        n_pure: usize,
        work: &mut [f64],
        states: &mut [f64],
    ) {
        if n_groups >= 2 {
            let at = crate::Instances::first(n_groups, n_pure, b.state_len());
            b.prolog_batch(pure, states, &at);
        } else if n_groups == 1 {
            b.prolog_into(pure, work, states);
        }
    }
    fn call_bundle_main_batch(
        b: &dyn crate::extern_fn::ExternBundle,
        args: &[f64],
        states: &[f64],
        n_groups: usize,
        n_args: usize,
        work: &mut [f64],
        out: &mut [f64],
    ) {
        if n_groups >= 2 {
            let at = crate::Instances::first(n_groups, n_args, b.state_len());
            b.main_batch(args, states, out, &at);
        } else if n_groups == 1 {
            b.main_into(args, states, work, out);
        }
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
        crate::semantics::powi_t(self, n)
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
