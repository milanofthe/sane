//! The `f64` twins of the dot-fold kernels: the same fold as the generic
//! reference in [`crate::semantics`] (four accumulators, merged as
//! `(0 + 1) + (2 + 3)`, then the tail in order), written once over a
//! four-lane vector [`Quad`] whose lane `l` is accumulator `l`. Bit-identical
//! to the reference by construction: every product and every sum rounds
//! once, nothing is fused. The generic reference stays the definition; the
//! tests hold the twins to it.
//!
//! A `Quad` is one 256-bit AVX register on x86-64 CPUs that have AVX
//! (detected at run time), else a pair of two-lane vectors: NEON on
//! AArch64, SSE2 on x86-64, two scalars elsewhere. `RSDAG_SSE2` set in the
//! environment keeps x86-64 to SSE2 (here and in the JIT), so both paths
//! can be tested on one machine.

/// Four `f64` lanes, every operation lane by lane and rounded once.
trait Quad: Copy {
    fn splat(v: f64) -> Self;
    /// # Safety
    /// `p` points at four readable `f64`.
    unsafe fn load(p: *const f64) -> Self;
    /// # Safety
    /// `p` points at four writable `f64`.
    unsafe fn store(self, p: *mut f64);
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    /// The magnitude, the sign bit cleared (as `f64::abs`).
    fn abs(self) -> Self;
    /// All bits set in the lanes where `self > o` (false where either is
    /// NaN), clear elsewhere.
    fn gt(self, o: Self) -> Self;
    /// All bits set in the lanes where `self == o`.
    fn eq(self, o: Self) -> Self;
    /// `t` in the lanes where `mask` (a comparison's) is set, `e` elsewhere.
    fn select(mask: Self, t: Self, e: Self) -> Self;
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(0.0)
    }
    #[inline(always)]
    fn of(v: [f64; 4]) -> Self {
        // SAFETY: `v` holds four.
        unsafe { Self::load(v.as_ptr()) }
    }
    #[inline(always)]
    fn lanes(self) -> [f64; 4] {
        let mut v = [0.0; 4];
        // SAFETY: `v` holds four.
        unsafe { self.store(v.as_mut_ptr()) };
        v
    }
}

/// Two `f64` lanes.
#[derive(Clone, Copy)]
struct V2(Inner);

#[cfg(target_arch = "aarch64")]
type Inner = std::arch::aarch64::float64x2_t;
#[cfg(target_arch = "x86_64")]
type Inner = std::arch::x86_64::__m128d;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
type Inner = [f64; 2];

// SAFETY (the `unsafe` blocks of both impls): NEON and SSE2 are part of
// the base architecture; loads and stores are the caller's contract.
#[cfg(target_arch = "aarch64")]
impl V2 {
    #[inline(always)]
    fn splat(v: f64) -> V2 {
        unsafe { V2(std::arch::aarch64::vdupq_n_f64(v)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> V2 {
        V2(std::arch::aarch64::vld1q_f64(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        std::arch::aarch64::vst1q_f64(p, self.0)
    }
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        unsafe { V2(std::arch::aarch64::vaddq_f64(self.0, o.0)) }
    }
    #[inline(always)]
    fn sub(self, o: V2) -> V2 {
        unsafe { V2(std::arch::aarch64::vsubq_f64(self.0, o.0)) }
    }
    #[inline(always)]
    fn mul(self, o: V2) -> V2 {
        unsafe { V2(std::arch::aarch64::vmulq_f64(self.0, o.0)) }
    }
    #[inline(always)]
    fn div(self, o: V2) -> V2 {
        unsafe { V2(std::arch::aarch64::vdivq_f64(self.0, o.0)) }
    }
    #[inline(always)]
    fn abs(self) -> V2 {
        unsafe { V2(std::arch::aarch64::vabsq_f64(self.0)) }
    }
    #[inline(always)]
    fn gt(self, o: V2) -> V2 {
        use std::arch::aarch64::*;
        unsafe { V2(vreinterpretq_f64_u64(vcgtq_f64(self.0, o.0))) }
    }
    #[inline(always)]
    fn eq(self, o: V2) -> V2 {
        use std::arch::aarch64::*;
        unsafe { V2(vreinterpretq_f64_u64(vceqq_f64(self.0, o.0))) }
    }
    #[inline(always)]
    fn select(mask: V2, t: V2, e: V2) -> V2 {
        use std::arch::aarch64::*;
        unsafe { V2(vbslq_f64(vreinterpretq_u64_f64(mask.0), t.0, e.0)) }
    }
}

#[cfg(target_arch = "x86_64")]
impl V2 {
    #[inline(always)]
    fn splat(v: f64) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_set1_pd(v)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> V2 {
        V2(std::arch::x86_64::_mm_loadu_pd(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        std::arch::x86_64::_mm_storeu_pd(p, self.0)
    }
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_add_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn sub(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_sub_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn mul(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_mul_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn div(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_div_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn abs(self) -> V2 {
        use std::arch::x86_64::*;
        unsafe { V2(_mm_andnot_pd(_mm_set1_pd(-0.0), self.0)) }
    }
    #[inline(always)]
    fn gt(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_cmpgt_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn eq(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_cmpeq_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn select(mask: V2, t: V2, e: V2) -> V2 {
        use std::arch::x86_64::*;
        unsafe {
            V2(_mm_or_pd(
                _mm_and_pd(mask.0, t.0),
                _mm_andnot_pd(mask.0, e.0),
            ))
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
impl V2 {
    #[inline(always)]
    fn splat(v: f64) -> V2 {
        V2([v; 2])
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> V2 {
        V2([*p, *p.add(1)])
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        *p = self.0[0];
        *p.add(1) = self.0[1];
    }
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        V2([self.0[0] + o.0[0], self.0[1] + o.0[1]])
    }
    #[inline(always)]
    fn sub(self, o: V2) -> V2 {
        V2([self.0[0] - o.0[0], self.0[1] - o.0[1]])
    }
    #[inline(always)]
    fn mul(self, o: V2) -> V2 {
        V2([self.0[0] * o.0[0], self.0[1] * o.0[1]])
    }
    #[inline(always)]
    fn div(self, o: V2) -> V2 {
        V2([self.0[0] / o.0[0], self.0[1] / o.0[1]])
    }
    #[inline(always)]
    fn abs(self) -> V2 {
        V2([self.0[0].abs(), self.0[1].abs()])
    }
    #[inline(always)]
    fn gt(self, o: V2) -> V2 {
        let m = |c: bool| f64::from_bits(if c { u64::MAX } else { 0 });
        V2([m(self.0[0] > o.0[0]), m(self.0[1] > o.0[1])])
    }
    #[inline(always)]
    fn eq(self, o: V2) -> V2 {
        let m = |c: bool| f64::from_bits(if c { u64::MAX } else { 0 });
        V2([m(self.0[0] == o.0[0]), m(self.0[1] == o.0[1])])
    }
    #[inline(always)]
    fn select(mask: V2, t: V2, e: V2) -> V2 {
        let pick = |l: usize| {
            if mask.0[l].to_bits() != 0 {
                t.0[l]
            } else {
                e.0[l]
            }
        };
        V2([pick(0), pick(1)])
    }
}

/// A [`Quad`] as two [`V2`], lanes `0, 1` and `2, 3`.
#[derive(Clone, Copy)]
struct Pair(V2, V2);

impl Quad for Pair {
    #[inline(always)]
    fn splat(v: f64) -> Pair {
        Pair(V2::splat(v), V2::splat(v))
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Pair {
        Pair(V2::load(p), V2::load(p.add(2)))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        self.0.store(p);
        self.1.store(p.add(2));
    }
    #[inline(always)]
    fn add(self, o: Pair) -> Pair {
        Pair(self.0.add(o.0), self.1.add(o.1))
    }
    #[inline(always)]
    fn sub(self, o: Pair) -> Pair {
        Pair(self.0.sub(o.0), self.1.sub(o.1))
    }
    #[inline(always)]
    fn mul(self, o: Pair) -> Pair {
        Pair(self.0.mul(o.0), self.1.mul(o.1))
    }
    #[inline(always)]
    fn div(self, o: Pair) -> Pair {
        Pair(self.0.div(o.0), self.1.div(o.1))
    }
    #[inline(always)]
    fn abs(self) -> Pair {
        Pair(self.0.abs(), self.1.abs())
    }
    #[inline(always)]
    fn gt(self, o: Pair) -> Pair {
        Pair(self.0.gt(o.0), self.1.gt(o.1))
    }
    #[inline(always)]
    fn eq(self, o: Pair) -> Pair {
        Pair(self.0.eq(o.0), self.1.eq(o.1))
    }
    #[inline(always)]
    fn select(mask: Pair, t: Pair, e: Pair) -> Pair {
        Pair(V2::select(mask.0, t.0, e.0), V2::select(mask.1, t.1, e.1))
    }
}

/// A [`Quad`] in one AVX register: separate multiplies and adds, never
/// FMA (the kernels enable `avx` alone).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Avx(std::arch::x86_64::__m256d);

// SAFETY (the `unsafe` blocks): an `Avx` is only made inside the AVX
// instances of `entry!`, which run after `avx()` found the CPU has AVX;
// loads and stores are the caller's contract.
#[cfg(target_arch = "x86_64")]
impl Quad for Avx {
    #[inline(always)]
    fn splat(v: f64) -> Avx {
        unsafe { Avx(std::arch::x86_64::_mm256_set1_pd(v)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Avx {
        Avx(std::arch::x86_64::_mm256_loadu_pd(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        std::arch::x86_64::_mm256_storeu_pd(p, self.0)
    }
    #[inline(always)]
    fn add(self, o: Avx) -> Avx {
        unsafe { Avx(std::arch::x86_64::_mm256_add_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn sub(self, o: Avx) -> Avx {
        unsafe { Avx(std::arch::x86_64::_mm256_sub_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn mul(self, o: Avx) -> Avx {
        unsafe { Avx(std::arch::x86_64::_mm256_mul_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn div(self, o: Avx) -> Avx {
        unsafe { Avx(std::arch::x86_64::_mm256_div_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn abs(self) -> Avx {
        use std::arch::x86_64::*;
        unsafe { Avx(_mm256_andnot_pd(_mm256_set1_pd(-0.0), self.0)) }
    }
    #[inline(always)]
    fn gt(self, o: Avx) -> Avx {
        use std::arch::x86_64::*;
        unsafe { Avx(_mm256_cmp_pd::<_CMP_GT_OQ>(self.0, o.0)) }
    }
    #[inline(always)]
    fn eq(self, o: Avx) -> Avx {
        use std::arch::x86_64::*;
        unsafe { Avx(_mm256_cmp_pd::<_CMP_EQ_OQ>(self.0, o.0)) }
    }
    #[inline(always)]
    fn select(mask: Avx, t: Avx, e: Avx) -> Avx {
        unsafe { Avx(std::arch::x86_64::_mm256_blendv_pd(e.0, t.0, mask.0)) }
    }
}

/// Whether the kernels run on [`Avx`]: the CPU has it and `RSDAG_SSE2` is
/// not set, decided once.
#[cfg(target_arch = "x86_64")]
fn avx() -> bool {
    static AVX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVX.get_or_init(|| {
        std::env::var_os("RSDAG_SSE2").is_none() && std::is_x86_feature_detected!("avx")
    })
}

/// `fn $name` running `$imp::<Avx>` compiled for AVX where [`avx`] holds,
/// `$imp::<Pair>` otherwise. `$imp` and everything it calls inline into
/// it, so the whole kernel is one AVX function. A closure does not inherit
/// the feature: one holding a [`Quad`] operation would call every AVX
/// intrinsic out of line, so none does (the solve's scratch is the
/// caller's, not borrowed from a thread local in `with`).
macro_rules! entry {
    ($(#[$m:meta])* fn $name:ident<$($g:ident: $b:path),*>($($a:ident: $t:ty),*) $(-> $r:ty)? = $imp:ident) => {
        $(#[$m])*
        pub(crate) fn $name<$($g: $b),*>($($a: $t),*) $(-> $r)? {
            #[cfg(target_arch = "x86_64")]
            if avx() {
                #[target_feature(enable = "avx")]
                fn wide<$($g: $b),*>($($a: $t),*) $(-> $r)? {
                    $imp::<Avx, $($g),*>($($a),*)
                }
                // SAFETY: the CPU has AVX.
                return unsafe { wide($($a),*) };
            }
            $imp::<Pair, $($g),*>($($a),*)
        }
    };
}

entry!(
    /// [`crate::semantics::dot_slice_t`] in `f64`.
    fn dot<>(a: &[f64], b: &[f64]) -> f64 = dot_q
);
entry!(
    /// [`crate::semantics::gemv_t`] in `f64`: four rows at a time, the
    /// vector loaded once per chunk for all four.
    fn gemv<>(a: &[f64], x: &[f64], m: usize, n: usize, out: &mut [f64]) = gemv_q
);
entry!(
    /// [`gemm`] with every entry stored through `st(index, value)`.
    fn gemm_with<F: FnMut(usize, f64)>(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, st: F) = gemm_q
);
entry!(
    /// [`crate::semantics::solve_batch_into`] in `f64`: four systems at a
    /// time side by side in the lanes where they are small, the rest one
    /// by one.
    fn solve_batch<>(a: &[f64], b: &[f64], n: usize, k: usize, count: usize, out: &mut [f64], scratch: &mut [f64]) = solve_batch_q
);
entry!(
    /// [`crate::semantics::solve_many_t`] in `f64`: the same panel-blocked
    /// elimination step for step (pivot search, swaps, panel updates, the
    /// panel's rows against the trailing columns, the trailing update
    /// through [`gemm`], the back-substitution), the row updates four
    /// lanes wide, the augmented matrix in the caller's scratch.
    fn solve_many<>(a: &[f64], b: &[f64], n: usize, k: usize, out: &mut [f64], scratch: &mut [f64]) = solve_q
);

/// The merge of the four accumulators and the tail past the last chunk
/// of four, in the reference order.
#[inline(always)]
fn finish<Q: Quad>(acc: Q, a: &[f64], b: &[f64], from: usize) -> f64 {
    let [l0, l1, l2, l3] = acc.lanes();
    let mut s = (l0 + l1) + (l2 + l3);
    for l in from..a.len() {
        s += a[l] * b[l];
    }
    s
}

#[inline(always)]
fn dot_q<Q: Quad>(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    assert!(b.len() >= n, "dot operands of one length");
    let ch = n / 4;
    let mut acc = Q::zero();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    for c in 0..ch {
        let o = 4 * c;
        // SAFETY: `o + 3 < 4 * ch <= n <= a.len(), b.len()`.
        unsafe { acc = acc.add(Q::load(pa.add(o)).mul(Q::load(pb.add(o)))) }
    }
    finish(acc, a, b, ch * 4)
}

#[inline(always)]
fn gemv_q<Q: Quad>(a: &[f64], x: &[f64], m: usize, n: usize, out: &mut [f64]) {
    assert!(a.len() >= m * n && x.len() >= n && out.len() >= m);
    let ch = n / 4;
    let px = x.as_ptr();
    let mut i = 0;
    while i + 4 <= m {
        let p: [*const f64; 4] = std::array::from_fn(|r| a[(i + r) * n..].as_ptr());
        let mut acc = [Q::zero(); 4];
        for c in 0..ch {
            let o = 4 * c;
            // SAFETY: row `r` starts at `(i + r) * n` with `n` elements,
            // `o + 3 < n`; `x` has `n`.
            unsafe {
                let xv = Q::load(px.add(o));
                for (r, ar) in acc.iter_mut().enumerate() {
                    *ar = ar.add(Q::load(p[r].add(o)).mul(xv));
                }
            }
        }
        for (r, &ar) in acc.iter().enumerate() {
            out[i + r] = finish(ar, &a[(i + r) * n..][..n], x, ch * 4);
        }
        i += 4;
    }
    for r in i..m {
        out[r] = dot_q::<Q>(&a[r * n..][..n], x);
    }
}

/// [`crate::semantics::gemm_t`] in `f64`: four rows of `a` against two
/// rows of `b` at a time, eight accumulators.
pub(crate) fn gemm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    assert!(a.len() >= m * k && b.len() >= n * k && out.len() >= m * n);
    gemm_with(a, b, m, k, n, |i, v| out[i] = v);
}

/// [`crate::semantics::gemm_fold_t`] in `f64`: the folds against an
/// operand at the store, the self folds after.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_fold(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    c: Option<&[f64]>,
    codes: &[u32],
    out: &mut [f64],
) {
    use crate::tape::Fold;
    assert!(a.len() >= m * k && b.len() >= n * k && out.len() >= m * n);
    let codes = &codes[..m * n];
    // One code for every entry, the common case, without a decode per
    // store.
    let uniform = codes.windows(2).all(|w| w[0] == w[1]);
    let f = Fold(codes[0]);
    match (uniform, f.0 & 3, f.is_self(), c) {
        (true, 0, _, _) => gemm_with(a, b, m, k, n, |i, v| out[i] = v),
        (true, 3, _, _) => gemm_with(a, b, m, k, n, |i, v| out[i] = -v),
        (true, 1, false, Some(c)) => {
            let c = &c[..m * n];
            gemm_with(a, b, m, k, n, |i, v| out[i] = c[i] - v)
        }
        (true, 2, false, Some(c)) => {
            let c = &c[..m * n];
            gemm_with(a, b, m, k, n, |i, v| out[i] = c[i] + v)
        }
        _ if codes.iter().any(|&code| Fold(code).is_self()) => {
            gemm_with(a, b, m, k, n, |i, v| out[i] = v);
            crate::semantics::fold_in_place(codes, c, out);
        }
        _ => gemm_with(a, b, m, k, n, |i, v| {
            out[i] = Fold(codes[i]).fold(c.map_or(v, |c| c[i]), v)
        }),
    }
}

#[inline(always)]
fn gemm_q<Q: Quad, F: FnMut(usize, f64)>(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    mut st: F,
) {
    assert!(a.len() >= m * k && b.len() >= n * k);
    let ch = k / 4;
    let mut i = 0;
    while i + 4 <= m {
        let pa: [*const f64; 4] = std::array::from_fn(|r| a[(i + r) * k..].as_ptr());
        let mut j = 0;
        while j + 2 <= n {
            let pb0 = b[j * k..].as_ptr();
            let pb1 = b[(j + 1) * k..].as_ptr();
            let mut acc = [[Q::zero(); 2]; 4];
            for c in 0..ch {
                let o = 4 * c;
                // SAFETY: every row has `k` elements and `o + 3 < k`.
                unsafe {
                    let b0 = Q::load(pb0.add(o));
                    let b1 = Q::load(pb1.add(o));
                    for (r, ar) in acc.iter_mut().enumerate() {
                        let av = Q::load(pa[r].add(o));
                        ar[0] = ar[0].add(av.mul(b0));
                        ar[1] = ar[1].add(av.mul(b1));
                    }
                }
            }
            for (r, ar) in acc.iter().enumerate() {
                let ra = &a[(i + r) * k..][..k];
                for (q, &aq) in ar.iter().enumerate() {
                    st(
                        (i + r) * n + j + q,
                        finish(aq, ra, &b[(j + q) * k..][..k], ch * 4),
                    );
                }
            }
            j += 2;
        }
        for r in 0..4 {
            for jj in j..n {
                st(
                    (i + r) * n + jj,
                    dot_q::<Q>(&a[(i + r) * k..][..k], &b[jj * k..][..k]),
                );
            }
        }
        i += 4;
    }
    for r in i..m {
        for j in 0..n {
            st(r * n + j, dot_q::<Q>(&a[r * k..][..k], &b[j * k..][..k]));
        }
    }
}

/// `row_i[j] -= l * row_k[j]` over `j`, four lanes at a time; each element
/// one product and one difference, as the reference.
#[inline(always)]
fn axpy_sub<Q: Quad>(row_i: &mut [f64], row_k: &[f64], l: f64) {
    let n = row_i.len().min(row_k.len());
    let ch = n / 4;
    let lv = Q::splat(l);
    let (pi, pk) = (row_i.as_mut_ptr(), row_k.as_ptr());
    for c in 0..ch {
        let o = 4 * c;
        // SAFETY: `o + 3 < n` within both rows.
        unsafe {
            let v = Q::load(pi.add(o)).sub(lv.mul(Q::load(pk.add(o))));
            v.store(pi.add(o));
        }
    }
    for j in ch * 4..n {
        row_i[j] -= l * row_k[j];
    }
}

#[inline(always)]
fn solve_q<Q: Quad>(
    a: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
    out: &mut [f64],
    scratch: &mut [f64],
) {
    use crate::semantics::{lu_panel, LU_PANEL_LARGE, LU_PANEL_SMALL};
    macro_rules! fixed {
        ($($n:literal)*) => {
            match n {
                $($n => return solve_fixed::<$n>(a, b, k, out),)*
                _ => {}
            }
        };
    }
    fixed!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    match lu_panel(n) {
        0 => solve_unblocked::<Q>(a, b, n, k, out, scratch),
        LU_PANEL_SMALL => solve_blocked::<Q, LU_PANEL_SMALL>(a, b, n, k, out, scratch),
        _ => solve_blocked::<Q, LU_PANEL_LARGE>(a, b, n, k, out, scratch),
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn solve_batch_q<Q: Quad>(
    a: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
    count: usize,
    out: &mut [f64],
    scratch: &mut [f64],
) {
    let (sa, sb) = (n * n, n * k);
    let mut c = 0;
    use crate::semantics::{SOLVE_BATCH_MAX_K, SOLVE_BATCH_MAX_N};
    if n <= SOLVE_BATCH_MAX_N && k <= SOLVE_BATCH_MAX_K {
        macro_rules! lanes {
            ($($n:literal)*) => {
                match n {
                    $($n => while c + 4 <= count {
                        solve_lanes::<Q, $n>(&a[c * sa..], &b[c * sb..], k, &mut out[c * sb..]);
                        c += 4;
                    },)*
                    _ => {}
                }
            };
        }
        lanes!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    }
    for c in c..count {
        solve_q::<Q>(
            &a[c * sa..(c + 1) * sa],
            &b[c * sb..(c + 1) * sb],
            n,
            k,
            &mut out[c * sb..(c + 1) * sb],
            scratch,
        );
    }
}

/// Four systems of `N` unknowns and `k` right-hand sides (at most
/// [`crate::semantics::SOLVE_BATCH_MAX_K`]) side by side, lane `q` system `q`: `a` holds their
/// matrices back to back, `b` and `out` their right-hand sides and
/// solutions. Each lane runs [`solve_fixed`] on its own system: its own
/// pivot, its rows swapped where its pivot says (a lane select), its own
/// products, differences and quotients in the reference's order, so every
/// solution is bit-identical to its system solved alone. Four independent
/// eliminations share one chain of latencies.
#[inline(always)]
fn solve_lanes<Q: Quad, const N: usize>(a: &[f64], b: &[f64], k: usize, out: &mut [f64]) {
    let (sa, sb) = (N * N, N * k);
    // `col[j][i]` is entry `(i, j)`, `x[c][i]` entry `i` of right-hand
    // side `c`, of every lane's system.
    let mut col = [[Q::zero(); N]; N];
    for i in 0..N {
        for (j, cj) in col.iter_mut().enumerate() {
            cj[i] = Q::of(std::array::from_fn(|q| a[q * sa + i * N + j]));
        }
    }
    let mut x = [[Q::zero(); N]; crate::semantics::SOLVE_BATCH_MAX_K];
    for (c, xc) in x.iter_mut().enumerate().take(k) {
        for (i, xi) in xc.iter_mut().enumerate() {
            *xi = Q::of(std::array::from_fn(|q| b[q * sb + c * N + i]));
        }
    }
    for kk in 0..N {
        let mut best = col[kk][kk].abs();
        let mut p = Q::splat(kk as f64);
        for i in kk + 1..N {
            let v = col[kk][i].abs();
            let gt = v.gt(best);
            best = Q::select(gt, v, best);
            p = Q::select(gt, Q::splat(i as f64), p);
        }
        for i in kk + 1..N {
            let swap = p.eq(Q::splat(i as f64));
            for cj in col.iter_mut() {
                let (u, v) = (cj[kk], cj[i]);
                cj[kk] = Q::select(swap, v, u);
                cj[i] = Q::select(swap, u, v);
            }
            for xc in x.iter_mut().take(k) {
                let (u, v) = (xc[kk], xc[i]);
                xc[kk] = Q::select(swap, v, u);
                xc[i] = Q::select(swap, u, v);
            }
        }
        let piv = col[kk][kk];
        let mut l = [Q::zero(); N];
        for i in kk + 1..N {
            l[i] = col[kk][i].div(piv);
            col[kk][i] = l[i];
        }
        for cj in col.iter_mut().skip(kk + 1) {
            let u = cj[kk];
            for i in kk + 1..N {
                cj[i] = cj[i].sub(l[i].mul(u));
            }
        }
        for xc in x.iter_mut().take(k) {
            let u = xc[kk];
            for i in kk + 1..N {
                xc[i] = xc[i].sub(l[i].mul(u));
            }
        }
    }
    for (c, xc) in x.iter_mut().enumerate().take(k) {
        for i in (0..N).rev() {
            let mut s = xc[i];
            for j in i + 1..N {
                s = s.sub(col[j][i].mul(xc[j]));
            }
            xc[i] = s.div(col[i][i]);
            for (q, v) in xc[i].lanes().into_iter().enumerate() {
                out[q * sb + c * N + i] = v;
            }
        }
    }
}

/// The unblocked elimination of `N` unknowns with the size a constant, so
/// its loops unroll, the matrix by columns on the stack: the pivot search,
/// the multipliers and each column's update run down a column, four rows
/// to a vector. The right-hand sides are eliminated in `out`. Every entry
/// sees the reference's products and differences in the reference's
/// order (a row update is entry by entry, the right-hand sides' columns
/// included), so the result is bit-identical.
#[inline(always)]
fn solve_fixed<const N: usize>(a: &[f64], b: &[f64], k: usize, out: &mut [f64]) {
    // `col[j][i]` is entry `(i, j)`.
    let mut col = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            col[j][i] = a[i * N + j];
        }
    }
    let x = &mut out[..N * k];
    x.copy_from_slice(&b[..N * k]);
    for kk in 0..N {
        let mut p = kk;
        let mut best = col[kk][kk].abs();
        for i in kk + 1..N {
            if col[kk][i].abs() > best {
                best = col[kk][i].abs();
                p = i;
            }
        }
        if p != kk {
            for c in col.iter_mut() {
                c.swap(kk, p);
            }
            for c in 0..k {
                x.swap(c * N + kk, c * N + p);
            }
        }
        let piv = col[kk][kk];
        let mut l = [0.0f64; N];
        for i in kk + 1..N {
            l[i] = col[kk][i] / piv;
            col[kk][i] = l[i];
        }
        for c in col.iter_mut().skip(kk + 1) {
            let u = c[kk];
            for i in kk + 1..N {
                c[i] -= l[i] * u;
            }
        }
        for c in 0..k {
            let u = x[c * N + kk];
            for i in kk + 1..N {
                x[c * N + i] -= l[i] * u;
            }
        }
    }
    for c in 0..k {
        let x = &mut x[c * N..(c + 1) * N];
        for i in (0..N).rev() {
            let mut s = x[i];
            for j in i + 1..N {
                s -= col[j][i] * x[j];
            }
            x[i] = s / col[i][i];
        }
    }
}

/// The right-looking elimination of [`crate::semantics::solve_many_generic`]
/// without panels.
#[inline(always)]
fn solve_unblocked<Q: Quad>(
    a: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
    out: &mut [f64],
    scratch: &mut [f64],
) {
    let [m, ..] = crate::semantics::solve_parts(scratch, a, b, n, k);
    let w = n + k;
    for kk in 0..n {
        let piv = pivot(m, n, w, kk);
        let (top, rest) = m.split_at_mut((kk + 1) * w);
        let row_k = &top[kk * w..(kk + 1) * w];
        for row_i in rest.chunks_exact_mut(w) {
            let l = row_i[kk] / piv;
            row_i[kk] = l;
            axpy_sub::<Q>(&mut row_i[kk + 1..w], &row_k[kk + 1..w], l);
        }
    }
    back_substitute::<Q>(m, n, k, w, out);
}

/// The partial pivot of column `kk`: the first row at or below `kk` of the
/// largest magnitude, swapped into row `kk`; returns the pivot.
#[inline(always)]
fn pivot(m: &mut [f64], n: usize, w: usize, kk: usize) -> f64 {
    let mut p = kk;
    let mut best = m[kk * w + kk].abs();
    for i in kk + 1..n {
        let v = m[i * w + kk].abs();
        if v > best {
            best = v;
            p = i;
        }
    }
    if p != kk {
        for j in 0..w {
            m.swap(kk * w + j, p * w + j);
        }
    }
    m[kk * w + kk]
}

/// Back substitution over the eliminated augmented matrix, four
/// right-hand sides per [`Quad`]: each lane is its column's own sequence
/// of products and differences.
#[inline(always)]
fn back_substitute<Q: Quad>(m: &[f64], n: usize, k: usize, w: usize, out: &mut [f64]) {
    let mut c = 0;
    while c + 4 <= k {
        let x = &mut out[c * n..(c + 4) * n];
        for i in (0..n).rev() {
            let mut s = Q::of(std::array::from_fn(|q| m[i * w + n + c + q]));
            for j in i + 1..n {
                let xj = Q::of(std::array::from_fn(|q| x[q * n + j]));
                s = s.sub(Q::splat(m[i * w + j]).mul(xj));
            }
            for (q, v) in s.lanes().into_iter().enumerate() {
                x[q * n + i] = v / m[i * w + i];
            }
        }
        c += 4;
    }
    while c < k {
        let x = &mut out[c * n..(c + 1) * n];
        for i in (0..n).rev() {
            let mut s = m[i * w + n + c];
            for j in i + 1..n {
                let t = m[i * w + j] * x[j];
                s -= t;
            }
            x[i] = s / m[i * w + i];
        }
        c += 1;
    }
}

#[inline(always)]
fn solve_blocked<Q: Quad, const NB: usize>(
    a: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
    out: &mut [f64],
    scratch: &mut [f64],
) {
    let [m, ut, lrows, prod] = crate::semantics::solve_parts(scratch, a, b, n, k);
    let w = n + k;
    let mut k0 = 0;
    while k0 < n {
        let k1 = (k0 + NB).min(n);
        for kk in k0..k1 {
            let piv = pivot(m, n, w, kk);
            let (top, rest) = m.split_at_mut((kk + 1) * w);
            let row_k = &top[kk * w..(kk + 1) * w];
            for row_i in rest.chunks_exact_mut(w) {
                let l = row_i[kk] / piv;
                row_i[kk] = l;
                axpy_sub::<Q>(&mut row_i[kk + 1..k1], &row_k[kk + 1..k1], l);
            }
        }
        // The panel's unit lower triangle against the columns right of it.
        for kk in k0..k1 {
            let (top, rest) = m.split_at_mut((kk + 1) * w);
            let row_k = &top[kk * w..(kk + 1) * w];
            for row_i in rest[..(k1 - kk - 1) * w].chunks_exact_mut(w) {
                let l = row_i[kk];
                axpy_sub::<Q>(&mut row_i[k1..w], &row_k[k1..w], l);
            }
        }
        if k1 < n {
            let nb = k1 - k0;
            let cols = w - k1;
            let ut = &mut ut[..cols * nb];
            for (jj, j) in (k1..w).enumerate() {
                for (q, kk) in (k0..k1).enumerate() {
                    ut[jj * nb + q] = m[kk * w + j];
                }
            }
            let mut i = k1;
            while i < n {
                let rows = (n - i).min(4);
                for r in 0..rows {
                    let at = (i + r) * w;
                    lrows[r * nb..(r + 1) * nb].copy_from_slice(&m[at + k0..at + k1]);
                }
                let pr = &mut prod[..rows * cols];
                gemm_q::<Q, _>(&lrows[..rows * nb], ut, rows, nb, cols, |e, v| pr[e] = v);
                for r in 0..rows {
                    let at = (i + r) * w + k1;
                    for jj in 0..cols {
                        m[at + jj] -= prod[r * cols + jj];
                    }
                }
                i += rows;
            }
        }
        k0 = k1;
    }
    back_substitute::<Q>(m, n, k, w, out);
}
