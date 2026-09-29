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
    match lu_panel(n) {
        0 => solve_unblocked::<Q>(a, b, n, k, out, scratch),
        LU_PANEL_SMALL => solve_blocked::<Q, LU_PANEL_SMALL>(a, b, n, k, out, scratch),
        _ => solve_blocked::<Q, LU_PANEL_LARGE>(a, b, n, k, out, scratch),
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
