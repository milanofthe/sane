//! The `f64` twins of the dot-fold kernels: the same fold as the generic
//! reference in [`crate::semantics`] (four accumulators, merged as
//! `(0 + 1) + (2 + 3)`, then the tail in order), as two-lane vectors where
//! the target has them. Bit-identical to the reference by construction:
//! lane `l` of the vector pair is accumulator `l`, every product and every
//! sum rounds once, nothing is fused. The generic reference stays the
//! definition; the tests hold the twins to it.

/// Two `f64` lanes: NEON on AArch64, SSE2 on x86-64 (part of the base
/// architecture, so no dispatch), two scalars elsewhere.
#[derive(Clone, Copy)]
struct V2(Inner);

#[cfg(target_arch = "aarch64")]
type Inner = std::arch::aarch64::float64x2_t;
#[cfg(target_arch = "x86_64")]
type Inner = std::arch::x86_64::__m128d;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
type Inner = [f64; 2];

#[cfg(target_arch = "aarch64")]
impl V2 {
    #[inline(always)]
    fn zero() -> V2 {
        unsafe { V2(std::arch::aarch64::vdupq_n_f64(0.0)) }
    }
    #[inline(always)]
    fn splat(v: f64) -> V2 {
        unsafe { V2(std::arch::aarch64::vdupq_n_f64(v)) }
    }
    #[inline(always)]
    fn pair(a: f64, b: f64) -> V2 {
        let v = [a, b];
        unsafe { V2(std::arch::aarch64::vld1q_f64(v.as_ptr())) }
    }
    /// # Safety
    /// `p` points at two writable `f64`.
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        std::arch::aarch64::vst1q_f64(p, self.0)
    }
    #[inline(always)]
    fn sub(self, o: V2) -> V2 {
        unsafe { V2(std::arch::aarch64::vsubq_f64(self.0, o.0)) }
    }
    /// # Safety
    /// `p` points at two readable `f64`.
    #[inline(always)]
    unsafe fn load(p: *const f64) -> V2 {
        V2(std::arch::aarch64::vld1q_f64(p))
    }
    #[inline(always)]
    fn mul(self, o: V2) -> V2 {
        unsafe { V2(std::arch::aarch64::vmulq_f64(self.0, o.0)) }
    }
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        unsafe { V2(std::arch::aarch64::vaddq_f64(self.0, o.0)) }
    }
    #[inline(always)]
    fn lanes(self) -> [f64; 2] {
        unsafe {
            [
                std::arch::aarch64::vgetq_lane_f64(self.0, 0),
                std::arch::aarch64::vgetq_lane_f64(self.0, 1),
            ]
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl V2 {
    #[inline(always)]
    fn zero() -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_setzero_pd()) }
    }
    #[inline(always)]
    fn splat(v: f64) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_set1_pd(v)) }
    }
    #[inline(always)]
    fn pair(a: f64, b: f64) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_set_pd(b, a)) }
    }
    /// # Safety
    /// `p` points at two writable `f64`.
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        std::arch::x86_64::_mm_storeu_pd(p, self.0)
    }
    #[inline(always)]
    fn sub(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_sub_pd(self.0, o.0)) }
    }
    /// # Safety
    /// `p` points at two readable `f64`.
    #[inline(always)]
    unsafe fn load(p: *const f64) -> V2 {
        V2(std::arch::x86_64::_mm_loadu_pd(p))
    }
    #[inline(always)]
    fn mul(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_mul_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_add_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn lanes(self) -> [f64; 2] {
        let mut out = [0.0; 2];
        unsafe { std::arch::x86_64::_mm_storeu_pd(out.as_mut_ptr(), self.0) };
        out
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
impl V2 {
    #[inline(always)]
    fn zero() -> V2 {
        V2([0.0; 2])
    }
    #[inline(always)]
    fn splat(v: f64) -> V2 {
        V2([v; 2])
    }
    #[inline(always)]
    fn pair(a: f64, b: f64) -> V2 {
        V2([a, b])
    }
    /// # Safety
    /// `p` points at two writable `f64`.
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        *p = self.0[0];
        *p.add(1) = self.0[1];
    }
    #[inline(always)]
    fn sub(self, o: V2) -> V2 {
        V2([self.0[0] - o.0[0], self.0[1] - o.0[1]])
    }
    /// # Safety
    /// `p` points at two readable `f64`.
    #[inline(always)]
    unsafe fn load(p: *const f64) -> V2 {
        V2([*p, *p.add(1)])
    }
    #[inline(always)]
    fn mul(self, o: V2) -> V2 {
        V2([self.0[0] * o.0[0], self.0[1] * o.0[1]])
    }
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        V2([self.0[0] + o.0[0], self.0[1] + o.0[1]])
    }
    #[inline(always)]
    fn lanes(self) -> [f64; 2] {
        self.0
    }
}

/// The merge of the four accumulators and the tail past the last chunk
/// of four, in the reference order.
#[inline(always)]
fn finish(c0: V2, c1: V2, a: &[f64], b: &[f64], from: usize) -> f64 {
    let [l0, l1] = c0.lanes();
    let [l2, l3] = c1.lanes();
    let mut s = (l0 + l1) + (l2 + l3);
    for l in from..a.len() {
        s += a[l] * b[l];
    }
    s
}

/// [`crate::semantics::dot_slice_t`] in `f64`.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    assert!(b.len() >= n, "dot operands of one length");
    let ch = n / 4;
    let (mut c0, mut c1) = (V2::zero(), V2::zero());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    for c in 0..ch {
        let o = 4 * c;
        // SAFETY: `o + 3 < 4 * ch <= n <= a.len(), b.len()`.
        unsafe {
            c0 = c0.add(V2::load(pa.add(o)).mul(V2::load(pb.add(o))));
            c1 = c1.add(V2::load(pa.add(o + 2)).mul(V2::load(pb.add(o + 2))));
        }
    }
    finish(c0, c1, a, b, ch * 4)
}

/// [`crate::semantics::gemv_t`] in `f64`: four rows at a time, each row
/// its own accumulator pair, the vector loaded once per chunk for all
/// four.
pub(crate) fn gemv(a: &[f64], x: &[f64], m: usize, n: usize, out: &mut [f64]) {
    assert!(a.len() >= m * n && x.len() >= n && out.len() >= m);
    let ch = n / 4;
    let px = x.as_ptr();
    let mut i = 0;
    while i + 4 <= m {
        let p = [
            a[i * n..].as_ptr(),
            a[(i + 1) * n..].as_ptr(),
            a[(i + 2) * n..].as_ptr(),
            a[(i + 3) * n..].as_ptr(),
        ];
        let mut acc = [[V2::zero(); 2]; 4];
        for c in 0..ch {
            let o = 4 * c;
            // SAFETY: row `r` starts at `(i + r) * n` with `n` elements,
            // `o + 3 < n`; `x` has `n`.
            unsafe {
                let x0 = V2::load(px.add(o));
                let x1 = V2::load(px.add(o + 2));
                for (r, ar) in acc.iter_mut().enumerate() {
                    ar[0] = ar[0].add(V2::load(p[r].add(o)).mul(x0));
                    ar[1] = ar[1].add(V2::load(p[r].add(o + 2)).mul(x1));
                }
            }
        }
        for (r, ar) in acc.iter().enumerate() {
            out[i + r] = finish(ar[0], ar[1], &a[(i + r) * n..][..n], x, ch * 4);
        }
        i += 4;
    }
    for r in i..m {
        out[r] = dot(&a[r * n..][..n], x);
    }
}

/// [`crate::semantics::gemm_t`] in `f64`: four rows of `a` against two
/// rows of `b` at a time, sixteen accumulator pairs.
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

/// [`gemm`] with every entry stored through `st(index, value)`.
fn gemm_with(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, mut st: impl FnMut(usize, f64)) {
    let ch = k / 4;
    let mut i = 0;
    while i + 4 <= m {
        let pa = [
            a[i * k..].as_ptr(),
            a[(i + 1) * k..].as_ptr(),
            a[(i + 2) * k..].as_ptr(),
            a[(i + 3) * k..].as_ptr(),
        ];
        let mut j = 0;
        while j + 2 <= n {
            let pb0 = b[j * k..].as_ptr();
            let pb1 = b[(j + 1) * k..].as_ptr();
            let mut acc = [[[V2::zero(); 2]; 2]; 4];
            for c in 0..ch {
                let o = 4 * c;
                // SAFETY: every row has `k` elements and `o + 3 < k`.
                unsafe {
                    let b00 = V2::load(pb0.add(o));
                    let b01 = V2::load(pb0.add(o + 2));
                    let b10 = V2::load(pb1.add(o));
                    let b11 = V2::load(pb1.add(o + 2));
                    for (r, ar) in acc.iter_mut().enumerate() {
                        let a0 = V2::load(pa[r].add(o));
                        let a1 = V2::load(pa[r].add(o + 2));
                        ar[0][0] = ar[0][0].add(a0.mul(b00));
                        ar[0][1] = ar[0][1].add(a1.mul(b01));
                        ar[1][0] = ar[1][0].add(a0.mul(b10));
                        ar[1][1] = ar[1][1].add(a1.mul(b11));
                    }
                }
            }
            for (r, ar) in acc.iter().enumerate() {
                for (q, aq) in ar.iter().enumerate() {
                    st(
                        (i + r) * n + j + q,
                        finish(
                            aq[0],
                            aq[1],
                            &a[(i + r) * k..][..k],
                            &b[(j + q) * k..][..k],
                            ch * 4,
                        ),
                    );
                }
            }
            j += 2;
        }
        for r in 0..4 {
            for jj in j..n {
                st(
                    (i + r) * n + jj,
                    dot(&a[(i + r) * k..][..k], &b[jj * k..][..k]),
                );
            }
        }
        i += 4;
    }
    for r in i..m {
        for j in 0..n {
            st(r * n + j, dot(&a[r * k..][..k], &b[j * k..][..k]));
        }
    }
}

/// `row_i[j] -= l * row_k[j]` over `j`, two lanes at a time; each element
/// one product and one difference, as the reference.
#[inline(always)]
fn axpy_sub(row_i: &mut [f64], row_k: &[f64], l: f64) {
    let n = row_i.len().min(row_k.len());
    let ch = n / 2;
    let lv = V2::splat(l);
    let (pi, pk) = (row_i.as_mut_ptr(), row_k.as_ptr());
    for c in 0..ch {
        let o = 2 * c;
        // SAFETY: `o + 1 < n` within both rows.
        unsafe {
            let v = V2::load(pi.add(o)).sub(lv.mul(V2::load(pk.add(o))));
            v.store(pi.add(o));
        }
    }
    for j in ch * 2..n {
        row_i[j] -= l * row_k[j];
    }
}

/// [`crate::semantics::solve_many_t`] in `f64`: the same panel-blocked
/// elimination step for step (pivot search, swaps, panel updates, the
/// panel's rows against the trailing columns, the trailing update through
/// [`gemm`], the back-substitution), the row updates two lanes wide, and
/// the augmented matrix in a scratch buffer kept per thread.
pub(crate) fn solve_many(a: &[f64], b: &[f64], n: usize, k: usize, out: &mut [f64]) {
    use crate::semantics::{LU_PANEL_LARGE, LU_PANEL_SMALL, LU_PANEL_SWITCH, LU_UNBLOCKED_MAX};
    if n <= LU_UNBLOCKED_MAX {
        solve_unblocked(a, b, n, k, out)
    } else if n < LU_PANEL_SWITCH {
        solve_blocked::<LU_PANEL_SMALL>(a, b, n, k, out)
    } else {
        solve_blocked::<LU_PANEL_LARGE>(a, b, n, k, out)
    }
}

/// The right-looking elimination of [`crate::semantics::solve_unblocked`],
/// rows two lanes wide, the augmented matrix in a scratch kept per thread.
fn solve_unblocked(a: &[f64], b: &[f64], n: usize, k: usize, out: &mut [f64]) {
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|cell| {
        let mut m = cell.borrow_mut();
        let w = n + k;
        m.clear();
        m.reserve(n * w);
        for i in 0..n {
            m.extend_from_slice(&a[i * n..(i + 1) * n]);
            for c in 0..k {
                m.push(b[c * n + i]);
            }
        }
        for kk in 0..n {
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
            let piv = m[kk * w + kk];
            let (top, rest) = m.split_at_mut((kk + 1) * w);
            let row_k = &top[kk * w..(kk + 1) * w];
            for row_i in rest.chunks_exact_mut(w) {
                let l = row_i[kk] / piv;
                row_i[kk] = l;
                axpy_sub(&mut row_i[kk + 1..w], &row_k[kk + 1..w], l);
            }
        }
        back_substitute(&m, n, k, w, out);
    });
}

/// Back substitution over the eliminated augmented matrix, two right-hand
/// sides per lane pair: each lane is its column's own sequence of products
/// and differences.
fn back_substitute(m: &[f64], n: usize, k: usize, w: usize, out: &mut [f64]) {
    let mut c = 0;
    while c + 2 <= k {
        let (x0, x1) = out[c * n..(c + 2) * n].split_at_mut(n);
        for i in (0..n).rev() {
            let mut sv = V2::pair(m[i * w + n + c], m[i * w + n + c + 1]);
            for j in i + 1..n {
                let mj = V2::splat(m[i * w + j]);
                sv = sv.sub(mj.mul(V2::pair(x0[j], x1[j])));
            }
            let [s0, s1] = sv.lanes();
            x0[i] = s0 / m[i * w + i];
            x1[i] = s1 / m[i * w + i];
        }
        c += 2;
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

fn solve_blocked<const NB: usize>(a: &[f64], b: &[f64], n: usize, k: usize, out: &mut [f64]) {
    thread_local! {
        static SCRATCH: std::cell::RefCell<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
    }
    SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (m, ut, lrows, prod) = &mut *guard;
        let w = n + k;
        m.clear();
        m.reserve(n * w);
        for i in 0..n {
            m.extend_from_slice(&a[i * n..(i + 1) * n]);
            for c in 0..k {
                m.push(b[c * n + i]);
            }
        }
        let mut k0 = 0;
        while k0 < n {
            let k1 = (k0 + NB).min(n);
            for kk in k0..k1 {
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
                let piv = m[kk * w + kk];
                let (top, rest) = m.split_at_mut((kk + 1) * w);
                let row_k = &top[kk * w..(kk + 1) * w];
                for (r, row_i) in rest.chunks_exact_mut(w).enumerate() {
                    let _ = r;
                    let l = row_i[kk] / piv;
                    row_i[kk] = l;
                    axpy_sub(&mut row_i[kk + 1..k1], &row_k[kk + 1..k1], l);
                }
            }
            // The panel's unit lower triangle against the columns right of it.
            for kk in k0..k1 {
                let (top, rest) = m.split_at_mut((kk + 1) * w);
                let row_k = &top[kk * w..(kk + 1) * w];
                for row_i in rest[..(k1 - kk - 1) * w].chunks_exact_mut(w) {
                    let l = row_i[kk];
                    axpy_sub(&mut row_i[k1..w], &row_k[k1..w], l);
                }
            }
            if k1 < n {
                let nb = k1 - k0;
                let cols = w - k1;
                ut.clear();
                ut.resize(cols * nb, 0.0);
                for (jj, j) in (k1..w).enumerate() {
                    for (q, kk) in (k0..k1).enumerate() {
                        ut[jj * nb + q] = m[kk * w + j];
                    }
                }
                lrows.clear();
                lrows.resize(4 * nb, 0.0);
                prod.clear();
                prod.resize(4 * cols, 0.0);
                let mut i = k1;
                while i < n {
                    let rows = (n - i).min(4);
                    for r in 0..rows {
                        let at = (i + r) * w;
                        lrows[r * nb..(r + 1) * nb].copy_from_slice(&m[at + k0..at + k1]);
                    }
                    gemm(
                        &lrows[..rows * nb],
                        ut,
                        rows,
                        nb,
                        cols,
                        &mut prod[..rows * cols],
                    );
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
        back_substitute(m, n, k, w, out);
    });
}
