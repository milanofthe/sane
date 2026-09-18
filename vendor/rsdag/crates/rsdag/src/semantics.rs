//! The reference semantics: what every op computes, in one place.
//!
//! The arena sweep, the tape interpreter, the native backend's host
//! routines and the numeric twin of [`Builder`](crate::Builder) all take
//! their arithmetic from here, so a value cannot depend on which of them
//! computed it. That covers the domain guards (the `exp` cap at
//! [`EXP_LIMIT`], the `ln` floor at [`LN_FLOOR`], the `sqrt` clamp), the
//! special functions, which are defined here rather than taken from a
//! platform library, and the fold orders of reductions and dot products. A
//! backend that reproduces these bit for bit reproduces every program.

use crate::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use crate::scalar::Scalar;
use crate::tape::Fold;

/// The reduction of a slice in `T`: four accumulators, merged as
/// `(a0 + a1) + (a2 + a3)`, then the tail in order. The one fold order every
/// evaluator uses, so results agree to the bit; for fewer than four operands
/// the accumulators start at the identity and the result is the sequential
/// fold's exactly.
pub fn reduce_slice_t<T: Scalar>(op: ReduceOp, xs: &[T]) -> T {
    match op {
        ReduceOp::Sum => {
            let mut a = [T::zero(); 4];
            let ch = xs.len() / 4;
            for c in 0..ch {
                for l in 0..4 {
                    a[l] = a[l].add(xs[4 * c + l]);
                }
            }
            let mut acc = (a[0].add(a[1])).add(a[2].add(a[3]));
            for &x in &xs[ch * 4..] {
                acc = acc.add(x);
            }
            acc
        }
        ReduceOp::Product => {
            let mut a = [T::one(); 4];
            let ch = xs.len() / 4;
            for c in 0..ch {
                for l in 0..4 {
                    a[l] = a[l].mul(xs[4 * c + l]);
                }
            }
            let mut acc = (a[0].mul(a[1])).mul(a[2].mul(a[3]));
            for &x in &xs[ch * 4..] {
                acc = acc.mul(x);
            }
            acc
        }
        ReduceOp::Min => {
            let mut it = xs.iter();
            match it.next() {
                None => T::from_f64(f64::INFINITY),
                Some(&f) => it.fold(f, |acc, &x| acc.min(x)),
            }
        }
        ReduceOp::Max => {
            let mut it = xs.iter();
            match it.next() {
                None => T::from_f64(f64::NEG_INFINITY),
                Some(&f) => it.fold(f, |acc, &x| acc.max(x)),
            }
        }
    }
}

/// The inner product in `T`, the same order over the products.
pub fn dot_slice_t<T: Scalar>(a: &[T], b: &[T]) -> T {
    let mut acc = [T::zero(); 4];
    let ch = a.len() / 4;
    for c in 0..ch {
        for l in 0..4 {
            acc[l] = acc[l].add(a[4 * c + l].mul(b[4 * c + l]));
        }
    }
    let mut s = (acc[0].add(acc[1])).add(acc[2].add(acc[3]));
    for k in ch * 4..a.len() {
        s = s.add(a[k].mul(b[k]));
    }
    s
}

/// [`reduce_slice_t`] in `f64`.
pub fn reduce_slice(op: ReduceOp, xs: &[f64]) -> f64 {
    reduce_slice_t(op, xs)
}

/// [`dot_slice_t`] in `f64`: the two-lane vector twin, bit-identical.
pub fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    <f64 as Scalar>::dot_slice(a, b)
}

/// Canonical evaluation of a [`BinOp`], the one reference for every backend.
pub fn binary_f64(op: BinOp, x: f64, y: f64) -> f64 {
    match op {
        BinOp::Powf => libm::pow(x, y),
        BinOp::Mod => libm::fmod(x, y),
        BinOp::Atan2 => libm::atan2(x, y),
        BinOp::Hypot => libm::hypot(x, y),
    }
}

/// Digamma `psi(x)`: recurrence into the asymptotic zone, then the series.
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + libm::log(x)
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)))
}

/// Trigamma `psi'(x)`: recurrence into the asymptotic zone, then the series.
pub fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result
        + inv
        + 0.5 * inv2
        + inv * inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 / 30.0)))
}

/// Counter-based uniform noise in `[0, 1)` from the bits of `key`
/// (splitmix64 finalizer; the top 53 bits become the mantissa).
pub fn rand_uniform(key: f64) -> f64 {
    let mut z = key.to_bits().wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Argument above which `exp` is linearly extrapolated (a "limited exponential",
/// SPICE `limexp`). `exp` overflows f64 near 709; compact models routinely drive
/// the argument far past that on intermediate Newton iterates (out-of-range
/// internal-node guesses), producing `inf`/`NaN` that poison the whole solve.
/// Linearizing above this threshold keeps the value and its (constant) derivative
/// finite with ample headroom against overflow, while leaving every physical
/// evaluation -- exp arguments are O(10) at any real operating point -- bit-for-bit
/// unchanged. 80 is the classic SPICE `limexp` knee. See [`unary_f64`].
pub const EXP_LIMIT: f64 = 80.0;

/// Argument at or below which `ln` is clamped (so `ln(x<=0)` returns a finite,
/// modest `ln(LN_FLOOR)` instead of `-inf`/`NaN` on an out-of-range Newton
/// iterate). A *clamp*, not a linear extrapolation: a steep extrapolation slope
/// (`1/LN_FLOOR`) would blow intermediate values up to `~1e30` and wreck
/// conditioning. In-range arguments are untouched. See [`unary_f64`].
pub const LN_FLOOR: f64 = 1e-30;

/// Evaluate a unary op on a real argument. Single source of truth shared by the
/// arena evaluator ([`crate::eval`](mod@crate::eval)) and the compiled tape ([`crate::tape`]).
///
/// `exp`/`ln`/`sqrt` are domain-guarded (limited exponential, log floor, and
/// `sqrt` of a negative argument clamped to 0): on an out-of-range intermediate
/// Newton iterate these return a finite, smoothly-extrapolated value instead of
/// `inf`/`NaN`, so a single out-of-range internal node cannot poison the residual
/// and Jacobian. Every in-range argument evaluates identically to the bare op.
/// The derivative rules in [`crate::autodiff`] mirror these guards exactly.
pub fn unary_f64(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Exp => {
            if x > EXP_LIMIT {
                EXP_LIMIT.exp() * (1.0 + (x - EXP_LIMIT))
            } else {
                x.exp()
            }
        }
        UnaryOp::Ln => {
            if x > LN_FLOOR {
                x.ln()
            } else {
                LN_FLOOR.ln()
            }
        }
        UnaryOp::Sqrt => {
            if x > 0.0 {
                x.sqrt()
            } else {
                0.0
            }
        }
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Atan => x.atan(),
        UnaryOp::Floor => x.floor(),
        UnaryOp::Tan => libm::tan(x),
        UnaryOp::Log10 => libm::log10(x),
        UnaryOp::Log2 => libm::log2(x),
        UnaryOp::Log1p => libm::log1p(x),
        UnaryOp::Expm1 => libm::expm1(x),
        UnaryOp::Cbrt => libm::cbrt(x),
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                x
            }
        }
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => libm::round(x),
        UnaryOp::Trunc => x.trunc(),
        UnaryOp::Asin => libm::asin(x),
        UnaryOp::Acos => libm::acos(x),
        UnaryOp::Asinh => libm::asinh(x),
        UnaryOp::Acosh => libm::acosh(x),
        UnaryOp::Atanh => libm::atanh(x),
        UnaryOp::Erf => libm::erf(x),
        UnaryOp::Erfc => libm::erfc(x),
        UnaryOp::Lgamma => libm::lgamma(x),
        UnaryOp::Tgamma => libm::tgamma(x),
        UnaryOp::Digamma => digamma(x),
        UnaryOp::Trigamma => trigamma(x),
        UnaryOp::RandUniform => rand_uniform(x),
    }
}

/// What a unary op *is*, as data rather than code: how it prints, what it is
/// called in generated C, and whether it is differentiable everywhere it is
/// defined.
///
/// The semantics live in [`unary_f64`] (and in [`crate::Scalar`] for the
/// other execution types), because a guard like the `exp` cap is code. What
/// is data lives here, so a new op is one row plus its semantics and its
/// lowerings, instead of an edit in the printer, the C backend, the JIT's
/// code mapping and the generator.
///
/// The aggregate ops (`Reduce`, `Dot`) are deliberately not in this table:
/// they carry an operand list rather than a fixed arity, they are how arrays
/// and matrices lower natively into the graph, and every backend emits them
/// as a loop rather than a call.
/// Evaluate a [`CmpOp`] on two ordered arguments. Single source of truth shared
/// by the arena evaluator, the complex evaluator, the compiled tape (all `f64`)
/// and the constant-folding interner (exact `BigRational`), so a `Cmp` node's
/// `1.0`/`0.0` result agrees across every backend.
pub fn cmp_bool<T: PartialOrd>(op: CmpOp, x: T, y: T) -> bool {
    match op {
        CmpOp::Gt => x > y,
        CmpOp::Ge => x >= y,
        CmpOp::Lt => x < y,
        CmpOp::Le => x <= y,
        CmpOp::Eq => x == y,
        CmpOp::Ne => x != y,
    }
}

/// A dense matrix-vector product: `out[i] = dot(a[i*n..(i+1)*n], x)` for
/// `m` rows, each row the same fold as [`dot_slice_t`], so a fused product
/// is bit-identical to its rows as separate dots.
pub fn gemv_t<T: Scalar>(a: &[T], x: &[T], m: usize, n: usize, out: &mut [T]) {
    // Four rows at a time, each with its own four accumulators: sixteen
    // independent chains for the core, and every row's fold is exactly
    // `dot_slice_t`'s.
    let ch = n / 4;
    let mut i = 0;
    while i + 4 <= m {
        let rows = [
            &a[i * n..(i + 1) * n],
            &a[(i + 1) * n..(i + 2) * n],
            &a[(i + 2) * n..(i + 3) * n],
            &a[(i + 3) * n..(i + 4) * n],
        ];
        let mut acc = [[T::zero(); 4]; 4];
        for c in 0..ch {
            for (r, row) in rows.iter().enumerate() {
                for l in 0..4 {
                    acc[r][l] = acc[r][l].add(row[4 * c + l].mul(x[4 * c + l]));
                }
            }
        }
        for (r, row) in rows.iter().enumerate() {
            let a4 = acc[r];
            let mut s = (a4[0].add(a4[1])).add(a4[2].add(a4[3]));
            for k in ch * 4..n {
                s = s.add(row[k].mul(x[k]));
            }
            out[i + r] = s;
        }
        i += 4;
    }
    for r in i..m {
        out[r] = dot_slice_t(&a[r * n..(r + 1) * n], x);
    }
}

/// [`gemv_t`] with its rows folded per `codes` against `c` (see
/// [`gemm_fold_t`]).
pub fn gemv_fold_t<T: Scalar>(
    a: &[T],
    x: &[T],
    m: usize,
    n: usize,
    c: Option<&[T]>,
    codes: &[u32],
    out: &mut [T],
) {
    gemv_t(a, x, m, n, out);
    fold_in_place(codes, c, out);
}

/// A dense matrix-matrix product against rows: `out[i*n + j] =
/// dot(a[i*k..], b[j*k..])` for `m` rows of `a` and `n` rows of `b`, each
/// of `k` (the right factor by columns, each contiguous). Every entry is
/// the fold of [`dot_slice_t`], so a fused product is bit-identical to
/// its entries as separate dots.
pub fn gemm_t<T: Scalar>(a: &[T], b: &[T], m: usize, k: usize, n: usize, out: &mut [T]) {
    gemm_with(a, b, m, k, n, |i, v| out[i] = v);
}

/// [`gemm_t`] with every entry stored through `st(index, value)`.
pub fn gemm_with<T: Scalar>(
    a: &[T],
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
    mut st: impl FnMut(usize, T),
) {
    // Four rows of `a` against two rows of `b` at a time: thirty-two
    // independent accumulators, six operand rows in the near cache.
    let ch = k / 4;
    let tail = |i: usize, j: usize, acc: [T; 4]| -> T {
        let mut s = (acc[0].add(acc[1])).add(acc[2].add(acc[3]));
        for l in ch * 4..k {
            s = s.add(a[i * k + l].mul(b[j * k + l]));
        }
        s
    };
    let mut i = 0;
    while i + 4 <= m {
        let mut j = 0;
        while j + 2 <= n {
            let mut acc = [[[T::zero(); 4]; 2]; 4];
            for c in 0..ch {
                for (r, ar) in acc.iter_mut().enumerate() {
                    let ra = &a[(i + r) * k + 4 * c..][..4];
                    for (q, aq) in ar.iter_mut().enumerate() {
                        let rb = &b[(j + q) * k + 4 * c..][..4];
                        for l in 0..4 {
                            aq[l] = aq[l].add(ra[l].mul(rb[l]));
                        }
                    }
                }
            }
            for (r, ar) in acc.iter().enumerate() {
                for (q, &aq) in ar.iter().enumerate() {
                    st((i + r) * n + j + q, tail(i + r, j + q, aq));
                }
            }
            j += 2;
        }
        for r in 0..4 {
            for jj in j..n {
                st(
                    (i + r) * n + jj,
                    dot_slice_t(&a[(i + r) * k..][..k], &b[jj * k..][..k]),
                );
            }
        }
        i += 4;
    }
    for r in i..m {
        for j in 0..n {
            st(r * n + j, dot_slice_t(&a[r * k..][..k], &b[j * k..][..k]));
        }
    }
}

/// [`gemm_t`] with its entries folded per `codes` (see
/// [`crate::tape::Fold`]) against the accumulator `c`: an entry folded
/// with an operand is folded as it is stored; a self fold (against another
/// entry's product) is folded after every product is stored, the products
/// it reads being those of entries that stay plain.
pub fn gemm_fold_t<T: Scalar>(
    a: &[T],
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
    c: Option<&[T]>,
    codes: &[u32],
    out: &mut [T],
) {
    if codes.iter().any(|&code| Fold(code).is_self()) {
        gemm_t(a, b, m, k, n, out);
        fold_in_place(codes, c, out);
    } else {
        gemm_with(a, b, m, k, n, |i, v| {
            out[i] = Fold(codes[i]).fold(c.map_or(v, |c| c[i]), v)
        });
    }
}

/// The self folds of `out` (products stored plain), in place.
pub fn fold_in_place<T: Scalar>(codes: &[u32], c: Option<&[T]>, out: &mut [T]) {
    for i in 0..out.len() {
        let f = Fold(codes[i]);
        if f.is_plain() {
            continue;
        }
        let acc = if f.is_self() {
            out[f.self_index()]
        } else {
            c.map_or(out[i], |c| c[i])
        };
        out[i] = f.fold(acc, out[i]);
    }
}

/// [`gemm_t`] in `f64`: the two-lane vector twin, bit-identical.
pub fn gemm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    <f64 as Scalar>::gemm(a, b, m, k, n, out)
}

/// [`gemm_fold_t`] in `f64`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fold(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    c: Option<&[f64]>,
    codes: &[u32],
    out: &mut [f64],
) {
    <f64 as Scalar>::gemm_fold(a, b, m, k, n, c, codes, out)
}

/// [`gemv_fold_t`] in `f64`.
pub fn gemv_fold(
    a: &[f64],
    x: &[f64],
    m: usize,
    n: usize,
    c: Option<&[f64]>,
    codes: &[u32],
    out: &mut [f64],
) {
    <f64 as Scalar>::gemv_fold(a, x, m, n, c, codes, out)
}

/// [`gemv_t`] in `f64`: the two-lane vector twin, bit-identical.
pub fn gemv(a: &[f64], x: &[f64], m: usize, n: usize, out: &mut [f64]) {
    <f64 as Scalar>::gemv(a, x, m, n, out)
}

/// The column-panel widths of the blocked elimination in [`solve_t`]: the
/// panel's own updates run one element at a time, the trailing update as
/// a product, and the balance moves with the size (measured: 16 wins
/// below about five hundred unknowns, 32 above). A width is a constant of
/// its instantiation so the panel loops unroll.
pub const LU_PANEL_SMALL: usize = 16;
/// See [`LU_PANEL_SMALL`].
pub const LU_PANEL_LARGE: usize = 32;
/// Systems of fewer unknowns than this use [`LU_PANEL_SMALL`].
pub const LU_PANEL_SWITCH: usize = 512;
/// Systems of at most this many unknowns are eliminated right-looking
/// without panels: each pivot updates the whole trailing matrix and the
/// right-hand sides row by row (one product and one difference per entry),
/// which at these sizes beats the panel machinery.
pub const LU_UNBLOCKED_MAX: usize = 64;

pub fn solve_t<T: Scalar>(a: &[T], b: &[T], n: usize, out: &mut [T]) {
    solve_many_t(a, b, n, 1, out)
}

/// [`solve_t`] for `k` right-hand sides at once: `b` holds `k` vectors of
/// `n` back to back, `out` receives the `k` solutions the same way. One
/// factorization serves every right-hand side; each solution is bit-identical
/// to its own [`solve_t`] (the pivots depend on `a` alone, and every
/// right-hand side column runs through the same updates in the same order).
pub fn solve_many_t<T: Scalar>(a: &[T], b: &[T], n: usize, k: usize, out: &mut [T]) {
    T::solve_many(a, b, n, k, out)
}

/// [`solve_many_t`] as the generic reference, for any scalar (the `f64`
/// twin in `simd` mirrors it step for step).
pub fn solve_many_generic<T: Scalar>(a: &[T], b: &[T], n: usize, k: usize, out: &mut [T]) {
    if n <= LU_UNBLOCKED_MAX {
        solve_unblocked(a, b, n, k, out)
    } else if n < LU_PANEL_SWITCH {
        solve_blocked::<T, LU_PANEL_SMALL>(a, b, n, k, out)
    } else {
        solve_blocked::<T, LU_PANEL_LARGE>(a, b, n, k, out)
    }
}

/// [`solve_many_t`] right-looking without panels (see [`LU_UNBLOCKED_MAX`]).
fn solve_unblocked<T: Scalar>(a: &[T], b: &[T], n: usize, k: usize, out: &mut [T]) {
    let w = n + k;
    let mut m: Vec<T> = Vec::with_capacity(n * w);
    for i in 0..n {
        m.extend_from_slice(&a[i * n..(i + 1) * n]);
        for c in 0..k {
            m.push(b[c * n + i]);
        }
    }
    for kk in 0..n {
        let mut p = kk;
        let mut best = m[kk * w + kk].magnitude();
        for i in kk + 1..n {
            let v = m[i * w + kk].magnitude();
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
        for i in kk + 1..n {
            let l = m[i * w + kk].div(piv);
            m[i * w + kk] = l;
            for j in kk + 1..w {
                let t = l.mul(m[kk * w + j]);
                m[i * w + j] = m[i * w + j].sub(t);
            }
        }
    }
    for c in 0..k {
        let x = &mut out[c * n..(c + 1) * n];
        for i in (0..n).rev() {
            let mut s = m[i * w + n + c];
            for j in i + 1..n {
                let t = m[i * w + j].mul(x[j]);
                s = s.sub(t);
            }
            x[i] = s.div(m[i * w + i]);
        }
    }
}

/// [`solve_many_t`] with column panels of `NB`.
fn solve_blocked<T: Scalar, const NB: usize>(a: &[T], b: &[T], n: usize, k: usize, out: &mut [T]) {
    let w = n + k;
    let mut m: Vec<T> = Vec::with_capacity(n * w);
    for i in 0..n {
        m.extend_from_slice(&a[i * n..(i + 1) * n]);
        for c in 0..k {
            m.push(b[c * n + i]);
        }
    }
    let mut k0 = 0;
    while k0 < n {
        let k1 = (k0 + NB).min(n);
        // The panel's columns, right-looking within the panel.
        for k in k0..k1 {
            let mut p = k;
            let mut best = m[k * w + k].magnitude();
            for i in k + 1..n {
                let v = m[i * w + k].magnitude();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if p != k {
                for j in 0..w {
                    m.swap(k * w + j, p * w + j);
                }
            }
            let piv = m[k * w + k];
            for i in k + 1..n {
                let l = m[i * w + k].div(piv);
                m[i * w + k] = l;
                for j in k + 1..k1 {
                    let t = l.mul(m[k * w + j]);
                    m[i * w + j] = m[i * w + j].sub(t);
                }
            }
        }
        // The panel's unit lower triangle applied to the columns right of
        // the panel, for the panel's own rows.
        for k in k0..k1 {
            for i in k + 1..k1 {
                let l = m[i * w + k];
                for j in k1..w {
                    let t = l.mul(m[k * w + j]);
                    m[i * w + j] = m[i * w + j].sub(t);
                }
            }
        }
        // The trailing block: each entry minus the dot of its row's
        // multipliers with its column's panel rows.
        if k1 < n {
            let nb = k1 - k0;
            let cols = w - k1;
            let mut ut: Vec<T> = vec![T::zero(); cols * nb];
            for (jj, j) in (k1..w).enumerate() {
                for (kk, k) in (k0..k1).enumerate() {
                    ut[jj * nb + kk] = m[k * w + j];
                }
            }
            let mut lrows: Vec<T> = vec![T::zero(); 4 * nb];
            let mut prod: Vec<T> = vec![T::zero(); 4 * cols];
            let mut i = k1;
            while i < n {
                let rows = (n - i).min(4);
                for r in 0..rows {
                    let at = (i + r) * w;
                    lrows[r * nb..(r + 1) * nb].copy_from_slice(&m[at + k0..at + k1]);
                }
                T::gemm(
                    &lrows[..rows * nb],
                    &ut,
                    rows,
                    nb,
                    cols,
                    &mut prod[..rows * cols],
                );
                for r in 0..rows {
                    let at = (i + r) * w + k1;
                    for jj in 0..cols {
                        m[at + jj] = m[at + jj].sub(prod[r * cols + jj]);
                    }
                }
                i += rows;
            }
        }
        k0 = k1;
    }
    for c in 0..k {
        let x = &mut out[c * n..(c + 1) * n];
        for i in (0..n).rev() {
            let mut s = m[i * w + n + c];
            for j in i + 1..n {
                let t = m[i * w + j].mul(x[j]);
                s = s.sub(t);
            }
            x[i] = s.div(m[i * w + i]);
        }
    }
}

/// [`solve_t`] in `f64`.
pub fn solve(a: &[f64], b: &[f64], n: usize, out: &mut [f64]) {
    solve_t(a, b, n, out)
}

/// [`solve_many_t`] in `f64`.
pub fn solve_many(a: &[f64], b: &[f64], n: usize, k: usize, out: &mut [f64]) {
    solve_many_t(a, b, n, k, out)
}
