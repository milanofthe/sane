//! Passivity assessment and enforcement for a fitted [`RationalModel`] admittance macromodel.
//!
//! A `p`-port admittance `Y(s)` is **passive** (positive-real) iff `Y` is analytic in the open
//! right half-plane (guaranteed: the fit is stable, poles in the LHP) and its Hermitian part
//! `Φ(ω) = ½(Y(jω) + Y(jω)^H) ⪰ 0` for all `ω`. For a reciprocal (symmetric `Y`) network this is
//! `Re{Y(jω)} ⪰ 0`. A non-passive macromodel can make a transient simulator blow up, so we both
//! **check** and (optionally) **enforce**.
//!
//! Check = two cross-checking mechanisms:
//! NOTE (√s models, rapidmom#43): the grid scan and bisection refine evaluate the FULL model —
//! including the optional √s skin-effect term — so violation detection covers it; only the
//! Hamiltonian crossing localisation below is blind to the (non-rational) √s part and may miss a
//! crossing it induces. The dense scan is the belt for exactly this reason.
//!
//! 1. **Hamiltonian imaginary-eigenvalue test** — build a real state-space realisation
//!    (complex-conjugate pole pairs → 2×2 real blocks), form the positive-real Hamiltonian matrix;
//!    its purely-imaginary eigenvalues `jω₀` localise the frequencies where an eigenvalue of `Φ`
//!    crosses zero (the violation-band boundaries).
//! 2. **Dense frequency-grid scan** — evaluate `λ_min(Φ(ω))` on a log grid over the band (and a
//!    margin beyond); the authoritative source for the passivity margin and the violation bands.
//!
//! Enforcement = Gustavsen FRP-style residue perturbation: at each violating (frequency,
//! eigenvector) a least-squares, minimum-norm perturbation of the (normalised) residues and the
//! constant term `D` pushes the offending eigenvalue up to a small positive margin; iterate,
//! re-scanning, until passive. Minimum-norm in normalised residue space keeps the fit quality:
//! we report the before/after margin AND fit error so the trade-off is explicit.

use std::f64::consts::TAU;

use faer::Mat;

use super::ratmodel::RationalModel;
use super::C;

/// A passivity assessment: the margin, where it occurs, and the violation bands.
#[derive(Clone, Debug)]
pub struct PassivityReport {
    /// `true` iff `λ_min(Φ(ω)) ≥ −tol` everywhere on the scan.
    pub passive: bool,
    /// The worst (smallest) eigenvalue of `Φ(ω)` over the scan [S]. `≥ 0` ⇒ passive.
    pub margin: f64,
    /// Frequency [Hz] at which `margin` occurs.
    pub worst_freq: f64,
    /// Violation bands `[(f_lo, f_hi)]` [Hz] where `λ_min(Φ) < 0`.
    pub bands: Vec<(f64, f64)>,
    /// Purely-imaginary Hamiltonian eigenfrequencies [Hz] (crossing-frequency localisation);
    /// empty if `D + Dᵀ` is singular (the scan is then the sole authority).
    pub crossings: Vec<f64>,
}

impl PassivityReport {
    /// A one-line human summary for the Verilog-A provenance header.
    pub fn summary(&self) -> String {
        if self.passive {
            format!(
                "passive (margin {:.3e} S @ {:.4} GHz)",
                self.margin,
                self.worst_freq / 1e9
            )
        } else {
            format!(
                "NON-PASSIVE (min eig {:.3e} S @ {:.4} GHz, {} band(s))",
                self.margin,
                self.worst_freq / 1e9,
                self.bands.len()
            )
        }
    }
}

/// Result of [`enforce`]: before/after margins and fit errors, and the effort spent.
#[derive(Clone, Debug)]
pub struct EnforceReport {
    pub before: PassivityReport,
    pub after: PassivityReport,
    pub fit_error_before: f64,
    pub fit_error_after: f64,
    pub iterations: usize,
    /// Whether the model is passive after enforcement.
    pub success: bool,
}

// ── small dense linear algebra on flat row-major Vec<f64> ─────────────────────

/// Invert a `p×p` row-major matrix by Gauss-Jordan with partial pivoting. `None` if singular.
fn inv(a: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; p * p];
    for i in 0..p {
        inv[i * p + i] = 1.0;
    }
    for col in 0..p {
        // pivot
        let mut piv = col;
        let mut best = m[col * p + col].abs();
        for r in col + 1..p {
            let v = m[r * p + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for k in 0..p {
                m.swap(col * p + k, piv * p + k);
                inv.swap(col * p + k, piv * p + k);
            }
        }
        let d = m[col * p + col];
        for k in 0..p {
            m[col * p + k] /= d;
            inv[col * p + k] /= d;
        }
        for r in 0..p {
            if r == col {
                continue;
            }
            let f = m[r * p + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..p {
                m[r * p + k] -= f * m[col * p + k];
                inv[r * p + k] -= f * inv[col * p + k];
            }
        }
    }
    Some(inv)
}

// ── Hermitian part and its spectrum ───────────────────────────────────────────

/// `Φ = ½(Y + Y^H)` reduced to its real symmetric part (exact for a reciprocal, symmetric `Y`;
/// the symmetric part of `Re{Y}` otherwise), as a `p×p` row-major matrix.
fn phi_matrix(y: &[C], p: usize) -> Vec<f64> {
    let mut phi = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            phi[i * p + j] = 0.5 * (y[i * p + j].re + y[j * p + i].re);
        }
    }
    phi
}

/// Smallest eigenvalue and its (unit) eigenvector of a `p×p` real symmetric matrix. Closed-form for
/// `p ≤ 2` (the common port counts, exact and fast); for `p > 2` the faer general eigensolver gives
/// the value and inverse iteration the vector.
fn sym_min_eig(phi: &[f64], p: usize) -> (f64, Vec<f64>) {
    if p == 1 {
        return (phi[0], vec![1.0]);
    }
    if p == 2 {
        let (a, b, d) = (phi[0], phi[1], phi[3]);
        let mid = 0.5 * (a + d);
        let r = (0.25 * (a - d) * (a - d) + b * b).sqrt();
        let lam = mid - r;
        // (Φ − λI) is rank-1; a null vector is [b, λ−a] (or an axis if b≈0).
        let v = if b.abs() > 1e-300 {
            [b, lam - a]
        } else if a <= d {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        };
        let n = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-300);
        return (lam, vec![v[0] / n, v[1] / n]);
    }
    // General p: value from faer, eigenvector by inverse iteration on the shifted matrix.
    let mat = Mat::<f64>::from_fn(p, p, |i, j| phi[i * p + j]);
    let lam = match mat.eigenvalues() {
        Ok(e) => e.iter().map(|c| c.re).fold(f64::INFINITY, f64::min),
        Err(_) => phi
            .iter()
            .step_by(p + 1)
            .cloned()
            .fold(f64::INFINITY, f64::min),
    };
    let spread = phi.iter().fold(0.0f64, |m, &v| m.max(v.abs())).max(1e-12);
    let mu = lam - 1e-4 * spread; // shift just below λ_min so (Φ − μI) is invertible
    let shifted: Vec<f64> = (0..p * p)
        .map(|k| phi[k] - if k / p == k % p { mu } else { 0.0 })
        .collect();
    let mut x = vec![1.0 / (p as f64).sqrt(); p];
    if let Some(inv_s) = inv(&shifted, p) {
        for _ in 0..8 {
            let mut nx = vec![0.0f64; p];
            for i in 0..p {
                nx[i] = (0..p).map(|j| inv_s[i * p + j] * x[j]).sum();
            }
            let nrm = nx.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
            x = nx.iter().map(|v| v / nrm).collect();
        }
    }
    (lam, x)
}

/// `(λ_min, eigenvector of λ_min)` of the Hermitian part of `Y` at one frequency.
fn min_eig(y: &[C], p: usize) -> (f64, Vec<f64>) {
    sym_min_eig(&phi_matrix(y, p), p)
}

/// Dense log-frequency grid over the model's **validity range**: the fitted band plus a modest
/// guard (a transient simulator can excite just outside the band), the near-DC limit, and a far
/// asymptotic point (`Y(∞) = D` must itself be positive-real). We deliberately do NOT chase
/// passivity far out of band — there the fit is pure extrapolation and forcing it corrupts the
/// in-band model.
fn scan_grid(model: &RationalModel, n: usize) -> Vec<f64> {
    let flo = model.fmin * 0.2;
    let fhi = model.fmax * 2.0;
    let mut g = Vec::with_capacity(2 * n + 2);
    g.push(model.fmax * 1e-6); // ω→0 proxy
    let (la, lb) = (flo.ln(), fhi.ln());
    for i in 0..n {
        g.push((la + (lb - la) * i as f64 / (n as f64 - 1.0)).exp());
    }
    // A dense LINEAR in-band grid too: log spacing under-samples the high end of the band where a
    // thin resonant violation notch can hide between log points (would make a "passive" claim
    // unreliable vs a linear-grid verifier).
    let nl = n.max(1);
    for i in 0..nl {
        g.push(model.fmin + (model.fmax - model.fmin) * i as f64 / (nl as f64 - 1.0).max(1.0));
    }
    g
}

/// Scan `λ_min(Φ(ω))` on the grid; return `(margin, worst_freq, bands)`.
fn scan(model: &RationalModel, grid: &[f64]) -> (f64, f64, Vec<(f64, f64)>) {
    let p = model.n_ports;
    let mut margin = f64::INFINITY;
    let mut worst_f = grid[0];
    let mut mins = Vec::with_capacity(grid.len());
    for &f in grid {
        let (lam, _) = min_eig(&model.eval(f), p);
        mins.push(lam);
        if lam < margin {
            margin = lam;
            worst_f = f;
        }
    }
    // Violation bands: contiguous runs of λ_min < 0 (boundaries bisection-refined).
    let tol = -1e-12 * model_scale(model);
    let mut bands = Vec::new();
    let mut i = 0;
    while i < grid.len() {
        if mins[i] < tol {
            let start = if i == 0 {
                grid[0]
            } else {
                refine(model, grid[i - 1], grid[i])
            };
            let mut j = i;
            while j + 1 < grid.len() && mins[j + 1] < tol {
                j += 1;
            }
            let end = if j + 1 < grid.len() {
                refine(model, grid[j], grid[j + 1])
            } else {
                grid[j]
            };
            bands.push((start, end));
            i = j + 1;
        } else {
            i += 1;
        }
    }
    (margin, worst_f, bands)
}

/// Bisect the sign change of `λ_min(Φ)` between `fa` (violating or not) and `fb` to locate a band
/// boundary frequency [Hz].
fn refine(model: &RationalModel, fa: f64, fb: f64) -> f64 {
    let p = model.n_ports;
    let (mut a, mut b) = (fa, fb);
    let sa = min_eig(&model.eval(a), p).0 < 0.0;
    for _ in 0..40 {
        let m = 0.5 * (a + b);
        let sm = min_eig(&model.eval(m), p).0 < 0.0;
        if sm == sa {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// A representative conductance scale: the max diagonal `Re{Y(jω)}` over the validity band (the
/// actual level `Re{Y}` sits at), for scale-relative tolerances and the enforcement target margin.
/// Falls back to `|D|` if the band evaluation is degenerate.
fn model_scale(model: &RationalModel) -> f64 {
    let p = model.n_ports;
    let mut s = 1e-12f64;
    for &f in &scan_grid(model, 48) {
        let y = model.eval(f);
        for i in 0..p {
            s = s.max(y[i * p + i].re.abs());
        }
    }
    for i in 0..p {
        s = s.max(model.d[i * p + i].abs());
    }
    s.max(1e-12)
}

// ── real state-space realisation + Hamiltonian crossings ──────────────────────

struct Realization {
    a: Vec<f64>,    // N×N
    b: Vec<f64>,    // N×p
    c: Vec<f64>,    // p×N
    dmat: Vec<f64>, // p×p
    n: usize,
    p: usize,
}

/// Build the **normalised** (`s_n = s/K`, `K = 2π·fmax`) real state-space realisation of `Y`
/// (ignoring the `sE` feedthrough, which contributes only to the imaginary part on `jω` and so
/// does not affect `Φ`). Real pole → `p` states; complex pair → `2p` real states (2×2 rotation
/// blocks). `C(s_nI − A)^{-1}B + D` reconstructs the (physical-equal) admittance.
fn realize(model: &RationalModel) -> Realization {
    let p = model.n_ports;
    let k = TAU * model.fmax;
    let nr = model.real_poles.len();
    let nc = model.cpx_poles.len();
    let n = nr * p + nc * 2 * p;
    let mut a = vec![0.0; n * n];
    let mut b = vec![0.0; n * p];
    let mut c = vec![0.0; p * n];
    let mut off = 0;
    for kk in 0..nr {
        let pr = model.real_poles[kk] / k;
        for comp in 0..p {
            let s = off + comp;
            a[s * n + s] = pr;
            b[s * p + comp] = 1.0;
            for i in 0..p {
                c[i * n + s] = model.res_real[i * p + comp][kk] / k;
            }
        }
        off += p;
    }
    for kk in 0..nc {
        let pc = model.cpx_poles[kk] / k;
        let (ar, br) = (pc.re, pc.im);
        for comp in 0..p {
            let x1 = off + comp;
            let x2 = off + p + comp;
            a[x1 * n + x1] = ar;
            a[x1 * n + x2] = br;
            a[x2 * n + x1] = -br;
            a[x2 * n + x2] = ar;
            b[x1 * p + comp] = 1.0;
            for i in 0..p {
                let rc = model.res_cpx[i * p + comp][kk] / k;
                c[i * n + x1] = 2.0 * rc.re;
                c[i * n + x2] = 2.0 * rc.im;
            }
        }
        off += 2 * p;
    }
    let dmat: Vec<f64> = (0..p * p).map(|e| model.d[e]).collect();
    Realization {
        a,
        b,
        c,
        dmat,
        n,
        p,
    }
}

/// Purely-imaginary eigenfrequencies [Hz] of the positive-real Hamiltonian matrix
/// `M = [[A − B R⁻¹C, −B R⁻¹Bᵀ],[Cᵀ R⁻¹C, −Aᵀ + Cᵀ R⁻¹Bᵀ]]`, `R = D + Dᵀ` — the frequencies
/// where an eigenvalue of `Φ` crosses zero. Empty if `R` is singular.
fn hamiltonian_crossings(model: &RationalModel) -> Vec<f64> {
    let r = realize(model);
    let (n, p) = (r.n, r.p);
    if n == 0 {
        return Vec::new();
    }
    // R = D + Dᵀ and its inverse.
    let mut rmat = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            rmat[i * p + j] = r.dmat[i * p + j] + r.dmat[j * p + i];
        }
    }
    let Some(rinv) = inv(&rmat, p) else {
        return Vec::new();
    };
    // helpers on r's flat blocks
    let bt = |i: usize, j: usize| r.b[j * p + i]; // Bᵀ[i,j] = B[j,i]
    let ct = |i: usize, j: usize| r.c[j * n + i]; // Cᵀ[i,j] = C[j,i]
                                                  // Rinv·C  (p×n), Rinv·Bᵀ (p×n)
    let mut ric = vec![0.0; p * n];
    let mut rib = vec![0.0; p * n];
    for i in 0..p {
        for j in 0..n {
            let mut sc = 0.0;
            let mut sb = 0.0;
            for l in 0..p {
                sc += rinv[i * p + l] * r.c[l * n + j];
                sb += rinv[i * p + l] * bt(l, j);
            }
            ric[i * n + j] = sc;
            rib[i * n + j] = sb;
        }
    }
    let dim = 2 * n;
    let mut m = vec![0.0f64; dim * dim];
    let set = |m: &mut Vec<f64>, i: usize, j: usize, v: f64| m[i * dim + j] = v;
    for i in 0..n {
        for j in 0..n {
            // M11 = A − B·(Rinv·C)
            let mut m11 = r.a[i * n + j];
            let mut m12 = 0.0; // -B·(Rinv·Bᵀ)
            let mut m21 = 0.0; // Cᵀ·(Rinv·C)
            let mut m22 = -r.a[j * n + i]; // −Aᵀ
            for l in 0..p {
                m11 -= r.b[i * p + l] * ric[l * n + j];
                m12 -= r.b[i * p + l] * rib[l * n + j];
                m21 += ct(i, l) * ric[l * n + j];
                m22 += ct(i, l) * rib[l * n + j];
            }
            set(&mut m, i, j, m11);
            set(&mut m, i, n + j, m12);
            set(&mut m, n + i, j, m21);
            set(&mut m, n + i, n + j, m22);
        }
    }
    let mat = Mat::<f64>::from_fn(dim, dim, |i, j| m[i * dim + j]);
    let Ok(eigs) = mat.eigenvalues() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in eigs {
        // purely imaginary: |Re| ≪ |Im|
        if e.im.abs() > 1e-9 && e.re.abs() <= 1e-6 * e.im.abs() {
            let wn = e.im.abs();
            out.push(wn * model.fmax); // s_n = j·wn, physical f = wn·fmax
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out.dedup_by(|a, b| (*a - *b).abs() <= 1e-3 * a.abs());
    out
}

// ── public check ──────────────────────────────────────────────────────────────

/// Assess passivity of `model`: dense-grid `λ_min(Φ)` scan (authoritative margin + bands) with the
/// Hamiltonian crossing frequencies as a localisation cross-check.
pub fn check(model: &RationalModel) -> PassivityReport {
    let crossings = hamiltonian_crossings(model);
    // Grid: dense log scan + a cluster of points around each Hamiltonian crossing (so a thin
    // violation notch between grid points is not missed).
    let mut grid = scan_grid(model, 600);
    for &fc in &crossings {
        for d in [-2e-2, -5e-3, 5e-3, 2e-2] {
            grid.push(fc * (1.0 + d));
        }
    }
    // Clamp to the validity range (≤ 2·fmax): a crossing/notch beyond it is pure extrapolation.
    let fhi = model.fmax * 2.0;
    grid.retain(|f| f.is_finite() && *f > 0.0 && *f <= fhi);
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (margin, worst_freq, bands) = scan(model, &grid);
    PassivityReport {
        passive: margin >= -1e-12 * model_scale(model),
        margin,
        worst_freq,
        bands,
        crossings,
    }
}

// ── enforcement (Gustavsen FRP-style residue perturbation) ───────────────────

/// The (normalised) residue/`D` degrees of freedom, laid out per symmetric matrix entry
/// `σ = (a,b)` with `a ≤ b`, then per pole. Perturbing a symmetric entry mirrors to `(b,a)`, so
/// `Y` stays reciprocal (and `Φ` stays real symmetric).
struct DofLayout {
    /// symmetric entries (a,b), a ≤ b
    sym: Vec<(usize, usize)>,
    nr: usize,
    nc: usize,
}
impl DofLayout {
    fn new(model: &RationalModel) -> Self {
        let p = model.n_ports;
        let mut sym = Vec::new();
        for a in 0..p {
            for b in a..p {
                sym.push((a, b));
            }
        }
        DofLayout {
            sym,
            nr: model.real_poles.len(),
            nc: model.cpx_poles.len(),
        }
    }
    /// DOFs per symmetric entry: 1 (D) + nr (real residues) + 2·nc (complex residues).
    fn per_entry(&self) -> usize {
        1 + self.nr + 2 * self.nc
    }
    fn n_dof(&self) -> usize {
        self.sym.len() * self.per_entry()
    }
}

/// Add `alpha·x` (normalised residue/`D` perturbation) into the physical model, mirroring each
/// symmetric entry so `Y` stays reciprocal (residues scale by `K` back to physical).
fn apply_perturbation(
    model: &mut RationalModel,
    layout: &DofLayout,
    x: &[f64],
    alpha: f64,
    k: f64,
) {
    let p = model.n_ports;
    for (si, &(a, b)) in layout.sym.iter().enumerate() {
        let base = si * layout.per_entry();
        let (eab, eba) = (a * p + b, b * p + a);
        let dd = alpha * x[base];
        model.d[eab] += dd;
        if a != b {
            model.d[eba] += dd;
        }
        for kk in 0..layout.nr {
            let dr = alpha * x[base + 1 + kk] * k;
            model.res_real[eab][kk] += dr;
            if a != b {
                model.res_real[eba][kk] += dr;
            }
        }
        for kk in 0..layout.nc {
            let dc = C::new(
                alpha * x[base + 1 + layout.nr + 2 * kk] * k,
                alpha * x[base + 1 + layout.nr + 2 * kk + 1] * k,
            );
            model.res_cpx[eab][kk] += dc;
            if a != b {
                model.res_cpx[eba][kk] += dc;
            }
        }
    }
}

/// Enforce passivity by iterated minimum-norm residue perturbation (Gustavsen FRP), with a
/// trust-region step so a fix never corrupts the in-band fit. Mutates `model` in place; returns the
/// before/after report. `max_iter` caps the outer iterations. Passivity is enforced over the
/// model's *validity range* only (see [`scan_grid`]) — chasing out-of-band extrapolation would
/// wreck the fit for no physical gain.
pub fn enforce(model: &mut RationalModel, max_iter: usize) -> EnforceReport {
    let before = check(model);
    let fit_before = model.fit_error;
    let p = model.n_ports;
    let k = TAU * model.fmax;
    let scale = model_scale(model);
    let target = 1e-4 * scale; // small positive margin (headroom for transient sim / thin notches)
                               // Fit-error ceiling for the trust region: passivity (a hard requirement for transient
                               // stability) is worth a bounded fit-error growth, but never a blow-up. The achieved error is
                               // reported so the caller sees the trade-off.
    let fit_cap = (fit_before * 20.0).max(fit_before + 3e-3);
    let layout = DofLayout::new(model);
    let ndof = layout.n_dof();

    let mut iters = 0;
    for _ in 0..max_iter {
        let rep = check(model);
        if rep.passive {
            break;
        }
        iters += 1;
        // Build well-separated control points where λ_min < target (dedup collinear clusters:
        // one point per ~3 % frequency step keeps the least-squares well-conditioned). The
        // Hamiltonian crossing frequencies are added densely so thin resonant violation notches
        // between the log-grid points are caught.
        let mut grid = scan_grid(model, 500);
        for &fc in &rep.crossings {
            for d in [-1.5e-2, -6e-3, -2e-3, 0.0, 2e-3, 6e-3, 1.5e-2] {
                grid.push(fc * (1.0 + d));
            }
        }
        let fhi = model.fmax * 2.0;
        grid.retain(|f| f.is_finite() && *f > 0.0 && *f <= fhi);
        grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut rhs: Vec<f64> = Vec::new();
        for &f in &grid {
            let (lam, u) = min_eig(&model.eval(f), p);
            if lam >= target {
                continue;
            }
            let sn = C::new(0.0, f / model.fmax);
            let mut row = vec![0.0f64; ndof];
            for (si, &(a, b)) in layout.sym.iter().enumerate() {
                let q = if a == b {
                    u[a] * u[a]
                } else {
                    2.0 * u[a] * u[b]
                };
                if q == 0.0 {
                    continue;
                }
                let base = si * layout.per_entry();
                row[base] += q; // D dof: φ = 1
                for kk in 0..layout.nr {
                    let prn = model.real_poles[kk] / k;
                    row[base + 1 + kk] += q * (C::new(1.0, 0.0) / (sn - C::new(prn, 0.0))).re;
                }
                for kk in 0..layout.nc {
                    let pcn = model.cpx_poles[kk] / k;
                    let t1 = C::new(1.0, 0.0) / (sn - pcn);
                    let t2 = C::new(1.0, 0.0) / (sn - pcn.conj());
                    row[base + 1 + layout.nr + 2 * kk] += q * (t1 + t2).re;
                    row[base + 1 + layout.nr + 2 * kk + 1] += q * (C::new(0.0, 1.0) * (t1 - t2)).re;
                }
            }
            rows.push(row);
            rhs.push(target - lam);
        }
        if rows.is_empty() {
            break;
        }
        // Regularised least-squares in DOF space (there are more violation constraints than DOFs):
        // minimise ‖A x − c‖² + λ‖x‖² ⇒ (AᵀA + λI) x = Aᵀc. The ridge both conditions the normal
        // matrix and keeps the perturbation ‖x‖ small (fit preservation).
        let nviol = rows.len();
        let mut ata = vec![0.0f64; ndof * ndof];
        let mut atc = vec![0.0f64; ndof];
        for r in 0..nviol {
            let row = &rows[r];
            let cr = rhs[r];
            for i in 0..ndof {
                atc[i] += row[i] * cr;
                for j in 0..ndof {
                    ata[i * ndof + j] += row[i] * row[j];
                }
            }
        }
        let ridge = 1e-6
            * (0..ndof)
                .map(|i| ata[i * ndof + i])
                .fold(0.0f64, f64::max)
                .max(1e-300);
        for i in 0..ndof {
            ata[i * ndof + i] += ridge;
        }
        let Some(ata_inv) = inv(&ata, ndof) else {
            break;
        };
        let mut x = vec![0.0f64; ndof];
        for i in 0..ndof {
            x[i] = (0..ndof).map(|j| ata_inv[i * ndof + j] * atc[j]).sum();
        }
        // Trust region: take the largest damped step that improves the margin without pushing the
        // fit error past the ceiling. Backtrack from a full step; break if none qualifies.
        // Take the largest damped step that strictly improves the global margin without pushing the
        // fit error past the ceiling. Strict (not target-sized) improvement lets small steps
        // accumulate over iterations toward a passive model.
        let base_margin = rep.margin;
        let snapshot = model.clone();
        let mut accepted = false;
        for &alpha in &[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03, 0.015, 0.007, 0.003] {
            *model = snapshot.clone();
            apply_perturbation(model, &layout, &x, alpha, k);
            let m2 = check(model).margin;
            let e2 = model.sample_error();
            if m2 > base_margin + 1e-9 * scale && e2 <= fit_cap {
                accepted = true;
                break;
            }
        }
        if !accepted {
            *model = snapshot;
            break;
        }
    }
    let after = check(model);
    model.fit_error = model.sample_error();
    EnforceReport {
        success: after.passive,
        before,
        after,
        fit_error_before: fit_before,
        fit_error_after: model.fit_error,
        iterations: iters,
    }
}

#[cfg(test)]
mod eigtests {
    use super::sym_min_eig;
    /// `Φ x = λ_min x` and `λ_min` correct for 2×2 and general symmetric matrices.
    fn check_pair(a: &[f64], p: usize, want_lam: f64) {
        let (lam, x) = sym_min_eig(a, p);
        assert!((lam - want_lam).abs() < 1e-9, "λ_min {lam} != {want_lam}");
        let nrm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((nrm - 1.0).abs() < 1e-9, "eigenvector not unit: {nrm}");
        for i in 0..p {
            let ax: f64 = (0..p).map(|k| a[i * p + k] * x[k]).sum();
            assert!((ax - lam * x[i]).abs() < 1e-7, "residual comp {i}");
        }
    }
    #[test]
    fn min_eig_2x2() {
        check_pair(&[2.0, 1.0, 1.0, 2.0], 2, 1.0); // eig 1, 3
        check_pair(&[3.0, 0.0, 0.0, 5.0], 2, 3.0); // diagonal
    }
    #[test]
    fn min_eig_3x3() {
        check_pair(&[2.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 1.0, 3.0], 3, 2.0); // eig 2,2,4
        check_pair(
            &[4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0],
            3,
            2.194397167422409,
        ); // numeric λ_min
    }
}
