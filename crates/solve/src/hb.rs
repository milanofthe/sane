//! Harmonic balance: periodic steady state solved in the frequency domain.
//!
//! Each unknown is a truncated Fourier series `x_i(t) = Σ_{k=0..K} X_{i,k}
//! e^{jkw0 t}` (with `X_{i,-k} = conj(X_{i,k})`, so the time signal is real).
//! The unknowns are the complex coefficients `X`. We require the residual to
//! vanish at every harmonic, evaluated by the **alternating frequency-time**
//! (AFT) scheme:
//!
//! 1. synthesise the time samples `x(t_m)` and `x'(t_m)` from `X` (an inverse
//!    real FFT per unknown; `x'` has coefficients `jkw0 X_k`),
//! 2. evaluate the *time-domain* residual [`CompiledDc::residual`] at each
//!    sample (the same compiled tape transient uses -- a `SIN` source enters
//!    through the time argument `t_m`, so the drive is free),
//! 3. forward real FFT the per-unknown sample sequence back to the harmonic
//!    residuals `R_k`.
//!
//! The Jacobian `dR/dX` reuses the device Jacobian tapes the same way. Its
//! block structure is the engine's strength: a *constant* (LTI) entry of
//! `dF/dx` is frequency-diagonal `(G + jkw0 C)` (no harmonic coupling), and
//! only the *variable* (device) entries -- flagged by [`CompiledDc`]'s
//! `jx_var` classification -- generate the dense Toeplitz blocks that couple
//! harmonics. The sparsity pattern of that block matrix is fixed (the DC
//! pattern promoted to harmonic blocks), so the symbolic factorisation is
//! built once and every Newton step only refills the value buffer.
//!
//! This is a single-tone, monolithic, direct-solve implementation. Matrix-free
//! GMRES with a per-harmonic block preconditioner, exact spectral convolution
//! through polynomial subgraphs, and multitone come later.

use crate::CompiledDc;
use num_complex::Complex64;
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rslab::{GeneralCsc, KluSettings, KluSolver, KluSymbolic};
use sane_core::constants::{
    GMIN_DC, HB_BACKTRACK_GROWTH, HB_CONT_DLAM0, HB_CONT_DLAM_MAX, HB_CONT_DLAM_MIN,
    HB_CONT_EASY_ITERS, HB_CONT_GROW, HB_CONT_LAMBDA_EPS, HB_CONT_SHRINK, LINE_SEARCH_TRIES,
};
use std::sync::Arc;

/// Result of a harmonic-balance solve.
pub struct HbResult {
    /// `spectra[i][k]` is the complex Fourier coefficient of harmonic `k`
    /// (`k = 0..=harmonics`, `k = 0` is DC) at unknown `i`.
    pub spectra: Vec<Vec<Complex64>>,
    pub converged: bool,
    pub iters: usize,
    /// Fundamental angular frequency used (rad/s).
    pub w0: f64,
    /// Harmonics solved for (so `harmonics + 1` coefficients including DC).
    pub harmonics: usize,
    /// Time samples per period on the AFT grid.
    pub samples: usize,
    /// Infinity norm of the final harmonic residual.
    pub residual_norm: f64,
}

/// Choose the AFT time-sample count for `harmonics` solved harmonics, given the
/// system's nonlinearity. For a polynomial residual of degree `d` the alias-free
/// count `2*d*K + 1` is exact; for an unbounded (transcendental / piecewise /
/// opaque) residual no exact bound exists, so we oversample `oversample`-fold.
/// Rounded up to a power of two for the FFT, and at least `2K+1` (the Jacobian
/// blocks need spectra out to index `K`).
pub fn hb_samples(nl: &rsdag::Nonlinearity, harmonics: usize, oversample: usize) -> usize {
    let base = 2 * harmonics + 1;
    let need = nl
        .alias_free_samples(harmonics as u32)
        .unwrap_or_else(|| oversample.max(1) * base);
    need.max(base).next_power_of_two()
}

/// How a single value-buffer entry of the harmonic-block Jacobian is computed.
enum Slot {
    /// `dR_k/dX_l` contribution from jx nonzero `nz`: harmonic `k-l` of the
    /// `dF/dx` entry's waveform.
    Jx { nz: usize, dk: i64 },
    /// `dR_k/dX_l` contribution from jxd nonzero `nz`, scaled by `j*l*w0`:
    /// harmonic `k-l` of the `dF/dx'` entry's waveform.
    Jxd { nz: usize, dk: i64, l: i64 },
    /// A `GMIN_DC` regularisation entry on a node-harmonic diagonal.
    Gmin,
}

/// A harmonic-balance problem compiled around a [`CompiledDc`]: holds the AFT
/// grid, the real-FFT plans, and the fixed harmonic-block Jacobian pattern with
/// its symbolic factorisation.
pub struct CompiledHb<'a> {
    cdc: &'a CompiledDc,
    /// Number of unknowns.
    n: usize,
    /// Number of harmonics `K` (coefficients `0..=K`).
    k: usize,
    /// Coefficients per unknown (`K + 1`).
    h: usize,
    /// Time samples per period (the AFT grid).
    m: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    /// Fixed Jacobian pattern (harmonic blocks over the DC sparsity) as a CSC
    /// skeleton plus KLU's symbolic analysis (BTF + per-block AMD); only the
    /// values change between Newton steps.
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    sym: KluSymbolic,
    /// One per pattern *entry*, in value-buffer order: how to fill it each
    /// step; `slot_of[e]` is the CSC value slot entry `e` sums into.
    slots: Vec<Slot>,
    slot_of: Vec<usize>,
}

/// The factorized two-sided Toeplitz harmonic-balance Jacobian plus the
/// parameter-Jacobian pattern and spectra. Built once by an AFT pass and shared
/// by the coefficient gradient and Hessian (both adjoint-solve against it).
struct ToeplitzJacobian {
    lu: KluSolver<Complex64>,
    pj_rows: Vec<usize>,
    pj_cols: Vec<usize>,
    pjs: Vec<Vec<Complex64>>,
    /// The synthesized state / state-derivative waveforms on the AFT grid, reused
    /// by the Hessian's second-order sampling pass.
    x_time: Vec<Vec<f64>>,
    xd_time: Vec<Vec<f64>>,
}

impl<'a> CompiledHb<'a> {
    /// One AFT pass producing the device Jacobian spectra (`dF/dx`, `dF/dx'`,
    /// `dF/dp`), then assembly and LU-factorization of the exact two-sided
    /// Toeplitz HB Jacobian. `None` if the AFT grid is too coarse (`m/2 < 2K`) or
    /// the system is singular. Shared by `coeff_gradient` and `coeff_hessian`.
    fn build_toeplitz_jacobian(
        &self,
        spectra: &[Vec<Complex64>],
        p: &[f64],
        w0: f64,
    ) -> Option<ToeplitzJacobian> {
        let n = self.n;
        let k = self.k as i64;
        let h2 = 2 * self.k + 1;
        let kmax = 2 * self.k;
        if self.m / 2 < kmax {
            return None;
        }
        let flat = |i: usize, kk: i64| -> usize { i * h2 + (kk + k) as usize };

        // dF/dx, dF/dx', dF/dp waveforms on the AFT grid (one sample loop).
        let (x_time, xd_time) = self.synth(spectra, w0);
        let period = 2.0 * std::f64::consts::PI / w0;
        let nzx = self.cdc.jx_rows.len();
        let nzd = self.cdc.jxd_rows.len();
        let mut jx_time = vec![vec![0.0; self.m]; nzx];
        let mut jxd_time = vec![vec![0.0; self.m]; nzd];
        let mut pj_rows: Vec<usize> = Vec::new();
        let mut pj_cols: Vec<usize> = Vec::new();
        let mut pj_time: Vec<Vec<f64>> = Vec::new();
        let (mut xv, mut xdv) = (vec![0.0; n], vec![0.0; n]);
        let (mut inb, mut wb, mut ob) = (Vec::new(), Vec::new(), Vec::new());
        let cn = self.cdc.n;
        for mm in 0..self.m {
            for i in 0..n {
                xv[i] = x_time[i][mm];
                xdv[i] = xd_time[i][mm];
            }
            let t = mm as f64 / self.m as f64 * period;
            self.cdc.fill_inputs(&xv, &xdv, p, t, &mut inb);
            self.cdc.tape_step.eval(&inb, &mut wb, &mut ob);
            for (nz, s) in jx_time.iter_mut().enumerate() {
                s[mm] = ob[cn + nz];
            }
            self.cdc.tape_jxd.eval(&inb, &mut wb, &mut ob);
            for (nz, s) in jxd_time.iter_mut().enumerate() {
                s[mm] = ob[nz];
            }
            let (pr, pc, pv) = self.cdc.jacobian_p_sparse(&xv, &xdv, p, t);
            if mm == 0 {
                pj_rows = pr;
                pj_cols = pc;
                pj_time = vec![vec![0.0; self.m]; pv.len()];
            }
            for (nz, &val) in pv.iter().enumerate() {
                pj_time[nz][mm] = val;
            }
        }
        let gx: Vec<Vec<Complex64>> = (0..nzx)
            .map(|nz| {
                if self.cdc.jx_var[nz] {
                    self.analyse_k(&jx_time[nz], kmax)
                } else {
                    let mut v = vec![Complex64::new(0.0, 0.0); kmax + 1];
                    v[0] = Complex64::new(jx_time[nz][0], 0.0);
                    v
                }
            })
            .collect();
        let gxd: Vec<Vec<Complex64>> = (0..nzd)
            .map(|nz| {
                if self.cdc.jxd_var[nz] {
                    self.analyse_k(&jxd_time[nz], kmax)
                } else {
                    let mut v = vec![Complex64::new(0.0, 0.0); kmax + 1];
                    v[0] = Complex64::new(jxd_time[nz][0], 0.0);
                    v
                }
            })
            .collect();
        let pjs: Vec<Vec<Complex64>> = (0..pj_time.len())
            .map(|nz| self.analyse_k(&pj_time[nz], self.k))
            .collect();

        // Assemble the exact two-sided Toeplitz Jacobian (size n*(2K+1)).
        let dim = n * h2;
        let (mut tr, mut tc, mut tv) = (Vec::new(), Vec::new(), Vec::<Complex64>::new());
        for nz in 0..nzx {
            let (i, j) = (self.cdc.jx_rows[nz], self.cdc.jx_cols[nz]);
            if self.cdc.jx_var[nz] {
                for kk in -k..=k {
                    for l in -k..=k {
                        tr.push(flat(i, kk));
                        tc.push(flat(j, l));
                        tv.push(spec_at(&gx[nz], kk - l));
                    }
                }
            } else {
                let g0 = gx[nz][0];
                for kk in -k..=k {
                    tr.push(flat(i, kk));
                    tc.push(flat(j, kk));
                    tv.push(g0);
                }
            }
        }
        for nz in 0..nzd {
            let (i, j) = (self.cdc.jxd_rows[nz], self.cdc.jxd_cols[nz]);
            if self.cdc.jxd_var[nz] {
                for kk in -k..=k {
                    for l in -k..=k {
                        tr.push(flat(i, kk));
                        tc.push(flat(j, l));
                        tv.push(Complex64::new(0.0, l as f64 * w0) * spec_at(&gxd[nz], kk - l));
                    }
                }
            } else {
                let c0 = gxd[nz][0];
                for kk in -k..=k {
                    tr.push(flat(i, kk));
                    tc.push(flat(j, kk));
                    tv.push(Complex64::new(0.0, kk as f64 * w0) * c0);
                }
            }
        }
        let nodes =
            (0..self.cdc.n).filter(|&i| self.cdc.kinds[i] == sane_dae::UnknownKind::NodeVoltage);
        for i in nodes {
            for kk in -k..=k {
                tr.push(flat(i, kk));
                tc.push(flat(i, kk));
                tv.push(Complex64::new(GMIN_DC, 0.0));
            }
        }
        let a = GeneralCsc::from_triplets(dim, &tr, &tc, &tv).ok()?;
        let lu = KluSolver::factor(&a, &KluSettings::default()).ok()?;
        Some(ToeplitzJacobian {
            lu,
            pj_rows,
            pj_cols,
            pjs,
            x_time,
            xd_time,
        })
    }

    /// Global index of harmonic `k` of unknown `i` in the flattened system.
    #[inline]
    fn idx(&self, i: usize, k: usize) -> usize {
        i * self.h + k
    }

    /// Build the harmonic-balance problem. `harmonics` is `K`; `samples` is the
    /// AFT grid size (use [`hb_samples`]). The Jacobian pattern is assembled once
    /// here: LTI jx entries contribute only block diagonals, device entries the
    /// full `(K+1)x(K+1)` Toeplitz block, jxd entries likewise (charge storage
    /// can be nonlinear), plus a gmin diagonal on every node harmonic.
    pub fn new(cdc: &'a CompiledDc, harmonics: usize, samples: usize) -> Option<Self> {
        let n = cdc.n;
        let k = harmonics;
        let h = k + 1;
        let m = samples;
        if m < 2 * k {
            return None; // need spectra out to index K (m/2 >= K)
        }
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(m);
        let c2r = planner.plan_fft_inverse(m);

        // Assemble the fixed pattern and the per-entry fill recipe together so
        // they stay aligned: (rows, cols)[e] <-> slots[e] <-> valbuf[e].
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut slots: Vec<Slot> = Vec::new();
        let idx = |i: usize, kk: usize| i * h + kk;

        // EXPERIMENT (`SANE_HB_BAND=B`): truncate the variable Toeplitz blocks
        // to bandwidth `|k-l| <= B` (quasi-Newton -- the residual stays exact,
        // so a converged solution satisfies the same tolerance; only the Newton
        // path and iteration count change). Off (full blocks) unless set.
        let band: i64 = sane_core::config()
            .hb_band
            .map(|b| b as i64)
            .unwrap_or(i64::MAX);

        // dF/dx blocks.
        for nz in 0..cdc.jx_rows.len() {
            let (i, j) = (cdc.jx_rows[nz], cdc.jx_cols[nz]);
            if cdc.jx_var[nz] {
                // Variable (device) entry: full Toeplitz block, couples harmonics.
                for kk in 0..h {
                    for l in 0..h {
                        let dk = kk as i64 - l as i64;
                        if dk.abs() > band {
                            continue;
                        }
                        rows.push(idx(i, kk));
                        cols.push(idx(j, l));
                        slots.push(Slot::Jx { nz, dk });
                    }
                }
            } else {
                // Constant (LTI) entry: frequency-diagonal, no coupling.
                for kk in 0..h {
                    rows.push(idx(i, kk));
                    cols.push(idx(j, kk));
                    slots.push(Slot::Jx { nz, dk: 0 });
                }
            }
        }
        // dF/dx' blocks, scaled per column harmonic by j*l*w0. A constant
        // (linear-capacitor) entry has a purely DC spectrum, so it is
        // frequency-diagonal like the LTI jx entries; only variable (nonlinear
        // charge) entries generate the dense Toeplitz block.
        for nz in 0..cdc.jxd_rows.len() {
            let (i, j) = (cdc.jxd_rows[nz], cdc.jxd_cols[nz]);
            if cdc.jxd_var[nz] {
                for kk in 0..h {
                    for l in 0..h {
                        let dk = kk as i64 - l as i64;
                        if dk.abs() > band {
                            continue;
                        }
                        rows.push(idx(i, kk));
                        cols.push(idx(j, l));
                        slots.push(Slot::Jxd {
                            nz,
                            dk,
                            l: l as i64,
                        });
                    }
                }
            } else {
                for kk in 0..h {
                    rows.push(idx(i, kk));
                    cols.push(idx(j, kk));
                    slots.push(Slot::Jxd {
                        nz,
                        dk: 0,
                        l: kk as i64,
                    });
                }
            }
        }
        // gmin regularisation on node-voltage harmonics (as the DC/AC solves do).
        let nodes = (0..cdc.n).filter(|&i| cdc.kinds[i] == sane_dae::UnknownKind::NodeVoltage);
        for i in nodes {
            for kk in 0..h {
                rows.push(idx(i, kk));
                cols.push(idx(i, kk));
                slots.push(Slot::Gmin);
            }
        }

        // Deduplicate into a CSC skeleton with per-entry slots (duplicate
        // positions sum), then analyze once (KLU BTF + per-block AMD).
        let dim = n * h;
        let total = rows.len();
        let mut order: Vec<usize> = (0..total).collect();
        order.sort_unstable_by_key(|&e| (cols[e], rows[e]));
        let mut col_ptr = vec![0usize; dim + 1];
        let mut row_idx: Vec<usize> = Vec::with_capacity(total);
        let mut slot_of = vec![0usize; total];
        let (mut prev_r, mut prev_c) = (usize::MAX, usize::MAX);
        for &e in &order {
            let (r, c) = (rows[e], cols[e]);
            if r != prev_r || c != prev_c {
                row_idx.push(r);
                col_ptr[c + 1] += 1;
                (prev_r, prev_c) = (r, c);
            }
            slot_of[e] = row_idx.len() - 1;
        }
        for j in 0..dim {
            col_ptr[j + 1] += col_ptr[j];
        }
        let skeleton = GeneralCsc {
            n: dim,
            col_ptr: col_ptr.clone(),
            row_idx: row_idx.clone(),
            values: vec![Complex64::new(1.0, 0.0); row_idx.len()],
        };
        let sym = KluSymbolic::analyze(&skeleton).ok()?;
        Some(CompiledHb {
            cdc,
            n,
            k,
            h,
            m,
            r2c,
            c2r,
            col_ptr,
            row_idx,
            sym,
            slots,
            slot_of,
        })
    }

    /// Synthesise the per-unknown time samples of `x` and `x'` from the spectra.
    /// `x'` has coefficients `jkw0 X_k`. Uses the inverse real FFT, which is the
    /// unnormalised sum `Σ_k X_k e^{jkw0 t_m}` -- exactly `x(t_m)`.
    fn synth(&self, spectra: &[Vec<Complex64>], w0: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        crate::HB_SYNTH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut x_time = vec![vec![0.0; self.m]; self.n];
        let mut xd_time = vec![vec![0.0; self.m]; self.n];
        let mut spec = self.c2r.make_input_vec();
        let mut sd = self.c2r.make_input_vec();
        for i in 0..self.n {
            for s in spec.iter_mut() {
                *s = Complex64::new(0.0, 0.0);
            }
            for s in sd.iter_mut() {
                *s = Complex64::new(0.0, 0.0);
            }
            for kk in 0..self.h {
                let xk = spectra[i][kk];
                spec[kk] = xk;
                sd[kk] = Complex64::new(0.0, kk as f64 * w0) * xk;
            }
            // DC bin must be real (and so must Nyquist, which is zero here).
            spec[0].im = 0.0;
            sd[0] = Complex64::new(0.0, 0.0);
            // `process` consumes its input as scratch; `spec` / `sd` are fully
            // rebuilt (zeroed + refilled) on the next unknown, so feed them
            // directly rather than cloning a fresh input vector per unknown.
            self.c2r.process(&mut spec, &mut x_time[i]).unwrap();
            self.c2r.process(&mut sd, &mut xd_time[i]).unwrap();
        }
        (x_time, xd_time)
    }

    /// Forward real FFT of a per-unknown sample sequence, normalised to the
    /// physical Fourier coefficients (`1/M`), keeping harmonics `0..=K`.
    fn analyse(&self, time: &[f64]) -> Vec<Complex64> {
        let mut buf = self.r2c.make_input_vec();
        buf.copy_from_slice(time);
        let mut spec = self.r2c.make_output_vec();
        self.r2c.process(&mut buf, &mut spec).unwrap();
        let inv_m = 1.0 / self.m as f64;
        spec[..self.h].iter().map(|c| c * inv_m).collect()
    }

    /// Lay out the tape inputs for `lanes` consecutive AFT samples (`base..base+lanes`)
    /// in structure-of-arrays form: `soa[k][lane]` is input `k` for sample `base+lane`.
    /// Unused lanes (only the last chunk, since `m` is a power of two) are padded
    /// with sample `base` so the batch eval stays in-domain; their outputs are
    /// discarded by the caller. This is the per-chunk feed for the SoA batch kernel.
    #[allow(clippy::too_many_arguments)]
    fn fill_soa<const L: usize>(
        &self,
        x_time: &[Vec<f64>],
        xd_time: &[Vec<f64>],
        p: &[f64],
        period: f64,
        base: usize,
        lanes: usize,
        xv: &mut [f64],
        xdv: &mut [f64],
        inbs: &mut [Vec<f64>],
        soa: &mut Vec<[f64; L]>,
    ) {
        let n = self.n;
        for lane in 0..lanes {
            let mm = base + lane;
            for i in 0..n {
                xv[i] = x_time[i][mm];
                xdv[i] = xd_time[i][mm];
            }
            let t = mm as f64 / self.m as f64 * period;
            self.cdc.fill_inputs(xv, xdv, p, t, &mut inbs[lane]);
        }
        if lanes < L {
            let src = inbs[0].clone();
            for b in inbs.iter_mut().take(L).skip(lanes) {
                b.clone_from(&src);
            }
        }
        let ninp = inbs[0].len();
        soa.clear();
        soa.resize(ninp, [0.0; L]);
        for (k, slot) in soa.iter_mut().enumerate() {
            for lane in 0..L {
                slot[lane] = inbs[lane][k];
            }
        }
    }

    /// One fused AFT evaluation pass per Newton iteration, from the
    /// already-synthesised state waveforms (see [`synth`](Self::synth)): the
    /// step tape emits `[residual (n) | jx values (nzx)]` in a single batched
    /// evaluation, so the residual comes for free with the Jacobian values --
    /// there is no separate residual-tape pass. The jxd tape rides the same
    /// `fill_soa` inputs. Results land in flat time-major buffers
    /// (`buf[e * m + mm]`, each entry's sample sequence contiguous for the
    /// FFTs), reused across iterations by the caller.
    ///
    /// The M AFT samples are independent, so they parallelise over the
    /// worker pool in chunks of `L` samples, each chunk one evaluation per
    /// sample through the step tape's current backend.
    fn eval_aft(
        &self,
        x_time: &[Vec<f64>],
        xd_time: &[Vec<f64>],
        p: &[f64],
        w0: f64,
        f_time: &mut Vec<f64>,
        jx_time: &mut Vec<f64>,
        jxd_time: &mut Vec<f64>,
    ) {
        let period = 2.0 * std::f64::consts::PI / w0;
        let n = self.n;
        let cn = self.cdc.n;
        let nzx = self.cdc.jx_rows.len();
        let nzd = self.cdc.jxd_rows.len();
        const L: usize = 4;
        let m = self.m;
        let width = n + nzx + nzd;
        // Each chunk returns one flat block `v[o * lanes + lane]` over its
        // outputs; the serial scatter below transposes into the time-major
        // buffers (disjoint strided writes, cheap at these sizes).
        let chunks: Vec<Vec<f64>> = sane_core::log_stage!(
            "hb/aft_eval",
            crate::parallel::install(|| {
                (0..m.div_ceil(L))
                    .into_par_iter()
                    .map_init(
                        || {
                            (
                                vec![0.0f64; n],
                                vec![0.0f64; n],
                                vec![Vec::<f64>::new(); L],
                                Vec::<[f64; L]>::new(),
                                (Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()),
                                Vec::<[f64; L]>::new(),
                            )
                        },
                        |(xv, xdv, inbs, soa, (fb, wb, rb), ob), c| {
                            let base = c * L;
                            let lanes = L.min(m - base);
                            self.fill_soa::<L>(
                                x_time, xd_time, p, period, base, lanes, xv, xdv, inbs, soa,
                            );
                            let mut v = vec![0.0f64; width * lanes];
                            // Step tape: residual ++ jx values in one evaluation.
                            self.cdc.tape_step.eval_lanes::<L>(soa, fb, wb, rb, ob);
                            for o in 0..n + nzx {
                                let src = if o < n { ob[o] } else { ob[cn + (o - n)] };
                                for lane in 0..lanes {
                                    v[o * lanes + lane] = src[lane];
                                }
                            }
                            self.cdc.tape_jxd.eval_lanes::<L>(soa, fb, wb, rb, ob);
                            for nz in 0..nzd {
                                for lane in 0..lanes {
                                    v[(n + nzx + nz) * lanes + lane] = ob[nz][lane];
                                }
                            }
                            v
                        },
                    )
                    .collect()
            })
        );
        f_time.clear();
        f_time.resize(n * m, 0.0);
        jx_time.clear();
        jx_time.resize(nzx * m, 0.0);
        jxd_time.clear();
        jxd_time.resize(nzd * m, 0.0);
        let mut base = 0usize;
        for v in &chunks {
            let lanes = v.len() / width;
            for o in 0..width {
                let dst = if o < n {
                    &mut f_time[o * m + base..]
                } else if o < n + nzx {
                    &mut jx_time[(o - n) * m + base..]
                } else {
                    &mut jxd_time[(o - n - nzx) * m + base..]
                };
                dst[..lanes].copy_from_slice(&v[o * lanes..o * lanes + lanes]);
            }
            base += lanes;
        }
    }

    /// The DC start spectrum: `X_{i,0} = x_dc[i]`, all higher harmonics zero.
    fn dc_start(&self, x_dc: &[f64]) -> Vec<Vec<Complex64>> {
        (0..self.n)
            .map(|i| {
                let mut v = vec![Complex64::new(0.0, 0.0); self.h];
                v[0] = Complex64::new(x_dc.get(i).copied().unwrap_or(0.0), 0.0);
                v
            })
            .collect()
    }

    /// Newton on the harmonic residual from a given start spectrum. Returns the
    /// updated spectra, whether it converged, the iteration count, and the final
    /// residual norm. The block-sparse Jacobian is refilled in place each step;
    /// the symbolic factorisation is reused.
    fn newton_from(
        &self,
        p: &[f64],
        mut spectra: Vec<Vec<Complex64>>,
        w0: f64,
        tol: f64,
        max_iter: usize,
    ) -> (Vec<Vec<Complex64>>, bool, usize, f64) {
        let dim = self.n * self.h;
        let mut last_norm = f64::INFINITY;
        let mut converged = false;
        let mut iters = 0;
        // Per-solve factorization state: the first iteration factors with full
        // pivoting, later ones replay the frozen pivot sequence (KLU numeric-only
        // refactor) on the refreshed harmonic-block values.
        let mut csc = GeneralCsc {
            n: dim,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values: vec![Complex64::new(0.0, 0.0); self.row_idx.len()],
        };
        let mut solver: Option<KluSolver<Complex64>> = None;
        // Fused-pass sample buffers (time-major, entry sequences contiguous),
        // reused across iterations.
        let (mut f_time, mut jx_time, mut jxd_time) = (Vec::new(), Vec::new(), Vec::new());
        // the previous iterate and its step, for the retroactive backtracking
        let mut retract: Option<(Vec<Vec<Complex64>>, Vec<Complex64>)> = None;
        let mut prev_norm = f64::INFINITY;
        let mut bt = crate::newton::Backtrack::new(LINE_SEARCH_TRIES);
        // `tol` is the absolute floor of both residual and update, with the
        // engine's relative tolerance (see `Convergence::from_tol`).
        let crit = self.cdc.criterion(&crate::Convergence::from_tol(tol));
        for it in 0..max_iter.max(1) {
            iters = it + 1;
            // Synthesise the state waveforms once per iteration (#51), then one
            // fused AFT pass evaluates residual and Jacobian values together --
            // the step tape emits both, so the residual costs no extra tape work.
            let (x_time, xd_time) = sane_core::log_stage!("hb/synth", self.synth(&spectra, w0));
            self.eval_aft(
                &x_time,
                &xd_time,
                p,
                w0,
                &mut f_time,
                &mut jx_time,
                &mut jxd_time,
            );
            let r: Vec<Vec<Complex64>> = sane_core::log_stage!(
                "hb/res_fft",
                (0..self.n)
                    .map(|i| self.analyse(&f_time[i * self.m..(i + 1) * self.m]))
                    .collect()
            );
            let rnorm = r.iter().flatten().map(|c| c.norm()).fold(0.0_f64, f64::max);
            // Retroactive backtracking (see `newton`): the residual of this
            // iterate is the probe of the previous step. A step that grew the
            // residual beyond `HB_BACKTRACK_GROWTH` is retracted to a fraction
            // and re-evaluated before its Jacobian is used; the transform pass
            // that produced the residual also produced that Jacobian, so
            // probing forward would cost the same pass twice. Ordinary
            // non-monotone steps pass: Newton on the harmonic residual is not
            // monotone in the max-norm near a solution.
            if let Some((prev, dx)) = &retract {
                if !(rnorm <= prev_norm * HB_BACKTRACK_GROWTH) {
                    if let Some(alpha) = bt.shrink() {
                        for i in 0..self.n {
                            for kk in 0..self.h {
                                spectra[i][kk] = prev[i][kk] + dx[self.idx(i, kk)] * alpha;
                            }
                            spectra[i][0].im = 0.0;
                        }
                        continue;
                    }
                }
            }
            bt = crate::newton::Backtrack::new(LINE_SEARCH_TRIES);
            last_norm = rnorm;
            // The residual half of the shared contract, per harmonic: every
            // row of unknown `i` within the floor of `i`'s kind.
            // The residual half of the shared contract, per harmonic: every
            // row of unknown `i` within the floor of `i`'s kind. The update
            // half is not demanded here (measured: a quarter more iterations
            // on the easy drives and 2.5x on a hard continuation for no change
            // in the spectrum); `tol` is the harmonic KCL tolerance.
            if (0..self.n).all(|i| {
                let floor = crit.residual_floor(i);
                r[i].iter().all(|c| c.norm() <= floor)
            }) {
                converged = true;
                break;
            }
            if !rnorm.is_finite() {
                break; // diverged
            }
            prev_norm = rnorm;

            // FFT the device Jacobian waveforms to the entry spectra used by the
            // Toeplitz blocks (LTI entries skip the FFT, DC bin only).
            let (gx_spec, gxd_spec) = self.jacobian_spectra(&jx_time, &jxd_time);

            // Scatter the values straight into the CSC skeleton from the slot
            // recipe (duplicate positions sum).
            sane_core::log_stage!("hb/scatter", {
                for v in csc.values.iter_mut() {
                    *v = Complex64::new(0.0, 0.0);
                }
                for (e, slot) in self.slots.iter().enumerate() {
                    let v = match *slot {
                        Slot::Jx { nz, dk } => spec_at(&gx_spec[nz], dk),
                        Slot::Jxd { nz, dk, l } => {
                            Complex64::new(0.0, l as f64 * w0) * spec_at(&gxd_spec[nz], dk)
                        }
                        Slot::Gmin => Complex64::new(GMIN_DC, 0.0),
                    };
                    csc.values[self.slot_of[e]] += v;
                }
            });
            // rhs = -R (flattened), solve J dX = -R.
            let mut rhs = vec![Complex64::new(0.0, 0.0); dim];
            for i in 0..self.n {
                for kk in 0..self.h {
                    rhs[self.idx(i, kk)] = -r[i][kk];
                }
            }
            let dx: Vec<Complex64> = {
                let refactored = sane_core::log_stage!(
                    "hb/refactor",
                    matches!(solver.as_mut().map(|sl| sl.refactor(&csc)), Some(Ok(())))
                );
                if !refactored {
                    solver = match sane_core::log_stage!(
                        "hb/factor",
                        self.sym.factor(&csc, &KluSettings::default())
                    ) {
                        Ok(sl) => Some(sl),
                        Err(_) => break, // singular block matrix; report non-convergence
                    };
                }
                let lu = solver.as_ref().unwrap();
                match sane_core::log_stage!("hb/backsolve", lu.solve(&rhs)) {
                    Ok(d) => d,
                    Err(_) => break,
                }
            };
            let prev = spectra.clone();
            for i in 0..self.n {
                for kk in 0..self.h {
                    spectra[i][kk] += dx[self.idx(i, kk)];
                }
                spectra[i][0].im = 0.0; // DC stays real
            }
            retract = Some((prev, dx));
        }
        (spectra, converged, iters, last_norm)
    }

    /// Solve the periodic steady state by Newton on the harmonic residual,
    /// warm-started from the DC operating point `x_dc` (all higher harmonics
    /// zero). `w0` is the fundamental angular frequency (must match the drive).
    pub fn solve(&self, p: &[f64], x_dc: &[f64], w0: f64, tol: f64, max_iter: usize) -> HbResult {
        let start = self.dc_start(x_dc);
        let (spectra, converged, iters, last_norm) = self.newton_from(p, start, w0, tol, max_iter);
        HbResult {
            spectra,
            converged,
            iters,
            w0,
            harmonics: self.k,
            samples: self.m,
            residual_norm: last_norm,
        }
    }

    /// Solve by **source-stepping continuation**: ramp the excitation amplitude
    /// (the parameters at indices `ramp`, e.g. the `SIN` amplitudes) from 0 to
    /// their target via a homotopy parameter `lambda` in `[0, 1]`. At
    /// `lambda = 0` the only periodic solution is the DC point (higher harmonics
    /// vanish), so the start is exact; each step warm-starts from the previous
    /// converged spectrum, and the step size adapts (grows on easy steps, halves
    /// on failure). This reaches periodic steady states a cold Newton from DC
    /// cannot -- large signal swings that cross device regions. `iters` returns
    /// the total Newton iterations across all continuation steps.
    pub fn solve_continuation(
        &self,
        p: &[f64],
        x_dc: &[f64],
        w0: f64,
        ramp: &[usize],
        tol: f64,
        max_iter: usize,
    ) -> HbResult {
        let mut spectra = self.dc_start(x_dc);
        let mut lambda = 0.0_f64;
        let mut step = HB_CONT_DLAM0;
        let mut total_iters = 0;
        let mut last_norm = f64::INFINITY;
        let mut converged = true;
        // Scale the ramped parameters by a factor and run Newton.
        let scaled = |factor: f64| -> Vec<f64> {
            let mut q = p.to_vec();
            for &i in ramp {
                q[i] = p[i] * factor;
            }
            q
        };
        while lambda < 1.0 - HB_CONT_LAMBDA_EPS {
            let trial = (lambda + step).min(1.0);
            let q = scaled(trial);
            let (s, conv, it, rn) = self.newton_from(&q, spectra.clone(), w0, tol, max_iter);
            total_iters += it;
            if conv {
                spectra = s;
                lambda = trial;
                last_norm = rn;
                if it <= HB_CONT_EASY_ITERS {
                    step = (step * HB_CONT_GROW).min(HB_CONT_DLAM_MAX); // grow on easy steps
                }
            } else {
                step *= HB_CONT_SHRINK;
                if step < HB_CONT_DLAM_MIN {
                    converged = false;
                    last_norm = rn;
                    break;
                }
            }
        }
        HbResult {
            spectra,
            converged,
            iters: total_iters,
            w0,
            harmonics: self.k,
            samples: self.m,
            residual_norm: last_norm,
        }
    }

    /// Reduce the fused pass's Jacobian sample sequences (see
    /// [`eval_aft`](Self::eval_aft)) to their Fourier coefficients (`0..=K`).
    /// An LTI `dF/dx` entry is constant in time, so we skip its FFT and keep
    /// only the DC bin.
    #[allow(clippy::type_complexity)]
    fn jacobian_spectra(
        &self,
        jx_time: &[f64],
        jxd_time: &[f64],
    ) -> (Vec<Vec<Complex64>>, Vec<Vec<Complex64>>) {
        let nzx = self.cdc.jx_rows.len();
        let nzd = self.cdc.jxd_rows.len();
        let m = self.m;
        let _fft_guard = sane_core::log::scope("hb/jac_fft");
        let gx: Vec<Vec<Complex64>> = (0..nzx)
            .map(|nz| {
                if self.cdc.jx_var[nz] {
                    self.analyse(&jx_time[nz * m..(nz + 1) * m])
                } else {
                    let mut v = vec![Complex64::new(0.0, 0.0); self.h];
                    v[0] = Complex64::new(jx_time[nz * m], 0.0);
                    v
                }
            })
            .collect();
        let gxd: Vec<Vec<Complex64>> = (0..nzd)
            .map(|nz| {
                if self.cdc.jxd_var[nz] {
                    self.analyse(&jxd_time[nz * m..(nz + 1) * m])
                } else {
                    let mut v = vec![Complex64::new(0.0, 0.0); self.h];
                    v[0] = Complex64::new(jxd_time[nz * m], 0.0);
                    v
                }
            })
            .collect();
        (gx, gxd)
    }

    /// Forward real FFT of a sample sequence to harmonics `0..=kmax` (physical
    /// coefficients, `1/M` normalised). `kmax` must satisfy `kmax <= m/2`.
    fn analyse_k(&self, time: &[f64], kmax: usize) -> Vec<Complex64> {
        let mut buf = self.r2c.make_input_vec();
        buf.copy_from_slice(time);
        let mut spec = self.r2c.make_output_vec();
        self.r2c.process(&mut buf, &mut spec).unwrap();
        let inv_m = 1.0 / self.m as f64;
        spec[..=kmax].iter().map(|c| c * inv_m).collect()
    }

    /// All-parameter sensitivity of the steady-state harmonic coefficients at
    /// unknown `out_idx`, by the implicit-function adjoint on the **exact**
    /// harmonic-balance Jacobian. Returns `grad[k][j] = dX_{out,k}/dp_j`
    /// (complex) for `k = 0..=K` and every parameter column `j` in
    /// `0..nparams`. `None` if the AFT grid is too coarse for the Jacobian
    /// spectra (`m/2 < 2K`) or the system is singular.
    ///
    /// At the periodic steady state `R(X, p) = 0`, so `dX/dp = -J^{-1} dR/dp`.
    /// The Newton block-Toeplitz `J` the solver assembles is only the
    /// *holomorphic* part `dR_k/dX_l = G_{k-l}` (it folds `X_{-l} = conj X_l`,
    /// dropping the conjugate coupling). The **exact** Jacobian is recovered in
    /// the two-sided coefficient space `k, l in [-K, K]` -- there `X_l` are
    /// independent and `dR_k/dX_l = G_{k-l}` is the *complete* derivative (a
    /// pure Toeplitz, needing `dF/dx` harmonics out to `2K`). `dR_k/dp_j` is the
    /// `k`-th Fourier coefficient of the time-sampled `dF/dp_j` -- the parameter
    /// Jacobian routed through the same AFT. One transpose solve per output
    /// harmonic then yields the adjoint, and a sparse contraction the gradient.
    /// Requires `ensure_param_jac` on the backing `CompiledDc`.
    pub fn coeff_gradient(
        &self,
        spectra: &[Vec<Complex64>],
        p: &[f64],
        w0: f64,
        out_idx: usize,
        nparams: usize,
    ) -> Option<Vec<Vec<Complex64>>> {
        let n = self.n;
        let k = self.k as i64;
        let h2 = 2 * self.k + 1;
        let dim = n * h2;
        let flat = |i: usize, kk: i64| -> usize { i * h2 + (kk + k) as usize };
        let ToeplitzJacobian {
            lu,
            pj_rows,
            pj_cols,
            pjs,
            ..
        } = self.build_toeplitz_jacobian(spectra, p, w0)?;

        // One transpose solve per output harmonic k0 = 0..K, then contract the
        // adjoint with dR/dp_j (sparse over the parameter-Jacobian pattern).
        let mut grad = vec![vec![Complex64::new(0.0, 0.0); nparams]; self.k + 1];
        for k0 in 0..=self.k {
            let mut e = vec![Complex64::new(0.0, 0.0); dim];
            e[flat(out_idx, k0 as i64)] = Complex64::new(1.0, 0.0);
            let lam = lu.solve_transpose(&e).ok()?; // J^T lambda = e_{(out, k0)}
            for nz in 0..pjs.len() {
                let (i, j) = (pj_rows[nz], pj_cols[nz]);
                if j >= nparams {
                    continue;
                }
                let mut acc = Complex64::new(0.0, 0.0);
                for kk in -k..=k {
                    acc += lam[flat(i, kk)] * spec_at(&pjs[nz], kk);
                }
                grad[k0][j] -= acc; // dX_{out,k0}/dp_j = -(J^{-T} e)^T dR/dp_j
            }
        }
        Some(grad)
    }

    /// Exact second-order-adjoint Hessian of the steady-state coefficient
    /// `X_{out_idx, k_metric}` w.r.t. the parameter subset `subset` (indices into
    /// `param_names`). Dense `|subset| x |subset|` complex matrix
    /// `H_ab = d^2 X_{out,k}/dp_a dp_b`, or `None` if `ensure_hessian` /
    /// `ensure_param_jac` were not called, the AFT grid is too coarse
    /// (`m/2 < 2K`), or the system is singular.
    ///
    /// The metric is linear in the unknowns, so with the adjoint `J^T lambda =
    /// e_{(out,k)}` and the forward state sensitivities `J s_a = -dR/dp_a`,
    /// `H_ab = -lambda^T ( R_XX[s_a,s_b] + R_Xp_b s_a + R_Xp_a s_b + R_pa pb )`.
    /// Every second-derivative contraction reduces, through the AFT, to a
    /// time-domain sum: the **adjoint waveform** `l_i(t) = sum_k lambda_{i,k}
    /// e^{-jk w0 t}` weights the device second derivatives (the same
    /// Lagrangian-Hessian blocks the DC solve uses, here sampled along the
    /// waveform), contracted with the **state-sensitivity waveforms**
    /// `sigma_a,j(t) = sum_k s_a[j,k] e^{jk w0 t}` and -- since a HB unknown
    /// drives both `x` and `x'` -- their time derivatives `sigma'_a,j(t)`. The
    /// `x'` Hessian blocks (`L_x'x', L_x x', L_x'p`) make this exact also for
    /// nonlinear charge storage.
    pub fn coeff_hessian(
        &self,
        spectra: &[Vec<Complex64>],
        p: &[f64],
        w0: f64,
        out_idx: usize,
        k_metric: usize,
        subset: &[usize],
    ) -> Option<Vec<Vec<Complex64>>> {
        let n = self.n;
        let k = self.k as i64;
        let h2 = 2 * self.k + 1;
        if k_metric > self.k {
            return None;
        }
        let flat = |i: usize, kk: i64| -> usize { i * h2 + (kk + k) as usize };
        let dim = n * h2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let period = two_pi / w0;
        let ToeplitzJacobian {
            lu,
            pj_rows,
            pj_cols,
            pjs,
            x_time,
            xd_time,
        } = self.build_toeplitz_jacobian(spectra, p, w0)?;
        // Scratch for the per-sample Lagrangian-Hessian evaluation below.
        let (mut xv, mut xdv) = (vec![0.0; n], vec![0.0; n]);

        // --- adjoint lambda (one transpose solve) and forward state sens s_a ---
        let mut e = vec![Complex64::new(0.0, 0.0); dim];
        e[flat(out_idx, k_metric as i64)] = Complex64::new(1.0, 0.0);
        let lam = lu.solve_transpose(&e).ok()?;
        let ns = subset.len();
        let mut svec: Vec<Vec<Complex64>> = Vec::with_capacity(ns);
        for &a in subset {
            let mut rhs = vec![Complex64::new(0.0, 0.0); dim];
            for nz in 0..pjs.len() {
                if pj_cols[nz] == a {
                    for kk in -k..=k {
                        rhs[flat(pj_rows[nz], kk)] -= spec_at(&pjs[nz], kk);
                    }
                }
            }
            svec.push(lu.solve(&rhs).ok()?);
        }

        // --- adjoint waveform l_i(t) and state-sensitivity waveforms sigma_a,j(t) ---
        // `sig` is the state-sensitivity waveform (the x path of dX/dp_a);
        // `sigd` its time derivative (the x' path, coefficient j*k*w0). A HB
        // unknown X_l drives both x(t) and x'(t), so an exact Hessian needs both.
        let mut ell = vec![vec![Complex64::new(0.0, 0.0); self.m]; n];
        let mut sig = vec![vec![vec![Complex64::new(0.0, 0.0); self.m]; n]; ns];
        let mut sigd = vec![vec![vec![Complex64::new(0.0, 0.0); self.m]; n]; ns];
        for mm in 0..self.m {
            let th = two_pi * mm as f64 / self.m as f64;
            for i in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for kk in -k..=k {
                    acc += lam[flat(i, kk)] * Complex64::from_polar(1.0, -(kk as f64) * th);
                }
                ell[i][mm] = acc;
            }
            for a in 0..ns {
                for j in 0..n {
                    let (mut acc, mut accd) = (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
                    for kk in -k..=k {
                        let basis = Complex64::from_polar(1.0, kk as f64 * th);
                        let xc = svec[a][flat(j, kk)] * basis;
                        acc += xc;
                        accd += Complex64::new(0.0, kk as f64 * w0) * xc; // d/dt
                    }
                    sig[a][j][mm] = acc;
                    sigd[a][j][mm] = accd;
                }
            }
        }

        // --- per-sample Lagrangian-Hessian contraction (re/im of l(t) separately) ---
        let [xx_rc, xp_rc, pp_rc, xdxd_rc, xxd_rc, xdp_rc] = self.cdc.hessian_block_pattern()?;
        // Map a parameter index to its position in `subset` (for the L_pp lookup).
        let np = self.cdc.param_count();
        let mut loc = vec![usize::MAX; np];
        for (a, &j) in subset.iter().enumerate() {
            loc[j] = a;
        }
        let inv_m = 1.0 / self.m as f64;
        let mut h = vec![vec![Complex64::new(0.0, 0.0); ns]; ns];
        let (mut ell_re, mut ell_im) = (vec![0.0; n], vec![0.0; n]);
        for mm in 0..self.m {
            for i in 0..n {
                ell_re[i] = ell[i][mm].re;
                ell_im[i] = ell[i][mm].im;
                xv[i] = x_time[i][mm];
                xdv[i] = xd_time[i][mm];
            }
            let t = mm as f64 / self.m as f64 * period;
            let [xxr, xpr, ppr, xdxdr, xxdr, xdpr] =
                self.cdc.hessian_block_values(&xv, &xdv, p, t, &ell_re)?;
            let [xxi, xpi, ppi, xdxdi, xxdi, xdpi] =
                self.cdc.hessian_block_values(&xv, &xdv, p, t, &ell_im)?;
            // L_pp restricted to the subset (mirrors the DC dense `lpp[ja][jb]` read).
            let mut lpp = vec![vec![Complex64::new(0.0, 0.0); ns]; ns];
            for kix in 0..pp_rc.0.len() {
                let (r, col) = (pp_rc.0[kix], pp_rc.1[kix]);
                if loc[r] != usize::MAX && loc[col] != usize::MAX {
                    lpp[loc[r]][loc[col]] = Complex64::new(ppr[kix], ppi[kix]);
                }
            }
            for a in 0..ns {
                for b in a..ns {
                    let (ja, jb) = (subset[a], subset[b]);
                    let mut acc = Complex64::new(0.0, 0.0);
                    // R_XX direction (a,b) through both the x and the x' paths:
                    // s_a^T L_xx s_b  +  L_x x'(sig_a sigd_b + sig_b sigd_a)  +  sigd_a^T L_x'x' sigd_b.
                    for kix in 0..xx_rc.0.len() {
                        let lc = Complex64::new(xxr[kix], xxi[kix]);
                        acc += lc * sig[a][xx_rc.0[kix]][mm] * sig[b][xx_rc.1[kix]][mm];
                    }
                    for kix in 0..xxd_rc.0.len() {
                        let lc = Complex64::new(xxdr[kix], xxdi[kix]);
                        let (r, c) = (xxd_rc.0[kix], xxd_rc.1[kix]); // r: x index, c: x' index
                        acc +=
                            lc * (sig[a][r][mm] * sigd[b][c][mm] + sig[b][r][mm] * sigd[a][c][mm]);
                    }
                    for kix in 0..xdxd_rc.0.len() {
                        let lc = Complex64::new(xdxdr[kix], xdxdi[kix]);
                        acc += lc * sigd[a][xdxd_rc.0[kix]][mm] * sigd[b][xdxd_rc.1[kix]][mm];
                    }
                    // R_Xp: x path (L_xp) and x' path (L_x'p), each for column jb (with s_a)
                    // and column ja (with s_b).
                    for kix in 0..xp_rc.0.len() {
                        let lc = Complex64::new(xpr[kix], xpi[kix]);
                        let (r, col) = (xp_rc.0[kix], xp_rc.1[kix]);
                        if col == jb {
                            acc += lc * sig[a][r][mm];
                        }
                        if col == ja {
                            acc += lc * sig[b][r][mm];
                        }
                    }
                    for kix in 0..xdp_rc.0.len() {
                        let lc = Complex64::new(xdpr[kix], xdpi[kix]);
                        let (r, col) = (xdp_rc.0[kix], xdp_rc.1[kix]); // r: x' index
                        if col == jb {
                            acc += lc * sigd[a][r][mm];
                        }
                        if col == ja {
                            acc += lc * sigd[b][r][mm];
                        }
                    }
                    acc += lpp[a][b]; // L_pp[ja][jb]
                    h[a][b] -= acc * inv_m;
                }
            }
        }
        for a in 0..ns {
            for b in (a + 1)..ns {
                h[b][a] = h[a][b];
            }
        }
        Some(h)
    }
}

/// Coefficient at signed harmonic index `dk` of a real signal's spectrum, using
/// conjugate symmetry for negative indices (`X_{-p} = conj(X_p)`).
#[inline]
fn spec_at(spec: &[Complex64], dk: i64) -> Complex64 {
    if dk >= 0 {
        spec[dk as usize]
    } else {
        spec[(-dk) as usize].conj()
    }
}
