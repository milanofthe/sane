//! AC and S-parameter analyses on [`Model`], plus their exact adjoint
//! machinery: responses, per-frequency sensitivities, all-parameter gradients
//! (VJPs over a cached weighted-adjoint tape) and the AC Hessian.

use std::collections::HashMap;
use std::f64::consts::PI;

use ndarray::{s, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3};
use num_complex::Complex64;
use rayon::prelude::*;
use rsdag::{differentiate, ExprId, Node, SymbolId};
use sane_core::log;
use sane_core::Graph;
use sane_dae::Dae as CoreDae;
use sane_solve::CompiledDc;

use crate::model::{Gradient, Model, ModelError, Point};

/// Frequencies per block of an AC sweep: one pivoting factorization each, a
/// unit of work for one worker.
pub const AC_BLOCK: usize = 64;

/// Cached weighted-adjoint tape for one AC input (see
/// `Model::ensure_ac_vjp_tape`): weight symbols and the once-differentiated
/// `dPsi/dx`, `dPsi/dp` expressions.
pub(crate) struct AcVjpTape {
    wb: Vec<SymbolId>,
    wg: Vec<SymbolId>,
    wc: Vec<SymbolId>,
    g_pos: Vec<(usize, usize)>,
    c_pos: Vec<(usize, usize)>,
    dpsi_dx: Vec<ExprId>,
    dpsi_dp: Vec<ExprId>,
}

/// Frequency-independent state shared by one AC adjoint call (see
/// `Model::ac_adjoint_setup`).
struct AcAdjointSetup {
    n: usize,
    input: String,
    gr_s: Vec<usize>,
    gc_s: Vec<usize>,
    gv_s: Vec<f64>,
    cr_s: Vec<usize>,
    cc_s: Vec<usize>,
    cv_s: Vec<f64>,
    b: Vec<Complex64>,
    pnames: Vec<String>,
    prr: Vec<usize>,
    prc: Vec<usize>,
    prv: Vec<f64>,
    gsys: crate::sparse_ac::AcSystem,
    env: HashMap<SymbolId, f64>,
}

impl Model {
    /// Small-signal AC response `H(j2*pi*f)` from source `input` to the unknown at
    /// `out_idx`, at the operating point `x`, solved numerically per frequency in
    /// Rust: `e_out^T (G + jwC)^{-1} (-dF/d(input))`. Returns `(re, im)` per
    /// frequency.
    ///
    /// The frequencies run in blocks of [`AC_BLOCK`] on SANE's worker pool
    /// (`SANE_THREADS`): a block factors at its first frequency with pivoting
    /// and refactors numerically after it. The blocks are fixed, so the result
    /// does not depend on the thread count.
    pub(crate) fn ac_response(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freqs_hz: Vec<f64>,
    ) -> Result<Vec<(f64, f64)>, ModelError> {
        self.inner.bound(&p)?;
        let rows = self.ac_solve(input, &[out_idx], &x, &p, &freqs_hz)?;
        Ok(rows.iter().map(|r| (r[0].re, r[0].im)).collect())
    }

    /// The small-signal responses of the unknowns `outs` to the source
    /// `input` at the operating point `(x, p)`, per frequency (`NaN` where
    /// `G + jwC` is singular): one factorization per frequency for all of
    /// them.
    pub(crate) fn ac_solve(
        &self,
        input: &str,
        outs: &[usize],
        x: &[f64],
        p: &[f64],
        freqs_hz: &[f64],
    ) -> Result<Vec<Vec<Complex64>>, ModelError> {
        let (x, p) = (x.to_vec(), p.to_vec());
        // Transport delays are exact in AC: `hist_k = x_src(t - τ_k)` becomes
        // `e^{-jωτ_k} X_src`, an extra (frequency-dependent) coupling entry.
        self.ensure_hist_jac_ready();
        let mut task = sane_core::log::task(
            "AC",
            "ac",
            &format!(
                "(points: {}, f: {:.3e}..{:.3e} Hz)",
                freqs_hz.len(),
                freqs_hz.first().copied().unwrap_or(0.0),
                freqs_hz.last().copied().unwrap_or(0.0)
            ),
        );
        let n = self.dae().dim();
        // Sparse A = G(+gmin) + jwC over the fixed pattern: symbolic analysis
        // once, numeric refactor per frequency (never densified).
        let (gr, gc, gv) = self.cdc().system_triplets_dc(&x, &p);
        let (cr, cc, cv) = self.cdc().jacobian_q_x_sparse(&x, &p, 0.0);
        let bin = self.jacobian_i_input(input, x.clone(), p.clone(), 0.0)?;
        let b: Vec<Complex64> = bin.iter().map(|v| Complex64::new(-v, 0.0)).collect();
        let (dr, dc, dv, dtau) = self.delay_ac_entries(&x, &p);
        let sym = crate::sparse_ac::SymbolicAc::new(
            n,
            (&gr, &gc, &gv),
            (&cr, &cc, &cv),
            (&dr, &dc, &dv, &dtau),
            false,
            2.0 * PI * freqs_hz.first().copied().unwrap_or(0.0),
        );
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let blocks: Vec<Vec<Vec<Complex64>>> = sane_solve::parallel::install(|| {
            freqs_hz
                .par_chunks(AC_BLOCK)
                .map(|block| {
                    let mut fac = sym.as_ref().map(|s| s.solver());
                    block
                        .iter()
                        .map(|&f| {
                            // A singular A = G + jwC (floating subnet, ideal
                            // VCVS/inductor loop, garbage DC point) has no
                            // small-signal solution at this frequency. Surface it
                            // as NaN -- a catchable signal the Python layer turns
                            // into a warning -- instead of silently coercing to
                            // 0.0, which reads as a flat ~-600 dB response and
                            // masquerades as a size limit (issue #39).
                            match fac.as_mut().and_then(|fa| fa.solve(2.0 * PI * f, &b)) {
                                Some(v) => outs.iter().map(|&i| v[i]).collect(),
                                None => vec![nan; outs.len()],
                            }
                        })
                        .collect()
                })
                .collect()
        });
        let out: Vec<Vec<Complex64>> = blocks.concat();
        let singular = out
            .iter()
            .filter(|r| r.iter().any(|h| h.re.is_nan()))
            .count();
        task.finish(if singular > 0 {
            format!("points: {}, singular: {singular}", out.len())
        } else {
            format!("points: {}", out.len())
        });
        Ok(out)
    }

    /// [`Model::ac_gradient`] over a sweep: per frequency the transfer `H`
    /// and `(dHre/dp, dHim/dp)` for every parameter (in the order of the
    /// returned names), the frequency-independent setup done once.
    #[allow(clippy::type_complexity)]
    pub(crate) fn ac_gradient_sweep(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freqs: &[f64],
    ) -> Result<(Vec<String>, Vec<((f64, f64), Vec<(f64, f64)>)>), ModelError> {
        self.inner.bound(&p)?;
        self.ensure_no_delays("ac_gradient")?;
        let _g = log::scope("sens/ac_gradient");
        let setup = self.ac_adjoint_setup(input, &x, &p)?;
        let np = setup.pnames.len();
        let mut rows = Vec::with_capacity(freqs.len());
        for &f in freqs {
            let w = 2.0 * PI * f;
            let sys = crate::sparse_ac::AcSystem::assemble(
                setup.n,
                (&setup.gr_s, &setup.gc_s, &setup.gv_s),
                (&setup.cr_s, &setup.cc_s, &setup.cv_s),
                w,
            );
            let lu = sys
                .factored()
                .ok_or_else(|| ModelError::Numeric("ac_gradient: singular A".to_string()))?;
            let v = lu
                .solve(&setup.b)
                .map_err(|_| ModelError::Numeric("ac_gradient: singular A".to_string()))?;
            let mut e_out = vec![Complex64::new(0.0, 0.0); setup.n];
            e_out[out_idx] = Complex64::new(1.0, 0.0);
            let lam = lu
                .solve_transpose(&e_out)
                .map_err(|_| ModelError::Numeric("ac_gradient: singular A^T".to_string()))?;

            // Two weight assignments on the SAME tape: the real and imaginary
            // parts of Psi = lam^T (b - A v) (see AcVjpTape).
            let (u_re, expl_re) = self.ac_adjoint_pull(&setup, &lam, &v, w, false)?;
            let (u_im, expl_im) = self.ac_adjoint_pull(&setup, &lam, &v, w, true)?;

            // Operating-point shift: one complex DC transpose solve carries re/im.
            let urhs: Vec<Complex64> = (0..setup.n)
                .map(|m| Complex64::new(u_re[m], u_im[m]))
                .collect();
            let mu = setup.gsys.solve_transpose(&urhs).ok_or_else(|| {
                ModelError::Numeric("ac_gradient: singular DC Jacobian".to_string())
            })?;
            let mut d: Vec<(f64, f64)> = (0..np).map(|k| (expl_re[k], expl_im[k])).collect();
            for t in 0..setup.prv.len() {
                let (i, k) = (setup.prr[t], setup.prc[t]);
                d[k].0 -= mu[i].re * setup.prv[t];
                d[k].1 -= mu[i].im * setup.prv[t];
            }
            rows.push(((v[out_idx].re, v[out_idx].im), d));
        }
        Ok((setup.pnames, rows))
    }

    /// Native VJP sweep for S-parameter (and general multi-output AC) losses:
    /// pull a per-frequency complex cotangent over a SET of output unknowns
    /// back to `dL/dp` in one call.
    ///
    /// `weights[k]` lists `(out_idx, c_re, c_im)` entries at `freqs[k]`; the
    /// VJP convention matches [`Model::ac_gradient`]'s cotangent contraction
    /// `c_re*d(Re H)/dp + c_im*d(Im H)/dp`, summed over the entries. Because
    /// `H_i = e_i^T A^{-1} b` is linear in `e_i`, that sum equals
    /// `Re(d/dp[e~^T A^{-1} b])` with the weighted adjoint seed
    /// `e~ = sum_i conj(c_i) e_i` -- ONE forward + one adjoint solve per
    /// frequency, regardless of how many outputs carry cotangent. All
    /// frequency-independent work (excitation, sparse patterns, the
    /// symbolic tape) is set up once; the per-frequency step is a factored
    /// solve pair plus read-only tape evaluation, so the cost per call is
    /// constant (no symbolic-context growth).
    pub(crate) fn ac_vjp_sweep(
        &self,
        input: &str,
        weights: &[Vec<(usize, f64, f64)>],
        x: Vec<f64>,
        p: Vec<f64>,
        freqs: Vec<f64>,
    ) -> Result<Vec<(String, f64)>, ModelError> {
        self.inner.bound(&p)?;
        self.ensure_no_delays("ac_vjp_sweep")?;
        let _g = log::scope("sens/ac_vjp_sweep");
        if weights.len() != freqs.len() {
            return Err(ModelError::Numeric(
                "ac_vjp_sweep: weights/freqs length mismatch".into(),
            ));
        }
        let setup = self.ac_adjoint_setup(input, &x, &p)?;
        let np = setup.pnames.len();
        let mut acc = vec![0.0; np];
        for (k, f) in freqs.iter().enumerate() {
            if weights[k].is_empty() {
                continue;
            }
            let w = 2.0 * PI * f;
            let sys = crate::sparse_ac::AcSystem::assemble(
                setup.n,
                (&setup.gr_s, &setup.gc_s, &setup.gv_s),
                (&setup.cr_s, &setup.cc_s, &setup.cv_s),
                w,
            );
            let lu = sys
                .factored()
                .ok_or_else(|| ModelError::Numeric("ac_vjp_sweep: singular A".to_string()))?;
            let v = lu
                .solve(&setup.b)
                .map_err(|_| ModelError::Numeric("ac_vjp_sweep: singular A".to_string()))?;
            // Weighted adjoint seed e~ = sum_i conj(c_i) e_i.
            let mut seed = vec![Complex64::new(0.0, 0.0); setup.n];
            for &(idx, cre, cim) in &weights[k] {
                seed[idx] += Complex64::new(cre, -cim);
            }
            let lam = lu
                .solve_transpose(&seed)
                .map_err(|_| ModelError::Numeric("ac_vjp_sweep: singular A^T".to_string()))?;

            let (u_re, expl_re) = self.ac_adjoint_pull(&setup, &lam, &v, w, false)?;
            let urhs: Vec<Complex64> = (0..setup.n).map(|m| Complex64::new(u_re[m], 0.0)).collect();
            let mu = setup.gsys.solve_transpose(&urhs).ok_or_else(|| {
                ModelError::Numeric("ac_vjp_sweep: singular DC Jacobian".to_string())
            })?;
            for t in 0..setup.prv.len() {
                let (i, kk) = (setup.prr[t], setup.prc[t]);
                acc[kk] -= mu[i].re * setup.prv[t];
            }
            for (kk, a) in acc.iter_mut().enumerate() {
                *a += expl_re[kk];
            }
        }
        Ok(setup.pnames.iter().cloned().zip(acc).collect())
    }

    /// S-parameter sweep at an explicit operating point `(x, p)`: `ports` are
    /// `(drive source, out_idx, z0)` triples in port order (the Thevenin form
    /// a deck `P` element lowers to). Returns per frequency the flattened
    /// row-major `n x n` scattering matrix as `(re, im)` pairs, from
    /// `S_ij = 2*sqrt(z0_j/z0_i) * H_ij - delta_ij` with `H_ij` the complex
    /// AC transfer from source j to output i. Signature mirrors
    /// [`Model::ac_response`]; [`Model::sp_sweep`] is the overrides-based
    /// convenience on top.
    pub(crate) fn sp_response(
        &self,
        ports: &[(String, usize, f64)],
        x: Vec<f64>,
        p: Vec<f64>,
        freqs_hz: Vec<f64>,
    ) -> Result<Vec<Vec<(f64, f64)>>, ModelError> {
        self.inner.bound(&p)?;
        let n = ports.len();
        if n == 0 {
            return Err(ModelError::Numeric("sp_response: no ports".into()));
        }
        let nf = freqs_hz.len();
        let mut s = vec![vec![(0.0, 0.0); n * n]; nf];
        for (j, (src, _, z0j)) in ports.iter().enumerate() {
            for (i, (_, out_idx, z0i)) in ports.iter().enumerate() {
                let h = self.ac_response(src, *out_idx, x.clone(), p.clone(), freqs_hz.clone())?;
                let scale = 2.0 * (z0j / z0i).sqrt();
                for (k, &(re, im)) in h.iter().enumerate() {
                    let mut v = (scale * re, scale * im);
                    if i == j {
                        v.0 -= 1.0;
                    }
                    s[k][i * n + j] = v;
                }
            }
        }
        Ok(s)
    }

    /// VJP of [`Model::sp_response`]: pull a complex cotangent
    /// `cot[k][i*n+j] = dL/dRe(S_ij) + j*dL/dIm(S_ij)` at `freqs_hz[k]` back
    /// to `dL/dp` over every parameter, exactly (adjoint through every port
    /// pair, incl. the operating-point shift). One weighted adjoint solve per
    /// (driving port, frequency) via [`Model::ac_vjp_sweep`]; the wave
    /// normalisation `2*sqrt(z0_j/z0_i)` is folded into the seeds (the
    /// `-delta_ij` term is constant and drops out of the derivative).
    pub(crate) fn sp_vjp(
        &self,
        ports: &[(String, usize, f64)],
        x: Vec<f64>,
        p: Vec<f64>,
        freqs_hz: Vec<f64>,
        cot: &[Vec<(f64, f64)>],
    ) -> Result<Vec<(String, f64)>, ModelError> {
        self.inner.bound(&p)?;
        let n = ports.len();
        if n == 0 {
            return Err(ModelError::Numeric("sp_vjp: no ports".into()));
        }
        if cot.len() != freqs_hz.len() || cot.iter().any(|row| row.len() != n * n) {
            return Err(ModelError::Numeric(format!(
                "sp_vjp: cotangent must be nf x {} entries",
                n * n
            )));
        }
        let mut acc: Vec<(String, f64)> = Vec::new();
        for (j, (src, _, z0j)) in ports.iter().enumerate() {
            let weights: Vec<Vec<(usize, f64, f64)>> = cot
                .iter()
                .map(|row| {
                    ports
                        .iter()
                        .enumerate()
                        .filter_map(|(i, (_, out_idx, z0i))| {
                            let (cre, cim) = row[i * n + j];
                            if cre == 0.0 && cim == 0.0 {
                                return None;
                            }
                            let scale = 2.0 * (z0j / z0i).sqrt();
                            Some((*out_idx, scale * cre, scale * cim))
                        })
                        .collect()
                })
                .collect();
            if weights.iter().all(|w| w.is_empty()) {
                continue;
            }
            let part = self.ac_vjp_sweep(src, &weights, x.clone(), p.clone(), freqs_hz.clone())?;
            if acc.is_empty() {
                acc = part;
            } else {
                for (a, b) in acc.iter_mut().zip(part) {
                    a.1 += b.1;
                }
            }
        }
        if acc.is_empty() {
            // all-zero cotangent: still return the parameter axis
            let pnames = self.cdc().param_names(&self.context_arc().lock().unwrap());
            acc = pnames.into_iter().map(|nm| (nm, 0.0)).collect();
        }
        Ok(acc)
    }

    /// Frequency-independent setup shared by the AC adjoint pulls: sparse
    /// numeric G/C (+ the DC shift system), the excitation, dF/dp triplets,
    /// the cached symbolic tape and the operating-point environment.
    fn ac_adjoint_setup(
        &self,
        input: &str,
        x: &[f64],
        p: &[f64],
    ) -> Result<AcAdjointSetup, ModelError> {
        let n = self.dae().dim();
        let (gr_s, gc_s, gv_s) = self.cdc().system_triplets_dc(x, p);
        let (cr_s, cc_s, cv_s) = self.cdc().jacobian_q_x_sparse(x, p, 0.0);
        let bin = self.jacobian_i_input(input, x.to_vec(), p.to_vec(), 0.0)?;
        let b: Vec<Complex64> = bin.iter().map(|&val| Complex64::new(-val, 0.0)).collect();
        self.cdc()
            .ensure_param_jac(&mut self.context_arc().lock().unwrap(), self.dae());
        let pnames = self.cdc().param_names(&self.context_arc().lock().unwrap());
        let ((prr, prc, prv), _) = self.cdc().jacobian_p_sparse(x, p, 0.0);
        let empty: (Vec<usize>, Vec<usize>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
        let gsys = crate::sparse_ac::AcSystem::assemble(
            n,
            (&gr_s, &gc_s, &gv_s),
            (&empty.0, &empty.1, &empty.2),
            0.0,
        );
        self.ensure_ac_vjp_tape(input)?;
        // Operating-point environment (weights are filled per pull).
        let arc = self.context_arc();
        let mut cg = arc.lock().unwrap();
        let c = &mut *cg;
        let mut env: HashMap<SymbolId, f64> = HashMap::new();
        for (i, &sy) in self.dae().x.iter().enumerate() {
            env.insert(sy, x.get(i).copied().unwrap_or(0.0));
        }
        for (j, name) in pnames.iter().enumerate() {
            let e = c.sym(name);
            if let Node::Symbol(sy) = c.node(e) {
                env.insert(*sy, p.get(j).copied().unwrap_or(0.0));
            }
        }
        env.insert(self.dae().t, 0.0);
        drop(cg);
        Ok(AcAdjointSetup {
            n,
            input: input.to_string(),
            gr_s,
            gc_s,
            gv_s,
            cr_s,
            cc_s,
            cv_s,
            b,
            pnames,
            prr,
            prc,
            prv,
            gsys,
            env,
        })
    }

    /// Evaluate one weight assignment of the cached tape: returns
    /// `(dPsi/dx, dPsi/dp)` for `Psi = lam^T (b - A v)` -- the real part with
    /// `imag = false`, the imaginary part with `imag = true`. Read-only on the
    /// symbolic context (no growth).
    fn ac_adjoint_pull(
        &self,
        setup: &AcAdjointSetup,
        lam: &[Complex64],
        v: &[Complex64],
        w: f64,
        imag: bool,
    ) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
        let cache = self.ac_vjp_tape_cache();
        let tapes = cache.lock().unwrap();
        let tape = tapes
            .get(&setup.input)
            .ok_or_else(|| ModelError::Numeric("ac adjoint tape missing".into()))?;
        let mut env = setup.env.clone();
        for (i, &sy) in tape.wb.iter().enumerate() {
            env.insert(sy, if imag { lam[i].im } else { lam[i].re });
        }
        for (t, &sy) in tape.wg.iter().enumerate() {
            let (i, j) = tape.g_pos[t];
            let pr = lam[i].re * v[j].re - lam[i].im * v[j].im;
            let pi = lam[i].re * v[j].im + lam[i].im * v[j].re;
            env.insert(sy, if imag { pi } else { pr });
        }
        for (t, &sy) in tape.wc.iter().enumerate() {
            let (i, j) = tape.c_pos[t];
            let pr = lam[i].re * v[j].re - lam[i].im * v[j].im;
            let pi = lam[i].re * v[j].im + lam[i].im * v[j].re;
            env.insert(sy, if imag { -pr * w } else { pi * w });
        }
        let arc = self.context_arc();
        let cg = arc.lock().unwrap();
        let u = rsdag::eval(&cg, &tape.dpsi_dx, &env);
        let expl = rsdag::eval(&cg, &tape.dpsi_dp, &env);
        Ok((u, expl))
    }

    /// Build (once per input) the weighted-adjoint tape: the Psi functional
    /// with weight SYMBOLS in place of the per-frequency `(lambda, v)`
    /// constants, differentiated w.r.t. every state and parameter. With
    /// weights as symbols the tape is frequency-independent, so every later
    /// pull is pure evaluation and the context stops growing per call (the
    /// old per-frequency `konst` embedding grew the arena on every VJP).
    ///
    /// Sign convention: `Psi = sum_i w_b[i]*b_i - sum_t w_g[t]*G_t + sum_t
    /// w_c[t]*C_t`; the assignments in [`Model::ac_adjoint_pull`] reproduce
    /// `Re` / `Im` of `lam^T (b - (G + jwC) v)`.
    fn ensure_ac_vjp_tape(&self, input: &str) -> Result<(), ModelError> {
        {
            let cache = self.ac_vjp_tape_cache();
            if cache.lock().unwrap().contains_key(input) {
                return Ok(());
            }
        }
        let arc = self.context_arc();
        let mut cg = arc.lock().unwrap();
        let c = &mut *cg;
        let dae = self.dae();
        let n = dae.dim();
        let ((gr, gc, ge), (cr, cc, ce)) = dae.jacobian_iq_coo(c);
        let isyms = self.inner.drive(c, input)?;
        let bsym: Vec<ExprId> = (dae.at_rest(c).0.iter())
            .map(|&r| {
                let d = crate::d_drive(c, r, &isyms);
                c.neg(d)
            })
            .collect();
        // Weight symbols, namespaced per input so tapes never collide.
        let mk = |c: &mut Graph, tag: &str, k: usize| -> (ExprId, SymbolId) {
            let e = c.sym(&format!("$acvjp${input}${tag}{k}"));
            match c.node(e) {
                Node::Symbol(sy) => (e, *sy),
                _ => unreachable!("fresh weight name is always a symbol"),
            }
        };
        let mut psi = c.zero();
        let mut wb = Vec::with_capacity(n);
        for (i, &bi) in bsym.iter().enumerate() {
            let (we, ws) = mk(c, "b", i);
            wb.push(ws);
            let term = c.mul(we, bi);
            psi = c.add(psi, term);
        }
        let mut wg = Vec::with_capacity(ge.len());
        let mut g_pos = Vec::with_capacity(ge.len());
        for (t, &gt) in ge.iter().enumerate() {
            let (we, ws) = mk(c, "g", t);
            wg.push(ws);
            g_pos.push((gr[t], gc[t]));
            let term = c.mul(we, gt);
            psi = c.sub(psi, term);
        }
        let mut wc = Vec::with_capacity(ce.len());
        let mut c_pos = Vec::with_capacity(ce.len());
        for (t, &ct) in ce.iter().enumerate() {
            let (we, ws) = mk(c, "c", t);
            wc.push(ws);
            c_pos.push((cr[t], cc[t]));
            let term = c.mul(we, ct);
            psi = c.add(psi, term);
        }
        let dpsi_dx: Vec<ExprId> = dae.x.iter().map(|&xm| differentiate(c, psi, xm)).collect();
        let pnames = self.cdc().param_names(c);
        let mut dpsi_dp = Vec::with_capacity(pnames.len());
        for name in &pnames {
            let e = c.sym(name);
            let ps = match c.node(e) {
                Node::Symbol(sy) => *sy,
                _ => return Err(ModelError::Numeric(format!("'{name}' is not a symbol"))),
            };
            dpsi_dp.push(differentiate(c, psi, ps));
        }
        drop(cg);
        let tape = AcVjpTape {
            wb,
            wg,
            wc,
            g_pos,
            c_pos,
            dpsi_dx,
            dpsi_dp,
        };
        self.ac_vjp_tape_cache()
            .lock()
            .unwrap()
            .insert(input.to_string(), tape);
        Ok(())
    }

    /// Exact analytic AC Hessian via the **second-order adjoint** -- no finite
    /// differences. The trick: AC analysis at one frequency is itself an algebraic
    /// system, so build the combined system `[I(x,p)=0; Re(Av-b)=0; Im(Av-b)=0]`
    /// in the unknowns `(x, v_re, v_im)`, then run the *existing* DC second-order
    /// adjoint (`hessian`) on it. The operating-point shift falls out for free
    /// because `x` is part of the combined solution. Returns, for the output node,
    /// `(v_re, v_im, grad_re[subset], grad_im[subset], H_re[subset^2], H_im[subset^2])`;
    /// the caller projects these onto the desired metric (mag/phase/real/imag).
    #[allow(clippy::type_complexity)]
    pub(crate) fn ac_hessian(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freq: f64,
        subset: Vec<String>,
    ) -> Result<(f64, f64, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>), ModelError> {
        self.inner.bound(&p)?;
        let _g = log::scope("sens/ac_hessian");
        let n = self.dae().dim();
        let w = 2.0 * PI * freq;
        // Numeric forward solve A v = b (sparse, KLU) to assemble the combined
        // solution y*.
        let (gr_s, gc_s, gv_s) = self.cdc().system_triplets_dc(&x, &p);
        let (cr_s, cc_s, cv_s) = self.cdc().jacobian_q_x_sparse(&x, &p, 0.0);
        let bin = self.jacobian_i_input(input, x.clone(), p.clone(), 0.0)?;
        let bvec: Vec<Complex64> = bin.iter().map(|&val| Complex64::new(-val, 0.0)).collect();
        let v = crate::sparse_ac::AcSystem::assemble(
            n,
            (&gr_s, &gc_s, &gv_s),
            (&cr_s, &cc_s, &cv_s),
            w,
        )
        .solve(&bvec)
        .ok_or_else(|| ModelError::Numeric("ac_hessian: singular A".to_string()))?;
        let pnames = self.cdc().param_names(&self.context_arc().lock().unwrap());

        let arc = self.context_arc();
        let mut cg = arc.lock().unwrap();
        let c = &mut *cg;
        let sym_id = |c: &mut Graph, name: &str| -> SymbolId {
            let e = c.sym(name);
            match c.node(e) {
                Node::Symbol(s) => *s,
                _ => unreachable!("sym() yields a Symbol node"),
            }
        };

        // Fresh AC phasor unknowns v_re_i, v_im_i.
        let (mut vre_e, mut vim_e) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut vre_s, mut vim_s) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let nr = format!("vre{i}");
            let ni = format!("vim{i}");
            vre_s.push(sym_id(c, &nr));
            vim_s.push(sym_id(c, &ni));
            vre_e.push(c.sym(&nr));
            vim_e.push(c.sym(&ni));
        }
        // Symbolic G, C (sparse) and b_i = -dI/d(input).
        let ((gr, gc, ge), (cr, cc, ce)) = self.dae().jacobian_iq_coo(c);
        let isyms = self.inner.drive(c, input)?;
        let bsym: Vec<ExprId> = (self.dae().at_rest(c).0.iter())
            .map(|&r| {
                let d = crate::d_drive(c, r, &isyms);
                c.neg(d)
            })
            .collect();
        let zero = c.zero();
        let wexpr = c.konst_f64(w);
        // AC rows: Re(Av-b) and Im(Av-b), A = G + jwC, b real.
        let mut ac_re = vec![zero; n];
        let mut ac_im = vec![zero; n];
        for k in 0..ge.len() {
            let (i, j) = (gr[k], gc[k]);
            let tr = c.mul(ge[k], vre_e[j]);
            ac_re[i] = c.add(ac_re[i], tr);
            let ti = c.mul(ge[k], vim_e[j]);
            ac_im[i] = c.add(ac_im[i], ti);
        }
        for k in 0..ce.len() {
            let (i, j) = (cr[k], cc[k]);
            let wc = c.mul(wexpr, ce[k]);
            let tr = c.mul(wc, vim_e[j]);
            ac_re[i] = c.sub(ac_re[i], tr); // - wC v_im
            let ti = c.mul(wc, vre_e[j]);
            ac_im[i] = c.add(ac_im[i], ti); // + wC v_re
        }
        for i in 0..n {
            ac_re[i] = c.sub(ac_re[i], bsym[i]); // - b
        }
        // Combined algebraic DAE in (x, v_re, v_im): the DC currents, then the
        // AC rows.
        let mut currents = self.dae().at_rest(c).0.clone();
        currents.extend(ac_re);
        currents.extend(ac_im);
        let mut xs = self.dae().x.clone();
        xs.extend(vre_s);
        xs.extend(vim_s);
        let mut names = self.dae().unknowns.clone();
        names.extend((0..n).map(|i| format!("vre{i}")));
        names.extend((0..n).map(|i| format!("vim{i}")));
        let zero = c.zero();
        let combined = CoreDae {
            charges: vec![zero; currents.len()],
            currents,
            assertions: self.dae().assertions.clone(),
            structure: self.dae().structure.clone(),
            aliases: self.dae().aliases.clone(),
            n_nodes: self.dae().n_nodes,
            param_defaults: self.dae().param_defaults.clone(),
            events: Vec::new(),
            delays: Vec::new(),
            kinds: self
                .dae()
                .kinds
                .iter()
                .copied()
                .cycle()
                .take(3 * n)
                .collect(),
            unknowns: names,
            x: xs,
            t: self.dae().t,
            companion: Vec::new(),
            observers: Default::default(),
            dc_seeds: Vec::new(),
            limits: Vec::new(),
            sources: Vec::new(),
            source_names: Vec::new(),
            labels: Default::default(),
            injection: Default::default(),
            rest: Default::default(),
        };
        let cdc_c = CompiledDc::new(c, &combined);
        cdc_c.ensure_param_jac(c, &combined);
        cdc_c.ensure_hessian(c, &combined);
        let cnames = cdc_c.param_names(c);

        // Combined solution y* = (x, v_re, v_im) and the parameter vector in the
        // combined system's parameter order.
        let mut ystar = x.clone();
        ystar.extend(v.iter().map(|z| z.re));
        ystar.extend(v.iter().map(|z| z.im));
        let valmap: HashMap<&str, f64> = pnames
            .iter()
            .map(|s| s.as_str())
            .zip(p.iter().copied())
            .collect();
        let p_c: Vec<f64> = cnames
            .iter()
            .map(|nm| {
                valmap
                    .get(nm.as_str())
                    .copied()
                    .or_else(|| self.values().get(nm).copied())
                    .unwrap_or(0.0)
            })
            .collect();
        let mut subset_cols = Vec::with_capacity(subset.len());
        for s in &subset {
            let col = cnames
                .iter()
                .position(|n| n == s)
                .ok_or_else(|| ModelError::Numeric(format!("unknown parameter '{s}'")))?;
            subset_cols.push(col);
        }
        let re_idx = n + out_idx;
        let im_idx = 2 * n + out_idx;
        let gr_all = cdc_c.sensitivity(re_idx, &ystar, &p_c, 0.0);
        let gi_all = cdc_c.sensitivity(im_idx, &ystar, &p_c, 0.0);
        if gr_all.is_empty() || gi_all.is_empty() {
            return Err(ModelError::Numeric(
                "ac_hessian: singular combined Jacobian".to_string(),
            ));
        }
        let g_re: Vec<f64> = subset_cols.iter().map(|&k| gr_all[k]).collect();
        let g_im: Vec<f64> = subset_cols.iter().map(|&k| gi_all[k]).collect();
        let h_re = cdc_c.hessian(re_idx, &subset_cols, &ystar, &p_c, 0.0);
        let h_im = cdc_c.hessian(im_idx, &subset_cols, &ystar, &p_c, 0.0);
        Ok((v[out_idx].re, v[out_idx].im, g_re, g_im, h_re, h_im))
    }
}

// --- the AC and S-parameter analyses at a point ---------------------------

/// Small-signal responses of outputs to a source (see [`Point::ac`]):
/// `h[[i, k]]` is `outputs[i]` per unit `input` at `freqs[k]`, linearized
/// at the operating point.
pub struct AcResponse {
    /// The point it was solved at (of the binding's structure), and its
    /// operating point.
    at: Point,
    x: Vec<f64>,
    pub input: String,
    pub outputs: Vec<String>,
    outs: Vec<usize>,
    pub freqs: Vec<f64>,
    pub h: Array2<Complex64>,
}

/// Sensitivities of AC responses: `grad[[i, k, j]]` is
/// `d outputs[i] / d params[j]` at `freqs[k]`.
#[derive(Clone, Debug)]
pub struct AcSensitivity {
    pub freqs: Vec<f64>,
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    pub param_values: Vec<f64>,
    /// `h[[i, k]]`, the responses.
    pub h: Array2<Complex64>,
    pub grad: Array3<Complex64>,
}

/// Hessians of AC responses: `h[[i, k, a, b]]` is
/// `d^2 outputs[i] / d params[a] d params[b]` at `freqs[k]` (the real
/// part's Hessian the real part, the imaginary part's the imaginary).
#[derive(Clone, Debug)]
pub struct AcHessian {
    pub freqs: Vec<f64>,
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    pub param_values: Vec<f64>,
    pub h: Array4<Complex64>,
}

/// Scattering parameters over the circuit's ports (see
/// [`Point::s_parameters`]): `s[[k, i, j]]` at `freqs[k]`, port `j` driven.
pub struct SParameters {
    at: Point,
    x: Vec<f64>,
    /// Per port its drive source, its node's unknown and its reference
    /// impedance.
    ports: Vec<(String, usize, f64)>,
    pub port_names: Vec<String>,
    pub z0: Vec<f64>,
    pub freqs: Vec<f64>,
    pub s: Array3<Complex64>,
}

/// Sensitivities of scattering parameters: `grad[[k, i, j, q]]` is
/// `d S_ij / d params[q]` at `freqs[k]`.
#[derive(Clone, Debug)]
pub struct SpSensitivity {
    pub freqs: Vec<f64>,
    pub params: Vec<String>,
    pub param_values: Vec<f64>,
    pub grad: Array4<Complex64>,
}

impl Point {
    /// The small-signal responses of `outputs` (nodes, unknowns, branch
    /// currents) to the source `input` at `freqs` (Hz), linearized at the
    /// operating point.
    pub fn ac(
        &self,
        input: &str,
        outputs: &[&str],
        freqs: &[f64],
    ) -> Result<AcResponse, ModelError> {
        if let Some((there, _)) = &self.other {
            return there.ac(input, outputs, freqs);
        }
        let x = self.operating_point()?.x;
        let outs = self.model.inner.outputs(outputs)?;
        let rows = self.model.ac_solve(input, &outs, &x, &self.p, freqs)?;
        let bad: Vec<bool> = rows
            .iter()
            .map(|r| r.iter().any(|h| h.re.is_nan()))
            .collect();
        super::warn_singular(&format!("AC from '{input}'"), freqs, &bad);
        Ok(AcResponse {
            at: self.clone(),
            x,
            input: input.to_string(),
            outputs: outputs.iter().map(|o| o.to_string()).collect(),
            h: Array2::from_shape_fn((outs.len(), freqs.len()), |(i, k)| rows[k][i]),
            outs,
            freqs: freqs.to_vec(),
        })
    }

    /// The scattering parameters over the circuit's ports (deck `P`
    /// elements, in order) at `freqs` (Hz): each port the Thevenin form it
    /// lowers to, an ideal drive source behind its `z0`, so that
    /// `S_ij = 2 sqrt(z0_j / z0_i) V_i - delta_ij` under unit drive of port
    /// `j`.
    pub fn s_parameters(&self, freqs: &[f64]) -> Result<SParameters, ModelError> {
        if let Some((there, _)) = &self.other {
            return there.s_parameters(freqs);
        }
        let inner = &self.model.inner;
        if inner.ports.is_empty() {
            return Err(ModelError::Numeric(
                "the circuit has no ports (`P` elements, e.g. `P1 in 0 Z0=50`)".into(),
            ));
        }
        let mut ports = Vec::with_capacity(inner.ports.len());
        for (src, node, z0) in &inner.ports {
            let i = inner
                .resolve(node)
                .ok_or_else(|| ModelError::UnknownRef(node.clone()))?;
            ports.push((src.clone(), i, *z0));
        }
        let x = self.operating_point()?.x;
        let flat = self
            .model
            .sp_response(&ports, x.clone(), self.p.clone(), freqs.to_vec())?;
        let n = ports.len();
        let s = Array3::from_shape_fn((flat.len(), n, n), |(k, i, j)| {
            Complex64::new(flat[k][i * n + j].0, flat[k][i * n + j].1)
        });
        Ok(SParameters {
            at: self.clone(),
            x,
            port_names: ports.iter().map(|p| p.0.clone()).collect(),
            z0: ports.iter().map(|p| p.2).collect(),
            ports,
            freqs: freqs.to_vec(),
            s,
        })
    }
}

impl AcResponse {
    /// The response of `output` at every frequency.
    pub fn of(&self, output: &str) -> Option<ArrayView1<'_, Complex64>> {
        let i = self.outputs.iter().position(|o| o == output)?;
        Some(self.h.row(i))
    }

    /// The magnitude of `output`'s response in dB.
    pub fn mag_db(&self, output: &str) -> Option<Vec<f64>> {
        Some(
            self.of(output)?
                .iter()
                .map(|h| 20.0 * h.norm().max(1e-30).log10())
                .collect(),
        )
    }

    /// The phase of `output`'s response in degrees.
    pub fn phase_deg(&self, output: &str) -> Option<Vec<f64>> {
        Some(
            self.of(output)?
                .iter()
                .map(|h| h.arg().to_degrees())
                .collect(),
        )
    }

    /// The derivatives of every response by the parameters under `wrt`
    /// (all for none) at every frequency, exact: one adjoint per output and
    /// frequency, the operating point's shift included.
    pub fn sensitivity(&self, wrt: &[&str]) -> Result<AcSensitivity, ModelError> {
        let model = &self.at.model;
        let cols = model.inner.columns(wrt)?;
        let names: Vec<String> = cols
            .iter()
            .map(|&c| model.inner.store.pnames[c].clone())
            .collect();
        let mut grad = Array3::zeros((self.outs.len(), self.freqs.len(), names.len()));
        for (i, &o) in self.outs.iter().enumerate() {
            let (all, rows) = model.ac_gradient_sweep(
                &self.input,
                o,
                self.x.clone(),
                self.at.p.clone(),
                &self.freqs,
            )?;
            for (j, n) in names.iter().enumerate() {
                let Some(a) = all.iter().position(|a| a == n) else {
                    continue;
                };
                for (k, (_, g)) in rows.iter().enumerate() {
                    grad[[i, k, j]] = Complex64::new(g[a].0, g[a].1);
                }
            }
        }
        Ok(AcSensitivity {
            freqs: self.freqs.clone(),
            outputs: self.outputs.clone(),
            param_values: cols.iter().map(|&c| self.at.p[c]).collect(),
            params: names,
            h: self.h.clone(),
            grad,
        })
    }

    /// The gradient by every parameter of a real scalar `L` of the
    /// responses, given its cotangents `cotangent[[i, k]] =
    /// dL/dRe h[[i, k]] + j dL/dIm h[[i, k]]`: one weighted adjoint per
    /// frequency, whatever the number of outputs and parameters.
    pub fn vjp(&self, cotangent: ArrayView2<'_, Complex64>) -> Result<Gradient, ModelError> {
        if cotangent.dim() != self.h.dim() {
            return Err(ModelError::Numeric(format!(
                "a cotangent per output and frequency: {:?} for {:?}",
                cotangent.dim(),
                self.h.dim()
            )));
        }
        let weights: Vec<Vec<(usize, f64, f64)>> = (0..self.freqs.len())
            .map(|k| {
                (self.outs.iter().enumerate())
                    .map(|(i, &o)| (o, cotangent[[i, k]].re, cotangent[[i, k]].im))
                    .collect()
            })
            .collect();
        let g = self.at.model.ac_vjp_sweep(
            &self.input,
            &weights,
            self.x.clone(),
            self.at.p.clone(),
            self.freqs.clone(),
        )?;
        Ok(Gradient::from_pairs(g))
    }

    /// The Hessians of every response by the parameters under `wrt` (all
    /// for none) at every frequency, exact (the second-order adjoint, the
    /// operating point's shift included).
    pub fn hessian(&self, wrt: &[&str]) -> Result<AcHessian, ModelError> {
        let model = &self.at.model;
        let cols = model.inner.columns(wrt)?;
        let names: Vec<String> = cols
            .iter()
            .map(|&c| model.inner.store.pnames[c].clone())
            .collect();
        let np = names.len();
        let mut h = Array4::zeros((self.outs.len(), self.freqs.len(), np, np));
        for (i, &o) in self.outs.iter().enumerate() {
            for (k, &f) in self.freqs.iter().enumerate() {
                let (_, _, _, _, hr, hi) = model.ac_hessian(
                    &self.input,
                    o,
                    self.x.clone(),
                    self.at.p.clone(),
                    f,
                    names.clone(),
                )?;
                for a in 0..np {
                    for b in 0..np {
                        h[[i, k, a, b]] = Complex64::new(hr[a][b], hi[a][b]);
                    }
                }
            }
        }
        Ok(AcHessian {
            freqs: self.freqs.clone(),
            outputs: self.outputs.clone(),
            param_values: cols.iter().map(|&c| self.at.p[c]).collect(),
            params: names,
            h,
        })
    }
}

impl SParameters {
    /// `S_ij` at every frequency.
    pub fn of(&self, i: usize, j: usize) -> Vec<Complex64> {
        self.s.slice(s![.., i, j]).to_vec()
    }

    /// The derivatives of every `S_ij` by the parameters under `wrt` (all
    /// for none), exact (an adjoint per port pair and frequency).
    pub fn sensitivity(&self, wrt: &[&str]) -> Result<SpSensitivity, ModelError> {
        let model = &self.at.model;
        let cols = model.inner.columns(wrt)?;
        let names: Vec<String> = cols
            .iter()
            .map(|&c| model.inner.store.pnames[c].clone())
            .collect();
        let n = self.ports.len();
        let mut grad = Array4::zeros((self.freqs.len(), n, n, names.len()));
        for (j, (src, _, zj)) in self.ports.iter().enumerate() {
            for (i, (_, oi, zi)) in self.ports.iter().enumerate() {
                let (all, rows) = model.ac_gradient_sweep(
                    src,
                    *oi,
                    self.x.clone(),
                    self.at.p.clone(),
                    &self.freqs,
                )?;
                let scale = 2.0 * (zj / zi).sqrt();
                for (k, (_, g)) in rows.iter().enumerate() {
                    for (q, name) in names.iter().enumerate() {
                        if let Some(a) = all.iter().position(|a| a == name) {
                            grad[[k, i, j, q]] = Complex64::new(g[a].0, g[a].1) * scale;
                        }
                    }
                }
            }
        }
        Ok(SpSensitivity {
            freqs: self.freqs.clone(),
            param_values: cols.iter().map(|&c| self.at.p[c]).collect(),
            params: names,
            grad,
        })
    }

    /// The gradient by every parameter of a real scalar `L` of the
    /// scattering parameters, given `cotangent[[k, i, j]] =
    /// dL/dRe S_ij + j dL/dIm S_ij` at `freqs[k]`.
    pub fn vjp(&self, cotangent: ArrayView3<'_, Complex64>) -> Result<Gradient, ModelError> {
        if cotangent.dim() != self.s.dim() {
            return Err(ModelError::Numeric(format!(
                "a cotangent per frequency and port pair: {:?} for {:?}",
                cotangent.dim(),
                self.s.dim()
            )));
        }
        let cot: Vec<Vec<(f64, f64)>> = (cotangent.outer_iter())
            .map(|m| m.iter().map(|c| (c.re, c.im)).collect())
            .collect();
        let g = self.at.model.sp_vjp(
            &self.ports,
            self.x.clone(),
            self.at.p.clone(),
            self.freqs.clone(),
            &cot,
        )?;
        Ok(Gradient::from_pairs(g))
    }
}
