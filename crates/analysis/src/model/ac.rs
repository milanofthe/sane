//! AC and S-parameter analyses on [`Model`], plus their exact adjoint
//! machinery: responses, per-frequency sensitivities, all-parameter gradients
//! (VJPs over a cached weighted-adjoint tape) and the AC Hessian.

use std::collections::HashMap;
use std::f64::consts::PI;

use num_complex::Complex64;
use rsdag::{differentiate, eval, ExprId, Graph, Node, SymbolId};
use sane_core::log;
use sane_dae::{ac_param_derivatives, small_signal_transfer, Dae as CoreDae};
use sane_solve::CompiledDc;

use crate::model::{Model, ModelError};
use crate::op_env;

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
    /// Exact total derivatives of the small-signal matrices w.r.t. `param`,
    /// including the operating-point shift, via AD: returns `(dG, dC, dB)`.
    pub fn ac_derivatives(
        &self,
        input: &str,
        param: &str,
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>), ModelError> {
        let arc = self.context_arc();
        let mut c = arc.lock().unwrap();
        let dae = self.dae();
        let pnames = self.cdc().param_names(&c);
        let col = pnames
            .iter()
            .position(|n| n == param)
            .ok_or_else(|| ModelError::Numeric(format!("unknown parameter '{param}'")))?;
        // Build dF/dp before the state-sensitivity solve; without it the
        // operating-point-shift term dx/dp is silently zero (issue #38).
        self.cdc().ensure_param_jac(&mut c, dae);
        let s = self.cdc().state_sensitivity(col, &x, &p, t);
        if s.is_empty() {
            return Err(ModelError::Numeric(
                "ac_derivatives: singular Jacobian".to_string(),
            ));
        }
        let sym_of = |ctx: &mut Graph, name: &str| -> Option<SymbolId> {
            let e = ctx.sym(name);
            match ctx.node(e) {
                Node::Symbol(sy) => Some(*sy),
                _ => None,
            }
        };
        let input_sym = sym_of(&mut c, input)
            .ok_or_else(|| ModelError::Numeric(format!("'{input}' is not a symbol")))?;
        let p_sym = sym_of(&mut c, param)
            .ok_or_else(|| ModelError::Numeric(format!("'{param}' is not a symbol")))?;
        let env = op_env(&mut c, dae, &pnames, &x, &[], &p, t);
        Ok(ac_param_derivatives(
            &mut c, dae, input_sym, p_sym, &s, &env,
        ))
    }

    /// Small-signal AC transfer `H(j2*pi*f)` from source parameter `input` to
    /// unknown `output`, evaluated at each frequency in `freqs_hz`. `values`
    /// binds parameters (and, for nonlinear circuits, operating-point unknowns);
    /// unbound symbols default to 0. Returns `(re, im)` pairs, or `None` if the
    /// output is unknown.
    pub fn ac_transfer(
        &self,
        input: &str,
        output: &str,
        values: HashMap<String, f64>,
        freqs_hz: Vec<f64>,
    ) -> Option<Vec<(f64, f64)>> {
        let arc = self.context_arc();
        let mut cg = arc.lock().unwrap();
        let c = &mut *cg;
        let h = small_signal_transfer(c, self.dae(), input, output)?;
        let syms: Vec<SymbolId> = c.free_symbols(h).into_iter().collect();
        let mut out = Vec::with_capacity(freqs_hz.len());
        for f in freqs_hz {
            let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
            for &s in &syms {
                let name = c.symbol_name(s);
                let v = if name == "s" {
                    Complex64::new(0.0, 2.0 * PI * f)
                } else {
                    Complex64::new(values.get(name).copied().unwrap_or(0.0), 0.0)
                };
                env.insert(s, v);
            }
            let z = eval(c, &[h], &env)[0];
            out.push((z.re, z.im));
        }
        Some(out)
    }

    /// Small-signal AC response `H(j2*pi*f)` from source `input` to the unknown at
    /// `out_idx`, at the operating point `x`, solved numerically per frequency in
    /// Rust: `e_out^T (G + jwC)^{-1} (-dF/d(input))`. Returns `(re, im)` per
    /// frequency.
    pub fn ac_response(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freqs_hz: Vec<f64>,
    ) -> Result<Vec<(f64, f64)>, ModelError> {
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
        let z = vec![0.0; n];
        // Sparse A = G(+gmin) + jwC over the fixed pattern: symbolic analysis
        // once, numeric refactor per frequency (never densified).
        let (gr, gc, gv) = self.cdc().system_triplets_dc(&x, &p);
        let (cr, cc, cv) = self.cdc().jacobian_xdot_sparse(&x, &z, &p, 0.0);
        let bin = self.input_jacobian(input, x.clone(), z.clone(), p.clone(), 0.0)?;
        let b: Vec<Complex64> = bin.iter().map(|v| Complex64::new(-v, 0.0)).collect();
        let (dr, dc, dv, dtau) = self.delay_ac_entries(&x, &p);
        let sym = crate::sparse_ac::SymbolicAc::new_with_delays(
            n,
            (&gr, &gc, &gv),
            (&cr, &cc, &cv),
            (&dr, &dc, &dv, &dtau),
            false,
        );
        let mut fac = sym.as_ref().map(|s| s.solver());
        let mut out = Vec::with_capacity(freqs_hz.len());
        for f in freqs_hz {
            let w = 2.0 * PI * f;
            // A singular A = G + jwC (floating subnet, ideal VCVS/inductor loop,
            // garbage DC point) has no small-signal solution at this frequency.
            // Surface it as NaN -- a catchable signal the Python layer turns into a
            // warning -- instead of silently coercing to 0.0, which reads as a flat
            // ~-600 dB response and masquerades as a size limit (issue #39).
            let h = fac
                .as_mut()
                .and_then(|fa| fa.solve(w, &b))
                .map(|xx| xx[out_idx])
                .unwrap_or(Complex64::new(f64::NAN, f64::NAN));
            out.push((h.re, h.im));
        }
        let singular = out.iter().filter(|(re, _)| re.is_nan()).count();
        task.finish(if singular > 0 {
            format!("points: {}, singular: {singular}", out.len())
        } else {
            format!("points: {}", out.len())
        });
        Ok(out)
    }

    /// Exact AC-transfer sensitivity `dH/dp(jw)` from `input` w.r.t. `param` at
    /// the operating point `x`, computed natively (the complex solves run in
    /// Rust). Returns `(re, im)` per frequency.
    pub fn ac_sensitivity(
        &self,
        input: &str,
        param: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freqs_hz: Vec<f64>,
    ) -> Result<Vec<(f64, f64)>, ModelError> {
        self.ensure_no_delays("ac_sensitivity")?;
        let n = self.dae().dim();
        let z = vec![0.0; n];
        let g = self.cdc().system_matrix_dc(&x, &z, &p, 0.0);
        let c = self.cdc().jacobian_xdot(&x, &z, &p, 0.0);
        // B = -dF/d(input); dB from the total small-signal derivatives.
        let bin = self.input_jacobian(input, x.clone(), z.clone(), p.clone(), 0.0)?;
        let b: Vec<f64> = bin.iter().map(|v| -v).collect();
        let (dg, dc, db) = self.ac_derivatives(input, param, x.clone(), p.clone(), 0.0)?;
        Ok(
            crate::ac_response_sensitivity(&g, &c, &b, &dg, &dc, &db, out_idx, &freqs_hz)
                .into_iter()
                .map(|r| (r[0], r[1]))
                .collect(),
        )
    }

    /// Exact AC sensitivity `dH/dp` of the transfer `H(jw) = v[out_idx]` w.r.t.
    /// **every** parameter at one frequency, by the adjoint. With the forward
    /// state `A v = b` and adjoint `A^T lambda = e_out` (`A = G + jwC`), the
    /// scalar functional `Psi(x,p) = lambda^T b - lambda^T A v` (lambda, v frozen)
    /// has total derivative `dH/dp_k = dPsi/dp_k - mu^T dF/dp_k`, where
    /// `G_dc^T mu = grad_x Psi` is one DC adjoint (the operating-point shift). All
    /// pieces are exact first-order autodiff of one scalar -- no finite
    /// differences, two complex solves plus one real solve total, all parameters
    /// at once. Returns `(name, dHre/dp, dHim/dp)`.
    pub fn ac_gradient(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freq: f64,
    ) -> Result<Vec<(String, f64, f64)>, ModelError> {
        self.ensure_no_delays("ac_gradient")?;
        let _g = log::scope("sens/ac_gradient");
        let setup = self.ac_adjoint_setup(input, &x, &p)?;
        let w = 2.0 * PI * freq;
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
        let mu = setup
            .gsys
            .solve_transpose(&urhs)
            .ok_or_else(|| ModelError::Numeric("ac_gradient: singular DC Jacobian".to_string()))?;
        let np = setup.pnames.len();
        let mut sh_re = vec![0.0; np];
        let mut sh_im = vec![0.0; np];
        for t in 0..setup.prv.len() {
            let (i, k) = (setup.prr[t], setup.prc[t]);
            sh_re[k] -= mu[i].re * setup.prv[t];
            sh_im[k] -= mu[i].im * setup.prv[t];
        }
        Ok(setup
            .pnames
            .iter()
            .enumerate()
            .map(|(k, name)| (name.clone(), expl_re[k] + sh_re[k], expl_im[k] + sh_im[k]))
            .collect())
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
    pub fn ac_vjp_sweep(
        &self,
        input: &str,
        weights: &[Vec<(usize, f64, f64)>],
        x: Vec<f64>,
        p: Vec<f64>,
        freqs: Vec<f64>,
    ) -> Result<Vec<(String, f64)>, ModelError> {
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
    pub fn sp_response(
        &self,
        ports: &[(String, usize, f64)],
        x: Vec<f64>,
        p: Vec<f64>,
        freqs_hz: Vec<f64>,
    ) -> Result<Vec<Vec<(f64, f64)>>, ModelError> {
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
    pub fn sp_vjp(
        &self,
        ports: &[(String, usize, f64)],
        x: Vec<f64>,
        p: Vec<f64>,
        freqs_hz: Vec<f64>,
        cot: &[Vec<(f64, f64)>],
    ) -> Result<Vec<(String, f64)>, ModelError> {
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
        let z = vec![0.0; n];
        let (gr_s, gc_s, gv_s) = self.cdc().system_triplets_dc(x, p);
        let (cr_s, cc_s, cv_s) = self.cdc().jacobian_xdot_sparse(x, &z, p, 0.0);
        let bin = self.input_jacobian(input, x.to_vec(), z.clone(), p.to_vec(), 0.0)?;
        let b: Vec<Complex64> = bin.iter().map(|&val| Complex64::new(-val, 0.0)).collect();
        self.cdc()
            .ensure_param_jac(&mut self.context_arc().lock().unwrap(), self.dae());
        let pnames = self.cdc().param_names(&self.context_arc().lock().unwrap());
        let (prr, prc, prv) = self.cdc().jacobian_p_sparse(x, &z, p, 0.0);
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
        for sy in self.dae().xdot.iter().flatten() {
            env.insert(*sy, 0.0);
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
        let (gr, gc, ge) = dae.jacobian_x_coo(c);
        let (cr, cc, ce) = dae.jacobian_xdot_coo(c);
        let ie = c.sym(input);
        let isym = match c.node(ie) {
            Node::Symbol(sy) => *sy,
            _ => return Err(ModelError::Numeric(format!("'{input}' is not a symbol"))),
        };
        let bsym: Vec<ExprId> = dae
            .residuals
            .iter()
            .map(|&r| {
                let d = differentiate(c, r, isym);
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
    /// system, so build the combined system `[F(x,0,p)=0; Re(Av-b)=0; Im(Av-b)=0]`
    /// in the unknowns `(x, v_re, v_im)`, then run the *existing* DC second-order
    /// adjoint (`hessian`) on it. The operating-point shift falls out for free
    /// because `x` is part of the combined solution. Returns, for the output node,
    /// `(v_re, v_im, grad_re[subset], grad_im[subset], H_re[subset^2], H_im[subset^2])`;
    /// the caller projects these onto the desired metric (mag/phase/real/imag).
    #[allow(clippy::type_complexity)]
    pub fn ac_hessian(
        &self,
        input: &str,
        out_idx: usize,
        x: Vec<f64>,
        p: Vec<f64>,
        freq: f64,
        subset: Vec<String>,
    ) -> Result<(f64, f64, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>), ModelError> {
        let _g = log::scope("sens/ac_hessian");
        let n = self.dae().dim();
        let z = vec![0.0; n];
        let w = 2.0 * PI * freq;
        // Numeric forward solve A v = b (sparse, KLU) to assemble the combined
        // solution y*.
        let (gr_s, gc_s, gv_s) = self.cdc().system_triplets_dc(&x, &p);
        let (cr_s, cc_s, cv_s) = self.cdc().jacobian_xdot_sparse(&x, &z, &p, 0.0);
        let bin = self.input_jacobian(input, x.clone(), z.clone(), p.clone(), 0.0)?;
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
        // Symbolic G, C (sparse) and b_i = -dF/d(input).
        let (gr, gc, ge) = self.dae().jacobian_x_coo(c);
        let (cr, cc, ce) = self.dae().jacobian_xdot_coo(c);
        let isym = sym_id(c, input);
        let bsym: Vec<ExprId> = self
            .dae()
            .residuals
            .iter()
            .map(|&r| {
                let d = differentiate(c, r, isym);
                c.neg(d)
            })
            .collect();
        let zero = c.zero();
        let wexpr = c.konst_f64(w);
        // DC residuals F(x, 0, p): every derivative symbol read as zero.
        let at_rest: rustc_hash::FxHashMap<SymbolId, ExprId> = self
            .dae()
            .xdot
            .iter()
            .flatten()
            .map(|&s| (s, zero))
            .collect();
        let dc_res: Vec<ExprId> = rsdag::substitute(c, &self.dae().residuals, &at_rest);
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
        // Combined algebraic DAE in (x, v_re, v_im).
        let mut residuals = dc_res;
        residuals.extend(ac_re);
        residuals.extend(ac_im);
        let mut xs = self.dae().x.clone();
        xs.extend(vre_s);
        xs.extend(vim_s);
        let mut names = self.dae().unknowns.clone();
        names.extend((0..n).map(|i| format!("vre{i}")));
        names.extend((0..n).map(|i| format!("vim{i}")));
        let combined = CoreDae {
            n_nodes: self.dae().n_nodes,
            param_defaults: self.dae().param_defaults.clone(),
            events: Vec::new(),
            delays: Vec::new(),
            residuals,
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
            xdot: vec![None; 3 * n],
            t: self.dae().t,
            stamps: Vec::new(),
            companion: Vec::new(),
            noise_sources: Vec::new(),
            op_vars: Vec::new(),
            dc_seeds: Vec::new(),
            limits: Vec::new(),
            sources: Vec::new(),
            source_names: Vec::new(),
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
