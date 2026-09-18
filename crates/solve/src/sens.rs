//! Exact sensitivity machinery on a compiled DAE: the lazily-built parameter /
//! history Jacobian tapes, first-order adjoint sensitivities, and the
//! second-order-adjoint Hessian over the compiled Lagrangian blocks.

use std::sync::atomic::Ordering;

use rsdag::{Graph, Tape};
use sane_core::constants::GMIN_DC;
use sane_core::log_stage;
use sane_dae::{lagrangian_hessian, Dae};

use crate::{bundle, sparse, CompiledDc, StepEval, FACTOR_FX_CALLS};

/// Compiled `∂F/∂hist` triplets (see [`CompiledDc::ensure_hist_jac`]).
pub(crate) struct HistJac {
    tape: StepEval,
    rows: Vec<usize>,
    cols: Vec<usize>,
}

/// Compiled parameter Jacobian `dF/dp` and its sparse coordinates.
pub(crate) struct ParamJac {
    tape: StepEval,
    rows: Vec<usize>,
    cols: Vec<usize>,
}

/// The Lagrangian-Hessian blocks `L_xx`, `L_xp`, `L_pp` compiled into one tape
/// over inputs `(x, xdot, p, t, λ)`, plus their sparse coordinates. Evaluated
/// and contracted with the state sensitivities to give the exact Hessian.
pub(crate) struct CompiledHessian {
    np: usize,
    nxx: usize,
    nxp: usize,
    npp: usize,
    nxdxd: usize,
    nxxd: usize,
    xx_rc: (Vec<usize>, Vec<usize>),
    xp_rc: (Vec<usize>, Vec<usize>),
    pp_rc: (Vec<usize>, Vec<usize>),
    // Rate (`x'`) second-derivative blocks: rate-rate, state-rate, rate-param.
    // Zero contribution when the reactive elements are linear; populated for
    // nonlinear charge storage so a periodic analysis is exact.
    xdxd_rc: (Vec<usize>, Vec<usize>),
    xxd_rc: (Vec<usize>, Vec<usize>),
    xdp_rc: (Vec<usize>, Vec<usize>),
    tape: StepEval,
}

impl CompiledDc {
    /// Build the history Jacobian `dF/dhist` tape on demand (idempotent):
    /// the frequency-domain coupling of a transport delay. Cheap (one entry
    /// per delay in practice) and only built for delay circuits.
    pub fn ensure_hist_jac(&self, ctx: &mut Graph, dae: &Dae) {
        if self.hjac.get().is_some() || dae.delays.is_empty() {
            return;
        }
        let (r, c, e) = dae.jacobian_hist_coo(ctx);
        let tape = StepEval::new(Tape::compile(ctx, &e, &self.base_inputs));
        let _ = self.hjac.set(HistJac {
            tape,
            rows: r,
            cols: c,
        });
    }

    /// `∂F/∂hist` at `(x, p)` as `(rows, delay indices, values)`; empty when
    /// the circuit has no delays or [`ensure_hist_jac`](Self::ensure_hist_jac)
    /// was not called.
    pub fn hist_jac_sparse(&self, x: &[f64], p: &[f64]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let Some(hj) = self.hjac.get() else {
            return (Vec::new(), Vec::new(), Vec::new());
        };
        let xdot = vec![0.0; self.n];
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        self.fill_inputs(x, &xdot, p, 0.0, &mut inputs);
        hj.tape.eval(&inputs, &mut work, &mut out);
        (hj.rows.clone(), hj.cols.clone(), out)
    }

    /// Build the parameter Jacobian `dF/dp` tape on demand (idempotent). Call
    /// before any sensitivity query; split out of `new` so it is paid only when a
    /// sensitivity is asked for, not on every extract.
    pub fn ensure_param_jac(&self, ctx: &mut Graph, dae: &Dae) {
        if self.pjac.get().is_some() {
            return;
        }
        let (pr, pc, pe) = log_stage!(
            "sens/param_jac_coo",
            if dae.stamps.is_empty() {
                dae.jacobian_p_coo(ctx, &self.param_syms)
            } else {
                dae.jacobian_p_coo_templated(ctx, &self.param_syms)
            }
        );
        bundle::ensure_function_bodies(ctx, &pe, &self.param_syms);
        let tape = log_stage!(
            "sens/param_jac_tape",
            StepEval::new(Tape::compile(ctx, &pe, &self.base_inputs))
        );
        let _ = self.pjac.set(ParamJac {
            tape,
            rows: pr,
            cols: pc,
        });
    }

    /// Build the Lagrangian-Hessian tape on demand (idempotent). Call before
    /// [`hessian`](Self::hessian); it is split out of `new` so the heavy O(n^2)
    /// second-order stage is paid only when a second-order sensitivity is asked
    /// for, not on every extract.
    pub fn ensure_hessian(&self, ctx: &mut Graph, dae: &Dae) {
        // The Hessian contraction uses the state sensitivities, which need dF/dp.
        self.ensure_param_jac(ctx, dae);
        if self.chess.get().is_some() {
            return;
        }
        let hs = log_stage!(
            "sens/hessian_lagrangian",
            lagrangian_hessian(ctx, dae, &self.param_syms)
        );
        let mut h_input_syms = self.base_inputs.clone();
        h_input_syms.extend(hs.lambda.iter().copied());
        // Root order is fixed: xx | xp | pp | xdxd | xxd | xdp. The DC Hessian
        // reads only the first three blocks (x' = 0 there); a periodic analysis
        // reads all six.
        let mut h_roots = Vec::new();
        h_roots.extend(hs.xx.2.iter().copied());
        h_roots.extend(hs.xp.2.iter().copied());
        h_roots.extend(hs.pp.2.iter().copied());
        h_roots.extend(hs.xdxd.2.iter().copied());
        h_roots.extend(hs.xxd.2.iter().copied());
        h_roots.extend(hs.xdp.2.iter().copied());
        bundle::ensure_function_bodies(ctx, &h_roots, &self.param_syms);
        let ch = CompiledHessian {
            np: self.param_syms.len(),
            nxx: hs.xx.2.len(),
            nxp: hs.xp.2.len(),
            npp: hs.pp.2.len(),
            nxdxd: hs.xdxd.2.len(),
            nxxd: hs.xxd.2.len(),
            xx_rc: (hs.xx.0, hs.xx.1),
            xp_rc: (hs.xp.0, hs.xp.1),
            pp_rc: (hs.pp.0, hs.pp.1),
            xdxd_rc: (hs.xdxd.0, hs.xdxd.1),
            xxd_rc: (hs.xxd.0, hs.xxd.1),
            xdp_rc: (hs.xdp.0, hs.xdp.1),
            tape: log_stage!(
                "sens/hessian_tape",
                StepEval::new(Tape::compile(ctx, &h_roots, &h_input_syms))
            ),
        };
        let _ = self.chess.set(ch);
    }

    /// Sparse `dF/dp` (residual rows x parameter columns) as `(rows, cols, values)`.
    pub fn jacobian_p_sparse(
        &self,
        x: &[f64],
        xdot: &[f64],
        p: &[f64],
        t: f64,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let pj = self.pjac.get().expect(
            "jacobian_p_sparse: parameter Jacobian not built -- call ensure_param_jac before any \
             sensitivity query. Returning empty triplets here silently zeros dF/dp, which makes \
             state_sensitivity return an all-zeros (not empty) vector that passes the is_empty / \
             len guards and drops the operating-point-shift term (see issue #38).",
        );
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        self.fill_inputs(x, xdot, p, t, &mut inputs);
        pj.tape.eval(&inputs, &mut work, &mut out);
        (pj.rows.clone(), pj.cols.clone(), out)
    }

    /// Exact first-order sensitivity `dy/dp` of the metric `y = x[metric]` (an
    /// unknown) w.r.t. every parameter, by the **adjoint method**: one transpose
    /// solve `(dF/dx)^T λ = e_metric`, then `dy/dp_k = -(dF/dp)^T λ`. Evaluated at
    /// the point `(x, p, t)` (use the DC operating point). Returns one value per
    /// parameter, in `param_names` order. Empty on a singular Jacobian.
    pub fn sensitivity(&self, metric: usize, x: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let n = self.n;
        let zeros = vec![0.0; n];
        // dF/dx at the point plus the baseline gmin shunt (same as the
        // operating-point solve); the adjoint runs as a transpose solve on the
        // shared factorization -- no separate transposed factor.
        let lu = match self.factor_fx(x, p, t) {
            Some(lu) => lu,
            None => return Vec::new(),
        };
        let mut e = vec![0.0; n];
        e[metric] = 1.0;
        let lambda = match lu.solve_transpose(&e) {
            Some(l) => l, // (dF/dx)^T λ = e_metric
            None => return Vec::new(),
        };

        let (pr, pc, pv) = self.jacobian_p_sparse(x, &zeros, p, t);
        let mut s = vec![0.0; self.param_syms.len()];
        for k in 0..pv.len() {
            s[pc[k]] -= pv[k] * lambda[pr[k]]; // dy/dp_k = -(dF/dp)^T λ
        }
        s
    }

    /// Factor `dF/dx + GMIN_DC*I` at the point, for the adjoint / forward
    /// sensitivity solves. Forward solves use [`KluSolver::solve`], adjoints
    /// [`KluSolver::solve_transpose`] on the same factorization.
    fn factor_fx(&self, x: &[f64], p: &[f64], t: f64) -> Option<sparse::TripletLu> {
        // Instrumentation for the "factor once" contract (#48): a full factorization
        // (triplet build + ordering + numeric LU) is the expensive step the
        // batched Hessian solve exists to amortise. Cheap relaxed counter, read by
        // `factor_fx_calls` for the count-based verification.
        FACTOR_FX_CALLS.fetch_add(1, Ordering::Relaxed);
        let n = self.n;
        let zeros = vec![0.0; n];
        let (mut jr, mut jc, mut jv) = self.jacobian_x_sparse(x, &zeros, p, t);
        for i in 0..n {
            jr.push(i);
            jc.push(i);
            jv.push(GMIN_DC);
        }
        sparse::factor_triplets_both(n, &jr, &jc, &jv)
    }

    /// State sensitivity `s_k = dx/dp_k` solving `(dF/dx) s_k = -dF/dp_k` at the
    /// point, for parameter column `param_col`. Empty on a singular Jacobian.
    pub fn state_sensitivity(&self, param_col: usize, x: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let n = self.n;
        let zeros = vec![0.0; n];
        let lu = match self.factor_fx(x, p, t) {
            Some(lu) => lu,
            None => return Vec::new(),
        };
        let (pr, pc, pv) = self.jacobian_p_sparse(x, &zeros, p, t);
        let mut b = vec![0.0; n];
        for k in 0..pv.len() {
            if pc[k] == param_col {
                b[pr[k]] -= pv[k]; // rhs = -dF/dp_k
            }
        }
        lu.solve(&b).unwrap_or_default()
    }

    /// Sparsity patterns of the six Lagrangian-Hessian blocks
    /// `(L_xx, L_xp, L_pp, L_x'x', L_x x', L_x'p)` as `((rows, cols), ...)`, or
    /// `None` if [`ensure_hessian`](Self::ensure_hessian) was not called. Fetch
    /// once; pair with [`hessian_block_values`]. The last three are the rate
    /// (`x'`) blocks (empty for linear reactances).
    #[allow(clippy::type_complexity)]
    pub fn hessian_block_pattern(&self) -> Option<[(Vec<usize>, Vec<usize>); 6]> {
        let ch = self.chess.get()?;
        Some([
            ch.xx_rc.clone(),
            ch.xp_rc.clone(),
            ch.pp_rc.clone(),
            ch.xdxd_rc.clone(),
            ch.xxd_rc.clone(),
            ch.xdp_rc.clone(),
        ])
    }

    /// Values of the six Lagrangian-Hessian blocks `L = sum_i lambda_i * F_i` at
    /// the point `(x, xdot, p, t)` for the multiplier `lambda`, as value vectors
    /// `[xx, xp, pp, x'x', x x', x'p]` aligned with [`hessian_block_pattern`].
    /// Unlike [`hessian`](Self::hessian) this evaluates at an arbitrary `xdot`
    /// and caller-supplied `lambda` (the tape is linear in `lambda`), so a
    /// periodic analysis can sample these blocks along its waveform -- including
    /// the rate blocks that make a nonlinear-charge-storage Hessian exact.
    /// `None` if `ensure_hessian` was not called.
    pub fn hessian_block_values(
        &self,
        x: &[f64],
        xdot: &[f64],
        p: &[f64],
        t: f64,
        lambda: &[f64],
    ) -> Option<[Vec<f64>; 6]> {
        let ch = self.chess.get()?;
        let mut inputs = Vec::new();
        self.fill_inputs(x, xdot, p, t, &mut inputs);
        inputs.extend_from_slice(lambda);
        let (mut work, mut out) = (Vec::new(), Vec::new());
        ch.tape.eval(&inputs, &mut work, &mut out);
        // Fixed root layout: xx | xp | pp | xdxd | xxd | xdp (the last fills the rest).
        let o1 = ch.nxx;
        let o2 = o1 + ch.nxp;
        let o3 = o2 + ch.npp;
        let o4 = o3 + ch.nxdxd;
        let o5 = o4 + ch.nxxd;
        Some([
            out[..o1].to_vec(),
            out[o1..o2].to_vec(),
            out[o2..o3].to_vec(),
            out[o3..o4].to_vec(),
            out[o4..o5].to_vec(),
            out[o5..].to_vec(),
        ])
    }

    /// Exact second-order-adjoint Hessian of the metric `y = x[metric]` w.r.t.
    /// the parameter columns `subset` (indices into `param_names`), at the
    /// operating point `(x, p, t)`. A pure leaf-value re-evaluation: the adjoint
    /// `λ` and state sensitivities `s_k` are solved, the compiled Lagrangian-
    /// Hessian tape is evaluated once, and the dense `|subset| x |subset|`
    /// Hessian is the numeric contraction
    /// `H_ab = -(s_aᵀ L_xx s_b + s_aᵀ L_xp[:,b] + s_bᵀ L_xp[:,a] + L_pp[a][b])`.
    /// Empty on a singular Jacobian.
    pub fn hessian(
        &self,
        metric: usize,
        subset: &[usize],
        x: &[f64],
        p: &[f64],
        t: f64,
    ) -> Vec<Vec<f64>> {
        let n = self.n;
        let ch = match self.chess.get() {
            Some(c) => c,
            None => return Vec::new(), // ensure_hessian was not called
        };
        let kk = subset.len();
        let np = ch.np;
        // Parameter index -> its position in `subset` (for building the forward RHS
        // and the L_xp column / L_pp lookups); MAX marks a parameter outside `subset`.
        let mut loc = vec![usize::MAX; np];
        for (a, &j) in subset.iter().enumerate() {
            loc[j] = a;
        }

        // Factor dF/dx + gmin*I ONCE and reuse it for every solve (#48). The old path
        // called `state_sensitivity` per parameter, each running a full sparse
        // factorization (triplet build + ordering + numeric LU) of the identical
        // matrix -- K factorizations for K parameters, plus one more in the adjoint.
        // Here we factor once: the forward sensitivities J s_k = -dF/dp_k become a
        // single multi-column solve (KLU sweeps all |subset| RHS columns through the
        // same factors), and the adjoint (dF/dx)^T λ = e_metric reuses the very
        // same factorization via a transpose solve.
        let zeros = vec![0.0; n];
        let lu = match self.factor_fx(x, p, t) {
            Some(lu) => lu,
            None => return Vec::new(), // singular Jacobian
        };
        // Adjoint via transpose-solve on the shared factorization.
        let mut e = vec![0.0; n];
        e[metric] = 1.0;
        let lambda = match lu.solve_transpose(&e) {
            Some(l) => l,
            None => return Vec::new(),
        };
        // Forward: assemble the n x |subset| RHS = -dF/dp[:, subset] (row-major
        // for KLU's batched multi-RHS solve), one solve.
        let (pr, pc, pv) = self.jacobian_p_sparse(x, &zeros, p, t);
        let mut rhs = vec![0.0; n * kk];
        for k in 0..pv.len() {
            let b = loc[pc[k]];
            if b != usize::MAX {
                rhs[pr[k] * kk + b] -= pv[k]; // column b == subset position of parameter pc[k]
            }
        }
        let s_mat = match lu.solve_many(&rhs, kk) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let s: Vec<Vec<f64>> = (0..kk)
            .map(|a| (0..n).map(|i| s_mat[i * kk + a]).collect())
            .collect();

        // Evaluate the Hessian-block tape: base inputs (xdot = 0) ++ multipliers.
        let mut inputs = Vec::new();
        self.fill_inputs(x, &zeros, p, t, &mut inputs);
        inputs.extend_from_slice(&lambda);
        let (mut work, mut out) = (Vec::new(), Vec::new());
        ch.tape.eval(&inputs, &mut work, &mut out);

        // Contract the sparse Lagrangian-Hessian blocks directly (#47): iterate the
        // (row, col, value) triplets and map parameter indices through `loc[]` into
        // the |subset| x |subset| result. Block value slices are in the fixed root
        // layout xx | xp | pp | (rate blocks).
        let xx = &out[..ch.nxx];
        let xp = &out[ch.nxx..ch.nxx + ch.nxp];
        let pp = &out[ch.nxx + ch.nxp..ch.nxx + ch.nxp + ch.npp];
        // L_pp restricted to subset x subset (sparse scatter through loc[]).
        let mut lpp = vec![vec![0.0; kk]; kk];
        for k in 0..ch.pp_rc.0.len() {
            let (r, c) = (ch.pp_rc.0[k], ch.pp_rc.1[k]);
            if loc[r] != usize::MAX && loc[c] != usize::MAX {
                lpp[loc[r]][loc[c]] = pp[k];
            }
        }
        let mut h = vec![vec![0.0; kk]; kk];
        for a in 0..kk {
            for b in a..kk {
                let (ja, jb) = (subset[a], subset[b]);
                let (sa, sb) = (&s[a], &s[b]);
                // s_aᵀ L_xx s_b over the sparse xx triplets (row m, col l).
                let mut t1 = 0.0;
                for k in 0..ch.nxx {
                    t1 += xx[k] * sa[ch.xx_rc.0[k]] * sb[ch.xx_rc.1[k]];
                }
                // s_aᵀ L_xp[:,jb] + s_bᵀ L_xp[:,ja] over the sparse xp triplets.
                let mut t2 = 0.0;
                for k in 0..ch.nxp {
                    let (r, col) = (ch.xp_rc.0[k], ch.xp_rc.1[k]);
                    let v = xp[k];
                    if col == jb {
                        t2 += v * sa[r];
                    }
                    if col == ja {
                        t2 += v * sb[r];
                    }
                }
                let val = -(t1 + t2 + lpp[a][b]);
                h[a][b] = val;
                h[b][a] = val;
            }
        }
        h
    }
}
