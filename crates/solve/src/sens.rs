//! Exact sensitivity machinery on a compiled DAE: the lazily-built parameter /
//! history Jacobian tapes, first-order adjoint sensitivities, and the
//! second-order-adjoint Hessian over the compiled Lagrangian blocks.

use rsdag::{SymbolId, Tape};
use sane_core::constants::GMIN_DC;
use sane_core::log_stage;
use sane_core::Graph;
use sane_dae::{lagrangian_hessian, Dae};

use crate::{sparse, CompiledDc, StepEval};

/// Compiled `∂I/∂hist` triplets (see [`CompiledDc::ensure_hist_jac`]).
pub(crate) struct HistJac {
    tape: StepEval,
    rows: Vec<usize>,
    cols: Vec<usize>,
}

/// A sparse Jacobian's coordinates.
pub type Pattern = (Vec<usize>, Vec<usize>);

/// Compiled parameter Jacobians `dI/dp` and `dQ/dp` (one tape, `dI/dp`
/// first) and their sparse coordinates.
pub(crate) struct ParamJac {
    tape: StepEval,
    i_rc: Pattern,
    q_rc: Pattern,
}

/// The Lagrangian-Hessian blocks `L_xx`, `L_xp`, `L_pp` of
/// `L = λ·I + μ·Q` compiled into one tape over inputs `(x, p, t, λ, μ)`, plus
/// their sparse coordinates. Evaluated and contracted with the state
/// sensitivities to give the exact Hessian.
pub(crate) struct CompiledHessian {
    np: usize,
    nxx: usize,
    nxp: usize,
    xx_rc: Pattern,
    xp_rc: Pattern,
    pp_rc: Pattern,
    tape: StepEval,
}

impl CompiledDc {
    /// Build the history Jacobian `dI/dhist` tape on demand (idempotent):
    /// the frequency-domain coupling of a transport delay. Cheap (one entry
    /// per delay in practice) and only built for delay circuits.
    pub fn ensure_hist_jac(&self, ctx: &mut Graph, dae: &Dae) {
        if self.hjac.get().is_some() || dae.delays.is_empty() {
            return;
        }
        let (r, c, e) = dae.jacobian_hist_coo(ctx);
        let tape = crate::eval::step_eval(Tape::compile(ctx, &e, &self.base_inputs));
        let _ = self.hjac.set(HistJac {
            tape,
            rows: r,
            cols: c,
        });
    }

    /// `∂I/∂hist` at `(x, p)` as `(rows, delay indices, values)`; empty when
    /// the circuit has no delays or [`ensure_hist_jac`](Self::ensure_hist_jac)
    /// was not called.
    pub fn hist_jac_sparse(&self, x: &[f64], p: &[f64]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let Some(hj) = self.hjac.get() else {
            return (Vec::new(), Vec::new(), Vec::new());
        };
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        self.fill_inputs(x, p, 0.0, &mut inputs);
        hj.tape.eval(&inputs, &mut work, &mut out);
        (hj.rows.clone(), hj.cols.clone(), out)
    }

    /// `dI/d(input)` at `(x, p, t)`: the currents' derivative in the symbol
    /// `input` (an independent source's value, the excitation of the AC and
    /// pole-zero analyses). Its tape is compiled on the first query for
    /// `input` and kept; later queries only evaluate it.
    pub fn jacobian_i_input(
        &self,
        ctx: &mut Graph,
        dae: &Dae,
        input: SymbolId,
        x: &[f64],
        p: &[f64],
        t: f64,
    ) -> Vec<f64> {
        let tape = {
            let mut cache = self.input_jacs.lock().unwrap();
            cache
                .entry(input)
                .or_insert_with(|| {
                    let roots: Vec<_> = dae
                        .currents
                        .iter()
                        .map(|&r| rsdag::differentiate(ctx, r, input))
                        .collect();
                    std::sync::Arc::new(crate::eval::step_eval(Tape::compile(
                        ctx,
                        &roots,
                        &self.base_inputs,
                    )))
                })
                .clone()
        };
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        self.fill_inputs(x, p, t, &mut inputs);
        tape.eval(&inputs, &mut work, &mut out);
        out
    }

    /// Build the parameter Jacobian tape (`dI/dp` and `dQ/dp`) on demand
    /// (idempotent). Call before any sensitivity query; split out of `new` so
    /// it is paid only when a sensitivity is asked for, not on every extract.
    pub fn ensure_param_jac(&self, ctx: &mut Graph, dae: &Dae) {
        if self.pjac.get().is_some() {
            return;
        }
        let ((ir, ic, ie), (qr, qc, qe)) = log_stage!(
            "sens/param_jac_coo",
            dae.jacobian_p_iq_coo(ctx, &self.param_syms)
        );
        let roots: Vec<_> = ie.into_iter().chain(qe).collect();
        let tape = log_stage!(
            "sens/param_jac_tape",
            crate::eval::step_eval(Tape::compile(ctx, &roots, &self.base_inputs))
        );
        let _ = self.pjac.set(ParamJac {
            tape,
            i_rc: (ir, ic),
            q_rc: (qr, qc),
        });
    }

    /// Build the Lagrangian-Hessian tape on demand (idempotent). Call before
    /// [`hessian`](Self::hessian); it is split out of `new` so the heavy O(n^2)
    /// second-order stage is paid only when a second-order sensitivity is asked
    /// for, not on every extract.
    pub fn ensure_hessian(&self, ctx: &mut Graph, dae: &Dae) {
        // The Hessian contraction uses the state sensitivities, which need dI/dp.
        self.ensure_param_jac(ctx, dae);
        if self.chess.get().is_some() {
            return;
        }
        let hs = log_stage!(
            "sens/hessian_lagrangian",
            lagrangian_hessian(ctx, dae, &self.param_syms)
        );
        let mut h_input_syms = self.base_inputs.clone();
        h_input_syms.extend(hs.lambda.iter().chain(&hs.mu).copied());
        // Root order is fixed: xx | xp | pp.
        let h_roots: Vec<_> = (hs.xx.2.iter().chain(&hs.xp.2).chain(&hs.pp.2))
            .copied()
            .collect();
        let ch = CompiledHessian {
            np: self.param_syms.len(),
            nxx: hs.xx.2.len(),
            nxp: hs.xp.2.len(),
            xx_rc: (hs.xx.0, hs.xx.1),
            xp_rc: (hs.xp.0, hs.xp.1),
            pp_rc: (hs.pp.0, hs.pp.1),
            tape: log_stage!(
                "sens/hessian_tape",
                crate::eval::step_eval(Tape::compile(ctx, &h_roots, &h_input_syms))
            ),
        };
        let _ = self.chess.set(ch);
    }

    /// `dI/dp` and `dQ/dp` (rows x parameter columns) at `(x, p, t)`, as
    /// `(rows, cols, values)` each.
    #[allow(clippy::type_complexity)]
    pub fn jacobian_p_sparse(
        &self,
        x: &[f64],
        p: &[f64],
        t: f64,
    ) -> (
        (Vec<usize>, Vec<usize>, Vec<f64>),
        (Vec<usize>, Vec<usize>, Vec<f64>),
    ) {
        let pj = self.pjac.get().expect(
            "jacobian_p_sparse: parameter Jacobian not built -- call ensure_param_jac before any \
             sensitivity query. Returning empty triplets here silently zeros dI/dp, which makes \
             state_sensitivity return an all-zeros (not empty) vector that passes the is_empty / \
             len guards and drops the operating-point-shift term (see issue #38).",
        );
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        self.fill_inputs(x, p, t, &mut inputs);
        pj.tape.eval(&inputs, &mut work, &mut out);
        let q = out.split_off(pj.i_rc.0.len());
        (
            (pj.i_rc.0.clone(), pj.i_rc.1.clone(), out),
            (pj.q_rc.0.clone(), pj.q_rc.1.clone(), q),
        )
    }

    /// Exact first-order sensitivity `dy/dp` of the metric `y = x[metric]` (an
    /// unknown) w.r.t. every parameter, by the **adjoint method**: one transpose
    /// solve `G^T λ = e_metric`, then `dy/dp_k = -(dI/dp)^T λ`. Evaluated at
    /// the point `(x, p, t)` (use the DC operating point). Returns one value per
    /// parameter, in `param_names` order. Empty on a singular Jacobian.
    pub fn sensitivity(&self, metric: usize, x: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let n = self.n;
        // G at the point plus the baseline gmin shunt (same as the
        // operating-point solve); the adjoint runs as a transpose solve on the
        // shared factorization -- no separate transposed factor.
        let lu = match self.factor_fx(x, p, t) {
            Some(lu) => lu,
            None => return Vec::new(),
        };
        let mut e = vec![0.0; n];
        e[metric] = 1.0;
        let lambda = match lu.solve_transpose(&e) {
            Some(l) => l, // G^T λ = e_metric
            None => return Vec::new(),
        };

        let ((pr, pc, pv), _) = self.jacobian_p_sparse(x, p, t);
        let mut s = vec![0.0; self.param_syms.len()];
        for k in 0..pv.len() {
            s[pc[k]] -= pv[k] * lambda[pr[k]]; // dy/dp_k = -(dI/dp)^T λ
        }
        s
    }

    /// Factor `G + GMIN_DC*I` at the point, for the adjoint / forward
    /// sensitivity solves. Forward solves use [`KluSolver::solve`], adjoints
    /// [`KluSolver::solve_transpose`] on the same factorization.
    fn factor_fx(&self, x: &[f64], p: &[f64], t: f64) -> Option<sparse::TripletLu> {
        let n = self.n;
        let (mut jr, mut jc, mut jv) = self.jacobian_i_x_sparse(x, p, t);
        for i in 0..n {
            jr.push(i);
            jc.push(i);
            jv.push(GMIN_DC);
        }
        sparse::factor_triplets(n, &jr, &jc, &jv)
    }

    /// State sensitivity `s_k = dx/dp_k` solving `G s_k = -dI/dp_k` at the
    /// point, for parameter column `param_col`. Empty on a singular Jacobian.
    pub fn state_sensitivity(&self, param_col: usize, x: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let n = self.n;
        let lu = match self.factor_fx(x, p, t) {
            Some(lu) => lu,
            None => return Vec::new(),
        };
        let ((pr, pc, pv), _) = self.jacobian_p_sparse(x, p, t);
        let mut b = vec![0.0; n];
        for k in 0..pv.len() {
            if pc[k] == param_col {
                b[pr[k]] -= pv[k]; // rhs = -dI/dp_k
            }
        }
        lu.solve(&b).unwrap_or_default()
    }

    /// Sparsity patterns of the Lagrangian-Hessian blocks
    /// `(L_xx, L_xp, L_pp)`, or `None` if
    /// [`ensure_hessian`](Self::ensure_hessian) was not called. Fetch once;
    /// pair with [`hessian_block_values`](Self::hessian_block_values).
    pub fn hessian_block_pattern(&self) -> Option<[Pattern; 3]> {
        let ch = self.chess.get()?;
        Some([ch.xx_rc.clone(), ch.xp_rc.clone(), ch.pp_rc.clone()])
    }

    /// Values of the Lagrangian-Hessian blocks of `L = λ·I + μ·Q` at the
    /// point `(x, p, t)` for the multipliers `lambda` (of the currents) and
    /// `mu` (of the charges), as `[xx, xp, pp]` aligned with
    /// [`hessian_block_pattern`](Self::hessian_block_pattern). The tape is
    /// linear in the multipliers, so a periodic analysis can sample these
    /// blocks along its waveform, the charges weighted by the adjoint's time
    /// derivative. `None` if `ensure_hessian` was not called.
    pub fn hessian_block_values(
        &self,
        x: &[f64],
        p: &[f64],
        t: f64,
        lambda: &[f64],
        mu: &[f64],
    ) -> Option<[Vec<f64>; 3]> {
        let ch = self.chess.get()?;
        let mut inputs = Vec::new();
        self.fill_inputs(x, p, t, &mut inputs);
        inputs.extend_from_slice(lambda);
        inputs.extend_from_slice(mu);
        let (mut work, mut out) = (Vec::new(), Vec::new());
        ch.tape.eval(&inputs, &mut work, &mut out);
        let pp = out.split_off(ch.nxx + ch.nxp);
        let xp = out.split_off(ch.nxx);
        Some([out, xp, pp])
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

        // Factor G + gmin*I ONCE and reuse it for every solve (#48). The old path
        // called `state_sensitivity` per parameter, each running a full sparse
        // factorization (triplet build + ordering + numeric LU) of the identical
        // matrix -- K factorizations for K parameters, plus one more in the adjoint.
        // Here we factor once: the forward sensitivities G s_k = -dI/dp_k become a
        // single multi-column solve (KLU sweeps all |subset| RHS columns through the
        // same factors), and the adjoint G^T λ = e_metric reuses the very
        // same factorization via a transpose solve.
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
        // Forward: assemble the n x |subset| RHS = -dI/dp[:, subset]
        // (column-major for the batched multi-RHS solve), one solve.
        let ((pr, pc, pv), _) = self.jacobian_p_sparse(x, p, t);
        let mut rhs = vec![0.0; n * kk];
        for k in 0..pv.len() {
            let b = loc[pc[k]];
            if b != usize::MAX {
                rhs[b * n + pr[k]] -= pv[k]; // column b == subset position of parameter pc[k]
            }
        }
        let s_mat = match lu.solve_many(&rhs, kk) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let s: Vec<Vec<f64>> = s_mat.chunks(n.max(1)).map(<[f64]>::to_vec).collect();

        // The Hessian blocks at the operating point: the currents' multiplier
        // the adjoint, the charges' zero (nothing moves).
        let Some([xx, xp, pp]) = self.hessian_block_values(x, p, t, &lambda, &vec![0.0; n]) else {
            return Vec::new();
        };

        // Contract the sparse Lagrangian-Hessian blocks directly (#47): iterate the
        // (row, col, value) triplets and map parameter indices through `loc[]` into
        // the |subset| x |subset| result.
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
