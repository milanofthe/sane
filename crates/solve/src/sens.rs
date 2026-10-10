//! Exact sensitivity machinery on a compiled DAE: the lazily-built parameter /
//! history Jacobian tapes, first-order adjoint sensitivities, and the
//! second-order-adjoint Hessian over the compiled Lagrangian blocks.

use std::collections::HashMap;

use rsdag::{ExprId, SymbolId, Tape};
use sane_core::constants::GMIN_DC;
use sane_core::log_stage;
use sane_core::Graph;
use sane_dae::{frozen, lagrangian_hessian, param_column, Dae};

use crate::{sparse, CompiledDc, PrologToken, StepEval};

/// Compiled `∂I/∂hist` triplets (see [`CompiledDc::ensure_hist_jac`]).
pub(crate) struct HistJac {
    tape: StepEval,
    rows: Vec<usize>,
    cols: Vec<usize>,
}

/// The noise sources compiled: one tape over the system's inputs computing
/// every source's level expressions (see `NoiseSource::exprs`) and then the
/// entries of where its generator enters the rows.
pub(crate) struct NoiseProgram {
    tape: StepEval,
    /// Per source, its number of level expressions.
    levels: Vec<usize>,
    /// Per source, the rows its generator enters.
    rows: Vec<Vec<usize>>,
}

/// A noise source at a point: its level and where it enters the rows, with
/// what gain.
pub struct NoiseAt {
    pub level: sane_dae::NoiseLevel,
    pub injection: Vec<(usize, f64)>,
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

/// The frozen Jacobians of a step start for the transient sensitivities
/// (see [`sane_dae::Frozen`]), laid out for contracting their gradients with
/// a stage's direction `w = (v, τ)` into `dE/dx`.
pub(crate) struct FrozenState {
    /// The entries' rows (the charges' offset by `n`) and the direction's
    /// component each takes (`n` for the time rate `τ`).
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    entries: Vec<ExprId>,
    /// The gradients' expressions, and per gradient entry its direction
    /// component and its place in `dE/dx`.
    hx_e: Vec<ExprId>,
    pub hx_w: Vec<usize>,
    pub hx_slot: Vec<usize>,
    /// `dE/dx`'s coordinates, the rows as the entries'.
    pub x_rc: Pattern,
}

/// One parameter's columns, symbolic: of the currents and charges
/// (`dI/dp ++ dQ/dp`, the charges' rows offset by `n`) and of the frozen
/// entries (`d(entry)/dp`, by the entry's place).
pub(crate) struct ParamExprs {
    iq_rows: Vec<usize>,
    iq_e: Vec<ExprId>,
    e_at: Vec<usize>,
    e_e: Vec<ExprId>,
}

/// The programs of the transient sensitivities by one set of parameters,
/// compiled together so the parameters share what they have in common (a
/// device's own evaluation): at a stage point the columns of the currents
/// and charges, `dI/dp ++ dQ/dp`; at a step start the frozen entries'
/// gradients by the states, then their columns. Each column entry carries
/// its row (the frozen entry's place) and the parameter's place in the set.
pub(crate) struct SensProgram {
    pub iq: StepEval,
    pub iq_rows: Vec<usize>,
    pub iq_of: Vec<usize>,
    pub e: StepEval,
    pub e_at: Vec<usize>,
    pub e_of: Vec<usize>,
}

/// A program of the system's inputs bound to one parameter vector: its
/// parameter prolog run once, each evaluation patching the state and the
/// time into the inputs and running the main phase.
pub(crate) struct Bound {
    tok: PrologToken,
    inputs: Vec<f64>,
    work: Vec<f64>,
}

impl Bound {
    /// `tape` at the parameters `p`.
    pub fn new(cdc: &CompiledDc, tape: &StepEval, p: &[f64]) -> Self {
        let mut inputs = Vec::new();
        cdc.fill_inputs(&[], p, 0.0, &mut inputs);
        let mut work = Vec::new();
        let tok = tape.eval_prolog(&inputs, &mut work);
        Bound { tok, inputs, work }
    }

    /// `tape` at `(x, t)` into `out`.
    pub fn eval(
        &mut self,
        cdc: &CompiledDc,
        tape: &StepEval,
        x: &[f64],
        t: f64,
        out: &mut Vec<f64>,
    ) {
        cdc.patch_inputs(x, t, &mut self.inputs);
        tape.eval_main(&mut self.tok, &self.inputs, &mut self.work, out);
    }
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

    /// Every noise source of `dae` at `(x, p)`: its level and its injection
    /// (see [`NoiseAt`]), in the order of its observers. The program is
    /// compiled on the first query and kept.
    pub fn noise_at(&self, ctx: &mut Graph, dae: &Dae, x: &[f64], p: &[f64]) -> Vec<NoiseAt> {
        let prog = self.noise.get_or_init(|| {
            let noise = dae.observers.noise(ctx);
            let injection = dae.noise_injection(ctx);
            let mut roots: Vec<ExprId> = Vec::new();
            let mut levels = Vec::with_capacity(noise.len());
            for n in noise.iter() {
                let e = n.exprs();
                levels.push(e.len());
                roots.extend(e);
            }
            let rows = (injection.iter())
                .map(|q| q.iter().map(|&(i, _)| i).collect())
                .collect();
            roots.extend(injection.iter().flat_map(|q| q.iter().map(|&(_, e)| e)));
            NoiseProgram {
                tape: crate::eval::step_eval(Tape::compile(ctx, &roots, &self.base_inputs)),
                levels,
                rows,
            }
        });
        let (mut inputs, mut work, mut out) = (Vec::new(), Vec::new(), Vec::new());
        self.fill_inputs(x, p, 0.0, &mut inputs);
        prog.tape.eval(&inputs, &mut work, &mut out);
        let mut vals = out.into_iter();
        let level: Vec<sane_dae::NoiseLevel> = (prog.levels.iter())
            .map(|&k| {
                let mut v: Vec<f64> = vals.by_ref().take(k).collect();
                match k {
                    2 => sane_dae::NoiseLevel::Spectral {
                        psd: v[0],
                        fexp: v[1],
                    },
                    _ => sane_dae::NoiseLevel::Table(
                        v.split_off(2).chunks(2).map(|c| (c[0], c[1])).collect(),
                    ),
                }
            })
            .collect();
        (level.into_iter().zip(&prog.rows))
            .map(|(level, rows)| NoiseAt {
                level,
                injection: rows
                    .iter()
                    .map(|&i| (i, vals.next().expect("one per entry")))
                    .collect(),
            })
            .collect()
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
                    let roots: Vec<_> = (dae.at_rest(ctx).0.iter())
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

    /// Build what the transient sensitivities by the parameter columns
    /// `cols` read, on demand: the frozen entries' gradients once, each
    /// parameter's columns the first time it is asked for, and the
    /// programs of the set (see [`SensProgram`]). Call before
    /// [`solve_transient_sensitivity`](Self::solve_transient_sensitivity).
    pub fn ensure_transient_sensitivity(&self, ctx: &mut Graph, dae: &Dae, cols: &[usize]) {
        if self.sens_programs.lock().unwrap().contains_key(cols) {
            return;
        }
        let fz = self.frozen.get_or_init(|| {
            let f = log_stage!("sens/frozen", frozen(ctx, dae));
            // dE/dx: an entry's gradient moves its row
            let mut at: HashMap<(usize, usize), usize> = HashMap::new();
            let (mut x_rc, mut hx_slot): (Pattern, Vec<usize>) = Default::default();
            for (&k, &c) in f.x.0.iter().zip(&f.x.1) {
                let slot = *at.entry((f.rows[k], c)).or_insert_with(|| {
                    x_rc.0.push(f.rows[k]);
                    x_rc.1.push(c);
                    x_rc.0.len() - 1
                });
                hx_slot.push(slot);
            }
            FrozenState {
                hx_w: f.x.0.iter().map(|&k| f.cols[k]).collect(),
                rows: f.rows,
                cols: f.cols,
                entries: f.entries,
                hx_e: f.x.2,
                hx_slot,
                x_rc,
            }
        });
        let mut exprs = self.param_exprs.lock().unwrap();
        for &c in cols {
            if exprs.contains_key(&c) {
                continue;
            }
            let sym = self.param_syms[c];
            let rest = dae.at_rest(ctx);
            let iq: Vec<ExprId> = rest.0.iter().chain(&rest.1).copied().collect();
            let (iq_rows, iq_e) = param_column(ctx, &iq, sym);
            let (e_at, e_e) = param_column(ctx, &fz.entries, sym);
            let col = ParamExprs {
                iq_rows,
                iq_e,
                e_at,
                e_e,
            };
            exprs.insert(c, std::sync::Arc::new(col));
        }
        let (mut iq_rows, mut iq_of, mut iq_e) = (Vec::new(), Vec::new(), Vec::new());
        let (mut e_at, mut e_of, mut e_e) = (Vec::new(), Vec::new(), fz.hx_e.clone());
        for (j, c) in cols.iter().enumerate() {
            let col = &exprs[c];
            iq_rows.extend_from_slice(&col.iq_rows);
            iq_of.extend(std::iter::repeat_n(j, col.iq_rows.len()));
            iq_e.extend_from_slice(&col.iq_e);
            e_at.extend_from_slice(&col.e_at);
            e_of.extend(std::iter::repeat_n(j, col.e_at.len()));
            e_e.extend_from_slice(&col.e_e);
        }
        drop(exprs);
        let split = |roots: &[ExprId]| {
            let tape = Tape::compile_split(ctx, roots, &self.base_inputs, &self.base_pure);
            crate::eval::step_eval(tape)
        };
        let prog = SensProgram {
            iq: log_stage!("sens/iq_tape", split(&iq_e)),
            iq_rows,
            iq_of,
            e: log_stage!("sens/e_tape", split(&e_e)),
            e_at,
            e_of,
        };
        let mut programs = self.sens_programs.lock().unwrap();
        programs.insert(cols.to_vec(), std::sync::Arc::new(prog));
    }

    /// The frozen entries' layout (see
    /// [`ensure_transient_sensitivity`](Self::ensure_transient_sensitivity)).
    pub(crate) fn frozen(&self) -> Option<&FrozenState> {
        self.frozen.get()
    }

    /// The programs of the transient sensitivities by `cols` (see
    /// [`ensure_transient_sensitivity`](Self::ensure_transient_sensitivity)).
    pub(crate) fn sens_program(&self, cols: &[usize]) -> Option<std::sync::Arc<SensProgram>> {
        self.sens_programs.lock().unwrap().get(cols).cloned()
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
    pub(crate) fn factor_fx(&self, x: &[f64], p: &[f64], t: f64) -> Option<sparse::TripletLu> {
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
