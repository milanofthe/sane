//! Exact sensitivity machinery on a compiled DAE: the lazily built parameter
//! and delay-history Jacobian tapes, the noise programs, first-order adjoint
//! sensitivities, the second-order-adjoint Hessian over the compiled
//! Lagrangian blocks, and the programs of the transient's forward
//! sensitivities.

use crate::sparse::Triplets;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rsdag::{ExprId, SymbolId, Tape};
use sane_core::constants::GMIN_DC;
use sane_core::log_stage;
use sane_core::Graph;
use sane_dae::{frozen, lagrangian_hessian, param_column, Dae};

use crate::{sparse, CompiledDc, StepEval};

/// Compiled `dI/dhist` triplets.
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

/// What the sensitivities and the small-signal adjoints compile on first
/// use, over the system's graph: each is heavy at scale, and an analysis
/// that does not ask for it pays nothing. Every builder takes the graph's
/// lock before a cache's, so builders on two threads do not deadlock.
#[derive(Default)]
pub(crate) struct SensCache {
    /// `dI/dp` and `dQ/dp`.
    pjac: OnceLock<ParamJac>,
    /// `dI/dhist`, the frequency-domain coupling of the transport delays.
    hjac: OnceLock<HistJac>,
    noise: OnceLock<NoiseProgram>,
    /// The Lagrangian-Hessian blocks, the heaviest (about `O(n^2)`) stage.
    chess: OnceLock<CompiledHessian>,
    /// The transient sensitivities': the frozen entries' gradients, each
    /// parameter's columns as asked for, the programs per set of columns.
    frozen: OnceLock<FrozenState>,
    param_exprs: Mutex<HashMap<usize, Arc<ParamExprs>>>,
    sens_programs: Mutex<HashMap<Vec<usize>, Arc<SensProgram>>>,
    /// `dI/d(input)` per input symbol (an independent source's value, the
    /// excitation of the AC and pole-zero analyses).
    input_jacs: Mutex<HashMap<SymbolId, Arc<StepEval>>>,
    /// The weighted pulls, by key (see [`CompiledDc::ensure_pull`]).
    pulls: Mutex<HashMap<String, Arc<StepEval>>>,
}

impl CompiledDc {
    /// `tape` over the system's inputs, as a hot program.
    fn program(&self, ctx: &Graph, roots: &[ExprId]) -> StepEval {
        crate::eval::step_eval(Tape::compile(ctx, roots, &self.base_inputs))
    }

    /// The history Jacobian `dI/dhist`; `None` without delays.
    fn hist_jac(&self) -> Option<&HistJac> {
        if self.dae.delays.is_empty() {
            return None;
        }
        Some(self.sens.hjac.get_or_init(|| {
            let mut ctx = self.graph();
            let (r, c, e) = self.dae.jacobian_hist_coo(&mut ctx);
            HistJac {
                tape: self.program(&ctx, &e),
                rows: r,
                cols: c,
            }
        }))
    }

    /// The noise program (see [`NoiseProgram`]).
    fn noise_program(&self) -> &NoiseProgram {
        self.sens.noise.get_or_init(|| {
            let mut guard = self.graph();
            let ctx = &mut *guard;
            let noise = self.dae.observers.noise(ctx);
            let injection = self.dae.noise_injection(ctx);
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
                tape: self.program(ctx, &roots),
                levels,
                rows,
            }
        })
    }

    /// Every noise source at `(x, p)`: its level and its injection (see
    /// [`NoiseAt`]), in the order of its observers.
    pub fn noise_at(&self, x: &[f64], p: &[f64]) -> Vec<NoiseAt> {
        let prog = self.noise_program();
        let mut vals = self.eval_once(&prog.tape, x, p, 0.0, &[]).into_iter();
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

    /// `dI/dhist` at `(x, p)`, the columns the delays; empty without delays.
    pub fn hist_jac_sparse(&self, x: &[f64], p: &[f64]) -> Triplets {
        let Some(hj) = self.hist_jac() else {
            return Triplets::default();
        };
        Triplets {
            rows: hj.rows.clone(),
            cols: hj.cols.clone(),
            vals: self.eval_once(&hj.tape, x, p, 0.0, &[]),
        }
    }

    /// `dI/d(input)` at `(x, p, t)`: the currents' derivative in the symbol
    /// `input` (an independent source's value, the excitation of the AC and
    /// pole-zero analyses). Compiled on the first query for `input`.
    pub fn jacobian_i_input(&self, input: SymbolId, x: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let cached = self.sens.input_jacs.lock().unwrap().get(&input).cloned();
        let tape = cached.unwrap_or_else(|| {
            let mut guard = self.graph();
            let ctx = &mut *guard;
            let roots: Vec<_> = (self.dae.at_rest(ctx).0.iter())
                .map(|&r| rsdag::differentiate(ctx, r, input))
                .collect();
            let tape = Arc::new(self.program(ctx, &roots));
            let mut cache = self.sens.input_jacs.lock().unwrap();
            cache.entry(input).or_insert(tape).clone()
        });
        self.eval_once(&tape, x, p, t, &[])
    }

    /// Build, once per `key`, the weighted pull of the expressions `entries`
    /// makes (on the first call, over the graph and the DAE): the gradient
    /// by the states and by the parameters of `Psi = sum_k w_k e_k`, its
    /// weights `w_k` inputs. Every adjoint of a small-signal quantity is such
    /// a pull: the entries are the Jacobians' (and the input coupling's, the
    /// noise levels'), the weights the products of an adjoint and a solution.
    /// Compiled once, a pull is an evaluation; the graph does not grow per
    /// call.
    pub fn ensure_pull(&self, key: &str, entries: impl FnOnce(&mut Graph, &Dae) -> Vec<ExprId>) {
        if self.sens.pulls.lock().unwrap().contains_key(key) {
            return;
        }
        let mut guard = self.graph();
        let ctx = &mut *guard;
        let entries = entries(ctx, &self.dae);
        let mut psi = Vec::with_capacity(entries.len());
        let mut inputs = self.base_inputs.clone();
        for (k, &e) in entries.iter().enumerate() {
            let w = ctx.sym(&format!("$pull${key}${k}"));
            let rsdag::Node::Symbol(ws) = *ctx.node(w) else {
                unreachable!("a fresh weight name is a symbol")
            };
            inputs.push(ws);
            psi.push(ctx.mul(w, e));
        }
        let psi = ctx.reduce(rsdag::ReduceOp::Sum, psi);
        let wrt: Vec<SymbolId> = (self.dae.x.iter().chain(&self.param_syms))
            .copied()
            .collect();
        let grad = log_stage!("sens/pull", rsdag::gradient(ctx, psi, &wrt));
        let mut pure = self.base_pure.clone();
        pure.resize(inputs.len(), false);
        let tape = Tape::compile_split(ctx, &grad, &inputs, &pure);
        let program = Arc::new(crate::eval::step_eval(tape));
        let mut pulls = self.sens.pulls.lock().unwrap();
        pulls.entry(key.to_string()).or_insert(program);
    }

    /// The pull `key` (see [`ensure_pull`](Self::ensure_pull)) at `(x, p)`
    /// and `t = 0` for the weights `w`: `(dPsi/dx, dPsi/dp)`. `None` if it
    /// was not built.
    pub fn pull(&self, key: &str, x: &[f64], p: &[f64], w: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let tape = self.sens.pulls.lock().unwrap().get(key)?.clone();
        let mut out = self.eval_once(&tape, x, p, 0.0, w);
        let dp = out.split_off(self.n);
        Some((out, dp))
    }

    /// The parameter Jacobians `dI/dp` and `dQ/dp`.
    fn param_jac(&self) -> &ParamJac {
        self.sens.pjac.get_or_init(|| {
            let mut ctx = self.graph();
            let ((ir, ic, ie), (qr, qc, qe)) = log_stage!(
                "sens/param_jac_coo",
                self.dae.jacobian_p_iq_coo(&mut ctx, &self.param_syms)
            );
            let roots: Vec<_> = ie.into_iter().chain(qe).collect();
            ParamJac {
                tape: log_stage!("sens/param_jac_tape", self.program(&ctx, &roots)),
                i_rc: (ir, ic),
                q_rc: (qr, qc),
            }
        })
    }

    /// The frozen entries of a step start (see [`FrozenState`]).
    pub(crate) fn frozen(&self) -> &FrozenState {
        self.sens.frozen.get_or_init(|| {
            let mut ctx = self.graph();
            let f = log_stage!("sens/frozen", frozen(&mut ctx, &self.dae));
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
        })
    }

    /// The programs of the transient sensitivities by the parameter columns
    /// `cols` (see [`SensProgram`]): the frozen entries' gradients once, each
    /// parameter's columns the first time it is asked for, the programs of
    /// the set the first time the set is.
    pub(crate) fn sens_program(&self, cols: &[usize]) -> Arc<SensProgram> {
        if let Some(p) = self.sens.sens_programs.lock().unwrap().get(cols) {
            return p.clone();
        }
        let fz = self.frozen();
        let mut guard = self.graph();
        let ctx = &mut *guard;
        let mut exprs = self.sens.param_exprs.lock().unwrap();
        for &c in cols {
            if exprs.contains_key(&c) {
                continue;
            }
            let sym = self.param_syms[c];
            let rest = self.dae.at_rest(ctx);
            let iq: Vec<ExprId> = rest.0.iter().chain(&rest.1).copied().collect();
            let (iq_rows, iq_e) = param_column(ctx, &iq, sym);
            let (e_at, e_e) = param_column(ctx, &fz.entries, sym);
            let col = ParamExprs {
                iq_rows,
                iq_e,
                e_at,
                e_e,
            };
            exprs.insert(c, Arc::new(col));
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
        let prog = Arc::new(SensProgram {
            iq: log_stage!("sens/iq_tape", split(&iq_e)),
            iq_rows,
            iq_of,
            e: log_stage!("sens/e_tape", split(&e_e)),
            e_at,
            e_of,
        });
        let mut programs = self.sens.sens_programs.lock().unwrap();
        programs.entry(cols.to_vec()).or_insert(prog).clone()
    }

    /// The Lagrangian-Hessian blocks (see [`CompiledHessian`]).
    fn compiled_hessian(&self) -> &CompiledHessian {
        self.sens.chess.get_or_init(|| {
            let mut ctx = self.graph();
            let hs = log_stage!(
                "sens/hessian_lagrangian",
                lagrangian_hessian(&mut ctx, &self.dae, &self.param_syms)
            );
            let mut h_input_syms = self.base_inputs.clone();
            h_input_syms.extend(hs.lambda.iter().chain(&hs.mu).copied());
            // Root order is fixed: xx | xp | pp.
            let h_roots: Vec<_> = (hs.xx.2.iter().chain(&hs.xp.2).chain(&hs.pp.2))
                .copied()
                .collect();
            CompiledHessian {
                np: self.param_syms.len(),
                nxx: hs.xx.2.len(),
                nxp: hs.xp.2.len(),
                xx_rc: (hs.xx.0, hs.xx.1),
                xp_rc: (hs.xp.0, hs.xp.1),
                pp_rc: (hs.pp.0, hs.pp.1),
                tape: log_stage!(
                    "sens/hessian_tape",
                    crate::eval::step_eval(Tape::compile(&ctx, &h_roots, &h_input_syms))
                ),
            }
        })
    }

    /// `dI/dp` and `dQ/dp` (rows x parameter columns) at `(x, p, t)`.
    pub fn jacobian_p_sparse(&self, x: &[f64], p: &[f64], t: f64) -> (Triplets, Triplets) {
        let pj = self.param_jac();
        let mut out = self.eval_once(&pj.tape, x, p, t, &[]);
        let q = out.split_off(pj.i_rc.0.len());
        let (ir, qr) = (&pj.i_rc, &pj.q_rc);
        (
            Triplets {
                rows: ir.0.clone(),
                cols: ir.1.clone(),
                vals: out,
            },
            Triplets {
                rows: qr.0.clone(),
                cols: qr.1.clone(),
                vals: q,
            },
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

        let (
            Triplets {
                rows: pr,
                cols: pc,
                vals: pv,
            },
            _,
        ) = self.jacobian_p_sparse(x, p, t);
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
        let Triplets {
            rows: mut jr,
            cols: mut jc,
            vals: mut jv,
        } = self.jacobian_i_x_sparse(x, p, t);
        for i in 0..n {
            jr.push(i);
            jc.push(i);
            jv.push(GMIN_DC);
        }
        sparse::factor_triplets(n, &jr, &jc, &jv)
    }

    /// Sparsity patterns of the Lagrangian-Hessian blocks
    /// `(L_xx, L_xp, L_pp)`. Fetch once; pair with
    /// [`hessian_block_values`](Self::hessian_block_values).
    pub(crate) fn hessian_block_pattern(&self) -> [Pattern; 3] {
        let ch = self.compiled_hessian();
        [ch.xx_rc.clone(), ch.xp_rc.clone(), ch.pp_rc.clone()]
    }

    /// Values of the Lagrangian-Hessian blocks of `L = λ·I + μ·Q` at the
    /// point `(x, p, t)` for the multipliers `lambda` (of the currents) and
    /// `mu` (of the charges), as `[xx, xp, pp]` aligned with
    /// [`hessian_block_pattern`](Self::hessian_block_pattern). The tape is
    /// linear in the multipliers, so a periodic analysis can sample these
    /// blocks along its waveform, the charges weighted by the adjoint's time
    /// derivative.
    pub(crate) fn hessian_block_values(
        &self,
        x: &[f64],
        p: &[f64],
        t: f64,
        lambda: &[f64],
        mu: &[f64],
    ) -> [Vec<f64>; 3] {
        let ch = self.compiled_hessian();
        let multipliers: Vec<f64> = lambda.iter().chain(mu).copied().collect();
        let mut out = self.eval_once(&ch.tape, x, p, t, &multipliers);
        let pp = out.split_off(ch.nxx + ch.nxp);
        let xp = out.split_off(ch.nxx);
        [out, xp, pp]
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
        let ch = self.compiled_hessian();
        let kk = subset.len();
        let np = ch.np;
        // Parameter index -> its position in `subset` (for building the forward RHS
        // and the L_xp column / L_pp lookups); MAX marks a parameter outside `subset`.
        let mut loc = vec![usize::MAX; np];
        for (a, &j) in subset.iter().enumerate() {
            loc[j] = a;
        }

        // One factorization of G + gmin*I serves every solve: the forward
        // sensitivities G s_k = -dI/dp_k as one multi-column solve, the adjoint
        // G^T lambda = e_metric as a transpose solve.
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
        let (
            Triplets {
                rows: pr,
                cols: pc,
                vals: pv,
            },
            _,
        ) = self.jacobian_p_sparse(x, p, t);
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
        let [xx, xp, pp] = self.hessian_block_values(x, p, t, &lambda, &vec![0.0; n]);

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
