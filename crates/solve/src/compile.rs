//! Construction of a [`CompiledDc`] from a DAE: sparse symbolic Jacobian
//! extraction, tape compilation (with the parameter-pure prolog split), the
//! reusable symbolic LU, and the linear/nonlinear block classification.

use std::collections::HashMap;

use rsdag::{Crossing, ExprId, Node, ReduceOp, SymbolId, Tape};
use sane_core::constants::*;
use sane_core::Graph;
use sane_core::{time_stage, Profile};
use sane_dae::Dae;

use crate::schur::Partition;
use crate::{sparse, CompiledDc, InputSrc, SolverTricks, StepEval, Symbolic};

impl CompiledDc {
    /// Build from a DAE, computing its sparse symbolic Jacobians and compiling
    /// evaluation tapes.
    pub fn new(ctx: &mut Graph, dae: &Dae) -> Self {
        Self::new_profiled(ctx, dae).0
    }

    /// As [`new`](Self::new), but also return a [`Profile`] of the per-stage
    /// build cost (sparse-Jacobian extraction, tape compilation, symbolic LU,
    /// block classification) -- the instrumented extract pipeline.
    pub fn new_profiled(ctx: &mut Graph, dae: &Dae) -> (Self, Profile) {
        sparse::install_log_bridge();
        let mut prof = Profile::new();
        // Default to sequential: measured, faer's rayon sparse-LU is neutral-to-
        // harmful on the narrow elimination trees of 2D parasitic meshes (up to
        // ~3x slower at n=1e4). The `set_parallelism` knob remains for opt-in
        // experimentation; real throughput parallelism belongs at the outer
        // loop (frequency / tolerance sweeps), not the inner LU.
        // Every row reads `I(x, t) + d/dt Q(x)`: the currents and charges,
        // and their Jacobians `G = dI/dx` and `C = dQ/dx`, one sparse
        // Jacobian over both.
        time_stage!(prof, "at_rest", dae.at_rest(ctx));
        let ((jr, jc, je), (xr, xc, xe)) =
            time_stage!(prof, "jac_iq_coo", dae.jacobian_iq_coo(ctx));
        // The rows' explicit time rates `dI/dt ++ dQ/dt`, by row (the
        // charges' rows after the currents'), then how they move with the
        // delayed signals, `dI/dhist ++ dQ/dhist`, by row and delay: a
        // delayed signal moves in time at its history's rate.
        let ((tr_i, _, te_i), (tr_q, _, te_q)) =
            time_stage!(prof, "jac_t_coo", dae.jacobian_t_iq_coo(ctx));
        let ((hr_i, hc_i, he_i), (hr_q, hc_q, he_q)) =
            time_stage!(prof, "jac_hist_coo", dae.jacobian_hist_iq_coo(ctx));
        let dt_rows: Vec<usize> = (tr_i.iter().copied())
            .chain(tr_q.iter().map(|&r| r + dae.dim()))
            .collect();
        let dt_hist: Vec<(usize, usize)> = (hr_i.iter().copied().zip(hc_i))
            .chain(hr_q.iter().map(|&r| r + dae.dim()).zip(hc_q))
            .collect();
        let dt_roots: Vec<ExprId> = (te_i.iter().chain(&te_q))
            .chain(he_i.iter().chain(&he_q))
            .copied()
            .collect();
        let param_syms = time_stage!(prof, "params", dae.params(ctx));

        // The system as a function with roles; every program over it takes
        // its inputs in the function's signature: x, the parameters, t, the
        // delay histories. The solver also reads what a guard is, and which
        // way it has to cross, off the roles.
        let sys = time_stage!(prof, "register", dae.register_function(ctx, "dae"));
        // The noise generators are no input of a program: the rows it
        // computes are at rest, and what the noise program reads (levels,
        // where a generator enters) none carries.
        let mut sig = rsdag::Signature::of(ctx.func(sys));
        let quiet: Vec<bool> = (sig.roles.iter())
            .map(|r| !matches!(r, rsdag::ParamRole::Noise { .. }))
            .collect();
        let mut keep = quiet.iter();
        sig.syms.retain(|_| *keep.next().unwrap());
        sig.roles
            .retain(|r| !matches!(r, rsdag::ParamRole::Noise { .. }));
        let input_syms = sig.syms.clone();
        let mut n_param = 0;
        let input_src: Vec<InputSrc> = sig
            .roles
            .iter()
            .map(|r| match *r {
                rsdag::ParamRole::State { id } => InputSrc::X(id as usize),
                rsdag::ParamRole::Param => {
                    n_param += 1;
                    InputSrc::P(n_param - 1)
                }
                rsdag::ParamRole::Time => InputSrc::T,
                rsdag::ParamRole::History { id } => InputSrc::Hist(id as usize),
                other => unreachable!("a DAE declares no {other:?} parameter"),
            })
            .collect();
        debug_assert_eq!(
            input_syms[sig.range(|r| matches!(r, rsdag::ParamRole::Param))],
            param_syms[..],
            "the parameters in the parameter vector's order"
        );
        let delay_src = Self::delay_sources(ctx, dae, &input_syms);
        // the Newton aids by unknown index: a limit's ends (ground, or a
        // voltage no longer an unknown, reads zero), the companion network's
        // entries over unknowns
        let index: HashMap<SymbolId, usize> =
            dae.x.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let at = |s: Option<SymbolId>| s.and_then(|s| index.get(&s).copied());
        let limits = (dae.limits.iter())
            .map(|l| crate::limiting::Limit {
                hi: at(l.hi),
                lo: at(l.lo),
                kind: l.kind,
            })
            .filter(|l| l.hi.is_some() || l.lo.is_some())
            .collect();
        let companion = (dae.companion.iter())
            .filter_map(|&(r, c, g)| Some((*index.get(&r)?, *index.get(&c)?, g)))
            .collect();

        let n_nodes = dae.n_nodes;
        let kinds = dae.unknown_kinds();
        debug_assert_eq!(kinds.len(), dae.dim());

        // Tape compilation is the biggest extraction stage on large circuits, and
        // the passes are independent (`Tape::compile` only reads `&Graph`), so
        // they are embarrassingly parallel. They are kept *sequential* on
        // purpose: the pass is allocation-bound, and parallelising it under the OS
        // allocator's global lock measured ~4.5x *slower* on the IBM grids. The
        // robust extraction speedup is the allocator itself (the SANE Python module
        // sets mimalloc; Rust embedders should do likewise -- see the crate docs),
        // not threads. Parallelism here would be a footgun for any embedder on the
        // default allocator, so the embeddable core stays single-threaded.
        // The hot tapes compile with a prolog split: parameter inputs are
        // solve-constant, so every op depending only on them (device-card
        // preprocessing, bin interpolation, temperature scalings) hoists into a
        // prefix the Newton loops evaluate once per parameter binding.
        let pure_inputs = sig.pure_mask();
        let guards = ctx
            .func(sys)
            .outputs_with_role(|r| matches!(r, rsdag::OutputRole::Guard { .. }));
        let event_roots: Vec<ExprId> = guards
            .iter()
            .map(|&o| match ctx.func(sys).outputs()[o as usize] {
                rsdag::Output::Expr(e) => e,
                _ => unreachable!("a guard output is an expression"),
            })
            .collect();
        let event_dirs: Vec<Crossing> = guards
            .iter()
            .map(|&o| match ctx.func(sys).output_roles()[o as usize] {
                rsdag::OutputRole::Guard { dir, .. } => dir,
                _ => unreachable!("selected by role"),
            })
            .collect();
        // The programs, one per way an analysis evaluates the system: the DC
        // Newton `I` and `I ++ G`, the transient (and harmonic balance) `I ++ Q`
        // and `I ++ Q ++ G ++ C`, `C` alone for the state rates, and the
        // switching surfaces. They compile the hierarchy as it stands: a
        // subcircuit body is a template appended per instance, and the device
        // calls of every instance run as one batch, as in a flat circuit;
        // everything symbolic stays on the hierarchy. All of them are views
        // of one program lowered once, each scheduled alone and its device
        // calls running the bodies of the outputs it reads.
        // The rows at rest (every noise generator zero, as in every
        // evaluation): what the programs compute.
        let rest = dae.at_rest(ctx);
        let (currents, charges) = (&rest.0, &rest.1);
        let n_rows = currents.len();
        let tran: Vec<ExprId> = (currents.iter().chain(charges))
            .chain(&je)
            .chain(&xe)
            .copied()
            .collect();
        let step_dc: Vec<ExprId> = currents.iter().chain(&je).copied().collect();
        let taus: Vec<ExprId> = dae.delays.iter().map(|dl| dl.tau).collect();
        let all: Vec<ExprId> = (tran.iter().chain(&event_roots).chain(&dt_roots))
            .copied()
            .collect();
        let lowered = time_stage!(
            prof,
            "tape_lower",
            rsdag::Lowered::new(&*ctx, &all, &input_syms, Some(&pure_inputs))
        );
        let view = |roots: &[ExprId]| crate::eval::step_eval(lowered.tape(&*ctx, roots));
        let tape_tau = (!taus.is_empty()).then(|| Tape::compile(ctx, &taus, &param_syms));
        let tape_res_dc = time_stage!(prof, "tape_res_dc", view(currents));
        let tape_step_dc = time_stage!(prof, "tape_step_dc", view(&step_dc));
        let tape_tran_res = time_stage!(prof, "tape_tran_res", view(&tran[..2 * n_rows]));
        let tape_tran_step = time_stage!(prof, "tape_tran_step", view(&tran));
        let tape_c = time_stage!(prof, "tape_c", view(&xe));
        let tape_dt = (!dt_roots.is_empty()).then(|| time_stage!(prof, "tape_dt", view(&dt_roots)));
        // The switching surfaces, evaluated once per candidate transient step.
        let tape_event = (!event_roots.is_empty())
            .then(|| time_stage!(prof, "tape_event", lowered.tape(&*ctx, &event_roots)));
        drop(lowered);
        let event_names: Vec<String> = dae.events.iter().map(|e| e.name.clone()).collect();

        // The parameter Jacobians and the Lagrangian-Hessian are only needed
        // for sensitivity / second-order sensitivity, and both are heavy at scale,
        // so they are built lazily (see `ensure_param_jac` / `ensure_hessian`).
        // Keep the base input symbols so they can be compiled on demand.
        let base_inputs = input_syms.clone();

        let n = dae.dim();
        let nnz_x = je.len();
        let symbolic = time_stage!(prof, "symbolic_lu", Self::build_symbolic(n, &jr, &jc));

        // Classify each Jacobian entry as constant (its symbolic derivative does
        // not depend on any unknown -> a linear element) or variable (depends on
        // x -> a nonlinear device). The unknowns touched by any variable entry
        // form the nonlinear block V; the rest are the linear block L.
        // Classify every jx nonzero as constant (LTI) or variable (device) once;
        // both the Schur partition below and harmonic balance read this.
        // "Variable" = not constant along a waveform: depends on x or t (an
        // entry depending only on parameters is solve-constant). Harmonic
        // balance keeps constant entries frequency-diagonal instead of dense
        // Toeplitz blocks; the Schur partition treats variable entries as the
        // nonlinear block.
        let vars: Vec<SymbolId> = dae
            .x
            .iter()
            .copied()
            .chain(std::iter::once(dae.t))
            .collect();
        // Same classification for the C nonzeros (charge storage): a linear
        // capacitor's entry is constant, so its harmonic block stays diagonal.
        let (jx_var, jxd_var): (Vec<bool>, Vec<bool>) = time_stage!(
            prof,
            "variable_entries",
            (ctx.depends_on(&je, &vars), ctx.depends_on(&xe, &vars))
        );
        // SPICE node-current scale tape: the additive current terms of each KCL
        // node row as separate roots, so summing their magnitudes per node yields
        // `sum|I_branch|` for the relative convergence test (no symbolic abs op
        // exists, so magnitudes are taken numerically after evaluation). Only the
        // node-adaptive fallback uses it, and only nonlinear circuits ever reach
        // that fallback (linear networks converge in the fast path), so it is built
        // solely for nonlinear circuits -- a linear power grid pays nothing.
        let (tape_iscale, iscale_rows): (Option<StepEval>, Vec<(usize, usize)>) =
            if jx_var.iter().any(|&v| v) {
                let mut terms: Vec<ExprId> = Vec::new();
                let mut rows: Vec<(usize, usize)> = Vec::with_capacity(n_nodes);
                for &r in currents.iter().take(n_nodes) {
                    let start = terms.len();
                    match ctx.node(r) {
                        Node::Reduce(ReduceOp::Sum, l) => terms.extend_from_slice(ctx.args(*l)),
                        _ => terms.push(r),
                    }
                    rows.push((start, terms.len() - start));
                }
                let tape = time_stage!(
                    prof,
                    "tape_iscale",
                    crate::eval::step_eval(Tape::compile(ctx, &terms, &input_syms))
                );
                (Some(tape), rows)
            } else {
                (None, Vec::new())
            };

        let partition = time_stage!(prof, "partition", {
            let var_entry = jx_var.clone();
            let mut is_nonlin = vec![false; n];
            for (k, &v) in var_entry.iter().enumerate() {
                if v {
                    is_nonlin[jr[k]] = true;
                    is_nonlin[jc[k]] = true;
                }
            }
            let nonlin: Vec<usize> = (0..n).filter(|&i| is_nonlin[i]).collect();
            let lin: Vec<usize> = (0..n).filter(|&i| !is_nonlin[i]).collect();
            // Only worthwhile when the nonlinear block is a small fraction of a
            // non-trivial system (otherwise the plain sparse LU is already best).
            if !nonlin.is_empty()
                && n >= PARTITION_MIN_DIM
                && nonlin.len() <= n / PARTITION_MAX_NONLIN_FRAC
            {
                Some(Partition {
                    var_entry,
                    lin,
                    nonlin,
                })
            } else {
                None
            }
        });

        // Independent-source DC values for the source-stepping ramp. The DAE
        // carries the structural list (`source_names`, every independent V/I
        // element, subcircuit-scoped ones like `Xop.I0` included -- a name test
        // cannot tell that source from `Q1.Is`, a saturation current, and the
        // old dot-free heuristic silently skipped every subcircuit bias source,
        // so the ramp excited a circuit around its own bias network). Hand-built
        // DAEs carry no element list; they keep the top-level name heuristic.
        let source_mask: Vec<bool> = if dae.source_names.is_empty() {
            param_syms
                .iter()
                .map(|&s| {
                    let nm = ctx.symbol_name(s);
                    !nm.contains('.') && matches!(nm.chars().next(), Some('V' | 'v' | 'I' | 'i'))
                })
                .collect()
        } else {
            let names: std::collections::HashSet<&str> =
                dae.source_names.iter().map(|s| s.as_str()).collect();
            param_syms
                .iter()
                .map(|&s| names.contains(ctx.symbol_name(s)))
                .collect()
        };
        sane_core::log::debug(&format!(
            "DC source ramp: {} of {} params are independent-source values ({} structural)",
            source_mask.iter().filter(|&&b| b).count(),
            source_mask.len(),
            dae.source_names.len()
        ));

        // Diagonal position of each unknown within the jacobian-x value array
        // (fixed by the pattern; the node-adaptive corrector loads these).
        let mut diag_idx: Vec<Option<usize>> = vec![None; n];
        for (k, (&r, &c)) in jr.iter().zip(&jc).enumerate() {
            if r == c {
                diag_idx[r] = Some(k);
            }
        }

        // Parameter name -> index, so the transient solver can resolve a source's
        // time parameters by their instance-scoped name when emitting breakpoints.
        let param_index: HashMap<String, usize> = param_syms
            .iter()
            .enumerate()
            .map(|(i, &s)| (ctx.symbol_name(s).to_string(), i))
            .collect();
        let index2 = crate::index2::index2_unknowns(n, (&jr, &jc), (&xr, &xc));
        let cdc = CompiledDc {
            n,
            hjac: std::sync::OnceLock::new(),
            noise: std::sync::OnceLock::new(),
            delay_src,
            tape_tau,
            kinds,
            tape_event,
            event_dirs,
            event_names,
            nnz_x,
            index2,
            jx_rows: jr,
            jx_cols: jc,
            diag_idx,
            jxd_rows: xr,
            jxd_cols: xc,
            tape_step_dc,
            tape_res_dc,
            tape_tran_res,
            tape_tran_step,
            tape_c,
            tape_dt,
            dt_rows,
            dt_hist,
            pjac: std::sync::OnceLock::new(),
            input_x_slots: input_src
                .iter()
                .enumerate()
                .filter_map(|(d, s)| match s {
                    InputSrc::X(i) => Some((d as u32, *i as u32)),
                    _ => None,
                })
                .collect(),
            input_t_slots: input_src
                .iter()
                .enumerate()
                .filter_map(|(d, s)| matches!(s, InputSrc::T).then_some(d as u32))
                .collect(),
            input_hist_slots: input_src
                .iter()
                .enumerate()
                .filter_map(|(d, s)| match s {
                    InputSrc::Hist(k) => Some((d as u32, *k as u32)),
                    _ => None,
                })
                .collect(),
            input_src,
            param_syms,
            source_mask,
            symbolic,
            jx_var,
            jxd_var,
            partition,
            chess: std::sync::OnceLock::new(),
            frozen: std::sync::OnceLock::new(),
            param_exprs: Default::default(),
            sens_programs: Default::default(),
            base_inputs,
            base_pure: pure_inputs.clone(),
            companion,
            companion_symbolic: std::sync::OnceLock::new(),
            input_jacs: Default::default(),
            limits,
            n_nodes,
            tape_iscale,
            iscale_rows,
            nodeset: Vec::new(),
            tricks: SolverTricks::default(),
            sources: dae.sources.clone(),
            param_index,
            last_gmin_hold: std::sync::atomic::AtomicU64::new(0),
            last_gmin_row: std::sync::atomic::AtomicUsize::new(usize::MAX),
            last_gmin_share: std::sync::atomic::AtomicU64::new(0),
        };
        (cdc, prof)
    }

    /// What the transport delays of `dae` delay (see
    /// [`DelaySources`](crate::delay::DelaySources)): unknowns where every
    /// source is one, else one tape over `input_syms`.
    fn delay_sources(
        ctx: &mut Graph,
        dae: &Dae,
        input_syms: &[SymbolId],
    ) -> crate::delay::DelaySources {
        let srcs: Vec<ExprId> = dae.delays.iter().map(|dl| dl.src).collect();
        let index: HashMap<SymbolId, usize> =
            dae.x.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let unknowns: Option<Vec<usize>> = (srcs.iter())
            .map(|&e| match *ctx.node(e) {
                Node::Symbol(s) => index.get(&s).copied(),
                _ => None,
            })
            .collect();
        if let Some(ix) = unknowns {
            return crate::delay::DelaySources::Unknowns(ix);
        }
        let mut roots = srcs.clone();
        roots.extend(srcs.iter().map(|&e| rsdag::differentiate(ctx, e, dae.t)));
        let mut jac = Vec::new();
        for (k, row) in rsdag::sparse_jacobian(ctx, &srcs, &dae.x)
            .into_iter()
            .enumerate()
        {
            for (j, g) in row {
                jac.push((k, j));
                roots.push(g);
            }
        }
        crate::delay::DelaySources::Exprs {
            n: srcs.len(),
            tape: Box::new(Tape::compile(ctx, &roots, input_syms)),
            jac,
        }
    }

    /// Build the reusable symbolic analysis over the augmented pattern
    /// (Jacobian nonzeros + full diagonal, so `gmin` injection is free). The
    /// value order callers must supply is: the `(rows, cols)` entries, then
    /// the `n` diagonal entries.
    pub(crate) fn build_symbolic(n: usize, rows: &[usize], cols: &[usize]) -> Option<Symbolic> {
        let mut r = rows.to_vec();
        let mut c = cols.to_vec();
        for i in 0..n {
            r.push(i);
            c.push(i);
        }
        let pattern = sparse::SparsePattern::new(n, &r, &c)?;
        Some(Symbolic { pattern })
    }
}
