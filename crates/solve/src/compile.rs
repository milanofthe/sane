//! Construction of a [`CompiledDc`] from a DAE: sparse symbolic Jacobian
//! extraction, tape compilation (with the parameter-pure prolog split), the
//! reusable symbolic LU, and the linear/nonlinear block classification.

use std::collections::HashMap;

use rsdag::{Crossing, ExprId, Graph, Node, ReduceOp, SymbolId, Tape};
use sane_core::constants::*;
use sane_core::{time_stage, Profile};
use sane_dae::Dae;

use crate::schur::Partition;
use crate::{bundle, sparse, CompiledDc, InputSrc, SolverTricks, StepEval, Symbolic};

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
        // Assemble the symbolic Jacobians from per-element stamp templates
        // (differentiate once per distinct element structure, instantiate the rest
        // by substitution). Transformed DAEs carry no stamps, so fall back to
        // differentiating the residuals directly.
        let templated = !dae.stamps.is_empty();
        let ((jr, jc, je), (xr, xc, xe)) = if templated {
            // One pass builds both blocks (shared canonicalisation and
            // instantiation per stamp).
            time_stage!(
                prof,
                "jac_x_xdot_coo",
                dae.jacobian_x_xdot_coo_templated(ctx)
            )
        } else {
            let jx = time_stage!(prof, "jac_x_coo", dae.jacobian_x_coo(ctx));
            let jxd = time_stage!(prof, "jac_xdot_coo", dae.jacobian_xdot_coo(ctx));
            (jx, jxd)
        };
        let param_syms = time_stage!(prof, "params", dae.params(ctx));

        // Input ordering: x, then differential xdot, then params, then t.
        let mut input_syms = Vec::new();
        let mut input_src = Vec::new();
        for (i, &s) in dae.x.iter().enumerate() {
            input_syms.push(s);
            input_src.push(InputSrc::X(i));
        }
        for (i, opt) in dae.xdot.iter().enumerate() {
            if let Some(s) = opt {
                input_syms.push(*s);
                input_src.push(InputSrc::Xdot(i));
            }
        }
        for (j, &s) in param_syms.iter().enumerate() {
            input_syms.push(s);
            input_src.push(InputSrc::P(j));
        }
        input_syms.push(dae.t);
        input_src.push(InputSrc::T);
        for (k, dl) in dae.delays.iter().enumerate() {
            input_syms.push(dl.hist);
            input_src.push(InputSrc::Hist(k));
        }
        let delay_src: Vec<usize> = dae.delays.iter().map(|dl| dl.src).collect();
        let tape_tau = (!dae.delays.is_empty()).then(|| {
            let roots: Vec<_> = dae.delays.iter().map(|dl| dl.tau).collect();
            Tape::compile(ctx, &roots, &param_syms)
        });

        let n_nodes = dae.n_nodes;
        let kinds = dae.unknown_kinds();
        debug_assert_eq!(kinds.len(), dae.dim());

        let mut step_roots = dae.residuals.clone();
        step_roots.extend(je.iter().copied());
        // Instance batching, phase 1: bind the bundle outputs the RESIDUAL
        // roots reach to a residual-only body, and compile `tape_res` while
        // that binding holds (the tape interns the bundle by Arc). The inner
        // Newton's residual-only evaluations then run a body without the
        // Jacobian partials -- most of a compact model's work.
        time_stage!(
            prof,
            "function_bodies_res",
            bundle::ensure_function_bodies(ctx, &dae.residuals, &param_syms)
        );
        // Tape compilation is the biggest extraction stage on large circuits, and
        // the three passes are independent (`Tape::compile` only reads `&Graph`),
        // so they are embarrassingly parallel. They are kept *sequential* on
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
        let pure_inputs: Vec<bool> = input_src
            .iter()
            .map(|s| matches!(s, InputSrc::P(_)))
            .collect();
        let tape_res = time_stage!(
            prof,
            "tape_res",
            StepEval::new(Tape::compile_split(
                ctx,
                &dae.residuals,
                &input_syms,
                &pure_inputs
            ))
        );
        // Instance batching, phase 2: rebind the union (outputs + partial
        // markers) for the Jacobian-bearing tapes. `tape_res` above keeps the
        // residual-only body it interned; both bodies compute identical bits
        // for the shared outputs (same DAG nodes, per-op deterministic).
        // The system as a function with roles: the solver then reads what a
        // guard is, and which way it has to cross, off the graph rather than
        // off SANE's own event list (which keeps only the names).
        let sys = dae.register_function(ctx, "dae");
        let guards = ctx
            .func(sys)
            .outputs_with_role(|r| matches!(r, rsdag::OutputRole::Guard { .. }));
        let event_roots: Vec<ExprId> = guards
            .iter()
            .map(|&o| match ctx.func(sys).outputs[o as usize] {
                rsdag::Output::Expr(e) => e,
                _ => unreachable!("a guard output is an expression"),
            })
            .collect();
        let event_dirs: Vec<Crossing> = guards
            .iter()
            .map(|&o| match ctx.func(sys).output_roles[o as usize] {
                rsdag::OutputRole::Guard { dir, .. } => dir,
                _ => unreachable!("selected by role"),
            })
            .collect();
        {
            let mut broots = step_roots.clone();
            broots.extend(xe.iter().copied());
            broots.extend(event_roots.iter().copied());
            time_stage!(
                prof,
                "function_bodies",
                bundle::ensure_function_bodies(ctx, &broots, &param_syms)
            );
        }
        let tape_step = time_stage!(
            prof,
            "tape_step",
            StepEval::new(Tape::compile_split(
                ctx,
                &step_roots,
                &input_syms,
                &pure_inputs
            ))
        );
        let tape_jxd = time_stage!(
            prof,
            "tape_jxd",
            StepEval::new(Tape::compile(ctx, &xe, &input_syms))
        );
        // The switching surfaces, evaluated once per candidate transient step.
        let tape_event = (!event_roots.is_empty()).then(|| {
            time_stage!(
                prof,
                "tape_event",
                Tape::compile(ctx, &event_roots, &input_syms)
            )
        });
        let event_names: Vec<String> = dae.events.iter().map(|e| e.name.clone()).collect();

        // The parameter Jacobian dF/dp and the Lagrangian-Hessian are only needed
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
        // "Variable" = not constant along a waveform: depends on x, x', or t
        // (an entry depending only on parameters is solve-constant). Harmonic
        // balance keeps constant entries frequency-diagonal instead of dense
        // Toeplitz blocks; the Schur partition treats variable entries as the
        // nonlinear block.
        let var_set: std::collections::HashSet<SymbolId> = dae
            .x
            .iter()
            .copied()
            .chain(dae.xdot.iter().flatten().copied())
            .chain(std::iter::once(dae.t))
            .collect();
        let jx_var: Vec<bool> = je
            .iter()
            .map(|&e| ctx.free_symbols(e).iter().any(|s| var_set.contains(s)))
            .collect();
        // Same classification for the jxd nonzeros (charge storage): a linear
        // capacitor's dC entry is constant, so its harmonic block stays diagonal.
        let jxd_var: Vec<bool> = xe
            .iter()
            .map(|&e| ctx.free_symbols(e).iter().any(|s| var_set.contains(s)))
            .collect();
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
                for &r in dae.residuals.iter().take(n_nodes) {
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
                    StepEval::new(Tape::compile(ctx, &terms, &input_syms))
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
        let cdc = CompiledDc {
            n,
            hjac: std::sync::OnceLock::new(),
            delay_src,
            tape_tau,
            kinds,
            tape_event,
            event_dirs,
            event_names,
            last_events: std::sync::Mutex::new(Vec::new()),
            nnz_x,
            jx_rows: jr,
            jx_cols: jc,
            diag_idx,
            jxd_rows: xr,
            jxd_cols: xc,
            tape_step,
            tape_res,
            tape_jxd,
            pjac: std::sync::OnceLock::new(),
            input_x_slots: input_src
                .iter()
                .enumerate()
                .filter_map(|(d, s)| match s {
                    InputSrc::X(i) => Some((d as u32, *i as u32)),
                    _ => None,
                })
                .collect(),
            input_xdot_slots: input_src
                .iter()
                .enumerate()
                .filter_map(|(d, s)| match s {
                    InputSrc::Xdot(i) => Some((d as u32, *i as u32)),
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
            base_inputs,
            companion: dae.companion.clone(),
            companion_symbolic: std::sync::OnceLock::new(),
            limits: dae.limits.clone(),
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
