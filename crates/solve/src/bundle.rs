//! Compiled function bodies: instance batching for templated compact models.
//!
//! A multiply-instantiated Verilog-A module is one function of the graph and
//! one call per instance (see `sane_core::func`); its Jacobian references the
//! function's derivative outputs. This module compiles, for every symbolic
//! function a set of roots calls, exactly the outputs referenced into one
//! body tape and registers it with the context. Each instance's `BundleCall`
//! then evaluates the one compiled body with its own arguments -- N small
//! calls into shared code instead of N cloned graph segments.
//!
//! A body's native form is compiled in the background as soon as the body
//! exists and adopted when it lands; the interpreted tape serves until then
//! and stays as the bit-identical fallback.
//!
//! The body compiles with a prolog split over its parameter-pure arguments.
//! A calling tape split over the same parameters runs each instance's
//! prolog in its own prolog and keeps the result per instance in its work
//! buffer (`rsdag::ExternBundle::state_len`), so per evaluation only the
//! rest of the body runs.
//!
//! `ensure_function_bodies` is idempotent and cheap when nothing new is
//! referenced; call it before compiling tapes whose roots may contain calls
//! (residuals, Jacobians, parameter Jacobians, Hessian blocks).

use std::sync::Arc;

use rsdag::{func::Body, ExprId, ExternBundle, FuncId, Output, Program, SymbolId, Tape};
use sane_core::Graph;

/// A compiled function body: the tape's roots are the function's outputs (in
/// registration order), its inputs the function's parameters.
///
/// Its prolog over the parameter-pure arguments is the body's state (see
/// `rsdag::ExternBundle::state_len`): a calling tape split over the
/// parameters keeps it per instance in its own work buffer, so an instance
/// with other parameters never costs another's prolog. The interpreter and
/// the native form share the state layout, so the native form takes over
/// mid-solve on a state the interpreter prepared.
struct TapeBundle {
    tape: Arc<Tape>,
    /// `None` = not compiled yet (job pending); `Some(None)` = compile failed
    /// (interpreter for good); `Some(Some(..))` = native ready.
    #[cfg(feature = "jit")]
    native: std::sync::OnceLock<Option<rsdag_jit::NativeTape>>,
    n_out: usize,
    /// One flag per parameter: solve-constant (a parameter of the circuit).
    pure: Vec<bool>,
}

/// Ops per emitted function for a device body: a body is one straight run
/// of a few thousand ops, best emitted whole.
#[cfg(feature = "jit")]
const CHUNK_OPS_BODY: usize = 16384;

impl TapeBundle {
    /// The program to run on `work`: the native form once it has landed and
    /// fits the scratch the caller sized from the interpreter's needs.
    fn program(&self, work: usize) -> &dyn Program {
        #[cfg(feature = "jit")]
        if let Some(Some(native)) = self.native.get() {
            if native.work_len() <= work {
                return native;
            }
        }
        let _ = work;
        &*self.tape
    }
}

impl ExternBundle for TapeBundle {
    fn n_outputs(&self) -> usize {
        self.n_out
    }
    fn work_len(&self) -> usize {
        // The program's buffer, then the arguments a prolog runs on.
        self.tape.work_len() + self.pure.len()
    }
    fn call_into(&self, args: &[f64], work: &mut [f64], out: &mut [f64]) {
        let (w, _) = work.split_at_mut(work.len() - self.pure.len());
        let p = self.program(w.len());
        p.eval_into(args, &mut w[..p.work_len()], out);
    }
    fn state_len(&self) -> usize {
        self.tape.state_len()
    }
    fn pure_args(&self) -> &[bool] {
        &self.pure
    }
    fn prolog_into(&self, pure: &[f64], work: &mut [f64], state: &mut [f64]) {
        let (w, args) = work.split_at_mut(work.len() - self.pure.len());
        // The prolog reads the pure arguments only; the others are NaN.
        let mut p = pure.iter();
        for (a, &is_pure) in args.iter_mut().zip(&self.pure) {
            *a = if is_pure {
                *p.next().expect("one value per pure argument")
            } else {
                f64::NAN
            };
        }
        let prog = self.program(w.len());
        let w = &mut w[..prog.work_len()];
        prog.eval_prolog_into(args, w);
        state.copy_from_slice(&w[..state.len()]);
    }
    fn main_into(&self, args: &[f64], state: &[f64], work: &mut [f64], out: &mut [f64]) {
        let (w, _) = work.split_at_mut(work.len() - self.pure.len());
        let prog = self.program(w.len());
        let w = &mut w[..prog.work_len()];
        w[..state.len()].copy_from_slice(state);
        prog.eval_main_into(args, w, out);
    }
}

/// Compile and register the body of every symbolic function whose outputs
/// `roots` call, covering exactly the outputs referenced (plus the ones an
/// earlier body already carried, so every tape compiled before keeps
/// working). `pure_syms` names the solve-constant symbols (the parameter
/// vector): parameters bound to them feed the body's prolog phase. Idempotent
/// and cheap when nothing new is referenced; call it before compiling tapes
/// whose roots may contain calls (residuals, Jacobians, parameter Jacobians,
/// Hessian blocks).
pub(crate) fn ensure_function_bodies(ctx: &mut Graph, roots: &[ExprId], pure_syms: &[SymbolId]) {
    use std::collections::{BTreeMap, BTreeSet};
    let mut needed: BTreeMap<FuncId, BTreeSet<u32>> = BTreeMap::new();
    for o in ctx.free_calls_in(roots) {
        let (f, out) = ctx.output(o);
        let func = ctx.func(f);
        if func.is_extern() || matches!(func.outputs[out as usize], Output::Zero) {
            continue;
        }
        needed.entry(f).or_default().insert(out);
    }
    let pure: std::collections::HashSet<SymbolId> = pure_syms.iter().copied().collect();
    for (f, outs) in needed {
        // The outputs some registered body carries (a function keeps every
        // body registered for it; a program takes the smallest covering one).
        let have: BTreeSet<u32> = ctx
            .func(f)
            .compiled
            .iter()
            .flat_map(|c| {
                c.slot_of
                    .iter()
                    .enumerate()
                    .filter_map(|(k, s)| s.map(|_| k as u32))
            })
            .collect();
        if outs.is_subset(&have) {
            continue;
        }
        let all: Vec<u32> = have.union(&outs).copied().collect();
        let exprs: Vec<ExprId> = all
            .iter()
            .map(|&k| ctx.output_expr(f, k).expect("symbolic output"))
            .collect();
        let params = ctx.func(f).params.clone();
        let pure_mask: Vec<bool> = params.iter().map(|s| pure.contains(s)).collect();
        let tape = Arc::new(Tape::compile_split(ctx, &exprs, &params, &pure_mask));
        let n_out = all.len();
        let n_pure = pure_mask.iter().filter(|&&p| p).count();
        let body = Arc::new(TapeBundle {
            tape,
            #[cfg(feature = "jit")]
            native: std::sync::OnceLock::new(),
            n_out,
            pure: pure_mask,
        });
        // The native form: one background job per body on the JIT compile
        // pool (the bodies of a circuit compile concurrently; the pool is the
        // compiler's own, see `rsdag_jit::background`), adopted by every instance
        // call as soon as it lands. A disabled JIT marks it as failed so the
        // dispatch never waits for it.
        #[cfg(feature = "jit")]
        {
            if crate::jit_enabled() {
                let b = body.clone();
                rsdag_jit::background::spawn(move || {
                    let job = || {
                        let t0 = sane_core::time::Instant::now();
                        let native =
                            rsdag_jit::NativeTape::compile_with(&b.tape, CHUNK_OPS_BODY).ok();
                        let _ = b.native.set(native);
                        sane_core::log::debug(&format!(
                            "function body ({} ops): native form compiled in {:.0} ms",
                            b.tape.n_ops(),
                            t0.elapsed().as_secs_f64() * 1e3
                        ));
                    };
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                });
            } else {
                let _ = body.native.set(None);
            }
        }
        let mut slot_of = vec![None; ctx.func(f).outputs.len()];
        for (i, &k) in all.iter().enumerate() {
            slot_of[k as usize] = Some(i as u32);
        }
        sane_core::log::debug(&format!(
            "function body '{}': {} outputs ({} parameters, {} pure)",
            ctx.func(f).name,
            n_out,
            params.len(),
            n_pure
        ));
        ctx.set_func_body(
            f,
            Body {
                bundle: body,
                slot_of,
            },
        );
    }
}
