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
//! The body compiles with a prolog split over its parameter-pure arguments:
//! within a solve the parameter binding is fixed, so every call of a solve
//! reuses one prolog buffer per evaluation slot. The slot keys on the actual
//! pure argument values, so correctness never rests on that sharing -- a
//! mismatch just recomputes the prolog.
//!
//! `ensure_function_bodies` is idempotent and cheap when nothing new is
//! referenced; call it before compiling tapes whose roots may contain calls
//! (residuals, Jacobians, parameter Jacobians, Hessian blocks).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rsdag::{func::Body, ExprId, ExternBundle, FuncId, Graph, Output, SymbolId, Tape};

/// A compiled function body: the tape's roots are the function's outputs (in
/// registration order), its inputs the function's parameters.
struct TapeBundle {
    tape: Arc<Tape>,
    /// `None` = not compiled yet (job pending); `Some(None)` = compile failed
    /// (interpreter for good); `Some(Some(..))` = native ready.
    #[cfg(feature = "jit")]
    native: std::sync::OnceLock<Option<rsdag_jit::NativeTape>>,
    n_out: usize,
    /// Positions of the solve-constant (parameter) arguments.
    pure_pos: Vec<usize>,
    /// Distinguishes this body's per-thread prolog state from other bundles'.
    uid: u64,
}

static BUNDLE_UID: AtomicU64 = AtomicU64::new(0);

/// Ops per emitted function for a device body: a body is one straight run
/// of a few thousand ops, best emitted whole.
#[cfg(feature = "jit")]
const CHUNK_OPS_BODY: usize = 16384;

/// One prolog-cached evaluation slot: a work buffer plus the pure argument
/// values it was prepared for.
#[derive(Default)]
struct Slot {
    /// Pure argument values the prolog in `work` was computed for (scalar
    /// slot).
    pure: Vec<f64>,
    work: Vec<f64>,
    out: Vec<f64>,
    primed: bool,
    /// The prolog in `work` was produced by the native scalar backend (its
    /// buffer layout differs from the interpreter's). A slot primed by the
    /// interpreter is re-primed when the native body lands mid-solve.
    primed_native: bool,
}

/// Per-(thread, body) evaluation state.
#[derive(Default)]
struct BodyState {
    scalar: Slot,
}

std::thread_local! {
    /// Bundle bodies contain no nested opaques, so a body evaluation never
    /// re-enters these slots.
    static STATES: RefCell<HashMap<u64, BodyState>> = RefCell::new(HashMap::new());
}

impl TapeBundle {
    fn eval_scalar(&self, args: &[f64], st: &mut Slot, out: &mut [f64]) {
        // Prolog reuse: recompute only when a pure argument moved (new solve /
        // new parameter binding); every instance call of a solve shares it.
        let dirty = !st.primed
            || self.pure_pos.len() != st.pure.len()
            || self
                .pure_pos
                .iter()
                .zip(&st.pure)
                .any(|(&i, &v)| args[i] != v);
        #[cfg(feature = "jit")]
        if let Some(Some(native)) = self.native.get() {
            if dirty || !st.primed_native {
                native.eval_prolog(args, &mut st.work);
                st.pure = self.pure_pos.iter().map(|&i| args[i]).collect();
                st.primed = true;
                st.primed_native = true;
            }
            native.eval_main(args, &mut st.work, &mut st.out);
            out.copy_from_slice(&st.out[..self.n_out]);
            return;
        }
        if dirty || st.primed_native {
            self.tape.eval_prolog(args, &mut st.work);
            st.pure = self.pure_pos.iter().map(|&i| args[i]).collect();
            st.primed = true;
            st.primed_native = false;
        }
        self.tape.eval_main(args, &mut st.work, &mut st.out);
        out.copy_from_slice(&st.out[..self.n_out]);
    }
}

impl ExternBundle for TapeBundle {
    fn n_outputs(&self) -> usize {
        self.n_out
    }

    fn call(&self, args: &[f64], out: &mut [f64]) {
        STATES.with(|s| {
            let mut map = s.borrow_mut();
            let st = map.entry(self.uid).or_default();
            self.eval_scalar(args, &mut st.scalar, out);
        });
    }

    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        STATES.with(|s| {
            let mut map = s.borrow_mut();
            let st = map.entry(self.uid).or_default();
            let mut g = 0usize;
            while g < n_groups {
                let (a, o) = (
                    &args[g * n_args..(g + 1) * n_args],
                    &mut out[g * self.n_out..(g + 1) * self.n_out],
                );
                self.eval_scalar(a, &mut st.scalar, o);
                g += 1;
            }
        });
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
        let pure_pos: Vec<usize> = pure_mask
            .iter()
            .enumerate()
            .filter(|(_, &p)| p)
            .map(|(i, _)| i)
            .collect();
        let tape = Arc::new(Tape::compile_split(ctx, &exprs, &params, &pure_mask));
        let n_out = all.len();
        let n_pure = pure_pos.len();
        let body = Arc::new(TapeBundle {
            tape,
            #[cfg(feature = "jit")]
            native: std::sync::OnceLock::new(),
            n_out,
            pure_pos,
            uid: BUNDLE_UID.fetch_add(1, Ordering::Relaxed),
        });
        // The native form: one background job per body on the JIT compile
        // pool (the bodies of a circuit compile concurrently; the pool is the
        // compiler's own, see `eval::jit_pool`), adopted by every instance
        // call as soon as it lands. A disabled JIT marks it as failed so the
        // dispatch never waits for it.
        #[cfg(feature = "jit")]
        {
            if crate::jit_enabled() {
                let b = body.clone();
                crate::eval::jit_pool().spawn(move || {
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
