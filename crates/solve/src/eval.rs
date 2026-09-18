//! The hot-tape evaluation backend ladder: plain interpreter, choice-specialized
//! shadow tape, and the background-compiled native (Cranelift) tape, arbitrated
//! per call. Every backend is bit-exact against the others, so swapping
//! mid-solve is sound. See [`StepEval`] for the full picture.

use rsdag::Tape;

/// Hot-tape evaluations before the background native compile is kicked off. A
/// couple of interpreted passes filter out one-shot tapes (a single residual
/// probe never pays compile CPU); anything on a real Newton path crosses the
/// threshold within its first iteration.
#[cfg(feature = "jit")]
const JIT_KICK_EVALS: u32 = 3;

/// Lazy native backend state for one hot tape. The first evaluations interpret;
/// once the tape proves hot, one background thread compiles it with the chunked
/// Cranelift backend (bit-identical results, see `sane-jit`) and `eval` swaps
/// over on the fly -- mid-solve swaps are sound *because* the backends agree to
/// the bit. `SANE_JIT=0` disables the kick entirely.
#[cfg(feature = "jit")]
struct JitState {
    compiled: std::sync::OnceLock<Option<rsdag_jit::NativeTape>>,
    kicked: std::sync::atomic::AtomicBool,
    evals: std::sync::atomic::AtomicU32,
}

#[cfg(feature = "jit")]
pub(crate) fn jit_enabled() -> bool {
    sane_core::config().jit
}

/// The JIT compiler's own rayon pool. Chunk compiles are long, unsplittable
/// jobs (tens of ms each); on the global pool they starve the latency-bound
/// work that runs there, the sparse factorizations and the tree-parallel
/// solves: a worker waiting in a `join` steals a compile chunk and holds the
/// solve's critical path for its duration, and a solve entered from outside
/// the pool queues behind every chunk already injected. Compiling here keeps
/// the two apart; the pool is half the machine, so a compile burst never
/// owns every core either.
#[cfg(feature = "jit")]
pub(crate) fn jit_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        rayon::ThreadPoolBuilder::new()
            .num_threads((cores / 2).max(1))
            .thread_name(|i| format!("sane-jit-{i}"))
            .build()
            .expect("failed to build the JIT compile pool")
    })
}

/// A unit of work for the background compiler.
#[cfg(feature = "jit")]
enum JitJob {
    /// Compile the full tape and publish into the `JitState`.
    Full(std::sync::Arc<Tape>, std::sync::Arc<JitState>),
    /// Compile a choice-specialized (shortened) tape and publish into the
    /// owning `SpecState`, tagged with the specialization epoch -- a stale
    /// result (the tape respecialized while compiling) is dropped on arrival.
    Spec(
        std::sync::Arc<rsdag::SpecializedTape>,
        std::sync::Arc<std::sync::Mutex<SpecState>>,
        u64,
    ),
}

/// The one background compiler thread, started on first use. Jobs queue in
/// kick order; each compiles (internally rayon-parallel over its chunks) and
/// publishes into its owner. A detached worker keeps the process free to exit
/// (the channel sender is static, the thread parks on `recv`).
#[cfg(feature = "jit")]
fn jit_queue() -> &'static std::sync::mpsc::Sender<JitJob> {
    static Q: std::sync::OnceLock<std::sync::mpsc::Sender<JitJob>> = std::sync::OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<JitJob>();
        std::thread::Builder::new()
            .name("sane-jit-compile".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    jit_pool().install(|| match job {
                        JitJob::Full(tape, state) => {
                            let compiled =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    rsdag_jit::NativeTape::compile(&tape).ok()
                                }))
                                .unwrap_or_default();
                            if compiled.is_some() {
                                sane_core::log::debug(
                                    "jit: hot tape compiled, native backend active",
                                );
                            } else {
                                sane_core::log::debug(
                                    "jit: tape compile failed, staying interpreted",
                                );
                            }
                            let _ = state.compiled.set(compiled);
                        }
                        JitJob::Spec(spec, cell, version) => {
                            // The solver reads the prolog guard slots straight
                            // from the work buffer after `eval_prolog`, so the
                            // liveness mask must keep them materialized.
                            let guards: Vec<u32> =
                                spec.prolog_guards().iter().map(|&(s, _)| s).collect();
                            let compiled =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    rsdag_jit::NativeTape::compile_live(spec.tape(), rsdag_jit::CHUNK_OPS, &guards).ok()
                                }))
                                .unwrap_or_default();
                            // Blocking lock is fine here: eval() only ever
                            // try-locks this mutex and falls back to the plain
                            // tape on contention.
                            let mut st = cell.lock().unwrap();
                            st.inflight = false;
                            if st.version != version {
                                sane_core::log::debug(
                                    "jit: specialized tape recompile outdated (region flipped), dropped",
                                );
                            } else if let Some(native) = compiled {
                                sane_core::log::debug(
                                    "jit: specialized tape compiled, native-spec backend active",
                                );
                                st.native = Some((version, std::sync::Arc::new(native)));
                            }
                        }
                    });
                }
            })
            .expect("spawn jit compile worker");
        tx
    })
}

/// Which backend prepared a solve episode's prolog. The token travels with
/// the caller's work buffer: main passes must stay on a layout-compatible
/// backend, so backend switches happen only at episode boundaries or through
/// the always-safe degradations (full native / plain full eval need no
/// prepared state). Carrying the `Arc`s keeps the per-iteration happy paths
/// lock-free; the state mutex is touched only for probation counters and
/// (rare) region flips.
#[derive(Clone)]
pub(crate) enum PrologToken {
    /// Interpreted full split tape prepared the buffer.
    Interp,
    /// Native full split tape; `true` once its prolog ran over the buffer
    /// (the main phase runs it first otherwise).
    #[cfg(feature = "jit")]
    NativeFull(bool),
    /// Interpreted specialized split tape (epoch pinned by the `Arc`).
    SpecInterp(
        std::sync::Arc<rsdag::SpecializedTape>,
        std::sync::Arc<std::sync::Mutex<SpecState>>,
        u64,
    ),
    /// Native specialized split tape.
    #[cfg(feature = "jit")]
    SpecNative(
        std::sync::Arc<rsdag_jit::NativeTape>,
        std::sync::Arc<rsdag::SpecializedTape>,
    ),
    /// Degraded mid-episode: plain full eval per iteration (always correct).
    Fallback,
}

/// A hot evaluation tape with two transparent, bit-exact accelerators.
///
/// Three backends serve one instruction stream, arbitrated per call:
///
/// 1. **Chunked native code** (`sane-jit`): after a few evaluations a
///    background thread compiles the tape; once ready it wins outright.
/// 2. **Choice-specialized shadow** (interpreter): device models branch by
///    operating region (`Select`) and the full tape evaluates both arms;
///    after a traced evaluation the tape is shortened against the current
///    choices ([`Tape::specialize`]) and runs guarded -- a region flip falls
///    back to a fresh trace + respecialization. Bridges the window until the
///    native code lands and carries builds without the `jit` feature. Tapes
///    with few selects, batched (SoA) evaluation, and thrashing choice
///    patterns stay on the plain tape.
/// 3. **Plain interpreter**, always correct, always available.
///
/// Every backend is bit-exact against the others (differentially fuzzed), so
/// swapping mid-solve is sound.
pub(crate) struct StepEval {
    tape: std::sync::Arc<Tape>,
    /// `None` when the tape has too few selects to be worth shortening.
    /// Shared with the background compiler for specialize-then-compile.
    spec: Option<std::sync::Arc<std::sync::Mutex<SpecState>>>,
    #[cfg(feature = "jit")]
    jit: std::sync::Arc<JitState>,
}

/// Specialization cache and its enable/disable policy counters.
#[derive(Default)]
pub(crate) struct SpecState {
    spec: Option<std::sync::Arc<rsdag::SpecializedTape>>,
    /// The choice trace the current specialization was built from.
    choices: Vec<u8>,
    /// Scratch trace buffer for flip re-traces (reused).
    scratch_choices: Vec<u8>,
    /// Selects observed flipping across respecializations: permanently
    /// unpinned, so they survive as real `Select`s in the shortened tape and
    /// stop costing a respecialization per region change (a driven transient
    /// hops regions every period; the bias structure never does).
    unpinned: Vec<bool>,
    evals: u64,
    flips: u64,
    /// Set when choices thrash (flips are a large fraction of evals), so a
    /// pathological circuit does not pay a respecialization per iteration.
    disabled: bool,
    /// Respecialization epoch: bumped whenever `spec` is replaced, so a
    /// background compile of an outdated specialization is discarded on
    /// arrival instead of serving stale choices.
    version: u64,
    /// Consecutive checked evaluations whose guards all held (the compile
    /// probation counter; reset on every flip).
    stable: u32,
    /// Native compiles already spent on this tape (budgeted: a flip after a
    /// compile costs a recompile, and a region-hopping circuit must not turn
    /// the background worker into a compile treadmill).
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    compiles: u32,
    /// Set while a spec-compile job for `version` is in flight.
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    inflight: bool,
    /// No specialization is rebuilt before this eval count (flip cooldown).
    cooldown_until: u64,
    /// The compiled shortened tape, tagged with the `version` it belongs to.
    /// `Arc` so a solve episode can pin it lock-free (see [`PrologToken`]).
    #[cfg(feature = "jit")]
    native: Option<(u64, std::sync::Arc<rsdag_jit::NativeTape>)>,
}

/// Below this many `Select`s the shortening cannot pay for its guard checks.
const SPEC_MIN_SELECTS: usize = 16;

/// Consecutive guard-holding checked evals before the shortened tape is worth
/// compiling natively (specialize-then-compile). Long steady phases -- exactly
/// the workloads where specialization pays -- cross this within one solve.
#[cfg(feature = "jit")]
const SPEC_COMPILE_AFTER: u32 = 16;

/// Native spec-compiles a single tape may spend across region flips.
#[cfg(feature = "jit")]
const SPEC_COMPILE_BUDGET: u32 = 4;

/// A specialization must remove at least this percentage of the instructions
/// to be worth its guard checks and bookkeeping; below that the tape opts out
/// and the full-tape backends serve it. (A drive-coupled circuit whose selects
/// all hop with the signal ends up mostly unpinned after flip learning -- the
/// "shortened" tape is then the full tape plus overhead.)
const SPEC_MIN_SHRINK_PCT: usize = 15;

/// Evals a tape must run flip-free before a (re)specialization is *built*.
/// Rebuilding is O(tape); a cold Newton cascade hops regions constantly, and
/// paying a rebuild per hop is exactly the churn this defers -- the full-tape
/// backends serve the turbulent phase, the specialization returns once the
/// solve settles (where it pays).
const SPEC_FLIP_COOLDOWN: u64 = 32;
/// Cost model: an interpreted tape op costs roughly this many compiled ops.
/// Once the FULL tape's native compile has landed, the *interpreted*
/// specialization rung only pays if the shortened tape is at least this factor
/// shorter -- otherwise a region-hopping circuit (a switching ring) pins the
/// hot path to the interpreter even though native code is sitting ready.
/// The compiled specialization rung is unaffected.
#[cfg(feature = "jit")]
const SPEC_INTERP_COST: usize = 4;

impl StepEval {
    pub(crate) fn new(tape: Tape) -> Self {
        let spec = (sane_core::config().tape_specialization
            && tape.n_selects() >= SPEC_MIN_SELECTS)
            .then(|| std::sync::Arc::new(std::sync::Mutex::new(SpecState::default())));
        StepEval {
            tape: std::sync::Arc::new(tape),
            spec,
            #[cfg(feature = "jit")]
            jit: std::sync::Arc::new(JitState {
                compiled: std::sync::OnceLock::new(),
                kicked: std::sync::atomic::AtomicBool::new(false),
                evals: std::sync::atomic::AtomicU32::new(0),
            }),
        }
    }

    #[inline]
    pub(crate) fn eval(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        // Backend ladder: native-specialized > native-full > specialized
        // interpreter > plain interpreter. Every rung is bit-exact against the
        // others, so any fallback (contention, region flip, compile pending)
        // is transparent. The specialized rungs need the cache lock;
        // contention (parallel sweep workers sharing one tape) falls past
        // them rather than serializing on the cache.
        // The full-tape compile is the ladder's backstop: count every eval
        // toward its kick, regardless of which rung serves it (specialized
        // rungs must not starve the backstop of its trigger).
        #[cfg(feature = "jit")]
        self.maybe_kick();
        if let Some(cache) = &self.spec {
            if let Ok(mut st) = cache.try_lock() {
                if !st.disabled && self.eval_spec(&mut st, inputs, work, out) {
                    return;
                }
            }
        }
        // Native full tape, once the background compile has landed.
        #[cfg(feature = "jit")]
        if let Some(Some(native)) = self.jit.compiled.get() {
            native.eval(inputs, work, out);
            return;
        }
        self.tape.eval(inputs, work, out);
    }

    /// The specialized rungs: the native shortened tape when compiled and
    /// current, else the checked interpreter with compile probation. Returns
    /// `false` when the caller should fall through to the full-tape ladder
    /// (specialization just thrash-disabled itself).
    fn eval_spec(
        &self,
        st: &mut SpecState,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) -> bool {
        st.evals += 1;
        // Native shortened tape (specialize-then-compile): evaluate and check
        // the guard outputs exactly like the interpreted `eval_checked`.
        #[cfg(feature = "jit")]
        if let (Some((ver, native)), Some(sp)) = (&st.native, &st.spec) {
            if *ver == st.version {
                native.eval(inputs, work, out);
                let ok = out[sp.n_real()..]
                    .iter()
                    .zip(sp.expected())
                    .all(|(&v, &e)| (v != 0.0) == (e != 0));
                out.truncate(sp.n_real());
                if ok {
                    st.stable = st.stable.saturating_add(1);
                    return true;
                }
                // Region flip under the compiled specialization: drop the
                // native code (the epoch bump in the flip bookkeeping below
                // also invalidates any in-flight compile) and re-check on the
                // interpreted specialization path.
                st.native = None;
            } else {
                st.native = None; // stale epoch (already respecialized)
            }
        }
        if let Some(sp) = &st.spec {
            // Backend cost model: with the full-tape native compile landed, run
            // the interpreted specialization only when it is genuinely shorter
            // than `native-full / SPEC_INTERP_COST`.
            #[cfg(feature = "jit")]
            if matches!(self.jit.compiled.get(), Some(Some(_)))
                && sp.n_ops() * SPEC_INTERP_COST > self.tape.n_ops()
            {
                return false; // the full-tape ladder (native) serves this eval
            }
            if sp.eval_checked(inputs, work, out) {
                st.stable = st.stable.saturating_add(1);
                // Probation passed: hand the *current* specialization to the
                // background compiler (budgeted, one job in flight per tape).
                #[cfg(feature = "jit")]
                if st.stable == SPEC_COMPILE_AFTER
                    && !st.inflight
                    && st.native.is_none()
                    && st.compiles < SPEC_COMPILE_BUDGET
                    && jit_enabled()
                {
                    if let Some(cache) = &self.spec {
                        st.inflight = true;
                        st.compiles += 1;
                        let _ =
                            jit_queue().send(JitJob::Spec(sp.clone(), cache.clone(), st.version));
                    }
                }
                return true;
            }
            // A region flipped: the shortened outputs are invalid. Re-trace on
            // the full tape below -- unless flips dominate the eval count.
            st.flips += 1;
            st.stable = 0;
            st.version += 1;
            st.cooldown_until = st.evals + SPEC_FLIP_COOLDOWN;
            if st.flips >= 8 && st.evals < st.flips * 8 {
                st.disabled = true;
                st.spec = None;
                #[cfg(feature = "jit")]
                {
                    st.native = None;
                }
                sane_core::log::debug(&format!(
                    "tape specialization disabled (thrash): {} flips in {} evals",
                    st.flips, st.evals
                ));
                return false; // fall through to the full-tape ladder
            }
        }
        if st.spec.is_none() && st.evals < st.cooldown_until {
            return false; // cooling down: the full-tape ladder serves
        }
        // The traced eval inside produces this iteration's correct result
        // even when the rebuild is deferred (cooldown) or opts out.
        self.retrace_respecialize(st, inputs, work, out);
        true
    }

    /// Prolog-split evaluation, phase 1. Prepares `work` for the episode's
    /// main passes and returns the [`PrologToken`] naming the backend that
    /// prepared it -- `eval_main` must stay layout-compatible with that
    /// backend, so the token travels with the caller's buffer.
    pub(crate) fn eval_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) -> PrologToken {
        // An episode counts toward the full-tape compile as an eval does: a
        // tape run only through episodes (the stage program) must reach
        // the native backend too.
        #[cfg(feature = "jit")]
        self.maybe_kick();
        // Specialized rungs (need the cache; skipped on contention/disable).
        if let Some(cache) = &self.spec {
            if let Ok(mut st) = cache.try_lock() {
                if !st.disabled {
                    // Episode entry advances the activity clock the flip
                    // cooldown is measured against (the lock-free native-spec
                    // happy path deliberately does no bookkeeping, so this is
                    // what lets a cooldown expire).
                    st.evals += 1;
                    // First episode (or post-cooldown rebuild): trace once (a
                    // full interpreted eval at the episode's start point) to
                    // obtain a specialization. During a cooldown neither trace
                    // nor rebuild -- the full ladder serves this episode.
                    if st.spec.is_none() {
                        if st.evals < st.cooldown_until {
                            drop(st);
                            #[cfg(feature = "jit")]
                            if let Some(Some(_)) = self.jit.compiled.get() {
                                return PrologToken::NativeFull(false);
                            }
                            self.tape.eval_prolog(inputs, work);
                            return PrologToken::Interp;
                        }
                        let mut tmp = Vec::new();
                        self.retrace_respecialize(&mut st, inputs, work, &mut tmp);
                        st.version += 1;
                        st.stable = 0;
                    }
                    #[cfg(feature = "jit")]
                    if let (Some((ver, native)), Some(sp)) = (st.native.clone(), st.spec.clone()) {
                        if ver == st.version {
                            native.eval_prolog(inputs, work);
                            if sp.check_prolog_guards(inputs, work) {
                                return PrologToken::SpecNative(native, sp);
                            }
                            // A *parameter* change flipped a pinned region:
                            // learn + respecialize, then retry the interpreted
                            // rung below on the fresh specialization.
                            st.native = None;
                            st.version += 1;
                            st.stable = 0;
                            st.flips += 1;
                            st.cooldown_until = st.evals + SPEC_FLIP_COOLDOWN;
                            let mut tmp = Vec::new();
                            self.retrace_respecialize(&mut st, inputs, work, &mut tmp);
                        } else {
                            st.native = None; // stale epoch
                        }
                    }
                    for _ in 0..2 {
                        if let Some(sp) = st.spec.clone() {
                            // Same backend cost model as `eval_spec`: with the
                            // full native tape landed, the interpreted rung
                            // only pays when genuinely shorter.
                            #[cfg(feature = "jit")]
                            if matches!(self.jit.compiled.get(), Some(Some(_)))
                                && sp.n_ops() * SPEC_INTERP_COST > self.tape.n_ops()
                            {
                                return PrologToken::NativeFull(false);
                            }
                            if sp.eval_prolog_checked(inputs, work) {
                                return PrologToken::SpecInterp(sp, cache.clone(), st.version);
                            }
                            st.version += 1;
                            st.stable = 0;
                            st.flips += 1;
                            st.cooldown_until = st.evals + SPEC_FLIP_COOLDOWN;
                            #[cfg(feature = "jit")]
                            {
                                st.native = None;
                            }
                            let mut tmp = Vec::new();
                            self.retrace_respecialize(&mut st, inputs, work, &mut tmp);
                        }
                    }
                }
            }
        }
        #[cfg(feature = "jit")]
        if let Some(Some(native)) = self.jit.compiled.get() {
            native.eval_prolog(inputs, work);
            return PrologToken::NativeFull(true);
        }
        self.tape.eval_prolog(inputs, work);
        PrologToken::Interp
    }

    /// Prolog-split evaluation, phase 2: per-iteration main pass over the
    /// buffer phase 1 prepared, on the backend the token names. Region flips
    /// degrade the token (to the full native tape when available, else the
    /// always-correct plain eval) for the remainder of the episode; the next
    /// episode's `eval_prolog` picks up the respecialization. Every rung is
    /// bit-exact, so degradation is invisible in the results.
    pub(crate) fn eval_main(
        &self,
        token: &mut PrologToken,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) {
        // Keep the full-tape backstop compile fed no matter which rung serves
        // this episode (see `eval`).
        #[cfg(feature = "jit")]
        self.maybe_kick();
        loop {
            match token {
                #[cfg(feature = "jit")]
                PrologToken::SpecNative(native, sp) => {
                    native.eval_main(inputs, work, out);
                    if sp.check_outputs(out) {
                        // No bookkeeping on the happy path: once native, the
                        // probation counters serve nothing, and a mutex per
                        // stage evaluation is measurable.
                        return;
                    }
                    // Main-phase region flip: retrace on the full tape (which
                    // also yields this iteration's correct result), publish
                    // the respecialization, degrade the episode.
                    self.flip_respecialize(inputs, work, out);
                    *token = self.degraded();
                    return;
                }
                PrologToken::SpecInterp(sp, cache, ver) => {
                    #[cfg(not(feature = "jit"))]
                    let _ = ver;
                    if sp.eval_main_checked(inputs, work, out) {
                        if let Ok(mut st) = cache.try_lock() {
                            st.evals += 1;
                            st.stable = st.stable.saturating_add(1);
                            // Probation passed: compile this specialization.
                            #[cfg(feature = "jit")]
                            if st.stable == SPEC_COMPILE_AFTER
                                && st.version == *ver
                                && !st.inflight
                                && st.native.is_none()
                                && st.compiles < SPEC_COMPILE_BUDGET
                                && jit_enabled()
                            {
                                st.inflight = true;
                                st.compiles += 1;
                                let _ =
                                    jit_queue().send(JitJob::Spec(sp.clone(), cache.clone(), *ver));
                            }
                        }
                        return;
                    }
                    self.flip_respecialize(inputs, work, out);
                    *token = self.degraded();
                    return;
                }
                #[cfg(feature = "jit")]
                PrologToken::NativeFull(prolog_done) => {
                    if let Some(Some(native)) = self.jit.compiled.get() {
                        // The split holds natively: the prolog once per
                        // episode (or once here, after a mid-episode switch),
                        // the main phase per call.
                        if !*prolog_done {
                            native.eval_prolog(inputs, work);
                            *prolog_done = true;
                        }
                        native.eval_main(inputs, work, out);
                    } else {
                        // unreachable in practice (the compile never un-lands);
                        // stay correct regardless.
                        self.tape.eval(inputs, work, out);
                    }
                    return;
                }
                PrologToken::Interp => {
                    // Upgrade to the full native tape as soon as it lands (it
                    // needs no prepared state, so the switch is always safe).
                    #[cfg(feature = "jit")]
                    if self.jit.compiled.get().is_some_and(|c| c.is_some()) {
                        *token = PrologToken::NativeFull(false);
                        continue;
                    }
                    self.tape.eval_main(inputs, work, out);
                    return;
                }
                PrologToken::Fallback => {
                    #[cfg(feature = "jit")]
                    if let Some(Some(native)) = self.jit.compiled.get() {
                        native.eval(inputs, work, out);
                        return;
                    }
                    self.tape.eval(inputs, work, out);
                    return;
                }
            }
        }
    }

    /// Retrace + respecialize with flip learning: a full traced eval (the
    /// bit-exact result for the current iteration), a diff of the new trace
    /// against the one the old specialization pinned -- every differing select
    /// is *permanently unpinned* and survives as a real `Select` -- and a
    /// partial respecialization over the stable remainder.
    fn retrace_respecialize(
        &self,
        st: &mut SpecState,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) {
        let mut fresh = std::mem::take(&mut st.scratch_choices);
        fresh.clear();
        self.tape.eval_with(inputs, work, out, &mut fresh);
        if st.unpinned.len() != fresh.len() {
            st.unpinned.resize(fresh.len(), false);
        }
        if st.choices.len() == fresh.len() {
            for (u, (a, b)) in st.unpinned.iter_mut().zip(st.choices.iter().zip(&fresh)) {
                if a != b {
                    *u = true;
                }
            }
        }
        if st.evals < st.cooldown_until {
            // Turbulent phase: learn (the diff above) but defer the O(tape)
            // rebuild; the full-tape backends serve until the flips calm down.
            st.spec = None;
            st.scratch_choices = std::mem::replace(&mut st.choices, fresh);
            return;
        }
        let pin: Vec<bool> = st.unpinned.iter().map(|&u| !u).collect();
        let spec = self.tape.specialize(&fresh, &pin);
        // Not enough shrink left after unpinning: specialization cannot pay
        // for itself on this tape -- opt out for good.
        if spec.n_ops() * 100 > self.tape.n_ops() * (100 - SPEC_MIN_SHRINK_PCT) {
            sane_core::log::debug(&format!(
                "tape specialization opted out (shrink too small: {} of {} ops)",
                spec.n_ops(),
                self.tape.n_ops()
            ));
            st.disabled = true;
            st.spec = None;
        } else {
            st.spec = Some(std::sync::Arc::new(spec));
        }
        st.scratch_choices = std::mem::replace(&mut st.choices, fresh);
    }

    /// Flip handling shared by the split path: learn + respecialize, or
    /// disable on thrash.
    fn flip_respecialize(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        if let Some(cache) = &self.spec {
            // Blocking lock: flips are rare and the state update must land.
            if let Ok(mut st) = cache.lock() {
                st.flips += 1;
                st.stable = 0;
                st.version += 1;
                st.cooldown_until = st.evals + SPEC_FLIP_COOLDOWN;
                #[cfg(feature = "jit")]
                {
                    st.native = None;
                }
                if st.flips >= 8 && st.evals < st.flips * 8 {
                    st.disabled = true;
                    st.spec = None;
                    self.tape.eval(inputs, work, out);
                    return;
                }
                self.retrace_respecialize(&mut st, inputs, work, out);
                return;
            }
        }
        self.tape.eval(inputs, work, out);
    }

    /// The safe episode backend after a mid-episode flip: full native when
    /// available, else the plain full eval.
    fn degraded(&self) -> PrologToken {
        #[cfg(feature = "jit")]
        if self.jit.compiled.get().is_some_and(|c| c.is_some()) {
            return PrologToken::NativeFull(false);
        }
        PrologToken::Fallback
    }

    /// Count interpreted evaluations and, past the threshold, enqueue the tape
    /// on the global background compiler. The kick itself is one channel send
    /// (~100 ns): a `thread::spawn` here would land its ~30 µs setup inside the
    /// caller's timed solve, which is measurable on microsecond-scale circuits.
    /// Failures (or a disabled JIT) leave the interpreter in place permanently.
    #[cfg(feature = "jit")]
    fn maybe_kick(&self) {
        use std::sync::atomic::Ordering;
        if self.jit.evals.fetch_add(1, Ordering::Relaxed) + 1 < JIT_KICK_EVALS
            || !jit_enabled()
            || self.jit.kicked.swap(true, Ordering::Relaxed)
        {
            return;
        }
        let _ = jit_queue().send(JitJob::Full(self.tape.clone(), self.jit.clone()));
    }

    /// Evaluate the tape on `L` independent input sets, lane-interleaved
    /// (`inputs[k][lane]` is input `k` of lane `lane`), one evaluation per
    /// lane through the current backend; `out[o][lane]` receives output
    /// `o` of lane `lane`. `flat` and `work` are the caller's scratch.
    pub(crate) fn eval_batch<const L: usize>(
        &self,
        inputs: &[[f64; L]],
        flat: &mut Vec<f64>,
        work: &mut Vec<f64>,
        out: &mut Vec<[f64; L]>,
    ) {
        let n_out = self.tape.n_outputs();
        out.clear();
        out.resize(n_out, [0.0; L]);
        let mut o = Vec::with_capacity(n_out);
        for lane in 0..L {
            flat.clear();
            flat.extend(inputs.iter().map(|k| k[lane]));
            self.eval(flat, work, &mut o);
            for (j, v) in o.iter().enumerate() {
                out[j][lane] = *v;
            }
        }
    }
}
