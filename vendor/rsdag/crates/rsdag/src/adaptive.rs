//! A hot program on the fastest backend that serves it, switched per call.
//!
//! Three backends evaluate one tape, and every one is bit-exact against the
//! others, so switching mid-solve is sound:
//!
//! 1. **Native code**, when a [`Compiler`] is given (`rsdag-jit` has one):
//!    after a few evaluations the tape is compiled in the background; once
//!    it lands it wins.
//! 2. **Choice specialization** (interpreter): models branch by operating
//!    region (`Select`) and the full tape evaluates both arms. After a
//!    traced evaluation the tape is shortened against the current choices
//!    ([`Tape::specialize`]) and runs guarded; a region flip retraces and
//!    respecializes, a select seen flipping is unpinned for good, and a
//!    tape whose choices thrash opts out. A specialization that holds long
//!    enough is compiled too.
//! 3. **The interpreter**, always correct, always available.
//!
//! The prolog/main split runs in episodes: [`Adaptive::eval_prolog`] once
//! per parameter binding, [`Adaptive::eval_main`] per iteration over the
//! same buffer. The interpreter and the native code share the state layout
//! ([`Tape::state_len`]), so an episode moves from one to the other at any
//! time; only a specialized episode is tied to its specialization, and
//! falls back to the full tape when a region flips.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::hooks::{log, Level};
use crate::{Program, SpecializedTape, Tape};

/// Native code for [`Adaptive`], from outside the core: `rsdag-jit`
/// implements it; without one the interpreter and its specialization serve.
pub trait Compiler: Send + Sync {
    /// `tape` as native code, keeping the slots `live` in the work array
    /// after a run (the prolog guards of a specialization are read there);
    /// `None` when it cannot be compiled.
    fn compile(&self, tape: &Tape, live: &[u32]) -> Option<Box<dyn Program>>;
    /// Run `job` in the background, after the jobs submitted before it.
    fn submit(&self, job: Box<dyn FnOnce() + Send>);
}

type Native = Arc<dyn Program>;

/// When to compile and when to specialize. The defaults are what a circuit
/// simulator's Newton loop measured best with.
#[derive(Clone, Debug)]
pub struct Policy {
    /// Compile natively at all.
    pub jit: bool,
    /// Evaluations before the background compile is kicked off: a couple of
    /// interpreted passes filter out one-shot tapes.
    pub kick_after: u32,
    /// Specialize over region choices at all.
    pub specialize: bool,
    /// Below this many `Select`s the shortening cannot pay for its guards.
    pub spec_min_selects: usize,
    /// A specialization must remove this percentage of the instructions.
    pub spec_min_shrink_pct: usize,
    /// Evaluations a tape must run flip-free before a specialization is
    /// rebuilt (rebuilding is O(tape); a cold Newton cascade hops regions).
    pub spec_flip_cooldown: u64,
    /// Guard-holding evaluations before a specialization is compiled.
    pub spec_compile_after: u32,
    /// Native compiles a specialization may spend across region flips.
    pub spec_compile_budget: u32,
    /// With the full native code landed, the interpreted specialization
    /// runs only when this many times shorter (an interpreted op costs
    /// about this many native ones).
    pub spec_interp_cost: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            jit: true,
            kick_after: 3,
            specialize: true,
            spec_min_selects: 16,
            spec_min_shrink_pct: 15,
            spec_flip_cooldown: 32,
            spec_compile_after: 16,
            spec_compile_budget: 4,
            spec_interp_cost: 4,
        }
    }
}

/// The native form of the full tape, compiled in the background.
struct Jit {
    /// `None` until the compile lands; `Some(None)` if it failed.
    compiled: OnceLock<Option<Native>>,
    kicked: AtomicBool,
    evals: AtomicU32,
}

/// Specialization cache and its policy counters.
#[derive(Default)]
struct Spec {
    spec: Option<Arc<SpecializedTape>>,
    /// The choice trace the current specialization was built from.
    choices: Vec<u8>,
    scratch_choices: Vec<u8>,
    /// Selects seen flipping: unpinned for good, they stay real `Select`s.
    unpinned: Vec<bool>,
    evals: u64,
    flips: u64,
    /// Choices thrash, or the shrink is too small: the full tape serves.
    disabled: bool,
    /// Bumped whenever `spec` is replaced, so a compile of an outdated
    /// specialization is dropped when it lands.
    version: u64,
    /// Consecutive guard-holding evaluations (the compile probation).
    stable: u32,
    compiles: u32,
    inflight: bool,
    /// No specialization is rebuilt before this evaluation count.
    cooldown_until: u64,
    native: Option<(u64, Native)>,
    built: u64,
}

/// What an [`Adaptive`] has done so far (see [`Adaptive::stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// The full tape's native code has landed.
    pub native: bool,
    /// Specializations built, and region flips seen.
    pub specializations: u64,
    pub flips: u64,
    /// A specialization runs as native code now.
    pub spec_native: bool,
    /// Specialization is off for this tape (thrash, or too little shrink).
    pub spec_off: bool,
}

/// Which specialization, if any, prepared an episode's buffer (see
/// [`Adaptive::eval_prolog`]). The full tape and its native code share
/// their layout, so they are one kind of episode.
#[derive(Clone)]
pub struct Episode(Kind);

#[derive(Clone)]
enum Kind {
    /// The full tape's layout: the interpreter or the native code.
    Full,
    SpecInterp(Arc<SpecializedTape>, u64),
    SpecNative(Native, Arc<SpecializedTape>),
}

/// A tape and its accelerated forms, arbitrated per call.
pub struct Adaptive {
    tape: Arc<Tape>,
    policy: Policy,
    compiler: Option<Arc<dyn Compiler>>,
    spec: Option<Arc<Mutex<Spec>>>,
    jit: Arc<Jit>,
}

impl Adaptive {
    /// `compiler` supplies the native code; without it the interpreter and
    /// its specialization serve.
    pub fn new(tape: Tape, policy: Policy, compiler: Option<Arc<dyn Compiler>>) -> Adaptive {
        let spec = (policy.specialize && tape.n_selects() >= policy.spec_min_selects)
            .then(|| Arc::new(Mutex::new(Spec::default())));
        Adaptive {
            tape: Arc::new(tape),
            policy,
            compiler,
            spec,
            jit: Arc::new(Jit {
                compiled: OnceLock::new(),
                kicked: AtomicBool::new(false),
                evals: AtomicU32::new(0),
            }),
        }
    }

    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// What the program has done so far (diagnostics; the specialization
    /// counters are read only when no evaluation holds them).
    pub fn stats(&self) -> Stats {
        let mut s = Stats {
            native: self.native().is_some(),
            ..Stats::default()
        };
        if let Some(Ok(st)) = self.spec.as_ref().map(|c| c.try_lock()) {
            s.specializations = st.built;
            s.flips = st.flips;
            s.spec_native = st.native.is_some();
            s.spec_off = st.disabled;
        }
        s
    }

    /// The native code of the full tape, once compiled.
    pub fn native(&self) -> Option<&dyn Program> {
        self.jit.compiled.get().and_then(Option::as_deref)
    }

    /// The whole program on the best rung: native specialization, native
    /// full tape, interpreted specialization, interpreter.
    pub fn eval(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        // Every call feeds the full-tape compile, whichever rung serves it.
        self.maybe_kick();
        if let Some(cache) = &self.spec {
            // Contention (parallel callers sharing one tape) falls past the
            // specialized rungs rather than serializing on the cache.
            if let Ok(mut st) = cache.try_lock() {
                if !st.disabled && self.eval_spec(&mut st, cache, inputs, work, out) {
                    return;
                }
            }
        }
        match self.native() {
            Some(native) => run(native, inputs, work, out),
            None => self.tape.eval(inputs, work, out),
        }
    }

    /// The specialized rungs; `false` sends the caller to the full tape.
    fn eval_spec(
        &self,
        st: &mut Spec,
        cache: &Arc<Mutex<Spec>>,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) -> bool {
        st.evals += 1;
        if let (Some((ver, native)), Some(sp)) = (st.native.clone(), st.spec.clone()) {
            if ver == st.version {
                run(&*native, inputs, work, out);
                let ok = out[sp.n_real()..]
                    .iter()
                    .zip(sp.expected())
                    .all(|(&v, &e)| (v != 0.0) == (e != 0));
                out.truncate(sp.n_real());
                if ok {
                    st.stable = st.stable.saturating_add(1);
                    return true;
                }
            }
            // A flip under the compiled specialization, or a stale epoch:
            // drop the native code and re-check interpreted.
            st.native = None;
        }
        if let Some(sp) = st.spec.clone() {
            if self.native().is_some()
                && sp.n_ops() * self.policy.spec_interp_cost > self.tape.n_ops()
            {
                return false; // the native full tape is faster
            }
            if sp.eval_checked(inputs, work, out) {
                st.stable = st.stable.saturating_add(1);
                let ver = st.version;
                self.maybe_compile_spec(st, cache, &sp, ver);
                return true;
            }
            // A region flipped: retrace below, unless flips dominate.
            st.flips += 1;
            st.stable = 0;
            st.version += 1;
            st.cooldown_until = st.evals + self.policy.spec_flip_cooldown;
            if st.flips >= 8 && st.evals < st.flips * 8 {
                self.disable(st, "choices thrash");
                return false;
            }
        }
        if st.spec.is_none() && st.evals < st.cooldown_until {
            return false; // cooling down: the full tape serves
        }
        // The traced evaluation is this call's correct result either way.
        self.retrace(st, inputs, work, out);
        true
    }

    /// Phase 1 of an episode: prepare `work` for the main passes and name
    /// the backend that prepared it, which [`eval_main`](Self::eval_main)
    /// stays compatible with.
    pub fn eval_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) -> Episode {
        // An episode counts toward the compile as an evaluation does.
        self.maybe_kick();
        if let Some(cache) = &self.spec {
            if let Ok(mut st) = cache.try_lock() {
                if !st.disabled {
                    if let Some(kind) = self.spec_prolog(&mut st, inputs, work) {
                        return Episode(kind);
                    }
                }
            }
        }
        self.full_prolog(inputs, work);
        Episode(Kind::Full)
    }

    /// The specialized episode starts, `None` for a full one.
    fn spec_prolog(&self, st: &mut Spec, inputs: &[f64], work: &mut Vec<f64>) -> Option<Kind> {
        // Episode entry advances the clock the flip cooldown runs on.
        st.evals += 1;
        if st.spec.is_none() {
            if st.evals < st.cooldown_until {
                return None;
            }
            let mut tmp = Vec::new();
            self.retrace(st, inputs, work, &mut tmp);
            st.version += 1;
            st.stable = 0;
        }
        if let (Some((ver, native)), Some(sp)) = (st.native.clone(), st.spec.clone()) {
            if ver == st.version {
                prolog(&*native, inputs, work);
                if sp.check_prolog_guards(inputs, work) {
                    return Some(Kind::SpecNative(native, sp));
                }
                // A parameter change flipped a pinned region.
                self.flipped(st);
                let mut tmp = Vec::new();
                self.retrace(st, inputs, work, &mut tmp);
            } else {
                st.native = None;
            }
        }
        for _ in 0..2 {
            let sp = st.spec.clone()?;
            if self.native().is_some()
                && sp.n_ops() * self.policy.spec_interp_cost > self.tape.n_ops()
            {
                return None;
            }
            if sp.eval_prolog_checked(inputs, work) {
                return Some(Kind::SpecInterp(sp, st.version));
            }
            self.flipped(st);
            let mut tmp = Vec::new();
            self.retrace(st, inputs, work, &mut tmp);
        }
        None
    }

    /// The full tape's prolog, natively when compiled; the state is the
    /// same either way.
    fn full_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) {
        match self.native() {
            Some(native) => prolog(native, inputs, work),
            None => self.tape.eval_prolog(inputs, work),
        }
    }

    /// Phase 2: a main pass over the buffer the episode's prolog prepared.
    /// A region flip under a specialization moves the episode to the full
    /// tape (the retrace that handles the flip leaves the full tape's state
    /// in `work`), for its remainder.
    pub fn eval_main(
        &self,
        ep: &mut Episode,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) {
        self.maybe_kick();
        match &ep.0 {
            Kind::SpecNative(native, sp) => {
                main(&**native, inputs, work, out);
                if sp.check_outputs(out) {
                    return;
                }
                self.flip_retrace(inputs, work, out);
                ep.0 = Kind::Full;
            }
            Kind::SpecInterp(sp, ver) => {
                if sp.eval_main_checked(inputs, work, out) {
                    if let Some(cache) = &self.spec {
                        if let Ok(mut st) = cache.try_lock() {
                            st.evals += 1;
                            st.stable = st.stable.saturating_add(1);
                            self.maybe_compile_spec(&mut st, cache, sp, *ver);
                        }
                    }
                    return;
                }
                self.flip_retrace(inputs, work, out);
                ep.0 = Kind::Full;
            }
            // Mid-episode, the interpreter's state serves the native code
            // as it is (`main` only lengthens the buffer).
            Kind::Full => match self.native() {
                Some(native) => main(native, inputs, work, out),
                None => self.tape.eval_main(inputs, work, out),
            },
        }
    }

    /// Evaluate `L` input sets, lane-interleaved (`inputs[k][lane]` is
    /// input `k` of lane `lane`), one evaluation per lane on the current
    /// backend; `out[o][lane]` is output `o` of lane `lane`. `flat`,
    /// `work` and `row` are the caller's scratch.
    pub fn eval_lanes<const L: usize>(
        &self,
        inputs: &[[f64; L]],
        flat: &mut Vec<f64>,
        work: &mut Vec<f64>,
        row: &mut Vec<f64>,
        out: &mut Vec<[f64; L]>,
    ) {
        let n_out = self.tape.n_outputs();
        out.clear();
        out.resize(n_out, [0.0; L]);
        for lane in 0..L {
            flat.clear();
            flat.extend(inputs.iter().map(|k| k[lane]));
            self.eval(flat, work, row);
            for (j, v) in row.iter().enumerate() {
                out[j][lane] = *v;
            }
        }
    }

    // --- bookkeeping --------------------------------------------------------

    fn flipped(&self, st: &mut Spec) {
        st.native = None;
        st.version += 1;
        st.stable = 0;
        st.flips += 1;
        st.cooldown_until = st.evals + self.policy.spec_flip_cooldown;
    }

    fn disable(&self, st: &mut Spec, why: &str) {
        st.disabled = true;
        st.spec = None;
        st.native = None;
        log(
            Level::Debug,
            &format!(
                "tape specialization off ({why}): {} flips in {} evals",
                st.flips, st.evals
            ),
        );
    }

    /// A full traced evaluation (the correct result of this call), a diff
    /// against the trace the current specialization pinned, whose differing
    /// selects are unpinned for good, and a respecialization over the rest
    /// (deferred during a cooldown).
    fn retrace(&self, st: &mut Spec, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
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
        if st.evals >= st.cooldown_until {
            let pin: Vec<bool> = st.unpinned.iter().map(|&u| !u).collect();
            let spec = self.tape.specialize(&fresh, &pin);
            if spec.n_ops() * 100 > self.tape.n_ops() * (100 - self.policy.spec_min_shrink_pct) {
                self.disable(st, "shrink too small");
            } else {
                st.spec = Some(Arc::new(spec));
                st.built += 1;
            }
        } else {
            st.spec = None;
        }
        st.scratch_choices = std::mem::replace(&mut st.choices, fresh);
    }

    /// A main-phase flip: learn and respecialize (or disable on thrash),
    /// with this call's result from the full tape.
    fn flip_retrace(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        if let Some(cache) = &self.spec {
            if let Ok(mut st) = cache.lock() {
                self.flipped(&mut st);
                if st.flips >= 8 && st.evals < st.flips * 8 {
                    self.disable(&mut st, "choices thrash");
                    self.tape.eval(inputs, work, out);
                } else {
                    self.retrace(&mut st, inputs, work, out);
                }
                return;
            }
        }
        self.tape.eval(inputs, work, out);
    }

    /// Past the probation, hand the current specialization to the
    /// background compiler (budgeted, one job in flight).
    fn maybe_compile_spec(
        &self,
        st: &mut Spec,
        cache: &Arc<Mutex<Spec>>,
        sp: &Arc<SpecializedTape>,
        ver: u64,
    ) {
        if !self.policy.jit
            || st.stable != self.policy.spec_compile_after
            || st.version != ver
            || st.inflight
            || st.native.is_some()
            || st.compiles >= self.policy.spec_compile_budget
        {
            return;
        }
        let Some(compiler) = self.compiler.clone() else {
            return;
        };
        st.inflight = true;
        st.compiles += 1;
        let (sp, cache) = (sp.clone(), cache.clone());
        let job = move || {
            // The prolog guards are read from the work buffer after a
            // prolog, so their slots stay materialized.
            let guards: Vec<u32> = sp.prolog_guards().iter().map(|&(s, _)| s).collect();
            let compiled = compiler.compile(sp.tape(), &guards);
            let mut st = cache.lock().unwrap_or_else(|e| e.into_inner());
            st.inflight = false;
            if st.version == ver {
                if let Some(native) = compiled {
                    log(
                        Level::Debug,
                        "specialized tape compiled, native backend active",
                    );
                    st.native = Some((ver, Arc::from(native)));
                }
            }
        };
        if let Some(c) = &self.compiler {
            c.submit(Box::new(job));
        }
    }

    /// Count the call; past the threshold, queue the full-tape compile. A
    /// failed compile leaves the interpreter in place for good.
    fn maybe_kick(&self) {
        let Some(compiler) = self.compiler.clone() else {
            return;
        };
        if !self.policy.jit
            || self.jit.evals.fetch_add(1, Ordering::Relaxed) + 1 < self.policy.kick_after
            || self.jit.kicked.swap(true, Ordering::Relaxed)
        {
            return;
        }
        let (tape, jit) = (self.tape.clone(), self.jit.clone());
        let c = compiler.clone();
        compiler.submit(Box::new(move || {
            let compiled = c.compile(&tape, &[]);
            log(
                Level::Debug,
                if compiled.is_some() {
                    "hot tape compiled, native backend active"
                } else {
                    "tape compile failed, staying interpreted"
                },
            );
            let _ = jit.compiled.set(compiled.map(Arc::from));
        }));
    }
}

/// The Vec-buffer forms over a [`Program`]: the buffers grow to its
/// lengths (never shrink, so a state a prolog left stays in place).
fn fit(p: &dyn Program, work: &mut Vec<f64>) {
    if work.len() < p.work_len() {
        work.resize(p.work_len(), 0.0);
    }
}

fn run(p: &dyn Program, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
    fit(p, work);
    out.resize(p.out_len(), 0.0);
    p.eval_into(inputs, work, out);
}

fn prolog(p: &dyn Program, inputs: &[f64], work: &mut Vec<f64>) {
    fit(p, work);
    p.eval_prolog_into(inputs, work);
}

fn main(p: &dyn Program, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
    fit(p, work);
    out.resize(p.out_len(), 0.0);
    p.eval_main_into(inputs, work, out);
}
