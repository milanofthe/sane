//! Compiled multi-output bodies behind extern functions.
//!
//! An extern function (see [`Graph::define_extern_func`](crate::graph::Graph::define_extern_func))
//! has no expressions for its outputs: each output is a slot of an
//! [`ExternBundle`], and a [`Node::Call`](crate::node::Node::Call) to it is
//! evaluated by calling the bundle. Both evaluation paths (the arena sweep
//! and the compiled [`Tape`](crate::tape::Tape)) do that, so a device
//! template body or an externally compiled model plugs in here.
//!
//! The trait is object-safe and shared as `Arc<dyn ExternBundle>`, so a
//! compiled body survives `Graph` mutation and crosses thread boundaries with
//! the per-thread tapes the solver clones.

/// The forms a backend compiled of a bundle, kept with the bundle so every
/// program that calls it shares them (a native body is emitted once, not
/// once per calling program). Keyed by the backend's options.
#[derive(Default)]
pub struct BackendCache(std::sync::Mutex<Vec<(u64, std::sync::Arc<dyn ExternBundle>)>>);

impl BackendCache {
    /// The form compiled under `key`, compiled by `make` on its first use.
    pub fn get_or_try_insert<E>(
        &self,
        key: u64,
        make: impl FnOnce() -> Result<std::sync::Arc<dyn ExternBundle>, E>,
    ) -> Result<std::sync::Arc<dyn ExternBundle>, E> {
        let find = |v: &[(u64, std::sync::Arc<dyn ExternBundle>)]| {
            v.iter().find(|(k, _)| *k == key).map(|(_, b)| b.clone())
        };
        if let Some(b) = find(&self.0.lock().unwrap()) {
            return Ok(b);
        }
        // Compiled outside the lock: a body's own calls take their bodies'.
        let b = make()?;
        let mut v = self.0.lock().unwrap();
        if let Some(first) = find(&v) {
            return Ok(first);
        }
        v.push((key, b.clone()));
        Ok(b)
    }
}

/// What a backend makes of a tape: a bundle evaluating it, given the
/// tape's pure-argument flags and its number of outputs; `None` where it
/// cannot.
pub type BodyCompiler = dyn Fn(&crate::tape::Tape, &[bool], usize) -> Option<std::sync::Arc<dyn ExternBundle>>
    + Send
    + Sync;

/// Where work runs that its caller does not wait for: a function taking a
/// job, which it runs later, in order with the jobs before it.
pub type Submit = std::sync::Arc<dyn Fn(Box<dyn FnOnce() + Send>) + Send + Sync>;

/// How a program runs the tapes its bundles carry of their own (see
/// [`ExternBundle::with_backend`]): what compiles them, where work runs
/// that nothing waits for, and how bodies specialize per binding.
#[derive(Clone, Default)]
pub struct BodyBackend {
    /// What a tape becomes; `None` interprets it.
    pub compile: Option<std::sync::Arc<BodyCompiler>>,
    /// Identifies what `compile` makes (its options), so the form it made
    /// of a tape is kept with the tape's bundle and shared by every program
    /// (see [`BackendCache`]).
    pub key: u64,
    /// Where compiles and other work go that an evaluation need not wait
    /// for; `None` does it in the evaluation that asks, so what runs is a
    /// function of the calls made alone.
    pub submit: Option<Submit>,
    /// How bodies specialize per binding; `None` keeps what the bundle has
    /// (a backend compiling a program its owner configured).
    pub variants: Option<crate::variant::VariantPolicy>,
}

/// Instances of a bundle a batch runs, in group-major arrays: instance `g`
/// has its arguments at `g * n_args`, its state at `g * stride` and its
/// outputs at `g * n_outputs`. The batch runs the instances `index` lists,
/// in its order, else the first `n`.
#[derive(Clone, Copy, Debug)]
pub struct Instances<'a> {
    pub n: usize,
    pub index: Option<&'a [u32]>,
    pub n_args: usize,
    /// Values from one instance's state to the next's, at least the
    /// bundle's [`state_len`](ExternBundle::state_len).
    pub stride: usize,
}

impl<'a> Instances<'a> {
    /// The first `n` instances, `n_args` arguments and `stride` state
    /// values apart.
    pub fn first(n: usize, n_args: usize, stride: usize) -> Self {
        Instances {
            n,
            index: None,
            n_args,
            stride,
        }
    }

    /// The instances `index` lists, in the arrays of these.
    pub fn listed(self, index: &'a [u32]) -> Self {
        Instances {
            n: index.len(),
            index: Some(index),
            ..self
        }
    }

    /// How many instances run.
    pub fn len(&self) -> usize {
        self.index.map_or(self.n, <[u32]>::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `k`th instance that runs.
    pub fn at(&self, k: usize) -> usize {
        self.index.map_or(k, |ix| ix[k] as usize)
    }

    /// The instances that run, in order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(|k| self.at(k))
    }
}

/// A multi-output compiled body shared by several opaque operators.
///
/// A compiled multi-output body typically produces many correlated outputs at
/// once: a device's terminal currents *and* the entries of its Jacobian.
/// Computing them in one call (the shared interior runs once) is the whole
/// point of compilation, so the outputs of one extern function are slots of a
/// single `ExternBundle`. The compiled tape calls the bundle once per distinct
/// argument list and scatters its outputs to every call that reads one.
pub trait ExternBundle: Send + Sync {
    /// Number of outputs this bundle writes.
    fn n_outputs(&self) -> usize;

    /// Evaluate all outputs from the arguments. `out` has length
    /// [`n_outputs`](Self::n_outputs); `args` holds one value per boundary
    /// input, in input order.
    ///
    /// The default routes through [`call_into`](Self::call_into) with a
    /// thread-local buffer, for callers that have none of their own; a
    /// caller in an inner loop holds one buffer per bundle and calls
    /// `call_into` directly.
    fn call(&self, args: &[f64], out: &mut [f64]) {
        crate::scratch::with_len(self.work_len(), 0.0, |w| self.call_into(args, w, out));
    }

    /// Scratch values [`call_into`](Self::call_into) needs; `0` for a bundle
    /// that keeps none (an opaque body with its own state, a native one).
    fn work_len(&self) -> usize {
        0
    }

    /// [`call`](Self::call) over a work buffer the caller owns: the form a
    /// solver drives per instance per iteration, with nothing allocated and
    /// nothing hidden in a thread-local.
    fn call_into(&self, args: &[f64], work: &mut [f64], out: &mut [f64]);

    /// Values a call keeps for its instance between evaluations: what the
    /// bundle computes from its parameter-pure arguments
    /// ([`pure_args`](Self::pure_args)) once per parameter binding. A
    /// calling tape with a prolog split keeps this block per call site in
    /// its own work buffer: [`prolog_into`](Self::prolog_into) fills it in
    /// the caller's prolog, [`main_into`](Self::main_into) reads it per
    /// evaluation. `0` for a bundle without such a phase.
    fn state_len(&self) -> usize {
        0
    }

    /// Which arguments the state depends on, one flag per argument; empty
    /// when [`state_len`](Self::state_len) is `0`.
    fn pure_args(&self) -> &[bool] {
        &[]
    }

    /// The state of an instance from its pure arguments (`pure`, in
    /// argument order, only the flagged ones). `work` is scratch of
    /// [`work_len`](Self::work_len) values.
    fn prolog_into(&self, _pure: &[f64], _work: &mut [f64], _state: &mut [f64]) {}

    /// The outputs from all arguments and the instance's state as
    /// [`prolog_into`](Self::prolog_into) left it. The default ignores the
    /// state and evaluates everything.
    fn main_into(&self, args: &[f64], _state: &[f64], work: &mut [f64], out: &mut [f64]) {
        self.call_into(args, work, out);
    }

    /// [`prolog_into`](Self::prolog_into) for the instances `at`: their
    /// pure arguments in `pure` (`at.n_args` each), their states in
    /// `states`. The default loops; implementations may run instances side
    /// by side, the states bit for bit the loop's.
    fn prolog_batch(&self, pure: &[f64], states: &mut [f64], at: &Instances) {
        let (na, sl, st) = (at.n_args, self.state_len(), at.stride);
        crate::scratch::with_len(self.work_len(), 0.0, |w| {
            for g in at.iter() {
                let p = &pure[g * na..(g + 1) * na];
                self.prolog_into(p, w, &mut states[g * st..g * st + sl]);
            }
        });
    }

    /// [`main_into`](Self::main_into) for the instances `at`: their
    /// arguments in `args`, their states in `states`, their outputs into
    /// `out`. The default loops; implementations may run instances side by
    /// side, the outputs bit for bit the loop's.
    fn main_batch(&self, args: &[f64], states: &[f64], out: &mut [f64], at: &Instances) {
        let (na, sl, st, no) = (at.n_args, self.state_len(), at.stride, self.n_outputs());
        crate::scratch::with_len(self.work_len(), 0.0, |w| {
            for g in at.iter() {
                let a = &args[g * na..(g + 1) * na];
                let s = &states[g * st..g * st + sl];
                self.main_into(a, s, w, &mut out[g * no..(g + 1) * no]);
            }
        });
    }

    /// Evaluate `n_groups` independent argument groups at once (instance
    /// batching): `args` is group-major (`n_groups * n_args`), `out` likewise
    /// (`n_groups * n_outputs`). The default loops over [`call`](Self::call);
    /// implementations may evaluate the groups as SIMD lanes -- results must
    /// stay bit-identical to the sequential loop.
    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        let n_out = self.n_outputs();
        for g in 0..n_groups {
            self.call(
                &args[g * n_args..(g + 1) * n_args],
                &mut out[g * n_out..(g + 1) * n_out],
            );
        }
    }
    /// The arguments [`main_into`](Self::main_into) reads, ascending, when
    /// it reads only some: the others need not be passed per evaluation
    /// once the prolog ran. `None` for all. The default is what the
    /// [`body`](Self::body)'s main phase reads ([`crate::Tape::main_reads`]).
    fn main_reads(&self) -> Option<Vec<u32>> {
        self.body().map(|t| t.main_reads())
    }
    /// The tape this bundle evaluates, when its body is one: a native
    /// backend compiles it and substitutes its own bundle, so a function
    /// body is emitted once and called per instance instead of being
    /// unrolled into every call site. `None` for an opaque body.
    fn body(&self) -> Option<&crate::tape::Tape> {
        None
    }
    /// Where a backend keeps what it compiled of this bundle's
    /// [`body`](Self::body); `None` compiles it per program.
    fn backend_cache(&self) -> Option<&BackendCache> {
        None
    }
    /// For a bundle that runs tapes of its own besides its
    /// [`body`](Self::body) (a body's per-binding variants, see
    /// [`crate::variant`]): the same bundle with each of them run as
    /// `backend` says. A program's owner and a backend that compiles tapes
    /// ask this before anything else of the bundle; `None` for a bundle
    /// without such tapes.
    fn with_backend(&self, _backend: &BodyBackend) -> Option<std::sync::Arc<dyn ExternBundle>> {
        None
    }
    /// A count that moves whenever background work lands that a prolog
    /// would now run on (a body's variant found, built or compiled, see
    /// [`crate::variant`]): states laid out before it moved stay correct,
    /// a prolog run again may lay out faster ones. `0` for a bundle without
    /// such work.
    fn forms_epoch(&self) -> u64 {
        0
    }
}
