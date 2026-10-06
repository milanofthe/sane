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

/// A multi-output compiled body shared by several opaque operators.
///
/// A compiled multi-output body typically produces many correlated outputs at
/// once: a device's terminal currents *and* the entries of its Jacobian.
/// Computing them in one call (the shared interior runs once) is the whole
/// point of compilation, so the outputs of one extern function are slots of a
/// single `ExternBundle`. The compiled tape calls the bundle once per distinct
/// argument list and scatters its outputs to every call that reads one.
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

    /// [`prolog_into`](Self::prolog_into) for `n_groups` instances: `pure`
    /// group-major (`n_pure` values each), `states` likewise
    /// ([`state_len`](Self::state_len) each). The default loops;
    /// implementations may run instances side by side, the states bit for
    /// bit the loop's.
    fn prolog_batch(&self, pure: &[f64], n_groups: usize, n_pure: usize, states: &mut [f64]) {
        let sl = self.state_len();
        crate::scratch::with_len(self.work_len(), 0.0, |w| {
            for g in 0..n_groups {
                let (p, st) = (
                    &pure[g * n_pure..(g + 1) * n_pure],
                    &mut states[g * sl..(g + 1) * sl],
                );
                self.prolog_into(p, w, st);
            }
        });
    }

    /// [`main_into`](Self::main_into) for `n_groups` instances: `args`,
    /// `states` and `out` group-major. The default loops; implementations
    /// may run instances side by side, the outputs bit for bit the loop's.
    fn main_batch(
        &self,
        args: &[f64],
        states: &[f64],
        n_groups: usize,
        n_args: usize,
        out: &mut [f64],
    ) {
        let (sl, no) = (self.state_len(), self.n_outputs());
        crate::scratch::with_len(self.work_len(), 0.0, |w| {
            for g in 0..n_groups {
                let a = &args[g * n_args..(g + 1) * n_args];
                let st = &states[g * sl..(g + 1) * sl];
                self.main_into(a, st, w, &mut out[g * no..(g + 1) * no]);
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
}
