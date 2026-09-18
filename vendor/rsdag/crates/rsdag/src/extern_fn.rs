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
pub trait ExternBundle: Send + Sync {
    /// Number of outputs this bundle writes.
    fn n_outputs(&self) -> usize;

    /// Evaluate all outputs from the arguments. `out` has length
    /// [`n_outputs`](Self::n_outputs); `args` holds one value per boundary
    /// input, in input order.
    fn call(&self, args: &[f64], out: &mut [f64]);

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
}
