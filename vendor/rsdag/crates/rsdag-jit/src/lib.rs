//! Native backend for the evaluation [`Tape`](rsdag::Tape).
//!
//! The tape is the evaluation IR; [`Tape::eval`](rsdag::Tape::eval) is the
//! interpreting backend and [`NativeTape`] the native one: the op stream
//! emitted straight to AArch64 or x86-64 machine code (compiled at around
//! 100 to 200 ns per op), with the interpreter's storage model as the register allocator's
//! spill model. The op stream is cut into chunks of [`CHUNK_OPS`]
//! instructions, each its own function, so compile stays linear and the
//! chunks build in parallel; a value crossing a chunk boundary simply stays
//! in the work array.
//!
//! Bit-exactness is a hard invariant: the same IEEE operation sequence as
//! the interpreter (`MulAdd` is a multiply and an add, two roundings, on
//! every backend), reductions folded in the reference
//! order, and every transcendental through the same
//! [`unary_f64`](rsdag::semantics::unary_f64) host routine, so the domain
//! guards hold identically. `tests/parity.rs` fuzzes arena == tape ==
//! native to the bit.

/// Ops per emitted function. A chunk boundary costs nothing but the call,
/// so the size is a matter of parallel build granularity.
pub const CHUNK_OPS: usize = 1024;

/// How a natively compiled function body runs a batch of instances (the
/// calls one `CallBatch` makes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Batch {
    /// One after the other on the calling thread.
    #[default]
    Serial,
    /// Split over the current rayon pool (the caller's `install`, else the
    /// global one) when the batch has at least `min_ops` ops, serially
    /// below. The result is the serial loop's, bit for bit.
    Parallel { min_ops: usize },
}

/// Whether the function bodies a program calls get lane code, running
/// several instances of a batch side by side in every register. The
/// results are the scalar code's, bit for bit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Lanes {
    /// Where it pays for the body, by its ops against the values moved
    /// into and out of the lanes.
    #[default]
    Auto,
    /// Wherever the body compiles so.
    Always,
    /// Never.
    Never,
}

/// What [`NativeTape::compile_opts`] builds.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Ops per emitted function.
    pub chunk_ops: usize,
    /// How the function bodies it calls run their batches.
    pub batch: Batch,
    /// Lane code for the function bodies.
    pub lanes: Lanes,
    /// Whether a body's per-binding variants (see [`rsdag::variant`]) are
    /// found, built and compiled on the [`background`] queue, an instance
    /// running the full body until its variant is ready. Off, a prolog
    /// does that work itself where it first needs it: it pays once, and
    /// what runs is a function of the calls made alone.
    pub background: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            chunk_ops: CHUNK_OPS,
            batch: Batch::Serial,
            lanes: Lanes::Auto,
            background: true,
        }
    }
}

/// Reasons a tape cannot be compiled (callers fall back to the interpreter).
#[derive(Debug)]
pub enum JitError {
    /// Executable memory could not be mapped.
    Codegen(String),
    /// No native backend for this target.
    Unsupported,
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Codegen(s) => write!(f, "native codegen error: {s}"),
            JitError::Unsupported => write!(f, "no native backend for this target"),
        }
    }
}
impl std::error::Error for JitError {}

#[cfg(target_arch = "aarch64")]
mod aarch64;
pub mod background;
mod host;
mod ir;
mod isa;
mod native;
#[cfg(target_arch = "x86_64")]
mod x86_64;

pub use native::NativeTape;

/// The native backend as an [`rsdag::Compiler`]: [`NativeTape`] compiled on
/// the [`background`] queue, for an [`rsdag::Adaptive`].
pub struct Jit {
    pub options: Options,
}

impl rsdag::Compiler for Jit {
    fn compile(&self, tape: &rsdag::Tape, live: &[u32]) -> Option<Box<dyn rsdag::Program>> {
        NativeTape::compile_opts(tape, &self.options, live)
            .ok()
            .map(|n| Box::new(n) as Box<dyn rsdag::Program>)
    }
    fn submit(&self, job: Box<dyn FnOnce() + Send>) {
        background::submit(job);
    }
}

/// [`Jit`] with the default [`Options`], shared.
pub fn compiler() -> std::sync::Arc<dyn rsdag::Compiler> {
    static C: std::sync::OnceLock<std::sync::Arc<dyn rsdag::Compiler>> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        std::sync::Arc::new(Jit {
            options: Options::default(),
        })
    })
    .clone()
}
