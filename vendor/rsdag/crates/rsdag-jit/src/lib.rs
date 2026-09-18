//! Native backend for the evaluation [`Tape`](rsdag::Tape).
//!
//! The tape is the evaluation IR; [`Tape::eval`](rsdag::Tape::eval) is the
//! interpreting backend and [`NativeTape`] the native one: the op stream
//! emitted straight to AArch64 or x86-64 machine code, at around 40 ns per
//! op, with the interpreter's storage model as the register allocator's
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
mod host;
mod ir;
mod isa;
mod native;
#[cfg(target_arch = "x86_64")]
mod x86_64;

pub use native::NativeTape;
