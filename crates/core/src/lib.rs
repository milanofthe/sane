//! What is SANE's alone: the physical and solver constants, the run-time
//! configuration, the logger, the profiler, the portable clock, the
//! source-snippet renderer and the lowering of named math calls of the
//! Verilog-A and SPICE front-ends. The expression graph, its
//! differentiation, the tape and the reference evaluation are `rsdag`
//! (vendored under `vendor/rsdag`), which the consumers use directly --
//! [`hooks`] wires its reporting into SANE's logger and clock.

pub mod config;
pub mod constants;
pub mod diag;
pub mod hooks;
pub mod log;
pub mod mathfn;
pub mod profile;
pub mod time;

pub use config::{config, set_config, update_config, Config};
pub use log::{LogLevel, ProgressTracker};
pub use mathfn::lower_math_call;
pub use profile::Profile;
