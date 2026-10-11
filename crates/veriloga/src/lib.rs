//! Native Verilog-A frontend for SANE.
//!
//! Parses Verilog-A source and lowers the analog block onto SANE's symbolic
//! DAG via the `DeviceModel::lower_behavioral` path, so a Verilog-A model
//! becomes an ordinary device the whole engine can analyze. Verilog-A is also
//! the source of truth for the built-in device models (diode, MOSFET, BJT,
//! JFET, MESFET, switch, transformer, transmission line): they ship as `.va`
//! source in `builtin/` and lower through exactly this pipeline. Implemented
//! clean-room from the Accellera Verilog-AMS LRM 2.4.0; no GPL code is reused.
//!
//! Pipeline: `lexer` -> `preprocessor` (include/define/ifdef) -> `parser` ->
//! AST -> `elaborate` -> `lower` (with `template` instance caching).

// The Verilog-A front-end is an internal pipeline: lexer -> preprocessor ->
// parser -> AST -> elaborate -> lower. Those stages are crate-private so their
// token / AST / elaborated representations can change without breaking
// downstream crates; the curated entry points below are the public surface.
pub(crate) mod ast;
pub mod builtin;
pub mod device;
pub(crate) mod elaborate;
pub(crate) mod error;
pub(crate) mod lexer;
pub(crate) mod lower;
mod ops;
pub(crate) mod parser;
pub(crate) mod preprocessor;
mod template;
pub(crate) mod token;
mod topology;

// Curated public surface.
pub use ast::Module;
pub use builtin::{builtin_device, builtin_module};
pub use elaborate::{elaborate, ElaboratedModule, ResolvedParam};
pub use error::{Diagnostic, Diagnostics, Span};
pub use lower::lower_analog;

use std::path::PathBuf;

/// Preprocess and parse Verilog-A source into modules. `search_dirs` are
/// searched for `\`include` files (after the built-in standard headers).
pub fn parse_modules(
    src: &str,
    file: &str,
    search_dirs: &[PathBuf],
) -> Result<Vec<ast::Module>, error::Diagnostics> {
    let (toks, map) =
        preprocessor::preprocess_mapped(src, file, search_dirs, &["__VAMS_COMPACT_MODELING__"])?;
    // Parser diagnostics carry the preprocessor's source map, so an error
    // inside a macro body or an included file renders its true source line
    // plus the whole expansion chain.
    parser::parse(&toks, file).map_err(|d| d.with_map(map))
}

#[cfg(test)]
mod tests;
