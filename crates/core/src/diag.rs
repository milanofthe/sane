//! Shared source-diagnostic rendering.
//!
//! A single caret-snippet renderer used by both of SANE's input front-ends —
//! the Verilog-A compiler (`sane_veriloga`) and the SPICE netlist parser
//! (`sane_netlist`) — so a parse error looks the same whichever language the
//! user is in: a `file:line:col: error: message` header, then the offending
//! source line, then a caret under the column.

use std::fmt::Write as _;

/// Append one rendered diagnostic to `out`.
///
/// `line`/`col` are 1-based positions into `source`. A `col` of 0 means the
/// column is unknown: the header is still written but no source line/caret is
/// drawn (there is nothing to point at). When `line` is out of range for
/// `source` the caret is likewise omitted.
pub fn render_snippet(
    out: &mut String,
    source: &str,
    file: &str,
    line: u32,
    col: u32,
    message: &str,
) {
    let _ = writeln!(out, "{file}:{line}:{col}: error: {message}");
    if col == 0 {
        return;
    }
    if let Some(src) = source.lines().nth(line.saturating_sub(1) as usize) {
        out.push_str(src);
        out.push('\n');
        let pad = col.saturating_sub(1) as usize;
        out.push_str(&" ".repeat(pad));
        out.push_str("^\n");
    }
}

/// Render a single diagnostic to an owned string (convenience over
/// [`render_snippet`]).
pub fn render_one(source: &str, file: &str, line: u32, col: u32, message: &str) -> String {
    let mut out = String::new();
    render_snippet(&mut out, source, file, line, col, message);
    out
}
