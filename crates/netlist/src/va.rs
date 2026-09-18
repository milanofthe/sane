//! Verilog-A intake: extraction of `.veriloga` blocks / `.va` file loads from
//! the deck, module parsing + elaboration, and the mapping of Verilog-A
//! diagnostic spans back onto deck coordinates.

use rustc_hash::FxHashMap as HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use sane_veriloga::elaborate as va_elaborate;
use sane_veriloga::{parse_modules, Diagnostics, ElaboratedModule};

use crate::{err, err_at, ParseError};

/// Where a `.veriloga` chunk originated, so Verilog-A diagnostic spans map back
/// into deck coordinates instead of being scraped from the rendered message.
pub(crate) enum VaOrigin {
    /// Inline `.veriloga ... .endveriloga` block. `directive_line` is the deck
    /// line of the `.veriloga` directive; the block body begins on the next
    /// line, so a VA span at line `L` lands at deck line `directive_line + L`.
    Inline { directive_line: usize },
    /// External `.va` file pulled in by `.veriloga "file"`. The span belongs to
    /// the file, not the deck, so the file's own caret rendering is embedded in
    /// the message and the deck error points at the `.veriloga` directive.
    File { directive_line: usize },
}

/// Parse a Verilog-A source string and register every module it defines into
/// `models`, keyed by lowercased module name. `include_dirs` is the search path
/// for `` `include `` (the file's own directory for the file form, empty for an
/// inline block); the built-in Accellera headers are always available on top.
pub(crate) fn register_va_modules(
    models: &mut HashMap<String, Arc<ElaboratedModule>>,
    src: &str,
    srcname: &str,
    include_dirs: &[PathBuf],
    origin: VaOrigin,
) -> Result<(), ParseError> {
    let ms = parse_modules(src, srcname, include_dirs)
        .map_err(|d| va_parse_error(&origin, "veriloga", &d, src))?;
    for m in &ms {
        let em =
            va_elaborate(m).map_err(|d| va_parse_error(&origin, "veriloga elaborate", &d, src))?;
        models.insert(m.name.to_ascii_lowercase(), Arc::new(em));
    }
    Ok(())
}

/// Convert a Verilog-A [`Diagnostics`] into a netlist [`ParseError`], carrying
/// the span (line/col) structurally rather than string-scraping the rendered
/// text. For an inline block the VA span is remapped into deck coordinates so
/// the deck's own caret points at the offending line; for a file include the
/// file's rendered caret snippet (which lives outside the deck) is embedded.
pub(crate) fn va_parse_error(
    origin: &VaOrigin,
    phase: &str,
    d: &Diagnostics,
    src: &str,
) -> ParseError {
    match origin {
        VaOrigin::Inline { directive_line } => match d.items.first() {
            Some(it) => err_at(
                directive_line + it.span.line as usize,
                it.span.col as usize,
                &format!("{phase}: {}", it.message),
            ),
            None => err(*directive_line, &format!("{phase}: unspecified error")),
        },
        VaOrigin::File { directive_line } => err(
            *directive_line,
            &format!("{phase}:\n{}", d.render(src).trim_end()),
        ),
    }
}

/// Split the arguments of a `.veriloga "a.va" "b.va"` directive into paths,
/// honouring double quotes (Windows paths may contain spaces) and falling back
/// to whitespace separation for bare tokens.
pub(crate) fn parse_va_paths(rest: &str) -> Vec<String> {
    let mut paths = Vec::new();
    if rest.contains('"') {
        let mut it = rest.chars().peekable();
        while let Some(c) = it.next() {
            if c == '"' {
                let s: String = it.by_ref().take_while(|&c| c != '"').collect();
                if !s.is_empty() {
                    paths.push(s);
                }
            }
        }
    } else {
        paths.extend(rest.split_whitespace().map(|s| s.to_string()));
    }
    paths
}

/// Pull `.veriloga` directives out of the deck and register every module they
/// define. Two forms are supported, both keyed into one registry by lowercased
/// module name and usable from `N` instances:
///
/// - inline block: `.veriloga ... .endveriloga` (no `` `include `` resolution)
/// - file include: `.veriloga "path/to/model.va"` reads each file and adds its
///   own directory to the `` `include `` search path, so real-world models that
///   pull in companion headers (e.g. EKV's `generalMacrosAndDefines.va`) load
///   directly. Relative paths resolve against the process working directory.
///
/// Block lines are blanked in the returned text to preserve line numbers.
pub(crate) fn extract_veriloga(
    text: &str,
) -> Result<(String, HashMap<String, Arc<ElaboratedModule>>), ParseError> {
    let mut models: HashMap<String, Arc<ElaboratedModule>> = HashMap::default();
    let mut out = String::new();
    let mut va_buf = String::new();
    let mut in_va = false;
    let mut start = 0usize;
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        // `.veriloga` either opens an inline block (bare directive) or, with
        // trailing arguments, includes one or more `.va` files.
        if !in_va
            && t.get(..9)
                .is_some_and(|h| h.eq_ignore_ascii_case(".veriloga"))
        {
            let rest = t[9..].trim();
            if rest.is_empty() {
                in_va = true;
                start = i + 1;
                va_buf.clear();
                out.push('\n');
                continue;
            }
            for p in parse_va_paths(rest) {
                let path = PathBuf::from(&p);
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| err(i + 1, &format!("veriloga file '{p}': {e}")))?;
                let dir = path.parent().map(|d| d.to_path_buf()).unwrap_or_default();
                register_va_modules(
                    &mut models,
                    &src,
                    &p,
                    &[dir],
                    VaOrigin::File {
                        directive_line: i + 1,
                    },
                )?;
            }
            out.push('\n');
            continue;
        }
        if in_va && t.eq_ignore_ascii_case(".endveriloga") {
            in_va = false;
            register_va_modules(
                &mut models,
                &va_buf,
                ".veriloga",
                &[],
                VaOrigin::Inline {
                    directive_line: start,
                },
            )?;
            out.push('\n');
            continue;
        }
        if in_va {
            va_buf.push_str(line);
            va_buf.push('\n');
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if in_va {
        return Err(err(start, ".veriloga block without .endveriloga"));
    }
    Ok((out, models))
}
