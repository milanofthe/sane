//! Text-level `.include` / `.lib` resolution, run before tokenisation.
//!
//! PDK decks pull model libraries and corner sections in by reference; those
//! files contain whole `.model` / `.subckt` / `.param` blocks, so they must be
//! spliced into the source text *before* `preprocess`/`extract_veriloga` see it.
//!
//! Supported forms (ngspice/HSPICE-compatible subset):
//! - `.include "file"` / `.inc "file"` — splice the whole file.
//! - `.lib "file" SECTION` — splice only the `SECTION` block of `file`
//!   (the lines between `.lib SECTION` and its matching `.endl`). This is the
//!   corner-selection idiom (`.lib "corners.lib" tt`).
//! - `.lib SECTION ... .endl` definition blocks encountered standalone are
//!   skipped (only pulled in when called by the two-argument form above).
//!
//! Relative paths resolve against `base_dir` (the directory of the file being
//! processed); the top-level deck uses the caller-provided base (or the CWD).

use std::path::{Path, PathBuf};

use crate::{err, ParseError};

/// Guards against include cycles and runaway recursion.
const MAX_INCLUDE_DEPTH: usize = 64;

/// Expand all `.include` / `.lib` references in `text`. `base_dir` is the
/// directory relative paths in the *top-level* text resolve against (`None`
/// uses the process CWD).
pub(crate) fn resolve_includes(text: &str, base_dir: Option<&Path>) -> Result<String, ParseError> {
    let mut out = String::new();
    let mut seen: Vec<PathBuf> = Vec::new();
    expand(text, base_dir, &mut out, &mut seen, 0)?;
    Ok(out)
}

fn expand(
    text: &str,
    base_dir: Option<&Path>,
    out: &mut String,
    seen: &mut Vec<PathBuf>,
    depth: usize,
) -> Result<(), ParseError> {
    if depth > MAX_INCLUDE_DEPTH {
        return Err(err(
            0,
            "`.include`/`.lib` nesting too deep (cyclic reference?)",
        ));
    }
    let raw: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < raw.len() {
        let line = raw[i];
        let trimmed = line.trim_start();
        let head = first_word(trimmed).to_ascii_lowercase();

        if head == ".include" || head == ".inc" {
            let args = split_args(after_word(trimmed));
            let file = args
                .first()
                .ok_or_else(|| err(i + 1, "`.include` needs a file path"))?;
            let path = resolve_path(base_dir, file);
            include_file(&path, out, seen, depth, i + 1)?;
            i += 1;
            continue;
        }

        if head == ".lib" {
            let args = split_args(after_word(trimmed));
            match args.len() {
                // `.lib file section` — pull in one section of an external file.
                n if n >= 2 => {
                    let path = resolve_path(base_dir, &args[0]);
                    include_section(&path, &args[1], out, seen, depth, i + 1)?;
                    i += 1;
                    continue;
                }
                // `.lib section` — a definition block; skip it (and its body)
                // unless it is reached via the call form above.
                1 => {
                    i = skip_lib_block(&raw, i);
                    continue;
                }
                _ => return Err(err(i + 1, "malformed `.lib` directive")),
            }
        }

        if head == ".endl" {
            i += 1; // stray `.endl` outside a block: drop it.
            continue;
        }

        out.push_str(line);
        out.push('\n');
        i += 1;
    }
    Ok(())
}

/// Splice an entire included file, recursively resolving its own references.
fn include_file(
    path: &Path,
    out: &mut String,
    seen: &mut Vec<PathBuf>,
    depth: usize,
    line: usize,
) -> Result<(), ParseError> {
    let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if seen.contains(&canon) {
        return Ok(()); // already included once; break the cycle.
    }
    seen.push(canon);
    let src = std::fs::read_to_string(path).map_err(|e| {
        err(
            line,
            &format!("cannot read `.include` file '{}': {e}", path.display()),
        )
    })?;
    expand(&src, path.parent(), out, seen, depth + 1)
}

/// Splice one `.lib SECTION ... .endl` block out of an external library file.
fn include_section(
    path: &Path,
    section: &str,
    out: &mut String,
    seen: &mut Vec<PathBuf>,
    depth: usize,
    line: usize,
) -> Result<(), ParseError> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        err(
            line,
            &format!("cannot read `.lib` file '{}': {e}", path.display()),
        )
    })?;
    let raw: Vec<&str> = src.lines().collect();
    let mut i = 0;
    while i < raw.len() {
        let t = raw[i].trim_start();
        if first_word(t).eq_ignore_ascii_case(".lib") {
            let args = split_args(after_word(t));
            if args.len() == 1 && args[0].eq_ignore_ascii_case(section) {
                // Found the section: capture its body up to the matching `.endl`.
                // `end` is the index past the `.endl` (or `raw.len()` if the file
                // is truncated with no `.endl`); clamp so a section at/near EOF
                // yields an empty body instead of panicking on an inverted range.
                let end = skip_lib_block(&raw, i);
                let start = i + 1;
                let stop = end.saturating_sub(1).max(start);
                let body = raw[start..stop].join("\n");
                return expand(&body, path.parent(), out, seen, depth + 1);
            }
        }
        i += 1;
    }
    Err(err(
        line,
        &format!(
            "section '{section}' not found in `.lib` file '{}'",
            path.display()
        ),
    ))
}

/// Given the index of a `.lib <name>` definition line, return the index of the
/// line just past its matching `.endl`, honouring nested definition blocks.
fn skip_lib_block(raw: &[&str], start: usize) -> usize {
    let mut depth = 0i32;
    let mut i = start;
    while i < raw.len() {
        let t = raw[i].trim_start();
        let head = first_word(t).to_ascii_lowercase();
        if head == ".lib" && split_args(after_word(t)).len() == 1 {
            depth += 1; // a nested definition block opens.
        } else if head == ".endl" {
            depth -= 1;
            if depth == 0 {
                return i + 1;
            }
        }
        i += 1;
    }
    raw.len()
}

/// Resolve `file` against `base_dir` (absolute paths pass through).
fn resolve_path(base_dir: Option<&Path>, file: &str) -> PathBuf {
    let p = Path::new(file);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    match base_dir {
        Some(dir) => dir.join(p),
        None => p.to_path_buf(),
    }
}

/// First whitespace-delimited word of a line (the directive keyword).
fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}

/// Everything after the first word (the directive arguments), comments stripped.
fn after_word(s: &str) -> &str {
    let rest = match s.find(char::is_whitespace) {
        Some(p) => &s[p..],
        None => "",
    };
    // Drop a trailing inline comment (`;` / `$`).
    let cut = rest.find([';', '$']).unwrap_or(rest.len());
    &rest[..cut]
}

/// Split directive arguments, honouring double-quoted paths (which may contain
/// spaces) and falling back to whitespace separation for bare tokens.
fn split_args(rest: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut it = rest.chars().peekable();
    while let Some(&c) = it.peek() {
        if c.is_whitespace() {
            it.next();
        } else if c == '"' {
            it.next();
            let s: String = it.by_ref().take_while(|&c| c != '"').collect();
            args.push(s);
        } else {
            let s: String =
                std::iter::from_fn(|| it.next_if(|c| !c.is_whitespace() && *c != '"')).collect();
            if !s.is_empty() {
                args.push(s);
            }
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_quoted_and_bare_args() {
        assert_eq!(split_args(r#" "a b.lib" tt "#), vec!["a b.lib", "tt"]);
        assert_eq!(split_args(" models.inc "), vec!["models.inc"]);
    }

    #[test]
    fn skips_definition_block() {
        let src = ".lib tt\nR1 1 0 1k\n.endl\nC1 0 1 1u\n";
        let out = resolve_includes(src, None).unwrap();
        // The standalone definition block is dropped; the trailing line stays.
        assert!(out.contains("C1 0 1 1u"));
        assert!(!out.contains("R1 1 0 1k"));
    }
}
