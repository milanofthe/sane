//! SPICE netlist preprocessing: strip comments, join continuation lines, and
//! tokenise into logical lines. Run before element parsing so the parser sees
//! clean token streams.
//!
//! Rules (close to ngspice/LTspice):
//! - A line whose first non-space character is `*` is a full-line comment.
//! - `$` and `;` start an inline comment (to end of line), unless inside `{...}`.
//! - A line whose first non-space character is `+` continues the previous
//!   logical line (its tokens are appended).
//! - `{...}` expression groups are kept as a single token even with spaces.
//! - Everything is otherwise whitespace-separated; case is preserved here and
//!   handled case-insensitively by the parser.

/// A logical line: the 1-based source line number of its first physical line,
/// the 1-based column of its first token (for caret diagnostics), and its
/// tokens.
#[derive(Debug, Clone)]
pub struct Line {
    pub no: usize,
    pub col: usize,
    pub tokens: Vec<String>,
}

/// Preprocess raw netlist text into logical lines.
pub fn preprocess(text: &str) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let no = i + 1;
        let content = strip_comment(raw);
        let trimmed = content.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('+') {
            // Continuation: append to the previous logical line.
            let toks = tokenize(rest);
            // A `+` continuation with no previous logical line has nothing to
            // continue; drop it entirely rather than turning its tokens into a
            // bogus standalone element (or pushing an empty-token Line).
            if let Some(last) = out.last_mut() {
                last.tokens.extend(toks);
            }
            continue;
        }
        // 1-based column of the first token: leading whitespace in the
        // comment-stripped line, which preserves the original indentation.
        let col = content.chars().take_while(|c| c.is_whitespace()).count() + 1;
        let tokens = tokenize(trimmed);
        if !tokens.is_empty() {
            out.push(Line { no, col, tokens });
        }
    }
    // Coalesce spaced `key = value` (and `key=`/`=value`) into single
    // `key=value` tokens, so parameter binding is robust to the spacing a deck
    // happens to use (the BSIM4 modelcards format params as `ua      = 5e-11`).
    // Done after continuation merge so an `=` may even span a `+` boundary.
    for line in &mut out {
        line.tokens = coalesce_eq(std::mem::take(&mut line.tokens));
    }
    // Invariant for all downstream consumers: every logical line has >=1 token.
    out.retain(|line| !line.tokens.is_empty());
    out
}

/// Merge `key`, `=`, `value` (in any spaced arrangement) into one `key=value`
/// token. Bracketed groups (`{..}`, `(..)`) are already single tokens and are
/// left untouched (they never start or end with a bare `=`).
fn coalesce_eq(tokens: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if t == "=" {
            let prev = out.pop().unwrap_or_default();
            let next = tokens.get(i + 1).cloned().unwrap_or_default();
            out.push(format!("{prev}={next}"));
            i += 2;
        } else if t.len() > 1 && t.ends_with('=') && !t.ends_with("}") {
            let next = tokens.get(i + 1).cloned().unwrap_or_default();
            out.push(format!("{t}{next}"));
            i += 2;
        } else if t.len() > 1 && t.starts_with('=') {
            let prev = out.pop().unwrap_or_default();
            out.push(format!("{prev}{t}"));
            i += 1;
        } else {
            out.push(t.clone());
            i += 1;
        }
    }
    out
}

/// Remove comments from one physical line. Returns the code portion.
fn strip_comment(raw: &str) -> String {
    if raw.trim_start().starts_with('*') {
        return String::new(); // full-line comment
    }
    let mut out = String::new();
    let mut depth: i32 = 0;
    for c in raw.chars() {
        match c {
            '{' => {
                depth += 1;
                out.push(c);
            }
            '}' => {
                depth -= 1;
                out.push(c);
            }
            '$' | ';' if depth <= 0 => break, // inline comment
            _ => out.push(c),
        }
    }
    out
}

/// Whitespace-tokenise, keeping `{...}` and `(...)` groups intact (so brace
/// expressions and function-like values such as `SIN(0 1 1k)` stay one token).
fn tokenize(s: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    let mut depth: u32 = 0;
    for c in s.chars() {
        match c {
            '{' | '(' => {
                depth += 1;
                cur.push(c);
            }
            '}' | ')' => {
                depth = depth.saturating_sub(1);
                cur.push(c);
            }
            c if c.is_whitespace() && depth == 0 => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    toks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_comments_and_blanks() {
        let lines = preprocess("* title\n\nR1 1 2 1k ; inline\nC1 2 0 1u $ also\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].tokens, vec!["R1", "1", "2", "1k"]);
        assert_eq!(lines[1].tokens, vec!["C1", "2", "0", "1u"]);
    }

    #[test]
    fn joins_continuations() {
        let lines = preprocess(".model M NPN\n+ Is=1e-15\n+ Bf=100\n");
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0].tokens,
            vec![".model", "M", "NPN", "Is=1e-15", "Bf=100"]
        );
    }

    #[test]
    fn groups_brace_expressions() {
        let lines = preprocess("R1 1 2 {R0 * 2}\n");
        assert_eq!(lines[0].tokens, vec!["R1", "1", "2", "{R0 * 2}"]);
    }
}
