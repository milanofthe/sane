//! Token-level Verilog-A preprocessor: `\`include`, `\`define` (object- and
//! function-like macros), `\`ifdef`/`\`ifndef`/`\`elsif`/`\`else`/`\`endif`,
//! `\`undef`. Operates on the lexer's token stream (not raw text); macro and
//! directive invocations are themselves backtick tokens (`Tok::Directive`), so
//! directives and macro calls are disambiguated by name.
//!
//! `\`include` resolves built-in standard headers (disciplines.vams,
//! constants.vams) first, then the provided search directories. A `\`define`
//! body extends to the end of its logical line (tracked via `Token::nl_before`,
//! so `\`-continued lines fold correctly).

use rustc_hash::FxHashMap as HashMap;
use std::path::PathBuf;

use sane_core::constants::PREPROCESSOR_MAX_DEPTH as MAX_DEPTH;

use crate::error::{Diagnostic, Diagnostics, SourceMap, Span};
use crate::lexer::lex;
use crate::token::{Tok, Token};

struct Macro {
    params: Option<Vec<String>>,
    /// Body tokens keep the spans (and contexts) of their definition site.
    body: Vec<Token>,
}

struct Cond {
    /// Emitting in this branch?
    active: bool,
    /// Has any branch of this if/elsif/else chain been taken yet?
    taken: bool,
    /// Was the enclosing region active (so `\`else` can re-enable)?
    parent_active: bool,
}

pub struct Preprocessor {
    macros: HashMap<String, Macro>,
    builtins: HashMap<&'static str, &'static str>,
    search_dirs: Vec<PathBuf>,
    cond: Vec<Cond>,
    out: Vec<Token>,
    /// Expansion contexts (root file, includes, macro frames) for diagnostics.
    map: SourceMap,
}

/// Preprocess `src` (named `file`), resolving includes from `search_dirs` plus
/// the built-in standard headers. `predef` names are pre-defined empty macros
/// (e.g. `__VAMS_COMPACT_MODELING__`).
/// Preprocess, additionally returning the [`SourceMap`] of expansion
/// contexts so diagnostics on the token stream can render the true source
/// line of every token plus its include/macro expansion chain.
pub fn preprocess_mapped(
    src: &str,
    file: &str,
    search_dirs: &[PathBuf],
    predef: &[&str],
) -> Result<(Vec<Token>, std::sync::Arc<SourceMap>), Diagnostics> {
    let mut builtins: HashMap<&'static str, &'static str> = HashMap::default();
    builtins.insert("constants.vams", include_str!("headers/constants.vams"));
    builtins.insert("disciplines.vams", include_str!("headers/disciplines.vams"));
    // common alternate spellings used by some model headers
    builtins.insert("constants.h", include_str!("headers/constants.vams"));
    builtins.insert("discipline.h", include_str!("headers/disciplines.vams"));
    builtins.insert("disciplines.h", include_str!("headers/disciplines.vams"));

    let mut pp = Preprocessor {
        macros: HashMap::default(),
        builtins,
        search_dirs: search_dirs.to_vec(),
        cond: Vec::new(),
        out: Vec::new(),
        map: SourceMap::root(file, src),
    };
    for name in predef {
        pp.macros.insert(
            (*name).to_string(),
            Macro {
                params: None,
                body: Vec::new(),
            },
        );
    }
    let fail = |pp: &Preprocessor, d: Diagnostic| {
        Diagnostics::one(file, d).with_map(std::sync::Arc::new(pp.map.clone()))
    };
    let toks = lex(src, file).map_err(|d| Diagnostics::one(file, d))?;
    // Root tokens carry context 0 (the default), which is the root file.
    if let Err(d) = pp.process(&toks, file, 0) {
        return Err(fail(&pp, d));
    }
    if !pp.cond.is_empty() {
        return Err(fail(
            &pp,
            Diagnostic::new(
                "unterminated `ifdef/`ifndef (missing `endif)",
                Span::default(),
            ),
        ));
    }
    pp.out.push(Token::new(Tok::Eof, Span::default()));
    Ok((pp.out, std::sync::Arc::new(pp.map)))
}

impl Preprocessor {
    fn active(&self) -> bool {
        self.cond.last().is_none_or(|c| c.active)
    }

    fn process(&mut self, toks: &[Token], file: &str, depth: usize) -> Result<(), Diagnostic> {
        if depth > MAX_DEPTH {
            return Err(Diagnostic::new(
                "macro/include expansion too deep (cycle?)",
                Span::default(),
            ));
        }
        let mut i = 0;
        while i < toks.len() {
            let t = &toks[i];
            match &t.tok {
                Tok::Eof => i += 1,
                Tok::Directive(name) => {
                    let name = name.clone();
                    match name.as_str() {
                        "ifdef" | "ifndef" => i = self.do_ifdef(toks, i, name == "ifdef")?,
                        "elsif" => i = self.do_elsif(toks, i)?,
                        "else" => {
                            self.do_else(t)?;
                            i += 1;
                        }
                        "endif" => {
                            self.do_endif(t)?;
                            i += 1;
                        }
                        _ if !self.active() => i += 1,
                        "define" => i = self.do_define(toks, i)?,
                        "undef" => i = self.do_undef(toks, i),
                        "include" => i = self.do_include(toks, i, depth)?,
                        // Irrelevant / unsupported line directives: skip the line.
                        "resetall" | "celldefine" | "endcelldefine" | "timescale"
                        | "default_discipline" | "default_nodetype" | "default_transition"
                        | "begin_keywords" | "end_keywords" | "pragma" | "line" => {
                            i = skip_line(toks, i + 1)
                        }
                        _ => i = self.expand_call(toks, i, &name, file, depth)?,
                    }
                }
                _ => {
                    if self.active() {
                        self.out.push(t.clone());
                    }
                    i += 1;
                }
            }
        }
        Ok(())
    }

    // --- conditionals ------------------------------------------------------

    fn do_ifdef(&mut self, toks: &[Token], i: usize, want: bool) -> Result<usize, Diagnostic> {
        let parent_active = self.active();
        let name = ident_at(toks, i + 1, "`ifdef/`ifndef requires a macro name")?;
        let defined = self.macros.contains_key(&name);
        let active = parent_active && (defined == want);
        self.cond.push(Cond {
            active,
            taken: active,
            parent_active,
        });
        Ok(i + 2)
    }

    fn do_elsif(&mut self, toks: &[Token], i: usize) -> Result<usize, Diagnostic> {
        let name = ident_at(toks, i + 1, "`elsif requires a macro name")?;
        let c = self
            .cond
            .last_mut()
            .ok_or_else(|| Diagnostic::new("`elsif without `ifdef", toks[i].span))?;
        let take = c.parent_active && !c.taken && self.macros.contains_key(&name);
        // borrow re-check: recompute defined before mutability above is fine since
        // macros not mutated here.
        c.active = take;
        if take {
            c.taken = true;
        }
        Ok(i + 2)
    }

    fn do_else(&mut self, t: &Token) -> Result<(), Diagnostic> {
        let c = self
            .cond
            .last_mut()
            .ok_or_else(|| Diagnostic::new("`else without `ifdef", t.span))?;
        c.active = c.parent_active && !c.taken;
        c.taken = true;
        Ok(())
    }

    fn do_endif(&mut self, t: &Token) -> Result<(), Diagnostic> {
        self.cond
            .pop()
            .ok_or_else(|| Diagnostic::new("`endif without `ifdef", t.span))?;
        Ok(())
    }

    // --- define / undef ----------------------------------------------------

    fn do_define(&mut self, toks: &[Token], i: usize) -> Result<usize, Diagnostic> {
        let name_tok = toks
            .get(i + 1)
            .ok_or_else(|| Diagnostic::new("`define requires a name", toks[i].span))?;
        let name = match &name_tok.tok {
            Tok::Ident(s) => s.clone(),
            _ => {
                return Err(Diagnostic::new(
                    "`define name must be an identifier",
                    name_tok.span,
                ))
            }
        };
        let mut j = i + 2;
        // Function-like macro: `(` immediately adjacent to the name (no space).
        let mut params: Option<Vec<String>> = None;
        if let Some(lp) = toks.get(j) {
            if matches!(lp.tok, Tok::LParen)
                && lp.span.offset == name_tok.span.offset + name.len() as u32
            {
                let (ps, next) = self.parse_param_list(toks, j)?;
                params = Some(ps);
                j = next;
            }
        }
        // Body: tokens to end of logical line.
        let start = j;
        while j < toks.len()
            && !matches!(toks[j].tok, Tok::Eof)
            && !(j > start && toks[j].nl_before)
        {
            // also stop if the very first body token starts a new line (empty body)
            if j == start && toks[j].nl_before {
                break;
            }
            j += 1;
        }
        let body = toks[start..j].to_vec();
        self.macros.insert(name, Macro { params, body });
        Ok(j)
    }

    fn parse_param_list(
        &self,
        toks: &[Token],
        lp: usize,
    ) -> Result<(Vec<String>, usize), Diagnostic> {
        // lp points at `(`
        let mut params = Vec::new();
        let mut j = lp + 1;
        loop {
            match toks.get(j).map(|t| &t.tok) {
                Some(Tok::RParen) => {
                    j += 1;
                    break;
                }
                Some(Tok::Ident(s)) => {
                    params.push(s.clone());
                    j += 1;
                    match toks.get(j).map(|t| &t.tok) {
                        Some(Tok::Comma) => j += 1,
                        Some(Tok::RParen) => {
                            j += 1;
                            break;
                        }
                        _ => {
                            return Err(Diagnostic::new(
                                "expected ',' or ')' in macro parameters",
                                toks[lp].span,
                            ))
                        }
                    }
                }
                _ => {
                    return Err(Diagnostic::new(
                        "malformed macro parameter list",
                        toks[lp].span,
                    ))
                }
            }
        }
        Ok((params, j))
    }

    fn do_undef(&mut self, toks: &[Token], i: usize) -> usize {
        if let Some(Tok::Ident(s)) = toks.get(i + 1).map(|t| &t.tok) {
            self.macros.remove(s);
        }
        skip_line(toks, i + 1)
    }

    // --- include -----------------------------------------------------------

    fn do_include(&mut self, toks: &[Token], i: usize, depth: usize) -> Result<usize, Diagnostic> {
        let path_tok = toks
            .get(i + 1)
            .ok_or_else(|| Diagnostic::new("`include requires a file name", toks[i].span))?;
        let path = match &path_tok.tok {
            Tok::Str(s) => s.clone(),
            _ => {
                return Err(Diagnostic::new(
                    "`include path must be a string",
                    path_tok.span,
                ))
            }
        };
        let next = skip_line(toks, i + 2);
        let base = path.rsplit(['/', '\\']).next().unwrap_or(&path);
        let src = if let Some(s) = self.builtins.get(base) {
            (*s).to_string()
        } else if let Some(found) = self.find_include(&path) {
            found
        } else {
            // Unknown include (often a simulator-private header): skip rather than
            // fail, so models that guard real content elsewhere still parse -- but
            // surface it as a captured (catchable) warning instead of dropping it
            // silently, since a missing header can remove real model content (#41).
            sane_core::log::warn_captured(&format!(
                "unresolved `include \"{path}\" skipped; any model content it provides is absent"
            ));
            return Ok(next);
        };
        let mut inc = lex(&src, &path)
            .map_err(|d| Diagnostic::new(format!("in include {path}: {}", d.message), d.span))?;
        // Register the include as an expansion context and stamp its tokens,
        // so their spans render against the include's own source with a
        // "included from here" note pointing at this directive.
        let ctx = self.map.push_file(&path, &src, toks[i].span);
        for t in &mut inc {
            t.span.ctx = ctx;
        }
        self.process(&inc, &path, depth + 1)?;
        Ok(next)
    }

    fn find_include(&self, path: &str) -> Option<String> {
        let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
        for dir in &self.search_dirs {
            for cand in [dir.join(path), dir.join(base)] {
                if let Ok(s) = std::fs::read_to_string(&cand) {
                    return Some(s);
                }
            }
        }
        None
    }

    // --- macro expansion ---------------------------------------------------

    fn expand_call(
        &mut self,
        toks: &[Token],
        i: usize,
        name: &str,
        file: &str,
        depth: usize,
    ) -> Result<usize, Diagnostic> {
        let span = toks[i].span;
        let (params, body) = match self.macros.get(name) {
            Some(m) => (m.params.clone(), m.body.clone()),
            None => {
                return Err(Diagnostic::new(format!("undefined macro `{name}"), span));
            }
        };
        let mut next = i + 1;
        let mut expanded: Vec<Token> = match params {
            None => body,
            Some(ps) => {
                // Collect actual argument token-lists from `(...)`.
                let lp = toks.get(next);
                if !matches!(lp.map(|t| &t.tok), Some(Tok::LParen)) {
                    return Err(Diagnostic::new(
                        format!("macro `{name} expects arguments"),
                        span,
                    ));
                }
                let (args, after) = collect_args(toks, next, span)?;
                next = after;
                if args.len() != ps.len() {
                    return Err(Diagnostic::new(
                        format!(
                            "macro `{name} expects {} args, got {}",
                            ps.len(),
                            args.len()
                        ),
                        span,
                    ));
                }
                substitute(&body, &ps, &args)
            }
        };
        // Give the body tokens a macro frame pointing at this invocation
        // (argument tokens keep their caller context: they appear verbatim at
        // the call site). One frame per distinct origin context, so nested
        // expansions chain naturally.
        let mut frame_of: HashMap<u32, u32> = HashMap::default();
        for t in &mut expanded {
            // Tokens originating at the invocation itself (substituted args)
            // are recognizable by their context matching the call site's.
            if t.span.ctx == span.ctx && t.span.offset >= span.offset {
                continue;
            }
            let frame = *frame_of
                .entry(t.span.ctx)
                .or_insert_with(|| self.map.push_macro(name, span, t.span.ctx));
            t.span.ctx = frame;
        }
        self.process(&expanded, file, depth + 1)?;
        Ok(next)
    }
}

// --- helpers ---------------------------------------------------------------

fn ident_at(toks: &[Token], i: usize, msg: &str) -> Result<String, Diagnostic> {
    match toks.get(i).map(|t| &t.tok) {
        Some(Tok::Ident(s)) => Ok(s.clone()),
        _ => Err(Diagnostic::new(
            msg,
            toks.get(i)
                .or_else(|| toks.last())
                .map(|t| t.span)
                .unwrap_or_default(),
        )),
    }
}

/// Index of the next token on a following line (or end), from `i`.
fn skip_line(toks: &[Token], i: usize) -> usize {
    let mut j = i;
    while j < toks.len() && !matches!(toks[j].tok, Tok::Eof) && !toks[j].nl_before {
        j += 1;
    }
    j
}

/// Collect comma-separated, paren-balanced argument token lists. `lp` points at
/// the opening `(`. Returns the args and the index just past the closing `)`.
fn collect_args(
    toks: &[Token],
    lp: usize,
    span: Span,
) -> Result<(Vec<Vec<Token>>, usize), Diagnostic> {
    let mut args: Vec<Vec<Token>> = Vec::new();
    let mut cur: Vec<Token> = Vec::new();
    let mut depth = 0i32;
    let mut j = lp;
    loop {
        let t = toks
            .get(j)
            .ok_or_else(|| Diagnostic::new("unterminated macro argument list", span))?;
        match &t.tok {
            Tok::Eof => return Err(Diagnostic::new("unterminated macro argument list", span)),
            Tok::LParen => {
                depth += 1;
                if depth > 1 {
                    cur.push(t.clone());
                }
                j += 1;
            }
            Tok::RParen => {
                depth -= 1;
                if depth == 0 {
                    if !cur.is_empty() || !args.is_empty() {
                        args.push(std::mem::take(&mut cur));
                    }
                    j += 1;
                    break;
                }
                cur.push(t.clone());
                j += 1;
            }
            Tok::Comma if depth == 1 => {
                args.push(std::mem::take(&mut cur));
                j += 1;
            }
            _ => {
                cur.push(t.clone());
                j += 1;
            }
        }
    }
    Ok((args, j))
}

/// Substitute parameter identifiers in `body` with the corresponding argument
/// token lists. Non-parameter tokens are copied verbatim.
fn substitute(body: &[Token], params: &[String], args: &[Vec<Token>]) -> Vec<Token> {
    let mut out = Vec::new();
    for t in body {
        if let Tok::Ident(s) = &t.tok {
            if let Some(idx) = params.iter().position(|p| p == s) {
                out.extend(args[idx].iter().cloned());
                continue;
            }
        }
        out.push(t.clone());
    }
    out
}
