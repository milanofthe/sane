//! Source positions and diagnostics for the Verilog-A frontend.
//!
//! Every token and AST node carries a [`Span`]; parse / elaboration errors are
//! [`Diagnostic`]s that render the offending source line with a caret, so the
//! user sees exactly where a model failed to compile.

use std::fmt;

/// A 1-based source position (and 0-based byte offset for slicing). `ctx`
/// names the expansion context the position refers to (a [`SourceMap`] index:
/// the root file, an `` `include ``d file, or a macro expansion); `0` is the
/// root file, so plain spans keep working without a map.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Span {
    pub line: u32,
    pub col: u32,
    pub offset: u32,
    pub ctx: u32,
}

impl Span {
    pub fn new(line: u32, col: u32, offset: u32) -> Self {
        Self {
            line,
            col,
            offset,
            ctx: 0,
        }
    }
}

/// One expansion context a [`Span`] can refer to.
#[derive(Clone, Debug)]
pub enum SpanCtx {
    /// A source file: the root, or an `` `include ``d file (then `included_at`
    /// points at the `` `include `` directive in its parent context).
    File {
        name: String,
        src: String,
        included_at: Option<Span>,
    },
    /// A macro expansion: the expanded tokens keep their definition-site
    /// line/col (which resolve through `def_ctx`); `invoked_at` points at the
    /// backtick call in its parent context.
    Macro {
        name: String,
        invoked_at: Span,
        def_ctx: u32,
    },
}

/// Table of expansion contexts, built by the preprocessor. Lets a diagnostic
/// render the true source line of a token that came out of an include or a
/// macro body, followed by the whole expansion chain -- the difference between
/// "somewhere in psp103.va line 12" and the actual macro body plus every
/// invocation site down to the deck.
#[derive(Clone, Debug, Default)]
pub struct SourceMap {
    ctxs: Vec<SpanCtx>,
}

impl SourceMap {
    /// Register the root file as context 0.
    pub fn root(name: &str, src: &str) -> SourceMap {
        SourceMap {
            ctxs: vec![SpanCtx::File {
                name: name.to_string(),
                src: src.to_string(),
                included_at: None,
            }],
        }
    }

    pub fn push_file(&mut self, name: &str, src: &str, included_at: Span) -> u32 {
        self.ctxs.push(SpanCtx::File {
            name: name.to_string(),
            src: src.to_string(),
            included_at: Some(included_at),
        });
        (self.ctxs.len() - 1) as u32
    }

    pub fn push_macro(&mut self, name: &str, invoked_at: Span, def_ctx: u32) -> u32 {
        self.ctxs.push(SpanCtx::Macro {
            name: name.to_string(),
            invoked_at,
            def_ctx,
        });
        (self.ctxs.len() - 1) as u32
    }

    pub fn ctx(&self, id: u32) -> Option<&SpanCtx> {
        self.ctxs.get(id as usize)
    }

    /// The file (name, source) a span's line/col actually refer to: macro
    /// contexts resolve through their definition site.
    pub fn file_of(&self, mut id: u32) -> Option<(&str, &str)> {
        loop {
            match self.ctxs.get(id as usize)? {
                SpanCtx::File { name, src, .. } => return Some((name, src)),
                SpanCtx::Macro { def_ctx, .. } => id = *def_ctx,
            }
        }
    }
}

/// A single compile error, located at a span.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
}

impl Diagnostic {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

/// One or more diagnostics, carrying the source file name for rendering and,
/// when the preprocessor ran, the [`SourceMap`] for expansion-chain rendering.
#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub file: String,
    pub items: Vec<Diagnostic>,
    /// Expansion contexts for the spans in `items` (set by the preprocessing
    /// pipeline; `None` for bare lexer/hand-built diagnostics).
    pub map: Option<std::sync::Arc<SourceMap>>,
}

impl Diagnostics {
    pub fn one(file: &str, d: Diagnostic) -> Self {
        Self {
            file: file.to_string(),
            items: vec![d],
            map: None,
        }
    }

    pub fn with_map(mut self, map: std::sync::Arc<SourceMap>) -> Self {
        self.map = Some(map);
        self
    }

    /// Render each diagnostic as `file:line:col: message` followed by the source
    /// line and a caret under the offending column, then -- with a source map --
    /// one note per expansion-chain frame (macro invocation sites, `` `include ``
    /// directives), each with its own caret. Falls back to `source` (the root
    /// text) when no map is attached. Delegates to the shared caret renderer in
    /// `sane_core::diag` so the netlist parser produces the same presentation.
    pub fn render(&self, source: &str) -> String {
        let mut out = String::new();
        for d in &self.items {
            match &self.map {
                Some(map) => render_chained(&mut out, map, d),
                None => sane_core::diag::render_snippet(
                    &mut out,
                    source,
                    &self.file,
                    d.span.line,
                    d.span.col,
                    &d.message,
                ),
            }
        }
        out
    }
}

/// Render one diagnostic with its expansion chain (see [`Diagnostics::render`]).
fn render_chained(out: &mut String, map: &SourceMap, d: &Diagnostic) {
    let (fname, src) = map.file_of(d.span.ctx).unwrap_or(("<unknown>", ""));
    sane_core::diag::render_snippet(out, src, fname, d.span.line, d.span.col, &d.message);
    let mut ctx = d.span.ctx;
    // Walk outward: macro frames to their invocation sites, included files to
    // their `include` directives, until the root file.
    let mut hops = 0;
    while hops < 32 {
        hops += 1;
        match map.ctx(ctx) {
            Some(SpanCtx::Macro {
                name, invoked_at, ..
            }) => {
                let (f, s) = map.file_of(invoked_at.ctx).unwrap_or(("<unknown>", ""));
                sane_core::diag::render_snippet(
                    out,
                    s,
                    f,
                    invoked_at.line,
                    invoked_at.col,
                    &format!("note: in expansion of `{name}"),
                );
                ctx = invoked_at.ctx;
            }
            Some(SpanCtx::File {
                included_at: Some(at),
                ..
            }) => {
                let (f, s) = map.file_of(at.ctx).unwrap_or(("<unknown>", ""));
                sane_core::diag::render_snippet(
                    out,
                    s,
                    f,
                    at.line,
                    at.col,
                    "note: included from here",
                );
                ctx = at.ctx;
            }
            _ => break,
        }
    }
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for d in &self.items {
            // Resolve the true file through the map when present; append a
            // compact expansion trail so even the plain Display names the chain.
            let file: &str = self
                .map
                .as_ref()
                .and_then(|m| m.file_of(d.span.ctx))
                .map(|(n, _)| n)
                .unwrap_or(&self.file);
            write!(f, "{}:{}:{}: {}", file, d.span.line, d.span.col, d.message)?;
            if let Some(map) = &self.map {
                let mut ctx = d.span.ctx;
                let mut hops = 0;
                while hops < 32 {
                    hops += 1;
                    match map.ctx(ctx) {
                        Some(SpanCtx::Macro {
                            name, invoked_at, ..
                        }) => {
                            let f2 = map
                                .file_of(invoked_at.ctx)
                                .map(|(n, _)| n)
                                .unwrap_or("<unknown>");
                            write!(
                                f,
                                " (in expansion of `{name} at {f2}:{}:{})",
                                invoked_at.line, invoked_at.col
                            )?;
                            ctx = invoked_at.ctx;
                        }
                        Some(SpanCtx::File {
                            included_at: Some(at),
                            ..
                        }) => {
                            let f2 = map.file_of(at.ctx).map(|(n, _)| n).unwrap_or("<unknown>");
                            write!(f, " (included from {f2}:{}:{})", at.line, at.col)?;
                            ctx = at.ctx;
                        }
                        _ => break,
                    }
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
