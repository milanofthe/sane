//! Token kinds for the Verilog-A lexer.

use crate::error::Span;

#[derive(Clone, PartialEq, Debug)]
pub enum Tok {
    /// Identifier or keyword (the parser distinguishes keywords by text).
    Ident(String),
    /// Numeric literal, already scaled (SI suffix / exponent folded in).
    Number(f64),
    /// String literal (stored without the surrounding quotes).
    Str(String),
    /// System identifier, e.g. `$temperature`, `$vt` (stored without the `$`).
    SysId(String),
    /// Compiler directive, e.g. `` `define`` (stored without the backtick).
    Directive(String),

    LParen,
    RParen,
    LBrace,
    RBrace,
    LBrack,
    RBrack,
    Comma,
    Semi,
    Colon,
    Dot,
    At,

    Assign,  // =
    Contrib, // <+

    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Pow, // **

    Lt,
    Gt,
    Le,
    Ge,
    EqEq,
    Ne,

    AndAnd,
    OrOr,
    Not, // !
    Question,

    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
    /// A hard (non-`\`-continued) newline preceded this token. The preprocessor
    /// uses this to bound a `\`define` body to its logical line.
    pub nl_before: bool,
}

impl Token {
    pub fn new(tok: Tok, span: Span) -> Self {
        Self {
            tok,
            span,
            nl_before: false,
        }
    }
}
