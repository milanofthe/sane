//! Hand-rolled scanner for Verilog-A source.
//!
//! Produces a flat `Vec<Token>` (terminated by `Eof`) with 1-based line/column
//! spans. Handles `//` and `/* */` comments, real/integer literals with SI
//! scale suffixes and exponents, system identifiers (`$...`), compiler
//! directives (`` `... ``), strings, and the analog operator set.

use crate::error::{Diagnostic, Span};
use crate::token::{Tok, Token};

pub fn lex(src: &str, file: &str) -> Result<Vec<Token>, Diagnostic> {
    Lexer::new(src, file).run()
}

struct Lexer<'a> {
    src: &'a [u8],
    text: &'a str,
    pos: usize,
    line: u32,
    col: u32,
    /// A hard newline has been seen since the last emitted token.
    pending_nl: bool,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str, _file: &str) -> Self {
        Self {
            src: src.as_bytes(),
            text: src,
            pos: 0,
            line: 1,
            col: 1,
            pending_nl: false,
        }
    }

    fn span(&self) -> Span {
        Span::new(self.line, self.col, self.pos as u32)
    }

    fn peek(&self) -> u8 {
        *self.src.get(self.pos).unwrap_or(&0)
    }
    fn peek2(&self) -> u8 {
        *self.src.get(self.pos + 1).unwrap_or(&0)
    }

    fn bump(&mut self) -> u8 {
        let c = self.peek();
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        c
    }

    fn run(mut self) -> Result<Vec<Token>, Diagnostic> {
        let mut out = Vec::new();
        loop {
            self.skip_trivia();
            let span = self.span();
            let c = self.peek();
            if c == 0 {
                out.push(Token::new(Tok::Eof, span));
                return Ok(out);
            }
            let tok = if c.is_ascii_alphabetic() || c == b'_' {
                self.ident()
            } else if c.is_ascii_digit() || (c == b'.' && self.peek2().is_ascii_digit()) {
                self.number(span)?
            } else if c == b'"' {
                self.string(span)?
            } else if c == b'$' {
                self.sys_id()
            } else if c == b'`' {
                self.directive()
            } else if c == b'\\' {
                self.escaped_ident()
            } else {
                self.operator(span)?
            };
            let mut tok = tok;
            tok.nl_before = self.pending_nl;
            self.pending_nl = false;
            out.push(tok);
        }
    }

    fn skip_trivia(&mut self) {
        loop {
            let c = self.peek();
            if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                if c == b'\n' {
                    self.pending_nl = true;
                }
                self.bump();
            } else if c == b'\\' && (self.peek2() == b'\n' || self.peek2() == b'\r') {
                // Line continuation `\<newline>`: fold the backslash AND the line
                // break away (without marking a hard newline).
                self.bump(); // backslash
                if self.peek() == b'\r' {
                    self.bump();
                }
                if self.peek() == b'\n' {
                    self.bump();
                }
            } else if c == b'/' && self.peek2() == b'/' {
                while self.peek() != b'\n' && self.peek() != 0 {
                    self.bump();
                }
            } else if c == b'/' && self.peek2() == b'*' {
                self.bump();
                self.bump();
                while !(self.peek() == b'*' && self.peek2() == b'/') && self.peek() != 0 {
                    self.bump();
                }
                self.bump();
                self.bump();
            } else {
                break;
            }
        }
    }

    fn ident(&mut self) -> Token {
        let span = self.span();
        let start = self.pos;
        while {
            let c = self.peek();
            c.is_ascii_alphanumeric() || c == b'_' || c == b'$'
        } {
            self.bump();
        }
        let s = self.text[start..self.pos].to_string();
        Token::new(Tok::Ident(s), span)
    }

    /// Escaped identifier: `\` then a run of non-whitespace, terminated by
    /// whitespace. The leading backslash is dropped; the name is the run.
    fn escaped_ident(&mut self) -> Token {
        let span = self.span();
        self.bump(); // backslash
        let start = self.pos;
        while {
            let c = self.peek();
            c != 0 && c != b' ' && c != b'\t' && c != b'\r' && c != b'\n'
        } {
            self.bump();
        }
        Token::new(Tok::Ident(self.text[start..self.pos].to_string()), span)
    }

    fn number(&mut self, span: Span) -> Result<Token, Diagnostic> {
        let start = self.pos;
        while self.peek().is_ascii_digit() {
            self.bump();
        }
        if self.peek() == b'.' {
            self.bump();
            while self.peek().is_ascii_digit() {
                self.bump();
            }
        }
        if self.peek() == b'e' || self.peek() == b'E' {
            self.bump();
            if self.peek() == b'+' || self.peek() == b'-' {
                self.bump();
            }
            while self.peek().is_ascii_digit() {
                self.bump();
            }
        }
        let mantissa: f64 = self.text[start..self.pos]
            .parse()
            .map_err(|_| Diagnostic::new("malformed number literal", span))?;
        // Optional SI scale suffix (case-sensitive, Verilog-AMS LRM): a letter
        // immediately following the digits. `M` = mega, `m` = milli.
        let scale = match self.peek() {
            b'T' => Some(1e12),
            b'G' => Some(1e9),
            b'M' => Some(1e6),
            b'K' | b'k' => Some(1e3),
            b'm' => Some(1e-3),
            b'u' => Some(1e-6),
            b'n' => Some(1e-9),
            b'p' => Some(1e-12),
            b'f' => Some(1e-15),
            b'a' => Some(1e-18),
            _ => None,
        };
        let value = if let Some(s) = scale {
            // Only consume the suffix if not part of a longer identifier-like run.
            let after = *self.src.get(self.pos + 1).unwrap_or(&0);
            if after.is_ascii_alphanumeric() || after == b'_' {
                mantissa
            } else {
                self.bump();
                mantissa * s
            }
        } else {
            mantissa
        };
        Ok(Token::new(Tok::Number(value), span))
    }

    fn string(&mut self, span: Span) -> Result<Token, Diagnostic> {
        self.bump(); // opening quote
                     // Scan with backslash-escape handling (LRM 2.4.0 §2.7.2): `\"` is a quote
                     // *inside* the string, not its terminator. Without this, an escaped quote
                     // in a `$strobe`/`$display` format string (common in BSIM6/BSIM-SOI) ends
                     // the string early and corrupts quote parity for the rest of the file.
        let mut s = String::new();
        loop {
            match self.peek() {
                0 => return Err(Diagnostic::new("unterminated string literal", span)),
                b'"' => break,
                b'\\' => {
                    self.bump(); // consume the backslash
                    let e = self.bump(); // and the escaped character
                    let ch = match e {
                        b'n' => '\n',
                        b't' => '\t',
                        0 => return Err(Diagnostic::new("unterminated string literal", span)),
                        // `\"`, `\\`, `\%` and any other escape keep the literal char.
                        other => other as char,
                    };
                    s.push(ch);
                }
                c => {
                    s.push(c as char);
                    self.bump();
                }
            }
        }
        self.bump(); // closing quote
        Ok(Token::new(Tok::Str(s), span))
    }

    fn sys_id(&mut self) -> Token {
        let span = self.span();
        self.bump(); // $
        let start = self.pos;
        while {
            let c = self.peek();
            c.is_ascii_alphanumeric() || c == b'_'
        } {
            self.bump();
        }
        Token::new(Tok::SysId(self.text[start..self.pos].to_string()), span)
    }

    fn directive(&mut self) -> Token {
        let span = self.span();
        self.bump(); // backtick
        let start = self.pos;
        while {
            let c = self.peek();
            c.is_ascii_alphanumeric() || c == b'_'
        } {
            self.bump();
        }
        Token::new(Tok::Directive(self.text[start..self.pos].to_string()), span)
    }

    fn operator(&mut self, span: Span) -> Result<Token, Diagnostic> {
        let c = self.bump();
        let d = self.peek();
        let two = |this: &mut Self, t: Tok| {
            this.bump();
            t
        };
        let tok = match c {
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b'{' => Tok::LBrace,
            b'}' => Tok::RBrace,
            b'[' => Tok::LBrack,
            b']' => Tok::RBrack,
            b',' => Tok::Comma,
            b';' => Tok::Semi,
            b':' => Tok::Colon,
            b'.' => Tok::Dot,
            b'@' => Tok::At,
            b'?' => Tok::Question,
            b'+' => Tok::Plus,
            b'-' => Tok::Minus,
            b'%' => Tok::Percent,
            b'*' => {
                if d == b'*' {
                    two(self, Tok::Pow)
                } else {
                    Tok::Star
                }
            }
            b'/' => Tok::Slash,
            b'<' => match d {
                b'+' => two(self, Tok::Contrib),
                b'=' => two(self, Tok::Le),
                _ => Tok::Lt,
            },
            b'>' => {
                if d == b'=' {
                    two(self, Tok::Ge)
                } else {
                    Tok::Gt
                }
            }
            b'=' => {
                if d == b'=' {
                    two(self, Tok::EqEq)
                } else {
                    Tok::Assign
                }
            }
            b'!' => {
                if d == b'=' {
                    two(self, Tok::Ne)
                } else {
                    Tok::Not
                }
            }
            b'&' if d == b'&' => two(self, Tok::AndAnd),
            b'|' if d == b'|' => two(self, Tok::OrOr),
            other => {
                return Err(Diagnostic::new(
                    format!("unexpected character '{}'", other as char),
                    span,
                ))
            }
        };
        Ok(Token::new(tok, span))
    }
}

#[cfg(test)]
mod tests {
    use super::lex;
    use crate::token::Tok;

    /// An escaped quote inside a string is content, not a terminator — so the
    /// string ends at the real closing quote and the rest of the file keeps its
    /// quote parity (the BSIM6 / BSIM-SOI `$strobe(...)` case).
    #[test]
    fn string_handles_escaped_quote() {
        let toks = lex(r#"$strobe("say \"hi\" now"); x"#, "t.va").expect("lex");
        let s = toks
            .iter()
            .find_map(|t| {
                if let Tok::Str(s) = &t.tok {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .expect("a string token");
        assert_eq!(s, "say \"hi\" now");
        // Lexing continued past the string: the trailing identifier is present.
        assert!(toks
            .iter()
            .any(|t| matches!(&t.tok, Tok::Ident(i) if i == "x")));
    }
}
