//! A small arithmetic expression evaluator for `.param` definitions and
//! `{...}` value expressions. Evaluates to a concrete `f64` against a parameter
//! environment (numbers, not symbols); the element's symbolic identity stays
//! its name. Supports `+ - * / ^`, unary minus, parentheses, engineering-suffix
//! numbers, named parameters/constants, and common functions.

use rustc_hash::FxHashMap as HashMap;

use crate::parse_value;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Tok {
    Num(f64),
    Ident(String),
    Op(char),
    /// A comparison operator: `<`, `>`, `<=`, `>=`, `==`, `!=`. Used by both the
    /// behavioral (`B`) source parser and the `.param` evaluator.
    Cmp(String),
    /// A logical operator: `&&` or `||`.
    Logic(String),
    /// Logical negation `!`.
    Not,
    /// Ternary punctuation `?` and `:`.
    Question,
    Colon,
    LParen,
    RParen,
    Comma,
}

pub(crate) fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            // number: digits/dot/exponent, then optional engineering suffix
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                i += 1;
            }
            if i < b.len() && (b[i] == 'e' || b[i] == 'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == '+' || b[j] == '-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            while i < b.len() && b[i].is_ascii_alphabetic() {
                i += 1; // engineering suffix (k, u, meg, ...)
            }
            let num: String = b[start..i].iter().collect();
            let v = parse_value(&num).ok_or_else(|| format!("bad number '{num}'"))?;
            out.push(Tok::Num(v));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len()
                && (b[i].is_ascii_alphanumeric()
                    || b[i] == '_'
                    // Hierarchical node names from subckt flattening (`x1.r0`): a
                    // dot continues the identifier only when an ident char follows,
                    // so a decimal number (`.5`) or a trailing dot is unaffected.
                    || (b[i] == '.'
                        && i + 1 < b.len()
                        && (b[i + 1].is_ascii_alphanumeric() || b[i + 1] == '_')))
            {
                i += 1;
            }
            out.push(Tok::Ident(b[start..i].iter().collect()));
        } else if matches!(c, '<' | '>' | '=' | '!') {
            // Comparison (`<`,`>`,`<=`,`>=`,`==`,`!=`) or logical negation (`!`).
            let mut op = String::from(c);
            i += 1;
            if i < b.len() && b[i] == '=' {
                op.push('=');
                i += 1;
            }
            if op == "!" {
                out.push(Tok::Not);
            } else if op == "=" {
                return Err("incomplete operator '=' (use == for comparison)".into());
            } else {
                out.push(Tok::Cmp(op));
            }
        } else if matches!(c, '&' | '|') {
            // Logical `&&` / `||`; a single `&`/`|` is not valid in expressions.
            i += 1;
            if i < b.len() && b[i] == c {
                i += 1;
                out.push(Tok::Logic(format!("{c}{c}")));
            } else {
                return Err(format!("incomplete logical operator '{c}' (use {c}{c})"));
            }
        } else {
            i += 1;
            match c {
                '(' => out.push(Tok::LParen),
                ')' => out.push(Tok::RParen),
                ',' => out.push(Tok::Comma),
                '?' => out.push(Tok::Question),
                ':' => out.push(Tok::Colon),
                '+' | '-' | '*' | '/' | '^' => out.push(Tok::Op(c)),
                _ => return Err(format!("unexpected char '{c}'")),
            }
        }
    }
    Ok(out)
}

struct P<'a> {
    t: Vec<Tok>,
    pos: usize,
    env: &'a HashMap<String, f64>,
}

impl P<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let v = self.t.get(self.pos).cloned();
        if v.is_some() {
            self.pos += 1;
        }
        v
    }

    /// Top of the grammar: a ternary `cond ? a : b` (right-associative) over the
    /// comparison/logic/arithmetic expression below it.
    fn ternary(&mut self) -> Result<f64, String> {
        let c = self.expr(0)?;
        if self.peek() == Some(&Tok::Question) {
            self.pos += 1;
            let then = self.ternary()?;
            match self.next() {
                Some(Tok::Colon) => {}
                _ => return Err("expected ':' in ternary expression".into()),
            }
            let els = self.ternary()?;
            Ok(if c != 0.0 { then } else { els })
        } else {
            Ok(c)
        }
    }

    fn expr(&mut self, min_bp: u8) -> Result<f64, String> {
        let mut lhs = self.atom()?;
        loop {
            // Precedence: `||` < `&&` < comparison < `+ -` < `* /` < `^`.
            // Comparisons/logic yield 1.0 / 0.0.
            let (lbp, rbp, kind) = match self.peek() {
                Some(Tok::Logic(s)) => {
                    let bp = if s == "||" { (1, 2) } else { (3, 4) };
                    (bp.0, bp.1, OpKind::Logic(s.clone()))
                }
                Some(Tok::Cmp(s)) => (5, 6, OpKind::Cmp(s.clone())),
                Some(Tok::Op(op)) => {
                    let bp = match op {
                        '+' | '-' => (7, 8),
                        '*' | '/' => (9, 10),
                        '^' => (14, 13),
                        _ => break,
                    };
                    (bp.0, bp.1, OpKind::Arith(*op))
                }
                _ => break,
            };
            if lbp < min_bp {
                break;
            }
            self.pos += 1;
            let rhs = self.expr(rbp)?;
            lhs = match kind {
                OpKind::Arith('+') => lhs + rhs,
                OpKind::Arith('-') => lhs - rhs,
                OpKind::Arith('*') => lhs * rhs,
                OpKind::Arith('/') => lhs / rhs,
                OpKind::Arith('^') => lhs.powf(rhs),
                OpKind::Arith(_) => unreachable!(),
                OpKind::Cmp(s) => bool_f64(compare(lhs, &s, rhs)),
                OpKind::Logic(s) => bool_f64(if s == "||" {
                    lhs != 0.0 || rhs != 0.0
                } else {
                    lhs != 0.0 && rhs != 0.0
                }),
            };
        }
        Ok(lhs)
    }

    fn atom(&mut self) -> Result<f64, String> {
        // Unary operators bind tighter than every binary operator except `^`
        // (so `-2^2 == -(2^2)`), hence the right-binding-power 13.
        match self.next().ok_or("unexpected end of expression")? {
            Tok::Num(n) => Ok(n),
            Tok::Op('-') => Ok(-self.expr(13)?),
            Tok::Op('+') => self.expr(13),
            Tok::Not => Ok(bool_f64(self.expr(13)? == 0.0)),
            Tok::LParen => {
                let v = self.ternary()?;
                match self.next() {
                    Some(Tok::RParen) => Ok(v),
                    _ => Err("expected ')'".into()),
                }
            }
            Tok::Ident(name) => {
                if self.peek() == Some(&Tok::LParen) {
                    self.pos += 1; // consume '('
                    let mut args = Vec::new();
                    if self.peek() != Some(&Tok::RParen) {
                        loop {
                            args.push(self.ternary()?);
                            match self.next() {
                                Some(Tok::Comma) => continue,
                                Some(Tok::RParen) => break,
                                _ => return Err("expected ',' or ')'".into()),
                            }
                        }
                    } else {
                        self.pos += 1; // consume ')'
                    }
                    call(&name.to_ascii_lowercase(), &args)
                } else {
                    let key = name.to_ascii_lowercase();
                    self.env
                        .get(&key)
                        .copied()
                        .ok_or_else(|| format!("unknown parameter '{name}'"))
                }
            }
            t => Err(format!("unexpected token {t:?}")),
        }
    }
}

/// Binary operator class resolved in the Pratt loop.
enum OpKind {
    Arith(char),
    Cmp(String),
    Logic(String),
}

fn bool_f64(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

fn compare(a: f64, op: &str, b: f64) -> bool {
    match op {
        "<" => a < b,
        ">" => a > b,
        "<=" => a <= b,
        ">=" => a >= b,
        "==" => a == b,
        "!=" => a != b,
        _ => false,
    }
}

fn call(name: &str, a: &[f64]) -> Result<f64, String> {
    let one = |f: fn(f64) -> f64| -> Result<f64, String> {
        if a.len() == 1 {
            Ok(f(a[0]))
        } else {
            Err(format!("{name} expects 1 arg"))
        }
    };
    // Statistical (Monte-Carlo mismatch) functions: in a nominal/typical run
    // they evaluate to their nominal value (the first argument); the spread
    // arguments only matter under an explicit Monte-Carlo sweep. PDK model cards
    // routinely gate these behind a mismatch switch (e.g. `MC_MM_SWITCH*agauss(
    // ...)`), so nominal evaluation makes the cards resolve.
    let nominal = |min_args: usize| -> Result<f64, String> {
        if a.len() >= min_args {
            Ok(a[0])
        } else {
            Err(format!("{name} expects at least {min_args} args"))
        }
    };
    match name {
        // `if(cond, then, else)` — the SPICE ternary in function form.
        "if" if a.len() == 3 => Ok(if a[0] != 0.0 { a[1] } else { a[2] }),
        "sin" => one(f64::sin),
        "cos" => one(f64::cos),
        "tan" => one(f64::tan),
        "asin" => one(f64::asin),
        "acos" => one(f64::acos),
        "atan" => one(f64::atan),
        "exp" => one(f64::exp),
        "ln" => one(f64::ln),
        "log" | "log10" => one(f64::log10),
        "sqrt" => one(f64::sqrt),
        "abs" => one(f64::abs),
        "pow" | "pwr" if a.len() == 2 => Ok(a[0].powf(a[1])),
        "min" if a.len() == 2 => Ok(a[0].min(a[1])),
        "max" if a.len() == 2 => Ok(a[0].max(a[1])),
        // Statistical: `agauss(nom, abs_var, sigma)`, `gauss(nom, rel_var,
        // sigma)`, `aunif(nom, abs_var)`, `unif(nom, rel_var)` all return `nom`
        // nominally; `limit(nom, abs)` likewise (deterministic nominal value).
        "agauss" | "gauss" => nominal(1),
        "aunif" | "unif" => nominal(1),
        "limit" => nominal(1),
        _ => Err(format!("unknown function '{name}'")),
    }
}

/// Evaluate an arithmetic expression against `env`.
pub fn eval_expr(s: &str, env: &HashMap<String, f64>) -> Result<f64, String> {
    let t = lex(s)?;
    let mut p = P { t, pos: 0, env };
    let v = p.ternary()?;
    if p.pos != p.t.len() {
        return Err("trailing tokens in expression".into());
    }
    Ok(v)
}

/// Resolve a value token to a number: a plain engineering number, or an
/// arithmetic expression (optionally wrapped in `{...}`) over `env`.
///
/// Goes through the expression evaluator, whose number lexer already handles
/// engineering suffixes. (Calling `parse_value` directly would be wrong here:
/// it leniently accepts a leading number and ignores the rest, so `2*r` would
/// wrongly resolve to `2`.)
pub fn resolve_value(tok: &str, env: &HashMap<String, f64>) -> Option<f64> {
    let inner = unwrap_expr(tok.trim());
    eval_expr(inner, env).ok()
}

/// Strip a single layer of expression delimiters: braces `{...}` (SPICE) or
/// single quotes `'...'` (HSPICE inline expressions, used by PDK netlists for
/// geometry and element values, e.g. `l='W*2'`, `C 'CAPACITOR_0'`).
pub fn unwrap_expr(tok: &str) -> &str {
    tok.strip_prefix('{')
        .and_then(|t| t.strip_suffix('}'))
        .or_else(|| tok.strip_prefix('\'').and_then(|t| t.strip_suffix('\'')))
        .unwrap_or(tok)
}

/// Extract the value *expression* from an element value string that may carry a
/// leading keyword (`r=`, `c =`) and trailing parameters (`tc1=...`): strip the
/// keyword, then return the body of the first balanced `{...}` or `'...}` block,
/// else the first whitespace-delimited token. So `r = {f(V)} tc1=0` yields
/// `f(V)`, `r={1k}` yields `1k`, and `1k tc1=0` yields `1k`.
pub(crate) fn value_expr(s: &str) -> &str {
    let s = s.trim();
    // Strip a leading `<alpha>=` keyword (the `=` must precede any expression
    // body, so a comparison `>=` inside a `{...}` value is never mistaken for it).
    let s = match s.split_once('=') {
        Some((lhs, rhs))
            if !lhs.trim().is_empty() && lhs.trim().chars().all(|c| c.is_ascii_alphabetic()) =>
        {
            rhs.trim_start()
        }
        _ => s,
    };
    if let Some(rest) = s.strip_prefix('{') {
        let mut depth = 1usize;
        for (i, c) in rest.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &rest[..i];
                    }
                }
                _ => {}
            }
        }
        return rest;
    }
    if let Some(rest) = s.strip_prefix('\'') {
        return rest.split_once('\'').map(|(b, _)| b).unwrap_or(rest);
    }
    s.split_whitespace().next().unwrap_or(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> HashMap<String, f64> {
        let mut e = HashMap::default();
        e.insert("pi".into(), std::f64::consts::PI);
        e.insert("r".into(), 1000.0);
        e.insert("fc".into(), 1000.0);
        e
    }

    #[test]
    fn arithmetic_and_precedence() {
        let e = env();
        assert!((eval_expr("1 + 2*3", &e).unwrap() - 7.0).abs() < 1e-12);
        assert!((eval_expr("2^3^2", &e).unwrap() - 512.0).abs() < 1e-9); // right assoc
        assert!((eval_expr("-2^2", &e).unwrap() + 4.0).abs() < 1e-12); // -(2^2)
        assert!((eval_expr("(1+1)*3", &e).unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn params_funcs_and_suffixes() {
        let e = env();
        // C = 1/(2*pi*R*fc)
        let got = eval_expr("1/(2*pi*r*fc)", &e).unwrap();
        let want = 1.0 / (2.0 * std::f64::consts::PI * 1000.0 * 1000.0);
        assert!((got - want).abs() <= want * 1e-12);
        assert!((eval_expr("sqrt(2)", &e).unwrap() - 2f64.sqrt()).abs() < 1e-12);
        assert!((eval_expr("2*1k", &e).unwrap() - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn agauss_mismatch_expression() {
        let mut e = HashMap::default();
        e.insert("mc_mm_switch".into(), 0.0);
        e.insert("l".into(), 1.0);
        e.insert("w".into(), 1.0);
        e.insert("mult".into(), 1.0);
        e.insert("my_toxe_slope".into(), 0.0);
        // parts
        assert_eq!(
            eval_expr("AGAUSS(0,1.0,1)", &e).ok(),
            Some(0.0),
            "AGAUSS nominal"
        );
        assert_eq!(eval_expr("sqrt(l*w*mult)", &e).ok(), Some(1.0), "sqrt");
        // full SKY130 toxe form
        let s =
            "4.148e-09+MC_MM_SWITCH*AGAUSS(0,1.0,1)*(4.148e-09*1.0*(my_toxe_slope/sqrt(l*w*mult)))";
        assert_eq!(eval_expr(s, &e).ok(), Some(4.148e-9), "toxe nominal");
    }

    #[test]
    fn single_quoted_hspice_expressions() {
        let e = env();
        // HSPICE inline expressions: 'expr' resolves like {expr}.
        assert!((resolve_value("'2*r'", &e).unwrap() - 2000.0).abs() < 1e-9);
        assert!((resolve_value("'r*1'", &e).unwrap() - 1000.0).abs() < 1e-9);
        assert!((resolve_value("{r*1}", &e).unwrap() - 1000.0).abs() < 1e-9);
        assert_eq!(unwrap_expr("'foo'"), "foo");
        assert_eq!(unwrap_expr("{foo}"), "foo");
        assert_eq!(unwrap_expr("foo"), "foo");
    }

    #[test]
    fn comparisons_logic_and_ternary() {
        let e = env();
        // comparisons yield 1.0 / 0.0
        assert_eq!(eval_expr("2 > 1", &e).unwrap(), 1.0);
        assert_eq!(eval_expr("2 < 1", &e).unwrap(), 0.0);
        assert_eq!(eval_expr("3 >= 3", &e).unwrap(), 1.0);
        assert_eq!(eval_expr("3 != 3", &e).unwrap(), 0.0);
        // logical and / or / not
        assert_eq!(eval_expr("(1 > 0) && (2 > 1)", &e).unwrap(), 1.0);
        assert_eq!(eval_expr("(1 > 2) || (2 > 1)", &e).unwrap(), 1.0);
        assert_eq!(eval_expr("!(1 > 2)", &e).unwrap(), 1.0);
        // ternary, both branches and nesting; binds looser than comparison
        assert_eq!(eval_expr("r > 500 ? 10 : 20", &e).unwrap(), 10.0);
        assert_eq!(eval_expr("r < 500 ? 10 : 20", &e).unwrap(), 20.0);
        assert_eq!(eval_expr("0 ? 1 : (0 ? 2 : 3)", &e).unwrap(), 3.0);
        // if(c,t,e) function form
        assert_eq!(eval_expr("if(r > 500, 1k, 2k)", &e).unwrap(), 1000.0);
        // arithmetic precedence still intact alongside the new operators
        assert!((eval_expr("1 + 2*3 > 6 ? 1 : 0", &e).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn statistical_functions_evaluate_to_nominal() {
        let e = env();
        // Nominal run: agauss/gauss/aunif/unif/limit return the first argument.
        assert_eq!(eval_expr("agauss(0.52, 1.0, 1)", &e).unwrap(), 0.52);
        assert_eq!(eval_expr("gauss(1e-9, 0.1, 3)", &e).unwrap(), 1e-9);
        assert_eq!(eval_expr("aunif(2.5, 0.3)", &e).unwrap(), 2.5);
        assert_eq!(eval_expr("unif(7, 0.05)", &e).unwrap(), 7.0);
        assert_eq!(eval_expr("limit(4, 0.5)", &e).unwrap(), 4.0);
        // The PDK idiom: mismatch term gated off by a switch resolves cleanly.
        assert_eq!(
            eval_expr("0.52 + 0*agauss(0, 1, 1)*sqrt(2)", &e).unwrap(),
            0.52
        );
    }

    #[test]
    fn resolve_value_handles_braces_numbers_exprs() {
        let e = env();
        assert_eq!(resolve_value("1k", &e), Some(1e3));
        assert_eq!(resolve_value("{2*r}", &e), Some(2000.0));
        assert!(resolve_value("nope", &e).is_none());
    }
}
