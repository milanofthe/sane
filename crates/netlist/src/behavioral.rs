//! Behavioral (`B`) source expression parser: turns `V=...` / `I=...` right-hand
//! sides into a [`sane_circuit::BExpr`] tree over node voltages `V(node)`, branch
//! currents `I(element)`, parameters and arithmetic / functions. Node names are
//! resolved to indices through the caller's node table; element names stay as
//! strings (resolved to branch currents in the DAE layer).

use rsdag::CmpOp;
use sane_circuit::{BCall, BExpr, BOp};
use sane_core::MathFn;

use crate::expr::{lex, Tok};

/// Parse a behavioral right-hand side into a [`BExpr`]. `resolve_node` maps a
/// node name to its index (and registers new nodes, as elsewhere in the parser).
pub(crate) fn parse_bexpr(
    s: &str,
    resolve_node: &mut dyn FnMut(&str) -> usize,
) -> Result<BExpr, String> {
    let t = lex(s)?;
    let mut p = BP {
        t,
        pos: 0,
        resolve_node,
    };
    let e = p.expr(0)?;
    if p.pos != p.t.len() {
        return Err("trailing tokens in B expression".into());
    }
    Ok(e)
}

struct BP<'a> {
    t: Vec<Tok>,
    pos: usize,
    resolve_node: &'a mut dyn FnMut(&str) -> usize,
}

impl BP<'_> {
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

    fn expr(&mut self, min_bp: u8) -> Result<BExpr, String> {
        let mut lhs = self.atom()?;
        loop {
            match self.peek().cloned() {
                // Arithmetic. Comparisons bind looser than everything (so
                // `a + b > c` parses as `(a + b) > c`), `^` is right-associative.
                Some(Tok::Op(op)) => {
                    let (lbp, rbp, op) = match op {
                        '+' => (3, 4, BOp::Add),
                        '-' => (3, 4, BOp::Sub),
                        '*' => (5, 6, BOp::Mul),
                        '/' => (5, 6, BOp::Div),
                        '^' => (8, 7, BOp::Pow),
                        _ => break,
                    };
                    if lbp < min_bp {
                        break;
                    }
                    self.pos += 1;
                    let rhs = self.expr(rbp)?;
                    lhs = BExpr::Bin(op, Box::new(lhs), Box::new(rhs));
                }
                // Comparison, yielding 1.0 / 0.0 (combine with `if(c, t, e)`).
                Some(Tok::Cmp(cmp)) => {
                    let (lbp, rbp) = (1u8, 2u8);
                    if lbp < min_bp {
                        break;
                    }
                    self.pos += 1;
                    let rhs = self.expr(rbp)?;
                    lhs = BExpr::Bin(BOp::Cmp(cmp_op(&cmp)), Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    /// A node reference inside `V(...)`: an identifier or a bare integer.
    fn node_ref(&mut self) -> Result<usize, String> {
        match self.next().ok_or("expected node name")? {
            Tok::Ident(n) => Ok((self.resolve_node)(&n)),
            Tok::Num(x) if x.fract() == 0.0 => Ok((self.resolve_node)(&format!("{}", x as i64))),
            t => Err(format!("expected node name, got {t:?}")),
        }
    }

    fn expect_rparen(&mut self) -> Result<(), String> {
        match self.next() {
            Some(Tok::RParen) => Ok(()),
            _ => Err("expected ')'".into()),
        }
    }

    fn atom(&mut self) -> Result<BExpr, String> {
        match self.next().ok_or("unexpected end of B expression")? {
            Tok::Num(n) => Ok(BExpr::Const(n)),
            Tok::Op('-') => Ok(BExpr::Neg(Box::new(self.expr(7)?))),
            Tok::Op('+') => self.expr(7),
            Tok::LParen => {
                let v = self.expr(0)?;
                match self.next() {
                    Some(Tok::RParen) => Ok(v),
                    _ => Err("expected ')'".into()),
                }
            }
            Tok::Ident(name) => {
                let lname = name.to_ascii_lowercase();
                if self.peek() == Some(&Tok::LParen) {
                    self.pos += 1; // consume '('
                    if lname == "v" {
                        // V(a) or V(a,b)
                        let a = self.node_ref()?;
                        let e = if self.peek() == Some(&Tok::Comma) {
                            self.pos += 1;
                            let b = self.node_ref()?;
                            BExpr::Bin(
                                BOp::Sub,
                                Box::new(BExpr::NodeV(a)),
                                Box::new(BExpr::NodeV(b)),
                            )
                        } else {
                            BExpr::NodeV(a)
                        };
                        self.expect_rparen()?;
                        Ok(e)
                    } else if lname == "i" {
                        // I(element): the branch current of a voltage-defined element
                        let elem = match self.next() {
                            Some(Tok::Ident(n)) => n,
                            t => return Err(format!("I() expects an element name, got {t:?}")),
                        };
                        self.expect_rparen()?;
                        Ok(BExpr::BranchI(elem))
                    } else {
                        // function call: f(arg, ...)
                        let mut args = Vec::new();
                        if self.peek() != Some(&Tok::RParen) {
                            loop {
                                args.push(self.expr(0)?);
                                match self.next() {
                                    Some(Tok::Comma) => continue,
                                    Some(Tok::RParen) => break,
                                    _ => return Err("expected ',' or ')'".into()),
                                }
                            }
                        } else {
                            self.pos += 1;
                        }
                        // Reject unknown functions / wrong arity at parse time,
                        // so the B-source never silently lowers to zero.
                        Ok(BExpr::Call(call_of(&lname, args.len())?, args))
                    }
                } else {
                    // a named parameter (stays symbolic, bound later)
                    Ok(BExpr::Param(name))
                }
            }
            t => Err(format!("unexpected token {t:?}")),
        }
    }
}

/// The comparison a comparison token spells.
fn cmp_op(op: &str) -> CmpOp {
    match op {
        "<" => CmpOp::Lt,
        ">" => CmpOp::Gt,
        "<=" => CmpOp::Le,
        ">=" => CmpOp::Ge,
        "==" => CmpOp::Eq,
        "!=" => CmpOp::Ne,
        _ => unreachable!("lexer only emits the six comparison operators"),
    }
}

/// What a behavioral (`B`) source calls by `name` with `argc` arguments:
/// an elementary function (see [`MathFn`]; `pwr` is `pow`) or `if(cond,
/// then, else)`. An unknown name or a wrong argument count is a hard parse
/// error rather than a silent zero.
fn call_of(name: &str, argc: usize) -> Result<BCall, String> {
    let (call, arity) = match name {
        "pwr" => (BCall::Math(MathFn::Pow), 2),
        "if" => (BCall::If, 3),
        _ => match MathFn::from_spice(name) {
            Some(f) => (BCall::Math(f), f.arity()),
            None => {
                let known: Vec<&str> = (MathFn::ALL.iter().map(|f| f.name()))
                    .chain(["log", "pwr", "if"])
                    .collect();
                return Err(format!(
                    "unknown function '{name}' in B source (supported: {})",
                    known.join(" ")
                ));
            }
        },
    };
    if argc != arity {
        return Err(format!(
            "function '{name}' expects {arity} argument(s), got {argc}"
        ));
    }
    Ok(call)
}
