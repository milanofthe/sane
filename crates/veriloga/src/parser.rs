//! Recursive-descent + Pratt parser for the Verilog-A analog subset.
//!
//! Consumes a preprocessed token stream and produces `ast::Module`s. Clean-room
//! from the Accellera LRM 2.4.0 Annex A (a presentation grammar, so the
//! disambiguation is hand-coded). `nature`/`discipline` items (from the built-in
//! headers) are parsed shallowly (name recorded, body skipped) since SANE treats
//! all nets as electrical with V/I access.

use crate::ast::*;
use crate::error::{Diagnostic, Diagnostics, Span};
use crate::token::{Tok, Token};

pub fn parse(toks: &[Token], file: &str) -> Result<Vec<Module>, Diagnostics> {
    let mut p = Parser {
        toks,
        pos: 0,
        diags: Vec::new(),
    };
    // Error-recovering parse: a malformed item/statement records a diagnostic,
    // synchronises to the next boundary, and parsing continues -- so all errors
    // in a model surface at once instead of one recompile per fix.
    let modules = p.source();
    if p.diags.is_empty() {
        Ok(modules)
    } else {
        Err(Diagnostics {
            file: file.to_string(),
            items: p.diags,
            map: None,
        })
    }
}

struct Parser<'a> {
    toks: &'a [Token],
    pos: usize,
    /// Accumulated diagnostics (recovery keeps parsing after each error).
    diags: Vec<Diagnostic>,
}

type PResult<T> = Result<T, Diagnostic>;

impl<'a> Parser<'a> {
    // --- token cursor ------------------------------------------------------

    fn cur(&self) -> &Token {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }
    fn tok(&self) -> &Tok {
        &self.cur().tok
    }
    fn span(&self) -> Span {
        self.cur().span
    }
    fn nth(&self, k: usize) -> &Tok {
        self.toks
            .get(self.pos + k)
            .map(|t| &t.tok)
            .unwrap_or(&Tok::Eof)
    }
    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos.min(self.toks.len() - 1)].clone();
        if self.pos < self.toks.len() {
            self.pos += 1;
        }
        t
    }
    fn at(&self, t: &Tok) -> bool {
        self.tok() == t
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.at(t) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn expect(&mut self, t: &Tok, what: &str) -> PResult<()> {
        if self.at(t) {
            self.bump();
            Ok(())
        } else {
            Err(self.err(&format!("expected {what}")))
        }
    }
    fn err(&self, msg: &str) -> Diagnostic {
        Diagnostic::new(format!("{msg} (found {:?})", self.tok()), self.span())
    }
    /// Is the current token the keyword `kw` (an identifier with that text)?
    fn at_kw(&self, kw: &str) -> bool {
        matches!(self.tok(), Tok::Ident(s) if s == kw)
    }
    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.at_kw(kw) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn ident(&mut self) -> PResult<String> {
        match self.tok().clone() {
            Tok::Ident(s) => {
                self.bump();
                Ok(s)
            }
            _ => Err(self.err("expected identifier")),
        }
    }

    // --- top level ---------------------------------------------------------

    fn source(&mut self) -> Vec<Module> {
        let mut modules = Vec::new();
        while !matches!(self.tok(), Tok::Eof) {
            let start = self.pos;
            if self.at_kw("module") || self.at_kw("macromodule") {
                match self.module() {
                    Ok(m) => modules.push(m),
                    Err(d) => {
                        self.diags.push(d);
                        // skip the rest of the broken module to `endmodule`.
                        while !matches!(self.tok(), Tok::Eof) && !self.eat_kw("endmodule") {
                            self.bump();
                        }
                    }
                }
            } else if self.at_kw("nature") {
                self.skip_or_record("endnature");
            } else if self.at_kw("discipline") {
                self.skip_or_record("enddiscipline");
            } else if self.eat(&Tok::Semi) {
                // stray
            } else {
                // skip unknown top-level token defensively
                self.bump();
            }
            // Guarantee forward progress so recovery can never loop.
            if self.pos == start {
                self.bump();
            }
        }
        modules
    }

    /// Skip to (and consume) a closing keyword; record a diagnostic if missing.
    fn skip_or_record(&mut self, end: &str) {
        if let Err(d) = self.skip_to_kw(end) {
            self.diags.push(d);
        }
    }

    /// Panic-mode recovery: skip to the next statement/declaration boundary --
    /// consume a terminating `;`, or stop before a block-closing keyword / EOF.
    fn recover(&mut self) {
        while !matches!(self.tok(), Tok::Eof) {
            if self.eat(&Tok::Semi) {
                return;
            }
            if self.at_kw("end")
                || self.at_kw("endmodule")
                || self.at_kw("endfunction")
                || self.at_kw("endcase")
            {
                return;
            }
            self.bump();
        }
    }

    fn skip_to_kw(&mut self, end: &str) -> PResult<()> {
        while !matches!(self.tok(), Tok::Eof) {
            if self.eat_kw(end) {
                return Ok(());
            }
            self.bump();
        }
        Err(self.err(&format!("missing {end}")))
    }

    // --- module ------------------------------------------------------------

    fn module(&mut self) -> PResult<Module> {
        let span = self.span();
        self.bump(); // module
        let name = self.ident()?;
        let mut ports = Vec::new();
        if self.eat(&Tok::LParen) {
            if !self.at(&Tok::RParen) {
                loop {
                    ports.push(self.ident()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            }
            self.expect(&Tok::RParen, "')'")?;
        }
        self.expect(&Tok::Semi, "';'")?;

        let mut m = Module {
            name,
            ports,
            nets: Vec::new(),
            params: Vec::new(),
            string_params: Vec::new(),
            aliases: Vec::new(),
            vars: Vec::new(),
            branches: Vec::new(),
            functions: Vec::new(),
            analog: Vec::new(),
            span,
        };

        while !self.at_kw("endmodule") && !matches!(self.tok(), Tok::Eof) {
            let start = self.pos;
            if let Err(d) = self.module_item(&mut m) {
                self.diags.push(d);
                self.recover();
            }
            if self.pos == start {
                self.bump();
            }
        }
        // A missing `endmodule` is recorded, not fatal (the module is best-effort).
        if !self.eat_kw("endmodule") {
            self.diags.push(self.err("expected 'endmodule'"));
        }
        Ok(m)
    }

    fn expect_kw(&mut self, kw: &str) -> PResult<()> {
        if self.eat_kw(kw) {
            Ok(())
        } else {
            Err(self.err(&format!("expected '{kw}'")))
        }
    }

    fn module_item(&mut self, m: &mut Module) -> PResult<()> {
        // Attribute instances (* ... *) annotate the FOLLOWING declaration
        // (`type="instance"`, `units`, `desc`; the compact-model OPP idiom).
        let mut attrs: Vec<(String, AttrVal)> = Vec::new();
        while self.at(&Tok::LParen) && matches!(self.nth(1), Tok::Star) {
            self.parse_attrs(&mut attrs)?;
        }
        if self.eat(&Tok::Semi) {
            return Ok(());
        }
        if self.at_kw("parameter") || self.at_kw("localparam") {
            self.param_decl(m, attrs)?;
        } else if self.at_kw("real") || self.at_kw("integer") || self.at_kw("genvar") {
            self.var_decl(m, attrs)?;
        } else if self.at_kw("branch") {
            self.branch_decl(m)?;
        } else if self.at_kw("aliasparam") {
            // aliasparam alias = target ;  (target: parameter or $sysfn)
            self.bump();
            let span = self.span();
            let alias = self.ident()?;
            self.expect(&Tok::Assign, "'='")?;
            let target = match self.tok().clone() {
                Tok::SysId(sys) => {
                    self.bump();
                    format!("${sys}")
                }
                _ => self.ident()?,
            };
            self.expect(&Tok::Semi, "';'")?;
            m.aliases.push(AliasDecl {
                alias,
                target,
                span,
            });
        } else if self.at_kw("analog")
            && matches!(self.nth(1), Tok::Ident(ref s) if s == "function")
        {
            self.analog_function(m)?;
        } else if self.at_kw("analog") && matches!(self.nth(1), Tok::Ident(ref s) if s == "initial")
        {
            // `analog initial <stmt>`: one-time initialization, same semantics
            // as `@(initial_step)` under SANE's single-model lowering.
            self.bump(); // analog
            self.bump(); // initial
            let s = self.stmt()?;
            m.analog.push(Stmt::InitialStep(Box::new(s)));
        } else if self.at_kw("analog") {
            self.bump(); // analog
            let s = self.stmt()?;
            match s {
                Stmt::Block(ss) => m.analog.extend(ss),
                other => m.analog.push(other),
            }
        } else if self.at_kw("inout") || self.at_kw("input") || self.at_kw("output") {
            // port direction declaration: consume to ';'
            self.skip_to(&Tok::Semi)?;
        } else if matches!(self.tok(), Tok::Ident(_)) {
            // discipline-typed net declaration (electrical/thermal/...) or ground:
            // `<disc> [name [array]] , ... ;` -- record the net names.
            self.net_decl(m)?;
        } else {
            return Err(self.err("unexpected module item"));
        }
        Ok(())
    }

    /// Parse one `(* name [= value], ... *)` attribute instance into `out`.
    /// Unrecognized value shapes are skipped permissively (attributes must
    /// never break a model that a simulator without attribute support parses).
    fn parse_attrs(&mut self, out: &mut Vec<(String, AttrVal)>) -> PResult<()> {
        self.bump(); // (
        self.bump(); // *
        loop {
            if matches!(self.tok(), Tok::Star) && matches!(self.nth(1), Tok::RParen) {
                self.bump();
                self.bump();
                return Ok(());
            }
            if matches!(self.tok(), Tok::Eof) {
                return Err(self.err("unterminated (* attribute *)"));
            }
            if let Tok::Ident(name) = self.tok().clone() {
                self.bump();
                let val = if self.eat(&Tok::Assign) {
                    match self.tok().clone() {
                        Tok::Str(s) => {
                            self.bump();
                            AttrVal::Str(s)
                        }
                        Tok::Ident(s) => {
                            self.bump();
                            AttrVal::Str(s)
                        }
                        Tok::Number(v) => {
                            self.bump();
                            AttrVal::Num(v)
                        }
                        Tok::Minus => {
                            self.bump();
                            match self.tok().clone() {
                                Tok::Number(v) => {
                                    self.bump();
                                    AttrVal::Num(-v)
                                }
                                _ => AttrVal::Flag,
                            }
                        }
                        _ => {
                            // value shape we do not model: skip to , or *)
                            while !(matches!(self.tok(), Tok::Comma | Tok::Eof)
                                || matches!(self.tok(), Tok::Star)
                                    && matches!(self.nth(1), Tok::RParen))
                            {
                                self.bump();
                            }
                            AttrVal::Flag
                        }
                    }
                } else {
                    AttrVal::Flag
                };
                out.push((name, val));
                let _ = self.eat(&Tok::Comma);
            } else {
                self.bump(); // permissive: unknown token inside attributes
            }
        }
    }

    fn skip_to(&mut self, t: &Tok) -> PResult<()> {
        while !matches!(self.tok(), Tok::Eof) {
            if self.eat(t) {
                return Ok(());
            }
            self.bump();
        }
        Err(self.err("unexpected end of input"))
    }

    // --- declarations ------------------------------------------------------

    fn param_decl(&mut self, m: &mut Module, attrs: Vec<(String, AttrVal)>) -> PResult<()> {
        self.bump(); // parameter / localparam
                     // `parameter string name = "lit";` -- compile-time model-variant
                     // selectors; folded during lowering, never a numeric symbol.
        if self.eat_kw("string") {
            loop {
                let name = self.ident()?;
                self.expect(&Tok::Assign, "'='")?;
                let lit = match self.tok().clone() {
                    Tok::Str(s) => {
                        self.bump();
                        s
                    }
                    _ => return Err(self.err("expected a string literal")),
                };
                m.string_params.push((name, lit));
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            self.expect(&Tok::Semi, "';'")?;
            return Ok(());
        }
        let ty = if self.eat_kw("integer") {
            VarType::Integer
        } else {
            let _ = self.eat_kw("real");
            VarType::Real
        };
        loop {
            let span = self.span();
            let name = self.ident()?;
            self.expect(&Tok::Assign, "'='")?;
            let default = self.expr()?;
            let ranges = self.ranges()?;
            m.params.push(ParamDecl {
                name,
                ty,
                default,
                ranges,
                attrs: attrs.clone(),
                span,
            });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::Semi, "';'")?;
        Ok(())
    }

    fn ranges(&mut self) -> PResult<Vec<RangeConstraint>> {
        let mut out = Vec::new();
        loop {
            let include = if self.at_kw("from") {
                true
            } else if self.at_kw("exclude") {
                false
            } else {
                break;
            };
            self.bump(); // from/exclude
                         // exclude may be a single value (no bracket)
            if !self.at(&Tok::LParen) && !self.at(&Tok::LBrack) {
                let v = self.expr()?;
                out.push(RangeConstraint {
                    include,
                    lo: Bound::Inclusive(v.clone()),
                    hi: Bound::Inclusive(v),
                });
                continue;
            }
            let lo_excl = self.eat(&Tok::LParen);
            if !lo_excl {
                self.expect(&Tok::LBrack, "'[' or '('")?;
            }
            let lo = self.bound(lo_excl)?;
            self.expect(&Tok::Colon, "':'")?;
            let hi = self.bound_hi()?;
            let hi_excl = self.eat(&Tok::RParen);
            if !hi_excl {
                self.expect(&Tok::RBrack, "']' or ')'")?;
            }
            out.push(RangeConstraint {
                include,
                lo: if lo_excl {
                    Bound::Exclusive(unwrap_bound(lo))
                } else {
                    lo
                },
                hi: if hi_excl { make_excl(hi) } else { hi },
            });
        }
        Ok(out)
    }

    fn bound(&mut self, _excl: bool) -> PResult<Bound> {
        if self.eat_kw("inf") {
            return Ok(Bound::Inf(true));
        }
        if self.at(&Tok::Minus) && matches!(self.nth(1), Tok::Ident(ref s) if s == "inf") {
            self.bump();
            self.bump();
            return Ok(Bound::Inf(false));
        }
        Ok(Bound::Inclusive(self.expr()?))
    }
    fn bound_hi(&mut self) -> PResult<Bound> {
        self.bound(false)
    }

    fn var_decl(&mut self, m: &mut Module, attrs: Vec<(String, AttrVal)>) -> PResult<()> {
        let ty = if self.eat_kw("integer") || self.eat_kw("genvar") {
            VarType::Integer
        } else {
            self.bump(); // real
            VarType::Real
        };
        loop {
            let span = self.span();
            let name = self.ident()?;
            // optional array range [a:b] or initializer = expr (skip)
            while self.at(&Tok::LBrack) {
                self.skip_bracket()?;
            }
            if self.eat(&Tok::Assign) {
                let _ = self.expr()?;
            }
            m.vars.push(VarDecl {
                name,
                ty,
                attrs: attrs.clone(),
                span,
            });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::Semi, "';'")?;
        Ok(())
    }

    fn skip_bracket(&mut self) -> PResult<()> {
        self.expect(&Tok::LBrack, "'['")?;
        let mut depth = 1;
        while depth > 0 && !matches!(self.tok(), Tok::Eof) {
            match self.tok() {
                Tok::LBrack => depth += 1,
                Tok::RBrack => depth -= 1,
                _ => {}
            }
            self.bump();
        }
        Ok(())
    }

    fn net_decl(&mut self, m: &mut Module) -> PResult<()> {
        let _disc = self.ident()?; // discipline keyword (electrical/thermal/ground/...)
                                   // optional vector range on the discipline (`electrical [n-1:0] bus;`)
        while self.at(&Tok::LBrack) {
            self.skip_bracket()?;
        }
        loop {
            let name = self.ident()?;
            while self.at(&Tok::LBrack) {
                self.skip_bracket()?;
            }
            m.nets.push(name);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::Semi, "';'")?;
        Ok(())
    }

    fn branch_decl(&mut self, m: &mut Module) -> PResult<()> {
        self.bump(); // branch
        self.expect(&Tok::LParen, "'('")?;
        let hi = self.ident()?;
        let lo = if self.eat(&Tok::Comma) {
            self.ident()?
        } else {
            "0".to_string()
        };
        self.expect(&Tok::RParen, "')'")?;
        loop {
            let name = self.ident()?;
            m.branches.push(BranchDecl {
                name,
                hi: hi.clone(),
                lo: lo.clone(),
            });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::Semi, "';'")?;
        Ok(())
    }

    fn analog_function(&mut self, m: &mut Module) -> PResult<()> {
        let span = self.span();
        self.bump(); // analog
        self.bump(); // function
        let ty = if self.eat_kw("integer") {
            VarType::Integer
        } else {
            let _ = self.eat_kw("real");
            VarType::Real
        };
        let name = self.ident()?;
        self.expect(&Tok::Semi, "';'")?;
        // input/output/declarations then a body statement, until endfunction
        let mut args = Vec::new();
        let mut outputs = Vec::new();
        let mut body = Vec::new();
        while !self.at_kw("endfunction") && !matches!(self.tok(), Tok::Eof) {
            if self.at_kw("input") || self.at_kw("output") || self.at_kw("inout") {
                let is_out = self.at_kw("output") || self.at_kw("inout");
                self.bump();
                loop {
                    let n = self.ident()?;
                    if is_out {
                        outputs.push(n.clone());
                    }
                    args.push(n);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::Semi, "';'")?;
            } else if self.at_kw("real") || self.at_kw("integer") {
                // local var decl inside function
                let mut tmp = Module {
                    name: String::new(),
                    ports: vec![],
                    nets: vec![],
                    params: vec![],
                    string_params: vec![],
                    aliases: vec![],
                    vars: vec![],
                    branches: vec![],
                    functions: vec![],
                    analog: vec![],
                    span,
                };
                self.var_decl(&mut tmp, Vec::new())?;
            } else {
                body.push(self.stmt()?);
            }
        }
        self.expect_kw("endfunction")?;
        m.functions.push(AnalogFunction {
            name,
            ty,
            args,
            outputs,
            body,
            span,
        });
        Ok(())
    }

    // --- statements --------------------------------------------------------

    fn stmt(&mut self) -> PResult<Stmt> {
        let span = self.span();
        if self.eat(&Tok::Semi) {
            return Ok(Stmt::Empty);
        }
        if self.at_kw("begin") {
            return self.block();
        }
        if self.at_kw("if") {
            return self.if_stmt();
        }
        if self.at_kw("case") {
            return self.case_stmt();
        }
        if self.at_kw("for") {
            return self.for_stmt();
        }
        if self.at_kw("while") {
            return self.while_stmt();
        }
        if self.at(&Tok::At) {
            return self.event_stmt();
        }
        // system task call as a statement. `$warning`/`$error`/`$fatal`/`$finish`
        // carry author-intended diagnostics (issue #41): preserve their args so
        // lowering can surface them. Every other task (`$strobe`, `$display`, ...)
        // is a simulation no-op whose args are discarded.
        if let Tok::SysId(name) = self.tok().clone() {
            let is_diag = matches!(name.as_str(), "warning" | "error" | "fatal" | "finish");
            self.bump();
            if is_diag {
                let args = if self.at(&Tok::LParen) {
                    self.call_args()?
                } else {
                    Vec::new()
                };
                self.expect(&Tok::Semi, "';'")?;
                return Ok(Stmt::SysTask { name, args, span });
            }
            if self.at(&Tok::LParen) {
                self.skip_paren()?;
            }
            self.expect(&Tok::Semi, "';'")?;
            return Ok(Stmt::IgnoredCall);
        }
        // contribution or assignment, both start with an identifier
        if let Tok::Ident(name) = self.tok().clone() {
            // access lvalue contribution: V(...)/I(...)/Pwr(...)/... <+ expr ;
            if access_kind(&name).is_some() && matches!(self.nth(1), Tok::LParen) {
                let (access, hi, lo) = self.access_head()?;
                if self.eat(&Tok::Contrib) {
                    let rhs = self.expr()?;
                    self.expect(&Tok::Semi, "';'")?;
                    return Ok(Stmt::Contribution {
                        access,
                        hi,
                        lo,
                        rhs,
                        span,
                    });
                }
                if self.eat(&Tok::Colon) {
                    // indirect: `access(hi,lo) : lhs == rhs ;`. Parse lhs above
                    // equality precedence so the `==` separator is not consumed
                    // as a comparison operator.
                    let lhs = self.bin(6)?;
                    self.expect(&Tok::EqEq, "'=='")?;
                    let rhs = self.expr()?;
                    self.expect(&Tok::Semi, "';'")?;
                    return Ok(Stmt::Indirect {
                        access,
                        hi,
                        lo,
                        lhs,
                        rhs,
                        span,
                    });
                }
                return Err(self.err("expected '<+' or ':' after an access lvalue"));
            }
            // assignment: lhs = expr ;   (lhs may have an array index, skipped)
            self.bump(); // name
            while self.at(&Tok::LBrack) {
                self.skip_bracket()?;
            }
            if self.eat(&Tok::Assign) {
                let rhs = self.expr()?;
                self.expect(&Tok::Semi, "';'")?;
                return Ok(Stmt::Assign {
                    lhs: name,
                    rhs,
                    span,
                });
            }
            // bare call statement `foo(args);` (analog function or task)
            if self.at(&Tok::LParen) {
                let args = self.call_args()?;
                self.expect(&Tok::Semi, "';'")?;
                return Ok(Stmt::Call { name, args, span });
            }
            return Err(self.err("expected '=' or '<+'"));
        }
        Err(self.err("unexpected statement"))
    }

    fn block(&mut self) -> PResult<Stmt> {
        self.bump(); // begin
        if self.eat(&Tok::Colon) {
            let _ = self.ident()?; // block label
        }
        // a named block may declare locals (real/integer) before statements
        let mut stmts = Vec::new();
        while !self.at_kw("end") && !matches!(self.tok(), Tok::Eof) {
            let start = self.pos;
            if self.at(&Tok::LParen) && matches!(self.nth(1), Tok::Star) {
                // block-local attribute instance: parse and drop (block locals
                // are not reportable op-vars).
                let mut ignored = Vec::new();
                if let Err(d) = self.parse_attrs(&mut ignored) {
                    self.diags.push(d);
                    self.recover();
                }
            } else if self.at_kw("real") || self.at_kw("integer") || self.at_kw("genvar") {
                let mut tmp = Module {
                    name: String::new(),
                    ports: vec![],
                    nets: vec![],
                    params: vec![],
                    string_params: vec![],
                    aliases: vec![],
                    vars: vec![],
                    branches: vec![],
                    functions: vec![],
                    analog: vec![],
                    span: self.span(),
                };
                if let Err(d) = self.var_decl(&mut tmp, Vec::new()) {
                    self.diags.push(d);
                    self.recover();
                }
            } else {
                match self.stmt() {
                    Ok(s) => stmts.push(s),
                    Err(d) => {
                        self.diags.push(d);
                        self.recover();
                    }
                }
            }
            if self.pos == start {
                self.bump();
            }
        }
        self.expect_kw("end")?;
        Ok(Stmt::Block(stmts))
    }

    fn if_stmt(&mut self) -> PResult<Stmt> {
        self.bump(); // if
        self.expect(&Tok::LParen, "'('")?;
        let cond = self.expr()?;
        self.expect(&Tok::RParen, "')'")?;
        let then = Box::new(self.stmt()?);
        let els = if self.eat_kw("else") {
            Some(Box::new(self.stmt()?))
        } else {
            None
        };
        Ok(Stmt::If { cond, then, els })
    }

    fn case_stmt(&mut self) -> PResult<Stmt> {
        self.bump(); // case
        self.expect(&Tok::LParen, "'('")?;
        let sel = self.expr()?;
        self.expect(&Tok::RParen, "')'")?;
        let mut items = Vec::new();
        let mut default = None;
        while !self.at_kw("endcase") && !matches!(self.tok(), Tok::Eof) {
            if self.eat_kw("default") {
                self.eat(&Tok::Colon);
                default = Some(Box::new(self.stmt()?));
                continue;
            }
            let mut labels = vec![self.expr()?];
            while self.eat(&Tok::Comma) {
                labels.push(self.expr()?);
            }
            self.expect(&Tok::Colon, "':'")?;
            let body = self.stmt()?;
            items.push((labels, body));
        }
        self.expect_kw("endcase")?;
        Ok(Stmt::Case {
            sel,
            items,
            default,
        })
    }

    fn for_stmt(&mut self) -> PResult<Stmt> {
        self.bump(); // for
        self.expect(&Tok::LParen, "'('")?;
        let init = Box::new(self.simple_assign()?);
        self.expect(&Tok::Semi, "';'")?;
        let cond = self.expr()?;
        self.expect(&Tok::Semi, "';'")?;
        let step = Box::new(self.simple_assign()?);
        self.expect(&Tok::RParen, "')'")?;
        let body = Box::new(self.stmt()?);
        Ok(Stmt::For {
            init,
            cond,
            step,
            body,
        })
    }

    fn while_stmt(&mut self) -> PResult<Stmt> {
        self.bump(); // while
        self.expect(&Tok::LParen, "'('")?;
        let cond = self.expr()?;
        self.expect(&Tok::RParen, "')'")?;
        let body = Box::new(self.stmt()?);
        Ok(Stmt::While { cond, body })
    }

    /// `lhs = expr` without a trailing semicolon (for `for` init/step).
    fn simple_assign(&mut self) -> PResult<Stmt> {
        let span = self.span();
        let lhs = self.ident()?;
        self.expect(&Tok::Assign, "'='")?;
        let rhs = self.expr()?;
        Ok(Stmt::Assign { lhs, rhs, span })
    }

    fn event_stmt(&mut self) -> PResult<Stmt> {
        let span = self.span();
        self.bump(); // @
                     // @(event_expr) — `initial_step`/`initial_model` are supported (a one-time
                     // init block); every other control (`cross`/`timer`/`final_step`/`above`)
                     // cannot be honored by SANE's analysis-agnostic lowering, so its name is
                     // captured and the body carried to lowering, which rejects it with a
                     // diagnostic instead of silently stripping the guard (issue #42).
        let mut is_initial = false;
        let mut control = String::new();
        let mut args = Vec::new();
        if self.eat(&Tok::LParen) {
            if let Tok::Ident(s) = self.tok() {
                control = s.clone();
            }
            if self.at_kw("initial_step") || self.at_kw("initial_model") {
                is_initial = true;
            }
            // `cross(expr [, dir [, tol...]])` / `above(expr [, ...])`: the
            // arguments are expressions lowering needs (the surface, its
            // direction); the rest of the control is skipped.
            if matches!(control.as_str(), "cross" | "above") {
                self.bump();
                if self.eat(&Tok::LParen) {
                    while !matches!(self.tok(), Tok::RParen | Tok::Eof) {
                        args.push(self.expr()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.skip_paren_rest()?;
                }
            }
            self.skip_paren_rest()?;
        }
        let body = self.stmt()?;
        if is_initial {
            Ok(Stmt::InitialStep(Box::new(body)))
        } else if control.is_empty() {
            // A bare `@(...)` with no recognizable control token: keep permissive.
            Ok(body)
        } else {
            Ok(Stmt::Event {
                control,
                args,
                body: Box::new(body),
                span,
            })
        }
    }

    /// Consume a balanced `( ... )` starting at the current `(`.
    fn skip_paren(&mut self) -> PResult<()> {
        self.expect(&Tok::LParen, "'('")?;
        self.skip_paren_rest()
    }
    /// Consume the rest of a `( ... )` whose `(` was already eaten.
    fn skip_paren_rest(&mut self) -> PResult<()> {
        let mut depth = 1;
        while depth > 0 && !matches!(self.tok(), Tok::Eof) {
            match self.tok() {
                Tok::LParen => depth += 1,
                Tok::RParen => depth -= 1,
                _ => {}
            }
            self.bump();
        }
        Ok(())
    }

    fn access_head(&mut self) -> PResult<(Access, String, Option<String>)> {
        let name = self.ident()?;
        let access = access_kind(&name).unwrap_or(Access::I);
        self.expect(&Tok::LParen, "'('")?;
        let hi = self.access_node()?;
        let lo = if self.eat(&Tok::Comma) {
            Some(self.access_node()?)
        } else {
            None
        };
        self.expect(&Tok::RParen, "')'")?;
        Ok((access, hi, lo))
    }

    /// An access-function node argument: a plain node name, or a port-flow
    /// reference `<port>` (the angle brackets are dropped).
    fn access_node(&mut self) -> PResult<String> {
        if self.eat(&Tok::Lt) {
            let n = self.ident()?;
            self.expect(&Tok::Gt, "'>'")?;
            Ok(n)
        } else {
            self.ident()
        }
    }

    // --- expressions (Pratt) ----------------------------------------------

    fn expr(&mut self) -> PResult<Expr> {
        self.ternary()
    }

    fn ternary(&mut self) -> PResult<Expr> {
        let cond = self.bin(0)?;
        if self.at(&Tok::Question) {
            let span = self.span();
            self.bump();
            let then = self.expr()?;
            self.expect(&Tok::Colon, "':'")?;
            let els = self.ternary()?;
            return Ok(Expr::Ternary {
                cond: Box::new(cond),
                then: Box::new(then),
                els: Box::new(els),
                span,
            });
        }
        Ok(cond)
    }

    fn bin(&mut self, min_bp: u8) -> PResult<Expr> {
        let mut lhs = self.unary()?;
        loop {
            let (op, lbp, rbp) = match bin_op(self.tok()) {
                Some(x) => x,
                None => break,
            };
            if lbp < min_bp {
                break;
            }
            let span = self.span();
            self.bump();
            let rhs = self.bin(rbp)?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> PResult<Expr> {
        let span = self.span();
        if self.eat(&Tok::Minus) {
            return Ok(Expr::Unary {
                op: UnOp::Neg,
                arg: Box::new(self.unary()?),
                span,
            });
        }
        if self.eat(&Tok::Not) {
            return Ok(Expr::Unary {
                op: UnOp::Not,
                arg: Box::new(self.unary()?),
                span,
            });
        }
        if self.eat(&Tok::Plus) {
            return self.unary();
        }
        self.primary()
    }

    fn primary(&mut self) -> PResult<Expr> {
        let span = self.span();
        match self.tok().clone() {
            Tok::Number(n) => {
                self.bump();
                Ok(Expr::Num(n))
            }
            // String literals in expressions: compile-time only (string-parameter
            // comparisons, system-function arguments); the lowerer enforces use.
            Tok::Str(s) => {
                self.bump();
                Ok(Expr::Str(s))
            }
            Tok::LParen => {
                self.bump();
                let e = self.expr()?;
                self.expect(&Tok::RParen, "')'")?;
                Ok(e)
            }
            Tok::LBrace => {
                // Array / coefficient-vector literal `{e0, e1, ...}`.
                self.bump();
                let mut elems = Vec::new();
                if !self.at(&Tok::RBrace) {
                    loop {
                        elems.push(self.expr()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RBrace, "'}'")?;
                Ok(Expr::Array(elems))
            }
            Tok::SysId(name) => {
                self.bump();
                let args = if self.at(&Tok::LParen) {
                    self.call_args()?
                } else {
                    Vec::new()
                };
                Ok(Expr::SysFn { name, args, span })
            }
            Tok::Ident(name) => {
                self.bump();
                if let (Some(access), true) = (access_kind(&name), self.at(&Tok::LParen)) {
                    // access function in an expression (probe)
                    self.bump(); // (
                    let hi = self.access_node()?;
                    let lo = if self.eat(&Tok::Comma) {
                        Some(self.access_node()?)
                    } else {
                        None
                    };
                    self.expect(&Tok::RParen, "')'")?;
                    return Ok(Expr::Access {
                        access,
                        hi,
                        lo,
                        span,
                    });
                }
                if self.at(&Tok::LParen) {
                    let args = self.call_args()?;
                    return Ok(Expr::Call { name, args, span });
                }
                Ok(Expr::Ident(name, span))
            }
            _ => Err(self.err("expected an expression")),
        }
    }

    fn call_args(&mut self) -> PResult<Vec<Expr>> {
        self.expect(&Tok::LParen, "'('")?;
        let mut args = Vec::new();
        if !self.at(&Tok::RParen) {
            loop {
                // string args (e.g. $simparam("gmin"), $strobe text) preserved.
                if let Tok::Str(s) = self.tok() {
                    let s = s.clone();
                    self.bump();
                    args.push(Expr::Str(s));
                } else {
                    args.push(self.expr()?);
                }
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen, "')'")?;
        Ok(args)
    }
}

/// Binary operator binding powers `(op, left_bp, right_bp)`. Higher binds
/// tighter; `**` is right-associative (right_bp < left_bp).
fn bin_op(t: &Tok) -> Option<(BinOp, u8, u8)> {
    Some(match t {
        Tok::OrOr => (BinOp::Or, 1, 2),
        Tok::AndAnd => (BinOp::And, 3, 4),
        Tok::EqEq => (BinOp::Eq, 5, 6),
        Tok::Ne => (BinOp::Ne, 5, 6),
        Tok::Lt => (BinOp::Lt, 7, 8),
        Tok::Gt => (BinOp::Gt, 7, 8),
        Tok::Le => (BinOp::Le, 7, 8),
        Tok::Ge => (BinOp::Ge, 7, 8),
        Tok::Plus => (BinOp::Add, 9, 10),
        Tok::Minus => (BinOp::Sub, 9, 10),
        Tok::Star => (BinOp::Mul, 11, 12),
        Tok::Slash => (BinOp::Div, 11, 12),
        Tok::Percent => (BinOp::Mod, 11, 12),
        Tok::Pow => (BinOp::Pow, 14, 13),
        _ => return None,
    })
}

/// Classify an identifier as a nature access function. `V`/`I` are electrical
/// (first-class); the rest are other-discipline accesses carried by name.
fn access_kind(name: &str) -> Option<Access> {
    match name {
        "V" => Some(Access::V),
        "I" => Some(Access::I),
        "Pwr" | "Temp" | "Q" | "Phi" | "Pot" | "Flow" | "MMF" => {
            Some(Access::Other(name.to_string()))
        }
        _ => None,
    }
}

fn unwrap_bound(b: Bound) -> Expr {
    match b {
        Bound::Inclusive(e) | Bound::Exclusive(e) => e,
        Bound::Inf(_) => Expr::Num(f64::INFINITY),
    }
}
fn make_excl(b: Bound) -> Bound {
    match b {
        Bound::Inclusive(e) => Bound::Exclusive(e),
        other => other,
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use crate::lexer::lex;

    fn diags(src: &str) -> Vec<String> {
        let toks = lex(src, "t.va").expect("lex");
        match parse(&toks, "t.va") {
            Ok(_) => Vec::new(),
            Err(d) => d.items.iter().map(|x| x.message.clone()).collect(),
        }
    }

    #[test]
    fn reports_multiple_errors_not_just_first() {
        // Two broken statements in one analog block: recovery must surface both.
        let src = "module m(a); electrical a; analog begin\n  x = ;\n  y = ;\nend endmodule";
        let ds = diags(src);
        assert!(
            ds.len() >= 2,
            "expected >=2 diagnostics, got {}: {ds:?}",
            ds.len()
        );
    }

    #[test]
    fn recovers_to_following_valid_statement() {
        // A broken statement followed by a valid one: the valid one still parses
        // (only the broken one is reported).
        let toks = lex(
            "module m(a); electrical a; analog begin\n  x = ;\n  I(a) <+ 1.0;\nend endmodule",
            "t.va",
        )
        .expect("lex");
        let r = parse(&toks, "t.va");
        assert!(r.is_err(), "the broken statement must be reported");
        assert_eq!(r.unwrap_err().items.len(), 1, "exactly one error expected");
    }

    #[test]
    fn clean_module_has_no_diagnostics() {
        assert!(diags("module m(a); electrical a; analog I(a) <+ 1.0; endmodule").is_empty());
    }
}
