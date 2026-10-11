//! Expressions: operators, the system functions and the module's own
//! analog functions, inlined.

use super::*;

impl<'a, 'b> Lower<'a, 'b> {
    pub(super) fn expr(&mut self, e: &Expr) -> Result<ExprId, String> {
        match e {
            Expr::Num(n) => Ok(self.ctx().konst_f64(*n)),
            Expr::Str(_) => {
                Err("string literal is only valid as a system-function argument".into())
            }
            Expr::Array(_) => {
                Err("array literal is only valid as a filter coefficient argument".into())
            }
            Expr::Ident(name, _) => {
                if let Some(v) = self.st.vars.get(name) {
                    return Ok(*v);
                }
                if let Some(v) = self.int_value(name) {
                    return Ok(v);
                }
                if self.em.params.iter().any(|p| &p.name == name) {
                    let sym = format!("{}.{}", self.inst, name);
                    let e = self.ctx().sym(&sym);
                    if let Some(s) = sym_of(self.ctx(), e) {
                        self.param_syms.insert(name.clone(), s);
                    }
                    return Ok(e);
                }
                if self.em.string_params.contains_key(name) {
                    return Err(format!(
                        "string parameter '{name}' can only be compared with ==/!= against \
                         a string"
                    ));
                }
                Err(format!("unknown identifier '{name}'"))
            }
            Expr::Access { access, hi, lo, .. } => {
                let (hn, ln) = self.resolve_pair(hi, lo);
                if is_potential(access) {
                    if hn == ln {
                        // collapsed branch: its voltage is identically zero
                        return Ok(self.ctx().zero());
                    }
                    let vhi = self.node_voltage(&hn)?;
                    let vlo = self.node_voltage(&ln)?;
                    Ok(self.ctx().sub(vhi, vlo))
                } else {
                    // Flow probe: the branch was promoted to an explicit current
                    // unknown during setup; return it (oriented).
                    let (key, sign) = canon(&hn, &ln);
                    match self.probe_of.get(&key).copied() {
                        Some(i) => Ok(if sign == 1.0 { i } else { self.ctx().neg(i) }),
                        None => Err("flow probe of a branch that was not pre-scanned".into()),
                    }
                }
            }
            Expr::Unary { op, arg, .. } => {
                let a = self.expr(arg)?;
                match op {
                    UnOp::Neg => Ok(self.ctx().neg(a)),
                    UnOp::Not => {
                        let zero = self.ctx().zero();
                        Ok(self.ctx().cmp(CmpOp::Eq, a, zero))
                    }
                }
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                // String comparison (`mode == "fast"`): compile-time only.
                if let Some(v) = self.fold_str_cmp(*op, lhs, rhs) {
                    return Ok(self.ctx().konst_f64(v));
                }
                if self.const_str(lhs).is_some() || self.const_str(rhs).is_some() {
                    return Err(
                        "string operands support only ==/!= against another compile-time \
                         string"
                            .into(),
                    );
                }
                let a = self.expr(lhs)?;
                let b = self.expr(rhs)?;
                Ok(self.binary(*op, a, b))
            }
            Expr::Ternary {
                cond, then, els, ..
            } => {
                // A condition the graph folds takes its arm statically, as in
                // `lower_if`: the discarded arm never enters the graph.
                let c = self.expr(cond)?;
                if let Some(v) = self.ctx().const_f64(c) {
                    return if v != 0.0 {
                        self.expr(then)
                    } else {
                        self.expr(els)
                    };
                }
                // Each arm on its path, as an `if`'s.
                self.path.push(c);
                let t = self.expr(then);
                self.path.pop();
                let nc = self.not(c);
                self.path.push(nc);
                let e = self.expr(els);
                self.path.pop();
                Ok(self.ctx().select(c, t?, e?))
            }
            Expr::Call { name, args, .. } => self.call(name, args),
            Expr::SysFn { name, args, .. } => self.sysfn(name, args),
        }
    }

    pub(super) fn binary(&mut self, op: BinOp, a: ExprId, b: ExprId) -> ExprId {
        // Both arms of a conditional are lowered, so a divide on a path not
        // taken must stay finite: its divisor is 1 off the path. Exact on it,
        // and finite off it, residual and derivatives alike (an `inf` there
        // would leak as `NaN` through `0 * inf` in a derivative product,
        // issue #43). Where a binding decides the path, its variant drops the
        // guard with the arm.
        if op == BinOp::Div && !self.path.is_empty() && !self.ctx().is_zero(b) {
            let reached = self.reached();
            let ctx = self.ctx();
            let one = ctx.one();
            let b = ctx.select(reached, b, one);
            return ctx.div(a, b);
        }
        let ctx = self.lo.ctx();
        match op {
            BinOp::Add => ctx.add(a, b),
            BinOp::Sub => ctx.sub(a, b),
            BinOp::Mul => ctx.mul(a, b),
            // A constant-zero divisor folds to 0: `recip(0)` would panic, and
            // such a divide sits on a path a constant decided not to take.
            BinOp::Div => {
                if ctx.is_zero(b) {
                    ctx.konst_f64(0.0)
                } else {
                    ctx.div(a, b)
                }
            }
            BinOp::Pow => sane_core::mathfn::pow(ctx, a, b),
            BinOp::Mod => {
                if ctx.is_zero(b) {
                    ctx.konst_f64(0.0)
                } else {
                    let q = ctx.div(a, b);
                    let fq = ctx.floor(q);
                    let bf = ctx.mul(b, fq);
                    ctx.sub(a, bf)
                }
            }
            BinOp::Lt => ctx.cmp(CmpOp::Lt, a, b),
            BinOp::Gt => ctx.cmp(CmpOp::Gt, a, b),
            BinOp::Le => ctx.cmp(CmpOp::Le, a, b),
            BinOp::Ge => ctx.cmp(CmpOp::Ge, a, b),
            BinOp::Eq => ctx.cmp(CmpOp::Eq, a, b),
            BinOp::Ne => ctx.cmp(CmpOp::Ne, a, b),
            BinOp::And => {
                let zero = ctx.zero();
                let na = ctx.cmp(CmpOp::Ne, a, zero);
                let nb = ctx.cmp(CmpOp::Ne, b, zero);
                ctx.mul(na, nb)
            }
            BinOp::Or => {
                let zero = ctx.zero();
                let na = ctx.cmp(CmpOp::Ne, a, zero);
                let nb = ctx.cmp(CmpOp::Ne, b, zero);
                let s = ctx.add(na, nb);
                ctx.cmp(CmpOp::Gt, s, zero)
            }
        }
    }

    pub(super) fn sysfn(&mut self, name: &str, args: &[Expr]) -> Result<ExprId, String> {
        match name {
            // Global circuit temperature [K]: a shared symbol (defaulting to
            // TEMP_NOMINAL_K) so a temperature sweep can vary it, rather than a
            // baked-in constant.
            "temperature" => Ok(self.ctx().sym(sane_core::constants::TEMP_SYMBOL)),
            "vt" => {
                let t = if args.is_empty() {
                    self.ctx().sym(sane_core::constants::TEMP_SYMBOL)
                } else {
                    self.expr(&args[0])?
                };
                let kq = self.ctx().konst_f64(K_OVER_Q);
                Ok(self.ctx().mul(kq, t))
            }
            "abstime" | "realtime" => Ok(self.ctx().sym("t")),
            // Parallel multiplicity `m` (instance `m=`): the simulator already
            // scales every flow contribution by `m` (see `contribute`); this
            // returns the value for any explicit model use (e.g. noise / R/m).
            "mfactor" => {
                let m = self.mfactor;
                Ok(self.ctx().konst_f64(m))
            }
            "port_connected" => Ok(self.ctx().konst_f64(1.0)),
            // Correct: was the parameter explicitly set on the instance/deck?
            "param_given" => {
                let set = matches!(args.first(), Some(Expr::Ident(p, _)) if self.given.contains(p));
                Ok(self.ctx().konst_f64(if set { 1.0 } else { 0.0 }))
            }
            "simparam" => {
                // Prefer the model-supplied default; otherwise fall back to a
                // small table of standard simulator options. Unknown option with
                // no default -> error (no silent guess).
                if args.len() >= 2 {
                    self.expr(&args[1])
                } else {
                    let opt = match args.first() {
                        Some(Expr::Str(s)) => s.as_str(),
                        _ => "",
                    };
                    let v = match opt {
                        "gmin" => Some(VA_SIMPARAM_GMIN),
                        "scale" | "shrink" | "sourceScaleFactor" | "mfactor" => Some(1.0),
                        "tnom" => Some(TEMP_NOMINAL_C),
                        _ => None,
                    };
                    match v {
                        Some(x) => Ok(self.ctx().konst_f64(x)),
                        None => Err(format!(
                            "$simparam(\"{opt}\") without a default is unsupported"
                        )),
                    }
                }
            }
            // `$limit(x, fn, ...)` is path-only Newton limiting: the limited
            // access is recorded here (only live, parameter-folded arms reach
            // this point) and applied by the DC solver between iterates; the
            // VALUE is exactly `x`, so the converged fixed point is untouched.
            "limit" => {
                if args.is_empty() {
                    return Err("$limit expects at least one argument".into());
                }
                if let (
                    Expr::Access {
                        access: Access::V,
                        hi,
                        lo,
                        ..
                    },
                    Some(Expr::Str(kind)),
                ) = (&args[0], args.get(1))
                {
                    let kind = match kind.as_str() {
                        "pnjlim" => Some(LimitKind::PnJunction),
                        "fetlim" => Some(LimitKind::Fet),
                        _ => None,
                    };
                    // Declared on a path over the parameters, the limit holds
                    // where the path does; on one over the solution, always.
                    let reached = self.reached();
                    let when = match self.ctx().const_f64(reached) {
                        Some(r) => (r != 0.0).then_some(None),
                        None if self.over_params(reached) => Some(Some(reached)),
                        None => Some(None),
                    };
                    if let (Some(kind), Some(when)) = (kind, when) {
                        let (hn, ln) = self.resolve_pair(hi, lo);
                        let vhi = self.node_voltage(&hn)?;
                        let vlo = self.node_voltage(&ln)?;
                        let (hi_s, lo_s) = (sym_of(self.lo.ctx(), vhi), sym_of(self.lo.ctx(), vlo));
                        let limit = FragmentLimit {
                            hi: hi_s,
                            lo: lo_s,
                            kind,
                            when,
                        };
                        if !self.limits.iter().any(|l| {
                            (l.hi, l.lo, l.kind, l.when)
                                == (limit.hi, limit.lo, limit.kind, limit.when)
                        }) {
                            self.limits.push(limit);
                        }
                    }
                }
                self.expr(&args[0])
            }
            // Solver hints: no effect on the residual.
            "bound_step" | "discontinuity" | "limit_step" => Ok(self.ctx().zero()),
            // Diagnostics: no effect on the residual.
            "strobe" | "display" | "write" | "fopen" | "fclose" | "fstrobe" | "fdisplay"
            | "debug" | "warning" | "error" | "finish" | "fwrite" | "monitor" => {
                Ok(self.ctx().zero())
            }
            _ => Err(format!("system function '${name}' is not supported")),
        }
    }

    pub(super) fn inline_fn(
        &mut self,
        func: &crate::ast::AnalogFunction,
        args: &[Expr],
    ) -> Result<ExprId, String> {
        if args.len() != func.args.len() {
            return Err(format!(
                "analog function '{}' expects {} args, got {}",
                func.name,
                func.args.len(),
                args.len()
            ));
        }
        let mut argmap: HashMap<String, ExprId> = HashMap::default();
        for (p, e) in func.args.iter().zip(args) {
            let v = self.expr(e)?;
            argmap.insert(p.clone(), v);
        }
        let saved = std::mem::replace(
            &mut self.st,
            State {
                vars: argmap,
                node_cur: HashMap::default(),
            },
        );
        self.cond_depth += 1;
        // The function body runs against a fresh, fully save/restored state, so
        // its journaled writes are discarded wholesale below -- mark the journal
        // so they can be truncated rather than left dangling against `saved`.
        let jmark = self.journal.len();
        let mut err = None;
        for s in &func.body {
            if let Err(e) = self.stmt(s) {
                err = Some(e);
                break;
            }
        }
        self.cond_depth -= 1;
        // Capture `output`/`inout` argument final values to write back to the
        // caller's variables (the actual args must be plain identifiers).
        let mut writebacks: Vec<(String, ExprId)> = Vec::new();
        for out in &func.outputs {
            if let Some(idx) = func.args.iter().position(|a| a == out) {
                if let (Some(Expr::Ident(caller, _)), Some(&v)) =
                    (args.get(idx), self.st.vars.get(out))
                {
                    writebacks.push((caller.clone(), v));
                }
            }
        }
        let result = self.st.vars.get(&func.name).copied();
        self.st = saved;
        // The body's writes were against the discarded function state; drop their
        // undo entries so the journal stays consistent with `self.st`.
        self.journal.truncate(jmark);
        // Apply the write-backs to the caller through the journaled helpers, so an
        // enclosing conditional can rewind them like any other write.
        for (caller, v) in writebacks {
            self.set_var(caller, v);
        }
        if let Some(e) = err {
            return Err(e);
        }
        result.ok_or_else(|| {
            format!(
                "analog function '{}' did not assign its return value",
                func.name
            )
        })
    }
}
