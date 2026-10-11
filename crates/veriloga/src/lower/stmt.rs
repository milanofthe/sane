//! The analog block's statements: assignments, conditionals (a branch per
//! reached path, merged where they meet), case, loops, events and the
//! system tasks.

use super::*;

impl<'a, 'b> Lower<'a, 'b> {
    /// Join the constant (string / numeric) arguments of a diagnostic system task
    /// into a human message. The runtime format specifiers a real simulator would
    /// interpolate are not available at load time, so only the literal parts show.
    pub(super) fn task_message(&self, args: &[Expr]) -> String {
        let parts: Vec<String> = args
            .iter()
            .filter_map(|a| match a {
                Expr::Str(s) => Some(s.clone()),
                Expr::Num(n) => Some(format!("{n}")),
                _ => None,
            })
            .collect();
        if parts.is_empty() {
            "(no constant message)".to_string()
        } else {
            parts.join(" ")
        }
    }

    /// Lower a diagnostic system task (`$warning`/`$error`/`$fatal`/`$finish`),
    /// issue #41. Where it is reached is the conjunction of the conditions
    /// of the arms around it ([`Self::path`]): constant, it fires or not;
    /// over the parameters only, an `$error` is an assertion that its path
    /// does not hold; over the solution, a runtime check SANE does not make.
    pub(super) fn sys_task(
        &mut self,
        name: &str,
        args: &[Expr],
        span: crate::error::Span,
    ) -> Result<(), String> {
        let msg = self.task_message(args);
        match name {
            // A warning always fires: surface it as a captured (catchable) warning.
            "warning" => {
                sane_core::log::warn_captured(&format!(
                    "$warning (module {}, line {}): {}",
                    self.em.name, span.line, msg
                ));
                Ok(())
            }
            // $error/$fatal reached unconditionally is the author rejecting this
            // configuration: a hard load failure. Reached on a path over the
            // parameters, it rejects the bindings that take the path.
            "error" | "fatal" => {
                let reached = self.reached();
                let text = format!(
                    "${name} (module {}, line {}): {}",
                    self.em.name, span.line, msg
                );
                match self.ctx().const_f64(reached) {
                    Some(r) if r != 0.0 => Err(text),
                    Some(_) => Ok(()),
                    None if self.over_params(reached) => {
                        let holds = self.not(reached);
                        self.assertions.push(Assertion {
                            holds,
                            message: text,
                        });
                        Ok(())
                    }
                    None => {
                        sane_core::log::warn_captured(&format!(
                            "{text} (a runtime assertion SANE does not enforce; the model is \
                             solved without it)"
                        ));
                        Ok(())
                    }
                }
            }
            // $finish ends a simulation run; it has no load-time residual meaning.
            // An unconditional one flags a model that expects to abort -- note it.
            "finish" => {
                if self.cond_depth == 0 {
                    sane_core::log::warn_captured(&format!(
                        "$finish (module {}, line {}) reached unconditionally at load; ignored \
                         (SANE runs no procedural time loop): {}",
                        self.em.name, span.line, msg
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub(super) fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        if !self.skip.is_empty() && self.skip.contains(&crate::topology::at(s)) {
            return Ok(());
        }
        match s {
            Stmt::Block(ss) => {
                for s in ss {
                    self.stmt(s)?;
                }
                Ok(())
            }
            Stmt::Empty | Stmt::IgnoredCall => Ok(()),
            Stmt::SysTask { name, args, span } => self.sys_task(name, args, *span),
            Stmt::Event {
                control,
                args,
                body,
                span,
            } => self.event(control, args, body, span.line),
            Stmt::Call { name, args, .. } => {
                // An analog-function call statement: inline it for its output-arg
                // side effects (the return value is discarded). Unknown call
                // statements (tasks) are ignored.
                if self.em.functions.contains_key(name) {
                    self.call(name, args).map(|_| ())
                } else {
                    Ok(())
                }
            }
            Stmt::Assign { lhs, rhs, .. } => {
                let v = self.expr(rhs)?;
                // A non-finite constant is baked into the residual as a
                // bias-independent NaN/Inf that poisons every Newton step.
                // Surface it UNCONDITIONALLY as a captured warning (re-raised
                // as a catchable SaneConvergenceWarning regardless of the log
                // level, #41), since the first non-finite variable is the root
                // cause. `SANE_VA_TRACE_NAN` adds per-assignment verbosity.
                if let Some(c) = self.ctx().const_f64(v).filter(|c| !c.is_finite()) {
                    sane_core::log::warn_captured(&format!(
                        "VA non-finite constant baked into '{lhs}' = {c} (module {}); it \
                         poisons every Newton step of this model",
                        self.em.name
                    ));
                    if sane_core::config().va_trace_nan {
                        sane_core::log::warning(&format!("VA non-finite const: {lhs} = {c}"));
                    }
                }
                self.set_var(lhs.clone(), v);
                Ok(())
            }
            Stmt::Contribution {
                access,
                hi,
                lo,
                rhs,
                ..
            } => self.contribute(access, hi, lo, rhs),
            Stmt::Indirect {
                hi, lo, lhs, rhs, ..
            } => self.indirect(hi, lo, lhs, rhs),
            Stmt::InitialStep(inner) => self.stmt(inner),
            Stmt::If { cond, then, els } => self.lower_if(cond, then, els.as_deref()),
            Stmt::Case {
                sel,
                items,
                default,
            } => self.lower_case(sel, items, default.as_deref()),
            Stmt::For {
                init,
                cond,
                step,
                body,
            } => {
                self.stmt(init)?;
                self.lower_loop(cond, body, Some(step))
            }
            Stmt::While { cond, body } => self.lower_loop(cond, body, None),
        }
    }

    pub(super) fn lower_if(
        &mut self,
        cond: &Expr,
        then: &Stmt,
        els: Option<&Stmt>,
    ) -> Result<(), String> {
        let c = self.expr(cond)?;
        // A condition the graph folds (literals, string parameters,
        // `$param_given`) takes its arm statically: no merge, and a voltage
        // contribution inside it stays top-level.
        if let Some(v) = self.ctx().const_f64(c) {
            if v != 0.0 {
                return self.stmt(then);
            } else if let Some(e) = els {
                return self.stmt(e);
            }
            return Ok(());
        }
        // A condition on the parameters that decides the topology (an arm
        // contributes a potential, a branch shorted or a source) takes its
        // arm by the instance's values, as structure.
        if let Some(v) = self.topology(c, then, els) {
            return match (v, els) {
                (true, _) => self.stmt(then),
                (false, Some(e)) => self.stmt(e),
                (false, None) => Ok(()),
            };
        }
        // Lower each arm against the live state, recording only what it
        // writes, then rewind to the pre-branch state. No environment clone.
        let then_w = self.lower_branch(c, |l| l.stmt(then))?;
        let else_w = match els {
            Some(e) => {
                let nc = self.not(c);
                self.lower_branch(nc, |l| l.stmt(e))?
            }
            None => Writes::default(),
        };
        self.merge_writes(c, &then_w, &else_w);
        Ok(())
    }

    /// Where `c` reads the parameters only and an arm of `then` / `els`
    /// contributes a potential, `c`'s truth at the instance's values,
    /// recorded as structure; else `None`.
    pub(super) fn topology(&mut self, c: ExprId, then: &Stmt, els: Option<&Stmt>) -> Option<bool> {
        if !contributes_potential(then) && !els.is_some_and(contributes_potential) {
            return None;
        }
        let free = self.lo.ctx().free_symbols(c);
        let mut env = std::collections::HashMap::new();
        let mut names = Vec::new();
        for (name, &s) in &self.param_syms {
            if free.contains(&s) {
                env.insert(s, *self.values.get(name)?);
                names.push(name.clone());
            }
        }
        if env.len() != free.len() {
            return None; // reads something else too (the temperature)
        }
        let v = rsdag::eval::<f64, _>(self.lo.ctx(), &[c], &env)[0];
        if !v.is_finite() {
            return None;
        }
        names.sort();
        let holds = if v != 0.0 { c } else { self.not(c) };
        let holds = {
            let ctx = self.ctx();
            let zero = ctx.zero();
            ctx.cmp(CmpOp::Ne, holds, zero)
        };
        self.structure.push(Assertion {
            holds,
            message: format!(
                "{} {} structural: the topology of module {} depends on {}; build the model \
                 at the new value to change it",
                names.join(", "),
                if names.len() == 1 { "is" } else { "are" },
                self.em.name,
                if names.len() == 1 { "it" } else { "them" },
            ),
        });
        Some(v != 0.0)
    }

    /// Integer parameter `name`'s value, folded as structure.
    pub(super) fn int_value(&mut self, name: &str) -> Option<ExprId> {
        let p = self.em.params.iter().find(|p| p.name == name)?;
        if p.ty != crate::ast::VarType::Integer {
            return None;
        }
        let v = *self.values.get(name)?;
        let sym = format!("{}.{}", self.inst, name);
        let ctx = self.ctx();
        let (e, k) = (ctx.sym(&sym), ctx.konst_f64(v));
        if let Some(s) = sym_of(self.ctx(), e) {
            if self.param_syms.insert(name.to_string(), s).is_none() {
                let holds = self.ctx().cmp(CmpOp::Eq, e, k);
                self.structure.push(Assertion {
                    holds,
                    message: format!(
                        "{name} is structural (built at {v}); build the model with the new \
                         value to change it"
                    ),
                });
            }
        }
        Some(k)
    }

    /// The conjunction of the conditions the statement being lowered is
    /// reached under (`1` at the top level).
    pub(super) fn reached(&mut self) -> ExprId {
        let path = self.path.clone();
        let ctx = self.ctx();
        let one = ctx.one();
        path.into_iter().fold(one, |acc, c| {
            let zero = ctx.zero();
            let nz = ctx.cmp(CmpOp::Ne, c, zero);
            ctx.mul(acc, nz)
        })
    }

    /// Logical negation (`!c`, as Verilog-A reads a truth).
    pub(super) fn not(&mut self, c: ExprId) -> ExprId {
        let ctx = self.ctx();
        let zero = ctx.zero();
        ctx.cmp(CmpOp::Eq, c, zero)
    }

    /// Whether `e` reads the instance's parameters (and the temperature) only:
    /// a value fixed by a binding.
    pub(super) fn over_params(&mut self, e: ExprId) -> bool {
        let temp = self.ctx().sym(sane_core::constants::TEMP_SYMBOL);
        let temp = sym_of(self.lo.ctx(), temp);
        let params: HashSet<SymbolId> = self.param_syms.values().copied().collect();
        self.lo
            .ctx()
            .free_symbols(e)
            .iter()
            .all(|s| params.contains(s) || Some(*s) == temp)
    }

    /// `@(cross(expr, dir))` / `@(above(expr))`: declare a switching surface
    /// `expr = 0` the transient integrator lands on (direction `0` either
    /// way, `+1` rising, `-1` falling; `above` is rising). The body must be
    /// passive (empty, `$discontinuity`, `$bound_step`, ignored tasks): an
    /// event body executes only at the event instant, which needs a discrete
    /// state the one continuous model does not carry. Other controls
    /// (`timer`, `final_step`) are rejected for the same reason.
    pub(super) fn event(
        &mut self,
        control: &str,
        args: &[Expr],
        body: &Stmt,
        line: u32,
    ) -> Result<(), String> {
        let default_dir = match control {
            "cross" => Crossing::Either,
            "above" => Crossing::Rising,
            _ => {
                return Err(format!(
                    "unsupported analog event '@({control} ...)' (module {}, line {}): SANE lowers one \
                     analysis-agnostic model and cannot honor timer/final_step event controls; \
                     running the guarded body unconditionally would silently change the model, \
                     so it is rejected. Supported: @(initial_step), @(cross ...), @(above ...).",
                    self.em.name, line
                ))
            }
        };
        if !passive_event_body(body) {
            return Err(format!(
                "analog event '@({control} ...)' with a body (module {}, line {}): the body runs \
                 only at the event instant, which needs a discrete state SANE does not carry; \
                 only an empty body (or $discontinuity / $bound_step) is supported.",
                self.em.name, line
            ));
        }
        let Some(surface) = args.first() else {
            return Err(format!(
                "'@({control}())' needs a surface expression (module {}, line {})",
                self.em.name, line
            ));
        };
        let g = self.expr(surface)?;
        let dir = match args.get(1) {
            None => default_dir,
            Some(d) => {
                let d = self.expr(d)?;
                match self.ctx().const_f64(d) {
                    Some(c) if c > 0.0 => Crossing::Rising,
                    Some(c) if c < 0.0 => Crossing::Falling,
                    Some(_) => Crossing::Either,
                    None => {
                        return Err(format!(
                            "'@({control} ...)' direction must be a compile-time constant \
                             (module {}, line {})",
                            self.em.name, line
                        ))
                    }
                }
            }
        };
        // A surface on a path the parameters decide exists only where the
        // path is taken: elsewhere it is held off zero and never crosses.
        let reached = self.reached();
        let g = match self.ctx().const_f64(reached) {
            Some(r) if r != 0.0 => g,
            Some(_) => return Ok(()),
            None if self.over_params(reached) => {
                let one = self.ctx().one();
                self.ctx().select(reached, g, one)
            }
            None => g,
        };
        let ev = FragmentEvent { g, dir };
        if !self.events.contains(&ev) {
            self.events.push(ev);
        }
        Ok(())
    }

    /// Lower a conditional arm, reached when `c` holds, at `cond_depth + 1`
    /// (so its writes are journaled and minting an unknown is forbidden),
    /// capture what it wrote, then rewind the state to before the arm.
    pub(super) fn lower_branch(
        &mut self,
        c: ExprId,
        arm: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<Writes, String> {
        let mark = self.journal.len();
        self.cond_depth += 1;
        self.path.push(c);
        let r = arm(self);
        self.path.pop();
        self.cond_depth -= 1;
        r?;
        let w = self.collect_writes(mark);
        self.rewind(mark);
        Ok(w)
    }

    /// Merge the two arms' writes into the (pre-branch) state with `select(c,..)`,
    /// touching only variables an arm actually wrote. A key unwritten by an arm
    /// keeps its pre-branch value.
    pub(super) fn merge_writes(&mut self, c: ExprId, then_w: &Writes, else_w: &Writes) {
        let zero = self.ctx().zero();
        let union = |a: &Writes, b: &Writes, pick: fn(&Writes) -> Vec<String>| {
            let mut ks = pick(a);
            for k in pick(b) {
                if !ks.contains(&k) {
                    ks.push(k);
                }
            }
            ks
        };
        // vars
        for k in union(then_w, else_w, |w| w.vars.keys().cloned().collect()) {
            let base = self.st.vars.get(&k).copied().unwrap_or(zero);
            let t = then_w.vars.get(&k).copied().unwrap_or(base);
            let e = else_w.vars.get(&k).copied().unwrap_or(base);
            let v = if t == e {
                t
            } else {
                self.ctx().select(c, t, e)
            };
            self.set_var(k, v);
        }
        // node currents
        for k in union(then_w, else_w, |w| w.node_cur.keys().cloned().collect()) {
            let base = self.st.node_cur.get(&k).copied().unwrap_or(zero);
            let t = then_w.node_cur.get(&k).copied().unwrap_or(base);
            let e = else_w.node_cur.get(&k).copied().unwrap_or(base);
            let v = if t == e {
                t
            } else {
                self.ctx().select(c, t, e)
            };
            if self.cond_depth > 0 {
                self.journal
                    .push(Undo::NodeCur(k.clone(), self.st.node_cur.get(&k).copied()));
            }
            self.st.node_cur.insert(k, v);
        }
    }

    pub(super) fn lower_case(
        &mut self,
        sel: &Expr,
        items: &[(Vec<Expr>, Stmt)],
        default: Option<&Stmt>,
    ) -> Result<(), String> {
        let chain = case_to_if_chain(sel, items, default);
        self.stmt(&chain)
    }

    /// A loop, its init run: `while (cond) { body; step }`. An iteration
    /// whose condition the graph folds runs or ends the loop; any other runs
    /// gated by its condition, as `if (cond) { body; step }`, so the unrolled
    /// loop is the module's wherever the loop ends within the iterations
    /// unrolled. Past [`VA_LOOP_GATED_CAP`] gated iterations the loop ends:
    /// with an assertion that it does when the condition reads parameters
    /// only, else it cannot be lowered.
    pub(super) fn lower_loop(
        &mut self,
        cond: &Expr,
        body: &Stmt,
        step: Option<&Stmt>,
    ) -> Result<(), String> {
        let (mut iters, mut gated) = (0usize, 0usize);
        loop {
            let c = self.expr(cond)?;
            match self.ctx().const_f64(c) {
                Some(0.0) => return Ok(()),
                Some(_) => {
                    self.stmt(body)?;
                    if let Some(st) = step {
                        self.stmt(st)?;
                    }
                }
                None if gated == VA_LOOP_GATED_CAP => {
                    if !self.over_params(c) {
                        return Err(format!(
                            "loop condition depends on the solution and does not settle \
                             within {VA_LOOP_GATED_CAP} iterations (module {})",
                            self.em.name
                        ));
                    }
                    let holds = self.not(c);
                    self.assertions.push(Assertion {
                        holds,
                        message: format!(
                            "a loop of module {} runs more than {VA_LOOP_GATED_CAP} \
                             iterations for these parameters",
                            self.em.name
                        ),
                    });
                    return Ok(());
                }
                None => {
                    gated += 1;
                    let w = self.lower_branch(c, |l| {
                        l.stmt(body)?;
                        match step {
                            Some(st) => l.stmt(st),
                            None => Ok(()),
                        }
                    })?;
                    self.merge_writes(c, &w, &Writes::default());
                }
            }
            iters += 1;
            if iters > MAX_UNROLL {
                return Err(format!(
                    "loop exceeded the unroll cap of {MAX_UNROLL} iterations (module {})",
                    self.em.name
                ));
            }
        }
    }

    /// Compile-time value of a string expression: a literal, or a string
    /// parameter's default. `None` for anything else.
    pub(super) fn const_str<'e>(&self, e: &'e Expr) -> Option<&'e str>
    where
        'a: 'e,
    {
        match e {
            Expr::Str(s) => Some(s),
            Expr::Ident(name, _) => self.em.string_params.get(name).map(String::as_str),
            _ => None,
        }
    }

    /// Fold a string comparison `a == b` / `a != b` where both sides are
    /// compile-time strings (a literal or a string parameter). String
    /// parameters exist only at compile time, so this is the ONLY operation
    /// they support -- the model-variant-selector pattern.
    pub(super) fn fold_str_cmp(&self, op: BinOp, lhs: &Expr, rhs: &Expr) -> Option<f64> {
        if !matches!(op, BinOp::Eq | BinOp::Ne) {
            return None;
        }
        let (a, b) = (self.const_str(lhs)?, self.const_str(rhs)?);
        let eq = a == b;
        Some(bool_f64(if matches!(op, BinOp::Eq) { eq } else { !eq }))
    }
}

/// Desugar a `case` statement to a nested if-chain
/// `if (sel==l0 || ...) body0 else if ... else default`.
pub(super) fn case_to_if_chain(
    sel: &Expr,
    items: &[(Vec<Expr>, Stmt)],
    default: Option<&Stmt>,
) -> Stmt {
    let mut acc: Stmt = default.cloned().unwrap_or(Stmt::Empty);
    for (labels, body) in items.iter().rev() {
        let mut cond: Option<Expr> = None;
        for l in labels {
            let eq = Expr::Binary {
                op: BinOp::Eq,
                lhs: Box::new(sel.clone()),
                rhs: Box::new(l.clone()),
                span: l.span(),
            };
            cond = Some(match cond {
                None => eq,
                Some(prev) => Expr::Binary {
                    op: BinOp::Or,
                    lhs: Box::new(prev),
                    rhs: Box::new(eq),
                    span: l.span(),
                },
            });
        }
        let cond = cond.unwrap_or(Expr::Num(1.0));
        acc = Stmt::If {
            cond,
            then: Box::new(body.clone()),
            els: Some(Box::new(acc)),
        };
    }
    acc
}

/// Is an event body free of model semantics (see `Lower::event`)?
pub(super) fn passive_event_body(s: &Stmt) -> bool {
    match s {
        Stmt::Empty | Stmt::IgnoredCall => true,
        Stmt::Block(ss) => ss.iter().all(passive_event_body),
        Stmt::SysTask { name, .. } => matches!(name.as_str(), "discontinuity" | "bound_step"),
        _ => false,
    }
}
