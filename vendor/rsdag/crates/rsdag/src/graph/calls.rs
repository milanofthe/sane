//! Functions, calls and the system layer over them: defining a function,
//! calling it, roles, role-selected Jacobians, feedthrough, inlining.
//!
//! A child module of `graph` so it reaches the arena's private fields; a
//! second `impl Graph` block rather than a second type, because a call is a
//! node like any other and the functions live in the same arena.

use super::*;

impl<K: Field> Graph<K> {
    // --- functions and calls ---------------------------------------------

    /// Append a function; the ids stay in definition order.
    pub(crate) fn push_function(&mut self, func: Function) -> FuncId {
        self.funcs.push(func);
        FuncId(self.funcs.len() as u32 - 1)
    }

    /// Append an output with its role to `f`; returns its index.
    pub(crate) fn push_output(&mut self, f: FuncId, output: Output, role: OutputRole) -> u32 {
        self.funcs[f.0 as usize].push_output(output, role)
    }

    /// Define a symbolic function: `outputs` are expressions over the formal
    /// `params` (free symbols of the outputs not listed in `params` are shared
    /// globals every call sees unchanged).
    pub fn define_func(
        &mut self,
        name: &str,
        params: Vec<SymbolId>,
        outputs: Vec<ExprId>,
    ) -> FuncId {
        let f = self.push_function(Function::new(name, params, None));
        for e in outputs {
            self.push_output(f, Output::Expr(e), OutputRole::Plain);
        }
        f
    }

    /// Close an open graph over `outputs` into a function: every symbol the
    /// outputs mention becomes a parameter, in symbol order; a called
    /// function's globals stay globals. The `Scope` idiom: build with named
    /// symbols, then close.
    pub fn close(&mut self, name: &str, outputs: Vec<ExprId>) -> FuncId {
        let params: Vec<SymbolId> = self.mentioned_symbols_in(&outputs).into_iter().collect();
        self.define_func(name, params, outputs)
    }

    /// Set the role of parameter `param` of function `f`.
    pub fn set_param_role(&mut self, f: FuncId, param: u32, role: ParamRole) {
        self.funcs[f.0 as usize].set_param_role(param, role);
    }

    /// Set the role of output `out` of function `f`.
    pub fn set_output_role(&mut self, f: FuncId, out: u32, role: OutputRole) {
        self.funcs[f.0 as usize].set_output_role(out, role);
    }

    /// Inline every call reachable from `roots`, to the bottom.
    ///
    /// The static fusion a consumer does before it compiles: a hierarchy of
    /// blocks or subcircuits becomes one expression per root, so the
    /// scheduler, the slot allocator and common-subexpression elimination
    /// see across the instance boundaries that the calls were hiding. The
    /// price is size -- an instance that was one `Call` becomes a copy of
    /// its body -- which is why it is a choice and not the default.
    ///
    /// Calls into an extern body cannot be inlined (there is no expression
    /// to inline) and are left as they are.
    pub fn inline_all(&mut self, roots: &[ExprId]) -> Vec<ExprId> {
        self.inline_where(roots, &mut |_, _| true)
    }

    /// Every call into a function whose body calls further functions,
    /// inlined, to the bottom; the calls of leaf functions (a device model's
    /// body) stay calls: the hierarchy flattened down to its leaves, as an
    /// expression. A compiled program needs no such rewrite: the tape
    /// compiler takes a composite function as a template and batches the
    /// leaf calls of every instance itself (see [`Tape::compile`](crate::Tape::compile)).
    pub fn inline_composite(&mut self, roots: &[ExprId]) -> Vec<ExprId> {
        let mut composite: HashMap<FuncId, bool> = HashMap::default();
        self.inline_where(roots, &mut |g, f| {
            *composite.entry(f).or_insert_with(|| {
                let exprs: Vec<ExprId> = g.funcs[f.0 as usize]
                    .outputs()
                    .iter()
                    .filter_map(|o| match *o {
                        Output::Expr(e) => Some(e),
                        _ => None,
                    })
                    .collect();
                !g.free_calls_in(&exprs).is_empty()
            })
        })
    }

    /// The calls into the functions `which` names inlined, to the bottom.
    fn inline_where(
        &mut self,
        roots: &[ExprId],
        which: &mut dyn FnMut(&Self, FuncId) -> bool,
    ) -> Vec<ExprId> {
        self.inline_with(
            roots,
            which,
            &mut HashMap::default(),
            &mut HashMap::default(),
        )
    }

    /// Every call under `roots` that passes constants, redirected to a copy
    /// of its function specialized to them: the copy's outputs are the
    /// function's with those parameters replaced by the constants (and
    /// folded), it takes the other arguments only. The calls that pass the
    /// same constants in the same places share one copy, so their instances
    /// still run as one batch, and calls in a copy's body are specialized
    /// too. A ground terminal, or the derivatives of a DC analysis set to
    /// zero, take their share of a device body away before it is compiled.
    /// A parameter with the `Param` role stays an argument when constant:
    /// its work is the body's prolog, once per binding, and the instances
    /// keep sharing one body whatever their parameter values.
    ///
    /// The copy keeps the roles of the parameters it keeps; its outputs are
    /// `Plain` apart from their non-derivative roles, a derivative of it is
    /// derived anew. Calls into an extern body are left as they are.
    pub fn specialize_calls(&mut self, roots: &[ExprId]) -> Vec<ExprId> {
        specialize_calls_in(self, roots, &mut HashMap::default())
    }

    /// [`inline_where`](Self::inline_where) with `bodies` holding each
    /// output of an inlined function already inlined over its parameters
    /// (all of a function's outputs at once, so their shared body is walked
    /// once), and `instances` the substitution of each instance, kept across
    /// its outputs: an instance's body is rewritten once, however many of
    /// its outputs are called.
    fn inline_with(
        &mut self,
        roots: &[ExprId],
        which: &mut dyn FnMut(&Self, FuncId) -> bool,
        bodies: &mut HashMap<OutputId, ExprId>,
        instances: &mut HashMap<(FuncId, u32, ArgList), Instance>,
    ) -> Vec<ExprId> {
        crate::transform::rewrite(self, roots, |g, _, node, ops| {
            let (Node::Call(o, _), Some(l)) = (node, ops.list) else {
                return g.rebuild(node, ops);
            };
            let (f, k) = g.output(o);
            match g.funcs[f.0 as usize].outputs()[k as usize] {
                Output::Zero => return g.zero,
                // An extern body, or a function not to inline, stays a call
                // over inlined arguments.
                Output::Slot(_) => return g.rebuild(node, ops),
                Output::Expr(_) if !which(g, f) => return g.rebuild(node, ops),
                Output::Expr(_) => {}
            }
            // the body of the output, whatever context it is called in
            let plain = g.output_id(f, k);
            if !bodies.contains_key(&plain) {
                let outs: Vec<(u32, ExprId)> = g.funcs[f.0 as usize]
                    .outputs()
                    .iter()
                    .enumerate()
                    .filter_map(|(k, out)| match *out {
                        Output::Expr(e) => Some((k as u32, e)),
                        _ => None,
                    })
                    .collect();
                let exprs: Vec<ExprId> = outs.iter().map(|&(_, e)| e).collect();
                let inlined = g.inline_with(&exprs, which, bodies, instances);
                for (&(k, _), b) in outs.iter().zip(inlined) {
                    let ok = g.output_id(f, k);
                    bodies.insert(ok, b);
                }
            }
            let body = bodies[&plain];
            let inst = instances
                .entry((f, ops.ctx, l))
                .or_insert_with(|| Instance {
                    bound: g.binding(f, &g.full_args_in(ops.ctx, l)),
                    memo: HashMap::default(),
                    lists: HashMap::default(),
                    contexts: HashMap::default(),
                });
            inst.apply(g, body)
        })
    }

    /// `f` with the globals `map` binds to something else than themselves
    /// substituted in its body, its nested calls' included: the function a
    /// call of `f` runs where `map` holds (a substitution, a caller binding
    /// a parameter some called body reads as a global). `None` when `map`
    /// binds none of them. One copy per function and binding; the copy
    /// keeps `f`'s parameters, roles and output indices, and the derivative
    /// roles with respect to its parameters.
    pub(crate) fn rebound(&mut self, f: FuncId, map: &HashMap<SymbolId, ExprId>) -> Option<FuncId> {
        let globals = self.globals(f);
        let mut binding: Vec<(SymbolId, ExprId)> = globals
            .iter()
            .filter_map(|&e| match *self.node(e) {
                Node::Symbol(s) => map.get(&s).filter(|&&v| v != e).map(|&v| (s, v)),
                _ => None,
            })
            .collect();
        if binding.is_empty() {
            return None;
        }
        binding.sort_unstable();
        if let Some(&copy) = self.rebound.get(&(f, binding.clone())) {
            return Some(copy);
        }
        let func = self.func(f);
        let (name, params) = (func.name().to_string(), func.params().to_vec());
        let roles = func.param_roles().to_vec();
        let (outputs, out_roles) = (func.outputs().to_vec(), func.output_roles().to_vec());
        let bound: HashMap<SymbolId, ExprId> = binding.iter().copied().collect();
        let exprs: Vec<ExprId> = outputs
            .iter()
            .filter_map(|o| match *o {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        let mut done = crate::transform::substitute(self, &exprs, &bound).into_iter();
        let copy = self.push_function(Function::new(&name, params.clone(), None));
        for (k, &r) in roles.iter().enumerate() {
            self.set_param_role(copy, k as u32, r);
        }
        for (o, role) in outputs.iter().zip(out_roles) {
            let out = match *o {
                Output::Expr(_) => Output::Expr(done.next().expect("one per expression")),
                other => other,
            };
            let role = match role {
                OutputRole::Derivative { wrt, .. } if wrt as usize >= params.len() => {
                    OutputRole::Plain
                }
                r => r,
            };
            self.push_output(copy, out, role);
        }
        self.rebound.insert((f, binding), copy);
        Some(copy)
    }

    /// The parameters of `f` bound to the arguments of a call.
    fn binding(&self, f: FuncId, args: &[ExprId]) -> HashMap<SymbolId, ExprId> {
        let params = self.funcs[f.0 as usize].params();
        params.iter().copied().zip(args.iter().copied()).collect()
    }

    /// The operands of a call (its parameters, then its globals, see
    /// [`globals`](Self::globals)) output `out` of `f` can have a nonzero
    /// derivative in, by index, ascending: its [`support_in`](Self::support_in)
    /// among them. Structural, read off the graph. The outputs of a function
    /// share their body, so the ones not known yet are found in one pass
    /// over it, and each is kept. An extern output is taken to read every
    /// parameter, a zero one none.
    pub fn output_support(&self, f: FuncId, out: u32) -> Arc<[u32]> {
        let func = &self.funcs[f.0 as usize];
        if let Some(s) = func.cached_support(out) {
            return s;
        }
        let n = func.params().len() + self.globals(f).len();
        let all: Vec<u32> = (0..n as u32).collect();
        let reads = self.reads(f, Through::Carries, &all);
        for (k, r) in reads.iter().enumerate() {
            func.cache_support(k as u32, r.clone());
        }
        func.cached_support(out).expect("just found")
    }

    /// Per root, whether its value reads any of `syms`, a comparison's
    /// operands and a selector's condition included: which entries of a
    /// Jacobian vary with the state, say. One pass over the roots' cone; a
    /// call reads them if an argument that does goes into its output's
    /// value, which the called function answers once per pattern of such
    /// arguments.
    pub fn depends_on(&self, roots: &[ExprId], syms: &[SymbolId]) -> Vec<bool> {
        let wanted: FxHashSet<SymbolId> = syms.iter().copied().collect();
        let flow = self.flow(roots, Through::Reads, |n| match *n {
            Node::Symbol(s) => wanted.contains(&s),
            _ => false,
        });
        roots.iter().map(|&r| *flow.get(r)).collect()
    }

    /// The symbols `exprs` can have a nonzero derivative in: their
    /// [`free_symbols_in`](Self::free_symbols_in) through the operands that
    /// carry a derivative (not a comparison's, not a selector's condition),
    /// and through a call only the arguments its output's
    /// [`output_support`](Self::output_support) names. The sparsity of every
    /// derivative of `exprs`, nested calls included.
    pub fn support_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        let flow = self.flow(exprs, Through::Carries, |n| match *n {
            Node::Symbol(s) => Set::one(s.0, self.n_symbols()),
            _ => Set::bottom(),
        });
        let mut all = Set::bottom();
        for &e in exprs {
            all.join(flow.get(e));
        }
        all.iter().map(SymbolId).collect()
    }

    /// Which outputs of `f` structurally read which of its parameters:
    /// `feedthrough(f)[out][param]`.
    ///
    /// Structural, not numeric: it asks whether the parameter occurs in the
    /// output's expression at all, so it costs one walk per output and needs
    /// no differentiation. That is the question a block scheduler asks --
    /// direct feedthrough decides the evaluation order, and a cycle among
    /// the blocks that have it is an algebraic loop.
    ///
    /// An extern output is opaque and is reported as reading every
    /// parameter, which is the safe direction: it can cost an ordering
    /// constraint, never a missed loop. A zero output reads nothing.
    pub fn feedthrough(&self, f: FuncId) -> Vec<Vec<bool>> {
        let n = self.funcs[f.0 as usize].params().len();
        let all: Vec<u32> = (0..n as u32).collect();
        self.reads(f, Through::Syntax, &all)
            .iter()
            .map(|r| {
                let mut row = vec![false; n];
                r.iter().for_each(|&p| row[p as usize] = true);
                row
            })
            .collect()
    }

    /// Jacobian of the outputs of `f` with a role against its parameters with
    /// a role: `(output index, param index, derivative output index)` for
    /// every structurally nonzero pair, differentiating on demand.
    pub fn jacobian_by_role(
        &mut self,
        f: FuncId,
        out_role: impl Fn(&OutputRole) -> bool,
        param_role: impl Fn(&ParamRole) -> bool,
    ) -> Vec<(u32, u32, u32)> {
        let outs = self.funcs[f.0 as usize].outputs_with_role(out_role);
        let pars = self.funcs[f.0 as usize].params_with_role(param_role);
        let mut entries = Vec::new();
        for &o in &outs {
            for (&p, k) in pars.iter().zip(self.derivative_outputs(f, o, &pars)) {
                if !matches!(self.funcs[f.0 as usize].outputs()[k as usize], Output::Zero) {
                    entries.push((o, p, k));
                }
            }
        }
        entries
    }

    /// Define an extern function over `arity` arguments whose outputs are the
    /// given slots of `body` (or [`Output::Zero`]). Derivative outputs the body
    /// carries are declared with [`declare_derivative`](Self::declare_derivative).
    pub fn define_extern_func(
        &mut self,
        name: &str,
        arity: usize,
        body: Arc<dyn ExternBundle>,
        outputs: Vec<Output>,
    ) -> FuncId {
        let params: Vec<SymbolId> = (0..arity)
            .map(|i| {
                let e = self.sym(&format!("{name}.${i}"));
                match *self.node(e) {
                    Node::Symbol(s) => s,
                    _ => unreachable!("sym yields a symbol"),
                }
            })
            .collect();
        self.define_extern_func_with_params(name, params, body, outputs)
    }

    /// Define an extern function over the given formal parameters (the
    /// symbols already exist), see [`define_extern_func`](Self::define_extern_func).
    pub fn define_extern_func_with_params(
        &mut self,
        name: &str,
        params: Vec<SymbolId>,
        body: Arc<dyn ExternBundle>,
        outputs: Vec<Output>,
    ) -> FuncId {
        let f = self.push_function(Function::new(name, params, Some(body)));
        for o in outputs {
            self.push_output(f, o, OutputRole::Plain);
        }
        f
    }

    /// Declare `d outputs[out] / d params[param]` of an extern function as
    /// `deriv` (a slot of its body, or zero). Undeclared derivatives are zero.
    pub fn declare_derivative(&mut self, f: FuncId, out: u32, param: u32, deriv: Output) -> u32 {
        self.push_output(
            f,
            deriv,
            OutputRole::Derivative {
                of: out,
                wrt: param,
            },
        )
    }

    pub fn func(&self, f: FuncId) -> &Function {
        &self.funcs[f.0 as usize]
    }

    /// Number of symbols.
    pub fn n_symbols(&self) -> usize {
        self.symbol_names.len()
    }

    /// Register a body a consumer compiled for the symbolic function `f`:
    /// from now on a tape calls `body` (see [`Function::compiled`]). A tape
    /// compiled before keeps the body it was compiled with.
    pub fn set_func_body(&mut self, f: FuncId, body: crate::func::Body) {
        assert!(
            !self.funcs[f.0 as usize].is_extern(),
            "an extern function is its own body"
        );
        // Several bodies may serve one function (a residual-only one and
        // one with the partials); a program takes the smallest that covers
        // the outputs it calls.
        let bodies = self.funcs[f.0 as usize].compiled_mut();
        let ptr = Arc::as_ptr(&body.bundle) as *const () as usize;
        if !bodies
            .iter()
            .any(|b| Arc::as_ptr(&b.bundle) as *const () as usize == ptr)
        {
            bodies.push(body);
        }
    }

    pub fn n_funcs(&self) -> usize {
        self.funcs.len()
    }

    /// The `(function, output index)` an output id names.
    #[inline]
    pub fn output(&self, o: OutputId) -> (FuncId, u32) {
        self.outputs[o.0 as usize]
    }

    /// The interned id of output `out` of `f`.
    pub fn output_id(&mut self, f: FuncId, out: u32) -> OutputId {
        self.output_in(f, out, NO_CONTEXT)
    }

    /// The interned id of output `out` of `f` called in context `ctx`.
    pub(crate) fn output_in(&mut self, f: FuncId, out: u32, ctx: u32) -> OutputId {
        if let Some(&o) = self.output_dedup.get(&(f, out, ctx)) {
            return o;
        }
        let o = OutputId(self.outputs.len() as u32);
        self.outputs.push((f, out));
        self.output_ctx.push(ctx);
        self.output_dedup.insert((f, out, ctx), o);
        o
    }

    /// The context a call of output `o` runs in (see [`bind`](Self::bind)):
    /// its function's bound parameters, ascending, and their expressions.
    pub fn context(&self, o: OutputId) -> Option<(&[u32], &[ExprId])> {
        match self.output_ctx[o.0 as usize] {
            NO_CONTEXT => None,
            c => {
                let c = &self.contexts[c as usize];
                Some((&c.at, self.args(c.exprs)))
            }
        }
    }

    /// The list of the bound expressions of context `c`.
    pub(crate) fn context_list(&self, c: u32) -> ArgList {
        self.contexts[c as usize].exprs
    }

    /// The context id of output `o` (`NO_CONTEXT` for none).
    pub(crate) fn context_of(&self, o: OutputId) -> u32 {
        self.output_ctx[o.0 as usize]
    }

    /// `f` with the parameters at the given positions bound to expressions:
    /// a call through it passes the others, in their order, and runs `f`'s
    /// body with the bound ones taken from the binding. A model card is a
    /// binding of its device function: a call carries an instance's own
    /// arguments, the card is the binding's, once. The body, its
    /// derivatives and the positions of its parameters stay `f`'s; a bound
    /// call is in every respect the call of `f` with all its arguments.
    pub fn bind(&mut self, f: FuncId, bound: &[(u32, ExprId)]) -> Bound {
        let mut pairs: Vec<(u32, ExprId)> = bound.to_vec();
        pairs.sort_unstable();
        pairs.dedup_by_key(|p| p.0);
        if pairs.is_empty() {
            return Bound {
                func: f,
                ctx: NO_CONTEXT,
            };
        }
        let at: Box<[u32]> = pairs.iter().map(|&(k, _)| k).collect();
        let exprs: Vec<ExprId> = pairs.iter().map(|&(_, e)| e).collect();
        let exprs = self.intern_args(&exprs);
        let key = (f, at.clone(), exprs);
        if let Some(&ctx) = self.context_dedup.get(&key) {
            return Bound { func: f, ctx };
        }
        let n = self.funcs[f.0 as usize].params().len();
        let mut slot = vec![0u32; n];
        let (mut next, mut j) = (0u32, 0usize);
        for (p, s) in slot.iter_mut().enumerate() {
            if at.get(j) == Some(&(p as u32)) {
                *s = BOUND | j as u32;
                j += 1;
            } else {
                *s = next;
                next += 1;
            }
        }
        let ctx = self.contexts.len() as u32;
        self.contexts.push(Context {
            f,
            at,
            exprs,
            slot: slot.into(),
        });
        self.context_dedup.insert(key, ctx);
        Bound { func: f, ctx }
    }

    /// Context `c` with its bound expressions `exprs` for a call of `f`
    /// (`f` itself, or a copy with its parameters): the context id.
    pub(crate) fn context_over(&mut self, c: u32, f: FuncId, exprs: &[ExprId]) -> u32 {
        if c == NO_CONTEXT {
            return NO_CONTEXT;
        }
        let ctx = &self.contexts[c as usize];
        if ctx.f == f && self.args(ctx.exprs) == exprs {
            return c;
        }
        let pairs: Vec<(u32, ExprId)> = ctx.at.iter().copied().zip(exprs.iter().copied()).collect();
        self.bind(f, &pairs).ctx
    }

    /// Context `c` for a call of `f` (a copy with the parameters of `c`'s).
    pub(crate) fn context_onto(&mut self, c: u32, f: FuncId) -> u32 {
        if c == NO_CONTEXT {
            return NO_CONTEXT;
        }
        let exprs = self.args(self.contexts[c as usize].exprs).to_vec();
        self.context_over(c, f, &exprs)
    }

    /// The operands of a call of `f` in context `ctx` over `l`, in `f`'s
    /// parameter order (see [`full_args`](Self::full_args)).
    pub(crate) fn full_args_in(&self, ctx: u32, l: ArgList) -> Vec<ExprId> {
        let args = self.args(l);
        match ctx {
            NO_CONTEXT => args.to_vec(),
            c => {
                let c = &self.contexts[c as usize];
                let bound = self.args(c.exprs);
                c.slot
                    .iter()
                    .map(|&s| match s & BOUND {
                        0 => args[s as usize],
                        _ => bound[(s & !BOUND) as usize],
                    })
                    .collect()
            }
        }
    }

    /// The parameters a call through `b` passes, in order.
    pub fn bound_arity(&self, b: Bound) -> usize {
        let n = self.funcs[b.func.0 as usize].params().len();
        match b.ctx {
            NO_CONTEXT => n,
            c => n - self.contexts[c as usize].at.len(),
        }
    }

    /// A call of output `out` through `b`, over the parameters it leaves.
    pub fn call_bound(&mut self, b: Bound, out: u32, args: &[ExprId]) -> ExprId {
        debug_assert_eq!(args.len(), self.bound_arity(b), "call arity");
        let l = self.intern_args(args);
        self.call_list_in(b.func, out, b.ctx, l)
    }

    /// [`call_bound`](Self::call_bound) for several outputs over one list.
    pub fn calls_bound(&mut self, b: Bound, outs: &[u32], args: &[ExprId]) -> Vec<ExprId> {
        debug_assert_eq!(args.len(), self.bound_arity(b), "call arity");
        let l = self.intern_args(args);
        outs.iter()
            .map(|&out| self.call_list_in(b.func, out, b.ctx, l))
            .collect()
    }

    /// The operands of a call of output `o` over `l` in `f`'s parameter
    /// order: an argument, or a bound expression where `o`'s context binds
    /// the parameter.
    pub(crate) fn full_args(&self, o: OutputId, l: ArgList) -> std::borrow::Cow<'_, [ExprId]> {
        match self.output_ctx[o.0 as usize] {
            NO_CONTEXT => std::borrow::Cow::Borrowed(self.args(l)),
            c => {
                let c = &self.contexts[c as usize];
                let (args, bound) = (self.args(l), self.args(c.exprs));
                std::borrow::Cow::Owned(
                    c.slot
                        .iter()
                        .map(|&s| match s & BOUND {
                            0 => args[s as usize],
                            _ => bound[(s & !BOUND) as usize],
                        })
                        .collect(),
                )
            }
        }
    }

    /// The expression of a symbolic output, `None` for a slot or zero output.
    pub fn output_expr(&self, f: FuncId, out: u32) -> Option<ExprId> {
        match self.funcs[f.0 as usize].outputs()[out as usize] {
            Output::Expr(e) => Some(e),
            _ => None,
        }
    }

    /// Output `out` of `f` applied to `args` (one argument per parameter). A
    /// zero output folds to the constant zero.
    pub fn call(&mut self, f: FuncId, out: u32, args: &[ExprId]) -> ExprId {
        let func = &self.funcs[f.0 as usize];
        debug_assert_eq!(args.len(), func.params().len(), "call arity");
        if matches!(func.outputs()[out as usize], Output::Zero) {
            return self.zero;
        }
        let l = self.intern_args(args);
        self.call_list(f, out, l)
    }

    /// Outputs `outs` of `f` called over one argument list: the calls of one
    /// instance. The list is interned once, so an instance of a wide body
    /// costs its width once, not once per output.
    pub fn calls(&mut self, f: FuncId, outs: &[u32], args: &[ExprId]) -> Vec<ExprId> {
        debug_assert_eq!(
            args.len(),
            self.funcs[f.0 as usize].params().len(),
            "call arity"
        );
        let l = self.intern_args(args);
        outs.iter().map(|&out| self.call_list(f, out, l)).collect()
    }

    /// [`call`](Self::call) over an argument list already interned.
    pub(crate) fn call_list(&mut self, f: FuncId, out: u32, l: ArgList) -> ExprId {
        self.call_list_in(f, out, NO_CONTEXT, l)
    }

    /// [`call_list`](Self::call_list) in context `ctx`.
    pub(crate) fn call_list_in(&mut self, f: FuncId, out: u32, ctx: u32, l: ArgList) -> ExprId {
        if matches!(
            self.funcs[f.0 as usize].outputs()[out as usize],
            Output::Zero
        ) {
            return self.zero;
        }
        let o = self.output_in(f, out, ctx);
        if ctx != NO_CONTEXT && !self.list_shape.contains_key(&(ctx, l)) {
            // a bound call's is the one over all its operands in parameter
            // order: the call it stands for
            let shape = self.full_args(o, l).iter().fold(0, |h, &a| {
                crate::node::shape::mix(h, self.shape[a.0 as usize])
            });
            self.list_shape.insert((ctx, l), shape);
        }
        self.intern(Node::Call(o, l))
    }

    /// [`call`](Self::call) by output id.
    pub fn call_output(&mut self, o: OutputId, args: &[ExprId]) -> ExprId {
        let (f, out) = self.output(o);
        let ctx = self.context_of(o);
        let l = self.intern_args(args);
        self.call_list_in(f, out, ctx, l)
    }

    /// The call of output `o` over [`operands`](Self::operands): its
    /// arguments, then its context's bound expressions (anew, whatever they
    /// are now).
    pub(crate) fn call_over_operands(&mut self, o: OutputId, ops: &[ExprId]) -> ExprId {
        let (f, out) = self.output(o);
        match self.context_of(o) {
            NO_CONTEXT => self.call(f, out, ops),
            c => {
                let nb = self.contexts[c as usize].at.len();
                let (args, bound) = ops.split_at(ops.len() - nb);
                let ctx = self.context_over(c, f, bound);
                let l = self.intern_args(args);
                self.call_list_in(f, out, ctx, l)
            }
        }
    }

    /// The index of the derivative output `d outputs[out] / d params[param]`,
    /// differentiating the body on first demand (symbolic functions) or
    /// looking up the declared slot (extern functions; zero if undeclared).
    pub fn derivative_output(&mut self, f: FuncId, out: u32, param: u32) -> u32 {
        let func = &self.funcs[f.0 as usize];
        if let Some(k) = func.derivative(out, param) {
            return k;
        }
        let output = func.outputs()[out as usize];
        let d = match output {
            Output::Expr(e) => {
                let wrt = self.operand_symbol(f, param);
                let de = crate::autodiff::differentiate(self, e, wrt);
                if self.is_zero(de) {
                    Output::Zero
                } else {
                    Output::Expr(de)
                }
            }
            Output::Slot(_) | Output::Zero => Output::Zero,
        };
        self.push_output(
            f,
            d,
            OutputRole::Derivative {
                of: out,
                wrt: param,
            },
        )
    }

    /// [`derivative_output`](Self::derivative_output) for several parameters
    /// of one output. The missing derivatives of a symbolic function are
    /// derived in one reverse sweep over the body when they are
    /// [`REVERSE_MIN_TOUCHED`](crate::autodiff::REVERSE_MIN_TOUCHED) or more
    /// (a device's parameters), in one forward sweep each otherwise.
    pub fn derivative_outputs(&mut self, f: FuncId, out: u32, params: &[u32]) -> Vec<u32> {
        let func = &self.funcs[f.0 as usize];
        if let Output::Expr(e) = func.outputs()[out as usize] {
            let missing: Vec<u32> = params
                .iter()
                .copied()
                .filter(|&p| func.derivative(out, p).is_none())
                .collect();
            if missing.len() >= crate::autodiff::REVERSE_MIN_TOUCHED {
                let wrt: Vec<SymbolId> =
                    missing.iter().map(|&p| self.operand_symbol(f, p)).collect();
                let grad = crate::autodiff::gradient(self, e, &wrt);
                for (&p, d) in missing.iter().zip(grad) {
                    let d = if self.is_zero(d) {
                        Output::Zero
                    } else {
                        Output::Expr(d)
                    };
                    self.push_output(f, d, OutputRole::Derivative { of: out, wrt: p });
                }
            }
        }
        params
            .iter()
            .map(|&p| self.derivative_output(f, out, p))
            .collect()
    }

    /// Inline a call: the output expression with the parameters replaced by
    /// `args`. `None` for an extern (slot) output, which has no body to inline.
    pub fn inline_call(&mut self, f: FuncId, out: u32, args: &[ExprId]) -> Option<ExprId> {
        match self.funcs[f.0 as usize].outputs()[out as usize] {
            Output::Slot(_) => None,
            _ => Some(self.inline_outputs(f, &[out], args)[0]),
        }
    }

    /// Inline several outputs of a symbolic function at once, with one shared
    /// substitution pass (the outputs of a device template share its core, so
    /// per-output substitution would rebuild that core per output).
    pub fn inline_outputs(&mut self, f: FuncId, outs: &[u32], args: &[ExprId]) -> Vec<ExprId> {
        let map = self.binding(f, args);
        let func = &self.funcs[f.0 as usize];
        let exprs: Vec<ExprId> = outs
            .iter()
            .map(|&o| match func.outputs()[o as usize] {
                Output::Expr(e) => e,
                Output::Zero => self.zero,
                Output::Slot(_) => panic!("cannot inline an extern function output"),
            })
            .collect();
        crate::transform::substitute(self, &exprs, &map)
    }

    /// Every `(function, output)` called anywhere in `exprs` (one pass over
    /// the forest). A solver uses it to compile exactly the outputs a set of
    /// roots reads.
    pub fn free_calls_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<OutputId> {
        self.cone_nodes(exprs, false)
            .into_iter()
            .filter_map(|e| match *self.node(e) {
                Node::Call(o, _) => Some(o),
                _ => None,
            })
            .collect()
    }

    /// The free symbols of `expr` (see
    /// [`free_symbols_in`](Self::free_symbols_in)).
    pub fn free_symbols(&self, expr: ExprId) -> std::collections::BTreeSet<SymbolId> {
        self.free_symbols_in(&[expr])
    }

    /// The symbols `exprs` read: what they mention and what the bodies they
    /// call read as globals (see [`globals`](Self::globals)), the inputs a
    /// program over them takes. One walk over the forest, each shared node
    /// once.
    pub fn free_symbols_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        self.symbols_of(self.cone_nodes(exprs, true))
    }

    /// The symbols `exprs` mention, the bodies they call not looked into: the
    /// parameters a function closed over them takes (see
    /// [`close`](Self::close)).
    pub fn mentioned_symbols_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        self.symbols_of(self.cone_nodes(exprs, false))
    }

    fn symbols_of(&self, nodes: Vec<ExprId>) -> std::collections::BTreeSet<SymbolId> {
        nodes
            .into_iter()
            .filter_map(|e| match *self.node(e) {
                Node::Symbol(s) => Some(s),
                _ => None,
            })
            .collect()
    }
}

/// The specialized copies made so far: `(function, constant arguments by
/// position)` to the copy.
type Specialized = HashMap<(FuncId, Vec<(u32, ExprId)>), FuncId>;

fn specialize_calls_in<K: Field>(
    g: &mut Graph<K>,
    roots: &[ExprId],
    made: &mut Specialized,
) -> Vec<ExprId> {
    // per instance (function, argument list): the copy it calls over the
    // arguments that are not constant, or none
    let mut instances: HashMap<(FuncId, u32, ArgList), Option<(FuncId, u32, ArgList)>> =
        HashMap::default();
    // per copy and context of the original, the context the copy keeps
    let mut kept: HashMap<(FuncId, u32), u32> = HashMap::default();
    crate::transform::rewrite(g, roots, |g, _e, node, ops| {
        let (Node::Call(o, _), Some(l)) = (node, ops.list) else {
            return g.rebuild(node, ops);
        };
        let (f, out) = g.output(o);
        let target = match instances.get(&(f, ops.ctx, l)) {
            Some(&t) => t,
            None => {
                let t = specialize_instance(g, f, ops.ctx, l, made, &mut kept);
                instances.insert((f, ops.ctx, l), t);
                t
            }
        };
        match target {
            Some((copy, ctx, rest)) => g.call_list_in(copy, out, ctx, rest),
            None => g.rebuild(node, ops),
        }
    })
}

/// The copy of `f` a call in context `ctx` over `l` runs, the context it
/// keeps (the bound parameters that are not constants) and the arguments
/// it keeps, interned; `None` when the call passes no constant to
/// specialize on.
fn specialize_instance<K: Field>(
    g: &mut Graph<K>,
    f: FuncId,
    ctx: u32,
    l: ArgList,
    made: &mut Specialized,
    kept: &mut HashMap<(FuncId, u32), u32>,
) -> Option<(FuncId, u32, ArgList)> {
    // A parameter stays an argument even when constant: its work is the
    // body's prolog, and specializing on it would split the instances of
    // one function into one copy per value.
    let full = g.full_args_in(ctx, l);
    let roles = g.func(f).param_roles();
    let consts: Vec<(u32, ExprId)> = full
        .iter()
        .enumerate()
        .filter(|&(k, &a)| {
            g.const_of(a).is_some() && !matches!(roles.get(k), Some(ParamRole::Param))
        })
        .map(|(k, &a)| (k as u32, a))
        .collect();
    if consts.is_empty() || g.func(f).is_extern() {
        return None;
    }
    let key = (f, consts);
    let copy = match made.get(&key) {
        Some(&c) => c,
        None => {
            let c = specialize_function(g, f, &key.1, made);
            made.insert(key.clone(), c);
            c
        }
    };
    // The kept parameters, in order: those `ctx` binds stay bound in the
    // copy, the others are the arguments. `key.1` is in parameter order:
    // one merge, not a search per parameter. The copy's context is the same
    // for every instance of the context, made once.
    let mut bound_at = vec![false; full.len()];
    if ctx != NO_CONTEXT {
        g.contexts[ctx as usize]
            .at
            .iter()
            .for_each(|&p| bound_at[p as usize] = true);
    }
    let mut consts_at = key.1.iter().map(|&(p, _)| p as usize).peekable();
    let (mut rest, mut rebound): (Vec<ExprId>, Vec<(u32, ExprId)>) = (Vec::new(), Vec::new());
    let known = kept.get(&(copy, ctx)).copied();
    let mut j = 0u32;
    for (k, &a) in full.iter().enumerate() {
        if consts_at.next_if_eq(&k).is_some() {
            continue;
        }
        if !bound_at[k] {
            rest.push(a);
        } else if known.is_none() {
            rebound.push((j, a));
        }
        j += 1;
    }
    let ctx = match known {
        Some(c) => c,
        None => {
            let c = g.bind(copy, &rebound).ctx;
            kept.insert((copy, ctx), c);
            c
        }
    };
    Some((copy, ctx, g.intern_args(&rest)))
}

/// The copy of `f` with the parameters `consts` names bound to their
/// constants.
fn specialize_function<K: Field>(
    g: &mut Graph<K>,
    f: FuncId,
    consts: &[(u32, ExprId)],
    made: &mut Specialized,
) -> FuncId {
    let func = g.func(f);
    let name = func.name().to_string();
    let params = func.params().to_vec();
    let roles = func.param_roles().to_vec();
    let outputs = func.outputs().to_vec();
    let out_roles = func.output_roles().to_vec();
    let bound: HashMap<SymbolId, ExprId> = consts
        .iter()
        .map(|&(k, c)| (params[k as usize], c))
        .collect();
    let exprs: Vec<ExprId> = outputs
        .iter()
        .filter_map(|o| match *o {
            Output::Expr(e) => Some(e),
            _ => None,
        })
        .collect();
    let folded = crate::transform::substitute(g, &exprs, &bound);
    let folded = specialize_calls_in(g, &folded, made);
    let kept: Vec<usize> = (0..params.len())
        .filter(|&k| !bound.contains_key(&params[k]))
        .collect();
    let copy = g.push_function(Function::new(
        &name,
        kept.iter().map(|&k| params[k]).collect(),
        None,
    ));
    for (j, &k) in kept.iter().enumerate() {
        g.set_param_role(copy, j as u32, roles[k]);
    }
    let mut next = folded.into_iter();
    for (o, role) in outputs.iter().zip(out_roles) {
        let out = match *o {
            Output::Expr(_) => {
                let e = next.next().expect("one folded output per expression");
                if g.is_zero(e) {
                    Output::Zero
                } else {
                    Output::Expr(e)
                }
            }
            _ => Output::Zero,
        };
        let role = match role {
            OutputRole::Derivative { .. } => OutputRole::Plain,
            r => r,
        };
        g.push_output(copy, out, role);
    }
    copy
}

/// One instance's substitution of its arguments into its function's body,
/// kept across the body's outputs (they share most of it): the parameters
/// bound, the nodes rewritten so far and the call lists.
struct Instance {
    bound: HashMap<SymbolId, ExprId>,
    memo: HashMap<ExprId, ExprId>,
    lists: HashMap<ArgList, ArgList>,
    /// Per context and function called in it, the context rewritten.
    contexts: HashMap<(u32, FuncId), u32>,
}

impl Instance {
    /// `root` with the instance's arguments substituted.
    fn apply<K: Field>(&mut self, g: &mut Graph<K>, root: ExprId) -> ExprId {
        let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];
        let mut ops: Vec<ExprId> = Vec::new();
        while let Some((e, expanded)) = stack.pop() {
            if self.memo.contains_key(&e) {
                continue;
            }
            let node = *g.node(e);
            if !expanded {
                stack.push((e, true));
                if let Node::Call(_, l) = node {
                    if self.lists.contains_key(&l) {
                        continue;
                    }
                }
                let pending = g.operands(e);
                stack.extend(
                    pending
                        .iter()
                        .rev()
                        .filter(|c| !self.memo.contains_key(c))
                        .map(|&c| (c, false)),
                );
                continue;
            }
            let r = match node {
                Node::Const(_) => e,
                Node::Symbol(s) => self.bound.get(&s).copied().unwrap_or(e),
                Node::Call(o, l) => {
                    let memo = &self.memo;
                    let nl = *self.lists.entry(l).or_insert_with(|| {
                        let new: Vec<ExprId> = g.args(l).iter().map(|c| memo[c]).collect();
                        g.intern_args(&new)
                    });
                    let (f0, k) = g.output(o);
                    let f = g.rebound(f0, &self.bound).unwrap_or(f0);
                    let (c, memo) = (g.context_of(o), &self.memo);
                    let ctx = match c {
                        NO_CONTEXT => NO_CONTEXT,
                        c => *self.contexts.entry((c, f)).or_insert_with(|| {
                            let exprs = g.args(g.context_list(c));
                            let new: Vec<ExprId> = exprs.iter().map(|e| memo[e]).collect();
                            g.context_over(c, f, &new)
                        }),
                    };
                    g.call_list_in(f, k, ctx, nl)
                }
                _ => {
                    ops.clear();
                    ops.extend(g.operands(e).iter().map(|c| self.memo[c]));
                    g.build(node, &ops)
                }
            };
            self.memo.insert(e, r);
        }
        self.memo[&root]
    }
}
