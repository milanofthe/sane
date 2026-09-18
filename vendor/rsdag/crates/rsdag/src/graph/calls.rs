//! Functions, calls and the system layer over them: defining a function,
//! calling it, roles, role-selected Jacobians, feedthrough, inlining.
//!
//! A child module of `graph` so it reaches the arena's private fields; a
//! second `impl Graph` block rather than a second type, because a call is a
//! node like any other and the functions live in the same arena.

use super::*;

impl<K: Field> Graph<K> {
    // --- functions and calls ---------------------------------------------

    /// Define a symbolic function: `outputs` are expressions over the formal
    /// `params` (free symbols of the outputs not listed in `params` are shared
    /// globals every call sees unchanged).
    pub fn define_func(
        &mut self,
        name: &str,
        params: Vec<SymbolId>,
        outputs: Vec<ExprId>,
    ) -> FuncId {
        let id = FuncId(self.funcs.len() as u32);
        let n_par = params.len();
        let n_out = outputs.len();
        self.funcs.push(Function {
            name: name.to_string(),
            params,
            param_roles: vec![ParamRole::Free; n_par],
            outputs: outputs.into_iter().map(Output::Expr).collect(),
            output_roles: vec![OutputRole::Plain; n_out],
            body: FunctionBody::Symbolic,
            compiled: Vec::new(),
            deriv_index: HashMap::default(),
        });
        id
    }

    /// Close an open graph over `outputs` into a function: every free symbol
    /// the outputs depend on becomes a parameter, in symbol order. The
    /// `Scope` idiom: build with named symbols, then close.
    pub fn close(&mut self, name: &str, outputs: Vec<ExprId>) -> FuncId {
        let params: Vec<SymbolId> = self.free_symbols_in(&outputs).into_iter().collect();
        self.define_func(name, params, outputs)
    }

    /// Set the role of parameter `param` of function `f`.
    pub fn set_param_role(&mut self, f: FuncId, param: u32, role: ParamRole) {
        self.funcs[f.0 as usize].param_roles[param as usize] = role;
    }

    /// Set the role of output `out` of function `f`.
    pub fn set_output_role(&mut self, f: FuncId, out: u32, role: OutputRole) {
        self.funcs[f.0 as usize].output_roles[out as usize] = role;
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
        let mut map: HashMap<ExprId, ExprId> = HashMap::default();
        roots
            .iter()
            .map(|&r| self.inline_rec(r, &mut map))
            .collect()
    }

    fn inline_rec(&mut self, e: ExprId, map: &mut HashMap<ExprId, ExprId>) -> ExprId {
        if let Some(&done) = map.get(&e) {
            return done;
        }
        let node = *self.node(e);
        // Operands first, so the rebuild below reads them from `map`.
        let operands: Vec<ExprId> = self.operands(e).to_vec();
        for a in operands {
            self.inline_rec(a, map);
        }
        let out = match node {
            Node::Call(o, l) => {
                let args: Vec<ExprId> = self.args(l).to_vec().iter().map(|a| map[a]).collect();
                let (f, k) = self.output(o);
                match self.funcs[f.0 as usize].outputs[k as usize] {
                    // An extern body stays a call, over inlined arguments.
                    Output::Slot(_) => self.call(f, k, &args),
                    Output::Zero => self.zero,
                    Output::Expr(_) => {
                        let body = self.inline_outputs(f, &[k], &args)[0];
                        // The body may itself contain calls.
                        self.inline_rec(body, map)
                    }
                }
            }
            _ => {
                let konst = match node {
                    Node::Const(c) => self.consts[c.0 as usize].clone(),
                    _ => K::zero(),
                };
                let args: Vec<ExprId> = match node {
                    Node::Reduce(_, l) | Node::Dot(l) | Node::Solve(l, _) => {
                        self.args(l).to_vec().iter().map(|a| map[a]).collect()
                    }
                    _ => Vec::new(),
                };
                // The call table is copied out first: a closure handed to
                // `rebuild_node` cannot hold a borrow of the graph it builds
                // in.
                let call_table = self.outputs.clone();
                self.rebuild_node(
                    &node,
                    |_| konst.clone(),
                    |s| s,
                    |x| map[&x],
                    |_| args.clone(),
                    |o| call_table[o.0 as usize],
                )
            }
        };
        map.insert(e, out);
        out
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
        let func = &self.funcs[f.0 as usize];
        let index: HashMap<SymbolId, usize> = func
            .params
            .iter()
            .enumerate()
            .map(|(k, &s)| (s, k))
            .collect();
        func.outputs
            .iter()
            .map(|out| {
                let mut row = vec![false; func.params.len()];
                match *out {
                    Output::Zero => {}
                    Output::Slot(_) => row.iter_mut().for_each(|r| *r = true),
                    Output::Expr(e) => {
                        for s in self.free_symbols(e) {
                            if let Some(&k) = index.get(&s) {
                                row[k] = true;
                            }
                        }
                    }
                }
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
            for &p in &pars {
                let k = self.derivative_output(f, o, p);
                if !matches!(self.funcs[f.0 as usize].outputs[k as usize], Output::Zero) {
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
        let id = FuncId(self.funcs.len() as u32);
        let n_out = outputs.len();
        self.funcs.push(Function {
            name: name.to_string(),
            param_roles: vec![ParamRole::Free; params.len()],
            params,
            outputs,
            output_roles: vec![OutputRole::Plain; n_out],
            body: FunctionBody::Extern(body),
            compiled: Vec::new(),
            deriv_index: HashMap::default(),
        });
        id
    }

    /// Declare `d outputs[out] / d params[param]` of an extern function as
    /// `deriv` (a slot of its body, or zero). Undeclared derivatives are zero.
    pub fn declare_derivative(&mut self, f: FuncId, out: u32, param: u32, deriv: Output) -> u32 {
        let func = &mut self.funcs[f.0 as usize];
        let k = func.outputs.len() as u32;
        func.outputs.push(deriv);
        func.output_roles.push(OutputRole::Derivative {
            of: out,
            wrt: param,
        });
        func.deriv_index.insert((out, param), k);
        k
    }

    pub fn func(&self, f: FuncId) -> &Function {
        &self.funcs[f.0 as usize]
    }

    /// Number of symbols.
    pub fn n_symbols(&self) -> usize {
        self.symbol_names.len()
    }

    /// Mutable access to a function (roles, memoised outputs).
    pub fn func_mut(&mut self, f: FuncId) -> &mut Function {
        &mut self.funcs[f.0 as usize]
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
        let bodies = &mut self.funcs[f.0 as usize].compiled;
        let ptr = Arc::as_ptr(&body.bundle) as *const () as usize;
        if !bodies
            .iter()
            .any(|b| Arc::as_ptr(&b.bundle) as *const () as usize == ptr)
        {
            bodies.push(body);
        }
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
        let id = FuncId(self.funcs.len() as u32);
        let n_out = outputs.len();
        self.funcs.push(Function {
            name: name.to_string(),
            param_roles: vec![ParamRole::Free; params.len()],
            params,
            outputs,
            output_roles: vec![OutputRole::Plain; n_out],
            body: FunctionBody::Extern(body),
            compiled: Vec::new(),
            deriv_index: HashMap::default(),
        });
        id
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
        if let Some(&o) = self.output_dedup.get(&(f, out)) {
            return o;
        }
        let o = OutputId(self.outputs.len() as u32);
        self.outputs.push((f, out));
        self.output_dedup.insert((f, out), o);
        o
    }

    /// The expression of a symbolic output, `None` for a slot or zero output.
    pub fn output_expr(&self, f: FuncId, out: u32) -> Option<ExprId> {
        match self.funcs[f.0 as usize].outputs[out as usize] {
            Output::Expr(e) => Some(e),
            _ => None,
        }
    }

    /// Output `out` of `f` applied to `args` (one argument per parameter). A
    /// zero output folds to the constant zero.
    pub fn call(&mut self, f: FuncId, out: u32, args: &[ExprId]) -> ExprId {
        debug_assert_eq!(
            args.len(),
            self.funcs[f.0 as usize].params.len(),
            "call arity"
        );
        if matches!(self.funcs[f.0 as usize].outputs[out as usize], Output::Zero) {
            return self.zero;
        }
        let o = self.output_id(f, out);
        let l = self.intern_args(args);
        self.intern(Node::Call(o, l))
    }

    /// [`call`](Self::call) by output id.
    pub fn call_output(&mut self, o: OutputId, args: &[ExprId]) -> ExprId {
        let (f, out) = self.output(o);
        self.call(f, out, args)
    }

    /// The index of the derivative output `d outputs[out] / d params[param]`,
    /// differentiating the body on first demand (symbolic functions) or
    /// looking up the declared slot (extern functions; zero if undeclared).
    pub fn derivative_output(&mut self, f: FuncId, out: u32, param: u32) -> u32 {
        if let Some(&k) = self.funcs[f.0 as usize].deriv_index.get(&(out, param)) {
            return k;
        }
        let d = match self.funcs[f.0 as usize].outputs[out as usize] {
            Output::Expr(e) => {
                let wrt = self.funcs[f.0 as usize].params[param as usize];
                let de = crate::autodiff::differentiate(self, e, wrt);
                if self.is_zero(de) {
                    Output::Zero
                } else {
                    Output::Expr(de)
                }
            }
            Output::Slot(_) | Output::Zero => Output::Zero,
        };
        let func = &mut self.funcs[f.0 as usize];
        let k = func.outputs.len() as u32;
        func.outputs.push(d);
        func.output_roles.push(OutputRole::Derivative {
            of: out,
            wrt: param,
        });
        func.deriv_index.insert((out, param), k);
        k
    }

    /// Inline a call: the output expression with the parameters replaced by
    /// `args`. `None` for an extern (slot) output, which has no body to inline.
    pub fn inline_call(&mut self, f: FuncId, out: u32, args: &[ExprId]) -> Option<ExprId> {
        match self.funcs[f.0 as usize].outputs[out as usize] {
            Output::Expr(e) => {
                let params = self.funcs[f.0 as usize].params.clone();
                let map: HashMap<SymbolId, ExprId> =
                    params.iter().copied().zip(args.iter().copied()).collect();
                Some(crate::transform::substitute(self, &[e], &map)[0])
            }
            Output::Zero => Some(self.zero),
            Output::Slot(_) => None,
        }
    }

    /// Inline several outputs of a symbolic function at once, with one shared
    /// substitution pass (the outputs of a device template share its core, so
    /// per-output substitution would rebuild that core per output).
    pub fn inline_outputs(&mut self, f: FuncId, outs: &[u32], args: &[ExprId]) -> Vec<ExprId> {
        let params = self.funcs[f.0 as usize].params.clone();
        let map: HashMap<SymbolId, ExprId> =
            params.iter().copied().zip(args.iter().copied()).collect();
        let exprs: Vec<ExprId> = outs
            .iter()
            .map(|&o| match self.funcs[f.0 as usize].outputs[o as usize] {
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
        let mut set = std::collections::BTreeSet::new();
        let mut visited = FxHashSet::default();
        let mut stack: Vec<ExprId> = exprs.to_vec();
        while let Some(e) = stack.pop() {
            if !visited.insert(e) {
                continue;
            }
            if let Node::Call(o, _) = *self.node(e) {
                set.insert(o);
            }
            stack.extend_from_slice(&self.operands(e));
        }
        set
    }

    /// The set of free symbols reachable from `expr`.
    ///
    /// Memoised over shared subexpressions (a `visited` set): in a hash-consed
    /// DAG a node may be reachable by exponentially many paths, so without this
    /// the traversal is super-linear in the node count. With it, each node is
    /// visited once -- O(nodes reachable from `expr`).
    pub fn free_symbols(&self, expr: ExprId) -> std::collections::BTreeSet<SymbolId> {
        self.free_symbols_in(&[expr])
    }

    /// Union of the free symbols across many expressions, sharing one `visited`
    /// set so a subexpression hash-consed into several of them is traversed once
    /// (a single pass over the forest, not one per expression).
    pub fn free_symbols_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        let mut set = std::collections::BTreeSet::new();
        let mut visited = FxHashSet::default();
        let mut stack: Vec<ExprId> = exprs.to_vec();
        while let Some(e) = stack.pop() {
            if !visited.insert(e) {
                continue;
            }
            match *self.node(e) {
                Node::Const(_) => {}
                Node::Symbol(s) => {
                    set.insert(s);
                }
                _ => stack.extend_from_slice(&self.operands(e)),
            }
        }
        set
    }
}
