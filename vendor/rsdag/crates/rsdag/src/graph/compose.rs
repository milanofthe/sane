//! Composition: programs built apart, called as one. A function of another
//! graph is imported with the functions it calls; calls into it compose it
//! into whatever this graph builds, and a program over the composition is
//! compiled as one (see [`Graph::inline_composite`] and
//! [`Tape::compose`](crate::Tape::compose)).

use super::*;

impl<K: Field> Graph<K> {
    /// Function `f` of graph `other` as a function of this graph, together
    /// with the functions it calls: its outputs rebuilt here (sharing what
    /// this graph already holds), its parameters, roles and derivative
    /// outputs kept. `imported` maps the functions of `other` already
    /// imported to theirs here, so a function called from several imports
    /// comes over once; pass the same map to import more from `other`.
    ///
    /// A symbolic function this graph already holds with the same
    /// parameters, roles and outputs is taken for it rather than a copy made.
    /// An extern function shares its body. A body a consumer registered for
    /// a symbolic function stays with `other`; here the function compiles
    /// its own.
    pub fn import(
        &mut self,
        other: &Graph<K>,
        f: FuncId,
        imported: &mut HashMap<FuncId, FuncId>,
    ) -> FuncId {
        if let Some(&g) = imported.get(&f) {
            return g;
        }
        let func = other.func(f);
        let params: Vec<SymbolId> = func
            .params()
            .iter()
            .map(|&s| self.symbol_id(other.symbol_name(s)))
            .collect();
        let exprs: Vec<ExprId> = func
            .outputs()
            .iter()
            .filter_map(|o| match *o {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        let mut built = self.import_exprs(other, &exprs, imported).into_iter();
        let outputs: Vec<Output> = func
            .outputs()
            .iter()
            .map(|o| match *o {
                Output::Expr(_) => Output::Expr(built.next().expect("one per expression")),
                other => other,
            })
            .collect();
        // A symbolic function this graph already holds, the same over the
        // same parameters, is the one to call: its instances, imported along
        // different paths, then batch together.
        let same = (0..self.funcs.len()).map(|k| FuncId(k as u32)).find(|&g| {
            let h = &self.funcs[g.0 as usize];
            func.extern_body().is_none()
                && h.extern_body().is_none()
                && h.params() == &params[..]
                && h.param_roles() == func.param_roles()
                && h.outputs() == &outputs[..]
                && h.output_roles() == func.output_roles()
        });
        if let Some(g) = same {
            imported.insert(f, g);
            return g;
        }
        let new = self.push_function(Function::new(
            func.name(),
            params,
            func.extern_body().cloned(),
        ));
        for (k, &role) in func.param_roles().iter().enumerate() {
            self.set_param_role(new, k as u32, role);
        }
        for (out, &role) in outputs.into_iter().zip(func.output_roles()) {
            self.push_output(new, out, role);
        }
        imported.insert(f, new);
        new
    }

    /// Expressions of graph `other` rebuilt here: their symbols by name,
    /// their calls into the functions of `other` imported (see
    /// [`import`](Self::import)).
    pub fn import_exprs(
        &mut self,
        other: &Graph<K>,
        roots: &[ExprId],
        imported: &mut HashMap<FuncId, FuncId>,
    ) -> Vec<ExprId> {
        // the cone of `roots` in `other`, ascending: operands first
        // (a call's list once: an instance's calls share it)
        let mut seen: FxHashSet<ExprId> = FxHashSet::default();
        let mut walked: FxHashSet<ArgList> = FxHashSet::default();
        let mut stack: Vec<ExprId> = roots.to_vec();
        let mut cone: Vec<ExprId> = Vec::new();
        while let Some(e) = stack.pop() {
            if !seen.insert(e) {
                continue;
            }
            cone.push(e);
            match *other.node(e) {
                Node::Call(_, l) => {
                    if walked.insert(l) {
                        stack.extend_from_slice(other.args(l));
                    }
                }
                _ => stack.extend_from_slice(&other.operands(e)),
            }
        }
        cone.sort_unstable();
        let mut map: HashMap<ExprId, ExprId> = HashMap::default();
        map.reserve(cone.len());
        let mut lists: HashMap<ArgList, ArgList> = HashMap::default();
        let mut ops: Vec<ExprId> = Vec::new();
        for &e in &cone {
            let node = *other.node(e);
            let r = match node {
                Node::Const(c) => self.konst(other.const_val(c).clone()),
                Node::Symbol(s) => self.sym(other.symbol_name(s)),
                Node::Call(o, l) => {
                    let (f, k) = other.output(o);
                    let g = self.import(other, f, imported);
                    let nl = match lists.get(&l) {
                        Some(&nl) => nl,
                        None => {
                            let args: Vec<ExprId> = other.args(l).iter().map(|a| map[a]).collect();
                            let nl = self.intern_args(&args);
                            lists.insert(l, nl);
                            nl
                        }
                    };
                    self.call_list(g, k, nl)
                }
                _ => {
                    ops.clear();
                    ops.extend(other.operands(e).iter().map(|a| map[a]));
                    self.build(node, &ops)
                }
            };
            map.insert(e, r);
        }
        roots.iter().map(|r| map[r]).collect()
    }

    /// The symbol named `name`, made if new.
    fn symbol_id(&mut self, name: &str) -> SymbolId {
        let e = self.sym(name);
        match *self.node(e) {
            Node::Symbol(s) => s,
            _ => unreachable!("sym yields a symbol"),
        }
    }
}
