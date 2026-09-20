//! Lower an elaborated Verilog-A module's analog block onto SANE's symbolic DAG
//! via the `DeviceModel::lower_behavioral` contract.
//!
//! Contributions are assembled MNA-style: current contributions stamp into
//! per-node current accumulators (ports -> terminal currents, internal nodes ->
//! KCL residuals); voltage contributions mint a branch-current unknown plus a
//! KVL residual. `ddt` becomes a symbolic time derivative.
//!
//! Procedural control flow is flattened to dataflow: the mutable lowering
//! [`State`] (variable values + node-current accumulators) is cloned across an
//! `if`/`case`'s arms and merged with `select` on the condition; `for` loops
//! with constant bounds are unrolled; `analog function`s are inlined. Minting a
//! DAE unknown (a voltage-source branch or an `idt` state) is only allowed at
//! the top level, not inside a conditional (the rare "switch branch" idiom is
//! deferred); attempting it is a clear error.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use rsdag::{differentiate, time_derivative, CmpOp, Crossing, ExprId, Graph, Node, SymbolId};
use sane_core::constants::{MAX_UNROLL, VERILOGA_K_OVER_Q, WHILE_MAX_UNROLL};
use sane_device::{
    BehavioralFragment, FragmentEvent, FragmentLimit, LimitKind, LoweredDelay, Lowerer,
    NoiseSource, OpVar,
};

use crate::ast::{Access, BinOp, Expr, Stmt, UnOp};
use crate::elaborate::{bool_f64, const_builtin, const_eval_with, ElaboratedModule};

pub fn lower_analog(
    em: &ElaboratedModule,
    inst: &str,
    param_values: &HashMap<String, f64>,
    given: &HashSet<String>,
    mfactor: f64,
    lo: &mut Lowerer,
    terminal_v: &[ExprId],
    terminal_vdot: &[ExprId],
) -> Result<BehavioralFragment, String> {
    // Compile-time environment for structural decisions (switch branches,
    // loop bounds): module defaults overridden by the instance's bound values.
    let mut param_env: HashMap<String, f64> = em
        .params
        .iter()
        .map(|p| (p.name.clone(), p.default))
        .collect();
    for (k, v) in param_values {
        param_env.insert(k.clone(), *v);
    }
    // Node collapsing: statically-reached `V(a,b) <+ 0` shorts merge their
    // nodes before lowering (see `compute_node_collapses`).
    let node_alias = compute_node_collapses(em, &param_env, given);
    let mut l = Lower {
        em,
        inst: inst.to_string(),
        lo,
        node_alias,
        node_v: HashMap::default(),
        deriv_of: HashMap::default(),
        param_env,
        param_syms: HashMap::default(),
        given: given.clone(),
        internal_resid_nodes: Vec::new(),
        branch_resid: Vec::new(),
        cond_depth: 0,
        st: State::default(),
        noise: Vec::new(),
        events: Vec::new(),
        limits: Vec::new(),
        cur_branch: None,
        probe_of: HashMap::default(),
        probe_order: Vec::new(),
        flow_sum: HashMap::default(),
        probe_potential: HashMap::default(),
        switch_of: HashMap::default(),
        switch_order: Vec::new(),
        switch_potential: std::collections::HashSet::default(),
        mfactor,
        journal: Vec::new(),
        facts: Vec::new(),
        flag_budget: HashMap::default(),
    };
    l.setup(terminal_v, terminal_vdot);
    // Verilog-A variables default to 0 (LRM 2.4.0 §3.3.1). Seed every declared
    // variable with that default so a read before its first assignment -- or on a
    // path where its guarding branch was not taken -- yields 0 and folds
    // structural `if`/loop decisions, instead of raising "unknown identifier".
    // Real compact models (BSIM3, MVSG) read config flags that are only assigned
    // inside a guarded block or later in source order. A variable assigned before
    // use simply overwrites this seed, so already-lowering models are unaffected.
    let zero = l.ctx().zero();
    for (name, _ty) in &em.vars {
        l.st.vars.insert(name.clone(), zero);
        l.st.const_vars.insert(name.clone(), 0.0);
    }
    // `em` is a shared reference independent of `l`'s mutable borrow, so the
    // analog block can be walked in place without cloning the whole AST.
    for s in &em.analog {
        l.stmt(s)?;
        // Top-level statements run at `cond_depth == 0`; any conditional inside
        // them has already rewound its own journal slice. A top-level loop leaves
        // committed writes journaled with no rewind -- drop them (never undone).
        l.journal.clear();
    }
    Ok(l.finish())
}

/// Mutable lowering state affected by control flow (everything cloned/merged
/// across conditional arms). Structural state (minted unknowns, residuals) lives
/// on `Lower` and is append-only.
#[derive(Clone, Default)]
struct State {
    /// procedural variable / loop var -> current value expr.
    vars: HashMap<String, ExprId>,
    /// node name -> accumulated current leaving the node into the device.
    node_cur: HashMap<String, ExprId>,
    /// compile-time-constant shadow of `vars`: a variable currently holding a
    /// value foldable from parameters/literals (e.g. an integer assigned a
    /// parameter) maps to that value, enabling structural loop bounds and `if`
    /// decisions through variables. Dropped when a variable becomes runtime.
    const_vars: HashMap<String, f64>,
}

/// One undoable write to [`State`], journaled while lowering inside a conditional
/// so a branch can be rewound to its pre-branch state. `old` is the value before
/// the write (`None` = the key was absent).
enum Undo {
    Var(String, Option<ExprId>),
    NodeCur(String, Option<ExprId>),
    Const(String, Option<f64>),
}

/// What a conditional arm changed, relative to the pre-branch state: only the
/// keys it actually wrote (so a merge is O(written), not O(all variables)).
#[derive(Default)]
struct Writes {
    vars: HashMap<String, ExprId>,
    node_cur: HashMap<String, ExprId>,
    /// Final compile-time-const state per touched key (`None` = became runtime).
    const_vars: HashMap<String, Option<f64>>,
}

struct Lower<'a, 'b> {
    em: &'a ElaboratedModule,
    inst: String,
    lo: &'a mut Lowerer<'b>,
    /// Collapsed-node aliases (node -> representative), from the static
    /// `V(a,b) <+ 0` pre-scan; applied by [`Self::resolve_pair`].
    node_alias: HashMap<String, String>,
    node_v: HashMap<String, ExprId>,
    deriv_of: HashMap<SymbolId, ExprId>,
    /// Compile-time parameter environment (defaults + instance overrides) for
    /// folding structural conditions and loop bounds.
    param_env: HashMap<String, f64>,
    /// Parameter symbols this instance's expressions reference, by name.
    param_syms: HashMap<String, SymbolId>,
    /// Parameter names the deck/instance explicitly set (for `$param_given`).
    given: HashSet<String>,
    internal_resid_nodes: Vec<String>,
    branch_resid: Vec<ExprId>,
    /// >0 while lowering inside a conditional / loop / function (minting an
    /// > unknown is then forbidden).
    cond_depth: usize,
    st: State,
    /// Collected small-signal noise sources.
    noise: Vec<NoiseSource>,
    /// Switching surfaces declared by `@(cross ...)` / `@(above ...)`.
    events: Vec<FragmentEvent>,
    /// Controlling-voltage Newton limits, recorded at each `$limit(V(a,b),
    /// "pnjlim"/"fetlim", ...)` site actually lowered -- a `$limit` in a
    /// structurally dead (parameter-folded) conditional arm contributes none,
    /// so polarity-switched models declare exactly their live orientation.
    limits: Vec<FragmentLimit>,
    /// The (hi, lo) node voltage symbols of the contribution currently being
    /// lowered, so a `white_noise`/`flicker_noise` in its RHS attaches to it.
    cur_branch: Option<(Option<SymbolId>, Option<SymbolId>)>,
    /// Branches whose current is probed `I(a,b)` somewhere: promoted to an
    /// explicit current unknown. Canonical (lo<=hi) key -> current expr.
    probe_of: HashMap<(String, String), ExprId>,
    /// Probed branches in mint order (for residual ordering).
    probe_order: Vec<(String, String)>,
    /// Accumulated flow contribution per probed branch (canonical orientation).
    flow_sum: HashMap<(String, String), ExprId>,
    /// Accumulated potential contribution per probed branch (canonical
    /// orientation): a probed branch that is voltage-contributed becomes a
    /// source (its residual is the KVL, not the open-branch `i - flow` form).
    probe_potential: HashMap<(String, String), ExprId>,
    /// Switch branches: a branch that receives a potential contribution
    /// `V(a,b) <+ ..` inside a conditional (the switchable-parasitic idiom). Its
    /// flow unknown is minted/stamped unconditionally so the per-arm constraint
    /// (flow `i - flow` vs potential `(V_hi-V_lo) - val`) can be merged with
    /// `select` instead of conditionally minting an unknown. Canonical key ->
    /// current expr.
    switch_of: HashMap<(String, String), ExprId>,
    /// Switch branches in mint order (for residual ordering).
    switch_order: Vec<(String, String)>,
    /// Switch branches that have received a potential contribution (a voltage
    /// source / node collapse). Once so, a later flow contribution to the SAME
    /// pair (e.g. a compact model's thermal-noise current on a named current
    /// branch over a node pair its V-collapse already shorted) must be stamped as
    /// a parallel node current, NOT overwrite the potential constraint.
    switch_potential: std::collections::HashSet<(String, String)>,
    /// Parallel multiplicity `m`: every flow contribution is scaled by it (the
    /// device acts as `m` parallel copies) and `$mfactor` returns it.
    mfactor: f64,
    /// Undo log of `State` writes made while `cond_depth > 0`, so a conditional
    /// arm can be rewound to its pre-branch state and only the written variables
    /// merged (pruned-SSA-style). Empty at the top level.
    journal: Vec<Undo>,
    /// Comparison facts known on the current conditional path (`expr OP const`),
    /// pushed per `if` arm. Read by the while-loop bound analysis: the HiSIM2
    /// exp-reduction loop `while (t >= 60) t = t - 60;` is bounded because the
    /// enclosing guard proves `t < 500`.
    facts: Vec<(ExprId, CmpOp, f64)>,
    /// Shared unroll budget per flag for NESTED same-flag while loops (the
    /// HiSIM2 goto-emulation `while(f){while(f){while(f){...}}}`): the
    /// outermost loop's bound analysis sizes it, inner loops draw from it
    /// instead of re-analysing against a by-then non-constant counter.
    flag_budget: HashMap<String, usize>,
}

impl<'a, 'b> Lower<'a, 'b> {
    fn ctx(&mut self) -> &mut Graph {
        self.lo.ctx()
    }

    fn setup(&mut self, terminal_v: &[ExprId], terminal_vdot: &[ExprId]) {
        let zero = self.ctx().zero();
        self.node_v.insert("0".to_string(), zero);
        for (k, port) in self.em.ports.iter().enumerate() {
            self.node_v.insert(port.clone(), terminal_v[k]);
            if let Some(sym) = sym_of(self.lo.ctx(), terminal_v[k]) {
                self.deriv_of.insert(sym, terminal_vdot[k]);
            }
        }
        for node in &self.em.internal_nodes {
            if self.node_alias.contains_key(node) {
                continue; // collapsed onto its representative: no unknown
            }
            let (v, vdot) = self.lo.internal_node_of(&self.inst, node);
            self.node_v.insert(node.clone(), v);
            if let Some(sym) = sym_of(self.lo.ctx(), v) {
                self.deriv_of.insert(sym, vdot);
            }
            self.internal_resid_nodes.push(node.clone());
        }
        // Pre-scan the analog block for probed branch currents `I(a,b)`; promote
        // each to an explicit current unknown (minted after the internal nodes).
        let mut keys: Vec<(String, String)> = Vec::new();
        // Copy out the shared module reference so the analog block can be scanned
        // in place while `self` is borrowed mutably (no full AST clone).
        let em = self.em;
        for s in &em.analog {
            collect_probe_keys(self, s, &mut keys);
        }
        for key in keys {
            let (i, _) = self
                .lo
                .branch_current_of(&self.inst, &format!("flow_{}_{}", key.0, key.1));
            self.probe_of.insert(key.clone(), i);
            self.probe_order.push(key);
        }
        // Pre-scan for switch branches (a potential contribution inside a
        // conditional); mint each one's current unknown unconditionally (after
        // the probes). A branch already promoted as a probe keeps the probe path.
        let mut sw_keys: Vec<(String, String)> = Vec::new();
        for s in &em.analog {
            collect_switch_keys(self, s, false, &mut sw_keys);
        }
        for key in sw_keys {
            if self.probe_of.contains_key(&key) {
                continue;
            }
            let (i, _) = self
                .lo
                .branch_current_of(&self.inst, &format!("sw_{}_{}", key.0, key.1));
            self.switch_of.insert(key.clone(), i);
            // Seed the merged-residual pseudo-variable with `i` (the open-branch
            // `i = 0` constraint). If an arm leaves this branch uncontributed, the
            // `select` merge then falls back to `i` rather than 0 -- otherwise the
            // unconstrained current would make the DAE structurally singular.
            self.st.vars.insert(switch_resid_key(&key), i);
            self.switch_order.push(key);
        }
    }

    fn finish(mut self) -> BehavioralFragment {
        // Stamp each promoted branch current into the node balances first (it is
        // part of KCL, so it must be present before terminal currents are read).
        for key in self.probe_order.clone() {
            let i = self.probe_of[&key];
            self.add_cur(&key.0, i);
            let ni = self.ctx().neg(i);
            self.add_cur(&key.1, ni);
            // A voltage-contributed probed branch replaces its constraint with
            // the KVL (below); any flow contributions on the same pair then
            // inject in parallel through the node balances instead.
            if self.probe_potential.contains_key(&key) {
                if let Some(&f) = self.flow_sum.get(&key) {
                    self.add_cur(&key.0, f);
                    let nf = self.ctx().neg(f);
                    self.add_cur(&key.1, nf);
                }
            }
        }
        // Switch-branch currents are likewise part of KCL: stamp them too.
        for key in self.switch_order.clone() {
            let i = self.switch_of[&key];
            self.add_cur(&key.0, i);
            let ni = self.ctx().neg(i);
            self.add_cur(&key.1, ni);
        }
        let zero = self.ctx().zero();
        let terminal_currents: Vec<ExprId> = self
            .em
            .ports
            .iter()
            .map(|p| self.st.node_cur.get(p).copied().unwrap_or(zero))
            .collect();
        // Residuals in extras' mint order: internal-node KCL, then promoted
        // branch-current constraints (I - sum_flow = 0), then walk-minted (V /
        // idt) residuals.
        let mut residuals = Vec::new();
        for node in &self.internal_resid_nodes.clone() {
            residuals.push(self.st.node_cur.get(node).copied().unwrap_or(zero));
        }
        for key in &self.probe_order.clone() {
            let i = self.probe_of[key];
            let f = self.flow_sum.get(key).copied().unwrap_or(zero);
            let r = if let Some(&pot) = self.probe_potential.get(key) {
                // Voltage-contributed probed branch: the residual is the KVL
                // `v(hi) - v(lo) - pot = 0` (canonical orientation); the flow
                // sum was already stamped in parallel above.
                let vhi = self.node_v.get(&key.0).copied().unwrap_or(zero);
                let vlo = self.node_v.get(&key.1).copied().unwrap_or(zero);
                let dv = self.ctx().sub(vhi, vlo);
                self.ctx().sub(dv, pot)
            } else {
                self.ctx().sub(i, f)
            };
            residuals.push(r);
        }
        // Switch-branch constraints follow the probes in mint order.
        for key in &self.switch_order.clone() {
            // The merged per-arm constraint (a pseudo-variable), seeded to the
            // open-branch `i = 0` residual in `setup` so it is always present.
            let r = self
                .st
                .vars
                .get(&switch_resid_key(key))
                .copied()
                .unwrap_or_else(|| self.switch_of[key]);
            residuals.push(r);
        }
        residuals.extend(self.branch_resid.iter().copied());
        // Operating-point variables: the final (post-analog-block) value of each
        // `(* desc *)`-annotated module variable, exported for OP readout.
        let mut op_vars = Vec::new();
        for ov in &self.em.opvars {
            let Some(&value) = self.st.vars.get(&ov.name) else {
                continue;
            };
            op_vars.push(OpVar {
                name: format!("{}.{}", self.inst, ov.name),
                short: ov.name.clone(),
                desc: ov.desc.clone(),
                units: ov.units.clone(),
                value,
            });
        }
        let mut param_syms: Vec<(String, SymbolId)> = self.param_syms.into_iter().collect();
        param_syms.sort();
        BehavioralFragment {
            terminal_currents,
            residuals,
            noise: self.noise,
            events: self.events,
            param_syms,
            op_vars,
            limits: self.limits,
        }
    }

    // --- node helpers ------------------------------------------------------

    fn node_voltage(&mut self, name: &str) -> Result<ExprId, String> {
        self.node_v
            .get(name)
            .copied()
            .ok_or_else(|| format!("unknown node '{name}'"))
    }

    fn resolve_pair(&self, hi: &str, lo: &Option<String>) -> (String, String) {
        let (h, l) = if lo.is_none() {
            if let Some((bh, bl)) = self.em.branches.get(hi) {
                (bh.clone(), bl.clone())
            } else {
                (hi.to_string(), "0".to_string())
            }
        } else {
            (
                hi.to_string(),
                lo.clone().unwrap_or_else(|| "0".to_string()),
            )
        };
        // Collapsed nodes resolve to their representative everywhere: probes,
        // contributions and KCL accumulation all see one node.
        let ch = self.node_alias.get(&h).cloned().unwrap_or(h);
        let cl = self.node_alias.get(&l).cloned().unwrap_or(l);
        (ch, cl)
    }

    fn add_cur(&mut self, node: &str, val: ExprId) {
        if node == "0" {
            return;
        }
        let zero = self.ctx().zero();
        let prev = self.st.node_cur.get(node).copied().unwrap_or(zero);
        let sum = self.ctx().add(prev, val);
        if self.cond_depth > 0 {
            self.journal.push(Undo::NodeCur(
                node.to_string(),
                self.st.node_cur.get(node).copied(),
            ));
        }
        self.st.node_cur.insert(node.to_string(), sum);
    }

    // --- journaled State writes -------------------------------------------
    // Inside a conditional (`cond_depth > 0`) each write records its undo so the
    // arm can be rewound; at the top level it is a plain insert/remove.

    fn set_var(&mut self, key: String, val: ExprId) {
        if self.cond_depth > 0 {
            self.journal
                .push(Undo::Var(key.clone(), self.st.vars.get(&key).copied()));
        }
        self.st.vars.insert(key, val);
    }

    fn set_const(&mut self, key: String, val: f64) {
        if self.cond_depth > 0 {
            self.journal.push(Undo::Const(
                key.clone(),
                self.st.const_vars.get(&key).copied(),
            ));
        }
        self.st.const_vars.insert(key, val);
    }

    fn drop_const(&mut self, key: &str) {
        if self.cond_depth > 0 {
            self.journal.push(Undo::Const(
                key.to_string(),
                self.st.const_vars.get(key).copied(),
            ));
        }
        self.st.const_vars.remove(key);
    }

    /// Capture the keys (and their current values) written since `mark`.
    fn collect_writes(&self, mark: usize) -> Writes {
        let mut w = Writes::default();
        for u in &self.journal[mark..] {
            match u {
                Undo::Var(k, _) => {
                    w.vars.entry(k.clone()).or_insert_with(|| self.st.vars[k]);
                }
                Undo::NodeCur(k, _) => {
                    w.node_cur
                        .entry(k.clone())
                        .or_insert_with(|| self.st.node_cur[k]);
                }
                Undo::Const(k, _) => {
                    w.const_vars
                        .entry(k.clone())
                        .or_insert_with(|| self.st.const_vars.get(k).copied());
                }
            }
        }
        w
    }

    /// Undo every journaled write back to `mark`, restoring the pre-branch state.
    fn rewind(&mut self, mark: usize) {
        while self.journal.len() > mark {
            match self.journal.pop().unwrap() {
                Undo::Var(k, old) => match old {
                    Some(v) => {
                        self.st.vars.insert(k, v);
                    }
                    None => {
                        self.st.vars.remove(&k);
                    }
                },
                Undo::NodeCur(k, old) => match old {
                    Some(v) => {
                        self.st.node_cur.insert(k, v);
                    }
                    None => {
                        self.st.node_cur.remove(&k);
                    }
                },
                Undo::Const(k, old) => match old {
                    Some(v) => {
                        self.st.const_vars.insert(k, v);
                    }
                    None => {
                        self.st.const_vars.remove(&k);
                    }
                },
            }
        }
    }

    /// Scale a flow expression by the parallel multiplicity `m` (no-op for m = 1).
    fn scale_m(&mut self, e: ExprId) -> ExprId {
        let mf = self.mfactor;
        if (mf - 1.0).abs() < f64::EPSILON {
            return e;
        }
        let m = self.ctx().konst_f64(mf);
        self.ctx().mul(m, e)
    }

    // --- statements --------------------------------------------------------

    /// Join the constant (string / numeric) arguments of a diagnostic system task
    /// into a human message. The runtime format specifiers a real simulator would
    /// interpolate are not available at load time, so only the literal parts show.
    fn task_message(&self, args: &[Expr]) -> String {
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
    /// issue #41. The lowerer statically takes compile-time-constant guards and
    /// only descends runtime guards at `cond_depth > 0`, so reaching a task at
    /// `cond_depth == 0` means it is unconditional or inside a guard that folded
    /// TRUE for this instance's parameters -- exactly when the author wants it to
    /// fire.
    fn sys_task(
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
            // $error/$fatal reached unconditionally (or under a compile-time-true
            // guard) is the author rejecting this configuration: hard load failure.
            // Under a runtime guard it is an assertion SANE's single symbolic model
            // cannot enforce -- do not drop it silently, note it at load time.
            "error" | "fatal" => {
                if self.cond_depth == 0 {
                    Err(format!(
                        "${name} (module {}, line {}): {}",
                        self.em.name, span.line, msg
                    ))
                } else {
                    sane_core::log::warn_captured(&format!(
                        "${name} (module {}, line {}) is a runtime assertion SANE cannot enforce; \
                         the model is solved without it: {}",
                        self.em.name, span.line, msg
                    ));
                    Ok(())
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

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
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
                self.set_var(lhs.clone(), v);
                // Maintain the compile-time-constant shadow.
                match self.const_of_expr(rhs) {
                    Some(c) => {
                        // A non-finite compile-time-constant assignment is baked
                        // into the residual as a bias-independent NaN/Inf that
                        // poisons every Newton step. Surface it UNCONDITIONALLY as
                        // a captured warning (re-raised as a catchable
                        // SaneConvergenceWarning regardless of the log level, #41),
                        // since the first non-finite variable is the root cause.
                        // `SANE_VA_TRACE_NAN` adds extra per-assignment verbosity.
                        if !c.is_finite() {
                            sane_core::log::warn_captured(&format!(
                                "VA non-finite constant baked into '{lhs}' = {c} (module {}); it \
                                 poisons every Newton step of this model",
                                self.em.name
                            ));
                            if sane_core::config().va_trace_nan {
                                sane_core::log::warning(&format!(
                                    "VA non-finite const: {lhs} = {c}"
                                ));
                            }
                        }
                        self.set_const(lhs.clone(), c)
                    }
                    None => self.drop_const(lhs),
                }
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
            } => self.lower_for(init, cond, step, body),
            Stmt::While { cond, body } => self.lower_while(cond, body),
        }
    }

    fn lower_if(&mut self, cond: &Expr, then: &Stmt, els: Option<&Stmt>) -> Result<(), String> {
        // If the condition is compile-time constant (a parameter-gated structural
        // decision, e.g. a switch branch), take that arm statically -- no merge,
        // and a voltage contribution inside it stays top-level.
        if let Some(c) = self.const_of_expr(cond) {
            if c != 0.0 {
                return self.stmt(then);
            } else if let Some(e) = els {
                return self.stmt(e);
            }
            return Ok(());
        }
        let c = self.expr(cond)?;
        // Lower each arm against the live state, recording only what it writes,
        // then rewind to the pre-branch state. No environment clone. Each arm
        // additionally knows the comparison facts its guard implies (see
        // `facts`), for the while-loop bound analysis.
        let mark = self.facts.len();
        self.cond_facts(cond, true)?;
        let then_w = self.lower_branch(then)?;
        self.facts.truncate(mark);
        let else_w = match els {
            Some(e) => {
                self.cond_facts(cond, false)?;
                let w = self.lower_branch(e)?;
                self.facts.truncate(mark);
                w
            }
            None => Writes::default(),
        };
        self.merge_writes(c, &then_w, &else_w);
        Ok(())
    }

    /// `@(cross(expr, dir))` / `@(above(expr))`: declare a switching surface
    /// `expr = 0` the transient integrator lands on (direction `0` either
    /// way, `+1` rising, `-1` falling; `above` is rising). The body must be
    /// passive (empty, `$discontinuity`, `$bound_step`, ignored tasks): an
    /// event body executes only at the event instant, which needs a discrete
    /// state the one continuous model does not carry. Other controls
    /// (`timer`, `final_step`) are rejected for the same reason.
    fn event(
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
            Some(d) => match self.const_of_expr(d) {
                Some(c) if c > 0.0 => Crossing::Rising,
                Some(c) if c < 0.0 => Crossing::Falling,
                Some(_) => Crossing::Either,
                None => {
                    return Err(format!(
                        "'@({control} ...)' direction must be a compile-time constant (module {}, \
                         line {})",
                        self.em.name, line
                    ))
                }
            },
        };
        let ev = FragmentEvent { g, dir };
        if !self.events.contains(&ev) {
            self.events.push(ev);
        }
        Ok(())
    }

    /// Extract the comparison facts a guard implies on the taken (`positive`)
    /// or not-taken arm: conjunctions decompose on the taken side, disjunctions
    /// on the negated side (`!(a||b) = !a && !b`). Each fact is
    /// `lowered-expr OP constant`; the expression side is hash-consed, so a
    /// later occurrence of the same source expression compares equal by id.
    fn cond_facts(&mut self, cond: &Expr, positive: bool) -> Result<(), String> {
        match cond {
            Expr::Binary {
                op: BinOp::And,
                lhs,
                rhs,
                ..
            } if positive => {
                self.cond_facts(lhs, true)?;
                self.cond_facts(rhs, true)
            }
            Expr::Binary {
                op: BinOp::Or,
                lhs,
                rhs,
                ..
            } if !positive => {
                self.cond_facts(lhs, false)?;
                self.cond_facts(rhs, false)
            }
            Expr::Unary {
                op: UnOp::Not, arg, ..
            } => self.cond_facts(arg, !positive),
            Expr::Binary { op, lhs, rhs, .. } => {
                let cmp = match op {
                    BinOp::Lt => Some(CmpOp::Lt),
                    BinOp::Le => Some(CmpOp::Le),
                    BinOp::Gt => Some(CmpOp::Gt),
                    BinOp::Ge => Some(CmpOp::Ge),
                    _ => None,
                };
                let Some(mut cmp) = cmp else { return Ok(()) };
                // Normalise to `expr OP const`.
                let (e, c) = if let Some(c) = self.const_of_expr(rhs) {
                    (lhs, c)
                } else if let Some(c) = self.const_of_expr(lhs) {
                    cmp = mirror_cmp(cmp);
                    (rhs, c)
                } else {
                    return Ok(());
                };
                let cmp = if positive { cmp } else { negate_cmp(cmp) };
                let id = self.expr(e)?;
                self.facts.push((id, cmp, c));
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Lower a conditional arm at `cond_depth + 1` (so its writes are journaled
    /// and minting an unknown is forbidden), capture what it wrote, then rewind
    /// the state to before the arm.
    fn lower_branch(&mut self, s: &Stmt) -> Result<Writes, String> {
        let mark = self.journal.len();
        self.cond_depth += 1;
        let r = self.stmt(s);
        self.cond_depth -= 1;
        r?;
        let w = self.collect_writes(mark);
        self.rewind(mark);
        Ok(w)
    }

    /// Merge the two arms' writes into the (pre-branch) state with `select(c,..)`,
    /// touching only variables an arm actually wrote. A key unwritten by an arm
    /// keeps its pre-branch value; a variable runtime in either arm (or whose
    /// arms disagree on a constant) loses its constant shadow.
    fn merge_writes(&mut self, c: ExprId, then_w: &Writes, else_w: &Writes) {
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
        // constant shadow: a variable stays constant only if both arms agree on
        // the same constant value; otherwise it becomes runtime.
        for k in union(then_w, else_w, |w| w.const_vars.keys().cloned().collect()) {
            let base = self.st.const_vars.get(&k).copied();
            let t = then_w.const_vars.get(&k).copied().unwrap_or(base);
            let e = else_w.const_vars.get(&k).copied().unwrap_or(base);
            match (t, e) {
                (Some(tv), Some(ev)) if tv == ev => self.set_const(k, tv),
                _ => self.drop_const(&k),
            }
        }
    }

    fn lower_case(
        &mut self,
        sel: &Expr,
        items: &[(Vec<Expr>, Stmt)],
        default: Option<&Stmt>,
    ) -> Result<(), String> {
        let chain = case_to_if_chain(sel, items, default);
        self.stmt(&chain)
    }

    fn lower_for(
        &mut self,
        init: &Stmt,
        cond: &Expr,
        step: &Stmt,
        body: &Stmt,
    ) -> Result<(), String> {
        // Constant-bounds unrolling. The loop variable is tracked numerically
        // (for the condition) and exposed to the body as a constant.
        let (var, init_rhs) = as_assign(init)?;
        let (svar, step_rhs) = as_assign(step)?;
        let mut val = self.eval_num(&init_rhs, &var, None)?;
        let mut iters = 0;
        loop {
            if self.eval_num(cond, &var, Some(val))? == 0.0 {
                break;
            }
            let kv = self.ctx().konst_f64(val);
            self.set_var(var.clone(), kv);
            self.set_const(var.clone(), val);
            self.cond_depth += 1;
            self.stmt(body)?;
            self.cond_depth -= 1;
            val = self.eval_num(&step_rhs, &svar, Some(val))?;
            iters += 1;
            if iters > MAX_UNROLL {
                return Err("for-loop exceeded unroll cap (non-constant bounds?)".into());
            }
        }
        Ok(())
    }

    fn lower_while(&mut self, cond: &Expr, body: &Stmt) -> Result<(), String> {
        // Unroll a while-loop whose termination is bounded by an integer counter.
        // We iterate as long as the condition is NOT provably false: runtime
        // (voltage-dependent) predicates are treated as "keep iterating", so a
        // guard like `(niter<=4) && (abs(dx)>tol)` stops exactly when the counter
        // expires. This is exact for fixed-point/Newton convergence loops -- once
        // the runtime early-exit would have fired, the fixed point is reached and
        // any further unrolled iterations are no-ops producing the same value.
        //
        // Loops whose condition never becomes provably false get a bound from
        // one of two structural analyses before erroring out:
        //
        // 1. Flag loops (HiSIM2 SCE): `while (flag)` where the body clears the
        //    flag and every re-trigger `flag = 1` sits under a guard conjunct
        //    `counter < C` with `counter` counting monotonically up from a
        //    known start. The trigger budget bounds the iterations; after
        //    unrolling that many, the flag is PROVABLY zero, so it is reset to
        //    a constant (which also terminates enclosing loops on the same
        //    flag -- the nested-idential-while continue idiom).
        // 2. Descent loops (HiSIM2 exp reduction): `while (v >= C)` where the
        //    body only decrements `v` by a constant and an enclosing guard
        //    proves an upper bound on `v`'s start value (see `facts`).
        //
        // Each unrolled iteration is guarded by the (runtime) condition through
        // the normal select-merge machinery, so unrolling the BOUND is exact:
        // real executions run <= bound iterations, and the extra unrolled ones
        // reproduce the settled state.
        if matches!(self.partial_cond(cond), Some(c) if c == 0.0) {
            return Ok(());
        }
        if let Expr::Ident(flag, _) = cond {
            // Inner same-flag loop: draw from the enclosing loop's budget.
            // Every level exits either because the flag is provably clear or
            // because the shared budget (an upper bound on the TOTAL number of
            // per-level iterations across the nest) is exhausted -- in both
            // cases the flag is provably zero afterwards.
            if self.flag_budget.contains_key(flag) {
                loop {
                    if matches!(self.partial_cond(cond), Some(c) if c == 0.0) {
                        break;
                    }
                    match self.flag_budget.get_mut(flag) {
                        Some(b) if *b > 0 => *b -= 1,
                        _ => break,
                    }
                    self.cond_depth += 1;
                    let r = self.stmt(body);
                    self.cond_depth -= 1;
                    r?;
                }
                let z = self.ctx().zero();
                self.set_var(flag.clone(), z);
                self.set_const(flag.clone(), 0.0);
                return Ok(());
            }
            if let Some(bound) = self.while_flag_bound(flag, body) {
                // Each level of a same-flag nest re-enters only on a fresh
                // trigger, so per level the iterations are <= bound; the shared
                // budget bound*levels covers the whole nest.
                let levels = 1 + count_same_flag_whiles(body, flag);
                self.flag_budget.insert(flag.clone(), bound * levels);
                loop {
                    if matches!(self.partial_cond(cond), Some(c) if c == 0.0) {
                        break;
                    }
                    match self.flag_budget.get_mut(flag.as_str()) {
                        Some(b) if *b > 0 => *b -= 1,
                        _ => break,
                    }
                    self.cond_depth += 1;
                    let r = self.stmt(body);
                    self.cond_depth -= 1;
                    r?;
                }
                self.flag_budget.remove(flag.as_str());
                // Trigger budget exhausted: the flag is provably clear.
                let z = self.ctx().zero();
                self.set_var(flag.clone(), z);
                self.set_const(flag.clone(), 0.0);
                return Ok(());
            }
        }
        if let Some(bound) = self.while_descent_bound(cond, body) {
            for _ in 0..bound {
                self.cond_depth += 1;
                let r = self.stmt(body);
                self.cond_depth -= 1;
                r?;
            }
            return Ok(());
        }
        let mut iters = 0;
        while !matches!(self.partial_cond(cond), Some(c) if c == 0.0) {
            self.cond_depth += 1;
            let r = self.stmt(body);
            self.cond_depth -= 1;
            r?;
            iters += 1;
            if iters > WHILE_MAX_UNROLL {
                let what = match cond {
                    Expr::Ident(n, _) => format!("flag '{n}'"),
                    Expr::Binary { op, lhs, .. } => match &**lhs {
                        Expr::Ident(n, _) => format!("'{n}' {op:?} ..."),
                        _ => format!("{op:?} expression"),
                    },
                    _ => "complex condition".to_string(),
                };
                return Err(format!(
                    "while-loop has no static iteration bound (cannot lower; condition: {what})"
                ));
            }
        }
        Ok(())
    }

    /// Bound analysis for flag loops (pattern 1 above). Returns the iteration
    /// bound, or `None` when the pattern does not apply.
    fn while_flag_bound(&self, flag: &str, body: &Stmt) -> Option<usize> {
        /// One candidate counter comparison from a trigger site's guard stack.
        struct Cand {
            counter: String,
            limit: f64,
            inclusive: bool,
        }
        /// A `flag = <nonzero>` site: every counter-comparison candidate found
        /// in the positive guard conjuncts above it. Which candidate is a real
        /// counter is decided after the scan (increment validation).
        struct Site {
            candidates: Vec<Cand>,
        }
        struct Scan<'a, 'b, 'c> {
            l: &'a Lower<'b, 'c>,
            flag: &'a str,
            /// (counter name, limit, inclusive) per trigger site; None = a
            /// trigger without a usable counter guard (pattern fails).
            sites: Option<Vec<Site>>,
            /// increments applied to counters (name -> min positive step);
            /// a non-increment assignment poisons the counter.
            incs: HashMap<String, Option<f64>>,
        }
        impl Scan<'_, '_, '_> {
            fn fail(&mut self) {
                self.sites = None;
            }
            /// `counter < limit` / `counter <= limit` in a guard conjunct.
            fn counter_guard(&self, e: &Expr) -> Option<Cand> {
                if let Expr::Binary { op, lhs, rhs, .. } = e {
                    let (name, lim, op) = match (&**lhs, &**rhs) {
                        (Expr::Ident(n, _), r) => (n, self.l.const_of_expr(r)?, *op),
                        (l, Expr::Ident(n, _)) => {
                            let m = match op {
                                BinOp::Lt => BinOp::Gt,
                                BinOp::Le => BinOp::Ge,
                                BinOp::Gt => BinOp::Lt,
                                BinOp::Ge => BinOp::Le,
                                o => *o,
                            };
                            (n, self.l.const_of_expr(l)?, m)
                        }
                        _ => return None,
                    };
                    let inclusive = match op {
                        BinOp::Lt => false,
                        BinOp::Le => true,
                        _ => return None,
                    };
                    return Some(Cand {
                        counter: name.clone(),
                        limit: lim,
                        inclusive,
                    });
                }
                None
            }
            fn guards_site(&self, guards: &[&Expr]) -> Option<Site> {
                // Collect every counter-shaped comparison on the guard stack;
                // region guards over runtime variables (`Vgs < 0`) also match
                // here and are weeded out later by the increment validation.
                let mut candidates = Vec::new();
                for g in guards {
                    let mut stack = vec![*g];
                    while let Some(e) = stack.pop() {
                        if let Expr::Binary {
                            op: BinOp::And,
                            lhs,
                            rhs,
                            ..
                        } = e
                        {
                            stack.push(lhs);
                            stack.push(rhs);
                            continue;
                        }
                        if let Some(c) = self.counter_guard(e) {
                            candidates.push(c);
                        }
                    }
                }
                (!candidates.is_empty()).then_some(Site { candidates })
            }
            fn stmt<'e>(&mut self, s: &'e Stmt, guards: &mut Vec<&'e Expr>) {
                if self.sites.is_none() {
                    return;
                }
                match s {
                    Stmt::Block(ss) => ss.iter().for_each(|x| self.stmt(x, guards)),
                    Stmt::Assign { lhs, rhs, .. } => {
                        if lhs == self.flag {
                            match self.l.const_of_expr(rhs) {
                                Some(0.0) => {}
                                _ => match self.guards_site(guards) {
                                    Some(site) => {
                                        if let Some(v) = &mut self.sites {
                                            v.push(site);
                                        }
                                    }
                                    None => self.fail(),
                                },
                            }
                            return;
                        }
                        // counter increment tracking: `c = c + d` (d const > 0)
                        let inc = match rhs {
                            Expr::Binary {
                                op: BinOp::Add,
                                lhs: a,
                                rhs: b,
                                ..
                            } => match (&**a, &**b) {
                                (Expr::Ident(n, _), d) if n == lhs => self.l.const_of_expr(d),
                                (d, Expr::Ident(n, _)) if n == lhs => self.l.const_of_expr(d),
                                _ => None,
                            },
                            _ => None,
                        };
                        let entry = self.incs.entry(lhs.clone()).or_insert(Some(f64::INFINITY));
                        match (inc, entry.as_mut()) {
                            (Some(d), Some(cur)) if d > 0.0 => *cur = cur.min(d),
                            _ => *entry = None, // non-increment write poisons it
                        }
                    }
                    Stmt::If { cond, then, els } => {
                        guards.push(cond);
                        self.stmt(then, guards);
                        guards.pop();
                        if let Some(e) = els {
                            self.stmt(e, guards);
                        }
                    }
                    Stmt::Case {
                        sel: _,
                        items,
                        default,
                    } => {
                        for (_, b) in items {
                            self.stmt(b, guards);
                        }
                        if let Some(d) = default {
                            self.stmt(d, guards);
                        }
                    }
                    Stmt::For {
                        init, step, body, ..
                    } => {
                        self.stmt(init, guards);
                        self.stmt(step, guards);
                        self.stmt(body, guards);
                    }
                    Stmt::While { body, .. } => self.stmt(body, guards),
                    Stmt::InitialStep(b) | Stmt::Event { body: b, .. } => self.stmt(b, guards),
                    Stmt::Contribution { .. }
                    | Stmt::Indirect { .. }
                    | Stmt::Call { .. }
                    | Stmt::SysTask { .. }
                    | Stmt::Empty
                    | Stmt::IgnoredCall => {}
                }
            }
        }
        let dbg = sane_core::config().va_debug_while;
        let mut sc = Scan {
            l: self,
            flag,
            sites: Some(Vec::new()),
            incs: HashMap::default(),
        };
        let mut guards: Vec<&Expr> = Vec::new();
        sc.stmt(body, &mut guards);
        if dbg {
            eprintln!(
                "[while_flag_bound] flag={flag} sites={:?}",
                sc.sites.as_ref().map(|v| v
                    .iter()
                    .map(|st| st
                        .candidates
                        .iter()
                        .map(|c| format!(
                            "{}<{}{}",
                            c.counter,
                            if c.inclusive { "=" } else { "" },
                            c.limit
                        ))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()),
            );
        }
        let sites = sc.sites?;
        if sites.is_empty() {
            // No re-trigger at all: the body clears the flag, one pass suffices.
            return Some(1);
        }
        let mut bound = 1usize;
        for site in &sites {
            // A candidate is a real counter iff its only writes in the body are
            // constant positive increments and its pre-loop value is constant.
            let site_bound = site
                .candidates
                .iter()
                .filter_map(|c| {
                    let delta = (*sc.incs.get(&c.counter)?)?;
                    if !(delta > 0.0) || !delta.is_finite() {
                        return None;
                    }
                    let start = self.const_lookup(&c.counter)?;
                    let span = c.limit - start + if c.inclusive { delta } else { 0.0 };
                    if span <= 0.0 {
                        return Some(0); // this guard can never fire again
                    }
                    Some((span / delta).ceil() as usize)
                })
                .min();
            match site_bound {
                Some(t) => bound = bound.max(t + 1),
                None => {
                    if dbg {
                        eprintln!("[while_flag_bound] trigger site without a valid counter guard");
                    }
                    return None;
                }
            }
        }
        (bound <= 64).then_some(bound)
    }

    /// Bound analysis for descent loops (pattern 2 above):
    /// `while (v >= C)` / `while (v > C)` where the body's only writes to `v`
    /// subtract a positive constant, and a path fact (see `cond_facts`) proves
    /// an upper bound on `v`'s current (runtime) value.
    fn while_descent_bound(&mut self, cond: &Expr, body: &Stmt) -> Option<usize> {
        let (var, floor) = match cond {
            Expr::Binary {
                op: BinOp::Ge | BinOp::Gt,
                lhs,
                rhs,
                ..
            } => match &**lhs {
                Expr::Ident(n, _) => (n.clone(), self.const_of_expr(rhs)?),
                _ => return None,
            },
            _ => return None,
        };
        // Every write to `var` in the body must be `var = var - d`, d const > 0.
        fn min_decrement(l: &Lower, var: &str, s: &Stmt, dec: &mut Option<f64>) -> bool {
            match s {
                Stmt::Block(ss) => ss.iter().all(|x| min_decrement(l, var, x, dec)),
                Stmt::Assign { lhs, rhs, .. } if lhs == var => {
                    let d = match rhs {
                        Expr::Binary {
                            op: BinOp::Sub,
                            lhs: a,
                            rhs: b,
                            ..
                        } => match &**a {
                            Expr::Ident(n, _) if n == var => l.const_of_expr(b),
                            _ => None,
                        },
                        _ => None,
                    };
                    match d {
                        Some(d) if d > 0.0 => {
                            *dec = Some(dec.map_or(d, |cur: f64| cur.min(d)));
                            true
                        }
                        _ => false,
                    }
                }
                Stmt::Assign { .. } | Stmt::Contribution { .. } | Stmt::Indirect { .. } => true,
                Stmt::If { then, els, .. } => {
                    min_decrement(l, var, then, dec)
                        && els.as_deref().is_none_or(|e| min_decrement(l, var, e, dec))
                }
                Stmt::Case { items, default, .. } => {
                    items.iter().all(|(_, b)| min_decrement(l, var, b, dec))
                        && default
                            .as_deref()
                            .is_none_or(|d| min_decrement(l, var, d, dec))
                }
                Stmt::For { body, .. } | Stmt::While { body, .. } => {
                    min_decrement(l, var, body, dec)
                }
                Stmt::InitialStep(b) | Stmt::Event { body: b, .. } => min_decrement(l, var, b, dec),
                Stmt::Call { .. } | Stmt::SysTask { .. } | Stmt::Empty | Stmt::IgnoredCall => true,
            }
        }
        let mut dec = None;
        if !min_decrement(self, &var, body, &mut dec) {
            return None;
        }
        let delta = dec?;
        // Upper bound of the loop variable's CURRENT value from the path facts.
        let v0 = *self.st.vars.get(&var)?;
        let upper = self
            .facts
            .iter()
            .filter(|(e, op, _)| *e == v0 && matches!(op, CmpOp::Lt | CmpOp::Le))
            .map(|(_, _, c)| *c)
            .fold(f64::INFINITY, f64::min);
        if !upper.is_finite() {
            return None;
        }
        let span = upper - floor;
        if span <= 0.0 {
            return Some(1); // provably below the floor after at most one test
        }
        let bound = (span / delta).ceil() as usize + 1;
        (bound <= 64).then_some(bound)
    }

    /// Partial evaluation of a while-condition over currently-constant values:
    /// parameters and any loop variable that presently holds a constant node.
    /// Returns `Some(0.0)`/`Some(non-zero)` when determined, `None` when it
    /// depends on runtime (voltage) values. `&&`/`||` short-circuit so a counter
    /// bound conjoined with a runtime predicate is still decidable once the
    /// counter expires.
    fn partial_cond(&mut self, e: &Expr) -> Option<f64> {
        match e {
            Expr::Num(n) => Some(*n),
            Expr::Str(_) | Expr::Array(_) => None,
            Expr::Ident(name, _) => {
                if let Some(v) = self.const_lookup(name) {
                    return Some(v);
                }
                let id = *self.st.vars.get(name)?;
                self.lo.ctx().const_f64(id)
            }
            Expr::Unary { op, arg, .. } => {
                let v = self.partial_cond(arg)?;
                Some(match op {
                    UnOp::Neg => -v,
                    UnOp::Not => bool_f64(v == 0.0),
                })
            }
            Expr::Binary { op, lhs, rhs, .. } => match op {
                BinOp::And => {
                    let a = self.partial_cond(lhs);
                    if matches!(a, Some(x) if x == 0.0) {
                        return Some(0.0);
                    }
                    let b = self.partial_cond(rhs);
                    if matches!(b, Some(x) if x == 0.0) {
                        return Some(0.0);
                    }
                    match (a, b) {
                        (Some(_), Some(_)) => Some(1.0),
                        _ => None,
                    }
                }
                BinOp::Or => {
                    let a = self.partial_cond(lhs);
                    if matches!(a, Some(x) if x != 0.0) {
                        return Some(1.0);
                    }
                    let b = self.partial_cond(rhs);
                    if matches!(b, Some(x) if x != 0.0) {
                        return Some(1.0);
                    }
                    match (a, b) {
                        (Some(_), Some(_)) => Some(0.0),
                        _ => None,
                    }
                }
                _ => {
                    let a = self.partial_cond(lhs)?;
                    let b = self.partial_cond(rhs)?;
                    Some(match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                        BinOp::Mod => a - b * (a / b).trunc(),
                        BinOp::Pow => a.powf(b),
                        BinOp::Lt => bool_f64(a < b),
                        BinOp::Gt => bool_f64(a > b),
                        BinOp::Le => bool_f64(a <= b),
                        BinOp::Ge => bool_f64(a >= b),
                        BinOp::Eq => bool_f64(a == b),
                        BinOp::Ne => bool_f64(a != b),
                        BinOp::And | BinOp::Or => unreachable!(),
                    })
                }
            },
            Expr::Ternary {
                cond, then, els, ..
            } => {
                if self.partial_cond(cond)? != 0.0 {
                    self.partial_cond(then)
                } else {
                    self.partial_cond(els)
                }
            }
            Expr::Call { name, args, .. } => {
                let v: Vec<f64> = args
                    .iter()
                    .map(|a| self.partial_cond(a))
                    .collect::<Option<_>>()?;
                const_builtin(name, &v)
            }
            // `$param_given` is a compile-time fact of the instance binding.
            Expr::SysFn { name, args, .. } if name == "param_given" => Some(bool_f64(
                matches!(args.first(), Some(Expr::Ident(p, _)) if self.given.contains(p)),
            )),
            Expr::Access { .. } | Expr::SysFn { .. } => None,
        }
    }

    /// Evaluate a compile-time-constant expression over the parameter defaults
    /// plus an optional bound loop variable.
    fn eval_num(&self, e: &Expr, var: &str, val: Option<f64>) -> Result<f64, String> {
        const_eval_with(e, &|n| {
            if let Some(v) = val {
                if n == var {
                    return Some(v);
                }
            }
            self.const_lookup(n)
        })
        .ok_or_else(|| "for-loop bound/step is not a compile-time constant".to_string())
    }

    /// Resolve an identifier to its compile-time value: a constant-shadowed
    /// variable (latest assignment) overrides a parameter default. The
    /// pseudo-name `$given(X)` (from `$param_given`, see `const_eval_with`)
    /// resolves to whether the instance/deck explicitly set parameter `X`.
    fn const_lookup(&self, name: &str) -> Option<f64> {
        if let Some(p) = name
            .strip_prefix("$given(")
            .and_then(|s| s.strip_suffix(')'))
        {
            return Some(bool_f64(self.given.contains(p)));
        }
        self.st
            .const_vars
            .get(name)
            .copied()
            .or_else(|| self.param_env.get(name).copied())
    }

    /// Compile-time value of a string expression: a literal, or a string
    /// parameter's default. `None` for anything else.
    fn const_str<'e>(&self, e: &'e Expr) -> Option<&'e str>
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
    fn fold_str_cmp(&self, op: BinOp, lhs: &Expr, rhs: &Expr) -> Option<f64> {
        if !matches!(op, BinOp::Eq | BinOp::Ne) {
            return None;
        }
        let (a, b) = (self.const_str(lhs)?, self.const_str(rhs)?);
        let eq = a == b;
        Some(bool_f64(if matches!(op, BinOp::Eq) { eq } else { !eq }))
    }

    /// Compile-time-constant value of an expression in the current scope, if any.
    fn const_of_expr(&self, e: &Expr) -> Option<f64> {
        if let Expr::Binary { op, lhs, rhs, .. } = e {
            if let Some(v) = self.fold_str_cmp(*op, lhs, rhs) {
                return Some(v);
            }
        }
        const_eval_with(e, &|n| self.const_lookup(n))
    }

    fn contribute(
        &mut self,
        access: &Access,
        hi: &str,
        lo: &Option<String>,
        rhs: &Expr,
    ) -> Result<(), String> {
        let (hn, ln) = self.resolve_pair(hi, lo);
        // A self-branch (both endpoints the same node, typically via a
        // collapse): a zero potential contribution IS the collapse (no-op); a
        // flow contribution circulates within one node (no-op); any other
        // potential contribution is inconsistent (excluded by the pre-scan,
        // kept as a defensive error).
        if hn == ln {
            if is_potential(access) && self.const_of_expr(rhs) != Some(0.0) {
                return Err(format!(
                    "potential contribution on collapsed branch ({hn},{ln}) is not zero"
                ));
            }
            return Ok(());
        }
        // Attach any noise sources in the RHS to this branch's node voltages.
        let hsym = self
            .node_v
            .get(&hn)
            .copied()
            .and_then(|e| sym_of(self.lo.ctx(), e));
        let lsym = self
            .node_v
            .get(&ln)
            .copied()
            .and_then(|e| sym_of(self.lo.ctx(), e));
        let saved_branch = self.cur_branch.take();
        self.cur_branch = Some((hsym, lsym));
        let val = self.expr(rhs)?;
        self.cur_branch = saved_branch;

        // Switch branch: its current unknown `i` is minted/stamped
        // unconditionally (in `setup`). Each arm only records its constraint as a
        // pseudo-variable, which the conditional machinery merges with `select`;
        // `finish` emits the merged constraint as this branch's residual. This is
        // what makes a `V(..) <+ ..` inside a conditional lowerable.
        let (skey, ssign) = canon(&hn, &ln);
        if let Some(&i) = self.switch_of.get(&skey) {
            // Reserved pseudo-variable names carry the per-branch residual / flow.
            let resid_key = switch_resid_key(&skey);
            if is_potential(access) {
                // V(hi,lo) <+ val  ->  (V(hi) - V(lo)) - val = 0. This branch is a
                // voltage source / node collapse; remember it so a later flow on
                // the same pair adds in parallel instead of clobbering this.
                self.switch_potential.insert(skey.clone());
                let vhi = self.node_voltage(&hn)?;
                let vlo = self.node_voltage(&ln)?;
                let dv = self.ctx().sub(vhi, vlo);
                let r = self.ctx().sub(dv, val);
                self.set_var(resid_key, r);
            } else if self.switch_potential.contains(&skey) {
                // The pair is already a voltage source (collapsed). A further flow
                // contribution (e.g. a compact model's thermal-noise current on a
                // separate named current branch over the same nodes) is a parallel
                // current injection: stamp it into the node balances and leave the
                // potential constraint intact. At DC a `white_noise` value is 0, so
                // this is harmless; the point is to not overwrite the collapse.
                let val = self.scale_m(val);
                self.add_cur(&hn, val);
                let neg = self.ctx().neg(val);
                self.add_cur(&ln, neg);
            } else {
                // I(hi,lo) <+ flow  ->  i - sum(flow) = 0, accumulating flows in
                // the canonical orientation (and scaled by parallel multiplicity).
                let val = self.scale_m(val);
                let signed = if ssign == 1.0 {
                    val
                } else {
                    self.ctx().neg(val)
                };
                let flow_key = switch_flow_key(&skey);
                let zero = self.ctx().zero();
                let prev = self.st.vars.get(&flow_key).copied().unwrap_or(zero);
                let sum = self.ctx().add(prev, signed);
                self.set_var(flow_key, sum);
                let r = self.ctx().sub(i, sum);
                self.set_var(resid_key, r);
            }
            return Ok(());
        }

        if is_potential(access) {
            // Potential (V / Temp / ...) contribution: a source branch. Mint a
            // flow unknown, stamp it into the two node balances, and add the
            // constraint `pot(hi) - pot(lo) - rhs = 0`.
            if self.cond_depth > 0 {
                // The switch-branch path (above) handles conditional potential
                // contributions; this is only reached when the same branch is
                // *also* a current probe, where the probe owns the unknown.
                return Err(
                    "potential contribution inside a conditional on a branch whose current is \
                     also probed is not supported"
                        .into(),
                );
            }
            let (key, sign) = canon(&hn, &ln);
            if self.probe_of.contains_key(&key) {
                // The branch current is probed (`I(a,b)` used somewhere): the
                // probe already owns the flow unknown and its node stamps, so
                // this contribution only REPLACES the probe's open-branch
                // constraint with the KVL -- the voltage-source-with-current-
                // sense idiom (ideal transformer, Branin transmission line).
                // Accumulated in the canonical branch orientation.
                let signed = if sign == 1.0 {
                    val
                } else {
                    self.ctx().neg(val)
                };
                let zero = self.ctx().zero();
                let prev = self.probe_potential.get(&key).copied().unwrap_or(zero);
                let sum = self.ctx().add(prev, signed);
                self.probe_potential.insert(key, sum);
                return Ok(());
            }
            let (i, _) = self
                .lo
                .branch_current_of(&self.inst, &format!("flow_{hn}_{ln}"));
            self.add_cur(&hn, i);
            let neg = self.ctx().neg(i);
            self.add_cur(&ln, neg);
            let vhi = self.node_voltage(&hn)?;
            let vlo = self.node_voltage(&ln)?;
            let dv = self.ctx().sub(vhi, vlo);
            let r = self.ctx().sub(dv, val);
            self.branch_resid.push(r);
        } else {
            // Parallel multiplicity: a flow contribution from `m` devices in
            // parallel is `m` times the single-device flow (the LRM `$mfactor`
            // rule). Potential contributions above set a voltage and are not
            // scaled. Internal-node KCL stays balanced (every flow term scales).
            let val = self.scale_m(val);
            let (key, sign) = canon(&hn, &ln);
            if self.probe_of.contains_key(&key) {
                // Probed branch: accumulate the flow; the current unknown is
                // stamped into the nodes at finish (with its constraint).
                let signed = if sign == 1.0 {
                    val
                } else {
                    self.ctx().neg(val)
                };
                let zero = self.ctx().zero();
                let prev = self.flow_sum.get(&key).copied().unwrap_or(zero);
                let s = self.ctx().add(prev, signed);
                self.flow_sum.insert(key, s);
            } else {
                // Flow (I / Pwr / ...) contribution: stamp into the node balances.
                self.add_cur(&hn, val);
                let neg = self.ctx().neg(val);
                self.add_cur(&ln, neg);
            }
        }
        Ok(())
    }

    /// Indirect contribution `access(hi,lo) : lhs == rhs`: introduce a branch
    /// flow unknown (stamped into the node balances) and the implicit-equation
    /// residual `lhs - rhs = 0` that determines it.
    fn indirect(
        &mut self,
        hi: &str,
        lo: &Option<String>,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<(), String> {
        if self.cond_depth > 0 {
            return Err("indirect contribution inside a conditional is not supported".into());
        }
        let (hn, ln) = self.resolve_pair(hi, lo);
        // Lower the implicit equation first (may mint nested states), then mint
        // this branch's flow unknown so residual order matches mint order.
        let l = self.expr(lhs)?;
        let r = self.expr(rhs)?;
        let (i, _) = self
            .lo
            .branch_current_of(&self.inst, &format!("ind_{hn}_{ln}"));
        self.add_cur(&hn, i);
        let neg = self.ctx().neg(i);
        self.add_cur(&ln, neg);
        let resid = self.ctx().sub(l, r);
        self.branch_resid.push(resid);
        Ok(())
    }

    // --- expressions -------------------------------------------------------

    fn expr(&mut self, e: &Expr) -> Result<ExprId, String> {
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
                // A compile-time-constant condition (a parameter-gated
                // structural decision, e.g. a polarity fold `type > 0 ? ..`)
                // takes its arm statically, exactly like `lower_if` -- the
                // discarded arm never enters the graph.
                if let Some(c) = self.const_of_expr(cond) {
                    return if c != 0.0 {
                        self.expr(then)
                    } else {
                        self.expr(els)
                    };
                }
                let c = self.expr(cond)?;
                let t = self.expr(then)?;
                let e = self.expr(els)?;
                Ok(self.ctx().select(c, t, e))
            }
            Expr::Call { name, args, .. } => self.call(name, args),
            Expr::SysFn { name, args, .. } => self.sysfn(name, args),
        }
    }

    fn binary(&mut self, op: BinOp, a: ExprId, b: ExprId) -> ExprId {
        // Whether we are lowering inside a conditional arm (both arms of a
        // ternary / `if` are lowered eagerly, so a divide in the not-taken arm
        // must stay finite). Read before borrowing the context (issue #43).
        let in_cond = self.cond_depth > 0;
        let ctx = self.lo.ctx();
        match op {
            BinOp::Add => ctx.add(a, b),
            BinOp::Sub => ctx.sub(a, b),
            BinOp::Mul => ctx.mul(a, b),
            // Guard the divisor. A constant-zero divisor folds to 0 (both arms of a
            // ternary / if are lowered eagerly, so a guarded `x/y` in the not-taken
            // arm where y folds to 0 must not hit `recip(0)`, which panics; the
            // `select` discards this value anyway). A RUNTIME divisor inside a
            // conditional can hit 0 in the not-taken arm at some operating point,
            // producing an `inf` that leaks as `NaN` through `0*inf` in derivative
            // products; clamp its magnitude away from zero (sign-preserving, finite
            // derivative), so the discarded arm's residual AND Jacobian stay finite
            // while the taken arm (|y| >= floor) is exact (issue #43).
            BinOp::Div => {
                if ctx.is_zero(b) {
                    ctx.konst_f64(0.0)
                } else if in_cond {
                    let safe = guarded_denom(ctx, b);
                    ctx.div(a, safe)
                } else {
                    ctx.div(a, b)
                }
            }
            BinOp::Pow => {
                let la = ctx.ln(a);
                let bla = ctx.mul(b, la);
                ctx.exp(bla)
            }
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

    fn sysfn(&mut self, name: &str, args: &[Expr]) -> Result<ExprId, String> {
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
                let kq = self.ctx().konst_f64(VERILOGA_K_OVER_Q);
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
                        "gmin" => Some(1e-12),
                        "scale" | "shrink" | "sourceScaleFactor" | "mfactor" => Some(1.0),
                        "tnom" => Some(27.0),
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
                    if let Some(kind) = kind {
                        let (hn, ln) = self.resolve_pair(hi, lo);
                        let vhi = self.node_voltage(&hn)?;
                        let vlo = self.node_voltage(&ln)?;
                        let (hi_s, lo_s) = (sym_of(self.lo.ctx(), vhi), sym_of(self.lo.ctx(), vlo));
                        if !self
                            .limits
                            .iter()
                            .any(|l| l.hi == hi_s && l.lo == lo_s && l.kind == kind)
                        {
                            self.limits.push(FragmentLimit {
                                hi: hi_s,
                                lo: lo_s,
                                kind,
                            });
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

    fn call(&mut self, name: &str, args: &[Expr]) -> Result<ExprId, String> {
        if name == "ddt" {
            if args.is_empty() {
                return Err("ddt expects one argument".into());
            }
            let q = self.expr(&args[0])?;
            let deriv = self.deriv_of.clone();
            return Ok(time_derivative(self.ctx(), q, &deriv));
        }
        if name == "white_noise" || name == "flicker_noise" {
            // Record a noise source on the enclosing contribution's branch; the
            // large-signal value is zero.
            if args.is_empty() {
                return Err(format!("{name} expects a power-spectral-density argument"));
            }
            let psd = self.expr(&args[0])?;
            let flicker_exp = if name == "flicker_noise" {
                match args.get(1) {
                    Some(e) => self.expr(e)?,
                    None => self.ctx().one(),
                }
            } else {
                self.ctx().zero()
            };
            if let Some((hi, lo)) = self.cur_branch {
                self.noise.push(NoiseSource {
                    hi,
                    lo,
                    psd,
                    flicker_exp,
                    table: Vec::new(),
                });
            }
            return Ok(self.ctx().zero());
        }
        if name == "noise_table" || name == "noise_table_log" {
            // Tabular noise: a flat {f0, p0, f1, p1, ...} coefficient array,
            // chunked into (frequency, psd) points. Large-signal value is zero.
            if args.is_empty() {
                return Err(format!("{name} expects a coefficient array argument"));
            }
            let flat = self.const_array(&args[0])?;
            let table: Vec<(f64, f64)> = flat
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| (c[0], c[1]))
                .collect();
            if let Some((hi, lo)) = self.cur_branch {
                let psd = self.ctx().zero();
                let flicker_exp = self.ctx().zero();
                self.noise.push(NoiseSource {
                    hi,
                    lo,
                    psd,
                    flicker_exp,
                    table,
                });
            }
            return Ok(self.ctx().zero());
        }
        if name == "laplace_nd" {
            // Continuous Laplace filter H(s) = N(s)/D(s) with numerator/
            // denominator coefficient vectors -> controllable-canonical state
            // space (one differential state per denominator order).
            if self.cond_depth > 0 {
                return Err("laplace filter inside a conditional is not supported".into());
            }
            if args.len() < 3 {
                return Err("laplace_nd expects (input, num_coeffs, den_coeffs)".into());
            }
            let u = self.expr(&args[0])?;
            let num = self.const_array(&args[1])?;
            let den = self.const_array(&args[2])?;
            return self.lower_laplace_nd(u, &num, &den);
        }
        if name == "ddx" {
            // ddx(f, V(node)) = exact symbolic partial derivative of f w.r.t. the
            // node potential -- a direct hit for SANE's `differentiate`.
            if args.len() != 2 {
                return Err("ddx expects (expr, access)".into());
            }
            let f = self.expr(&args[0])?;
            if let Expr::Access { hi, lo, .. } = &args[1] {
                let (hn, _) = self.resolve_pair(hi, lo);
                let vexpr = self.node_voltage(&hn)?;
                if let Some(sym) = sym_of(self.lo.ctx(), vexpr) {
                    return Ok(differentiate(self.ctx(), f, sym));
                }
                // ground / non-symbol target: derivative is zero.
                return Ok(self.ctx().zero());
            }
            return Err("ddx second argument must be a node access like V(n)".into());
        }
        if name == "idt" || name == "idtmod" {
            // idt(u[,ic]) -> state s with ds/dt = u (residual `sdot - u = 0`); the
            // integral value is s. At DC (xdot = 0) this enforces u = 0, the
            // steady-state condition for the usual NQS-charge integrand.
            // idtmod(u, ic, modulus[, offset]) additionally wraps the output.
            if self.cond_depth > 0 {
                return Err(format!("{name} inside a conditional is not supported"));
            }
            if args.is_empty() {
                return Err(format!("{name} expects at least one argument"));
            }
            // The ic argument: DC enforces the steady-state `u = 0` (matching
            // OpenVAF/OSDI), so ic is not a constraint -- a constant ic is
            // routed as the state's DC Newton seed (`.nodeset` semantics: it
            // breaks symmetry / picks the branch on multi-stable integrands,
            // and the transient then starts from that DC point). A
            // non-constant ic cannot seed a numeric solve; surface it as a
            // captured (catchable) warning instead of silently dropping it
            // (issue #43).
            let mut dc_seed = None;
            if args.len() >= 2 {
                match self.const_of_expr(&args[1]) {
                    Some(c) if c != 0.0 => dc_seed = Some(c),
                    Some(_) => {}
                    None => sane_core::log::warn_captured(&format!(
                        "{name} initial condition is not a constant (module {}): it cannot \
                         seed the DC solve and is ignored; DC enforces the steady state u = 0",
                        self.em.name
                    )),
                }
            }
            let u = self.expr(&args[0])?;
            let sname = format!("idt{}", self.lo.extras.len());
            let (s, sdot) = self.lo.unknown_of(&self.inst, &sname, true);
            if let Some(c) = dc_seed {
                self.lo.extras.last_mut().expect("just minted").dc_seed = Some(c);
            }
            let r = self.ctx().sub(sdot, u);
            self.branch_resid.push(r);
            if name == "idtmod" && args.len() >= 3 {
                // offset + ((s - offset) mod modulus)
                let modulus = self.expr(&args[2])?;
                let offset = if args.len() >= 4 {
                    self.expr(&args[3])?
                } else {
                    self.ctx().zero()
                };
                let shifted = self.ctx().sub(s, offset);
                let m = self.binary(BinOp::Mod, shifted, modulus);
                return Ok(self.ctx().add(offset, m));
            }
            return Ok(s);
        }
        if name == "absdelay" {
            // absdelay(u, td[, maxdelay]) -> transport delay. The source is
            // pinned into an auxiliary algebraic unknown y = u so the
            // integrator's dense output covers it; the delayed value d enters
            // through a history input the transient loop fills with y(t - td)
            // (residual `d - hist = 0`). td must be a constant/parameter
            // expression -- it compiles into the delay tape over the parameter
            // vector and is checked positive at transient setup. maxdelay is
            // redundant under that restriction (the history window adapts to
            // the actual delay) and is intentionally ignored.
            if self.cond_depth > 0 {
                return Err("absdelay inside a conditional is not supported".into());
            }
            if args.len() < 2 {
                return Err("absdelay expects (expr, td)".into());
            }
            let u = self.expr(&args[0])?;
            let td = self.expr(&args[1])?;
            let k = self.lo.delays.len();
            let hist_suffix = format!("dly{k}_hist");
            let hname = format!("{}.{hist_suffix}", self.inst);
            let src_extra = self.lo.extras.len();
            let (y, _) = self
                .lo
                .unknown_of(&self.inst, &format!("dly{k}_src"), false);
            let r = self.ctx().sub(y, u);
            self.branch_resid.push(r);
            let out_extra = self.lo.extras.len();
            let (d, _) = self.lo.unknown_of(&self.inst, &format!("dly{k}"), false);
            let hexpr = self.ctx().sym(&hname);
            let hist = sym_of(self.lo.ctx(), hexpr).expect("sym() yields a Symbol node");
            let r = self.ctx().sub(d, hexpr);
            self.branch_resid.push(r);
            self.lo.delays.push(LoweredDelay {
                src_extra,
                out_extra,
                hist,
                hist_suffix,
                tau: td,
            });
            return Ok(d);
        }
        if name == "analysis" {
            // `analysis("name", ...)` reports whether the active simulator analysis
            // matches. SANE lowers ONE analysis-agnostic symbolic model that all
            // analyses share (noise is extracted separately from the same model),
            // so there is no single "active analysis": this folds to 0. The model
            // keeps its full large-signal behaviour and skips analysis-specific
            // simplifications (e.g. BSIM3's `if (analysis("noise"))` derivative
            // pruning). The string argument is intentionally not lowered.
            return Ok(self.ctx().zero());
        }
        // User analog function: inline.
        if let Some(func) = self.em.functions.get(name).cloned() {
            return self.inline_fn(&func, args);
        }
        let a: Vec<ExprId> = args
            .iter()
            .map(|e| self.expr(e))
            .collect::<Result<_, _>>()?;
        self.builtin(name, &a)
    }

    fn inline_fn(
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
        let mut cargmap: HashMap<String, f64> = HashMap::default();
        for (p, e) in func.args.iter().zip(args) {
            let v = self.expr(e)?;
            argmap.insert(p.clone(), v);
            if let Some(c) = self.const_of_expr(e) {
                cargmap.insert(p.clone(), c);
            }
        }
        let saved = std::mem::replace(
            &mut self.st,
            State {
                vars: argmap,
                node_cur: HashMap::default(),
                const_vars: cargmap,
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
        let mut writebacks: Vec<(String, ExprId, Option<f64>)> = Vec::new();
        for out in &func.outputs {
            if let Some(idx) = func.args.iter().position(|a| a == out) {
                if let (Some(Expr::Ident(caller, _)), Some(&v)) =
                    (args.get(idx), self.st.vars.get(out))
                {
                    let c = self.st.const_vars.get(out).copied();
                    writebacks.push((caller.clone(), v, c));
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
        for (caller, v, c) in writebacks {
            self.set_var(caller.clone(), v);
            match c {
                Some(c) => self.set_const(caller, c),
                None => self.drop_const(&caller),
            }
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

    fn const_array(&self, e: &Expr) -> Result<Vec<f64>, String> {
        match e {
            Expr::Array(elems) => elems
                .iter()
                .map(|x| {
                    self.const_of_expr(x).ok_or_else(|| {
                        "laplace coefficient is not a compile-time constant".to_string()
                    })
                })
                .collect(),
            _ => Err("expected a coefficient vector {..}".into()),
        }
    }

    /// Lower H(s)=N(s)/D(s) (coefficient vectors, ascending powers of s) to a
    /// controllable-canonical state space: states s_i = w^(i) where D(s)w = u,
    /// output y = sum_i num_i * w^(i). One differential unknown per denominator
    /// order; the integrator chain and the defining equation are residual rows.
    fn lower_laplace_nd(&mut self, u: ExprId, num: &[f64], den: &[f64]) -> Result<ExprId, String> {
        let k = match den.iter().rposition(|&x| x != 0.0) {
            Some(k) => k,
            None => return Err("laplace_nd denominator is empty or all zero".into()),
        };
        if num.len() > k + 1 {
            return Err(
                "laplace_nd with numerator order > denominator order is not supported".into(),
            );
        }
        if k == 0 {
            let g = num.first().copied().unwrap_or(0.0) / den[0];
            let kg = self.ctx().konst_f64(g);
            return Ok(self.ctx().mul(kg, u));
        }
        let base = self.lo.extras.len();
        let mut s = Vec::with_capacity(k);
        let mut sdot = Vec::with_capacity(k);
        for i in 0..k {
            let (si, sdi) = self
                .lo
                .unknown_of(&self.inst, &format!("lap{base}_{i}"), true);
            s.push(si);
            sdot.push(sdi);
        }
        for i in 0..k - 1 {
            let r = self.ctx().sub(sdot[i], s[i + 1]);
            self.branch_resid.push(r);
        }
        // a_k * s_{k-1}' + sum_{i<k} a_i s_i - u = 0
        let mut acc = {
            let a0 = self.ctx().konst_f64(den[0]);
            self.ctx().mul(a0, s[0])
        };
        for i in 1..k {
            let ai = self.ctx().konst_f64(den[i]);
            let t = self.ctx().mul(ai, s[i]);
            acc = self.ctx().add(acc, t);
        }
        let last = {
            let akc = self.ctx().konst_f64(den[k]);
            let lead = self.ctx().mul(akc, sdot[k - 1]);
            let s1 = self.ctx().add(lead, acc);
            self.ctx().sub(s1, u)
        };
        self.branch_resid.push(last);
        // output y = sum_{i<k} b_i s_i (+ b_k * s_{k-1}' if deg num == k)
        let mut y = self.ctx().zero();
        for i in 0..k {
            let bi = *num.get(i).unwrap_or(&0.0);
            if bi != 0.0 {
                let bic = self.ctx().konst_f64(bi);
                let t = self.ctx().mul(bic, s[i]);
                y = self.ctx().add(y, t);
            }
        }
        if num.len() == k + 1 && num[k] != 0.0 {
            let bkc = self.ctx().konst_f64(num[k]);
            let t = self.ctx().mul(bkc, sdot[k - 1]);
            y = self.ctx().add(y, t);
        }
        Ok(y)
    }

    fn builtin(&mut self, name: &str, a: &[ExprId]) -> Result<ExprId, String> {
        let ctx = self.lo.ctx();
        // Verilog-A-specific names first; everything else is the shared
        // elementary-function table in `sane_core::mathfn` (one canonical
        // construction per function, used by every expression frontend).
        Ok(match (name, a.len()) {
            ("limexp", 1) => sane_device::safe_exp(ctx, a[0]),
            // `$limit(x, ...)` is path-only (junction limiting): it shapes Newton
            // convergence, never the converged result, so identity is exact.
            ("limit", _) if !a.is_empty() => a[0],
            // Time-domain filters change the dynamic response; not supported, so
            // reject rather than silently use the unfiltered signal. (absdelay
            // is supported and handled in `call` before argument lowering.)
            ("transition" | "slew" | "last_crossing", _) => {
                return Err(format!("time-domain filter '{name}' is not supported"))
            }
            (
                "laplace_nd" | "laplace_zd" | "laplace_np" | "laplace_zp" | "zi_nd" | "zi_zd"
                | "zi_np" | "zi_zp",
                _,
            ) => return Err(format!("Laplace/Z filter '{name}' is not supported")),
            _ => match sane_core::lower_math_call(ctx, name, a) {
                Some(e) => e,
                None => return Err(format!("function '{name}/{}' not yet lowered", a.len())),
            },
        })
    }
}

/// Canonical branch key (sorted node names) and the orientation sign of the
/// written `(a, b)` relative to it (+1 if already sorted, -1 if swapped).
/// Reserved pseudo-variable name carrying a switch branch's merged residual.
/// The `\u{1}` prefix and separators cannot occur in a Verilog-A identifier, so
/// these never collide with a real variable nor with another branch's key
/// (joining node names with `_` would alias `(a_b,c)` and `(a,b_c)`).
fn switch_resid_key(key: &(String, String)) -> String {
    format!("\u{1}swr\u{1}{}\u{1}{}", key.0, key.1)
}

/// Reserved pseudo-variable name accumulating a switch branch's flow.
fn switch_flow_key(key: &(String, String)) -> String {
    format!("\u{1}swf\u{1}{}\u{1}{}", key.0, key.1)
}

/// Collect branches that receive a potential contribution `V(..) <+ ..` inside a
/// conditional (the switch-branch idiom). `cond` is true once inside any
/// `if`/`case`/`for`/`while` body. The returned canonical keys get their flow
/// unknown minted unconditionally so the per-arm constraint can be merged.
fn collect_switch_keys(l: &Lower, s: &Stmt, cond: bool, out: &mut Vec<(String, String)>) {
    match s {
        Stmt::Contribution { access, hi, lo, .. } => {
            if cond && is_potential(access) {
                let (hn, ln) = l.resolve_pair(hi, lo);
                if hn == ln {
                    return; // collapsed self-branch: nothing to switch
                }
                let (key, _) = canon(&hn, &ln);
                if !out.contains(&key) {
                    out.push(key);
                }
            }
        }
        Stmt::Block(ss) => ss.iter().for_each(|s| collect_switch_keys(l, s, cond, out)),
        Stmt::If { then, els, .. } => {
            collect_switch_keys(l, then, true, out);
            if let Some(e) = els {
                collect_switch_keys(l, e, true, out);
            }
        }
        Stmt::Case { items, default, .. } => {
            for (_, body) in items {
                collect_switch_keys(l, body, true, out);
            }
            if let Some(d) = default {
                collect_switch_keys(l, d, true, out);
            }
        }
        Stmt::For { body, .. } | Stmt::While { body, .. } => {
            collect_switch_keys(l, body, true, out)
        }
        Stmt::InitialStep(b) => collect_switch_keys(l, b, cond, out),
        Stmt::Event { body, .. } => collect_switch_keys(l, body, true, out),
        _ => {}
    }
}

/// Node collapsing (OpenVAF-style, static per instance): a potential
/// contribution `V(a,b) <+ 0` that is statically reached under this instance's
/// parameters shorts its two nodes -- the compact-model geometry-switch idiom
/// (PSP/HiSIM `SWGEO` variants) that removes unused internal nodes. Instead of
/// minting a branch-current unknown plus a `V(a)-V(b)=0` constraint row, the
/// two nodes become ONE node before lowering: smaller systems and no
/// near-singular source rows. Conservative by construction: a branch is only
/// collapsed when every potential contribution it receives is a
/// statically-reached constant zero, it is never current-probed, and at least
/// one endpoint is an internal node (ports keep their identity).
///
/// Returns the flattened alias map `node -> representative`.
fn compute_node_collapses(
    em: &ElaboratedModule,
    param_env: &HashMap<String, f64>,
    given: &HashSet<String>,
) -> HashMap<String, String> {
    // Escape hatch and differential reference (`Config::node_collapse`): with
    // collapsing off, every static zero-volt branch lowers as an explicit
    // source (flow unknown + constraint row).
    if !sane_core::config().node_collapse {
        return HashMap::default();
    }
    struct Scan<'a> {
        em: &'a ElaboratedModule,
        param_env: &'a HashMap<String, f64>,
        /// explicitly-set parameter names (for `$param_given` folding).
        given: &'a HashSet<String>,
        /// compile-time-constant variable shadow (mirrors the lowering's).
        shadow: HashMap<String, f64>,
        /// statically-reached `V(hi,lo) <+ 0` pairs, in reach order.
        zero: Vec<(String, String)>,
        /// canonical branch keys that must NOT collapse.
        blocked: HashSet<(String, String)>,
    }
    impl Scan<'_> {
        fn raw_pair(&self, hi: &str, lo: &Option<String>) -> (String, String) {
            if lo.is_none() {
                if let Some((bh, bl)) = self.em.branches.get(hi) {
                    return (bh.clone(), bl.clone());
                }
            }
            (
                hi.to_string(),
                lo.clone().unwrap_or_else(|| "0".to_string()),
            )
        }
        fn ckey(&self, hi: &str, lo: &Option<String>) -> (String, String) {
            let (h, l) = self.raw_pair(hi, lo);
            canon(&h, &l).0
        }
        /// Compile-time value over parameters + the constant shadow, with
        /// string-parameter comparison folding (mirrors `Lower::const_of_expr`).
        fn ceval(&self, e: &Expr) -> Option<f64> {
            if let Expr::Binary { op, lhs, rhs, .. } = e {
                if matches!(op, BinOp::Eq | BinOp::Ne) {
                    fn cs<'x>(em: &'x ElaboratedModule, x: &'x Expr) -> Option<&'x str> {
                        match x {
                            Expr::Str(s) => Some(s.as_str()),
                            Expr::Ident(n, _) => em.string_params.get(n).map(String::as_str),
                            _ => None,
                        }
                    }
                    if let (Some(a), Some(b)) = (cs(self.em, lhs), cs(self.em, rhs)) {
                        let eq = a == b;
                        return Some(bool_f64(if matches!(op, BinOp::Eq) { eq } else { !eq }));
                    }
                }
            }
            const_eval_with(e, &|n| {
                if let Some(p) = n.strip_prefix("$given(").and_then(|s| s.strip_suffix(')')) {
                    return Some(bool_f64(self.given.contains(p)));
                }
                self.shadow
                    .get(n)
                    .copied()
                    .or_else(|| self.param_env.get(n).copied())
            })
        }
        /// Block every current-probed branch (`I(a,b)` in an expression).
        fn block_probes(&mut self, e: &Expr) {
            match e {
                Expr::Access { access, hi, lo, .. } => {
                    if !is_potential(access) {
                        let k = self.ckey(hi, lo);
                        self.blocked.insert(k);
                    }
                }
                Expr::Unary { arg, .. } => self.block_probes(arg),
                Expr::Binary { lhs, rhs, .. } => {
                    self.block_probes(lhs);
                    self.block_probes(rhs);
                }
                Expr::Ternary {
                    cond, then, els, ..
                } => {
                    self.block_probes(cond);
                    self.block_probes(then);
                    self.block_probes(els);
                }
                Expr::Call { args, .. } | Expr::SysFn { args, .. } | Expr::Array(args) => {
                    args.iter().for_each(|a| self.block_probes(a));
                }
                Expr::Num(_) | Expr::Str(_) | Expr::Ident(_, _) => {}
            }
        }
        /// Walk statements. `decided`: this statement is statically reached
        /// under the instance parameters (no runtime guard above it).
        fn stmt(&mut self, s: &Stmt, decided: bool) {
            match s {
                Stmt::Block(ss) => ss.iter().for_each(|x| self.stmt(x, decided)),
                Stmt::Contribution {
                    access,
                    hi,
                    lo,
                    rhs,
                    ..
                } => {
                    self.block_probes(rhs);
                    if is_potential(access) {
                        let is_zero = self.ceval(rhs) == Some(0.0);
                        if decided && is_zero {
                            self.zero.push(self.raw_pair(hi, lo));
                        } else {
                            let k = self.ckey(hi, lo);
                            self.blocked.insert(k);
                        }
                    }
                }
                Stmt::Indirect {
                    access,
                    hi,
                    lo,
                    lhs,
                    rhs,
                    ..
                } => {
                    self.block_probes(lhs);
                    self.block_probes(rhs);
                    if is_potential(access) {
                        let k = self.ckey(hi, lo);
                        self.blocked.insert(k);
                    }
                }
                Stmt::Assign { lhs, rhs, .. } => {
                    self.block_probes(rhs);
                    match (decided, self.ceval(rhs)) {
                        (true, Some(c)) => {
                            self.shadow.insert(lhs.clone(), c);
                        }
                        _ => {
                            self.shadow.remove(lhs);
                        }
                    }
                }
                Stmt::If { cond, then, els } => {
                    self.block_probes(cond);
                    match self.ceval(cond) {
                        Some(c) if c != 0.0 => self.stmt(then, decided),
                        Some(_) => {
                            if let Some(e) = els {
                                self.stmt(e, decided);
                            }
                        }
                        None => {
                            self.stmt(then, false);
                            if let Some(e) = els {
                                self.stmt(e, false);
                            }
                        }
                    }
                }
                Stmt::Case {
                    sel,
                    items,
                    default,
                } => {
                    let chain = case_to_if_chain(sel, items, default.as_deref());
                    self.stmt(&chain, decided);
                }
                Stmt::For {
                    init,
                    cond,
                    step,
                    body,
                } => {
                    self.block_probes(cond);
                    self.stmt(init, false);
                    self.stmt(step, false);
                    self.stmt(body, false);
                }
                Stmt::While { cond, body } => {
                    self.block_probes(cond);
                    self.stmt(body, false);
                }
                Stmt::InitialStep(b) => self.stmt(b, decided),
                Stmt::Event { body, .. } => self.stmt(body, false),
                Stmt::Call { args, .. } | Stmt::SysTask { args, .. } => {
                    args.iter().for_each(|a| self.block_probes(a));
                }
                Stmt::Empty | Stmt::IgnoredCall => {}
            }
        }
    }

    let mut sc = Scan {
        em,
        param_env,
        given,
        shadow: HashMap::default(),
        zero: Vec::new(),
        blocked: HashSet::default(),
    };
    // Analog functions may contain contributions? (not legal VA; ignore.)
    for s in &em.analog {
        sc.stmt(s, true);
    }

    // Union the surviving zero pairs, ground/ports as preferred representatives.
    let rank = |n: &str| -> u8 {
        if n == "0" {
            0
        } else if em.ports.iter().any(|p| p == n) {
            1
        } else {
            2
        }
    };
    let mut alias: HashMap<String, String> = HashMap::default();
    fn find(alias: &HashMap<String, String>, n: &str) -> String {
        let mut cur = n.to_string();
        while let Some(next) = alias.get(&cur) {
            cur = next.clone();
        }
        cur
    }
    for (h, l) in &sc.zero {
        let key = canon(h, l).0;
        if sc.blocked.contains(&key) {
            continue;
        }
        let (a, b) = (find(&alias, h), find(&alias, l));
        if a == b {
            continue;
        }
        // Collapse the internal node into the lower-ranked (more "external")
        // representative; never merge two ports (or a port into ground).
        let (keep, gone) = if rank(&a) <= rank(&b) { (a, b) } else { (b, a) };
        if rank(&gone) < 2 {
            continue; // both endpoints are ports/ground: keep the source branch
        }
        alias.insert(gone, keep);
    }
    // Flatten chains so lookups are single-step.
    let flat: HashMap<String, String> =
        alias.keys().map(|k| (k.clone(), find(&alias, k))).collect();
    flat
}

/// Desugar a `case` statement to a nested if-chain
/// `if (sel==l0 || ...) body0 else if ... else default` -- shared by the
/// lowering and the node-collapse pre-scan so both fold identically.
fn case_to_if_chain(sel: &Expr, items: &[(Vec<Expr>, Stmt)], default: Option<&Stmt>) -> Stmt {
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

/// Number of `while (<flag>)` loops nested anywhere inside `s` (the same-flag
/// goto-emulation nest depth, for sizing the shared unroll budget).
fn count_same_flag_whiles(s: &Stmt, flag: &str) -> usize {
    match s {
        Stmt::Block(ss) => ss.iter().map(|x| count_same_flag_whiles(x, flag)).sum(),
        Stmt::If { then, els, .. } => {
            count_same_flag_whiles(then, flag)
                + els
                    .as_deref()
                    .map_or(0, |e| count_same_flag_whiles(e, flag))
        }
        Stmt::Case { items, default, .. } => {
            items
                .iter()
                .map(|(_, b)| count_same_flag_whiles(b, flag))
                .sum::<usize>()
                + default
                    .as_deref()
                    .map_or(0, |d| count_same_flag_whiles(d, flag))
        }
        Stmt::For { body, .. } => count_same_flag_whiles(body, flag),
        Stmt::While { cond, body } => {
            let own = matches!(cond, Expr::Ident(n, _) if n == flag) as usize;
            own + count_same_flag_whiles(body, flag)
        }
        Stmt::InitialStep(b) | Stmt::Event { body: b, .. } => count_same_flag_whiles(b, flag),
        _ => 0,
    }
}

/// Is an event body free of model semantics (see `Lower::event`)?
fn passive_event_body(s: &Stmt) -> bool {
    match s {
        Stmt::Empty | Stmt::IgnoredCall => true,
        Stmt::Block(ss) => ss.iter().all(passive_event_body),
        Stmt::SysTask { name, .. } => matches!(name.as_str(), "discontinuity" | "bound_step"),
        _ => false,
    }
}

/// Mirror a comparison across its operands (`c OP e` -> `e OP' c`).
fn mirror_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
        other => other,
    }
}

/// Logical negation of a comparison.
fn negate_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Ge,
        CmpOp::Le => CmpOp::Gt,
        CmpOp::Gt => CmpOp::Le,
        CmpOp::Ge => CmpOp::Lt,
        CmpOp::Eq => CmpOp::Ne,
        CmpOp::Ne => CmpOp::Eq,
    }
}

fn canon(a: &str, b: &str) -> ((String, String), f64) {
    if a <= b {
        ((a.to_string(), b.to_string()), 1.0)
    } else {
        ((b.to_string(), a.to_string()), -1.0)
    }
}

/// Collect the canonical keys of branches whose current is probed `I(a,b)` in a
/// statement (recursively), so they can be promoted to explicit unknowns.
fn collect_probe_keys(l: &Lower, s: &Stmt, out: &mut Vec<(String, String)>) {
    match s {
        Stmt::Block(ss) => ss.iter().for_each(|s| collect_probe_keys(l, s, out)),
        Stmt::Assign { rhs, .. } => collect_expr_probes(l, rhs, out),
        Stmt::Contribution { rhs, .. } => collect_expr_probes(l, rhs, out),
        Stmt::Indirect { lhs, rhs, .. } => {
            collect_expr_probes(l, lhs, out);
            collect_expr_probes(l, rhs, out);
        }
        Stmt::If { cond, then, els } => {
            collect_expr_probes(l, cond, out);
            collect_probe_keys(l, then, out);
            if let Some(e) = els {
                collect_probe_keys(l, e, out);
            }
        }
        Stmt::Case {
            sel,
            items,
            default,
        } => {
            collect_expr_probes(l, sel, out);
            for (labels, body) in items {
                labels.iter().for_each(|e| collect_expr_probes(l, e, out));
                collect_probe_keys(l, body, out);
            }
            if let Some(d) = default {
                collect_probe_keys(l, d, out);
            }
        }
        Stmt::For {
            init,
            cond,
            step,
            body,
        } => {
            collect_probe_keys(l, init, out);
            collect_expr_probes(l, cond, out);
            collect_probe_keys(l, step, out);
            collect_probe_keys(l, body, out);
        }
        Stmt::While { cond, body } => {
            collect_expr_probes(l, cond, out);
            collect_probe_keys(l, body, out);
        }
        Stmt::InitialStep(b) => collect_probe_keys(l, b, out),
        Stmt::Event { body, .. } => collect_probe_keys(l, body, out),
        Stmt::Call { args, .. } => args.iter().for_each(|a| collect_expr_probes(l, a, out)),
        Stmt::SysTask { args, .. } => args.iter().for_each(|a| collect_expr_probes(l, a, out)),
        Stmt::Empty | Stmt::IgnoredCall => {}
    }
}

fn collect_expr_probes(l: &Lower, e: &Expr, out: &mut Vec<(String, String)>) {
    match e {
        Expr::Access { access, hi, lo, .. } => {
            if !is_potential(access) {
                let (hn, ln) = l.resolve_pair(hi, lo);
                let (key, _) = canon(&hn, &ln);
                if !out.contains(&key) {
                    out.push(key);
                }
            }
        }
        Expr::Unary { arg, .. } => collect_expr_probes(l, arg, out),
        Expr::Binary { lhs, rhs, .. } => {
            collect_expr_probes(l, lhs, out);
            collect_expr_probes(l, rhs, out);
        }
        Expr::Ternary {
            cond, then, els, ..
        } => {
            collect_expr_probes(l, cond, out);
            collect_expr_probes(l, then, out);
            collect_expr_probes(l, els, out);
        }
        Expr::Call { args, .. } | Expr::SysFn { args, .. } | Expr::Array(args) => {
            args.iter().for_each(|a| collect_expr_probes(l, a, out))
        }
        Expr::Num(_) | Expr::Str(_) | Expr::Ident(_, _) => {}
    }
}

/// Whether an access function is a potential (V/Temp/...) vs a flow (I/Pwr/...).
/// This is what makes electrical and thermal (and other conservative
/// disciplines) lower uniformly: a potential contribution is a source branch, a
/// flow contribution stamps into the node balance.
fn is_potential(a: &Access) -> bool {
    match a {
        Access::V => true,
        Access::I => false,
        Access::Other(n) => matches!(n.as_str(), "Temp" | "Phi" | "Pot"),
    }
}

/// Magnitude floor for a runtime divisor lowered inside a conditional arm. Small
/// enough to leave every physical divisor untouched, large enough that `1/floor`
/// (`1e30`) stays a finite `f64` rather than an `inf` (issue #43).
const VA_DENOM_FLOOR: f64 = 1e-30;

/// Clamp a divisor's magnitude to `VA_DENOM_FLOOR` away from zero, preserving its
/// sign, so an eagerly-lowered not-taken conditional arm can never divide by a
/// runtime zero. Exact for `|b| >= floor`; within the tiny band the value
/// saturates to `±floor` (a flat clamp, so the derivative is 0 there), keeping
/// both the residual and the Jacobian finite. Every piece is a `select`/`cmp`
/// with a well-defined subgradient, so autodiff stays consistent.
fn guarded_denom(ctx: &mut Graph, b: ExprId) -> ExprId {
    let zero = ctx.zero();
    let floor = ctx.konst_f64(VA_DENOM_FLOOR);
    let nfloor = ctx.konst_f64(-VA_DENOM_FLOOR);
    // b >= 0: clamp up to +floor;  b < 0: clamp down to -floor.
    let ge_floor = ctx.cmp(CmpOp::Ge, b, floor);
    let hi = ctx.select(ge_floor, b, floor);
    let le_nfloor = ctx.cmp(CmpOp::Le, b, nfloor);
    let lo = ctx.select(le_nfloor, b, nfloor);
    let pos = ctx.cmp(CmpOp::Ge, b, zero);
    ctx.select(pos, hi, lo)
}

pub(crate) fn sym_of(ctx: &Graph, e: ExprId) -> Option<SymbolId> {
    match ctx.node(e) {
        Node::Symbol(s) => Some(*s),
        _ => None,
    }
}

/// Destructure a `var = expr` assignment statement.
fn as_assign(s: &Stmt) -> Result<(String, Expr), String> {
    match s {
        Stmt::Assign { lhs, rhs, .. } => Ok((lhs.clone(), rhs.clone())),
        _ => Err("for-loop init/step must be an assignment".into()),
    }
}
