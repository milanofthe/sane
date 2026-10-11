//! Lower an elaborated Verilog-A module's analog block onto SANE's symbolic DAG
//! via the `DeviceModel::lower_behavioral` contract.
//!
//! Contributions are assembled MNA-style: current contributions stamp into
//! per-node current accumulators (ports -> terminal currents, internal nodes ->
//! KCL residuals); voltage contributions mint a branch-current unknown plus a
//! KVL residual. Every row leaves as a current `i` and a charge `q` (`i +
//! d/dt q = 0`): `ddt(q)` puts `q` into the charge and reads zero in the
//! current.
//!
//! The lowering is exact: the graph is the one evaluator of the module, and
//! a parameter is a symbol of it, never a value. A condition the graph folds
//! to a constant (literals, loop counters, string parameters, `$param_given`)
//! takes its arm; any other condition, a parameter's included, keeps both
//! arms and merges what they write with `select` on it. Which arm a binding of
//! the parameters takes is the backend's to decide (rsdag specializes a body
//! per binding), so the lowered model is the module for every parameter
//! value. Only the structure is the instance's: its integer parameters (mode
//! selectors) and a condition on parameters with a potential contribution in
//! an arm (a branch shorted or a source, its topology) fold at the instance's
//! values, each with an assertion that a binding keeps them. Loops unroll:
//! an iteration whose condition folds runs or ends the loop, any other runs
//! gated by its condition; a loop still running after [`VA_LOOP_GATED_CAP`]
//! gated iterations ends there with an assertion, over the parameters, that
//! it does. `analog function`s are inlined.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use rsdag::{differentiate, CmpOp, Crossing, ExprId, Node, SymbolId};
use sane_core::constants::{
    K_OVER_Q, MAX_UNROLL, TEMP_NOMINAL_C, VA_LOOP_GATED_CAP, VA_SIMPARAM_GMIN,
};
use sane_core::Graph;
use sane_device::{
    Assertion, BehavioralFragment, FragmentEvent, FragmentLimit, LimitKind, LoweredDelay, Lowerer,
    NoiseSource, OpVar,
};

use crate::ast::{Access, BinOp, Expr, Stmt, UnOp};
use crate::elaborate::{bool_f64, ElaboratedModule};
use crate::ops::AnalogOp;
use sane_core::MathFn;

mod analog;
mod contribute;
mod expr;
mod stmt;
use contribute::{
    collect_probe_keys, collect_switch_keys, contributes_potential, switch_resid_key,
};

/// Lower `em`'s analog block as instance `inst` (its parameters the symbols
/// `inst.name`), `given` the parameters the deck set (for `$param_given`).
///
/// A branch shorted for good, a potential contribution of the constant zero
/// reached unconditionally (`V(a, ai) <+ 0` where the condition around it
/// folds), merges its nodes: the block is lowered again over one node where
/// the first lowering found two, so the short costs no unknown. The nodes a
/// short merges are found by the lowering itself, so whether a branch is
/// shorted is decided exactly as everything else. That first lowering, the
/// topology pass, lowers only what the branches depend on (the topology
/// slice, see [`crate::topology`]); a module without a potential
/// contribution has nothing to find and is lowered once.
pub fn lower_analog(
    em: &ElaboratedModule,
    inst: &str,
    given: &HashSet<String>,
    values: &HashMap<String, f64>,
    mfactor: f64,
    lo: &mut Lowerer,
    terminal_v: &[ExprId],
) -> Result<BehavioralFragment, String> {
    let none = HashSet::default();
    if !em.analog.iter().any(contributes_potential) {
        let l = walk(
            em,
            inst,
            given,
            values,
            mfactor,
            lo,
            terminal_v,
            Branches::default(),
            &none,
        )?;
        return l.finish();
    }
    let skip = crate::topology::outside_slice(em);
    let (shorts, found) = topology_pass(em, inst, given, values, mfactor, lo, terminal_v, skip)?;
    let alias = node_aliases(em, &shorts);
    // The branches a potential contribution reaches under a condition that
    // does not fold, over the merged nodes: the ones that switch.
    let mut switched: Vec<(String, String)> = Vec::new();
    for (a, b) in &found {
        let at = |n: &String| alias.get(n).unwrap_or(n).clone();
        let (a, b) = (at(a), at(b));
        let key = canon(&a, &b).0;
        if a != b && !switched.contains(&key) {
            switched.push(key);
        }
    }
    let exact = Branches {
        alias,
        switched: Some(switched),
    };
    walk(
        em, inst, given, values, mfactor, lo, terminal_v, exact, &none,
    )?
    .finish()
}

/// The topology pass of [`lower_analog`] over the block, the statements
/// `skip` names left out: the branches shorted for good, and those a
/// potential contribution reaches under a condition that does not fold.
/// What it mints is dropped again.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn topology_pass(
    em: &ElaboratedModule,
    inst: &str,
    given: &HashSet<String>,
    values: &HashMap<String, f64>,
    mfactor: f64,
    lo: &mut Lowerer,
    terminal_v: &[ExprId],
    skip: &HashSet<usize>,
) -> Result<(Vec<(String, String)>, Vec<(String, String)>), String> {
    let (extras, delays) = (lo.extras.len(), lo.delays.len());
    let found = {
        let l = walk(
            em,
            inst,
            given,
            values,
            mfactor,
            lo,
            terminal_v,
            Branches::default(),
            skip,
        )?;
        let shorts: Vec<(String, String)> = (l.shorts.iter())
            .filter(|k| !l.open.contains(*k))
            .cloned()
            .collect();
        (shorts, l.switched)
    };
    lo.extras.truncate(extras);
    lo.delays.truncate(delays);
    Ok(found)
}

/// What a lowering knows of the branches before it starts (see
/// [`lower_analog`]): the merged nodes, and the branches that switch
/// (`None`: every one a potential contribution reaches in a conditional).
#[derive(Default)]
struct Branches {
    alias: HashMap<String, String>,
    switched: Option<Vec<(String, String)>>,
}

/// The nodes the shorts `shorts` merge, each onto its representative: a
/// port or ground where the merged set has one (they keep their identity),
/// else its first internal node in declaration order.
fn node_aliases(em: &ElaboratedModule, shorts: &[(String, String)]) -> HashMap<String, String> {
    let mut parent: HashMap<String, String> = HashMap::default();
    fn find(parent: &HashMap<String, String>, n: &str) -> String {
        let mut n = n.to_string();
        while let Some(p) = parent.get(&n) {
            n = p.clone();
        }
        n
    }
    let rank = |n: &str| -> (usize, usize) {
        match em.ports.iter().position(|p| p == n) {
            _ if n == "0" => (0, 0),
            Some(k) => (1, k),
            None => (
                2,
                em.internal_nodes
                    .iter()
                    .position(|p| p == n)
                    .unwrap_or(usize::MAX),
            ),
        }
    };
    for (a, b) in shorts {
        let (ra, rb) = (find(&parent, a), find(&parent, b));
        if ra == rb {
            continue;
        }
        let (keep, merge) = if rank(&ra) <= rank(&rb) {
            (ra, rb)
        } else {
            (rb, ra)
        };
        if rank(&merge).0 < 2 {
            continue; // two ports, or a port and ground: a source, not a short
        }
        parent.insert(merge, keep);
    }
    let nodes: Vec<String> = parent.keys().cloned().collect();
    nodes
        .into_iter()
        .map(|n| (n.clone(), find(&parent, &n)))
        .collect()
}

/// One lowering over what `branches` knows of the branches.
#[allow(clippy::too_many_arguments)]
/// The block lowered as instance `inst` over the branches `branches`, the
/// statements `skip` names left out (see [`lower_analog`]).
fn walk<'a, 'b>(
    em: &'a ElaboratedModule,
    inst: &str,
    given: &HashSet<String>,
    values: &HashMap<String, f64>,
    mfactor: f64,
    lo: &'a mut Lowerer<'b>,
    terminal_v: &[ExprId],
    branches: Branches,
    skip: &'a HashSet<usize>,
) -> Result<Lower<'a, 'b>, String> {
    let mut l = Lower {
        em,
        inst: inst.to_string(),
        lo,
        node_alias: branches.alias,
        shorts: Vec::new(),
        open: HashSet::default(),
        switched: Vec::new(),
        node_v: HashMap::default(),
        param_syms: HashMap::default(),
        given: given.clone(),
        values: values.clone(),
        structure: Vec::new(),
        internal_resid_nodes: Vec::new(),
        branch_resid: Vec::new(),
        cond_depth: 0,
        path: Vec::new(),
        assertions: Vec::new(),
        st: State::default(),
        noise: Vec::new(),
        events: Vec::new(),
        limits: Vec::new(),
        noise_scale: None,
        probe_of: HashMap::default(),
        probe_order: Vec::new(),
        flow_sum: HashMap::default(),
        probe_potential: HashMap::default(),
        switch_of: HashMap::default(),
        switch_order: Vec::new(),
        switch_potential: std::collections::HashSet::default(),
        mfactor,
        ddts: Vec::new(),
        journal: Vec::new(),
        skip,
    };
    l.setup(terminal_v, branches.switched);
    // Verilog-A variables default to 0 (LRM 2.4.0 §3.3.1): a read before the
    // first assignment, or on a path whose assigning branch was not taken,
    // reads 0.
    let zero = l.ctx().zero();
    for (name, _ty) in &em.vars {
        l.st.vars.insert(name.clone(), zero);
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
    Ok(l)
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
}

/// One undoable write to [`State`], journaled while lowering inside a conditional
/// so a branch can be rewound to its pre-branch state. `old` is the value before
/// the write (`None` = the key was absent).
enum Undo {
    Var(String, Option<ExprId>),
    NodeCur(String, Option<ExprId>),
}

/// What a conditional arm changed, relative to the pre-branch state: only the
/// keys it actually wrote (so a merge is O(written), not O(all variables)).
#[derive(Default)]
struct Writes {
    vars: HashMap<String, ExprId>,
    node_cur: HashMap<String, ExprId>,
}

struct Lower<'a, 'b> {
    em: &'a ElaboratedModule,
    /// The statements of the block this lowering leaves out (the topology
    /// pass's, see [`lower_analog`]).
    skip: &'a HashSet<usize>,
    inst: String,
    lo: &'a mut Lowerer<'b>,
    /// Merged nodes (node -> representative), from the shorts a lowering
    /// before this one found (see [`lower_analog`]).
    node_alias: HashMap<String, String>,
    /// Branches (canonical) a potential contribution of the constant zero
    /// shorts unconditionally, and those something else keeps open (another
    /// potential contribution, a probe, a conditional one).
    shorts: Vec<(String, String)>,
    open: HashSet<(String, String)>,
    /// Branches a potential contribution reached under a condition that does
    /// not fold (in reach order).
    switched: Vec<(String, String)>,
    node_v: HashMap<String, ExprId>,
    /// Parameter symbols this instance's expressions reference, by name.
    param_syms: HashMap<String, SymbolId>,
    /// Parameter names the deck/instance explicitly set (for `$param_given`).
    given: HashSet<String>,
    /// The instance's parameter values. Its integer parameters (mode
    /// selectors) fold, and so does a condition on parameters that decides
    /// the topology: both are the structure of the instance, like `given`
    /// (see `template`).
    values: HashMap<String, f64>,
    /// What the structure folded rests on, as assertions over the
    /// parameters: a binding where one fails needs another lowering.
    structure: Vec<Assertion>,
    internal_resid_nodes: Vec<String>,
    branch_resid: Vec<ExprId>,
    /// The `ddt` placeholders and their charges (see [`Lower::ddt`]).
    ddts: Vec<(SymbolId, ExprId)>,
    /// >0 while lowering inside a conditional / loop / function (minting an
    /// > unknown is then forbidden).
    cond_depth: usize,
    /// The conditions of the conditional arms being lowered, outermost first:
    /// the path a statement is reached on is their conjunction.
    path: Vec<ExprId>,
    /// What must hold of the parameters for the lowered model to be the
    /// module's (see [`Assertion`]).
    assertions: Vec<Assertion>,
    st: State,
    /// Collected small-signal noise sources.
    noise: Vec<NoiseSource>,
    /// Switching surfaces declared by `@(cross ...)` / `@(above ...)`.
    events: Vec<FragmentEvent>,
    /// Controlling-voltage Newton limits, recorded at each `$limit(V(a,b),
    /// "pnjlim"/"fetlim", ...)` site lowered (a limit only shapes the Newton
    /// path, so one in an arm a binding does not take is harmless).
    limits: Vec<FragmentLimit>,
    /// Inside a contribution's right-hand side, the factor its value is
    /// scaled by (the multiplicity of a flow, `1` for a potential): a noise
    /// generator in it is scaled with it, its density divided so the
    /// parallel devices' noise adds as uncorrelated noise does.
    noise_scale: Option<f64>,
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
}

impl<'a, 'b> Lower<'a, 'b> {
    fn ctx(&mut self) -> &mut Graph {
        self.lo.ctx()
    }

    fn setup(&mut self, terminal_v: &[ExprId], switched: Option<Vec<(String, String)>>) {
        let zero = self.ctx().zero();
        self.node_v.insert("0".to_string(), zero);
        for (k, port) in self.em.ports.iter().enumerate() {
            self.node_v.insert(port.clone(), terminal_v[k]);
        }
        for node in &self.em.internal_nodes {
            if self.node_alias.contains_key(node) {
                continue; // merged onto its representative: no unknown
            }
            let v = self.lo.internal_node_of(&self.inst, node);
            self.node_v.insert(node.clone(), v);
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
            let i = self
                .lo
                .branch_current_of(&self.inst, &format!("flow_{}_{}", key.0, key.1));
            self.probe_of.insert(key.clone(), i);
            self.probe_order.push(key);
        }
        // Pre-scan for switch branches (a potential contribution inside a
        // conditional); mint each one's current unknown unconditionally (after
        // the probes). A branch already promoted as a probe keeps the probe path.
        let sw_keys = switched.unwrap_or_else(|| {
            let mut keys: Vec<(String, String)> = Vec::new();
            for s in &em.analog {
                collect_switch_keys(self, s, false, &mut keys);
            }
            keys
        });
        for key in sw_keys {
            if self.probe_of.contains_key(&key) {
                continue;
            }
            let i = self
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

    fn finish(mut self) -> Result<BehavioralFragment, String> {
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
        // idt) currents.
        let mut currents = Vec::new();
        for node in &self.internal_resid_nodes.clone() {
            currents.push(self.st.node_cur.get(node).copied().unwrap_or(zero));
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
            currents.push(r);
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
            currents.push(r);
        }
        currents.extend(self.branch_resid.iter().copied());
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
        // The rows as currents and charges; whatever else reads a `ddt` (an
        // op-var, a noise PSD, a switching surface) is read at rest, where
        // every time derivative is zero.
        let n_cur = terminal_currents.len();
        let rows: Vec<ExprId> = terminal_currents.into_iter().chain(currents).collect();
        let (rows, mut charges) = self.resolve_ddt(&rows)?;
        let (terminal_currents, currents) = (rows[..n_cur].to_vec(), rows[n_cur..].to_vec());
        let terminal_charges = if charges.is_empty() {
            Vec::new()
        } else {
            charges.drain(..n_cur).collect()
        };
        let mut noise = std::mem::take(&mut self.noise);
        // The rows at rest, and where each noise generator enters them: the
        // rows carry the generators where the contributions put them; the
        // device hands over its rows without, and each generator's
        // coefficient per row (see `BehavioralFragment::noise_rows`).
        let (terminal_currents, currents, noise_rows) = if noise.is_empty() {
            (terminal_currents, currents, Vec::new())
        } else {
            let rows: Vec<ExprId> = terminal_currents.iter().chain(&currents).copied().collect();
            let inputs: Vec<SymbolId> = noise.iter().map(|n| n.input).collect();
            let mut noise_rows = vec![Vec::new(); inputs.len()];
            for (r, row) in rsdag::sparse_jacobian(self.ctx(), &rows, &inputs)
                .into_iter()
                .enumerate()
            {
                for (q, coeff) in row {
                    noise_rows[q].push((r, coeff));
                }
            }
            let zero = self.ctx().zero();
            let quiet: HashMap<SymbolId, ExprId> = inputs.iter().map(|&s| (s, zero)).collect();
            let mut rows = rsdag::substitute(self.ctx(), &rows, &quiet);
            let currents = rows.split_off(n_cur);
            (rows, currents, noise_rows)
        };
        let mut events = std::mem::take(&mut self.events);
        let observed: Vec<ExprId> = op_vars
            .iter()
            .map(|v| v.value)
            .chain(noise.iter().flat_map(|n| n.exprs()))
            .chain(events.iter().map(|e| e.g))
            .collect();
        let zero = self.ctx().zero();
        let rest: HashMap<SymbolId, ExprId> = self.ddts.iter().map(|&(m, _)| (m, zero)).collect();
        let mut observed = rsdag::substitute(self.ctx(), &observed, &rest).into_iter();
        for v in &mut op_vars {
            v.value = observed.next().expect("one per op-var");
        }
        for n in &mut noise {
            *n = n.with_exprs(n.input, &mut observed);
        }
        for e in &mut events {
            e.g = observed.next().expect("one per event");
        }
        let mut param_syms: Vec<(String, SymbolId)> = self.param_syms.into_iter().collect();
        param_syms.sort();
        let mut collapsed: Vec<(String, ExprId)> = (self.node_alias.iter())
            .map(|(node, rep)| (node.clone(), self.node_v[rep]))
            .collect();
        collapsed.sort();
        Ok(BehavioralFragment {
            terminal_currents,
            currents,
            terminal_charges,
            charges,
            noise,
            noise_rows,
            events,
            param_syms,
            op_vars,
            limits: self.limits,
            assertions: self.assertions,
            structural: self.structure,
            collapsed,
        })
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
        // Merged nodes resolve to their representative everywhere: probes,
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
            }
        }
    }

    // --- statements --------------------------------------------------------

    // --- expressions -------------------------------------------------------
}

fn canon(a: &str, b: &str) -> ((String, String), f64) {
    if a <= b {
        ((a.to_string(), b.to_string()), 1.0)
    } else {
        ((b.to_string(), a.to_string()), -1.0)
    }
}

pub(crate) fn is_potential(a: &Access) -> bool {
    match a {
        Access::V => true,
        Access::I => false,
        Access::Other(n) => matches!(n.as_str(), "Temp" | "Phi" | "Pot"),
    }
}

pub(crate) fn sym_of(ctx: &Graph, e: ExprId) -> Option<SymbolId> {
    match ctx.node(e) {
        Node::Symbol(s) => Some(*s),
        _ => None,
    }
}

/// The symbol of noise generator `k` of the device instance `inst`.
pub(crate) fn noise_symbol_name(inst: &str, k: usize) -> String {
    format!("{inst}#noise{k}")
}
