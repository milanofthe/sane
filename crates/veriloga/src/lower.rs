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
use sane_core::constants::{MAX_UNROLL, VA_LOOP_GATED_CAP, VERILOGA_K_OVER_Q};
use sane_core::Graph;
use sane_device::{
    Assertion, BehavioralFragment, FragmentEvent, FragmentLimit, LimitKind, LoweredDelay, Lowerer,
    NoiseSource, OpVar,
};

use crate::ast::{Access, BinOp, Expr, Stmt, UnOp};
use crate::elaborate::{bool_f64, ElaboratedModule};

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

    /// `ddt(q)`: a placeholder for the time derivative of the charge `q`,
    /// resolved row by row once the rows are whole (see
    /// [`Self::resolve_ddt`]).
    fn ddt(&mut self, q: ExprId) -> ExprId {
        let name = format!("{}ddt#{}", self.inst, self.ddts.len());
        let m = self.ctx().sym(&name);
        let s = sym_of(self.lo.ctx(), m).expect("sym() yields a Symbol");
        self.ddts.push((s, q));
        m
    }

    /// `rows` as `i + d/dt q`: per row its current `i`, the placeholders
    /// read as zero, and its charge `q = sum_m (d row / d m) q_m` (empty
    /// without any `ddt`). A row must be affine in the placeholders; `ddt`
    /// inside a nonlinear function has no charge.
    fn resolve_ddt(&mut self, rows: &[ExprId]) -> Result<(Vec<ExprId>, Vec<ExprId>), String> {
        if self.ddts.is_empty() {
            return Ok((rows.to_vec(), Vec::new()));
        }
        let ddts = self.ddts.clone();
        let ctx = self.lo.ctx();
        let zero = ctx.zero();
        let rest: HashMap<SymbolId, ExprId> = ddts.iter().map(|&(m, _)| (m, zero)).collect();
        let markers: HashSet<SymbolId> = ddts.iter().map(|&(m, _)| m).collect();
        let mut charges = Vec::with_capacity(rows.len());
        for &r in rows {
            let free = ctx.free_symbols(r);
            let mut terms = Vec::new();
            for &(m, q) in ddts.iter().filter(|(m, _)| free.contains(m)) {
                let c = differentiate(ctx, r, m);
                if ctx.free_symbols(c).iter().any(|s| markers.contains(s)) {
                    return Err(format!(
                        "module {}: ddt in a nonlinear expression has no charge",
                        self.em.name
                    ));
                }
                terms.push(ctx.mul(c, q));
            }
            charges.push(ctx.reduce(rsdag::ReduceOp::Sum, terms));
        }
        Ok((rsdag::substitute(ctx, rows, &rest), charges))
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

    /// Scale a flow expression by the parallel multiplicity `m` (no-op for m = 1).
    /// A noise generator of density `psd` (a table's densities `table`) in
    /// the contribution being lowered: its symbol `{inst}#noise{k}`, the
    /// density divided by the contribution's scale (see `noise_scale`).
    fn noise_generator(
        &mut self,
        name: &str,
        psd: ExprId,
        flicker_exp: ExprId,
        table: Vec<(ExprId, ExprId)>,
    ) -> Result<ExprId, String> {
        let Some(scale) = self.noise_scale else {
            return Err(format!("{name} outside a contribution statement"));
        };
        let name = noise_symbol_name(&self.inst, self.noise.len());
        let g = self.ctx().sym(&name);
        let input = sym_of(self.lo.ctx(), g).expect("sym() yields a Symbol node");
        let per = |ctx: &mut Graph, e: ExprId| {
            if scale == 1.0 {
                e
            } else {
                let s = ctx.konst_f64(scale);
                ctx.div(e, s)
            }
        };
        let psd = per(self.ctx(), psd);
        let table = (table.into_iter())
            .map(|(f, p)| (f, per(self.ctx(), p)))
            .collect();
        self.noise.push(NoiseSource {
            input,
            psd,
            flicker_exp,
            table,
        });
        Ok(g)
    }

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
    /// issue #41. Where it is reached is the conjunction of the conditions
    /// of the arms around it ([`Self::path`]): constant, it fires or not;
    /// over the parameters only, an `$error` is an assertion that its path
    /// does not hold; over the solution, a runtime check SANE does not make.
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

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
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

    fn lower_if(&mut self, cond: &Expr, then: &Stmt, els: Option<&Stmt>) -> Result<(), String> {
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
    fn topology(&mut self, c: ExprId, then: &Stmt, els: Option<&Stmt>) -> Option<bool> {
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
    fn int_value(&mut self, name: &str) -> Option<ExprId> {
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
    fn reached(&mut self) -> ExprId {
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
    fn not(&mut self, c: ExprId) -> ExprId {
        let ctx = self.ctx();
        let zero = ctx.zero();
        ctx.cmp(CmpOp::Eq, c, zero)
    }

    /// Whether `e` reads the instance's parameters (and the temperature) only:
    /// a value fixed by a binding.
    fn over_params(&mut self, e: ExprId) -> bool {
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
    fn lower_branch(
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

    /// A loop, its init run: `while (cond) { body; step }`. An iteration
    /// whose condition the graph folds runs or ends the loop; any other runs
    /// gated by its condition, as `if (cond) { body; step }`, so the unrolled
    /// loop is the module's wherever the loop ends within the iterations
    /// unrolled. Past [`VA_LOOP_GATED_CAP`] gated iterations the loop ends:
    /// with an assertion that it does when the condition reads parameters
    /// only, else it cannot be lowered.
    fn lower_loop(&mut self, cond: &Expr, body: &Stmt, step: Option<&Stmt>) -> Result<(), String> {
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

    fn contribute(
        &mut self,
        access: &Access,
        hi: &str,
        lo: &Option<String>,
        rhs: &Expr,
    ) -> Result<(), String> {
        let (hn, ln) = self.resolve_pair(hi, lo);
        // A self-branch (both endpoints the same node): a zero potential
        // contribution is a no-op, as is a flow circulating within one node;
        // any other potential contribution is inconsistent.
        if hn == ln {
            if is_potential(access) {
                let v = self.expr(rhs)?;
                if self.ctx().const_f64(v) != Some(0.0) {
                    return Err(format!(
                        "potential contribution on the self-branch ({hn},{ln}) is not zero"
                    ));
                }
            }
            return Ok(());
        }
        // A noise generator in the RHS enters the residuals where the value
        // does, scaled with it (see `noise_scale`).
        let scale = if is_potential(access) {
            1.0
        } else {
            self.mfactor
        };
        let saved_scale = self.noise_scale.replace(scale);
        let val = self.expr(rhs);
        self.noise_scale = saved_scale;
        let val = val?;

        // Switch branch: its current unknown `i` is minted/stamped
        // unconditionally (in `setup`). Each arm only records its constraint as a
        // pseudo-variable, which the conditional machinery merges with `select`;
        // `finish` emits the merged constraint as this branch's residual. This is
        // what makes a `V(..) <+ ..` inside a conditional lowerable.
        let (skey, ssign) = canon(&hn, &ln);
        if is_potential(access) {
            if self.cond_depth > 0 && !self.switched.contains(&skey) {
                self.switched.push(skey.clone());
            }
            let short = self.cond_depth == 0
                && !self.probe_of.contains_key(&skey)
                && self.ctx().const_f64(val) == Some(0.0);
            if short {
                if !self.shorts.contains(&skey) {
                    self.shorts.push(skey.clone());
                }
            } else {
                self.open.insert(skey.clone());
            }
        }
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
            let i = self
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
        let i = self
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

    fn binary(&mut self, op: BinOp, a: ExprId, b: ExprId) -> ExprId {
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

    fn call(&mut self, name: &str, args: &[Expr]) -> Result<ExprId, String> {
        if name == "ddt" {
            if args.is_empty() {
                return Err("ddt expects one argument".into());
            }
            let q = self.expr(&args[0])?;
            return Ok(self.ddt(q));
        }
        if name == "white_noise" || name == "flicker_noise" {
            // A noise generator: a symbol, zero in every evaluation, that
            // enters the residuals where the contribution puts it.
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
            // A source on a conditional path is there where the path is.
            let reached = self.reached();
            let psd = self.ctx().mul(psd, reached);
            return self.noise_generator(name, psd, flicker_exp, Vec::new());
        }
        if name == "noise_table" || name == "noise_table_log" {
            // Tabular noise: a flat {f0, p0, f1, p1, ...} coefficient array,
            // chunked into (frequency, psd) points. Large-signal value is zero.
            if args.is_empty() {
                return Err(format!("{name} expects a coefficient array argument"));
            }
            let flat = self.coeffs(&args[0])?;
            let table: Vec<(ExprId, ExprId)> = flat
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| (c[0], c[1]))
                .collect();
            let reached = self.reached();
            let table = table
                .into_iter()
                .map(|(f, p)| (f, self.ctx().mul(p, reached)))
                .collect();
            let (psd, flicker_exp) = (self.ctx().zero(), self.ctx().zero());
            return self.noise_generator(name, psd, flicker_exp, table);
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
            let num = self.coeffs(&args[1])?;
            let den = self.coeffs(&args[2])?;
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
            // idt(u[,ic]) -> state s with ds/dt = u (current `-u`, charge `s`);
            // the integral value is s. At DC (at rest) this enforces u = 0,
            // the steady-state condition for the usual NQS-charge integrand.
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
                let ic = self.expr(&args[1])?;
                match self.ctx().const_f64(ic) {
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
            let s = self.lo.unknown_of(&self.inst, &sname);
            if let Some(c) = dc_seed {
                self.lo.extras.last_mut().expect("just minted").dc_seed = Some(c);
            }
            let ds = self.ddt(s);
            let r = self.ctx().sub(ds, u);
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
            let y = self.lo.unknown_of(&self.inst, &format!("dly{k}_src"));
            let r = self.ctx().sub(y, u);
            self.branch_resid.push(r);
            let out_extra = self.lo.extras.len();
            let d = self.lo.unknown_of(&self.inst, &format!("dly{k}"));
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

    /// A coefficient vector `{e0, e1, ...}`, each entry lowered.
    fn coeffs(&mut self, e: &Expr) -> Result<Vec<ExprId>, String> {
        match e {
            Expr::Array(elems) => elems.iter().map(|x| self.expr(x)).collect(),
            _ => Err("expected a coefficient vector {..}".into()),
        }
    }

    /// Lower H(s)=N(s)/D(s) (coefficient vectors, ascending powers of s) to a
    /// controllable-canonical state space: states s_i = w^(i) where D(s)w = u,
    /// output y = sum_i num_i * w^(i). One differential unknown per denominator
    /// order (its trailing coefficients that are the constant zero dropped);
    /// the integrator chain and the defining equation are residual rows. The
    /// coefficients are expressions, a parameter's included.
    fn lower_laplace_nd(
        &mut self,
        u: ExprId,
        num: &[ExprId],
        den: &[ExprId],
    ) -> Result<ExprId, String> {
        let zero_coeff = |l: &mut Self, c: ExprId| l.ctx().const_f64(c) == Some(0.0);
        let Some(k) = (0..den.len()).rev().find(|&i| !zero_coeff(self, den[i])) else {
            return Err("laplace_nd denominator is empty or all zero".into());
        };
        let n_num = (0..num.len())
            .rev()
            .find(|&i| !zero_coeff(self, num[i]))
            .map_or(0, |i| i + 1);
        if n_num > k + 1 {
            return Err(
                "laplace_nd with numerator order > denominator order is not supported".into(),
            );
        }
        let coeff = |i: usize| num.get(i).copied().filter(|_| i < n_num);
        if k == 0 {
            let Some(b0) = coeff(0) else {
                return Ok(self.ctx().zero());
            };
            let g = self.ctx().div(b0, den[0]);
            return Ok(self.ctx().mul(g, u));
        }
        let base = self.lo.extras.len();
        let mut s = Vec::with_capacity(k);
        let mut sdot = Vec::with_capacity(k);
        for i in 0..k {
            let si = self.lo.unknown_of(&self.inst, &format!("lap{base}_{i}"));
            s.push(si);
            let dsi = self.ddt(si);
            sdot.push(dsi);
        }
        for i in 0..k - 1 {
            let r = self.ctx().sub(sdot[i], s[i + 1]);
            self.branch_resid.push(r);
        }
        // a_k * s_{k-1}' + sum_{i<k} a_i s_i - u = 0
        let ctx = self.ctx();
        let mut terms: Vec<ExprId> = (0..k).map(|i| ctx.mul(den[i], s[i])).collect();
        terms.push(ctx.mul(den[k], sdot[k - 1]));
        let lhs = ctx.reduce(rsdag::ReduceOp::Sum, terms);
        let last = ctx.sub(lhs, u);
        self.branch_resid.push(last);
        // output y = sum_{i<k} b_i s_i (+ b_k * s_{k-1}' if deg num == k)
        let ctx = self.lo.ctx();
        let mut terms: Vec<ExprId> = (0..k)
            .filter_map(|i| coeff(i).map(|b| ctx.mul(b, s[i])))
            .collect();
        if let Some(bk) = coeff(k) {
            terms.push(ctx.mul(bk, sdot[k - 1]));
        }
        Ok(ctx.reduce(rsdag::ReduceOp::Sum, terms))
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
                    return; // self-branch: nothing to switch
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

/// Desugar a `case` statement to a nested if-chain
/// `if (sel==l0 || ...) body0 else if ... else default`.
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

/// Is an event body free of model semantics (see `Lower::event`)?
fn passive_event_body(s: &Stmt) -> bool {
    match s {
        Stmt::Empty | Stmt::IgnoredCall => true,
        Stmt::Block(ss) => ss.iter().all(passive_event_body),
        Stmt::SysTask { name, .. } => matches!(name.as_str(), "discontinuity" | "bound_step"),
        _ => false,
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
/// Whether `s` contributes a potential anywhere, whatever its branches
/// merge to (a short already merged reads as a self-branch).
fn contributes_potential(s: &Stmt) -> bool {
    match s {
        Stmt::Contribution { access, .. } => is_potential(access),
        Stmt::Block(ss) => ss.iter().any(contributes_potential),
        Stmt::If { then, els, .. } => {
            contributes_potential(then) || els.as_deref().is_some_and(contributes_potential)
        }
        Stmt::Case { items, default, .. } => {
            items.iter().any(|(_, b)| contributes_potential(b))
                || default.as_deref().is_some_and(contributes_potential)
        }
        Stmt::For { body, .. }
        | Stmt::While { body, .. }
        | Stmt::InitialStep(body)
        | Stmt::Event { body, .. } => contributes_potential(body),
        _ => false,
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
