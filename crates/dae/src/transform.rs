//! Graph transformations on an extracted [`Dae`]: node merging (shorts), exact
//! resistive-node elimination, and operating-point-guided pruning (opens) --
//! all performed on the symbolic graph, never by re-extraction from a netlist.

use std::collections::{HashMap, HashSet};

use rustc_hash::FxHashMap;

use rsdag::{differentiate, ExprId, Node, ReduceOp, SymbolId};
use sane_core::Graph;

use crate::{stamp, Dae, UnknownKind};

impl Dae {
    /// A transformed copy of this DAE over a (possibly reduced) unknown layout,
    /// with the Jacobian stamps rebuilt from the new residuals. The analysis-only
    /// registries a graph transform cannot preserve (delays, events, companion
    /// network, noise sources, op-vars, device limits, source shapes) are
    /// cleared; the time symbol carries over.
    fn transformed(
        &self,
        ctx: &mut Graph,
        n_nodes: usize,
        residuals: Vec<ExprId>,
        unknowns: Vec<String>,
        kinds: Vec<UnknownKind>,
        x: Vec<SymbolId>,
        xdot: Vec<Option<SymbolId>>,
    ) -> Dae {
        let stamps = stamp::stamps_from_residuals(ctx, &residuals);
        Dae {
            residuals,
            n_nodes,
            param_defaults: self.param_defaults.clone(),
            unknowns,
            kinds,
            x,
            xdot,
            t: self.t,
            events: Vec::new(),
            delays: Vec::new(),
            stamps,
            companion: Vec::new(),
            noise_sources: Vec::new(),
            op_vars: Vec::new(),
            // Name-keyed: seeds of surviving unknowns stay valid, the rest are
            // ignored at lookup.
            dc_seeds: self.dc_seeds.clone(),
            limits: Vec::new(),
            sources: Vec::new(),
            source_names: Vec::new(),
        }
    }
}

/// Merge node-voltage unknowns by shorting them together: for each pair, the two
/// nodes become one (the lower index survives). Replaces the eliminated node's
/// voltage / derivative symbols with the survivor's everywhere, fuses their KCL
/// residuals (the shorting branch's current cancels), and drops the eliminated
/// unknown -- so the dimension shrinks. A graph transformation, no re-extraction.
/// Also returns, for each surviving unknown, its original index (to remap an
/// operating point onto the reduced system).
pub(crate) fn merge_nodes(
    ctx: &mut Graph,
    dae: &Dae,
    pairs: &[(usize, usize)],
) -> (Dae, Vec<usize>) {
    let m = dae.x.len();
    let nn = dae.n_nodes;

    // Union-find over node indices; the smallest index in a class survives.
    let mut rep: Vec<usize> = (0..m).collect();
    fn find(rep: &mut [usize], mut i: usize) -> usize {
        while rep[i] != i {
            rep[i] = rep[rep[i]];
            i = rep[i];
        }
        i
    }
    for &(a, b) in pairs {
        if a < nn && b < nn {
            let (ra, rb) = (find(&mut rep, a), find(&mut rep, b));
            if ra != rb {
                rep[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    for i in 0..m {
        rep[i] = find(&mut rep, i);
    }

    // Substitute every eliminated node's symbols with its representative's,
    // in one pass over the residuals.
    let mut merge: FxHashMap<SymbolId, ExprId> = FxHashMap::default();
    for i in 0..nn {
        let r = rep[i];
        if r != i {
            merge.insert(dae.x[i], ctx.symbol_expr(dae.x[r]));
            if let (Some(si), Some(sr)) = (dae.xdot[i], dae.xdot[r]) {
                merge.insert(si, ctx.symbol_expr(sr));
            }
        }
    }
    let residuals = rsdag::substitute(ctx, &dae.residuals, &merge);

    // Fuse the KCL residuals of each class into the representative, then keep
    // only representative node residuals + all branch/internal residuals.
    let mut new_res = Vec::new();
    let mut new_unknowns = Vec::new();
    let mut new_x = Vec::new();
    let mut new_xdot = Vec::new();
    let mut survivors = Vec::new();
    for i in 0..nn {
        if rep[i] == i {
            // Gather the (substituted) terms of every node merged into i.
            let mut terms: Vec<ExprId> = Vec::new();
            for j in 0..nn {
                if rep[j] == i {
                    match *ctx.node(residuals[j]) {
                        Node::Reduce(ReduceOp::Sum, l) => terms.extend_from_slice(ctx.args(l)),
                        other => {
                            let _ = other;
                            terms.push(residuals[j]);
                        }
                    }
                }
            }
            new_res.push(ctx.reduce(ReduceOp::Sum, terms));
            new_unknowns.push(dae.unknowns[i].clone());
            new_x.push(dae.x[i]);
            new_xdot.push(dae.xdot[i]);
            survivors.push(i);
        }
    }
    for i in nn..m {
        new_res.push(residuals[i]);
        new_unknowns.push(dae.unknowns[i].clone());
        new_x.push(dae.x[i]);
        new_xdot.push(dae.xdot[i]);
        survivors.push(i);
    }

    let n_nodes = survivors.iter().filter(|&&i| i < nn).count();
    let new_kinds: Vec<UnknownKind> = survivors.iter().map(|&i| dae.kinds[i]).collect();
    let reduced = dae.transformed(
        ctx,
        n_nodes,
        new_res,
        new_unknowns,
        new_kinds,
        new_x,
        new_xdot,
    );
    (reduced, survivors)
}

/// Exactly eliminate internal resistive nodes by Gaussian (Schur) elimination
/// on the graph -- the series/star-mesh reduction that collapses a chain of
/// series resistors into one branch, losslessly and frequency-independently.
///
/// A node `N` is eliminable when it is purely resistive-linear: no incident
/// capacitance (its derivative symbol appears in no residual), no incident
/// branch current (no source or inductor anchored there), and a self-conductance
/// `A = ∂KCL_N/∂v_N` that is a nonzero constant -- so every incident branch is a
/// linear resistor, not a nonlinear device that merely looks resistive at the
/// bias. Its KCL `A·v_N + B = 0` then solves to `v_N = -B/A` (a linear
/// combination of its neighbors' voltages); inlining that everywhere folds its
/// series branches into direct branches between the neighbors. Because the
/// elimination is exact, the response between *all* remaining nodes is preserved
/// identically. `keep` (unknown names) protects probe nodes from removal.
///
/// Nodes are eliminated greedily one at a time, re-scanning between each (a
/// neighbor's degree changes as chains collapse). Returns the reduced DAE and
/// the names of the eliminated nodes, in elimination order.
pub fn eliminate_nodes(ctx: &mut Graph, dae: &Dae, keep: &HashSet<String>) -> (Dae, Vec<String>) {
    let nn = dae.n_nodes;
    let mut residuals = dae.residuals.clone();
    let mut unknowns = dae.unknowns.clone();
    let mut kinds = dae.kinds.clone();
    let mut x = dae.x.clone();
    let mut xdot = dae.xdot.clone();
    let mut is_node: Vec<bool> = (0..x.len()).map(|i| i < nn).collect();

    // Branch-current symbols never change (we only remove node unknowns); a node
    // whose KCL touches one anchors a source or inductor and is not eliminable.
    let branch_set: HashSet<SymbolId> = (0..x.len())
        .filter(|&i| !is_node[i])
        .map(|i| x[i])
        .collect();

    let mut eliminated = Vec::new();
    loop {
        let volt_set: HashSet<SymbolId> =
            (0..x.len()).filter(|&i| is_node[i]).map(|i| x[i]).collect();
        // Every symbol that currently appears anywhere: a node's derivative
        // symbol showing up here means an incident capacitor (not resistive).
        let mut live: HashSet<SymbolId> = HashSet::new();
        for &r in &residuals {
            live.extend(ctx.free_symbols(r));
        }

        let mut target = None;
        for i in 0..x.len() {
            if !is_node[i] || keep.contains(&unknowns[i]) {
                continue;
            }
            if let Some(d) = xdot[i] {
                if live.contains(&d) {
                    continue; // incident capacitance -> not purely resistive
                }
            }
            let fs = ctx.free_symbols(residuals[i]);
            if fs.iter().any(|s| branch_set.contains(s)) {
                continue; // incident branch current -> source/inductor node
            }
            let a = differentiate(ctx, residuals[i], x[i]);
            if ctx.is_zero(a) {
                continue; // no self-conductance
            }
            if ctx.free_symbols(a).iter().any(|s| volt_set.contains(s)) {
                continue; // voltage-dependent conductance -> nonlinear device
            }
            target = Some(i);
            break;
        }
        let Some(i) = target else { break };

        // KCL_N = A*v_N + B = 0  ->  v_N = -B/A, with A = ∂KCL/∂v_N (constant)
        // and B = KCL|_{v_N=0}. Inline v_N everywhere else, then drop node N.
        let a = differentiate(ctx, residuals[i], x[i]);
        let zero = ctx.zero();
        let vi = x[i];
        let at_zero: FxHashMap<SymbolId, ExprId> = [(vi, zero)].into_iter().collect();
        let b = rsdag::substitute(ctx, &[residuals[i]], &at_zero)[0];
        let nb = ctx.neg(b);
        let ainv = ctx.pow_i(a, -1);
        let v_expr = ctx.mul(nb, ainv);
        let inline: FxHashMap<SymbolId, ExprId> = [(vi, v_expr)].into_iter().collect();
        let inlined = rsdag::substitute(ctx, &residuals, &inline);
        for (j, res) in residuals.iter_mut().enumerate() {
            if j != i {
                *res = inlined[j];
            }
        }
        eliminated.push(unknowns[i].clone());
        residuals.remove(i);
        unknowns.remove(i);
        kinds.remove(i);
        x.remove(i);
        xdot.remove(i);
        is_node.remove(i);
    }

    let n_nodes = nn - eliminated.len();
    let reduced = dae.transformed(ctx, n_nodes, residuals, unknowns, kinds, x, xdot);
    (reduced, eliminated)
}

/// A branch term's importance at the operating point: `(|t|, Σ_j |∂t/∂x_j|,
/// Σ_j |∂t/∂x'_j|)`. The first is the **DC current** the branch carries (bias
/// relevance -- a constant-current branch has zero conductance but still sets
/// the operating point, so it must not be pruned on admittance alone); the
/// second its small-signal conductance, the third its capacitance.
fn term_importance(
    ctx: &mut Graph,
    t: ExprId,
    x_set: &HashSet<SymbolId>,
    xdot_set: &HashSet<SymbolId>,
    env: &HashMap<SymbolId, f64>,
) -> (f64, f64, f64) {
    // Build all derivative expressions first (this mutates the arena), then
    // evaluate the term and every derivative in a *single* arena sweep instead
    // of one full sweep per entry.
    let mut roots = vec![t];
    let mut is_cond = Vec::new(); // true: conductance (x), false: capacitance (xdot)
    for s in ctx.free_symbols(t) {
        if x_set.contains(&s) {
            roots.push(differentiate(ctx, t, s));
            is_cond.push(true);
        } else if xdot_set.contains(&s) {
            roots.push(differentiate(ctx, t, s));
            is_cond.push(false);
        }
    }
    let vals = rsdag::eval(ctx, &roots, env);
    let i = vals[0].abs();
    let (mut g, mut c) = (0.0, 0.0);
    for (k, &cond) in is_cond.iter().enumerate() {
        let v = vals[k + 1].abs();
        if cond {
            g += v;
        } else {
            c += v;
        }
    }
    (i, g, c)
}

/// Operating-point-guided graph reduction. Each KCL residual is a
/// `Reduce(Sum, [branch currents])`; at the linearization point `(x_op, p)` each
/// branch term has a conductance and capacitance importance. A branch that is
/// negligible (relative to the dominant term) in *both* regimes at *all* nodes
/// it touches is dropped -- consistently from both KCL residuals it couples
/// (the term and its hash-consed negation), so charge conservation holds.
///
/// Returns the reduced DAE (same unknowns; only the residual sums shrink) and
/// the list of pruned branches as `(element_name, relative_importance)`, sorted
/// by importance ascending -- the order in which they fall away (least relevant
/// first), so the caller can reuse it (e.g. to map back to netlist elements). No
/// re-extraction from the netlist; the reduction happens on the graph.
///
/// DC operating-point evaluation environment (`xdot = 0`, `t = 0`): each state
/// bound to its operating value, every differential variable and time to zero,
/// and the given parameters. Shared by the graph-reduction passes.
fn dc_op_env(dae: &Dae, x_op: &[f64], p: &[(SymbolId, f64)]) -> HashMap<SymbolId, f64> {
    let mut env: HashMap<SymbolId, f64> = HashMap::new();
    for (i, &s) in dae.x.iter().enumerate() {
        env.insert(s, x_op.get(i).copied().unwrap_or(0.0));
    }
    for opt in dae.xdot.iter().flatten() {
        env.insert(*opt, 0.0);
    }
    for &(s, v) in p {
        env.insert(s, v);
    }
    env.insert(dae.t, 0.0);
    env
}

pub(crate) fn prune_graph(
    ctx: &mut Graph,
    dae: &Dae,
    x_op: &[f64],
    p: &[(SymbolId, f64)],
    omegas: &[f64],
    rel_tol: f64,
) -> (Dae, Vec<(String, f64)>) {
    // Evaluation environment at the operating point (xdot = 0 linearization).
    let env = dc_op_env(dae, x_op, p);

    // Admittance is differentiated only w.r.t. node voltages (not branch-current
    // unknowns, whose self-derivative would be a spurious conductance of 1).
    let nn = dae.n_nodes;
    let x_set: HashSet<SymbolId> = dae.x[..nn].iter().copied().collect();
    let xdot_set: HashSet<SymbolId> = dae.xdot[..nn].iter().flatten().copied().collect();
    let x_all: HashSet<SymbolId> = dae.x.iter().copied().collect();
    let xdot_all: HashSet<SymbolId> = dae.xdot.iter().flatten().copied().collect();
    // Branch-current unknowns (i_V1, i_L1, ...): these couple voltage-defined
    // branches and must never be pruned, even when they carry no current at the
    // operating point (e.g. an AC-only source at DC).
    let branch_i: HashSet<SymbolId> = dae.x[nn..].iter().copied().collect();

    // Collect the KCL term lists and per-term importances + per-residual scales.
    let kcl: Vec<Option<Vec<ExprId>>> = dae
        .residuals
        .iter()
        .map(|&r| match ctx.node(r) {
            Node::Reduce(ReduceOp::Sum, l) => Some(ctx.args(*l).to_vec()),
            _ => None,
        })
        .collect();
    let n = dae.residuals.len();
    let nw = omegas.len();
    let mut imp: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); n];
    let mut term_res: HashMap<ExprId, usize> = HashMap::new();
    for i in 0..n {
        if let Some(ts) = &kcl[i] {
            for &t in ts {
                term_res.insert(t, i);
                imp[i].push(term_importance(ctx, t, &x_set, &xdot_set, &env));
            }
        }
    }

    // Per-node scales: the DC-current scale (max branch current) and the
    // admittance scale at each frequency (|g + jω c|, so a parasitic is measured
    // against the node's conductance too, not only against other capacitances).
    let adm = |g: f64, c: f64, om: f64| (g * g + (om * c) * (om * c)).sqrt();
    let mut iscale = vec![0.0f64; n];
    let mut nscale = vec![vec![0.0f64; nw]; n];
    for i in 0..n {
        for &(ii, g, c) in &imp[i] {
            iscale[i] = iscale[i].max(ii);
            for (w, &om) in omegas.iter().enumerate() {
                nscale[i][w] = nscale[i][w].max(adm(g, c, om));
            }
        }
    }

    // Drop a branch only if it is negligible relative to its node's admittance
    // at every frequency and at every node it couples -- consistently from both
    // KCL residuals (the term and its hash-consed negation). Record each pruned
    // branch by element name and its peak relative importance.
    let mut drop_term: HashSet<(usize, ExprId)> = HashSet::new();
    let mut pruned: HashMap<String, f64> = HashMap::new();
    for i in 0..n {
        if let Some(ts) = &kcl[i] {
            for (k, &t) in ts.iter().enumerate() {
                let (ii, g, c) = imp[i][k];
                let nc = ctx.neg(t);
                let other = term_res.get(&nc).copied().filter(|&j| j != i);
                // DC-current relevance (at both nodes), then admittance over the band.
                let mut max_rel = ii / iscale[i].max(1e-300);
                if let Some(j) = other {
                    max_rel = max_rel.max(ii / iscale[j].max(1e-300));
                }
                max_rel = omegas.iter().enumerate().fold(max_rel, |acc, (w, &om)| {
                    let a = adm(g, c, om);
                    let mut rel = a / nscale[i][w].max(1e-300);
                    if let Some(j) = other {
                        rel = rel.max(a / nscale[j][w].max(1e-300));
                    }
                    acc.max(rel)
                });
                // Never prune a term coupling a branch-current unknown.
                let structural = ctx.free_symbols(t).iter().any(|s| branch_i.contains(s));
                if max_rel < rel_tol && !structural {
                    drop_term.insert((i, t));
                    if let Some(j) = other {
                        drop_term.insert((j, nc));
                    }
                    // The branch's element is the parameter symbol in the term.
                    let elem: Vec<String> = ctx
                        .free_symbols(t)
                        .into_iter()
                        .filter(|s| !x_all.contains(s) && !xdot_all.contains(s) && *s != dae.t)
                        .map(|s| ctx.symbol_name(s).to_string())
                        .collect();
                    let name = if elem.is_empty() {
                        "?".to_string()
                    } else {
                        elem.join("+")
                    };
                    pruned
                        .entry(name)
                        .and_modify(|v| *v = v.min(max_rel))
                        .or_insert(max_rel);
                }
            }
        }
    }
    let mut pruned: Vec<(String, f64)> = pruned.into_iter().collect();
    pruned.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Rebuild the residuals without the dropped branch terms.
    let new_residuals: Vec<ExprId> = (0..n)
        .map(|i| {
            if let Some(ts) = &kcl[i] {
                let kept: Vec<ExprId> = ts
                    .iter()
                    .copied()
                    .filter(|&t| !drop_term.contains(&(i, t)))
                    .collect();
                ctx.reduce(ReduceOp::Sum, kept)
            } else {
                dae.residuals[i]
            }
        })
        .collect();

    let reduced = dae.transformed(
        ctx,
        dae.n_nodes,
        new_residuals,
        dae.unknowns.clone(),
        dae.kinds.clone(),
        dae.x.clone(),
        dae.xdot.clone(),
    );
    (reduced, pruned)
}

/// Identify branches whose two nodes are electrically the *same* node, to merge.
///
/// A linear two-terminal branch makes its nodes one node when the voltage across
/// it is forced negligible -- i.e. (voltage division) its admittance swamps the
/// *total* remaining admittance of both of its nodes, at every frequency in the
/// band. This is a symmetric "same node" test on the branch and the Thévenin
/// load it sees: not a pairwise dominance contest against the single largest
/// neighbor. Returns the node-index pairs to merge and the element names.
fn detect_shorts(
    ctx: &mut Graph,
    dae: &Dae,
    env: &HashMap<SymbolId, f64>,
    x_set: &HashSet<SymbolId>,
    xdot_set: &HashSet<SymbolId>,
    omegas: &[f64],
    rel_tol: f64,
) -> (Vec<(usize, usize)>, Vec<String>) {
    let nn = dae.n_nodes;
    let x_idx: HashMap<SymbolId, usize> = dae.x[..nn]
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();
    let x_all: HashSet<SymbolId> = dae.x.iter().copied().collect();
    let xdot_all: HashSet<SymbolId> = dae.xdot.iter().flatten().copied().collect();
    let adm = |g: f64, c: f64, om: f64| (g * g + (om * c) * (om * c)).sqrt();

    // Collect node-to-node admittance branches and (g, c) per node.
    let mut branches: Vec<(usize, usize, f64, f64, String)> = Vec::new();
    let mut node_adm: Vec<Vec<(f64, f64)>> = vec![Vec::new(); nn];
    let mut seen: HashSet<(usize, usize, String)> = HashSet::new();
    for i in 0..nn {
        if let Node::Reduce(ReduceOp::Sum, l) = *ctx.node(dae.residuals[i]) {
            let ts = ctx.args(l).to_vec();
            for t in ts {
                let fs = ctx.free_symbols(t);
                let nodes: Vec<usize> = fs.iter().filter_map(|s| x_idx.get(s).copied()).collect();
                let (_, g, c) = term_importance(ctx, t, x_set, xdot_set, env);
                node_adm[i].push((g, c));
                // Only LINEAR admittance branches are shortable: a node-to-node
                // term whose conductance does not depend on any node voltage
                // (a resistor, not a forward-biased diode that merely looks like
                // a wire at this operating point).
                let linear = nodes.len() == 2 && {
                    let d = differentiate(ctx, t, dae.x[nodes[0]]);
                    !ctx.free_symbols(d).iter().any(|s| x_set.contains(s))
                };
                if linear {
                    let (a, b) = (nodes[0].min(nodes[1]), nodes[0].max(nodes[1]));
                    let elem: Vec<String> = fs
                        .iter()
                        .filter(|s| !x_all.contains(s) && !xdot_all.contains(s) && **s != dae.t)
                        .map(|s| ctx.symbol_name(*s).to_string())
                        .collect();
                    let name = elem.join("+");
                    if seen.insert((a, b, name.clone())) {
                        branches.push((a, b, g, c, name));
                    }
                }
            }
        }
    }
    // Total admittance at a node, at frequency `om`, of every branch OTHER than
    // this one -- the Thévenin load the branch sees (one instance of its own
    // (g, c) removed). Magnitudes summed: a conservative bound (|sum| <= sum|.|),
    // so we merge only when the branch truly swamps everything else.
    let node_other_sum = |node: usize, gc: (f64, f64), om: f64| {
        let mut removed = false;
        let mut s = 0.0f64;
        for &(g, c) in &node_adm[node] {
            if !removed && (g, c) == gc {
                removed = true;
                continue;
            }
            s += adm(g, c, om);
        }
        s
    };

    let mut pairs = Vec::new();
    let mut names = Vec::new();
    for (a, b, g, c, name) in branches {
        // Same node: at every frequency the branch admittance swamps the total
        // remaining admittance of both of its nodes by 1/rel_tol, so the voltage
        // it drops (the voltage-division ratio) stays below rel_tol across the
        // whole band -- the two nodes move together and are one node.
        let merge = omegas.iter().all(|&om| {
            let s = adm(g, c, om);
            s * rel_tol > node_other_sum(a, (g, c), om)
                && s * rel_tol > node_other_sum(b, (g, c), om)
        });
        if merge {
            pairs.push((a, b));
            names.push(name);
        }
    }
    (pairs, names)
}

/// The full operating-point-guided graph transformation: **open** negligible
/// branches (drop their terms) and **short** near-wire branches (merge their
/// nodes), both on the graph. Returns the reduced DAE and the list of
/// `(element, operation)` applied (`"open"` / `"short"`), opens first (in
/// importance order), then shorts.
pub fn reduce_graph(
    ctx: &mut Graph,
    dae: &Dae,
    x_op: &[f64],
    p: &[(SymbolId, f64)],
    omegas: &[f64],
    rel_tol: f64,
) -> (Dae, Vec<(String, String)>) {
    // SHORT first: merge near-wire branches' nodes, so the remaining admittance
    // scales are sane before the OPEN pass (a wire would otherwise dominate a
    // node's scale and make every real branch there look negligible).
    let env = dc_op_env(dae, x_op, p);
    let nn = dae.n_nodes;
    let v_set: HashSet<SymbolId> = dae.x[..nn].iter().copied().collect();
    let vdot_set: HashSet<SymbolId> = dae.xdot[..nn].iter().flatten().copied().collect();
    let (pairs, short_names) = detect_shorts(ctx, dae, &env, &v_set, &vdot_set, omegas, rel_tol);
    let (merged, survivors) = merge_nodes(ctx, dae, &pairs);
    let x_merged: Vec<f64> = survivors
        .iter()
        .map(|&oi| x_op.get(oi).copied().unwrap_or(0.0))
        .collect();

    // OPEN pass on the merged graph.
    let (opened, opened_list) = prune_graph(ctx, &merged, &x_merged, p, omegas, rel_tol);

    let mut transforms: Vec<(String, String)> = short_names
        .into_iter()
        .map(|e| (e, "short".to_string()))
        .collect();
    transforms.extend(
        opened_list
            .into_iter()
            .map(|(e, _)| (e, "open".to_string())),
    );
    (opened, transforms)
}
