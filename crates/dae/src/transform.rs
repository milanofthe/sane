//! Graph transformations on an extracted [`Dae`]: node merging (shorts), exact
//! resistive-node elimination, and operating-point-guided pruning (opens) --
//! all performed on the symbolic graph, never by re-extraction from a netlist.

use std::collections::{HashMap, HashSet};

use rustc_hash::FxHashMap;

use rsdag::{differentiate, ExprId, Node, ReduceOp, SymbolId};
use sane_core::Graph;

use crate::rewrite::Rewrite;
use crate::Dae;

/// A row's terms: the summands of a `Reduce(Sum)`, nothing for zero, else the
/// row itself.
fn terms_of(ctx: &Graph, e: ExprId) -> Vec<ExprId> {
    match *ctx.node(e) {
        Node::Reduce(ReduceOp::Sum, l) => ctx.args(l).to_vec(),
        _ if ctx.is_zero(e) => Vec::new(),
        _ => vec![e],
    }
}

/// Merge node-voltage unknowns by shorting them together: for each pair, the two
/// nodes become one (the lower index survives). Replaces the eliminated node's
/// voltage symbol with the survivor's everywhere, fuses their KCL currents and
/// charges (the shorting branch's current cancels), and drops the eliminated
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

    // Substitute every eliminated node's voltage with its representative's,
    // in one pass over the currents and charges.
    let mut merge: FxHashMap<SymbolId, ExprId> = FxHashMap::default();
    for i in 0..nn {
        if rep[i] != i {
            merge.insert(dae.x[i], ctx.symbol_expr(dae.x[rep[i]]));
        }
    }
    let currents = rsdag::substitute(ctx, &dae.currents, &merge);
    let charges = rsdag::substitute(ctx, &dae.charges, &merge);

    // Fuse the KCL rows of each class into the representative, then keep only
    // representative node rows + all branch/internal rows.
    let (mut new_i, mut new_q) = (Vec::new(), Vec::new());
    let mut survivors = Vec::new();
    for i in 0..nn {
        if rep[i] == i {
            // Gather the (substituted) terms of every node merged into i.
            let (mut is, mut qs) = (Vec::new(), Vec::new());
            for j in (0..nn).filter(|&j| rep[j] == i) {
                is.extend(terms_of(ctx, currents[j]));
                qs.extend(terms_of(ctx, charges[j]));
            }
            new_i.push(ctx.reduce(ReduceOp::Sum, is));
            new_q.push(ctx.reduce(ReduceOp::Sum, qs));
            survivors.push(i);
        }
    }
    for i in nn..m {
        new_i.push(currents[i]);
        new_q.push(charges[i]);
        survivors.push(i);
    }

    let n_nodes = survivors.iter().filter(|&&i| i < nn).count();
    let reduced = dae.rewrite(
        ctx,
        Rewrite {
            rows: Some((new_i, new_q)),
            keep: survivors.clone(),
            n_nodes,
            subst: merge,
        },
    );
    (reduced, survivors)
}

/// Exactly eliminate internal resistive nodes by Gaussian (Schur) elimination
/// on the graph -- the series/star-mesh reduction that collapses a chain of
/// series resistors into one branch, losslessly and frequency-independently.
///
/// A node `N` is eliminable when it is purely resistive-linear: no incident
/// capacitance (no charge reads its voltage), no incident branch current (no
/// source or inductor anchored there), and a self-conductance
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
    let mut currents = dae.currents.clone();
    let mut charges = dae.charges.clone();
    let mut unknowns = dae.unknowns.clone();
    let mut x = dae.x.clone();
    let mut is_node: Vec<bool> = (0..x.len()).map(|i| i < nn).collect();
    // the original index of each unknown left, and what each eliminated
    // voltage is in the ones left
    let mut left: Vec<usize> = (0..x.len()).collect();
    let mut subst: FxHashMap<SymbolId, ExprId> = FxHashMap::default();

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
        // Every voltage some charge reads: a node among them has an incident
        // capacitance and is not resistive.
        let stored = ctx.free_symbols_in(&charges);

        let mut target = None;
        for i in 0..x.len() {
            if !is_node[i] || keep.contains(&unknowns[i]) {
                continue;
            }
            if stored.contains(&x[i]) || !ctx.is_zero(charges[i]) {
                continue; // incident capacitance -> not purely resistive
            }
            let fs = ctx.free_symbols(currents[i]);
            if fs.iter().any(|s| branch_set.contains(s)) {
                continue; // incident branch current -> source/inductor node
            }
            let a = differentiate(ctx, currents[i], x[i]);
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
        let a = differentiate(ctx, currents[i], x[i]);
        let zero = ctx.zero();
        let vi = x[i];
        let at_zero: FxHashMap<SymbolId, ExprId> = [(vi, zero)].into_iter().collect();
        let b = rsdag::substitute(ctx, &[currents[i]], &at_zero)[0];
        let nb = ctx.neg(b);
        let ainv = ctx.pow_i(a, -1);
        let v_expr = ctx.mul(nb, ainv);
        let inline: FxHashMap<SymbolId, ExprId> = [(vi, v_expr)].into_iter().collect();
        let inlined = rsdag::substitute(ctx, &currents, &inline);
        // the voltages eliminated before, now in the ones left
        let (syms, exprs): (Vec<SymbolId>, Vec<ExprId>) = subst.drain().unzip();
        let exprs = rsdag::substitute(ctx, &exprs, &inline);
        subst.extend(syms.into_iter().zip(exprs));
        subst.insert(vi, v_expr);
        for (j, row) in currents.iter_mut().enumerate() {
            if j != i {
                *row = inlined[j];
            }
        }
        eliminated.push(unknowns[i].clone());
        currents.remove(i);
        charges.remove(i);
        unknowns.remove(i);
        x.remove(i);
        is_node.remove(i);
        left.remove(i);
    }

    let n_nodes = nn - eliminated.len();
    let reduced = dae.rewrite(
        ctx,
        Rewrite {
            rows: Some((currents, charges)),
            keep: left,
            n_nodes,
            subst,
        },
    );
    (reduced, eliminated)
}

/// A branch term's importance at the operating point: `(|t|, Σ_j |∂t/∂x_j|)`
/// over the node voltages `x_set`. For a current term the first is the **DC
/// current** the branch carries (bias relevance -- a constant-current branch
/// has zero conductance but still sets the operating point, so it must not be
/// pruned on admittance alone) and the second its small-signal conductance;
/// for a charge term the second is its capacitance.
fn term_importance(
    ctx: &mut Graph,
    t: ExprId,
    x_set: &HashSet<SymbolId>,
    env: &HashMap<SymbolId, f64>,
) -> (f64, f64) {
    // Build all derivative expressions first (this mutates the arena), then
    // evaluate the term and every derivative in a *single* arena sweep instead
    // of one full sweep per entry.
    let mut roots = vec![t];
    for s in ctx.free_symbols(t) {
        if x_set.contains(&s) {
            roots.push(differentiate(ctx, t, s));
        }
    }
    let vals = rsdag::eval(ctx, &roots, env);
    (vals[0].abs(), vals[1..].iter().map(|v| v.abs()).sum())
}

/// A KCL row's branch term with its importance at the operating point: its
/// DC current, conductance and capacitance (a current term has no
/// capacitance, a charge term neither current nor conductance).
struct Term {
    e: ExprId,
    charge: bool,
    i: f64,
    g: f64,
    c: f64,
}

/// Every branch term of the rows `rows` (the current terms, then the charge
/// terms), with its importance.
fn branch_terms(
    ctx: &mut Graph,
    dae: &Dae,
    rows: std::ops::Range<usize>,
    x_set: &HashSet<SymbolId>,
    env: &HashMap<SymbolId, f64>,
) -> Vec<Vec<Term>> {
    rows.map(|r| {
        let mut out = Vec::new();
        for (charge, row) in [(false, dae.currents[r]), (true, dae.charges[r])] {
            for e in terms_of(ctx, row) {
                let (i, g) = term_importance(ctx, e, x_set, env);
                out.push(if charge {
                    Term {
                        e,
                        charge,
                        i: 0.0,
                        g: 0.0,
                        c: g,
                    }
                } else {
                    Term {
                        e,
                        charge,
                        i,
                        g,
                        c: 0.0,
                    }
                });
            }
        }
        out
    })
    .collect()
}

/// A term's element: the parameter symbols it reads.
fn element_of(ctx: &Graph, dae: &Dae, t: ExprId, x_all: &HashSet<SymbolId>) -> String {
    let elem: Vec<String> = ctx
        .free_symbols(t)
        .into_iter()
        .filter(|s| !x_all.contains(s) && *s != dae.t)
        .map(|s| ctx.symbol_name(s).to_string())
        .collect();
    elem.join("+")
}

/// DC operating-point evaluation environment (`t = 0`): each state bound to
/// its operating value, time to zero, and the given parameters. Shared by the
/// graph-reduction passes.
fn dc_op_env(
    ctx: &mut Graph,
    dae: &Dae,
    x_op: &[f64],
    p: &[(SymbolId, f64)],
) -> HashMap<SymbolId, f64> {
    let mut env: HashMap<SymbolId, f64> = HashMap::new();
    // noise generators are zero in every evaluation
    for g in dae.observers.generators(ctx) {
        env.insert(g, 0.0);
    }
    for (i, &s) in dae.x.iter().enumerate() {
        env.insert(s, x_op.get(i).copied().unwrap_or(0.0));
    }
    for &(s, v) in p {
        env.insert(s, v);
    }
    env.insert(dae.t, 0.0);
    env
}

/// Operating-point-guided graph reduction. Each KCL row's current and charge
/// are `Reduce(Sum, [branch terms])`; at the linearization point `(x_op, p)`
/// each current term has a DC current and a conductance, each charge term a
/// capacitance. A term that is negligible (relative to its node's dominant
/// admittance) at *all* nodes it touches is dropped -- consistently from both
/// KCL rows it couples (the term and its hash-consed negation), so charge
/// conservation holds.
///
/// Returns the reduced DAE (same unknowns; only the sums shrink) and the list
/// of pruned branches as `(element_name, relative_importance)`, sorted by
/// importance ascending -- the order in which they fall away (least relevant
/// first), so the caller can reuse it (e.g. to map back to netlist elements). No
/// re-extraction from the netlist; the reduction happens on the graph.
pub(crate) fn prune_graph(
    ctx: &mut Graph,
    dae: &Dae,
    x_op: &[f64],
    p: &[(SymbolId, f64)],
    omegas: &[f64],
    rel_tol: f64,
) -> (Dae, Vec<(String, f64)>) {
    let env = dc_op_env(ctx, dae, x_op, p);
    // The noise generators and the element each belongs to (`R1` of
    // `R1#noise`, `M1` of `M1#noise2`).
    let owner: HashMap<SymbolId, String> = (dae.observers.generators(ctx).into_iter())
        .map(|g| {
            let name = ctx.symbol_name(g);
            (
                g,
                name.split_once("#noise")
                    .map_or(name, |(o, _)| o)
                    .to_string(),
            )
        })
        .collect();

    // Admittance is differentiated only w.r.t. node voltages (not branch-current
    // unknowns, whose self-derivative would be a spurious conductance of 1).
    let nn = dae.n_nodes;
    let x_set: HashSet<SymbolId> = dae.x[..nn].iter().copied().collect();
    let x_all: HashSet<SymbolId> = dae.x.iter().copied().collect();
    // Branch-current unknowns (i_V1, i_L1, ...): these couple voltage-defined
    // branches and must never be pruned, even when they carry no current at the
    // operating point (e.g. an AC-only source at DC).
    let branch_i: HashSet<SymbolId> = dae.x[nn..].iter().copied().collect();

    let n = dae.dim();
    let nw = omegas.len();
    let terms = branch_terms(ctx, dae, 0..n, &x_set, &env);
    let row_of: HashMap<(bool, ExprId), usize> = (terms.iter().enumerate())
        .flat_map(|(r, ts)| ts.iter().map(move |t| ((t.charge, t.e), r)))
        .collect();
    // A node's terms can be dropped only where it has more than one: its
    // dominant term stays.
    let prunable = |r: usize| terms[r].len() > 1;

    // Per-node scales: the DC-current scale (max branch current) and the
    // admittance scale at each frequency (|g + jω c|, so a parasitic is measured
    // against the node's conductance too, not only against other capacitances).
    let adm = |g: f64, c: f64, om: f64| (g * g + (om * c) * (om * c)).sqrt();
    let mut iscale = vec![0.0f64; n];
    let mut nscale = vec![vec![0.0f64; nw]; n];
    for (r, ts) in terms.iter().enumerate() {
        for t in ts {
            iscale[r] = iscale[r].max(t.i);
            for (w, &om) in omegas.iter().enumerate() {
                nscale[r][w] = nscale[r][w].max(adm(t.g, t.c, om));
            }
        }
    }

    // Drop a term only if it is negligible relative to its node's admittance
    // at every frequency and at every node it couples -- consistently from both
    // KCL rows (the term and its hash-consed negation). Record each pruned
    // branch by element name and its peak relative importance.
    let mut drop: HashSet<(usize, bool, ExprId)> = HashSet::new();
    let mut pruned: HashMap<String, f64> = HashMap::new();
    for (r, ts) in terms.iter().enumerate() {
        for t in ts {
            if !prunable(r) {
                continue;
            }
            let nt = ctx.neg(t.e);
            let other = row_of.get(&(t.charge, nt)).copied().filter(|&j| j != r);
            if other.is_some_and(|j| !prunable(j)) {
                continue;
            }
            // DC-current relevance (at both nodes), then admittance over the band.
            let rows = std::iter::once(r).chain(other);
            let mut max_rel = rows
                .clone()
                .map(|j| t.i / iscale[j].max(1e-300))
                .fold(0.0, f64::max);
            for (w, &om) in omegas.iter().enumerate() {
                let a = adm(t.g, t.c, om);
                for j in rows.clone() {
                    max_rel = max_rel.max(a / nscale[j][w].max(1e-300));
                }
            }
            // Never prune a term coupling a branch-current unknown, nor a
            // noise generator on its own (it goes with its element, below).
            let structural = (ctx.free_symbols(t.e).iter())
                .any(|s| branch_i.contains(s) || owner.contains_key(s));
            if max_rel < rel_tol && !structural {
                drop.insert((r, t.charge, t.e));
                if let Some(j) = other {
                    drop.insert((j, t.charge, nt));
                }
                let name = element_of(ctx, dae, t.e, &x_all);
                let name = if name.is_empty() {
                    "?".to_string()
                } else {
                    name
                };
                pruned
                    .entry(name)
                    .and_modify(|v| *v = v.min(max_rel))
                    .or_insert(max_rel);
            }
        }
    }
    // A noise generator goes where its element went: out of every row the
    // element's terms all left.
    for (r, ts) in terms.iter().enumerate() {
        for t in ts.iter().filter(|t| !t.charge) {
            let fs = ctx.free_symbols(t.e);
            let Some(o) = fs.iter().find_map(|s| owner.get(s)) else {
                continue;
            };
            let of_owner = |u: &&Term| {
                !u.charge
                    && u.e != t.e
                    && (ctx.free_symbols(u.e).iter())
                        .filter(|s| !owner.contains_key(s))
                        .any(|s| {
                            let n = ctx.symbol_name(*s);
                            n == o
                                || n.strip_prefix(o.as_str())
                                    .is_some_and(|rest| rest.starts_with('.'))
                        })
            };
            let mut own = ts.iter().filter(of_owner).peekable();
            if own.peek().is_some() && own.all(|u| drop.contains(&(r, false, u.e))) {
                drop.insert((r, false, t.e));
            }
        }
    }
    let mut pruned: Vec<(String, f64)> = pruned.into_iter().collect();
    pruned.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Rebuild the rows without the dropped branch terms.
    let [currents, charges] = [false, true].map(|charge| {
        (0..n)
            .map(|r| {
                let row = if charge {
                    dae.charges[r]
                } else {
                    dae.currents[r]
                };
                if !(terms[r].iter())
                    .any(|t| t.charge == charge && drop.contains(&(r, charge, t.e)))
                {
                    return row;
                }
                let kept: Vec<ExprId> = (terms[r].iter())
                    .filter(|t| t.charge == charge && !drop.contains(&(r, charge, t.e)))
                    .map(|t| t.e)
                    .collect();
                ctx.reduce(ReduceOp::Sum, kept)
            })
            .collect::<Vec<_>>()
    });

    let reduced = dae.rewrite(
        ctx,
        Rewrite {
            rows: Some((currents, charges)),
            keep: (0..n).collect(),
            n_nodes: dae.n_nodes,
            subst: FxHashMap::default(),
        },
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
    omegas: &[f64],
    rel_tol: f64,
) -> (Vec<(usize, usize)>, Vec<String>) {
    let nn = dae.n_nodes;
    let x_set: HashSet<SymbolId> = dae.x[..nn].iter().copied().collect();
    let x_idx: HashMap<SymbolId, usize> = dae.x[..nn]
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();
    let x_all: HashSet<SymbolId> = dae.x.iter().copied().collect();
    let adm = |g: f64, c: f64, om: f64| (g * g + (om * c) * (om * c)).sqrt();

    // Collect node-to-node admittance branches and (g, c) per node.
    let terms = branch_terms(ctx, dae, 0..nn, &x_set, env);
    let mut branches: Vec<(usize, usize, f64, f64, String)> = Vec::new();
    let mut node_adm: Vec<Vec<(f64, f64)>> = vec![Vec::new(); nn];
    let mut seen: HashSet<(usize, usize, String)> = HashSet::new();
    for (i, ts) in terms.iter().enumerate() {
        for t in ts {
            node_adm[i].push((t.g, t.c));
            let fs = ctx.free_symbols(t.e);
            let nodes: Vec<usize> = fs.iter().filter_map(|s| x_idx.get(s).copied()).collect();
            // Only LINEAR admittance branches are shortable: a node-to-node
            // term whose conductance (or capacitance) does not depend on any
            // node voltage (a resistor, not a forward-biased diode that merely
            // looks like a wire at this operating point).
            let linear = nodes.len() == 2 && {
                let d = differentiate(ctx, t.e, dae.x[nodes[0]]);
                !ctx.free_symbols(d).iter().any(|s| x_set.contains(s))
            };
            if linear {
                let (a, b) = (nodes[0].min(nodes[1]), nodes[0].max(nodes[1]));
                let name = element_of(ctx, dae, t.e, &x_all);
                if seen.insert((a, b, name.clone())) {
                    branches.push((a, b, t.g, t.c, name));
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
    let env = dc_op_env(ctx, dae, x_op, p);
    let (pairs, short_names) = detect_shorts(ctx, dae, &env, omegas, rel_tol);
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
