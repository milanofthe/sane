//! Element-stamp templates for assembling the symbolic Jacobian.
//!
//! Every element -- a linear MNA element (R/C/L/source/controlled source) or a
//! nonlinear device -- contributes one or more current / residual terms to the
//! DAE. Each contribution is a [`Stamp`]: an expression summed into one residual
//! row. The Jacobian of the whole system is the sum of the per-stamp local
//! Jacobians, because differentiation is linear over the residual sum.
//!
//! Two stamps that are *structurally* identical -- same DAG shape, differing only
//! in which node-voltage / parameter symbols they reference (two resistors, two
//! transistors of the same model and configuration) -- share a **template**: we
//! canonicalize the contribution (renaming its symbols to a fixed pool of
//! placeholders in structural encounter order), differentiate the canonical form
//! **once**, and instantiate every other occurrence by substituting the
//! placeholders back to that instance's symbols.
//!
//! This is exact, not an approximation: a symbol renaming `sigma` commutes with
//! differentiation, so `subst(d/d$p canon, sigma) == d/d(sigma $p) instance`.
//! The assembled entries are therefore identical (after hash-consing) to
//! differentiating each contribution directly -- it is purely a build-time win,
//! and it preserves the exact expression structure (so the numerically robust
//! `select`/clamp forms are kept, unlike reverse-mode AD).

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use rsdag::{differentiate, ExprId, Graph, Node, ReduceOp, SymbolId};

/// One element contribution to the DAE: `expr` is summed into residual `row`.
/// Ports (the unknown symbols it depends on) and parameters are auto-detected.
pub struct Stamp {
    pub row: usize,
    pub expr: ExprId,
}

impl Stamp {
    pub fn new(row: usize, expr: ExprId) -> Self {
        Self { row, expr }
    }
}

/// Derive the per-element stamps of a residual system: a node KCL row is a
/// `Reduce(Sum)` of its incident element currents, so each summand is one stamp;
/// any other row (a branch constraint, an internal-node residual) is a single
/// stamp. Because `d(sum) = sum(d)`, summing the stamps' derivatives reproduces
/// the row's Jacobian exactly -- so this is the single, uniform way to obtain
/// stamps for *any* DAE, whether freshly assembled or produced by a graph
/// transform (node merge / elimination / branch pruning), with no per-transform
/// bookkeeping.
pub fn stamps_from_residuals(ctx: &Graph, residuals: &[ExprId]) -> Vec<Stamp> {
    let mut stamps = Vec::new();
    for (row, &r) in residuals.iter().enumerate() {
        match ctx.node(r) {
            Node::Reduce(ReduceOp::Sum, l) => {
                for &t in ctx.args(*l) {
                    stamps.push(Stamp::new(row, t));
                }
            }
            _ => stamps.push(Stamp::new(row, r)),
        }
    }
    stamps
}

/// A growable pool of canonical placeholder symbols, shared across all stamps so
/// that structurally-identical canonical forms intern to the *same* `ExprId`.
#[derive(Default)]
struct Placeholders {
    ports: Vec<ExprId>,
    params: Vec<ExprId>,
}

impl Placeholders {
    fn port(&mut self, ctx: &mut Graph, k: usize) -> ExprId {
        while self.ports.len() <= k {
            let i = self.ports.len();
            let s = ctx.sym(&format!("$p{i}"));
            self.ports.push(s);
        }
        self.ports[k]
    }
    fn param(&mut self, ctx: &mut Graph, k: usize) -> ExprId {
        while self.params.len() <= k {
            let i = self.params.len();
            let s = ctx.sym(&format!("$q{i}"));
            self.params.push(s);
        }
        self.params[k]
    }
}

/// The canonical form of a stamp expression plus the maps needed to instantiate.
struct Canon {
    /// Canonicalized expression (over the placeholder pool); the template key.
    canon: ExprId,
    /// Placeholder symbol -> instance expression (to substitute a template back).
    back: HashMap<SymbolId, ExprId>,
    /// Every placeholder symbol (ports *and* parameters), in encounter order,
    /// paired with the instance symbol it stands for. Which ones are columns of a
    /// given Jacobian block is decided by the caller's `col_of`, so the same
    /// canonical template serves dF/dx, dF/dx' and dF/dp.
    vars: Vec<(SymbolId, SymbolId)>,
}

/// Symbols in `expr` in deterministic structural first-encounter order. Two
/// expressions that differ only in their leaf symbols yield corresponding orders,
/// which is what makes the canonical form structural.
fn symbols_in_order(ctx: &Graph, expr: ExprId) -> Vec<SymbolId> {
    let mut seen: HashSet<ExprId> = HashSet::default();
    let mut seen_sym: HashSet<SymbolId> = HashSet::default();
    let mut order: Vec<SymbolId> = Vec::new();
    fn go(
        ctx: &Graph,
        e: ExprId,
        seen: &mut HashSet<ExprId>,
        seen_sym: &mut HashSet<SymbolId>,
        order: &mut Vec<SymbolId>,
    ) {
        if !seen.insert(e) {
            return;
        }
        match ctx.node(e) {
            Node::Const(_) => {}
            Node::Symbol(s) => {
                if seen_sym.insert(*s) {
                    order.push(*s);
                }
            }
            Node::Neg(a) | Node::Pow(a, _) | Node::Unary(_, a) => {
                go(ctx, *a, seen, seen_sym, order)
            }
            Node::Add(a, b) | Node::Mul(a, b) | Node::Cmp(_, a, b) | Node::Binary(_, a, b) => {
                go(ctx, *a, seen, seen_sym, order);
                go(ctx, *b, seen, seen_sym, order);
            }
            Node::Select(c, t, e2) => {
                go(ctx, *c, seen, seen_sym, order);
                go(ctx, *t, seen, seen_sym, order);
                go(ctx, *e2, seen, seen_sym, order);
            }
            Node::Reduce(_, l) | Node::Call(_, l) | Node::Dot(l) | Node::Solve(l, _) => {
                for &a in ctx.args(*l) {
                    go(ctx, a, seen, seen_sym, order);
                }
            }
        }
    }
    go(ctx, expr, &mut seen, &mut seen_sym, &mut order);
    order
}

/// Canonicalize a stamp expression: rename every non-time symbol to a placeholder
/// (a port placeholder if it is a differentiation variable per `is_var`, else a
/// parameter placeholder), in structural encounter order. `t` is left untouched
/// (it is globally shared and never a differentiation variable here).
fn canonicalize(
    ctx: &mut Graph,
    expr: ExprId,
    t: SymbolId,
    is_var: &impl Fn(SymbolId) -> bool,
    pool: &mut Placeholders,
) -> Canon {
    let order = symbols_in_order(ctx, expr);
    let mut fwd: HashMap<SymbolId, ExprId> = HashMap::default();
    let mut back: HashMap<SymbolId, ExprId> = HashMap::default();
    let mut vars: Vec<(SymbolId, SymbolId)> = Vec::new();
    let (mut n_port, mut n_param) = (0usize, 0usize);
    for s in order {
        if s == t {
            continue; // leave time as-is
        }
        let inst_expr = ctx.symbol_expr(s);
        // Ports (differentiation variables) and parameters draw from separate
        // placeholder pools so the two never alias, but both are recorded as
        // `vars` -- `col_of` later decides which are columns of this block.
        let cp = if is_var(s) {
            let p = pool.port(ctx, n_port);
            n_port += 1;
            p
        } else {
            let p = pool.param(ctx, n_param);
            n_param += 1;
            p
        };
        fwd.insert(s, cp);
        let cps = sym_of(ctx, cp);
        back.insert(cps, inst_expr);
        vars.push((cps, s));
    }
    let canon = rsdag::substitute(ctx, &[expr], &fwd)[0];
    Canon { canon, back, vars }
}

/// The `SymbolId` of a symbol expression (panics if not a bare symbol).
fn sym_of(ctx: &Graph, e: ExprId) -> SymbolId {
    match ctx.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!("placeholder expr is not a symbol"),
    }
}

/// One sparse Jacobian block as `(rows, cols, exprs)`.
pub type SparseBlock = (Vec<usize>, Vec<usize>, Vec<ExprId>);

/// Assemble a sparse Jacobian `(rows, cols, exprs)` from element stamps via
/// per-structure templates. `is_var` classifies a symbol as a differentiation
/// variable (port) vs. a parameter; `col_of` gives the column for the unknowns
/// that this Jacobian differentiates against (e.g. the `x` symbols for dF/dx, the
/// `xdot` symbols for dF/dx'). Ports whose instance symbol is absent from
/// `col_of` are simply skipped, so the same stamps and templates serve every
/// Jacobian block.
pub fn assemble_jacobian(
    ctx: &mut Graph,
    stamps: &[Stamp],
    t: SymbolId,
    is_var: &impl Fn(SymbolId) -> bool,
    col_of: &impl Fn(SymbolId) -> Option<usize>,
) -> SparseBlock {
    assemble_jacobians(ctx, stamps, t, is_var, &[col_of])
        .pop()
        .expect("one block requested")
}

/// [`assemble_jacobian`] for several blocks at once (dF/dx and dF/dx' of one
/// system): each stamp is canonicalised once, the derivative cache is shared,
/// and the derivatives of all blocks are instantiated in ONE substitution pass
/// per stamp. Block `b` differentiates against the columns `col_ofs[b]` names.
pub fn assemble_jacobians(
    ctx: &mut Graph,
    stamps: &[Stamp],
    t: SymbolId,
    is_var: &impl Fn(SymbolId) -> bool,
    col_ofs: &[&dyn Fn(SymbolId) -> Option<usize>],
) -> Vec<SparseBlock> {
    let mut pool = Placeholders::default();
    // (canon, placeholder) -> derivative of the canonical form w.r.t. that
    // placeholder (None if structurally zero). Differentiated once per distinct
    // element structure and reused across every instance of it.
    let mut cache: HashMap<(ExprId, SymbolId), Option<ExprId>> = HashMap::default();
    // Per block: (row, col) -> accumulated derivative expression.
    let mut accs: Vec<HashMap<(usize, usize), ExprId>> =
        (0..col_ofs.len()).map(|_| HashMap::default()).collect();

    // Per stamp: the derivatives to instantiate (deduplicated) and, per block,
    // which of them lands in which column.
    let mut dcanons: Vec<ExprId> = Vec::new();
    let mut targets: Vec<(usize, usize, usize)> = Vec::new(); // (block, col, dcanon idx)
    for stamp in stamps {
        let c = canonicalize(ctx, stamp.expr, t, is_var, &mut pool);
        dcanons.clear();
        targets.clear();
        for &(canon_sym, inst_sym) in &c.vars {
            for (b, col_of) in col_ofs.iter().enumerate() {
                let Some(col) = col_of(inst_sym) else {
                    continue;
                };
                // Only the placeholders that are columns of SOME block are
                // differentiated, so dF/dx never pays for parameter columns.
                let dcanon = match cache.get(&(c.canon, canon_sym)) {
                    Some(&d) => d,
                    None => {
                        let dd = differentiate(ctx, c.canon, canon_sym);
                        let d = if ctx.is_zero(dd) { None } else { Some(dd) };
                        cache.insert((c.canon, canon_sym), d);
                        d
                    }
                };
                let Some(dcanon) = dcanon else { continue };
                let k = match dcanons.iter().position(|&d| d == dcanon) {
                    Some(k) => k,
                    None => {
                        dcanons.push(dcanon);
                        dcanons.len() - 1
                    }
                };
                targets.push((b, col, k));
            }
        }
        // Instantiate every column's derivative in ONE substitution pass: the
        // derivatives w.r.t. different ports share the stamp's primal
        // subgraph, which a per-column pass would rebuild per column.
        let dinsts = rsdag::substitute(ctx, &dcanons, &c.back);
        for &(b, col, k) in &targets {
            let dinst = dinsts[k];
            if ctx.is_zero(dinst) {
                continue;
            }
            let key = (stamp.row, col);
            let merged = match accs[b].get(&key) {
                Some(&prev) => ctx.add(prev, dinst),
                None => dinst,
            };
            accs[b].insert(key, merged);
        }
    }

    // Flatten each block, dropping any entries that cancelled to zero.
    accs.into_iter()
        .map(|acc| {
            let mut rows = Vec::with_capacity(acc.len());
            let mut cols = Vec::with_capacity(acc.len());
            let mut exprs = Vec::with_capacity(acc.len());
            let mut entries: Vec<((usize, usize), ExprId)> = acc.into_iter().collect();
            entries.sort_unstable_by_key(|&((r, col), _)| (r, col));
            for ((r, col), e) in entries {
                if ctx.is_zero(e) {
                    continue;
                }
                rows.push(r);
                cols.push(col);
                exprs.push(e);
            }
            (rows, cols, exprs)
        })
        .collect()
}
