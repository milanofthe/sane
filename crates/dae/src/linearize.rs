//! Linearisation transform: the small-signal *linear mass-matrix DAE*.
//!
//! The nonlinear system `I(x, t) + d/dt Q(x) = 0` is rewritten into a linear
//! DAE
//!
//! ```text
//!     G dx + d/dt (C dx) = 0,    G = dI/dx,   C = dQ/dx
//! ```
//!
//! represented in the *same* graph and the *same* [`Dae`] abstraction: the
//! currents and charges are linear forms in the perturbation unknowns `dx`,
//! whose coefficients are the operating-point Jacobians. The bias unknowns
//! inside those coefficients are *frozen* to constant operating-point symbols
//! (`name#op`), so the system is genuinely linear in `dx` -- differentiating a
//! linearised row recovers `G`/`C` exactly rather than `G + (dG/dx) x`.

use rustc_hash::FxHashMap;

use rsdag::{ExprId, SymbolId};
use sane_core::Graph;

use crate::Dae;

/// The operating-point freeze map: every unknown symbol `x_j` is mapped to a
/// fresh constant symbol named `"{name}#op"`. Applied to a Jacobian entry, it
/// turns the bias unknowns into operating-point constants so the resulting
/// coefficient is constant w.r.t. the perturbation unknowns.
pub fn freeze_op_point(ctx: &mut Graph, dae: &Dae) -> FxHashMap<SymbolId, ExprId> {
    (dae.x.iter())
        .map(|&xs| {
            let fname = format!("{}#op", ctx.symbol_name(xs));
            (xs, ctx.sym(&fname))
        })
        .collect()
}

/// Linearise the DAE about its operating point into the linear DAE
/// `G dx + d/dt (C dx) + H dh + B_n n + B_u du = 0` (see the module docs).
/// The returned [`Dae`] keeps the same unknowns and time symbol; row `i`
/// carries the current `sum_j G_ij x_j`, the charge `sum_j C_ij x_j`, and
/// what drives the small signal: every transport delay's history `h_k`
/// (each delay keeps its time, its signal linearized), every noise generator
/// `n_q` (each source keeps its level, frozen at the operating point), and
/// every independent source's drive `u` as its deviation `u - u#op`. The
/// coefficients are frozen at the operating point. A small-signal model has
/// no switching surfaces, Newton aids or waveforms: those it drops.
pub fn linearize(ctx: &mut Graph, dae: &Dae) -> Dae {
    let freeze = freeze_op_point(ctx, dae);
    let zero = ctx.zero();
    let generators = dae.observers.generators(ctx);
    // a coefficient reads every generator at zero, as in every evaluation
    let mut frozen = freeze.clone();
    frozen.extend(generators.iter().map(|&g| (g, zero)));
    let (g, c) = dae.jacobian_iq_coo(ctx);
    let mut terms: [Vec<Vec<ExprId>>; 2] =
        [vec![Vec::new(); dae.dim()], vec![Vec::new(); dae.dim()]];
    for (k, (rows, cols, coefs)) in [g, c].into_iter().enumerate() {
        let coefs = rsdag::substitute(ctx, &coefs, &frozen);
        for ((&row, &col), coef) in rows.iter().zip(&cols).zip(coefs) {
            let xj = ctx.symbol_expr(dae.x[col]);
            terms[k][row].push(ctx.mul(coef, xj));
        }
    }
    // the small signal's drives: the histories, the generators, the sources
    // (each as its deviation from the operating point)
    let mut drives: Vec<(SymbolId, ExprId)> = Vec::new();
    for s in dae
        .delays
        .iter()
        .map(|d| d.hist)
        .chain(generators.iter().copied())
    {
        drives.push((s, ctx.symbol_expr(s)));
    }
    let params = dae.params(ctx);
    for name in &dae.source_names {
        let value = sane_circuit::value_symbol_name(name);
        let prefix = format!("{name}.");
        for &p in &params {
            let pn = ctx.symbol_name(p).to_string();
            let drive = pn == value
                || pn
                    .strip_prefix(&prefix)
                    .is_some_and(sane_circuit::SourceFn::is_level);
            if drive {
                let u = ctx.symbol_expr(p);
                let op = ctx.sym(&format!("{pn}#op"));
                drives.push((p, ctx.sub(u, op)));
            }
        }
    }
    let wrt: Vec<SymbolId> = drives.iter().map(|&(s, _)| s).collect();
    for (row, entries) in rsdag::sparse_jacobian(ctx, &dae.currents, &wrt)
        .into_iter()
        .enumerate()
    {
        for (k, coef) in entries {
            let coef = rsdag::substitute(ctx, &[coef], &frozen)[0];
            terms[0][row].push(ctx.mul(coef, drives[k].1));
        }
    }
    let [currents, charges] = terms.map(|rows| {
        (rows.into_iter())
            .map(|t| ctx.reduce(rsdag::ReduceOp::Sum, t))
            .collect::<Vec<_>>()
    });
    // each delay's signal, linearized
    let srcs: Vec<ExprId> = dae.delays.iter().map(|d| d.src).collect();
    let src_lin: Vec<ExprId> = (rsdag::sparse_jacobian(ctx, &srcs, &dae.x).into_iter())
        .map(|entries| {
            let t: Vec<ExprId> = (entries.into_iter())
                .map(|(j, coef)| {
                    let coef = rsdag::substitute(ctx, &[coef], &frozen)[0];
                    let xj = ctx.symbol_expr(dae.x[j]);
                    ctx.mul(coef, xj)
                })
                .collect();
            ctx.reduce(rsdag::ReduceOp::Sum, t)
        })
        .collect();

    // the observers frozen at the operating point, the rest as rewritten
    let mut lin = dae.rewrite(
        ctx,
        crate::rewrite::Rewrite {
            rows: Some((currents, charges)),
            keep: (0..dae.dim()).collect(),
            n_nodes: dae.n_nodes,
            subst: freeze,
        },
    );
    for (d, src) in lin.delays.iter_mut().zip(src_lin) {
        d.src = src;
    }
    lin.events.clear();
    lin.limits.clear();
    lin.companion.clear();
    lin.dc_seeds.clear();
    lin.sources.clear();
    lin.source_names.clear();
    lin
}

/// Shared verification helpers for the small-signal transforms (used by both the
/// linearisation and the canonical-network tests): reference circuits and a
/// multi-point numeric matrix-equality check that is robust to the associativity
/// of the assembled sums.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;
    use crate::DeviceInstance;
    use rsdag::eval;
    use sane_circuit::Elements;
    use std::collections::HashMap;

    /// A parallel RLC tank driven by a current source: a genuine second-order
    /// linear DAE (node KCL with the capacitor's charge `C*v`, plus the
    /// inductor's branch constraint `v - d/dt (L*i_L)`).
    pub(crate) fn rlc(ctx: &mut Graph) -> Dae {
        let mut c = Elements::new();
        c.current_source("I1", 0, 1)
            .resistor("R1", 1, 0)
            .inductor("L1", 1, 0)
            .capacitor("C1", 1, 0);
        crate::assemble(ctx, &sane_circuit::Circuit::flat(&c, &[])).unwrap()
    }

    /// A diode-loaded RC node: a nonlinear DAE whose small-signal entries are the
    /// junction conductance and capacitance.
    pub(crate) fn diode_rc(ctx: &mut Graph) -> Dae {
        let mut c = Elements::new();
        c.voltage_source("V1", 2, 0)
            .resistor("R1", 1, 2)
            .capacitor("C1", 1, 0);
        let devs = vec![DeviceInstance::new(
            std::sync::Arc::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
            vec![1, 0],
        )];
        crate::assemble(ctx, &sane_circuit::Circuit::flat(&c, &devs)).unwrap()
    }

    /// Deterministic pseudo-random value for a symbol name (FNV-1a of the
    /// name, salted per evaluation point) -- lets the verification probe `G` and
    /// `C` at several points without `Math::random`.
    fn pseudo(name: &str, salt: u64) -> f64 {
        let mut h = 0xcbf29ce484222325u64 ^ salt;
        for b in name.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        // Kept on the thermal-voltage scale (~0.02..0.05) so a diode/MOSFET
        // exponent v/(N*Vt) stays modest and the entries evaluate finite.
        (h % 30) as f64 / 1000.0 + 0.02
    }

    /// Bind every free symbol of `mats` for evaluation: an operating-point freeze
    /// symbol `"X#op"` gets the *same* value as its base `"X"`, so evaluating
    /// frozen and unfrozen matrices at one point is a like-for-like comparison.
    fn env_for(ctx: &Graph, mats: &[&[Vec<ExprId>]], salt: u64) -> HashMap<SymbolId, f64> {
        use sane_core::constants::{TEMP_NOMINAL_K, TEMP_SYMBOL};
        let mut env = HashMap::new();
        for mat in mats {
            for row in *mat {
                for &e in row {
                    for s in ctx.free_symbols(e) {
                        let name = ctx.symbol_name(s);
                        let base = name.strip_suffix("#op").unwrap_or(name);
                        // Temperature symbols must stay physical (~300 K): a random
                        // ~0.02 would make the thermal voltage tiny and overflow the
                        // diode's exp temperature scaling.
                        let val =
                            if base == TEMP_SYMBOL || base.to_ascii_lowercase().ends_with("tnom") {
                                TEMP_NOMINAL_K
                            } else {
                                pseudo(base, salt)
                            };
                        env.insert(s, val);
                    }
                }
            }
        }
        env
    }

    /// Two small-signal matrices are equal as operators: probed at several random
    /// operating points (frozen `#op` symbols pinned to their base unknown's
    /// value), robust to the associativity of the assembled sums.
    pub(crate) fn assert_matrix_eq(ctx: &mut Graph, a: &[Vec<ExprId>], b: &[Vec<ExprId>]) {
        let n = a.len();
        assert_eq!(n, b.len(), "dimension mismatch");
        for salt in [1u64, 7, 31, 131] {
            let refs: [&[Vec<ExprId>]; 2] = [a, b];
            let env = env_for(ctx, &refs, salt);
            for i in 0..n {
                for j in 0..n {
                    let lhs = eval(ctx, &[a[i][j]], &env)[0];
                    let rhs = eval(ctx, &[b[i][j]], &env)[0];
                    let tol = 1e-9 * (1.0 + lhs.abs());
                    assert!(
                        (lhs - rhs).abs() <= tol,
                        "[{i}][{j}] mismatch at salt {salt}: {lhs} vs {rhs}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::{assert_matrix_eq, diode_rc, rlc};
    use super::*;

    /// `G = dI/dx` and `C = dQ/dx`, dense.
    fn gc(ctx: &mut Graph, dae: &Dae) -> [Vec<Vec<ExprId>>; 2] {
        let n = dae.dim();
        let zero = ctx.zero();
        let (g, c) = dae.jacobian_iq_coo(ctx);
        [g, c].map(|(rows, cols, exprs)| {
            let mut m = vec![vec![zero; n]; n];
            for ((r, c), e) in rows.into_iter().zip(cols).zip(exprs) {
                m[r][c] = e;
            }
            m
        })
    }

    /// The linearised DAE's `G` and `C` reproduce the original DAE's (the
    /// matrix assembly carried into the linear graph), with the dimension
    /// preserved.
    fn assert_reassembles(ctx: &mut Graph, dae: &Dae) {
        let orig = gc(ctx, dae);
        let lin = linearize(ctx, dae);
        assert_eq!(dae.dim(), lin.dim(), "dimension preserved");
        let lin = gc(ctx, &lin);
        for (a, b) in orig.iter().zip(&lin) {
            assert_matrix_eq(ctx, a, b);
        }
    }

    #[test]
    fn linear_rlc_reassembles() {
        let mut ctx = Graph::new();
        let dae = rlc(&mut ctx);
        assert_reassembles(&mut ctx, &dae);
    }

    #[test]
    fn nonlinear_diode_reassembles() {
        let mut ctx = Graph::new();
        let dae = diode_rc(&mut ctx);
        assert_reassembles(&mut ctx, &dae);
    }

    /// The linearised rows are genuinely linear: differentiating row `i`
    /// w.r.t. unknown `j` yields a coefficient free of every perturbation
    /// unknown (the hallmark of a constant-coefficient linear DAE).
    #[test]
    fn coefficients_are_constant() {
        let mut ctx = Graph::new();
        let dae = diode_rc(&mut ctx);
        let lin = linearize(&mut ctx, &dae);
        let perturbations: std::collections::BTreeSet<SymbolId> = lin.x.iter().copied().collect();
        for mat in gc(&mut ctx, &lin) {
            for row in mat {
                for entry in row {
                    let fs = ctx.free_symbols(entry);
                    assert!(
                        fs.is_disjoint(&perturbations),
                        "linear coefficient still depends on a perturbation unknown"
                    );
                }
            }
        }
    }
}
