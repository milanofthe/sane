//! Linearisation transform: the small-signal *linear mass-matrix DAE*.
//!
//! The nonlinear system `F(x, x', t) = 0` is rewritten into a linear DAE
//!
//! ```text
//!     G dx + C dx' = 0,    G = dF/dx,   C = dF/dx'
//! ```
//!
//! represented in the *same* graph and the *same* [`Dae`] abstraction: the
//! residual rows are linear forms in the perturbation unknowns `dx`, whose
//! coefficients are the operating-point Jacobians. The bias unknowns inside
//! those coefficients are *frozen* to constant operating-point symbols
//! (`name#op`), so the system is genuinely linear in `dx` -- differentiating a
//! linearised residual recovers `G`/`C` exactly rather than `G + (dG/dx) x`.
//!
//! This is the matrix assembly `A(s) = G + sC` living in the graph exactly the
//! way the nonlinear residual assembly does: [`crate::small_signal_matrix`] on
//! the linearised DAE reproduces `A(s)` node-for-node (the verification test).
//! Mapping the per-element stamps to canonical R/C/L and controlled sources is
//! then a further read of this linear DAE, not a separate representation.

use rustc_hash::FxHashMap;

use rsdag::{differentiate, ExprId, SymbolId};
use sane_core::Graph;

use crate::stamp::Stamp;
use crate::Dae;

/// The operating-point freeze map: every unknown symbol `x_j` and its derivative
/// symbol `x'_j` is mapped to a fresh constant symbol named `"{name}#op"`. Applied
/// to a Jacobian entry, it turns the bias unknowns into operating-point constants
/// so the resulting coefficient is constant w.r.t. the perturbation unknowns.
pub fn freeze_op_point(ctx: &mut Graph, dae: &Dae) -> FxHashMap<SymbolId, ExprId> {
    let mut map: FxHashMap<SymbolId, ExprId> = FxHashMap::default();
    for &xs in &dae.x {
        let fname = format!("{}#op", ctx.symbol_name(xs));
        let fe = ctx.sym(&fname);
        map.insert(xs, fe);
    }
    for &xds in dae.xdot.iter().flatten() {
        let fname = format!("{}#op", ctx.symbol_name(xds));
        let fe = ctx.sym(&fname);
        map.insert(xds, fe);
    }
    map
}

/// How finely the linearised DAE's stamps are split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Granularity {
    /// One stamp per source-element contribution (the linearised image of each
    /// original stamp). Mirrors the nonlinear DAE's structure.
    #[default]
    PerElement,
    /// One stamp per canonical matrix entry: each stamp is a single
    /// `coefficient * port` (instantaneous) or `coefficient * port'` (reactive)
    /// monomial. The canonical small-signal elements then live *directly in the
    /// graph* as the DAE's stamps -- no separate structure needed to hold them.
    Canonical,
}

/// Linearise the DAE about its operating point into the linear mass-matrix DAE
/// `G dx + C dx' = 0`, with per-element stamps ([`Granularity::PerElement`]). See
/// [`linearize_with`].
pub fn linearize(ctx: &mut Graph, dae: &Dae) -> Dae {
    linearize_with(ctx, dae, Granularity::PerElement)
}

/// Linearise the DAE about its operating point into the linear mass-matrix DAE
/// `G dx + C dx' = 0` (see the module docs). The returned [`Dae`] keeps the same
/// unknowns, derivative symbols and time symbol; its residuals are linear forms.
/// The stamps are the small-signal contributions -- grouped per source element
/// ([`Granularity::PerElement`]) or split to one canonical element each
/// ([`Granularity::Canonical`]). Either way, summing a row's stamps reproduces
/// that row's residual exactly.
pub fn linearize_with(ctx: &mut Graph, dae: &Dae, gran: Granularity) -> Dae {
    let freeze = freeze_op_point(ctx, dae);

    // Frozen perturbation-expression for each unknown / derivative (the dx_j and
    // dx'_j the linear forms multiply -- the original symbols, now meaning phasor
    // perturbations, since the bias copies inside the coefficients were frozen).
    let x_expr: Vec<ExprId> = dae.x.iter().map(|&s| ctx.symbol_expr(s)).collect();
    let xdot_expr: Vec<Option<ExprId>> = dae
        .xdot
        .iter()
        .map(|opt| opt.map(|s| ctx.symbol_expr(s)))
        .collect();

    // Linearise each source stamp into its `coef*port` monomials. The residual
    // rows fall out by summing the monomials; the stamps are either one summed
    // stamp per source element or one stamp per monomial (canonical elements).
    let mut stamps = Vec::with_capacity(dae.stamps.len());
    let n = dae.dim();
    let zero = ctx.zero();
    let mut residuals = vec![zero; n];
    for st in &dae.stamps {
        let terms = linear_terms(ctx, st.expr, dae, &freeze, &x_expr, &xdot_expr);
        match gran {
            Granularity::PerElement => {
                let mut acc = zero;
                for t in &terms {
                    acc = ctx.add(acc, *t);
                }
                residuals[st.row] = ctx.add(residuals[st.row], acc);
                stamps.push(Stamp::new(st.row, acc));
            }
            Granularity::Canonical => {
                for t in terms {
                    residuals[st.row] = ctx.add(residuals[st.row], t);
                    stamps.push(Stamp::new(st.row, t));
                }
            }
        }
    }

    Dae {
        n_nodes: dae.n_nodes,
        param_defaults: dae.param_defaults.clone(),
        residuals,
        events: Vec::new(),
        delays: dae.delays.clone(),
        unknowns: dae.unknowns.clone(),
        kinds: dae.kinds.clone(),
        x: dae.x.clone(),
        xdot: dae.xdot.clone(),
        t: dae.t,
        stamps,
        // A linear DAE carries no homotopy companion network, device limits or
        // (re-derived) noise sources; those stay with the nonlinear DAE the
        // operating point was solved on. Source shapes (transient breakpoints / HB
        // fundamental) likewise belong to the time-domain DAE, not this AC form.
        companion: Vec::new(),
        noise_sources: Vec::new(),
        op_vars: Vec::new(),
        dc_seeds: Vec::new(),
        limits: Vec::new(),
        sources: Vec::new(),
        source_names: Vec::new(),
    }
}

/// The monomials of the first-order Taylor of one expression about the operating
/// point: each nonzero `(dexpr/dx_j)|_op dx_j` and `(dexpr/dx'_j)|_op dx'_j`,
/// where `|_op` freezes the bias unknowns to their operating-point constants.
/// Each returned term is one `coefficient * port` canonical element. The constant
/// term `expr|_op` is dropped (the operating-point residual, zero by KCL).
fn linear_terms(
    ctx: &mut Graph,
    expr: ExprId,
    dae: &Dae,
    freeze: &FxHashMap<SymbolId, ExprId>,
    x_expr: &[ExprId],
    xdot_expr: &[Option<ExprId>],
) -> Vec<ExprId> {
    let fs = ctx.free_symbols(expr);
    let mut terms = Vec::new();
    for (j, &xs) in dae.x.iter().enumerate() {
        if fs.contains(&xs) {
            let d = differentiate(ctx, expr, xs);
            let df = rsdag::substitute(ctx, &[d], freeze)[0];
            let term = ctx.mul(df, x_expr[j]);
            if !ctx.is_zero(term) {
                terms.push(term);
            }
        }
    }
    for (j, opt) in dae.xdot.iter().enumerate() {
        if let (Some(xds), Some(xe)) = (opt, xdot_expr[j]) {
            if fs.contains(xds) {
                let d = differentiate(ctx, expr, *xds);
                let df = rsdag::substitute(ctx, &[d], freeze)[0];
                let term = ctx.mul(df, xe);
                if !ctx.is_zero(term) {
                    terms.push(term);
                }
            }
        }
    }
    terms
}

/// Shared verification helpers for the small-signal transforms (used by both the
/// linearisation and the canonical-network tests): reference circuits and a
/// multi-point numeric matrix-equality check that is robust to the associativity
/// of the assembled sums.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;
    use crate::{assemble_dae, DeviceInstance};
    use num_complex::Complex64;
    use rsdag::eval;
    use sane_mna::Circuit;
    use std::collections::HashMap;

    /// A parallel RLC tank driven by a current source: a genuine second-order
    /// linear DAE (node KCL with the capacitor's `C*vdot`, plus the inductor's
    /// branch constraint `v - L*idot_L`).
    pub(crate) fn rlc(ctx: &mut Graph) -> Dae {
        let mut c = Circuit::new();
        c.current_source("I1", 0, 1)
            .resistor("R1", 1, 0)
            .inductor("L1", 1, 0)
            .capacitor("C1", 1, 0);
        assemble_dae(ctx, &c, &[])
    }

    /// A diode-loaded RC node: a nonlinear DAE whose small-signal entries are the
    /// junction conductance and capacitance.
    pub(crate) fn diode_rc(ctx: &mut Graph) -> Dae {
        let mut c = Circuit::new();
        c.voltage_source("V1", 2, 0)
            .resistor("R1", 1, 2)
            .capacitor("C1", 1, 0);
        let devs = vec![DeviceInstance::new(
            Box::new(sane_veriloga::builtin_device("sane_diode", "D1", &[])),
            vec![1, 0],
        )];
        assemble_dae(ctx, &c, &devs)
    }

    /// Deterministic pseudo-random complex value for a symbol name (FNV-1a of the
    /// name, salted per evaluation point) -- lets the verification probe `A(s)` at
    /// several points without `Math::random`.
    fn pseudo(name: &str, salt: u64) -> Complex64 {
        let mut h = 0xcbf29ce484222325u64 ^ salt;
        for b in name.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        // Kept on the thermal-voltage scale (~0.02..0.05) so a diode/MOSFET
        // exponent v/(N*Vt) stays modest and the entries evaluate finite.
        let re = (h % 30) as f64 / 1000.0 + 0.02;
        let im = ((h >> 17) % 30) as f64 / 1000.0 + 0.005;
        Complex64::new(re, im)
    }

    /// Bind every free symbol of `mats` for evaluation: an operating-point freeze
    /// symbol `"X#op"` gets the *same* value as its base `"X"`, so evaluating
    /// frozen and unfrozen matrices at one point is a like-for-like comparison.
    fn env_for(ctx: &Graph, mats: &[&[Vec<ExprId>]], salt: u64) -> HashMap<SymbolId, Complex64> {
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
                                Complex64::new(TEMP_NOMINAL_K, 0.0)
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
                    let tol = 1e-9 * (1.0 + lhs.norm());
                    assert!(
                        (lhs - rhs).norm() <= tol,
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
    use crate::small_signal_matrix;

    /// The linearised DAE's `A(s)` reproduces the original DAE's `A(s)` (the matrix
    /// assembly carried into the linear graph), with the dimension preserved -- at
    /// both stamp granularities (canonical splitting must not change the operator).
    fn assert_reassembles(ctx: &mut Graph, dae: &Dae) {
        let a_orig = small_signal_matrix(ctx, dae);
        for gran in [Granularity::PerElement, Granularity::Canonical] {
            let lin = linearize_with(ctx, dae, gran);
            assert_eq!(dae.dim(), lin.dim(), "dimension preserved");
            let a_lin = small_signal_matrix(ctx, &lin);
            assert_matrix_eq(ctx, &a_orig, &a_lin);
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

    /// With canonical granularity each stamp is a single `coef * port` monomial:
    /// exactly one perturbation port appears in it. This is the canonical element
    /// living directly in the graph.
    #[test]
    fn canonical_stamps_are_atomic() {
        let mut ctx = Graph::new();
        let dae = rlc(&mut ctx);
        let lin = linearize_with(&mut ctx, &dae, Granularity::Canonical);
        let ports: std::collections::BTreeSet<SymbolId> = lin
            .x
            .iter()
            .copied()
            .chain(lin.xdot.iter().flatten().copied())
            .collect();
        for st in &lin.stamps {
            let touched = ctx.free_symbols(st.expr).intersection(&ports).count();
            assert_eq!(touched, 1, "a canonical stamp touches exactly one port");
        }
    }

    /// The linearised residuals are genuinely linear: differentiating row `i`
    /// w.r.t. unknown `j` yields a coefficient free of every perturbation unknown
    /// (the hallmark of a constant-coefficient linear DAE).
    #[test]
    fn coefficients_are_constant() {
        let mut ctx = Graph::new();
        let dae = diode_rc(&mut ctx);
        let lin = linearize(&mut ctx, &dae);
        let g = lin.jacobian_x(&mut ctx);
        let c = lin.jacobian_xdot(&mut ctx);
        let perturbations: std::collections::BTreeSet<SymbolId> = lin
            .x
            .iter()
            .copied()
            .chain(lin.xdot.iter().flatten().copied())
            .collect();
        for mat in [&g, &c] {
            for row in mat {
                for &entry in row {
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
