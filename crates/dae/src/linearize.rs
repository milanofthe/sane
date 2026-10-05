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

use rustc_hash::FxHashMap;

use rsdag::{ExprId, SymbolId};
use sane_core::Graph;

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

/// Linearise the DAE about its operating point into the linear mass-matrix DAE
/// `G dx + C dx' = 0` (see the module docs). The returned [`Dae`] keeps the same
/// unknowns, derivative symbols and time symbol; row `i` of its residuals is
/// `sum_j G_ij x_j + C_ij x'_j`, the Jacobians frozen at the operating point.
pub fn linearize(ctx: &mut Graph, dae: &Dae) -> Dae {
    let freeze = freeze_op_point(ctx, dae);
    let (jx, jxd) = dae.jacobian_x_xdot_coo(ctx);
    let coefs: Vec<ExprId> = jx.2.iter().chain(&jxd.2).copied().collect();
    let frozen = rsdag::substitute(ctx, &coefs, &freeze);
    let ports = jx.1.iter().map(|&j| dae.x[j]).chain(
        jxd.1
            .iter()
            .map(|&j| dae.xdot[j].expect("a differential unknown")),
    );
    let mut terms: Vec<Vec<ExprId>> = vec![Vec::new(); dae.dim()];
    for ((&row, port), coef) in jx.0.iter().chain(&jxd.0).zip(ports).zip(frozen) {
        let p = ctx.symbol_expr(port);
        terms[row].push(ctx.mul(coef, p));
    }
    let residuals = terms
        .into_iter()
        .map(|t| ctx.reduce(rsdag::ReduceOp::Sum, t))
        .collect();

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
        labels: dae.labels.clone(),
    }
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
    /// assembly carried into the linear graph), with the dimension preserved.
    fn assert_reassembles(ctx: &mut Graph, dae: &Dae) {
        let a_orig = small_signal_matrix(ctx, dae);
        let lin = linearize(ctx, dae);
        assert_eq!(dae.dim(), lin.dim(), "dimension preserved");
        let a_lin = small_signal_matrix(ctx, &lin);
        assert_matrix_eq(ctx, &a_orig, &a_lin);
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
