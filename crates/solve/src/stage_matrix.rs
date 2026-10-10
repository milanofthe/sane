//! The stage matrix of the transient methods, `G + α·C + gmin·I`, over one
//! pattern fixed for the integration: the union of the Jacobian patterns of
//! `G = dI/dx` and `C = dQ/dx` and the full diagonal, analyzed once (see
//! [`CompiledDc::stage_symbolic`]).
//!
//! An assembly takes the values of `G` and `C` an evaluation computed and
//! keeps them: every factorization after it (a new `α`, the step size moved)
//! scatters them straight into the factor's values and refactors in place,
//! KLU's numeric-only refactor over the frozen pivot sequence or the
//! supernodal refactor over the shared analysis. The kept values serve the
//! products with `C` and the rounding magnitudes of the rows, which belong to
//! the same matrix the solves invert. Nothing allocates once the first
//! assembly has sized the buffers.

use sane_core::constants::GMIN_DC;

use crate::{sparse, CompiledDc, Symbolic};

pub(crate) struct StageMatrix<'a> {
    cdc: &'a CompiledDc,
    fac: sparse::Refactorable<'a>,
    /// `G` and `C` of the last assembly, in their Jacobian patterns.
    g: Vec<f64>,
    c: Vec<f64>,
}

impl<'a> StageMatrix<'a> {
    /// The stage matrix of `cdc` on the stage pattern's analysis `sym`, not
    /// yet assembled.
    pub fn new(cdc: &'a CompiledDc, sym: &'a Symbolic) -> Self {
        StageMatrix {
            cdc,
            fac: sym.pattern.factorizer(),
            g: vec![0.0; cdc.nnz_x],
            c: vec![0.0; cdc.jxd_rows.len()],
        }
    }

    /// Take `G` and `C` and factor `G + α·C + gmin·I`; `false` when singular.
    pub fn assemble(&mut self, g: &[f64], c: &[f64], alpha: f64) -> bool {
        self.g.copy_from_slice(g);
        self.c.copy_from_slice(c);
        self.factor(alpha)
    }

    /// Factor `G + α·C + gmin·I` over the kept `G` and `C`; `false` when
    /// singular.
    pub fn factor(&mut self, alpha: f64) -> bool {
        let row_scaling = self.cdc.tricks.row_equilibration;
        self.fac
            .factor_scaled(&[(&self.g, 1.0), (&self.c, alpha)], GMIN_DC, row_scaling)
    }

    /// `out = (G + α·C + gmin·I)⁻¹ rhs` on the last factorization; `false`
    /// without one.
    pub fn solve(&mut self, rhs: &[f64], out: &mut [f64]) -> bool {
        self.fac.solve_into(rhs, out)
    }

    /// The kept `G` and `C`, in their Jacobian patterns.
    pub fn g_values(&self) -> &[f64] {
        &self.g
    }
    pub fn c_values(&self) -> &[f64] {
        &self.c
    }

    /// `out = C v` with the kept `C`.
    pub fn c_mul(&self, v: &[f64], out: &mut [f64]) {
        let cdc = self.cdc;
        out.fill(0.0);
        for (k, &c) in self.c.iter().enumerate() {
            out[cdc.jxd_rows[k]] += c * v[cdc.jxd_cols[k]];
        }
    }

    /// Per row, `|G| |x| + |C| |x| / hγ` over the kept `G` and `C`: the size
    /// of the terms a stage residual sums, which bounds its rounding.
    pub fn row_magnitudes(&self, x: &[f64], hg: f64, out: &mut [f64]) {
        let cdc = self.cdc;
        out.fill(0.0);
        for (k, &g) in self.g.iter().enumerate() {
            out[cdc.jx_rows[k]] += g.abs() * x[cdc.jx_cols[k]].abs();
        }
        for (k, &c) in self.c.iter().enumerate() {
            out[cdc.jxd_rows[k]] += c.abs() * x[cdc.jxd_cols[k]].abs() / hg;
        }
    }
}
