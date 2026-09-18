//! The fitted macromodel produced by the vector fit: a common pole set with per-entry residues
//! and const/proportional terms. Evaluated at `s = jω` (normalised) to reconstruct the response.

use super::C;

/// Fitted model: common `poles` (expanded, incl. both members of each c.c. pair), per-entry
/// `res[e]` residues, and per-entry constant `cst` / proportional `dif` (s·h) terms.
pub struct VfModel {
    pub(super) poles: Vec<C>,
    pub(super) res: Vec<Vec<C>>,
    pub(super) cst: Vec<C>,
    pub(super) dif: Vec<C>,
    /// Per-entry √s (skin-effect branch) coefficient; all-zero unless the fit
    /// ran with the sqrt term enabled ([`crate::fit_auto_with`]).
    pub(super) sqt: Vec<C>,
    pub(super) d: usize,
}

impl VfModel {
    pub fn n_support(&self) -> usize {
        self.poles.len() + 1
    }
    pub fn eval(&self, s: C) -> Vec<C> {
        (0..self.d)
            .map(|e| {
                let mut v = self.cst[e] + s * self.dif[e] + s.sqrt() * self.sqt[e];
                for (p, r) in self.poles.iter().zip(&self.res[e]) {
                    v += r / (s - p);
                }
                v
            })
            .collect()
    }
}
