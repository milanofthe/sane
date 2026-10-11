//! The DC system at a binding as the Newton core solves it: the shunt it
//! adds to `F(x)` and the evaluations and factorizations behind it.

use super::*;

/// What a DC Newton adds to `F(x)` to make its system: a shunt to ground,
/// pin springs, or the companion homotopy.
pub(super) enum Shunt {
    /// `gmin x` on every unknown.
    Uniform(f64),
    /// `diag x - anchor`: `GMIN_DC` on every unknown and a spring
    /// `g_k (x_k - t_k)` on each pinned one (the node-set phase, the
    /// pseudo-transient anchor).
    Pinned { diag: Vec<f64>, anchor: Vec<f64> },
    /// The companion homotopy at `lambda`: `(1 - lambda) (G_comp x +
    /// GMIN_START x)` on the node rows; `GMIN_DC` regularizes the matrix only.
    Companion(f64),
}

/// The DC system at a binding as the Newton core solves it (see
/// [`newton::System`]): `F(x)` and its Jacobian from the binding's episodes
/// (the step tape for both, the residual tape for the probes), what the
/// [`Shunt`] adds, the reused symbolic pattern refactored in place (or the
/// cached Schur partition of a large mostly-linear system, or a one-shot
/// triplet LU on a degenerate pattern), and the rounding the residual rows
/// sum.
pub(super) struct Dc<'a, 'b> {
    pub(super) cdc: &'a CompiledDc,
    pub(super) p: &'a Binding<'b>,
    pub(super) shunt: Shunt,
    pub(super) tapes: std::cell::RefMut<'a, Episodes>,
    pub(super) valbuf: Vec<f64>,
    pub(super) fac: Option<sparse::Refactorable<'a>>,
    /// The cached linear block of the partition, built at the first factor.
    pub(super) lin: Option<LinCache>,
    pub(super) partition: bool,
    pub(super) terms: Vec<f64>,
    /// Load the weak diagonals of the matrix (the node-adaptive fallback):
    /// each up to `ADAPT_DIAG_FRAC` of the strongest, decaying with the
    /// residual norm.
    pub(super) load_weak: bool,
    /// Grant the residual half where the SPICE-relative test holds (see
    /// [`CompiledDc::residual_relative_ok`]).
    pub(super) relative: Option<Convergence>,
    /// The residual norm at the last Jacobian evaluation.
    pub(super) fnorm: f64,
    pub(super) iscale: (Vec<f64>, Vec<f64>),
}

impl<'a, 'b> Dc<'a, 'b> {
    pub(super) fn new(
        cdc: &'a CompiledDc,
        p: &'a Binding<'b>,
        shunt: Shunt,
        tricks: SolverTricks,
    ) -> Self {
        let partition =
            matches!(shunt, Shunt::Uniform(_)) && cdc.dc.partition.is_some() && tricks.partition;
        Dc {
            cdc,
            p,
            shunt,
            tapes: p.tapes.borrow_mut(),
            valbuf: Vec::new(),
            fac: None,
            lin: None,
            partition,
            terms: Vec::new(),
            load_weak: false,
            relative: None,
            fnorm: f64::INFINITY,
            iscale: (Vec::new(), Vec::new()),
        }
    }

    /// The node-adaptive form: weak diagonals loaded, the relative test
    /// granted, no partition.
    pub(super) fn adaptive(mut self, conv: Convergence) -> Self {
        self.load_weak = true;
        self.relative = Some(conv);
        self.partition = false;
        self
    }

    /// The Jacobian nonzeros of the last Jacobian evaluation.
    pub(super) fn jac(&self) -> &[f64] {
        &self.tapes.step.out[self.cdc.n..]
    }

    /// The diagonal the shunt puts on the matrix, row `i`.
    pub(super) fn diag(&self, i: usize) -> f64 {
        match &self.shunt {
            Shunt::Uniform(g) => *g,
            Shunt::Pinned { diag, .. } => diag[i],
            Shunt::Companion(l) => {
                let base = if self.cdc.is_node(i) {
                    (1.0 - l) * GMIN_START
                } else {
                    0.0
                };
                base + GMIN_DC
            }
        }
    }

    /// The companion entries `(row, col, value)` the homotopy puts on the
    /// matrix (none outside it).
    pub(super) fn extra(&self) -> Vec<(usize, usize, f64)> {
        match self.shunt {
            Shunt::Companion(l) => (self.cdc.dc.companion.iter())
                .map(|&(r, c, g)| (r, c, (1.0 - l) * g))
                .collect(),
            _ => Vec::new(),
        }
    }
}

impl newton::System<f64> for Dc<'_, '_> {
    fn eval(&mut self, x: &[f64], jacobian: bool, res: &mut [f64]) -> bool {
        let (cdc, p) = (self.cdc, self.p);
        let tb = match jacobian {
            true => &mut self.tapes.step,
            false => &mut self.tapes.res,
        };
        let tape = if jacobian {
            &cdc.tape_step_dc
        } else {
            &cdc.tape_res_dc
        };
        cdc.eval_episode(tape, x, p, tb);
        let f = &tb.out;
        match &self.shunt {
            Shunt::Uniform(g) => {
                for (i, r) in res.iter_mut().enumerate() {
                    *r = f[i] + g * x[i];
                }
            }
            Shunt::Pinned { diag, anchor } => {
                for (i, r) in res.iter_mut().enumerate() {
                    *r = f[i] + diag[i] * x[i] - anchor[i];
                }
            }
            Shunt::Companion(l) => {
                let s = 1.0 - l;
                for (i, r) in res.iter_mut().enumerate() {
                    *r = f[i]
                        + if cdc.is_node(i) {
                            s * GMIN_START * x[i]
                        } else {
                            0.0
                        };
                }
                for &(r, c, g) in &cdc.dc.companion {
                    res[r] += s * g * x[c];
                }
            }
        }
        if jacobian {
            self.fnorm = norm2(res);
        }
        true
    }

    fn factor(&mut self) -> bool {
        let cdc = self.cdc;
        let n = cdc.n;
        if self.partition {
            if let Shunt::Uniform(g) = self.shunt {
                if self.lin.is_none() {
                    let part = cdc.dc.partition.as_ref().unwrap();
                    self.lin = cdc.build_lin_cache(part, self.jac(), g);
                }
            }
            if self.lin.is_some() {
                return true;
            }
            self.partition = false; // degenerate: the full LU
        }
        let companion = matches!(self.shunt, Shunt::Companion(_));
        let sym = match companion {
            true => cdc.companion_symbolic(),
            false => cdc.dc.symbolic.as_ref(),
        };
        let Some(sym) = sym else {
            return true; // the triplet fallback factors as it solves
        };
        // values: the Jacobian nonzeros, the companion entries, the diagonal
        let mut vals = std::mem::take(&mut self.valbuf);
        vals.clear();
        vals.extend_from_slice(self.jac());
        if self.load_weak {
            let out = &self.tapes.step.out;
            let maxd = (cdc.diag_idx.iter().flatten())
                .map(|&di| out[n + di].abs())
                .fold(0.0f64, f64::max);
            // strong far from the solution, vanishing as the residual does
            let floor = ADAPT_DIAG_FRAC * maxd * self.fnorm.min(1.0);
            for &di in cdc.diag_idx.iter().flatten() {
                let jii = vals[di].abs();
                if jii < floor {
                    vals[di] += floor - jii;
                }
            }
        }
        vals.extend(self.extra().into_iter().map(|(_, _, v)| v));
        vals.extend((0..n).map(|i| self.diag(i)));
        let f = self.fac.get_or_insert_with(|| sym.pattern.factorizer());
        let ok = f.factor(&vals, cdc.tricks.row_equilibration);
        self.valbuf = vals;
        ok
    }

    fn solve(&mut self, rhs: &[f64], dx: &mut [f64]) -> bool {
        let cdc = self.cdc;
        if self.partition {
            let (part, lin) = (
                cdc.dc.partition.as_ref().unwrap(),
                self.lin.as_ref().unwrap(),
            );
            return (cdc.solve_partitioned(lin, part, self.jac(), rhs))
                .map(|d| dx.copy_from_slice(&d))
                .is_some();
        }
        if let Some(f) = &mut self.fac {
            return f.solve_into(rhs, dx);
        }
        let diag: Vec<f64> = (0..cdc.n).map(|i| self.diag(i)).collect();
        (cdc.solve_triplets(self.jac(), &self.extra(), &diag, rhs))
            .map(|d| dx.copy_from_slice(&d))
            .is_some()
    }

    fn probe(&mut self, rhs: &[f64], dx: &mut [f64]) -> bool {
        match (&mut self.fac, self.partition) {
            (Some(f), false) => f.solve_into(rhs, dx),
            _ => false,
        }
    }

    fn chord(&mut self, rhs: &[f64], dx: &mut [f64]) -> bool {
        self.probe(rhs, dx)
    }

    fn rounding(&mut self, x: &[f64], _res: &[f64]) -> (bool, f64) {
        let n = self.cdc.n;
        let out = &self.tapes.step.out;
        match self.shunt {
            Shunt::Uniform(g) => self
                .cdc
                .rounding(&out[..n], &out[n..], x, g, &mut self.terms),
            // the rounding of `F + GMIN_DC x`: a floor for the line search,
            // no acceptance (the spring rows sum terms it does not count)
            Shunt::Pinned { .. } => (
                false,
                self.cdc
                    .rounding(&out[..n], &out[n..], x, GMIN_DC, &mut self.terms)
                    .1,
            ),
            // waypoints take the residual half alone, which the rounding
            // never keeps from passing
            Shunt::Companion(_) => (false, 0.0),
        }
    }

    fn accept_residual(&mut self, x: &[f64], _res: &[f64]) -> bool {
        let (Some(conv), Some(tape)) = (self.relative, &self.cdc.tape_iscale) else {
            return false;
        };
        // the terms at the inputs of the residual tested, the latest step's
        let (work, terms) = &mut self.iscale;
        let step = &self.tapes.step;
        let Some(bound) = &step.bound else {
            return false;
        };
        tape.eval(bound.inputs(), work, terms);
        let n = self.cdc.n;
        self.cdc
            .residual_relative_ok(&step.out[..n], terms, x, GMIN_DC, &conv)
    }

    fn limit(&self, x: &[f64], step: &mut [f64]) -> f64 {
        newton::limit_step(&self.cdc.limits, x, step)
    }
}
