//! What one transient integration carries from step to step, and the
//! implicit Euler stage of its consistent starts and restarts.
//!
//! Every row of the DAE reads `I(x, t) + d/dt Q(x)`: `I` the residual at
//! rest, `Q` the charge it stores. The integration carries the charges,
//! `dQ/dt = −Ĩ` with `Ĩ = I + gmin·x`. A consistent start from a caller's
//! state, and the restart past a discontinuity, are one implicit Euler step
//! of a sliver `δ`,
//!
//! `r(X) = (Q(X) − qₙ)/δ + Ĩ(X, t) = 0`,
//!
//! solved by the modified Newton of [`CompiledDc::stage_newton`] against the
//! stage matrix `C(X)/δ + G(X) + gmin`, `G = dI/dx` and `C = dQ/dx`. What a
//! step needs beyond its time and size -- the programs, the stage
//! factorization and whether it is fresh, the charge at the step start, the
//! mass matrix, the scratch buffers, the statistics and the tolerances --
//! lives in one [`StageWorkspace`].

use sane_core::constants::*;

use crate::program::{Need, Program};
use crate::stage_matrix::StageMatrix;
use crate::{newton, sparse, CompiledDc, Convergence, Stats, Symbolic};

/// The mass matrix `C = dQ/dx` for the state rates `x' = C⁻¹ f`: when `C`
/// is regular, its factorization at any state.
pub(crate) struct MassMatrix<'a> {
    /// `None` when `C` is singular (a genuine DAE with algebraic unknowns).
    lu: Option<sparse::Refactorable<'a>>,
    /// `C` vanished at the start: no dynamic element at all.
    zero: bool,
    /// `C` evaluated for a rate.
    at: Vec<f64>,
}

impl<'a> MassMatrix<'a> {
    /// `C` at `(x, t)`, factorized on `sym` when regular.
    fn new(sym: Option<&'a Symbolic>, program: &mut Program<'_>, x: &[f64], t: f64) -> Self {
        let mut m = MassMatrix {
            lu: sym.map(|s| s.pattern.factorizer()),
            at: program.eval_c(x, t).to_vec(),
            zero: false,
        };
        let zero = m.at.iter().all(|v| *v == 0.0);
        if !m.factor() {
            m.lu = None;
        }
        m.zero = zero;
        m
    }

    /// Factorize `C` at the values `at` holds.
    fn factor(&mut self) -> bool {
        let Some(lu) = self.lu.as_mut() else {
            return false;
        };
        lu.factor_scaled(&[(&self.at, 1.0)], 0.0, false)
    }

    /// No dynamic element at all: every accepted point is an algebraic solve.
    pub fn is_zero(&self) -> bool {
        self.zero
    }

    /// The state rate `x' = C(x)⁻¹ slope` into `out`, `C` evaluated at
    /// `(x, t)`; `false` when `C` is singular.
    pub fn rate_at(
        &mut self,
        program: &mut Program<'_>,
        x: &[f64],
        t: f64,
        slope: &[f64],
        out: &mut [f64],
    ) -> bool {
        if self.lu.is_none() {
            return false;
        }
        self.at.copy_from_slice(program.eval_c(x, t));
        self.factor() && self.lu.as_mut().is_some_and(|lu| lu.solve_into(slope, out))
    }
}

/// One implicit Euler stage as the Newton core solves it (see
/// [`CompiledDc::stage_newton`]): `r(X) = (Q(X) − qₙ)/hg + Ĩ(X)` against the
/// transient-wide stage factorization, which an evaluation with the
/// Jacobian refactors.
struct Stage<'s, 'w, 'a> {
    cdc: &'s CompiledDc,
    ws: &'w mut StageWorkspace<'a>,
    ti: f64,
    hg: f64,
    /// Every row of the last residual within the rounding of its terms.
    rounded: bool,
}

impl newton::System<f64> for Stage<'_, '_, '_> {
    fn eval(&mut self, x: &[f64], jacobian: bool, res: &mut [f64]) -> bool {
        let (cdc, ws, hg) = (self.cdc, &mut *self.ws, self.hg);
        let n = cdc.n;
        ws.stats.iters += 1;
        if jacobian {
            ws.stats.refacs += 1;
            if !cdc.refactor_stage(ws, x, self.ti, hg) {
                return false;
            }
            // the rounding of the stage's updates, on the new factors
            if !cdc.rounding_floor(ws, x, hg, 1.0) {
                return false;
            }
            ws.floor_fresh = true;
        } else {
            ws.program.eval(x, self.ti, Need::Residual);
        }
        // r = (Q(x) - qn)/hg + Ĩ, Ĩ = I + gmin·x; within the rounding of
        // what it is computed from in every row, it is solved.
        ws.matrix.row_magnitudes(x, hg, &mut ws.rowmag);
        let (cur, chg) = (ws.program.currents(), ws.program.charges());
        let mut rounded = true;
        for k in 0..n {
            let (i, q) = (cur[k] + GMIN_DC * x[k], chg[k]);
            let r = (q - ws.qn[k]) / hg + i;
            res[k] = r;
            let terms = (q.abs() + ws.qn[k].abs()) / hg + ws.rowmag[k];
            rounded &= r.abs() <= NEWTON_ROUNDOFF * f64::EPSILON * terms;
        }
        self.rounded = rounded;
        true
    }

    fn factor(&mut self) -> bool {
        true // the evaluation with the Jacobian refactored
    }

    fn solve(&mut self, rhs: &[f64], dx: &mut [f64]) -> bool {
        self.ws.matrix.solve(rhs, dx)
    }

    fn rounding(&mut self, _x: &[f64], _res: &[f64]) -> (bool, f64) {
        (self.rounded, 0.0)
    }

    fn update_rounding(&self) -> Option<&[f64]> {
        self.ws.floor_fresh.then_some(self.ws.floor.as_slice())
    }

    fn limit(&self, x: &[f64], step: &mut [f64]) -> f64 {
        newton::limit_step(&self.cdc.limits, x, step)
    }

    fn stale(&self) -> bool {
        !self.ws.fac_fresh
    }

    fn mark_stale(&mut self) {
        self.ws.fac_fresh = false;
    }
}

/// Everything one transient integration carries into every stage solve.
pub(crate) struct StageWorkspace<'a> {
    /// The transient's programs, bound to the integration's parameters.
    pub program: Program<'a>,
    /// The stage matrix (see [`StageMatrix`]), and whether its factors
    /// belong to the current iterate and step size.
    pub matrix: StageMatrix<'a>,
    pub fac_fresh: bool,
    pub mass: MassMatrix<'a>,
    pub cdx: Vec<f64>,
    /// The charge at the step start, and at the latest evaluation (a
    /// solved stage's).
    pub qn: Vec<f64>,
    pub q: Vec<f64>,
    /// Per row the magnitude `|G| |x| + |C| |x| / hγ` its residual is
    /// computed from (the residual's rounding floor).
    pub rowmag: Vec<f64>,
    /// Per unknown, the rounding floor of the stage solves (see
    /// [`CompiledDc::rounding_floor`]), and whether it belongs to the
    /// current factorization.
    pub floor: Vec<f64>,
    pub floor_fresh: bool,
    /// A step's right-hand sides, and its error estimate.
    pub rhs: Vec<f64>,
    pub step: Vec<f64>,
    /// The slope `f = −Ĩ(x)` of the latest residual slope.
    pub f: Vec<f64>,
    /// The slopes `−Ĩ` at a step's start and end.
    pub slopes: [Vec<f64>; 2],
    pub stats: Stats,
    /// A delay-history knot's delayed values and rates.
    pub delay_vals: Vec<f64>,
    pub delay_rate: Vec<f64>,
    /// Per unknown, whether it is index-2: outside the error norm.
    pub index2: &'a [bool],
    pub rtol: f64,
    pub atol: f64,
    pub trace: bool,
    /// The Newton core's vectors, kept across the stage solves.
    pub newton: newton::Scratch<f64>,
}

impl<'a> StageWorkspace<'a> {
    /// The workspace of an integration from `(t0, x0)`: the stage
    /// factorization on `sym`, the mass matrix's on `mass_sym`, and the
    /// charge at `x0` as the first step's start.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cdc: &'a CompiledDc,
        sym: &'a Symbolic,
        mass_sym: Option<&'a Symbolic>,
        x0: &[f64],
        p: &[f64],
        t0: f64,
        taus: Vec<f64>,
        dhist: Option<crate::delay::DelayHistory>,
        rtol: f64,
        atol: f64,
    ) -> Self {
        let n = cdc.n;
        let mut program = Program::new(cdc, x0, p, t0);
        if let Some(h) = dhist {
            program.set_delays(h, taus);
        }
        let mass = MassMatrix::new(mass_sym, &mut program, x0, t0);
        let mut ws = StageWorkspace {
            index2: &cdc.index2,
            program,
            matrix: StageMatrix::new(cdc, sym),
            fac_fresh: false,
            mass,
            cdx: vec![0.0; n],
            qn: vec![0.0; n],
            q: vec![0.0; n],
            rowmag: vec![0.0; n],
            floor: vec![0.0; n],
            floor_fresh: false,
            rhs: vec![0.0; n],
            step: vec![0.0; n],
            f: vec![0.0; n],
            slopes: [vec![0.0; n], vec![0.0; n]],
            stats: Stats::default(),
            delay_vals: Vec::new(),
            delay_rate: Vec::new(),
            rtol,
            atol,
            trace: sane_core::config().tran_trace,
            newton: newton::Scratch::default(),
        };
        cdc.residual_slope(&mut ws, x0, t0);
        ws.advance();
        ws
    }

    /// The state moved on: its charge `ws.q` is the next step's start.
    pub fn advance(&mut self) {
        self.qn.copy_from_slice(&self.q);
    }

    /// The integration tolerances as the shared convergence contract (one
    /// floor for every kind: a transient carries `atol` alone).
    pub fn tolerances(&self) -> Convergence {
        Convergence {
            reltol: self.rtol,
            abstol: self.atol,
            vntol: self.atol,
        }
    }
}

impl CompiledDc {
    /// The shunted slope `f = −Ĩ(x, t)` at `(x, t)` into `ws.f` and the
    /// charge `Q(x)` into `ws.q`, through the residual-only tape (an
    /// explicit stage needs no Jacobian).
    pub(crate) fn residual_slope(&self, ws: &mut StageWorkspace<'_>, x: &[f64], t: f64) {
        ws.program.eval(x, t, Need::Residual);
        let cur = ws.program.currents();
        for (k, f) in ws.f.iter_mut().enumerate() {
            *f = -(cur[k] + GMIN_DC * x[k]);
        }
        ws.q.copy_from_slice(ws.program.charges());
    }

    /// Evaluate the Jacobians at `(x, t)` and factorize the stage matrix
    /// `C/(hγ) + G + gmin` there, `Q(x)` into `ws.q`; `false` when singular.
    pub(crate) fn refactor_stage(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &[f64],
        t: f64,
        hg: f64,
    ) -> bool {
        ws.program.eval(x, t, Need::Jacobian);
        ws.floor_fresh = false;
        ws.q.copy_from_slice(ws.program.charges());
        let (g, c) = ws.program.jacobians();
        let ok = ws.matrix.assemble(g, c, 1.0 / hg);
        ws.fac_fresh = ok;
        ok
    }

    /// Solve the implicit Euler stage `r(X) = (Q(X) − qₙ)/hg + Ĩ(X, ti) = 0`
    /// (`qₙ` is `ws.qn`) by the limiting *modified* Newton of the core: the
    /// frozen factorization reused (evaluating only `I` and `Q` via the
    /// cheap residual tape), refactorized -- and the Jacobians re-evaluated
    /// -- only when a stale one contracts too slowly. No line search.
    /// Converged at the update fraction `tol` of the integration
    /// tolerances, or within the rounding of every row. `x` is the guess on
    /// entry and the stage value `X` on return; `true` when converged.
    pub(crate) fn stage_newton(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &mut [f64],
        ti: f64,
        hg: f64,
    ) -> bool {
        let tol = (10.0 * f64::EPSILON / ws.rtol).max(STAGE_TOL_MAX.min(ws.rtol.sqrt()));
        let contract = newton::Contract {
            criterion: self.criterion(&ws.tolerances()),
            residual: false,
            update: Some(tol),
        };
        let policy = newton::Policy {
            max_iter: STAGE_MAX_ITER,
            jacobian: newton::Jacobian::Modified {
                theta: STAGE_STALL_THETA,
            },
            globalization: newton::Globalization::None,
            limiting: self.tricks.device_limiting,
            stall: false,
            early_accept: false,
            composite: false,
            keep_best: false,
            trace: ws.trace,
        };
        self.stage_solve(ws, x, ti, hg, &contract, &policy)
    }

    /// One implicit stage at `ti` by the Newton core as `policy` says, to
    /// `contract`: `x` the guess on entry and the stage value on return.
    /// `true` when converged.
    pub(crate) fn stage_solve(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &mut [f64],
        ti: f64,
        hg: f64,
        contract: &newton::Contract<'_>,
        policy: &newton::Policy,
    ) -> bool {
        let mut scratch = std::mem::take(&mut ws.newton);
        let out = {
            let mut stage = Stage {
                cdc: self,
                ws,
                ti,
                hg,
                rounded: false,
            };
            newton::solve(&mut stage, x, contract, policy, &mut scratch)
        };
        ws.newton = scratch;
        out.converged
    }

    /// The rounding floor of the stage solves into `ws.floor`, times `gain`:
    /// per unknown, the move the residual's own rounding (the ulps of the
    /// terms each row sums, as `stage_newton` bounds them) makes through the
    /// stage matrix at the last stage `(x, h gamma)`. A mode the circuit pins only
    /// algebraically and weakly -- a floating bridge held by megaohms against
    /// a large capacitor's charge -- amplifies that rounding, as `1/h`; no
    /// tolerance below it is resolvable. `false` when the solve fails.
    pub(crate) fn rounding_floor(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &[f64],
        hg: f64,
        gain: f64,
    ) -> bool {
        ws.floor_fresh = false;
        ws.matrix.row_magnitudes(x, hg, &mut ws.rowmag);
        for k in 0..self.n {
            let terms = (ws.q[k].abs() + ws.qn[k].abs()) / hg + ws.rowmag[k];
            ws.rhs[k] = NEWTON_ROUNDOFF * f64::EPSILON * terms;
        }
        if !ws.matrix.solve(&ws.rhs, &mut ws.floor) {
            return false;
        }
        ws.floor.iter_mut().for_each(|v| *v = gain * v.abs());
        true
    }

    /// Consistent restart after a discontinuity at `(t, x)`: one implicit-Euler
    /// step of the vanishing size `delta` into the region beyond it. The
    /// algebraic part of the state jumps there (a switch closes, a source kinks)
    /// while the charges are pinned by `C/delta`; the returned state is the
    /// right-limit state the next step starts from. `ws.qn` is the charge
    /// carried to `x`. The state goes to `out`, its charge into `ws.q` and
    /// its slope into `ws.f`; `false` when the Newton fails (the caller then
    /// restarts unreinitialised).
    pub(crate) fn reinit_step(
        &self,
        ws: &mut StageWorkspace<'_>,
        x: &[f64],
        t: f64,
        delta: f64,
        out: &mut [f64],
    ) -> bool {
        ws.fac_fresh = false;
        out.copy_from_slice(x);
        let ok = self.stage_newton(ws, out, t + delta, delta);
        // the factorization now belongs to `delta`, not to any step size
        ws.fac_fresh = false;
        if ok {
            self.residual_slope(ws, out, t + delta);
        }
        ok
    }
}

/// The WRMS norm of a filtered error estimate against the integration
/// tolerances widened by a rounding `floor` (empty: none), over the
/// unknowns that are not index-2, `(norm, index of the worst component)`.
pub(crate) fn ws_error_norm(
    ws: &StageWorkspace<'_>,
    x_new: &[f64],
    e_filt: &[f64],
    scale: f64,
    floor: &[f64],
) -> (f64, usize) {
    let (mut acc, mut m) = (0.0, 0usize);
    let (mut worst, mut worst_e) = (0usize, 0.0f64);
    for r in 0..x_new.len() {
        // an index-2 unknown is a rate of the others, its error an order
        // lower than theirs: theirs control it
        if ws.index2[r] {
            continue;
        }
        m += 1;
        let floor = floor.get(r).copied().unwrap_or(0.0);
        let sc = (ws.atol + ws.rtol * x_new[r].abs() + floor).max(f64::MIN_POSITIVE);
        let e = (e_filt[r] / scale) / sc;
        acc += e * e;
        if e.abs() > worst_e {
            worst_e = e.abs();
            worst = r;
        }
    }
    ((acc / m.max(1) as f64).sqrt(), worst)
}
