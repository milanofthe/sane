//! Forward sensitivities of the transient: the derivative of the computed
//! trajectory by a set of parameters, by differentiating every step the
//! integration takes (internal differentiation).
//!
//! A Rodas4 step from `yₙ = (qₙ, xₙ)` solves its stages `i` (see
//! [`crate::rosenbrock`])
//!
//! `u_qᵢ/(hγ) + G u_xᵢ + Ĩ(Xᵢ, tᵢ) + h dᵢ Iₜ − Σⱼ (Cᵢⱼ/h) u_qⱼ = 0`,
//! `u_qᵢ − C u_xᵢ + Qᵢ − Q(Xᵢ) − h dᵢ Qₜ = 0`,
//!
//! `G`, `C`, `Iₜ`, `Qₜ` at the step start, `(Qᵢ, Xᵢ) = yₙ + Σⱼ aᵢⱼ uⱼ`.
//! Differentiated by a parameter, a stage's derivatives `(u_qᵢ', u_xᵢ')`
//! solve the same stage matrix `K = C/(hγ) + G`,
//!
//! `K u_xᵢ' = R_q − R_x/(hγ)`,  `u_qᵢ' = C u_xᵢ' + R_x`,
//!
//! `R_q = −Eᴵᵢ' − Gᵢ Xᵢ' − dI/dp(Xᵢ) + Σⱼ (Cᵢⱼ/h) u_qⱼ'`,
//! `R_x = Eᵠᵢ' − Qᵢ' + Cᵢ Xᵢ' + dQ/dp(Xᵢ)`,
//!
//! with `Gᵢ`, `Cᵢ` the Jacobians at the stage point and `Eᵢ' = dE/dx xₙ' +
//! dE/dp` how the start's frozen Jacobians and time rates move along the
//! stage's direction `(u_xᵢ, h dᵢ)`: the frozen entries' gradients, the
//! step's, contracted with it (see [`sane_dae::Frozen`]). The step's
//! derivative is
//! `yₙ₊₁' = yₙ' + Σ bᵢ uᵢ'` (from `Q(xₙ)` where the step projected its
//! charge, its derivative `C xₙ' + dQ/dp`), the output's the continuous
//! extension differentiated. A consistent (re)start, the implicit Euler
//! stage `(Q(X) − qₐ)/δ + Ĩ(X) = 0`, differentiates to
//! `(C/δ + G) X' = (qₐ' − dQ/dp)/δ − dI/dp`, an operating-point start to
//! `G x₀' = −dI/dp`.
//!
//! Every derivative solves with a factorization the integration has: the
//! step's stage matrix, a restart's own. The step sequence is the
//! integration's, held: the derivative is the computed trajectory's at its
//! steps. Each parameter costs a solve, a few sparse products and its own
//! columns' programs per stage, as sparse as the elements it enters; the
//! Jacobians at the stage points (the stepper's, evaluated with its stages'
//! currents and charges) and the frozen entries' gradients are per step,
//! whatever the number of parameters.

use std::sync::Arc;

use sane_core::constants::GMIN_DC;

use crate::program::{Need, Program};
use crate::rosenbrock::{Rosenbrock, RODAS4};
use crate::sens::{Bound, FrozenState, SensProgram};
use crate::stage_matrix::StageMatrix;
use crate::transient::fraction;
use crate::{CompiledDc, Symbolic};

/// The sensitivities of one integration by the parameter columns `cols`,
/// and their rows at the requested times for the unknowns `outputs`.
pub(crate) struct Sens<'a> {
    cdc: &'a CompiledDc,
    program: Program<'a>,
    /// A restart's stage matrix `C/δ + G` at its point.
    euler: StageMatrix<'a>,
    np: usize,
    outputs: Vec<usize>,
    /// Per parameter, the state's and the charge's derivative, and the
    /// state's at the start of the latest segment.
    sx: Vec<Vec<f64>>,
    sq: Vec<Vec<f64>>,
    sx_prev: Vec<Vec<f64>>,
    /// Per parameter, the charge's derivative the latest step started from.
    sqb: Vec<Vec<f64>>,
    /// Per stage and parameter, the stage increments' derivatives.
    ux: Vec<Vec<Vec<f64>>>,
    uq: Vec<Vec<Vec<f64>>>,
    /// The set's programs, each bound to the parameters, and per parameter
    /// its entries in their columns' outputs: `(row, place)` of the
    /// currents and charges, `(row, direction component, place)` of the
    /// frozen entries.
    prog: Arc<SensProgram>,
    frozen: &'a FrozenState,
    iq_bound: Bound,
    e_bound: Bound,
    iq_of: Vec<Vec<(usize, usize)>>,
    e_of: Vec<Vec<(usize, usize, usize)>>,
    /// The latest evaluations' values: the Jacobians at a stage point, the
    /// frozen entries' gradients and columns at the step start, the
    /// parameters' columns at a stage point.
    g: Vec<f64>,
    c: Vec<f64>,
    ej: Vec<f64>,
    piq: Vec<f64>,
    /// A stage's direction `(u_x, h d)` and `dE/dx` along it.
    w: Vec<f64>,
    m: Vec<f64>,
    /// Scratch.
    yx: Vec<f64>,
    yq: Vec<f64>,
    rq: Vec<f64>,
    rx: Vec<f64>,
    rhs: Vec<f64>,
    point: Vec<f64>,
    /// The rows at the requested times, `outputs × parameters` each.
    t_eval: &'a [f64],
    cursor: usize,
    rows: Vec<Vec<f64>>,
}

impl<'a> Sens<'a> {
    /// The sensitivities by the parameter columns `cols` of an integration
    /// from `(t0, x0)`, at `t_eval` for the unknowns `outputs`: from an
    /// operating point (`dc_start`) the point's, from a given state zero.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cdc: &'a CompiledDc,
        sym: &'a Symbolic,
        p: &[f64],
        x0: &[f64],
        dc_start: bool,
        cols: &[usize],
        outputs: &[usize],
        t_eval: &'a [f64],
    ) -> Result<Self, String> {
        let (n, np) = (cdc.n, cols.len());
        let t0 = t_eval.first().copied().unwrap_or(0.0);
        let prog = cdc
            .sens_program(cols)
            .ok_or("transient sensitivity: call ensure_transient_sensitivity first")?;
        let by_param = |rows: &[usize], of: &[usize], offset: usize| {
            let mut by = vec![Vec::new(); np];
            for (k, (&r, &j)) in rows.iter().zip(of).enumerate() {
                by[j].push((r, offset + k));
            }
            by
        };
        let iq_of = by_param(&prog.iq_rows, &prog.iq_of, 0);
        let frozen = cdc
            .frozen()
            .ok_or("transient sensitivity: call ensure_transient_sensitivity first")?;
        let mut e_of = vec![Vec::new(); np];
        for (k, (&at, &j)) in prog.e_at.iter().zip(&prog.e_of).enumerate() {
            let place = frozen.hx_w.len() + k;
            e_of[j].push((frozen.rows[at], frozen.cols[at], place));
        }
        let mut s = Sens {
            cdc,
            program: Program::new(cdc, x0, p, t0),
            euler: StageMatrix::new(cdc, sym),
            np,
            outputs: outputs.to_vec(),
            sx: vec![vec![0.0; n]; np],
            sq: vec![vec![0.0; n]; np],
            sx_prev: vec![vec![0.0; n]; np],
            sqb: vec![vec![0.0; n]; np],
            ux: vec![vec![vec![0.0; n]; np]; RODAS4.b.len()],
            uq: vec![vec![vec![0.0; n]; np]; RODAS4.b.len()],
            iq_bound: Bound::new(cdc, &prog.iq, p),
            e_bound: Bound::new(cdc, &prog.e, p),
            prog,
            frozen,
            iq_of,
            e_of,
            g: Vec::new(),
            c: Vec::new(),
            ej: Vec::new(),
            piq: Vec::new(),
            w: vec![0.0; n + 1],
            m: vec![0.0; frozen.x_rc.0.len()],
            yx: vec![0.0; n],
            yq: vec![0.0; n],
            rq: vec![0.0; n],
            rx: vec![0.0; n],
            rhs: vec![0.0; n],
            point: vec![0.0; n],
            t_eval,
            cursor: 0,
            rows: vec![vec![0.0; outputs.len() * np]; t_eval.len()],
        };
        // the operating point's, `(G + gmin) x0' = -dI/dp`, and the charge
        // `Q(x0)`'s, `C x0' + dQ/dp`
        let lu = if dc_start {
            Some(
                cdc.factor_fx(x0, p, t0)
                    .ok_or("transient sensitivity: the operating point's Jacobian is singular")?,
            )
        } else {
            None
        };
        s.program.eval(x0, t0, Need::Jacobian);
        s.c.clear();
        s.c.extend_from_slice(s.program.jacobians().1);
        s.iq_bound.eval(cdc, &s.prog.iq, x0, t0, &mut s.piq);
        for j in 0..np {
            if let Some(lu) = &lu {
                s.rhs.fill(0.0);
                for &(r, k) in &s.iq_of[j] {
                    if r < n {
                        s.rhs[r] -= s.piq[k];
                    }
                }
                s.sx[j] = lu
                    .solve(&s.rhs)
                    .ok_or("transient sensitivity: the operating point's solve failed")?;
            }
            let sq = &mut s.sq[j];
            sq.fill(0.0);
            for (k, &v) in s.c.iter().enumerate() {
                sq[cdc.jxd_rows[k]] += v * s.sx[j][cdc.jxd_cols[k]];
            }
            for &(r, k) in &s.iq_of[j] {
                if r >= n {
                    sq[r - n] += s.piq[k];
                }
            }
        }
        Ok(s)
    }

    /// The step the integration just accepted, of size `h` from `(tn, xn)`
    /// by `method`, on its stage matrix `matrix`. `false` when a solve
    /// fails.
    pub fn rodas(
        &mut self,
        matrix: &mut StageMatrix<'_>,
        method: &Rosenbrock,
        xn: &[f64],
        tn: f64,
        h: f64,
    ) -> bool {
        let (cdc, tab, n) = (self.cdc, &RODAS4, self.cdc.n);
        let hg = h * tab.gamma;
        let ux = method.stage_ux();
        for j in 0..self.np {
            self.sx_prev[j].copy_from_slice(&self.sx[j]);
        }
        // the frozen entries' gradients and columns at the step start
        self.e_bound.eval(cdc, &self.prog.e, xn, tn, &mut self.ej);
        let fz = self.frozen;
        for i in 0..tab.b.len() {
            // the stage point, its Jacobians and parameter Jacobians (the
            // first stage's are the start's)
            let ti = tn + tab.c[i] * h;
            self.point.copy_from_slice(xn);
            for (jx, &a) in tab.a[i].iter().enumerate() {
                for k in 0..n {
                    self.point[k] += a * ux[jx][k];
                }
            }
            if i == 0 {
                self.g.clear();
                self.g.extend_from_slice(matrix.g_values());
                self.c.clear();
                self.c.extend_from_slice(matrix.c_values());
            } else {
                let (g, c) = method
                    .stage_jacobians(i)
                    .expect("the stepper keeps its stage Jacobians");
                self.g.clear();
                self.g.extend_from_slice(g);
                self.c.clear();
                self.c.extend_from_slice(c);
            }
            // how the start's frozen Jacobians and time rates move with it,
            // along the stage's direction: dE/dx
            self.w[..n].copy_from_slice(&ux[i]);
            self.w[n] = h * tab.d[i];
            self.m.fill(0.0);
            for (q, (&slot, &c)) in fz.hx_slot.iter().zip(&fz.hx_w).enumerate() {
                self.m[slot] += self.ej[q] * self.w[c];
            }
            // the parameters' own at the stage point: dI/dp, dQ/dp
            self.iq_bound
                .eval(cdc, &self.prog.iq, &self.point, ti, &mut self.piq);
            for j in 0..self.np {
                // the charge the step started from
                if i == 0 {
                    if method.projected() {
                        let sqb = &mut self.sqb[j];
                        matrix.c_mul(&self.sx[j], sqb);
                        for &(r, k) in &self.iq_of[j] {
                            if r >= n {
                                sqb[r - n] += self.piq[k];
                            }
                        }
                    } else {
                        self.sqb[j].copy_from_slice(&self.sq[j]);
                    }
                }
                // the stage point's derivative
                self.yx.copy_from_slice(&self.sx[j]);
                self.yq.copy_from_slice(&self.sqb[j]);
                for (jx, &a) in tab.a[i].iter().enumerate() {
                    if a != 0.0 {
                        for k in 0..n {
                            self.yx[k] += a * self.ux[jx][j][k];
                            self.yq[k] += a * self.uq[jx][j][k];
                        }
                    }
                }
                let (rq, rx) = (&mut self.rq, &mut self.rx);
                rq.fill(0.0);
                rx.fill(0.0);
                // E' = dE/dx xn' + dE/dp: currents' rows into R_q, charges' into R_x
                let sxj = &self.sx[j];
                let x_rc = &fz.x_rc;
                for (k, (&r, &c)) in x_rc.0.iter().zip(&x_rc.1).enumerate() {
                    let v = self.m[k] * sxj[c];
                    if r < n {
                        rq[r] -= v;
                    } else {
                        rx[r - n] += v;
                    }
                }
                for &(r, c, k) in &self.e_of[j] {
                    let v = self.ej[k] * self.w[c];
                    if r < n {
                        rq[r] -= v;
                    } else {
                        rx[r - n] += v;
                    }
                }
                // the stage point's Jacobians along its derivative
                for (k, &g) in self.g.iter().enumerate() {
                    rq[cdc.jx_rows[k]] -= g * self.yx[cdc.jx_cols[k]];
                }
                for k in 0..n {
                    rq[k] -= GMIN_DC * self.yx[k];
                }
                for (k, &c) in self.c.iter().enumerate() {
                    rx[cdc.jxd_rows[k]] += c * self.yx[cdc.jxd_cols[k]];
                }
                // the parameter's own at the stage point
                for &(r, k) in &self.iq_of[j] {
                    if r < n {
                        rq[r] -= self.piq[k];
                    } else {
                        rx[r - n] += self.piq[k];
                    }
                }
                for k in 0..n {
                    rx[k] -= self.yq[k];
                }
                for (jx, &cij) in tab.cm[i].iter().enumerate() {
                    for k in 0..n {
                        rq[k] += cij / h * self.uq[jx][j][k];
                    }
                }
                for k in 0..n {
                    self.rhs[k] = rq[k] - rx[k] / hg;
                }
                if !matrix.solve(&self.rhs, &mut self.ux[i][j]) {
                    return false;
                }
                matrix.c_mul(&self.ux[i][j], &mut self.uq[i][j]);
                for k in 0..n {
                    self.uq[i][j][k] += self.rx[k];
                }
            }
        }
        for j in 0..self.np {
            self.sq[j].copy_from_slice(&self.sqb[j]);
            for (i, &b) in tab.b.iter().enumerate() {
                for k in 0..n {
                    self.sx[j][k] += b * self.ux[i][j][k];
                    self.sq[j][k] += b * self.uq[i][j][k];
                }
            }
        }
        self.sx.iter().flatten().all(|v| v.is_finite())
    }

    /// A consistent (re)start just taken: the implicit Euler stage of size
    /// `hg` at `ts` that ended in `x`. `false` when singular.
    pub fn euler(&mut self, x: &[f64], ts: f64, hg: f64) -> bool {
        let (cdc, n) = (self.cdc, self.cdc.n);
        for j in 0..self.np {
            self.sx_prev[j].copy_from_slice(&self.sx[j]);
        }
        self.program.eval(x, ts, Need::Jacobian);
        let (g, c) = self.program.jacobians();
        if !self.euler.assemble(g, c, 1.0 / hg) {
            return false;
        }
        self.iq_bound.eval(cdc, &self.prog.iq, x, ts, &mut self.piq);
        for j in 0..self.np {
            // (C/δ + G) X' = (qₐ' − dQ/dp)/δ − dI/dp
            for k in 0..n {
                self.rhs[k] = self.sq[j][k] / hg;
            }
            for &(r, k) in &self.iq_of[j] {
                if r < n {
                    self.rhs[r] -= self.piq[k];
                } else {
                    self.rhs[r - n] -= self.piq[k] / hg;
                }
            }
            if !self.euler.solve(&self.rhs, &mut self.sx[j]) {
                return false;
            }
            // its charge out, Q(X): C X' + dQ/dp
            self.euler.c_mul(&self.sx[j], &mut self.sq[j]);
            for &(r, k) in &self.iq_of[j] {
                if r >= n {
                    self.sq[j][r - n] += self.piq[k];
                }
            }
        }
        true
    }

    /// The rows at the requested times at or before the start: the current
    /// derivatives.
    pub fn emit_start(&mut self, t0: f64) {
        while self.cursor < self.t_eval.len() && self.t_eval[self.cursor] <= t0 {
            self.row_now(self.cursor);
            self.cursor += 1;
        }
    }

    /// The rows at the requested times in the segment `(ta, tb]` just
    /// taken: on the step's continuous extension differentiated (`own`),
    /// linear over a restart's sliver.
    pub fn emit(&mut self, own: bool, ta: f64, tb: f64) {
        let (tab, no, np) = (&RODAS4, self.outputs.len(), self.np);
        while self.cursor < self.t_eval.len() && self.t_eval[self.cursor] <= tb {
            let th = fraction(self.t_eval[self.cursor], ta, tb);
            let row = &mut self.rows[self.cursor];
            for (o, &r) in self.outputs.iter().enumerate() {
                for j in 0..np {
                    let (a, b) = (self.sx_prev[j][r], self.sx[j][r]);
                    row[o * np + j] = if own {
                        let (mut d2, mut d3) = (0.0, 0.0);
                        for i in 0..tab.d2.len() {
                            d2 += tab.d2[i] * self.ux[i][j][r];
                            d3 += tab.d3[i] * self.ux[i][j][r];
                        }
                        a * (1.0 - th) + th * (b + (1.0 - th) * (d2 + th * d3))
                    } else {
                        a * (1.0 - th) + b * th
                    };
                }
            }
            debug_assert_eq!(row.len(), no * np);
            self.cursor += 1;
        }
    }

    /// The rows past the last step at the final derivatives, and all rows.
    pub fn finish(mut self) -> Vec<Vec<f64>> {
        while self.cursor < self.t_eval.len() {
            self.row_now(self.cursor);
            self.cursor += 1;
        }
        self.rows
    }

    fn row_now(&mut self, k: usize) {
        let np = self.np;
        for (o, &r) in self.outputs.iter().enumerate() {
            for j in 0..np {
                self.rows[k][o * np + j] = self.sx[j][r];
            }
        }
    }
}
