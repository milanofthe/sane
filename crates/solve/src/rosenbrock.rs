//! Rosenbrock-Wanner methods for the SANE DAE `I(x, t) + d/dt Q(x) = 0`,
//! charge-oriented: linearly implicit, one Jacobian evaluation and one
//! factorization per step and no Newton iteration.
//!
//! The DAE is integrated as the index-1 system of the charges and the
//! unknowns, `y = (q, x)`,
//!
//! `q' = −Ĩ(x, t)`,  `0 = q − Q(x)`,
//!
//! with the constant mass matrix `M = diag(I, 0)` and `Ĩ = I + gmin·x`. Its
//! Jacobian `J = [[0, −G], [I, −C]]` (`G = dĨ/dx`, `C = dQ/dx` at the step
//! start) makes a stage's matrix `M/(hγ) − J`, which the charge rows reduce
//! to the transient's stage matrix: `(C/(hγ) + G) u_x = r_q + r_x/(hγ)`,
//! `u_q = C u_x − r_x`.
//!
//! The method is Rodas4 in Hairer and Wanner's transformed form (VI.4, their
//! RODAS code): stage `i` solves
//!
//! `(M/(hγ) − J) uᵢ = f(tₙ + cᵢh, yₙ + Σⱼ aᵢⱼ uⱼ) + M Σⱼ (Cᵢⱼ/h) uⱼ + h dᵢ f_t`,
//!
//! the step is `yₙ₊₁ = yₙ + Σ bᵢ uᵢ` and its error `Σ eᵢ uᵢ`. Both the
//! method and its embedded solution are stiffly accurate, L-stable and made
//! for index-1 DAEs: the last stage starts at the embedded solution and ends
//! at the step's. `f_t` is the
//! sources' explicit time dependence, `(−dI/dt, −dQ/dt)`, differentiated with
//! the rest of the DAE. Without device limiting, an evaluation that leaves
//! the finite numbers rejects the step.
//!
//! The charge is carried from step to step, as integrated: an oscillating
//! circuit keeps its phase by the charge it conserves. The carried charge is
//! off the constraint by the steps' errors, `δ = q − Q(xₙ)`, and a step
//! starting from it moves the state by `(C/(hγ) + G)⁻¹ δ/(hγ)` to meet it,
//! whatever else it does; damped by `G` at the step sizes of the
//! integration, undamped as `h` shrinks (`C⁻¹ δ`, which a capacitance
//! network coupling a node to its neighbours can make large). Where that
//! move exceeds the step's tolerance the step starts from the start's own
//! charge `Q(xₙ)` instead, on the constraint. A candidate within tolerance
//! evaluates the Jacobians at its end: its end slope for the dense output,
//! and the next step's stage matrix and charge when the next step starts
//! there. A rejected step's retry refactors the Jacobians it kept at the new
//! step size.

use sane_core::constants::*;

use crate::program::Need;
use crate::stage::{ws_error_norm, StageWorkspace};
use crate::CompiledDc;

/// A Rodas tableau (see the module docs); row `i` of `a` and `cm` holds the
/// coefficients of the stages before stage `i`.
pub(crate) struct Tableau {
    pub gamma: f64,
    pub a: &'static [&'static [f64]],
    pub cm: &'static [&'static [f64]],
    pub b: &'static [f64],
    pub e: &'static [f64],
    pub c: &'static [f64],
    pub d: &'static [f64],
    /// The order of the error estimate's leading term, `p̂ + 1`.
    pub err_order: f64,
    /// The continuous extension: over a step,
    /// `y(tₙ + θh) = yₙ (1 − θ) + θ (yₙ₊₁ + (1 − θ)(d₂ + θ d₃))` with
    /// `d₂ = Σ D₂ⱼ uⱼ`, `d₃ = Σ D₃ⱼ uⱼ`.
    pub d2: &'static [f64],
    pub d3: &'static [f64],
}

const R4A5: [f64; 4] = [
    1.221_224_509_226_641,
    6.019_134_481_288_629,
    12.537_083_329_320_87,
    -0.687_886_036_105_895,
];

/// Rodas4 (Hairer and Wanner, the RODAS code's method 1): six stages, order
/// 4 with an embedded order-3 solution.
pub(crate) const RODAS4: Tableau = Tableau {
    gamma: 0.25,
    a: &[
        &[],
        &[1.544],
        &[0.946_678_528_081_582_6, 0.255_701_169_898_328_4],
        &[
            3.314_825_187_068_521,
            2.896_124_015_972_201,
            0.998_641_913_997_781_7,
        ],
        &R4A5,
        &[R4A5[0], R4A5[1], R4A5[2], R4A5[3], 1.0],
    ],
    cm: &[
        &[],
        &[-5.668_8],
        &[-2.430_093_356_833_875, -0.206_359_915_709_191_5],
        &[
            -0.107_352_905_815_137_5,
            -9.594_562_251_023_355,
            -20.470_286_148_096_16,
        ],
        &[
            7.496_443_313_967_647,
            -10.246_804_314_643_52,
            -33.999_903_528_199_05,
            11.708_908_932_061_6,
        ],
        &[
            8.083_246_795_921_522,
            -7.981_132_988_064_893,
            -31.521_594_328_743_71,
            16.319_305_431_231_36,
            -6.058_818_238_834_054,
        ],
    ],
    b: &[R4A5[0], R4A5[1], R4A5[2], R4A5[3], 1.0, 1.0],
    e: &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    c: &[0.0, 0.386, 0.21, 0.63, 1.0, 1.0],
    d: &[0.25, -0.1043, 0.1035, -0.0362, 0.0, 0.0],
    err_order: 4.0,
    d2: &[
        10.126_235_083_445_86,
        -7.487_995_877_610_167,
        -34.800_918_615_557_47,
        -7.992_771_707_568_823,
        1.025_137_723_295_662,
    ],
    d3: &[
        -0.676_280_339_280_125_3,
        6.087_714_651_680_015,
        16.430_843_208_924_78,
        24.767_225_114_183_86,
        -6.594_389_125_716_872,
    ],
};

/// The Rosenbrock steps' vectors: the stage increments of the charges and the
/// unknowns, a stage's point, its right-hand side's charge rows, and at the
/// step start the currents, the charges and the sources' time rates.
#[derive(Default)]
struct RosWork {
    uq: Vec<Vec<f64>>,
    ux: Vec<Vec<f64>>,
    yq: Vec<f64>,
    yx: Vec<f64>,
    rx: Vec<f64>,
    i0: Vec<f64>,
    q0: Vec<f64>,
    /// The charge the step starts from.
    qb: Vec<f64>,
    it: Vec<f64>,
    qt: Vec<f64>,
    /// The continuous extension's `d₂`, `d₃` of the last step within
    /// tolerance.
    d2: Vec<f64>,
    d3: Vec<f64>,
}

impl RosWork {
    fn fit(&mut self, stages: usize, n: usize) {
        self.uq.resize(stages, Vec::new());
        self.ux.resize(stages, Vec::new());
        for v in self.uq.iter_mut().chain(self.ux.iter_mut()) {
            v.resize(n, 0.0);
        }
        for v in [
            &mut self.yq,
            &mut self.yx,
            &mut self.rx,
            &mut self.i0,
            &mut self.q0,
            &mut self.qb,
            &mut self.it,
            &mut self.qt,
            &mut self.d2,
            &mut self.d3,
        ] {
            v.resize(n, 0.0);
        }
    }
}

/// A Rosenbrock method of tableau `tab`, with what it keeps between steps:
/// the step start its stage matrix's Jacobians belong to, and the Jacobian
/// evaluation at the end of the latest candidate within tolerance, which
/// starts the next step when that step starts there.
pub(crate) struct Rosenbrock {
    tab: &'static Tableau,
    w: RosWork,
    /// The time and the state the stage matrix's `G`, `C` and `w.i0`,
    /// `w.q0`, `w.it`, `w.qt` were evaluated at.
    start: Option<f64>,
    start_x: Vec<f64>,
    /// The time, the state and the program's evaluation count of the end
    /// evaluation.
    end: Option<(f64, u64)>,
    end_x: Vec<f64>,
    /// Whether the latest step started from the start's own charge `Q(xₙ)`
    /// rather than the carried one.
    projected: bool,
    /// Per stage, the Jacobians `G`, `C` at its point, where kept (see
    /// [`keep_stage_jacobians`](Self::keep_stage_jacobians)).
    stage_jac: Option<Vec<(Vec<f64>, Vec<f64>)>>,
}

impl Rosenbrock {
    pub fn new(tab: &'static Tableau) -> Self {
        Rosenbrock {
            tab,
            w: RosWork::default(),
            start: None,
            start_x: Vec::new(),
            end: None,
            end_x: Vec::new(),
            projected: false,
            stage_jac: None,
        }
    }

    /// Evaluate the Jacobians at the stage points with their currents and
    /// charges, and keep them: what the step's sensitivities read (see
    /// [`stage_jacobians`](Self::stage_jacobians)), at the cost of the
    /// Jacobians over the residuals the stages need alone.
    pub fn keep_stage_jacobians(&mut self) {
        self.stage_jac = Some(vec![Default::default(); self.tab.b.len()]);
    }

    /// The Jacobians `G`, `C` at the point of the latest step's stage `i`
    /// (from `1`; the first stage's are the step start's), where kept.
    pub(crate) fn stage_jacobians(&self, i: usize) -> Option<(&[f64], &[f64])> {
        let (g, c) = &self.stage_jac.as_ref()?[i];
        Some((g, c))
    }

    /// The stage matrix at the step start `(t, xn)` for `hγ`: the end
    /// evaluation of the step before when the step starts where it ended,
    /// the kept Jacobians refactored when it starts where the last one did
    /// (a reject, a retake), an evaluation otherwise. `false` when singular.
    fn prepare(&mut self, ws: &mut StageWorkspace<'_>, xn: &[f64], t: f64, hg: f64) -> bool {
        let w = &mut self.w;
        let ended_here = self.end.is_some_and(|(te, k)| {
            te == t && k == ws.program.evals() && self.end_x.as_slice() == xn
        });
        self.end = None;
        if !ended_here && self.start == Some(t) && self.start_x.as_slice() == xn {
            ws.stats.refacs += 1;
            return ws.matrix.factor(1.0 / hg);
        }
        if !ended_here {
            ws.program.eval(xn, t, Need::Jacobian);
            ws.stats.iters += 1;
        }
        w.i0.copy_from_slice(ws.program.currents());
        w.q0.copy_from_slice(ws.program.charges());
        let (g, c) = ws.program.jacobians();
        ws.stats.refacs += 1;
        let ok = ws.matrix.assemble(g, c, 1.0 / hg);
        ws.program.eval_time_rates(xn, t, &mut w.it, &mut w.qt);
        self.start = ok.then_some(t);
        self.start_x.clear();
        self.start_x.extend_from_slice(xn);
        ok
    }

    /// The stages of a step of size `h` from `(t, xn)` and the charge
    /// `w.qb`, on the factored stage matrix and the start's `w.i0`, `w.q0`,
    /// `w.it`, `w.qt`: their increments into `w.ux`, `w.uq`. `false` when a
    /// stage leaves the finite numbers or its solve fails.
    fn run_stages(&mut self, ws: &mut StageWorkspace<'_>, xn: &[f64], t: f64, h: f64) -> bool {
        let (n, tab) = (xn.len(), self.tab);
        let (stages, hg) = (tab.b.len(), h * tab.gamma);
        let (w, kept) = (&mut self.w, &mut self.stage_jac);
        let need = if kept.is_some() {
            Need::Jacobian
        } else {
            Need::Residual
        };
        for i in 0..stages {
            // The stage's point, and the currents and charges there (the
            // first stage's are the step start's), with the Jacobians where
            // kept.
            w.yq.copy_from_slice(&w.qb);
            w.yx.copy_from_slice(xn);
            for (j, &a) in tab.a[i].iter().enumerate() {
                if a != 0.0 {
                    for k in 0..n {
                        w.yq[k] += a * w.uq[j][k];
                        w.yx[k] += a * w.ux[j][k];
                    }
                }
            }
            let (cur, chg) = if i == 0 {
                (w.i0.as_slice(), w.q0.as_slice())
            } else {
                let ti = t + tab.c[i] * h;
                ws.program.eval(&w.yx, ti, need);
                ws.stats.iters += 1;
                if let Some(kept) = kept.as_mut() {
                    let (g, c) = ws.program.jacobians();
                    kept[i].0.clear();
                    kept[i].0.extend_from_slice(g);
                    kept[i].1.clear();
                    kept[i].1.extend_from_slice(c);
                }
                (ws.program.currents(), ws.program.charges())
            };
            // r_q = −Ĩ(Yᵢ) + Σⱼ (Cᵢⱼ/h) u_qⱼ − h dᵢ dI/dt,
            // r_x = q(Yᵢ) − Q(X(Yᵢ)) − h dᵢ dQ/dt,
            // solved as (C/(hγ) + G) u_x = r_q + r_x/(hγ), u_q = C u_x − r_x.
            let hd = h * tab.d[i];
            for k in 0..n {
                let mut rq = -(cur[k] + GMIN_DC * w.yx[k]) - hd * w.it[k];
                for (j, &cij) in tab.cm[i].iter().enumerate() {
                    rq += cij / h * w.uq[j][k];
                }
                let rx = w.yq[k] - chg[k] - hd * w.qt[k];
                w.rx[k] = rx;
                ws.rhs[k] = rq + rx / hg;
            }
            if !ws.rhs.iter().all(|v| v.is_finite()) {
                return false;
            }
            if !ws.matrix.solve(&ws.rhs, &mut w.ux[i]) {
                return false;
            }
            ws.matrix.c_mul(&w.ux[i], &mut w.uq[i]);
            for k in 0..n {
                w.uq[i][k] -= w.rx[k];
            }
        }
        true
    }

    /// The unknowns' stage increments of the latest step.
    pub(crate) fn stage_ux(&self) -> &[Vec<f64>] {
        &self.w.ux
    }

    /// Whether the latest step started from `Q(xₙ)` (see the module docs).
    pub(crate) fn projected(&self) -> bool {
        self.projected
    }

    /// The order in `h` of the error estimate's leading term, by which the
    /// step size controller scales.
    pub fn err_order(&self) -> f64 {
        self.tab.err_order
    }

    /// The integration restarts at a discontinuity: no evaluation before it
    /// carries over.
    pub fn restart(&mut self) {
        self.start = None;
        self.end = None;
    }

    /// The state at `te` in the last step `[ta, tb]` from `xa` to `xb`, on
    /// the continuous extension, into `out`.
    pub fn interpolate(&self, te: f64, ta: f64, tb: f64, xa: &[f64], xb: &[f64], out: &mut [f64]) {
        let th = if tb > ta {
            ((te - ta) / (tb - ta)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let w = &self.w;
        for (k, o) in out.iter_mut().enumerate() {
            *o = xa[k] * (1.0 - th) + th * (xb[k] + (1.0 - th) * (w.d2[k] + th * w.d3[k]));
        }
    }

    /// The state rates at both ends of the last step `[ta, tb]`, the
    /// continuous extension's, into `ra` and `rb`.
    pub fn rates(&self, ta: f64, tb: f64, xa: &[f64], xb: &[f64], ra: &mut [f64], rb: &mut [f64]) {
        let (w, h) = (&self.w, tb - ta);
        for k in 0..xa.len() {
            ra[k] = (xb[k] - xa[k] + w.d2[k]) / h;
            rb[k] = (xb[k] - xa[k] - w.d2[k] - w.d3[k]) / h;
        }
    }

    /// One step from `(t, xn)` with size `h`: `x_{n+1}` into `x`, its
    /// charge into `ws.q`, the slopes at
    /// both ends into `ws.slopes[0]` and `ws.slopes[last]` (the loop's
    /// dense-output contract), the scaled error, or `None` when the stage
    /// matrix is singular or a stage leaves the finite numbers.
    pub fn step(
        &mut self,
        cdc: &CompiledDc,
        ws: &mut StageWorkspace<'_>,
        xn: &[f64],
        t: f64,
        h: f64,
        x: &mut [f64],
    ) -> Option<f64> {
        let (n, tab) = (cdc.n, self.tab);
        let stages = tab.b.len();
        self.w.fit(stages, n);
        let hg = h * tab.gamma;
        if !self.prepare(ws, xn, t, hg) {
            return None;
        }
        let w = &mut self.w;
        for k in 0..n {
            ws.slopes[0][k] = -(w.i0[k] + GMIN_DC * xn[k]);
            ws.rhs[k] = (ws.qn[k] - w.q0[k]) / hg;
        }
        // The charge the step starts from: the carried one, unless the step
        // would move the state by more than its tolerance to meet it -- then
        // the start's own, `Q(xₙ)` (see the module docs).
        if !ws.matrix.solve(&ws.rhs, &mut ws.cdx) {
            return None;
        }
        let (off, _) = ws_error_norm(ws, xn, &ws.cdx, 1.0, &[]);
        self.projected = off > 1.0;
        let base = if self.projected { &w.q0 } else { &ws.qn };
        w.qb.copy_from_slice(base);
        if !self.run_stages(ws, xn, t, h) {
            return None;
        }
        let w = &mut self.w;
        // The step and its error, from the stage increments.
        x.copy_from_slice(xn);
        ws.q.copy_from_slice(&w.qb);
        ws.step.fill(0.0);
        for i in 0..stages {
            let (b, e) = (tab.b[i], tab.e[i]);
            for k in 0..n {
                x[k] += b * w.ux[i][k];
                ws.q[k] += b * w.uq[i][k];
                ws.step[k] += e * w.ux[i][k];
            }
        }
        let (mut err, _) = ws_error_norm(ws, x, &ws.step, 1.0, &[]);
        if !err.is_finite() {
            return None;
        }
        // Rejected for truncation error, not for the rounding of the stage
        // solves: the estimate moves by `sum |e_i|` times their floor.
        if err > 1.0 {
            let gain = tab.e.iter().map(|e| e.abs()).sum::<f64>();
            if cdc.rounding_floor(ws, x, hg, gain) {
                (err, _) = ws_error_norm(ws, x, &ws.step, 1.0, &ws.floor);
            }
        }
        // Within tolerance: the Jacobians at the end, the step's end slope
        // and the next step's start.
        if err <= 1.0 {
            let te = t + h;
            ws.program.eval(x, te, Need::Jacobian);
            ws.stats.iters += 1;
            let (cur, last) = (ws.program.currents(), ws.slopes.len() - 1);
            for k in 0..n {
                ws.slopes[last][k] = -(cur[k] + GMIN_DC * x[k]);
            }
            if !ws.slopes[last].iter().all(|v| v.is_finite()) {
                return None;
            }
            self.end = Some((te, ws.program.evals()));
            self.end_x.clear();
            self.end_x.extend_from_slice(x);
            for k in 0..n {
                let (mut a, mut b) = (0.0, 0.0);
                for j in 0..tab.d2.len() {
                    a += tab.d2[j] * w.ux[j][k];
                    b += tab.d3[j] * w.ux[j][k];
                }
                w.d2[k] = a;
                w.d3[k] = b;
            }
        }
        Some(err.max(STEP_ERR_FLOOR))
    }
}
