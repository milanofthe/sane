//! The transient's programs over one input layout: the currents and charges
//! `I ++ Q`, the same with their Jacobians `I ++ Q ++ G ++ C`, and `C`
//! alone, each a view of the one lowered program (see `compile`).
//!
//! A [`Program`] holds the input vector in the programs' layout, with the
//! parameters written once, and each program's prolog token and work: the
//! parameter-pure prefix ran when the program was bound, an evaluation
//! patches the state, the time and the delayed values into the inputs and
//! runs the main phase into buffers it keeps. Nothing allocates once the
//! first evaluations have sized them. With transport delays the program
//! keeps their accepted history, and every evaluation at `t` reads the
//! delayed values at `t − τ` off it: no evaluation sees another time's.

use crate::delay::DelayHistory;
use crate::{CompiledDc, PrologToken};

/// The transport delays of an integration: their accepted history, the
/// delays, and the delayed values last published, at `at` (NaN: none since
/// the history changed), as publication `published`.
struct Delays {
    history: DelayHistory,
    taus: Vec<f64>,
    vals: Vec<f64>,
    rates: Vec<f64>,
    at: f64,
    published: u64,
}

/// What an evaluation computes beyond `I` and `Q`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Need {
    /// The currents and the charges.
    Residual,
    /// With their Jacobians `G = dI/dx` and `C = dQ/dx`.
    Jacobian,
}

pub(crate) struct Program<'a> {
    cdc: &'a CompiledDc,
    inputs: Vec<f64>,
    res_tok: PrologToken,
    res_work: Vec<f64>,
    step_tok: PrologToken,
    step_work: Vec<f64>,
    c_work: Vec<f64>,
    /// `I ++ Q` of the latest evaluation, `++ G ++ C` after one with the
    /// Jacobians.
    out: Vec<f64>,
    jacobian: bool,
    /// The evaluations into `out` so far.
    evals: u64,
    c_out: Vec<f64>,
    dt_work: Vec<f64>,
    dt_out: Vec<f64>,
    delays: Option<Delays>,
}

impl<'a> Program<'a> {
    /// The programs of `cdc` bound to the parameters `p`, the inputs at
    /// `(x, t)`.
    pub fn new(cdc: &'a CompiledDc, x: &[f64], p: &[f64], t: f64) -> Self {
        let mut inputs = Vec::new();
        cdc.fill_inputs(x, p, t, &mut inputs);
        let (mut res_work, mut step_work) = (Vec::new(), Vec::new());
        let res_tok = cdc.tape_tran_res.eval_prolog(&inputs, &mut res_work);
        let step_tok = cdc.tape_tran_step.eval_prolog(&inputs, &mut step_work);
        Program {
            cdc,
            inputs,
            res_tok,
            res_work,
            step_tok,
            step_work,
            c_work: Vec::new(),
            out: Vec::new(),
            jacobian: false,
            evals: 0,
            c_out: Vec::new(),
            dt_work: Vec::new(),
            dt_out: Vec::new(),
            delays: None,
        }
    }

    /// Integrate with transport delays `taus`, their accepted history so far
    /// in `history`.
    pub fn set_delays(&mut self, history: DelayHistory, taus: Vec<f64>) {
        let k = taus.len();
        self.delays = Some(Delays {
            history,
            taus,
            vals: vec![0.0; k],
            rates: vec![0.0; k],
            at: f64::NAN,
            published: 0,
        });
    }

    /// The delays' accepted history, to extend: the values it published are
    /// stale from here on.
    pub fn history_mut(&mut self) -> Option<&mut DelayHistory> {
        self.delays.as_mut().map(|d| {
            d.at = f64::NAN;
            &mut d.history
        })
    }

    /// The delayed values at `t` published for the programs' inputs, unless
    /// they are already.
    fn publish(&mut self, t: f64) {
        let Some(d) = self.delays.as_mut() else {
            return;
        };
        if d.at == t && d.published == crate::delay::hist_publications() {
            return;
        }
        for (k, (v, tau)) in d.vals.iter_mut().zip(&d.taus).enumerate() {
            *v = d.history.eval(k, t - tau);
        }
        crate::delay::set_hist_values(&d.vals);
        d.at = t;
        d.published = crate::delay::hist_publications();
    }

    /// Evaluate at `(x, t)` what `need` asks for.
    pub fn eval(&mut self, x: &[f64], t: f64, need: Need) {
        let cdc = self.cdc;
        self.publish(t);
        cdc.patch_inputs(x, t, &mut self.inputs);
        self.jacobian = need == Need::Jacobian;
        self.evals += 1;
        if self.jacobian {
            cdc.tape_tran_step.eval_main(
                &mut self.step_tok,
                &self.inputs,
                &mut self.step_work,
                &mut self.out,
            );
        } else {
            cdc.tape_tran_res.eval_main(
                &mut self.res_tok,
                &self.inputs,
                &mut self.res_work,
                &mut self.out,
            );
        }
    }

    /// `C` alone at `(x, t)` (the state rates' mass matrix), in the
    /// jacobian-x' pattern.
    pub fn eval_c(&mut self, x: &[f64], t: f64) -> &[f64] {
        let cdc = self.cdc;
        self.publish(t);
        cdc.patch_inputs(x, t, &mut self.inputs);
        cdc.tape_c
            .eval(&self.inputs, &mut self.c_work, &mut self.c_out);
        &self.c_out
    }

    /// The rows' time rates at a fixed state, `dI/dt` and `dQ/dt` at
    /// `(x, t)`, into `it` and `qt`: the sources' explicit ones, and those
    /// of the delayed signals, which move at their history's rate at
    /// `t − τ`.
    pub fn eval_time_rates(&mut self, x: &[f64], t: f64, it: &mut [f64], qt: &mut [f64]) {
        let cdc = self.cdc;
        it.fill(0.0);
        qt.fill(0.0);
        let Some(tape) = &cdc.tape_dt else {
            return;
        };
        self.publish(t);
        let hist_rates: &[f64] = match self.delays.as_mut() {
            Some(d) => {
                for (k, (r, tau)) in d.rates.iter_mut().zip(&d.taus).enumerate() {
                    *r = d.history.rate(k, t - tau);
                }
                &d.rates
            }
            None => &[],
        };
        cdc.patch_inputs(x, t, &mut self.inputs);
        tape.eval(&self.inputs, &mut self.dt_work, &mut self.dt_out);
        let n = cdc.n;
        let (direct, hist) = self.dt_out.split_at(cdc.dt_rows.len());
        let terms = (cdc.dt_rows.iter().zip(direct).map(|(&row, &v)| (row, v)))
            .chain((cdc.dt_hist.iter().zip(hist)).map(|(&(row, k), &v)| (row, v * hist_rates[k])));
        for (row, v) in terms {
            if row < n {
                it[row] += v;
            } else {
                qt[row - n] += v;
            }
        }
    }

    /// The evaluations so far by [`eval`](Self::eval): what the accessors
    /// read belongs to the evaluation while this count is unchanged.
    pub fn evals(&self) -> u64 {
        self.evals
    }

    /// The currents `I` of the latest evaluation.
    pub fn currents(&self) -> &[f64] {
        &self.out[..self.cdc.n]
    }

    /// The charges `Q` of the latest evaluation.
    pub fn charges(&self) -> &[f64] {
        let n = self.cdc.n;
        &self.out[n..2 * n]
    }

    /// `G` and `C` of the latest evaluation, in their Jacobian patterns; it
    /// asked for them.
    pub fn jacobians(&self) -> (&[f64], &[f64]) {
        debug_assert!(self.jacobian, "the latest evaluation computed no Jacobian");
        let n = self.cdc.n;
        self.out[2 * n..].split_at(self.cdc.nnz_x)
    }
}
