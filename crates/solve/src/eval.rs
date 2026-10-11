//! The hot tapes run as `rsdag::Adaptive`: the interpreter, its choice
//! specialization and (with the `jit` feature) native code compiled in the
//! background, chosen per call and bit-exact against each other. SANE's
//! part is the configuration: `Config::jit` and `Config::tape_specialization`
//! switch the rungs, the thresholds are rsdag's defaults.

use rsdag::{Adaptive, Policy, Tape};

use crate::CompiledDc;

pub(crate) type StepEval = Adaptive;
pub(crate) type PrologToken = rsdag::Episode;

/// A hot tape under SANE's configuration.
pub(crate) fn step_eval(tape: Tape) -> StepEval {
    let cfg = sane_core::config();
    let policy = Policy {
        jit: cfg!(feature = "jit") && cfg.jit,
        specialize: cfg.tape_specialization,
        variants: rsdag::variant::VariantPolicy {
            enabled: cfg.variants,
            ..Default::default()
        },
        ..Policy::default()
    };
    #[cfg(feature = "jit")]
    let compiler = Some(rsdag_jit::compiler());
    #[cfg(not(feature = "jit"))]
    let compiler = None;
    Adaptive::new(tape, policy, compiler)
}

/// A program of the system's inputs bound to one parameter vector: the one
/// way every hot program of the system is evaluated. Binding fills the
/// inputs and runs the parameter-pure prolog once; every evaluation patches
/// the state, the time and the delayed values into the inputs and runs the
/// main phase. It belongs to one tape, which it does not hold, so a holder
/// may keep the tape next to it.
pub(crate) struct Bound {
    tok: PrologToken,
    inputs: Vec<f64>,
    work: Vec<f64>,
}

impl Bound {
    /// `tape` of `cdc` at the parameters `p`, with `extra` after the
    /// system's inputs (a pull's weights, a Hessian's multipliers).
    pub fn new(cdc: &CompiledDc, tape: &StepEval, p: &[f64], extra: &[f64]) -> Self {
        let mut inputs = Vec::new();
        cdc.fill_inputs(&[], p, 0.0, &mut inputs);
        inputs.extend_from_slice(extra);
        let mut work = Vec::new();
        let tok = tape.eval_prolog(&inputs, &mut work);
        Bound { tok, inputs, work }
    }

    /// `tape` at `(x, t)` into `out`.
    pub fn eval(
        &mut self,
        cdc: &CompiledDc,
        tape: &StepEval,
        x: &[f64],
        t: f64,
        out: &mut Vec<f64>,
    ) {
        cdc.patch_inputs(x, t, &mut self.inputs);
        tape.eval_main(&mut self.tok, &self.inputs, &mut self.work, out);
    }

    /// The inputs of the latest evaluation, for a program over the same
    /// inputs to read the same point.
    pub fn inputs(&self) -> &[f64] {
        &self.inputs
    }
}

/// The system at a state, evaluated the way the Newton iterations do (see
/// [`Bound`]), into a buffer this keeps. The DC program gives `I ++ G`
/// (the residual at rest and its Jacobian), the transient program
/// `I ++ Q ++ G ++ C` (with the charges and their Jacobian), as a DC and a
/// transient Newton iteration need them.
pub struct Evaluator<'a> {
    tape: &'a StepEval,
    cdc: &'a CompiledDc,
    binding: Bound,
    out: Vec<f64>,
}

impl CompiledDc {
    /// An evaluator of the DC program at the parameters `p`.
    pub fn dc_evaluator(&self, p: &[f64]) -> Evaluator<'_> {
        Evaluator::new(self, &self.tape_step_dc, p)
    }

    /// An evaluator of the transient program at the parameters `p`.
    pub fn transient_evaluator(&self, p: &[f64]) -> Evaluator<'_> {
        Evaluator::new(self, &self.tape_tran_step, p)
    }

    /// `tape` evaluated once at `(x, p, t)`, `extra` after the system's
    /// inputs.
    pub(crate) fn eval_once(
        &self,
        tape: &StepEval,
        x: &[f64],
        p: &[f64],
        t: f64,
        extra: &[f64],
    ) -> Vec<f64> {
        let mut out = Vec::new();
        Bound::new(self, tape, p, extra).eval(self, tape, x, t, &mut out);
        out
    }
}

impl<'a> Evaluator<'a> {
    fn new(cdc: &'a CompiledDc, tape: &'a StepEval, p: &[f64]) -> Self {
        Evaluator {
            tape,
            cdc,
            binding: Bound::new(cdc, tape, p, &[]),
            out: Vec::new(),
        }
    }

    /// The program's outputs at the state `x`.
    pub fn eval(&mut self, x: &[f64]) -> &[f64] {
        (self.binding).eval(self.cdc, self.tape, x, 0.0, &mut self.out);
        &self.out
    }

    /// Whether native code serves the evaluation yet (it is compiled in the
    /// background while the interpreter serves).
    pub fn native(&self) -> bool {
        self.tape.native().is_some()
    }
}
