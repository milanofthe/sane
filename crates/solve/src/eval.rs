//! The hot tapes run as `rsdag::Adaptive`: the interpreter, its choice
//! specialization and (with the `jit` feature) native code compiled in the
//! background, chosen per call and bit-exact against each other. SANE's
//! part is the configuration: `Config::jit` and `Config::tape_specialization`
//! switch the rungs, the thresholds are rsdag's defaults.

use rsdag::{Adaptive, Policy, Tape};

pub(crate) type StepEval = Adaptive;
pub(crate) type PrologToken = rsdag::Episode;

/// A hot tape under SANE's configuration.
pub(crate) fn step_eval(tape: Tape) -> StepEval {
    let cfg = sane_core::config();
    let policy = Policy {
        jit: cfg!(feature = "jit") && cfg.jit,
        specialize: cfg.tape_specialization,
        ..Policy::default()
    };
    #[cfg(feature = "jit")]
    let compiler = Some(rsdag_jit::compiler());
    #[cfg(not(feature = "jit"))]
    let compiler = None;
    Adaptive::new(tape, policy, compiler)
}

/// The system at a state, evaluated the way the Newton iterations do: the
/// parameter prolog once, then per call only the main phase, into buffers
/// this keeps. The DC program gives `I ++ G` (the residual at rest and its
/// Jacobian), the transient program `I ++ Q ++ G ++ C` (with the charges
/// and their Jacobian), as a DC and a transient Newton iteration need them.
pub struct Evaluator<'a> {
    tape: &'a StepEval,
    cdc: &'a crate::CompiledDc,
    tok: PrologToken,
    inputs: Vec<f64>,
    work: Vec<f64>,
    out: Vec<f64>,
}

impl crate::CompiledDc {
    /// An evaluator of the DC program at the parameters `p`.
    pub fn dc_evaluator(&self, p: &[f64]) -> Evaluator<'_> {
        Evaluator::new(self, &self.tape_step_dc, p)
    }

    /// An evaluator of the transient program at the parameters `p`.
    pub fn transient_evaluator(&self, p: &[f64]) -> Evaluator<'_> {
        Evaluator::new(self, &self.tape_tran_step, p)
    }
}

impl<'a> Evaluator<'a> {
    fn new(cdc: &'a crate::CompiledDc, tape: &'a StepEval, p: &[f64]) -> Self {
        let (mut inputs, mut work) = (Vec::new(), Vec::new());
        cdc.fill_inputs(&[], p, 0.0, &mut inputs);
        let tok = tape.eval_prolog(&inputs, &mut work);
        Evaluator {
            tape,
            cdc,
            tok,
            inputs,
            work,
            out: Vec::new(),
        }
    }

    /// The program's outputs at the state `x`.
    pub fn eval(&mut self, x: &[f64]) -> &[f64] {
        self.cdc.patch_inputs(x, 0.0, &mut self.inputs);
        self.tape
            .eval_main(&mut self.tok, &self.inputs, &mut self.work, &mut self.out);
        &self.out
    }

    /// Whether native code serves the evaluation yet (it is compiled in the
    /// background while the interpreter serves).
    pub fn native(&self) -> bool {
        self.tape.native().is_some()
    }
}
