//! The hot tapes run as `rsdag::Adaptive`: the interpreter, its choice
//! specialization and (with the `jit` feature) native code compiled in the
//! background, chosen per call and bit-exact against each other. SANE's
//! part is the configuration: `Config::jit` and `Config::tape_specialization`
//! switch the rungs, the thresholds are rsdag's defaults.

use rsdag::{Adaptive, Policy, Tape};

pub(crate) type StepEval = Adaptive;
pub(crate) type PrologToken = rsdag::Episode;

#[cfg(feature = "jit")]
pub(crate) fn jit_enabled() -> bool {
    sane_core::config().jit
}

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
