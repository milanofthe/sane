//! Roles: the system layer as a signature.
//!
//! A role is metadata on a parameter or an output of a [`Function`]; nothing
//! in the graph changes. A DAE is a function with `Free`, `State` and `Time`
//! parameters and `Residual` outputs (in charge form, a `Charge` output beside
//! each, the residual its current), a block diagram block one with `Input`,
//! `State`, `Time`, `Memory` and `Param` parameters and `Output`,
//! `StateDeriv` and `MemoryWrite` outputs, an event a `Guard` output (with
//! the [`Crossing`] direction that counts) plus an effect function with
//! `StateWrite` outputs. A transport delay is a `History` parameter beside
//! a `DelaySource` and a `DelayTime` output, a noise generator a `Noise`
//! parameter (zero in every evaluation, the system's response to it its
//! transfer) beside its `NoiseLevel` outputs, and a quantity read off the
//! system without entering it an `Observer` output. Consumers keep their own
//! integrators and schedulers; the backend guarantees that a function with
//! roles can be evaluated, differentiated with respect to any role subset,
//! specialized and lowered.
//!
//! [`Function`]: crate::func::Function

/// What a parameter of a function stands for.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ParamRole {
    /// A symbolic unknown or a formal leaf without further meaning.
    Free,
    /// An element of an input port.
    Input { port: u32, elem: u32 },
    /// A tunable parameter (bound to a persistent buffer at lowering).
    Param,
    /// A continuous state.
    State { id: u32 },
    /// The time derivative of a continuous state (DAE conventions).
    StateDot { id: u32 },
    /// The independent variable.
    Time,
    /// A discrete memory slot element.
    Memory { slot: u32, offset: u32 },
    /// A past value a consumer supplies, delay line `id`'s output (a
    /// transport delay's interpolated history): its
    /// [`DelaySource`](OutputRole::DelaySource) output
    /// [`DelayTime`](OutputRole::DelayTime) earlier.
    History { id: u32 },
    /// Noise generator `id`: an input that is zero in every evaluation; the
    /// system's small-signal response to it, weighted by its
    /// [`NoiseLevel`](OutputRole::NoiseLevel) outputs, is its noise.
    Noise { id: u32 },
}

impl ParamRole {
    /// Where the role's group sits in a [`Signature`], and the parameter's
    /// place within it.
    fn place(&self) -> (u8, u32, u32) {
        match *self {
            ParamRole::State { id } => (0, id, 0),
            ParamRole::StateDot { id } => (1, id, 0),
            ParamRole::Input { port, elem } => (2, port, elem),
            ParamRole::Param => (3, 0, 0),
            ParamRole::Time => (4, 0, 0),
            ParamRole::Memory { slot, offset } => (5, slot, offset),
            ParamRole::History { id } => (6, id, 0),
            ParamRole::Noise { id } => (7, id, 0),
            ParamRole::Free => (8, 0, 0),
        }
    }
}

/// A function's parameters in the order a program over it takes its inputs:
/// grouped by role (states, their derivatives, inputs, parameters, time,
/// memory, histories, noise generators, the rest), each group by its index (a state's id, an
/// input's port and element) and otherwise in declaration order. One
/// ordering, so the function's roles and the programs compiled over it
/// state the same signature, and a consumer fills a group as one slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// The input symbols, in input order.
    pub syms: Vec<crate::node::SymbolId>,
    /// Their roles.
    pub roles: Vec<ParamRole>,
}

impl Signature {
    /// The signature of `f`'s parameters.
    pub fn of(f: &crate::func::Function) -> Signature {
        let mut order: Vec<usize> = (0..f.params().len()).collect();
        // Stable: parameters of one place keep their declaration order.
        order.sort_by_key(|&i| f.param_roles()[i].place());
        Signature {
            syms: order.iter().map(|&i| f.params()[i]).collect(),
            roles: order.iter().map(|&i| f.param_roles()[i]).collect(),
        }
    }

    /// The inputs whose role satisfies `role`, as one range: the groups
    /// are contiguous, so a group (or a run of groups) is.
    pub fn range(&self, role: impl Fn(&ParamRole) -> bool) -> std::ops::Range<usize> {
        let first = self
            .roles
            .iter()
            .position(&role)
            .unwrap_or(self.roles.len());
        let len = self.roles[first..].iter().take_while(|r| role(r)).count();
        first..first + len
    }

    /// The prolog split a program over the signature takes: the `Param`
    /// inputs are pure.
    pub fn pure_mask(&self) -> Vec<bool> {
        self.roles
            .iter()
            .map(|r| matches!(r, ParamRole::Param))
            .collect()
    }
}

/// The direction of a sign change that counts as a crossing: Verilog-A's
/// `@(cross(expr, dir))` argument, and what a switch declares about its
/// threshold.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Crossing {
    /// Either direction (Verilog-A `0`).
    #[default]
    Either,
    /// Negative to positive (`+1`).
    Rising,
    /// Positive to negative (`-1`).
    Falling,
}

impl Crossing {
    /// Whether the step from `before` to `after` reaches or crosses the
    /// surface in this direction.
    ///
    /// Deliberately generous at zero: touching the surface counts, from
    /// either side. A guard exists so that a hard `Select` in the residual
    /// flips on a step boundary, and a `Select` may compare with `>` or with
    /// `>=`, so no single rule about the exact zero is right for both. The
    /// cost of the two choices is not symmetric: a missed flip puts a jump
    /// inside a step, where Newton and the error estimate see nonsense, while
    /// a surplus landing costs one step. Keeping a consumer off a surface it
    /// has already fired on is that consumer's business (arm the surface
    /// again once the trajectory has left it), not this test's.
    pub fn crosses(self, before: f64, after: f64) -> bool {
        if before == after {
            return false;
        }
        let rising = before <= 0.0 && after >= 0.0;
        let falling = before >= 0.0 && after <= 0.0;
        match self {
            Crossing::Either => rising || falling,
            Crossing::Rising => rising,
            Crossing::Falling => falling,
        }
    }
}

/// What an output of a function computes.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OutputRole {
    /// A value without further meaning (the default).
    Plain,
    /// An element of an output port.
    Output { port: u32, elem: u32 },
    /// `dx/dt` of a continuous state.
    StateDeriv { id: u32 },
    /// A residual `F(x, x', t) = 0`. In charge form, `F = i(x, t) + d/dt
    /// q(x)`, the current `i` of row `id`, its charge the
    /// [`Charge`](Self::Charge) output of the same `id`.
    Residual { id: u32 },
    /// The charge `q(x)` of residual row `id` in charge form: the row reads
    /// the [`Residual`](Self::Residual) output of the same `id` plus the time
    /// derivative of this one. A row without one stores no charge.
    Charge { id: u32 },
    /// A discrete state assignment (event effects).
    StateWrite { id: u32 },
    /// A memory slot assignment.
    MemoryWrite { slot: u32, offset: u32 },
    /// A guard `g(x, t)` whose sign change in `dir` is an event: the
    /// consumer's integrator lands a step on the crossing and runs the
    /// effect function (the [`StateWrite`](Self::StateWrite) outputs).
    Guard { id: u32, dir: Crossing },
    /// The signal delay line `id` delays: its
    /// [`History`](ParamRole::History) input reads this output
    /// [`DelayTime`](Self::DelayTime) earlier.
    DelaySource { id: u32 },
    /// The delay of delay line `id`.
    DelayTime { id: u32 },
    /// The level of noise generator `id` (see [`ParamRole::Noise`]):
    /// `elem` 0 the power spectral density at 1 Hz, 1 the exponent of its
    /// frequency dependence (`level / f^exponent`), then, for a tabulated
    /// one, `(frequency, density)` pairs from `elem` 2 on.
    NoiseLevel { id: u32, elem: u32 },
    /// A quantity read off the system without entering it (an operating
    /// point variable).
    Observer { id: u32 },
    /// A derivative output `d outputs[of] / d params[wrt]` (memoised by
    /// [`Graph::derivative_output`](crate::graph::Graph::derivative_output)).
    Derivative { of: u32, wrt: u32 },
}
