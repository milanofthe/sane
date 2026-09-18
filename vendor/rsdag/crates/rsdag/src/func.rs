//! Functions in the graph: a sub-DAG over formal parameters, applied through
//! [`Node::Call`](crate::node::Node::Call) nodes.
//!
//! A compact model instantiated a thousand times is one function and a
//! thousand calls. The function's outputs are its terminal currents, internal
//! residual rows, noise densities, operating-point variables -- whatever the
//! frontend lowers once over the formal leaves; a call binds those leaves to
//! one instance's expressions. Everything the engine does with a call is a
//! property of this one construct:
//!
//! - **differentiation** is the chain rule over *derivative outputs*
//!   (`d out_j / d param_i`), which the context differentiates from the body
//!   on first demand and memoises, so the Jacobian of a circuit references
//!   derivative calls into the same function -- no marker names, no parsing;
//! - **inlining** substitutes the arguments into the output expression, which
//!   is how a single-instance module keeps a fully symbolic fragment and how
//!   the differential reference (every instance as its own graph) is produced;
//! - **evaluation** runs the body once per distinct argument list and reads
//!   the outputs, in the arena evaluator through a per-function tape and in a
//!   compiled tape through its body (its native forms
//!   and lane batching); a batch of calls into one function is what the SIMD
//!   lanes evaluate.
//!
//! An *extern* function has no symbolic body: its outputs, including the
//! derivative outputs it can supply, are slots of a numeric
//! [`crate::extern_fn::ExternBundle`] (a compiled OSDI model).
//! A derivative an extern cannot supply is the zero output.

use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::extern_fn::ExternBundle;
use crate::node::{ExprId, SymbolId};
use crate::role::{OutputRole, ParamRole};

/// Index of a function in a [`Graph`](crate::graph::Graph).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct FuncId(pub u32);

/// An interned `(function, output index)` pair -- what a `Call` node names.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct OutputId(pub u32);

/// One output of a function.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Output {
    /// A symbolic expression over the function's parameters.
    Expr(ExprId),
    /// Output slot `k` of an extern body.
    Slot(u32),
    /// Identically zero (a derivative the body does not carry).
    Zero,
}

/// How a function's outputs are computed.
pub enum FunctionBody {
    /// Outputs are expressions over the parameters.
    Symbolic,
    /// Outputs are slots of a numeric bundle.
    Extern(Arc<dyn ExternBundle>),
}

/// A function as something to call: a bundle and which of its slots
/// carries each output, so a tape or a sweep picks outputs without the
/// symbolic expressions. An extern function is its bundle; a symbolic one
/// is its interpreted body.
#[derive(Clone)]
pub struct Body {
    pub bundle: Arc<dyn ExternBundle>,
    /// `slot_of[out]` is the bundle slot holding output `out`, `None` when
    /// the body was compiled without it.
    pub slot_of: Vec<Option<u32>>,
}

pub struct Function {
    pub name: String,
    /// Formal leaves in argument order.
    pub params: Vec<SymbolId>,
    /// One role per parameter (`Free` unless set).
    pub param_roles: Vec<ParamRole>,
    pub outputs: Vec<Output>,
    /// One role per output (`Plain` unless set; derivative outputs are
    /// tagged `Derivative`).
    pub output_roles: Vec<OutputRole>,
    pub body: FunctionBody,
    /// A body a consumer compiled itself and registered with
    /// [`Graph::set_func_body`](crate::Graph::set_func_body), used in place
    /// of the interpreted body of a symbolic function (a consumer's body may
    /// cache work over its solve-constant arguments, say) by every program
    /// whose calls it covers ([`Function::body_for`]); a program that calls
    /// an expression output it lacks (a derivative demanded later) takes the
    /// interpreted body until the consumer registers one that covers it.
    /// The symbolic outputs stay: differentiation and printing read them,
    /// only the evaluation goes through the registered bundle.
    pub compiled: Vec<Body>,
    /// Derivative output `d outputs[out] / d params[param]`, by index.
    pub(crate) deriv_index: HashMap<(u32, u32), u32>,
}

/// A function body evaluated by the interpreter: the fallback every consumer
/// can build from the symbolic outputs alone, so a tape or an arena sweep is
/// total without a solver-registered body (which only upgrades this to native
/// code and lane batching).
pub struct InterpretedBody {
    tape: crate::tape::Tape,
    n_out: usize,
}

impl ExternBundle for InterpretedBody {
    fn n_outputs(&self) -> usize {
        self.n_out
    }
    fn call(&self, args: &[f64], out: &mut [f64]) {
        // A pool rather than one buffer: a body that calls a body nests.
        thread_local! {
            static POOL: std::cell::RefCell<Vec<(Vec<f64>, Vec<f64>)>> = Default::default();
        }
        let (mut work, mut o) = POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
        self.tape.eval(args, &mut work, &mut o);
        out.copy_from_slice(&o[..self.n_out]);
        POOL.with(|p| p.borrow_mut().push((work, o)));
    }
    fn body(&self) -> Option<&crate::tape::Tape> {
        Some(&self.tape)
    }
}

impl Function {
    /// The function as something to call: an extern function is its bundle
    /// with the slot of each output, a symbolic one is its body compiled to
    /// a tape and interpreted (see [`InterpretedBody`]), every expression
    /// output a slot.
    pub fn body<K: crate::field::Field>(&self, ctx: &crate::graph::Graph<K>) -> Body {
        let all: Vec<u32> = (0..self.outputs.len() as u32).collect();
        self.body_for(ctx, &all)
    }

    /// [`body`](Self::body) for a program that calls the outputs `needed`:
    /// among the registered bodies that carry each of them that is an
    /// expression, the one computing the fewest outputs; the interpreted
    /// body when none covers them.
    pub fn body_for<K: crate::field::Field>(
        &self,
        ctx: &crate::graph::Graph<K>,
        needed: &[u32],
    ) -> Body {
        let covering = self.compiled.iter().filter(|c| {
            needed.iter().all(|&k| {
                !matches!(self.outputs[k as usize], Output::Expr(_))
                    || c.slot_of.get(k as usize).is_some_and(|s| s.is_some())
            })
        });
        if let Some(c) = covering.min_by_key(|c| c.bundle.n_outputs()) {
            return c.clone();
        }
        if let FunctionBody::Extern(b) = &self.body {
            return Body {
                bundle: b.clone(),
                slot_of: self
                    .outputs
                    .iter()
                    .map(|o| match o {
                        Output::Slot(k) => Some(*k),
                        _ => None,
                    })
                    .collect(),
            };
        }
        let mut roots = Vec::new();
        let mut slot_of = Vec::with_capacity(self.outputs.len());
        for o in &self.outputs {
            match o {
                Output::Expr(e) => {
                    slot_of.push(Some(roots.len() as u32));
                    roots.push(*e);
                }
                _ => slot_of.push(None),
            }
        }
        let tape = crate::tape::Tape::compile(ctx, &roots, &self.params);
        Body {
            bundle: Arc::new(InterpretedBody {
                tape,
                n_out: roots.len(),
            }),
            slot_of,
        }
    }

    /// Indices of the parameters carrying `role`, in argument order.
    pub fn params_with_role(&self, role: impl Fn(&ParamRole) -> bool) -> Vec<u32> {
        (0..self.params.len() as u32)
            .filter(|&i| role(&self.param_roles[i as usize]))
            .collect()
    }

    /// Indices of the outputs carrying `role`, in output order.
    pub fn outputs_with_role(&self, role: impl Fn(&OutputRole) -> bool) -> Vec<u32> {
        (0..self.outputs.len() as u32)
            .filter(|&i| role(&self.output_roles[i as usize]))
            .collect()
    }

    pub fn is_extern(&self) -> bool {
        matches!(self.body, FunctionBody::Extern(_))
    }
}
