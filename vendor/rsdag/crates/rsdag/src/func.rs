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

/// A function of a [`Graph`](crate::graph::Graph), read through its
/// accessors and changed only through the graph, so the derivative index
/// always says what the output roles say.
pub struct Function {
    name: String,
    params: Vec<SymbolId>,
    param_roles: Vec<ParamRole>,
    outputs: Vec<Output>,
    output_roles: Vec<OutputRole>,
    /// The numeric bundle of an extern function; `None` for a symbolic one.
    extern_body: Option<Arc<dyn ExternBundle>>,
    compiled: Vec<Body>,
    /// The interpreted bodies [`body_for`](Function::body_for) built, one per
    /// set of outputs a program asked for: every later program over the
    /// graph that calls a subset of one takes it, so a body is compiled once
    /// per function rather than once per program.
    interpreted: std::sync::Mutex<Vec<Body>>,
    /// Derivative output `d outputs[out] / d params[param]`, by index: the
    /// outputs whose role is [`OutputRole::Derivative`].
    deriv_index: HashMap<(u32, u32), u32>,
    /// Per output, its support once asked for (see
    /// [`Graph::output_support`](crate::Graph::output_support)). An output
    /// never changes once pushed, so neither does its support.
    support: std::sync::Mutex<Vec<Option<Arc<[u32]>>>>,
}

/// A function body evaluated by the interpreter: the fallback every consumer
/// can build from the symbolic outputs alone, so a tape or an arena sweep is
/// total without a solver-registered body (which only upgrades this to native
/// code and lane batching).
///
/// Parameters with [`ParamRole::Param`] are the body's pure arguments: its
/// tape is split over them, and a caller that has them in its own prolog
/// keeps the body's prolog result per instance (see
/// [`ExternBundle::state_len`]).
pub struct InterpretedBody {
    tape: crate::tape::Tape,
    n_out: usize,
    /// One flag per parameter; empty when no parameter is pure.
    pure: Vec<bool>,
    /// What backends compiled of the tape (see [`ExternBundle::backend_cache`]).
    backends: crate::extern_fn::BackendCache,
}

impl InterpretedBody {
    /// The work layout: the tape's buffer, its outputs, then the arguments
    /// a prolog is run on.
    fn parts<'w>(&self, work: &'w mut [f64]) -> (&'w mut [f64], &'w mut [f64], &'w mut [f64]) {
        let (w, rest) = work.split_at_mut(self.tape.work_len());
        let (o, a) = rest.split_at_mut(self.tape.out_len());
        (w, o, &mut a[..self.pure.len()])
    }
}

impl ExternBundle for InterpretedBody {
    fn n_outputs(&self) -> usize {
        self.n_out
    }
    fn work_len(&self) -> usize {
        self.tape.work_len() + self.tape.out_len() + self.pure.len()
    }
    fn call_into(&self, args: &[f64], work: &mut [f64], out: &mut [f64]) {
        let (w, o, _) = self.parts(work);
        self.tape.eval_into(args, w, o);
        out.copy_from_slice(&o[..self.n_out]);
    }
    fn state_len(&self) -> usize {
        self.tape.state_len()
    }
    fn pure_args(&self) -> &[bool] {
        &self.pure
    }
    fn prolog_into(&self, pure: &[f64], work: &mut [f64], state: &mut [f64]) {
        let (w, _, args) = self.parts(work);
        // The prolog reads the pure arguments only; the others are NaN.
        let mut p = pure.iter();
        for (a, &is_pure) in args.iter_mut().zip(&self.pure) {
            *a = if is_pure {
                *p.next().expect("one value per pure argument")
            } else {
                f64::NAN
            };
        }
        self.tape.eval_prolog_into(args, w);
        state.copy_from_slice(&w[..state.len()]);
    }
    fn main_into(&self, args: &[f64], state: &[f64], work: &mut [f64], out: &mut [f64]) {
        let (w, o, _) = self.parts(work);
        w[..state.len()].copy_from_slice(state);
        self.tape.eval_main_into(args, w, o);
        out.copy_from_slice(&o[..self.n_out]);
    }
    fn body(&self) -> Option<&crate::tape::Tape> {
        Some(&self.tape)
    }
    fn backend_cache(&self) -> Option<&crate::extern_fn::BackendCache> {
        Some(&self.backends)
    }
}

impl Function {
    /// A function without outputs over `params`, every role `Free`.
    pub(crate) fn new(
        name: &str,
        params: Vec<SymbolId>,
        extern_body: Option<Arc<dyn ExternBundle>>,
    ) -> Function {
        Function {
            name: name.to_string(),
            param_roles: vec![ParamRole::Free; params.len()],
            params,
            outputs: Vec::new(),
            output_roles: Vec::new(),
            extern_body,
            compiled: Vec::new(),
            interpreted: Default::default(),
            deriv_index: HashMap::default(),
            support: Default::default(),
        }
    }

    /// Append an output with its role; returns its index.
    pub(crate) fn push_output(&mut self, output: Output, role: OutputRole) -> u32 {
        let k = self.outputs.len() as u32;
        self.outputs.push(output);
        self.output_roles.push(OutputRole::Plain);
        self.set_output_role(k, role);
        k
    }

    /// Set the role of output `out`, keeping the derivative index.
    pub(crate) fn set_output_role(&mut self, out: u32, role: OutputRole) {
        if let OutputRole::Derivative { of, wrt } = self.output_roles[out as usize] {
            self.deriv_index.remove(&(of, wrt));
        }
        if let OutputRole::Derivative { of, wrt } = role {
            self.deriv_index.insert((of, wrt), out);
        }
        self.output_roles[out as usize] = role;
    }

    pub(crate) fn set_param_role(&mut self, param: u32, role: ParamRole) {
        self.param_roles[param as usize] = role;
    }

    /// The derivative output `d outputs[out] / d params[param]`, if there is one.
    pub fn derivative(&self, out: u32, param: u32) -> Option<u32> {
        self.deriv_index.get(&(out, param)).copied()
    }

    pub(crate) fn cached_support(&self, out: u32) -> Option<Arc<[u32]>> {
        self.support
            .lock()
            .unwrap()
            .get(out as usize)
            .cloned()
            .flatten()
    }

    pub(crate) fn cache_support(&self, out: u32, support: Arc<[u32]>) {
        let mut cache = self.support.lock().unwrap();
        if cache.len() <= out as usize {
            cache.resize(out as usize + 1, None);
        }
        cache[out as usize] = Some(support);
    }

    pub(crate) fn compiled_mut(&mut self) -> &mut Vec<Body> {
        &mut self.compiled
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    /// Formal leaves in argument order.
    pub fn params(&self) -> &[SymbolId] {
        &self.params
    }
    /// One role per parameter (`Free` unless set).
    pub fn param_roles(&self) -> &[ParamRole] {
        &self.param_roles
    }
    pub fn outputs(&self) -> &[Output] {
        &self.outputs
    }
    /// One role per output (`Plain` unless set; derivative outputs are
    /// tagged `Derivative`).
    pub fn output_roles(&self) -> &[OutputRole] {
        &self.output_roles
    }
    /// The numeric bundle of an extern function; `None` for a symbolic one.
    pub fn extern_body(&self) -> Option<&Arc<dyn ExternBundle>> {
        self.extern_body.as_ref()
    }
    /// The bodies a consumer compiled itself and registered with
    /// [`Graph::set_func_body`](crate::Graph::set_func_body), used in place
    /// of the interpreted body of a symbolic function (a consumer's body may
    /// cache work over its solve-constant arguments, say) by every program
    /// whose calls it covers ([`Function::body_for`]); a program that calls
    /// an expression output it lacks (a derivative demanded later) takes the
    /// interpreted body until the consumer registers one that covers it.
    /// The symbolic outputs stay: differentiation and printing read them,
    /// only the evaluation goes through the registered bundle.
    pub fn compiled(&self) -> &[Body] {
        &self.compiled
    }

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
    /// expression, the one computing the fewest outputs; else the body of
    /// exactly those outputs, interpreted.
    pub fn body_for<K: crate::field::Field>(
        &self,
        ctx: &crate::graph::Graph<K>,
        needed: &[u32],
    ) -> Body {
        let covering = self.compiled.iter().filter(|c| self.covers(c, needed));
        if let Some(c) = covering.min_by_key(|c| c.bundle.n_outputs()) {
            return c.clone();
        }
        if self.extern_body.is_none() {
            let built = self.interpreted.lock().unwrap();
            let covering = built.iter().filter(|c| self.covers(c, needed));
            if let Some(c) = covering.min_by_key(|c| c.bundle.n_outputs()) {
                return c.clone();
            }
        }
        if let Some(b) = &self.extern_body {
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
        let mut slot_of = vec![None; self.outputs.len()];
        for &k in needed {
            if let (Output::Expr(e), None) = (self.outputs[k as usize], slot_of[k as usize]) {
                slot_of[k as usize] = Some(roots.len() as u32);
                roots.push(e);
            }
        }
        // Split over the parameters the roles call pure, when there are any.
        let pure: Vec<bool> = self
            .param_roles
            .iter()
            .map(|r| matches!(r, ParamRole::Param))
            .collect();
        let (tape, pure) = if pure.iter().any(|&p| p) {
            (
                crate::tape::Tape::compile_split(ctx, &roots, &self.params, &pure),
                pure,
            )
        } else {
            (
                crate::tape::Tape::compile(ctx, &roots, &self.params),
                Vec::new(),
            )
        };
        let body = Body {
            bundle: Arc::new(InterpretedBody {
                tape,
                n_out: roots.len(),
                pure,
                backends: Default::default(),
            }),
            slot_of,
        };
        self.interpreted.lock().unwrap().push(body.clone());
        body
    }

    /// Whether `body` carries every output of `needed` that is an expression.
    fn covers(&self, body: &Body, needed: &[u32]) -> bool {
        needed.iter().all(|&k| {
            !matches!(self.outputs[k as usize], Output::Expr(_))
                || body.slot_of.get(k as usize).is_some_and(|s| s.is_some())
        })
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
        self.extern_body.is_some()
    }
}
