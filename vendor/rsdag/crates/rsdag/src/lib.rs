//! Rust symbolic graph backend: a hash-consed expression DAG over free
//! symbols and exact or floating constants, with symbolic differentiation, a
//! flat tape and its interpreter, and a symbolic layer. The shared substrate
//! of SANE and fastsim, so an optimization here lands in every consumer.

// egg times its runs with `instant`, which on wasm32 links against a host
// `now` a plain browser module does not have: say so here instead of at link.
#[cfg(all(target_arch = "wasm32", feature = "egraph"))]
compile_error!("the `egraph` feature needs a clock (egg reads one) and does not build for wasm32");

pub mod adaptive;
pub mod autodiff;
pub mod builder;
pub mod display;
pub mod dot;
pub mod eval;
pub mod extern_fn;
pub mod field;
pub mod func;
pub mod graph;
pub mod hooks;
pub mod mathfn;
pub mod module;
pub mod node;
pub mod nonlinearity;
pub mod role;
pub mod scalar;
pub mod scope;
pub mod semantics;
mod simd;
pub mod simplify;
pub mod symbolic;
#[cfg(any(test, feature = "synth"))]
pub mod synth;
pub mod tape;
// Expression substitution is exposed only through the curated `substitute*`
// re-exports below, not as a module path.
pub(crate) mod transform;

pub use adaptive::{Adaptive, Compiler, Episode, Policy, Stats};
pub use autodiff::{
    differentiate, gradient, hessian, sparse_jacobian, time_derivative, SparseRows,
};
pub use builder::{Builder, Numeric};
pub use display::to_string;
pub use eval::{eval, eval_named};
pub use extern_fn::ExternBundle;
#[cfg(feature = "exact")]
pub use field::ratio_powi;
pub use field::{Field, F64};
pub use func::{Body, FuncId, Function, FunctionBody, Output, OutputId};
pub use graph::Graph;
pub use mathfn::lower_call;
pub use module::{IdMap, Module, ModuleError, MODULE_VERSION};
pub use node::{
    ArgList, BinOp, CmpOp, ConstId, ExprId, Node, Operands, ReduceOp, SymbolId, UnaryOp,
};
pub use nonlinearity::{nonlinearity, nonlinearity_of, Degree, Nonlinearity};
/// The exact constant field (feature `exact`).
#[cfg(feature = "exact")]
pub use num_rational::BigRational;
pub use role::{Crossing, OutputRole, ParamRole, Signature};
pub use scalar::Scalar;
pub use scope::Scope;
pub use semantics::{
    binary_f64, cmp_bool, dot_slice, reduce_slice, unary_f64, EXP_LIMIT, LN_FLOOR,
};
pub use simplify::rebuild;
#[cfg(feature = "egraph")]
pub use symbolic::simplify_egraph;
pub use symbolic::{collect, determinant, newton_step, rational_form};
pub use tape::{NoTrace, Program, SpecializedTape, Tape, TapeVisitor, TraceSink};

pub use transform::substitute;
