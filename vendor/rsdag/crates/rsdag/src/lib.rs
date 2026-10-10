//! Rust symbolic graph backend: a hash-consed expression DAG over free
//! symbols and exact or floating constants, with symbolic differentiation and
//! a flat tape and its interpreter. The shared substrate of SANE and fastsim,
//! so an optimization here lands in every consumer.

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
pub mod module;
pub mod node;
pub mod nonlinearity;
pub mod parallel;
pub mod role;
pub mod scalar;
pub mod scope;
pub mod scratch;
pub mod semantics;
mod simd;
#[cfg(any(test, feature = "synth"))]
pub mod synth;
pub mod tape;
pub mod variant;
// The rewrite machinery stays internal; substitution is exposed through the
// `substitute` re-export below, not as a module path.
pub(crate) mod transform;

pub use adaptive::{Adaptive, Compiler, Episode, Policy, Stats};
pub use autodiff::{differentiate, gradient, sparse_jacobian, SparseRows};
pub use builder::{Builder, Numeric};
pub use display::to_string;
pub use eval::eval;
pub use extern_fn::{BackendCache, BodyBackend, BodyCompiler, ExternBundle, Instances, Submit};
pub use field::{Field, F64};
pub use func::{Body, FuncId, Function, Output, OutputId};
pub use graph::{Bound, Graph};
pub use module::{IdMap, Module, ModuleError, MODULE_VERSION};
pub use node::{
    ArgList, BinOp, CmpOp, ConstId, ExprId, Node, Operands, ReduceOp, SymbolId, UnaryOp,
};
pub use nonlinearity::{nonlinearity, nonlinearity_of, Degree, Nonlinearity, UnarySet};
/// The exact constant field (feature `exact`).
#[cfg(feature = "exact")]
pub use num_rational::BigRational;
pub use role::{Crossing, OutputRole, ParamRole, Signature};
pub use scalar::Scalar;
pub use scope::Scope;
pub use semantics::{
    binary_f64, cmp_bool, dot_slice, reduce_slice, unary_f64, EXP_LIMIT, LN_FLOOR,
};
pub use tape::{Lowered, NoTrace, ParamSelects, Program, SpecializedTape, Tape, TraceSink};

pub use transform::substitute;
