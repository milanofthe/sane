//! Symbolic analysis: what a symbolic circuit or system tool needs beyond
//! numeric evaluation, as transformations over a graph with an exact field.
//!
//! - [`det`]: symbolic determinants (Laplace expansion, structural skipping).
//! - [`poly`]: an expression as a polynomial in one symbol, rational
//!   canonical forms `N(s)/D(s)`, term expansion and numeric pruning.
//! - `egraph` (feature `egraph`): equality-saturation simplification (algebraic rewrites with
//!   constant folding as an e-graph analysis) for `Graph<BigRational>`.

pub mod det;
#[cfg(feature = "egraph")]
pub mod egraph;
pub mod poly;

pub use det::{count_det_terms, determinant};
#[cfg(feature = "egraph")]
pub use egraph::simplify_egraph;
pub use poly::{
    collect, expand_terms, poly_add, poly_mul, poly_to_expr, prune_poly, rational_form,
};
