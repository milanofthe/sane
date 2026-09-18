//! Symbolic analysis: what a symbolic circuit or system tool needs beyond
//! numeric evaluation, as transformations over a graph with an exact field.
//!
//! - [`det`]: symbolic determinants (Laplace expansion, structural skipping).
//! - [`poly`]: an expression as a polynomial in one symbol, rational
//!   canonical forms `N(s)/D(s)`, term expansion and numeric pruning.
//! - [`egraph`]: equality-saturation simplification (algebraic rewrites with
//!   constant folding as an e-graph analysis) for `Graph<BigRational>`.

pub mod det;
pub mod egraph;
pub mod poly;
pub mod solve;

pub use det::{count_det_terms, determinant};
pub use egraph::simplify_egraph;
pub use poly::{
    collect, expand_terms, poly_add, poly_mul, poly_to_expr, prune_poly, rational_form,
};
pub use solve::{
    lu_static, newton_step, newton_step_planned, pattern_of, plan, solve_planned, sparse_rows,
    NewtonStep, Pattern, Plan, Solved, StaticLu,
};
