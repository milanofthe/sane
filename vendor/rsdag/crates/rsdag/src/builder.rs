//! Write a model once, run it two ways.
//!
//! A block's or a device's mathematics is a function of its inputs. Written
//! against [`Builder`], the same code either *computes* the value -- over
//! plain `f64`, through [`Numeric`], the hot path when nothing symbolic is
//! wanted -- or *records* it into a [`Graph`], where it can be
//! differentiated, specialized and compiled. One body, no drift between the
//! numeric model and its symbolic twin.
//!
//! The trait is the op vocabulary: the required methods are one per node
//! kind, and the named functions (`sin`, `atan2`, `min`, ...) are provided
//! on top of them, so an implementation is a dozen methods and a model reads
//! like arithmetic.
//!
//! ```
//! use rsdag::{Builder, Graph, Numeric, Scope, Tape, F64};
//!
//! /// A diode current, written once.
//! fn diode<B: Builder>(b: &mut B, v: B::N, is: B::N) -> B::N {
//!     let vt = b.cst(0.025);
//!     let e = { let q = b.div(v, vt); b.exp(q) };
//!     let one = b.cst(1.0);
//!     let m = b.sub(e, one);
//!     b.mul(is, m)
//! }
//!
//! // Computed:
//! let i = diode(&mut Numeric, 0.6, 1e-14);
//! // Recorded, then differentiated and compiled:
//! let mut g: Graph<F64> = Graph::new();
//! let mut s = Scope::new(&mut g, "diode");
//! let (v, is) = (s.param("v"), s.param("is"));
//! let expr = diode(&mut *s, v, is);
//! let f = s.close(vec![expr]);
//! # let _ = (i, f, Tape::compile(&g, &[expr], &[]));
//! ```
//!
//! Methods take `&mut self` because recording mutates the graph; a numeric
//! implementation simply ignores it.

use crate::field::Field;
use crate::graph::Graph;
use crate::node::{BinOp, CmpOp, ExprId, ReduceOp, UnaryOp};
use crate::semantics::{binary_f64, cmp_bool, dot_slice, reduce_slice, unary_f64};

/// The op vocabulary as a trait. See the module docs.
pub trait Builder {
    /// A value: `f64` when computing, [`ExprId`] when recording.
    type N: Copy;

    fn cst(&mut self, v: f64) -> Self::N;
    fn add(&mut self, a: Self::N, b: Self::N) -> Self::N;
    fn mul(&mut self, a: Self::N, b: Self::N) -> Self::N;
    fn neg(&mut self, a: Self::N) -> Self::N;
    /// `a^n` for an integer `n`; `n = -1` is the reciprocal.
    fn powi(&mut self, a: Self::N, n: i64) -> Self::N;
    fn unary(&mut self, op: UnaryOp, a: Self::N) -> Self::N;
    fn binary(&mut self, op: BinOp, a: Self::N, b: Self::N) -> Self::N;
    /// `1` or `0`.
    fn cmp(&mut self, op: CmpOp, a: Self::N, b: Self::N) -> Self::N;
    /// `cond != 0 ? t : e`.
    fn select(&mut self, cond: Self::N, t: Self::N, e: Self::N) -> Self::N;
    fn reduce(&mut self, op: ReduceOp, args: &[Self::N]) -> Self::N;
    /// `sum_i a[i] * b[i]`.
    fn dot(&mut self, a: &[Self::N], b: &[Self::N]) -> Self::N;
    /// `x` with `A x = b`, for a square `a` given by rows.
    ///
    /// The implicit step a block writes in the same code as its model: the
    /// numeric builder solves with a pivoted dense LU, the recording builder
    /// builds a static-pivot LU in a fill-reducing order as graph ops (see
    /// [`crate::symbolic::solve`]), so the recorded twin carries the solve
    /// and differentiates through it.
    fn solve(&mut self, a: &[Vec<Self::N>], b: &[Self::N]) -> Vec<Self::N>;

    // --- the ring, spelled the way a model reads ---------------------------

    fn sub(&mut self, a: Self::N, b: Self::N) -> Self::N {
        let nb = self.neg(b);
        self.add(a, nb)
    }
    /// `a / b`, as the graph builds it: `a * b^-1`, two roundings. The
    /// numeric builder does the same, so both paths agree to the bit.
    fn div(&mut self, a: Self::N, b: Self::N) -> Self::N {
        let r = self.powi(b, -1);
        self.mul(a, r)
    }
    fn recip(&mut self, a: Self::N) -> Self::N {
        self.powi(a, -1)
    }
    fn powf(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.binary(BinOp::Powf, a, b)
    }
    fn modulo(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.binary(BinOp::Mod, a, b)
    }
    fn atan2(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.binary(BinOp::Atan2, a, b)
    }
    fn hypot(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.binary(BinOp::Hypot, a, b)
    }
    fn min(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.reduce(ReduceOp::Min, &[a, b])
    }
    fn max(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.reduce(ReduceOp::Max, &[a, b])
    }
    fn sum(&mut self, args: &[Self::N]) -> Self::N {
        self.reduce(ReduceOp::Sum, args)
    }
    fn product(&mut self, args: &[Self::N]) -> Self::N {
        self.reduce(ReduceOp::Product, args)
    }

    // --- comparisons --------------------------------------------------------

    fn gt(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.cmp(CmpOp::Gt, a, b)
    }
    fn ge(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.cmp(CmpOp::Ge, a, b)
    }
    fn lt(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.cmp(CmpOp::Lt, a, b)
    }
    fn le(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.cmp(CmpOp::Le, a, b)
    }
    fn eq(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.cmp(CmpOp::Eq, a, b)
    }
    fn ne(&mut self, a: Self::N, b: Self::N) -> Self::N {
        self.cmp(CmpOp::Ne, a, b)
    }

    // --- the elementary functions, one line each -----------------------------

    fn exp(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Exp, a)
    }
    fn ln(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Ln, a)
    }
    fn log10(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Log10, a)
    }
    fn log2(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Log2, a)
    }
    fn log1p(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Log1p, a)
    }
    fn expm1(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Expm1, a)
    }
    fn sqrt(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Sqrt, a)
    }
    fn cbrt(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Cbrt, a)
    }
    fn sin(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Sin, a)
    }
    fn cos(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Cos, a)
    }
    fn tan(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Tan, a)
    }
    fn asin(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Asin, a)
    }
    fn acos(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Acos, a)
    }
    fn atan(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Atan, a)
    }
    fn sinh(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Sinh, a)
    }
    fn cosh(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Cosh, a)
    }
    fn tanh(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Tanh, a)
    }
    fn asinh(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Asinh, a)
    }
    fn acosh(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Acosh, a)
    }
    fn atanh(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Atanh, a)
    }
    fn abs(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Abs, a)
    }
    fn sign(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Sign, a)
    }
    fn floor(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Floor, a)
    }
    fn ceil(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Ceil, a)
    }
    fn round(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Round, a)
    }
    fn trunc(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Trunc, a)
    }
    fn erf(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Erf, a)
    }
    fn erfc(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Erfc, a)
    }
    fn lgamma(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Lgamma, a)
    }
    fn tgamma(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::Tgamma, a)
    }
    fn rand_uniform(&mut self, a: Self::N) -> Self::N {
        self.unary(UnaryOp::RandUniform, a)
    }
}

/// The builder that computes: every op is the reference `f64` routine the
/// interpreter and the compiled backends use, so a model run through it
/// gives the same value its recorded twin evaluates to.
#[derive(Clone, Copy, Debug, Default)]
pub struct Numeric;

impl Builder for Numeric {
    type N = f64;

    fn cst(&mut self, v: f64) -> f64 {
        v
    }
    fn add(&mut self, a: f64, b: f64) -> f64 {
        a + b
    }
    fn mul(&mut self, a: f64, b: f64) -> f64 {
        a * b
    }
    fn neg(&mut self, a: f64) -> f64 {
        -a
    }
    fn powi(&mut self, a: f64, n: i64) -> f64 {
        a.powi(n as i32)
    }
    fn unary(&mut self, op: UnaryOp, a: f64) -> f64 {
        unary_f64(op, a)
    }
    fn binary(&mut self, op: BinOp, a: f64, b: f64) -> f64 {
        binary_f64(op, a, b)
    }
    fn cmp(&mut self, op: CmpOp, a: f64, b: f64) -> f64 {
        if cmp_bool(op, a, b) {
            1.0
        } else {
            0.0
        }
    }
    fn select(&mut self, cond: f64, t: f64, e: f64) -> f64 {
        if cond != 0.0 {
            t
        } else {
            e
        }
    }
    fn reduce(&mut self, op: ReduceOp, args: &[f64]) -> f64 {
        reduce_slice(op, args)
    }
    fn dot(&mut self, a: &[f64], b: &[f64]) -> f64 {
        dot_slice(a, b)
    }
    fn solve(&mut self, a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let flat: Vec<f64> = a.iter().flatten().copied().collect();
        let mut out = vec![0.0; n];
        crate::semantics::solve(&flat, b, n, &mut out);
        out
    }
}

/// The builder that records: every op is the graph's smart constructor, so
/// the recorded twin folds, shares and canonicalises exactly as a graph
/// built by hand would. A [`crate::Scope`] derefs to its graph, so a model
/// records into a scope through `&mut *scope`.
impl<K: Field> Builder for Graph<K> {
    type N = ExprId;

    fn cst(&mut self, v: f64) -> ExprId {
        self.konst_f64(v)
    }
    fn add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        Graph::add(self, a, b)
    }
    fn mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        Graph::mul(self, a, b)
    }
    fn neg(&mut self, a: ExprId) -> ExprId {
        Graph::neg(self, a)
    }
    fn powi(&mut self, a: ExprId, n: i64) -> ExprId {
        self.pow_i(a, n)
    }
    fn unary(&mut self, op: UnaryOp, a: ExprId) -> ExprId {
        Graph::unary(self, op, a)
    }
    fn binary(&mut self, op: BinOp, a: ExprId, b: ExprId) -> ExprId {
        Graph::binary(self, op, a, b)
    }
    fn cmp(&mut self, op: CmpOp, a: ExprId, b: ExprId) -> ExprId {
        Graph::cmp(self, op, a, b)
    }
    fn select(&mut self, cond: ExprId, t: ExprId, e: ExprId) -> ExprId {
        Graph::select(self, cond, t, e)
    }
    fn reduce(&mut self, op: ReduceOp, args: &[ExprId]) -> ExprId {
        Graph::reduce(self, op, args.to_vec())
    }
    fn dot(&mut self, a: &[ExprId], b: &[ExprId]) -> ExprId {
        Graph::dot(self, a.to_vec(), b.to_vec())
    }
    fn solve(&mut self, a: &[Vec<ExprId>], b: &[ExprId]) -> Vec<ExprId> {
        // A dense matrix is one pivoting kernel; a sparse one is the
        // structural solve, its static LU as ops.
        use crate::symbolic::solve::{pattern_of, plan, solve_planned, sparse_rows};
        let rows = sparse_rows(self, a);
        let n = b.len();
        let nnz: usize = rows.iter().map(Vec::len).sum();
        // A structurally singular system has no static LU; it goes to the
        // dense kernel too, which is exactly what the numeric builder runs
        // on it, so both paths still agree to the bit.
        let plan = (nnz < n * n).then(|| plan(&pattern_of(&rows))).flatten();
        match plan {
            Some(plan) => solve_planned(self, &rows, &plan, b).x,
            None => Graph::solve_dense(self, a.iter().flatten().copied().collect(), b.to_vec()),
        }
    }
}
