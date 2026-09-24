//! Python bindings for SANE's symbolic core: the hash-consed expression DAG.
//!
//! Exposes a shared [`Context`] (the DAG arena + symbol table) and [`Expr`]
//! handles into it. `Expr` overloads the Python arithmetic operators, so a
//! symbolic expression reads like ordinary Python:
//!
//! ```python
//! x, y = sane.symbols("x y")
//! f = 2 * x + (y ** 2).exp()
//! df = f.diff(x)
//! ```
//!
//! Every `Expr` carries a refcounted handle to the `Context` it lives in;
//! mixing `Expr`s from different contexts raises.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyTuple};

use rsdag::simplify_egraph as core_simplify;
use rsdag::{
    differentiate as core_diff, eval as core_eval, to_string, ExprId, Node, SymbolId,
    Tape as CoreTape,
};
use sane_core::Graph as CoreCtx;

// CONTEXT ==============================================================================

/// Poison-recovering lock for the shared symbolic arena.
///
/// A panic while the mutex is held (e.g. a numerical edge case deep in a
/// solver) poisons a `std::sync::Mutex`; with a plain `.lock().unwrap()` every
/// subsequent call on the handle would then re-panic, permanently bricking the
/// Python object. Arena mutations are append-only interning, so a panic never
/// leaves the graph structurally broken -- recovering the guard is safe and
/// keeps the handle usable after the original panic surfaced to Python.
pub trait LockCtx {
    fn lock_ctx(&self) -> MutexGuard<'_, CoreCtx>;
}

impl LockCtx for Mutex<CoreCtx> {
    fn lock_ctx(&self) -> MutexGuard<'_, CoreCtx> {
        self.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A hash-consed symbolic DAG arena and symbol table. Build expressions with
/// [`sym`](Context::sym) / [`symbols`](Context::symbols) / [`const`] and the
/// arithmetic operators on the returned [`Expr`] handles.
///
/// The arena lives behind `Arc<Mutex<..>>` so the *same* graph can be shared
/// between this Python handle and a Rust-side owner (e.g. `sane_analysis::Model`):
/// a `Model` hands out `Expr` that operate on the very graph it solves on.
#[pyclass]
pub struct Context {
    pub inner: Arc<Mutex<CoreCtx>>,
}

impl Context {
    /// Wrap a shared core context (e.g. from `Model::context_arc`) as a Python
    /// `Context`, so expression handles operate on that same graph.
    pub fn from_arc(inner: Arc<Mutex<CoreCtx>>) -> Self {
        Context { inner }
    }

    /// Lock the shared arena for mutation/inspection, recovering from poisoning.
    pub fn lock(&self) -> MutexGuard<'_, CoreCtx> {
        self.inner.lock_ctx()
    }
}

#[pymethods]
impl Context {
    #[new]
    fn new() -> Self {
        Context {
            inner: Arc::new(Mutex::new(CoreCtx::new())),
        }
    }

    /// Create (or look up) a free symbol named `name`.
    fn sym(slf: Bound<'_, Context>, name: &str) -> Expr {
        let id = slf.borrow().inner.lock_ctx().sym(name);
        Expr::new(slf.unbind(), id)
    }

    /// Create several symbols at once from a space-separated string
    /// (`"x y z"`). Returns a list of [`Expr`].
    fn symbols(slf: Bound<'_, Context>, names: &str) -> Vec<Expr> {
        let py = slf.py();
        let ctx = slf.unbind();
        let mut out = Vec::new();
        for n in names.split_whitespace() {
            let id = ctx.borrow(py).inner.lock_ctx().sym(n);
            out.push(Expr::new(ctx.clone_ref(py), id));
        }
        out
    }

    /// A constant expression from an int or float.
    #[pyo3(name = "const")]
    fn konst(slf: Bound<'_, Context>, value: Bound<'_, PyAny>) -> PyResult<Expr> {
        let id = {
            let c = slf.borrow();
            let mut c = c.inner.lock_ctx();
            if let Ok(i) = value.extract::<i64>() {
                c.konst_int(i)
            } else if let Ok(f) = value.extract::<f64>() {
                c.konst_f64(f)
            } else {
                return Err(PyValueError::new_err("const expects an int or float"));
            }
        };
        Ok(Expr::new(slf.unbind(), id))
    }

    /// The number of interned nodes in the DAG.
    fn __len__(&self) -> usize {
        self.inner.lock_ctx().len()
    }
}

// EXPR =================================================================================

/// A handle to a node in a [`Context`]'s expression DAG. Supports the Python
/// arithmetic operators (`+ - * / **`, unary `-`) with other `Expr`s and with
/// Python ints / floats, plus the elementary functions as methods.
#[pyclass]
pub struct Expr {
    ctx: Py<Context>,
    id: u32,
}

impl Expr {
    pub(crate) fn new(ctx: Py<Context>, id: ExprId) -> Self {
        Expr { ctx, id: id.0 }
    }

    fn same(&self, py: Python<'_>, id: ExprId) -> Expr {
        Expr {
            ctx: self.ctx.clone_ref(py),
            id: id.0,
        }
    }

    /// Resolve a list of `wrt` handles to symbol ids, checking they are symbols
    /// from this expression's context (shared by `grad` / `hess`).
    fn symbol_ids(&self, py: Python<'_>, wrt: &[PyRef<Expr>]) -> PyResult<Vec<SymbolId>> {
        let cg = self.ctx.borrow(py);
        let c = cg.inner.lock_ctx();
        let mut syms = Vec::with_capacity(wrt.len());
        for w in wrt {
            if !w.ctx.is(&self.ctx) {
                return Err(PyValueError::new_err("`wrt` is from a different Context"));
            }
            match c.node(ExprId(w.id)) {
                Node::Symbol(s) => syms.push(*s),
                _ => return Err(PyValueError::new_err("`wrt` entries must be symbols")),
            }
        }
        Ok(syms)
    }

    /// Resolve `other` (an `Expr` in the same context, or an int / float) to an
    /// `ExprId`, creating a constant node for scalars.
    fn coerce(
        &self,
        py: Python<'_>,
        c: &mut CoreCtx,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<ExprId> {
        if let Ok(e) = other.extract::<PyRef<Expr>>() {
            if !e.ctx.is(&self.ctx) {
                return Err(PyValueError::new_err(
                    "cannot combine Expr from a different Context",
                ));
            }
            return Ok(ExprId(e.id));
        }
        if let Ok(i) = other.extract::<i64>() {
            return Ok(c.konst_int(i));
        }
        if let Ok(f) = other.extract::<f64>() {
            return Ok(c.konst_f64(f));
        }
        let _ = py;
        Err(PyValueError::new_err("expected Expr, int or float"))
    }
}

#[pymethods]
impl Expr {
    // --- arithmetic operators ---------------------------------------------

    fn __add__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Expr> {
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let b = self.coerce(py, &mut c, &other)?;
        let r = c.add(ExprId(self.id), b);
        drop(c);
        drop(cg);
        Ok(self.same(py, r))
    }

    fn __radd__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Expr> {
        self.__add__(py, other)
    }

    fn __sub__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Expr> {
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let b = self.coerce(py, &mut c, &other)?;
        let r = c.sub(ExprId(self.id), b);
        drop(c);
        drop(cg);
        Ok(self.same(py, r))
    }

    fn __rsub__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Expr> {
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let b = self.coerce(py, &mut c, &other)?;
        let r = c.sub(b, ExprId(self.id));
        drop(c);
        drop(cg);
        Ok(self.same(py, r))
    }

    fn __mul__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Expr> {
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let b = self.coerce(py, &mut c, &other)?;
        let r = c.mul(ExprId(self.id), b);
        drop(c);
        drop(cg);
        Ok(self.same(py, r))
    }

    fn __rmul__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Expr> {
        self.__mul__(py, other)
    }

    fn __truediv__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Expr> {
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let b = self.coerce(py, &mut c, &other)?;
        let r = c.div(ExprId(self.id), b);
        drop(c);
        drop(cg);
        Ok(self.same(py, r))
    }

    fn __rtruediv__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Expr> {
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let b = self.coerce(py, &mut c, &other)?;
        let r = c.div(b, ExprId(self.id));
        drop(c);
        drop(cg);
        Ok(self.same(py, r))
    }

    fn __neg__(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().neg(ExprId(self.id));
        self.same(py, r)
    }

    /// `self ** exponent`. An integer exponent uses the exact integer-power
    /// node; a symbolic or fractional exponent lowers to ``exp(exponent*ln(self))``.
    fn __pow__(
        &self,
        py: Python<'_>,
        exponent: Bound<'_, PyAny>,
        _modulo: Bound<'_, PyAny>,
    ) -> PyResult<Expr> {
        if let Ok(n) = exponent.extract::<i64>() {
            let r = self
                .ctx
                .borrow(py)
                .inner
                .lock_ctx()
                .pow_i(ExprId(self.id), n);
            return Ok(self.same(py, r));
        }
        // a ** b  =  exp(b * ln(a))
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let b = self.coerce(py, &mut c, &exponent)?;
        let la = c.ln(ExprId(self.id));
        let bla = c.mul(b, la);
        let r = c.exp(bla);
        drop(c);
        drop(cg);
        Ok(self.same(py, r))
    }

    /// `base ** self` for an `Expr` exponent: lowers to ``exp(self*ln(base))``.
    fn __rpow__(
        &self,
        py: Python<'_>,
        base: Bound<'_, PyAny>,
        _modulo: Bound<'_, PyAny>,
    ) -> PyResult<Expr> {
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let a = self.coerce(py, &mut c, &base)?;
        let la = c.ln(a);
        let bla = c.mul(ExprId(self.id), la);
        let r = c.exp(bla);
        drop(c);
        drop(cg);
        Ok(self.same(py, r))
    }

    // --- elementary functions ---------------------------------------------

    fn exp(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().exp(ExprId(self.id));
        self.same(py, r)
    }
    fn ln(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().ln(ExprId(self.id));
        self.same(py, r)
    }
    fn sqrt(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().sqrt(ExprId(self.id));
        self.same(py, r)
    }
    fn sin(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().sin(ExprId(self.id));
        self.same(py, r)
    }
    fn cos(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().cos(ExprId(self.id));
        self.same(py, r)
    }
    fn sinh(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().sinh(ExprId(self.id));
        self.same(py, r)
    }
    fn cosh(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().cosh(ExprId(self.id));
        self.same(py, r)
    }
    fn tanh(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().tanh(ExprId(self.id));
        self.same(py, r)
    }
    fn floor(&self, py: Python<'_>) -> Expr {
        let r = self.ctx.borrow(py).inner.lock_ctx().floor(ExprId(self.id));
        self.same(py, r)
    }

    // --- calculus / simplify ----------------------------------------------

    /// Partial derivative w.r.t. the symbol `wrt` (an `Expr` that is a symbol).
    fn diff(&self, py: Python<'_>, wrt: PyRef<Expr>) -> PyResult<Expr> {
        if !wrt.ctx.is(&self.ctx) {
            return Err(PyValueError::new_err("`wrt` is from a different Context"));
        }
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let s = match c.node(ExprId(wrt.id)) {
            Node::Symbol(s) => *s,
            _ => return Err(PyValueError::new_err("`wrt` must be a symbol")),
        };
        let r = core_diff(&mut c, ExprId(self.id), s);
        drop(c);
        drop(cg);
        Ok(self.same(py, r))
    }

    /// Gradient w.r.t. a list of symbols, via ONE reverse-mode (adjoint) sweep
    /// over the DAG -- the right shape for a scalar over many leaves. Returns
    /// one `Expr` per symbol; each is an ordinary expression that can be
    /// differentiated again (`f.grad([x, y])[0].diff(x)` is a second
    /// derivative, and so on to any order).
    fn grad(&self, py: Python<'_>, wrt: Vec<PyRef<Expr>>) -> PyResult<Vec<Expr>> {
        let syms = self.symbol_ids(py, &wrt)?;
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let g = rsdag::gradient(&mut c, ExprId(self.id), &syms);
        drop(c);
        drop(cg);
        Ok(g.into_iter().map(|r| self.same(py, r)).collect())
    }

    /// Symbolic Hessian `d2f/d(wrt[i])d(wrt[j])` (forward-over-reverse).
    fn hess(&self, py: Python<'_>, wrt: Vec<PyRef<Expr>>) -> PyResult<Vec<Vec<Expr>>> {
        let syms = self.symbol_ids(py, &wrt)?;
        let cg = self.ctx.borrow(py);
        let mut c = cg.inner.lock_ctx();
        let h = rsdag::hessian(&mut c, ExprId(self.id), &syms);
        drop(c);
        drop(cg);
        Ok(h.into_iter()
            .map(|row| row.into_iter().map(|r| self.same(py, r)).collect())
            .collect())
    }

    /// Algebraic simplification via equality saturation (egg). Defined on the
    /// linear-algebra fragment; raises on comparison / select / opaque nodes.
    fn simplify(&self, py: Python<'_>) -> PyResult<Expr> {
        let id = ExprId(self.id);
        let ctx_cell = &self.ctx;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            core_simplify(&mut ctx_cell.borrow(py).inner.lock_ctx(), id)
        }))
        .map_err(|_| {
            PyValueError::new_err(
                "simplify is not defined on this expression (comparison/select/opaque)",
            )
        })?;
        Ok(self.same(py, r))
    }

    /// Render this expression as a LaTeX string (e.g. for a transfer function
    /// or an approximated `H(s)`). Call ``.simplify().latex()`` for a compact form.
    fn latex(&self, py: Python<'_>) -> String {
        sane_export::latex_expr(&self.ctx.borrow(py).inner.lock_ctx(), ExprId(self.id))
    }

    /// Render the hash-consed DAG of this expression as a Graphviz DOT string
    /// (feed to ``dot -Tpdf``). Shared subexpressions appear as a single node
    /// with several incoming edges, so common-subexpression sharing is visible.
    ///
    /// ``label`` names this root. ``others`` is an optional list of further
    /// ``Expr`` roots in the same context (with ``labels``), drawn in the same
    /// graph -- e.g. a residual together with its symbolic derivative, to show
    /// the structure forward-mode differentiation reuses.
    ///
    /// ``highlight`` is an optional list of ``Expr`` (same context): only the
    /// nodes reachable from them are drawn at full opacity, and every other node
    /// and the edges touching it are faded to alpha ~0.2 -- handy for showing
    /// exactly which nodes a transform (e.g. the Newton g_min shift) adds.
    #[pyo3(signature = (label="F", others=None, labels=None, highlight=None))]
    fn to_dot(
        &self,
        py: Python<'_>,
        label: &str,
        others: Option<Vec<PyRef<Expr>>>,
        labels: Option<Vec<String>>,
        highlight: Option<Vec<PyRef<Expr>>>,
    ) -> PyResult<String> {
        let mut roots: Vec<(ExprId, String)> = vec![(ExprId(self.id), label.to_string())];
        if let Some(extra) = others {
            let names = labels.unwrap_or_default();
            for (i, e) in extra.iter().enumerate() {
                if !e.ctx.is(&self.ctx) {
                    return Err(PyValueError::new_err(
                        "every Expr must share the same Context",
                    ));
                }
                let name = names.get(i).cloned().unwrap_or_default();
                roots.push((ExprId(e.id), name));
            }
        }
        let mut hi: Vec<ExprId> = Vec::new();
        if let Some(hs) = highlight {
            for e in &hs {
                if !e.ctx.is(&self.ctx) {
                    return Err(PyValueError::new_err(
                        "every highlight Expr must share the same Context",
                    ));
                }
                hi.push(ExprId(e.id));
            }
        }
        Ok(sane_export::export_dot(
            &self.ctx.borrow(py).inner.lock_ctx(),
            &roots,
            &hi,
        ))
    }

    // --- evaluation -------------------------------------------------------

    /// Evaluate to a real number, binding free symbols by name via keyword
    /// arguments: ``expr.eval(x=1.0, y=2.0)``. Raises if a free symbol is unbound.
    #[pyo3(signature = (**kwargs))]
    fn eval(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<f64> {
        let cg = self.ctx.borrow(py);
        let c = cg.inner.lock_ctx();
        let env = self.real_env(&c, kwargs)?;
        let out = rsdag::eval(&c, &[ExprId(self.id)], &env);
        let v = out[0];
        if v.is_nan() {
            return Err(PyValueError::new_err(
                "evaluation produced NaN (unbound symbol or opaque node)",
            ));
        }
        Ok(v)
    }

    /// Evaluate to a complex number (the s-domain / AC case). Keyword values may
    /// be real or complex: ``H.eval_complex(s=1j*omega, R=1e3, C=1e-9)``.
    #[pyo3(signature = (**kwargs))]
    fn eval_complex(
        &self,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let cg = self.ctx.borrow(py);
        let c = cg.inner.lock_ctx();
        let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
        for s in c.free_symbols(ExprId(self.id)) {
            let name = c.symbol_name(s).to_string();
            let v = match kwargs.and_then(|d| d.get_item(name.as_str()).ok().flatten()) {
                Some(v) => extract_complex(&v)?,
                None => return Err(PyValueError::new_err(format!("unbound symbol '{name}'"))),
            };
            env.insert(s, v);
        }
        let z = core_eval(&c, &[ExprId(self.id)], &env)[0];
        Ok(PyComplex::from_doubles_bound(py, z.re, z.im).into())
    }

    // --- introspection ----------------------------------------------------

    /// The free symbols of this expression, as names.
    #[getter]
    fn free_symbols(&self, py: Python<'_>) -> Vec<String> {
        let cg = self.ctx.borrow(py);
        let c = cg.inner.lock_ctx();
        c.free_symbols(ExprId(self.id))
            .into_iter()
            .map(|s| c.symbol_name(s).to_string())
            .collect()
    }

    /// The [`Context`] this expression lives in (to create more symbols /
    /// constants compatible with it).
    #[getter]
    fn context(&self, py: Python<'_>) -> Py<Context> {
        self.ctx.clone_ref(py)
    }

    fn is_zero(&self, py: Python<'_>) -> bool {
        self.ctx
            .borrow(py)
            .inner
            .lock_ctx()
            .is_zero(ExprId(self.id))
    }
    fn is_one(&self, py: Python<'_>) -> bool {
        self.ctx.borrow(py).inner.lock_ctx().is_one(ExprId(self.id))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        to_string(&self.ctx.borrow(py).inner.lock_ctx(), ExprId(self.id))
    }
}

impl Expr {
    /// Build the real evaluation environment from kwargs, covering exactly the
    /// free symbols of this expression.
    fn real_env(
        &self,
        c: &CoreCtx,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<HashMap<SymbolId, f64>> {
        let mut env = HashMap::new();
        for s in c.free_symbols(ExprId(self.id)) {
            let name = c.symbol_name(s).to_string();
            let v = match kwargs.and_then(|d| d.get_item(name.as_str()).ok().flatten()) {
                Some(v) => v.extract::<f64>()?,
                None => return Err(PyValueError::new_err(format!("unbound symbol '{name}'"))),
            };
            env.insert(s, v);
        }
        Ok(env)
    }
}

fn extract_complex(v: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    if let Ok(f) = v.extract::<f64>() {
        return Ok(Complex64::new(f, 0.0));
    }
    if let Ok(z) = v.downcast::<PyComplex>() {
        return Ok(Complex64::new(z.real(), z.imag()));
    }
    Err(PyValueError::new_err("expected a real or complex value"))
}

// MODULE-LEVEL HELPERS =================================================================

/// Create symbols in a fresh [`Context`]: ``x, y = sane.symbols("x y")``.
/// Returns a single [`Expr`] for one name, else a list.
#[pyfunction]
pub fn symbols(py: Python<'_>, names: &str) -> PyResult<PyObject> {
    let ctx = Py::new(py, Context::new())?;
    let mut out = Vec::new();
    for n in names.split_whitespace() {
        let id = ctx.borrow(py).inner.lock_ctx().sym(n);
        out.push(Expr::new(ctx.clone_ref(py), id));
    }
    let objs: Vec<PyObject> = out.into_iter().map(|e| e.into_py(py)).collect();
    if objs.len() == 1 {
        Ok(objs.into_iter().next().unwrap())
    } else {
        Ok(PyTuple::new_bound(py, objs).into())
    }
}

/// The symbolic Jacobian `d(residuals[i])/d(wrt[j])`. All expressions must share
/// one context; `wrt` must be symbols.
#[pyfunction]
pub fn jacobian(
    py: Python<'_>,
    residuals: Vec<PyRef<Expr>>,
    wrt: Vec<PyRef<Expr>>,
) -> PyResult<Vec<Vec<Expr>>> {
    if residuals.is_empty() {
        return Ok(vec![]);
    }
    let ctxobj = residuals[0].ctx.clone_ref(py);
    for e in residuals.iter().chain(wrt.iter()) {
        if !e.ctx.is(&ctxobj) {
            return Err(PyValueError::new_err(
                "all expressions must share one Context",
            ));
        }
    }
    let rs: Vec<ExprId> = residuals.iter().map(|e| ExprId(e.id)).collect();
    let cg = ctxobj.borrow(py);
    let mut c = cg.inner.lock_ctx();
    let mut syms = Vec::with_capacity(wrt.len());
    for w in &wrt {
        match c.node(ExprId(w.id)) {
            Node::Symbol(s) => syms.push(*s),
            _ => return Err(PyValueError::new_err("`wrt` entries must be symbols")),
        }
    }
    let rows = rsdag::sparse_jacobian(&mut c, &rs, &syms);
    let zero = c.zero();
    let jac: Vec<Vec<ExprId>> = rows
        .into_iter()
        .map(|row| {
            let mut dense = vec![zero; syms.len()];
            for (j, e) in row {
                dense[j] = e;
            }
            dense
        })
        .collect();
    drop(c);
    drop(cg);
    Ok(jac
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|id| Expr::new(ctxobj.clone_ref(py), id))
                .collect()
        })
        .collect())
}

/// The structural sparsity pattern of a symbolic Jacobian (`True` where the
/// entry is not identically zero).
#[pyfunction]
pub fn sparsity(py: Python<'_>, jac: Vec<Vec<PyRef<Expr>>>) -> PyResult<Vec<Vec<bool>>> {
    if jac.is_empty() {
        return Ok(vec![]);
    }
    let ctxobj = jac[0][0].ctx.clone_ref(py);
    let ids: Vec<Vec<ExprId>> = jac
        .iter()
        .map(|row| row.iter().map(|e| ExprId(e.id)).collect())
        .collect();
    let arc = ctxobj.borrow(py).inner.clone();
    let c = arc.lock_ctx();
    Ok(ids
        .iter()
        .map(|row| row.iter().map(|&e| !c.is_zero(e)).collect())
        .collect())
}

// TAPE =================================================================================

/// A compiled flat evaluator over a fixed list of input symbols. Built with
/// [`compile`](crate::symbolic::compile); evaluate with [`eval`](Tape::eval).
#[pyclass]
pub struct Tape {
    inner: CoreTape,
    n_in: usize,
}

#[pymethods]
impl Tape {
    /// Evaluate the compiled roots at the given input values (in the order the
    /// input symbols were given to `compile`). Returns one value per root.
    fn eval(&self, inputs: Vec<f64>) -> PyResult<Vec<f64>> {
        if inputs.len() != self.n_in {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.n_in,
                inputs.len()
            )));
        }
        let mut work = Vec::new();
        let mut out = Vec::new();
        self.inner.eval(&inputs, &mut work, &mut out);
        Ok(out)
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.inner.n_outputs()
    }
}

/// Compile a list of root expressions into a fast [`Tape`] over the named input
/// symbols. Symbols outside `inputs` evaluate to NaN.
#[pyfunction]
pub fn compile(
    py: Python<'_>,
    roots: Vec<PyRef<Expr>>,
    inputs: Vec<PyRef<Expr>>,
) -> PyResult<Tape> {
    if roots.is_empty() {
        return Err(PyValueError::new_err("compile needs at least one root"));
    }
    let ctxobj = roots[0].ctx.clone_ref(py);
    let cg = ctxobj.borrow(py);
    let c = cg.inner.lock_ctx();
    let root_ids: Vec<ExprId> = roots.iter().map(|e| ExprId(e.id)).collect();
    let mut in_syms = Vec::with_capacity(inputs.len());
    for e in &inputs {
        match c.node(ExprId(e.id)) {
            Node::Symbol(s) => in_syms.push(*s),
            _ => return Err(PyValueError::new_err("`inputs` entries must be symbols")),
        }
    }
    let tape = CoreTape::compile(&c, &root_ids, &in_syms);
    Ok(Tape {
        inner: tape,
        n_in: inputs.len(),
    })
}

// REGISTRATION =========================================================================

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Context>()?;
    m.add_class::<Expr>()?;
    m.add_class::<Tape>()?;
    m.add_function(wrap_pyfunction!(symbols, m)?)?;
    m.add_function(wrap_pyfunction!(jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(sparsity, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    Ok(())
}
