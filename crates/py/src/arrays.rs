//! numpy at the boundary. A result's arrays reach Python without a copy: a
//! read-only numpy array over the result's own memory, which it keeps
//! alive. Inputs come in as any array-like.

use num_complex::Complex64;
use numpy::ndarray::{ArrayBase, ArrayView1, Data, Dimension};
use numpy::npyffi::flags::NPY_ARRAY_WRITEABLE;
use numpy::{AllowTypeChange, Element, PyArray, PyArray1, PyArrayLike1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// `a`, owned by `owner`, as a read-only numpy array over the same memory.
pub fn view<'py, T, S, D>(
    owner: &Bound<'py, PyAny>,
    a: &ArrayBase<S, D>,
) -> Bound<'py, PyArray<T, D>>
where
    T: Element,
    S: Data<Elem = T>,
    D: Dimension,
{
    // SAFETY: a result never changes its arrays once built, and the numpy
    // array holds `owner` (and so the memory) for as long as it lives.
    let arr = unsafe { PyArray::borrow_from_array_bound(a, owner.clone()) };
    unsafe { (*arr.as_array_ptr()).flags &= !NPY_ARRAY_WRITEABLE };
    arr
}

/// [`view`] of a vector.
pub fn view1<'py, T: Element>(owner: &Bound<'py, PyAny>, v: &[T]) -> Bound<'py, PyArray1<T>> {
    view(owner, &ArrayView1::from(v))
}

/// A vector handed over to numpy (moved, not copied).
pub fn vec1<T: Element>(py: Python<'_>, v: Vec<T>) -> Bound<'_, PyArray1<T>> {
    PyArray1::from_vec_bound(py, v)
}

/// An array-like of floats.
pub type Floats<'py> = PyArrayLike1<'py, f64, AllowTypeChange>;

/// An array-like of floats as a vector.
pub fn floats(a: &Floats<'_>) -> Vec<f64> {
    a.as_array().to_vec()
}

/// Complex values as Python understands them.
pub type C64 = Complex64;

/// Names: one string, or a sequence of them.
pub struct Names(pub Vec<String>);

impl<'py> FromPyObject<'py> for Names {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if ob.is_instance_of::<PyString>() {
            return Ok(Names(vec![ob.extract::<String>()?]));
        }
        ob.extract::<Vec<String>>()
            .map(Names)
            .map_err(|_| PyValueError::new_err("expected a name or a sequence of names"))
    }
}

impl Names {
    pub fn refs(&self) -> Vec<&str> {
        self.0.iter().map(|s| s.as_str()).collect()
    }
}

/// The names of an optional `wrt`: none for all.
pub fn wrt(w: &Option<Names>) -> Vec<&str> {
    w.as_ref().map_or_else(Vec::new, Names::refs)
}
