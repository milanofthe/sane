//! Buffers a thread keeps between calls, so an evaluation in a loop
//! allocates on its first round only. One stack per thread of values of
//! any type: a call takes the newest of its type (or a new one) and puts it
//! back, so a body that calls a body, or runs a stage of its own, takes one
//! of its own.

use std::any::Any;
use std::cell::RefCell;

thread_local! {
    static FREE: RefCell<Vec<Box<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` on a `T` of this thread's.
pub fn with<T: Default + 'static, R>(f: impl FnOnce(&mut T) -> R) -> R {
    let mut buf: Box<T> = FREE
        .with(|s| {
            let mut s = s.borrow_mut();
            let at = s.iter().rposition(|b| b.is::<T>())?;
            s.swap_remove(at).downcast().ok()
        })
        .unwrap_or_default();
    let r = f(&mut buf);
    FREE.with(|s| s.borrow_mut().push(buf));
    r
}

/// Run `f` on `len` values of this thread's, `zero` where newly grown.
pub fn with_len<T: Copy + 'static, R>(len: usize, zero: T, f: impl FnOnce(&mut [T]) -> R) -> R {
    with(|b: &mut Vec<T>| {
        if b.len() < len {
            b.resize(len, zero);
        }
        f(&mut b[..len])
    })
}
