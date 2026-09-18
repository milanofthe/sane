//! Configurable worker pool for SANE's data-parallel sweeps.
//!
//! The engine's outer loops that are embarrassingly parallel -- AC / noise over
//! a frequency grid, harmonic-balance device sampling over the period -- run on
//! one shared [`rayon::ThreadPool`] so the parallelism is bounded and explicit
//! rather than grabbing every core.
//!
//! The default is **4 worker threads**. Override it either with the
//! `SANE_THREADS` environment variable or programmatically via [`configure`]
//! (both must take effect before the pool is first used).
//!
//! Sweep parallelism is *outer*: each task solves its systems sequentially
//! (the graph solve's programs are sequential by construction), so the sweep
//! never nests with another pool and oversubscribes the machine.

use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Worker-thread count when neither `SANE_THREADS` nor [`configure`] is set.
pub const DEFAULT_THREADS: usize = 4;

static POOL: OnceLock<ThreadPool> = OnceLock::new();

/// The configured thread count (`Config::threads`), otherwise
/// [`DEFAULT_THREADS`].
fn requested_threads() -> usize {
    sane_core::config().threads.unwrap_or(DEFAULT_THREADS)
}

fn build(n: usize) -> ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(n.max(1))
        .thread_name(|i| format!("sane-worker-{i}"))
        .build()
        .expect("failed to build SANE worker pool")
}

/// Set the worker-thread count. Effective only if called before the pool is
/// first used (the first [`pool`]/[`install`] call); returns `false` if the
/// pool was already built, in which case the count is unchanged.
pub fn configure(threads: usize) -> bool {
    if cfg!(target_arch = "wasm32") {
        return false;
    }
    POOL.set(build(threads)).is_ok()
}

/// The shared worker pool, built on first use with the configured thread count.
pub fn pool() -> &'static ThreadPool {
    POOL.get_or_init(|| build(requested_threads()))
}

/// Number of worker threads in the (possibly lazily built) pool.
pub fn threads() -> usize {
    if cfg!(target_arch = "wasm32") {
        return 1;
    }
    pool().current_num_threads()
}

/// Run `f` on the shared worker pool. rayon parallel iterators created inside
/// `f` use this pool's threads.
///
/// On wasm32 there is no custom pool (a `ThreadPoolBuilder::build` would fail:
/// the browser sandbox cannot spawn threads without cross-origin isolation), so
/// `f` runs inline -- rayon's parallel iterators then execute sequentially on
/// the global pool's current-thread fallback. Same results, single-threaded.
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if cfg!(target_arch = "wasm32") {
        return f();
    }
    pool().install(f)
}
