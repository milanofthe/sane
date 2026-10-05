//! Configurable worker pools for SANE's parallel work.
//!
//! The engine's outer loops that are embarrassingly parallel -- AC / noise over
//! a frequency grid, harmonic-balance device sampling over the period -- run on
//! one shared [`rayon::ThreadPool`] so the parallelism is bounded and explicit
//! rather than grabbing every core. The solvers ([`solve`]) run the device
//! instances of each evaluation on rsdag's [`Workers`](rsdag::parallel::Workers)
//! (`rsdag::parallel`), bit for bit as serially; its workers stay awake across
//! the short serial stretches of a Newton loop.
//!
//! The default is **4 threads**, in each pool. Override it either with the
//! `SANE_THREADS` environment variable or programmatically via [`configure`]
//! (both must take effect before the pools are first used).
//!
//! Sweep parallelism is *outer*: each task solves its systems sequentially
//! (rslab's KLU is sequential, and inside a sweep no solver runs its devices
//! on the workers), so the two never nest and oversubscribe the machine.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use rayon::{ThreadPool, ThreadPoolBuilder};
use rsdag::parallel::{Parallel, Workers};

/// Worker-thread count when neither `SANE_THREADS` nor [`configure`] is set.
pub const DEFAULT_THREADS: usize = 4;

static POOL: OnceLock<ThreadPool> = OnceLock::new();
static WORKERS: OnceLock<Arc<Workers>> = OnceLock::new();
/// The count [`configure`] set (0: none).
static CONFIGURED: AtomicUsize = AtomicUsize::new(0);

/// The configured thread count: [`configure`], `Config::threads`, otherwise
/// [`DEFAULT_THREADS`].
fn requested_threads() -> usize {
    match CONFIGURED.load(Ordering::Relaxed) {
        0 => sane_core::config().threads.unwrap_or(DEFAULT_THREADS),
        n => n,
    }
}

fn build(n: usize) -> ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(n.max(1))
        .thread_name(|i| format!("sane-worker-{i}"))
        .build()
        .expect("failed to build SANE worker pool")
}

/// Set the thread count. Effective only if called before the pools are
/// first used; returns `false` if one was already built, in which case the
/// count is unchanged.
pub fn configure(threads: usize) -> bool {
    if cfg!(target_arch = "wasm32") || WORKERS.get().is_some() {
        return false;
    }
    CONFIGURED.store(threads.max(1), Ordering::Relaxed);
    POOL.set(build(threads)).is_ok()
}

/// The shared sweep pool, built on first use with the configured thread count.
pub fn pool() -> &'static ThreadPool {
    POOL.get_or_init(|| build(requested_threads()))
}

/// The solvers' workers, built on first use with the configured thread count.
fn workers() -> &'static Arc<Workers> {
    WORKERS.get_or_init(|| Arc::new(Workers::new(requested_threads())))
}

/// The configured number of threads.
pub fn threads() -> usize {
    if cfg!(target_arch = "wasm32") {
        return 1;
    }
    requested_threads()
}

/// Run `f` on the shared sweep pool. rayon parallel iterators created inside
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

/// Run the solver `f` with the workers lent to rsdag: the programs `f`
/// evaluates run their independent calls (the device instances) on them
/// (`rsdag::parallel`), with the same results as serially. Serial with one
/// thread, inside another `solve`, and on wasm32.
pub fn solve<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if cfg!(target_arch = "wasm32") || rsdag::parallel::current().is_some() || threads() < 2 {
        return f();
    }
    rsdag::parallel::install(Parallel::new(workers().clone()), f)
}
