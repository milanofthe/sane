//! The background compiler: one thread that takes compile jobs in order,
//! and the thread pool the jobs (and a consumer's own compiles) run on.
//!
//! Compiling is long, unsplittable work (tens of milliseconds per chunk).
//! On the global rayon pool it would starve the latency-bound work that
//! runs there: a worker waiting in a `join` steals a compile chunk and
//! holds the caller's critical path for its duration. Compiles therefore
//! run on a pool of their own, half the machine by default, so a compile
//! burst never owns every core either.

use std::sync::mpsc::{channel, Sender};
use std::sync::OnceLock;

type Job = Box<dyn FnOnce() + Send>;

/// The compile pool: half the cores (at least one) unless
/// [`set_threads`] ran first.
pub fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        let n = THREADS.get().copied().unwrap_or_else(|| {
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            (cores / 2).max(1)
        });
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("rsdag-jit-{i}"))
            .build()
            .expect("failed to build the compile pool")
    })
}

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
static THREADS: OnceLock<usize> = OnceLock::new();

/// Size the compile pool before its first use; `false` when it is too
/// late (the pool exists, or a size was set already).
pub fn set_threads(n: usize) -> bool {
    POOL.get().is_none() && THREADS.set(n.max(1)).is_ok()
}

/// Run `job` on the compile pool, after every job submitted before it.
/// The submission is one channel send, so it costs the caller nothing
/// measurable; a panic in the job is caught and ends only the job.
pub fn submit(job: impl FnOnce() + Send + 'static) {
    let _ = queue().send(Box::new(job));
}

/// Run `job` on the compile pool now, beside whatever the queue is running
/// (for independent compiles, a consumer's function bodies say).
pub fn spawn(job: impl FnOnce() + Send + 'static) {
    pool().spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
    });
}

fn queue() -> &'static Sender<Job> {
    static Q: OnceLock<Sender<Job>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = channel::<Job>();
        // A detached worker parked on `recv` keeps the process free to exit.
        std::thread::Builder::new()
            .name("rsdag-jit-queue".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    pool().install(|| {
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                    });
                }
            })
            .expect("spawn the compile queue");
        tx
    })
}
