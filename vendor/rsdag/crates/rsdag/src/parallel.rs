//! Parallel execution of a program's independent calls.
//!
//! A tape groups consecutive calls that read nothing another one of them
//! writes into stages (see [`Tape::stages`](crate::Tape::stages)); every
//! instance of every call in a stage is a piece of work of its own. When
//! a [`Pool`] is installed on the calling thread ([`install`]) and a stage
//! carries at least [`Parallel::min_ops`] ops, its instances run on the
//! pool, each over scratch of its worker's; otherwise one after the other.
//! Either way an instance computes exactly what the serial loop computes,
//! so the results are the same bit for bit, whatever the thread count.
//! Both backends, the interpreter and the native code, run stages so.
//!
//! The host lends the pool: [`Workers`], this module's own, keeps its
//! workers awake across the short serial stretches between a solver's
//! evaluations (a factorization, a step decision) and lets the calling
//! thread take a share, so a stage costs no thread put to sleep and woken;
//! with the `rayon` feature every [`rayon::ThreadPool`] is a [`Pool`] too.
//! Work running on a pool, and a stage's calls, see no pool installed, so
//! a call inside a parallel stage runs its own stages serially.

use std::cell::RefCell;
use std::sync::Arc;

/// Workers to run independent pieces of work on.
pub trait Pool: Send + Sync {
    /// Number of workers.
    fn threads(&self) -> usize;
    /// Run `f(item)` for every `item` in `0..n`, returning when all ran.
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync));
    /// Run `f` where handing work to the pool is cheapest: on one of its
    /// workers, for a pool whose hand-off from outside costs more (a
    /// thread put to sleep and woken). The default runs it here.
    fn enter(&self, f: &mut (dyn FnMut() + Send)) {
        f()
    }
}

/// A pool and from what size of stage on it is worth the hand-off.
#[derive(Clone)]
pub struct Parallel {
    pub pool: Arc<dyn Pool>,
    /// Ops a stage carries (instances times their body's ops) from which on
    /// it runs on the pool; smaller stages run serially.
    pub min_ops: usize,
}

/// [`Parallel::min_ops`] by default: about ten microseconds of work.
pub const MIN_OPS: usize = 20_000;

impl Parallel {
    pub fn new(pool: Arc<dyn Pool>) -> Self {
        Parallel {
            pool,
            min_ops: MIN_OPS,
        }
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Parallel>> = const { RefCell::new(None) };
    /// This thread is running pieces of a stage: a stage inside runs serially.
    static IN_STAGE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with `p` installed: the programs `f` evaluates run their stages
/// on `p`'s pool. `f` runs where the pool hands work out cheapest
/// ([`Pool::enter`]: for a rayon pool, on one of its workers), so a solver
/// that evaluates its programs many times should run whole inside one
/// `install`, not install per evaluation. The previous installation comes
/// back when `f` returns (or unwinds).
pub fn install<R: Send>(p: Parallel, f: impl FnOnce() -> R + Send) -> R {
    struct Restore(Option<Parallel>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let p = self.0.take();
            CURRENT.with(|c| *c.borrow_mut() = p);
        }
    }
    let pool = p.pool.clone();
    let (mut f, mut r) = (Some(f), None);
    pool.enter(&mut || {
        let _restore = Restore(CURRENT.with(|c| c.replace(Some(p.clone()))));
        r = Some((f.take().expect("entered once"))());
    });
    r.expect("the pool ran it")
}

/// What is installed on this thread.
pub fn current() -> Option<Parallel> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Whether work of `ops` ops runs on the installed pool: there is one, with
/// at least two workers, and the work reaches its [`Parallel::min_ops`].
pub fn worth(ops: usize) -> bool {
    !IN_STAGE.with(|f| f.get())
        && CURRENT.with(|c| {
            c.borrow()
                .as_ref()
                .is_some_and(|p| ops >= p.min_ops && p.pool.threads() >= 2)
        })
}

/// Run `n` pieces of work of `ops` ops together: on the installed pool when
/// there is one and the work is worth it, else serially on this thread.
pub fn run(n: usize, ops: usize, f: &(dyn Fn(usize) + Sync)) {
    let p = if n >= 2 && worth(ops) {
        CURRENT.with(|c| c.borrow().as_ref().map(|p| p.pool.clone()))
    } else {
        None
    };
    match p {
        Some(pool) => {
            struct Clear;
            impl Drop for Clear {
                fn drop(&mut self) {
                    IN_STAGE.with(|f| f.set(false));
                }
            }
            IN_STAGE.with(|f| f.set(true));
            let _clear = Clear;
            pool.run(n, f)
        }
        None => (0..n).for_each(f),
    }
}

/// Instances per piece of work for a stage of `n` instances: about four
/// pieces per worker of the installed pool (balance over instances of
/// unequal cost), all in one piece when there is none.
pub fn block(n: usize) -> usize {
    let threads = CURRENT.with(|c| c.borrow().as_ref().map_or(0, |p| p.pool.threads()));
    if threads < 2 {
        return n.max(1);
    }
    (n / (threads * 4)).max(1)
}

/// Scratch of values of type `T` for the work this thread runs, kept between
/// calls: a buffer per thread and type, grown on demand.
pub fn with_scratch<T: Copy + 'static, R>(len: usize, zero: T, f: impl FnOnce(&mut [T]) -> R) -> R {
    use std::any::{Any, TypeId};
    thread_local! {
        static BUFS: RefCell<Vec<(TypeId, Box<dyn Any>)>> = const { RefCell::new(Vec::new()) };
    }
    // Taken out for the call, so a body that runs a stage of its own (on
    // this thread, serially) takes a buffer of its own.
    let mut buf: Box<Vec<T>> = BUFS.with(|b| {
        let mut b = b.borrow_mut();
        let at = b.iter().position(|(t, _)| *t == TypeId::of::<T>());
        at.and_then(|i| b.swap_remove(i).1.downcast::<Vec<T>>().ok())
            .unwrap_or_default()
    });
    if buf.len() < len {
        buf.resize(len, zero);
    }
    let r = f(&mut buf[..len]);
    BUFS.with(|b| b.borrow_mut().push((TypeId::of::<T>(), buf)));
    r
}

#[cfg(feature = "rayon")]
impl Pool for rayon::ThreadPool {
    fn threads(&self) -> usize {
        self.current_num_threads()
    }
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        use rayon::prelude::*;
        self.install(|| (0..n).into_par_iter().for_each(f));
    }
    fn enter(&self, f: &mut (dyn FnMut() + Send)) {
        self.install(f)
    }
}

/// How long [`Workers`] keep waiting for the next stage before they sleep:
/// longer than the serial stretch between a solver's evaluations.
pub const SPIN: std::time::Duration = std::time::Duration::from_micros(300);

/// A pool of threads for stages: `threads - 1` workers and the thread that
/// runs a stage, which takes pieces of it too. After a stage the workers
/// keep waiting for the next one for [`SPIN`] (as OpenMP's active wait),
/// then sleep until one comes; so the stages of a solve, a few tens of
/// microseconds apart, reach awake workers, and an idle pool costs nothing.
pub struct Workers {
    shared: Arc<Shared>,
    threads: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
    /// One stage at a time.
    dispatch: std::sync::Mutex<()>,
}

/// What the workers and the dispatching thread share.
struct Shared {
    /// Bumped per stage; a worker that sees it change takes part.
    epoch: AtomicU64,
    /// The stage's work, a pointer to the caller's `&dyn Fn`; null once it
    /// is done, so a late worker leaves it alone.
    job: AtomicPtr<&'static (dyn Fn(usize) + Sync)>,
    n: AtomicUsize,
    next: AtomicUsize,
    done: AtomicUsize,
    /// Workers inside the stage; the caller returns only at zero.
    active: AtomicUsize,
    shutdown: AtomicBool,
    spin: std::time::Duration,
    /// Sleeping workers wait here for the next epoch.
    sleep: std::sync::Mutex<usize>,
    wake: std::sync::Condvar,
    panic: std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst};

impl Workers {
    /// A pool of `threads` threads (the caller one of them), waiting
    /// [`SPIN`] between stages.
    pub fn new(threads: usize) -> Self {
        Self::with_spin(threads, SPIN)
    }

    /// [`new`](Self::new) with the workers waiting `spin` between stages.
    pub fn with_spin(threads: usize, spin: std::time::Duration) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            n: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            spin,
            sleep: std::sync::Mutex::new(0),
            wake: std::sync::Condvar::new(),
            panic: std::sync::Mutex::new(None),
        });
        let handles = (1..threads)
            .map(|k| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("rsdag-worker-{k}"))
                    .spawn(move || sh.work())
                    .expect("a worker thread")
            })
            .collect();
        Workers {
            shared,
            threads,
            handles,
            dispatch: std::sync::Mutex::new(()),
        }
    }
}

impl Shared {
    /// Take pieces of the current stage until none is left.
    fn take(&self, f: &(dyn Fn(usize) + Sync)) {
        let n = self.n.load(SeqCst);
        loop {
            let i = self.next.fetch_add(1, SeqCst);
            if i >= n {
                return;
            }
            if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(i))) {
                let mut slot = self.panic.lock().unwrap();
                slot.get_or_insert(p);
            }
            self.done.fetch_add(1, SeqCst);
        }
    }

    /// A worker's life: wait for a stage (awake for `spin`, then asleep),
    /// take part, repeat.
    fn work(&self) {
        let mut seen = self.epoch.load(SeqCst);
        loop {
            let since = std::time::Instant::now();
            let mut spins = 0u32;
            loop {
                if self.shutdown.load(SeqCst) {
                    return;
                }
                if self.epoch.load(SeqCst) != seen {
                    break;
                }
                if since.elapsed() >= self.spin {
                    let mut sleeping = self.sleep.lock().unwrap();
                    *sleeping += 1;
                    while self.epoch.load(SeqCst) == seen && !self.shutdown.load(SeqCst) {
                        sleeping = self.wake.wait(sleeping).unwrap();
                    }
                    *sleeping -= 1;
                    break;
                }
                spins += 1;
                if spins.is_multiple_of(64) {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
            }
            seen = self.epoch.load(SeqCst);
            // Announce first, then look: the caller clears the job before it
            // waits for the announced ones, so either this sees no job or
            // the caller sees this worker (both sequentially consistent).
            self.active.fetch_add(1, SeqCst);
            let job = self.job.load(SeqCst);
            if !job.is_null() {
                // SAFETY: the job lives until the caller has seen `active`
                // drop to zero, which this worker holds above zero here.
                let f: &(dyn Fn(usize) + Sync) = unsafe { *job };
                self.take(f);
            }
            self.active.fetch_sub(1, SeqCst);
        }
    }
}

impl Pool for Workers {
    fn threads(&self) -> usize {
        self.threads
    }

    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.threads < 2 || n < 2 {
            (0..n).for_each(f);
            return;
        }
        // A second thread dispatching at the same time runs its stage itself.
        let Ok(_one) = self.dispatch.try_lock() else {
            (0..n).for_each(f);
            return;
        };
        let sh = &*self.shared;
        // SAFETY: `f` outlives this call, and the workers drop their last use
        // of it before this returns (see `work`); the `'static` is a lie that
        // never leaves this frame.
        let f_static: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let cell = f_static;
        sh.n.store(n, SeqCst);
        sh.next.store(0, SeqCst);
        sh.done.store(0, SeqCst);
        sh.job.store(&cell as *const _ as *mut _, SeqCst);
        sh.epoch.fetch_add(1, SeqCst);
        if *sh.sleep.lock().unwrap() > 0 {
            sh.wake.notify_all();
        }
        sh.take(f);
        while sh.done.load(SeqCst) < n {
            std::hint::spin_loop();
        }
        sh.job.store(std::ptr::null_mut(), SeqCst);
        while sh.active.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
        if let Some(p) = sh.panic.lock().unwrap().take() {
            std::panic::resume_unwind(p);
        }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, SeqCst);
        {
            let _g = self.shared.sleep.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
