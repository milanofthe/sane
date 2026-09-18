//! Monotonic clock for the engine's timing instrumentation, portable to hosts
//! without an OS clock (wasm32: `std::time::Instant::now()` panics in the
//! browser). Native builds re-export [`std::time::Instant`] unchanged, so the
//! instrumentation cost and behavior stay exactly as before. On wasm32 the
//! host registers a millisecond clock once ([`set_clock_ms`], e.g. JS
//! `Date.now`); until then the clock reads zero and every duration is `0` --
//! timing lines become inert instead of trapping.
//!
//! Dependency-free by design, like the rest of `sane-core`: the wasm host
//! brings its own clock instead of this crate linking a JS interop layer.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub use wasm::{set_clock_ms, Instant};

/// Seconds since the UNIX epoch, for wall-clock log timestamps. Native uses
/// [`std::time::SystemTime`]; wasm32 uses the registered clock (JS `Date.now`
/// is already ms since the epoch, so the time-of-day in log lines is real).
pub fn epoch_secs() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (wasm::now_ms() / 1e3) as u64
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// The registered ms clock as a raw fn pointer (0 = none). An atomic usize
    /// keeps this dependency-free and safe to read from any context.
    static CLOCK: AtomicUsize = AtomicUsize::new(0);

    /// Register the host clock: a plain function returning milliseconds on a
    /// monotone (or epoch) scale, e.g. a trampoline to JS `Date.now`.
    pub fn set_clock_ms(f: fn() -> f64) {
        CLOCK.store(f as usize, Ordering::Relaxed);
    }

    pub(super) fn now_ms() -> f64 {
        let p = CLOCK.load(Ordering::Relaxed);
        if p == 0 {
            0.0
        } else {
            // Safety: the only writer is `set_clock_ms`, which stores a valid
            // `fn() -> f64`.
            unsafe { std::mem::transmute::<usize, fn() -> f64>(p)() }
        }
    }

    /// Drop-in stand-in for `std::time::Instant` over the registered ms clock.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Instant(f64);

    impl Instant {
        pub fn now() -> Self {
            Instant(now_ms())
        }

        pub fn elapsed(&self) -> Duration {
            Duration::from_secs_f64((now_ms() - self.0).max(0.0) / 1e3)
        }

        pub fn duration_since(&self, earlier: Instant) -> Duration {
            Duration::from_secs_f64((self.0 - earlier.0).max(0.0) / 1e3)
        }
    }
}
