//! rsdag's reporting hooks wired to SANE's logger and portable clock.
//!
//! rsdag has no logging dependency and no opinion about time: it reports its
//! compile stages through `rsdag::hooks` and reads whatever clock the host
//! installed. Its default clock is `std::time::Instant`, which panics on
//! wasm32 -- so the wiring is not decoration, it is what lets the engine
//! compile a tape in the browser at all.
//!
//! The sink reports SANE's own threshold back to rsdag, so with logging off
//! rsdag skips the clock and the formatting altogether.
//!
//! [`install`] is idempotent and cheap; SANE calls it from every entry point
//! that can produce logs ([`crate::log::set_level`] and the hosts), so a
//! consumer never has to think about it.

use std::sync::Once;

use crate::log::{self, LogLevel};
use crate::time::Instant;

/// rsdag's stage reports routed into SANE's logger. They are compiler
/// internals from SANE's point of view (one line per tape pass), so its
/// `Debug` and `Info` land at `Debug`; warnings keep their level. The message
/// carries an `rsdag:` prefix, like the solver's `rslab:` lines.
struct Sink;

impl rsdag::hooks::Log for Sink {
    fn enabled(&self, level: rsdag::hooks::Level) -> bool {
        log::enabled(match level {
            rsdag::hooks::Level::Debug | rsdag::hooks::Level::Info => LogLevel::Debug,
            rsdag::hooks::Level::Warn => LogLevel::Warning,
        })
    }

    fn log(&self, level: rsdag::hooks::Level, msg: &str) {
        let line = format!("rsdag: {msg}");
        match level {
            rsdag::hooks::Level::Debug | rsdag::hooks::Level::Info => log::debug(&line),
            rsdag::hooks::Level::Warn => log::warning(&line),
        }
    }
}

/// SANE's portable clock as rsdag's, so the stage timings are real on wasm32
/// too (where the host registers `Date.now`) instead of trapping.
struct Clock;

impl rsdag::hooks::Clock for Clock {
    fn now_ns(&self) -> u64 {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

static SINK: Sink = Sink;
static CLOCK: Clock = Clock;

/// Install both hooks once per process.
pub fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        rsdag::hooks::set_log(&SINK);
        rsdag::hooks::set_clock(&CLOCK);
    });
}
