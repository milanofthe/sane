//! Native logging for the whole SANE tool, in the fastsim / pathsim style.
//!
//! Dependency-free. A process-global level (an atomic) gates everything, so the
//! many stateless analysis functions can emit progress without threading a
//! logger handle through every signature. INFO/WARNING go to stdout (they are
//! normal progress, not errors); ERROR goes to stderr. The line format mirrors
//! Python's `logging`: `HH:MM:SS - LEVEL - message`.
//!
//! Disabled by default, so library use and the test/bench suites stay silent.
//! Hosts opt in via [`set_level`] / [`set_enabled`] (the Python API exposes
//! `sane.set_log_level(...)`).

use crate::time::Instant;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

/// Log levels matching Python's `logging` module numeric values.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LogLevel {
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
    Disabled = 100,
}

impl LogLevel {
    /// Numeric severity (the discriminant), for comparisons against the global.
    #[inline]
    pub fn num(self) -> u8 {
        self as u8
    }

    /// Uppercase name as used in the log line.
    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARNING",
            LogLevel::Error => "ERROR",
            LogLevel::Disabled => "DISABLED",
        }
    }

    /// Parse a level name (case-insensitive); also accepts the numeric values.
    /// Returns `None` for anything unrecognized.
    pub fn parse(s: &str) -> Option<LogLevel> {
        match s.trim().to_ascii_uppercase().as_str() {
            "DEBUG" | "10" => Some(LogLevel::Debug),
            "INFO" | "20" => Some(LogLevel::Info),
            "WARNING" | "WARN" | "30" => Some(LogLevel::Warning),
            "ERROR" | "40" => Some(LogLevel::Error),
            "DISABLED" | "OFF" | "NONE" | "100" => Some(LogLevel::Disabled),
            _ => None,
        }
    }
}

/// The single process-global threshold. Messages at or above it are emitted.
static LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Disabled as u8);

/// Set the global log threshold. Messages with severity `>= level` are emitted.
/// Every hook registered with [`on_level_change`] is told the new level, and
/// rsdag's reporting hooks are installed if they are not yet
/// ([`crate::hooks`]).
pub fn set_level(level: LogLevel) {
    crate::hooks::install();
    LEVEL.store(level.num(), Ordering::Relaxed);
    if let Ok(hooks) = LEVEL_HOOKS.lock() {
        for h in hooks.iter() {
            h(level);
        }
    }
}

/// Level-change hooks: a dependency with its own logger (the sparse solver)
/// mirrors SANE's threshold into it, so one `set_level` governs the whole
/// process.
static LEVEL_HOOKS: Mutex<Vec<fn(LogLevel)>> = Mutex::new(Vec::new());

/// Register `hook` to be called on every [`set_level`]; it is called once
/// immediately with the current level.
pub fn on_level_change(hook: fn(LogLevel)) {
    if let Ok(mut hooks) = LEVEL_HOOKS.lock() {
        hooks.push(hook);
    }
    hook(level());
}

/// Convenience: enable at INFO, or disable entirely.
pub fn set_enabled(on: bool) {
    set_level(if on {
        LogLevel::Info
    } else {
        LogLevel::Disabled
    });
}

/// The current global threshold.
pub fn level() -> LogLevel {
    match LEVEL.load(Ordering::Relaxed) {
        10 => LogLevel::Debug,
        20 => LogLevel::Info,
        30 => LogLevel::Warning,
        40 => LogLevel::Error,
        _ => LogLevel::Disabled,
    }
}

/// Would a message at `level` be emitted under the current threshold?
#[inline]
pub fn enabled(level: LogLevel) -> bool {
    level.num() >= LEVEL.load(Ordering::Relaxed)
}

#[inline]
fn timestamp() -> String {
    let secs = crate::time::epoch_secs() % 86_400; // time of day
    format!(
        "{:02}:{:02}:{:02}",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

/// Host log sink: when registered, every emitted line goes to it instead of
/// stdout/stderr. The wasm host uses this to stream engine logs into the UI
/// (stdout does not exist in the browser); embedders can route into their own
/// logging. The sink receives the raw message; formatting stays with the host.
static SINK: Mutex<Option<fn(LogLevel, &str)>> = Mutex::new(None);

/// Redirect emitted log lines to `sink` (replacing stdout/stderr). `None`
/// restores the default console output.
pub fn set_sink(sink: Option<fn(LogLevel, &str)>) {
    if let Ok(mut s) = SINK.lock() {
        *s = sink;
    }
}

// --- stage context ---------------------------------------------------------
// A thread-local stack of the stages currently in flight ("where are we?").
// Every emitted line carries the joined path as a `[dc/gmin]`-style prefix, so
// nested solver internals (a DC Newton inside a transient IC inside a
// sensitivity sweep) read unambiguously. `scope()` pushes automatically, so
// the existing stage timers double as context markers.
std::thread_local! {
    static STAGES: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The current stage path, e.g. `"tran/irk_step"`; empty outside any scope.
pub fn stage_path() -> String {
    STAGES.with(|s| s.borrow().join("/"))
}

fn push_stage(name: &'static str) {
    STAGES.with(|s| s.borrow_mut().push(name));
}

fn pop_stage() {
    STAGES.with(|s| {
        s.borrow_mut().pop();
    });
}

#[inline]
fn emit(level: LogLevel, msg: &str) {
    let path = stage_path();
    let line = if path.is_empty() {
        msg.to_string()
    } else {
        format!("[{path}] {msg}")
    };
    if let Ok(s) = SINK.lock() {
        if let Some(f) = *s {
            f(level, &line);
            return;
        }
    }
    if level == LogLevel::Error {
        eprintln!("{} - {} - {}", timestamp(), level.as_str(), line);
    } else {
        println!("{} - {} - {}", timestamp(), level.as_str(), line);
    }
}

/// Log at DEBUG.
#[inline]
pub fn debug(msg: &str) {
    if enabled(LogLevel::Debug) {
        emit(LogLevel::Debug, msg);
    }
}

/// Log at INFO (stdout).
#[inline]
pub fn info(msg: &str) {
    if enabled(LogLevel::Info) {
        emit(LogLevel::Info, msg);
    }
}

/// Log at WARNING (stdout).
#[inline]
pub fn warning(msg: &str) {
    if enabled(LogLevel::Warning) {
        emit(LogLevel::Warning, msg);
    }
}

/// Log at ERROR (stderr). Suppressed only when logging is fully disabled.
#[inline]
pub fn error(msg: &str) {
    if enabled(LogLevel::Error) {
        emit(LogLevel::Error, msg);
    }
}

/// Captured correctness-affecting warnings, drained by the host (the Python
/// layer) and re-emitted as catchable `warnings.warn`. Independent of the log
/// threshold, so a diagnostic the user must not miss (a gmin-regularized point,
/// an out-of-range device parameter) reaches them even with logging disabled --
/// the default. See [`warn_captured`] / [`drain_captured`] (issue #54).
static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Emit a WARNING (gated by the log level, as usual) **and** unconditionally
/// capture it for the host to re-raise as a catchable warning. Use for
/// correctness-affecting diagnostics that must reach the caller regardless of
/// [`set_level`] (issue #54).
pub fn warn_captured(msg: &str) {
    warning(msg);
    if let Ok(mut c) = CAPTURED.lock() {
        c.push(msg.to_string());
    }
}

/// Drain and return every message captured by [`warn_captured`] since the last
/// drain (clearing the buffer). The Python layer calls this at analysis
/// boundaries and re-emits each as a `SaneConvergenceWarning`.
pub fn drain_captured() -> Vec<String> {
    CAPTURED
        .lock()
        .map(|mut c| std::mem::take(&mut *c))
        .unwrap_or_default()
}

/// Emit a stage-timing line at DEBUG, e.g. `stage ac/assemble_gc : 12.345 ms`.
/// The single sink behind the [`time_stage!`](crate::time_stage) and
/// [`log_stage!`](crate::log_stage) macros, so every instrumented stage in the
/// engine logs in one consistent format, gated by the global level.
#[inline]
pub fn stage(name: &str, dur: std::time::Duration) {
    // Always offer the span to the global profile sink (a no-op unless a
    // benchmark has collection active), then emit to the log if DEBUG is on.
    crate::profile::record_global(name, dur);
    if enabled(LogLevel::Debug) {
        emit(
            LogLevel::Debug,
            &format!("stage {name} : {:.3} ms", dur.as_secs_f64() * 1e3),
        );
    }
}

/// RAII scope timer AND stage-context marker: pushes `name` onto the
/// thread-local stage stack (so every log line inside carries the
/// `[a/b/c]` path) and logs `stage <name>` with the elapsed wall time when
/// dropped, so a function with many `?`/early-return paths still reports its
/// total on every path. Put `let _g = log::scope("sens/ac_gradient");` at the
/// top of the function. Cheap (one `Instant`) and silent unless DEBUG is on.
#[must_use]
pub struct ScopeTimer {
    name: &'static str,
    start: Instant,
}

impl Drop for ScopeTimer {
    fn drop(&mut self) {
        pop_stage();
        stage(self.name, self.start.elapsed());
    }
}

/// Start a [`ScopeTimer`] for `name` (see its docs). Bind it to a `_`-prefixed
/// local so it lives to the end of the scope.
#[inline]
pub fn scope(name: &'static str) -> ScopeTimer {
    push_stage(name);
    ScopeTimer {
        name,
        start: Instant::now(),
    }
}

/// RAII lifecycle scope for a top-level analysis, in the fastsim / pathsim
/// vocabulary: `STARTING -> <NAME> <details>` at creation, `FINISHED -> <NAME>
/// (<details>, runtime: ...)` on drop -- and the stage context pushed for
/// everything in between. `finish_details` set before drop lands in the
/// closing line:
///
/// ```text
/// let mut t = log::task("HB", "hb", "(f0: 1.0e3, harmonics: 8)");
/// // ... solve ...
/// t.finish(format!("converged: {conv}, iters: {iters}"));
/// ```
///
/// Emits at INFO; a no-op (beyond one `Instant`) below that.
#[must_use]
pub struct TaskScope {
    name: &'static str,
    start: Instant,
    finish_details: Option<String>,
}

impl TaskScope {
    /// Set the result details for the FINISHED line (e.g. iteration counts).
    pub fn finish(&mut self, details: String) {
        self.finish_details = Some(details);
    }
}

impl Drop for TaskScope {
    fn drop(&mut self) {
        pop_stage();
        let ms = self.start.elapsed().as_secs_f64() * 1e3;
        match self.finish_details.take() {
            Some(d) => info(&format!(
                "FINISHED -> {} ({d}, runtime: {ms:.1} ms)",
                self.name
            )),
            None => info(&format!("FINISHED -> {} (runtime: {ms:.1} ms)", self.name)),
        }
    }
}

/// Open a [`TaskScope`]: logs `STARTING -> <name> <details>` and pushes
/// `stage` onto the context stack (use a short lowercase stage tag, e.g.
/// `"hb"`, and the uppercase analysis name for the lifecycle lines).
pub fn task(name: &'static str, stage_tag: &'static str, details: &str) -> TaskScope {
    if details.is_empty() {
        info(&format!("STARTING -> {name}"));
    } else {
        info(&format!("STARTING -> {name} {details}"));
    }
    push_stage(stage_tag);
    TaskScope {
        name,
        start: Instant::now(),
        finish_details: None,
    }
}

/// Initialise the global level from the `SANE_LOG` environment variable (a level
/// name or number, see [`LogLevel::parse`]); leaves it unchanged if unset or
/// unrecognised. Lets any host or example opt into logging without code changes.
pub fn init_from_env() {
    if let Some(lvl) = std::env::var("SANE_LOG")
        .ok()
        .and_then(|s| LogLevel::parse(&s))
    {
        set_level(lvl);
    }
}

// ======================================================================================
// Progress tracker -- ASCII bar + ETA + EMA rate, mirroring pathsim's ProgressTracker.
// ======================================================================================

/// Lightweight progress stats accumulated over a run.
#[derive(Clone, Debug, Default)]
pub struct ProgressStats {
    pub total_steps: usize,
    pub successful_steps: usize,
    pub rejected_steps: usize,
    pub runtime_ms: f64,
}

/// Progress tracker for a long-running solve: prints an ASCII bar with ETA and
/// rate, throttled so it never floods the log. A no-op when logging is below
/// INFO, so it is cheap to leave in the hot path.
pub struct ProgressTracker {
    pub description: String,
    /// Parenthesized parameter details for the STARTING line (may be empty).
    pub details: String,
    pub total: f64,
    pub stats: ProgressStats,
    start_time: Instant,
    last_log_time: Instant,
    last_log_progress: f64,
    ema_rate: f64,
    min_log_interval: f64,
    update_log_every: f64,
    bar_width: usize,
    ema_alpha: f64,
    interrupted: bool,
}

impl ProgressTracker {
    /// `total` is a human-facing magnitude for the start line (e.g. simulated
    /// seconds, number of points); progress passed to [`update`] is a fraction
    /// in `[0, 1]`.
    pub fn new(total: f64, description: &str) -> Self {
        Self::with_details(total, description, "")
    }

    /// Like [`new`](Self::new), with a separate details clause: the STARTING
    /// line reads `STARTING -> <description> <details>`, the closing line
    /// `FINISHED -> <description> (steps: .., ok: .., runtime: ..)` -- the
    /// pathsim/fastsim vocabulary.
    pub fn with_details(total: f64, description: &str, details: &str) -> Self {
        let now = Instant::now();
        Self {
            description: description.to_string(),
            details: details.to_string(),
            total,
            stats: ProgressStats::default(),
            start_time: now,
            last_log_time: now,
            last_log_progress: 0.0,
            ema_rate: 0.0,
            min_log_interval: 1.0,
            update_log_every: 0.2,
            bar_width: 20,
            ema_alpha: 0.3,
            interrupted: false,
        }
    }

    /// Reset the clock and print the STARTING line.
    pub fn start(&mut self) {
        let now = Instant::now();
        self.start_time = now;
        self.last_log_time = now;
        if self.details.is_empty() {
            info(&format!("STARTING -> {}", self.description));
        } else {
            info(&format!(
                "STARTING -> {} {}",
                self.description, self.details
            ));
        }
    }

    /// Record one step worth of progress (`progress` is the overall fraction in
    /// `[0, 1]`). Logging is throttled by time and progress deltas.
    pub fn update(&mut self, progress: f64, success: bool) {
        self.stats.total_steps += 1;
        if success {
            self.stats.successful_steps += 1;
        } else {
            self.stats.rejected_steps += 1;
        }

        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let instant_rate = self.stats.total_steps as f64 / elapsed;
            if self.ema_rate == 0.0 {
                self.ema_rate = instant_rate;
            } else {
                self.ema_rate =
                    self.ema_alpha * instant_rate + (1.0 - self.ema_alpha) * self.ema_rate;
            }
        }

        if !enabled(LogLevel::Info) {
            return;
        }
        let now = Instant::now();
        let time_trigger =
            now.duration_since(self.last_log_time).as_secs_f64() >= self.min_log_interval;
        let progress_trigger = progress >= self.last_log_progress + self.update_log_every;
        if (time_trigger || progress_trigger) && progress > 0.0 {
            self.log_progress(progress, elapsed);
            self.last_log_time = now;
            self.last_log_progress = progress;
        }
    }

    /// Mark the run as interrupted (changes the closing line).
    pub fn interrupt(&mut self) {
        self.interrupted = true;
    }

    /// Print the closing line with totals and runtime.
    pub fn close(&mut self) {
        self.stats.runtime_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        let status = if self.interrupted {
            "INTERRUPTED"
        } else {
            "FINISHED"
        };
        if self.stats.rejected_steps > 0 {
            info(&format!(
                "{} -> {} (steps: {}, ok: {}, rejected: {}, runtime: {:.1} ms)",
                status,
                self.description,
                self.stats.total_steps,
                self.stats.successful_steps,
                self.stats.rejected_steps,
                self.stats.runtime_ms
            ));
        } else {
            info(&format!(
                "{} -> {} (steps: {}, ok: {}, runtime: {:.1} ms)",
                status,
                self.description,
                self.stats.total_steps,
                self.stats.successful_steps,
                self.stats.runtime_ms
            ));
        }
    }

    fn log_progress(&self, progress: f64, elapsed: f64) {
        let pct = (progress * 100.0) as usize;
        let filled = (progress * self.bar_width as f64) as usize;
        let filled = filled.min(self.bar_width);
        let bar: String = "#".repeat(filled) + &"-".repeat(self.bar_width - filled);
        let elapsed_str = format_time(elapsed);
        let eta = if progress > 0.01 {
            format_time(elapsed / progress * (1.0 - progress))
        } else {
            "--:--".to_string()
        };
        info(&format!(
            "{} {:3}% | {}<{} | {}",
            bar,
            pct,
            elapsed_str,
            eta,
            format_rate(self.ema_rate)
        ));
    }
}

fn format_time(secs: f64) -> String {
    if secs < 0.0 || secs.is_nan() || secs.is_infinite() {
        return "--:--".to_string();
    }
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        format!("{:02}:{:02}", (secs / 60.0) as u64, (secs % 60.0) as u64)
    } else {
        format!(
            "{:02}:{:02}:{:02}",
            (secs / 3600.0) as u64,
            ((secs % 3600.0) / 60.0) as u64,
            (secs % 60.0) as u64
        )
    }
}

fn format_rate(rate: f64) -> String {
    if rate <= 0.0 || rate.is_nan() {
        "N/A".to_string()
    } else if rate < 0.1 {
        format!("{:.1} it/min", rate * 60.0)
    } else if rate < 1.0 {
        format!("{:.2} it/s", rate)
    } else {
        format!("{:.1} it/s", rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_names_roundtrip() {
        assert_eq!(LogLevel::parse("info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::parse("WARN"), Some(LogLevel::Warning));
        assert_eq!(LogLevel::parse("40"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("off"), Some(LogLevel::Disabled));
        assert_eq!(LogLevel::parse("bogus"), None);
        assert_eq!(LogLevel::Info.as_str(), "INFO");
    }

    #[test]
    fn threshold_gates_levels() {
        set_level(LogLevel::Warning);
        assert!(!enabled(LogLevel::Info));
        assert!(enabled(LogLevel::Warning));
        assert!(enabled(LogLevel::Error));
        set_level(LogLevel::Disabled);
        assert!(!enabled(LogLevel::Error));
        set_enabled(true);
        assert!(enabled(LogLevel::Info));
        // Restore silence for the rest of the suite.
        set_enabled(false);
    }

    #[test]
    fn stage_stack_nests_and_unwinds() {
        assert_eq!(stage_path(), "");
        {
            let _a = scope("outer");
            assert_eq!(stage_path(), "outer");
            {
                let _b = scope("inner");
                assert_eq!(stage_path(), "outer/inner");
            }
            assert_eq!(stage_path(), "outer");
        }
        assert_eq!(stage_path(), "");
    }

    #[test]
    fn format_helpers() {
        assert_eq!(format_time(5.2), "5.2s");
        assert_eq!(format_time(65.0), "01:05");
        assert_eq!(format_time(3661.0), "01:01:01");
        assert_eq!(format_rate(0.05), "3.0 it/min");
        assert_eq!(format_rate(42.0), "42.0 it/s");
    }
}
