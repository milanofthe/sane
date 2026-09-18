//! Lightweight, always-on stage timing for the extract / compile pipeline.
//!
//! A [`Profile`] records `(stage, duration)` pairs in the order they happen, so
//! the cost of each pipeline phase (graph assembly, sparse-Jacobian extraction,
//! tape compilation, symbolic LU, ...) is visible end to end without an external
//! profiler. The overhead is one [`std::time::Instant`] per stage -- negligible
//! against the work it brackets -- so it stays on in normal runs and the timings
//! can be surfaced (e.g. to Python) for scaling studies.

use std::sync::Mutex;
use std::time::Duration;

/// An ordered list of `(stage name, wall-clock duration)` measurements.
#[derive(Default, Clone, Debug)]
pub struct Profile {
    spans: Vec<(String, Duration)>,
}

/// Process-global collection sink. When active (between [`collect_begin`] and
/// [`collect_take`]), every stage that flows through [`crate::log::stage`] -- and
/// thus every `time_stage!` / `log_stage!` / `log::scope` -- is also recorded
/// here, so a benchmark can capture the full stage breakdown of an analysis
/// programmatically rather than by parsing log lines. Independent of the log
/// level (collection works with logging fully off).
static SINK: Mutex<Option<Profile>> = Mutex::new(None);

/// Begin collecting stage timings into the global sink (clearing any prior run).
pub fn collect_begin() {
    *SINK.lock().unwrap() = Some(Profile::new());
}

/// Stop collecting and return what was gathered (empty if collection was off).
pub fn collect_take() -> Profile {
    SINK.lock().unwrap().take().unwrap_or_default()
}

// --- TEMPORARY: split Verilog-A template build vs clone time (cross-crate) ------
thread_local! {
    static TPL: std::cell::Cell<(u128, u128, u32, u32)> = const { std::cell::Cell::new((0, 0, 0, 0)) };
}
/// Add a template-build span (ns).
pub fn record_tpl_build(ns: u128) {
    TPL.with(|c| {
        let (b, cl, nb, nc) = c.get();
        c.set((b + ns, cl, nb + 1, nc));
    });
}
/// Add a template-clone (instantiate) span (ns).
pub fn record_tpl_clone(ns: u128) {
    TPL.with(|c| {
        let (b, cl, nb, nc) = c.get();
        c.set((b, cl + ns, nb, nc + 1));
    });
}
/// `(build_ns, clone_ns, builds, clones)` since the last call; resets.
pub fn take_tpl_stats() -> (u128, u128, u32, u32) {
    TPL.with(|c| {
        let v = c.get();
        c.set((0, 0, 0, 0));
        v
    })
}

/// Record `(name, dur)` into the global sink if collection is active. Called by
/// [`crate::log::stage`]; cheap (an uncontended lock) and a no-op when inactive.
pub(crate) fn record_global(name: &str, dur: Duration) {
    if let Ok(mut g) = SINK.lock() {
        if let Some(p) = g.as_mut() {
            p.record(name, dur);
        }
    }
}

impl Profile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a finished stage.
    pub fn record(&mut self, name: impl Into<String>, dur: Duration) {
        self.spans.push((name.into(), dur));
    }

    /// Append another profile's spans, each prefixed with `prefix` (e.g. merging
    /// a sub-stage profile into the top-level one under `compile/`).
    pub fn extend_prefixed(&mut self, prefix: &str, other: &Profile) {
        for (n, d) in &other.spans {
            self.spans.push((format!("{prefix}{n}"), *d));
        }
    }

    /// `(stage, milliseconds)` pairs, in record order.
    pub fn millis(&self) -> Vec<(String, f64)> {
        self.spans
            .iter()
            .map(|(n, d)| (n.clone(), d.as_secs_f64() * 1e3))
            .collect()
    }

    /// `(stage, milliseconds)` summed per distinct stage name, in first-seen
    /// order (a stage hit N times -- e.g. `dc/newton` across continuation steps --
    /// collapses to one row with the total).
    pub fn aggregated_millis(&self) -> Vec<(String, f64)> {
        let mut order: Vec<String> = Vec::new();
        let mut idx: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        let mut totals: Vec<f64> = Vec::new();
        for (n, d) in &self.spans {
            let ms = d.as_secs_f64() * 1e3;
            match idx.get(n.as_str()) {
                Some(&i) => totals[i] += ms,
                None => {
                    idx.insert(n.as_str(), totals.len());
                    order.push(n.clone());
                    totals.push(ms);
                }
            }
        }
        order.into_iter().zip(totals).collect()
    }

    /// Total wall time across all recorded spans (ms). Note: nested stages
    /// double-count, so this is a sum of spans, not an exclusive total.
    pub fn total_millis(&self) -> f64 {
        self.spans.iter().map(|(_, d)| d.as_secs_f64() * 1e3).sum()
    }
}

/// Time an expression under `prof`, recording it as `name`, and yield its value:
/// `let coo = time_stage!(prof, "jac_x_coo", dae.jacobian_x_coo(ctx));`.
/// The duration is also sent to the global logger (DEBUG), so a profiled
/// pipeline's stages show up in the log without surfacing the [`Profile`].
#[macro_export]
macro_rules! time_stage {
    ($prof:expr, $name:expr, $body:expr) => {{
        let __t = $crate::time::Instant::now();
        let __r = $body;
        let __d = __t.elapsed();
        $prof.record($name, __d);
        $crate::log::stage($name, __d);
        __r
    }};
}

/// Time an expression and log it as a stage at DEBUG, with no [`Profile`] handle
/// to thread: `let b = log_stage!("ac/eval_b", eval_real(ctx, &env, &db));`.
/// For the many stateless analysis functions that have nowhere to keep a
/// `Profile` but should still report where their time goes.
#[macro_export]
macro_rules! log_stage {
    ($name:expr, $body:expr) => {{
        let __t = $crate::time::Instant::now();
        let __r = $body;
        $crate::log::stage($name, __t.elapsed());
        __r
    }};
}
