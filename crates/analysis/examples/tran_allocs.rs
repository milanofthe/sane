//! The heap allocations of a transient: a counting allocator around the
//! system one, the solve of a deck at a capped step, and the allocations
//! per step. The loop-closing tool for an allocation-free stepping loop.
//!
//! ```text
//! cargo run -q --release --example tran_allocs -- graetz.cir 1 1e-6 trap
//! ```
//!
//! Args: `deck tstop dt_max [esdirk32|trap]`. With `ALLOC_SITES=N` every
//! `N`th allocation of the timed run records its backtrace, and the most
//! frequent call sites are listed; build with `--profile profiling` for
//! file and line.

use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use sane_analysis::Model;
use sane_solve::TransientMethod;

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
/// Sample every `SAMPLE`th allocation, 0 for none.
static SAMPLE: AtomicU64 = AtomicU64::new(0);
static SITES: Mutex<Vec<Backtrace>> = Mutex::new(Vec::new());

std::thread_local! {
    /// Inside the sampler: its own allocations are not sampled.
    static SAMPLING: Cell<bool> = const { Cell::new(false) };
}

fn count(size: usize) {
    let k = ALLOCS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(size as u64, Ordering::Relaxed);
    let every = SAMPLE.load(Ordering::Relaxed);
    if every > 0 && k.is_multiple_of(every) && !SAMPLING.with(|s| s.replace(true)) {
        let bt = Backtrace::force_capture();
        SITES.lock().unwrap().push(bt);
        SAMPLING.with(|s| s.set(false));
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The first frames of a backtrace outside the allocator and the standard
/// library, one per line.
fn site(bt: &Backtrace, depth: usize) -> String {
    let text = bt.to_string();
    let mut frames = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let Some((_, name)) = line.trim().split_once(": ") else {
            continue;
        };
        let at = match lines.peek() {
            Some(l) if l.trim().starts_with("at ") => lines.next().unwrap().trim(),
            _ => "",
        };
        let foreign = [
            "std::", "core::", "alloc::", "<alloc::", "<core::", "<std::",
        ];
        if name.contains("tran_allocs") || foreign.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        let at = at.rsplit(['\\', '/']).next().unwrap_or("");
        frames.push(format!("  {name} ({at})"));
        if frames.len() == depth {
            break;
        }
    }
    frames.join("\n")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: tran_allocs deck.cir tstop dt_max [esdirk32|trap]";
    let path = args.next().expect(usage);
    let tstop: f64 = args.next().and_then(|a| a.parse().ok()).expect(usage);
    let dt_max: f64 = args.next().and_then(|a| a.parse().ok()).expect(usage);
    let method = TransientMethod::from_name(&args.next().unwrap_or_default()).expect(usage);
    let every: u64 = std::env::var("ALLOC_SITES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let model = Model::from_netlist(&std::fs::read_to_string(&path).expect("read deck"))
        .expect("build model");
    let steps = (tstop / dt_max).round() as usize;
    let npts = (steps + 1).min(100_001);
    let t: Vec<f64> = (0..npts)
        .map(|k| tstop * k as f64 / (npts - 1) as f64)
        .collect();
    let p = model.pvec(&[]);
    // a short run first: the programs compiled, the buffers grown
    let warm: Vec<f64> = t.iter().copied().take(3).collect();
    model
        .solve_transient(method, p.clone(), warm, None, 1e-4, 1e-7, Some(dt_max))
        .expect("transient");

    let (a0, b0) = (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    );
    SAMPLE.store(every, Ordering::Relaxed);
    let t0 = Instant::now();
    model
        .solve_transient(method, p, t, None, 1e-4, 1e-7, Some(dt_max))
        .expect("transient");
    let secs = t0.elapsed().as_secs_f64();
    SAMPLE.store(0, Ordering::Relaxed);
    let (a, b) = (
        ALLOCS.load(Ordering::Relaxed) - a0,
        BYTES.load(Ordering::Relaxed) - b0,
    );
    println!(
        "{path} {method:?}: {secs:.2} s, {a} allocations ({:.1} per step), {:.1} MB",
        a as f64 / steps.max(1) as f64,
        b as f64 / 1e6
    );

    let sites = std::mem::take(&mut *SITES.lock().unwrap());
    if sites.is_empty() {
        return;
    }
    let mut tally: HashMap<String, usize> = HashMap::new();
    for bt in &sites {
        *tally.entry(site(bt, 4)).or_default() += 1;
    }
    let mut tally: Vec<_> = tally.into_iter().collect();
    tally.sort_by_key(|s| std::cmp::Reverse(s.1));
    for (site, k) in tally.iter().take(12) {
        println!(
            "{:5.1}% ({k} samples)\n{site}",
            100.0 * *k as f64 / sites.len() as f64
        );
    }
}
