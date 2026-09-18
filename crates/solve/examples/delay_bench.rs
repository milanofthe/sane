//! Microbenchmark for the DelayHistory hot paths: monotone stage queries
//! (the inner loop), retreat/random queries, and push with eviction.
//!
//!     cargo run --release -p sane-solve --example delay_bench

use std::time::Instant;

use sane_solve::DelayHistory;

fn main() {
    // realistic shape: 4 delayed signals, ~200 in-horizon knots (adaptive
    // steps inside one delay window), millions of stage queries
    let nsig = 4;
    let mut h = DelayHistory::new(nsig, 1.0);
    for k in 0..200 {
        let t = k as f64 * 0.005;
        let vals: Vec<f64> = (0..nsig).map(|s| (t + s as f64).sin()).collect();
        let ders: Vec<f64> = (0..nsig).map(|s| (t + s as f64).cos()).collect();
        h.push(t, &vals, &ders);
    }

    // monotone queries (the stage pattern: forward march, tiny increments)
    let n = 20_000_000usize;
    let t0 = Instant::now();
    let mut acc = 0.0;
    for k in 0..n {
        let tq = 0.9 * (k as f64 / n as f64) + 0.05;
        acc += h.eval(k % nsig, tq);
    }
    let per = t0.elapsed().as_secs_f64() / n as f64 * 1e9;
    println!("eval monotone : {per:.2} ns/op   (acc {acc:.3})");

    // retreat-heavy queries (rejected-step pattern)
    let t0 = Instant::now();
    let mut acc = 0.0;
    for k in 0..n {
        let base = 0.9 * (k as f64 / n as f64) + 0.05;
        let tq = if k % 7 == 0 { base - 0.01 } else { base };
        acc += h.eval(k % nsig, tq);
    }
    let per = t0.elapsed().as_secs_f64() / n as f64 * 1e9;
    println!("eval w/retreat: {per:.2} ns/op   (acc {acc:.3})");

    // random access (bisect fallback)
    let t0 = Instant::now();
    let mut acc = 0.0;
    let mut state = 0x9e3779b97f4a7c15u64;
    for k in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let tq = (state % 1000) as f64 * 1e-3;
        acc += h.eval(k % nsig, tq);
    }
    let per = t0.elapsed().as_secs_f64() / n as f64 * 1e9;
    println!("eval random   : {per:.2} ns/op   (acc {acc:.3})");

    // push with eviction active (long run, bounded window)
    let m = 20_000_000usize;
    let mut h = DelayHistory::new(nsig, 0.5);
    let vals = vec![1.0; nsig];
    let ders = vec![0.0; nsig];
    let t0 = Instant::now();
    for k in 0..m {
        h.push(k as f64 * 1e-3, &vals, &ders);
    }
    let per = t0.elapsed().as_secs_f64() / m as f64 * 1e9;
    println!("push + evict  : {per:.2} ns/op   (knots kept: {})", h.len());
}
