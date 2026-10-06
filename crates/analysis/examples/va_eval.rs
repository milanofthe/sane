//! What a deck's devices cost: the build (netlist, Verilog-A or OSDI
//! models, extraction, tapes), the DC solve, the time until native code
//! serves, and per call the DC program (`I`, `G`) and the transient
//! program (`I`, `Q`, `G`, `C`), interpreted and native. Run on the same
//! deck with `.veriloga` and with `.osdi` models (or against
//! `sane-osdi`'s `osdi_ring`), it sets SANE's Verilog-A compiler against
//! OpenVAF's.
//!
//! ```text
//! cargo run -q --release --example va_eval -- ring.cir [threads]
//! ```
//!
//! Prints one line of `key=value` pairs.

use std::time::{Duration, Instant};

use sane_analysis::Model;
use sane_solve::Evaluator;

/// The interpreted median and the native best time of `ev` per call, in
/// microseconds, and the time until native code served.
fn timing(ev: &mut Evaluator<'_>, xs: [&[f64]; 2]) -> (f64, f64, f64) {
    let (mut interp, t) = (Vec::new(), Instant::now());
    while !ev.native() && t.elapsed() < Duration::from_secs(300) {
        let s = Instant::now();
        std::hint::black_box(ev.eval(xs[interp.len() % 2]));
        interp.push(s.elapsed().as_secs_f64());
    }
    let to_native = t.elapsed().as_secs_f64();
    interp.sort_by(f64::total_cmp);
    let s = Instant::now();
    std::hint::black_box(ev.eval(xs[0]));
    let k = ((0.02 / s.elapsed().as_secs_f64()) as usize).clamp(1, 100_000);
    let mut best = f64::INFINITY;
    for _ in 0..25 {
        let s = Instant::now();
        for i in 0..k {
            std::hint::black_box(ev.eval(xs[i % 2]));
        }
        best = best.min(s.elapsed().as_secs_f64() / k as f64);
    }
    let median = interp.get(interp.len() / 2).copied().unwrap_or(f64::NAN);
    (median * 1e6, best * 1e6, to_native)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: va_eval deck.cir [threads]");
    if let Some(n) = args.next().and_then(|a| a.parse().ok()) {
        sane_solve::set_parallelism(n);
    }
    let src = std::fs::read_to_string(&path).expect("read deck");

    let t = Instant::now();
    let model = Model::from_netlist(&src).expect("build model");
    let build = t.elapsed();
    let cdc = model.cdc();
    let p = model.pvec(&[]);

    let t = Instant::now();
    let (x, converged, iters) = cdc.solve_dc(&p, &[], 1e-9, 200);
    let dc = t.elapsed();
    assert!(converged, "DC did not converge");

    // Two states a microvolt apart, alternated, so no evaluation sees the
    // inputs of the one before.
    let x2: Vec<f64> = x.iter().map(|v| v + 1e-6).collect();
    let xs = [&x[..], &x2[..]];
    let (dc_interp, dc_native, dc_ready) = timing(&mut cdc.dc_evaluator(&p), xs);
    let (tr_interp, tr_native, tr_ready) = timing(&mut cdc.transient_evaluator(&p), xs);

    println!(
        "dim={} build_s={:.4} dc_s={:.4} dc_iters={} \
         dc_interp_us={dc_interp:.2} dc_native_us={dc_native:.2} dc_ready_s={dc_ready:.3} \
         tran_interp_us={tr_interp:.2} tran_native_us={tr_native:.2} tran_ready_s={tr_ready:.3}",
        cdc.dim(),
        build.as_secs_f64(),
        dc.as_secs_f64(),
        iters,
    );
}
