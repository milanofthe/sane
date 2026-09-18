//! c6288 DC study: which cascade stage (if any) reaches the operating point of
//! a 10k-transistor digital CMOS net, and how long it takes.
//!
//! Usage: cargo run --release -p sane-analysis --example c6288_ptc -- <deck> [mode]
//!   mode = full (default: the stock cascade) | ptc (pseudo-transient only)
//!
//! The deck comes from `paper/benchmarks/bench/bench_vacask.py:c6288_deck()`
//! (write it out with `--emit-deck`); passing it in keeps this example free of
//! benchmark paths.

use sane_core::time::Instant;
use sane_solve::SolverTricks;

fn main() {
    sane_core::log::init_from_env();
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: c6288_ptc <deck.cir> [full|ptc]");
    let mode = args.next().unwrap_or_else(|| "full".into());
    let deck = std::fs::read_to_string(&path).expect("read deck");

    let t0 = Instant::now();
    let model = sane_analysis::Model::from_netlist(&deck).expect("model");
    println!(
        "extract: {:.1} s, dim {}",
        t0.elapsed().as_secs_f64(),
        model.dim()
    );

    // `ptc`: skip every static continuation so the pseudo-transient stage runs
    // first -- the question is whether it converges at all, not whether the
    // cascade eventually reaches it.
    let tricks = if mode == "ptc" {
        SolverTricks {
            gmin_continuation: false,
            source_continuation: false,
            companion_continuation: false,
            node_adaptive: false,
            pseudo_transient: true,
            ..SolverTricks::default()
        }
    } else {
        SolverTricks::default()
    };
    let cdc = model.cdc();
    let p = model.pvec(&[]);
    let x0 = vec![0.0; cdc.dim()];
    let t0 = Instant::now();
    let (x, ok, iters) = cdc.solve_dc_with(&p, &x0, 1e-9, 200, tricks);
    let dt = t0.elapsed().as_secs_f64();
    if ok {
        let (lo, hi) = x
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        println!(
            "DC[{mode}]: converged in {dt:.1} s ({iters} iterations), nodes {lo:.3} .. {hi:.3} V"
        );
    } else {
        println!("DC[{mode}]: FAILED after {dt:.1} s ({iters} iterations)");
    }
}
