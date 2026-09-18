//! Transient of a netlist deck with the step statistics and the switching
//! events: the loop-closing tool for event-driven circuits (switched-mode
//! converters, comparators, relays).
//!
//! ```text
//! SANE_LOG=debug cargo run -q --release --example deck_tran -- buck.cir 2e-3 2001 vout
//! SANE_EVENTS=0 cargo run -q --release --example deck_tran -- buck.cir 2e-3 2001 vout
//! ```
//!
//! Args: `deck [tstop] [npts] [node...]`. Prints the build and solve wall
//! times, the events (count, first few by name and time) and, for each named
//! node, its value at the last point and its min/max over the run.

use std::time::Instant;

use sane_analysis::Model;
use sane_solve::TransientMethod;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: deck_tran deck.cir [tstop] [npts] [node...]");
    let tstop: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(1e-3);
    let npts: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1001);
    let nodes: Vec<String> = args.collect();

    sane_analysis::logging::init_from_env();
    let deck = std::fs::read_to_string(&path).expect("read deck");
    let t0 = Instant::now();
    let model = Model::from_netlist(&deck).expect("build model");
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t: Vec<f64> = (0..npts)
        .map(|k| tstop * k as f64 / (npts - 1).max(1) as f64)
        .collect();
    let t1 = Instant::now();
    let traj = model
        .transient(TransientMethod::Esdirk32, &[], &t, 1e-4, 1e-7)
        .expect("transient");
    let solve_ms = t1.elapsed().as_secs_f64() * 1e3;
    let events = model.transient_events();

    println!(
        "{path}: dim={} build={build_ms:.1} ms solve={solve_ms:.1} ms events={}",
        model.dim(),
        events.len()
    );
    for (name, te, dir) in events.iter().take(6) {
        println!(
            "  event {name} at {te:.6e} ({})",
            if *dir > 0 { "rising" } else { "falling" }
        );
    }
    if events.len() > 6 {
        println!("  ... {} more", events.len() - 6);
    }
    let rows = traj.rows();
    for node in &nodes {
        let Some(k) = model.resolve(node) else {
            println!("  {node}: unknown node");
            continue;
        };
        let last = rows.last().map(|r| r[k]).unwrap_or(f64::NAN);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for r in rows {
            lo = lo.min(r[k]);
            hi = hi.max(r[k]);
        }
        println!("  {node}: last={last:.6} min={lo:.6} max={hi:.6}");
    }
}
