//! Profile the extraction pipeline (parse + DAE build + CompiledDc::new) for one
//! netlist, breaking the wall time into the instrumented stages and the
//! unaccounted remainder (parse + DAE assembly, which carry no `time_stage!`).
//!
//!   cargo run --release --example prof_extract -- path.cir
use sane_analysis::Model;
use sane_core::profile::{self, Profile};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: prof_extract <path.cir>");
    let src = std::fs::read_to_string(&path).expect("read netlist");

    profile::collect_begin();
    let t0 = Instant::now();
    let model = Model::from_netlist(&src).expect("build model");
    let wall = t0.elapsed().as_secs_f64() * 1e3;
    let prof: Profile = profile::collect_take();

    let mut stages = prof.aggregated_millis();
    stages.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let staged: f64 = stages.iter().map(|(_, t)| *t).sum();

    println!("circuit {}  dim {}", path, model.dim());
    println!("  wall (extract) {wall:.1} ms");
    for (n, t) in stages.iter().filter(|(_, t)| *t >= 0.5) {
        println!("    {n:<16} {t:>8.1} ms  ({:.0}%)", t / wall * 100.0);
    }
    println!(
        "    {:<16} {:>8.1} ms  ({:.0}%)  [parse + DAE build, uninstrumented]",
        "unaccounted",
        wall - staged,
        (wall - staged) / wall * 100.0
    );
}
