//! VACASK ring benchmark (9-stage PSP103 CMOS ring, 1 us at dt_max = 50 ps),
//! runnable over both device paths for a same-methodology comparison:
//!
//!   cargo run --release -p sane-analysis --example ring_bench -- osdi <psp103.osdi> <models.inc> [t_end] [rtol] [atol] [method]
//!   cargo run --release -p sane-analysis --example ring_bench -- va   <psp103.va>   <models.inc> [t_end] [rtol] [atol] [method]
//!
//! Methodology mirrors `paper/benchmarks/bench/bench_vacask.py` (`ring` case):
//! rtol 1e-4, atol 1e-7, forced dt_max 5e-11, DC start, swing check on n1.
//! `t_end` (seconds, default 1e-6) scales the span for quick probes.

use sane_analysis::Model;
use sane_solve::TransientMethod;

const STAGES: usize = 9;
const VDD: f64 = 1.2;
const W: f64 = 10e-6;
const L: f64 = 1e-6;
const PFACT: f64 = 2.0;
const LDS: f64 = 0.5e-6;
const DT_MAX: f64 = 5e-11;

fn geo(w: f64) -> String {
    let ad = w * LDS;
    let pd = 2.0 * (w + LDS);
    format!("W={w:e} L={L:e} AD={ad:e} AS={ad:e} PD={pd:e} PS={pd:e}")
}

fn deck(mode: &str, model_path: &str, cards: &str) -> String {
    let load = match mode {
        "osdi" => format!(".osdi \"{model_path}\""),
        "va" => format!(".veriloga \"{model_path}\""),
        m => panic!("mode must be osdi|va, got {m}"),
    };
    let mut lines = vec![
        "* 9-stage ring oscillator -- VACASK benchmark suite (ring)".to_string(),
        load,
        cards.to_string(),
        format!("vdd vdd 0 {VDD}"),
        "i0 0 n1 dc 0 pulse 0 10u 1n 1n 1n 1n 1".to_string(),
    ];
    for k in 1..=STAGES {
        let (inn, out) = (format!("n{k}"), format!("n{}", k % STAGES + 1));
        lines.push(format!(
            "NMP{k} {out} {inn} vdd vdd psp103p {}",
            geo(W * PFACT)
        ));
        lines.push(format!("NMN{k} {out} {inn} 0 0 psp103n {}", geo(W)));
    }
    lines.push(".end".to_string());
    lines.join("\n")
}

fn main() {
    sane_core::log::init_from_env();
    let args: Vec<String> = std::env::args().collect();
    let (mode, model_path, cards_path) = (&args[1], &args[2], &args[3]);
    let t_end: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1e-6);
    let rtol: f64 = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(1e-4);
    let atol: f64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1e-7);
    let method = TransientMethod::from_name(args.get(7).map(String::as_str).unwrap_or(""))
        .expect("method: esdirk32|trap");
    let cards = std::fs::read_to_string(cards_path).expect("model cards");

    let t0 = std::time::Instant::now();
    let model = Model::from_netlist(&deck(mode, model_path, &cards)).expect("model");
    let built = t0.elapsed().as_secs_f64();
    println!("[{mode}] dim {}  built {built:.2} s", model.dim());

    let npts = ((t_end / DT_MAX).round() as usize + 1).min(100_001);
    let t_eval: Vec<f64> = (0..npts)
        .map(|i| t_end * i as f64 / (npts - 1) as f64)
        .collect();
    let p = model.pvec(&[]);

    let t0 = std::time::Instant::now();
    let rows = model
        .solve_transient(method, p, t_eval.clone(), None, rtol, atol, Some(DT_MAX))
        .expect("transient");
    let solve = t0.elapsed().as_secs_f64();

    let i = model.resolve("n1").expect("n1");
    let tail: Vec<f64> = rows[rows.len() / 2..].iter().map(|r| r[i]).collect();
    let swing = tail.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - tail.iter().cloned().fold(f64::INFINITY, f64::min);
    let ok = (0.6 * VDD..=1.2 * VDD).contains(&swing);
    println!(
        "[{mode}] solve {solve:.2} s  span {t_end:e}  rtol {rtol:e}  steps>= {}  swing {swing:.3} {}",
        npts - 1,
        if ok { "OK" } else { "FAIL" }
    );
}
