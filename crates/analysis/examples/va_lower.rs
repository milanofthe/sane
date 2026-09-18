//! Lowering benchmark for the bundled Verilog-A compact models (EKV, VBIC,
//! HICUM, BSIM4). Builds a minimal one-instance deck per model and times
//! `Model::from_netlist` (which parses + lowers the model). The lowering happens
//! at build, before any DC solve, so a non-converging topology is fine.
//!
//! ```text
//! SANE_LOG=debug cargo run -q --release --example va_lower -- ekv
//! ```
//! Args: model name(s) from {ekv, vbic, hicum, bsim4}; default all but bsim4.

use std::time::Instant;

use sane_analysis::Model;

const M: &str = "paper/benchmarks/models";

fn deck(model: &str) -> Option<String> {
    Some(match model {
        "ekv" => format!(
            ".veriloga \"{M}/ekv/vacode/ekv26.va\"\n\
             Vd d 0 1\nVg g 0 1\nVs s 0 0\nVb b 0 0\n\
             N1 d g s b ekv26_va W=10u L=1u\n.end\n"
        ),
        "vbic" => format!(
            ".veriloga \"{M}/vbic/vacode/vbic_1p3.va\"\n\
             Vc c 0 5\nVb b 0 0.8\nVe e 0 0\nVx s 0 0\n\
             N1 c b e s vbic13_4t\n.end\n"
        ),
        "hicum" => format!(
            ".veriloga \"{M}/hicum0/vacode/hicumL0_v2p1p0.va\"\n\
             Vc c 0 1\nVb b 0 0.8\nVe e 0 0\nVx s 0 0\nVt t 0 0\n\
             N1 c b e s t hicumL0va\n.end\n"
        ),
        "bsim4" => format!(
            ".veriloga \"{M}/bsim4/bsim4.va\"\n\
             Vd d 0 1\nVg g 0 1\nVs s 0 0\nVb b 0 0\n\
             N1 d g s b bsim4va L=1u W=1u\n.end\n"
        ),
        _ => return None,
    })
}

fn main() {
    sane_analysis::logging::init_from_env();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<String> = if args.is_empty() {
        ["ekv", "vbic", "hicum"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        args
    };
    for m in &models {
        let Some(src) = deck(m) else {
            eprintln!("{m}: unknown model");
            continue;
        };
        let t0 = Instant::now();
        match Model::from_netlist(&src) {
            Ok(model) => println!(
                "{m:<8} dim={:<5} build={:.1} ms",
                model.dim(),
                t0.elapsed().as_secs_f64() * 1e3
            ),
            Err(e) => println!(
                "{m:<8} FAILED ({:.1} ms): {e:?}",
                t0.elapsed().as_secs_f64() * 1e3
            ),
        }
    }
}
