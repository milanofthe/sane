//! DC operating-point benchmark over a directory of decks: Newton iterations,
//! convergence and wall time per deck under the default tricks and with the
//! composite (Traub) step off, so a solver change is measured on the corpus
//! before it becomes the default.
//!
//! ```text
//! cargo run -q --release --example dc_bench -- crates/netlist/tests/fixtures
//! ```

use std::time::Instant;

use rsdag::Graph;
use sane_core::constants::DC_OP_MAXIT;
use sane_dae::assemble_dae;
use sane_netlist::parse;
use sane_solve::{CompiledDc, Convergence, SolverTricks};

fn main() {
    sane_analysis::logging::init_from_env();
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "crates/netlist/tests/fixtures".to_string());
    let mut decks: Vec<_> = std::fs::read_dir(&dir)
        .expect("deck dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "cir" || x == "spice"))
        .collect();
    decks.sort();
    let variants: [(&str, SolverTricks); 3] = [
        ("default", SolverTricks::default()),
        (
            "no-composite",
            SolverTricks {
                composite_step: false,
                ..SolverTricks::default()
            },
        ),
        (
            "limit-correctors",
            SolverTricks {
                device_limiting: true,
                ..SolverTricks::default()
            },
        ),
    ];
    println!(
        "{:<28} {:>5} | {:>10} {:>9} | {:>10} {:>9} | {:>10} {:>9}",
        "deck", "dim", "iters(def)", "ms(def)", "iters(nc)", "ms(nc)", "iters(lc)", "ms(lc)"
    );
    let mut totals = [(0usize, 0.0f64, 0usize); 3];
    for path in &decks {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(parsed) = parse(&src) else {
            println!(
                "{:<28} parse error",
                path.file_name().unwrap().to_string_lossy()
            );
            continue;
        };
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        let mut cdc = CompiledDc::new(&mut ctx, &dae);
        let p = parsed.pvec(&cdc.param_names(&ctx));
        let mut row = format!(
            "{:<28} {:>5} |",
            path.file_name().unwrap().to_string_lossy(),
            cdc.dim()
        );
        // Warm the JIT and caches, then three rounds alternating the variants;
        // the best time of each is reported (order and background compiles
        // otherwise bias the first variant).
        for (_, tricks) in &variants {
            cdc.set_tricks(*tricks);
            let _ = cdc.solve_dc_conv_with(&p, &[], Convergence::default(), DC_OP_MAXIT, *tricks);
        }
        let mut best = [(0usize, false, f64::INFINITY); 3];
        for _round in 0..3 {
            for (k, (_, tricks)) in variants.iter().enumerate() {
                cdc.set_tricks(*tricks);
                let t = Instant::now();
                let (_, conv, iters) =
                    cdc.solve_dc_conv_with(&p, &[], Convergence::default(), DC_OP_MAXIT, *tricks);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if ms < best[k].2 {
                    best[k] = (iters, conv, ms);
                }
            }
        }
        for (k, _) in variants.iter().enumerate() {
            let (iters, conv, ms) = best[k];
            totals[k].0 += iters;
            totals[k].1 += ms;
            totals[k].2 += usize::from(!conv);
            row.push_str(&format!(
                " {:>9}{} {:>9.3} |",
                iters,
                if conv { " " } else { "!" },
                ms
            ));
        }
        println!("{row}");
    }
    println!(
        "{:<28} {:>5} | {:>10} {:>9} | {:>10} {:>9} | {:>10} {:>9}   (! = not converged; failures {} / {} / {})",
        "total",
        "",
        totals[0].0,
        format!("{:.1}", totals[0].1),
        totals[1].0,
        format!("{:.1}", totals[1].1),
        totals[2].0,
        format!("{:.1}", totals[2].1),
        totals[0].2,
        totals[1].2,
        totals[2].2
    );
}
