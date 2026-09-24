//! Warm-DC gate benchmark for the lazy JIT wiring (M2): repeated warm
//! `solve_dc` on a real deck, comparing interpreted vs native hot tapes in one
//! process (before/after the background compile swaps in), plus bit-identity
//! of the solved operating point.
//!
//!   cargo run -q --release --example dc_jit_bench -- <deck.cir>

use sane_core::Graph;
use sane_dae::assemble_dae;
use sane_solve::CompiledDc;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dc_jit_bench <deck.cir>");
    let text = std::fs::read_to_string(&path).expect("read deck");
    let parsed = sane_netlist::parse(&text).expect("parse deck");
    let mut ctx = Graph::new();
    let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    // Parameter vector from the deck's bound values (the same source the
    // analysis-layer ParamStore uses), unbound entries at 0, $temp nominal.
    let p: Vec<f64> = cdc
        .param_names(&ctx)
        .iter()
        .map(|n| {
            if n == "$temp" {
                sane_core::constants::TEMP_NOMINAL_K
            } else {
                parsed.param_value(n).unwrap_or(0.0)
            }
        })
        .collect();

    // Cold solve (interpreted; also the reference solution).
    let (x0, conv, _it) = cdc.solve_dc(&p, &[], 1e-10, 200);
    assert!(conv, "cold DC did not converge");

    // Warm loop, interpreted phase: the first samples run before the background
    // compile lands (kick threshold is a handful of evals).
    // Settle: spin (untimed) until the backend choice is stable and the
    // process sits on performance cores -- a sleeping process gets demoted to
    // efficiency cores on Apple Silicon and phase-to-phase comparisons inside
    // one process measure the migration, not the backend. The meaningful
    // comparison is therefore across processes: run once with SANE_JIT=0 and
    // once with SANE_JIT=1 and compare the settled numbers.
    let settle_until = Instant::now() + std::time::Duration::from_millis(2500);
    while Instant::now() < settle_until {
        let (_, c, _) = cdc.solve_dc(&p, &x0, 1e-10, 200);
        assert!(c);
    }
    let mut last = Vec::new();
    let mut reps = 0usize;
    let t0 = Instant::now();
    let run_until = t0 + std::time::Duration::from_millis(2000);
    while Instant::now() < run_until {
        let (x, c, _) = cdc.solve_dc(&p, &x0, 1e-10, 200);
        assert!(c);
        last = x;
        reps += 1;
    }
    let per = t0.elapsed().as_secs_f64() / reps as f64;
    // FNV-1a over the solution bits: identical across a SANE_JIT=0 and a
    // SANE_JIT=1 run iff the solved operating points are bit-identical.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in &last {
        for b in v.to_bits().to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x1000_0000_01b3);
        }
    }
    // Cold solves exercise the multi-iteration Newton loop (the prolog-split
    // regime: the parameter-pure prefix runs once, every further iteration
    // skips it).
    let mut cold_reps = 0usize;
    let t0 = Instant::now();
    let run_until = t0 + std::time::Duration::from_millis(2000);
    while Instant::now() < run_until {
        let (_, c, _) = cdc.solve_dc(&p, &[], 1e-10, 200);
        assert!(c);
        cold_reps += 1;
    }
    let cold = t0.elapsed().as_secs_f64() / cold_reps as f64;

    println!(
        "dim {} | warm DC {:.4} ms ({} reps) | cold DC {:.4} ms ({} reps) | solution fnv {h:016x}",
        x0.len(),
        per * 1e3,
        reps,
        cold * 1e3,
        cold_reps,
    );
}
