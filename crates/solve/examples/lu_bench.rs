//! Sparse-LU backend benchmark: times the engine paths that are dominated by
//! the sparse factorization (DC Newton on a diode ladder, DC on a large linear
//! RC mesh with a few devices, transient on an RC ladder), for A/B comparison
//! of the linear-solver backend across commits.
//!
//!   cargo run -q --release --example lu_bench
//!
//! Prints one line per case: geometry, wall-clock per solve, and (where
//! applicable) Newton iteration counts, so backend swaps can be compared
//! commit-vs-commit on identical circuits.

use std::time::Instant;

use sane_core::Graph;
use sane_dae::{assemble_dae, Dae, DeviceInstance};
use sane_mna::Circuit;
use sane_solve::{CompiledDc, TransientMethod};

fn params(ctx: &Graph, dae: &Dae, vsrc: f64, r: f64, cval: f64) -> Vec<f64> {
    let tnom = sane_core::constants::TEMP_NOMINAL_K;
    dae.params(ctx)
        .iter()
        .map(|&s| {
            let n = ctx.symbol_name(s);
            if n.starts_with('V') && !n.contains('.') {
                vsrc
            } else if n.starts_with('R') && !n.contains('.') {
                r
            } else if n.starts_with('C') && !n.contains('.') {
                cval
            } else if n.ends_with(".Is") {
                1e-12
            } else if n.ends_with(".N") {
                1.0
            } else if n == sane_core::constants::TEMP_SYMBOL || n.ends_with(".Tnom") {
                tnom
            } else if n.ends_with(".Eg") {
                1.11
            } else if n.ends_with(".XTI") {
                3.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Diode ladder (nonlinear DC Newton: every node has a junction).
fn diode_ladder(k: usize) -> (Graph, Dae) {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0);
    let mut devs = Vec::new();
    for i in 1..=k {
        c.resistor(&format!("R{i}"), i, i + 1);
        devs.push(DeviceInstance::new(
            Box::new(sane_veriloga::builtin_device(
                "sane_diode",
                format!("D{i}"),
                &[],
            )),
            vec![i + 1, 0],
        ));
    }
    let dae = assemble_dae(&mut ctx, &c, &devs);
    (ctx, dae)
}

/// Large linear RC grid (w x h nodes, resistors along both axes, C to ground)
/// with a handful of diodes embedded -- exercises the partitioned/linear path
/// and the stage factorization at scale.
fn rc_grid(w: usize, h: usize, diodes: usize) -> (Graph, Dae) {
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    let id = |x: usize, y: usize| 1 + y * w + x;
    c.voltage_source("V1", id(0, 0), 0);
    let mut r = 0usize;
    for y in 0..h {
        for x in 0..w {
            if x + 1 < w {
                r += 1;
                c.resistor(&format!("R{r}"), id(x, y), id(x + 1, y));
            }
            if y + 1 < h {
                r += 1;
                c.resistor(&format!("R{r}"), id(x, y), id(x, y + 1));
            }
            c.capacitor(&format!("C{}", id(x, y)), id(x, y), 0);
        }
    }
    let mut devs = Vec::new();
    for d in 0..diodes {
        let node = id((d * 7 + 3) % w, (d * 5 + 2) % h);
        devs.push(DeviceInstance::new(
            Box::new(sane_veriloga::builtin_device(
                "sane_diode",
                format!("D{d}"),
                &[],
            )),
            vec![node, 0],
        ));
    }
    let dae = assemble_dae(&mut ctx, &c, &devs);
    (ctx, dae)
}

fn time_best<R>(reps: usize, mut f: impl FnMut() -> R) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let _ = f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    // 1) Nonlinear DC Newton: diode ladders of growing size.
    for &k in &[200usize, 1000, 4000] {
        let (mut ctx, dae) = diode_ladder(k);
        let t_ext = Instant::now();
        let cdc = CompiledDc::new(&mut ctx, &dae);
        let ext_ms = t_ext.elapsed().as_secs_f64() * 1e3;
        let p = params(&ctx, &dae, 3.0, 1000.0, 0.0);
        let (_, conv, iters) = cdc.solve_dc(&p, &[], 1e-9, 200);
        assert!(conv, "diode ladder k={k} must converge");
        let best = time_best(5, || cdc.solve_dc(&p, &[], 1e-9, 200));
        println!(
            "dc  diode-ladder  k={k:5}  dim={:6}  extract {ext_ms:8.1} ms  solve {:9.2} ms  ({iters} iters)",
            cdc.dim(),
            best * 1e3
        );
    }

    // 2) Mostly-linear DC at scale (partition + full LU fallback paths). The
    //    optional argument adds a `g x g` mesh case (power-grid regime).
    let big: Option<usize> = std::env::args().nth(1).and_then(|s| s.parse().ok());
    let mut grids: Vec<(usize, usize)> = vec![(40, 40), (90, 90)];
    if let Some(g) = big {
        grids.push((g, g));
    }
    for &(w, h) in &grids {
        // Pure linear grid (no devices): the power-grid regime, no partition,
        // Newton converges in 1-2 iterations -- factor-dominated.
        {
            let (mut ctx, dae) = rc_grid(w, h, 0);
            let cdc = CompiledDc::new(&mut ctx, &dae);
            let p = params(&ctx, &dae, 1.0, 100.0, 1e-9);
            let (_, conv, iters) = cdc.solve_dc(&p, &[], 1e-9, 200);
            assert!(conv, "linear rc grid {w}x{h} must converge");
            let best = time_best(3, || cdc.solve_dc(&p, &[], 1e-9, 200));
            println!(
                "dc  rc-grid-lin  {w:3}x{h:<3}  dim={:6}  solve {:9.2} ms  ({iters} iters)",
                cdc.dim(),
                best * 1e3
            );
        }
        let (mut ctx, dae) = rc_grid(w, h, 8);
        let t_ext = Instant::now();
        let cdc = CompiledDc::new(&mut ctx, &dae);
        let ext_ms = t_ext.elapsed().as_secs_f64() * 1e3;
        let p = params(&ctx, &dae, 1.0, 100.0, 1e-9);
        let (_, conv, iters) = cdc.solve_dc(&p, &[], 1e-9, 200);
        assert!(conv, "rc grid {w}x{h} must converge");
        let best = time_best(3, || cdc.solve_dc(&p, &[], 1e-9, 200));
        println!(
            "dc  rc-grid      {w:3}x{h:<3}  dim={:6}  extract {ext_ms:8.1} ms  solve {:9.2} ms  ({iters} iters)",
            cdc.dim(),
            best * 1e3
        );
    }

    // 3) Transient (stage factorization + repeated stage solves).
    {
        let (mut ctx, dae) = rc_grid(30, 30, 4);
        let cdc = CompiledDc::new(&mut ctx, &dae);
        let p = params(&ctx, &dae, 1.0, 100.0, 1e-9);
        let t_eval: Vec<f64> = (0..=100).map(|i| i as f64 * 1e-9).collect();
        let best = time_best(3, || {
            cdc.solve_transient(
                TransientMethod::Esdirk32,
                &p,
                &[],
                &t_eval,
                1e-6,
                1e-9,
                None,
            )
            .expect("transient")
        });
        println!(
            "tran rc-grid      30x30   dim={:6}  {:9.2} ms / 100ns span",
            cdc.dim(),
            best * 1e3
        );
    }
}
