//! Scaling benchmark against external simulators (circulax comparison):
//! diode-ladder DC and RC-ladder transient, sizes from argv.
//!
//!     cargo run --release -p sane-analysis --example ladder_bench -- 10 50 200

use std::time::Instant;

use sane_analysis::Model;

fn rc_ladder(n: usize) -> String {
    let mut s = String::from("* rc ladder\nV1 n0 0 PULSE(0 1 1u 1u 1u 1m 2m)\n");
    for i in 1..=n {
        s.push_str(&format!("R{i} n{} n{i} 1k\n", i - 1));
        s.push_str(&format!("C{i} n{i} 0 100n\n"));
    }
    s.push_str(".end\n");
    s
}

fn diode_ladder(n: usize) -> String {
    let mut s = String::from("* diode ladder\n.model Dmod D(Is=1e-12 N=1)\nV1 n0 0 5\n");
    for i in 1..=n {
        s.push_str(&format!("R{i} n{} n{i} 1k\n", i - 1));
        s.push_str(&format!("D{i} n{i} 0 Dmod\n"));
    }
    s.push_str(".end\n");
    s
}

fn main() {
    sane_core::log::init_from_env();
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![10, 50, 200]
    } else {
        sizes
    };

    for n in sizes {
        // --- diode ladder DC -------------------------------------------------
        let t0 = Instant::now();
        let model = Model::from_netlist(&diode_ladder(n)).expect("model");
        let build_s = t0.elapsed().as_secs_f64();
        let p = model.pvec(&[]);
        let t0 = Instant::now();
        let (x, conv, iters) = model.cdc().solve_dc(&p, &[], 1e-9, 200);
        let dc_first = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let (x2, _, _) = model.cdc().solve_dc(&p, &[], 1e-9, 200);
        let dc_s = t0.elapsed().as_secs_f64();
        let out_idx = model.resolve(&format!("n{n}")).expect("out");
        let vout = x2.get(out_idx).copied().unwrap_or(f64::NAN);
        println!(
            "N={n} dc: build={build_s:.4}s first={dc_first:.4}s steady={dc_s:.4}s conv={conv} iters={iters} dim={} vout={vout:.6}",
            model.dim()
        );
        let _ = x;

        // --- RC ladder transient --------------------------------------------
        let t0 = Instant::now();
        let model = Model::from_netlist(&rc_ladder(n)).expect("model");
        let build_s = t0.elapsed().as_secs_f64();
        let p = model.pvec(&[]);
        let t_eval: Vec<f64> = (0..200).map(|k| 2e-3 * k as f64 / 199.0).collect();
        let out_idx = model.resolve(&format!("n{n}")).expect("out");

        // fixed 1 us steps (same grid as the circulax ConstantStepSize run)
        std::env::set_var("SANE_TRAN_FIXED", "1");
        let t0f = Instant::now();
        let rows = model
            .cdc()
            .solve_transient(
                sane_solve::TransientMethod::Esdirk32,
                &p,
                &[],
                &t_eval,
                1e-4,
                1e-7,
                Some(1e-6),
            )
            .expect("tran fixed");
        let tr_fixed = t0f.elapsed().as_secs_f64();
        let v_end_fixed = rows
            .last()
            .and_then(|r| r.get(out_idx))
            .copied()
            .unwrap_or(f64::NAN);

        // long fixed run: 50k steps of 1 us -> marginal cost per step
        let t_long: Vec<f64> = (0..200).map(|k| 5e-2 * k as f64 / 199.0).collect();
        let t0l = Instant::now();
        let _ = model
            .cdc()
            .solve_transient(
                sane_solve::TransientMethod::Esdirk32,
                &p,
                &[],
                &t_long,
                1e-4,
                1e-7,
                Some(1e-6),
            )
            .expect("tran long");
        let tr_long = t0l.elapsed().as_secs_f64();
        std::env::remove_var("SANE_TRAN_FIXED");

        // adaptive, rtol 1e-4 / atol 1e-7 (same tolerances as the PID run)
        let t0a = Instant::now();
        let rows = model
            .cdc()
            .solve_transient(
                sane_solve::TransientMethod::Esdirk32,
                &p,
                &[],
                &t_eval,
                1e-4,
                1e-7,
                None,
            )
            .expect("tran adaptive");
        let tr_adapt = t0a.elapsed().as_secs_f64();
        let v_end = rows
            .last()
            .and_then(|r| r.get(out_idx))
            .copied()
            .unwrap_or(f64::NAN);

        println!(
            "N={n} tran: build={build_s:.4}s fixed(2000)={tr_fixed:.4}s long(50k)={tr_long:.4}s ({:.2}us/step) adaptive={tr_adapt:.4}s dim={} vend_fixed={v_end_fixed:.6} vend={v_end:.6}",
            (tr_long - tr_fixed) / 48_000.0 * 1e6,
            model.dim()
        );
    }
}
