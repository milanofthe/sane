//! Demonstrates the fold payoff: folding the parameters you do not tune collapses
//! the dF/dp (param-Jacobian) work, so sensitivity on a big BSIM4 model goes from
//! seconds to milliseconds. Keeps a few parameters symbolic and folds the rest.
//!
//! ```text
//! cargo run -q --release --example fold_bench -- paper/figures/data/_sane_HoiLee_AFFC_Pin_3.cir vout 3
//! ```
//! Args: <netlist> <output> [n_keep].

use std::time::Instant;

use sane_analysis::Model;

fn main() {
    sane_analysis::logging::init_from_env();
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("netlist");
    let output = a.next().expect("output node");
    let n_keep: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let src = std::fs::read_to_string(&path).expect("read");
    let model = Model::from_netlist(&src).expect("build");
    let all = model.params().to_vec();
    println!("params total = {}", all.len());

    // Baseline: sensitivity over ALL parameters.
    let op = model.operating_point(&[]).expect("op");
    let t0 = Instant::now();
    let s0 = op.sensitivity(&output);
    let dt0 = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "baseline sensitivity (all {} params): {:.1} ms ({})",
        all.len(),
        dt0,
        if s0.is_ok() { "ok" } else { "err" }
    );

    // Fold everything except the first `n_keep` parameters.
    let fold: Vec<&str> = all.iter().skip(n_keep).map(|s| s.as_str()).collect();
    let t1 = Instant::now();
    let folded = model.fold(&fold).expect("fold");
    let dt_fold = t1.elapsed().as_secs_f64() * 1e3;
    println!(
        "fold {} params -> {} remain, build {:.1} ms",
        fold.len(),
        folded.params().len(),
        dt_fold
    );

    let op2 = folded.operating_point(&[]).expect("op2");
    let t2 = Instant::now();
    let s1 = op2.sensitivity(&output);
    let dt1 = t2.elapsed().as_secs_f64() * 1e3;
    println!(
        "folded sensitivity ({} params): {:.1} ms ({})",
        folded.params().len(),
        dt1,
        if s1.is_ok() { "ok" } else { "err" }
    );
}
