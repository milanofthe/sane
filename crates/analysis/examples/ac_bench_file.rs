//! AC-sweep parallelism benchmark on a real netlist file.
//!
//! Loads a `.cir`, solves the DC operating point, then times the AC sweep (which
//! linearises at the OP and solves `G + jwC` per frequency on the worker pool).
//! Thread count comes from `SANE_THREADS` (default 4); run the same binary under
//! different settings to A/B:
//!
//! ```text
//! SANE_THREADS=1 cargo run -q --release --example ac_bench_file -- path.cir Vd 22 8000
//! SANE_THREADS=4 cargo run -q --release --example ac_bench_file -- path.cir Vd 22 8000
//! ```
//!
//! Args: `<netlist> <input-source> <output-node> [points] [fstart] [fstop]`.

use std::time::Instant;

use sane_analysis::Model;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: <netlist> <input> <output> [points] [fstart] [fstop]");
    let input = args.next().expect("input source name");
    let output = args.next().expect("output node name");
    let points: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(8000);
    let fstart: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(1.0);
    let fstop: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(1e7);

    sane_analysis::logging::init_from_env(); // SANE_LOG=debug surfaces stage timings
    let src = std::fs::read_to_string(&path).expect("read netlist");
    let model = match Model::from_netlist(&src) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{path}: build failed: {e:?}");
            std::process::exit(1);
        }
    };

    // Warm up: DC solve + first pool build, so timing is the sweep itself.
    match model.ac(&[], &input, &output, fstart, fstop, 16) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("{path}: AC failed (input={input}, output={output}): {e:?}");
            std::process::exit(1);
        }
    }

    let threads = sane_solve::parallel::threads();
    let t0 = Instant::now();
    let ac = model
        .ac(&[], &input, &output, fstart, fstop, points)
        .expect("ac sweep");
    let dt = t0.elapsed();

    let sum: f64 = ac.mag_db.iter().sum();
    let name = std::path::Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&path);
    println!(
        "{name:<22} points={points} threads={threads} \
         elapsed={:>8.3} ms  mag_db[0]={:>8.2}  mag_db[last]={:>8.2}  checksum={sum:.4}",
        dt.as_secs_f64() * 1e3,
        ac.mag_db.first().copied().unwrap_or(0.0),
        ac.mag_db.last().copied().unwrap_or(0.0),
    );
}
