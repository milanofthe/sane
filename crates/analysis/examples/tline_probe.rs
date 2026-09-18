//! Debug probe: internal tline unknowns over the first nanoseconds.
use sane_analysis::Model;
use sane_solve::TransientMethod;

fn main() {
    sane_core::log::init_from_env();
    let deck =
        "* step\nV1 s 0 PULSE(0 1 0 50p 50p 1 2)\nR1 s in 50\nT1 in 0 out 0 Z0=50 TD=5n\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    println!("unknowns: {:?}", model.unknowns());
    let t: Vec<f64> = (0..60).map(|k| 12e-9 * k as f64 / 59.0).collect();
    let traj = model
        .transient(TransientMethod::Esdirk32, &[], &t, 1e-6, 1e-9)
        .expect("tran");
    let idx = |n: &str| model.resolve(n).expect(n);
    // The line is the builtin Verilog-A Branin module: two probed port
    // currents plus two absdelay source/output pairs.
    let (vin, vout) = (idx("in"), idx("out"));
    let (w1, w2) = (idx("T1.dly0_src"), idx("T1.dly1_src"));
    let (d1, d2) = (idx("T1.dly0"), idx("T1.dly1"));
    let (i1, i2) = (idx("T1.flow_n1_p1"), idx("T1.flow_n2_p2"));
    println!(
        "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "t/ns", "in", "out", "w1", "w2", "d1", "d2", "i1", "i2"
    );
    for (tk, row) in t.iter().zip(traj.rows()) {
        println!(
            "{:8.3} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:9.5} {:9.5}",
            tk * 1e9,
            row[vin],
            row[vout],
            row[w1],
            row[w2],
            row[d1],
            row[d2],
            row[i1],
            row[i2]
        );
    }
}
