//! FNV-1a hash over HB spectra bits for a few decks -- A/B bit-identity probe
//! for changes to the harmonic-balance solve path.

use sane_analysis::Model;

fn fnv(h: &mut u64, x: f64) {
    for b in x.to_bits().to_le_bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn main() {
    let mut decks: Vec<(String, f64, usize, Vec<String>)> = Vec::new();
    // Diode ladder (the hb_timing deck), two sizes/orders.
    for (k, kh) in [(10usize, 8usize), (40, 16)] {
        let mut d = String::from("V1 1 0 SIN(1.2 0.4 1000)\n");
        for i in 1..=k {
            d.push_str(&format!("R{i} {i} {} 1k\nD{i} {} 0 dm\n", i + 1, i + 1));
        }
        d.push_str(".model dm D(Is=1e-14 N=1 Vt=0.025852)\n.end\n");
        decks.push((d, 1000.0, kh, vec!["2".into(), format!("{}", k + 1)]));
    }
    // Single-diode rectifier with charge storage (exercises jxd blocks).
    decks.push((
        "V1 1 0 SIN(0 1 1k)\nR1 1 2 100\nD1 2 3 dm\nC1 3 0 1u\nR2 3 0 10k\n.model dm D(Is=1e-12 N=1.5 Vt=0.025852 Cj0=1e-9)\n.end\n".into(),
        1000.0,
        12,
        vec!["2".into(), "3".into()],
    ));
    // MOS common-source stage driven into nonlinearity.
    decks.push((
        "VDD 1 0 5\nVIN 2 0 SIN(2 0.5 10k)\nRD 1 3 1k\nM1 3 2 0 0 nm W=1 L=1\n.model nm NMOS(Vto=1 Kp=2e-3 lambda=0.01)\n.end\n".into(),
        10_000.0,
        10,
        vec!["3".into()],
    ));

    let mut h: u64 = 0xcbf29ce484222325;
    for (deck, f0, kh, refs) in &decks {
        let model = Model::from_netlist(deck).expect("build");
        let hb = model.harmonic_balance(&[], *f0, *kh, None).expect("hb");
        assert!(hb.converged, "HB must converge");
        for r in refs {
            for &v in hb.magnitude(r).expect("mag") {
                fnv(&mut h, v);
            }
            for &v in hb.phase(r).expect("phase") {
                fnv(&mut h, v);
            }
        }
    }
    println!("hb spectra fnv {h:016x}");
}
