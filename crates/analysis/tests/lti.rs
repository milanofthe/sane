//! Noise, poles, zeros, state space and reduced models at a `Point`.

use sane_analysis::{log_grid, Complex64, Model};

const BOLTZMANN: f64 = 1.380649e-23;

#[test]
fn thermal_noise_follows_the_temperature() {
    // A lone resistor: no device reads the temperature, yet its thermal noise
    // 4kTR does, so `$temp` is a parameter of the model.
    let m = Model::from_netlist("R1 out 0 1k\n").unwrap();
    for t in [300.15, 600.3] {
        let ns = m
            .at(&[("$temp", t)])
            .unwrap()
            .noise(&["out"], &[1e3])
            .unwrap();
        let want = 4.0 * BOLTZMANN * t * 1e3;
        assert!(
            (ns.psd[[0, 0]] - want).abs() < 1e-3 * want,
            "T={t}: {} vs {want}",
            ns.psd[[0, 0]]
        );
    }
}

#[test]
fn noise_outputs_and_rms() {
    // An RC divider: every output in one run, as each alone.
    let m = Model::from_netlist("V1 in 0 1\nR1 in mid 1k\nR2 mid 0 3k\nC1 mid 0 1n\n").unwrap();
    let pt = m.at(&[]).unwrap();
    let freqs = log_grid(1.0, 1e7, 30);
    let both = pt.noise(&["mid", "in"], &freqs).unwrap();
    let one = pt.noise(&["mid"], &freqs).unwrap();
    assert_eq!(both.of("mid").unwrap(), one.of("mid").unwrap());
    // `in` is held by the ideal source: no noise.
    assert!(both.of("in").unwrap().iter().all(|&v| v.abs() < 1e-30));
    // Low-frequency density: 4kT (R1 || R2).
    let want = (4.0 * BOLTZMANN * 300.15 * 750.0_f64).sqrt();
    let d0 = one.density("mid").unwrap()[0];
    assert!((d0 - want).abs() < 1e-3 * want, "{d0} vs {want}");
    // kT/C: the band covers the whole roll-off.
    let rms = one.rms("mid").unwrap();
    let ktc = (BOLTZMANN * 300.15 / 1e-9_f64).sqrt();
    assert!((rms - ktc).abs() < 0.05 * ktc, "{rms} vs {ktc}");
}

#[test]
fn noise_sensitivity_matches_finite_differences() {
    let m = Model::from_netlist("V1 in 0 1\nR1 in mid 1k\nR2 mid 0 3k\nC1 mid 0 1n\n").unwrap();
    let freqs = [1e3, 1e5];
    let ns = m.at(&[]).unwrap().noise(&["mid"], &freqs).unwrap();
    let g = ns.sensitivity(&["R2", "C1"]).unwrap();
    assert_eq!(g.params, ["R2", "C1"]);
    for (j, (name, v)) in [("R2", 3e3), ("C1", 1e-9)].into_iter().enumerate() {
        let h = v * 1e-6;
        let psd = |x: f64| {
            m.at(&[(name, x)])
                .unwrap()
                .noise(&["mid"], &freqs)
                .unwrap()
                .psd
                .row(0)
                .to_vec()
        };
        let (up, dn) = (psd(v + h), psd(v - h));
        for k in 0..freqs.len() {
            let fd = (up[k] - dn[k]) / (2.0 * h);
            let d = g.grad[[0, k, j]];
            assert!(
                (d - fd).abs() < 1e-5 * fd.abs().max(1e-40),
                "{name} f={}: {d} vs {fd}",
                freqs[k]
            );
        }
    }
}

#[test]
fn lead_network_zero_and_its_sensitivity() {
    // R1 || C1 in series, R2 to ground: H = R2 (1 + s R1 C1) / (R1 + R2 + s R1 R2 C1),
    // a zero at -1/(R1 C1) and a pole at -(R1 + R2)/(R1 R2 C1).
    let m = Model::from_netlist("V1 in 0 1\nR1 in out 1k\nC1 in out 1u\nR2 out 0 1k\n").unwrap();
    let pt = m.at(&[]).unwrap();
    let zeros = pt.zeros("V1", &["out"]).unwrap();
    let z = zeros.of("out").unwrap();
    assert_eq!(z.len(), 1, "{z:?}");
    assert!(
        (z[0] - Complex64::new(-1000.0, 0.0)).norm() < 1e-6 * 1000.0,
        "{z:?}"
    );
    let poles = pt.poles().unwrap();
    let near = |p: &Complex64| (p - Complex64::new(-2000.0, 0.0)).norm() < 1e-6 * 2000.0;
    assert!(poles.poles.iter().any(near), "{:?}", poles.poles);
    // d z / d R1 = 1 / (R1^2 C1), d z / d R2 = 0.
    let g = &zeros.sensitivity(&["R1", "R2"]).unwrap()[0];
    assert!((g.grad[[0, 0]].re - 1.0).abs() < 1e-6, "{:?}", g.grad);
    assert!(g.grad[[0, 1]].norm() < 1e-9, "{:?}", g.grad);
    // d p / d R2 = 1 / (R2^2 C1) for p = -1/(R1 C1) - 1/(R2 C1).
    let gp = poles.sensitivity(&["R2"]).unwrap();
    let k = gp.roots.iter().position(near).unwrap();
    assert!(
        (gp.grad[[k, 0]].re - 1.0).abs() < 1e-6,
        "{:?}",
        gp.grad.row(k)
    );
}

#[test]
fn state_space_reproduces_the_ac_response() {
    // Two sources, two outputs: C (jwE - A)^-1 B is the AC response of
    // every pair.
    let net = "V1 a 0 1\nI1 0 out 0\nR1 a mid 1k\nC1 mid 0 1u\nR2 mid out 2k\nC2 out 0 100n\n";
    let m = Model::from_netlist(net).unwrap();
    let pt = m.at(&[]).unwrap();
    let ss = pt.state_space(&["V1", "I1"], &["mid", "out"]).unwrap();
    let freqs = [10.0, 1e3, 1e5];
    let n = ss.a.nrows();
    for (col, input) in ["V1", "I1"].into_iter().enumerate() {
        let ac = pt.ac(input, &["mid", "out"], &freqs).unwrap();
        for (k, &f) in freqs.iter().enumerate() {
            let w = 2.0 * std::f64::consts::PI * f;
            let mut a: Vec<Vec<Complex64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| Complex64::new(-ss.a[[i, j]], w * ss.e[[i, j]]))
                        .collect()
                })
                .collect();
            let mut b: Vec<Complex64> = (0..n)
                .map(|i| Complex64::new(ss.b[[i, col]], 0.0))
                .collect();
            let x = gauss(&mut a, &mut b);
            for (row, out) in ["mid", "out"].into_iter().enumerate() {
                let y: Complex64 = (0..n).map(|j| x[j] * ss.c[[row, j]]).sum();
                let h = ac.of(out).unwrap()[k];
                assert!(
                    (y - h).norm() < 1e-9 * h.norm().max(1e-12),
                    "{input}->{out} f={f}: {y} vs {h}"
                );
            }
        }
    }
}

#[test]
fn reduction_is_exact_at_full_order() {
    let m = Model::from_netlist("V1 in 0 1\nR1 in a 10\nL1 a out 1m\nC1 out 0 1u\n").unwrap();
    let r = m
        .at(&[])
        .unwrap()
        .reduce("V1", "out", 2, &log_grid(10.0, 1e6, 40))
        .unwrap();
    assert_eq!(r.poles.len(), 2);
    // Exact up to the gmin that the full transfer carries.
    assert!(r.max_error_db() < 1e-4, "{}", r.max_error_db());
    assert!(m.at(&[]).unwrap().reduce("nope", "out", 2, &[1.0]).is_err());
}

/// Dense complex Gaussian elimination with partial pivoting.
fn gauss(a: &mut [Vec<Complex64>], b: &mut [Complex64]) -> Vec<Complex64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].norm().total_cmp(&a[j][c].norm()))
            .unwrap();
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                let v = a[c][k];
                a[r][k] -= f * v;
            }
            let v = b[c];
            b[r] -= f * v;
        }
    }
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for r in (0..n).rev() {
        let s: Complex64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    x
}
