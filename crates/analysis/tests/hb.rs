//! Harmonic balance at a `Point`: spectra, distortion and their exact
//! derivatives.

use sane_analysis::ndarray::{s, Axis};
use sane_analysis::{HbOptions, Model};

const RECTIFIER: &str = "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 dm\nC1 mid 0 100n\n\
                         .model dm D(Is=1e-14 N=1 Vt=0.025852)\n.end";

#[test]
fn linear_circuit_has_no_distortion() {
    let m =
        Model::from_netlist("V1 in 0 SIN(0 1 1000)\nR1 in out 1k\nC1 out 0 100n\n.end").unwrap();
    let hb = m
        .at(&[])
        .unwrap()
        .harmonic_balance(&HbOptions::default())
        .unwrap();
    assert!((hb.f0 - 1000.0).abs() < 1e-9, "inferred f0 = {}", hb.f0);
    // |H(j w0)| of the RC low-pass.
    let w = 2.0 * std::f64::consts::PI * 1000.0;
    let want = 1.0 / (1.0 + (w * 1e3 * 100e-9_f64).powi(2)).sqrt();
    let x1 = hb.amplitude("out").unwrap()[1];
    assert!((hb.magnitude("out").unwrap()[1] - x1 / 2.0).abs() < 1e-15);
    assert!((x1 - want).abs() < 1e-9, "{x1} vs {want}");
    assert!(hb.thd("out").unwrap() < 1e-9);
}

#[test]
fn not_converging_is_an_error() {
    let m = Model::from_netlist(RECTIFIER).unwrap();
    let opts = HbOptions {
        max_iter: 1,
        continuation: Some(false),
        ..HbOptions::default()
    };
    let err = m
        .at(&[])
        .unwrap()
        .harmonic_balance(&opts)
        .err()
        .expect("one iteration is not enough");
    assert!(err.to_string().contains("did not converge"), "{err}");
}

#[test]
fn spectrum_sensitivity_matches_finite_differences() {
    let m = Model::from_netlist(RECTIFIER).unwrap();
    // Tight enough that the differences resolve the derivative.
    let opts = HbOptions {
        tol: 1e-13,
        ..HbOptions::default()
    };
    let hb = m.at(&[]).unwrap().harmonic_balance(&opts).unwrap();
    assert!(hb.thd("mid").unwrap() > 1e-3, "the diode distorts");
    let g = hb.sensitivity(&["mid", "in"], &["R1", "C1"]).unwrap();
    let one = hb.sensitivity(&["mid"], &["R1", "C1"]).unwrap();
    assert_eq!(
        g.grad.index_axis(Axis(0), 0),
        one.grad.index_axis(Axis(0), 0)
    );
    for (j, (name, v)) in [("R1", 1e3), ("C1", 100e-9)].into_iter().enumerate() {
        let h = v * 1e-6;
        let x = |v: f64| {
            let pt = m.at(&[(name, v)]).unwrap();
            pt.harmonic_balance(&opts)
                .unwrap()
                .spectrum("mid")
                .unwrap()
                .to_vec()
        };
        let (up, dn) = (x(v + h), x(v - h));
        for k in 0..=3 {
            let fd = (up[k] - dn[k]) / (2.0 * h);
            let d = g.grad[[0, k, j]];
            assert!(
                (d - fd).norm() < 1e-5 * fd.norm().max(1e-12 / v),
                "{name} k={k}: {d} vs {fd}"
            );
        }
        // The source node is held: no sensitivity beyond rounding.
        assert!((0..=hb.harmonics).all(|k| g.grad[[1, k, j]].norm() < 1e-12 / v));
    }
}

#[test]
fn spectrum_hessian_matches_the_gradient() {
    let m = Model::from_netlist(RECTIFIER).unwrap();
    let opts = HbOptions::default();
    let hb = m.at(&[]).unwrap().harmonic_balance(&opts).unwrap();
    let hs = hb.hessian("mid", 1, &["R1", "C1"]).unwrap();
    assert_eq!(hs.params.len(), 2);
    let (a, b) = (hs.h[[0, 1]], hs.h[[1, 0]]);
    assert!((a - b).norm() < 1e-6 * a.norm(), "asymmetric: {a} vs {b}");
    // Each column against central differences of the gradient.
    for (j, name) in hs.params.iter().enumerate() {
        let v = hs.param_values[j];
        let h = v * 1e-6;
        let grad = |x: f64| {
            let pt = m.at(&[(name.as_str(), x)]).unwrap();
            let hb = pt.harmonic_balance(&opts).unwrap();
            let refs: Vec<&str> = hs.params.iter().map(|s| s.as_str()).collect();
            hb.sensitivity(&["mid"], &refs)
                .unwrap()
                .grad
                .slice(s![0, 1, ..])
                .to_vec()
        };
        let (up, dn) = (grad(v + h), grad(v - h));
        for i in 0..hs.params.len() {
            let fd = (up[i] - dn[i]) / (2.0 * h);
            let d = hs.h[[i, j]];
            assert!(
                (d - fd).norm() < 1e-4 * fd.norm().max(1e-30),
                "({i},{j}): {d} vs {fd}"
            );
        }
    }
}
