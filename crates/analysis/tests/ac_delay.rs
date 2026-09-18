//! Transport delays in AC: `hist_k(t) = x_src(t - tau)` is exactly
//! `e^{-j w tau} X_src` in the frequency domain, so a matched line is a pure
//! phase shift and a mismatched one shows the analytic standing-wave ripple.

use sane_analysis::Model;

fn ac(model: &Model, input: &str, out: &str, freqs: &[f64]) -> Vec<(f64, f64)> {
    let op = model.operating_point(&[]).expect("dc");
    let x = op.vector().to_vec();
    let p = model.pvec(&[]);
    let idx = model.resolve(out).expect("out");
    model
        .ac_response(input, idx, x, p, freqs.to_vec())
        .expect("ac")
}

/// Matched source and load: |H| is flat at 1/2 (the source divider) and the
/// phase is exactly -w*TD, the delay's signature.
#[test]
fn matched_line_is_a_pure_phase_shift() {
    let td = 1e-9;
    let deck = "* matched\nV1 s 0 AC 1\nR1 s in 50\nT1 in 0 out 0 Z0=50 TD=1n\nRL out 0 50\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    let freqs: Vec<f64> = (1..=6).map(|k| 5e7 * k as f64).collect();
    for (f, (re, im)) in freqs.iter().zip(ac(&model, "V1", "out", &freqs)) {
        let mag = (re * re + im * im).sqrt();
        assert!((mag - 0.5).abs() < 1e-6, "f={f:.3e}: |H|={mag}");
        let w = 2.0 * std::f64::consts::PI * f;
        let want = -(w * td) % (2.0 * std::f64::consts::PI);
        let got = im.atan2(re);
        let d = ((got - want).abs() % (2.0 * std::f64::consts::PI)).min(
            (2.0 * std::f64::consts::PI) - ((got - want).abs() % (2.0 * std::f64::consts::PI)),
        );
        assert!(d < 1e-6, "f={f:.3e}: phase {got} vs {want}");
    }
}

/// Open-ended line, matched source: the input sees Z_in = -j Z0 cot(w TD), so
/// |H| = |v_in / v_s| runs the analytic |cos| resonance pattern -- a rational
/// (lumped) model cannot reproduce these repeating nulls.
#[test]
fn open_line_shows_the_analytic_resonances() {
    let td = 1e-9;
    let deck = "* open\nV1 s 0 AC 1\nR1 s in 50\nT1 in 0 out 0 Z0=50 TD=1n\n.end\n";
    let model = Model::from_netlist(deck).expect("model");
    // quarter-wave (w*TD = pi/2) shorts the input; half-wave passes it fully
    let f_quarter = 1.0 / (4.0 * td);
    let f_half = 1.0 / (2.0 * td);
    let r = ac(&model, "V1", "in", &[f_quarter, f_half]);
    let mag = |c: (f64, f64)| (c.0 * c.0 + c.1 * c.1).sqrt();
    assert!(mag(r[0]) < 1e-6, "quarter-wave null: |H|={}", mag(r[0]));
    assert!(
        (mag(r[1]) - 1.0).abs() < 1e-6,
        "half-wave pass: |H|={}",
        mag(r[1])
    );
}

/// A Verilog-A `absdelay` behaves identically (same machinery, arbitrary
/// behavioural quantity): unity magnitude, linear phase.
#[test]
fn absdelay_is_exact_in_ac() {
    let deck = "\
* va delay
.veriloga
module dline(in, out);
  inout in, out; electrical in, out;
  parameter real td = 1e-9;
  analog V(out) <+ absdelay(V(in), td);
endmodule
.endveriloga
V1 in 0 AC 1
N1 in out dline td=2n
R1 out 0 1k
.end
";
    let model = Model::from_netlist(deck).expect("model");
    let f = 1.25e8;
    let r = ac(&model, "V1", "out", &[f]);
    let (re, im) = r[0];
    let mag = (re * re + im * im).sqrt();
    assert!((mag - 1.0).abs() < 1e-9, "|H| = {mag}");
    let want = -(2.0 * std::f64::consts::PI * f * 2e-9);
    let got = im.atan2(re);
    let two_pi = 2.0 * std::f64::consts::PI;
    let d = ((got - want) % two_pi + two_pi) % two_pi;
    assert!(d.min(two_pi - d) < 1e-9, "phase {got} vs {want}");
}
