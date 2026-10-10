//! A transform keeps what belongs to the circuit: its ports, the circuit
//! itself, and the structures its parameters decide.

use sane_analysis::{Model, ModelError};

fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-8 * (1.0 + want.abs()),
        "got {got}, want {want}"
    );
}

/// S-parameters read the circuit's ports, a folded model's too.
#[test]
fn a_kept_model_keeps_its_ports() {
    let model =
        Model::from_netlist("P1 in 0 z0=50\nR1 in out 30\nC1 out 0 1p\nP2 out 0 z0=50\n").unwrap();
    // the port sources drive the S-parameters: folded, they drive nothing
    let folded = model.keep(&["R1"]).unwrap();
    assert!(matches!(
        folded.at(&[]).unwrap().s_parameters(&[1e6]),
        Err(ModelError::Invalid(_))
    ));
    let kept = model.keep(&["R1", "P1", "P2"]).unwrap();
    assert_eq!(kept.params().len(), 3);
    let freqs = [1e6, 1e9];
    let want = model.at(&[]).unwrap().s_parameters(&freqs).unwrap();
    let got = kept.at(&[]).unwrap().s_parameters(&freqs).unwrap();
    for (a, b) in got.s.iter().zip(want.s.iter()) {
        assert!((a - b).norm() < 1e-12, "{a} vs {b}");
    }
    assert!(kept.circuit().is_some(), "the circuit it derives from");
}

const SW: &str = "\
.veriloga
module sw(a, b);
  inout a, b; electrical a, b, ai;
  parameter real R = 1.0;
  analog begin
    if (R > 0)
      I(a, ai) <+ V(a, ai) / R;
    else
      V(a, ai) <+ 0.0;
    I(ai, b) <+ V(ai, b);
  end
endmodule
.endveriloga
V1 a 0 1
N1 a 0 sw R=0
";

fn current(model: &Model, binding: &[(&str, f64)]) -> Result<f64, ModelError> {
    let op = model.at(binding).and_then(|pt| pt.operating_point())?;
    Ok(-op.vector()[model.resolve("V1").unwrap()])
}

/// A folded model crosses a device's structure as the circuit's does: the
/// other structure set up and folded alike, at the values it folded.
#[test]
fn a_folded_model_restructures() {
    let model = Model::from_netlist(SW).unwrap();
    model.set("V1", 2.0).unwrap();
    let folded = model.fold(&["V1"]).unwrap();
    assert!(!folded.params().iter().any(|p| p == "V1"));
    close(current(&folded, &[]).unwrap(), 2.0);
    close(current(&folded, &[("N1.R", 1.0)]).unwrap(), 1.0);
    close(current(&folded, &[("N1.R", 3.0)]).unwrap(), 0.5);
    close(current(&folded, &[]).unwrap(), 2.0);
    // twice derived: folded, then the structure's own parameter kept
    let kept = folded.keep(&["N1.R"]).unwrap();
    close(current(&kept, &[("N1.R", 1.0)]).unwrap(), 1.0);
}

/// A pruned model rests on one structure's operating point: it keeps that
/// structure and says so.
#[test]
fn a_pruned_model_keeps_its_structure() {
    let model = Model::from_netlist(SW).unwrap();
    let (pruned, _) = model.at(&[]).unwrap().prune(1e-9, &[]).unwrap();
    match current(&pruned, &[("N1.R", 1.0)]) {
        Err(ModelError::Invalid(m)) => assert!(m.contains("pruned"), "{m}"),
        other => panic!("crossed: {other:?}"),
    }
}

/// A source driven by a waveform drives the small-signal response through
/// its level, as its value would: the same response as a DC source's.
#[test]
fn a_waveform_source_drives_ac_through_its_level() {
    let rc = "R1 in out 1k\nC1 out 0 100n\n";
    let freqs = [100.0, 1e3, 1e4];
    let h = |deck: String| {
        let model = Model::from_netlist(&deck).unwrap();
        let ac = model.at(&[]).unwrap().ac("V1", &["out"], &freqs).unwrap();
        ac.of("out").unwrap().to_vec()
    };
    let want = h(format!("V1 in 0 0\n{rc}"));
    for shape in ["SIN(0 1 1k)", "PULSE(0 1 1u 1u 1u 1m 2m)", "PWL(0 0 1m 1)"] {
        for (a, b) in h(format!("V1 in 0 {shape}\n{rc}")).iter().zip(&want) {
            assert!(
                (a - b).norm() < 1e-12 && b.norm() > 0.1,
                "{shape}: {a} vs {b}"
            );
        }
    }
    // folded, a source drives nothing, and says so
    let model = Model::from_netlist(&format!("V1 in 0 SIN(0 1 1k)\n{rc}")).unwrap();
    let folded = model.keep(&["R1"]).unwrap();
    assert!(matches!(
        folded.at(&[]).unwrap().ac("V1", &["out"], &freqs),
        Err(ModelError::Invalid(_))
    ));
}

// --- what a transform carries: delays, events, noise -------------------------

use sane_analysis::TransientOptions;

fn tran(model: &Model, t: &[f64], out: &str) -> (Vec<f64>, Vec<sane_analysis::Event>) {
    let opts = TransientOptions {
        rtol: 1e-7,
        atol: 1e-10,
        ..Default::default()
    };
    let tr = model
        .at(&[])
        .unwrap()
        .transient(t, &opts)
        .expect("transient");
    (tr.signal(out).expect("signal"), tr.events)
}

fn noise(model: &Model, binding: &[(&str, f64)], out: &str, freqs: &[f64]) -> Vec<f64> {
    let n = model
        .at(binding)
        .unwrap()
        .noise(&[out], freqs)
        .expect("noise");
    n.psd.row(0).to_vec()
}

fn all_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().chain(b).fold(0.0f64, |m, v| m.max(v.abs()));
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol * scale, "{x} vs {y}");
    }
}

/// A divider in front of a transmission line: its middle node eliminable.
const LINE: &str = "V1 a 0 PULSE(0 1 0 50p 50p 1 2)\nR1 a m 30\nR2 m b 20\n\
                    T1 b 0 out 0 Z0=50 TD=1n\nR3 out 0 75\nC1 out 0 0.2p\n";

/// Eliminating a node and folding parameters keep the transport delay: the
/// same bounce, at the same times.
#[test]
fn a_transform_keeps_the_delays() {
    let model = Model::from_netlist(LINE).unwrap();
    let t: Vec<f64> = (0..=120).map(|k| k as f64 * 5e-11).collect();
    let (want, _) = tran(&model, &t, "out");
    let (elim, gone) = model.eliminate_nodes(&[]);
    assert!(
        gone.iter().any(|n| n == "v2"),
        "the divider's middle: {gone:?}"
    );
    let (got, _) = tran(&elim, &t, "out");
    all_close(&got, &want, 1e-6);
    let folded = model.fold(&["R1", "R2"]).unwrap();
    let (got, _) = tran(&folded, &t, "out");
    all_close(&got, &want, 1e-9);
}

/// A switch closing on a ramp: its event lands where it did, through an
/// elimination and a fold.
#[test]
fn a_transform_keeps_the_events() {
    let deck = "Vc ctrl 0 PWL(0 0 1m 2)\nVdd vdd 0 5\nRa vdd s 500\nRb s s2 500\n\
                S1 s2 out ctrl 0 SW\nR1 out 0 1k\n.model SW SW(Vt=1 Vh=0 Ron=1 Roff=1meg)\n";
    let model = Model::from_netlist(deck).unwrap();
    let t: Vec<f64> = (0..=100).map(|k| k as f64 * 1e-5).collect();
    let (want, ev) = tran(&model, &t, "out");
    assert_eq!(ev.len(), 1);
    for derived in [model.eliminate_nodes(&[]).0, model.fold(&["Ra"]).unwrap()] {
        let (got, got_ev) = tran(&derived, &t, "out");
        assert_eq!(got_ev.len(), 1, "the event is still there");
        assert!((got_ev[0].t - ev[0].t).abs() < 1e-12, "at the same time");
        all_close(&got, &want, 1e-6);
    }
}

/// The noise of the eliminated resistors is still there, where it goes.
#[test]
fn an_elimination_keeps_the_noise() {
    let deck = "V1 a 0 1\nR1 a m 1k\nR2 m n 2k\nR3 n b 500\nR4 b 0 3k\nC1 b 0 1n\n";
    let model = Model::from_netlist(deck).unwrap();
    let freqs = [10.0, 1e5, 1e7];
    let want = noise(&model, &[], "b", &freqs);
    let (elim, gone) = model.eliminate_nodes(&["v4".to_string()]);
    assert_eq!(gone.len(), 2, "m and n: {gone:?}");
    // exact but for the gmin shunt the eliminated nodes no longer carry
    all_close(&noise(&elim, &[], "b", &freqs), &want, 1e-7);
}

/// The small-signal model is the circuit's small signal: the same AC
/// (through a delay) and the same noise, bound to the operating point.
#[test]
fn a_linearization_keeps_delays_noise_and_drives() {
    let deck = "V1 in 0 0.6\nR1 in a 1k\nD1 a 0 dm\nR2 a b 2k\n\
                T1 b 0 out 0 Z0=50 TD=1n\nR3 out 0 50\n.model dm D(Is=1e-14)\n";
    let model = Model::from_netlist(deck).unwrap();
    let x = model
        .at(&[])
        .unwrap()
        .operating_point()
        .unwrap()
        .vector()
        .to_vec();
    let lin = model.linearize();
    let mut binding: Vec<(String, f64)> = (model.unknowns().iter())
        .zip(x.iter())
        .map(|(u, &v)| (format!("{u}#op"), v))
        .collect();
    binding.push(("V1#op".to_string(), 0.6));
    // the operating point the coefficients read
    let binding: Vec<(&str, f64)> = (binding.iter())
        .filter(|(k, _)| lin.params().contains(k))
        .map(|(k, v)| (k.as_str(), *v))
        .collect();
    assert!(
        binding.iter().any(|(k, _)| *k == "V1#op"),
        "the drive is a deviation"
    );
    let freqs = [1e6, 1e8, 3e8, 1e9];
    let ac = |m: &Model, b: &[(&str, f64)]| {
        let r = m.at(b).unwrap().ac("V1", &["out"], &freqs).expect("ac");
        r.of("out").unwrap().to_vec()
    };
    let (want, got) = (ac(&model, &[]), ac(&lin, &binding));
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).norm() <= 1e-9 * b.norm().max(1e-30), "{a} vs {b}");
    }
    all_close(
        &noise(&lin, &binding, "a", &freqs),
        &noise(&model, &[], "a", &freqs),
        1e-9,
    );
}
