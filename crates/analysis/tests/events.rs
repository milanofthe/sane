//! Transient events: a hard switch flips at a located step boundary. The
//! switch thresholds are switching surfaces the integrator lands on, so the
//! crossing time is resolved far below the step size and the waveform is a
//! clean step; the events are reported by surface name.

use sane_analysis::Model;
use sane_solve::TransientMethod;

fn tran(model: &Model, tstop: f64, npts: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let t: Vec<f64> = (0..npts)
        .map(|k| tstop * k as f64 / (npts - 1) as f64)
        .collect();
    let rows = model
        .transient(TransientMethod::Esdirk32, &[], &t, 1e-6, 1e-9)
        .expect("transient")
        .rows()
        .to_vec();
    (t, rows)
}

/// A hard voltage-controlled switch (Vh = 0) driven by a ramp closes at the
/// instant the ramp crosses Vt; the event is located to nanoseconds on a
/// millisecond span and the output is on the correct side of the threshold
/// at every reported point.
#[test]
fn hard_vswitch_event_is_located() {
    let deck = "\
* hard switch closing on a ramp
Vc ctrl 0 PWL(0 0 1m 2)
Vdd vdd 0 5
S1 vdd out ctrl 0 SW
R1 out 0 1k
.model SW SW(Vt=1 Vh=0 Ron=1 Roff=1meg)
.end
";
    let model = Model::from_netlist(deck).expect("model");
    let out = model.resolve("out").expect("out");
    let (t, rows) = tran(&model, 1e-3, 201);
    let events = model.transient_events();
    assert_eq!(events.len(), 1, "one crossing: {events:?}");
    let (name, te, dir) = &events[0];
    assert_eq!(name, "S1#0");
    assert_eq!(*dir, 1, "the surface V(ctrl) - Vt rises through zero");
    assert!(
        (te - 0.5e-3).abs() < 1e-9,
        "event at {te:.9e}, expected 0.5 ms"
    );
    let v_off = 5.0 * 1e3 / (1e6 + 1e3);
    let v_on = 5.0 * 1e3 / (1e3 + 1.0);
    for (tk, row) in t.iter().zip(&rows) {
        // the landing is on the near side of the surface: the point at the
        // event instant itself is the last one in the old mode
        if (*tk - 0.5e-3).abs() < 1e-9 {
            continue;
        }
        let want = if *tk < 0.5e-3 { v_off } else { v_on };
        assert!(
            (row[out] - want).abs() < 1e-3 * v_on,
            "t={tk:.3e}: out={} expected {want}",
            row[out]
        );
    }
}

/// The current-controlled switch declares its threshold the same way; with a
/// positive window both edges are surfaces and both fire on a ramp.
#[test]
fn cswitch_window_edges_are_events() {
    let deck = "\
* current switch with a soft window on a ramping sense current
V1 1 0 PWL(0 0 1m 4)
R1 1 2 1k
Vsense 2 0 0
V2 3 0 5
W1 3 out Vsense CSW
Rl out 0 1k
.model CSW CSW(It=2m Ih=1m Ron=1 Roff=1meg)
.end
";
    let model = Model::from_netlist(deck).expect("model");
    let _ = tran(&model, 1e-3, 51);
    let mut events = model.transient_events();
    events.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    assert_eq!(events.len(), 2, "both window edges: {events:?}");
    // I(Vsense) = V1/1k ramps 0 -> 4 mA; the edges It -/+ Ih = 1 mA, 3 mA
    // are crossed at 0.25 ms and 0.75 ms.
    assert!((events[0].1 - 0.25e-3).abs() < 1e-8, "{events:?}");
    assert!((events[1].1 - 0.75e-3).abs() < 1e-8, "{events:?}");
    assert!(events.iter().all(|e| e.0.starts_with("W1#")));
}

/// A hard switch toggled by a pulse train: one event per edge, each within
/// the edge's ramp, and no spurious events between edges.
#[test]
fn pulse_driven_hard_switch_counts_edges() {
    let deck = "\
* hard switch chopping a DC rail
Vg g 0 PULSE(0 5 1u 10n 10n 4u 10u)
Vdd vdd 0 12
S1 vdd sw g 0 SW
R1 sw 0 100
.model SW SW(Vt=2.5 Vh=0 Ron=0.1 Roff=1meg)
.end
";
    let model = Model::from_netlist(deck).expect("model");
    let _ = tran(&model, 50e-6, 101);
    let events = model.transient_events();
    // edges at 1u + k*10u (rising) and 1u + 4u + 10n + k*10u (falling), k < 5
    assert_eq!(events.len(), 10, "{events:?}");
    for (k, (_, te, dir)) in events.iter().enumerate() {
        let period = (k / 2) as f64 * 10e-6;
        let (edge, want_dir) = if k % 2 == 0 {
            (1e-6 + 5e-9 + period, 1)
        } else {
            (1e-6 + 4e-6 + 10e-9 + 5e-9 + period, -1)
        };
        assert_eq!(*dir, want_dir, "event {k}: {events:?}");
        assert!(
            (te - edge).abs() < 6e-9,
            "event {k} at {te:.4e}, edge midpoint {edge:.4e}"
        );
    }
}
