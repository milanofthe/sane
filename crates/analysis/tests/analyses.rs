//! Analyses end to end on small decks: poles, transient and DC sweeps, AC,
//! reduction, noise, temperature sweeps, state space.

use sane_analysis::{Model, TransientOptions};
use sane_core::Graph;
use sane_netlist::parse;

/// A user initial condition: a node voltage `V(net)` or a branch current
/// `I(element)` (e.g. an inductor) to start the transient from.
enum IcTarget {
    V(String),
    I(String),
}

/// Parse `.ic V(net)=value I(L1)=value ...` directives.
fn parse_ic(netlist: &str) -> Vec<(IcTarget, f64)> {
    let mut out = Vec::new();
    for raw in netlist.lines() {
        let line = raw.trim();
        if !line.to_ascii_lowercase().starts_with(".ic") {
            continue;
        }
        for tok in line.split_whitespace().skip(1) {
            let Some(eq) = tok.find('=') else { continue };
            let lhs = tok[..eq].trim();
            let Some(val) = sane_netlist::parse_value(tok[eq + 1..].trim()) else {
                continue;
            };
            let low = lhs.to_ascii_lowercase();
            if low.starts_with("v(") && lhs.ends_with(')') {
                out.push((IcTarget::V(lhs[2..lhs.len() - 1].to_string()), val));
            } else if low.starts_with("i(") && lhs.ends_with(')') {
                out.push((IcTarget::I(lhs[2..lhs.len() - 1].to_string()), val));
            }
        }
    }
    out
}

fn resolve_out_idx(
    parsed: &sane_circuit::Circuit,
    dae: &sane_dae::Dae,
    output: &str,
) -> Option<usize> {
    dae.node(parsed.find_node(output)?)
}

fn nearest_real(poles: &[[f64; 2]], target: f64) -> f64 {
    poles
        .iter()
        .map(|p| (p[0] - target).abs() + p[1].abs())
        .fold(f64::INFINITY, f64::min)
}

/// Relative magnitude AC sensitivity `d ln|H| / d ln param` at `freq`,
/// reconstructed from the exact `Model` AC response and per-parameter
/// `dH/dp` (the same quantity the deleted `ac_sensitivity` façade ranked).
fn ac_rel_sens(net: &str, input: &str, output: &str, freq: f64, param: &str) -> f64 {
    let m = Model::from_netlist(net).expect("model");
    let ac = (m.at(&[]).unwrap())
        .ac(input, &[output], &[freq])
        .expect("ac");
    let s = ac.sensitivity(&[param]).expect("ac sensitivity");
    let (h0, dhc) = (ac.h[[0, 0]], s.grad[[0, 0, 0]]);
    let dmag_dp = (h0.conj() * dhc).re / (h0.norm() * h0.norm());
    m.get(param).expect("param value") * dmag_dp
}

/// Relative transient sensitivity `d ln y(tstop) / d ln param` from the
/// state the deck's `.ic` sets (every other unknown at zero): its pinned
/// values do not move with the parameter.
fn transient_sens_rel(net: &str, output: &str, tstop: f64, tstep: f64, param: &str) -> f64 {
    let m = Model::from_netlist(net).expect("model");
    let mut x0 = vec![0.0; m.dim()];
    for (tgt, val) in parse_ic(net) {
        let idx = match tgt {
            IcTarget::V(node) => m.resolve(&node),
            IcTarget::I(elem) => m.resolve(&elem),
        };
        if let Some(i) = idx {
            x0[i] = val;
        }
    }
    let npts = ((tstop / tstep).round() as usize).clamp(1, 100_000);
    let t_eval: Vec<f64> = (0..=npts).map(|k| k as f64 * tstop / npts as f64).collect();
    let opts = TransientOptions {
        dt_max: Some(tstop / npts as f64),
        x0: Some(x0),
        ..TransientOptions::default()
    };
    let s = (m.at(&[]).unwrap())
        .transient(&t_eval, &opts)
        .and_then(|tr| tr.sensitivity(&[output], &[param]))
        .expect("transient sensitivity");
    let k = t_eval.len() - 1;
    s.grad[[0, k, 0]] * m.get(param).expect("param value") / s.values[[0, k]]
}

#[test]
fn rc_lowpass_pole() {
    // R=1k, C=1u -> single real pole at s = -1/(RC) = -1000 rad/s.
    let net = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n";
    let m = Model::from_netlist(net).expect("model");
    let poles = m.at(&[]).unwrap().poles().expect("poles").poles;
    let roots: Vec<[f64; 2]> = poles.iter().map(|s| [s.re, s.im]).collect();
    let d = nearest_real(&roots, -1000.0);
    assert!(
        d < 1.0,
        "expected a pole near -1000, got {poles:?} (err {d})"
    );
}

#[test]
fn transient_sensitivity_rc_discharge() {
    // RC discharge from V(out)=1 (UIC) through R to ground: V(t)=exp(-t/RC).
    // The relative sensitivity d ln V / d ln R = t/(RC) (larger R holds charge
    // longer), so at t = RC it is +1.0.
    let net = ".ic V(out)=1\nR1 out 0 1k\nC1 out 0 1u\n";
    let sr = transient_sens_rel(net, "out", 1e-3, 2e-5, "R1");
    assert!(
        (sr - 1.0).abs() < 0.05,
        "d ln V/d ln R = {sr}, expected ~ +1.0 at t=RC"
    );
}

// Run `cargo test -p sane-analysis bench_ua741 -- --ignored --nocapture`.
#[test]
#[ignore = "long-running benchmark (pole-zero on dim-109 is ~20 min)"]
fn bench_ua741() {
    let net = include_str!("../../../crates/netlist/tests/fixtures/ua741.cir").to_string();
    let m = Model::from_netlist(&net).expect("model");
    let t = |label: &str, f: &dyn Fn() -> String| {
        let t0 = sane_core::time::Instant::now();
        let info = f();
        eprintln!(
            "  {:<24} {:>9.1} ms   {}",
            label,
            t0.elapsed().as_secs_f64() * 1000.0,
            info
        );
    };
    eprintln!("\n=== uA741 analysis breakdown (24 BJT, transistor level, dim 109) ===");
    t("Operating point", &|| {
        let op = m.at(&[]).and_then(|pt| pt.operating_point());
        format!("ok={} dim={}", op.is_ok(), m.dim())
    });
    t("Transient (1us/10ns)", &|| {
        let npts = ((1e-6 / 1e-8_f64).round() as usize).clamp(1, 100_000);
        let te: Vec<f64> = (0..=npts).map(|k| k as f64 * 1e-6 / npts as f64).collect();
        let opts = sane_analysis::TransientOptions::default();
        let tr = m.at(&[]).and_then(|pt| pt.transient(&te, &opts));
        format!("ok={} traces={}", tr.is_ok(), m.unknowns().len())
    });
    t("AC (50 pts, 1Hz-1MHz)", &|| {
        let ac = (m.at(&[]))
            .and_then(|pt| pt.ac("Vd", &["22"], &sane_analysis::log_grid(1.0, 1e6, 50)))
            .map(|ac| ac.mag_db("22").unwrap());
        format!(
            "ok={} |H0|={:.1}dB",
            ac.is_ok(),
            ac.as_ref()
                .ok()
                .and_then(|a| a.first().copied())
                .unwrap_or(0.0)
        )
    });
    t("Pole-Zero (std-reduce, dim 109)", &|| {
        let pt = m.at(&[]).unwrap();
        match pt.poles().and_then(|p| Ok((p, pt.zeros("Vd", &["22"])?))) {
            Ok((p, z)) => format!("ok=true poles={} zeros={}", p.poles.len(), z.zeros[0].len()),
            Err(e) => format!("ok=false {e}"),
        }
    });
    t("DC sensitivity (adjoint)", &|| {
        let s = (m.at(&[]).and_then(|pt| pt.operating_point()))
            .and_then(|op| op.sensitivity(&["22"], &[]));
        format!(
            "ok={} params={}",
            s.is_ok(),
            s.as_ref().map(|s| s.params.len()).unwrap_or(0)
        )
    });
    t("Noise (50 pts)", &|| {
        let ns =
            (m.at(&[])).and_then(|pt| pt.noise(&["22"], &sane_analysis::log_grid(1.0, 1e6, 50)));
        format!("ok={}", ns.is_ok())
    });
    eprintln!("====================================================================\n");
}

#[test]
fn dc_sweep_cmos_inverter_vtc() {
    // The default example: sweeping the input must produce the inverter
    // transfer curve -- Vout high (~5V) at Vin=0, low (~0V) at Vin=5.
    let net = "VDD vdd 0 DC 5\nVIN vin 0 DC 2.5\nM1 vout vin 0 0 NMOS1\nM2 vout vin vdd vdd PMOS1\n.model NMOS1 NMOS(Kp=120u W=2 L=1 Vto=0.7)\n.model PMOS1 PMOS(Kp=40u W=4 L=1 Vto=-0.7)\n";
    let m = Model::from_netlist(net).expect("model");
    let vin: Vec<f64> = (0..=20).map(|k| 0.25 * k as f64).collect();
    let ds = m.at(&[]).unwrap().dc_sweep("VIN", &vin).expect("dc_sweep");
    let vout = ds.signal("vout").expect("vout trace");
    assert!(vout[0] > 4.0, "Vout at Vin=0 = {}", vout[0]);
    assert!(
        *vout.last().unwrap() < 1.0,
        "Vout at Vin=5 = {}",
        vout.last().unwrap()
    );
}

#[test]
fn ac_sallen_key_rolls_off() {
    // The default example, as the studio regenerates it. AC must succeed and
    // the output must roll off (2nd-order low-pass).
    let net = "VS in 0 AC 1\nR1 in x 11.2k\nR2 x y 11.2k\nC1 x vo 2000pF\nC2 y 0 1000pF\nE_OP vo 0 y vo 1Meg\n";
    let m = Model::from_netlist(net).expect("model");
    let ac = (m.at(&[]).unwrap())
        .ac("VS", &["vo"], &sane_analysis::log_grid(10.0, 1e6, 50))
        .expect("AC");
    let mag_db = ac.mag_db("vo").unwrap();
    assert!(mag_db.len() == 50);
    assert!(
        mag_db[49] < mag_db[0] - 20.0,
        "no rolloff: {} -> {}",
        mag_db[0],
        mag_db[49]
    );
}

#[test]
fn ac_sensitivity_rc_pole() {
    // RC lowpass at its pole f = 1/(2 pi RC) = 159.155 Hz: d ln|H|/d ln R and
    // d ln|H|/d ln C are both -0.5 there (|H| = 1/sqrt(1+(wRC)^2), wRC=1).
    let net = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n";
    let sr = ac_rel_sens(net, "V1", "out", 159.1549, "R1");
    let sc = ac_rel_sens(net, "V1", "out", 159.1549, "C1");
    assert!((sr + 0.5).abs() < 0.03, "dln|H|/dlnR = {sr}, expected -0.5");
    assert!((sc + 0.5).abs() < 0.03, "dln|H|/dlnC = {sc}, expected -0.5");
}

#[test]
fn model_reduce_exact_when_order_sufficient() {
    // RLC (2 poles): reducing to an order >= the true order must reproduce the
    // full magnitude response to within rounding.
    let net = "V1 in 0 1\nR1 in a 10\nL1 a out 1m\nC1 out 0 1u\n";
    let m = Model::from_netlist(net).expect("model");
    let r = (m.at(&[]).unwrap())
        .reduce("V1", "out", 4, &sane_analysis::log_grid(1.0, 1e6, 40))
        .expect("reduce");
    assert!(
        r.max_error_db() < 0.1,
        "expected exact reconstruction, max_err {} dB",
        r.max_error_db()
    );
}

#[test]
fn noise_single_resistor_johnson() {
    // A lone 1k resistor to ground: output noise = sqrt(4kTR) ~= 4.07 nV/rtHz,
    // flat in frequency.
    let net = "R1 out 0 1k\n";
    let m = Model::from_netlist(net).expect("model");
    let r = (m.at(&[]).unwrap())
        .noise(&["out"], &[1.0, 1e3, 1e6])
        .expect("noise");
    let expected = (4.0 * 1.380649e-23 * 300.15 * 1000.0_f64).sqrt();
    for nv in &r.density("out").unwrap() {
        assert!(
            (nv - expected).abs() / expected < 0.02,
            "noise {nv} vs expected {expected}"
        );
    }
}

#[test]
fn noise_diode_shot() {
    // A diode biased at I by an ideal current source, output at the diode node.
    // The node sees only the diode (rd = Vt/I), driven by shot noise 2q*I, so
    // the output PSD is 2q*I*rd^2 = 2q*Vt^2/I; the resistor-free setup isolates
    // the diode's own shot noise (no thermal term to subtract).
    let i_bias = 1e-3_f64;
    let net = format!("I1 0 a {i_bias}\nD1 a 0\n");
    let m = Model::from_netlist(&net).expect("model");
    let r = (m.at(&[]).unwrap())
        .noise(&["a"], &[1.0, 1e3, 1e6])
        .expect("noise");
    let q = 1.602176634e-19_f64;
    let vt = (1.380649e-23 / 1.602176634e-19) * 300.15;
    let expected = (2.0 * q * vt * vt / i_bias).sqrt(); // sqrt(2q Vt^2 / I)
    for nv in &r.density("a").unwrap() {
        assert!(
            (nv - expected).abs() / expected < 0.05,
            "diode shot noise {nv} vs expected {expected}"
        );
    }
}

#[test]
fn temp_sweep_diode_tempco() {
    // Forward-biased diode (~1 mA). Vf falls with temperature at about
    // -2 mV/degC (the classic silicon junction tempco).
    let net = "V1 in 0 5\nR1 in a 4.3k\nD1 a 0\n";
    let m = Model::from_netlist(net).expect("model");
    let kelvin = [300.15, 350.15]; // 27 and 77 degC
    let r = m
        .at(&[])
        .unwrap()
        .dc_sweep("$temp", &kelvin)
        .expect("temperature sweep");
    let v = r.signal("a").unwrap();
    assert_eq!(v.len(), 2);
    let slope_mv = (v[1] - v[0]) / (kelvin[1] - kelvin[0]) * 1000.0;
    assert!(
        slope_mv > -3.0 && slope_mv < -1.0,
        "expected ~ -2 mV/degC, got {slope_mv}"
    );
}

#[test]
fn temp_sweep_reaches_veriloga_model() {
    // A Verilog-A diode whose thermal voltage is `$vt = k*T/q`: the device
    // temperature must reach the model (via the global `$temp` symbol), so
    // the operating point moves with temperature rather than staying flat.
    let net = "\
.veriloga
module vadio(a,c); inout a,c; electrical a,c;
parameter real Is=1e-14;
analog I(a,c) <+ Is*(exp(V(a,c)/$vt) - 1.0);
endmodule
.endveriloga
V1 in 0 5
R1 in a 4.3k
N1 a 0 vadio
";
    let m = Model::from_netlist(net).expect("model");
    let kelvin = [233.15, 288.15, 343.15, 398.15]; // -40 .. 125 degC
    let r = m
        .at(&[])
        .unwrap()
        .dc_sweep("$temp", &kelvin)
        .expect("temperature sweep");
    let v = r.signal("a").unwrap();
    assert_eq!(v.len(), 4);
    let span =
        v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        span > 1e-3,
        "temperature did not reach the VA model (flat: {v:?})"
    );
}

#[test]
fn dt_max_is_robust() {
    use sane_solve::CompiledDc;
    // dt_max must never crash and must be honored. the transient caps each step at
    // the requested 50 ns natively (no backend fallback). Both the uncapped and
    // capped runs are independent, error-controlled integrations of the same
    // 1 MHz-driven RC; with the solution-space error estimate the uncapped run
    // already resolves the source, so the two agree closely (not bit-identical,
    // as they take different step sequences).
    let parsed = parse("Vin a 0 SIN(0 1 1e6)\nR1 a b 1k\nC1 b 0 1n\n").unwrap();
    let mut ctx = Graph::new();
    let dae = sane_dae::assemble(&mut ctx, &parsed).unwrap();
    let cdc = CompiledDc::new(ctx, dae);
    let pnames = cdc.param_names();
    let p: Vec<f64> = parsed.pvec(&pnames);
    let t: Vec<f64> = (0..=4).map(|k| k as f64 * 0.25e-6).collect();
    let without = cdc
        .solve_transient(&p, &[], &t, 1e-4, 1e-7, None)
        .unwrap()
        .rows;
    let withcap = cdc
        .solve_transient(&p, &[], &t, 1e-4, 1e-7, Some(50e-9))
        .unwrap()
        .rows;
    assert_eq!(withcap.len(), t.len());
    // Both are accurate integrations of the same circuit, so they agree to a
    // tight (but not bit-exact) tolerance.
    let b = resolve_out_idx(&parsed, cdc.dae(), "b").unwrap();
    for (u, w) in without.iter().zip(&withcap) {
        assert!(
            (u[b] - w[b]).abs() < 1e-4,
            "capped and uncapped disagree: {} vs {}",
            u[b],
            w[b]
        );
    }
}

#[test]
fn state_space_rc_descriptor() {
    // RC: E=C (one reactive state), A=-G, D=0, output = e_out^T.
    let net = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n";
    let m = Model::from_netlist(net).expect("model");
    let ss = m
        .at(&[])
        .unwrap()
        .state_space(&["V1"], &["out"])
        .expect("state_space");
    let n = m.unknowns().len();
    assert!(n >= 1 && ss.a.dim() == (n, n) && ss.e.dim() == (n, n));
    assert!(ss.b.dim() == (n, 1) && ss.c.dim() == (1, n));
    assert_eq!(ss.d, ndarray::array![[0.0]]);
    // E must have at least one nonzero entry (the capacitor's dynamics).
    assert!(ss.e.iter().any(|&v| v.abs() > 0.0), "E is all zero");
    // exactly one output pick.
    assert!((ss.c.row(0).sum() - 1.0).abs() < 1e-9);
}

#[test]
fn rlc_series_complex_poles() {
    // R=10, L=1m, C=1u: s = -R/2L +/- j*sqrt(1/LC - (R/2L)^2) = -5000 +/- j31225.
    let net = "V1 in 0 1\nR1 in a 10\nL1 a out 1m\nC1 out 0 1u\n";
    let m = Model::from_netlist(net).expect("model");
    let poles = m.at(&[]).unwrap().poles().expect("poles").poles;
    let hit = (poles.iter())
        .any(|p| (p.re + 5000.0).abs() < 200.0 && (p.im.abs() - 31225.0).abs() < 500.0);
    assert!(hit, "expected poles near -5000 +/- j31225, got {poles:?}");
}
