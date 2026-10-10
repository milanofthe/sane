//! Noise generators enter the rows where the circuit puts them: parallel
//! devices add uncorrelated noise, a subcircuit's noise is the flat
//! circuit's, and a noise voltage is a voltage.

use sane_analysis::Model;

fn psd(deck: &str, out: &str, freqs: &[f64]) -> Vec<f64> {
    let model = Model::from_netlist(deck).expect("model");
    let n = model.at(&[]).unwrap().noise(&[out], freqs).expect("noise");
    n.psd.row(0).to_vec()
}

fn close(a: &[f64], b: &[f64], tol: f64) {
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol * x.abs().max(y.abs()), "{x} vs {y}");
    }
}

const FREQS: [f64; 3] = [10.0, 1e3, 1e6];

/// `M=4` is four devices in parallel: four times the noise power of one, not
/// the noise of one scaled up four times in amplitude.
#[test]
fn a_multiplicity_is_parallel_devices() {
    let head = ".model nm NMOS(level=1 vto=0.5 kp=100u lambda=0.02)\n\
                V1 vdd 0 1.8\nVg g 0 0.9\nR1 vdd d 2k\n";
    let one = format!("{head}M1 d g 0 0 nm W=10u L=1u M=4\n");
    let four: String = (1..=4)
        .map(|k| format!("M{k} d g 0 0 nm W=10u L=1u\n"))
        .collect();
    close(
        &psd(&one, "d", &FREQS),
        &psd(&format!("{head}{four}"), "d", &FREQS),
        1e-12,
    );
}

/// A subcircuit's resistors are as noisy as the same resistors placed flat.
#[test]
fn a_subcircuit_is_as_noisy_as_its_flat_circuit() {
    let flat = "V1 a 0 1\nR1 a m 1k\nR2 m b 2k\nR3 b 0 3k\nC1 b 0 1n\n";
    let sub = ".subckt pi in out\nR1 in m 1k\nR2 m out 2k\n.ends\n\
               V1 a 0 1\nX1 a b pi\nR3 b 0 3k\nC1 b 0 1n\n";
    close(&psd(flat, "b", &FREQS), &psd(sub, "b", &FREQS), 1e-12);
}

/// A noise voltage in series with a resistor is the Norton current noise
/// across it: `S_i = S_v / R^2`.
#[test]
fn a_noise_voltage_is_a_voltage() {
    let module = |body: &str| {
        format!(
            ".veriloga\nmodule nr(a, b);\n  inout a, b; electrical a, b, m;\n  \
             parameter real R = 1k;\n  analog begin\n{body}  end\nendmodule\n.endveriloga\n\
             V1 x 0 0\nR0 x a 500\nN1 a 0 nr\nC1 a 0 1n\n"
        )
    };
    let series = module("    V(a, m) <+ white_noise(4e-18, \"e\");\n    I(m, b) <+ V(m, b) / R;\n");
    let norton = module(
        "    V(a, m) <+ 0;\n    I(m, b) <+ V(m, b) / R + white_noise(4e-18 / (R * R), \"i\");\n",
    );
    close(&psd(&series, "a", &FREQS), &psd(&norton, "a", &FREQS), 1e-9);
}
