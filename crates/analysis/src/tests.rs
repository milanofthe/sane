//! In-crate integration tests over the `Model` front end (Verilog-A engine
//! parity, deck directives, sensitivity oracles). These need crate internals,
//! so they live here rather than in `tests/`.

use super::*;

mod veriloga_engine_tests {
    //! WP5 gate: a Verilog-A model in a deck must flow through every engine
    //! path (DC bias, AC, pole-zero, transient) and match the equivalent native
    //! device, proving the lowered VA model is an ordinary device to the engine.
    use super::*;

    fn poles_of(net: &str, input: &str, output: &str) -> Vec<[f64; 2]> {
        Model::from_netlist(net)
            .expect("model")
            .poles_zeros(&[], input, output)
            .expect("pole_zero")
            .poles
    }
    fn ac_mag_db(net: &str, input: &str, output: &str, f0: f64, f1: f64, n: usize) -> Vec<f64> {
        Model::from_netlist(net)
            .expect("model")
            .ac(&[], input, output, f0, f1, n)
            .expect("ac")
            .mag_db
    }
    fn noise_psd(net: &str, output: &str, f0: f64, f1: f64, n: usize) -> Vec<f64> {
        Model::from_netlist(net)
            .expect("model")
            .noise(&[], output, f0, f1, n)
            .expect("noise")
            .psd
    }

    const VA: &str = r#"
.veriloga
module diode(a, c);
  inout a, c; electrical a, c;
  parameter real Is = 1e-14; parameter real N = 1.0; parameter real Vt = 0.025852;
  analog I(a, c) <+ Is * (limexp(V(a,c)/(N*Vt)) - 1.0);
endmodule
.endveriloga
V1 in 0 1 AC 1
R1 in n1 1k
N1 n1 0 diode
C1 n1 0 1u
.end
"#;

    const NATIVE: &str = r#"
V1 in 0 1 AC 1
R1 in n1 1k
D1 n1 0 mydiode
C1 n1 0 1u
.model mydiode D(Is=1e-14 N=1)
.end
"#;

    fn sorted_poles(p: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let mut v: Vec<[f64; 2]> = p.to_vec();
        v.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        v
    }

    #[test]
    fn va_diode_matches_native_pole_zero() {
        let va = poles_of(VA, "V1", "n1");
        let nat = poles_of(NATIVE, "V1", "n1");
        let (pv, pn) = (sorted_poles(&va), sorted_poles(&nat));
        assert_eq!(pv.len(), pn.len(), "pole count: {pv:?} vs {pn:?}");
        // The native and Verilog-A diodes are independent lowerings of the same
        // junction physics (both with thermal voltage k*T/q at $temp). They agree
        // to ~0.1%, not bit-for-bit: the native model carries a GMIN companion
        // shunt and the overflow-safe exp, which shift the small-signal
        // conductance slightly versus the VA `limexp`. The tolerance catches a
        // real divergence (a wrong VA lowering) while allowing this.
        for (a, b) in pv.iter().zip(&pn) {
            let scale = a[0].abs().max(1.0);
            assert!(
                (a[0] - b[0]).abs() < 5e-3 * scale && (a[1] - b[1]).abs() < 5e-3 * scale,
                "pole {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn va_diode_matches_native_ac() {
        let va = ac_mag_db(VA, "V1", "n1", 1.0, 1e6, 12);
        let nat = ac_mag_db(NATIVE, "V1", "n1", 1.0, 1e6, 12);
        assert_eq!(va.len(), nat.len());
        // ~0.01 dB agreement between the two independent diode lowerings (see
        // the pole-zero test for why this is not bit-exact).
        for (a, b) in va.iter().zip(&nat) {
            assert!((a - b).abs() < 0.05, "AC mag_db {a} vs {b}");
        }
    }

    #[test]
    fn va_diode_transient_runs() {
        let m = Model::from_netlist(VA).expect("model");
        let npts = ((5e-3 / 5e-5_f64).round() as usize).clamp(1, 100_000);
        let t: Vec<f64> = (0..=npts).map(|k| k as f64 * 5e-3 / npts as f64).collect();
        let tr = m
            .transient(sane_solve::TransientMethod::default(), &[], &t, 1e-4, 1e-7)
            .expect("transient");
        assert!(!tr.t.is_empty(), "no transient samples");
    }

    // WP7: a Verilog-A resistor that declares its own thermal white_noise must
    // produce the same output noise as the equivalent native resistor (whose
    // thermal noise the engine adds automatically).
    const VA_NOISE: &str = r#"
.veriloga
module nres(a, b);
  inout a, b; electrical a, b;
  parameter real R = 1k;
  analog begin
    I(a,b) <+ V(a,b)/R;
    I(a,b) <+ white_noise(4.0 * 1.380649e-23 * 300.15 / R, "thermal");
  end
endmodule
.endveriloga
V1 in 0 0
N1 in out nres
R2 out 0 1k
.end
"#;
    const NATIVE_NOISE: &str = r#"
V1 in 0 0
R1 in out 1k
R2 out 0 1k
.end
"#;

    #[test]
    fn va_laplace_lowpass_pole() {
        // H(s) = 1/(1 + tau*s): a single pole at -1/tau = -1000 rad/s.
        let deck = "\
.veriloga
module lpf(in, out); inout in,out; electrical in,out;
parameter real tau = 1e-3;
analog V(out) <+ laplace_nd(V(in), {1.0}, {1.0, tau});
endmodule
.endveriloga
V1 in 0 1 AC 1
N1 in out lpf
Rl out 0 1meg
.end
";
        let poles = poles_of(deck, "V1", "out");
        let hit = poles
            .iter()
            .any(|p| (p[0] + 1000.0).abs() < 5.0 && p[1].abs() < 1.0);
        assert!(hit, "expected a pole near -1000, got {poles:?}");
    }

    #[test]
    fn va_noise_table_matches_native_resistor() {
        // A VA resistor whose noise_table is a flat PSD = 4kT/R must match the
        // native resistor's thermal noise.
        let va = "\
.veriloga
module nres(a, b); inout a,b; electrical a,b; parameter real R = 1k;
analog begin
  I(a,b) <+ V(a,b)/R;
  I(a,b) <+ noise_table({1.0, 4.0*1.380649e-23*300.15/R, 1e9, 4.0*1.380649e-23*300.15/R});
end
endmodule
.endveriloga
V1 in 0 0
N1 in out nres
R2 out 0 1k
.end
";
        let nat = "V1 in 0 0\nR1 in out 1k\nR2 out 0 1k\n.end\n";
        let rv = noise_psd(va, "out", 1.0, 1e5, 8);
        let rn = noise_psd(nat, "out", 1.0, 1e5, 8);
        for (a, b) in rv.iter().zip(&rn) {
            assert!(
                (a - b).abs() < 1e-12 * b.max(1e-12) + 1e-15,
                "noise {a} vs {b}"
            );
        }
    }

    #[test]
    fn va_white_noise_matches_native_resistor() {
        let va = noise_psd(VA_NOISE, "out", 1.0, 1e5, 8);
        let nat = noise_psd(NATIVE_NOISE, "out", 1.0, 1e5, 8);
        assert_eq!(va.len(), nat.len());
        for (a, b) in va.iter().zip(&nat) {
            assert!(
                (a - b).abs() < 1e-12 * b.max(1e-12) + 1e-15,
                "noise {a} vs {b}"
            );
        }
    }
}

mod bsim4_validation {
    //! BSIM4 (Verilog-A) end-to-end validation against the ECL/CC reference deck
    //! in TEMP. Ignored by default (environment-specific path; run locally with
    //! `--ignored --nocapture`). The Verilog-A model is loaded via `.veriloga`,
    //! bound through a SPICE `.model` card, and driven through DC / AC / transient.
    use super::*;

    // Local, `Model`-backed reconstructions of the deleted netlist-string façades,
    // kept so these (mostly #[ignore], environment-specific) BSIM4 decks still drive
    // the engine through the same result shapes. Faithful: traces and signals are
    // labeled via the retained `label_unknown`, exactly as the old façades did.
    struct Trace {
        name: String,
        kind: String,
        values: Vec<f64>,
    }
    struct Signal {
        name: String,
        value: f64,
    }
    struct SweepResult {
        ok: bool,
        message: String,
        sweep: Vec<f64>,
        traces: Vec<Trace>,
    }
    struct SimResult {
        ok: bool,
        message: String,
        dim: usize,
        signals: Vec<Signal>,
    }
    struct TranResult {
        ok: bool,
        message: String,
        traces: Vec<Trace>,
    }
    struct AcResult {
        ok: bool,
        message: String,
        f: Vec<f64>,
        mag_db: Vec<f64>,
    }

    fn label_traces(m: &Model, rows: &[Vec<f64>]) -> Vec<Trace> {
        m.unknowns()
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let (name, kind) = label_unknown(u, m.node_names());
                Trace {
                    name,
                    kind: kind.into(),
                    values: rows
                        .iter()
                        .map(|r| r.get(i).copied().unwrap_or(0.0))
                        .collect(),
                }
            })
            .collect()
    }

    fn dc_sweep(deck: String, source: String, a: f64, b: f64, step: f64) -> SweepResult {
        let m = match Model::from_netlist(&deck) {
            Ok(m) => m,
            Err(e) => {
                return SweepResult {
                    ok: false,
                    message: e.to_string(),
                    sweep: vec![],
                    traces: vec![],
                }
            }
        };
        match m.dc_sweep(&source, a, b, step) {
            Ok(ds) => SweepResult {
                ok: true,
                message: "ok".into(),
                sweep: ds.sweep.clone(),
                traces: label_traces(&m, ds.rows()),
            },
            Err(e) => SweepResult {
                ok: false,
                message: e.to_string(),
                sweep: vec![],
                traces: vec![],
            },
        }
    }

    fn simulate(deck: String) -> SimResult {
        let m = match Model::from_netlist(&deck) {
            Ok(m) => m,
            Err(e) => {
                return SimResult {
                    ok: false,
                    message: e.to_string(),
                    dim: 0,
                    signals: vec![],
                }
            }
        };
        let dim = m.dim();
        match m.operating_point(&[]) {
            Ok(op) => {
                let signals = m
                    .unknowns()
                    .iter()
                    .enumerate()
                    .map(|(i, u)| {
                        let (name, _kind) = label_unknown(u, m.node_names());
                        Signal {
                            name,
                            value: op.vector().get(i).copied().unwrap_or(0.0),
                        }
                    })
                    .collect();
                SimResult {
                    ok: true,
                    message: "ok".into(),
                    dim,
                    signals,
                }
            }
            Err(e) => SimResult {
                ok: false,
                message: e.to_string(),
                dim,
                signals: vec![],
            },
        }
    }

    fn transient(deck: String, tstop: f64, tstep: f64) -> TranResult {
        if !(tstop > 0.0) || !(tstep > 0.0) || tstep > tstop {
            return TranResult {
                ok: false,
                message: "transient needs 0 < tstep <= tstop".into(),
                traces: vec![],
            };
        }
        let m = match Model::from_netlist(&deck) {
            Ok(m) => m,
            Err(e) => {
                return TranResult {
                    ok: false,
                    message: e.to_string(),
                    traces: vec![],
                }
            }
        };
        let n = ((tstop / tstep).round() as usize).clamp(1, 100_000);
        let t: Vec<f64> = (0..=n).map(|k| k as f64 * tstop / n as f64).collect();
        match m.transient(sane_solve::TransientMethod::default(), &[], &t, 1e-4, 1e-7) {
            Ok(tr) => TranResult {
                ok: true,
                message: "ok".into(),
                traces: label_traces(&m, tr.rows()),
            },
            Err(e) => TranResult {
                ok: false,
                message: e.to_string(),
                traces: vec![],
            },
        }
    }

    fn ac_analysis(
        deck: String,
        input: String,
        output: String,
        f0: f64,
        f1: f64,
        n: usize,
    ) -> AcResult {
        let m = match Model::from_netlist(&deck) {
            Ok(m) => m,
            Err(e) => {
                return AcResult {
                    ok: false,
                    message: e.to_string(),
                    f: vec![],
                    mag_db: vec![],
                }
            }
        };
        match m.ac(&[], &input, &output, f0, f1, n) {
            Ok(ac) => AcResult {
                ok: true,
                message: "ok".into(),
                f: ac.freqs,
                mag_db: ac.mag_db,
            },
            Err(e) => AcResult {
                ok: false,
                message: e.to_string(),
                f: vec![],
                mag_db: vec![],
            },
        }
    }

    // The full BSIM4 model + modelcards come from a machine-provisioned
    // VA-Models checkout (github.com/dwarning/VA-Models) — a test asset, not
    // vendored. Point SANE_VA_MODELS at its root; the tests self-skip when it
    // is absent. (They previously hardcoded a developer-machine path, which
    // made them FAIL instead of skip on any other machine / in CI.)
    fn va_models() -> Option<(String, String)> {
        let base = std::path::PathBuf::from(std::env::var_os("SANE_VA_MODELS")?);
        let va = base.join("code/bsim4/vacode/bsim4.va");
        let card = base.join("examples/bsim4/Modelcards/modelcard.nmos");
        (va.exists() && card.exists()).then(|| {
            (
                va.to_string_lossy().into_owned(),
                card.to_string_lossy().into_owned(),
            )
        })
    }

    // Modelcard path is overridable for bisection diagnostics against the OSDI
    // oracle (run the same card through ngspice + SANE).
    fn card_path(default_card: &str) -> String {
        std::env::var("BSIM4_CARD").unwrap_or_else(|_| default_card.to_string())
    }
    fn card(path: &str) -> String {
        std::fs::read_to_string(path).unwrap_or_default()
    }
    fn dump(name: &str, sweep: &[f64], id: &[f64]) {
        if let Ok(dir) = std::env::var("BSIM4_OUT") {
            let mut out = String::new();
            for (x, y) in sweep.iter().zip(id) {
                out.push_str(&format!(
                    "{x:.6e} {y:.8e}
"
                ));
            }
            let _ = std::fs::write(format!("{dir}/{name}"), out);
        }
    }

    /// Build a single-NMOS deck: VA model + nmos card + the 4 bias sources.
    /// `None` when the VA-Models corpus is not provisioned (test self-skips).
    fn nmos_deck(vd: f64, vg: f64, w: &str, l: &str) -> Option<String> {
        let (va, card_default) = va_models()?;
        Some(format!(
            ".veriloga \"{va}\"\n{card}\n\
             vd d 0 dc {vd}\nvg g 0 dc {vg}\nvs s 0 dc 0\nvb b 0 dc 0\n\
             NM1 d g s b n1 W={w} L={l} nf=1\n.end\n",
            card = card(&card_path(&card_default)),
        ))
    }

    /// Pull a current trace by element name from a DC sweep result.
    fn cur(r: &SweepResult, name: &str) -> Vec<f64> {
        r.traces
            .iter()
            .find(|t| t.kind == "current" && t.name.eq_ignore_ascii_case(name))
            .map(|t| t.values.clone())
            .unwrap_or_default()
    }

    // The relaxed BSIM4 .va bundled with the paper corpus (a test asset, not
    // shipped with the engine). Used to validate that a raw, level=54 SPICE
    // `.model` card -- the form PDKs actually ship -- imports natively via the
    // `M` element + `.model_alias`, with no external Python flattener.
    fn bundled_va() -> String {
        format!(
            "{}/../../paper/benchmarks/models/bsim4/bsim4.va",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    /// Regression guard for SANE's internal-node-collapse lowering: a
    /// gate-controlled channel lives between INTERNAL nodes di/si/gi, connected to
    /// the terminals only by `V(branch) <+ 0` collapses inside mode-gated
    /// conditionals -- the pattern compact models (BSIM4) use to wire S/D and the
    /// gate. The collapse path (rdsmod=0) must conduct identically to the explicit
    /// series-resistor path (rdsmod=1).
    #[test]
    fn va_internal_node_collapse_conducts() {
        let build = |rdsmod: f64| -> f64 {
            let deck = format!(
                ".veriloga\nmodule cmos(d,g,s,b); inout d,g,s,b; electrical d,g,s,b,di,si,gi;\n\
                 branch (d,di) br_v_d_di, br_i_d_di;\n\
                 branch (s,si) br_v_s_si, br_i_s_si;\n\
                 parameter real rdsmod={rdsmod}; parameter real gain=1e-3; real md;\n\
                 analog begin md=rdsmod;\n\
                 if (md != 0) begin I(br_i_d_di)<+V(d,di)*1.0; I(br_i_s_si)<+V(s,si)*1.0; end\n\
                 else begin V(br_v_d_di)<+0.0; V(br_v_s_si)<+0.0; end\n\
                 V(g,gi)<+0.0; I(di,si)<+gain*V(gi,si)*V(di,si); end endmodule\n.endveriloga\n\
                 vd d 0 dc 0.1\nvg g 0 dc 1.0\nvs s 0 dc 0\nvb b 0 dc 0\nM1 d g s b cmos\n.end\n"
            );
            let r = dc_sweep(deck, "vg".into(), 1.0, 1.0, 1.0);
            assert!(
                r.ok,
                "collapse deck (rdsmod={rdsmod}) DC failed: {}",
                r.message
            );
            cur(&r, "vd").first().map(|x| -x).unwrap_or(0.0)
        };
        let collapse = build(0.0);
        let resistor = build(1.0);
        assert!(
            collapse > 1e-7,
            "collapse (rdsmod=0) conducts: {collapse:.3e}"
        );
        let rel = (collapse - resistor).abs() / (resistor.abs() + 1e-300);
        assert!(
            rel < 1e-6,
            "collapse {collapse:.6e} == resistor {resistor:.6e}"
        );
    }

    /// Full native BSIM4 conduction on the real SKY130 nominal nmos card (247
    /// params, tests/data/sky130_nfet_tt.card): a raw `level=54` card imports via
    /// `M` + `.model_alias` (no Python flattener) and the 9723-line compact model
    /// lowers to a physical transistor -- Id-Vg turns on around Vth (~0.5 V) and
    /// rises into strong inversion. Needs the switch-branch node-collapse fix
    /// (else S/D float at the gmin floor).
    #[test]
    #[ignore]
    fn bsim4_sky130_card_conducts() {
        let va = bundled_va();
        let card_path = format!(
            "{}/tests/data/sky130_nfet_tt.card",
            env!("CARGO_MANIFEST_DIR")
        );
        if !std::path::Path::new(&va).exists() || !std::path::Path::new(&card_path).exists() {
            eprintln!("bundled bsim4.va / sky130 card missing, skipping");
            return;
        }
        let card = std::fs::read_to_string(&card_path).unwrap();
        let deck = format!(
            ".veriloga \"{va}\"\n.model_alias level=54 bsim4va\n{card}\n\
             vd d 0 dc 0.05\nvg g 0 dc 1.0\nvs s 0 dc 0\nvb b 0 dc 0\n\
             M1 d g s b nch W=1e-6 L=0.15e-6\n.end\n"
        );
        let r = dc_sweep(deck, "vg".into(), 0.0, 1.8, 0.3);
        assert!(r.ok, "SKY130 nmos native import/DC failed: {}", r.message);
        let id: Vec<f64> = cur(&r, "vd").iter().map(|x| -x).collect();
        assert!(!id.is_empty(), "vd current trace present");
        eprintln!("SKY130 nominal nmos Id-Vg (native level=54 import):");
        for (k, vg) in r.sweep.iter().enumerate() {
            eprintln!("  Vg={:.2}  Id={:.6e}", vg, id[k]);
        }
        // Physical transistor: off below Vth, strong current in inversion, and a
        // large on/off ratio over the gate sweep (the channel tracks Vgs).
        let lo = id[0];
        let hi = *id.last().unwrap();
        assert!(hi > 1e-5, "strong-inversion drain current: {hi:.3e}");
        assert!(
            hi > 1e6 * lo.max(1e-300),
            "Id-Vg turns on (on/off {:.1e})",
            hi / lo.max(1e-300)
        );
    }

    /// Native compact-model import: a legacy `level=54` nmos card, bound to the
    /// VA module by `.model_alias`, instantiated with `M` (d g s b). No `N`
    /// element, no hand-rewriting of the card -- exactly a PDK deck's form.
    #[test]
    #[ignore]
    fn bsim4_native_level54_m_element_dc() {
        let va = bundled_va();
        if !std::path::Path::new(&va).exists() {
            eprintln!("bundled bsim4.va missing, skipping");
            return;
        }
        // tnoimod=0: the noise model is DC/AC-irrelevant and SKY130's out-of-range
        // tnoia/tnoib otherwise poison the setup. A minimal card; the .va supplies
        // physical defaults for everything else.
        let card = "level=54 version=4.5 tnoimod=0 vth0=0.4 u0=0.05 toxe=4e-9 vsat=8e4";
        let bias = "vd d 0 dc 0.05\nvg g 0 dc 1.0\nvs s 0 dc 0\nvb b 0 dc 0";
        // M element: legacy nmos card routed by .model_alias.
        let m_deck = format!(
            ".veriloga \"{va}\"\n.model_alias level=54 bsim4va\n\
             .model nch nmos {card}\n{bias}\nM1 d g s b nch W=10e-6 L=1e-6\n.end\n"
        );
        // N element: same module/params, the already-supported direct idiom.
        let n_deck = format!(
            ".veriloga \"{va}\"\n.model nch bsim4va type=1 {card}\n\
             {bias}\nNM1 d g s b nch W=10e-6 L=1e-6\n.end\n"
        );
        let rm = dc_sweep(m_deck, "vg".into(), 0.0, 1.2, 0.1);
        let rn = dc_sweep(n_deck, "vg".into(), 0.0, 1.2, 0.1);
        assert!(rm.ok, "native level=54 (M) DC failed: {}", rm.message);
        assert!(rn.ok, "reference (N) DC failed: {}", rn.message);
        let idm: Vec<f64> = cur(&rm, "vd").iter().map(|x| -x).collect();
        let idn: Vec<f64> = cur(&rn, "vd").iter().map(|x| -x).collect();
        assert!(!idm.is_empty(), "no M vd current trace");
        eprintln!("native BSIM4 Id-Vg (M routed via .model_alias  vs  N direct):");
        for (k, vg) in rm.sweep.iter().enumerate() {
            eprintln!("  Vg={:.2}  M={:.6e}  N={:.6e}", vg, idm[k], idn[k]);
        }
        // The point of this test: the real, 9723-line BSIM4 .va imports natively
        // from a raw level=54 card via `M` + `.model_alias`, and the routed path
        // matches the already-supported `N`-direct idiom BIT-FOR-BIT (M is pure
        // SPICE sugar over N). Physical accuracy of a given card is owned by the
        // golden tests below; here routing correctness is what is asserted.
        for (k, (&a, &b)) in idm.iter().zip(&idn).enumerate() {
            let rel = (a - b).abs() / (b.abs() + 1e-300);
            assert!(
                rel < 1e-9,
                "M/N parity @Vg={:.2}: M={a:.6e} N={b:.6e}",
                rm.sweep[k]
            );
        }
    }

    /// End-to-end conduction through the native import path: a compact MOS
    /// (EKV, from the corpus) routed by `.model_alias` + `M` must produce a real,
    /// rising channel current -- proving the path is useful, not just wired.
    #[test]
    #[ignore]
    fn ekv_native_level_routing_conducts() {
        let va = format!(
            "{}/../../paper/benchmarks/models/ekv/vacode/ekv26.va",
            env!("CARGO_MANIFEST_DIR")
        );
        if !std::path::Path::new(&va).exists() {
            eprintln!("EKV .va missing, skipping");
            return;
        }
        // EKV has no standard SPICE level; bind an arbitrary one to the module.
        let deck = format!(
            ".veriloga \"{va}\"\n.model_alias level=99 ekv26_va\n\
             .model nch nmos level=99 VTO=0.5\n\
             vd d 0 dc 0.5\nvg g 0 dc 1.0\nvs s 0 dc 0\nvb b 0 dc 0\n\
             M1 d g s b nch W=10e-6 L=1e-6\n.end\n"
        );
        let r = dc_sweep(deck, "vg".into(), 0.0, 1.2, 0.1);
        assert!(r.ok, "EKV native DC failed: {}", r.message);
        let id: Vec<f64> = cur(&r, "vd").iter().map(|x| -x).collect();
        assert!(!id.is_empty(), "no vd current trace");
        eprintln!("native EKV Id-Vg (M element, level=99 -> ekv26_va):");
        for (k, vg) in r.sweep.iter().enumerate() {
            eprintln!("  Vg={:.2}  Id={:.6e}", vg, id[k]);
        }
        let (lo, hi) = (id[0], *id.last().unwrap());
        assert!(hi > 1e-9, "EKV conducts in strong inversion: {hi:.3e}");
        assert!(hi > lo, "EKV Id rises with Vg: lo={lo:.3e} hi={hi:.3e}");
    }

    // --- Native AnalogGym SKY130 suite -------------------------------------
    //
    // The 12 SKY130 AnalogGym amplifiers ingested NATIVELY: raw PDK netlist +
    // binned `level=54` corner + design variables + a device-subckt library,
    // bound through `.model_alias` + the relaxed BSIM4 `.va`. No Python
    // flattener -- exactly the deck a PDK ships. Each amp is wired into the
    // open-loop testbench (DC follower via a 1T feedback inductor) and DC-solved.
    // Ignored by default (heavy: the 9723-line BSIM4 lowered per transistor).
    const ANALOGGYM: &[&str] = &[
        "Alfio_RAFFC_Pin_3",
        "Fan_SMC_Pin_3",
        "HoiLee_AFFC_Pin_3",
        "Leung_DFCFC1_Pin_3",
        "Leung_DFCFC2_Pin_3",
        "Leung_NMCF_Pin_3",
        "Leung_NMCNR_Pin_3",
        "Peng_ACBC_Pin_3",
        "Qu2017_AZC_Pin_3",
        "Ramos_PFC_Pin_3",
        "Sau_CFCC_Pin_3",
        "Yan_AZ_Pin_3",
    ];

    fn analoggym_dir() -> String {
        format!(
            "{}/../../paper/benchmarks/circuits/analoggym",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    /// Build the native open-loop testbench for one amp (mirrors the Python
    /// `bench_analoggym.testbench`, but feeds SANE the RAW decks via `.include`).
    fn native_tb(amp: &str) -> String {
        let va = bundled_va();
        let dir = analoggym_dir();
        // The device library pins the nominal-corner mismatch slopes to zero, so
        // the corner's toxe/vth0/voff card expressions resolve to their nominal
        // values (matching the reference flow) instead of silently dropping.
        format!(
            ".veriloga \"{va}\"\n\
             .model_alias level=54 bsim4va\n\
             .option scale=1u\n\
             .include \"{dir}/tt.spice\"\n\
             .include \"{dir}/design_variables/{amp}\"\n\
             .include \"{dir}/sky130_devices.spice\"\n\
             .include \"{dir}/netlist/{amp}\"\n\
             .param mc_mm_switch=0\n.param mc_pr_switch=0\n\
             V1 vdd 0 1.8\nVindc opin 0 0.45\nVin signal_in 0 dc 0.45 ac 1\n\
             Lfb vout opout_dc 1T\nCin opout_dc signal_in 1T\n\
             Xop 0 vdd opout_dc opin vout {amp_l}\nCload vout 0 500p\n\
             .nodeset v(vout)=0.45\n.end\n",
            amp_l = amp.to_lowercase(),
        )
    }

    #[test]
    #[ignore]
    fn analoggym_one_native() {
        let va = bundled_va();
        if !std::path::Path::new(&va).exists() {
            eprintln!("bundled bsim4.va missing, skipping");
            return;
        }
        let amp = "HoiLee_AFFC_Pin_3"; // a representative converging amp
        let t0 = sane_core::time::Instant::now();
        let r = simulate(native_tb(amp));
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "{amp}: ok={} dim={} ({:.1}s) msg={}",
            r.ok, r.dim, dt, r.message
        );
        if r.ok {
            for s in &r.signals {
                if matches!(s.name.as_str(), "vout" | "vdd" | "opin" | "opout_dc") {
                    eprintln!("  {} = {:.4} V", s.name, s.value);
                }
            }
        }
        assert!(r.ok, "{amp} native DC failed: {}", r.message);
    }

    // All twelve amplifiers now converge natively: correct nominal card parameters
    // (the model-card expression-tokeniser fix), the SPICE gmin-step-down fallback
    // in source continuation (which rescues the auto-zeroing choppers whose
    // high-impedance output is unstable at the bare 1e-12 floor), and a follower
    // node-set that selects the mid-rail basin for the multi-solution Qu2017_AZC.
    const KNOWN_NONCONVERGENT: &[&str] = &[];

    // Equivalence of the native import to the historical Python flattener, both
    // solved by the SAME current engine -- the strongest available oracle (ngspice
    // is absent; the stored JSON predates engine/model changes). The native deck
    // (M -> device-subckt -> binned card -> VA) and the flattened deck (N -> VA,
    // params baked numerically) assemble to the IDENTICAL system: equal DAE
    // dimension, and -- now that the card-expression parser keeps statistical
    // (AGAUSS) params intact -- the same nominal device parameters, so where both
    // converge the operating point agrees to <5e-3 -- except the genuinely
    // ill-determined low-/moderate-gain amps (Alfio, Fan), which sit in a slightly
    // different valid basin. The flattened deck is given the same follower node-set
    // so the multi-solution choppers select the same mid-rail basin. Reads
    // $FLAT_DIR/_flat_<amp>.cir (from sky130.py emit_sane); skipped when unset.
    #[test]
    #[ignore]
    fn analoggym_native_matches_flattened() {
        let Ok(flat_dir) = std::env::var("FLAT_DIR") else {
            eprintln!("set FLAT_DIR=<dir with _flat_<amp>.cir> to run; skipping");
            return;
        };
        eprintln!("\nnative vs flattened (same engine):");
        for &amp in ANALOGGYM {
            let flat_path = format!("{flat_dir}/_flat_{amp}.cir");
            let Ok(flat_txt) = std::fs::read_to_string(&flat_path) else {
                continue;
            };
            // Apples-to-apples: give the flattened deck the SAME follower node-set the
            // native testbench carries, so the multi-solution choppers select the
            // same (mid-rail) basin instead of an arbitrary other valid one.
            let flat_txt = flat_txt.replacen(".end", ".nodeset v(vout)=0.45\n.end", 1);
            let rn = simulate(native_tb(amp));
            let rf = simulate(flat_txt);
            if !rn.ok || !rf.ok {
                eprintln!("  {amp:<22} native_ok={} flat_ok={}", rn.ok, rf.ok);
                continue;
            }
            // Same circuit -> same assembled system dimension.
            assert_eq!(
                rn.dim, rf.dim,
                "{amp}: native/flattened DAE dimension must match"
            );
            let get =
                |r: &SimResult, n: &str| r.signals.iter().find(|s| s.name == n).map(|s| s.value);
            let mut worst = 0.0f64;
            for node in ["vout", "opin", "opout_dc", "vdd"] {
                if let (Some(a), Some(b)) = (get(&rn, node), get(&rf, node)) {
                    worst = worst.max((a - b).abs());
                }
            }
            // The low-/moderate-gain amps with an ill-determined output bias
            // (Alfio is the -18.8 dB outlier) settle in a slightly different but
            // valid basin; the well-conditioned majority agree to <5e-3.
            let different_basin = matches!(amp, "Alfio_RAFFC_Pin_3" | "Fan_SMC_Pin_3");
            eprintln!(
                "  {amp:<22} dim={} vout(native)={:.4} vout(flat)={:.4}  max|Δ|={worst:.2e}{}",
                rn.dim,
                get(&rn, "vout").unwrap_or(0.0),
                get(&rf, "vout").unwrap_or(0.0),
                if worst < 5e-3 {
                    ""
                } else {
                    "  (different basin)"
                }
            );
            if !different_basin {
                assert!(
                    worst < 5e-3,
                    "{amp}: native vs flattened OP mismatch {worst:.3e}"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn analoggym_suite_native() {
        let va = bundled_va();
        if !std::path::Path::new(&va).exists() {
            eprintln!("bundled bsim4.va missing, skipping");
            return;
        }
        let mut pass = 0usize;
        let mut unexpected = Vec::new();
        eprintln!("\nnative SKY130 AnalogGym suite (DC operating point):");
        eprintln!(
            "  {:<22} {:>5} {:>6} {:>8} {:>7}",
            "amp", "ok", "dim", "vout/V", "t/s"
        );
        for &amp in ANALOGGYM {
            let t0 = sane_core::time::Instant::now();
            let r = simulate(native_tb(amp));
            let dt = t0.elapsed().as_secs_f64();
            let vout = r.signals.iter().find(|s| s.name == "vout").map(|s| s.value);
            eprintln!(
                "  {:<22} {:>5} {:>6} {:>8} {:>7.1}{}",
                amp,
                r.ok,
                r.dim,
                vout.map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "-".into()),
                dt,
                if r.ok { "" } else { "  <-- non-convergent" },
            );
            if r.ok {
                pass += 1;
                // A converged operating point must sit within the supply rails;
                // the exact bias is amp-specific (the low-gain Alfio sits near the
                // bottom, high-gain amps near mid-rail), so only the rail bound is
                // asserted -- numeric agreement with the reference is the job of
                // analoggym_native_matches_flattened.
                let v = vout.expect("converged amp reports vout");
                assert!(
                    (-0.1..=1.9).contains(&v),
                    "{amp} vout {v:.3} V outside supply rails"
                );
            } else if !KNOWN_NONCONVERGENT.contains(&amp) {
                unexpected.push(amp);
            }
        }
        eprintln!(
            "\n{pass}/{} converged natively ({} known pre-existing non-convergent: {:?})",
            ANALOGGYM.len(),
            KNOWN_NONCONVERGENT.len(),
            KNOWN_NONCONVERGENT,
        );
        assert!(
            unexpected.is_empty(),
            "amps regressed (were converging): {unexpected:?}"
        );
        assert_eq!(
            pass,
            ANALOGGYM.len() - KNOWN_NONCONVERGENT.len(),
            "all but the known pre-existing non-convergent amps solve natively",
        );
    }

    #[test]
    #[ignore]
    fn bsim4_id_vg_dc() {
        // Id-Vg at Vds = 50 mV, Vg 0 -> 1.2 V (the reference id_nmos sweep).
        let Some(deck) = nmos_deck(0.05, 1.0, "10e-6", "1e-6") else {
            eprintln!("VA-Models corpus missing (set SANE_VA_MODELS), skipping");
            return;
        };
        let r = dc_sweep(deck, "vg".into(), 0.0, 1.2, 0.02);
        assert!(r.ok, "DC sweep failed: {}", r.message);
        let names: Vec<&str> = r.traces.iter().map(|t| t.name.as_str()).collect();
        eprintln!("traces: {names:?}");
        // Drain current = current delivered by vd (Id = -i(vd)).
        let ivd = cur(&r, "vd");
        assert!(!ivd.is_empty(), "no vd current trace");
        eprintln!("Vg, Id(=-i(vd)) [A]:");
        for (k, vg) in r.sweep.iter().enumerate() {
            if k % 5 == 0 {
                eprintln!("  {:.3}  {:.6e}", vg, -ivd[k]);
            }
        }
        // Physical sanity: Id rises monotonically with Vg and is positive in
        // strong inversion.
        let id: Vec<f64> = ivd.iter().map(|x| -x).collect();
        dump("sane_idvg.dat", &r.sweep, &id);
        // Golden anchors from the SAME .va compiled to OSDI and run in ngspice
        // (and confirmed equal to ngspice native BSIM4 level=14): (Vg, Id).
        for (vg, gold) in [(0.4, 2.7645e-5), (0.8, 9.4323e-5), (1.2, 1.1368e-4)] {
            let k = r
                .sweep
                .iter()
                .position(|x| (x - vg).abs() < 1e-6)
                .expect("Vg point");
            let rel = (id[k] - gold).abs() / gold;
            assert!(
                rel < 5e-3,
                "Id-Vg @Vg={vg}: SANE {:.5e} vs OSDI {gold:.5e} (rel {:.2}%)",
                id[k],
                rel * 100.0
            );
        }
    }

    #[test]
    #[ignore]
    fn bsim4_id_vd_dc() {
        // Output characteristic: Id-Vd at Vg = 1.0 V, Vd 0 -> 1.2 V.
        let Some(deck) = nmos_deck(0.05, 1.0, "10e-6", "1e-6") else {
            eprintln!("VA-Models corpus missing (set SANE_VA_MODELS), skipping");
            return;
        };
        let r = dc_sweep(deck, "vd".into(), 0.0, 1.2, 0.02);
        assert!(r.ok, "DC sweep failed: {}", r.message);
        let ivd = cur(&r, "vd");
        assert!(!ivd.is_empty(), "no vd current trace");
        eprintln!("Vd, Id [A]:");
        for (k, vd) in r.sweep.iter().enumerate() {
            if k % 5 == 0 {
                eprintln!("  {:.3}  {:.6e}", vd, -ivd[k]);
            }
        }
        let id: Vec<f64> = ivd.iter().map(|x| -x).collect();
        dump("sane_idvd.dat", &r.sweep, &id);
        // Golden anchors (OSDI of the same .va) for the output characteristic.
        for (vd, gold) in [(0.2, 3.76193e-4), (0.6, 6.31018e-4), (1.2, 6.70658e-4)] {
            let k = r
                .sweep
                .iter()
                .position(|x| (x - vd).abs() < 1e-6)
                .expect("Vd point");
            let rel = (id[k] - gold).abs() / gold;
            assert!(
                rel < 5e-3,
                "Id-Vd @Vd={vd}: SANE {:.5e} vs OSDI {gold:.5e} (rel {:.2}%)",
                id[k],
                rel * 100.0
            );
        }
    }

    /// Common-source gain stage deck (gstage): single NMOS, Rsource, Rload.
    /// `None` when the VA-Models corpus is not provisioned (test self-skips).
    fn gstage_deck(extra_vin: &str) -> Option<String> {
        let (va, card_default) = va_models()?;
        Some(format!(
            ".veriloga \"{va}\"\n{card}\n\
             NM1 d g 0 0 N1 L=0.09u W=4u\n\
             Rsource 1 g 100k\nRload d vdd 1k\nVdd vdd 0 1.8\n\
             Vin 1 0 dc 0.8 ac 1 {extra_vin}\n.end\n",
            card = card(&card_path(&card_default)),
        ))
    }

    #[test]
    #[ignore]
    fn bsim4_gstage_ac() {
        let Some(deck) = gstage_deck("") else {
            eprintln!("VA-Models corpus missing (set SANE_VA_MODELS), skipping");
            return;
        };
        let r = ac_analysis(deck, "Vin".into(), "d".into(), 100.0, 1e9, 10);
        assert!(r.ok, "AC failed: {}", r.message);
        for (f, m) in r.f.iter().zip(&r.mag_db) {
            if (f.log10().fract()).abs() < 1e-6 {
                eprintln!("  f={:.3e}  |H|={:.4} dB", f, m);
            }
        }
        // Golden (OSDI of the same .va): low-frequency gain 8.7485 dB, and the
        // amplifier rolls off (gain at 1 GHz well below the passband).
        let lf = r.mag_db[0];
        assert!(
            (lf - 8.7485).abs() < 0.05,
            "low-freq gain {lf} dB vs OSDI 8.7485 dB"
        );
        let hf = *r.mag_db.last().unwrap();
        assert!(hf < lf - 10.0, "expected roll-off: hf={hf} lf={lf}");
    }

    #[test]
    #[ignore]
    fn bsim4_gstage_tran() {
        // Transient integration of the full BSIM4 stage. The drive is constant
        // DC, so the engine starts from a consistent operating point and the
        // adaptive BDF advances the device + its capacitances in the time domain,
        // remaining stable at the operating point. (The lowered charge model is
        // separately validated to be exact by the AC roll-off match above; a
        // continuous time-varying-source swing is undersampled by the engine's
        // adaptive integrator -- a pre-existing limitation, not a BSIM4 lowering
        // issue, reproducible on a plain native RC.)
        let Some(deck) = gstage_deck("") else {
            eprintln!("VA-Models corpus missing (set SANE_VA_MODELS), skipping");
            return;
        };
        let r = transient(deck, 5e-8, 1e-10);
        assert!(r.ok, "transient failed: {}", r.message);
        let vd = r
            .traces
            .iter()
            .find(|t| t.name == "d")
            .expect("node d trace");
        assert!(!vd.values.is_empty(), "no transient samples");
        // Stable time integration: the BSIM4 stage holds its DC operating point
        // (V(d) ~ 0.77 V from the validated DC solution) across the whole window.
        let (mn, mx) = vd
            .values
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        eprintln!("V(d) transient range: [{mn:.4}, {mx:.4}]");
        assert!(
            mn > 0.70 && mx < 0.84,
            "V(d) not stable near the DC op: [{mn}, {mx}]"
        );
    }
}

mod nonlinearity_reality {
    //! The DAG nonlinearity analysis against real parsed circuits: it is what a
    //! harmonic-balance solve reads to pick its harmonic count and time sampling.
    use super::*;
    use rsdag::UnaryOp;

    fn classify(net: &str) -> rsdag::Nonlinearity {
        let parsed = parse(net).expect("parse");
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        dae.nonlinearity(&ctx)
    }

    #[test]
    fn linear_rc_is_polynomial_degree_one() {
        // A pure R/C/V network is linear: degree-1 in the unknowns, no structural
        // features, so AFT needs only 2*1*K+1 samples and is even avoidable.
        let nl = classify("V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.end");
        assert_eq!(nl.degree, rsdag::Degree::Finite(1));
        assert!(nl.is_polynomial());
        assert!(nl.transcendental.is_empty());
        assert!(!nl.piecewise);
        let d = 1; // polynomial degree of the linear residual
        assert_eq!(nl.alias_free_samples(8), Some(2 * d * 8 + 1));
    }

    #[test]
    fn diode_clipper_is_transcendental_and_piecewise() {
        // The diode model is exp(v/vt) guarded by a select above vcrit: both
        // transcendental (exp over an unknown) and piecewise (the region branch).
        // No finite harmonic bound -> the solver must oversample.
        let net = "V1 1 0 AC 5\nR1 1 2 1k\nD1 0 2 DMOD\nD2 2 0 DMOD\n\
                   .model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end";
        let nl = classify(net);
        assert!(!nl.is_polynomial());
        assert!(nl.transcendental.contains(&UnaryOp::Exp));
        assert!(nl.piecewise, "the diode's region select is piecewise");
        assert_eq!(nl.alias_free_samples(8), None);
    }

    #[test]
    fn mosfet_is_piecewise_and_transcendental() {
        // The level-1 MOSFET is a square law within strong inversion, switched
        // between cutoff/triode/saturation by selects (piecewise), AND carries an
        // exponential weak-inversion (subthreshold) tail below von -- so it is
        // both piecewise and transcendental, like the diode: no finite harmonic
        // bound, the HB solver must oversample.
        let net = "M1 1 2 0 0 NMOS1\nVin 2 0 DC 0.5\nV1 1 0 DC 24\n\
                   .model NMOS1 NMOS(Kp=20u W=1 L=1 Vto=-1)\n.end";
        let nl = classify(net);
        assert!(
            !nl.is_polynomial(),
            "region switching makes it non-polynomial"
        );
        assert!(nl.piecewise);
        assert!(
            nl.transcendental.contains(&UnaryOp::Exp),
            "subthreshold is exponential"
        );
        assert_eq!(nl.alias_free_samples(8), None);
    }
}

mod hb_verify {
    //! Harmonic balance against independent references.
    use super::*;
    use num_complex::Complex64;
    use sane_solve::hb::{hb_samples, CompiledHb};

    fn unknown_idx(dae: &sane_dae::Dae, parsed: &sane_netlist::ParsedCircuit, node: &str) -> usize {
        let k = parsed.node(node).unwrap();
        dae.unknowns
            .iter()
            .position(|u| *u == format!("v{k}"))
            .unwrap()
    }

    #[test]
    fn linear_rc_matches_analytic_transfer() {
        // A linear RC driven by a 1 kHz tone: harmonic balance must reproduce the
        // exact transfer H(jw0) = 1/(1 + jw0 R C) as the ratio of the fundamental
        // coefficients out/in -- a direct analytic anchor (no AFT aliasing, since
        // the system is degree 1, and Newton is exact in one step).
        let net = "V1 in 0 SIN(0 1 1000)\nR1 in out 1k\nC1 out 0 1u\n.end";
        let parsed = parse(net).unwrap();
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        let cdc = CompiledDc::new(&mut ctx, &dae);
        let pnames = cdc.param_names(&ctx);
        let p: Vec<f64> = parsed.pvec(&pnames);
        let (x_dc, conv, _) = cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
        assert!(conv);
        let w0 = p[pnames.iter().position(|n| n == "V1.sin_w").unwrap()];

        let nl = dae.nonlinearity(&ctx);
        assert_eq!(nl.degree, rsdag::Degree::Finite(1));
        let k = 2;
        let m = hb_samples(&nl, k, 8);
        let hb = CompiledHb::new(&cdc, k, m).unwrap();
        let res = hb.solve(&p, &x_dc, w0, 1e-12, 20);
        assert!(
            res.converged,
            "HB did not converge (||R|| = {:.2e})",
            res.residual_norm
        );

        let in_i = unknown_idx(&dae, &parsed, "in");
        let out_i = unknown_idx(&dae, &parsed, "out");
        let h_hb = res.spectra[out_i][1] / res.spectra[in_i][1];
        let (r, c) = (1e3, 1e-6);
        let h_ref = Complex64::new(1.0, 0.0) / Complex64::new(1.0, w0 * r * c);
        assert!(
            (h_hb - h_ref).norm() < 1e-6,
            "HB transfer {h_hb:?} vs analytic {h_ref:?}"
        );

        // A linear system generates no second harmonic.
        assert!(res.spectra[out_i][2].norm() < 1e-9, "spurious 2nd harmonic");
    }

    /// Physical Fourier coefficient `X_k` of an evenly sampled real period:
    /// `(1/M) Σ_m s_m e^{-2pi i k m / M}` -- same convention as the HB spectra.
    fn dft_coeff(samples: &[f64], k: usize) -> Complex64 {
        let m = samples.len() as f64;
        let mut acc = Complex64::new(0.0, 0.0);
        for (j, &s) in samples.iter().enumerate() {
            let ang = -2.0 * PI * k as f64 * j as f64 / m;
            acc += Complex64::from_polar(s, ang);
        }
        acc / m
    }

    #[test]
    fn diode_harmonics_match_transient_fft() {
        // A diode driven through an RC: weakly nonlinear, so it generates real
        // harmonics. Harmonic balance must reproduce the magnitudes of the first
        // few harmonics that a settled transient simulation produces -- the AFT
        // loop's end-to-end correctness, including the jw reactive coupling.
        // Biased into conduction so the diode dominates the node: a strong, smooth
        // nonlinearity (kept below vcrit) that generates clear harmonics. A small
        // shunt cap keeps the integrator well-behaved without filtering the signal.
        let net = "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 DMOD\n\
                   C1 mid 0 100n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end";
        let parsed = parse(net).unwrap();
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        let cdc = CompiledDc::new(&mut ctx, &dae);
        let pnames = cdc.param_names(&ctx);
        let p: Vec<f64> = parsed.pvec(&pnames);
        let (x_dc, conv, _) = cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
        assert!(conv);
        let w0 = p[pnames.iter().position(|n| n == "V1.sin_w").unwrap()];
        let mid = unknown_idx(&dae, &parsed, "mid");

        // Harmonic balance.
        let nl = dae.nonlinearity(&ctx);
        assert!(!nl.is_polynomial(), "diode is transcendental");
        let k = 8;
        let m = hb_samples(&nl, k, 16);
        let hb = CompiledHb::new(&cdc, k, m).unwrap();
        let res = hb.solve(&p, &x_dc, w0, 1e-10, 60);
        assert!(
            res.converged,
            "HB did not converge (||R|| = {:.2e})",
            res.residual_norm
        );

        // Settled transient, FFT of the last period.
        let period = 2.0 * PI / w0;
        let (n_per, mfft) = (20usize, 64usize);
        let t_eval: Vec<f64> = (0..n_per * mfft)
            .map(|j| j as f64 * period / mfft as f64)
            .collect();
        let traj = cdc
            .solve_transient(
                sane_solve::TransientMethod::Esdirk32,
                &p,
                &[],
                &t_eval,
                1e-6,
                1e-9,
                None,
            )
            .expect("transient");
        let last: Vec<f64> = (0..mfft)
            .map(|j| traj[(n_per - 1) * mfft + j][mid])
            .collect();

        // The fundamental must match tightly; the small harmonics a bit looser.
        for (kk, tol) in [(1usize, 2e-2), (2, 8e-2), (3, 1.5e-1)] {
            let hb_mag = res.spectra[mid][kk].norm();
            let tr_mag = dft_coeff(&last, kk).norm();
            let rel = (hb_mag - tr_mag).abs() / tr_mag.max(1e-12);
            assert!(
                rel < tol,
                "harmonic {kk}: HB |X| = {hb_mag:.3e}, transient |X| = {tr_mag:.3e} (rel {rel:.2e})"
            );
        }
        // The nonlinearity must actually produce a second harmonic.
        assert!(res.spectra[mid][2].norm() > 1e-6, "no HD2 generated");
    }

    #[test]
    fn hb_coeff_gradient_matches_finite_difference() {
        // The analytic adjoint gradient of a steady-state harmonic coefficient
        // w.r.t. a parameter, finite-difference checked. This guards the shared
        // Toeplitz-Jacobian assembly used by both coeff_gradient and coeff_hessian
        // (which otherwise had no numerical test).
        let net = "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 DMOD\n\
                   C1 mid 0 100n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end";
        let parsed = parse(net).unwrap();
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        let cdc = CompiledDc::new(&mut ctx, &dae);
        cdc.ensure_param_jac(&mut ctx, &dae);
        let pnames = cdc.param_names(&ctx);
        let p: Vec<f64> = parsed.pvec(&pnames);
        let (x_dc, conv, _) = cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
        assert!(conv);
        let w0 = p[pnames.iter().position(|n| n == "V1.sin_w").unwrap()];
        let mid = unknown_idx(&dae, &parsed, "mid");

        let k = 5usize;
        let m = hb_samples(&dae.nonlinearity(&ctx), k, 16);
        let hb = CompiledHb::new(&cdc, k, m).unwrap();
        let solve =
            |pv: &[f64]| -> Vec<Vec<Complex64>> { hb.solve(pv, &x_dc, w0, 1e-12, 80).spectra };

        let spec = solve(&p);
        let grad = hb
            .coeff_gradient(&spec, &p, w0, mid, pnames.len())
            .expect("gradient");

        // Central-difference d X_{mid,1} / d R1 against the analytic gradient.
        let j = pnames.iter().position(|n| n == "R1").unwrap();
        let eps = 1e-4 * p[j];
        let mut pp = p.clone();
        pp[j] += eps;
        let mut pm = p.clone();
        pm[j] -= eps;
        let fd = (solve(&pp)[mid][1] - solve(&pm)[mid][1]) / (2.0 * eps);
        let an = grad[1][j];
        let err = (fd - an).norm() / an.norm().max(1e-12);
        assert!(err < 5e-3, "analytic {an:?} vs FD {fd:?} (rel {err:.2e})");
    }

    #[test]
    fn hb_coeff_hessian_matches_gradient_finite_difference() {
        // The analytic second-order-adjoint Hessian of a harmonic coefficient,
        // finite-difference checked against the gradient. Guards coeff_hessian
        // (which reuses the shared AFT waveforms), previously untested.
        let net = "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 DMOD\n\
                   C1 mid 0 100n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end";
        let parsed = parse(net).unwrap();
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        let cdc = CompiledDc::new(&mut ctx, &dae);
        cdc.ensure_param_jac(&mut ctx, &dae);
        cdc.ensure_hessian(&mut ctx, &dae);
        let pnames = cdc.param_names(&ctx);
        let p: Vec<f64> = parsed.pvec(&pnames);
        let (x_dc, conv, _) = cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
        assert!(conv);
        let w0 = p[pnames.iter().position(|n| n == "V1.sin_w").unwrap()];
        let mid = unknown_idx(&dae, &parsed, "mid");
        let k = 5usize;
        let m = hb_samples(&dae.nonlinearity(&ctx), k, 16);
        let hb = CompiledHb::new(&cdc, k, m).unwrap();

        let j = pnames.iter().position(|n| n == "R1").unwrap();
        let k_metric = 1usize;
        let grad_at = |pv: &[f64]| -> Complex64 {
            let spec = hb.solve(pv, &x_dc, w0, 1e-12, 80).spectra;
            hb.coeff_gradient(&spec, pv, w0, mid, pnames.len()).unwrap()[k_metric][j]
        };
        let spec = hb.solve(&p, &x_dc, w0, 1e-12, 80).spectra;
        let hess = hb
            .coeff_hessian(&spec, &p, w0, mid, k_metric, &[j])
            .expect("hessian");

        let eps = 1e-3 * p[j];
        let mut pp = p.clone();
        pp[j] += eps;
        let mut pm = p.clone();
        pm[j] -= eps;
        let fd = (grad_at(&pp) - grad_at(&pm)) / (2.0 * eps);
        let an = hess[0][0];
        let err = (fd - an).norm() / an.norm().max(1e-12);
        assert!(
            err < 2e-2,
            "hessian analytic {an:?} vs FD {fd:?} (rel {err:.2e})"
        );
    }

    fn setup(
        net: &str,
    ) -> (
        Graph,
        sane_dae::Dae,
        CompiledDc,
        Vec<f64>,
        Vec<f64>,
        f64,
        Vec<usize>,
    ) {
        let parsed = parse(net).unwrap();
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        let cdc = CompiledDc::new(&mut ctx, &dae);
        let pnames = cdc.param_names(&ctx);
        let p: Vec<f64> = parsed.pvec(&pnames);
        let (x_dc, conv, _) = cdc.solve_dc(&p, &[], DC_OP_TOL, DC_OP_MAXIT);
        assert!(conv);
        let w0 = p[pnames.iter().position(|n| n == "V1.sin_w").unwrap()];
        let ramp: Vec<usize> = pnames
            .iter()
            .enumerate()
            .filter(|(_, n)| n.ends_with(".sin_amp"))
            .map(|(i, _)| i)
            .collect();
        (ctx, dae, cdc, p, x_dc, w0, ramp)
    }

    #[test]
    fn hb_infers_fundamental_from_source() {
        // With f0 <= 0 the fundamental is read from the periodic source (the 1 kHz
        // SIN), so an auto run must match an explicit f0 = 1000 run exactly.
        let net = "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 DMOD\n\
                   C1 mid 0 100n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end";
        let m = Model::from_netlist(net).expect("model");
        let auto = m.harmonic_balance(&[], 0.0, 8, None).expect("auto HB");
        let explicit = m
            .harmonic_balance(&[], 1000.0, 8, None)
            .expect("explicit HB");
        // The inferred fundamental (f0 <= 0 path) must be the source's 1 kHz.
        let f0 = m
            .cdc()
            .source_fundamental(&m.pvec(&[]))
            .expect("periodic source");
        assert!((f0 - 1000.0).abs() < 1e-6, "inferred f0 = {f0}");
        // Auto and explicit must produce identical per-unknown magnitude spectra.
        for u in m.unknowns() {
            let ma = auto.magnitude(u).expect("auto spectrum");
            let me = explicit.magnitude(u).expect("explicit spectrum");
            for (a, e) in ma.iter().zip(me) {
                assert!((a - e).abs() < 1e-9, "spectra differ: {a} vs {e}");
            }
        }
    }

    #[test]
    fn continuation_matches_direct_on_easy_circuit() {
        // Where a cold Newton already converges, source-stepping must reach the
        // exact same periodic steady state (it only changes the path, not the
        // fixed point).
        let net = "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 DMOD\n\
                   C1 mid 0 100n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end";
        let (ctx, _dae, cdc, p, x_dc, w0, ramp) = setup(net);
        let _ = &ctx;
        let nl = rsdag::Nonlinearity {
            degree: rsdag::Degree::Unbounded,
            ..Default::default()
        };
        let m = hb_samples(&nl, 8, 16);
        let hb = CompiledHb::new(&cdc, 8, m).unwrap();
        let direct = hb.solve(&p, &x_dc, w0, 1e-11, 60);
        let cont = hb.solve_continuation(&p, &x_dc, w0, &ramp, 1e-11, 60);
        assert!(direct.converged && cont.converged);
        let d: f64 = (0..direct.spectra.len())
            .flat_map(|i| (0..=8).map(move |k| (i, k)))
            .map(|(i, k)| (direct.spectra[i][k] - cont.spectra[i][k]).norm())
            .fold(0.0, f64::max);
        assert!(d < 1e-7, "continuation diverged from direct by {d:.2e}");
    }

    #[test]
    fn damped_cold_newton_reaches_the_hard_drive() {
        // A hard, large-swing drive (2 V into a 50 ohm + diode). The undamped
        // cold Newton from DC diverged here and only source-stepping
        // converged; with the retroactive backtracking the cold Newton
        // converges too, and both land on the same spectrum.
        let net = "V1 in 0 SIN(0 2.0 1000)\nR1 in mid 50\nD1 mid 0 DMOD\n\
                   C1 mid 0 100n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end";
        let (ctx, _dae, cdc, p, x_dc, w0, ramp) = setup(net);
        let _ = &ctx;
        let nl = rsdag::Nonlinearity {
            degree: rsdag::Degree::Unbounded,
            ..Default::default()
        };
        let m = hb_samples(&nl, 8, 16);
        let hb = CompiledHb::new(&cdc, 8, m).unwrap();
        let direct = hb.solve(&p, &x_dc, w0, 1e-9, 60);
        assert!(
            direct.converged && direct.residual_norm < 1e-9,
            "damped cold Newton should converge (conv={}, ||R||={:.2e}, {} iterations)",
            direct.converged,
            direct.residual_norm,
            direct.iters
        );
        let cont = hb.solve_continuation(&p, &x_dc, w0, &ramp, 1e-9, 60);
        assert!(
            cont.converged && cont.residual_norm < 1e-9,
            "continuation should converge (conv={}, ||R||={:.2e})",
            cont.converged,
            cont.residual_norm
        );
        let d: f64 = (0..direct.spectra.len())
            .flat_map(|i| (0..=8).map(move |k| (i, k)))
            .map(|(i, k)| (direct.spectra[i][k] - cont.spectra[i][k]).norm())
            .fold(0.0, f64::max);
        assert!(d < 1e-6, "direct and continuation differ by {d:.2e}");
    }
}

mod sensitivity_ad_tests {
    use super::*;
    use num_complex::Complex64;

    /// Exact AC sensitivity of an RC lowpass at its corner (wRC = 1):
    /// |H|^2 = 1/(1+(wRC)^2), so d ln|H|/d ln R = d ln|H|/d ln C = -(wRC)^2/(1+(wRC)^2)
    /// = -1/2 at the corner. No finite differences -- this checks the autodiff path.
    #[test]
    fn ac_sensitivity_rc_matches_analytic() {
        let net = "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n";
        let fc = 1.0 / (2.0 * std::f64::consts::PI * 1e3 * 1e-6); // wRC = 1
        let m = Model::from_netlist(net).expect("model");
        let op = m.operating_point(&[]).expect("operating point");
        let x = op.vector().to_vec();
        let p = m.pvec(&[]);
        let out_idx = m.resolve("out").expect("output");
        let h = m
            .ac_response("V1", out_idx, x.clone(), p.clone(), vec![fc])
            .expect("ac")[0];
        let h0 = Complex64::new(h.0, h.1);
        // d ln|H|/d ln p = p * Re(conj(H) dH/dp) / |H|^2 (the deleted façade's rel).
        let rel = |param: &str| -> f64 {
            let dh = m
                .ac_sensitivity("V1", param, out_idx, x.clone(), p.clone(), vec![fc])
                .expect("ac_sens")[0];
            let dhc = Complex64::new(dh.0, dh.1);
            m.get(param).unwrap() * (h0.conj() * dhc).re / (h0.norm() * h0.norm())
        };
        let r1 = rel("R1");
        let c1 = rel("C1");
        assert!(
            (r1 + 0.5).abs() < 1e-3,
            "d ln|H|/d ln R1 = {r1}, expected -0.5"
        );
        assert!(
            (c1 + 0.5).abs() < 1e-3,
            "d ln|H|/d ln C1 = {c1}, expected -0.5"
        );
    }

    /// In the settled limit the exact transient sensitivity must converge to the
    /// exact DC sensitivity. A capacitively-loaded divider settles to
    /// out = V*R2/(R1+R2); both methods (augmented-DAE forward sensitivity and the
    /// adjoint DC sensitivity) are pure autodiff and must agree.
    #[test]
    fn transient_sensitivity_settles_to_dc() {
        let net = "V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\nC1 out 0 1u\n";
        let m = Model::from_netlist(net).expect("model");
        let n = m.dim();
        let out_idx = m.resolve("out").expect("output");
        // RC = (R1||R2)*C = 500 us; integrate well past it.
        let (tstop, tstep) = (1e-2_f64, 5e-5_f64);
        let npts = ((tstop / tstep).round() as usize).clamp(1, 100_000);
        let t_eval: Vec<f64> = (0..=npts).map(|k| k as f64 * tstop / npts as f64).collect();
        // Forward transient sensitivity: d ln(out(tstop))/d ln R1, normalised by the
        // baseline output at tstop (row [0..n) of the augmented trajectory).
        let (nn, traj) = m
            .transient_sensitivity(vec!["R1".to_string()], t_eval, 1e-4, 1e-7, None)
            .expect("transient_sensitivity");
        assert_eq!(nn, n);
        let last = traj.last().expect("trajectory");
        let (y, dydp) = (last[out_idx], last[n + out_idx]);
        let tr_r1 = dydp * m.get("R1").unwrap() / y;
        // Adjoint DC sensitivity at the operating point: rel = grad * p / value.
        let op = m.operating_point(&[]).expect("operating point");
        let sens = op.sensitivity("out").expect("sensitivity");
        let i = sens
            .names
            .iter()
            .position(|nm| nm == "R1")
            .expect("R1 column");
        let dc_r1 = sens.grad[i] * m.get("R1").unwrap() / sens.value;
        // Analytic d ln(out)/d ln R1 = -R1/(R1+R2) = -0.5.
        assert!((tr_r1 + 0.5).abs() < 1e-2, "transient R1 sens {tr_r1}");
        assert!(
            (tr_r1 - dc_r1).abs() < 1e-2,
            "transient {tr_r1} vs DC {dc_r1}"
        );
    }
}

mod nodeset_tests {
    //! `.nodeset` flows from the directive through `parse_nodeset` and `prepare`
    //! into every cold DC solve: the same bistable latch settles on whichever
    //! branch the node-set selects.
    use super::*;

    // High-impedance cross-coupled inverters (see crates/netlist/tests/nodeset.rs):
    // V(a)=g(V(b)), V(b)=g(V(a)) with a steep decreasing sigmoid; the symmetric
    // root a=b=0.5 is an unstable saddle.
    const LATCH: &str = "\
R1 a 0 1
R2 b 0 1
B1 0 a I=0.5 - 0.3183098862*atan(10*(V(b)-0.5))
B2 0 b I=0.5 - 0.3183098862*atan(10*(V(a)-0.5))
";

    fn va(op: &OperatingPoint, node: &str) -> f64 {
        op.get(node)
            .unwrap_or_else(|| panic!("no voltage for {node}"))
    }

    fn op_of(net: String) -> OperatingPoint {
        Model::from_netlist(&net)
            .expect("build model")
            .operating_point(&[])
            .expect("operating point")
    }

    #[test]
    fn parse_nodeset_reads_voltage_targets() {
        let ns = parse_nodeset(".nodeset V(a)=1 V(b)=0\n");
        assert_eq!(ns, vec![("a".to_string(), 1.0), ("b".to_string(), 0.0)]);
        // `.ic`-style branch currents are not node-sets.
        assert!(parse_nodeset(".nodeset I(L1)=1m\n").is_empty());
    }

    #[test]
    fn nodeset_directive_selects_the_branch() {
        // No node-set: the symmetric (metastable) root.
        let sym = op_of(format!("{LATCH}.op\n"));
        assert!((va(&sym, "a") - 0.5).abs() < 1e-6, "a = {}", va(&sym, "a"));

        // `.nodeset V(a)=1`: the a-high / b-low branch.
        let hi = op_of(format!("{LATCH}.nodeset V(a)=1\n.op\n"));
        assert!(
            va(&hi, "a") > 0.6 && va(&hi, "b") < 0.4,
            "a={} b={}",
            va(&hi, "a"),
            va(&hi, "b")
        );

        // `.nodeset V(a)=0`: the mirror branch.
        let lo = op_of(format!("{LATCH}.nodeset V(a)=0\n.op\n"));
        assert!(
            va(&lo, "a") < 0.4 && va(&lo, "b") > 0.6,
            "a={} b={}",
            va(&lo, "a"),
            va(&lo, "b")
        );
    }
}
