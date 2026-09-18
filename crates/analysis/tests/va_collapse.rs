//! Differential check of Verilog-A node collapsing on real compact models:
//! the DC operating point with collapsing ON must match the reference lowering
//! (`SANE_NO_COLLAPSE`, explicit zero-volt source branches) on the shared
//! unknowns. Ignored by default (needs SANE_VA_CORPUS, and it toggles a
//! process-global env var); run with `--ignored --test-threads=1`.

use sane_analysis::Model;

fn corpus() -> Option<std::path::PathBuf> {
    std::env::var_os("SANE_VA_CORPUS").map(std::path::PathBuf::from)
}

/// DC-solve `deck` and return (converged, port-node voltages by name).
fn dc(deck: &str, nodes: &[&str]) -> (bool, Vec<f64>) {
    let m = Model::from_netlist(deck).expect("model");
    match m.operating_point(&[]) {
        Ok(op) => (
            true,
            nodes
                .iter()
                .map(|n| op.get(n).unwrap_or(f64::NAN))
                .collect(),
        ),
        Err(_) => (false, Vec::new()),
    }
}

#[test]
#[ignore]
fn collapse_matches_reference_on_corpus_models() {
    let Some(base) = corpus() else {
        println!("SKIP (set SANE_VA_CORPUS)");
        return;
    };
    // diode_cmc: two terminals, defaults converge; bsim4: the collapse-heavy
    // 4-terminal case (16 -> 0 extras at defaults).
    let cases = [
        (
            "DIODE_CMC/diode_cmc.va",
            "diode_cmc",
            "N1 a 0 diode_cmc\nV1 a 0 0.5\nR1 a 0 1k\n",
            vec!["a"],
        ),
        (
            "BSIM4/bsim4.va",
            "bsim4",
            "N1 d g 0 0 bsim4va\nVd d 0 1.0\nVg g 0 0.8\n",
            vec!["d", "g", "Vd"],
        ),
        (
            "PSP103/psp103.va",
            "psp103",
            "N1 d g 0 0 psp103va\nVd d 0 1.0\nVg g 0 0.8\n",
            vec!["d", "g", "Vd"],
        ),
        (
            "HiSIM2/hisim2.va",
            "hisim2",
            "N1 d g 0 0 hisim2_va\nVd d 0 1.0\nVg g 0 0.8\n",
            vec!["d", "g", "Vd"],
        ),
        (
            "PSP102/psp102.va",
            "psp102",
            "N1 d g 0 0 psp102va\nVd d 0 1.0\nVg g 0 0.8\n",
            vec!["d", "g", "Vd"],
        ),
    ];
    for (rel, module, body, nodes) in cases {
        let path = base.join(rel);
        let Ok(_) = std::fs::read_to_string(&path) else {
            println!("{module}: SKIP (not present)");
            continue;
        };
        let deck = format!(".veriloga \"{}\"\n{body}.end\n", path.display());
        std::env::remove_var("SANE_NO_COLLAPSE");
        let (c_on, v_on) = dc(&deck, &nodes);
        std::env::set_var("SANE_NO_COLLAPSE", "1");
        let (c_off, v_off) = dc(&deck, &nodes);
        std::env::remove_var("SANE_NO_COLLAPSE");
        assert_eq!(c_on, c_off, "{module}: convergence must not differ");
        if c_on {
            for (i, n) in nodes.iter().enumerate() {
                let (a, b) = (v_on[i], v_off[i]);
                assert!(
                    (a - b).abs() <= 1e-9 * (1.0 + b.abs()),
                    "{module}: node {n} differs: collapsed {a} vs reference {b}"
                );
            }
            for (i, n) in nodes.iter().enumerate() {
                if *n == "Vd" {
                    assert!(
                        v_on[i].is_finite() && v_on[i].abs() > 1e-15,
                        "{module}: drain current should be finite and nonzero: {}",
                        v_on[i]
                    );
                }
            }
            println!("{module}: OK ({v_on:?})");
        } else {
            println!("{module}: both modes non-convergent (consistent)");
        }
    }
}
