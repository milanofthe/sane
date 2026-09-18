//! Deck-level OSDI test: `.osdi "lib"` + `N` instance through the full
//! netlist -> DAE -> DC pipeline. Ignored; needs SANE_OPENVAF_BIN and
//! SANE_VA_CORPUS.

use sane_analysis::Model;

#[test]
#[ignore]
fn osdi_deck_psp103_dc() {
    let (Some(corpus), Some(bin)) = (
        std::env::var_os("SANE_VA_CORPUS").map(std::path::PathBuf::from),
        std::env::var_os("SANE_OPENVAF_BIN").map(std::path::PathBuf::from),
    ) else {
        println!("SKIP (set SANE_VA_CORPUS and SANE_OPENVAF_BIN)");
        return;
    };
    let out = std::env::temp_dir().join("sane_osdi_deck_psp103.osdi");
    let status = std::process::Command::new(&bin)
        .arg(corpus.join("PSP103/psp103.va"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("run openvaf-r");
    assert!(status.success(), "openvaf-r failed");

    let deck = format!(
        ".osdi \"{}\"\nN1 d g 0 0 PSP103VA\nVd d 0 1.0\nVg g 0 0.8\n.end\n",
        out.display()
    );
    let m = Model::from_netlist(&deck).expect("model");
    let op = m.operating_point(&[]).expect("dc");
    let id = op.get("Vd").expect("drain current");
    let reference = -7.163534299181468e-5; // symbolic frontend, same bias
    assert!(
        (id - reference).abs() < 1e-9 * reference.abs() + 1e-12,
        "deck OSDI drain current {id} vs symbolic {reference}"
    );
    println!("osdi deck OK: I(Vd) = {id}");
}

#[test]
#[ignore]
fn osdi_noise_matches_symbolic_frontend() {
    // The same corpus diode once through the symbolic VA frontend and once as
    // an OSDI library: the output noise spectra must agree (shot/flicker on
    // the junction, thermal on the series resistance, plus R1's own 4kT/R).
    let (Some(corpus), Some(bin)) = (
        std::env::var_os("SANE_VA_CORPUS").map(std::path::PathBuf::from),
        std::env::var_os("SANE_OPENVAF_BIN").map(std::path::PathBuf::from),
    ) else {
        println!("SKIP (set SANE_VA_CORPUS and SANE_OPENVAF_BIN)");
        return;
    };
    let va = corpus.join("DIODE/diode.va");
    let osdi = std::env::temp_dir().join("sane_osdi_noise_diode.osdi");
    let status = std::process::Command::new(&bin)
        .arg(&va)
        .arg("-o")
        .arg(&osdi)
        .status()
        .expect("run openvaf-r");
    assert!(status.success(), "openvaf-r failed");

    let body = "N1 a 0 0 diode_va rs=10 is=1e-14\nV1 in 0 0.7\nR1 in a 1k\n.end\n";
    let deck_va = format!(".veriloga \"{}\"\n{body}", va.display());
    let deck_osdi = format!(".osdi \"{}\"\n{body}", osdi.display());

    let run = |deck: &str| -> Vec<f64> {
        let m = Model::from_netlist(deck).expect("model");
        m.noise(&[], "a", 1e3, 1e6, 7).expect("noise").psd
    };
    let sym = run(&deck_va);
    let os = run(&deck_osdi);
    assert_eq!(sym.len(), os.len());
    for (k, (a, b)) in sym.iter().zip(&os).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6 * b.abs() + 1e-30,
            "noise point {k}: symbolic {a} vs osdi {b}"
        );
    }
    println!(
        "osdi noise OK: {} points, e.g. {:.3e} V/rtHz at 1 kHz",
        os.len(),
        os[0]
    );
}

/// The SPICE model-card idiom over an OSDI module: `.model <name> <module> ...`
/// + `N... <name>` must route to the compiled module with the card's
/// parameters bound (same path a Verilog-A card takes).
#[test]
#[ignore]
fn osdi_deck_model_card_idiom() {
    let (Some(corpus), Some(bin)) = (
        std::env::var_os("SANE_VA_CORPUS").map(std::path::PathBuf::from),
        std::env::var_os("SANE_OPENVAF_BIN").map(std::path::PathBuf::from),
    ) else {
        println!("SKIP (set SANE_VA_CORPUS and SANE_OPENVAF_BIN)");
        return;
    };
    let out = std::env::temp_dir().join("sane_osdi_deck_psp103.osdi");
    let status = std::process::Command::new(&bin)
        .arg(corpus.join("PSP103/psp103.va"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("run openvaf-r");
    assert!(status.success(), "openvaf-r failed");

    // A p-type card (type=-1) through the card idiom vs the direct-module
    // instance with the inline parameter: identical DC by construction.
    let card = format!(
        ".osdi \"{}\"\n.model pmos_card PSP103VA type=-1\nN1 d g 0 0 pmos_card\nVd d 0 -1.0\nVg g 0 -0.8\n.end\n",
        out.display()
    );
    let direct = format!(
        ".osdi \"{}\"\nN1 d g 0 0 PSP103VA type=-1\nVd d 0 -1.0\nVg g 0 -0.8\n.end\n",
        out.display()
    );
    let id_of = |deck: &str| {
        let m = Model::from_netlist(deck).expect("model");
        m.operating_point(&[])
            .expect("dc")
            .get("Vd")
            .expect("drain current")
    };
    let (a, b) = (id_of(&card), id_of(&direct));
    assert!(a.is_finite() && a.abs() > 1e-15, "pmos conducts: {a}");
    assert!(
        (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
        "card {a} vs direct {b}"
    );
    println!("osdi model-card idiom OK: I(Vd) = {a}");
}
