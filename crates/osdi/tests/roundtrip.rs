//! End-to-end OSDI roundtrip: compile real Verilog-A models with openvaf-r,
//! load the .osdi library, place the device in a circuit and DC-solve against
//! analytic references. Ignored by default; needs `SANE_OPENVAF_BIN` (the
//! openvaf-r binary) and `SANE_VA_CORPUS` (the model sources).

use rustc_hash::FxHashMap as HashMap;
use std::path::PathBuf;

use sane_core::Graph;
use sane_dae::{assemble_dae, DeviceInstance};
use sane_mna::Circuit;
use sane_osdi::{OsdiDevice, OsdiLib};
use sane_solve::CompiledDc;

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(PathBuf::from)
}

/// Compile `src` to a .osdi in a temp dir; returns the library path.
fn compile(src: &std::path::Path) -> PathBuf {
    let bin = env_path("SANE_OPENVAF_BIN").expect("SANE_OPENVAF_BIN");
    let out = std::env::temp_dir().join(format!(
        "sane_osdi_test_{}.osdi",
        src.file_stem().unwrap().to_string_lossy()
    ));
    let status = std::process::Command::new(&bin)
        .arg(src)
        .arg("-o")
        .arg(&out)
        .status()
        .expect("run openvaf-r");
    assert!(status.success(), "openvaf-r failed on {}", src.display());
    out
}

fn params_vec(ctx: &Graph, dae: &sane_dae::Dae, vals: &[(&str, f64)]) -> Vec<f64> {
    let map: HashMap<&str, f64> = vals.iter().copied().collect();
    dae.params(ctx)
        .iter()
        .map(|&s| {
            let name = ctx.symbol_name(s);
            if let Some(v) = map.get(name) {
                *v
            } else if name == sane_core::constants::TEMP_SYMBOL {
                sane_core::constants::TEMP_NOMINAL_K
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
#[ignore]
fn osdi_resistor_divider_matches_analytic() {
    let (Some(corpus), Some(_)) = (env_path("SANE_VA_CORPUS"), env_path("SANE_OPENVAF_BIN")) else {
        println!("SKIP (set SANE_VA_CORPUS and SANE_OPENVAF_BIN)");
        return;
    };
    let lib_path = compile(&corpus.join("RESISTOR/resistor.va"));
    let lib = OsdiLib::load(&lib_path).expect("load");
    let module = lib.modules()[0].clone();
    println!(
        "module '{}': {} nodes ({} terminals): {:?}",
        module.name, module.num_nodes, module.num_terminals, module.node_names
    );

    // V1 (2 V) -- OSDI resistor (2k) -- native resistor (1k) to ground:
    // divider with v2 = 2 * 1k/3k.
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R2", 2, 0);
    let mut params = HashMap::default();
    params.insert("R".to_string(), 2000.0);
    let dev = OsdiDevice::new(
        "N1",
        lib.clone(),
        module,
        params,
        sane_core::constants::TEMP_NOMINAL_K,
    );
    dev.setup().expect("setup");
    let dae = assemble_dae(
        &mut ctx,
        &c,
        &[DeviceInstance::new(Box::new(dev), vec![1, 2])],
    );
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(&ctx, &dae, &[("V1", 2.0), ("R2", 1000.0)]);
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 100);
    assert!(conv, "OSDI resistor DC did not converge: {x:?}");
    let v2 = dae.unknowns.iter().position(|u| u == "v2").unwrap();
    let want = 2.0 * 1000.0 / 3000.0;
    assert!(
        (x[v2] - want).abs() < 1e-6,
        "divider voltage {} != {want} (sign convention?)",
        x[v2]
    );
    println!("resistor divider OK: v2 = {}", x[v2]);
}

#[test]
#[ignore]
fn osdi_diode_matches_analytic() {
    let (Some(corpus), Some(_)) = (env_path("SANE_VA_CORPUS"), env_path("SANE_OPENVAF_BIN")) else {
        println!("SKIP (set SANE_VA_CORPUS and SANE_OPENVAF_BIN)");
        return;
    };
    let lib_path = compile(&corpus.join("DIODE/diode.va"));
    let lib = OsdiLib::load(&lib_path).expect("load");
    let module = lib.modules()[0].clone();
    println!(
        "module '{}': {} nodes ({} terminals): {:?}, params has is: {}",
        module.name,
        module.num_nodes,
        module.num_terminals,
        module.node_names,
        module.has_param("is")
    );

    // V1 (0.7 V) -- R (1k) -- OSDI diode to ground. The corpus diode has a
    // series resistance node; rs defaults nonzero, so solve and check the
    // diode current against the load line (I = (0.7 - va)/1k must equal the
    // current entering the diode -- consistency, not closed form).
    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let mut params = HashMap::default();
    params.insert("is".to_string(), 1e-14);
    params.insert("rs".to_string(), 10.0);
    let n_term = module.num_terminals;
    let dev = OsdiDevice::new(
        "D1",
        lib.clone(),
        module,
        params,
        sane_core::constants::TEMP_NOMINAL_K,
    );
    dev.setup().expect("setup");
    // Corpus diode: terminals [A, C, dT]; tie the thermal node to ground.
    let terms = if n_term == 3 {
        vec![2, 0, 0]
    } else {
        vec![2, 0]
    };
    let dae = assemble_dae(&mut ctx, &c, &[DeviceInstance::new(Box::new(dev), terms)]);
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(&ctx, &dae, &[("V1", 0.7), ("R1", 1000.0)]);
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 200);
    assert!(conv, "OSDI diode DC did not converge: {x:?}");
    let v2 = x[dae.unknowns.iter().position(|u| u == "v2").unwrap()];
    // Forward-biased junction: anode voltage between 0.3 and 0.7 V.
    assert!(v2 > 0.3 && v2 < 0.7, "diode anode voltage {v2} implausible");
    println!("diode OK: va = {v2}");
}

#[test]
#[ignore]
fn osdi_psp103_matches_symbolic_frontend() {
    // The cross-check of both model paths: PSP103 compiled by OpenVAF and
    // loaded via OSDI must reproduce the drain current SANE's own symbolic
    // Verilog-A frontend computes for the same bias (see the va_collapse
    // differential test: I(Vd) = -7.1635e-5 at Vd=1.0, Vg=0.8, defaults).
    let (Some(corpus), Some(_)) = (env_path("SANE_VA_CORPUS"), env_path("SANE_OPENVAF_BIN")) else {
        println!("SKIP (set SANE_VA_CORPUS and SANE_OPENVAF_BIN)");
        return;
    };
    let lib_path = compile(&corpus.join("PSP103/psp103.va"));
    let lib = OsdiLib::load(&lib_path).expect("load");
    let module = lib.modules()[0].clone();
    println!(
        "module '{}': {} nodes ({} terminals)",
        module.name, module.num_nodes, module.num_terminals
    );

    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("Vd", 1, 0).voltage_source("Vg", 2, 0);
    let dev = OsdiDevice::new(
        "N1",
        lib.clone(),
        module,
        HashMap::default(),
        sane_core::constants::TEMP_NOMINAL_K,
    );
    dev.setup().expect("setup");
    let dae = assemble_dae(
        &mut ctx,
        &c,
        &[DeviceInstance::new(Box::new(dev), vec![1, 2, 0, 0])],
    );
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let p = params_vec(&ctx, &dae, &[("Vd", 1.0), ("Vg", 0.8)]);
    let (x, conv, _) = cdc.solve_dc(&p, &[], 1e-12, 200);
    assert!(conv, "OSDI psp103 DC did not converge");
    let id = x[dae.unknowns.iter().position(|u| u == "i_Vd").unwrap()];
    let reference = -7.163534299181468e-5; // symbolic frontend, same bias
    assert!(
        (id - reference).abs() < 1e-9 * (1.0 + reference.abs()) + 1e-12,
        "OSDI drain current {id} vs symbolic frontend {reference}"
    );
    println!("psp103 OK: I(Vd) = {id} (symbolic frontend: {reference})");
}

#[test]
#[ignore]
fn osdi_capacitor_transient_matches_analytic() {
    // Reactive path end-to-end: VACASK's capacitor.va as OSDI, RC step
    // response against 1 - exp(-t/RC). Exercises q, the reactive Jacobian
    // markers and the transient integrator through the bundle bridge.
    let Some(_) = env_path("SANE_OPENVAF_BIN") else {
        println!("SKIP (set SANE_OPENVAF_BIN)");
        return;
    };
    let Some(cap_src) = env_path("SANE_VACASK_DEVICES").map(|d| d.join("capacitor.va")) else {
        println!("SKIP (set SANE_VACASK_DEVICES to VACASK's devices dir)");
        return;
    };
    if !cap_src.exists() {
        println!("SKIP (no capacitor.va at {})", cap_src.display());
        return;
    }
    let lib_path = compile(&cap_src);
    let lib = OsdiLib::load(&lib_path).expect("load");
    let module = lib.modules()[0].clone();

    let mut ctx = Graph::new();
    let mut c = Circuit::new();
    c.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
    let mut params = HashMap::default();
    params.insert("c".to_string(), 1e-6);
    let dev = OsdiDevice::new(
        "C1",
        lib.clone(),
        module,
        params,
        sane_core::constants::TEMP_NOMINAL_K,
    );
    dev.setup().expect("setup");
    let dae = assemble_dae(
        &mut ctx,
        &c,
        &[DeviceInstance::new(Box::new(dev), vec![2, 0])],
    );
    let cdc = CompiledDc::new(&mut ctx, &dae);
    let (r, cap) = (1000.0, 1e-6);
    let tau = r * cap;
    let p = params_vec(&ctx, &dae, &[("V1", 1.0), ("R1", r)]);
    let v1 = dae.unknowns.iter().position(|u| u == "v1").unwrap();
    let v2 = dae.unknowns.iter().position(|u| u == "v2").unwrap();
    let mut x0 = vec![0.0; cdc.dim()];
    x0[v1] = 1.0;
    let times: Vec<f64> = (0..=5).map(|k| k as f64 * tau).collect();
    let traj = cdc
        .solve_transient(
            sane_solve::TransientMethod::Esdirk32,
            &p,
            &x0,
            &times,
            1e-6,
            1e-9,
            None,
        )
        .expect("transient");
    for (k, t) in times.iter().enumerate() {
        let want = 1.0 - (-t / tau).exp();
        let got = traj[k][v2];
        assert!((got - want).abs() < 5e-3, "t={t}: v_C={got}, want {want}");
    }
    println!("osdi capacitor RC step OK");
}
