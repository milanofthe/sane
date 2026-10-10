//! A netlist's DAE as an rsdag module (JSON): the graph with every device
//! body as a function and the system as the function `circuit` (parameters
//! with their roles: states, time, circuit parameters; the currents, charges
//! and guards as outputs), the circuit parameters' values by name and the
//! operating point. rsdag's module benchmarks read it.
//!
//!   cargo run --release -p sane-analysis --example export_module -- <deck.cir> <out.json>

use std::collections::BTreeMap;

use rsdag::{Field, Module, ParamRole, F64};
use sane_analysis::Model;
use sane_core::constants::{TEMP_NOMINAL_C, TEMP_SYMBOL, ZERO_CELSIUS_K};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(deck), Some(out)) = (args.next(), args.next()) else {
        eprintln!("usage: export_module <deck.cir> <out.json>");
        std::process::exit(2);
    };
    let src = std::fs::read_to_string(&deck).expect("read the deck");
    let model = Model::from_netlist(&src).expect("build the model");
    let op = model
        .at(&[])
        .and_then(|pt| pt.operating_point())
        .expect("operating point");
    let p = model.pvec(&[]);
    let arc = model.context_arc();
    let mut ctx = arc.lock().unwrap();
    let mut values: BTreeMap<String, f64> =
        model.cdc().param_names(&ctx).into_iter().zip(p).collect();
    let circuit = model.dae().register_function(&mut ctx, "circuit");
    // A parameter the solve does not vary (no device reads it as a DC
    // parameter) takes its module default, the temperature its nominal.
    let func = ctx.func(circuit);
    for (&s, role) in func.params().iter().zip(func.param_roles()) {
        if *role != ParamRole::Param {
            continue;
        }
        let name = ctx.symbol_name(s).to_string();
        if values.contains_key(&name) {
            continue;
        }
        let v = match model.dae().param_defaults.get(&s) {
            Some(&d) => d,
            None if name == TEMP_SYMBOL => TEMP_NOMINAL_C + ZERO_CELSIUS_K,
            None => panic!("no value for the parameter {name}"),
        };
        values.insert(name, v);
    }
    // The constants as the doubles every execution type starts from.
    let module: Module<F64> = ctx.to_module().map_consts(|c| F64::new(c.to_f64()));
    let json = serde_json::json!({
        "module": module,
        "circuit": circuit.0,
        "params": values,
        "x0": op.vector(),
    });
    std::fs::write(&out, serde_json::to_string(&json).unwrap()).expect("write the module");
    println!(
        "{deck}: {} unknowns, {} nodes, {} functions -> {out}",
        model.dim(),
        module_len(&json),
        ctx.n_funcs()
    );
}

fn module_len(json: &serde_json::Value) -> usize {
    json["module"]["nodes"].as_array().map_or(0, Vec::len)
}
