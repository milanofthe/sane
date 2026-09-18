//! Built-in device models, shipped as Verilog-A source.
//!
//! Verilog-A is SANE's source of truth for constitutive device equations: the
//! standard SPICE elements (diode, square-law MOSFET, Gummel-Poon BJT, JFET,
//! MESFET, switches, ideal transformer, ideal transmission line) live in
//! `crates/veriloga/builtin/*.va` and lower through exactly the same pipeline
//! as user-provided compact models -- one elaboration per module (lazy,
//! process-wide), then per-instance lowering with the template cache, instance
//! batching, `$limit` collection and noise extraction all shared.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::{Arc, OnceLock};

use crate::elaborate::{elaborate, ElaboratedModule};

/// The embedded builtin sources. Each file may define one or more modules.
const SOURCES: &[(&str, &str)] = &[
    ("diode.va", include_str!("../builtin/diode.va")),
    ("mosfet.va", include_str!("../builtin/mosfet.va")),
    ("bjt.va", include_str!("../builtin/bjt.va")),
    ("jfet.va", include_str!("../builtin/jfet.va")),
    ("mesfet.va", include_str!("../builtin/mesfet.va")),
    ("vswitch.va", include_str!("../builtin/vswitch.va")),
    ("transformer.va", include_str!("../builtin/transformer.va")),
    ("tline.va", include_str!("../builtin/tline.va")),
];

/// Look up a built-in module by its module name (`sane_diode`, `sane_mos`,
/// ...). Elaborates the whole builtin library once per process on first use;
/// the sources are compiled in, so a failure is a build defect and panics.
pub fn builtin_module(name: &str) -> Option<Arc<ElaboratedModule>> {
    static REGISTRY: OnceLock<HashMap<String, Arc<ElaboratedModule>>> = OnceLock::new();
    REGISTRY
        .get_or_init(|| {
            let mut map = HashMap::default();
            for (file, src) in SOURCES {
                let modules = crate::parse_modules(src, file, &[])
                    .unwrap_or_else(|d| panic!("builtin {file} does not parse: {d}"));
                for m in modules {
                    let em = elaborate(&m)
                        .unwrap_or_else(|d| panic!("builtin {file} does not elaborate: {d}"));
                    map.insert(em.name.clone(), Arc::new(em));
                }
            }
            map
        })
        .get(name)
        .cloned()
}

/// Construct a device instance of a built-in module by module name, with the
/// given instance parameters bound (they also count as `$param_given`). The
/// programmatic analogue of a netlist element line, used by tests, benches and
/// the Python circuit builder. Panics on an unknown module name (the builtin
/// set is fixed at compile time).
pub fn builtin_device(
    module: &str,
    inst: impl Into<String>,
    params: &[(&str, f64)],
) -> crate::device::VerilogADevice {
    let em = builtin_module(module).unwrap_or_else(|| panic!("no builtin module '{module}'"));
    let pvals: HashMap<String, f64> = params.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    let given: HashSet<String> = pvals.keys().cloned().collect();
    crate::device::VerilogADevice::with_instance(inst, em, pvals, given)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_builtins_elaborate_and_lower() {
        for name in [
            "sane_diode",
            "sane_mos",
            "sane_bjt",
            "sane_jfet",
            "sane_mesfet",
            "sane_vswitch",
            "sane_transformer",
            "sane_tline",
        ] {
            let em = builtin_module(name).unwrap_or_else(|| panic!("missing builtin {name}"));
            let dev = crate::device::VerilogADevice::new("X1", em);
            dev.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
