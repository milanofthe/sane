//! Device-instance placement: native models and Verilog-A modules, parameter
//! binding from model cards + inline overrides, geometry scaling, multiplicity.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::Arc;

use sane_device::{DeviceInstance, DeviceModel};
use sane_veriloga::device::VerilogADevice;
use sane_veriloga::ElaboratedModule;

use crate::expr::resolve_value;
use crate::model_cards::ModelCard;
use crate::source::alias_param;
use crate::{err_at, CompatReport, ParseError};

/// Fill any parameter the deck left unspecified with the device model's default
/// (e.g. junction thermal voltage, default beta, MOSFET W/L). Explicit values
/// already in `values` win.
pub(crate) fn apply_defaults(
    name: &str,
    model: &dyn DeviceModel,
    values: &mut HashMap<String, f64>,
) {
    for (k, v) in model.default_params() {
        values.entry(format!("{name}.{k}")).or_insert(v);
    }
}

/// Parallel multiplicity of a device instance from its tokens: `m=` / `mult=`
/// times, for MOSFETs, the finger count `nf` (which multiplies the device the
/// same way SPICE does). Bare model/node tokens (no `=`) are ignored.
pub(crate) fn instance_mfactor(toks: &[&str], env: &HashMap<String, f64>, with_nf: bool) -> f64 {
    let kv = |key: &str| {
        toks.iter().find_map(|t| {
            let (k, v) = t.split_once('=')?;
            k.eq_ignore_ascii_case(key)
                .then(|| resolve_value(v, env))
                .flatten()
        })
    };
    let m = kv("m").or_else(|| kv("mult")).unwrap_or(1.0);
    let nf = if with_nf {
        kv("nf").unwrap_or(1.0)
    } else {
        1.0
    };
    m * nf
}

/// Place a nonlinear device: push the instance (with its parallel multiplicity),
/// then bind its `.model`-card and inline parameters and fill model defaults.
/// Shared by every device arm (D/M/Q/J/Z and the switches) so the four-step
/// push/bind/defaults/mfactor sequence -- and its `.last().unwrap()` foot-guns --
/// live in one place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn place_device(
    devices: &mut Vec<DeviceInstance>,
    values: &mut HashMap<String, f64>,
    report: &mut CompatReport,
    name: &str,
    model: Box<dyn DeviceModel>,
    terminals: Vec<usize>,
    extras: &[&str],
    card: Option<&ModelCard>,
    params: &HashMap<String, f64>,
    mfactor: f64,
) {
    devices.push(DeviceInstance::new(model, terminals).with_mfactor(mfactor));
    let m = devices.last().unwrap().model.as_ref();
    bind_device_params(name, extras, card, params, values, m, report);
    let m = devices.last().unwrap().model.as_ref();
    apply_defaults(name, m, values);
}

/// Build and register a Verilog-A device instance from an elaborated module.
///
/// Shared by the built-in device elements (`D`/`M`/`Q`/`J`/`Z`/`S`/`T`, whose
/// models ship as Verilog-A in `sane_veriloga::builtin`), the `N` element
/// (direct VA instantiation) and the `M` element when its `.model` card is a
/// compact model (`level > 3`) routed to a VA module via `.model_alias`.
/// `inst_kv` are the instance `key=value` tokens (geometry, `m=`, ...).
/// `type_override`, when set, forces the module's `type` parameter from the
/// card's polarity token (`nmos`/`npn`/... -> +1, `pmos`/`pnp`/... -> -1),
/// since a SPICE `.model` card carries polarity in its type token, not a
/// param. `nf_scales_mfactor` folds the SPICE MOSFET finger count `nf=` into
/// the parallel multiplicity (the square-law `M` element idiom; compact
/// modules model `nf` internally and keep it a parameter).
#[allow(clippy::too_many_arguments)]
pub(crate) fn place_va_device(
    devices: &mut Vec<DeviceInstance>,
    values: &mut HashMap<String, f64>,
    report: &mut CompatReport,
    validated: &mut HashSet<String>,
    name: &str,
    modelname: &str,
    em: &Arc<ElaboratedModule>,
    terminals: Vec<usize>,
    inst_kv: &[&str],
    mcard: Option<&ModelCard>,
    params: &HashMap<String, f64>,
    type_override: Option<f64>,
    geom_scale: Option<f64>,
    nf_scales_mfactor: bool,
    line_no: usize,
    line_col: usize,
) -> Result<(), ParseError> {
    // Bind the `.model` card then the inline instance `key=value` params, using a
    // probe device for its `default_params`; the resolved instance values let
    // structural decisions (switch branches) fold at lowering.
    let mut extras: Vec<&str> = vec![modelname];
    extras.extend_from_slice(inst_kv);
    let probe = VerilogADevice::new(name, em.clone());
    bind_device_params(name, &extras, mcard, params, values, &probe, report);
    // MOS polarity from the card's nmos/pmos token -> module `type` parameter.
    if let Some(t) = type_override {
        if em
            .params
            .iter()
            .any(|p| p.name.eq_ignore_ascii_case("type"))
        {
            values.insert(format!("{name}.type"), t);
        }
    }
    // `.option scale` converts drawn geometry to the meters the model expects.
    // Applied here, before the parameter snapshot, using the module's own L/W
    // parameter names (compact-model VA modules do not apply scale internally).
    if let Some(scale) = geom_scale.filter(|&s| s != 1.0) {
        for key in ["L", "W"] {
            if let Some(p) = em.params.iter().find(|p| p.name.eq_ignore_ascii_case(key)) {
                if let Some(v) = values.get_mut(&format!("{name}.{}", p.name)) {
                    *v *= scale;
                }
            }
        }
    }
    // Parameters explicitly set (card or instance) -> the `$param_given` set
    // and the instance's value map. Defaults are NOT expanded per instance:
    // the lowering fills them from the module, and a model's parameter store
    // asks the device (`DeviceModel::param_default`) for an unbound symbol.
    // A compact model has hundreds of parameters; expanding them per instance
    // was the largest memory and time item of a 10k-transistor deck.
    let given: HashSet<String> = em
        .params
        .iter()
        .filter(|p| values.contains_key(&format!("{name}.{}", p.name)))
        .map(|p| p.name.clone())
        .collect();
    let pvals: HashMap<String, f64> = em
        .params
        .iter()
        .filter_map(|p| {
            values
                .get(&format!("{name}.{}", p.name))
                .map(|v| (p.name.clone(), *v))
        })
        .collect();
    // Enforce the module's declared parameter ranges (`from`/`exclude`): a value
    // outside its valid range is a likely mis-entered card, so surface it loudly
    // instead of silently producing a bad model.
    // Capture (not just log) so the Python layer re-raises these as catchable
    // SaneConvergenceWarnings regardless of the log level -- an out-of-range
    // device parameter is a likely mis-entered card the user must not miss (#54).
    for v in em.check_param_ranges(&pvals) {
        sane_core::log::warn_captured(&format!("{name} ({modelname}): {v}"));
    }
    // Parallel multiplicity (instance `m=` / `mult=`, or a module-declared
    // `aliasparam <name> = $mfactor`), default 1 -- a built-in instance
    // property, not a module parameter.
    let is_mfactor_key = |k: &str| {
        k.eq_ignore_ascii_case("m")
            || k.eq_ignore_ascii_case("mult")
            || em
                .aliases
                .iter()
                .any(|(a, t)| t == "$mfactor" && k.eq_ignore_ascii_case(a))
    };
    let mut mfactor = inst_kv
        .iter()
        .find_map(|t| {
            let (k, v) = t.split_once('=')?;
            if is_mfactor_key(k) {
                resolve_value(v, params)
            } else {
                None
            }
        })
        .unwrap_or(1.0);
    if nf_scales_mfactor {
        mfactor *= inst_kv
            .iter()
            .find_map(|t| {
                let (k, v) = t.split_once('=')?;
                k.eq_ignore_ascii_case("nf")
                    .then(|| resolve_value(v, params))
                    .flatten()
            })
            .unwrap_or(1.0);
    }
    let mut dev = VerilogADevice::with_instance(name, em.clone(), pvals, given);
    dev.mfactor = mfactor;
    // Surface any unsupported construct up front (logged + parse error) rather
    // than producing unexpected results during analysis. Done once per module
    // (see `validated`): the trial lowering is a module-level property and
    // re-running it per instance dominates parse time for large compact models.
    if validated.insert(em.name.clone()) {
        if let Err(e) = dev.validate() {
            return Err(err_at(
                line_no,
                line_col,
                &format!("veriloga model '{modelname}': {e}"),
            ));
        }
    }
    devices.push(DeviceInstance::new(Box::new(dev), terminals));
    Ok(())
}

/// Instance modifiers that are not device-model parameters (so they are not
/// flagged as "unknown" when they appear on an element line).
const INSTANCE_MODIFIERS: &[&str] = &["m", "mult", "nf"];

/// Bind a device's parameter values: from its referenced `.model` card, then
/// any inline `key=value` tokens (which override the model). Keys land as
/// `"{name}.{key}"` to match the instance-scoped device symbols.
pub(crate) fn bind_device_params(
    name: &str,
    extras: &[&str],
    card: Option<&ModelCard>,
    env: &HashMap<String, f64>,
    values: &mut HashMap<String, f64>,
    model: &dyn DeviceModel,
    report: &mut CompatReport,
) {
    // Resolve each deck key to the model's exact-case parameter name: first
    // the semantic SPICE alias (`BF` -> `betaF`, `VTO` -> `Vto`, ...), then the
    // model's own case-insensitive lookup (which also covers its `aliasparam`
    // aliases; a `$`-target alias such as `m = $mfactor` is a valid key that
    // binds no parameter -- the instance-modifier handling reads it from the
    // raw tokens). This keeps binding robust to the case a deck happens to use
    // instead of silently dropping the value to the model default.
    let resolve = |k: &str| -> String {
        let aliased = alias_param(k);
        model
            .canonical_param(&aliased)
            .map(str::to_string)
            .unwrap_or(aliased)
    };
    // A key the model recognises (after aliasing); unknown ones are dropped and
    // worth reporting (typo or unsupported parameter).
    let known =
        |k: &str| model.canonical_param(&alias_param(k)).is_some() || model.is_sysfn_alias(k);
    let is_modifier = |k: &str| INSTANCE_MODIFIERS.iter().any(|m| k.eq_ignore_ascii_case(m));

    if let Some(card) = card {
        for (k, v) in &card.params {
            if !known(k) {
                report.unknown_param(name, k);
            }
            values.insert(format!("{name}.{}", resolve(k)), *v);
        }
    }
    for (k, v) in extras
        .iter()
        .filter_map(|t| t.split_once('='))
        .filter_map(|(k, v)| resolve_value(v, env).map(|val| (k, val)))
    {
        if !known(k) && !is_modifier(k) {
            report.unknown_param(name, k);
        }
        values.insert(format!("{name}.{}", resolve(k)), v);
    }
}

/// Build and register an OSDI compiled-model instance. Parameters come from
/// the `.model` card (if any) and inline `key=value` tokens, bound numerically
/// (OSDI parameters are baked at load time, not symbolic). `m=`/`mult=` maps
/// to the OSDI `$mfactor` builtin when the model exposes it.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn place_osdi_device(
    devices: &mut Vec<DeviceInstance>,
    report: &mut CompatReport,
    name: &str,
    _modelname: &str,
    terminals: Vec<usize>,
    lib: std::sync::Arc<sane_osdi::OsdiLib>,
    module: std::sync::Arc<sane_osdi::OsdiModule>,
    inst_kv: &[&str],
    mcard: Option<&ModelCard>,
    env: &HashMap<String, f64>,
    temp_c: f64,
    line_no: usize,
    line_col: usize,
) -> Result<(), ParseError> {
    let mut params: HashMap<String, f64> = HashMap::default();
    if let Some(card) = mcard {
        for (k, v) in &card.params {
            if module.has_param(k) {
                params.insert(k.clone(), *v);
            } else {
                report.unknown_param(name, k);
            }
        }
    }
    for (k, v) in inst_kv
        .iter()
        .filter_map(|t| t.split_once('='))
        .filter_map(|(k, v)| resolve_value(v, env).map(|val| (k, val)))
    {
        let key = if k.eq_ignore_ascii_case("m") || k.eq_ignore_ascii_case("mult") {
            "$mfactor"
        } else {
            k
        };
        if module.has_param(key) {
            params.insert(key.to_string(), v);
        } else if !key.starts_with('$') {
            report.unknown_param(name, k);
        }
    }
    let temperature = temp_c + 273.15;
    let dev = sane_osdi::OsdiDevice::new(name, lib, module, params, temperature);
    // Set up eagerly so parameter/collapse errors surface at parse time.
    dev.setup().map_err(|e| err_at(line_no, line_col, &e))?;
    devices.push(DeviceInstance::new(Box::new(dev), terminals));
    Ok(())
}
