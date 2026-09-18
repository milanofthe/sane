//! Global deck options and parameters: `.temp` / `.option` collection,
//! `.model` alias maps, and iterative `.param` resolution.

use rustc_hash::FxHashMap as HashMap;

use crate::expr::resolve_value;
use crate::preprocess::Line;

/// Global simulation options gathered from `.temp` / `.option` directives.
#[derive(Clone, Default)]
pub(crate) struct NetlistOptions {
    /// Operating temperature in Celsius (`.temp` or `.option temp=`).
    pub(crate) temp_c: Option<f64>,
    /// Geometric scale factor (`.option scale=`); multiplies device L/W.
    pub(crate) scale: Option<f64>,
    /// Minimum conductance hint (`.option gmin=`); surfaced in diagnostics.
    pub(crate) gmin: Option<f64>,
}

impl NetlistOptions {
    /// Operating temperature [Celsius] (nominal 27 if no directive set it).
    pub(crate) fn temp_c(&self) -> f64 {
        self.temp_c.unwrap_or(sane_core::constants::TEMP_NOMINAL_C)
    }
    /// Operating temperature [Kelvin] for the global `$temp` symbol.
    pub(crate) fn temp_kelvin(&self) -> f64 {
        self.temp_c() + sane_core::constants::ZERO_CELSIUS_K
    }
    /// Geometric scale factor (default 1.0). Consumed by geometric device
    /// binding (instance modifiers); see the device arms.
    pub(crate) fn scale(&self) -> f64 {
        self.scale.unwrap_or(1.0)
    }
}

/// Whether a directive is acted on somewhere in the pipeline (a pre-pass, or
/// stripped before tokenisation). Anything not in this set is skipped by the
/// parser and reported as ignored.
pub(crate) fn is_handled_directive(head: &str) -> bool {
    // NB: `.global` is intentionally absent -- there is no global-net
    // implementation, so it must surface in the compatibility report rather than
    // be silently swallowed as "handled".
    const HANDLED: &[&str] = &[
        ".model",
        ".model_alias",
        ".param",
        ".subckt",
        ".ends",
        ".end",
        ".temp",
        ".option",
        ".options",
        ".veriloga",
        ".endveriloga",
        ".include",
        ".inc",
        ".lib",
        ".endl",
    ];
    HANDLED.iter().any(|d| head.eq_ignore_ascii_case(d))
}

/// Scan for global `.temp` / `.option` directives (the last occurrence wins,
/// matching SPICE). `.option scale/gmin/temp` and a bare `.temp <value>` are
/// recognised; unknown options are left for the compatibility report.
pub(crate) fn collect_options(lines: &[Line]) -> NetlistOptions {
    let mut opt = NetlistOptions::default();
    let env: HashMap<String, f64> = {
        let mut e = HashMap::default();
        e.insert("pi".to_string(), std::f64::consts::PI);
        e
    };
    for line in lines {
        let head = line.tokens[0].to_ascii_lowercase();
        if head == ".temp" {
            if let Some(v) = line.tokens.get(1).and_then(|t| resolve_value(t, &env)) {
                opt.temp_c = Some(v);
            }
        } else if head == ".option" || head == ".options" {
            for t in &line.tokens[1..] {
                let Some((k, v)) = t.split_once('=') else {
                    continue;
                };
                let Some(val) = resolve_value(v, &env) else {
                    continue;
                };
                match k.to_ascii_lowercase().as_str() {
                    "scale" => opt.scale = Some(val),
                    "gmin" => opt.gmin = Some(val),
                    "temp" | "temper" => opt.temp_c = Some(val),
                    _ => {}
                }
            }
        }
    }
    opt
}

/// Collect `.model_alias level=<n> <module>` bindings: map a SPICE compact-model
/// `level` to a loaded Verilog-A module name. This is how a legacy `.model ...
/// <type> level=<n>` card (instantiated with `M`) routes to a VA module's
/// symbolic lowering -- explicit, since there is no universal "level N = model X"
/// truth (different `.va` files implement the same level differently).
pub(crate) fn collect_model_aliases(lines: &[Line]) -> HashMap<u32, String> {
    let mut map = HashMap::default();
    for line in lines {
        if !line.tokens[0].eq_ignore_ascii_case(".model_alias") {
            continue;
        }
        let mut level: Option<u32> = None;
        let mut module: Option<String> = None;
        for t in &line.tokens[1..] {
            if let Some((k, v)) = t.split_once('=') {
                if k.eq_ignore_ascii_case("level") {
                    level = v.parse::<f64>().ok().map(|x| x.round() as u32);
                }
            } else if let Ok(n) = t.parse::<f64>() {
                level = Some(n.round() as u32);
            } else {
                module = Some(t.to_ascii_lowercase());
            }
        }
        if let (Some(l), Some(m)) = (level, module) {
            map.insert(l, m);
        }
    }
    map
}

/// Resolve `.param NAME=expr` definitions into a numeric environment. Parameters
/// may reference each other / constants (`pi`, `temp`); resolved iteratively to
/// a fixpoint. `temp_c` seeds the `temp` constant (set by `.temp`/`.option`).
pub(crate) fn resolve_params(lines: &[Line], temp_c: f64) -> HashMap<String, f64> {
    let mut env: HashMap<String, f64> = HashMap::default();
    env.insert("pi".into(), std::f64::consts::PI);
    env.insert("temp".into(), temp_c);

    let mut raw: Vec<(String, String)> = Vec::new();
    let mut depth = 0i32;
    for line in lines {
        let head = &line.tokens[0];
        if head.eq_ignore_ascii_case(".subckt") {
            depth += 1;
            continue;
        }
        if head.eq_ignore_ascii_case(".ends") {
            depth -= 1;
            continue;
        }
        if depth > 0 || !head.eq_ignore_ascii_case(".param") {
            continue; // only top-level .param are global
        }
        for t in &line.tokens[1..] {
            if let Some((k, v)) = t.split_once('=') {
                raw.push((k.to_ascii_lowercase(), v.to_string()));
            }
        }
    }
    // Iterate to a fixpoint so params can depend on earlier-defined params.
    for _ in 0..=raw.len() {
        let mut progress = false;
        for (k, vexpr) in &raw {
            if env.contains_key(k) {
                continue;
            }
            if let Some(val) = resolve_value(vexpr, &env) {
                env.insert(k.clone(), val);
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }
    env
}
