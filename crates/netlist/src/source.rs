//! Independent-source value parsing: DC values, source-function (`SIN`,
//! `PULSE`, `PWL`, ...) recognition and parameter aliasing, and the SPICE
//! engineering-notation number parser.

use rustc_hash::FxHashMap as HashMap;

use sane_mna::SourceFn;

use crate::expr::resolve_value;

/// Whether a value string references a node voltage / branch current, so the
/// element is behavioral (a voltage-dependent resistor) rather than a constant.
pub(crate) fn value_is_behavioral(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    l.contains("v(") || l.contains("i(")
}

/// The DC operating-point value of an independent source: a leading bare
/// number, or the value following a `DC` keyword. The `AC <mag>` small-signal
/// magnitude is skipped (it is not a DC value), as are transient functions.
pub(crate) fn dc_value(toks: &[&str], env: &HashMap<String, f64>) -> Option<f64> {
    let mut dc = None;
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        let tl = t.to_ascii_lowercase();
        if t.eq_ignore_ascii_case("dc") {
            dc = dc.or_else(|| toks.get(i + 1).and_then(|s| resolve_value(s, env)));
            i += 2;
        } else if let Some(rest) = tl.strip_prefix("dc=") {
            // `DC=<value>` keyword form (HSPICE/ngspice testbenches).
            dc = dc.or_else(|| resolve_value(&t[t.len() - rest.len()..], env));
            i += 1;
        } else if t.eq_ignore_ascii_case("ac") {
            i += 2; // skip the AC magnitude
        } else if tl.starts_with("ac=") {
            i += 1; // `AC=<mag>` keyword form -- the AC magnitude is not a DC value
        } else if is_source_fn(t) {
            break;
        } else {
            if dc.is_none() {
                if let Some(v) = resolve_value(t, env) {
                    dc = Some(v);
                }
            }
            i += 1;
        }
    }
    dc
}

/// Is this token a transient source function (`SIN(...)`, `PULSE(...)`, ...)?
pub(crate) fn is_source_fn(tok: &str) -> bool {
    let l = tok.to_ascii_lowercase();
    ["sin(", "pulse(", "exp(", "pwl("]
        .iter()
        .any(|f| l.starts_with(f))
}

/// Is this token the bare (unparenthesised) ngspice form of a source-function
/// keyword (`... dc 0 sin 0 1 1k`)?
pub(crate) fn is_bare_source_kw(tok: &str) -> bool {
    matches!(
        tok.to_ascii_lowercase().as_str(),
        "sin" | "pulse" | "exp" | "pwl"
    )
}

/// Parse a transient source-function spec, bind its parameters as
/// instance-scoped symbols, and return the [`SourceFn`] shape. Missing trailing
/// parameters are simply left unbound (the symbol stays free).
pub(crate) fn parse_source_fn(
    name: &str,
    spec: &str,
    env: &HashMap<String, f64>,
    values: &mut HashMap<String, f64>,
) -> Option<SourceFn> {
    let (func, rest) = spec.split_once('(')?;
    let inner = rest.trim_end_matches(')');
    // Keep a hole (`None`) for an unresolvable argument rather than compacting the
    // vector, so a single unknown value cannot shift every later positional
    // binding (e.g. `PWL(0 0 {undef} 1)` must not bind `1` to the third slot).
    let args: Vec<Option<f64>> = inner
        .split_whitespace()
        .map(|a| resolve_value(a, env))
        .collect();
    // The source type owns its parameter convention (suffix names, the SIN 2*pi
    // transform): bind the positional args to instance-scoped symbols here.
    let src = SourceFn::from_func(func, args.len())?;
    for (suffix, v) in src.bind_params(&args) {
        values.insert(format!("{name}.{suffix}"), v);
    }
    Some(src)
}

/// Map common SPICE model-parameter names to SANE's device symbol suffixes,
/// so real `.model` cards bind numerically. Unknown keys are kept verbatim.
pub(crate) fn alias_param(key: &str) -> String {
    match key.to_ascii_lowercase().as_str() {
        "is" => "Is",
        "n" => "N",
        "vt" => "Vt",
        // SPICE-canonical junction-capacitance spelling is `CJO` (letter O);
        // the model reads `Cj0` (digit zero). Accept both.
        "cjo" | "cj0" => "Cj0",
        "vj" | "pb" => "Vj",
        "bf" => "betaF",
        "br" => "betaR",
        "vaf" | "va" => "VAf",
        "var" => "VAr",
        "ikf" => "IKF",
        "ikr" => "IKR",
        "ise" => "ISE",
        "isc" => "ISC",
        "ne" => "NE",
        "nc" => "NC",
        "rb" => "Rb",
        "rc" => "Rc",
        "re" => "Re",
        // SPICE's threshold/pinch-off `VTO` is the canonical name for every FET
        // model (MOSFET, JFET, MESFET, EKV all read `Vto`); `Vth` is accepted as
        // an alias. (Previously aliased to `Vth`, which silently dropped the deck
        // value for J/Z/EKV, whose models read `Vto`.)
        "vto" | "vth" => "Vto",
        "kp" => "Kp",
        "b" => "B",
        "w" => "W",
        "l" => "L",
        "lambda" => "lambda",
        "ron" => "Ron",
        "roff" => "Roff",
        _ => return key.to_string(),
    }
    .to_string()
}

/// Parse a SPICE engineering value, e.g. `1k`, `2.2u`, `1Meg`, `1e3`, `4.7nF`.
///
/// Trailing unit text after the multiplier is ignored (`1kOhm` -> 1000).
pub fn parse_value(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let bytes = s.as_bytes();
    let mut split = bytes.len();
    let mut seen_exp = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_digit() || c == '.' {
            i += 1;
        } else if (c == '+' || c == '-') && i == 0 {
            i += 1;
        } else if (c == 'e' || c == 'E') && !seen_exp {
            // Treat as exponent only if followed by an optional sign and a digit.
            let mut j = i + 1;
            if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] as char).is_ascii_digit() {
                seen_exp = true;
                i = j;
            } else {
                split = i;
                break;
            }
        } else {
            split = i;
            break;
        }
    }
    let num: f64 = s[..split].parse().ok()?;
    let suffix = s[split..].to_ascii_lowercase();
    let mult = if suffix.starts_with("meg") {
        1e6
    } else {
        match suffix.chars().next() {
            Some('t') => 1e12,
            Some('g') => 1e9,
            Some('k') => 1e3,
            Some('m') => 1e-3,
            Some('u') => 1e-6,
            Some('n') => 1e-9,
            Some('p') => 1e-12,
            Some('f') => 1e-15,
            _ => 1.0,
        }
    };
    Some(num * mult)
}
