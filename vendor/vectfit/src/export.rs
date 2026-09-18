//! Deterministic **Verilog-A** code generation for a fitted [`RationalModel`] admittance
//! macromodel, targeting the AD-friendly grammar subset that the SANE analog engine lowers
//! (issue #37): no `$table_model`, no mid-range conditionals, only `ddt` + constant-coefficient
//! linear contributions (so the whole device is trivially smooth for gradient/Hessian AD).
//!
//! Realisation: the common-pole form `Y(s) = D + sE + Σ_k R_k/(s − p_k)` becomes a real
//! state-space with one internal node per state. Each **real pole** → a 1st-order section; each
//! **complex-conjugate pair** → a real 2nd-order section (2×2 rotation block). States are scaled by
//! `K = 2π·fmax` so the internal node voltages are `O(1)` (well-conditioned for the analog solver)
//! while `ddt` stays physical. Port currents `I(pᵢ, gnd) <+ Σ …` realise the admittance directly
//! (natural for MNA). An explicit `gnd` terminal is emitted because a Verilog-A module references
//! only its own terminals (SANE requirement), not the netlist global reference.
//!
//! Output is deterministic given the same model + date: stable node ordering and fixed-precision
//! float formatting, so a byte-for-byte diff detects any model change.

use std::f64::consts::TAU;
use std::fmt::Write as _;

use crate::passivity::{self, PassivityReport};
use crate::ratmodel::RationalModel;

/// Fixed-precision, sign-normalised float literal (no `-0`, 12 significant digits) — the basis of
/// the deterministic, byte-stable output.
fn f(x: f64) -> String {
    let x = if x == 0.0 { 0.0 } else { x }; // normalise -0.0 → 0.0
    format!("{:.12e}", x)
}

/// A valid Verilog-A identifier from an arbitrary string: keep `[A-Za-z0-9_]`, prefix `n_` if it
/// does not start with a letter, fall back to `fallback` if empty.
fn ident(s: &str, fallback: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        return fallback.to_string();
    }
    if !out.chars().next().unwrap().is_ascii_alphabetic() {
        out = format!("n_{out}");
    }
    out
}

/// Export error: the fit is too poor to trust and `force` was not set.
#[derive(Debug)]
pub struct ExportError {
    pub fit_error: f64,
    pub threshold: f64,
    /// `true` when the rejection is the √s skin-effect term (not realisable in
    /// the ddt-only grammar), not the fit error.
    pub sqrt_term: bool,
}
impl std::fmt::Display for ExportError {
    fn fmt(&self, fo: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.sqrt_term {
            return write!(
                fo,
                "model carries a significant \u{221a}s skin-effect term, which the ddt-only \
                 Verilog-A grammar cannot realise; refit without the sqrt term (fit_auto) or \
                 pass force=true to export the rational part only (\u{221a}s contribution \
                 dropped, fit error understates the model)"
            );
        }
        write!(
            fo,
            "fit error {:.3e} exceeds export threshold {:.3e}; pass force=true to export anyway \
             (the achieved error is embedded in the model header)",
            self.fit_error, self.threshold
        )
    }
}
impl std::error::Error for ExportError {}

/// Generate the Verilog-A source for `model`. `name` is the module/geometry name, `date` a caller-
/// supplied provenance date string (kept explicit for deterministic codegen — no wall clock).
/// `threshold` gates export on the achieved relative fit error unless `force`. The passivity of the
/// model is assessed and recorded in the header/metadata.
pub fn export_verilog_a(
    model: &RationalModel,
    name: &str,
    date: &str,
    force: bool,
    threshold: f64,
) -> Result<String, ExportError> {
    if model.fit_error > threshold && !force {
        return Err(ExportError {
            fit_error: model.fit_error,
            threshold,
            sqrt_term: false,
        });
    }
    // A √s branch term has no exact ddt-only realisation. "Significant" is
    // judged against the model's own scale at the top of the fitted band,
    // where √f is largest.
    if !force {
        let wtop = TAU * model.fmax;
        let scale = model
            .d
            .iter()
            .zip(&model.sq)
            .map(|(&d, &sq)| (sq.abs() * wtop.sqrt()) / d.abs().max(1e-300))
            .fold(0.0f64, f64::max);
        let has_sqrt = model.sq.iter().any(|&v| v != 0.0) && scale > 1e-9;
        if has_sqrt {
            return Err(ExportError {
                fit_error: model.fit_error,
                threshold,
                sqrt_term: true,
            });
        }
    }
    let pass = passivity::check(model);
    Ok(render(model, name, date, threshold, &pass))
}

/// The pure, deterministic string builder (no I/O, no clock) — the unit-test seam.
fn render(
    model: &RationalModel,
    name: &str,
    date: &str,
    threshold: f64,
    pass: &PassivityReport,
) -> String {
    let p = model.n_ports;
    let modname = ident(name, "rapidmom_model");
    let k = TAU * model.fmax;
    let nr = model.real_poles.len();
    let nc = model.cpx_poles.len();

    // Port terminal identifiers p0..p{p-1} (positional; real names live in the header/metadata).
    let port = |i: usize| format!("p{i}");
    let v = |node: &str| format!("V({node} , gnd)");

    let mut s = String::new();
    // ── provenance header ────────────────────────────────────────────────────
    let _ = writeln!(
        s,
        "// ============================================================================"
    );
    let _ = writeln!(
        s,
        "//  rapidmom - Verilog-A rational macromodel (auto-generated; do not edit)"
    );
    let _ = writeln!(s, "//  geometry   : {name}");
    let _ = writeln!(s, "//  generated  : {date}");
    let names: Vec<String> = (0..p)
        .map(|i| {
            format!(
                "p{i}={}",
                model.port_names.get(i).cloned().unwrap_or_default()
            )
        })
        .collect();
    let _ = writeln!(s, "//  ports      : {p}  ({})", names.join(", "));
    let z0s: Vec<String> = model.z0.iter().map(|z| format!("{z}")).collect();
    let _ = writeln!(s, "//  z0 [Ohm]   : {}", z0s.join(", "));
    let _ = writeln!(
        s,
        "//  fit band   : {:.6e} .. {:.6e} Hz",
        model.fmin, model.fmax
    );
    let _ = writeln!(
        s,
        "//  fit error  : {:.6e} (relative, max over band; bound {:.1e})",
        model.fit_error, threshold
    );
    let _ = writeln!(
        s,
        "//  model order: {} poles ({nr} real + {nc} conj-pair)",
        model.n_poles()
    );
    let _ = writeln!(s, "//  passivity  : {}", pass.summary());
    let _ = writeln!(
        s,
        "//  form       : common-pole vector fit of Y(s); s = j*2*pi*f; K = 2*pi*fmax"
    );
    let _ = writeln!(
        s,
        "// ============================================================================"
    );
    // ── machine-readable metadata (fixed grammar; issue #37) ─────────────────
    let _ = writeln!(s, "// >>> RAPIDMOM-META");
    let meta = [
        ("schema".to_string(), "1".to_string()),
        ("model".to_string(), "rational_common_pole".to_string()),
        ("domain".to_string(), "admittance_Y".to_string()),
        ("n_ports".to_string(), p.to_string()),
        ("port_names".to_string(), model.port_names.join(";")),
        ("z0_ohm".to_string(), z0s.join(";")),
        ("freq_min_hz".to_string(), f(model.fmin)),
        ("freq_max_hz".to_string(), f(model.fmax)),
        ("fit_error_rel".to_string(), f(model.fit_error)),
        ("fit_error_bound".to_string(), f(threshold)),
        (
            "passivity".to_string(),
            if pass.passive {
                "passive".into()
            } else {
                "violated".into()
            },
        ),
        ("passivity_margin_s".to_string(), f(pass.margin)),
        ("n_real_poles".to_string(), nr.to_string()),
        ("n_cpx_pairs".to_string(), nc.to_string()),
    ];
    for (kk, vv) in &meta {
        let _ = writeln!(s, "// {kk}: {vv}");
    }
    let _ = writeln!(s, "// <<< RAPIDMOM-META");
    let _ = writeln!(s);
    let _ = writeln!(s, "`include \"disciplines.vams\"");
    let _ = writeln!(s);

    // ── module + terminals ───────────────────────────────────────────────────
    let terms: Vec<String> = (0..p)
        .map(port)
        .chain(std::iter::once("gnd".to_string()))
        .collect();
    let _ = writeln!(s, "module {modname}({});", terms.join(", "));
    let _ = writeln!(s, "    inout {};", terms.join(", "));
    let _ = writeln!(s, "    electrical {};", terms.join(", "));

    // internal state nodes
    let mut states: Vec<String> = Vec::new();
    for kk in 0..nr {
        for comp in 0..p {
            states.push(format!("xr_{kk}_{comp}"));
        }
    }
    for kk in 0..nc {
        for comp in 0..p {
            states.push(format!("xc_{kk}_{comp}"));
            states.push(format!("xd_{kk}_{comp}"));
        }
    }
    if !states.is_empty() {
        // wrap long node lists for readability, deterministically (8 per line)
        for chunk in states.chunks(8) {
            let _ = writeln!(s, "    electrical {};", chunk.join(", "));
        }
    }
    let _ = writeln!(s, "    real Kf;");
    let _ = writeln!(s);
    let _ = writeln!(s, "    analog begin");
    let _ = writeln!(s, "        Kf = {};", f(k));

    // ── state equations ──────────────────────────────────────────────────────
    // Real pole kk, component comp: ddt(X) - pr*X - K*V(p_comp) = 0.
    for kk in 0..nr {
        let pr = model.real_poles[kk];
        for comp in 0..p {
            let node = format!("xr_{kk}_{comp}");
            let _ = writeln!(
                s,
                "        I({node} , gnd) <+ ddt({}) + ({})*{} + (-Kf)*{};",
                v(&node),
                f(-pr),
                v(&node),
                v(&port(comp)),
            );
        }
    }
    // Complex pair kk (a=Re, b=Im), component comp: two coupled 1st-order equations.
    for kk in 0..nc {
        let a = model.cpx_poles[kk].re;
        let b = model.cpx_poles[kk].im;
        for comp in 0..p {
            let n1 = format!("xc_{kk}_{comp}");
            let n2 = format!("xd_{kk}_{comp}");
            // ddt(X1) - a*X1 - b*X2 - K*v = 0
            let _ = writeln!(
                s,
                "        I({n1} , gnd) <+ ddt({}) + ({})*{} + ({})*{} + (-Kf)*{};",
                v(&n1),
                f(-a),
                v(&n1),
                f(-b),
                v(&n2),
                v(&port(comp)),
            );
            // ddt(X2) + b*X1 - a*X2 = 0
            let _ = writeln!(
                s,
                "        I({n2} , gnd) <+ ddt({}) + ({})*{} + ({})*{};",
                v(&n2),
                f(b),
                v(&n1),
                f(-a),
                v(&n2),
            );
        }
    }
    let _ = writeln!(s);

    // ── port currents ────────────────────────────────────────────────────────
    // I(p_i,gnd) <+ Σ_states C*V(state) + Σ_j D_ij V(p_j) + Σ_j E_ij ddt(V(p_j)).
    for i in 0..p {
        let _ = writeln!(s, "        I({} , gnd) <+", port(i));
        let mut terms: Vec<String> = Vec::new();
        // real-pole state outputs: coeff = res_real / K  (= normalised residue)
        for kk in 0..nr {
            for comp in 0..p {
                let coeff = model.res_real[i * p + comp][kk] / k;
                if coeff != 0.0 {
                    terms.push(format!(
                        "            ({})*{}",
                        f(coeff),
                        v(&format!("xr_{kk}_{comp}"))
                    ));
                }
            }
        }
        // complex-pole state outputs: coeff1 = 2*Re(rc)/K, coeff2 = 2*Im(rc)/K
        for kk in 0..nc {
            for comp in 0..p {
                let rc = model.res_cpx[i * p + comp][kk] / k;
                terms.push(format!(
                    "            ({})*{}",
                    f(2.0 * rc.re),
                    v(&format!("xc_{kk}_{comp}"))
                ));
                terms.push(format!(
                    "            ({})*{}",
                    f(2.0 * rc.im),
                    v(&format!("xd_{kk}_{comp}"))
                ));
            }
        }
        // direct feedthrough D and proportional E
        for j in 0..p {
            let d = model.d[i * p + j];
            if d != 0.0 {
                terms.push(format!("            ({})*{}", f(d), v(&port(j))));
            }
        }
        for j in 0..p {
            let e = model.e[i * p + j];
            if e != 0.0 {
                terms.push(format!("            ({})*ddt({})", f(e), v(&port(j))));
            }
        }
        if terms.is_empty() {
            terms.push("            0.0".to_string());
        }
        let _ = writeln!(s, "{};", terms.join(" +\n"));
    }
    let _ = writeln!(s, "    end");
    let _ = writeln!(s, "endmodule");
    s
}
