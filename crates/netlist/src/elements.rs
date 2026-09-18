//! Element placement: one method per SPICE element letter over the parser's
//! shared state, called from the per-line dispatch of `parse_impl`.

use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use sane_device::{CSwitch, DeviceInstance};
use sane_mna::{BKind, Circuit};
use sane_veriloga::builtin_module;

use crate::behavioral::parse_bexpr;
#[cfg(not(target_arch = "wasm32"))]
use crate::devices::place_osdi_device;
use crate::devices::{instance_mfactor, place_device, place_va_device};
use crate::expr::{resolve_value, value_expr};
use crate::model_cards::ModelLib;
use crate::options::NetlistOptions;
use crate::source::{
    dc_value, is_bare_source_kw, is_source_fn, parse_source_fn, value_is_behavioral,
};
use crate::{err_at, CompatReport, NodeMap, ParseError, PortDef};

/// One element line: the name, its type letter, the tokens and the source
/// position for diagnostics.
pub(crate) struct Elem<'t> {
    pub name: &'t str,
    /// The name past the last `.` (subcircuit flattening prefixes the path).
    pub base: &'t str,
    pub kind: char,
    pub tok: &'t [&'t str],
    pub line_no: usize,
    pub line_col: usize,
}

impl Elem<'_> {
    /// At least `n` tokens on the line.
    pub fn need(&self, n: usize) -> Result<(), ParseError> {
        if self.tok.len() < n {
            Err(err_at(
                self.line_no,
                self.line_col,
                &format!("expected at least {} tokens", n),
            ))
        } else {
            Ok(())
        }
    }
}

/// The parser's state while elements are placed: the circuit under
/// construction, the placed devices, the bound values, and the read-only
/// tables (parameters, model cards, options, Verilog-A modules) the placement
/// consults.
pub(crate) struct Placer<'a> {
    pub nodes: NodeMap,
    pub circuit: Circuit,
    pub devices: Vec<DeviceInstance>,
    pub values: HashMap<String, f64>,
    /// Verilog-A modules already trial-lowered by `validate()`: the check
    /// surfaces unsupported constructs in the module source, a module-level
    /// property, so it runs once per module rather than once per instance.
    pub validated: HashSet<String>,
    pub report: CompatReport,
    pub ports: Vec<PortDef>,
    pub params: &'a HashMap<String, f64>,
    pub models: &'a ModelLib,
    pub model_aliases: &'a HashMap<u32, String>,
    pub options: &'a NetlistOptions,
    pub va_models: &'a HashMap<String, Arc<sane_veriloga::ElaboratedModule>>,
    /// OSDI compiled modules (`.osdi "lib.osdi"`), by lowercased module name.
    #[cfg(not(target_arch = "wasm32"))]
    pub osdi_models: HashMap<String, (Arc<sane_osdi::OsdiLib>, Arc<sane_osdi::OsdiModule>)>,
}

impl Placer<'_> {
    pub(crate) fn place_passive(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, tok, line_no, line_col) = (el.name, el.tok, el.line_no, el.line_col);
        el.need(3)?;
        let a = self.nodes.resolve(tok[1]);
        let b = self.nodes.resolve(tok[2]);
        // Behavioral resistor: a value that references a node voltage /
        // branch current (a foundry voltage-dependent resistor, e.g.
        // `R n1 n2 r={rbody*(1+vc1*abs(V(n1,n2)))}`) is the linear
        // element's nonlinear generalisation. Its branch current is
        // `I(n1,n2) = V(n1,n2)/value`, so lower it onto the existing
        // behavioral-source machinery rather than a constant resistance.
        if el.kind == 'R' && tok[3..].iter().any(|t| value_is_behavioral(t)) {
            let raw = tok[3..].join(" ");
            let expr = value_expr(&raw);
            let bstr = format!("(V({},{}))/({})", tok[1], tok[2], expr);
            let mut resolve = |n: &str| self.nodes.resolve(n);
            let bexpr =
                parse_bexpr(&bstr, &mut resolve).map_err(|m| err_at(line_no, line_col, &m))?;
            self.circuit.behavioral_source(name, a, b, BKind::I, bexpr);
            return Ok(());
        }
        match el.kind {
            'R' => self.circuit.resistor(name, a, b),
            'C' => self.circuit.capacitor(name, a, b),
            'L' => self.circuit.inductor(name, a, b),
            'V' => self.circuit.voltage_source(name, a, b),
            'I' => self.circuit.current_source(name, a, b),
            _ => unreachable!(),
        };
        // Element value. For sources, the DC operating-point value is a
        // leading bare number or the token after `DC`; the `AC`
        // magnitude is NOT a DC value. For R/C/L the value is positional
        // or in the `r=`/`c=`/`l=` keyword form.
        let val = if matches!(el.kind, 'V' | 'I') {
            dc_value(&tok[3..], self.params)
        } else {
            tok[3..].iter().find_map(|s| {
                resolve_value(s, self.params).or_else(|| resolve_value(value_expr(s), self.params))
            })
        };
        if let Some(v) = val {
            self.values.insert(sane_mna::value_symbol_name(name), v);
        }
        // Transient source functions on independent sources. Both the
        // parenthesised form (`SIN(0 1 1k)`, one token after grouping)
        // and the bare ngspice form (`sin 0 1 1k`, keyword followed by
        // positional numbers) are accepted; the bare form is joined
        // into the parenthesised spec before parsing.
        if matches!(el.kind, 'V' | 'I') {
            let spec: Option<String> = tok[3..]
                .iter()
                .find(|t| is_source_fn(t))
                .map(|s| s.to_string())
                .or_else(|| {
                    tok[3..].iter().position(|t| is_bare_source_kw(t)).map(|i| {
                        let kw = tok[3 + i];
                        let args: Vec<&str> = tok[3 + i + 1..]
                            .iter()
                            .take_while(|t| !t.contains('='))
                            .copied()
                            .collect();
                        format!("{kw}({})", args.join(" "))
                    })
                });
            if let Some(spec) = spec {
                if let Some(src) = parse_source_fn(name, &spec, self.params, &mut self.values) {
                    self.circuit.set_source(src);
                }
            }
        }
        Ok(())
    }
    pub(crate) fn place_controlled_voltage(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, tok) = (el.name, el.tok);
        el.need(5)?;
        let a = self.nodes.resolve(tok[1]);
        let b = self.nodes.resolve(tok[2]);
        let cp = self.nodes.resolve(tok[3]);
        let cm = self.nodes.resolve(tok[4]);
        match el.kind {
            'E' => self.circuit.vcvs(name, a, b, cp, cm),
            'G' => self.circuit.vccs(name, a, b, cp, cm),
            _ => unreachable!(),
        };
        if let Some(v) = tok[5..].iter().find_map(|s| resolve_value(s, self.params)) {
            self.values.insert(sane_mna::value_symbol_name(name), v);
        }
        Ok(())
    }
    pub(crate) fn place_ideal_line(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, base, tok, line_no, line_col) =
            (el.name, el.base, el.tok, el.line_no, el.line_col);
        // Ideal (lossless) transmission line, exact via the method of
        // characteristics (Branin) and the transient delay engine:
        //   T<name> p1 n1 p2 n2 Z0=<ohms> TD=<seconds>
        // Z0 defaults to 50. Modelled by the built-in Verilog-A module
        // (two absdelay'd outgoing waves). An explicit N=<segments>
        // requests the lumped RLGC-ladder approximation instead (next
        // arm), which flows through analyses without delay support.
        el.need(5)?;
        let a = self.nodes.resolve(tok[1]);
        let b = self.nodes.resolve(tok[2]);
        let cp = self.nodes.resolve(tok[3]);
        let cm = self.nodes.resolve(tok[4]);
        let td = tok[5..].iter().find_map(|s| {
            let (k, vs) = s.split_once('=')?;
            k.eq_ignore_ascii_case("td")
                .then(|| resolve_value(vs, self.params))
                .flatten()
        });
        let td =
            td.ok_or_else(|| err_at(line_no, line_col, "transmission line needs TD=<delay>"))?;
        if !(td > 0.0) {
            return Err(err_at(
                line_no,
                line_col,
                "transmission line TD must be positive",
            ));
        }
        let em = builtin_module("sane_tline").expect("builtin tline");
        place_va_device(
            &mut self.devices,
            &mut self.values,
            &mut self.report,
            &mut self.validated,
            name,
            base,
            &em,
            vec![a, b, cp, cm],
            &tok[5..],
            None,
            self.params,
            None,
            None,
            false,
            line_no,
            line_col,
        )?;
        Ok(())
    }
    pub(crate) fn place_vswitch(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, base, tok, line_no, line_col) =
            (el.name, el.base, el.tok, el.line_no, el.line_col);
        // Voltage-controlled switch: S name n+ n- nc+ nc- [model].
        // Built-in Verilog-A model (smooth log-conductance transition).
        el.need(5)?;
        let a = self.nodes.resolve(tok[1]);
        let b = self.nodes.resolve(tok[2]);
        let cp = self.nodes.resolve(tok[3]);
        let cm = self.nodes.resolve(tok[4]);
        let model_tok = tok[5..].iter().find(|t| !t.contains('=')).copied();
        let card = model_tok.and_then(|m| self.models.select(m, None, None));
        let modelname = model_tok.unwrap_or(base);
        let em = builtin_module("sane_vswitch").expect("builtin vswitch");
        place_va_device(
            &mut self.devices,
            &mut self.values,
            &mut self.report,
            &mut self.validated,
            name,
            modelname,
            &em,
            vec![a, b, cp, cm],
            &tok[5..],
            card,
            self.params,
            None,
            None,
            false,
            line_no,
            line_col,
        )?;
        Ok(())
    }
    pub(crate) fn place_cswitch(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, tok) = (el.name, el.tok);
        // Current-controlled switch: W name n+ n- Vctrl [model].
        el.need(4)?;
        let a = self.nodes.resolve(tok[1]);
        let b = self.nodes.resolve(tok[2]);
        let ctrl = tok[3];
        let card = tok[4..]
            .iter()
            .find(|t| !t.contains('='))
            .and_then(|m| self.models.select(m, None, None));
        let mf = instance_mfactor(&tok, self.params, el.kind == 'M');
        place_device(
            &mut self.devices,
            &mut self.values,
            &mut self.report,
            name,
            Box::new(CSwitch::new(name, ctrl)),
            vec![a, b],
            &tok[4..],
            card,
            self.params,
            mf,
        );
        Ok(())
    }
    pub(crate) fn place_port(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, tok, line_no, line_col) = (el.name, el.tok, el.line_no, el.line_col);
        // Power port: P<name> n+ n- [Z0=<ohms>]. Lowers to the Thevenin
        // form the S-parameter extraction assumes: an ideal source
        // (named like the port, so it doubles as the AC/SP drive)
        // behind a Z0 series resistor onto n+; n- is the reference.
        // The port is registered in `ParsedCircuit::self.ports`.
        el.need(3)?;
        let a = self.nodes.resolve(tok[1]);
        let b = self.nodes.resolve(tok[2]);
        let z0 = tok[3..]
            .iter()
            .find_map(|s| {
                let (k, v) = s.split_once('=')?;
                if k.eq_ignore_ascii_case("z0") {
                    resolve_value(v, self.params)
                } else {
                    None
                }
            })
            .unwrap_or(50.0);
        if !(z0 > 0.0) {
            return Err(err_at(line_no, line_col, "port Z0 must be positive"));
        }
        let t = self.nodes.resolve(&format!("{name}.t"));
        self.circuit.voltage_source(name, t, b);
        self.values.insert(sane_mna::value_symbol_name(name), 0.0);
        let rn = format!("{name}.z0");
        self.circuit.resistor(&rn, t, a);
        self.values.insert(sane_mna::value_symbol_name(&rn), z0);
        self.ports.push(PortDef {
            name: name.to_string(),
            node: tok[1].to_string(),
            z0,
        });
        Ok(())
    }
    pub(crate) fn place_coupling(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, tok) = (el.name, el.tok);
        // Mutual inductance: K name Lx Ly coupling. Args are inductor
        // element names, not self.nodes.
        el.need(4)?;
        self.circuit.mutual(name, tok[1], tok[2]);
        if let Some(v) = tok[3..].iter().find_map(|s| resolve_value(s, self.params)) {
            self.values.insert(sane_mna::value_symbol_name(name), v);
        }
        Ok(())
    }
    pub(crate) fn place_controlled_current(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, tok) = (el.name, el.tok);
        // Current-controlled sources: F/H n+ n- Vctrl gain.
        el.need(4)?;
        let a = self.nodes.resolve(tok[1]);
        let b = self.nodes.resolve(tok[2]);
        let ctrl = tok[3]; // controlling (voltage-defined) element name
        match el.kind {
            'F' => self.circuit.cccs(name, a, b, ctrl),
            'H' => self.circuit.ccvs(name, a, b, ctrl),
            _ => unreachable!(),
        };
        if let Some(v) = tok[4..].iter().find_map(|s| resolve_value(s, self.params)) {
            self.values.insert(sane_mna::value_symbol_name(name), v);
        }
        Ok(())
    }
    /// Nonlinear devices. A trailing model-card token is allowed and
    /// ignored for now (parameters stay symbolic, e.g. `D1.Is`).
    pub(crate) fn place_diode(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, base, tok, line_no, line_col) =
            (el.name, el.base, el.tok, el.line_no, el.line_col);
        // Built-in Verilog-A diode; charge storage, breakdown and the
        // series-resistance internal node fold structurally from the
        // bound parameters, so a bare card lowers to the bare junction.
        el.need(3)?;
        let a = self.nodes.resolve(tok[1]);
        let k = self.nodes.resolve(tok[2]);
        let model_tok = tok[3..].iter().find(|t| !t.contains('=')).copied();
        let card = model_tok.and_then(|m| self.models.select(m, None, None));
        let modelname = model_tok.unwrap_or(base);
        let em = builtin_module("sane_diode").expect("builtin diode");
        place_va_device(
            &mut self.devices,
            &mut self.values,
            &mut self.report,
            &mut self.validated,
            name,
            modelname,
            &em,
            vec![a, k],
            &tok[3..],
            card,
            self.params,
            None,
            None,
            false,
            line_no,
            line_col,
        )?;
        Ok(())
    }
    pub(crate) fn place_mosfet(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, base, tok, line_no, line_col) =
            (el.name, el.base, el.tok, el.line_no, el.line_col);
        // SPICE MOSFET: M name drain gate source body [model] [self.params].
        el.need(5)?;
        let d = self.nodes.resolve(tok[1]);
        let g = self.nodes.resolve(tok[2]);
        let s = self.nodes.resolve(tok[3]);
        let body = self.nodes.resolve(tok[4]);
        let extras = &tok[5..];
        // The `M` element is the built-in square-law MOSFET (levels 1-3,
        // body ignored). Compact self.models (level > 3, BSIM/EKV/PSP/...) have
        // no native closed form in SANE; they are provided as Verilog-A and
        // instantiated through the `N` element (see the error below), so a
        // compact-level `M` is rejected rather than silently mis-modelled.
        let model_name = extras.iter().find(|t| !t.contains('=')).copied();
        // Instance geometry selects the binned model card (if any). Bin
        // ranges (`lmin/lmax/wmin/wmax`) are in meters, so the drawn L/W
        // must be scaled by `.option scale` before the window comparison --
        // exactly as the geometry is scaled when bound onto the device.
        let scale = self.options.scale();
        let inst_geom = |key: &str| {
            extras
                .iter()
                .find_map(|t| {
                    let (k, v) = t.split_once('=')?;
                    k.eq_ignore_ascii_case(key)
                        .then(|| resolve_value(v, self.params))
                        .flatten()
                })
                .map(|g| g * scale)
        };
        let card = model_name.and_then(|m| self.models.select(m, inst_geom("L"), inst_geom("W")));
        let level = card.and_then(|c| {
            c.params
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("level"))
                .map(|(_, v)| *v)
        });
        // Polarity from the model type (PMOS -> P, else N-channel).
        let is_p = card.is_some_and(|c| c.mtype.eq_ignore_ascii_case("PMOS"));
        let lvl = level.unwrap_or(1.0);
        if lvl > 3.5 {
            // Compact model: no native square law. Route it to a Verilog-A
            // module if one is bound -- either the card's type token names
            // a loaded module directly (the OSDI/Spectre idiom), or a
            // `.model_alias level=<n> <module>` binds this level. The card
            // carries polarity in its `nmos`/`pmos` token, mapped to the
            // module's `type` parameter (+1 / -1).
            let module = card
                .map(|c| c.mtype.to_ascii_lowercase())
                .filter(|t| self.va_models.contains_key(t))
                .or_else(|| self.model_aliases.get(&(lvl.round() as u32)).cloned());
            if let Some(em) = module.as_deref().and_then(|m| self.va_models.get(m)) {
                let modelname = model_name.unwrap_or(base);
                let terminals = vec![d, g, s, body];
                if terminals.len() != em.ports.len() {
                    return Err(err_at(line_no, line_col, &format!(
                        "compact MOSFET '{name}' routed to Verilog-A module '{}' with {} self.ports, \
                         but `M` provides 4 terminals (d g s b)",
                        em.name, em.ports.len())));
                }
                let sign = if card.is_some_and(|c| c.mtype.eq_ignore_ascii_case("pmos")) {
                    -1.0
                } else {
                    1.0
                };
                place_va_device(
                    &mut self.devices,
                    &mut self.values,
                    &mut self.report,
                    &mut self.validated,
                    name,
                    modelname,
                    em,
                    terminals,
                    extras,
                    card,
                    self.params,
                    Some(sign),
                    Some(self.options.scale()),
                    false,
                    line_no,
                    line_col,
                )?;
                return Ok(());
            }
            let mname = model_name.unwrap_or("?");
            return Err(err_at(
                line_no,
                line_col,
                &format!(
                    "MOSFET '{name}' model '{mname}' is a compact model (level {lvl}); SANE has \
                     no native compact MOSFET (only the square-law levels 1-3). Provide it as \
                     Verilog-A: load the module with `.veriloga \"<file>.va\"` and bind this \
                     level with `.model_alias level={} {mname}`, or use the `N` element with a \
                     `.model {mname} <va-module> ...` card.",
                    lvl.round() as u32
                ),
            ));
        }
        // Level 1-3 square law (level 2/3 effects fold into the self.params):
        // the built-in Verilog-A module, polarity via its `type` param.
        let sign = if is_p { -1.0 } else { 1.0 };
        let em = builtin_module("sane_mos").expect("builtin mosfet");
        place_va_device(
            &mut self.devices,
            &mut self.values,
            &mut self.report,
            &mut self.validated,
            name,
            model_name.unwrap_or(base),
            &em,
            vec![d, g, s, body],
            extras,
            card,
            self.params,
            Some(sign),
            Some(self.options.scale()),
            true,
            line_no,
            line_col,
        )?;
        Ok(())
    }
    pub(crate) fn place_bjt(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, base, tok, line_no, line_col) =
            (el.name, el.base, el.tok, el.line_no, el.line_col);
        // Q name collector base emitter [model] [area].
        el.need(4)?;
        let c = self.nodes.resolve(tok[1]);
        let b = self.nodes.resolve(tok[2]);
        let e = self.nodes.resolve(tok[3]);
        let extras = &tok[4..];
        // Model name = first token that is neither `key=val` nor a bare
        // number (the area multiplier is a bare number after the model).
        let card = extras
            .iter()
            .find(|t| !t.contains('=') && resolve_value(t, self.params).is_none())
            .and_then(|m| self.models.select(m, None, None));
        let is_p = card.is_some_and(|c| c.mtype.eq_ignore_ascii_case("PNP"));
        // Built-in Verilog-A Gummel-Poon BJT; parasitic series
        // resistances and charge storage fold structurally from the
        // bound parameters. Polarity via the module's `type` param.
        let sign = if is_p { -1.0 } else { 1.0 };
        let modelname = extras
            .iter()
            .find(|t| !t.contains('=') && resolve_value(t, self.params).is_none())
            .copied()
            .unwrap_or(base);
        let em = builtin_module("sane_bjt").expect("builtin bjt");
        place_va_device(
            &mut self.devices,
            &mut self.values,
            &mut self.report,
            &mut self.validated,
            name,
            modelname,
            &em,
            vec![c, b, e],
            extras,
            card,
            self.params,
            Some(sign),
            None,
            false,
            line_no,
            line_col,
        )?;
        // Area multiplier (bare number after the model) = N parallel
        // devices: saturation/leakage currents and knee currents scale
        // by area, series resistances by 1/area (after defaults, so it
        // scales defaulted self.values too).
        if let Some(area) = extras
            .iter()
            .filter(|t| !t.contains('='))
            .find_map(|t| resolve_value(t, self.params))
        {
            // A parameter the deck left at its default is bound here
            // (scaled), since defaults are not expanded per instance.
            let mul = |values: &mut HashMap<String, f64>, key: &str, f: f64| {
                let full = format!("{name}.{key}");
                let base = values
                    .get(&full)
                    .copied()
                    .or_else(|| em.default_map.get(key).copied());
                if let Some(v) = base {
                    values.insert(full, v * f);
                }
            };
            for k in ["Is", "IKF", "IKR", "ISE", "ISC"] {
                mul(&mut self.values, k, area);
            }
            for k in ["Rb", "Rc", "Re"] {
                mul(&mut self.values, k, 1.0 / area);
            }
        }
        Ok(())
    }
    pub(crate) fn place_jfet(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, base, tok, line_no, line_col) =
            (el.name, el.base, el.tok, el.line_no, el.line_col);
        // JFET: J name drain gate source [model]. Polarity from the model
        // type (PJF -> P, else N-channel).
        el.need(4)?;
        let d = self.nodes.resolve(tok[1]);
        let g = self.nodes.resolve(tok[2]);
        let s = self.nodes.resolve(tok[3]);
        let extras = &tok[4..];
        let card = extras
            .iter()
            .find(|t| !t.contains('='))
            .and_then(|m| self.models.select(m, None, None));
        let is_p = card.is_some_and(|c| c.mtype.eq_ignore_ascii_case("PJF"));
        let sign = if is_p { -1.0 } else { 1.0 };
        let modelname = extras
            .iter()
            .find(|t| !t.contains('='))
            .copied()
            .unwrap_or(base);
        let em = builtin_module("sane_jfet").expect("builtin jfet");
        place_va_device(
            &mut self.devices,
            &mut self.values,
            &mut self.report,
            &mut self.validated,
            name,
            modelname,
            &em,
            vec![d, g, s],
            extras,
            card,
            self.params,
            Some(sign),
            None,
            false,
            line_no,
            line_col,
        )?;
        Ok(())
    }
    pub(crate) fn place_behavioral(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, tok, line_no, line_col) = (el.name, el.tok, el.line_no, el.line_col);
        // Behavioral source: B name n+ n- V=<expr> | I=<expr>. The
        // expression spans the rest of the line (may contain spaces).
        el.need(4)?;
        let np = self.nodes.resolve(tok[1]);
        let nm = self.nodes.resolve(tok[2]);
        let rest: String = tok[3..].join(" ");
        let (bkind, expr_str) = match rest.chars().next() {
            Some('V' | 'v') if rest[1..].starts_with('=') => (BKind::V, &rest[2..]),
            Some('I' | 'i') if rest[1..].starts_with('=') => (BKind::I, &rest[2..]),
            _ => {
                return Err(err_at(
                    line_no,
                    line_col,
                    "B source needs V=<expr> or I=<expr>",
                ))
            }
        };
        let mut resolve = |n: &str| self.nodes.resolve(n);
        let bexpr =
            parse_bexpr(expr_str, &mut resolve).map_err(|m| err_at(line_no, line_col, &m))?;
        self.circuit.behavioral_source(name, np, nm, bkind, bexpr);
        Ok(())
    }
    pub(crate) fn place_zener(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, base, tok, line_no, line_col) =
            (el.name, el.base, el.tok, el.line_no, el.line_col);
        // MESFET: Z name drain gate source [model]. PMF -> P-channel.
        el.need(4)?;
        let d = self.nodes.resolve(tok[1]);
        let g = self.nodes.resolve(tok[2]);
        let s = self.nodes.resolve(tok[3]);
        let extras = &tok[4..];
        let card = extras
            .iter()
            .find(|t| !t.contains('='))
            .and_then(|m| self.models.select(m, None, None));
        let is_p = card.is_some_and(|c| c.mtype.eq_ignore_ascii_case("PMF"));
        let sign = if is_p { -1.0 } else { 1.0 };
        let modelname = extras
            .iter()
            .find(|t| !t.contains('='))
            .copied()
            .unwrap_or(base);
        let em = builtin_module("sane_mesfet").expect("builtin mesfet");
        place_va_device(
            &mut self.devices,
            &mut self.values,
            &mut self.report,
            &mut self.validated,
            name,
            modelname,
            &em,
            vec![d, g, s],
            extras,
            card,
            self.params,
            Some(sign),
            None,
            false,
            line_no,
            line_col,
        )?;
        Ok(())
    }
    pub(crate) fn place_osdi(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, tok, line_no, line_col) = (el.name, el.tok, el.line_no, el.line_col);
        // Verilog-A instance: `N<name> node... MODULENAME [p=v ...]`.
        let bare: Vec<&str> = tok[1..]
            .iter()
            .take_while(|t| !t.contains('='))
            .copied()
            .collect();
        if bare.len() < 2 {
            return Err(err_at(
                line_no,
                line_col,
                "veriloga instance needs node(s) and a model name",
            ));
        }
        let modelname = bare[bare.len() - 1];
        let node_toks = &bare[..bare.len() - 1];
        // The instance either names a Verilog-A module directly, or it
        // names a `.model` card whose type is a Verilog-A module (the
        // SPICE idiom: `.model n1 bsim4va ...` + `NM1 d g s b n1 ...`).
        let mkey = modelname.to_ascii_lowercase();
        let mcard = self.models.select(&mkey, None, None);
        // OSDI compiled module? Either named directly, or through a
        // `.model` card whose type is an OSDI module (the same SPICE
        // idiom as the Verilog-A path below).
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((lib, module)) = self
            .osdi_models
            .get(&mkey)
            .or_else(|| mcard.and_then(|c| self.osdi_models.get(&c.mtype.to_ascii_lowercase())))
        {
            if node_toks.len() != module.num_terminals {
                return Err(err_at(
                    line_no,
                    line_col,
                    &format!(
                        "osdi model '{modelname}' has {} terminals, {} connections given",
                        module.num_terminals,
                        node_toks.len()
                    ),
                ));
            }
            let terminals: Vec<usize> = node_toks.iter().map(|t| self.nodes.resolve(t)).collect();
            place_osdi_device(
                &mut self.devices,
                &mut self.report,
                name,
                modelname,
                terminals,
                lib.clone(),
                module.clone(),
                &tok[1 + node_toks.len()..],
                mcard,
                self.params,
                self.options.temp_c(),
                line_no,
                line_col,
            )?;
            return Ok(());
        }
        let em = if let Some(em) = self.va_models.get(&mkey) {
            em
        } else if let Some(card) = mcard {
            self.va_models
                .get(&card.mtype.to_ascii_lowercase())
                .ok_or_else(|| {
                    err_at(
                        line_no,
                        line_col,
                        &format!(
                            "model '{modelname}' has type '{}', which is not a Verilog-A module",
                            card.mtype
                        ),
                    )
                })?
        } else {
            return Err(err_at(
                line_no,
                line_col,
                &format!("unknown veriloga model '{modelname}'"),
            ));
        };
        if node_toks.len() != em.ports.len() {
            return Err(err_at(
                line_no,
                line_col,
                &format!(
                    "veriloga model '{modelname}' has {} self.ports, {} connections given",
                    em.ports.len(),
                    node_toks.len()
                ),
            ));
        }
        let terminals: Vec<usize> = node_toks.iter().map(|t| self.nodes.resolve(t)).collect();
        place_va_device(
            &mut self.devices,
            &mut self.values,
            &mut self.report,
            &mut self.validated,
            name,
            modelname,
            em,
            terminals,
            &tok[1 + node_toks.len()..],
            mcard,
            self.params,
            None,
            None,
            false,
            line_no,
            line_col,
        )?;
        Ok(())
    }
    pub(crate) fn place_lossy_line(&mut self, el: &Elem<'_>) -> Result<(), ParseError> {
        let (name, tok) = (el.name, el.tok);
        // Transmission line as an N-segment RLGC ladder ('T' reaches
        // this arm only with an explicit N=; without it the exact
        // delay-line arm above applies). An exact line is a delay
        // e^{-sTD}: transcendental in s (so it does not fit the
        // rational small-signal matrix A(s) = G + sC). The lumped
        // ladder is pure R/L/C, so it flows through every analysis
        // (DC/AC/transient/HB) unchanged; accuracy grows with the
        // segment count N (keep ~10-20 segments per wavelength of
        // interest). This is exactly how SPICE realizes the `U` (URC) line.
        //   T name n1 n2 n3 n4 [Z0= TD= | R= L= G= C= | ...] [N=]
        // Port 1 is (n1,n2), port 2 is (n3,n4); n2/n4 are the common
        // series/shunt reference (tie them externally for a 2-wire line).
        el.need(5)?;
        let n_in = self.nodes.resolve(tok[1]);
        let n_out = self.nodes.resolve(tok[3]);
        let n_ref = self.nodes.resolve(tok[2]); // shunt/series reference = n2
        let _n4 = self.nodes.resolve(tok[4]);
        let kv = |key: &str| {
            tok[5..].iter().find_map(|t| {
                let (k, v) = t.split_once('=')?;
                if k.eq_ignore_ascii_case(key) {
                    resolve_value(v, self.params)
                } else {
                    None
                }
            })
        };
        let nseg = kv("N").map(|v| (v as usize).clamp(1, 4096)).unwrap_or(16);
        // Total series (R, L) and shunt (G, C) of the whole line.
        let (r_tot, l_tot, g_tot, c_tot) = match el.kind {
            'T' => {
                // Lossless: L = Z0*TD, C = TD/Z0.
                let z0 = kv("Z0").unwrap_or(50.0);
                let td = kv("TD").unwrap_or(1e-9);
                (0.0, z0 * td, 0.0, td / z0)
            }
            'U' => {
                // Distributed RC (URC): series R, shunt C totals.
                (kv("R").unwrap_or(1.0), 0.0, 0.0, kv("C").unwrap_or(1e-12))
            }
            _ => {
                // Lossy (LTRA-like): R, L, G, C given as line totals.
                (
                    kv("R").unwrap_or(0.0),
                    kv("L").unwrap_or(0.0),
                    kv("G").unwrap_or(0.0),
                    kv("C").unwrap_or(0.0),
                )
            }
        };
        let (rs, ls) = (r_tot / nseg as f64, l_tot / nseg as f64);
        let (gs, cs) = (g_tot / nseg as f64, c_tot / nseg as f64);
        // L-section ladder: series (R then L) into each internal node,
        // shunt (C, and 1/G) from it to the reference.
        let mut prev = n_in;
        for k in 0..nseg {
            let node = if k + 1 == nseg {
                n_out
            } else {
                self.nodes.resolve(&format!("{name}.x{k}"))
            };
            // Series branch prev -> node: R and/or L in series. A mid node
            // is only needed when BOTH are present; with neither, a tiny R
            // keeps the ladder regular (a plain wire would short the self.ports).
            let rn = format!("{name}.R{k}");
            let ln = format!("{name}.L{k}");
            match (rs != 0.0, ls != 0.0) {
                (true, true) => {
                    let mid = self.nodes.resolve(&format!("{name}.r{k}"));
                    self.circuit.resistor(&rn, prev, mid);
                    self.values.insert(sane_mna::value_symbol_name(&rn), rs);
                    self.circuit.inductor(&ln, mid, node);
                    self.values.insert(sane_mna::value_symbol_name(&ln), ls);
                }
                (true, false) => {
                    self.circuit.resistor(&rn, prev, node);
                    self.values.insert(sane_mna::value_symbol_name(&rn), rs);
                }
                (false, true) => {
                    self.circuit.inductor(&ln, prev, node);
                    self.values.insert(sane_mna::value_symbol_name(&ln), ls);
                }
                (false, false) => {
                    self.circuit.resistor(&rn, prev, node);
                    self.values.insert(sane_mna::value_symbol_name(&rn), 1e-9);
                }
            }
            if cs != 0.0 {
                let cn = format!("{name}.C{k}");
                self.circuit.capacitor(&cn, node, n_ref);
                self.values.insert(sane_mna::value_symbol_name(&cn), cs);
            }
            if gs != 0.0 {
                let gn = format!("{name}.G{k}");
                self.circuit.resistor(&gn, node, n_ref);
                self.values
                    .insert(sane_mna::value_symbol_name(&gn), 1.0 / gs);
            }
            prev = node;
        }
        Ok(())
    }
}
