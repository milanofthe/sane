//! SPICE-like netlist parser.
//!
//! Lines are `Name node+ node- [more nodes] [value]`. The leading letter of
//! the name selects the element type (R/C/L/V/I/E/G). Node names are mapped to
//! integers with ground (`0`, `gnd`, `GND`, `ground`) fixed at index 0. The
//! element name is its symbolic parameter; an optional trailing value (with the
//! usual engineering suffixes) is captured for later numeric evaluation.
//!
//! ```text
//! * RC lowpass
//! V1 in 0 1
//! R1 in out 1k
//! C1 out 0 1u
//! .end
//! ```

use rustc_hash::FxHashMap as HashMap;
use std::path::Path;
use std::sync::Arc;

use sane_circuit::Circuit;

mod behavioral;
mod compat;
// Device-instance placement and parameter binding.
mod devices;
mod elements;
use elements::{Elem, Placer};
mod expr;
// `.model` card parsing and the binned model library.
mod model_cards;
// `.temp` / `.option` / `.param` collection and resolution.
mod options;
mod preprocess;
mod resolve;
// Independent-source value / source-function parsing.
mod source;
mod subckt;
// Verilog-A block extraction and module registration.
#[cfg(test)]
mod tests;
mod va;

use model_cards::parse_model_cards;
use options::{collect_model_aliases, collect_options, is_handled_directive, resolve_params};
pub use source::parse_value;
use va::extract_veriloga;

pub use compat::CompatReport;
use preprocess::preprocess;
use preprocess::Line;
use subckt::{hierarchy, Inst, Item};

/// A parse failure located in the deck.
///
/// Carries the 1-based `line` and `col` of the offending token. A `col` of 0
/// means the column is unknown (line-level error); [`ParseError::render`] then
/// omits the caret. Use `render` with the original deck text for a Verilog-A
/// style `file:line:col` header with a source line and caret.
#[derive(Debug, PartialEq)]
pub struct ParseError {
    pub line: usize,
    pub col: usize,
    pub msg: String,
}

impl ParseError {
    /// Render this error against the deck `source` as a `netlist:line:col:
    /// error: message` header followed, when the column is known, by the
    /// offending source line and a caret. Mirrors the Verilog-A frontend's
    /// diagnostics so both input languages read the same.
    pub fn render(&self, source: &str) -> String {
        sane_core::diag::render_one(
            source,
            "netlist",
            self.line as u32,
            self.col as u32,
            &self.msg,
        )
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.col == 0 {
            write!(f, "line {}: {}", self.line, self.msg)
        } else {
            write!(f, "line {}:{}: {}", self.line, self.col, self.msg)
        }
    }
}

/// Parse a netlist into a [`Circuit`]. Any failure is logged once here (the
/// single boundary, via `sane_core::log::error`) and returned as a typed
/// `ParseError`, so every parse error surfaces uniformly through the logging
/// system as well as the return value.
pub fn parse(text: &str) -> Result<Circuit, ParseError> {
    parse_with_base(text, None)
}

/// Like [`parse`], but resolves relative `.include` / `.lib` paths in the deck
/// against `base_dir` (typically the directory of the netlist file). `None`
/// uses the process working directory.
pub fn parse_with_base(text: &str, base_dir: Option<&Path>) -> Result<Circuit, ParseError> {
    parse_report(text, base_dir).map(|(c, _)| c)
}

/// [`parse_with_base`], with the report of what the deck states that the
/// parser does not take (see [`CompatReport`]).
pub fn parse_report(
    text: &str,
    base_dir: Option<&Path>,
) -> Result<(Circuit, CompatReport), ParseError> {
    parse_impl(text, base_dir).inspect_err(|e| {
        sane_core::log::error(&format!("netlist parse: line {}: {}", e.line, e.msg));
    })
}

fn parse_impl(text: &str, base_dir: Option<&Path>) -> Result<(Circuit, CompatReport), ParseError> {
    // Splice in `.include` / `.lib` references first (they carry whole
    // `.model` / `.subckt` / `.veriloga` blocks), then pull out inline
    // `.veriloga ... .endveriloga` blocks and compile them.
    let text = resolve::resolve_includes(text, base_dir)?;
    let (text, va_models) = extract_veriloga(&text)?;
    let lines = preprocess(&text);
    // Global simulation options (`.temp`, `.option scale/gmin/temp`). Collected
    // up front so the temperature feeds both the `.param` `temp` constant and
    // the global `$temp` symbol, and `scale` reaches geometric device binding.
    let options = collect_options(&lines);
    // Compact-model `level` -> Verilog-A module bindings (`.model_alias`).
    let model_aliases = collect_model_aliases(&lines);
    // Resolve top-level `.param` first, expand the `.subckt`/`X` hierarchy,
    // then collect `.model` cards over the top level and every body.
    let params = resolve_params(&lines, options.temp_c());
    let (items, body_models) = hierarchy(&lines, &params)?;
    let card_lines: Vec<Line> = items
        .iter()
        .filter_map(|it| match it {
            Item::Line(l) => Some(l.clone()),
            Item::Inst(_) => None,
        })
        .chain(body_models)
        .collect();
    let models = parse_model_cards(&card_lines, &params);

    let mut st = Placer {
        circuit: Circuit::new(),
        report: CompatReport::default(),
        params: &params,
        models: &models,
        model_aliases: &model_aliases,
        options: &options,
        va_models: &va_models,
        #[cfg(not(target_arch = "wasm32"))]
        osdi_models: HashMap::default(),
        bodies: HashMap::default(),
    };

    st.place_items(&items, base_dir)?;
    let Placer {
        mut circuit,
        mut report,
        ..
    } = st;

    // The shared global temperature symbol `$temp` [K] feeds both Verilog-A
    // `$temperature`/`$vt` and the native device temperature model (thermal
    // voltage, Is(T), mobility). A `.temp` / `.option temp=` directive sets it
    // (Celsius -> Kelvin); otherwise it defaults to nominal (27 degC). A sweep
    // can still override it downstream. Always bound (not only for VA), since
    // native junction devices read it; it is an unused parameter for a purely
    // linear deck.
    circuit.values.insert(
        sane_core::constants::TEMP_SYMBOL.to_string(),
        options.temp_kelvin(),
    );

    // `.option gmin=` is recorded but not yet applied by the parser (the solver
    // owns gmin); note it so the user knows it was seen, not silently dropped.
    if let Some(g) = options.gmin {
        report.notes.push(format!(
            ".option gmin={g} recorded (applied by the solver, not the netlist)"
        ));
    }

    circuit.dc = options.dc;
    circuit.nodeset = options::collect_nodeset(&text);
    Ok((circuit, report))
}

/// A subcircuit body placed: what every instance of it shares, in the
/// body's names.
pub(crate) struct Placed {
    body: Arc<Circuit>,
    report: CompatReport,
}

impl Placer<'_> {
    /// Place `items` in order. Stops at `.end`.
    fn place_items(&mut self, items: &[Item], base_dir: Option<&Path>) -> Result<(), ParseError> {
        for item in items {
            match item {
                Item::Line(line) => {
                    if !self.place_line(line, base_dir)? {
                        break;
                    }
                }
                Item::Inst(inst) => self.place_instance(inst, base_dir)?,
            }
        }
        Ok(())
    }

    /// Wire a subcircuit instance into this frame (see
    /// [`Circuit::instance`]). The body is placed once, for every instance
    /// of it.
    fn place_instance(&mut self, inst: &Inst, base_dir: Option<&Path>) -> Result<(), ParseError> {
        let key = Arc::as_ptr(&inst.body);
        let placed = match self.bodies.get(&key) {
            Some(placed) => placed.clone(),
            None => {
                let placed = std::rc::Rc::new(self.place_body(&inst.body, base_dir)?);
                self.bodies.insert(key, placed.clone());
                placed
            }
        };
        let conn: Vec<&str> = inst.conn.iter().map(|c| c.as_str()).collect();
        let instance =
            (self.circuit.instance(&inst.name, &placed.body, &conn)).map_err(|m| err(0, &m))?;
        let unknown: Vec<String> = (placed.report.unknown_params.iter())
            .map(|p| instance.rename(p))
            .collect();
        let report = &placed.report;
        self.report.unknown_params.extend(unknown);
        self.report
            .ignored_directives
            .extend(report.ignored_directives.iter().cloned());
        self.report.notes.extend(report.notes.iter().cloned());
        Ok(())
    }

    /// Place a subcircuit body over its own nodes and names.
    fn place_body(
        &mut self,
        body: &subckt::Body,
        base_dir: Option<&Path>,
    ) -> Result<Placed, ParseError> {
        let mut circuit = Circuit::new();
        circuit.ns = body.ns.clone();
        circuit.pins = body.ports.clone();
        let mut placer = Placer {
            circuit,
            report: CompatReport::default(),
            params: self.params,
            models: self.models,
            model_aliases: self.model_aliases,
            options: self.options,
            va_models: self.va_models,
            #[cfg(not(target_arch = "wasm32"))]
            osdi_models: std::mem::take(&mut self.osdi_models),
            bodies: std::mem::take(&mut self.bodies),
        };
        let placed = placer.place_items(&body.items, base_dir);
        self.bodies = std::mem::take(&mut placer.bodies);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.osdi_models = std::mem::take(&mut placer.osdi_models);
        }
        placed?;
        Ok(Placed {
            body: Arc::new(placer.circuit),
            report: placer.report,
        })
    }

    /// Place one element or directive line. `false` at `.end`.
    fn place_line(&mut self, line: &Line, base_dir: Option<&Path>) -> Result<bool, ParseError> {
        let line_no = line.no;
        let line_col = line.col;
        let tok: Vec<&str> = line.tokens.iter().map(|s| s.as_str()).collect();
        let head = tok[0];
        if head.starts_with('.') {
            if head.eq_ignore_ascii_case(".end") {
                return Ok(false);
            }
            if head.eq_ignore_ascii_case(".osdi") {
                // `.osdi "file.osdi"`: load an OpenVAF-compiled shared library
                // and register its modules for `N` instances.
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let raw = tok
                        .get(1)
                        .map(|t| t.trim_matches('"'))
                        .filter(|t| !t.is_empty())
                        .ok_or_else(|| err_at(line_no, line_col, ".osdi needs a library path"))?;
                    let mut path = std::path::PathBuf::from(raw);
                    if path.is_relative() {
                        if let Some(base) = base_dir {
                            path = base.join(path);
                        }
                    }
                    let lib = sane_osdi::OsdiLib::load(&path)
                        .map_err(|e| err_at(line_no, line_col, &e))?;
                    for module in lib.modules() {
                        self.osdi_models.insert(
                            module.name.to_ascii_lowercase(),
                            (lib.clone(), module.clone()),
                        );
                    }
                    return Ok(true);
                }
                #[cfg(target_arch = "wasm32")]
                let _ = base_dir;
                #[cfg(target_arch = "wasm32")]
                return Err(err_at(
                    line_no,
                    line_col,
                    ".osdi compiled models are not available in the browser build",
                ));
            }
            // Directives handled in a pre-pass (or stripped earlier) are honoured
            // elsewhere; anything else is genuinely skipped by this parser, so
            // record it for the compatibility report instead of dropping it
            // silently.
            if !is_handled_directive(head) {
                self.report
                    .ignored_directives
                    .insert(head.to_ascii_lowercase());
            }
            return Ok(true);
        }
        let name = head;
        // The type letter is the first char of the base name; a subcircuit
        // body's names carry its namespace (`__inv__.R1`), so look past the
        // last `.`.
        let base = name.rsplit('.').next().unwrap_or(name);
        let kind = base
            .chars()
            .next()
            .map(|c| c.to_ascii_uppercase())
            .ok_or_else(|| err_at(line_no, line_col, "empty element name"))?;
        let el = Elem {
            name,
            base,
            kind,
            tok: &tok,
            line_no,
            line_col,
        };
        match kind {
            'R' | 'C' | 'L' | 'V' | 'I' => self.place_passive(&el)?,
            'E' | 'G' => self.place_controlled_voltage(&el)?,
            'T' if !el.tok[5..].iter().any(|s| {
                s.split_once('=')
                    .is_some_and(|(k, _)| k.eq_ignore_ascii_case("n"))
            }) =>
            {
                self.place_ideal_line(&el)?
            }
            'S' => self.place_vswitch(&el)?,
            'W' => self.place_cswitch(&el)?,
            'P' => self.place_port(&el)?,
            'K' => self.place_coupling(&el)?,
            'F' | 'H' => self.place_controlled_current(&el)?,
            'D' => self.place_diode(&el)?,
            'M' => self.place_mosfet(&el)?,
            'Q' => self.place_bjt(&el)?,
            'J' => self.place_jfet(&el)?,
            'B' => self.place_behavioral(&el)?,
            'Z' => self.place_zener(&el)?,
            'N' => self.place_osdi(&el)?,
            'T' | 'O' | 'U' => self.place_lossy_line(&el)?,
            other => {
                return Err(err_at(
                    line_no,
                    line_col,
                    &format!("unsupported element type '{}'", other),
                ));
            }
        }
        Ok(true)
    }
}

pub(crate) fn err(line: usize, msg: &str) -> ParseError {
    ParseError {
        line,
        col: 0,
        msg: msg.to_string(),
    }
}

pub(crate) fn err_at(line: usize, col: usize, msg: &str) -> ParseError {
    ParseError {
        line,
        col,
        msg: msg.to_string(),
    }
}
