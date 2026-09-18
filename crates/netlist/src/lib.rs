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

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::path::Path;

use sane_device::DeviceInstance;
use sane_mna::Circuit;

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
use subckt::flatten;

/// Result of parsing a netlist.
pub struct ParsedCircuit {
    /// The topological circuit (linear elements), ready for MNA assembly.
    pub circuit: Circuit,
    /// Nonlinear device placements (diodes, transistors).
    pub devices: Vec<DeviceInstance>,
    /// Node name -> integer index (ground is 0).
    pub node_index: HashMap<String, usize>,
    /// Integer index -> node name (index 0 is ground, `"0"`).
    pub node_names: Vec<String>,
    /// Element name -> numeric value, when a value was given.
    pub values: HashMap<String, f64>,
    /// Power ports (`P` elements) in deck order, for S-parameter extraction.
    pub ports: Vec<PortDef>,
    /// Diagnostics: directives ignored and parameters dropped (see [`CompatReport`]).
    pub report: CompatReport,
}

/// A power port (`P<name> n+ n- [Z0=..]`): the Thevenin form the S-parameter
/// extraction assumes. The element lowers to an ideal source (named like the
/// port, so it doubles as the AC drive) behind a `Z0` series resistor onto
/// `node`; the source value symbol is the port name, the resistor's is
/// `<name>.z0`.
#[derive(Debug, Clone)]
pub struct PortDef {
    /// Element name (`P1`, ...), also the drive-source name for AC/SP.
    pub name: String,
    /// Network-side terminal (the deck's `n+` token, original spelling).
    pub node: String,
    /// Reference impedance in ohms.
    pub z0: f64,
}

impl std::fmt::Debug for ParsedCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParsedCircuit")
            .field("circuit", &self.circuit)
            .field("devices", &self.devices.len())
            .field("node_names", &self.node_names)
            .field("values", &self.values)
            .finish()
    }
}

impl ParsedCircuit {
    /// Parameter value by symbol name (`R1`, `M1.W`): the deck's bound value,
    /// else the placed device's module default. `None` for a symbol neither
    /// the deck nor a device gives a value.
    pub fn param_value(&self, name: &str) -> Option<f64> {
        self.values
            .get(name)
            .copied()
            .or_else(|| sane_device::ParamDefaults::new(&self.devices).get(name))
    }

    /// The parameter vector for `names` (the engine's column order): bound
    /// values, device defaults for unstated parameters, `0.0` for anything
    /// still unbound. The one way to turn a parsed deck into a `p` vector.
    pub fn pvec(&self, names: &[String]) -> Vec<f64> {
        let defaults = sane_device::ParamDefaults::new(&self.devices);
        names
            .iter()
            .map(|n| {
                self.values
                    .get(n)
                    .copied()
                    .or_else(|| defaults.get(n))
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Look up the integer index of a node by name (case-insensitive).
    pub fn node(&self, name: &str) -> Option<usize> {
        self.node_index.get(&name.to_ascii_lowercase()).copied()
    }
}

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

struct NodeMap {
    index: HashMap<String, usize>,
    names: Vec<String>,
}

impl NodeMap {
    fn new() -> Self {
        let mut index = HashMap::default();
        // Keys are lowercased for case-insensitive node identity.
        for g in ["0", "gnd", "ground"] {
            index.insert(g.to_string(), 0);
        }
        Self {
            index,
            names: vec!["0".to_string()],
        }
    }

    fn resolve(&mut self, name: &str) -> usize {
        let key = name.to_ascii_lowercase();
        if let Some(&idx) = self.index.get(&key) {
            return idx;
        }
        let idx = self.names.len();
        self.names.push(name.to_string());
        self.index.insert(key, idx);
        idx
    }
}

/// Parse a netlist into a [`ParsedCircuit`]. Any failure is logged once here
/// (the single boundary, via `sane_core::log::error`) and returned as a typed
/// `ParseError`, so every parse error surfaces uniformly through the logging
/// system as well as the return value.
pub fn parse(text: &str) -> Result<ParsedCircuit, ParseError> {
    parse_with_base(text, None)
}

/// Like [`parse`], but resolves relative `.include` / `.lib` paths in the deck
/// against `base_dir` (typically the directory of the netlist file). `None`
/// uses the process working directory.
pub fn parse_with_base(text: &str, base_dir: Option<&Path>) -> Result<ParsedCircuit, ParseError> {
    parse_impl(text, base_dir).inspect_err(|e| {
        sane_core::log::error(&format!("netlist parse: line {}: {}", e.line, e.msg));
    })
}

fn parse_impl(text: &str, base_dir: Option<&Path>) -> Result<ParsedCircuit, ParseError> {
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
    // Resolve top-level `.param` first, flatten the `.subckt`/`X` hierarchy,
    // then collect `.model` cards over the flattened netlist.
    let params = resolve_params(&lines, options.temp_c());
    let flat = flatten(&lines, &params)?;
    let models = parse_model_cards(&flat, &params);

    let mut st = Placer {
        nodes: NodeMap::new(),
        circuit: Circuit::new(),
        devices: Vec::new(),
        values: HashMap::default(),
        validated: HashSet::default(),
        report: CompatReport::default(),
        ports: Vec::new(),
        params: &params,
        models: &models,
        model_aliases: &model_aliases,
        options: &options,
        va_models: &va_models,
        #[cfg(not(target_arch = "wasm32"))]
        osdi_models: HashMap::default(),
    };

    for line in &flat {
        let line_no = line.no;
        let line_col = line.col;
        let tok: Vec<&str> = line.tokens.iter().map(|s| s.as_str()).collect();
        let head = tok[0];
        if head.starts_with('.') {
            if head.eq_ignore_ascii_case(".end") {
                break;
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
                        st.osdi_models.insert(
                            module.name.to_ascii_lowercase(),
                            (lib.clone(), module.clone()),
                        );
                    }
                    continue;
                }
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
                st.report
                    .ignored_directives
                    .insert(head.to_ascii_lowercase());
            }
            continue;
        }
        let name = head;
        // The type letter is the first char of the base name; subcircuit
        // flattening prefixes names with the instance path (`X1.R1`), so look
        // past the last `.`.
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
            'R' | 'C' | 'L' | 'V' | 'I' => st.place_passive(&el)?,
            'E' | 'G' => st.place_controlled_voltage(&el)?,
            'T' if !el.tok[5..].iter().any(|s| {
                s.split_once('=')
                    .is_some_and(|(k, _)| k.eq_ignore_ascii_case("n"))
            }) =>
            {
                st.place_ideal_line(&el)?
            }
            'S' => st.place_vswitch(&el)?,
            'W' => st.place_cswitch(&el)?,
            'P' => st.place_port(&el)?,
            'K' => st.place_coupling(&el)?,
            'F' | 'H' => st.place_controlled_current(&el)?,
            'D' => st.place_diode(&el)?,
            'M' => st.place_mosfet(&el)?,
            'Q' => st.place_bjt(&el)?,
            'J' => st.place_jfet(&el)?,
            'B' => st.place_behavioral(&el)?,
            'Z' => st.place_zener(&el)?,
            'N' => st.place_osdi(&el)?,
            'T' | 'O' | 'U' => st.place_lossy_line(&el)?,
            other => {
                return Err(err_at(
                    line_no,
                    line_col,
                    &format!("unsupported element type '{}'", other),
                ));
            }
        }
    }

    let Placer {
        nodes,
        circuit,
        devices,
        mut values,
        mut report,
        ports,
        ..
    } = st;

    // The shared global temperature symbol `$temp` [K] feeds both Verilog-A
    // `$temperature`/`$vt` and the native device temperature model (thermal
    // voltage, Is(T), mobility). A `.temp` / `.option temp=` directive sets it
    // (Celsius -> Kelvin); otherwise it defaults to nominal (27 degC). A sweep
    // can still override it downstream. Always bound (not only for VA), since
    // native junction devices read it; it is an unused parameter for a purely
    // linear deck.
    values.insert(
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

    Ok(ParsedCircuit {
        circuit,
        devices,
        node_index: nodes.index,
        node_names: nodes.names,
        values,
        ports,
        report,
    })
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
