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

use sane_dae::Instance;
use sane_device::{DeviceInstance, ParamDefaults};
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
use preprocess::Line;
use subckt::{hierarchy, Inst, Item};

/// Result of parsing a netlist.
pub struct ParsedCircuit {
    /// The topological circuit (linear elements), ready for MNA assembly.
    pub circuit: Circuit,
    /// Nonlinear device placements (diodes, transistors).
    pub devices: Vec<DeviceInstance>,
    /// Subcircuit instances, each its body placed in the body's own names.
    pub instances: Vec<Instance>,
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
            .field("instances", &self.instances.len())
            .field("node_names", &self.node_names)
            .field("values", &self.values)
            .finish()
    }
}

impl ParsedCircuit {
    /// The circuit's DAE: the top level assembled, every subcircuit instance
    /// a call of its body's function.
    pub fn assemble(&self, ctx: &mut sane_core::Graph) -> sane_dae::Dae {
        sane_dae::assemble(ctx, &self.circuit, &self.devices, &self.instances)
    }

    /// Every placed device, subcircuit bodies' included, under its name in
    /// the top frame (see [`ParamDefaults`]).
    fn defaults(&self) -> ParamDefaults<'_> {
        fn add<'a>(d: &mut ParamDefaults<'a>, instances: &'a [Instance], outer: &dyn Fn(&str) -> String) {
            for inst in instances {
                let rename = |n: &str| outer(&inst.rename(n));
                d.add(&inst.devices, &rename);
                add(d, &inst.instances, &rename);
            }
        }
        let mut d = ParamDefaults::new(&self.devices);
        add(&mut d, &self.instances, &|n| n.to_string());
        d
    }

    /// Parameter value by symbol name (`R1`, `M1.W`): the deck's bound value,
    /// else the placed device's module default. `None` for a symbol neither
    /// the deck nor a device gives a value.
    pub fn param_value(&self, name: &str) -> Option<f64> {
        self.values
            .get(name)
            .copied()
            .or_else(|| self.defaults().get(name))
    }

    /// The name of the parameter symbol device instance `inst` reads its
    /// module parameter `param` from: the card's (`nmos.vth0`, shared by
    /// the card's instances) or its own (`M1.w`, `X1.M1.w`). `None` when no
    /// placed device has that instance name.
    pub fn param_symbol(&self, inst: &str, param: &str) -> Option<String> {
        fn find(
            devices: &[DeviceInstance],
            instances: &[Instance],
            rename: &dyn Fn(&str) -> String,
            inst: &str,
            param: &str,
        ) -> Option<String> {
            let hit = devices
                .iter()
                .find(|d| d.model.instance_name().is_some_and(|n| rename(n) == inst));
            if let Some(d) = hit {
                return d.model.param_symbol(param).map(|s| rename(&s));
            }
            instances.iter().find_map(|i| {
                let r = |n: &str| rename(&i.rename(n));
                find(&i.devices, &i.instances, &r, inst, param)
            })
        }
        find(&self.devices, &self.instances, &|n| n.to_string(), inst, param)
    }

    /// The parameter vector for `names` (the engine's column order): bound
    /// values, device defaults for unstated parameters, `0.0` for anything
    /// still unbound. The one way to turn a parsed deck into a `p` vector.
    pub fn pvec(&self, names: &[String]) -> Vec<f64> {
        let defaults = self.defaults();
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

    let instances = st.place_items(&items, base_dir)?;
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
        instances,
        node_index: nodes.index,
        node_names: nodes.names,
        values,
        ports,
        report,
    })
}

impl Placer<'_> {
    /// Place `items` in order; the subcircuit instances among them are
    /// returned, placed. Stops at `.end`.
    fn place_items(
        &mut self,
        items: &[Item],
        base_dir: Option<&Path>,
    ) -> Result<Vec<Instance>, ParseError> {
        let mut instances = Vec::new();
        for item in items {
            match item {
                Item::Line(line) => {
                    if !self.place_line(line, base_dir)? {
                        break;
                    }
                }
                Item::Inst(inst) => instances.push(self.place_instance(inst, base_dir)?),
            }
        }
        Ok(instances)
    }

    /// Place a subcircuit instance's body over its own nodes and names, then
    /// wire it into this frame: its ports onto the nodes they connect to, its
    /// internal nodes and bound values under the instance's names.
    fn place_instance(&mut self, inst: &Inst, base_dir: Option<&Path>) -> Result<Instance, ParseError> {
        let mut body = Placer {
            nodes: NodeMap::new(),
            circuit: Circuit::new(),
            devices: Vec::new(),
            values: HashMap::default(),
            validated: std::mem::take(&mut self.validated),
            report: CompatReport::default(),
            ports: Vec::new(),
            params: self.params,
            models: self.models,
            model_aliases: self.model_aliases,
            options: self.options,
            va_models: self.va_models,
            #[cfg(not(target_arch = "wasm32"))]
            osdi_models: std::mem::take(&mut self.osdi_models),
        };
        let instances = body.place_items(&inst.body, base_dir)?;
        self.validated = body.validated;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.osdi_models = body.osdi_models;
        }
        let mut placed = Instance {
            name: inst.name.clone(),
            ns: inst.ns.clone(),
            nodes: Vec::new(),
            node_names: body.nodes.names[1..]
                .iter()
                .map(|n| n.strip_prefix(inst.ns.as_str()).unwrap_or(n).to_string())
                .collect(),
            circuit: body.circuit,
            devices: body.devices,
            instances,
        };
        placed.nodes = body.nodes.names[1..]
            .iter()
            .map(|node| {
                match inst.ports.iter().position(|p| p.eq_ignore_ascii_case(node)) {
                    Some(i) => self.nodes.resolve(&inst.conn[i]),
                    None => self.nodes.resolve(&placed.rename(node)),
                }
            })
            .collect();
        for (k, v) in body.values {
            self.values.insert(placed.rename(&k), v);
        }
        let report = body.report;
        self.report.ignored_directives.extend(report.ignored_directives);
        self.report
            .unknown_params
            .extend(report.unknown_params.iter().map(|p| placed.rename(p)));
        self.report.notes.extend(report.notes);
        self.ports.extend(body.ports.into_iter().map(|p| PortDef {
            name: placed.rename(&p.name),
            node: placed.rename(&p.node),
            z0: p.z0,
        }));
        Ok(placed)
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
