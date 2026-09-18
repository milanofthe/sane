//! The engine configuration: every switch the engine consults at run time, in
//! one struct, read from the environment once and settable by the host.
//!
//! Each field is a knob with a default that is right for a solve on a
//! developer machine; the environment variable named on the field overrides it
//! for A/B benchmarking and debugging without a code change, and a host (the
//! Python binding, the browser build, an embedding application) sets the
//! struct directly through [`set_config`] / [`update_config`]. Nothing in the
//! engine reads an environment variable itself: [`config`] is the one source,
//! so a switch is documented exactly once, here, and there is no guessing
//! which variable applies where.
//!
//! Test harnesses keep their own environment inputs (`SANE_VA_CORPUS`,
//! `SANE_OPENVAF_BIN`): those locate external data, they do not configure the
//! engine. The log level (`SANE_LOG`) lives with the logger.

use std::sync::RwLock;

/// The engine's run-time switches. See the module docs; every field names the
/// environment variable that overrides it.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Compile hot tapes to native code in the background (`SANE_JIT`, `0`
    /// disables; needs the `jit` feature of the solver).
    pub jit: bool,
    /// Choice-specialise the circuit-level tapes after a solve settles
    /// (`SANE_TAPE_SPEC`, `0` disables).
    pub tape_specialization: bool,
    /// Solve the Newton systems with rsdag's graph solve, the static LU as
    /// one program over the Jacobian entries, with the sparse LU library as
    /// the fallback for patterns beyond its range (`SANE_GRAPH_SOLVE`, `0`
    /// selects the library for every system: the A/B reference).
    pub graph_solve: bool,
    /// Lower the instances of a multiply-instantiated Verilog-A module as
    /// calls into one shared compiled body (`SANE_NO_DEVBUNDLE` disables: every
    /// instance becomes its own graph clone -- the differential reference).
    pub device_bundles: bool,
    /// Lower a Verilog-A module once per structure and clone the instances by
    /// substitution (`SANE_NO_TEMPLATE` disables: every instance is lowered
    /// from scratch -- the differential reference).
    pub device_templates: bool,
    /// Merge the nodes of statically zero-volt Verilog-A branches before
    /// lowering (`SANE_NO_COLLAPSE` disables: every such branch lowers as an
    /// explicit source with its own unknown).
    pub node_collapse: bool,
    /// Worker threads of the engine's pool (`SANE_THREADS`); `None` picks the
    /// solver's default.
    pub threads: Option<usize>,
    /// Force fixed transient steps at the step cap instead of adaptive control
    /// (`SANE_TRAN_FIXED`).
    pub transient_fixed_step: bool,
    /// Locate the declared switching surfaces and land transient steps on
    /// them (`SANE_EVENTS`, `0` disables: the controller discovers every mode
    /// change through rejects -- the A/B reference).
    pub events: bool,
    /// Print the per-iteration DC Newton residual and step trace
    /// (`SANE_DC_TRACE`).
    pub dc_trace: bool,
    /// Print every transient candidate step: time, size, error, verdict and
    /// the located events (`SANE_TRAN_TRACE`).
    pub tran_trace: bool,
    /// Harmonic-balance Jacobian bandwidth `|k-l| <= B` (`SANE_HB_BAND`);
    /// `None` keeps the full blocks.
    pub hb_band: Option<u32>,
    /// Report non-finite compile-time constants during Verilog-A lowering
    /// (`SANE_VA_TRACE_NAN`).
    pub va_trace_nan: bool,
    /// Trace the Verilog-A `while`-loop unrolling scan (`SANE_VA_DEBUG_WHILE`).
    pub va_debug_while: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            jit: true,
            tape_specialization: true,
            graph_solve: true,
            device_bundles: true,
            device_templates: true,
            node_collapse: true,
            threads: None,
            transient_fixed_step: false,
            events: true,
            dc_trace: false,
            tran_trace: false,
            hb_band: None,
            va_trace_nan: false,
            va_debug_while: false,
        }
    }
}

impl Config {
    /// The defaults with every environment override applied.
    pub fn from_env() -> Self {
        let mut c = Config::default();
        let off = |name: &str| {
            matches!(
                std::env::var(name).as_deref(),
                Ok("0") | Ok("off") | Ok("false")
            )
        };
        let set = |name: &str| std::env::var_os(name).is_some();
        if off("SANE_JIT") {
            c.jit = false;
        }
        if off("SANE_TAPE_SPEC") {
            c.tape_specialization = false;
        }
        if off("SANE_GRAPH_SOLVE") {
            c.graph_solve = false;
        }
        if set("SANE_NO_DEVBUNDLE") {
            c.device_bundles = false;
        }
        if set("SANE_NO_TEMPLATE") {
            c.device_templates = false;
        }
        if set("SANE_NO_COLLAPSE") {
            c.node_collapse = false;
        }
        c.threads = std::env::var("SANE_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n >= 1);
        c.transient_fixed_step = set("SANE_TRAN_FIXED");
        if off("SANE_EVENTS") {
            c.events = false;
        }
        c.dc_trace = set("SANE_DC_TRACE");
        c.tran_trace = set("SANE_TRAN_TRACE");
        c.hb_band = std::env::var("SANE_HB_BAND")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        c.va_trace_nan = set("SANE_VA_TRACE_NAN");
        c.va_debug_while = set("SANE_VA_DEBUG_WHILE");
        c
    }
}

static CONFIG: RwLock<Option<Config>> = RwLock::new(None);

/// The active configuration (a copy; the struct is small). Initialised from
/// the environment on first use.
pub fn config() -> Config {
    if let Some(c) = CONFIG.read().unwrap().as_ref() {
        return c.clone();
    }
    let mut w = CONFIG.write().unwrap();
    w.get_or_insert_with(Config::from_env).clone()
}

/// Replace the active configuration. Takes effect for everything compiled or
/// solved from now on; tapes and bodies already built keep their choices.
pub fn set_config(c: Config) {
    *CONFIG.write().unwrap() = Some(c);
}

/// Modify the active configuration in place (initialising it from the
/// environment first if needed).
pub fn update_config(f: impl FnOnce(&mut Config)) {
    let mut w = CONFIG.write().unwrap();
    let c = w.get_or_insert_with(Config::from_env);
    f(c);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_documented_ones() {
        let c = Config::default();
        assert!(c.jit && c.tape_specialization && c.device_bundles && c.device_templates);
        assert!(c.threads.is_none() && c.hb_band.is_none());
    }

    #[test]
    fn update_is_visible_to_config() {
        update_config(|c| c.va_trace_nan = true);
        assert!(config().va_trace_nan);
        update_config(|c| c.va_trace_nan = false);
        assert!(!config().va_trace_nan);
    }
}
