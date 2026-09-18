//! The device-model contract: every device lowers onto SANE's symbolic DAG
//! through ONE path, [`DeviceModel::lower_behavioral`], producing a
//! [`BehavioralFragment`] (terminal currents, extra-unknown residuals, noise
//! generators, Newton limits, operating-point variables).
//!
//! The constitutive equations of the standard devices live as Verilog-A source
//! in `sane_veriloga::builtin` -- Verilog-A is the source of truth for device
//! physics. This crate only defines the contract and the lowering support
//! types ([`Lowerer`], minted unknowns, delays, bundle templates), plus the one
//! device that is topological rather than physical (the current-controlled
//! switch, which references another element's branch current).
//!
//! Model parameters are instance-scoped free symbols named
//! `"{instance}.{param}"` (e.g. `D1.Is`), so numeric values are bound later.

use rsdag::{ExprId, SymbolId};
use sane_core::constants::COMPANION_G;

// Shared symbolic building blocks (overflow-safe exponential, smooth switch).
pub(crate) mod common;
// Behavioral-lowering support (`Lowerer`, minted unknowns, bundle templates).
mod lowering;

mod switch;
#[cfg(test)]
mod tests;

pub use common::safe_exp;
pub use lowering::{LoweredDelay, LoweredUnknown, Lowerer};
pub use switch::CSwitch;

/// Kind of curve-aware Newton-step limiting for a device's controlling voltage
/// (the classic SPICE convergence aids `pnjlim` / `fetlim`). The device declares
/// only *which* voltage to limit and its kind; the numeric thresholds (thermal
/// voltage, critical / threshold voltage) live as centralized solver constants,
/// because they are *path-only* (they shape the Newton iteration, never the
/// converged fixed point) and only logarithmically sensitive to the model
/// parameters, so a representative value per kind suffices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LimitKind {
    /// Forward PN-junction voltage: logarithmic limiting above the critical
    /// voltage (`pnjlim`).
    PnJunction,
    /// FET channel control voltage `vgs`: bounded step around threshold
    /// (`fetlim`).
    Fet,
}

/// A small-signal noise source contributed by a behavioral device: a current
/// noise generator across nodes `hi`/`lo` (by voltage symbol; `None` = ground)
/// with power spectral density `psd` and flicker exponent `flicker_exp` (both
/// expressions evaluated at the operating point; a constant-zero exponent is
/// white noise, `S(f) = psd / f^exp` otherwise). Consumed by noise analysis.
pub struct NoiseSource {
    pub hi: Option<SymbolId>,
    pub lo: Option<SymbolId>,
    pub psd: ExprId,
    pub flicker_exp: ExprId,
    /// Tabular noise: `(frequency, psd)` points, linearly interpolated. Empty for
    /// white (`flicker_exp == 0`) / flicker sources, which use `psd`.
    pub table: Vec<(f64, f64)>,
}

/// An operating-point variable a device exports: a named, documented internal
/// quantity (`gm`, `vth`, `ids`, ... -- the compact-model `(* desc *)` OPP
/// idiom) whose lowered expression analyses evaluate at a solved point.
#[derive(Clone)]
pub struct OpVar {
    /// Instance-qualified name (`M1.gm`).
    pub name: String,
    /// The variable's own name (`gm`).
    pub short: String,
    pub desc: String,
    pub units: Option<String>,
    pub value: ExprId,
}

/// A controlling-voltage limit contributed by a lowered device fragment:
/// `v(hi) - v(lo)` (by node-voltage symbol; `None` = ground) that the DC Newton
/// solver should curve-limit per step, oriented forward-positive. Recorded at
/// the `$limit(...)` site during lowering, so only the live (polarity-folded)
/// arm of a conditional contributes, and internal-node limits work like
/// terminal ones. Addressed like [`NoiseSource`] terminals and mapped to global
/// unknown indices during DAE assembly.
#[derive(Clone, Copy, Debug)]
pub struct FragmentLimit {
    pub hi: Option<SymbolId>,
    pub lo: Option<SymbolId>,
    pub kind: LimitKind,
}

/// What a device contributes to the DAE: a current leaving each external
/// terminal (summed into node KCL) plus one residual equation (`= 0`) per
/// extra unknown minted on the [`Lowerer`], in mint order. Residuals may
/// reference the extra unknowns' derivative symbols, so `ddt`/`idt`/laplace
/// lower directly. `noise` carries any small-signal noise generators; `limits`
/// any Newton-step controlling-voltage limits.
/// What an unknown physically is. `assemble_dae` mints them in blocks -- node
/// voltages `v{k}`, then one branch current `i_{elem}` per voltage source /
/// inductor / controlled source, then device-internal states -- and every
/// transform preserves that order.
///
/// The distinction matters wherever a bound has to be put on an unknown: volts,
/// amperes and Verilog-A states share no scale, so any such bound must be taken
/// per kind rather than from one constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownKind {
    /// A node potential (the leading KCL block).
    NodeVoltage,
    /// A branch current the assembler minted for an element.
    BranchCurrent,
    /// A device-internal state (Verilog-A state, transmission-line wave).
    DeviceState,
}

/// A switching surface a device declares: the transient integrator lands a
/// step on every zero crossing of `g` in direction `dir` (`0` either way,
/// `+1` rising, `-1` falling), so a hard mode change in the device's
/// expressions happens at a step boundary, never inside a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FragmentEvent {
    pub g: ExprId,
    pub dir: i8,
}

pub struct BehavioralFragment {
    pub terminal_currents: Vec<ExprId>,
    pub residuals: Vec<ExprId>,
    pub noise: Vec<NoiseSource>,
    /// Switching surfaces (see [`FragmentEvent`]).
    pub events: Vec<FragmentEvent>,
    /// The instance's parameter symbols by parameter name (the ones its
    /// expressions reference), so parameters are known by symbol rather than
    /// recovered from a qualified name.
    pub param_syms: Vec<(String, SymbolId)>,
    /// Exported operating-point variables (empty for most devices).
    pub op_vars: Vec<OpVar>,
    /// Controlling-voltage Newton limits (`$limit` sites; empty for most devices).
    pub limits: Vec<FragmentLimit>,
}

/// A nonlinear device: lowers to a DAE fragment over its terminal voltages.
///
/// `Send + Sync` so device instances can live inside types shared across FFI
/// boundaries (e.g. the Python `Circuit`).
pub trait DeviceModel: Send + Sync {
    /// Template-group key for instance counting (the assembler pre-counts
    /// instances per group so the frontend can decide whether the first
    /// instance also routes through the shared compiled bundle). `None` for
    /// devices without a template/bundle path.
    fn template_group(&self) -> Option<String> {
        None
    }

    /// Number of terminals (length of the voltage/current vectors).
    fn n_terminals(&self) -> usize;

    /// Names of controlling (voltage-defined) elements whose branch currents
    /// the device needs. The DAE assembler resolves these and passes their
    /// current expressions as `control_i`.
    fn control_currents(&self) -> Vec<String> {
        Vec::new()
    }

    /// Companion conductance network for homotopy continuation: the device's
    /// *easy* linear form at `lambda = 0`, as `(terminal_i, terminal_j, G)`
    /// conductances (local terminal indices). The DC solver deforms
    /// `H(x, lambda) = F(x) + (1 - lambda) * companion(x)` from this trivially
    /// solvable linear network at `lambda = 0` to the real device at `lambda = 1`
    /// -- a per-device nonlinearity homotopy expressed natively in the graph,
    /// rather than the global gmin shunt. The default is a conductance star to
    /// terminal 0 (so all terminals are connected and the block is regular);
    /// nonlinear models that want a more physical start can override.
    fn companion(&self) -> Vec<(usize, usize, f64)> {
        (1..self.n_terminals())
            .map(|k| (k, 0, COMPANION_G))
            .collect()
    }

    /// Default parameter values (unsuffixed names, e.g. `("Vt", 0.025852)`),
    /// applied by the netlist front-end for any parameter the deck and its
    /// `.model` card leave unspecified. Real SPICE decks omit temperature-derived
    /// and conventional-default parameters (thermal voltage, default beta, etc.),
    /// so without these the corresponding symbols would evaluate to zero.
    fn default_params(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }

    /// The instance name this model was placed under (`M1`, `Xop.M3`), when
    /// the model scopes its parameter symbols by it; `None` for a model whose
    /// parameters are bound numerically (nothing to look up by name).
    fn instance_name(&self) -> Option<&str> {
        None
    }

    /// The module default of one parameter (unsuffixed name). The netlist
    /// front end binds only the values a deck states; a parameter symbol the
    /// deck leaves unspecified takes its value from here at model build time,
    /// so no per-instance copy of a compact model's hundreds of defaults is
    /// ever materialised.
    fn param_default(&self, name: &str) -> Option<f64> {
        self.default_params()
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| *v)
    }

    /// Canonical (exact-case) parameter name for a deck key, after the model's
    /// own aliases (`aliasparam`); `None` when the model has no such parameter.
    /// The default matches case-insensitively against `default_params`; a
    /// model with hundreds of parameters keeps a lookup table instead.
    fn canonical_param(&self, key: &str) -> Option<&'static str> {
        self.default_params()
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(k, _)| *k)
    }

    /// Whether `key` aliases a system function (`aliasparam m = $mfactor`): a
    /// valid deck key that binds no parameter.
    fn is_sysfn_alias(&self, key: &str) -> bool {
        self.param_aliases()
            .iter()
            .any(|(a, t)| a.eq_ignore_ascii_case(key) && t.starts_with('$'))
    }

    /// Alias parameter names (`aliasparam alias = target` in Verilog-A): a deck
    /// key equal to `alias` binds the `target` parameter instead. A target with
    /// a `$` prefix names a system function (e.g. `$mfactor`, the parallel
    /// multiplicity). Default: no aliases.
    fn param_aliases(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Lower this device to a DAE fragment -- THE device integration path.
    /// The builder `lo` mints extra unknowns (internal nodes, branch currents
    /// for voltage contributions, `idt`/laplace states);
    /// `terminal_v`/`terminal_vdot` are the device's terminal node-voltage and
    /// node-voltage-derivative expressions (ground terminals get a zero
    /// derivative); `control_i` holds the controlling branch currents (aligned
    /// with [`control_currents`](Self::control_currents)).
    fn lower_behavioral(
        &self,
        lo: &mut Lowerer,
        terminal_v: &[ExprId],
        terminal_vdot: &[ExprId],
        control_i: &[ExprId],
    ) -> BehavioralFragment;
}

/// A placement of a device: its model and the node indices wired to its
/// terminals, in the model's terminal order (0 = ground).
/// Default-value lookup over the placed devices: `inst.param` resolves through
/// the device placed as `inst` (see [`DeviceModel::param_default`]). The one
/// place the "unstated parameter takes the module default" rule lives; the
/// netlist front end binds only what a deck states.
pub struct ParamDefaults<'a> {
    by_inst: rustc_hash::FxHashMap<&'a str, &'a dyn DeviceModel>,
}

impl<'a> ParamDefaults<'a> {
    pub fn new(devices: &'a [DeviceInstance]) -> Self {
        let by_inst = devices
            .iter()
            .filter_map(|d| d.model.instance_name().map(|n| (n, &*d.model)))
            .collect();
        Self { by_inst }
    }

    /// The device default of parameter symbol `name` (`M1.W`), if the device
    /// placed as its instance prefix declares it.
    pub fn get(&self, name: &str) -> Option<f64> {
        let (inst, p) = name.rsplit_once('.')?;
        self.by_inst.get(inst)?.param_default(p)
    }
}

pub struct DeviceInstance {
    pub model: Box<dyn DeviceModel>,
    pub terminals: Vec<usize>,
    /// Parallel multiplicity (SPICE `M=` times MOSFET `nf`). The device's
    /// terminal currents are scaled by this at DAE assembly, modelling
    /// `mfactor` identical devices in parallel. Defaults to 1.0. Verilog-A
    /// devices scale themselves via `$mfactor`, so this stays 1.0 for them.
    pub mfactor: f64,
}

impl DeviceInstance {
    pub fn new(model: Box<dyn DeviceModel>, terminals: Vec<usize>) -> Self {
        Self {
            model,
            terminals,
            mfactor: 1.0,
        }
    }

    /// Set the parallel multiplicity (`M=` * `nf`).
    pub fn with_mfactor(mut self, mfactor: f64) -> Self {
        self.mfactor = mfactor;
        self
    }
}
