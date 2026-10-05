//! `VerilogADevice`: an elaborated Verilog-A module wrapped as a SANE
//! `DeviceModel`. It lowers via `lower_behavioral` (the behavioral DAE-fragment
//! path), so a Verilog-A model flows through the whole engine like any device.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::Arc;

use rsdag::ExprId;
use sane_core::Graph;
use sane_device::{BehavioralFragment, DeviceModel, Lowerer};

use crate::elaborate::ElaboratedModule;
use crate::lower::lower_analog;

pub struct VerilogADevice {
    /// Instance name, scoping the parameter / internal-unknown symbols.
    pub name: String,
    pub module: Arc<ElaboratedModule>,
    /// Resolved instance parameter values (bare name -> value). Used to fold
    /// compile-time-structural decisions (switch branches) at lowering.
    pub params: HashMap<String, f64>,
    /// Parameter names the deck/instance explicitly set, for `$param_given`.
    pub given: HashSet<String>,
    /// The `.model` card the instance uses: a module parameter the instance
    /// does not set itself is the card's, the symbol `{card}.{param}` shared
    /// by every instance of the card.
    pub card: Option<String>,
    /// Parameters the instance sets itself (`{name}.{param}` symbols).
    pub inline: HashSet<String>,
    /// Parallel multiplicity `m` (instance `m=`/`mult=`): the device behaves as
    /// `m` identical devices in parallel, so every flow contribution scales by `m`
    /// and `$mfactor` returns it. Default 1.0.
    pub mfactor: f64,
}

impl VerilogADevice {
    pub fn new(name: impl Into<String>, module: Arc<ElaboratedModule>) -> Self {
        Self {
            name: name.into(),
            module,
            params: HashMap::default(),
            given: HashSet::default(),
            card: None,
            inline: HashSet::default(),
            mfactor: 1.0,
        }
    }

    pub fn with_instance(
        name: impl Into<String>,
        module: Arc<ElaboratedModule>,
        params: HashMap<String, f64>,
        given: HashSet<String>,
    ) -> Self {
        Self {
            name: name.into(),
            module,
            params,
            given,
            card: None,
            inline: HashSet::default(),
            mfactor: 1.0,
        }
    }

    /// The name of the symbol module parameter `param` binds to: the
    /// instance's when it sets it or uses no card, else the card's.
    pub fn param_symbol(&self, param: &str) -> String {
        match &self.card {
            Some(card) if !self.inline.contains(param) => format!("{card}.{param}"),
            _ => format!("{}.{param}", self.name),
        }
    }

    /// Trial-lower the device into a scratch context to surface any unsupported
    /// construct (a flow probe, a time-domain filter, an unsupported system
    /// function, a runtime switch branch, ...) UP FRONT, with a clear message,
    /// instead of failing later during analysis. The result is discarded.
    pub fn validate(&self) -> Result<(), String> {
        let mut ctx = Graph::new();
        let np = self.module.ports.len();
        let term_v: Vec<ExprId> = (0..np).map(|k| ctx.sym(&format!("__v{k}"))).collect();
        let term_vdot: Vec<ExprId> = (0..np).map(|k| ctx.sym(&format!("__vd{k}"))).collect();
        let mut lo = Lowerer::new(&mut ctx);
        lower_analog(
            &self.module,
            &self.name,
            &self.params,
            &self.given,
            self.mfactor,
            &mut lo,
            &term_v,
            &term_vdot,
        )
        .map(|_| ())
    }
}

impl DeviceModel for VerilogADevice {
    fn n_terminals(&self) -> usize {
        self.module.ports.len()
    }

    fn default_params(&self) -> Vec<(&'static str, f64)> {
        self.module.default_params_static.clone()
    }

    fn instance_name(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn card_name(&self) -> Option<&str> {
        self.card.as_deref()
    }

    fn param_symbol(&self, param: &str) -> Option<String> {
        Some(VerilogADevice::param_symbol(self, param))
    }

    fn param_default(&self, name: &str) -> Option<f64> {
        self.module.default_map.get(name).copied()
    }

    fn canonical_param(&self, key: &str) -> Option<&'static str> {
        self.module.canon.get(&key.to_ascii_lowercase()).copied()
    }

    fn is_sysfn_alias(&self, key: &str) -> bool {
        self.module
            .sysfn_aliases
            .contains(&key.to_ascii_lowercase())
    }

    fn param_aliases(&self) -> Vec<(String, String)> {
        self.module
            .aliases
            .iter()
            .map(|(a, t)| (a.clone(), t.clone()))
            .collect()
    }

    fn lower_behavioral(
        &self,
        lo: &mut Lowerer,
        terminal_v: &[ExprId],
        terminal_vdot: &[ExprId],
        _control_i: &[ExprId],
    ) -> BehavioralFragment {
        // A call of the module's function for this structure (see
        // `crate::template`), built on the first such instance. Unsupported
        // constructs are caught at load time by `validate()`, so a lowering
        // error here is a bug and panics inside.
        crate::template::lower_templated(self, lo, terminal_v, terminal_vdot)
    }
}
