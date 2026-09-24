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
            mfactor: 1.0,
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
    fn template_group(&self) -> Option<String> {
        // Group by module: instances of the same module share the bundling
        // decision even when parameter signatures split their cache keys.
        Some(self.module.name.clone())
    }

    fn n_terminals(&self) -> usize {
        self.module.ports.len()
    }

    fn default_params(&self) -> Vec<(&'static str, f64)> {
        self.module.default_params_static.clone()
    }

    fn instance_name(&self) -> Option<&str> {
        Some(&self.name)
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
        // Lower via the model-instance template cache (see `crate::template`): the
        // first instance of a (module, structure) is lowered directly, every
        // sibling is cloned by symbol substitution rather than re-walking the
        // analog block. Unsupported constructs are caught at load time by
        // `validate()`, so a lowering error here is a bug and panics inside.
        crate::template::lower_templated(self, lo, terminal_v, terminal_vdot)
    }
}
