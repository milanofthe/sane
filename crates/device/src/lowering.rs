//! The behavioral-lowering support types: the extra DAE unknowns a device
//! mints, transport delays, shared device-bundle templates, and the [`Lowerer`]
//! state a frontend drives while lowering a device onto the symbolic graph.

use std::any::Any;
use std::collections::HashMap;

use rsdag::{ExprId, Graph, Node, SymbolId};

/// An extra DAE unknown a behavioral device mints beyond its terminals (a branch
/// current for a voltage contribution, or an `idt`/laplace state variable), with
/// its minted value and derivative symbols and whether it is differential (its
/// time derivative appears in a residual, so the integrator must advance it).
pub struct LoweredUnknown {
    pub name: String,
    /// What the unknown physically is (a device-internal node potential, a
    /// branch current, a state): the solver's tolerances and shunts are
    /// taken per kind.
    pub kind: crate::UnknownKind,
    /// The name relative to the minting instance (`br0` of `U1.br0`), so a
    /// template re-mints it for a clone without parsing the qualified name.
    pub suffix: String,
    pub value: ExprId,
    pub xdot: ExprId,
    pub value_sym: SymbolId,
    pub xdot_sym: SymbolId,
    pub differential: bool,
    /// DC Newton seed (`idt(u, ic)`: the ic routed as a `.nodeset`-style
    /// starting value, not a constraint -- DC still enforces `u = 0`).
    pub dc_seed: Option<f64>,
}

/// A transport delay minted by a behavioral device (`absdelay`): the delayed
/// source lives in the extra unknown at `src_extra`, its delayed copy in
/// `out_extra` (both indices into this device's `extras`, in mint order; the
/// assembler shifts them onto the global layout). `hist` is the impure history
/// input symbol the transient loop fills with `src(t - tau)`; `tau` is the
/// delay expression (constants and parameters only).
pub struct LoweredDelay {
    pub src_extra: usize,
    pub out_extra: usize,
    pub hist: SymbolId,
    /// Instance-relative name of the history symbol (see `LoweredUnknown::suffix`).
    pub hist_suffix: String,
    pub tau: ExprId,
}

/// Builder handed to a behavioral device during DAE assembly. It mints extra
/// unknowns into the shared `Graph` and records them so the assembler can
/// append them to the unknown vector and wire their derivative symbols. The
/// device builds its residual expressions over the returned value/derivative
/// exprs. See [`DeviceModel::lower_behavioral`].
pub struct Lowerer<'a> {
    ctx: &'a mut Graph,
    /// Instance count per template group (see
    /// [`DeviceModel::template_group`](crate::DeviceModel::template_group)),
    /// filled by the assembler before the device loop. A frontend uses it to
    /// decide whether the FIRST instance of a template also routes through the
    /// shared compiled bundle (multi-instance circuits: SIMD-lane execution for
    /// all instances) or keeps its fully symbolic fragment (single-instance
    /// circuits: transparent to symbolic tooling).
    pub instance_groups: std::collections::HashMap<String, usize>,
    pub extras: Vec<LoweredUnknown>,
    /// Transport delays minted alongside `extras` (drained together per device).
    pub delays: Vec<LoweredDelay>,
    /// Device-bundle templates minted by the frontend (one per template group,
    /// keyed by the template cache key; drained once at end of assembly).
    /// Per-extract lowering cache, persisting across device instances (the
    /// assembler builds one `Lowerer` for the whole device loop). Type-erased so
    /// this crate stays agnostic of what a frontend caches; the Verilog-A
    /// frontend stores model-instance templates here, keyed by
    /// (module, terminal pattern, parameter signature), so a large compact model
    /// is lowered once and every further instance is cloned by symbol
    /// substitution instead of re-walking its analog block.
    cache: HashMap<String, Box<dyn Any>>,
}

impl<'a> Lowerer<'a> {
    pub fn new(ctx: &'a mut Graph) -> Self {
        Self {
            ctx,
            instance_groups: std::collections::HashMap::new(),
            extras: Vec::new(),
            delays: Vec::new(),
            cache: HashMap::new(),
        }
    }

    /// The shared symbolic context, for building contribution expressions.
    pub fn ctx(&mut self) -> &mut Graph {
        self.ctx
    }

    /// Fetch a clone of a cached value (the cache is read then the caller mutates
    /// the `Lowerer`, so a borrow cannot be held across instantiation).
    pub fn cache_get<T: Clone + 'static>(&self, key: &str) -> Option<T> {
        self.cache
            .get(key)
            .and_then(|b| b.downcast_ref::<T>())
            .cloned()
    }

    /// Store a value in the per-extract lowering cache.
    pub fn cache_put<T: 'static>(&mut self, key: String, val: T) {
        self.cache.insert(key, Box::new(val));
    }

    /// Mint an extra unknown by its fully-qualified `name` (the device scopes it
    /// with its instance name, e.g. `"U1.br0"`) and its derivative symbol.
    /// Returns `(value_expr, xdot_expr)`. `differential = true` marks it as a
    /// state the integrator advances (its `xdot` appears in a residual).
    pub fn unknown(&mut self, name: &str, differential: bool) -> (ExprId, ExprId) {
        self.unknown_with_suffix(name, name, crate::UnknownKind::DeviceState, differential)
    }

    /// Mint a device state scoped by `inst`: named `{inst}.{suffix}`, with the
    /// suffix recorded (see `LoweredUnknown::suffix`).
    pub fn unknown_of(&mut self, inst: &str, suffix: &str, differential: bool) -> (ExprId, ExprId) {
        self.unknown_kind_of(inst, suffix, crate::UnknownKind::DeviceState, differential)
    }

    /// Mint a device-internal node potential `{inst}.{suffix}` (its residual
    /// is a KCL row).
    pub fn internal_node_of(&mut self, inst: &str, suffix: &str) -> (ExprId, ExprId) {
        self.unknown_kind_of(inst, suffix, crate::UnknownKind::NodeVoltage, false)
    }

    /// Mint a branch current `{inst}.{suffix}` (its residual is a KVL or a
    /// flow-definition row).
    pub fn branch_current_of(&mut self, inst: &str, suffix: &str) -> (ExprId, ExprId) {
        self.unknown_kind_of(inst, suffix, crate::UnknownKind::BranchCurrent, false)
    }

    /// Mint an extra unknown `{inst}.{suffix}` of an explicit kind.
    pub fn unknown_kind_of(
        &mut self,
        inst: &str,
        suffix: &str,
        kind: crate::UnknownKind,
        differential: bool,
    ) -> (ExprId, ExprId) {
        let name = format!("{inst}.{suffix}");
        self.unknown_with_suffix(&name, suffix, kind, differential)
    }

    /// Mint an extra unknown by full name and explicit kind.
    pub fn unknown_kind(
        &mut self,
        name: &str,
        kind: crate::UnknownKind,
        differential: bool,
    ) -> (ExprId, ExprId) {
        self.unknown_with_suffix(name, name, kind, differential)
    }

    fn unknown_with_suffix(
        &mut self,
        name: &str,
        suffix: &str,
        kind: crate::UnknownKind,
        differential: bool,
    ) -> (ExprId, ExprId) {
        let (value, value_sym) = mint(self.ctx, name);
        let (xdot, xdot_sym) = mint(self.ctx, &format!("vdot_{name}"));
        self.extras.push(LoweredUnknown {
            name: name.to_string(),
            kind,
            suffix: suffix.to_string(),
            value,
            xdot,
            value_sym,
            xdot_sym,
            differential,
            dc_seed: None,
        });
        (value, xdot)
    }
}

/// Mint (or reuse) a free symbol, returning both its expression and its id.
fn mint(ctx: &mut Graph, name: &str) -> (ExprId, SymbolId) {
    let e = ctx.sym(name);
    let s = match ctx.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!("sym() yields a Symbol node"),
    };
    (e, s)
}
