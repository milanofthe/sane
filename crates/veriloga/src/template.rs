//! Model functions: a Verilog-A module as a function of the graph.
//!
//! A Verilog-A compact model (BSIM4 is ~10k lines) lowers its analog block into
//! thousands of DAG nodes. It is lowered once per *structure* into a function
//! over formal leaves -- the terminal voltages, the extra unknowns it mints (internal nodes, probed and switched branch currents,
//! `idt` states), its parameters and its delay histories -- and every instance,
//! the first and a single one alike, is a call of that function with its own
//! arguments. The solver runs one compiled body per function over all its
//! calls; symbolic tooling that wants the expressions inlines them
//! (`Graph::inline_all`).
//!
//! Two instances share a function when their lowering is identical. The
//! lowering is exact in the parameters (a parameter is a symbol, every branch
//! on one kept in the graph), so what tells two lowerings apart is only:
//! - the terminal pattern: which terminals are ground and which are tied
//!   together, since those collapse `V(a,b)` terms and change the graph shape;
//! - the multiplicity, baked into every flow;
//! - which of the parameters `$param_given` asks about the instance set;
//! - its structure: the integer parameters it reads (mode selectors), and
//!   the truth of each condition on parameters that decides the topology (a
//!   branch shorted or open), folded at the instance's values. The function
//!   keeps them as assertions, and an instance takes the function whose
//!   assertions hold at its values.
//! Every instance with the same of those is a call of one function, its
//! parameters the call's arguments, whatever their values.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use rsdag::{Crossing, ExprId, FuncId, ParamRole, SymbolId};
use sane_core::Graph;
use sane_device::{
    BehavioralFragment, FragmentEvent, FragmentLimit, LoweredDelay, Lowerer, NoiseSource, OpVar,
};

use crate::device::VerilogADevice;
use crate::lower::{lower_analog, sym_of};

/// An extra unknown the function's body reads, minted afresh per instance.
#[derive(Clone)]
struct ExtraMeta {
    /// Name relative to the instance (`di` of `M1.di`).
    suffix: String,
    kind: sane_device::UnknownKind,
    value_sym: SymbolId,
    /// DC Newton seed carried by the state (`idt(u, ic)` with constant ic).
    dc_seed: Option<f64>,
}

/// A recorded `absdelay`: extras-relative positions plus the formal history
/// symbol (minted per instance from `hist_suffix`).
#[derive(Clone)]
struct DelayMeta {
    src_extra: usize,
    out_extra: usize,
    hist: SymbolId,
    hist_suffix: String,
}

#[derive(Clone)]
struct NoiseMeta {
    hi: Option<SymbolId>,
    lo: Option<SymbolId>,
    /// The source as lowered, for its shape (its expressions are outputs).
    source: NoiseSource,
}

/// A module lowered once for one structure: the function, its formal leaves
/// and the per-instance registries (noise generators, limits, op-vars,
/// delays, events) over those leaves.
#[derive(Clone)]
struct ModelFn {
    func: FuncId,
    /// Per terminal, the formal voltage symbol (`None`: a ground terminal, a
    /// constant in the body).
    terminal_syms: Vec<Option<SymbolId>>,
    extras: Vec<ExtraMeta>,
    delays: Vec<DelayMeta>,
    /// The formal parameter symbols, by parameter name.
    param_syms: Vec<(String, SymbolId)>,
    noise: Vec<NoiseMeta>,
    limits: Vec<FragmentLimit>,
    /// Operating-point variables: (name, description, units).
    op_vars: Vec<(String, String, Option<String>)>,
    events: Vec<Crossing>,
    n_cur: u32,
    /// The function's parameters, in order. A call binds each one to the
    /// instance's expression, or passes a shared global (`$temp`, `t`) as
    /// itself.
    leaves: Vec<SymbolId>,
    /// Where the extras' rows' currents start.
    out_rows: u32,
    /// Where the stored charges start, and per row (terminals, then
    /// extras) whether it stores one (empty without charges).
    out_charge: u32,
    charge_out: Vec<bool>,
    out_noise: u32,
    out_opvar: u32,
    out_tau: u32,
    out_event: u32,
    /// Its assertions, over the formal parameters.
    assertions: Vec<sane_device::Assertion>,
    /// What its structure rests on (see the module docs), over the formal
    /// parameters.
    structure: Vec<sane_device::Assertion>,
    /// The internal nodes its structure merged, with their formal voltage.
    collapsed: Vec<(String, ExprId)>,
}

/// `dev`'s parameter values its structure is decided at: those of the
/// binding `lo` lowers at, else its own.
fn values_of(dev: &VerilogADevice, lo: &Lowerer) -> rustc_hash::FxHashMap<String, f64> {
    let mut values = dev.values();
    for (name, v) in values.iter_mut() {
        if let Some(b) = lo.value(&dev.param_symbol(name)) {
            *v = b;
        }
    }
    values
}

impl ModelFn {
    /// Whether `values` are of this function's structure.
    fn fits(&self, values: &rustc_hash::FxHashMap<String, f64>, ctx: &Graph) -> bool {
        if self.structure.is_empty() {
            return true;
        }
        let env: std::collections::HashMap<SymbolId, f64> = (self.param_syms.iter())
            .filter_map(|(name, s)| Some((*s, *values.get(name)?)))
            .collect();
        let holds: Vec<ExprId> = self.structure.iter().map(|a| a.holds).collect();
        rsdag::eval::<f64, _>(ctx, &holds, &env)
            .iter()
            .all(|&h| h == 1.0)
    }
}

/// Lower `dev` as a call of its module's function, building the function on
/// the first instance of its structure (see the module docs); an error
/// names what of the module does not lower.
pub(crate) fn lower_templated(
    dev: &VerilogADevice,
    lo: &mut Lowerer,
    term_v: &[ExprId],
) -> Result<BehavioralFragment, String> {
    let key = cache_key(dev, lo.ctx(), term_v);
    let mut fns = lo.cache_get::<Vec<ModelFn>>(&key).unwrap_or_default();
    let values = values_of(dev, lo);
    let mf = match fns.iter().find(|mf| mf.fits(&values, lo.ctx())) {
        Some(mf) => mf.clone(),
        None => {
            let t = sane_core::time::Instant::now();
            let ns = format!("{}#{}", dev.module.name, lo.ctx().n_funcs());
            let mf = build_function(dev, &values, lo, term_v, &ns)?;
            fns.push(mf.clone());
            lo.cache_put(key, fns);
            sane_core::profile::record_tpl_build(t.elapsed().as_nanos());
            mf
        }
    };
    let t = sane_core::time::Instant::now();
    let frag = instantiate(&mf, dev, lo, term_v);
    sane_core::profile::record_tpl_clone(t.elapsed().as_nanos());
    Ok(frag)
}

/// Lower `dev`'s module into a function named `ns` over formal leaves, the
/// terminals laid out by `term_v`'s connectivity pattern.
fn build_function(
    dev: &VerilogADevice,
    values: &rustc_hash::FxHashMap<String, f64>,
    lo: &mut Lowerer,
    term_v: &[ExprId],
    ns: &str,
) -> Result<ModelFn, String> {
    // Formal terminals: one symbol per distinct connected node (tied
    // terminals share it), the actual constant for a ground terminal.
    let mut fv = Vec::new();
    let mut terminal_syms = Vec::new();
    let mut seen: Vec<SymbolId> = Vec::new();
    for &e in term_v {
        let ctx = lo.ctx();
        match sym_of(ctx, e) {
            Some(s) => {
                let k = seen.iter().position(|&x| x == s).unwrap_or_else(|| {
                    seen.push(s);
                    seen.len() - 1
                });
                let v = ctx.sym(&format!("{ns}.t{k}"));
                terminal_syms.push(sym_of(ctx, v));
                fv.push(v);
            }
            None => {
                terminal_syms.push(None);
                fv.push(e);
            }
        }
    }

    // The body mints its extras as formal leaves: set aside the instance
    // registries while it lowers, keep what it minted as the function's.
    let saved_extras = std::mem::take(&mut lo.extras);
    let saved_delays = std::mem::take(&mut lo.delays);
    let lowered = lower_analog(&dev.module, ns, &dev.given, values, dev.mfactor, lo, &fv);
    let minted = std::mem::replace(&mut lo.extras, saved_extras);
    let minted_delays = std::mem::replace(&mut lo.delays, saved_delays);
    let frag = lowered?;
    let extras: Vec<ExtraMeta> = minted
        .iter()
        .map(|u| ExtraMeta {
            suffix: u.suffix.clone(),
            kind: u.kind,
            value_sym: u.value_sym,
            dc_seed: u.dc_seed,
        })
        .collect();
    let delays: Vec<DelayMeta> = minted_delays
        .iter()
        .map(|d| DelayMeta {
            src_extra: d.src_extra,
            out_extra: d.out_extra,
            hist: d.hist,
            hist_suffix: d.hist_suffix.clone(),
        })
        .collect();

    // Every per-instance quantity is one output.
    let mut outs: Vec<ExprId> = frag.terminal_currents.clone();
    let n_cur = outs.len() as u32;
    outs.extend(frag.currents.iter().copied());
    // The rows' charges, terminals first, after the currents; a charge that
    // is zero is no output.
    let out_charge = outs.len() as u32;
    let charge_out: Vec<bool> = frag
        .terminal_charges
        .iter()
        .chain(&frag.charges)
        .map(|&q| {
            let stored = !lo.ctx().is_zero(q);
            if stored {
                outs.push(q);
            }
            stored
        })
        .collect();
    let out_noise = outs.len() as u32;
    for n in &frag.noise {
        outs.extend(n.exprs());
    }
    let out_opvar = outs.len() as u32;
    outs.extend(frag.op_vars.iter().map(|v| v.value));
    let out_tau = outs.len() as u32;
    outs.extend(minted_delays.iter().map(|d| d.tau));
    let out_event = outs.len() as u32;
    outs.extend(frag.events.iter().map(|e| e.g));

    // The parameters in a fixed order: terminals, the extras, the model
    // parameters by name, then what else the body reads (the temperature,
    // time, delay histories).
    let ctx = lo.ctx();
    let free = ctx.free_symbols_in(&outs);
    let mut param_syms = frag.param_syms.clone();
    param_syms.sort();
    let mut leaves: Vec<SymbolId> = Vec::new();
    let push = |s: SymbolId, leaves: &mut Vec<SymbolId>| {
        if free.contains(&s) && !leaves.contains(&s) {
            leaves.push(s);
        }
    };
    for &s in terminal_syms.iter().flatten() {
        push(s, &mut leaves);
    }
    for x in &extras {
        push(x.value_sym, &mut leaves);
    }
    for &(_, s) in &param_syms {
        push(s, &mut leaves);
    }
    for &s in &free {
        push(s, &mut leaves);
    }
    // named after its module; `ns` keeps the formal leaves apart
    let func = ctx.define_func(&dev.module.name, leaves.clone(), outs);
    // The parameters (the model's, and the temperature) are the body's pure
    // arguments: a compiled body splits over them, so their work runs once
    // per parameter binding rather than in every evaluation.
    let pure: std::collections::HashSet<SymbolId> = param_syms.iter().map(|&(_, s)| s).collect();
    for (k, &s) in leaves.iter().enumerate() {
        if pure.contains(&s) || ctx.symbol_name(s) == sane_core::constants::TEMP_SYMBOL {
            ctx.set_param_role(func, k as u32, ParamRole::Param);
        }
    }

    let mf = ModelFn {
        func,
        terminal_syms,
        extras,
        delays,
        param_syms,
        noise: frag
            .noise
            .iter()
            .map(|n| NoiseMeta {
                hi: n.hi,
                lo: n.lo,
                source: n.clone(),
            })
            .collect(),
        limits: frag.limits.clone(),
        op_vars: frag
            .op_vars
            .iter()
            .map(|v| (v.short.clone(), v.desc.clone(), v.units.clone()))
            .collect(),
        events: frag.events.iter().map(|e| e.dir).collect(),
        n_cur,
        leaves,
        out_rows: n_cur,
        out_charge,
        charge_out,
        out_noise,
        out_opvar,
        out_tau,
        out_event,
        assertions: frag.assertions.clone(),
        structure: frag.structural.clone(),
        collapsed: frag.collapsed.clone(),
    };
    Ok(mf)
}

/// `dev` as a call of `mf`: its extras minted, its leaves bound, one call per
/// output.
fn instantiate(
    mf: &ModelFn,
    dev: &VerilogADevice,
    lo: &mut Lowerer,
    term_v: &[ExprId],
) -> BehavioralFragment {
    let mut map: HashMap<SymbolId, ExprId> = HashMap::default();
    // Mint the extras in the function's order so the DAE rows line up.
    for ex in &mf.extras {
        let val = lo.unknown_kind_of(&dev.name, &ex.suffix, ex.kind);
        lo.extras.last_mut().expect("just minted").dc_seed = ex.dc_seed;
        map.insert(ex.value_sym, val);
    }
    for k in 0..term_v.len() {
        if let Some(s) = mf.terminal_syms[k] {
            map.insert(s, term_v[k]);
        }
    }
    let ctx = lo.ctx();
    let zero = ctx.zero();
    let mut param_syms: Vec<(String, SymbolId)> = Vec::with_capacity(mf.param_syms.len());
    // what every instance of the card reads alike: the card's parameters
    // and the leaves no instance binds (the temperature)
    let mut shared: HashSet<SymbolId> = HashSet::default();
    for (name, formal) in &mf.param_syms {
        let actual = ctx.sym(&dev.param_symbol(name));
        map.insert(*formal, actual);
        if dev.card_of(name).is_some() {
            shared.insert(*formal);
        }
        if let Some(s) = sym_of(ctx, actual) {
            param_syms.push((name.clone(), s));
        }
    }
    let hist_syms: Vec<(SymbolId, String)> = mf
        .delays
        .iter()
        .map(|dl| {
            let he = ctx.sym(&format!("{}.{}", dev.name, dl.hist_suffix));
            map.insert(dl.hist, he);
            (
                sym_of(ctx, he).expect("sym() yields a Symbol"),
                dl.hist_suffix.clone(),
            )
        })
        .collect();

    // The shared leaves are a binding of the function, once per card; a
    // call passes the instance's own.
    let (mut bound, mut args) = (Vec::new(), Vec::new());
    for (k, &s) in mf.leaves.iter().enumerate() {
        match map.get(&s) {
            Some(&a) if !shared.contains(&s) => args.push(a),
            a => bound.push((k as u32, a.copied().unwrap_or_else(|| ctx.symbol_expr(s)))),
        }
    }
    let func = ctx.bind(mf.func, &bound);
    let mut call = |range: std::ops::Range<u32>| -> Vec<ExprId> {
        ctx.calls_bound(func, &range.collect::<Vec<u32>>(), &args)
    };
    let terminal_currents = call(0..mf.n_cur);
    let currents = call(mf.out_rows..mf.out_charge);
    let mut stored = call(mf.out_charge..mf.out_noise).into_iter();
    let mut charges: Vec<ExprId> = mf
        .charge_out
        .iter()
        .map(|&q| {
            if q {
                stored.next().expect("a stored charge")
            } else {
                zero
            }
        })
        .collect();
    let terminal_charges: Vec<ExprId> = charges
        .drain(..charges.len().min(mf.n_cur as usize))
        .collect();
    let noise_vals = call(mf.out_noise..mf.out_opvar);
    let opvar_vals = call(mf.out_opvar..mf.out_tau);
    let taus = call(mf.out_tau..mf.out_event);
    let event_vals = call(mf.out_event..mf.out_event + mf.events.len() as u32);

    let events = event_vals
        .into_iter()
        .zip(&mf.events)
        .map(|(g, &dir)| FragmentEvent { g, dir })
        .collect();
    let inst_delays: Vec<LoweredDelay> = mf
        .delays
        .iter()
        .zip(hist_syms)
        .zip(taus)
        .map(|((dl, (hist, hist_suffix)), tau)| LoweredDelay {
            src_extra: dl.src_extra,
            out_extra: dl.out_extra,
            hist,
            hist_suffix,
            tau,
        })
        .collect();
    let mut noise_vals = noise_vals.into_iter();
    let noise = mf
        .noise
        .iter()
        .map(|n| {
            let (hi, lo) = (remap_sym(n.hi, &map, ctx), remap_sym(n.lo, &map, ctx));
            n.source.with_exprs(hi, lo, &mut noise_vals)
        })
        .collect();
    let op_vars = mf
        .op_vars
        .iter()
        .zip(opvar_vals)
        .map(|((short, desc, units), value)| OpVar {
            name: format!("{}.{}", dev.name, short),
            short: short.clone(),
            desc: desc.clone(),
            units: units.clone(),
            value,
        })
        .collect();
    let rename: HashMap<SymbolId, ExprId> = mf
        .param_syms
        .iter()
        .filter_map(|(_, s)| map.get(s).map(|&a| (*s, a)))
        .collect();
    let limits = mf
        .limits
        .iter()
        .map(|l| FragmentLimit {
            hi: remap_sym(l.hi, &map, ctx),
            lo: remap_sym(l.lo, &map, ctx),
            kind: l.kind,
            when: l.when.map(|w| rsdag::substitute(ctx, &[w], &rename)[0]),
        })
        .collect();
    let renamed =
        |ctx: &mut Graph, list: &[sane_device::Assertion]| -> Vec<sane_device::Assertion> {
            let holds: Vec<ExprId> = list.iter().map(|a| a.holds).collect();
            let holds = rsdag::substitute(ctx, &holds, &rename);
            (list.iter().zip(holds))
                .map(|(a, holds)| sane_device::Assertion {
                    holds,
                    message: format!("{}: {}", dev.name, a.message),
                })
                .collect()
        };
    let structural = renamed(ctx, &mf.structure);
    let voltages: Vec<ExprId> = mf.collapsed.iter().map(|&(_, v)| v).collect();
    let voltages = rsdag::substitute(ctx, &voltages, &map);
    let collapsed = (mf.collapsed.iter().zip(voltages))
        .map(|((node, _), v)| (format!("{}.{}", dev.name, node), v))
        .collect();
    let assertions = renamed(ctx, &mf.assertions);
    lo.delays.extend(inst_delays);
    BehavioralFragment {
        terminal_currents,
        currents,
        terminal_charges,
        charges,
        noise,
        events,
        param_syms,
        op_vars,
        limits,
        assertions,
        structural,
        collapsed,
    }
}

/// `(module, terminal pattern, multiplicity, given-set)` -- see the module
/// docs. `mfactor` is baked into the graph (every flow scales by it), and
/// `$param_given` answers fold to constants, so both are structure; the
/// functions under one key tell their structure apart by their assertions.
fn cache_key(dev: &VerilogADevice, ctx: &Graph, term_v: &[ExprId]) -> String {
    let pattern = terminal_pattern(ctx, term_v);
    let given: String = (dev.module.given_reads.iter())
        .map(|p| if dev.given.contains(p) { '1' } else { '0' })
        .collect();
    format!(
        "va\u{1}{}\u{1}{}\u{1}{}\u{1}{}",
        dev.module.name,
        pattern,
        dev.mfactor.to_bits(),
        given,
    )
}

/// Canonical terminal connectivity: `G` for a grounded/constant terminal, else a
/// small integer that is the same for terminals tied to the same node.
fn terminal_pattern(ctx: &Graph, term_v: &[ExprId]) -> String {
    let mut seen: Vec<SymbolId> = Vec::new();
    let mut out = String::new();
    for &e in term_v {
        match sym_of(ctx, e) {
            Some(s) => {
                let idx = seen.iter().position(|x| *x == s).unwrap_or_else(|| {
                    seen.push(s);
                    seen.len() - 1
                });
                out.push_str(&idx.to_string());
            }
            None => out.push('G'),
        }
        out.push(',');
    }
    out
}

/// Remap a formal node symbol (a noise generator's, a limit's) to the
/// instance's. Unmapped symbols pass through.
fn remap_sym(
    s: Option<SymbolId>,
    map: &HashMap<SymbolId, ExprId>,
    ctx: &Graph,
) -> Option<SymbolId> {
    s.map(|sym| match map.get(&sym) {
        Some(&e) => sym_of(ctx, e).unwrap_or(sym),
        None => sym,
    })
}
