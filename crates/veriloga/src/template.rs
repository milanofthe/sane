//! Model functions: a Verilog-A module as a function of the graph.
//!
//! A Verilog-A compact model (BSIM4 is ~10k lines) lowers its analog block into
//! thousands of DAG nodes. It is lowered once per *structure* into a function
//! over formal leaves -- the terminal voltages and their derivatives, the extra
//! unknowns it mints (internal nodes, probed and switched branch currents,
//! `idt` states), its parameters and its delay histories -- and every instance,
//! the first and a single one alike, is a call of that function with its own
//! arguments. The solver runs one compiled body per function over all its
//! calls; symbolic tooling that wants the expressions inlines them
//! (`Graph::inline_all`).
//!
//! Two instances share a function only if their lowering is provably
//! identical. The key is `(module, terminal-connectivity pattern,
//! multiplicity)`, and under it each function carries the values of the
//! parameters that decided its structure (and, for `$param_given`, whether
//! they were set):
//! - a parameter's value enters the graph only through a structural decision
//!   (a branch, a loop bound, a static short, a folded constant); everywhere
//!   else it is the function's parameter. The lowering records which
//!   parameters its decisions read, through the constant shadow of every
//!   variable they went into;
//! - the terminal pattern captures which terminals are ground and which are tied
//!   together, since those collapse `V(a,b)` terms and change the graph shape.
//! Same key and the same values on those parameters => the same decisions =>
//! the same function. Instances that differ only in the others (a transistor's
//! W and L, say) share it, and their parameters are its arguments.

use rustc_hash::FxHashMap as HashMap;

use rsdag::{Crossing, ExprId, FuncId, ParamRole, SymbolId};
use sane_core::Graph;
use sane_device::{
    BehavioralFragment, FragmentEvent, FragmentLimit, LoweredDelay, Lowerer, NoiseSource, OpVar,
};

use crate::device::VerilogADevice;
use crate::lower::{lower_analog_structural, sym_of};

/// An extra unknown the function's body reads, minted afresh per instance.
#[derive(Clone)]
struct ExtraMeta {
    /// Name relative to the instance (`di` of `M1.di`).
    suffix: String,
    kind: sane_device::UnknownKind,
    value_sym: SymbolId,
    xdot_sym: SymbolId,
    differential: bool,
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
    table: Vec<(f64, f64)>,
}

/// A module lowered once for one structure: the function, its formal leaves
/// and the per-instance registries (noise generators, limits, op-vars,
/// delays, events) over those leaves.
#[derive(Clone)]
struct ModelFn {
    func: FuncId,
    /// Per terminal, the formal voltage and derivative symbols (`None`: a
    /// ground terminal, a constant in the body).
    terminal_syms: Vec<Option<SymbolId>>,
    terminal_vdot_syms: Vec<Option<SymbolId>>,
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
    out_res: u32,
    out_noise: u32,
    out_opvar: u32,
    out_tau: u32,
    out_event: u32,
}

/// The functions of one (module, terminal pattern, multiplicity), each with
/// the values of the parameters that decided its structure (`$given(X)`:
/// whether `X` was set, as 0 or 1).
type Bucket = std::rc::Rc<std::cell::RefCell<Vec<(Vec<(String, u64)>, ModelFn)>>>;

/// Lower `dev` as a call of its module's function, building the function on
/// the first instance of its (module, structure).
pub(crate) fn lower_templated(
    dev: &VerilogADevice,
    lo: &mut Lowerer,
    term_v: &[ExprId],
    term_vdot: &[ExprId],
) -> BehavioralFragment {
    let key = cache_key(dev, lo.ctx(), term_v);
    let bucket = match lo.cache_get::<Bucket>(&key) {
        Some(b) => b,
        None => {
            let b = Bucket::default();
            lo.cache_put(key.clone(), b.clone());
            b
        }
    };
    let env = instance_param_env(dev);
    // A structural read: a parameter's value, or whether it was set.
    let bits_of = |name: &str| -> u64 {
        match name
            .strip_prefix("$given(")
            .and_then(|s| s.strip_suffix(')'))
        {
            Some(p) => dev.given.contains(p) as u64,
            None => env.get(name).map_or(u64::MAX, |v| v.to_bits()),
        }
    };
    let hit = bucket.borrow().iter().find_map(|(sig, mf)| {
        sig.iter()
            .all(|(name, bits)| bits_of(name) == *bits)
            .then(|| mf.clone())
    });
    let mf = match hit {
        Some(mf) => mf,
        None => {
            let t = sane_core::time::Instant::now();
            let ns = format!("{}#{}", dev.module.name, lo.ctx().n_funcs());
            let (mf, structural) = build_function(dev, lo, term_v, &ns);
            let sig: Vec<(String, u64)> = structural
                .into_iter()
                .map(|name| {
                    let bits = bits_of(&name);
                    (name, bits)
                })
                .collect();
            sane_core::log::debug(&format!(
                "model function '{ns}' ({}): structure fixed by {} parameters: {}",
                dev.name,
                sig.len(),
                sig.iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
            bucket.borrow_mut().push((sig, mf.clone()));
            sane_core::profile::record_tpl_build(t.elapsed().as_nanos());
            mf
        }
    };
    let t = sane_core::time::Instant::now();
    let frag = instantiate(&mf, dev, lo, term_v, term_vdot);
    sane_core::profile::record_tpl_clone(t.elapsed().as_nanos());
    frag
}

/// Lower `dev`'s module into a function named `ns` over formal leaves, the
/// terminals laid out by `term_v`'s connectivity pattern. Returns it and the
/// parameters whose values fixed its structure.
fn build_function(
    dev: &VerilogADevice,
    lo: &mut Lowerer,
    term_v: &[ExprId],
    ns: &str,
) -> (ModelFn, std::collections::BTreeSet<String>) {
    // Formal terminals: one symbol per distinct connected node (tied
    // terminals share it), the actual constant for a ground terminal.
    let (mut fv, mut fvd) = (Vec::new(), Vec::new());
    let (mut terminal_syms, mut terminal_vdot_syms) = (Vec::new(), Vec::new());
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
                let vd = ctx.sym(&format!("{ns}.t{k}'"));
                terminal_syms.push(sym_of(ctx, v));
                terminal_vdot_syms.push(sym_of(ctx, vd));
                fv.push(v);
                fvd.push(vd);
            }
            None => {
                terminal_syms.push(None);
                terminal_vdot_syms.push(None);
                fv.push(e);
                fvd.push(ctx.zero());
            }
        }
    }

    // The body mints its extras as formal leaves: set aside the instance
    // registries while it lowers, keep what it minted as the function's.
    let saved_extras = std::mem::take(&mut lo.extras);
    let saved_delays = std::mem::take(&mut lo.delays);
    let lowered = lower_analog_structural(
        &dev.module,
        ns,
        &dev.params,
        &dev.given,
        dev.mfactor,
        lo,
        &fv,
        &fvd,
    );
    let minted = std::mem::replace(&mut lo.extras, saved_extras);
    let minted_delays = std::mem::replace(&mut lo.delays, saved_delays);
    let (frag, structural) = match lowered {
        Ok(r) => r,
        Err(e) => {
            // validate() runs at load time, so an unsupported construct is
            // already a parse error; reaching here is a real bug.
            sane_core::log::error(&format!("Verilog-A lowering of '{}': {e}", dev.module.name));
            panic!("Verilog-A lowering of '{}': {e}", dev.module.name);
        }
    };
    let extras: Vec<ExtraMeta> = minted
        .iter()
        .map(|u| ExtraMeta {
            suffix: u.suffix.clone(),
            kind: u.kind,
            value_sym: u.value_sym,
            xdot_sym: u.xdot_sym,
            differential: u.differential,
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
    outs.extend(frag.residuals.iter().copied());
    let out_noise = outs.len() as u32;
    for n in &frag.noise {
        outs.push(n.psd);
        outs.push(n.flicker_exp);
    }
    let out_opvar = outs.len() as u32;
    outs.extend(frag.op_vars.iter().map(|v| v.value));
    let out_tau = outs.len() as u32;
    outs.extend(minted_delays.iter().map(|d| d.tau));
    let out_event = outs.len() as u32;
    outs.extend(frag.events.iter().map(|e| e.g));

    // The parameters in a fixed order: terminals, their derivatives, the
    // extras and theirs, the model parameters by name, then what else the
    // body reads (the temperature, time, delay histories).
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
    for &s in terminal_vdot_syms.iter().flatten() {
        push(s, &mut leaves);
    }
    for x in &extras {
        push(x.value_sym, &mut leaves);
    }
    for x in &extras {
        push(x.xdot_sym, &mut leaves);
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
        terminal_vdot_syms,
        extras,
        delays,
        param_syms,
        noise: frag
            .noise
            .iter()
            .map(|n| NoiseMeta {
                hi: n.hi,
                lo: n.lo,
                table: n.table.clone(),
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
        out_res: n_cur,
        out_noise,
        out_opvar,
        out_tau,
        out_event,
    };
    (mf, structural)
}

/// `dev` as a call of `mf`: its extras minted, its leaves bound, one call per
/// output.
fn instantiate(
    mf: &ModelFn,
    dev: &VerilogADevice,
    lo: &mut Lowerer,
    term_v: &[ExprId],
    term_vdot: &[ExprId],
) -> BehavioralFragment {
    let mut map: HashMap<SymbolId, ExprId> = HashMap::default();
    // Mint the extras in the function's order so the DAE rows line up.
    for ex in &mf.extras {
        let (val, xdot) = lo.unknown_kind_of(&dev.name, &ex.suffix, ex.kind, ex.differential);
        lo.extras.last_mut().expect("just minted").dc_seed = ex.dc_seed;
        map.insert(ex.value_sym, val);
        map.insert(ex.xdot_sym, xdot);
    }
    for k in 0..term_v.len() {
        if let Some(s) = mf.terminal_syms[k] {
            map.insert(s, term_v[k]);
        }
        if let Some(s) = mf.terminal_vdot_syms[k] {
            map.insert(s, term_vdot[k]);
        }
    }
    let ctx = lo.ctx();
    let mut param_syms: Vec<(String, SymbolId)> = Vec::with_capacity(mf.param_syms.len());
    for (name, formal) in &mf.param_syms {
        let actual = ctx.sym(&dev.param_symbol(name));
        map.insert(*formal, actual);
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

    let args: Vec<ExprId> = mf
        .leaves
        .iter()
        .map(|&s| map.get(&s).copied().unwrap_or_else(|| ctx.symbol_expr(s)))
        .collect();
    let mut call = |range: std::ops::Range<u32>| -> Vec<ExprId> {
        ctx.calls(mf.func, &range.collect::<Vec<u32>>(), &args)
    };
    let terminal_currents = call(0..mf.n_cur);
    let residuals = call(mf.out_res..mf.out_noise);
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
    let noise = mf
        .noise
        .iter()
        .zip(noise_vals.chunks(2))
        .map(|(n, pf)| NoiseSource {
            hi: remap_sym(n.hi, &map, ctx),
            lo: remap_sym(n.lo, &map, ctx),
            psd: pf[0],
            flicker_exp: pf[1],
            table: n.table.clone(),
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
    let limits = mf
        .limits
        .iter()
        .map(|l| FragmentLimit {
            hi: remap_sym(l.hi, &map, ctx),
            lo: remap_sym(l.lo, &map, ctx),
            kind: l.kind,
        })
        .collect();
    lo.delays.extend(inst_delays);
    BehavioralFragment {
        terminal_currents,
        residuals,
        noise,
        events,
        param_syms,
        op_vars,
        limits,
    }
}

/// `(module, terminal pattern, multiplicity)` -- see module docs. `mfactor`
/// is in the key because it is baked into the graph (every flow scales by
/// it), so instances of different multiplicity need distinct functions. The
/// parameter values and `$param_given` answers a function depends on are
/// checked per function within the key (see [`Bucket`]).
fn cache_key(dev: &VerilogADevice, ctx: &Graph, term_v: &[ExprId]) -> String {
    let pattern = terminal_pattern(ctx, term_v);
    format!(
        "va\u{1}{}\u{1}{}\u{1}{}",
        dev.module.name,
        pattern,
        dev.mfactor.to_bits()
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

fn instance_param_env(dev: &VerilogADevice) -> HashMap<String, f64> {
    let mut env: HashMap<String, f64> = dev
        .module
        .params
        .iter()
        .map(|p| (p.name.clone(), p.default))
        .collect();
    for (k, v) in &dev.params {
        env.insert(k.clone(), *v);
    }
    env
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
