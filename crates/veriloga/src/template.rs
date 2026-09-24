//! Model-instance templating.
//!
//! A Verilog-A compact model (BSIM4 is ~10k lines) lowers its analog block into
//! thousands of DAG nodes. Doing that per transistor dominates DAE assembly. But
//! every instance of the same model with the same *structure* lowers to a graph
//! that is identical up to its leaf symbols: terminal node voltages, parameter
//! symbols (`{inst}.{param}`), and the minted internal unknowns. So we lower one
//! representative instance, cache the result, and clone every further instance by
//! substituting those leaves (`substitute_many`, a cheap memoised DAG copy)
//! instead of re-walking the analog block.
//!
//! Correctness rests on the cache: two instances share a template only if
//! their lowering is provably identical. The key is
//! `(module, terminal-connectivity pattern, multiplicity)`, and under it each
//! template carries the values of the parameters that decided its structure
//! (and, for `$param_given`, whether they were set):
//! - a parameter's value enters the graph only through a structural decision
//!   (a branch, a loop bound, a static short, a folded constant); everywhere
//!   else it is the instance's parameter symbol. The lowering records which
//!   parameters its decisions read, through the constant shadow of every
//!   variable they went into;
//! - the terminal pattern captures which terminals are ground and which are tied
//!   together, since those collapse `V(a,b)` terms and change the graph shape.
//! Same key and the same values on those parameters ⇒ the same decisions ⇒ the
//! same graph up to leaf symbols, which substitution restores. Instances that
//! differ only in the others (a transistor's W and L, say) share one template
//! and so one function body, and their parameters are its arguments.
//!
//! `SANE_NO_TEMPLATE` in the environment bypasses the cache (every instance is
//! lowered directly) -- the escape hatch and the differential cross-check.

use rustc_hash::FxHashMap as HashMap;

use rsdag::{Crossing, ExprId, FuncId, SymbolId};
use rustc_hash::FxHashMap;
use sane_core::Graph;
use sane_device::{
    BehavioralFragment, FragmentEvent, FragmentLimit, LoweredDelay, Lowerer, NoiseSource, OpVar,
};

use crate::device::VerilogADevice;
use crate::lower::{lower_analog_structural, sym_of};

/// A minted extra unknown, recorded so a cloned instance re-mints the same one.
#[derive(Clone)]
struct ExtraMeta {
    /// Name with the template instance's `{inst}.` prefix stripped.
    suffix: String,
    kind: sane_device::UnknownKind,
    value_sym: SymbolId,
    xdot_sym: SymbolId,
    differential: bool,
    /// DC Newton seed carried by the state (`idt(u, ic)` with constant ic).
    dc_seed: Option<f64>,
}

/// A recorded `absdelay`: extras-relative positions plus the template's history
/// symbol (re-minted per clone from `hist_suffix`) and delay expression.
#[derive(Clone)]
struct TemplDelay {
    src_extra: usize,
    out_extra: usize,
    hist: SymbolId,
    hist_suffix: String,
    tau: ExprId,
}

#[derive(Clone)]
struct TemplNoise {
    hi: Option<SymbolId>,
    lo: Option<SymbolId>,
    table: Vec<(f64, f64)>,
}

/// One lowered representative of a model+structure, cloned for sibling instances.
#[derive(Clone)]
struct VaTemplate {
    terminal_currents: Vec<ExprId>,
    noise: Vec<TemplNoise>,
    /// `$limit` sites (node symbols remapped per clone, like the noise nodes).
    limits: Vec<FragmentLimit>,
    /// Operating-point variables: (name suffix, desc, units, value expression).
    op_vars: Vec<(String, String, Option<String>, ExprId)>,
    /// Minted extras in mint order (internal nodes, probe currents, idt states).
    extras: Vec<ExtraMeta>,
    /// Transport delays minted alongside the extras (`absdelay`).
    delays: Vec<TemplDelay>,
    /// Switching-surface directions (the surfaces are function outputs).
    events: Vec<Crossing>,
    /// Per-terminal value / derivative symbols (`None` = ground or tied-off).
    terminal_syms: Vec<Option<SymbolId>>,
    terminal_vdot_syms: Vec<Option<SymbolId>>,
    /// Parameters actually present in the graph: (name, template-instance symbol).
    param_syms: Vec<(String, SymbolId)>,
    /// The template as a function of the graph: one output per terminal
    /// current, residual, noise density pair, op-var value, delay time and
    /// switching surface, in that order (`out_cur`, `out_res`, `out_noise`,
    /// `out_opvar`, `out_tau`, `out_event` are the offsets), over the leaves
    /// as parameters.
    func: FuncId,
    leaves: Vec<SymbolId>,
    out_res: u32,
    out_noise: u32,
    out_opvar: u32,
    out_tau: u32,
    out_event: u32,
}

/// Instance batching is on unless disabled (`Config::device_bundles`; with it
/// off every instance inlines the template -- the differential reference).
fn bundling() -> bool {
    sane_core::config().device_bundles
}

/// The argument list binding a template's leaves for one instance: the
/// instance's expression where `map` has one, the leaf itself (a shared
/// global such as `$temp` or `t`) otherwise.
fn call_args(
    ctx: &mut Graph,
    leaves: &[SymbolId],
    map: &FxHashMap<SymbolId, ExprId>,
) -> Vec<ExprId> {
    leaves
        .iter()
        .map(|&s| map.get(&s).copied().unwrap_or_else(|| ctx.symbol_expr(s)))
        .collect()
}

/// The outputs `outs` of the template applied to `args`: calls when instance
/// batching is on, inlined expressions (one substitution pass) otherwise.
fn apply(
    ctx: &mut Graph,
    tpl: &VaTemplate,
    outs: &[u32],
    args: &[ExprId],
    inline: bool,
) -> Vec<ExprId> {
    if inline {
        ctx.inline_outputs(tpl.func, outs, args)
    } else {
        outs.iter().map(|&k| ctx.call(tpl.func, k, args)).collect()
    }
}

/// Lower `dev` either by cloning a cached template or, on the first instance of
/// its (module, structure), by lowering directly and caching the result.
pub(crate) fn lower_templated(
    dev: &VerilogADevice,
    lo: &mut Lowerer,
    term_v: &[ExprId],
    term_vdot: &[ExprId],
) -> BehavioralFragment {
    if !sane_core::config().device_templates {
        return lower_direct(dev, lo, term_v, term_vdot).0;
    }

    let key = cache_key(dev, lo.ctx(), term_v);
    let bucket = match lo.cache_get::<Bucket>(&key) {
        Some(b) => b,
        None => {
            let b = Bucket::default();
            lo.cache_put(key, b.clone());
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
    let hit = bucket.borrow().iter().find_map(|(sig, tpl)| {
        sig.iter()
            .all(|(name, bits)| bits_of(name) == *bits)
            .then(|| tpl.clone())
    });
    if let Some(tpl) = hit {
        let t = sane_core::time::Instant::now();
        let r = instantiate(&tpl, dev, lo, term_v, term_vdot);
        sane_core::profile::record_tpl_clone(t.elapsed().as_nanos());
        return r;
    }

    // First instance of this (module, structure): lower directly, then capture a
    // template from the freshly minted extras (the loop's `lo.extras` was emptied
    // by the assembler after the previous instance, so what is there now is ours).
    let t = sane_core::time::Instant::now();
    let (mut frag, structural) = lower_direct(dev, lo, term_v, term_vdot);
    // The template holds for every instance with these values on the
    // parameters that decided its structure.
    let sig: Vec<(String, u64)> = structural
        .into_iter()
        .map(|name| {
            let bits = bits_of(&name);
            (name, bits)
        })
        .collect();
    sane_core::log::debug(&format!(
        "template '{}' ({}): structure fixed by {} parameters: {}",
        dev.module.name,
        dev.name,
        sig.len(),
        sig.iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    ));
    let tpl = build_template(dev, lo, term_v, term_vdot, &frag);
    // On a multiply-instantiated module the REPRESENTATIVE also routes through
    // calls (identity argument binding), so all instances execute in the
    // shared body and its SIMD lanes -- otherwise instance 1 would stay a
    // scalar copy carrying the whole model once more. A single-instance
    // module keeps its fully symbolic fragment: no lanes to share, and the
    // circuit stays transparent to symbolic tooling.
    let multi = lo
        .instance_groups
        .get(&dev.module.name)
        .is_some_and(|&n| n >= 2);
    if multi && bundling() {
        let ctx = lo.ctx();
        let identity = FxHashMap::default();
        let args = call_args(ctx, &tpl.leaves, &identity);
        let n_cur = frag.terminal_currents.len() as u32;
        let cur: Vec<u32> = (0..n_cur).collect();
        frag.terminal_currents = apply(ctx, &tpl, &cur, &args, false);
        let res: Vec<u32> = (tpl.out_res..tpl.out_noise).collect();
        frag.residuals = apply(ctx, &tpl, &res, &args, false);
        for (j, n) in frag.noise.iter_mut().enumerate() {
            let k = tpl.out_noise + 2 * j as u32;
            n.psd = ctx.call(tpl.func, k, &args);
            n.flicker_exp = ctx.call(tpl.func, k + 1, &args);
        }
        for (j, v) in frag.op_vars.iter_mut().enumerate() {
            v.value = ctx.call(tpl.func, tpl.out_opvar + j as u32, &args);
        }
        for (j, e) in frag.events.iter_mut().enumerate() {
            e.g = ctx.call(tpl.func, tpl.out_event + j as u32, &args);
        }
    }
    bucket.borrow_mut().push((sig, tpl));
    sane_core::profile::record_tpl_build(t.elapsed().as_nanos());
    frag
}

/// The templates of one (module, terminal pattern, multiplicity), each with
/// the values of the parameters that decided its structure (`$given(X)`:
/// whether `X` was set, as 0 or 1).
type Bucket = std::rc::Rc<std::cell::RefCell<Vec<(Vec<(String, u64)>, VaTemplate)>>>;

fn lower_direct(
    dev: &VerilogADevice,
    lo: &mut Lowerer,
    term_v: &[ExprId],
    term_vdot: &[ExprId],
) -> (BehavioralFragment, std::collections::BTreeSet<String>) {
    match lower_analog_structural(
        &dev.module,
        &dev.name,
        &dev.params,
        &dev.given,
        dev.mfactor,
        lo,
        term_v,
        term_vdot,
    ) {
        Ok(r) => r,
        Err(e) => {
            // validate() runs at load time, so an unsupported construct is already
            // a parse error; reaching here means a real bug -- fail loudly.
            sane_core::log::error(&format!("Verilog-A lowering of '{}': {e}", dev.module.name));
            panic!("Verilog-A lowering of '{}': {e}", dev.module.name);
        }
    }
}

/// `(module, terminal pattern, multiplicity)` -- see module docs. `mfactor`
/// is in the key because it is baked into the graph (every flow scales by
/// it), so instances of different multiplicity need distinct templates. The
/// parameter values and `$param_given` answers a template depends on are
/// checked per template within the key (see [`Bucket`]).
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

fn build_template(
    dev: &VerilogADevice,
    lo: &mut Lowerer,
    term_v: &[ExprId],
    term_vdot: &[ExprId],
    frag: &BehavioralFragment,
) -> VaTemplate {
    let extras = lo
        .extras
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

    // Delay registry (positions are device-relative: `lo.delays`, like
    // `lo.extras`, was drained by the assembler after the previous instance).
    let delays: Vec<TemplDelay> = lo
        .delays
        .iter()
        .map(|d| TemplDelay {
            src_extra: d.src_extra,
            out_extra: d.out_extra,
            hist: d.hist,
            hist_suffix: d.hist_suffix.clone(),
            tau: d.tau,
        })
        .collect();

    let ctx = lo.ctx();
    let terminal_syms: Vec<Option<SymbolId>> = term_v.iter().map(|&e| sym_of(ctx, e)).collect();
    let terminal_vdot_syms: Vec<Option<SymbolId>> =
        term_vdot.iter().map(|&e| sym_of(ctx, e)).collect();

    let param_syms: Vec<(String, SymbolId)> = frag.param_syms.clone();

    let noise = frag
        .noise
        .iter()
        .map(|n| TemplNoise {
            hi: n.hi,
            lo: n.lo,
            table: n.table.clone(),
        })
        .collect();

    // The template as a function: every per-instance quantity is one output
    // over the leaves (free symbols of all outputs, in deterministic sorted
    // id order); an instance is a call with its own leaf bindings.
    let mut outs: Vec<ExprId> = frag.terminal_currents.clone();
    let out_res = outs.len() as u32;
    outs.extend(frag.residuals.iter().copied());
    let out_noise = outs.len() as u32;
    for n in &frag.noise {
        outs.push(n.psd);
        outs.push(n.flicker_exp);
    }
    let out_opvar = outs.len() as u32;
    outs.extend(frag.op_vars.iter().map(|v| v.value));
    let out_tau = outs.len() as u32;
    outs.extend(delays.iter().map(|d| d.tau));
    let out_event = outs.len() as u32;
    outs.extend(frag.events.iter().map(|e| e.g));
    let leaves: Vec<SymbolId> = ctx.free_symbols_in(&outs).into_iter().collect();
    let func = ctx.define_func(&dev.module.name, leaves.clone(), outs);

    let op_vars = frag
        .op_vars
        .iter()
        .map(|v| (v.short.clone(), v.desc.clone(), v.units.clone(), v.value))
        .collect();

    VaTemplate {
        terminal_currents: frag.terminal_currents.clone(),
        noise,
        limits: frag.limits.clone(),
        op_vars,
        extras,
        delays,
        events: frag.events.iter().map(|e| e.dir).collect(),
        terminal_syms,
        terminal_vdot_syms,
        param_syms,
        func,
        leaves,
        out_res,
        out_noise,
        out_opvar,
        out_tau,
        out_event,
    }
}

fn instantiate(
    tpl: &VaTemplate,
    dev: &VerilogADevice,
    lo: &mut Lowerer,
    term_v: &[ExprId],
    term_vdot: &[ExprId],
) -> BehavioralFragment {
    let mut map: FxHashMap<SymbolId, ExprId> = FxHashMap::default();

    // Re-mint the extras in the template's order so the DAE rows line up, and map
    // each template extra symbol onto this instance's freshly minted one.
    for ex in &tpl.extras {
        let (val, xdot) = lo.unknown_kind_of(&dev.name, &ex.suffix, ex.kind, ex.differential);
        lo.extras.last_mut().expect("just minted").dc_seed = ex.dc_seed;
        map.insert(ex.value_sym, val);
        map.insert(ex.xdot_sym, xdot);
    }
    for k in 0..term_v.len() {
        if let Some(s) = tpl.terminal_syms[k] {
            map.insert(s, term_v[k]);
        }
        if let Some(s) = tpl.terminal_vdot_syms[k] {
            map.insert(s, term_vdot[k]);
        }
    }

    let ctx = lo.ctx();
    // The clone's parameter symbols: the template's, mapped to this instance.
    let mut param_syms: Vec<(String, SymbolId)> = Vec::with_capacity(tpl.param_syms.len());
    for (name, tsym) in &tpl.param_syms {
        let isym = ctx.sym(&format!("{}.{}", dev.name, name));
        map.insert(*tsym, isym);
        if let Some(s) = sym_of(ctx, isym) {
            param_syms.push((name.clone(), s));
        }
    }

    // Re-mint per-instance history symbols so the substituted residuals bind
    // this instance's delay inputs (the map must be complete before the
    // substitution pass below).
    let hist_syms: Vec<(SymbolId, String)> = tpl
        .delays
        .iter()
        .map(|dl| {
            let he = ctx.sym(&format!("{}.{}", dev.name, dl.hist_suffix));
            let hist = sym_of(ctx, he).expect("sym() yields a Symbol node");
            map.insert(dl.hist, he);
            (hist, dl.hist_suffix.clone())
        })
        .collect();

    // Every per-instance quantity is an output of the template function
    // applied to this instance's argument list: a call when instance batching
    // is on (one node each; the solver evaluates the shared body per instance
    // and per lane), otherwise the inlined expression, all outputs in one
    // substitution pass.
    let args = call_args(ctx, &tpl.leaves, &map);
    let inline = !bundling();
    let n_cur = tpl.terminal_currents.len() as u32;
    let cur: Vec<u32> = (0..n_cur).collect();
    let terminal_currents = apply(ctx, tpl, &cur, &args, inline);
    let res: Vec<u32> = (tpl.out_res..tpl.out_noise).collect();
    let residuals = apply(ctx, tpl, &res, &args, inline);
    let nz: Vec<u32> = (tpl.out_noise..tpl.out_opvar).collect();
    let noise_vals = apply(ctx, tpl, &nz, &args, inline);
    let ov: Vec<u32> = (tpl.out_opvar..tpl.out_tau).collect();
    let opvar_vals = apply(ctx, tpl, &ov, &args, inline);
    let tv: Vec<u32> = (tpl.out_tau..tpl.out_event).collect();
    let taus = apply(ctx, tpl, &tv, &args, inline);
    let n_out = tpl.out_event + tpl.events.len() as u32;
    let ev: Vec<u32> = (tpl.out_event..n_out).collect();
    let events: Vec<FragmentEvent> = apply(ctx, tpl, &ev, &args, inline)
        .into_iter()
        .zip(&tpl.events)
        .map(|(g, &dir)| FragmentEvent { g, dir })
        .collect();

    let inst_delays: Vec<LoweredDelay> = tpl
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

    let noise: Vec<NoiseSource> = tpl
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

    let op_vars: Vec<OpVar> = tpl
        .op_vars
        .iter()
        .zip(opvar_vals)
        .map(|((suffix, desc, units, _), value)| OpVar {
            name: format!("{}.{}", dev.name, suffix),
            short: suffix.clone(),
            desc: desc.clone(),
            units: units.clone(),
            value,
        })
        .collect();

    let limits: Vec<FragmentLimit> = tpl
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

/// Remap a noise generator's node symbol through the substitution (its target is
/// always a node-voltage / internal symbol). Unmapped symbols (ground) pass through.
fn remap_sym(
    s: Option<SymbolId>,
    map: &FxHashMap<SymbolId, ExprId>,
    ctx: &Graph,
) -> Option<SymbolId> {
    s.map(|sym| match map.get(&sym) {
        Some(&e) => sym_of(ctx, e).unwrap_or(sym),
        None => sym,
    })
}
