//! Assembly of the symbolic DAE from a parsed circuit: node KCL rows,
//! branch constraints, device lowering (every device -- built-in Verilog-A,
//! user Verilog-A, OSDI, behavioral -- through the one `lower_behavioral`
//! fragment path), noise generators, and the companion / limit / source
//! registries the solver reads.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use rsdag::{CmpOp, ExprId, FuncId, ParamRole, ReduceOp, SymbolId};
use sane_circuit::{BExpr, BKind, Circuit, Element, Instance, Kind, SourceFn};
use sane_core::Graph;
use sane_device::Lowerer;

use crate::observers::Observers;
use crate::{sym2, Dae, DelaySpec, EventSpec, NoiseSource, UnknownKind};

/// Symbol for an element's value parameter, mapped out of the reserved unknown
/// namespace (see [`sane_circuit::value_symbol_name`]) so e.g. a voltage source
/// named `v91` cannot be hash-consed onto node 91's voltage unknown.
fn value_sym(ctx: &mut Graph, name: &str) -> ExprId {
    ctx.sym(&sane_circuit::value_symbol_name(name))
}

/// The (possibly time-dependent) value of an independent source.
///
/// Numeric parameters are instance-scoped symbols (bound separately), so the
/// expression stays symbolic. Region splits use `Select`; periodicity uses
/// `floor`.
fn source_value(ctx: &mut Graph, e: &Element, t: ExprId) -> ExprId {
    // The constitutive waveform lives with the source type (see `sane_circuit::SourceFn`);
    // a constant element (no source shape) is just its own value symbol.
    match e.source {
        None => value_sym(ctx, &e.name),
        Some(src) => src.lower(ctx, &e.name, t),
    }
}

/// Accumulate a branch current `c` (or a charge, into the charge lists) into
/// the KCL term lists: `+c` leaves node `a`, `-c` enters node `b` (ground = 0
/// is skipped). The per-node lists are folded into one fused `Reduce(Sum)` at
/// the end, instead of a binary Add-tree.
fn add_current(ctx: &mut Graph, node_terms: &mut [Vec<ExprId>], a: usize, b: usize, c: ExprId) {
    if a != 0 {
        node_terms[a - 1].push(c);
    }
    if b != 0 {
        let nc = ctx.neg(c);
        node_terms[b - 1].push(nc);
    }
}

/// Translate a behavioral (`B`) source expression tree into the symbolic graph,
/// resolving `V(node)` to the node-voltage expression, `I(elem)` to the branch
/// current, `time` to the time symbol, and other identifiers to free parameter
/// symbols. The result is an ordinary expression, differentiable like any other.
fn translate_bexpr(
    ctx: &mut Graph,
    e: &BExpr,
    v: &[ExprId],
    branch_i_of_name: &HashMap<String, ExprId>,
    t_e: ExprId,
) -> ExprId {
    match e {
        BExpr::Const(c) => ctx.konst_f64(*c),
        BExpr::NodeV(n) => v[*n],
        BExpr::BranchI(name) => branch_i_of_name
            .get(&name.to_ascii_lowercase())
            .copied()
            .unwrap_or_else(|| ctx.zero()),
        BExpr::Param(name) => {
            if name.eq_ignore_ascii_case("time") {
                t_e
            } else {
                ctx.sym(name)
            }
        }
        BExpr::Neg(a) => {
            let x = translate_bexpr(ctx, a, v, branch_i_of_name, t_e);
            ctx.neg(x)
        }
        BExpr::Bin(op, a, b) => {
            let x = translate_bexpr(ctx, a, v, branch_i_of_name, t_e);
            let y = translate_bexpr(ctx, b, v, branch_i_of_name, t_e);
            match op {
                '+' => ctx.add(x, y),
                '-' => ctx.sub(x, y),
                '*' => ctx.mul(x, y),
                '/' => ctx.div(x, y),
                '^' => sane_core::mathfn::pow(ctx, x, y),
                // Comparison sentinels (see `sane_netlist::behavioral::cmp_sentinel`):
                // each yields 1.0 / 0.0, to be consumed by `if(cond, then, else)`.
                '<' => ctx.cmp(CmpOp::Lt, x, y),
                '>' => ctx.cmp(CmpOp::Gt, x, y),
                'l' => ctx.cmp(CmpOp::Le, x, y),
                'g' => ctx.cmp(CmpOp::Ge, x, y),
                'e' => ctx.cmp(CmpOp::Eq, x, y),
                'n' => ctx.cmp(CmpOp::Ne, x, y),
                _ => x,
            }
        }
        BExpr::Call(name, args) => {
            let a: Vec<ExprId> = args
                .iter()
                .map(|e| translate_bexpr(ctx, e, v, branch_i_of_name, t_e))
                .collect();
            match name.as_str() {
                "if" if a.len() == 3 => ctx.select(a[0], a[1], a[2]),
                // Everything else is the shared elementary-function table in
                // `sane_core::mathfn` (`pwr` is the B-source spelling of `pow`).
                // The `None` arm is unreachable for parsed netlists: the netlist
                // parser validates every B-source function name and arity (see
                // `sane_netlist::behavioral::validate_call`), so an unknown name
                // is a hard parse error, never a silent zero. It only guards a
                // programmatically constructed `BExpr`.
                other => {
                    let canonical = if other == "pwr" { "pow" } else { other };
                    sane_core::lower_math_call(ctx, canonical, &a).unwrap_or_else(|| ctx.zero())
                }
            }
        }
    }
}

/// An unknown a body mints beyond its nodes (a branch current, a device
/// extra), with its row `current + d/dt charge = 0`.
struct Unknown {
    name: String,
    kind: UnknownKind,
    x: SymbolId,
    current: ExprId,
    /// Zero for an algebraic row.
    charge: ExprId,
}

/// A row of a part a noise generator enters: a node's KCL (`k - 1` for
/// node `k`), or an extra's (a device's internal node or branch).
#[derive(Clone, Copy)]
enum RowRef {
    Node(usize),
    Extra(usize),
}

/// A body assembled over its own nodes: the current terms into each node and
/// everything else it contributes, in its own names. The top level becomes
/// the DAE; a subcircuit body becomes a function.
#[derive(Default)]
struct Part {
    /// Per node (`k - 1` for node `k`), the signed current and charge terms
    /// leaving it.
    node_terms: Vec<Vec<ExprId>>,
    node_charges: Vec<Vec<ExprId>>,
    branches: Vec<Unknown>,
    extras: Vec<Unknown>,
    /// The noise generators entering the rows: the row, the generator and
    /// its coefficient there. The rows above are at rest; the top level adds
    /// `coefficient * generator` to its own, a subcircuit body hands them to
    /// its instances (see `Lowered::noise`), so no call carries a generator.
    noise_terms: Vec<(RowRef, SymbolId, ExprId)>,
    /// Transport delays: the extra (body-relative) whose value is delayed,
    /// the history symbol, the delay.
    delays: Vec<(usize, SymbolId, ExprId)>,
    events: Vec<EventSpec>,
    /// The noise sources and op-vars: its own, and its instances'.
    observers: Observers,
    param_defaults: rustc_hash::FxHashMap<SymbolId, f64>,
    dc_seeds: Vec<(String, f64)>,
    /// `(row, col, g)` over node indices `k - 1`.
    companion: Vec<(usize, usize, f64)>,
    limits: Vec<sane_device::FragmentLimit>,
    sources: Vec<(String, SourceFn)>,
    source_names: Vec<String>,
    /// Display names (see [`Dae::labels`]).
    labels: Vec<(ExprId, String)>,
    /// See [`Dae::assertions`].
    assertions: Vec<sane_device::Assertion>,
    /// See [`Dae::structure`].
    structure: Vec<sane_device::Assertion>,
    /// The collapsed internal nodes and their voltages (see
    /// [`Dae::aliases`]).
    aliases: Vec<(String, ExprId)>,
}

/// The subcircuit functions built so far, by what they compute: a body's
/// outputs over its parameters. Instances whose bodies lower to the same
/// expressions share one function.
#[derive(Default)]
struct Bodies {
    funcs: HashMap<(Vec<ExprId>, Vec<SymbolId>), FuncId>,
    /// The subcircuit bodies assembled so far, by the body.
    lowered: HashMap<*const Circuit, Rc<Lowered>>,
    /// Per body, at how many places of the hierarchy it is instantiated
    /// (each body counted once, however often its parent is).
    sites: HashMap<*const Circuit, usize>,
    /// Time spent in device `lower_behavioral`, for the stage log.
    lower_t: std::time::Duration,
}

/// Per body under `c`, at how many places it is instantiated: each body
/// visited once, so an instance in a body counts once however often the
/// body is instantiated itself.
fn count_sites(
    c: &Circuit,
    seen: &mut HashSet<*const Circuit>,
    sites: &mut HashMap<*const Circuit, usize>,
) {
    for inst in &c.instances {
        let body = Arc::as_ptr(&inst.body);
        *sites.entry(body).or_default() += 1;
        if seen.insert(body) {
            count_sites(&inst.body, seen, sites);
        }
    }
}

fn sym_of(ctx: &Graph, e: ExprId) -> Option<SymbolId> {
    match ctx.node(e) {
        rsdag::Node::Symbol(s) => Some(*s),
        _ => None,
    }
}

/// Assemble the DAE of a circuit with its subcircuit instances. Every node
/// of the hierarchy is a top-level node (`v{k}`); every other unknown keeps
/// its block (branch currents, then device extras) in the instance's names.
/// An error names the device whose model does not lower.
pub fn assemble(ctx: &mut Graph, circuit: &Circuit) -> Result<Dae, String> {
    assemble_at(ctx, circuit, &|_| None)
}

/// [`assemble`] with the devices' structure decided at `values` (by
/// parameter symbol name) where those set a parameter.
pub fn assemble_at(
    ctx: &mut Graph,
    circuit: &Circuit,
    values: &dyn Fn(&str) -> Option<f64>,
) -> Result<Dae, String> {
    let n = circuit.node_count();
    let zero = ctx.zero();
    let (t_e, t) = sym2(ctx, "t");
    // Node voltage symbols (index 0 = ground = 0).
    let mut v = vec![zero; n + 1];
    let mut x = Vec::new();
    for k in 1..=n {
        let (e, s) = sym2(ctx, &format!("v{k}"));
        v[k] = e;
        x.push(s);
    }

    let dev_t0 = sane_core::time::Instant::now();
    let mut bodies = Bodies::default();
    count_sites(circuit, &mut HashSet::new(), &mut bodies.sites);
    let mut lo = Lowerer::at(ctx, values);
    let part = lower_body(&mut lo, &mut bodies, circuit, &v, t_e)?;
    drop(lo); // release the &mut Graph borrow before reusing `ctx` below
    sane_core::log::stage("dae/devices", dev_t0.elapsed());
    sane_core::log::stage("dae/devices_lower", bodies.lower_t);
    let (tpl_b, tpl_c, n_b, n_c) = sane_core::profile::take_tpl_stats();
    sane_core::log::stage(
        "dae/tpl_build",
        std::time::Duration::from_nanos(tpl_b as u64),
    );
    sane_core::log::stage(
        "dae/tpl_clone",
        std::time::Duration::from_nanos(tpl_c as u64),
    );
    sane_core::log::debug(&format!(
        "dae/devices: {n_b} template builds, {n_c} clones, {} subcircuit functions",
        bodies.funcs.len()
    ));

    // Layout: node KCL, then branch constraints, then device extras. The
    // rows at rest, and the rows with every noise generator where it enters
    // (`coefficient * generator`): the DAE's own.
    let rows_t0 = sane_core::time::Instant::now();
    let nb = part.branches.len();
    let mut noise: Vec<Vec<ExprId>> = vec![Vec::new(); n + nb + part.extras.len()];
    for &(row, g, coeff) in &part.noise_terms {
        let r = match row {
            RowRef::Node(k) => k,
            RowRef::Extra(j) => n + nb + j,
        };
        let g = ctx.symbol_expr(g);
        noise[r].push(ctx.mul(coeff, g));
    }
    let mut rest: Vec<ExprId> = (part.node_terms.iter())
        .map(|terms| ctx.reduce(ReduceOp::Sum, terms.clone()))
        .collect();
    rest.extend(part.branches.iter().chain(&part.extras).map(|u| u.current));
    let currents: Vec<ExprId> = (rest.iter().zip(&noise).enumerate())
        .map(|(r, (&row, ns))| match (ns.is_empty(), r < n) {
            (true, _) => row,
            // a node's terms and the generators in one sum
            (false, true) => {
                let terms = part.node_terms[r].iter().chain(ns).copied().collect();
                ctx.reduce(ReduceOp::Sum, terms)
            }
            (false, false) => {
                let terms = std::iter::once(row).chain(ns.iter().copied()).collect();
                ctx.reduce(ReduceOp::Sum, terms)
            }
        })
        .collect();
    let mut charges: Vec<ExprId> = part
        .node_charges
        .into_iter()
        .map(|terms| ctx.reduce(ReduceOp::Sum, terms))
        .collect();
    charges.extend(part.branches.iter().chain(&part.extras).map(|u| u.charge));

    let mut unknowns: Vec<String> = (1..=n).map(|k| format!("v{k}")).collect();
    let mut kinds = vec![UnknownKind::NodeVoltage; n];
    for u in &part.branches {
        unknowns.push(u.name.clone());
        kinds.push(UnknownKind::BranchCurrent);
        x.push(u.x);
    }
    for u in &part.extras {
        unknowns.push(u.name.clone());
        kinds.push(u.kind);
        x.push(u.x);
    }
    sane_core::log::stage("dae/rows", rows_t0.elapsed());
    let mut events = part.events;
    let n_rows = currents.len();
    let both: Vec<ExprId> = currents.into_iter().chain(rest).collect();
    // a circuit without devices and instances has no call to specialize
    let spec_t0 = sane_core::time::Instant::now();
    let (mut currents, charges) = if circuit.devices.is_empty() && circuit.instances.is_empty() {
        (both, charges)
    } else {
        specialized(ctx, both, charges, &mut events)
    };
    sane_core::log::stage("dae/specialize", spec_t0.elapsed());
    let rest = Arc::new((currents.split_off(n_rows), charges.clone()));

    // a delay's source is the extra unknown it was minted as
    let extra_base = n + part.branches.len();
    let delays: Vec<DelaySpec> = (part.delays.iter())
        .map(|&(src, hist, tau)| DelaySpec {
            src: ctx.symbol_expr(x[extra_base + src]),
            hist,
            tau,
        })
        .collect();

    let unknown_of: HashMap<SymbolId, usize> = x.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    // the companion network by node voltage (its indices are node rows)
    let companion = (part.companion.iter())
        .map(|&(r, c, g)| (x[r], x[c], g))
        .collect();

    // A collapsed node's voltage is a kept node's, or ground's.
    let aliases = (part.aliases.iter())
        .filter_map(|&(ref name, v)| match ctx.node(v) {
            rsdag::Node::Symbol(s) => unknown_of
                .get(s)
                .map(|&k| (name.clone(), Some(unknowns[k].clone()))),
            _ if ctx.const_f64(v) == Some(0.0) => Some((name.clone(), None)),
            _ => None,
        })
        .collect();
    // The circuit temperature is nominal unless the circuit states it.
    let mut param_defaults = part.param_defaults;
    let (_, temp) = sym2(ctx, sane_core::constants::TEMP_SYMBOL);
    param_defaults.insert(temp, sane_core::constants::TEMP_NOMINAL_K);
    Ok(Dae {
        currents,
        charges,
        assertions: part.assertions,
        structure: part.structure,
        aliases,
        n_nodes: n,
        param_defaults,
        events,
        delays,
        unknowns,
        kinds,
        x,
        t,
        companion,
        observers: part.observers,
        dc_seeds: part.dc_seeds,
        limits: part.limits,
        sources: part.sources,
        source_names: part.source_names,
        labels: part.labels.into_iter().collect(),
        injection: Default::default(),
        rest: std::sync::OnceLock::from(rest),
    })
}

/// The currents, the charges and the switching surfaces with every call that
/// passes constants (a ground terminal of a subcircuit) specialized to them,
/// in one pass over all, so a body is specialized once.
fn specialized(
    ctx: &mut Graph,
    currents: Vec<ExprId>,
    charges: Vec<ExprId>,
    events: &mut [EventSpec],
) -> (Vec<ExprId>, Vec<ExprId>) {
    let (nc, nq) = (currents.len(), charges.len());
    let roots: Vec<ExprId> = (currents.into_iter().chain(charges))
        .chain(events.iter().map(|e| e.g))
        .collect();
    let mut out = ctx.specialize_calls(&roots);
    for (ev, g) in events.iter_mut().zip(out.split_off(nc + nq)) {
        ev.g = g;
    }
    let charges = out.split_off(nc);
    (out, charges)
}

/// Assemble one body over node voltages `v` (index 0 is ground): its
/// elements, devices and subcircuit instances.
#[allow(clippy::too_many_arguments)]
fn lower_body(
    lo: &mut Lowerer,
    bodies: &mut Bodies,
    c: &Circuit,
    v: &[ExprId],
    t_e: ExprId,
) -> Result<Part, String> {
    let (circuit, devices, instances) = (&c.elements, &c.devices[..], &c.instances[..]);
    let n = v.len() - 1;
    let mut part = Part {
        node_terms: vec![Vec::new(); n],
        node_charges: vec![Vec::new(); n],
        ..Part::default()
    };
    let ctx = lo.ctx();

    // Branch-current unknowns for voltage-defined / dynamic elements.
    let mut branch_elem_idx: Vec<usize> = Vec::new();
    let mut branch_i: Vec<(ExprId, SymbolId)> = Vec::new();
    let mut branch_i_of_name: HashMap<String, ExprId> = HashMap::new();
    for (idx, e) in circuit.elements().iter().enumerate() {
        if matches!(
            e.kind,
            Kind::VoltageSource | Kind::Inductor | Kind::Vcvs | Kind::Ccvs
        ) {
            let (ie, is) = sym2(ctx, &format!("i_{}", e.name));
            branch_elem_idx.push(idx);
            branch_i.push((ie, is));
            branch_i_of_name.insert(e.name.to_ascii_lowercase(), ie);
        }
    }
    let branch_pos = |idx: usize| branch_elem_idx.iter().position(|&j| j == idx).unwrap();

    // Behavioral `V=` sources are voltage-defined, so each also gets a
    // branch-current unknown (registered so `I(Bname)` can reference it).
    let mut bhv_v: Vec<(usize, ExprId, SymbolId)> = Vec::new();
    for (bi, b) in circuit.behavioral().iter().enumerate() {
        if b.kind == BKind::V {
            let (ie, is) = sym2(ctx, &format!("i_{}", b.name));
            branch_i_of_name.insert(b.name.to_ascii_lowercase(), ie);
            bhv_v.push((bi, ie, is));
        }
    }

    // KCL: collect the signed branch-current terms per node; they are fused
    // into one Reduce(Sum) per node row by the caller.
    let node_terms = &mut part.node_terms;
    for (idx, e) in circuit.elements().iter().enumerate() {
        match e.kind {
            Kind::Resistor => {
                let s = value_sym(ctx, &e.name);
                let g = ctx.recip(s);
                let dv = ctx.sub(v[e.a], v[e.b]);
                let c = ctx.mul(dv, g);
                add_current(ctx, node_terms, e.a, e.b, c);
            }
            Kind::Capacitor => {
                let cs = value_sym(ctx, &e.name);
                let dv = ctx.sub(v[e.a], v[e.b]);
                let q = ctx.mul(cs, dv);
                add_current(ctx, &mut part.node_charges, e.a, e.b, q);
            }
            Kind::Inductor | Kind::VoltageSource | Kind::Vcvs | Kind::Ccvs => {
                let il = branch_i[branch_pos(idx)].0;
                add_current(ctx, node_terms, e.a, e.b, il);
            }
            Kind::CurrentSource => {
                let iv = source_value(ctx, e, t_e);
                add_current(ctx, node_terms, e.a, e.b, iv);
            }
            Kind::Vccs => {
                let gm = value_sym(ctx, &e.name);
                let (cp, cm) = e.ctrl.expect("VCCS control nodes");
                let dvc = ctx.sub(v[cp], v[cm]);
                let c = ctx.mul(gm, dvc);
                add_current(ctx, node_terms, e.a, e.b, c);
            }
            Kind::Cccs => {
                // I(a->b) = gain * I(ctrl).
                let gain = value_sym(ctx, &e.name);
                let cname = e.ctrl_elem.as_deref().expect("CCCS needs controller");
                let ictrl = branch_i_of_name[&cname.to_ascii_lowercase()];
                let c = ctx.mul(gain, ictrl);
                add_current(ctx, node_terms, e.a, e.b, c);
            }
        }
    }

    // Resistor thermal noise: each resistor R is a current-noise generator
    // `R#noise` across its two nodes with white PSD 4*k_B*T/R, a current the
    // rows carry beside the resistor's. Every noise source, stamped or
    // behavioral, lives in one registry that the noise analysis consumes
    // uniformly. Temperature is the shared `$temp` symbol (not a fixed
    // constant), so a temperature sweep moves resistor noise exactly as it
    // moves diode/MOSFET noise.
    {
        let k4 = ctx.konst_f64(4.0 * sane_core::constants::BOLTZMANN);
        let temp = ctx.sym(sane_core::constants::TEMP_SYMBOL);
        let coeff = ctx.mul(k4, temp); // 4*k_B*T
        for (name, a, b) in circuit.resistors() {
            let r = value_sym(ctx, &name);
            let g = ctx.recip(r);
            let psd = ctx.mul(coeff, g);
            let flicker_exp = ctx.zero();
            let (_, input) = sym2(ctx, &format!("{name}#noise"));
            for (node, sign) in [(a, 1.0), (b, -1.0)] {
                if node != 0 {
                    let c = ctx.konst_f64(sign);
                    part.noise_terms.push((RowRef::Node(node - 1), input, c));
                }
            }
            part.observers.noise.push(NoiseSource {
                input,
                psd,
                flicker_exp,
                table: Vec::new(),
            });
        }
    }

    // Every nonlinear device lowers to a fragment that mints "extra" unknowns
    // (device-internal nodes for native models; branch currents / idt / laplace
    // states for behavioral ones) with one row each.
    for inst in devices {
        // Companion conductance network for homotopy continuation (node-KCL
        // row x node-voltage col): each device's linear `lambda = 0` form,
        // stamped as a conductance between its terminals (ground contributes
        // nothing).
        for (li, lj, g) in inst.model.companion() {
            let (a, b) = (inst.terminals[li], inst.terminals[lj]);
            if a != 0 {
                part.companion.push((a - 1, a - 1, g));
            }
            if b != 0 {
                part.companion.push((b - 1, b - 1, g));
            }
            if a != 0 && b != 0 {
                part.companion.push((a - 1, b - 1, -g));
                part.companion.push((b - 1, a - 1, -g));
            }
        }
        let term_v: Vec<ExprId> = inst.terminals.iter().map(|&nd| v[nd]).collect();
        let ctrl_i: Vec<ExprId> = inst
            .model
            .control_currents()
            .iter()
            .map(|nm| {
                *branch_i_of_name
                    .get(&nm.to_ascii_lowercase())
                    .unwrap_or_else(|| {
                        panic!("device controller '{nm}' is not a voltage-defined element")
                    })
            })
            .collect();
        // ONE lowering path for every device: a `BehavioralFragment` (terminal
        // currents and charges + one row per minted extra unknown), so internal
        // nodes and behavioral states are treated uniformly.
        let lt = sane_core::time::Instant::now();
        let mut frag =
            (inst.model.lower_behavioral(lo, &term_v, &ctrl_i)).map_err(|e| {
                match inst.model.instance_name() {
                    Some(name) => format!("{name}: {e}"),
                    None => e,
                }
            })?;
        bodies.lower_t += lt.elapsed();
        // Parallel multiplicity (`M=` * `nf`): scale the terminal currents by m,
        // modelling m identical devices in parallel. Internal-node rows stay
        // per-device (one representative internal state). Verilog-A devices apply
        // `$mfactor` internally, so their `inst.mfactor` is 1.0 (no double count).
        if inst.mfactor != 1.0 {
            let c = lo.ctx();
            let m = c.konst_f64(inst.mfactor);
            for ti in frag
                .terminal_currents
                .iter_mut()
                .chain(frag.terminal_charges.iter_mut())
            {
                *ti = c.mul(m, *ti);
            }
        }
        let dev_extra_base = part.extras.len();
        let extras = std::mem::take(&mut lo.extras);
        debug_assert_eq!(
            frag.currents.len(),
            extras.len(),
            "device fragment must return one row per extra unknown"
        );
        // Transport delays (`absdelay`) minted by this device: shift the
        // device-relative extras positions onto the body's extras.
        for dl in std::mem::take(&mut lo.delays) {
            part.delays
                .push((dev_extra_base + dl.src_extra, dl.hist, dl.tau));
        }
        let zero = lo.ctx().zero();
        for (j, (ex, &current)) in extras.iter().zip(&frag.currents).enumerate() {
            if let Some(c) = ex.dc_seed {
                part.dc_seeds.push((ex.name.clone(), c));
            }
            part.extras.push(Unknown {
                name: ex.name.clone(),
                kind: ex.kind,
                x: ex.value_sym,
                current,
                charge: frag.charges.get(j).copied().unwrap_or(zero),
            });
        }
        for (k, &nd) in inst.terminals.iter().enumerate() {
            if nd != 0 {
                part.node_terms[nd - 1].push(frag.terminal_currents[k]);
                if let Some(&q) = frag.terminal_charges.get(k) {
                    part.node_charges[nd - 1].push(q);
                }
            }
        }
        // Its noise generators where they enter: a terminal's row onto the
        // node it is on (scaled by the multiplicity like the terminal
        // currents, the density divided, so parallel devices' noise adds
        // uncorrelated), an extra's onto the extra.
        let n_term = inst.terminals.len();
        let m = (inst.mfactor != 1.0).then(|| lo.ctx().konst_f64(inst.mfactor));
        for (src, rows) in frag.noise.iter_mut().zip(&frag.noise_rows) {
            let ctx = lo.ctx();
            for &(r, coeff) in rows {
                if r < n_term {
                    let nd = inst.terminals[r];
                    if nd != 0 {
                        let coeff = m.map_or(coeff, |m| ctx.mul(m, coeff));
                        part.noise_terms
                            .push((RowRef::Node(nd - 1), src.input, coeff));
                    }
                } else {
                    part.noise_terms.push((
                        RowRef::Extra(dev_extra_base + r - n_term),
                        src.input,
                        coeff,
                    ));
                }
            }
            if let Some(m) = m {
                src.psd = ctx.div(src.psd, m);
                src.table = (src.table.iter())
                    .map(|&(f, p)| (f, ctx.div(p, m)))
                    .collect();
            }
        }
        part.observers.noise.extend(frag.noise);
        for (k, ev) in frag.events.iter().enumerate() {
            let inst_name = inst
                .model
                .instance_name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("device{}", part.events.len()));
            part.events.push(EventSpec {
                g: ev.g,
                dir: ev.dir,
                name: format!("{inst_name}#{k}"),
            });
        }
        for (name, sym) in &frag.param_syms {
            if let Some(v) = inst.model.param_default(name) {
                part.param_defaults.insert(*sym, v);
            }
        }
        if let Some(name) = inst.model.instance_name() {
            let c = lo.ctx();
            let calls = frag
                .terminal_currents
                .iter()
                .chain(&frag.currents)
                .chain(&frag.terminal_charges)
                .chain(&frag.charges)
                .filter(|&&e| matches!(c.node(e), rsdag::Node::Call(..)));
            part.labels.extend(calls.map(|&e| (e, name.to_string())));
        }
        part.observers.op_vars.extend(frag.op_vars);
        part.assertions.extend(frag.assertions);
        part.structure.extend(frag.structural);
        part.aliases.extend(frag.collapsed);
        part.limits.extend(frag.limits);
    }

    // Behavioral sources into the node KCL: `I=` injects its current expression,
    // `V=` injects its branch current (the constraint is added as a branch row).
    let ctx = lo.ctx();
    for b in circuit.behavioral() {
        match b.kind {
            BKind::I => {
                let ix = translate_bexpr(ctx, &b.expr, v, &branch_i_of_name, t_e);
                add_current(ctx, &mut part.node_terms, b.a, b.b, ix);
            }
            BKind::V => {
                let ib = branch_i_of_name[&b.name.to_ascii_lowercase()];
                add_current(ctx, &mut part.node_terms, b.a, b.b, ib);
            }
        }
    }

    // Branch constraint rows.
    // Inductor name -> (index in branches, its current), for mutuals.
    let mut inductor: HashMap<String, (usize, ExprId)> = HashMap::new();
    let zero = ctx.zero();
    for (pos, &idx) in branch_elem_idx.iter().enumerate() {
        let e = &circuit.elements()[idx];
        let dv = ctx.sub(v[e.a], v[e.b]);
        let mut charge = zero;
        let current = match e.kind {
            Kind::VoltageSource => {
                let val = source_value(ctx, e, t_e);
                ctx.sub(dv, val)
            }
            Kind::Inductor => {
                let ls = value_sym(ctx, &e.name);
                let il = branch_i[pos].0;
                inductor.insert(e.name.to_ascii_lowercase(), (pos, il));
                // the flux, as the charge of `v - d/dt (L i) = 0`
                let flux = ctx.mul(ls, il);
                charge = ctx.neg(flux);
                dv
            }
            Kind::Vcvs => {
                let gain = value_sym(ctx, &e.name);
                let (cp, cm) = e.ctrl.expect("VCVS control nodes");
                let dvc = ctx.sub(v[cp], v[cm]);
                let g_dvc = ctx.mul(gain, dvc);
                ctx.sub(dv, g_dvc)
            }
            Kind::Ccvs => {
                // V(a) - V(b) - gain*I(ctrl) = 0.
                let gain = value_sym(ctx, &e.name);
                let cname = e.ctrl_elem.as_deref().expect("CCVS needs controller");
                let ictrl = branch_i_of_name[&cname.to_ascii_lowercase()];
                let g_i = ctx.mul(gain, ictrl);
                ctx.sub(dv, g_i)
            }
            _ => unreachable!(),
        };
        part.branches.push(Unknown {
            name: format!("i_{}", e.name),
            kind: UnknownKind::BranchCurrent,
            x: branch_i[pos].1,
            current,
            charge,
        });
    }

    // Mutual inductance: add M * i_other to each coupled inductor's flux,
    // where M = k * sqrt(L1*L2).
    for cpl in circuit.couplings() {
        let (l1, l2) = (cpl.l1.to_ascii_lowercase(), cpl.l2.to_ascii_lowercase());
        if let (Some(&(ix, i_x)), Some(&(iy, i_y))) = (inductor.get(&l1), inductor.get(&l2)) {
            let k = value_sym(ctx, &cpl.name);
            let ls1 = value_sym(ctx, &cpl.l1);
            let ls2 = value_sym(ctx, &cpl.l2);
            let prod = ctx.mul(ls1, ls2);
            let sq = ctx.sqrt(prod);
            let m = ctx.mul(k, sq);
            let m_i_y = ctx.mul(m, i_y);
            part.branches[ix].charge = ctx.sub(part.branches[ix].charge, m_i_y);
            let m_i_x = ctx.mul(m, i_x);
            part.branches[iy].charge = ctx.sub(part.branches[iy].charge, m_i_x);
        }
    }

    // Behavioral `V=` constraints: `v(np) - v(nm) - expr = 0`, one branch row
    // each (algebraic, like an independent voltage source).
    for &(bi, _, is) in &bhv_v {
        let b = &circuit.behavioral()[bi];
        let dv = ctx.sub(v[b.a], v[b.b]);
        let val = translate_bexpr(ctx, &b.expr, v, &branch_i_of_name, t_e);
        part.branches.push(Unknown {
            name: format!("i_{}", b.name),
            kind: UnknownKind::BranchCurrent,
            x: is,
            current: ctx.sub(dv, val),
            charge: zero,
        });
    }

    // Retain each independent source's stimulus shape (by element name) for the
    // analyses that need the structural facts lowering destroys (transient
    // breakpoints, the HB fundamental), and every independent V/I element by
    // name: the element's value symbol doubles as its DC value, and the DC
    // source-stepping continuation ramps exactly this set.
    for e in circuit.elements() {
        if let Some(s) = e.source {
            part.sources.push((e.name.clone(), s));
        }
        if matches!(e.kind, Kind::VoltageSource | Kind::CurrentSource) {
            part.source_names
                .push(sane_circuit::value_symbol_name(&e.name));
        }
    }

    for inst in instances {
        instantiate(lo, bodies, inst, v, t_e, &mut part)?;
    }
    Ok(part)
}

/// A subcircuit body assembled once, for every instance of it: its part in
/// its own names over formal nodes, and the functions its outputs are. A
/// body instantiated at one place only is no function: its outputs are
/// carried into that instance's frame (see `instantiate`); a function
/// shares a body between instances and has nothing to share there.
struct Lowered {
    part: Part,
    /// The noise sources and op-vars.
    observers: Arc<crate::observers::Body>,
    /// The formal node voltages (`[0]` ground).
    fv: Vec<ExprId>,
    /// Where each kind of output starts among the outputs of the functions
    /// in order: the unknowns' currents (the node currents before them),
    /// the charges (the nodes', then the unknowns'), the delays, the events.
    out_unknown: usize,
    out_charge: usize,
    out_tau: usize,
    out_event: usize,
    /// The functions, each with its parameters and the number of outputs
    /// an instance calls (the observers' after them it does not).
    funcs: Vec<(FuncId, Vec<SymbolId>, usize)>,
    /// The body's noise terms (see `Part::noise_terms`), each coefficient a
    /// constant (`Err`) or the output it is computed in (`Ok`).
    noise: Vec<(RowRef, SymbolId, Result<usize, ExprId>)>,
    /// A body carried into its one instance's frame: its outputs, in the
    /// order the functions' would be.
    inline: Option<Vec<ExprId>>,
}

/// `body` assembled over formal nodes and closed into its functions: once
/// per body, the ones before looked up. A body's labels go to the parent
/// that assembles it first, once.
fn lowered(
    lo: &mut Lowerer,
    bodies: &mut Bodies,
    body: &Arc<Circuit>,
    t_e: ExprId,
    labels: &mut Vec<(ExprId, String)>,
) -> Result<Rc<Lowered>, String> {
    if let Some(l) = bodies.lowered.get(&Arc::as_ptr(body)) {
        return Ok(l.clone());
    }
    let n = body.node_count();
    let ctx = lo.ctx();
    let zero = ctx.zero();
    let mut fv = vec![zero; n + 1];
    for k in 1..=n {
        fv[k] = ctx.sym(&format!("{}v#{k}", body.ns));
    }
    let mut part = lower_body(lo, bodies, body, &fv, t_e)?;
    let inline = bodies.sites.get(&Arc::as_ptr(body)) == Some(&1);
    if !inline {
        for (k, name) in body.node_names().iter().enumerate().skip(1) {
            let name = name.strip_prefix(body.ns.as_str()).unwrap_or(name);
            part.labels.push((fv[k], name.to_string()));
        }
        labels.append(&mut part.labels);
    }
    let ctx = lo.ctx();

    // Every per-instance quantity is one output.
    let mut outs: Vec<ExprId> = part
        .node_terms
        .iter()
        .map(|terms| ctx.reduce(ReduceOp::Sum, terms.clone()))
        .collect();
    let out_unknown = outs.len();
    outs.extend(part.branches.iter().chain(&part.extras).map(|u| u.current));
    let out_charge = outs.len();
    for terms in &part.node_charges {
        outs.push(ctx.reduce(ReduceOp::Sum, terms.clone()));
    }
    outs.extend(part.branches.iter().chain(&part.extras).map(|u| u.charge));
    let out_tau = outs.len();
    outs.extend(part.delays.iter().map(|d| d.2));
    let out_event = outs.len();
    outs.extend(part.events.iter().map(|e| e.g));
    // the noise coefficients that are no constant, read per instance
    let mut noise = Vec::with_capacity(part.noise_terms.len());
    for &(row, g, coeff) in &part.noise_terms {
        let at = match ctx.const_f64(coeff) {
            Some(_) => Err(coeff),
            None => {
                outs.push(coeff);
                Ok(outs.len() - 1)
            }
        };
        noise.push((row, g, at));
    }

    // The rows and the observers are two functions, so a call of the rows
    // reads only what they read. An instance calls the delays, the events
    // and the noise coefficients; the noise sources' levels and the op-vars
    // after them are called where asked for (see `Observers`).
    let n_called = outs.len() - out_tau;
    let observed = part.observers.exprs();
    let firsts: Vec<SymbolId> = fv[1..]
        .iter()
        .map(|&e| sym_of(ctx, e).expect("formal node"))
        .chain(part.branches.iter().chain(&part.extras).map(|u| u.x))
        .collect();
    // What is neither a node, an unknown, time nor a delay history is a
    // parameter: the body's pure arguments, its work run once per parameter
    // binding (see `ParamRole::Param`).
    let impure: HashSet<SymbolId> = firsts
        .iter()
        .copied()
        .chain(sym_of(ctx, t_e))
        .chain(part.delays.iter().map(|d| d.1))
        .collect();
    // named after the subcircuit (its namespace `__name__.`)
    let subckt = body.ns.trim_end_matches('.').trim_matches('_');
    let names = [subckt.to_string(), format!("{subckt}, observers")];
    let mut funcs = Vec::new();
    let own = if inline {
        // the observers alone a function, called where asked for
        (!observed.is_empty()).then(|| {
            let (f, leaves) = close(ctx, bodies, &body.ns, &names[1], observed, &firsts, &impure);
            (f, leaves, 0)
        })
    } else {
        let mut outs = outs.clone();
        outs.extend(observed.iter().copied());
        for (group, name) in [&outs[..out_tau], &outs[out_tau..]].into_iter().zip(&names) {
            if !group.is_empty() {
                let (f, leaves) = close(
                    ctx,
                    bodies,
                    &body.ns,
                    name,
                    group.to_vec(),
                    &firsts,
                    &impure,
                );
                funcs.push((f, leaves, group.len()));
            }
        }
        match funcs.last_mut() {
            Some((f, leaves, n)) if !observed.is_empty() => {
                *n = n_called;
                Some((*f, leaves.clone(), n_called as u32))
            }
            _ => None,
        }
    };
    let observers = std::mem::take(&mut part.observers).into_body(ctx, &body.ns, own);
    let l = Rc::new(Lowered {
        part,
        observers,
        fv,
        out_unknown,
        out_charge,
        out_tau,
        out_event,
        funcs,
        noise,
        inline: inline.then_some(outs),
    });
    bodies.lowered.insert(Arc::as_ptr(body), l.clone());
    Ok(l)
}

/// A subcircuit instance into its parent's `part`: its body's functions
/// called with the instance's nodes and names, or the body's outputs
/// carried into the parent's frame where it is instantiated here only.
#[allow(clippy::too_many_arguments)]
fn instantiate(
    lo: &mut Lowerer,
    bodies: &mut Bodies,
    inst: &Instance,
    v: &[ExprId],
    t_e: ExprId,
    part: &mut Part,
) -> Result<(), String> {
    let low = lowered(lo, bodies, &inst.body, t_e, &mut part.labels)?;
    let body = &low.part;
    let ctx = lo.ctx();

    // A formal node binds to the parent's node; every other leaf to its name
    // in the parent's frame.
    let binding: Vec<(SymbolId, ExprId)> = (1..=inst.nodes.len())
        .map(|k| {
            (
                sym_of(ctx, low.fv[k]).expect("formal node"),
                v[inst.node(k)],
            )
        })
        .collect();
    let mut map: rustc_hash::FxHashMap<SymbolId, ExprId> = binding.iter().copied().collect();
    let mut actual = |ctx: &mut Graph, s: SymbolId| rebind(ctx, inst, &mut map, s);
    let mut calls: Vec<ExprId> = Vec::with_capacity(low.out_event + body.events.len());
    match &low.inline {
        Some(outs) => {
            // the outputs, and the labels of the calls in them in the
            // instance's names
            let roots: Vec<ExprId> = (outs.iter().copied())
                .chain(body.labels.iter().map(|&(e, _)| e))
                .collect();
            let free: Vec<SymbolId> = ctx.free_symbols_in(&roots).into_iter().collect();
            let subst: rustc_hash::FxHashMap<SymbolId, ExprId> =
                free.into_iter().map(|s| (s, actual(ctx, s))).collect();
            calls = rsdag::substitute(ctx, &roots, &subst);
            let keys = calls.split_off(outs.len());
            part.labels.extend(
                keys.into_iter()
                    .zip(&body.labels)
                    .map(|(e, (_, name))| (e, inst.rename(name))),
            );
        }
        None => {
            for (func, leaves, n_out) in low.funcs.iter().filter(|f| f.2 > 0) {
                let args: Vec<ExprId> = leaves.iter().map(|&s| actual(ctx, s)).collect();
                let outs: Vec<u32> = (0..*n_out as u32).collect();
                calls.extend(ctx.calls(*func, &outs, &args));
            }
            part.labels
                .extend(calls.iter().map(|&e| (e, inst.name.clone())));
        }
    }
    let mut actual_sym = |ctx: &mut Graph, s: SymbolId| {
        let e = actual(ctx, s);
        sym_of(ctx, e)
    };

    let n_nodes = inst.nodes.len();
    for k in 1..=n_nodes {
        let p = inst.node(k);
        if p != 0 && !body.node_terms[k - 1].is_empty() {
            part.node_terms[p - 1].push(calls[k - 1]);
        }
        if p != 0 && !body.node_charges[k - 1].is_empty() {
            part.node_charges[p - 1].push(calls[low.out_charge + k - 1]);
        }
    }
    let zero = ctx.zero();
    let extra_base = part.extras.len();
    let n_branch = body.branches.len();
    for (j, u) in body.branches.iter().chain(&body.extras).enumerate() {
        let u = Unknown {
            name: inst.rename(&u.name),
            kind: u.kind,
            x: actual_sym(ctx, u.x).expect("an unknown is a symbol"),
            current: calls[low.out_unknown + j],
            charge: if ctx.is_zero(u.charge) {
                zero
            } else {
                calls[low.out_charge + n_nodes + j]
            },
        };
        if j < n_branch {
            part.branches.push(u);
        } else {
            part.extras.push(u);
        }
    }
    for &(row, g, at) in &low.noise {
        let row = match row {
            RowRef::Node(k) => match inst.node(k + 1) {
                0 => continue,
                p => RowRef::Node(p - 1),
            },
            RowRef::Extra(j) => RowRef::Extra(extra_base + j),
        };
        let g = actual_sym(ctx, g).expect("a noise generator is a symbol");
        part.noise_terms
            .push((row, g, at.map_or_else(|c| c, |o| calls[o])));
    }
    for (j, &(src, hist, _)) in body.delays.iter().enumerate() {
        let hist = actual_sym(ctx, hist).expect("a history is a symbol");
        part.delays
            .push((extra_base + src, hist, calls[low.out_tau + j]));
    }
    for (j, e) in body.events.iter().enumerate() {
        part.events.push(EventSpec {
            g: calls[low.out_event + j],
            dir: e.dir,
            name: inst.rename(&e.name),
        });
    }
    for (&s, &val) in &body.param_defaults {
        if let Some(a) = actual_sym(ctx, s) {
            part.param_defaults.insert(a, val);
        }
    }
    part.dc_seeds.extend(
        body.dc_seeds
            .iter()
            .map(|(name, c)| (inst.rename(name), *c)),
    );
    for &(r, c, g) in &body.companion {
        let (pr, pc) = (inst.node(r + 1), inst.node(c + 1));
        if pr != 0 && pc != 0 {
            part.companion.push((pr - 1, pc - 1, g));
        }
    }
    for l in &body.limits {
        let when = l.when.map(|w| {
            let rename: rustc_hash::FxHashMap<SymbolId, ExprId> = ctx
                .free_symbols_in(&[w])
                .into_iter()
                .filter_map(|s| actual_sym(ctx, s).map(|a| (s, ctx.symbol_expr(a))))
                .collect();
            rsdag::substitute(ctx, &[w], &rename)[0]
        });
        part.limits.push(sane_device::FragmentLimit {
            hi: l.hi.and_then(|s| actual_sym(ctx, s)),
            lo: l.lo.and_then(|s| actual_sym(ctx, s)),
            kind: l.kind,
            when,
        });
    }
    part.sources
        .extend(body.sources.iter().map(|(name, s)| (inst.rename(name), *s)));
    part.source_names
        .extend(body.source_names.iter().map(|s| inst.rename(s)));
    // The body's assertions and structure over the instance's parameters,
    // and its collapsed nodes in the instance's names and nodes.
    let exprs: Vec<ExprId> = (body.assertions.iter().chain(&body.structure))
        .map(|a| a.holds)
        .chain(body.aliases.iter().map(|&(_, v)| v))
        .collect();
    let rename: rustc_hash::FxHashMap<SymbolId, ExprId> = ctx
        .free_symbols_in(&exprs)
        .into_iter()
        .map(|s| (s, actual(ctx, s)))
        .collect();
    let mut exprs = rsdag::substitute(ctx, &exprs, &rename).into_iter();
    for (list, into) in [
        (&body.assertions, &mut part.assertions),
        (&body.structure, &mut part.structure),
    ] {
        into.extend(list.iter().map(|a| sane_device::Assertion {
            holds: exprs.next().expect("one per assertion"),
            message: a.message.clone(),
        }));
    }
    part.aliases.extend(
        (body.aliases.iter())
            .map(|(name, _)| (inst.rename(name), exprs.next().expect("one per alias"))),
    );
    // what each body symbol the instance reads is in the parent's frame,
    // where its observers are found when asked for
    part.observers.place(&inst.name, map, &low.observers);
    Ok(())
}

/// `outs` as a function of their free symbols, `firsts` leading in their
/// order, the rest by id: the one built before for the same outputs over the
/// same parameters, else a new one. Returns it with its parameters. A symbol
/// outside the body's namespace `ns` is the same one in every instance (a
/// model card's parameter, the temperature): a global of the function, not
/// a parameter.
fn close(
    ctx: &mut Graph,
    bodies: &mut Bodies,
    ns: &str,
    name: &str,
    outs: Vec<ExprId>,
    firsts: &[SymbolId],
    impure: &HashSet<SymbolId>,
) -> (FuncId, Vec<SymbolId>) {
    let free = ctx.free_symbols_in(&outs);
    let mut seen: HashSet<SymbolId> = HashSet::new();
    let own = |s: &SymbolId| ctx.symbol_name(*s).contains(ns);
    let leaves: Vec<SymbolId> = firsts
        .iter()
        .copied()
        .chain(free.iter().copied().filter(own))
        .filter(|s| free.contains(s) && seen.insert(*s))
        .collect();
    let key = (outs, leaves);
    if let Some(&f) = bodies.funcs.get(&key) {
        return (f, key.1);
    }
    let f = ctx.define_func(name, key.1.clone(), key.0.clone());
    for (k, s) in key.1.iter().enumerate() {
        if !impure.contains(s) {
            ctx.set_param_role(f, k as u32, ParamRole::Param);
        }
    }
    let leaves = key.1.clone();
    bodies.funcs.insert(key, f);
    (f, leaves)
}

/// A body symbol's expression in the parent's frame: the binding in `map` (a
/// formal node), else the symbol named by the instance's renaming.
fn rebind(
    ctx: &mut Graph,
    inst: &Instance,
    map: &mut rustc_hash::FxHashMap<SymbolId, ExprId>,
    s: SymbolId,
) -> ExprId {
    if let Some(&e) = map.get(&s) {
        return e;
    }
    let name = inst.rename(ctx.symbol_name(s));
    let e = ctx.sym(&name);
    map.insert(s, e);
    e
}
