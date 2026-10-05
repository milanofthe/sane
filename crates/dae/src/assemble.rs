//! Assembly of the symbolic DAE from a parsed circuit: node KCL residuals,
//! branch constraints, device lowering (every device -- built-in Verilog-A,
//! user Verilog-A, OSDI, behavioral -- through the one `lower_behavioral`
//! fragment path), noise generators, and the companion / limit / source
//! registries the solver reads.

use std::collections::{HashMap, HashSet};

use rsdag::{CmpOp, ExprId, FuncId, ParamRole, ReduceOp, SymbolId};
use sane_core::Graph;
use sane_device::Lowerer;
use sane_mna::{BExpr, BKind, Circuit, Element, Kind, SourceFn};

use crate::hierarchy::Instance;
use crate::{sym2, Dae, DelaySpec, DeviceInstance, EventSpec, Limit, NoiseSource, UnknownKind};

/// Symbol for an element's value parameter, mapped out of the reserved unknown
/// namespace (see [`sane_mna::value_symbol_name`]) so e.g. a voltage source
/// named `v91` cannot be hash-consed onto node 91's voltage unknown.
fn value_sym(ctx: &mut Graph, name: &str) -> ExprId {
    ctx.sym(&sane_mna::value_symbol_name(name))
}

/// The (possibly time-dependent) value of an independent source.
///
/// Numeric parameters are instance-scoped symbols (bound separately), so the
/// expression stays symbolic. Region splits use `Select`; periodicity uses
/// `floor`.
fn source_value(ctx: &mut Graph, e: &Element, t: ExprId) -> ExprId {
    // The constitutive waveform lives with the source type (see `sane_mna::SourceFn`);
    // a constant element (no source shape) is just its own value symbol.
    match e.source {
        None => value_sym(ctx, &e.name),
        Some(src) => src.lower(ctx, &e.name, t),
    }
}

/// Add a current `c` flowing from node `a` to node `b` into the KCL residuals.
/// Accumulate a branch current `c` into the KCL term lists: `+c` leaves node
/// `a`, `-c` enters node `b` (ground = 0 is skipped). The per-node lists are
/// folded into one fused `Reduce(Sum)` at the end, instead of a binary Add-tree.
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
                '^' => {
                    // integer constant exponent -> pow_i, else exp(b*ln(a))
                    if let BExpr::Const(c) = **b {
                        if c.fract() == 0.0 && c.abs() < 64.0 {
                            return ctx.pow_i(x, c as i64);
                        }
                    }
                    let l = ctx.ln(x);
                    let yl = ctx.mul(y, l);
                    ctx.exp(yl)
                }
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

/// Assemble a flat circuit (no subcircuit instances).
pub fn assemble_dae(ctx: &mut Graph, circuit: &Circuit, devices: &[DeviceInstance]) -> Dae {
    assemble(ctx, circuit, devices, &[])
}

/// An unknown a body mints beyond its nodes (a branch current, a device
/// extra), with its residual row.
struct Unknown {
    name: String,
    kind: UnknownKind,
    x: SymbolId,
    /// The derivative symbol: an inductor's `idot`, a device extra's (kept
    /// only if it appears in the residuals, decided once the DAE is whole),
    /// `None` for an algebraic branch.
    xdot: Option<SymbolId>,
    residual: ExprId,
}

/// A body assembled over its own nodes: the current terms into each node and
/// everything else it contributes, in its own names. The top level becomes
/// the DAE; a subcircuit body becomes a function.
#[derive(Default)]
struct Part {
    /// Per node (`k - 1` for node `k`), the signed current terms leaving it.
    node_terms: Vec<Vec<ExprId>>,
    branches: Vec<Unknown>,
    extras: Vec<Unknown>,
    /// `src` / `out` index `extras`.
    delays: Vec<DelaySpec>,
    events: Vec<EventSpec>,
    noise: Vec<NoiseSource>,
    op_vars: Vec<sane_device::OpVar>,
    param_defaults: rustc_hash::FxHashMap<SymbolId, f64>,
    dc_seeds: Vec<(String, f64)>,
    /// `(row, col, g)` over node indices `k - 1`.
    companion: Vec<(usize, usize, f64)>,
    limits: Vec<sane_device::FragmentLimit>,
    sources: Vec<(String, SourceFn)>,
    source_names: Vec<String>,
    /// Display names (see [`Dae::labels`]).
    labels: Vec<(ExprId, String)>,
}

/// The subcircuit functions built so far, by what they compute: a body's
/// outputs over its parameters. Instances whose bodies lower to the same
/// expressions share one function.
#[derive(Default)]
struct Bodies {
    funcs: HashMap<(Vec<ExprId>, Vec<SymbolId>), FuncId>,
    /// Time spent in device `lower_behavioral`, for the stage log.
    lower_t: std::time::Duration,
}

fn sym_of(ctx: &Graph, e: ExprId) -> Option<SymbolId> {
    match ctx.node(e) {
        rsdag::Node::Symbol(s) => Some(*s),
        _ => None,
    }
}

/// The nodes a body spans: its elements', its devices' terminals and its
/// instances' connections (a node only a device or an instance touches is
/// not counted by the element graph).
fn body_nodes(circuit: &Circuit, devices: &[DeviceInstance], instances: &[Instance]) -> usize {
    let dev = devices.iter().flat_map(|d| d.terminals.iter().copied());
    let inst = instances.iter().flat_map(|i| i.nodes.iter().copied());
    dev.chain(inst).max().unwrap_or(0).max(circuit.node_count())
}

/// Assemble the DAE of a circuit with its subcircuit instances. Every node
/// of the hierarchy is a top-level node (`v{k}`); every other unknown keeps
/// its block (branch currents, then device extras) in the instance's names.
pub fn assemble(
    ctx: &mut Graph,
    circuit: &Circuit,
    devices: &[DeviceInstance],
    instances: &[Instance],
) -> Dae {
    let n = body_nodes(circuit, devices, instances);
    let zero = ctx.zero();
    let (t_e, t) = sym2(ctx, "t");
    // Node voltage and derivative symbols (index 0 = ground = 0).
    let mut v = vec![zero; n + 1];
    let mut vdot = vec![zero; n + 1];
    let mut x = Vec::new();
    let mut xdot = Vec::new();
    for k in 1..=n {
        let (e, s) = sym2(ctx, &format!("v{k}"));
        let (de, ds) = sym2(ctx, &format!("vdot{k}"));
        (v[k], vdot[k]) = (e, de);
        x.push(s);
        // Node voltages always carry a derivative slot (a capacitor onto the
        // node is common; the mass-matrix column is simply zero otherwise).
        xdot.push(Some(ds));
    }

    let dev_t0 = sane_core::time::Instant::now();
    let mut bodies = Bodies::default();
    let mut lo = Lowerer::new(ctx);
    let part = lower_body(&mut lo, &mut bodies, circuit, devices, instances, &v, &vdot, t_e);
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

    // Layout: node KCL, then branch constraints, then device extras.
    let mut residuals: Vec<ExprId> = part
        .node_terms
        .into_iter()
        .map(|terms| ctx.reduce(ReduceOp::Sum, terms))
        .collect();
    residuals.extend(part.branches.iter().map(|u| u.residual));
    residuals.extend(part.extras.iter().map(|u| u.residual));

    // Unified differential-classification: an extra is a differential DOF iff
    // its derivative symbol actually appears in the assembled residuals (a
    // charge dq/dt, a Verilog-A ddt/idt). One rule for device-internal nodes
    // and behavioral states alike.
    let diff_syms = ctx.free_symbols_in(&residuals);

    let mut unknowns: Vec<String> = (1..=n).map(|k| format!("v{k}")).collect();
    let mut kinds = vec![UnknownKind::NodeVoltage; n];
    for u in &part.branches {
        unknowns.push(u.name.clone());
        kinds.push(UnknownKind::BranchCurrent);
        x.push(u.x);
        xdot.push(u.xdot);
    }
    for u in &part.extras {
        unknowns.push(u.name.clone());
        kinds.push(u.kind);
        x.push(u.x);
        xdot.push(u.xdot.filter(|s| diff_syms.contains(s)));
    }

    // delay indices were extra-relative; shift onto the final layout
    let extra_base = n + part.branches.len();
    let mut delays = part.delays;
    for dl in delays.iter_mut() {
        dl.src += extra_base;
        dl.out += extra_base;
    }

    // Controlling-voltage limits: node symbols -> global unknown indices, now
    // that the layout is final (covers external terminals and, for behavioral
    // devices, internal nodes alike; ground / non-unknown symbols -> None).
    let unknown_of: HashMap<SymbolId, usize> = x.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let limits: Vec<Limit> = part
        .limits
        .iter()
        .map(|fl| Limit {
            hi: fl.hi.and_then(|s| unknown_of.get(&s).copied()),
            lo: fl.lo.and_then(|s| unknown_of.get(&s).copied()),
            kind: fl.kind,
        })
        .collect();

    Dae {
        residuals,
        n_nodes: n,
        param_defaults: part.param_defaults,
        events: part.events,
        delays,
        unknowns,
        kinds,
        x,
        xdot,
        t,
        companion: part.companion,
        noise_sources: part.noise,
        op_vars: part.op_vars,
        dc_seeds: part.dc_seeds,
        limits,
        sources: part.sources,
        source_names: part.source_names,
        labels: part.labels.into_iter().collect(),
    }
}

/// Assemble one body over node voltages `v` / derivatives `vdot` (index 0 is
/// ground): its elements, devices and subcircuit instances.
#[allow(clippy::too_many_arguments)]
fn lower_body(
    lo: &mut Lowerer,
    bodies: &mut Bodies,
    circuit: &Circuit,
    devices: &[DeviceInstance],
    instances: &[Instance],
    v: &[ExprId],
    vdot: &[ExprId],
    t_e: ExprId,
) -> Part {
    let n = v.len() - 1;
    let mut part = Part {
        node_terms: vec![Vec::new(); n],
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
                let dvd = ctx.sub(vdot[e.a], vdot[e.b]);
                let c = ctx.mul(cs, dvd);
                add_current(ctx, node_terms, e.a, e.b, c);
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

    // Resistor thermal noise: each resistor R is a current-noise generator across
    // its two nodes with white PSD 4*k_B*T/R. Emitted here (alongside behavioral
    // device noise below) so every noise source, stamped or behavioral, lives in
    // one registry that the noise analysis consumes uniformly. Temperature is the
    // shared `$temp` symbol (not a fixed constant), so a temperature sweep moves
    // resistor noise exactly as it moves diode/MOSFET noise.
    {
        let k4 = ctx.konst_f64(4.0 * sane_core::constants::BOLTZMANN);
        let temp = ctx.sym(sane_core::constants::TEMP_SYMBOL);
        let coeff = ctx.mul(k4, temp); // 4*k_B*T
        for (name, a, b) in circuit.resistors() {
            let r = value_sym(ctx, &name);
            let g = ctx.recip(r);
            let psd = ctx.mul(coeff, g);
            let flicker_exp = ctx.zero();
            part.noise.push(NoiseSource {
                hi: sym_of(ctx, v[a]),
                lo: sym_of(ctx, v[b]),
                psd,
                flicker_exp,
                table: Vec::new(),
            });
        }
    }

    // Every nonlinear device lowers to a fragment that mints "extra" unknowns
    // (device-internal nodes for native models; branch currents / idt / laplace
    // states for behavioral ones) with one residual row each.
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
        // Terminal node-voltage derivatives (for a device's own ddt; ground -> 0).
        let term_vdot: Vec<ExprId> = inst.terminals.iter().map(|&nd| vdot[nd]).collect();

        // ONE lowering path for every device: a `BehavioralFragment` (terminal
        // currents + one residual per minted extra unknown), so internal nodes
        // and behavioral states are treated uniformly.
        let lt = sane_core::time::Instant::now();
        let mut frag = inst
            .model
            .lower_behavioral(lo, &term_v, &term_vdot, &ctrl_i);
        bodies.lower_t += lt.elapsed();
        // Parallel multiplicity (`M=` * `nf`): scale the terminal currents by m,
        // modelling m identical devices in parallel. Internal-node residuals stay
        // per-device (one representative internal state). Verilog-A devices apply
        // `$mfactor` internally, so their `inst.mfactor` is 1.0 (no double count).
        if inst.mfactor != 1.0 {
            let c = lo.ctx();
            let m = c.konst_f64(inst.mfactor);
            for ti in frag.terminal_currents.iter_mut() {
                *ti = c.mul(m, *ti);
            }
        }
        let dev_extra_base = part.extras.len();
        let extras = std::mem::take(&mut lo.extras);
        debug_assert_eq!(
            frag.residuals.len(),
            extras.len(),
            "device fragment must return one residual per extra unknown"
        );
        // Transport delays (`absdelay`) minted by this device: shift the
        // device-relative extras positions onto the body's extras.
        for dl in std::mem::take(&mut lo.delays) {
            part.delays.push(DelaySpec {
                src: dev_extra_base + dl.src_extra,
                out: dev_extra_base + dl.out_extra,
                hist: dl.hist,
                tau: dl.tau,
            });
        }
        for (ex, &residual) in extras.iter().zip(&frag.residuals) {
            if let Some(c) = ex.dc_seed {
                part.dc_seeds.push((ex.name.clone(), c));
            }
            part.extras.push(Unknown {
                name: ex.name.clone(),
                kind: ex.kind,
                x: ex.value_sym,
                xdot: Some(ex.xdot_sym),
                residual,
            });
        }
        for (k, &nd) in inst.terminals.iter().enumerate() {
            if nd != 0 {
                part.node_terms[nd - 1].push(frag.terminal_currents[k]);
            }
        }
        part.noise.extend(frag.noise);
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
                .chain(&frag.residuals)
                .filter(|&&e| matches!(c.node(e), rsdag::Node::Call(..)));
            part.labels.extend(calls.map(|&e| (e, name.to_string())));
        }
        part.op_vars.extend(frag.op_vars);
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

    // Branch constraint residuals.
    // Inductor name -> (index in branches, its idot expression), for mutuals.
    let mut inductor: HashMap<String, (usize, ExprId)> = HashMap::new();
    for (pos, &idx) in branch_elem_idx.iter().enumerate() {
        let e = &circuit.elements()[idx];
        let dv = ctx.sub(v[e.a], v[e.b]);
        let mut xdot = None;
        let residual = match e.kind {
            Kind::VoltageSource => {
                let val = source_value(ctx, e, t_e);
                ctx.sub(dv, val)
            }
            Kind::Inductor => {
                let ls = value_sym(ctx, &e.name);
                let (idote, idots) = sym2(ctx, &format!("idot_{}", e.name));
                let lidot = ctx.mul(ls, idote);
                inductor.insert(e.name.to_ascii_lowercase(), (pos, idote));
                xdot = Some(idots);
                ctx.sub(dv, lidot)
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
            xdot,
            residual,
        });
    }

    // Mutual inductance: add M * d/dt(i_other) to each coupled inductor's
    // constraint, where M = k * sqrt(L1*L2).
    for cpl in circuit.couplings() {
        let (l1, l2) = (cpl.l1.to_ascii_lowercase(), cpl.l2.to_ascii_lowercase());
        if let (Some(&(ix, idot_x)), Some(&(iy, idot_y))) = (inductor.get(&l1), inductor.get(&l2)) {
            let k = value_sym(ctx, &cpl.name);
            let ls1 = value_sym(ctx, &cpl.l1);
            let ls2 = value_sym(ctx, &cpl.l2);
            let prod = ctx.mul(ls1, ls2);
            let sq = ctx.sqrt(prod);
            let m = ctx.mul(k, sq);
            let m_idot_y = ctx.mul(m, idot_y);
            part.branches[ix].residual = ctx.sub(part.branches[ix].residual, m_idot_y);
            let m_idot_x = ctx.mul(m, idot_x);
            part.branches[iy].residual = ctx.sub(part.branches[iy].residual, m_idot_x);
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
            xdot: None,
            residual: ctx.sub(dv, val),
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
            part.source_names.push(sane_mna::value_symbol_name(&e.name));
        }
    }

    for inst in instances {
        instantiate(lo, bodies, inst, v, vdot, t_e, &mut part);
    }
    part
}

/// A subcircuit instance into its parent's `part`: the body assembled over
/// formal nodes, closed into a function (or found as one), called with the
/// instance's nodes and names.
#[allow(clippy::too_many_arguments)]
fn instantiate(
    lo: &mut Lowerer,
    bodies: &mut Bodies,
    inst: &Instance,
    v: &[ExprId],
    vdot: &[ExprId],
    t_e: ExprId,
    part: &mut Part,
) {
    let n = inst.nodes.len();
    let ctx = lo.ctx();
    let zero = ctx.zero();
    let (mut fv, mut fvd) = (vec![zero; n + 1], vec![zero; n + 1]);
    for k in 1..=n {
        fv[k] = ctx.sym(&format!("{}v#{k}", inst.ns));
        fvd[k] = ctx.sym(&format!("{}vdot#{k}", inst.ns));
    }
    let mut body = lower_body(
        lo,
        bodies,
        &inst.circuit,
        &inst.devices,
        &inst.instances,
        &fv,
        &fvd,
        t_e,
    );
    for (k, name) in inst.node_names.iter().enumerate() {
        body.labels.push((fv[k + 1], name.clone()));
        body.labels.push((fvd[k + 1], format!("{name}'")));
    }
    part.labels.append(&mut body.labels);
    let ctx = lo.ctx();

    // Every per-instance quantity is one output.
    let mut outs: Vec<ExprId> = body
        .node_terms
        .iter()
        .map(|terms| ctx.reduce(ReduceOp::Sum, terms.clone()))
        .collect();
    let out_unknown = outs.len();
    outs.extend(body.branches.iter().chain(&body.extras).map(|u| u.residual));
    let out_noise = outs.len();
    for ns in &body.noise {
        outs.push(ns.psd);
        outs.push(ns.flicker_exp);
    }
    let out_opvar = outs.len();
    outs.extend(body.op_vars.iter().map(|o| o.value));
    let out_tau = outs.len();
    outs.extend(body.delays.iter().map(|d| d.tau));
    let out_event = outs.len();
    outs.extend(body.events.iter().map(|e| e.g));

    // The residuals and the observers (noise, op-vars, delays, events) are
    // two functions, so a call of the residuals reads only what they read
    // (a resistor's noise reads the temperature, its current does not).
    let unknowns = body.branches.iter().chain(&body.extras);
    let firsts: Vec<SymbolId> = fv[1..]
        .iter()
        .chain(&fvd[1..])
        .map(|&e| sym_of(ctx, e).expect("formal node"))
        .chain(unknowns.clone().map(|u| u.x))
        .chain(unknowns.filter_map(|u| u.xdot))
        .collect();
    // What is neither a node, an unknown, time nor a delay history is a
    // parameter: the body's pure arguments, its work run once per parameter
    // binding (see `ParamRole::Param`).
    let impure: HashSet<SymbolId> = firsts
        .iter()
        .copied()
        .chain(sym_of(ctx, t_e))
        .chain(body.delays.iter().map(|d| d.hist))
        .collect();

    // A formal node binds to the parent's node; every other leaf to its name
    // in the parent's frame.
    let mut map: HashMap<SymbolId, ExprId> = HashMap::new();
    for k in 1..=n {
        let p = inst.node(k);
        map.insert(sym_of(ctx, fv[k]).unwrap(), v[p]);
        map.insert(sym_of(ctx, fvd[k]).unwrap(), vdot[p]);
    }
    let mut actual = |ctx: &mut Graph, s: SymbolId| rebind(ctx, inst, &mut map, s);
    let mut calls: Vec<ExprId> = Vec::with_capacity(outs.len());
    // named after the subcircuit (its namespace `__name__.`)
    let subckt = inst.ns.trim_end_matches('.').trim_matches('_');
    let names = [subckt.to_string(), format!("{subckt}, observers")];
    for (group, name) in [&outs[..out_noise], &outs[out_noise..]].into_iter().zip(&names) {
        if group.is_empty() {
            continue;
        }
        let (func, leaves) = close(ctx, bodies, name, group.to_vec(), &firsts, &impure);
        let args: Vec<ExprId> = leaves.iter().map(|&s| actual(ctx, s)).collect();
        let outs: Vec<u32> = (0..group.len() as u32).collect();
        calls.extend(ctx.calls(func, &outs, &args));
    }
    part.labels.extend(calls.iter().map(|&e| (e, inst.name.clone())));
    let mut actual_sym = |ctx: &mut Graph, s: SymbolId| {
        let e = actual(ctx, s);
        sym_of(ctx, e)
    };

    for k in 1..=n {
        let p = inst.node(k);
        if p != 0 && !body.node_terms[k - 1].is_empty() {
            part.node_terms[p - 1].push(calls[k - 1]);
        }
    }
    let extra_base = part.extras.len();
    let n_branch = body.branches.len();
    for (j, u) in body.branches.into_iter().chain(body.extras).enumerate() {
        let u = Unknown {
            name: inst.rename(&u.name),
            kind: u.kind,
            x: actual_sym(ctx, u.x).expect("an unknown is a symbol"),
            xdot: u.xdot.and_then(|s| actual_sym(ctx, s)),
            residual: calls[out_unknown + j],
        };
        if j < n_branch {
            part.branches.push(u);
        } else {
            part.extras.push(u);
        }
    }
    for (j, ns) in body.noise.into_iter().enumerate() {
        part.noise.push(NoiseSource {
            hi: ns.hi.and_then(|s| actual_sym(ctx, s)),
            lo: ns.lo.and_then(|s| actual_sym(ctx, s)),
            psd: calls[out_noise + 2 * j],
            flicker_exp: calls[out_noise + 2 * j + 1],
            table: ns.table,
        });
    }
    for (j, o) in body.op_vars.into_iter().enumerate() {
        part.op_vars.push(sane_device::OpVar {
            name: inst.rename(&o.name),
            value: calls[out_opvar + j],
            ..o
        });
    }
    for (j, d) in body.delays.into_iter().enumerate() {
        part.delays.push(DelaySpec {
            src: extra_base + d.src,
            out: extra_base + d.out,
            hist: actual_sym(ctx, d.hist).expect("a history is a symbol"),
            tau: calls[out_tau + j],
        });
    }
    for (j, e) in body.events.into_iter().enumerate() {
        part.events.push(EventSpec {
            g: calls[out_event + j],
            dir: e.dir,
            name: inst.rename(&e.name),
        });
    }
    for (s, val) in body.param_defaults {
        if let Some(a) = actual_sym(ctx, s) {
            part.param_defaults.insert(a, val);
        }
    }
    part.dc_seeds.extend(
        body.dc_seeds
            .into_iter()
            .map(|(name, c)| (inst.rename(&name), c)),
    );
    for (r, c, g) in body.companion {
        let (pr, pc) = (inst.node(r + 1), inst.node(c + 1));
        if pr != 0 && pc != 0 {
            part.companion.push((pr - 1, pc - 1, g));
        }
    }
    for l in body.limits {
        part.limits.push(sane_device::FragmentLimit {
            hi: l.hi.and_then(|s| actual_sym(ctx, s)),
            lo: l.lo.and_then(|s| actual_sym(ctx, s)),
            kind: l.kind,
        });
    }
    part.sources.extend(
        body.sources
            .into_iter()
            .map(|(name, s)| (inst.rename(&name), s)),
    );
    part.source_names.extend(body.source_names.iter().map(|s| inst.rename(s)));
}

/// `outs` as a function of their free symbols, `firsts` leading in their
/// order, the rest by id: the one built before for the same outputs over the
/// same parameters, else a new one. Returns it with its parameters.
fn close(
    ctx: &mut Graph,
    bodies: &mut Bodies,
    name: &str,
    outs: Vec<ExprId>,
    firsts: &[SymbolId],
    impure: &HashSet<SymbolId>,
) -> (FuncId, Vec<SymbolId>) {
    let free = ctx.free_symbols_in(&outs);
    let mut seen: HashSet<SymbolId> = HashSet::new();
    let leaves: Vec<SymbolId> = firsts
        .iter()
        .chain(&free)
        .copied()
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
    map: &mut HashMap<SymbolId, ExprId>,
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
