//! Assembly of the symbolic DAE from a parsed circuit: node KCL residuals,
//! branch constraints, device lowering (every device -- built-in Verilog-A,
//! user Verilog-A, OSDI, behavioral -- through the one `lower_behavioral`
//! fragment path), noise generators, and the companion / limit / source
//! registries the solver reads.

use std::collections::HashMap;

use rsdag::{CmpOp, ExprId, Graph, ReduceOp, SymbolId};
use sane_device::Lowerer;
use sane_mna::{BExpr, BKind, Circuit, Element, Kind};

use crate::{
    stamp, sym2, Dae, DelaySpec, DeviceInstance, EventSpec, Limit, NoiseSource, UnknownKind,
};

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

pub fn assemble_dae(ctx: &mut Graph, circuit: &Circuit, devices: &[DeviceInstance]) -> Dae {
    // Node count must cover nodes that appear only on devices (which live
    // outside the mna::Circuit and so are not counted by node_count()).
    let dev_max = devices
        .iter()
        .flat_map(|d| d.terminals.iter().copied())
        .max()
        .unwrap_or(0);
    let n = circuit.node_count().max(dev_max);
    let zero = ctx.zero();
    let (t_e, t) = sym2(ctx, "t");

    // Node voltage and derivative symbols (index 0 = ground = 0).
    let mut v = vec![zero; n + 1];
    let mut v_sym = vec![None; n + 1];
    let mut vdot = vec![zero; n + 1];
    let mut vdot_sym = vec![None; n + 1];
    for k in 1..=n {
        let (e, s) = sym2(ctx, &format!("v{k}"));
        v[k] = e;
        v_sym[k] = Some(s);
        let (de, ds) = sym2(ctx, &format!("vdot{k}"));
        vdot[k] = de;
        vdot_sym[k] = Some(ds);
    }

    // Branch-current unknowns for voltage-defined / dynamic elements.
    let mut branch_elem_idx: Vec<usize> = Vec::new();
    let mut branch_names: Vec<String> = Vec::new();
    let mut branch_i: Vec<ExprId> = Vec::new();
    let mut branch_i_sym: Vec<SymbolId> = Vec::new();
    let mut branch_i_of_name: std::collections::HashMap<String, ExprId> =
        std::collections::HashMap::new();
    for (idx, e) in circuit.elements().iter().enumerate() {
        if matches!(
            e.kind,
            Kind::VoltageSource | Kind::Inductor | Kind::Vcvs | Kind::Ccvs
        ) {
            let (ie, is) = sym2(ctx, &format!("i_{}", e.name));
            branch_elem_idx.push(idx);
            branch_names.push(format!("i_{}", e.name));
            branch_i.push(ie);
            branch_i_sym.push(is);
            branch_i_of_name.insert(e.name.to_ascii_lowercase(), ie);
        }
    }
    let branch_pos = |idx: usize| branch_elem_idx.iter().position(|&j| j == idx).unwrap();

    // Behavioral `V=` sources are voltage-defined, so each also gets a
    // branch-current unknown (registered so `I(Bname)` can reference it).
    let mut bhv_v_i: Vec<ExprId> = Vec::new();
    let mut bhv_v_sym: Vec<SymbolId> = Vec::new();
    let mut bhv_v_src: Vec<usize> = Vec::new();
    for (bi, b) in circuit.behavioral().iter().enumerate() {
        if b.kind == BKind::V {
            let (ie, is) = sym2(ctx, &format!("i_{}", b.name));
            branch_i_of_name.insert(b.name.to_ascii_lowercase(), ie);
            bhv_v_i.push(ie);
            bhv_v_sym.push(is);
            bhv_v_src.push(bi);
        }
    }

    // KCL residual per non-ground node: collect the signed branch-current
    // terms, then fuse them into one Reduce(Sum) below.
    let mut node_terms: Vec<Vec<ExprId>> = vec![Vec::new(); n];
    for (idx, e) in circuit.elements().iter().enumerate() {
        match e.kind {
            Kind::Resistor => {
                let s = value_sym(ctx, &e.name);
                let g = ctx.recip(s);
                let dv = ctx.sub(v[e.a], v[e.b]);
                let c = ctx.mul(dv, g);
                add_current(ctx, &mut node_terms, e.a, e.b, c);
            }
            Kind::Capacitor => {
                let cs = value_sym(ctx, &e.name);
                let dvd = ctx.sub(vdot[e.a], vdot[e.b]);
                let c = ctx.mul(cs, dvd);
                add_current(ctx, &mut node_terms, e.a, e.b, c);
            }
            Kind::Inductor | Kind::VoltageSource | Kind::Vcvs | Kind::Ccvs => {
                let il = branch_i[branch_pos(idx)];
                add_current(ctx, &mut node_terms, e.a, e.b, il);
            }
            Kind::CurrentSource => {
                let iv = source_value(ctx, e, t_e);
                add_current(ctx, &mut node_terms, e.a, e.b, iv);
            }
            Kind::Vccs => {
                let gm = value_sym(ctx, &e.name);
                let (cp, cm) = e.ctrl.expect("VCCS control nodes");
                let dvc = ctx.sub(v[cp], v[cm]);
                let c = ctx.mul(gm, dvc);
                add_current(ctx, &mut node_terms, e.a, e.b, c);
            }
            Kind::Cccs => {
                // I(a->b) = gain * I(ctrl).
                let gain = value_sym(ctx, &e.name);
                let cname = e.ctrl_elem.as_deref().expect("CCCS needs controller");
                let ictrl = branch_i_of_name[&cname.to_ascii_lowercase()];
                let c = ctx.mul(gain, ictrl);
                add_current(ctx, &mut node_terms, e.a, e.b, c);
            }
        }
    }

    // Every nonlinear device lowers to a fragment that mints "extra" unknowns
    // (device-internal nodes for native models; branch currents / idt / laplace
    // states for behavioral ones) with one residual row each, appended after the
    // branch rows.
    let mut extra_names: Vec<String> = Vec::new();
    let mut extra_kinds: Vec<UnknownKind> = Vec::new();
    let mut extra_x: Vec<SymbolId> = Vec::new();
    let mut extra_xdot: Vec<SymbolId> = Vec::new();
    let mut extra_res: Vec<ExprId> = Vec::new();
    // transport delays (tline lowering, behavioral `absdelay`); indices are
    // extra-relative and fixed up to global unknown positions once the layout
    // is final
    let mut delays: Vec<DelaySpec> = Vec::new();
    let mut events: Vec<EventSpec> = Vec::new();
    let mut noise_sources: Vec<NoiseSource> = Vec::new();
    let mut op_vars: Vec<sane_device::OpVar> = Vec::new();
    let mut param_defaults: rustc_hash::FxHashMap<SymbolId, f64> = Default::default();
    let mut dc_seeds: Vec<(String, f64)> = Vec::new();
    // Resistor thermal noise: each resistor R is a current-noise generator across
    // its two nodes with white PSD 4*k_B*T/R. Emitted here (alongside behavioral
    // device noise below) so every noise source, stamped or behavioral, lives in
    // one registry that the noise analysis consumes uniformly. Temperature is the
    // shared `$temp` symbol (not a fixed constant), so a temperature sweep moves
    // resistor noise exactly as it moves diode/MOSFET noise.
    {
        let k4 = ctx.konst_f64(4.0 * sane_core::constants::BOLTZMANN);
        let t = ctx.sym(sane_core::constants::TEMP_SYMBOL);
        let coeff = ctx.mul(k4, t); // 4*k_B*T
        for (name, a, b) in circuit.resistors() {
            let r = value_sym(ctx, &name);
            let g = ctx.recip(r);
            let psd = ctx.mul(coeff, g);
            let flicker_exp = ctx.zero();
            noise_sources.push(NoiseSource {
                hi: if a != 0 { v_sym[a] } else { None },
                lo: if b != 0 { v_sym[b] } else { None },
                psd,
                flicker_exp,
                table: Vec::new(),
            });
        }
    }
    // Companion conductance network for homotopy continuation (node-KCL row x
    // node-voltage col): each device's linear `lambda = 0` form, stamped as a
    // conductance between its terminals (ground = node 0 contributes nothing).
    let mut companion: Vec<(usize, usize, f64)> = Vec::new();
    // Per-device controlling-voltage limits, collected from the fragments by
    // node-voltage SYMBOL and mapped onto global unknown indices once the final
    // layout is known (below).
    let mut frag_limits: Vec<sane_device::FragmentLimit> = Vec::new();
    // One `Lowerer` for the whole device loop so its lowering cache (model-instance
    // templates) persists across instances within this single extraction.
    let mut lo = Lowerer::new(ctx);
    // Pre-count instances per template group so the frontends' first-instance
    // bundling decision (scalar-symbolic vs shared SIMD bundle) sees the whole
    // circuit rather than only the instances lowered so far.
    for inst in devices {
        if let Some(g) = inst.model.template_group() {
            *lo.instance_groups.entry(g).or_insert(0) += 1;
        }
    }
    // Localise the device-loop cost: total vs. the `lower_behavioral` portion
    // (the rest is residual/charge consumption). Aggregated, logged once.
    let dev_t0 = sane_core::time::Instant::now();
    let mut lower_t = std::time::Duration::ZERO;
    for inst in devices {
        for (li, lj, g) in inst.model.companion() {
            let (a, b) = (inst.terminals[li], inst.terminals[lj]);
            if a != 0 {
                companion.push((a - 1, a - 1, g));
            }
            if b != 0 {
                companion.push((b - 1, b - 1, g));
            }
            if a != 0 && b != 0 {
                companion.push((a - 1, b - 1, -g));
                companion.push((b - 1, a - 1, -g));
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
        // and behavioral states are treated uniformly (one set of arrays, one
        // differential-classification rule).
        let lt = sane_core::time::Instant::now();
        let mut frag = inst
            .model
            .lower_behavioral(&mut lo, &term_v, &term_vdot, &ctrl_i);
        lower_t += lt.elapsed();
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
        let dev_extra_base = extra_x.len();
        let extras = std::mem::take(&mut lo.extras);
        debug_assert_eq!(
            frag.residuals.len(),
            extras.len(),
            "device fragment must return one residual per extra unknown"
        );
        // Transport delays (`absdelay`) minted by this device: shift the
        // device-relative extras positions onto the extra-unknown layout (the
        // global fixup below the loop adds the node/branch offset).
        for dl in std::mem::take(&mut lo.delays) {
            delays.push(DelaySpec {
                src: dev_extra_base + dl.src_extra,
                out: dev_extra_base + dl.out_extra,
                hist: dl.hist,
                tau: dl.tau,
            });
        }
        for ex in &extras {
            if let Some(c) = ex.dc_seed {
                dc_seeds.push((ex.name.clone(), c));
            }
            extra_names.push(ex.name.clone());
            extra_kinds.push(ex.kind);
            extra_x.push(ex.value_sym);
            // Keep the derivative symbol; whether it is a differential DOF is
            // decided uniformly below from whether it appears in the residuals.
            extra_xdot.push(ex.xdot_sym);
        }
        for (k, &nd) in inst.terminals.iter().enumerate() {
            if nd != 0 {
                node_terms[nd - 1].push(frag.terminal_currents[k]);
            }
        }
        extra_res.extend(frag.residuals);
        noise_sources.extend(frag.noise);
        for (k, ev) in frag.events.iter().enumerate() {
            let inst_name = inst
                .model
                .instance_name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("device{}", events.len()));
            events.push(EventSpec {
                g: ev.g,
                dir: ev.dir,
                name: format!("{inst_name}#{k}"),
            });
        }
        for (name, sym) in &frag.param_syms {
            if let Some(v) = inst.model.param_default(name) {
                param_defaults.insert(*sym, v);
            }
        }
        op_vars.extend(frag.op_vars);
        frag_limits.extend(frag.limits);
    }
    drop(lo); // release the &mut Graph borrow before reusing `ctx` below

    // Behavioral sources into the node KCL: `I=` injects its current expression,
    // `V=` injects its branch current (the constraint is added as a branch row).
    for b in circuit.behavioral() {
        match b.kind {
            BKind::I => {
                let ix = translate_bexpr(ctx, &b.expr, &v, &branch_i_of_name, t_e);
                add_current(ctx, &mut node_terms, b.a, b.b, ix);
            }
            BKind::V => {
                let pos = bhv_v_src
                    .iter()
                    .position(|&s| circuit.behavioral()[s].name == b.name);
                if let Some(p) = pos {
                    add_current(ctx, &mut node_terms, b.a, b.b, bhv_v_i[p]);
                }
            }
        }
    }
    sane_core::log::stage("dae/devices", dev_t0.elapsed());
    sane_core::log::stage("dae/devices_lower", lower_t);
    let (tpl_b, tpl_c, n_b, n_c) = sane_core::profile::take_tpl_stats();
    sane_core::log::stage(
        "dae/tpl_build",
        std::time::Duration::from_nanos(tpl_b as u64),
    );
    sane_core::log::stage(
        "dae/tpl_clone",
        std::time::Duration::from_nanos(tpl_c as u64),
    );
    sane_core::log::debug(&format!("dae/devices: {n_b} template builds, {n_c} clones"));

    // Fuse each node's signed current terms into one Reduce(Sum) residual.
    let node_res: Vec<ExprId> = node_terms
        .into_iter()
        .map(|terms| ctx.reduce(ReduceOp::Sum, terms))
        .collect();

    // Branch constraint residuals.
    let mut branch_res = Vec::with_capacity(branch_elem_idx.len());
    let mut branch_xdot: Vec<Option<SymbolId>> = Vec::with_capacity(branch_elem_idx.len());
    // Inductor name -> (index in branch_res, its idot expression), for mutuals.
    let mut inductor: std::collections::HashMap<String, (usize, ExprId)> =
        std::collections::HashMap::new();
    for &idx in &branch_elem_idx {
        let e = &circuit.elements()[idx];
        let dv = ctx.sub(v[e.a], v[e.b]);
        match e.kind {
            Kind::VoltageSource => {
                let val = source_value(ctx, e, t_e);
                let res = ctx.sub(dv, val);
                branch_res.push(res);
                branch_xdot.push(None);
            }
            Kind::Inductor => {
                let ls = value_sym(ctx, &e.name);
                let (idote, idots) = sym2(ctx, &format!("idot_{}", e.name));
                let lidot = ctx.mul(ls, idote);
                let res = ctx.sub(dv, lidot);
                inductor.insert(e.name.to_ascii_lowercase(), (branch_res.len(), idote));
                branch_res.push(res);
                branch_xdot.push(Some(idots));
            }
            Kind::Vcvs => {
                let gain = value_sym(ctx, &e.name);
                let (cp, cm) = e.ctrl.expect("VCVS control nodes");
                let dvc = ctx.sub(v[cp], v[cm]);
                let g_dvc = ctx.mul(gain, dvc);
                let res = ctx.sub(dv, g_dvc);
                branch_res.push(res);
                branch_xdot.push(None);
            }
            Kind::Ccvs => {
                // V(a) - V(b) - gain*I(ctrl) = 0.
                let gain = value_sym(ctx, &e.name);
                let cname = e.ctrl_elem.as_deref().expect("CCVS needs controller");
                let ictrl = branch_i_of_name[&cname.to_ascii_lowercase()];
                let g_i = ctx.mul(gain, ictrl);
                let res = ctx.sub(dv, g_i);
                branch_res.push(res);
                branch_xdot.push(None);
            }
            _ => unreachable!(),
        }
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
            branch_res[ix] = ctx.sub(branch_res[ix], m_idot_y);
            let m_idot_x = ctx.mul(m, idot_x);
            branch_res[iy] = ctx.sub(branch_res[iy], m_idot_x);
        }
    }

    // Behavioral `V=` constraints: `v(np) - v(nm) - expr = 0`, one branch row
    // each (algebraic, like an independent voltage source).
    for (p, &bi) in bhv_v_src.iter().enumerate() {
        let b = &circuit.behavioral()[bi];
        let dv = ctx.sub(v[b.a], v[b.b]);
        let val = translate_bexpr(ctx, &b.expr, &v, &branch_i_of_name, t_e);
        let res = ctx.sub(dv, val);
        branch_res.push(res);
        branch_names.push(format!("i_{}", b.name));
        branch_i_sym.push(bhv_v_sym[p]);
        branch_xdot.push(None);
        branch_i.push(bhv_v_i[p]);
    }

    // Stitch residuals and the unknown/derivative vectors together: node KCL,
    // then branch constraints, then device extra-unknown rows (internal nodes +
    // behavioral states, both minted as fragment extras).
    let mut residuals = node_res;
    residuals.extend(branch_res);
    residuals.extend(extra_res);

    // Unified differential-classification: an unknown is a differential DOF iff
    // its derivative symbol actually appears in the assembled residuals (a charge
    // dq/dt, a Verilog-A ddt/idt, an inductor flux). This single rule governs
    // device-internal nodes and behavioral extras alike, replacing the former
    // per-path heuristics (a native `charge_dyn` set vs. a device-declared
    // `differential` flag) that could -- and did -- disagree.
    let diff_syms = ctx.free_symbols_in(&residuals);

    // Element stamps for template-based Jacobian assembly, derived uniformly from
    // the residual rows (node KCL sums split into their incident-current terms).
    let stamps = stamp::stamps_from_residuals(ctx, &residuals);

    let total = n + branch_i.len() + extra_x.len();
    let mut unknowns = Vec::with_capacity(total);
    let mut kinds = Vec::with_capacity(total);
    let mut x = Vec::with_capacity(total);
    let mut xdot = Vec::with_capacity(total);
    // Node voltages always carry a derivative slot (a capacitor onto the node is
    // common; the mass-matrix column is simply zero when none stores there).
    for k in 1..=n {
        unknowns.push(format!("v{k}"));
        kinds.push(UnknownKind::NodeVoltage);
        x.push(v_sym[k].unwrap());
        xdot.push(vdot_sym[k]);
    }
    for pos in 0..branch_i.len() {
        unknowns.push(branch_names[pos].clone());
        kinds.push(UnknownKind::BranchCurrent);
        x.push(branch_i_sym[pos]);
        xdot.push(branch_xdot[pos]);
    }
    for pos in 0..extra_x.len() {
        unknowns.push(extra_names[pos].clone());
        kinds.push(extra_kinds[pos]);
        x.push(extra_x[pos]);
        xdot.push(
            diff_syms
                .contains(&extra_xdot[pos])
                .then_some(extra_xdot[pos]),
        );
    }

    // Retain each independent source's stimulus shape (by element name) for the
    // analyses that need the structural facts lowering destroys (transient
    // breakpoints, the HB fundamental).
    let sources = circuit
        .elements()
        .iter()
        .filter_map(|e| e.source.map(|s| (e.name.clone(), s)))
        .collect();
    // Every independent V/I element by name (shaped or plain DC, top-level or
    // subcircuit-scoped): the element's value symbol doubles as its DC value,
    // and the DC source-stepping continuation ramps exactly this set.
    // Controlled sources are dependent elements and stay out.
    let source_names = circuit
        .elements()
        .iter()
        .filter(|e| matches!(e.kind, Kind::VoltageSource | Kind::CurrentSource))
        .map(|e| sane_mna::value_symbol_name(&e.name))
        .collect();

    // delay indices were extra-relative; shift onto the final layout
    let extra_base = n + branch_i.len();
    for dl in delays.iter_mut() {
        dl.src += extra_base;
        dl.out += extra_base;
    }

    // Controlling-voltage limits: node symbols -> global unknown indices, now
    // that the layout is final (covers external terminals and, for behavioral
    // devices, internal nodes alike; ground / non-unknown symbols -> None).
    let unknown_of: HashMap<SymbolId, usize> = x.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let limits: Vec<Limit> = frag_limits
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
        param_defaults,
        events,
        delays,
        unknowns,
        kinds,
        x,
        xdot,
        t,
        stamps,
        companion,
        noise_sources,
        op_vars,
        dc_seeds,
        limits,
        sources,
        source_names,
    }
}
