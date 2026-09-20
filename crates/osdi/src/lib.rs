//! OSDI compiled-compact-model loader: OpenVAF-compiled `.osdi` shared
//! libraries as SANE devices.
//!
//! An OSDI model exposes residuals `f(x)` (resistive) and `q(x)` (reactive,
//! `f + dq/dt = 0`) plus their first-order Jacobians through C callbacks. SANE
//! bridges them as an [`ExternBundle`]: each node's `f_i` and `q_i` become
//! opaque operators over the device's node voltages, the terminal current is
//! `f_i + d/dt q_i` (the `d[q_i]/dv_j` partial markers SANE's autodiff mints
//! are bound to the reactive Jacobian), and one `eval` call per Newton point
//! serves every output of the group.
//!
//! Trade-offs versus the symbolic Verilog-A frontend (`sane-veriloga`):
//! - parameters are baked numerically at load time (no parameter
//!   sensitivities, no temperature sweeps through the model),
//! - second derivatives of `f` are unavailable (Hessian-based analyses see
//!   NaN); second derivatives of `q` are bound to zero, matching what every
//!   OSDI-based simulator computes,
//! - noise sources are not imported yet.
//! What it buys: instant coverage of anything OpenVAF compiles, at native
//! evaluation speed. Requires OSDI 0.4 libraries (OpenVAF-reloaded).

use rustc_hash::FxHashMap as HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::{Arc, Mutex};

use rsdag::{time_derivative, ExprId, Graph};
use sane_device::{BehavioralFragment, DeviceModel, Lowerer};

mod ffi;
use ffi::*;

/// The host-side `osdi_log` sink: routed into SANE's logger.
unsafe extern "C" fn osdi_log_sink(_handle: *mut c_void, msg: *const c_char, lvl: u32) {
    if msg.is_null() {
        return;
    }
    let text = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    let text = format!("osdi: {}", text.trim_end());
    match lvl & LOG_LVL_MASK {
        0..=2 => sane_core::log::debug(&text),
        3 => sane_core::log::warn_captured(&text),
        _ => sane_core::log::error(&text),
    }
}

/// A loaded `.osdi` shared library and its module descriptors.
pub struct OsdiLib {
    /// Keeps the dylib mapped for as long as any module handle lives.
    _lib: libloading::Library,
    modules: Vec<Arc<OsdiModule>>,
}

/// One OSDI module descriptor inside a loaded library.
pub struct OsdiModule {
    /// Raw descriptor pointer; valid for the lifetime of the owning library
    /// (modules are only handed out inside the `Arc` that also keeps the
    /// library alive through `OsdiLib`).
    desc: *const OsdiDescriptor,
    pub name: String,
    /// Node names (terminals first), descriptor order.
    pub node_names: Vec<String>,
    pub num_terminals: usize,
    pub num_nodes: usize,
    /// Parameter name (lowercased, aliases included) -> descriptor id.
    param_ids: HashMap<String, u32>,
    /// Parameter id -> type (PARA_TY_*).
    param_ty: HashMap<u32, u32>,
}

// SAFETY: the descriptor is immutable static data inside the mapped dylib;
// calls that mutate model/instance state go through `EvalState`'s mutex.
unsafe impl Send for OsdiModule {}
unsafe impl Sync for OsdiModule {}

impl OsdiModule {
    /// Whether the module declares a parameter with this (case-insensitive) name.
    pub fn has_param(&self, name: &str) -> bool {
        self.param_ids.contains_key(&name.to_ascii_lowercase())
    }

    fn d(&self) -> &OsdiDescriptor {
        // SAFETY: see the `desc` field invariant.
        unsafe { &*self.desc }
    }
}

fn cstr(p: *mut c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: descriptor strings are NUL-terminated static data.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

impl OsdiLib {
    /// Load a `.osdi` shared library, check the OSDI version (0.4 required)
    /// and wire the log callback.
    pub fn load(path: &std::path::Path) -> Result<Arc<OsdiLib>, String> {
        // SAFETY: loading a shared object; the file is user-supplied, same
        // trust level as the deck itself.
        let lib = unsafe { libloading::Library::new(path) }
            .map_err(|e| format!("cannot load '{}': {e}", path.display()))?;
        let get_u32 = |name: &[u8]| -> Result<u32, String> {
            // SAFETY: symbol lookup of an exported u32 global.
            unsafe {
                lib.get::<*const u32>(name)
                    .map(|s| **s)
                    .map_err(|e| format!("missing symbol {}: {e}", String::from_utf8_lossy(name)))
            }
        };
        let major = get_u32(b"OSDI_VERSION_MAJOR\0")?;
        let minor = get_u32(b"OSDI_VERSION_MINOR\0")?;
        if (major, minor) < (0, 4) {
            return Err(format!(
                "'{}' exposes OSDI {major}.{minor}; SANE requires OSDI 0.4 \
                 (compile with OpenVAF-reloaded / openvaf-r)",
                path.display()
            ));
        }
        let n = get_u32(b"OSDI_NUM_DESCRIPTORS\0")? as usize;
        let stride = get_u32(b"OSDI_DESCRIPTOR_SIZE\0")? as usize;
        if stride < std::mem::size_of::<OsdiDescriptor>() {
            return Err(format!(
                "'{}': descriptor size {stride} smaller than the OSDI 0.4 layout",
                path.display()
            ));
        }
        // SAFETY: exported descriptor array of `n` entries with `stride` bytes each.
        let base = unsafe {
            lib.get::<*const u8>(b"OSDI_DESCRIPTORS\0")
                .map(|s| *s)
                .map_err(|e| format!("missing OSDI_DESCRIPTORS: {e}"))?
        };
        // Route model logging into SANE's logger.
        // SAFETY: `osdi_log` is an exported function-pointer variable.
        unsafe {
            if let Ok(slot) =
                lib.get::<*mut unsafe extern "C" fn(*mut c_void, *const c_char, u32)>(b"osdi_log\0")
            {
                slot.write(osdi_log_sink);
            }
        }

        let mut modules = Vec::with_capacity(n);
        for i in 0..n {
            // SAFETY: in-bounds descriptor slot; prefix layout per OSDI 0.4.
            let desc = unsafe { base.add(i * stride) } as *const OsdiDescriptor;
            // SAFETY: valid descriptor for the library's lifetime.
            let d = unsafe { &*desc };
            let node_names: Vec<String> = (0..d.num_nodes as usize)
                .map(|k| {
                    // SAFETY: `nodes` has `num_nodes` entries.
                    cstr(unsafe { &*d.nodes.add(k) }.name)
                })
                .collect();
            let mut param_ids = HashMap::default();
            let mut param_ty = HashMap::default();
            // `num_params` already includes the instance params; opvars follow.
            for id in 0..d.num_params {
                // SAFETY: `param_opvar` has `num_params + num_opvars` entries;
                // ids below `num_params` are parameters.
                let p = unsafe { &*d.param_opvar.add(id as usize) };
                param_ty.insert(id, p.flags & PARA_TY_MASK);
                for a in 0..=(p.num_alias as usize) {
                    // SAFETY: `name` holds `num_alias + 1` strings.
                    let nm = cstr(unsafe { *p.name.add(a) });
                    if !nm.is_empty() {
                        param_ids.insert(nm.to_ascii_lowercase(), id);
                    }
                }
            }
            modules.push(Arc::new(OsdiModule {
                desc,
                name: cstr(d.name),
                node_names,
                num_terminals: d.num_terminals as usize,
                num_nodes: d.num_nodes as usize,
                param_ids,
                param_ty,
            }));
        }
        Ok(Arc::new(OsdiLib { _lib: lib, modules }))
    }

    /// The modules this library defines.
    pub fn modules(&self) -> &[Arc<OsdiModule>] {
        &self.modules
    }
}

/// 16-byte-aligned zeroed byte buffer for model/instance data.
struct RawData {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}
impl RawData {
    fn new(size: usize) -> RawData {
        let layout = std::alloc::Layout::from_size_align(size.max(1), 16).expect("layout");
        // SAFETY: non-zero size, valid alignment.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "osdi data allocation failed");
        RawData { ptr, layout }
    }
    fn as_ptr(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }
}
impl Drop for RawData {
    fn drop(&mut self) {
        // SAFETY: allocated with the stored layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}
// SAFETY: exclusive ownership of the allocation; concurrent use is guarded by
// the `EvalState` mutex.
unsafe impl Send for RawData {}

/// Simulator options passed to setup/eval, with stable storage for the arrays.
struct SimParas {
    _names: Vec<CString>,
    name_ptrs: Vec<*mut c_char>,
    vals: Vec<f64>,
    empty_strs: Vec<*mut c_char>,
}
impl SimParas {
    fn new() -> SimParas {
        let names = vec![CString::new("gmin").unwrap()];
        let name_ptrs: Vec<*mut c_char> = names
            .iter()
            .map(|c| c.as_ptr() as *mut c_char)
            .chain(std::iter::once(std::ptr::null_mut()))
            .collect();
        SimParas {
            _names: names,
            name_ptrs,
            vals: vec![sane_core::constants::GMIN_DC],
            empty_strs: vec![std::ptr::null_mut()],
        }
    }
    fn raw(&mut self) -> OsdiSimParas {
        OsdiSimParas {
            names: self.name_ptrs.as_mut_ptr(),
            vals: self.vals.as_mut_ptr(),
            names_str: self.empty_strs.as_mut_ptr(),
            vals_str: self.empty_strs.as_mut_ptr(),
        }
    }
}
// SAFETY: the raw pointers point into the owned vectors above.
unsafe impl Send for SimParas {}

/// Where a descriptor node landed after applying the model's collapse hints:
/// its representative active node, or ground.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeRepr {
    Active(usize),
    Ground,
}

/// A set-up OSDI instance: model + instance data, the collapse-resolved node
/// topology, and the eval scratch. One per placed device; the bundle locks it
/// per evaluation.
struct EvalState {
    module: Arc<OsdiModule>,
    /// Keeps the dylib mapped for as long as the bundle (and thus any compiled
    /// tape referencing it) lives -- the descriptor and code pointers point
    /// into the mapped library.
    _lib: Arc<OsdiLib>,
    model: RawData,
    inst: RawData,
    paras: SimParas,
    /// Active descriptor nodes (repr == itself), in descriptor order:
    /// terminals first, then surviving internals. Bundle argument order.
    active: Vec<usize>,
    /// Noise sources, collapse-resolved: `(hi, lo)` as active-node indices
    /// (`None` = ground) plus the OSDI noise type (white / flicker; table
    /// sources are skipped with a warning at setup). Order matches
    /// `load_noise_params`' arrays.
    noise: Vec<(Option<usize>, Option<usize>, u32)>,
    pow_buf: Vec<f64>,
    exp_buf: Vec<f64>,
    /// Terminals the model collapsed away, as `(terminal, target)`:
    /// `None` = shorted to ground, `Some(t)` = shorted to terminal `t`. The
    /// device exposes each as a zero-volt source branch (flow unknown +
    /// constraint), since SANE terminals cannot merge with circuit ground.
    terminal_shorts: Vec<(usize, Option<usize>)>,
    /// Aggregated resistive Jacobian cells (row, col) in `active` indices and
    /// the descriptor entry slots summing into each.
    jr_cells: Vec<((usize, usize), Vec<usize>)>,
    jq_cells: Vec<((usize, usize), Vec<usize>)>,
    /// Descriptor-entry-order buffers for `write_jacobian_array_*`.
    jr_buf: Vec<f64>,
    jq_buf: Vec<f64>,
    res_buf: Vec<f64>,
    prev_solve: Vec<f64>,
    states: Vec<f64>,
    next_states: Vec<f64>,
}

impl EvalState {
    /// Allocate, bind parameters, run model+instance setup, resolve collapses.
    fn new(
        lib: Arc<OsdiLib>,
        module: Arc<OsdiModule>,
        inst_name: &str,
        params: &HashMap<String, f64>,
        temperature: f64,
    ) -> Result<EvalState, String> {
        let d = module.d();
        let model = RawData::new(d.model_size as usize);
        let inst = RawData::new(d.instance_size as usize);
        let mut paras = SimParas::new();

        // Bind parameters by name (case-insensitive, aliases included).
        for (name, &val) in params {
            let Some(&id) = module.param_ids.get(&name.to_ascii_lowercase()) else {
                sane_core::log::warn_captured(&format!(
                    "{inst_name} (osdi {}): unknown parameter '{name}' ignored",
                    module.name
                ));
                continue;
            };
            let ty = module.param_ty.get(&id).copied().unwrap_or(PARA_TY_REAL);
            // Try the model side first, then the instance side.
            // SAFETY: access() returns a pointer into model/instance data (or
            // null); writes match the declared parameter type.
            unsafe {
                let mut p = (d.access)(inst.as_ptr(), model.as_ptr(), id, ACCESS_FLAG_SET);
                if p.is_null() {
                    p = (d.access)(
                        inst.as_ptr(),
                        model.as_ptr(),
                        id,
                        ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE,
                    );
                }
                if p.is_null() {
                    sane_core::log::warn_captured(&format!(
                        "{inst_name} (osdi {}): parameter '{name}' not settable",
                        module.name
                    ));
                    continue;
                }
                match ty {
                    PARA_TY_INT => *(p as *mut i32) = val as i32,
                    _ => *(p as *mut f64) = val,
                }
            }
        }

        let mut info = OsdiInitInfo {
            flags: 0,
            num_errors: 0,
            errors: std::ptr::null_mut(),
        };
        let mut sp = paras.raw();
        // SAFETY: valid model/instance allocations and sim-paras arrays.
        unsafe {
            (d.setup_model)(std::ptr::null_mut(), model.as_ptr(), &mut sp, &mut info);
        }
        check_init(inst_name, &module, "setup_model", &info)?;
        let mut info = OsdiInitInfo {
            flags: 0,
            num_errors: 0,
            errors: std::ptr::null_mut(),
        };
        // SAFETY: as above; all terminals reported connected.
        unsafe {
            (d.setup_instance)(
                std::ptr::null_mut(),
                inst.as_ptr(),
                model.as_ptr(),
                temperature,
                d.num_terminals,
                &mut sp,
                &mut info,
            );
        }
        check_init(inst_name, &module, "setup_instance", &info)?;

        // Resolve the collapse decisions this setup made.
        let n = module.num_nodes;
        let mut parent: Vec<NodeRepr> = (0..n).map(NodeRepr::Active).collect();
        fn find(parent: &mut [NodeRepr], mut i: usize) -> NodeRepr {
            loop {
                match parent[i] {
                    NodeRepr::Ground => return NodeRepr::Ground,
                    NodeRepr::Active(p) if p == i => return NodeRepr::Active(i),
                    NodeRepr::Active(p) => {
                        parent[i] = parent[p];
                        i = p;
                    }
                }
            }
        }
        let mut terminal_shorts: Vec<(usize, Option<usize>)> = Vec::new();
        // SAFETY: `collapsed` is a bool array of `num_collapsible` in instance data.
        let collapsed = unsafe {
            std::slice::from_raw_parts(
                (inst.as_ptr() as *const u8).add(d.collapsed_offset as usize) as *const bool,
                d.num_collapsible as usize,
            )
        };
        for (k, &c) in collapsed.iter().enumerate() {
            if !c {
                continue;
            }
            // SAFETY: `collapsible` has `num_collapsible` entries.
            let pair = unsafe { &*d.collapsible.add(k) };
            let a = pair.node_1 as usize;
            let b = pair.node_2;
            let ra = find(&mut parent, a);
            let rb = if b == u32::MAX {
                NodeRepr::Ground
            } else {
                find(&mut parent, b as usize)
            };
            let nt = module.num_terminals;
            match (ra, rb) {
                (NodeRepr::Active(x), NodeRepr::Active(y)) if x != y => {
                    // Keep terminals as representatives. Two terminals merging
                    // becomes a zero-volt source branch on the circuit side.
                    let (keep, gone) = if x < nt && y < nt {
                        terminal_shorts.push((x.max(y), Some(x.min(y))));
                        (x.min(y), x.max(y))
                    } else if x < nt {
                        (x, y)
                    } else if y < nt {
                        (y, x)
                    } else {
                        (x.min(y), x.max(y))
                    };
                    parent[gone] = NodeRepr::Active(keep);
                }
                (NodeRepr::Active(x), NodeRepr::Ground)
                | (NodeRepr::Ground, NodeRepr::Active(x)) => {
                    if x < nt {
                        terminal_shorts.push((x, None));
                    }
                    parent[x] = NodeRepr::Ground;
                }
                _ => {}
            }
        }
        let repr: Vec<NodeRepr> = (0..n).map(|i| find(&mut parent, i)).collect();
        let active: Vec<usize> = (0..n).filter(|&i| repr[i] == NodeRepr::Active(i)).collect();
        let mut active_pos = vec![usize::MAX; n];
        for (pos, &i) in active.iter().enumerate() {
            active_pos[i] = pos;
        }

        // node_mapping: descriptor node -> index into prev_solve. Collapsed
        // nodes read their representative's voltage; ground reads a pinned 0
        // slot appended at the end.
        let ground_slot = active.len();
        // SAFETY: `node_mapping` is a u32 array of `num_nodes` in instance data.
        let mapping = unsafe {
            std::slice::from_raw_parts_mut(
                (inst.as_ptr() as *mut u8).add(d.node_mapping_offset as usize) as *mut u32,
                n,
            )
        };
        for i in 0..n {
            mapping[i] = match repr[i] {
                NodeRepr::Active(r) => active_pos[r] as u32,
                NodeRepr::Ground => ground_slot as u32,
            };
        }

        // Aggregated Jacobian cells, in the write_jacobian_array_* entry order.
        // SAFETY: `jacobian_entries` has `num_jacobian_entries` entries.
        let entries = unsafe {
            std::slice::from_raw_parts(d.jacobian_entries, d.num_jacobian_entries as usize)
        };
        let mut jr_cells: Vec<((usize, usize), Vec<usize>)> = Vec::new();
        let mut jq_cells: Vec<((usize, usize), Vec<usize>)> = Vec::new();
        let (mut jr_slot, mut jq_slot) = (0usize, 0usize);
        for e in entries {
            let cell = |a: u32, b: u32| -> Option<(usize, usize)> {
                let ra = repr[a as usize];
                let rb = repr[b as usize];
                match (ra, rb) {
                    (NodeRepr::Active(x), NodeRepr::Active(y)) => {
                        Some((active_pos[x], active_pos[y]))
                    }
                    _ => None, // ground row/col: no unknown behind it
                }
            };
            if e.flags & JACOBIAN_ENTRY_RESIST != 0 {
                if let Some(c) = cell(e.nodes.node_1, e.nodes.node_2) {
                    match jr_cells.iter_mut().find(|(k, _)| *k == c) {
                        Some((_, slots)) => slots.push(jr_slot),
                        None => jr_cells.push((c, vec![jr_slot])),
                    }
                }
                jr_slot += 1;
            }
            if e.flags & JACOBIAN_ENTRY_REACT != 0 {
                if let Some(c) = cell(e.nodes.node_1, e.nodes.node_2) {
                    match jq_cells.iter_mut().find(|(k, _)| *k == c) {
                        Some((_, slots)) => slots.push(jq_slot),
                        None => jq_cells.push((c, vec![jq_slot])),
                    }
                }
                jq_slot += 1;
            }
        }
        debug_assert_eq!(jr_slot, d.num_resistive_jacobian_entries as usize);
        debug_assert_eq!(jq_slot, d.num_reactive_jacobian_entries as usize);

        // Noise sources: node pair through the collapse mapping, plus the type.
        // SAFETY: `noise_sources` / `noise_source_type` have `num_noise_src`
        // entries (OSDI 0.4).
        let mut noise: Vec<(Option<usize>, Option<usize>, u32)> = Vec::new();
        let nsrc = d.num_noise_src as usize;
        for k in 0..nsrc {
            let src = unsafe { &*d.noise_sources.add(k) };
            let ty = unsafe { *d.noise_source_type.add(k) };
            let map_node = |raw: u32| -> Option<usize> {
                if raw == u32::MAX {
                    return None; // ground
                }
                match repr[raw as usize] {
                    NodeRepr::Active(r) => Some(active_pos[r]),
                    NodeRepr::Ground => None,
                }
            };
            if ty == NOISE_TYPE_TABLE {
                sane_core::log::warn_captured(&format!(
                    "{inst_name} (osdi {}): tabular noise source '{}' not imported yet",
                    module.name,
                    cstr(src.name)
                ));
            }
            noise.push((map_node(src.nodes.node_1), map_node(src.nodes.node_2), ty));
        }

        Ok(EvalState {
            model,
            inst,
            paras,
            jr_buf: vec![0.0; jr_slot],
            jq_buf: vec![0.0; jq_slot],
            res_buf: vec![0.0; ground_slot + 1],
            pow_buf: vec![0.0; nsrc],
            exp_buf: vec![0.0; nsrc],
            noise,
            prev_solve: vec![0.0; ground_slot + 1],
            states: vec![0.0; d.num_states as usize],
            next_states: vec![0.0; d.num_states as usize],
            jr_cells,
            jq_cells,
            active,
            terminal_shorts,
            module,
            _lib: lib,
        })
    }

    /// One full evaluation at the node voltages `v` (active order): residuals
    /// `f`/`q` per active node, the aggregated Jacobian cell values, and the
    /// per-source noise power / exponent.
    fn eval(
        &mut self,
        v: &[f64],
        f: &mut [f64],
        q: &mut [f64],
        jr: &mut [f64],
        jq: &mut [f64],
        npow: &mut [f64],
        nexp: &mut [f64],
    ) {
        let d = self.module.d();
        let n_active = self.active.len();
        self.prev_solve[..n_active].copy_from_slice(&v[..n_active]);
        self.prev_solve[n_active] = 0.0; // ground slot
        let sp = self.paras.raw();
        let mut info = OsdiSimInfo {
            paras: OsdiSimParas {
                names: sp.names,
                vals: sp.vals,
                names_str: sp.names_str,
                vals_str: sp.vals_str,
            },
            abstime: 0.0,
            prev_solve: self.prev_solve.as_mut_ptr(),
            prev_state: self.states.as_mut_ptr(),
            next_state: self.next_states.as_mut_ptr(),
            flags: CALC_RESIST_RESIDUAL
                | CALC_REACT_RESIDUAL
                | CALC_RESIST_JACOBIAN
                | CALC_REACT_JACOBIAN
                | CALC_NOISE,
        };
        // SAFETY: model/instance were set up in `new`; buffers are sized to the
        // descriptor's counts; node_mapping points into prev_solve.
        unsafe {
            let ret = (d.eval)(
                std::ptr::null_mut(),
                self.inst.as_ptr(),
                self.model.as_ptr(),
                &mut info,
            );
            if ret & (EVAL_RET_FLAG_FATAL | EVAL_RET_FLAG_FINISH | EVAL_RET_FLAG_STOP) != 0 {
                sane_core::log::warn_captured(&format!(
                    "osdi {}: eval requested stop/fatal (flags {ret:#x})",
                    self.module.name
                ));
            }
            // Residuals arrive MATRIX-MAPPED: the model scatters node i's row
            // into dst[node_mapping[i]], so collapsed nodes already sum into
            // their representative's slot and the ground row lands in the
            // trailing ground slot (ignored). Active slot k is simply dst[k].
            let n_active = self.active.len();
            self.res_buf.fill(0.0);
            (d.load_residual_resist)(
                self.inst.as_ptr(),
                self.model.as_ptr(),
                self.res_buf.as_mut_ptr(),
            );
            f.copy_from_slice(&self.res_buf[..n_active]);
            self.res_buf.fill(0.0);
            (d.load_residual_react)(
                self.inst.as_ptr(),
                self.model.as_ptr(),
                self.res_buf.as_mut_ptr(),
            );
            q.copy_from_slice(&self.res_buf[..n_active]);
            (d.write_jacobian_array_resist)(
                self.inst.as_ptr(),
                self.model.as_ptr(),
                self.jr_buf.as_mut_ptr(),
            );
            (d.write_jacobian_array_react)(
                self.inst.as_ptr(),
                self.model.as_ptr(),
                self.jq_buf.as_mut_ptr(),
            );
            if !self.noise.is_empty() {
                self.pow_buf.fill(0.0);
                self.exp_buf.fill(0.0);
                (d.load_noise_params)(
                    self.inst.as_ptr(),
                    self.model.as_ptr(),
                    self.pow_buf.as_mut_ptr(),
                    self.exp_buf.as_mut_ptr(),
                );
            }
        }
        for (k, (_, slots)) in self.jr_cells.iter().enumerate() {
            jr[k] = slots.iter().map(|&s| self.jr_buf[s]).sum();
        }
        for (k, (_, slots)) in self.jq_cells.iter().enumerate() {
            jq[k] = slots.iter().map(|&s| self.jq_buf[s]).sum();
        }
        npow.copy_from_slice(&self.pow_buf);
        nexp.copy_from_slice(&self.exp_buf);
    }
}

fn check_init(
    inst: &str,
    module: &OsdiModule,
    what: &str,
    info: &OsdiInitInfo,
) -> Result<(), String> {
    if info.num_errors == 0 {
        return Ok(());
    }
    let mut msgs = Vec::new();
    for k in 0..info.num_errors as usize {
        // SAFETY: `errors` holds `num_errors` entries when num_errors > 0.
        let e = unsafe { &*info.errors.add(k) };
        if e.code == INIT_ERR_OUT_OF_BOUNDS {
            // SAFETY: payload is the parameter id for this error code.
            let id = unsafe { e.payload.parameter_id };
            let name = module
                .param_ids
                .iter()
                .find(|(_, &v)| v == id)
                .map(|(k, _)| k.clone())
                .unwrap_or_else(|| format!("#{id}"));
            msgs.push(format!("parameter '{name}' out of bounds"));
        } else {
            msgs.push(format!("error code {}", e.code));
        }
    }
    Err(format!(
        "{inst} (osdi {}): {what}: {}",
        module.name,
        msgs.join("; ")
    ))
}

/// The [`rsdag::ExternBundle`] serving one placed OSDI instance: outputs
/// are `[f_0..f_{n-1}, q_0..q_{n-1}, jr cells..., jq cells..., noise powers...,
/// noise exponents..., zero]` over the active node voltages as arguments.
struct OsdiBundle {
    state: Mutex<EvalState>,
    n_active: usize,
    n_jr: usize,
    n_jq: usize,
    n_noise: usize,
}

impl rsdag::ExternBundle for OsdiBundle {
    fn n_outputs(&self) -> usize {
        2 * self.n_active + self.n_jr + self.n_jq + 2 * self.n_noise + 1
    }
    fn call_into(&self, args: &[f64], _work: &mut [f64], out: &mut [f64]) {
        // The compact model keeps its own state behind the lock; nothing of
        // ours lives in the caller's buffer.
        let mut st = self.state.lock().unwrap();
        let (n, njr, njq, nn) = (self.n_active, self.n_jr, self.n_jq, self.n_noise);
        let (f, rest) = out.split_at_mut(n);
        let (q, rest) = rest.split_at_mut(n);
        let (jr, rest) = rest.split_at_mut(njr);
        let (jq, rest) = rest.split_at_mut(njq);
        let (npow, rest) = rest.split_at_mut(nn);
        let (nexp, rest) = rest.split_at_mut(nn);
        st.eval(args, f, q, jr, jq, npow, nexp);
        rest[0] = 0.0; // the shared zero slot (unavailable second derivatives)
    }
}

/// A placed OSDI device: wraps the shared set-up instance as a SANE
/// [`DeviceModel`] via the bundle bridge.
pub struct OsdiDevice {
    pub name: String,
    module: Arc<OsdiModule>,
    /// Kept so the dylib stays mapped while devices exist.
    _lib: Arc<OsdiLib>,
    state: Mutex<Option<Arc<OsdiBundle>>>,
    params: HashMap<String, f64>,
    temperature: f64,
}

impl OsdiDevice {
    pub fn new(
        name: impl Into<String>,
        lib: Arc<OsdiLib>,
        module: Arc<OsdiModule>,
        params: HashMap<String, f64>,
        temperature: f64,
    ) -> OsdiDevice {
        OsdiDevice {
            name: name.into(),
            module,
            _lib: lib,
            state: Mutex::new(None),
            params,
            temperature,
        }
    }

    /// Evaluate residuals/Jacobian at explicit active-node voltages (debug /
    /// test aid; active order = terminals then surviving internals).
    pub fn debug_eval(&self, v: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let bundle = self.state.lock().unwrap().clone().expect("setup first");
        let mut st = bundle.state.lock().unwrap();
        let n = st.n_active();
        let (mut f, mut q) = (vec![0.0; n], vec![0.0; n]);
        let mut jr = vec![0.0; bundle.n_jr];
        let mut jq = vec![0.0; bundle.n_jq];
        let mut np = vec![0.0; bundle.n_noise];
        let mut ne = vec![0.0; bundle.n_noise];
        st.eval(v, &mut f, &mut q, &mut jr, &mut jq, &mut np, &mut ne);
        (f, q, jr)
    }

    /// Set up the instance (parameters, collapse resolution) eagerly so load
    /// errors surface at parse time. Idempotent.
    pub fn setup(&self) -> Result<(), String> {
        let mut slot = self.state.lock().unwrap();
        if slot.is_some() {
            return Ok(());
        }
        let st = EvalState::new(
            self._lib.clone(),
            self.module.clone(),
            &self.name,
            &self.params,
            self.temperature,
        )?;
        let (n_active, n_jr, n_jq, n_noise) = (
            st.active.len(),
            st.jr_cells.len(),
            st.jq_cells.len(),
            st.noise.len(),
        );
        *slot = Some(Arc::new(OsdiBundle {
            state: Mutex::new(st),
            n_active,
            n_jr,
            n_jq,
            n_noise,
        }));
        Ok(())
    }
}

impl DeviceModel for OsdiDevice {
    fn n_terminals(&self) -> usize {
        self.module.num_terminals
    }

    fn lower_behavioral(
        &self,
        lo: &mut Lowerer,
        terminal_v: &[ExprId],
        terminal_vdot: &[ExprId],
        _control_i: &[ExprId],
    ) -> BehavioralFragment {
        self.setup().expect("osdi setup validated at load time");
        let bundle = self.state.lock().unwrap().clone().expect("set up above");
        let st = bundle.state.lock().unwrap();
        let nt = self.module.num_terminals;
        let n_active = st.n_active();
        let prefix = format!("osdi.{}", self.name);

        // Arguments: terminal voltages, then freshly minted internal unknowns
        // (active order); derivative expressions alongside for ddt. A shorted
        // terminal is not active (its model row merged away); handled below.
        let terminal_shorts = st.terminal_shorts.clone();
        let mut args: Vec<ExprId> = Vec::with_capacity(n_active);
        let mut arg_dots: Vec<ExprId> = Vec::with_capacity(n_active);
        let mut deriv_of = rustc_hash::FxHashMap::default();
        for &node in st.active.iter() {
            if node < nt {
                args.push(terminal_v[node]);
                arg_dots.push(terminal_vdot[node]);
            } else {
                let label = format!("{}.{}", self.name, self.module.node_names[node]);
                let (v, vdot) =
                    lo.unknown_kind(&label, sane_device::UnknownKind::NodeVoltage, false);
                args.push(v);
                arg_dots.push(vdot);
            }
        }
        let ctx = lo.ctx();
        for (k, &a) in args.iter().enumerate() {
            if let rsdag::Node::Symbol(s) = ctx.node(a) {
                deriv_of.insert(*s, arg_dots[k]);
            }
        }

        // Register the bundle and bind every output / derivative-marker slot.
        let (n_jr, n_jq) = (bundle.n_jr, bundle.n_jq);
        let jr_cells: Vec<(usize, usize)> = st.jr_cells.iter().map(|(c, _)| *c).collect();
        let jq_cells: Vec<(usize, usize)> = st.jq_cells.iter().map(|(c, _)| *c).collect();
        let noise_meta = st.noise.clone();
        drop(st);
        // The compiled model as an extern function of the graph: outputs are
        // the bundle's slots (f_i, q_i, then the noise powers and exponents);
        // the derivative outputs it can supply are the aggregated Jacobian
        // cells, declared per (output, argument), zero where structurally
        // absent. Second derivatives of q are unavailable from OSDI and stay
        // zero -- exactly the term every f/q-formulation simulator omits
        // (state-dependent-capacitance curvature); so do the PSD partials.
        let n_noise = bundle.n_noise;
        let mut outputs: Vec<rsdag::Output> = Vec::with_capacity(2 * n_active + 2 * n_noise);
        for i in 0..n_active {
            outputs.push(rsdag::Output::Slot(i as u32));
        }
        for i in 0..n_active {
            outputs.push(rsdag::Output::Slot((n_active + i) as u32));
        }
        for k in 0..n_noise {
            outputs.push(rsdag::Output::Slot((2 * n_active + n_jr + n_jq + k) as u32));
        }
        for k in 0..n_noise {
            outputs.push(rsdag::Output::Slot(
                (2 * n_active + n_jr + n_jq + n_noise + k) as u32,
            ));
        }
        let fid = ctx.define_extern_func(&prefix, n_active, bundle.clone(), outputs);
        let mut jr_of = HashMap::default();
        for (slot, &(r, c)) in jr_cells.iter().enumerate() {
            jr_of.insert((r, c), (2 * n_active + slot) as u32);
        }
        let mut jq_of = HashMap::default();
        for (slot, &(r, c)) in jq_cells.iter().enumerate() {
            jq_of.insert((r, c), (2 * n_active + n_jr + slot) as u32);
        }
        for i in 0..n_active {
            for k in 0..n_active {
                let df = jr_of
                    .get(&(i, k))
                    .map(|&s| rsdag::Output::Slot(s))
                    .unwrap_or(rsdag::Output::Zero);
                ctx.declare_derivative(fid, i as u32, k as u32, df);
                let dq = jq_of
                    .get(&(i, k))
                    .map(|&s| rsdag::Output::Slot(s))
                    .unwrap_or(rsdag::Output::Zero);
                ctx.declare_derivative(fid, (n_active + i) as u32, k as u32, dq);
            }
        }
        let out_f = |i: usize| i as u32;
        let out_q = |i: usize| (n_active + i) as u32;
        let out_pow = |k: usize| (2 * n_active + k) as u32;
        let out_exp = |k: usize| (2 * n_active + n_noise + k) as u32;

        // Rows per ACTIVE node: f_i + d/dt q_i.
        let active_nodes: Vec<usize> = {
            let st = bundle.state.lock().unwrap();
            st.active.clone()
        };
        let mut rows: Vec<ExprId> = Vec::with_capacity(n_active);
        for i in 0..n_active {
            let f = ctx.call(fid, out_f(i), &args);
            let q = ctx.call(fid, out_q(i), &args);
            let dq = time_derivative(ctx, q, &deriv_of);
            rows.push(ctx.add(f, dq));
        }
        let zero = ctx.zero();
        let mut terminal_currents = vec![zero; nt];
        let mut residuals: Vec<ExprId> = Vec::new();
        for (pos, &node) in active_nodes.iter().enumerate() {
            if node < nt {
                terminal_currents[node] = rows[pos];
            } else {
                residuals.push(rows[pos]);
            }
        }
        // Model-collapsed terminals: a zero-volt source branch pins the
        // terminal to its target (ground or the representative terminal); the
        // branch current carries whatever the external circuit pushes through.
        for &(t, target) in &terminal_shorts {
            let (i, _) = lo.unknown_kind(
                &format!("{}.short_{}", self.name, self.module.node_names[t]),
                sane_device::UnknownKind::BranchCurrent,
                false,
            );
            let ctx = lo.ctx();
            terminal_currents[t] = ctx.add(terminal_currents[t], i);
            let constraint = match target {
                Some(rep) => {
                    let ni = ctx.neg(i);
                    terminal_currents[rep] = ctx.add(terminal_currents[rep], ni);
                    ctx.sub(terminal_v[t], terminal_v[rep])
                }
                None => terminal_v[t],
            };
            residuals.push(constraint);
        }
        // Noise sources: power / exponent as bundle outputs over the same
        // argument group (computed in the same eval as the residuals), node
        // pair by voltage symbol. Table sources and collapsed self-pairs are
        // skipped; the PSD's derivative markers (w.r.t. node voltages) bind to
        // zero -- the operating-point dependence of the PSD is not
        // differentiated through OSDI, matching other OSDI hosts.
        let ctx = lo.ctx();
        let mut noise = Vec::new();
        let sym_of = |ctx: &Graph, e: ExprId| match ctx.node(e) {
            rsdag::Node::Symbol(s) => Some(*s),
            _ => None,
        };
        for (k, &(hi, lo_n, ty)) in noise_meta.iter().enumerate() {
            if ty == ffi::NOISE_TYPE_TABLE || (hi == lo_n) {
                continue; // table: unsupported; self-pair: no net injection
            }
            let psd = ctx.call(fid, out_pow(k), &args);
            let flicker_exp = if ty == ffi::NOISE_TYPE_FLICKER {
                ctx.call(fid, out_exp(k), &args)
            } else {
                ctx.zero()
            };
            let node_sym = |idx: Option<usize>| idx.and_then(|i| sym_of(ctx, args[i]));
            noise.push(sane_device::NoiseSource {
                hi: node_sym(hi),
                lo: node_sym(lo_n),
                psd,
                flicker_exp,
                table: Vec::new(),
            });
        }
        BehavioralFragment {
            param_syms: Vec::new(),
            events: Vec::new(),
            terminal_currents,
            residuals,
            noise,
            op_vars: Vec::new(),
            limits: Vec::new(),
        }
    }
}

impl EvalState {
    fn n_active(&self) -> usize {
        self.active.len()
    }
}
