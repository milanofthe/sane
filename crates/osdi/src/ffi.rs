//! OSDI 0.4 ABI layout (clean-room from the published `osdi_0_4.h` interface
//! definition; the 0.3 fields form a layout-compatible prefix). Only the fields
//! SANE reads are given precise types; unused function pointers stay opaque so
//! the struct size/offsets match without pulling in their signatures.

#![allow(non_snake_case, dead_code)]

use std::os::raw::{c_char, c_void};

// --- eval flags --------------------------------------------------------------
pub const CALC_RESIST_RESIDUAL: u32 = 1;
pub const CALC_REACT_RESIDUAL: u32 = 2;
pub const CALC_RESIST_JACOBIAN: u32 = 4;
pub const CALC_REACT_JACOBIAN: u32 = 8;
pub const CALC_NOISE: u32 = 16;
pub const CALC_OP: u32 = 32;

pub const EVAL_RET_FLAG_FATAL: u32 = 2;
pub const EVAL_RET_FLAG_FINISH: u32 = 4;
pub const EVAL_RET_FLAG_STOP: u32 = 8;

// --- parameter flags ---------------------------------------------------------
pub const PARA_TY_MASK: u32 = 3;
pub const PARA_TY_REAL: u32 = 0;
pub const PARA_TY_INT: u32 = 1;
pub const PARA_TY_STR: u32 = 2;
pub const PARA_KIND_MASK: u32 = 3 << 30;
pub const PARA_KIND_MODEL: u32 = 0 << 30;
pub const PARA_KIND_INST: u32 = 1 << 30;
pub const PARA_KIND_OPVAR: u32 = 2 << 30;

pub const ACCESS_FLAG_READ: u32 = 0;
pub const ACCESS_FLAG_SET: u32 = 1;
pub const ACCESS_FLAG_INSTANCE: u32 = 4;

// --- jacobian entry flags ----------------------------------------------------
pub const JACOBIAN_ENTRY_RESIST_CONST: u32 = 1;
pub const JACOBIAN_ENTRY_REACT_CONST: u32 = 2;
pub const JACOBIAN_ENTRY_RESIST: u32 = 4;
pub const JACOBIAN_ENTRY_REACT: u32 = 8;

pub const INIT_ERR_OUT_OF_BOUNDS: u32 = 1;

// --- noise source types (OSDI 0.4) -------------------------------------------
pub const NOISE_TYPE_WHITE: u32 = 0;
pub const NOISE_TYPE_FLICKER: u32 = 1;
pub const NOISE_TYPE_TABLE: u32 = 2;

pub const LOG_LVL_MASK: u32 = 7;
pub const LOG_FMT_ERR: u32 = 16;

// --- structs -----------------------------------------------------------------

#[repr(C)]
pub struct OsdiSimParas {
    /// NULL-terminated array of option names.
    pub names: *mut *mut c_char,
    pub vals: *mut f64,
    /// NULL-terminated array of string-option names.
    pub names_str: *mut *mut c_char,
    pub vals_str: *mut *mut c_char,
}

#[repr(C)]
pub struct OsdiSimInfo {
    pub paras: OsdiSimParas,
    pub abstime: f64,
    pub prev_solve: *mut f64,
    pub prev_state: *mut f64,
    pub next_state: *mut f64,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union OsdiInitErrorPayload {
    pub parameter_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OsdiInitError {
    pub code: u32,
    pub payload: OsdiInitErrorPayload,
}

#[repr(C)]
pub struct OsdiInitInfo {
    pub flags: u32,
    pub num_errors: u32,
    pub errors: *mut OsdiInitError,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OsdiNodePair {
    pub node_1: u32,
    pub node_2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OsdiJacobianEntry {
    pub nodes: OsdiNodePair,
    pub react_ptr_off: u32,
    pub flags: u32,
}

#[repr(C)]
pub struct OsdiNode {
    pub name: *mut c_char,
    pub units: *mut c_char,
    pub residual_units: *mut c_char,
    pub resist_residual_off: u32,
    pub react_residual_off: u32,
    pub resist_limit_rhs_off: u32,
    pub react_limit_rhs_off: u32,
    pub is_flow: bool,
}

#[repr(C)]
pub struct OsdiParamOpvar {
    /// Array of `num_alias + 1` names (the first is canonical).
    pub name: *mut *mut c_char,
    pub num_alias: u32,
    pub description: *mut c_char,
    pub units: *mut c_char,
    pub flags: u32,
    pub len: u32,
}

#[repr(C)]
pub struct OsdiNoiseSource {
    pub name: *mut c_char,
    pub nodes: OsdiNodePair,
}

/// The OSDI 0.4 module descriptor. The prefix up to `load_jacobian_tran` is
/// layout-identical to OSDI 0.3; SANE requires the 0.4 tail (in particular
/// `write_jacobian_array_*`).
#[repr(C)]
pub struct OsdiDescriptor {
    pub name: *mut c_char,

    pub num_nodes: u32,
    pub num_terminals: u32,
    pub nodes: *mut OsdiNode,

    pub num_jacobian_entries: u32,
    pub jacobian_entries: *mut OsdiJacobianEntry,

    pub num_collapsible: u32,
    pub collapsible: *mut OsdiNodePair,
    /// Offset of the per-pair `bool collapsed[]` array in the instance data.
    pub collapsed_offset: u32,

    pub noise_sources: *mut OsdiNoiseSource,
    pub num_noise_src: u32,

    pub num_params: u32,
    pub num_instance_params: u32,
    pub num_opvars: u32,
    pub param_opvar: *mut OsdiParamOpvar,

    /// Offset of the `uint32_t node_mapping[num_nodes]` array in instance data.
    pub node_mapping_offset: u32,
    pub jacobian_ptr_resist_offset: u32,

    pub num_states: u32,
    pub state_idx_off: u32,

    pub bound_step_offset: u32,

    pub instance_size: u32,
    pub model_size: u32,

    pub access: unsafe extern "C" fn(
        inst: *mut c_void,
        model: *mut c_void,
        id: u32,
        flags: u32,
    ) -> *mut c_void,
    pub setup_model: unsafe extern "C" fn(
        handle: *mut c_void,
        model: *mut c_void,
        sim_params: *mut OsdiSimParas,
        res: *mut OsdiInitInfo,
    ),
    pub setup_instance: unsafe extern "C" fn(
        handle: *mut c_void,
        inst: *mut c_void,
        model: *mut c_void,
        temperature: f64,
        num_terminals: u32,
        sim_params: *mut OsdiSimParas,
        res: *mut OsdiInitInfo,
    ),
    pub eval: unsafe extern "C" fn(
        handle: *mut c_void,
        inst: *mut c_void,
        model: *mut c_void,
        info: *mut OsdiSimInfo,
    ) -> u32,
    pub load_noise: unsafe extern "C" fn(
        inst: *mut c_void,
        model: *mut c_void,
        freq: f64,
        noise_dens: *mut f64,
    ),
    pub load_residual_resist:
        unsafe extern "C" fn(inst: *mut c_void, model: *mut c_void, dst: *mut f64),
    pub load_residual_react:
        unsafe extern "C" fn(inst: *mut c_void, model: *mut c_void, dst: *mut f64),
    pub load_limit_rhs_resist: *const c_void,
    pub load_limit_rhs_react: *const c_void,
    pub load_spice_rhs_dc: *const c_void,
    pub load_spice_rhs_tran: *const c_void,
    pub load_jacobian_resist: *const c_void,
    pub load_jacobian_react: *const c_void,
    pub load_jacobian_tran: *const c_void,

    // ---- OSDI 0.4 additions ----
    pub given_flag_model: unsafe extern "C" fn(model: *mut c_void, id: u32) -> u32,
    pub given_flag_instance: unsafe extern "C" fn(inst: *mut c_void, id: u32) -> u32,
    pub num_resistive_jacobian_entries: u32,
    pub num_reactive_jacobian_entries: u32,
    pub write_jacobian_array_resist:
        unsafe extern "C" fn(inst: *mut c_void, model: *mut c_void, destination: *mut f64),
    pub write_jacobian_array_react:
        unsafe extern "C" fn(inst: *mut c_void, model: *mut c_void, destination: *mut f64),
    pub num_inputs: u32,
    pub inputs: *mut OsdiNodePair,
    pub load_jacobian_with_offset_resist: *const c_void,
    pub load_jacobian_with_offset_react: *const c_void,
    pub unknown_nature: *const c_void,
    pub residual_nature: *const c_void,
    pub noise_source_type: *mut u32,
    /// Per noise source, after an eval with `CALC_NOISE`: power density into
    /// `dens[k]` and (flicker) frequency exponent into `exp[k]`.
    pub load_noise_params:
        unsafe extern "C" fn(inst: *mut c_void, model: *mut c_void, dens: *mut f64, exp: *mut f64),
    pub module_flags: u32,
}
