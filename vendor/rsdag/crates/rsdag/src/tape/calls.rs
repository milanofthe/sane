//! Calls as every executor runs them: the tape interpreter and the native
//! code's host routine hand a call's description to [`run_call`], and a
//! stage of independent calls to [`run_stage`], so gathering, phases,
//! batches and stages have one implementation.
//!
//! A call works in three regions of its program's work buffer: the slots
//! (its operands, outputs and instance states), the gather area (its
//! arguments, group-major at the offset the tape gave it, apart from the
//! other calls of its stage) and the scratch its bundle works in.

use crate::extern_fn::ExternBundle;
use crate::scalar::Scalar;

use super::input_index;

/// Which entry of its bundle a call runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Everything, per instance.
    Whole,
    /// The main phase over the instances' states.
    Main,
    /// The instances' states from their pure arguments.
    Prolog,
}

/// A call over `n_groups` instances of `bundle`.
pub struct Call<'a> {
    pub bundle: &'a dyn ExternBundle,
    pub phase: Phase,
    pub n_groups: usize,
    /// Arguments per instance: the bundle's, or a prolog's pure ones.
    pub n_args: usize,
    /// Outputs per instance: the bundle's, or a prolog's state block.
    pub n_out: usize,
    /// The operands, `n_groups` times `n_in` (a slot, or a tagged input);
    /// `None` when the arguments were gathered already.
    pub operands: Option<&'a [u32]>,
    /// The argument each of an instance's `n_in` operands is, when an
    /// instance gathers only some (see [`Op::Call`](super::Op::Call));
    /// `None` for all, in order.
    pub places: Option<&'a [u32]>,
    /// Where the arguments are in the gather area, the outputs and (a main
    /// phase) the states among the slots.
    pub args: usize,
    pub out: usize,
    pub state: usize,
    /// Ops it carries: its instances times its body's.
    pub ops: u64,
}

/// A region of the work buffer the pieces of work of a stage share.
#[derive(Clone, Copy)]
pub struct Shared<T>(*mut T, usize);

// SAFETY: the pieces of work of a stage access disjoint parts of a region,
// or read the same ones; see `plan_stages`.
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Sync> Sync for Shared<T> {}

impl<T: Copy> Shared<T> {
    pub fn new(region: &mut [T]) -> Self {
        Shared(region.as_mut_ptr(), region.len())
    }

    /// # Safety
    /// `ptr` holds `len` values, and nothing else accesses them in a way
    /// the calls run over it do not allow (see [`run_call`]).
    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        Shared(ptr, len)
    }

    /// # Safety
    /// No piece of work writes slot `k` while this runs.
    unsafe fn get(self, k: usize) -> T {
        assert!(k < self.1);
        unsafe { *self.0.add(k) }
    }

    /// # Safety
    /// No piece of work writes `k` while another reads it.
    unsafe fn set(self, k: usize, v: T) {
        assert!(k < self.1);
        unsafe { *self.0.add(k) = v }
    }

    /// # Safety
    /// No piece of work writes `at..at + n` while the slice lives.
    unsafe fn slice<'a>(self, at: usize, n: usize) -> &'a [T] {
        assert!(at + n <= self.1);
        unsafe { std::slice::from_raw_parts(self.0.add(at), n) }
    }

    /// # Safety
    /// No other piece of work reads or writes `at..at + n` while the slice
    /// lives.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut<'a>(self, at: usize, n: usize) -> &'a mut [T] {
        assert!(at + n <= self.1);
        unsafe { std::slice::from_raw_parts_mut(self.0.add(at), n) }
    }
}

/// Instances `groups` of `c`: their operands gathered into their arguments,
/// then its bundle's phase over them, batched for two or more. `scratch`
/// holds the bundle's [`work_len`](ExternBundle::work_len).
///
/// # Safety
/// `slots` and `gather` are the program's regions; nothing else accesses
/// the instances' arguments, outputs and states while this runs, and
/// nothing writes what they read.
pub unsafe fn run_call<T: Scalar>(
    c: &Call,
    slots: Shared<T>,
    gather: Shared<T>,
    inputs: &[T],
    groups: std::ops::Range<usize>,
    scratch: &mut [T],
) {
    let (na, no, b) = (c.n_args, c.n_out, c.bundle);
    let (g0, ng) = (groups.start, groups.len());
    let shape = match c.phase {
        Phase::Prolog => b.state_len(),
        _ => b.n_outputs(),
    };
    assert_eq!(shape, no, "bundle changed since compile");
    if let Some(ops) = c.operands {
        let ni = ops.len() / c.n_groups.max(1);
        let value = |k: u32| match input_index(k) {
            Some(i) => inputs.get(i as usize).copied().unwrap_or(T::nan()),
            None => unsafe { slots.get(k as usize) },
        };
        for g in groups.clone() {
            let (from, to) = (&ops[g * ni..(g + 1) * ni], c.args + g * na);
            match c.places {
                None => from
                    .iter()
                    .enumerate()
                    .for_each(|(j, &k)| unsafe { gather.set(to + j, value(k)) }),
                Some(p) => from
                    .iter()
                    .zip(p)
                    .for_each(|(&k, &j)| unsafe { gather.set(to + j as usize, value(k)) }),
            }
        }
    }
    let args = unsafe { gather.slice(c.args + g0 * na, ng * na) };
    let out = unsafe { slots.slice_mut(c.out + g0 * no, ng * no) };
    match c.phase {
        Phase::Whole if ng >= 2 => T::call_bundle_batch(b, args, ng, na, out),
        Phase::Whole => T::call_bundle_whole(b, args, scratch, out),
        Phase::Main => {
            let sl = b.state_len();
            let states = unsafe { slots.slice(c.state + g0 * sl, ng * sl) };
            T::call_bundle_main_batch(b, args, states, ng, na, scratch, out);
        }
        Phase::Prolog => T::call_bundle_prolog_batch(b, args, ng, na, scratch, out),
    }
}

/// A stage of `n` independent calls, the `k`th of them `call(k)`: on the
/// installed pool when it is worth it ([`crate::parallel`]), every block of
/// instances over scratch of its thread's, else one call after the other.
/// Either way the results are the serial ones, and nothing is allocated.
///
/// # Safety
/// As [`run_call`], for every call; and the calls of the stage read nothing
/// another one writes, their arguments apart in the gather area.
pub unsafe fn run_stage<'a, T: Scalar>(
    n: usize,
    call: impl Fn(usize) -> Call<'a> + Sync,
    slots: Shared<T>,
    gather: Shared<T>,
    inputs: &[T],
) {
    let ops: u64 = (0..n).map(|k| call(k).ops).sum();
    let run_groups = |c: &Call, groups: std::ops::Range<usize>| {
        crate::scratch::with_len(c.bundle.work_len(), T::zero(), |s| unsafe {
            run_call(c, slots, gather, inputs, groups, s)
        })
    };
    if !crate::parallel::worth(ops as usize) {
        for k in 0..n {
            let c = call(k);
            run_groups(&c, 0..c.n_groups);
        }
        return;
    }
    // Pieces of `bs` instances, call after call; the `it`th found by
    // counting through the calls (a stage has few).
    let bs = crate::parallel::block((0..n).map(|k| call(k).n_groups).sum());
    let pieces = |c: &Call| c.n_groups.div_ceil(bs);
    let run = |mut it: usize| {
        let mut k = 0;
        let c = loop {
            let c = call(k);
            if it < pieces(&c) {
                break c;
            }
            it -= pieces(&c);
            k += 1;
        };
        let g0 = it * bs;
        run_groups(&c, g0..(g0 + bs).min(c.n_groups));
    };
    let total = (0..n).map(|k| pieces(&call(k))).sum();
    crate::parallel::run(total, ops as usize, &run);
}
