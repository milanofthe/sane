//! The native backend: the op stream emitted straight to machine code.
//!
//! The tape already knows what a code generator needs: it is straight-line
//! code over a work array of slots, with every value's lifetime computed.
//! So this backend does the least a compiler can do: for each op, operands
//! come from a small register cache or a load from the work array, one
//! instruction computes, and the result stays in the cache. It is written
//! to its slot in the work array only when it has to be: on eviction, at a
//! host call (which clobbers the caller-saved part of the cache) and at the
//! chunk's end, and then only if some later op still reads it, which the
//! liveness knows. A value that dies inside the chunk never touches
//! memory; nothing is ever spilled anywhere but where the interpreter keeps
//! it anyway. Compile time is linear in the op count, and the chunks are
//! emitted in parallel into one executable mapping.
//!
//! A large program is executed once per evaluation, straight through, so
//! its cost is instruction fetch: the bytes per op. That is why nothing is
//! stored that need not be, why the frequent host routines sit in
//! callee-saved registers, and why the AArch64 side addresses the work
//! array through a moving window.
//!
//! Everything that is not one instruction goes to a host routine: the
//! transcendentals, `Min`/`Max` reductions and bundle calls, the last two
//! with their operands gathered into a scratch area at the end of the work
//! array, so no chunk ever touches the stack beyond its own frame.
//! Bit-exactness against the interpreter is the invariant: the same IEEE
//! operation sequence, reductions folded in the reference order, every
//! transcendental through the same host function.

use rayon::prelude::*;
use rsdag::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use rsdag::tape::input_index;
use rsdag::{ExternBundle, Instances, Tape};
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::host::{self, Bundles};
use crate::ir::{Dense, Kernel, KernelKind, Liveness, ROp, StageRole};
use crate::isa::{Arg, Arith, Base, IArg, Isa, Round};
use crate::{Batch, JitError, Lanes, Options};

#[cfg(target_arch = "aarch64")]
type Arch = crate::aarch64::A64;
#[cfg(target_arch = "x86_64")]
type Arch = crate::x86_64::X64<1>;

/// An emitted chunk: `(work, inputs, bundles)`. Unsafe to call: the code
/// trusts the work array to have the layout's length and the inputs the
/// ones it reads, which [`NativeTape::run`] checks.
type ChunkFn = unsafe extern "C" fn(*mut f64, *const f64, *const Bundles);

/// A tape compiled to native code. Evaluation mirrors [`Tape`]: a
/// caller-owned work buffer, inputs padded with NaN, and the prolog/main
/// split of a specialized tape.
pub struct NativeTape {
    /// Every chunk's code in one mapping, held while the chunks point into
    /// it, and each chunk's entry in it.
    _code: Mapping,
    chunks: Vec<ChunkFn>,
    prolog_chunks: usize,
    /// The call and kernel descriptors the code holds the addresses of.
    _descs: Vec<(Vec<host::CallDesc>, Vec<host::KernelDesc>)>,
    bundles: Bundles,
    /// The fold code tables of the accumulating kernels; the code holds
    /// their addresses.
    _tables: Vec<Box<[u32]>>,
    outputs: Vec<u32>,
    layout: Layout,
    n_inputs: usize,
    n_ops: usize,
    /// The tape's state prefix (see [`Tape::state_len`]); the slot layout
    /// is the tape's, so an interpreter's state serves here and back.
    state_len: usize,
    /// Instances per run: `1`, or the lane code's width (see
    /// [`compile_lanes`](NativeTape::compile_lanes)).
    lanes: usize,
    /// The inputs the prolog and the main phase read (a body's main phase
    /// reads its states' arguments, not the parameters its prolog took).
    reads: [Vec<u32>; 2],
    /// Per phase (prolog, main): its ops, and those of them that call the
    /// host (lane code calls it once per lane).
    phase_ops: [(usize, usize); 2],
}

/// The gather area lane code needs for `op`: a host call's float
/// arguments and results lane by lane, or a min/max's terms, one lane's
/// row of them and the results.
fn lane_gather_len(op: &ROp, l: usize) -> usize {
    match op {
        ROp::Reduce(_, rsdag::node::ReduceOp::Min | rsdag::node::ReduceOp::Max, a) => {
            a.len() * l + a.len() + l
        }
        _ if op.host().is_some() => 3 * l,
        _ => 0,
    }
}

/// A function body compiled natively, behind the bundle interface the
/// tape calls bodies through: emitted once, called per instance. A batch
/// of instances runs as its [`Batch`] says; either way the result is the
/// serial loop's, bit for bit.
struct NativeBody {
    tape: NativeTape,
    /// The same body over several instances at once, widest first, each
    /// where it compiles so and pays for some phase.
    lanes: Vec<LaneCode>,
    /// What an instance of the scalar code costs per phase (see
    /// [`lane_costs`]).
    scalar: [usize; 3],
    n_out: usize,
    /// The pure-argument flags of the body it replaces: its prolog runs on
    /// those, the rest NaN.
    pure: Vec<bool>,
    /// Per argument, its rank among the pure ones (a prolog's arguments
    /// are those alone).
    rank: Vec<Option<usize>>,
    batch: Batch,
}

/// A min or max over at most this many terms is a chain of instructions
/// where the ISA has one; longer ones go through the host routine, whose
/// call costs about as much as this many terms.
const INLINE_MINMAX_MAX: usize = 16;

impl NativeBody {
    /// The instances `groups` of a batch, one after the other on one work
    /// buffer: nothing is cleared or resized between them, since a tape
    /// writes every slot it reads.
    fn run_groups(
        &self,
        work: &mut Vec<f64>,
        args: &[f64],
        n_args: usize,
        out: &mut [f64],
        groups: std::ops::Range<usize>,
    ) {
        work.resize(self.tape.layout.total, 0.0);
        self.run_groups_into(work, args, n_args, out, groups);
    }

    /// [`run_groups`](Self::run_groups) over a buffer the caller sized to
    /// [`work_len`](ExternBundle::work_len).
    fn run_groups_into(
        &self,
        work: &mut [f64],
        args: &[f64],
        n_args: usize,
        out: &mut [f64],
        groups: std::ops::Range<usize>,
    ) {
        let n_out = self.n_out;
        for g in groups {
            let ins = &args[g * n_args..(g + 1) * n_args];
            self.tape.run(0..self.tape.chunks.len(), ins, work);
            for (k, &slot) in self.tape.outputs[..n_out].iter().enumerate() {
                out[g * n_out + k] = match input_index(slot) {
                    Some(i) => ins.get(i as usize).copied().unwrap_or(f64::NAN),
                    None => work[slot as usize],
                };
            }
        }
    }
}

/// `ops` in chunks of about `chunk_ops`, each ending after an op outside a
/// stage or after a stage's last call.
fn chunked(ops: &[ROp], chunk_ops: usize) -> Vec<&[ROp]> {
    let mut out = Vec::new();
    let mut start = 0;
    for (k, op) in ops.iter().enumerate() {
        let inside = matches!(op, ROp::Call(c) if c.stage == StageRole::Deferred);
        if k + 1 - start >= chunk_ops.max(1) && !inside {
            out.push(&ops[start..=k]);
            start = k + 1;
        }
    }
    if start < ops.len() {
        out.push(&ops[start..]);
    }
    out
}

/// The phase a run of lane code covers, in the order of [`LaneCode::pays`].
#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Prolog,
    Main,
    Whole,
}

/// A body's lane code of one width.
struct LaneCode {
    tape: NativeTape,
    /// What a block of it costs per phase (see [`lane_costs`]).
    block: [usize; 3],
    /// The inputs each phase reads, in the order of [`Phase`]: what moves
    /// into the lanes.
    reads: [Vec<u32>; 3],
}

impl LaneCode {
    /// Whether it pays for `phase` against the scalar code's `one`.
    fn pays(&self, phase: usize, one: usize) -> bool {
        self.block[phase] < self.tape.lanes * one
    }
}

/// What lane code costs per phase (prolog, main phase, whole call): a
/// block of its width, and an instance of the scalar code, in
/// op-equivalents measured on SANE's device bodies. A block does its ops'
/// work once for all its lanes, but moves the values it reads and writes
/// into and out of the lanes (inputs read, the state, the outputs) and
/// makes its host calls lane by lane; it costs as much with lanes left
/// empty.
fn lane_costs(lt: &NativeTape, n_state: usize, n_out: usize) -> ([usize; 3], [usize; 3]) {
    const MOVE: usize = 2;
    const HOST: usize = 6;
    let [(o0, h0), (o1, h1)] = lt.phase_ops;
    let [r0, r1] = [lt.reads[0].len(), lt.reads[1].len()];
    let phases = [
        (o0, h0, r0 + n_state),
        (o1, h1, r1 + n_state + n_out),
        (o0 + o1, h0 + h1, r0 + r1 + n_out),
    ];
    let block =
        phases.map(|(ops, host, moved)| ops - host + lt.lanes * (host * (1 + HOST) + moved * MOVE));
    (block, phases.map(|(ops, _, _)| ops))
}

impl NativeBody {
    /// The instances of a batch of `n` that lane code takes, handed to
    /// `run` as ranges of blocks: the widest lane code that pays
    /// for `phase` (the cheapest per instance) fills its blocks, and the
    /// rest goes the cheapest way by [`lane_costs`], in blocks of any width
    /// (a short one runs with lanes empty) or scalar. Returns the first
    /// instance left to the scalar code.
    fn lanes_first(
        &self,
        phase: Phase,
        n: usize,
        mut run: impl FnMut(&LaneCode, std::ops::Range<usize>),
    ) -> usize {
        let (k, one) = (phase as usize, self.scalar[phase as usize]);
        let pays = |lc: &&LaneCode| lc.pays(k, one);
        let Some(widest) = self.lanes.iter().find(pays) else {
            return 0;
        };
        let full = n / widest.tape.lanes * widest.tape.lanes;
        if full > 0 {
            run(widest, 0..full);
        }
        // The rest, fewer than a block: `best[m]` the cheapest way to take
        // `m` of it, and the lane code it starts with (`None`: scalar).
        let rest = n - full;
        let mut best = [(0usize, None::<usize>); 8];
        for m in 1..=rest {
            best[m] = (best[m - 1].0 + one, None);
            for (i, lc) in self.lanes.iter().enumerate().filter(|(_, lc)| pays(lc)) {
                let c = best[m.saturating_sub(lc.tape.lanes)].0 + lc.block[k];
                if c < best[m].0 {
                    best[m] = (c, Some(i));
                }
            }
        }
        let (mut g, mut m) = (full, rest);
        while m > 0 {
            match best[m].1 {
                Some(i) => {
                    let w = self.lanes[i].tape.lanes.min(m);
                    run(&self.lanes[i], g..g + w);
                    g += w;
                    m -= w;
                }
                None => m -= 1,
            }
        }
        g
    }

    /// The instances `at` lists at positions `block` through the lane
    /// code `lc`, its width at a time: lane `l` of every slot and input is
    /// the block's instance `l` (a short last block repeats its last
    /// instance, whose copies are dropped). A prolog's arguments are the
    /// pure ones, completed with NaN as
    /// [`prolog_into`](ExternBundle::prolog_into) completes them, and it
    /// writes `states`; a main phase reads `states`; a main phase or a
    /// whole run writes `out`.
    #[allow(clippy::too_many_arguments)]
    fn run_lanes(
        &self,
        lc: &LaneCode,
        phase: Phase,
        args: &[f64],
        at: &Instances,
        block: std::ops::Range<usize>,
        states: Option<&[f64]>,
        mut states_out: Option<&mut [f64]>,
        mut out: Option<&mut [f64]>,
    ) {
        let lt = &lc.tape;
        let (l, sl, n_out) = (lt.lanes, self.tape.state_len, self.n_out);
        let (n_args, stride) = (at.n_args, at.stride);
        let n_in = lt.n_inputs.max(n_args).max(self.pure.len());
        // Only the inputs the phase reads move into the lanes; a prolog's
        // arguments are the pure ones, input `k` the pure argument of its
        // rank among them.
        let read = &lc.reads[phase as usize];
        let range = match phase {
            Phase::Prolog => 0..lt.prolog_chunks,
            Phase::Main => lt.prolog_chunks..lt.chunks.len(),
            Phase::Whole => 0..lt.chunks.len(),
        };
        rsdag::scratch::with_len(lt.layout.total, 0.0, |work: &mut [f64]| {
            rsdag::scratch::with_len(n_in * l, f64::NAN, |ins: &mut [f64]| {
                let end = block.end;
                for c in block.step_by(l) {
                    for lane in 0..l {
                        let g = at.at((c + lane).min(end - 1));
                        let a = &args[g * n_args..(g + 1) * n_args];
                        for &k in read {
                            let k = k as usize;
                            ins[k * l + lane] = if phase == Phase::Prolog {
                                self.rank
                                    .get(k)
                                    .copied()
                                    .flatten()
                                    .map_or(f64::NAN, |r| a[r])
                            } else {
                                a.get(k).copied().unwrap_or(f64::NAN)
                            };
                        }
                        if let Some(st) = states {
                            for s in 0..sl {
                                work[s * l + lane] = st[g * stride + s];
                            }
                        }
                    }
                    lt.run(range.clone(), ins, work);
                    for lane in 0..l.min(end - c) {
                        let g = at.at(c + lane);
                        if let Some(st) = states_out.as_deref_mut() {
                            for s in 0..sl {
                                st[g * stride + s] = work[s * l + lane];
                            }
                        }
                        if let Some(o) = out.as_deref_mut() {
                            for (k, &slot) in lt.outputs[..n_out].iter().enumerate() {
                                o[g * n_out + k] = match input_index(slot) {
                                    Some(i) => args
                                        .get(g * n_args + i as usize)
                                        .copied()
                                        .filter(|_| (i as usize) < n_args)
                                        .unwrap_or(f64::NAN),
                                    None => work[slot as usize * l + lane],
                                };
                            }
                        }
                    }
                }
            })
        });
    }
}

impl ExternBundle for NativeBody {
    fn n_outputs(&self) -> usize {
        self.n_out
    }
    fn work_len(&self) -> usize {
        self.tape.layout.total + self.pure.len()
    }
    fn call_into(&self, args: &[f64], work: &mut [f64], out: &mut [f64]) {
        let w = &mut work[..self.tape.layout.total];
        self.run_groups_into(w, args, args.len(), out, 0..1);
    }
    fn state_len(&self) -> usize {
        self.tape.state_len
    }
    fn pure_args(&self) -> &[bool] {
        &self.pure
    }
    fn prolog_into(&self, pure: &[f64], work: &mut [f64], state: &mut [f64]) {
        let (w, a) = work.split_at_mut(self.tape.layout.total);
        let a = &mut a[..self.pure.len()];
        let mut p = pure.iter();
        for (x, &is_pure) in a.iter_mut().zip(&self.pure) {
            *x = if is_pure {
                *p.next().expect("one value per pure argument")
            } else {
                f64::NAN
            };
        }
        self.tape.run(0..self.tape.prolog_chunks, a, w);
        let sl = self.tape.state_len;
        state[..sl].copy_from_slice(&w[..sl]);
    }
    fn main_into(&self, args: &[f64], state: &[f64], work: &mut [f64], out: &mut [f64]) {
        let w = &mut work[..self.tape.layout.total];
        let sl = self.tape.state_len;
        w[..sl].copy_from_slice(&state[..sl]);
        self.tape
            .run(self.tape.prolog_chunks..self.tape.chunks.len(), args, w);
        for (k, &slot) in self.tape.outputs[..self.n_out].iter().enumerate() {
            out[k] = match input_index(slot) {
                Some(i) => args.get(i as usize).copied().unwrap_or(f64::NAN),
                None => w[slot as usize],
            };
        }
    }
    fn prolog_batch(&self, pure: &[f64], states: &mut [f64], at: &Instances) {
        let k0 = self.lanes_first(Phase::Prolog, at.len(), |lc, r| {
            self.run_lanes(
                lc,
                Phase::Prolog,
                pure,
                at,
                r,
                None,
                Some(&mut *states),
                None,
            )
        });
        let (na, sl, st) = (at.n_args, self.tape.state_len, at.stride);
        rsdag::scratch::with_len(self.work_len(), 0.0, |w| {
            for g in (k0..at.len()).map(|k| at.at(k)) {
                let p = &pure[g * na..(g + 1) * na];
                self.prolog_into(p, w, &mut states[g * st..g * st + sl]);
            }
        });
    }
    fn main_batch(&self, args: &[f64], states: &[f64], out: &mut [f64], at: &Instances) {
        let k0 = self.lanes_first(Phase::Main, at.len(), |lc, r| {
            self.run_lanes(
                lc,
                Phase::Main,
                args,
                at,
                r,
                Some(states),
                None,
                Some(&mut *out),
            )
        });
        let (na, sl, st, no) = (at.n_args, self.tape.state_len, at.stride, self.n_out);
        rsdag::scratch::with_len(self.work_len(), 0.0, |w| {
            for g in (k0..at.len()).map(|k| at.at(k)) {
                let a = &args[g * na..(g + 1) * na];
                let s = &states[g * st..g * st + sl];
                self.main_into(a, s, w, &mut out[g * no..(g + 1) * no]);
            }
        });
    }
    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        if self.batch == Batch::Serial {
            let at = Instances::first(n_groups, n_args, 0);
            let g0 = self.lanes_first(Phase::Whole, n_groups, |lc, r| {
                self.run_lanes(lc, Phase::Whole, args, &at, r, None, None, Some(&mut *out))
            });
            rsdag::scratch::with::<Vec<f64>, _>(|work| {
                self.run_groups(work, args, n_args, out, g0..n_groups)
            });
            return;
        }
        let parallel = match self.batch {
            Batch::Serial => false,
            Batch::Parallel { min_ops } => {
                n_groups >= 2 && self.n_out > 0 && n_groups * self.tape.n_ops >= min_ops
            }
        };
        if !parallel {
            rsdag::scratch::with::<Vec<f64>, _>(|work| {
                self.run_groups(work, args, n_args, out, 0..n_groups)
            });
            return;
        }
        // Blocks of instances per task, so a thread amortises the
        // scheduler's hand-offs over many bodies.
        let block = blocks(n_groups);
        out.par_chunks_mut(block * self.n_out)
            .enumerate()
            .for_each(|(b, dst)| {
                let g0 = b * block;
                let g1 = (g0 + block).min(n_groups);
                let ins = &args[g0 * n_args..g1 * n_args];
                rsdag::scratch::with::<Vec<f64>, _>(|work| {
                    self.run_groups(work, ins, n_args, dst, 0..g1 - g0)
                });
            });
    }
}

/// `body` as native code behind the bundle interface ([`NativeBody`]), with
/// its lane form where that pays.
fn native_body(
    body: &Tape,
    pure: &[bool],
    no: usize,
    opts: &Options,
) -> Result<Arc<dyn ExternBundle>, JitError> {
    let sl = body.state_len();
    let lanes = match opts.lanes {
        Lanes::Never => Vec::new(),
        _ => NativeTape::compile_lanes(body),
    };
    // Forced, the scalar code is never the cheaper way.
    let mut scalar = [usize::MAX / 16; 3];
    let lanes: Vec<LaneCode> = lanes
        .into_iter()
        .map(|lt| {
            let (block, one) = lane_costs(&lt, sl, no);
            if opts.lanes == Lanes::Auto {
                scalar = one;
            }
            let mut whole = [lt.reads[0].as_slice(), lt.reads[1].as_slice()].concat();
            whole.sort_unstable();
            whole.dedup();
            let reads = [lt.reads[0].clone(), lt.reads[1].clone(), whole];
            LaneCode {
                tape: lt,
                block,
                reads,
            }
        })
        .collect();
    let lanes = lanes
        .into_iter()
        .filter(|lc| (0..3).any(|k| lc.pays(k, scalar[k])))
        .collect();
    Ok(Arc::new(NativeBody {
        tape: NativeTape::compile_opts(body, opts, &[])?,
        lanes,
        scalar,
        n_out: no,
        pure: pure.to_vec(),
        rank: pure
            .iter()
            .scan(0, |next, &p| {
                *next += usize::from(p);
                Some(p.then(|| *next - 1))
            })
            .collect(),
        batch: opts.batch,
    }))
}

/// The options a native body's code was emitted under, as a key of the
/// body's backend cache.
fn options_key(opts: &Options) -> u64 {
    let batch = match opts.batch {
        Batch::Serial => 0,
        Batch::Parallel { min_ops } => 1 + min_ops as u64,
    };
    let lanes = match opts.lanes {
        Lanes::Auto => 0,
        Lanes::Always => 1,
        Lanes::Never => 2,
    };
    ((opts.chunk_ops as u64) << 32) ^ batch ^ (lanes << 62)
}

/// Instances per parallel task: about four tasks per thread of the
/// current pool.
fn blocks(n: usize) -> usize {
    (n / (rayon::current_num_threads() * 4)).clamp(1, 4096)
}

/// The work array: the tape's slots, then the gather area for host calls,
/// then the scratch a called bundle gets.
#[derive(Clone, Copy)]
struct Layout {
    /// First element of the gather area (after the slots).
    gather: usize,
    /// First element of the bundle scratch, and its length.
    scratch: usize,
    scratch_len: usize,
    total: usize,
}

/// One chunk as emitted: its machine code and the call descriptors the
/// code holds the addresses of.
struct Emitted {
    bytes: Vec<u8>,
    calls: Vec<host::CallDesc>,
    kernels: Vec<host::KernelDesc>,
    gathers: Vec<Box<[u32]>>,
}

// The mapping is immutable after `Mapping::new`, so calling the code from
// any thread is sound.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl NativeTape {
    pub fn compile(tape: &Tape) -> Result<NativeTape, JitError> {
        Self::compile_opts(tape, &Options::default(), &[])
    }

    /// Compile as `opts` says, keeping the slots `live` written to the work
    /// array at the end of the program, as the outputs are: what a consumer
    /// that reads a specialized tape's prolog guards from `work` after
    /// [`eval_prolog`](Self::eval_prolog) passes
    /// ([`rsdag::SpecializedTape::prolog_guards`]).
    pub fn compile_opts(tape: &Tape, opts: &Options, live: &[u32]) -> Result<NativeTape, JitError> {
        Self::compile_isa::<Arch>(tape, opts, live)
    }

    /// The tape as code over several instances at once, each in its lane
    /// of every register and every slot that many values side by side (see
    /// [`NativeBody`]'s batches), widest first: four lanes with AVX, and
    /// two. None for bodies that call bodies or run dense kernels, nor off
    /// x86-64.
    fn compile_lanes(tape: &Tape) -> Vec<NativeTape> {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::x86_64::X64;
            let opts = Options::default();
            let wide = crate::x86_64::features()
                .avx
                .then(|| Self::compile_isa::<X64<4>>(tape, &opts, &[]));
            let two = Some(Self::compile_isa::<X64<2>>(tape, &opts, &[]));
            wide.into_iter().chain(two).filter_map(Result::ok).collect()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = tape;
            Vec::new()
        }
    }

    fn compile_isa<A: Isa>(
        tape: &Tape,
        opts: &Options,
        live: &[u32],
    ) -> Result<NativeTape, JitError> {
        let chunk_ops = opts.chunk_ops;
        if !cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            return Err(JitError::Unsupported);
        }
        let (mut ops, split) = crate::ir::record(tape);
        let lanes = A::LANES;
        if lanes > 1
            && ops
                .iter()
                .any(|op| matches!(op, ROp::Call(_) | ROp::Kernel(_)))
        {
            return Err(JitError::Unsupported);
        }
        // Function bodies that are tapes become native bodies of their own,
        // emitted once per body and options and shared by every program that
        // calls it; a body with tapes of its own (per-binding variants) has
        // them compiled the same way.
        let o = *opts;
        let backend = rsdag::BodyBackend {
            compile: Some(Arc::new(move |t: &Tape, pure: &[bool], n_out: usize| {
                native_body(t, pure, n_out, &o).ok()
            })),
            key: options_key(opts),
            submit: opts
                .background
                .then(|| -> rsdag::Submit { Arc::new(crate::background::submit) }),
            variants: None,
        };
        let bundles: Result<Bundles, JitError> = tape
            .bundles()
            .iter()
            .map(|b| {
                let make = || -> Result<Arc<dyn ExternBundle>, JitError> {
                    if let Some(v) = b.with_backend(&backend) {
                        return Ok(v);
                    }
                    match b.body() {
                        Some(body) => native_body(body, b.pure_args(), b.n_outputs(), opts),
                        None => Ok(b.clone()),
                    }
                };
                // The form of the bundle the program runs: its code by the
                // options, and where its work goes.
                let key = options_key(opts) ^ (u64::from(opts.background) << 61);
                match b.backend_cache() {
                    Some(cache) => cache.get_or_try_insert(key, make),
                    None => make(),
                }
            })
            .collect();
        let bundles = bundles?;
        // The fold code tables, boxed so their addresses hold for the
        // tape's life; the ops carry the addresses.
        let mut tables: Vec<Box<[u32]>> = Vec::new();
        for op in ops.iter_mut() {
            if let ROp::Kernel(Kernel {
                codes: Some((codes, table)),
                ..
            }) = op
            {
                let b: Box<[u32]> = codes.clone().into_boxed_slice();
                *table = b.as_ptr() as usize;
                tables.push(b);
            }
        }
        // Inputs the code reads: tagged operands, dense runs, outputs.
        let mut n_inputs = 0usize;
        for op in &ops {
            let mut top = 0usize;
            op.for_each_operand(|k| {
                if let Some(i) = input_index(k) {
                    top = top.max(i as usize + 1);
                }
            });
            n_inputs = n_inputs.max(top);
            if let ROp::Kernel(k) = op {
                for (d, len) in &k.operands {
                    if let Dense::Inputs(i) = d {
                        n_inputs = n_inputs.max((i + len) as usize);
                    }
                }
            }
        }
        for &o in tape.outputs() {
            if let Some(i) = input_index(o) {
                n_inputs = n_inputs.max(i as usize + 1);
            }
        }
        // The inputs each phase reads, and an output read from the inputs
        // counts for the main phase.
        let mut reads: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for (k, op) in ops.iter().enumerate() {
            let r = &mut reads[usize::from(k >= split)];
            op.for_each_operand(|o| {
                if let Some(i) = input_index(o) {
                    r.push(i);
                }
            });
            if let ROp::Kernel(kn) = op {
                for (d, len) in &kn.operands {
                    if let Dense::Inputs(i) = d {
                        r.extend(*i..i + len);
                    }
                }
            }
        }
        for r in &mut reads {
            r.sort_unstable();
            r.dedup();
        }
        let mut phase_ops = [(0usize, 0usize); 2];
        for (k, op) in ops.iter().enumerate() {
            let p = &mut phase_ops[usize::from(k >= split)];
            p.0 += 1;
            p.1 += usize::from(op.host().is_some());
        }
        let gather_len = ops
            .iter()
            .map(|op| {
                if lanes == 1 {
                    op.gather_len()
                } else {
                    lane_gather_len(op, lanes)
                }
            })
            .max()
            .unwrap_or(0);
        let n_work = tape.n_slots();
        // When each value dies; the outputs are read after the program.
        let liveness = Liveness::new(&ops, n_work, tape.outputs().iter().chain(live).copied());
        // The scratch lent to a called bundle, or to a dense solve.
        let solves = ops.iter().map(|op| match op {
            ROp::Kernel(Kernel {
                kind: KernelKind::Solve { n, k, .. },
                ..
            }) => rsdag::semantics::solve_scratch_len(*n as usize, *k as usize),
            _ => 0,
        });
        let scratch_len = bundles
            .iter()
            .map(|b| b.work_len())
            .chain(solves)
            .max()
            .unwrap_or(0);
        let layout = Layout {
            gather: n_work * lanes,
            scratch: n_work * lanes + gather_len,
            scratch_len,
            total: (n_work * lanes + gather_len + scratch_len).max(1),
        };
        // Chunk the prolog and main phases separately so no chunk straddles
        // the split, nor a stage: its last call runs the stage's calls from
        // its chunk's descriptors.
        let (pro, main) = ops.split_at(split);
        let pro = chunked(pro, chunk_ops);
        let prolog_chunks = pro.len();
        let jobs: Vec<&[ROp]> = pro.into_iter().chain(chunked(main, chunk_ops)).collect();
        let starts: Vec<usize> = jobs
            .iter()
            .scan(0, |acc, ops| {
                let s = *acc;
                *acc += ops.len();
                Some(s)
            })
            .collect();
        let emitted: Vec<Emitted> = jobs
            .par_iter()
            .zip(&starts)
            .map(|(ops, &start)| emit_chunk::<A>(ops, start, layout, &liveness))
            .collect();
        // One mapping for all of them: a chunk is position independent (it
        // reaches host routines, descriptors and tables by absolute
        // address), so it runs from wherever it lands, 16-byte aligned.
        let size = emitted.iter().map(|e| e.bytes.len() + 15).sum();
        let mut bytes: Vec<u8> = Vec::with_capacity(size);
        let mut offsets = Vec::with_capacity(emitted.len());
        let mut descs = Vec::with_capacity(emitted.len());
        for e in emitted {
            bytes.resize(bytes.len().next_multiple_of(16), 0);
            offsets.push(bytes.len());
            bytes.extend_from_slice(&e.bytes);
            descs.push((e.calls, e.kernels));
            tables.extend(e.gathers);
        }
        let code = Mapping::new(&bytes)?;
        let chunks = offsets
            .iter()
            // SAFETY: each offset is the entry of a function emitted for the
            // `ChunkFn` convention, inside the mapping, which the tape keeps.
            .map(|&o| unsafe { std::mem::transmute::<*mut u8, ChunkFn>(code.ptr.add(o)) })
            .collect();
        Ok(NativeTape {
            _code: code,
            chunks,
            _descs: descs,
            prolog_chunks,
            bundles,
            _tables: tables,
            outputs: tape.outputs().to_vec(),
            layout,
            n_inputs,
            n_ops: tape.n_ops(),
            state_len: tape.state_len(),
            lanes,
            reads,
            phase_ops,
        })
    }

    fn padded<'a>(&self, inputs: &'a [f64], buf: &'a mut Vec<f64>) -> &'a [f64] {
        if inputs.len() < self.n_inputs {
            buf.clear();
            buf.extend_from_slice(inputs);
            buf.resize(self.n_inputs, f64::NAN);
            buf
        } else {
            inputs
        }
    }

    /// Run the chunks in `range`; `inputs` has at least `n_inputs` values
    /// and `work` the layout's length.
    fn run(&self, range: std::ops::Range<usize>, inputs: &[f64], work: &mut [f64]) {
        assert!(
            work.len() >= self.layout.total && inputs.len() >= self.n_inputs * self.lanes,
            "buffers not prepared by this tape"
        );
        let (wp, ip, bp) = (
            work.as_mut_ptr(),
            inputs.as_ptr(),
            &self.bundles as *const Bundles,
        );
        for c in &self.chunks[range] {
            // SAFETY: the buffers were checked above, the code and the
            // tables it addresses live as long as `self`.
            unsafe { c(wp, ip, bp) };
            host::resume_panic();
        }
    }

    fn collect(&self, inputs: &[f64], work: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.extend(self.outputs.iter().map(|&s| match input_index(s) {
            Some(i) => inputs.get(i as usize).copied().unwrap_or(f64::NAN),
            None => work[s as usize],
        }));
    }

    /// The bundles the calls call, by index, each as this backend runs it;
    /// mirrors [`Tape::bundles`].
    pub fn bundles(&self) -> &[Arc<dyn ExternBundle>] {
        &self.bundles
    }

    /// Evaluate the parameter-pure prolog into `work` (grown here to the
    /// tape's layout; nothing is cleared, every slot is written before it is
    /// read); mirrors [`Tape::eval_prolog`]. Pair with [`eval_main`](Self::eval_main).
    pub fn eval_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) {
        let mut buf = Vec::new();
        let ins = self.padded(inputs, &mut buf);
        if work.len() < self.layout.total {
            work.resize(self.layout.total, 0.0);
        }
        self.run(0..self.prolog_chunks, ins, work);
    }

    /// Evaluate the main phase over a buffer prepared by
    /// [`eval_prolog`](Self::eval_prolog); mirrors [`Tape::eval_main`].
    pub fn eval_main(&self, inputs: &[f64], work: &mut [f64], out: &mut Vec<f64>) {
        let mut buf = Vec::new();
        let ins = self.padded(inputs, &mut buf);
        self.run(self.prolog_chunks..self.chunks.len(), ins, work);
        self.collect(ins, work, out);
    }

    /// Evaluate the whole tape; mirrors [`Tape::eval`].
    pub fn eval(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        let mut buf = Vec::new();
        let ins = self.padded(inputs, &mut buf);
        if work.len() < self.layout.total {
            work.resize(self.layout.total, 0.0);
        }
        self.run(0..self.chunks.len(), ins, work);
        self.collect(ins, work, out);
    }

    /// Write the outputs from `work` (and `inputs`, for an output that is
    /// an input) into a slice the caller sized.
    fn write(&self, inputs: &[f64], work: &[f64], out: &mut [f64]) {
        assert_eq!(
            out.len(),
            self.outputs.len(),
            "output buffer of the wrong size"
        );
        for (dst, &s) in out.iter_mut().zip(&self.outputs) {
            *dst = match input_index(s) {
                Some(i) => inputs.get(i as usize).copied().unwrap_or(f64::NAN),
                None => work[s as usize],
            };
        }
    }

    /// Evaluate many instances in parallel: `inputs` holds `n` input
    /// vectors of `stride` values back to back (NaN-padded when shorter
    /// than the program's), `out` receives the `n` output vectors back to
    /// back. Instances share nothing, so blocks of them run on the current
    /// rayon pool (the caller's `install`, else the global one), each over
    /// a work buffer of its thread's; this is what a batch of identical
    /// devices, a parameter sweep or an ensemble amounts to. The serial
    /// form is [`Program::eval_many_into`](rsdag::Program::eval_many_into).
    pub fn eval_many(&self, inputs: &[f64], stride: usize, out: &mut Vec<f64>) {
        use rsdag::Program;
        let n = inputs.len().checked_div(stride).unwrap_or(0);
        let n_out = self.outputs.len();
        out.clear();
        out.resize(n * n_out, 0.0);
        if n == 0 || n_out == 0 {
            return;
        }
        let block = blocks(n);
        let n_in = self.n_inputs.max(stride);
        out.par_chunks_mut(block * n_out)
            .zip(inputs.par_chunks(block * stride))
            .for_each(|(dst, ins)| {
                rsdag::scratch::with::<Vec<f64>, _>(|work| {
                    work.resize(self.layout.total, 0.0);
                    if n_in == stride {
                        self.eval_many_into(ins, stride, work, dst);
                        return;
                    }
                    let mut padded = vec![f64::NAN; ins.len() / stride * n_in];
                    for (p, i) in padded.chunks_exact_mut(n_in).zip(ins.chunks_exact(stride)) {
                        p[..stride].copy_from_slice(i);
                    }
                    self.eval_many_into(&padded, n_in, work, dst);
                });
            });
    }
}

// --- the emitter --------------------------------------------------------------

/// The value cache over an architecture's instruction layer: which slot
/// each cache register holds, whether memory has it yet, and which
/// registers the current op still needs. Every operand `get` pins its
/// register and every temporary is pinned by its maker, so an op can never
/// evict what it is about to use; the pins clear when the op is done.
struct Emitter<'a, I: Isa> {
    isa: I,
    layout: Layout,
    /// When each value dies.
    live: &'a Liveness,
    /// Global index of the op being emitted.
    pos: u32,
    /// Global index of the next op that calls a host routine (`u32::MAX`
    /// when the chunk has none left).
    next_call: u32,
    /// Slot held by each cache register (by cache index).
    held: Vec<Option<u32>>,
    /// Whether the register's value is newer than the slot in memory.
    dirty: Vec<bool>,
    /// The last op reading the register's value (see [`Liveness::death`]).
    death: Vec<u32>,
    /// Cache index holding each slot.
    at: FxHashMap<u32, usize>,
    /// Cache index of each register number.
    index: [u8; 32],
    /// Round-robin victim pointers of the callee-saved and caller-saved pools.
    next: [usize; 2],
    pinned: Vec<bool>,
    /// The chunk's call and kernel descriptors, allocated for all of them before the
    /// first is emitted (their addresses go into the code).
    descs: Vec<host::CallDesc>,
    kernels: Vec<host::KernelDesc>,
    /// The operand tables of the calls whose arguments the host gathers.
    gathers: Vec<Box<[u32]>>,
}

/// Calls with this many arguments over all their instances and more have
/// the host gather them from a table; narrower ones gather in the code,
/// whose size would otherwise grow with every argument.
const GATHER_TABLE: usize = 64;

impl<'a, I: Isa> Emitter<'a, I> {
    fn new(layout: Layout, live: &'a Liveness, hot: &[*const ()]) -> Emitter<'a, I> {
        let mut index = [0u8; 32];
        for (i, &r) in I::CACHE.iter().enumerate() {
            index[r as usize] = i as u8;
        }
        Emitter {
            isa: I::new(hot),
            layout,
            live,
            pos: 0,
            next_call: u32::MAX,
            held: vec![None; I::CACHE.len()],
            dirty: vec![false; I::CACHE.len()],
            death: vec![0; I::CACHE.len()],
            at: Default::default(),
            index,
            next: [0, 0],
            pinned: vec![false; I::CACHE.len()],
            descs: Vec::new(),
            kernels: Vec::new(),
            gathers: Vec::new(),
        }
    }

    /// Forget what cache index `i` holds, writing it back first if memory
    /// does not have it and some later op reads it. The current op counts
    /// as later: an operand it has not fetched yet may be what is evicted
    /// to make room for another.
    fn drop_index(&mut self, i: usize) {
        if let Some(s) = self.held[i].take() {
            if self.dirty[i] && self.death[i] >= self.pos {
                self.isa.store(I::CACHE[i], Base::Work, Self::slot_off(s));
            }
            self.dirty[i] = false;
            self.at.remove(&s);
        }
    }
    /// Write back everything still owed to memory (the chunk's end).
    fn flush(&mut self) {
        for i in 0..I::CACHE.len() {
            self.drop_index(i);
        }
    }
    /// A register to write a temporary into: the next unpinned one round
    /// robin, evicting whatever it held; pinned for the rest of the op.
    fn fresh(&mut self) -> u8 {
        self.fresh_in(I::SAVED..I::CACHE.len())
    }
    /// A register for the value op `pos` writes to `slot`.
    fn fresh_for(&mut self, slot: u32) -> u8 {
        self.fresh_until(self.live.death(slot, self.pos, true))
    }
    /// A register for a value that dies at `death`: preferably callee-saved
    /// when a host call comes before, so the call does not cost it a store
    /// and a reload; preferably caller-saved otherwise.
    fn fresh_until(&mut self, death: u32) -> u8 {
        let keep = I::SAVED > 0 && self.next_call <= death;
        if keep {
            self.fresh_in(0..I::SAVED)
        } else {
            self.fresh_in(I::SAVED..I::CACHE.len())
        }
    }
    /// A register from the preferred pool: first one that is empty or holds
    /// a dead value, in either pool; else the preferred pool's round-robin
    /// victim; else any unpinned register.
    fn fresh_in(&mut self, pool: std::ops::Range<usize>) -> u8 {
        let n = I::CACHE.len();
        let (lo, len) = (pool.start, pool.len().max(1));
        let other = if lo == 0 { I::SAVED..n } else { 0..I::SAVED };
        let take = |e: &mut Self, i: usize| {
            e.drop_index(i);
            e.pinned[i] = true;
            I::CACHE[i]
        };
        for i in pool.clone().chain(other) {
            let dead = match self.held[i] {
                None => true,
                Some(s) => match input_index(s) {
                    Some(_) => true, // an input reloads from the inputs
                    None => self.death[i] < self.pos,
                },
            };
            if !self.pinned[i] && dead {
                return take(self, i);
            }
        }
        let cursor = &mut self.next[usize::from(lo != 0 || I::SAVED == 0)];
        for _ in 0..len {
            let i = lo + *cursor % len;
            *cursor = (*cursor + 1) % len;
            if !self.pinned[i] {
                return take(self, i);
            }
        }
        for i in 0..n {
            if !self.pinned[i] {
                return take(self, i);
            }
        }
        unreachable!("an op pins fewer than {n} registers");
    }
    fn pin(&mut self, r: u8) {
        self.pinned[self.index[r as usize] as usize] = true;
    }
    /// Unpin every register except `keep`.
    fn release_except(&mut self, keep: &[u8]) {
        self.pinned.iter_mut().for_each(|p| *p = false);
        keep.iter().for_each(|&r| self.pin(r));
    }
    /// The byte offset of slot (or input) `s`: its `LANES` values side by side.
    fn slot_off(s: u32) -> usize {
        s as usize * 8 * I::LANES
    }
    /// The register holding `slot`, loading it if the cache does not have it.
    fn get(&mut self, slot: u32) -> u8 {
        if let Some(&i) = self.at.get(&slot) {
            self.pinned[i] = true;
            return I::CACHE[i];
        }
        match input_index(slot) {
            Some(k) => {
                // An input: read in place, cached, never written back.
                let r = self.fresh();
                self.isa.load(r, Base::Inputs, Self::slot_off(k));
                self.bind(r, slot, 0);
                r
            }
            None => {
                let death = self.live.death(slot, self.pos, false);
                let r = self.fresh_until(death);
                self.isa.load(r, Base::Work, Self::slot_off(slot));
                self.bind(r, slot, death);
                r
            }
        }
    }
    /// A kernel wrote the slots `dst .. dst+n`: whatever the cache held
    /// for them is stale.
    fn invalidate(&mut self, dst: u32, n: u32) {
        for s in dst..dst + n {
            if let Some(i) = self.at.remove(&s) {
                self.held[i] = None;
                self.dirty[i] = false;
            }
        }
    }
    fn bind(&mut self, r: u8, slot: u32, death: u32) {
        // A slot rebound to a new value: the old one is dead by the tape's
        // construction, so it is dropped without a write-back.
        if let Some(i) = self.at.remove(&slot) {
            self.held[i] = None;
            self.dirty[i] = false;
        }
        let i = self.index[r as usize] as usize;
        self.drop_index(i);
        self.held[i] = Some(slot);
        self.death[i] = death;
        self.at.insert(slot, i);
    }
    /// `r` is the value of `slot` now; memory will get it when it must.
    fn put(&mut self, slot: u32, r: u8) {
        let death = self.live.death(slot, self.pos, true);
        self.bind(r, slot, death);
        self.dirty[self.index[r as usize] as usize] = true;
    }
    fn fconst(&mut self, v: f64) -> u8 {
        let r = self.fresh();
        self.isa.fconst(r, v);
        r
    }
    /// A host call; the caller-saved part of the cache is written back
    /// where owed before, and gone afterwards.
    fn call(&mut self, addr: *const (), args: &[Arg]) {
        for i in I::SAVED..I::CACHE.len() {
            self.drop_index(i);
        }
        self.isa.call(addr, args);
    }
    /// A host call whose result is the value of `dst`. With lanes, one call
    /// per lane: the float arguments' lanes go to the gather area, each
    /// call takes its lane's values, and the results come back together.
    fn call_into(&mut self, dst: u32, addr: *const (), args: &[Arg]) {
        if I::LANES == 1 {
            self.call(addr, args);
            let r = self.fresh_for(dst);
            self.isa.mov(r, I::RESULT);
            self.put(dst, r);
            return;
        }
        let l = I::LANES;
        let base = self.layout.gather * 8;
        let mut nf = 0;
        for a in args {
            if let Arg::F(r) = *a {
                self.isa.store(r, Base::Work, base + nf * l * 8);
                nf += 1;
            }
        }
        let res = base + nf * l * 8;
        for lane in 0..l {
            self.release_except(&[]);
            let mut j = 0;
            let lane_args: Vec<Arg> = args
                .iter()
                .map(|a| match *a {
                    Arg::F(_) => {
                        let r = self.fresh();
                        self.isa.load_lane(r, Base::Work, base + (j * l + lane) * 8);
                        j += 1;
                        Arg::F(r)
                    }
                    Arg::I(i) => Arg::I(i),
                })
                .collect();
            self.call(addr, &lane_args);
            self.isa.store_lane(I::RESULT, Base::Work, res + lane * 8);
        }
        self.release_except(&[]);
        let r = self.fresh_for(dst);
        self.isa.load(r, Base::Work, res);
        self.put(dst, r);
    }
    /// Copy `slots` into the gather area; its byte offset.
    fn gather(&mut self, slots: &[u32]) -> usize {
        self.gather_at(slots, None, 0)
    }
    /// Copy `slots` into the gather area from element `at` on; the byte
    /// offset of the copy.
    fn gather_at(&mut self, slots: &[u32], places: Option<&[u32]>, at: usize) -> usize {
        debug_assert_eq!(I::LANES, 1, "lane code gathers lane by lane");
        let base = (self.layout.gather + at) * 8;
        for (k, &s) in slots.iter().enumerate() {
            let r = self.get(s);
            let to = places.map_or(k, |p| p[k] as usize);
            self.isa.store(r, Base::Work, base + to * 8);
            self.release_except(&[]);
        }
        base
    }

    /// The slots among `slots` whose newest value is only in a register,
    /// written back, so a host routine reading memory finds them.
    fn publish(&mut self, slots: &[u32]) {
        for s in slots {
            if let Some(&i) = self.at.get(s) {
                if self.dirty[i] {
                    self.isa.store(I::CACHE[i], Base::Work, Self::slot_off(*s));
                    self.dirty[i] = false;
                }
            }
        }
    }

    /// A dense operand's address: in place (inputs, a consecutive run of
    /// work slots), or gathered into the gather area from `*at`, so the
    /// area holds exactly the slots a kernel gathers
    /// ([`ROp::gather_len`]).
    fn dense_arg(&mut self, d: &Dense, at: &mut usize) -> IArg {
        let mut arg = |this: &mut Self, d: &Dense| match d {
            Dense::Inputs(k) => IArg::InputAddr(*k as usize * 8),
            Dense::Run(s, len) => {
                // Read in place: whatever of the run the register cache
                // still owes to memory is written back first.
                for slot in *s..s + len {
                    if let Some(&i) = this.at.get(&slot) {
                        this.drop_index(i);
                    }
                }
                IArg::WorkAddr(*s as usize * 8)
            }
            Dense::Slots(s) => {
                let p = IArg::WorkAddr(this.gather_at(s, None, *at));
                *at += s.len();
                p
            }
        };
        arg(self, d)
    }

    fn op(&mut self, op: &ROp) {
        self.op_inner(op);
        self.release_except(&[]);
    }

    fn op_inner(&mut self, op: &ROp) {
        match *op {
            ROp::Const(dst, v) => {
                let r = self.fconst(v);
                self.put(dst, r);
            }
            ROp::Add(dst, a, b) => self.bin2(Arith::Add, dst, a, b),
            ROp::Sub(dst, a, b) => self.bin2(Arith::Sub, dst, a, b),
            ROp::Mul(dst, a, b) => self.bin2(Arith::Mul, dst, a, b),
            ROp::MulAdd(dst, a, b, c) => {
                // Two roundings, like the interpreter.
                let x = self.get(a);
                let m = self.fresh();
                self.arith_rm(Arith::Mul, m, x, b);
                let r = self.fresh_for(dst);
                self.arith_rm(Arith::Add, r, m, c);
                self.put(dst, r);
            }
            ROp::Neg(dst, a) => {
                let x = self.get(a);
                let r = self.fresh_for(dst);
                self.isa.neg(r, x);
                self.put(dst, r);
            }
            ROp::Powi(dst, a, n) => match n {
                -1 => {
                    let x = self.get(a);
                    let one = self.fconst(1.0);
                    let r = self.fresh_for(dst);
                    self.isa.arith(Arith::Div, r, one, x);
                    self.put(dst, r);
                }
                2 => self.bin2(Arith::Mul, dst, a, a),
                n if crate::ir::powi_inline(n) => {
                    // `semantics::powi_t`'s binary exponentiation, product
                    // for product, so the result is the reference's bits.
                    let x = self.get(a);
                    let (mut base, mut acc, mut e) = (x, None::<u8>, n.unsigned_abs());
                    while e > 0 {
                        if e & 1 == 1 {
                            acc = Some(match acc {
                                None => base,
                                Some(p) => {
                                    let r = self.fresh();
                                    self.isa.arith(Arith::Mul, r, p, base);
                                    r
                                }
                            });
                        }
                        e >>= 1;
                        if e > 0 {
                            let r = self.fresh();
                            self.isa.arith(Arith::Mul, r, base, base);
                            base = r;
                        }
                        let keep: Vec<u8> = [Some(base), acc].into_iter().flatten().collect();
                        self.release_except(&keep);
                    }
                    let p = acc.expect("a nonzero exponent");
                    let r = self.fresh_for(dst);
                    if n < 0 {
                        let one = self.fconst(1.0);
                        self.isa.arith(Arith::Div, r, one, p);
                    } else {
                        self.isa.mov(r, p);
                    }
                    self.put(dst, r);
                }
                _ => {
                    let x = self.get(a);
                    self.call_into(
                        dst,
                        host::h_powi as *const (),
                        &[Arg::F(x), Arg::I(IArg::Imm(n as i64 as u64))],
                    );
                }
            },
            ROp::Unary(dst, uop, a) => self.unary(dst, uop, a),
            ROp::Binary(dst, bop, a, b) => {
                let (x, y) = (self.get(a), self.get(b));
                let code = Arg::I(IArg::Imm(BinOp::code(bop) as u64));
                self.call_into(
                    dst,
                    host::h_binary as *const (),
                    &[code, Arg::F(x), Arg::F(y)],
                );
            }
            ROp::Cmp(dst, cop, a, b) => {
                let (x, y) = (self.get(a), self.get(b));
                let one = self.fconst(1.0);
                let zero = self.fconst(0.0);
                let r = self.fresh_for(dst);
                self.isa.cmp_select(cop, x, y, one, zero, r);
                self.put(dst, r);
            }
            ROp::Select(dst, c, t, e) => {
                let (cv, tv, ev) = (self.get(c), self.get(t), self.get(e));
                let r = self.fresh_for(dst);
                self.isa.select_nz(cv, tv, ev, r);
                self.put(dst, r);
            }
            ROp::Reduce(dst, rop, ref args) => match rop {
                ReduceOp::Sum => {
                    let r = self.fold(Arith::Add, 0.0, args, None);
                    self.put(dst, r);
                }
                ReduceOp::Product => {
                    let r = self.fold(Arith::Mul, 1.0, args, None);
                    self.put(dst, r);
                }
                ReduceOp::Min | ReduceOp::Max
                    if I::MINMAX && !args.is_empty() && args.len() <= INLINE_MINMAX_MAX =>
                {
                    // The reference's left fold, one instruction per term.
                    let first = self.get(args[0]);
                    let mut acc = self.fresh();
                    self.isa.mov(acc, first);
                    for &k in &args[1..] {
                        let t = self.get(k);
                        let r = self.fresh();
                        self.isa.minmax(rop, r, acc, t);
                        acc = r;
                        self.release_except(&[acc]);
                    }
                    self.put(dst, acc);
                }
                ReduceOp::Min | ReduceOp::Max if I::LANES > 1 => {
                    // Lane by lane: each term's lanes to the gather area, one
                    // lane's terms side by side for the routine, its result
                    // into its lane.
                    let (l, m) = (I::LANES, args.len());
                    let base = self.layout.gather * 8;
                    for (j, &a) in args.iter().enumerate() {
                        let r = self.get(a);
                        self.isa.store(r, Base::Work, base + j * l * 8);
                        self.release_except(&[]);
                    }
                    let (row, res) = (base + m * l * 8, base + (m * l + m) * 8);
                    for lane in 0..l {
                        for j in 0..m {
                            let t = self.fresh();
                            self.isa.load_lane(t, Base::Work, base + (j * l + lane) * 8);
                            self.isa.store_lane(t, Base::Work, row + j * 8);
                            self.release_except(&[]);
                        }
                        let call = [
                            Arg::I(IArg::Imm(host::reduce_code(rop))),
                            Arg::I(IArg::WorkAddr(row)),
                            Arg::I(IArg::Imm(m as u64)),
                        ];
                        self.call(host::h_reduce as *const (), &call);
                        self.isa.store_lane(I::RESULT, Base::Work, res + lane * 8);
                    }
                    let r = self.fresh_for(dst);
                    self.isa.load(r, Base::Work, res);
                    self.put(dst, r);
                }
                ReduceOp::Min | ReduceOp::Max => {
                    let at = self.gather(args);
                    let args = [
                        Arg::I(IArg::Imm(host::reduce_code(rop))),
                        Arg::I(IArg::WorkAddr(at)),
                        Arg::I(IArg::Imm(args.len() as u64)),
                    ];
                    self.call_into(dst, host::h_reduce as *const (), &args);
                }
            },
            ROp::Dot(dst, ref a, ref b) => {
                let r = self.fold(Arith::Add, 0.0, a, Some(b));
                self.put(dst, r);
            }
            ROp::Call(ref c) => {
                // A call of a stage gathers apart from the others of it; the
                // stage's last one hands them all to `h_stage`.
                let table = if c.args.len() < GATHER_TABLE {
                    // where each operand goes among the groups' arguments
                    let places: Option<Vec<u32>> = c.positions.as_ref().map(|p| {
                        (0..c.args.len())
                            .map(|k| ((k / p.len()) * c.n_args as usize) as u32 + p[k % p.len()])
                            .collect()
                    });
                    self.gather_at(&c.args, places.as_deref(), c.gather_at);
                    None
                } else {
                    self.publish(&c.args);
                    Some(())
                };
                let mut keep = |v: &[u32]| {
                    let t: Box<[u32]> = v.into();
                    let p = t.as_ptr() as u64;
                    self.gathers.push(t);
                    p
                };
                let (table, places) = match table {
                    None => (0, 0),
                    Some(()) => (keep(&c.args), c.positions.as_deref().map_or(0, &mut keep)),
                };
                let n_inputs = c
                    .args
                    .iter()
                    .filter_map(|&s| input_index(s))
                    .max()
                    .map_or(0, |i| i as u64 + 1);
                let d = self.descs.len();
                assert!(
                    d < self.descs.capacity(),
                    "call descriptors counted before emission"
                );
                let desc = host::CallDesc {
                    bundle: c.bundle as u64,
                    kind: c.kind as u64,
                    n_groups: c.n_groups as u64,
                    n_args: c.n_args as u64,
                    n_in: (c.args.len() / c.n_groups as usize) as u64,
                    n_out: c.n_out as u64,
                    args: c.gather_at as u64,
                    out: c.dst as u64,
                    state: c.state as u64,
                    gather: self.layout.gather as u64,
                    scratch: self.layout.scratch as u64,
                    scratch_len: self.layout.scratch_len as u64,
                    ops: c.ops,
                    table,
                    places,
                    n_inputs,
                };
                // The table was sized up front: pushing never moves it, so
                // the address baked into the code stays valid.
                self.descs.push(desc);
                match c.stage {
                    StageRole::Alone => {
                        let ptr = &self.descs[d] as *const host::CallDesc as u64;
                        let args = [
                            Arg::I(IArg::Bundles),
                            Arg::I(IArg::Imm(ptr)),
                            Arg::I(IArg::WorkAddr(0)),
                            Arg::I(IArg::InputAddr(0)),
                        ];
                        self.call(host::h_call as *const (), &args);
                        self.invalidate(c.dst, c.n_groups * c.n_out);
                    }
                    StageRole::Deferred => {}
                    StageRole::Last(n) => {
                        let first = d + 1 - n as usize;
                        let ptr = &self.descs[first] as *const host::CallDesc as u64;
                        let args = [
                            Arg::I(IArg::Bundles),
                            Arg::I(IArg::Imm(ptr)),
                            Arg::I(IArg::Imm(n as u64)),
                            Arg::I(IArg::WorkAddr(0)),
                            Arg::I(IArg::InputAddr(0)),
                        ];
                        self.call(host::h_stage as *const (), &args);
                        let written: Vec<(u32, u32)> = self.descs[first..]
                            .iter()
                            .map(|e| (e.out as u32, (e.n_groups * e.n_out) as u32))
                            .collect();
                        for (dst, len) in written {
                            self.invalidate(dst, len);
                        }
                    }
                }
            }
            ROp::Kernel(ref kn) => {
                let mut at = 0usize;
                let mut operands = [host::Place::NONE; 3];
                for (p, (d, _)) in operands.iter_mut().zip(&kn.operands) {
                    *p = match self.dense_arg(d, &mut at) {
                        IArg::InputAddr(off) => host::Place {
                            base: host::Place::INPUTS,
                            off: off as u64,
                        },
                        IArg::WorkAddr(off) => host::Place {
                            base: host::Place::WORK,
                            off: off as u64,
                        },
                        _ => unreachable!("a dense operand is in the work array or the inputs"),
                    };
                }
                let (kind, m, k, n) = match kn.kind {
                    KernelKind::Gemv { m, n } => (0, m, 0, n),
                    KernelKind::Gemm { m, k, n } => (1, m, k, n),
                    KernelKind::Solve { n, k, count } => (2, count, k, n),
                };
                let desc = host::KernelDesc {
                    kind,
                    m: m as u64,
                    k: k as u64,
                    n: n as u64,
                    operands,
                    codes: kn.codes.as_ref().map_or(0, |c| c.1 as u64),
                    out: kn.dst as u64 * 8,
                    scratch: self.layout.scratch as u64 * 8,
                    scratch_len: self.layout.scratch_len as u64,
                };
                let d = self.kernels.len();
                assert!(
                    d < self.kernels.capacity(),
                    "kernel descriptors counted before emission"
                );
                // Sized up front: pushing never moves the table, so the
                // address baked into the code stays valid.
                self.kernels.push(desc);
                let ptr = &self.kernels[d] as *const host::KernelDesc as u64;
                let args = [
                    Arg::I(IArg::WorkAddr(0)),
                    Arg::I(IArg::InputAddr(0)),
                    Arg::I(IArg::Imm(ptr)),
                ];
                self.call(host::h_kernel as *const (), &args);
                self.invalidate(kn.dst, kn.width());
            }
        }
    }

    fn bin2(&mut self, op: Arith, dst: u32, a: u32, b: u32) {
        let x = self.get(a);
        let r = self.fresh_for(dst);
        self.arith_rm(op, r, x, b);
        self.put(dst, r);
    }

    /// `d = a op slot`: the second operand from memory when the ISA reads
    /// memory operands and the cache does not hold a value this op reads
    /// last (a load would take a register and an instruction for nothing),
    /// else from its register.
    fn arith_rm(&mut self, op: Arith, d: u8, a: u8, slot: u32) {
        if I::MEM_OPERANDS
            && input_index(slot).is_none()
            && !self.at.contains_key(&slot)
            && self.live.death(slot, self.pos, false) == self.pos
        {
            self.isa
                .arith_mem(op, d, a, Base::Work, Self::slot_off(slot));
            return;
        }
        let b = self.get(slot);
        self.isa.arith(op, d, a, b);
    }

    fn unary(&mut self, dst: u32, uop: UnaryOp, a: u32) {
        let x = self.get(a);
        // The result register only for an op done inline: a host call
        // returns its result in its own.
        let inline = match uop {
            UnaryOp::Sqrt => {
                // x <= 0 ? 0 : sqrt(x), the reference's guard (NaN stays NaN).
                let r = self.fresh_for(dst);
                let zero = self.fconst(0.0);
                let s = self.fresh();
                self.isa.sqrt(s, x);
                self.isa.cmp_select(CmpOp::Le, x, zero, zero, s, r);
                Some(r)
            }
            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Trunc => {
                let mode = match uop {
                    UnaryOp::Floor => Round::Floor,
                    UnaryOp::Ceil => Round::Ceil,
                    _ => Round::Trunc,
                };
                let r = self.fresh_for(dst);
                self.isa.round(mode, r, x).then_some(r)
            }
            UnaryOp::Abs => {
                let r = self.fresh_for(dst);
                self.isa.abs(r, x);
                Some(r)
            }
            UnaryOp::Sign => {
                let r = self.fresh_for(dst);
                let zero = self.fconst(0.0);
                let one = self.fconst(1.0);
                let minus = self.fconst(-1.0);
                let t = self.fresh();
                self.isa.cmp_select(CmpOp::Lt, x, zero, minus, x, t);
                self.isa.cmp_select(CmpOp::Gt, x, zero, one, t, r);
                Some(r)
            }
            _ => None,
        };
        match inline {
            Some(r) => self.put(dst, r),
            None => {
                let (addr, code) = host::unary_addr(uop);
                // The coded routine takes the op first: `h_unary_ext(op, x)`.
                let args: Vec<Arg> = code
                    .into_iter()
                    .map(|c| Arg::I(IArg::Imm(c as u64)))
                    .chain([Arg::F(x)])
                    .collect();
                self.call_into(dst, addr, &args);
            }
        }
    }

    /// A reduction (or, with `b`, a dot) in the reference order: four
    /// accumulators, merged as `(a0 + a1) + (a2 + a3)`, then the tail.
    /// Accumulators stay pinned across the terms; a term's registers are
    /// released once it is folded in, so a long list needs seven registers,
    /// not one per operand. Under four terms the merged accumulators are
    /// the identity itself (`+0` or `1`, exactly), so the fold starts there.
    fn fold(&mut self, op: Arith, ident: f64, a: &[u32], b: Option<&[u32]>) -> u8 {
        let n = a.len();
        let ch = n / 4;
        let mut s = if ch == 0 {
            self.fconst(ident)
        } else {
            let acc: [u8; 4] = std::array::from_fn(|_| self.fconst(ident));
            for c in 0..ch {
                for (k, &ak) in acc.iter().enumerate() {
                    self.term(op, ak, ak, a, b, 4 * c + k);
                    self.release_except(&acc);
                }
            }
            let l = self.fresh();
            self.isa.arith(op, l, acc[0], acc[1]);
            let r = self.fresh();
            self.isa.arith(op, r, acc[2], acc[3]);
            let s = self.fresh();
            self.isa.arith(op, s, l, r);
            s
        };
        for k in ch * 4..n {
            let s2 = self.fresh();
            self.term(op, s2, s, a, b, k);
            s = s2;
            self.release_except(&[s]);
        }
        s
    }

    /// `d = acc op term k` of a fold: the term is the operand, or for a dot
    /// the product.
    fn term(&mut self, op: Arith, d: u8, acc: u8, a: &[u32], b: Option<&[u32]>, k: usize) {
        match b {
            None => self.arith_rm(op, d, acc, a[k]),
            Some(bb) => {
                let x = self.get(a[k]);
                let p = self.fresh();
                self.arith_rm(Arith::Mul, p, x, bb[k]);
                self.isa.arith(op, d, acc, p);
            }
        }
    }
}

/// The host routines a chunk calls, most frequent first.
fn hot_routines(ops: &[ROp]) -> Vec<*const ()> {
    let mut count: Vec<(*const (), usize)> = Vec::new();
    for op in ops {
        if let Some(h) = op.host() {
            match count.iter_mut().find(|(a, _)| *a == h) {
                Some(c) => c.1 += 1,
                None => count.push((h, 1)),
            }
        }
    }
    // A routine called once is not worth a register load in the prologue.
    count.retain(|&(_, n)| n >= 2);
    count.sort_by_key(|a| std::cmp::Reverse(a.1));
    count.into_iter().map(|(a, _)| a).collect()
}

fn emit_chunk<A: Isa>(ops: &[ROp], start: usize, layout: Layout, live: &Liveness) -> Emitted {
    let hot = hot_routines(ops);
    // For each op, the next op at or after it that calls out.
    let mut next_call = vec![u32::MAX; ops.len() + 1];
    for k in (0..ops.len()).rev() {
        next_call[k] = if ops[k].host().is_some() {
            (start + k) as u32
        } else {
            next_call[k + 1]
        };
    }
    let mut e: Emitter<A> = Emitter::new(layout, live, &hot);
    let count = |f: fn(&ROp) -> bool| ops.iter().filter(|op| f(op)).count();
    e.descs = Vec::with_capacity(count(|op| matches!(op, ROp::Call(_))));
    e.kernels = Vec::with_capacity(count(|op| matches!(op, ROp::Kernel(_))));
    e.isa.prologue();
    for (k, op) in ops.iter().enumerate() {
        e.pos = (start + k) as u32;
        e.next_call = next_call[k];
        e.op(op);
    }
    // Past the chunk: what a later chunk reads is written back.
    e.pos = (start + ops.len()) as u32;
    e.flush();
    e.isa.epilogue();
    let (calls, kernels) = (std::mem::take(&mut e.descs), std::mem::take(&mut e.kernels));
    let gathers = std::mem::take(&mut e.gathers);
    Emitted {
        bytes: e.isa.finish(),
        calls,
        kernels,
        gathers,
    }
}

// --- executable memory -----------------------------------------------------------

struct Mapping {
    ptr: *mut u8,
    /// `VirtualFree` releases by base address alone; `munmap` wants the length.
    #[cfg_attr(windows, allow(dead_code))]
    len: usize,
}

#[cfg(unix)]
impl Mapping {
    /// The code is never writable and executable at once: mapped writable,
    /// filled, then switched to read and execute (macOS toggles its JIT
    /// write protection per thread instead, `MAP_JIT` requires it).
    fn new(bytes: &[u8]) -> Result<Mapping, JitError> {
        let len = bytes.len().max(1);
        unsafe {
            #[cfg(target_os = "macos")]
            let (flags, prot) = (
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            );
            #[cfg(not(target_os = "macos"))]
            let (flags, prot) = (
                libc::MAP_PRIVATE | libc::MAP_ANON,
                libc::PROT_READ | libc::PROT_WRITE,
            );
            let ptr = libc::mmap(std::ptr::null_mut(), len, prot, flags, -1, 0);
            if ptr == libc::MAP_FAILED {
                return Err(JitError::Codegen("mmap of executable memory failed".into()));
            }
            let ptr = ptr as *mut u8;
            let map = Mapping { ptr, len };
            #[cfg(target_os = "macos")]
            pthread_jit_write_protect_np(0);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            #[cfg(target_os = "macos")]
            {
                pthread_jit_write_protect_np(1);
                sys_icache_invalidate(ptr as *mut libc::c_void, bytes.len());
            }
            #[cfg(not(target_os = "macos"))]
            {
                let rx = libc::PROT_READ | libc::PROT_EXEC;
                if libc::mprotect(ptr as *mut libc::c_void, len, rx) != 0 {
                    return Err(JitError::Codegen(
                        "mprotect of the code to read and execute failed".into(),
                    ));
                }
                __clear_cache(
                    ptr as *mut libc::c_char,
                    ptr.add(bytes.len()) as *mut libc::c_char,
                );
            }
            Ok(map)
        }
    }
}

#[cfg(unix)]
impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_jit_write_protect_np(enabled: libc::c_int);
    fn sys_icache_invalidate(start: *mut libc::c_void, len: libc::size_t);
}
#[cfg(all(unix, not(target_os = "macos")))]
extern "C" {
    fn __clear_cache(start: *mut libc::c_char, end: *mut libc::c_char);
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn VirtualAlloc(addr: *mut u8, size: usize, kind: u32, protect: u32) -> *mut u8;
    fn VirtualFree(addr: *mut u8, size: usize, kind: u32) -> i32;
    fn VirtualProtect(addr: *mut u8, size: usize, protect: u32, old: *mut u32) -> i32;
    fn GetCurrentProcess() -> isize;
    fn FlushInstructionCache(process: isize, addr: *const u8, size: usize) -> i32;
}

#[cfg(windows)]
impl Mapping {
    /// Committed writable, filled, then switched to read and execute.
    fn new(bytes: &[u8]) -> Result<Mapping, JitError> {
        const MEM_COMMIT_RESERVE: u32 = 0x1000 | 0x2000;
        const PAGE_READWRITE: u32 = 0x04;
        const PAGE_EXECUTE_READ: u32 = 0x20;
        let len = bytes.len().max(1);
        unsafe {
            let ptr = VirtualAlloc(
                std::ptr::null_mut(),
                len,
                MEM_COMMIT_RESERVE,
                PAGE_READWRITE,
            );
            if ptr.is_null() {
                return Err(JitError::Codegen(
                    "VirtualAlloc of executable memory failed".into(),
                ));
            }
            let map = Mapping { ptr, len };
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            let mut old = 0u32;
            if VirtualProtect(ptr, len, PAGE_EXECUTE_READ, &mut old) == 0 {
                return Err(JitError::Codegen(
                    "VirtualProtect of the code to read and execute failed".into(),
                ));
            }
            FlushInstructionCache(GetCurrentProcess(), ptr, bytes.len());
            Ok(map)
        }
    }
}

#[cfg(windows)]
impl Drop for Mapping {
    fn drop(&mut self) {
        const MEM_RELEASE: u32 = 0x8000;
        unsafe {
            VirtualFree(self.ptr, 0, MEM_RELEASE);
        }
    }
}

impl rsdag::Program for NativeTape {
    fn n_inputs(&self) -> usize {
        self.n_inputs
    }
    fn work_len(&self) -> usize {
        self.layout.total
    }
    fn out_len(&self) -> usize {
        self.outputs.len()
    }
    fn state_len(&self) -> usize {
        self.state_len
    }
    fn eval_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]) {
        self.run(0..self.chunks.len(), inputs, work);
        self.write(inputs, work, out);
    }
    fn eval_prolog_into(&self, inputs: &[f64], work: &mut [f64]) {
        self.run(0..self.prolog_chunks, inputs, work);
    }
    fn eval_main_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]) {
        self.run(self.prolog_chunks..self.chunks.len(), inputs, work);
        self.write(inputs, work, out);
    }
}
