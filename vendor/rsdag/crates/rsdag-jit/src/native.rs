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
//! last-use table knows. A value that dies inside the chunk never touches
//! memory; nothing is ever spilled anywhere but where the interpreter keeps
//! it anyway. Compile time is linear in the op count, around 40 ns per op;
//! chunks are emitted in parallel.
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
use rsdag::{ExternBundle, Tape};
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::host::{self, Bundles};
use crate::ir::{Dense, ROp, Recorder};
use crate::isa::{Arg, Arith, Base, IArg, Isa, Round};
use crate::{JitError, CHUNK_OPS};

#[cfg(target_arch = "aarch64")]
type Arch = crate::aarch64::A64;
#[cfg(target_arch = "x86_64")]
type Arch = crate::x86_64::X64;

type ChunkFn = extern "C" fn(*mut f64, *const f64, *const Bundles);

/// A tape compiled to native code. Evaluation mirrors [`Tape`]: a
/// caller-owned work buffer, inputs padded with NaN, and the prolog/main
/// split of a specialized tape.
pub struct NativeTape {
    chunks: Vec<Code>,
    prolog_chunks: usize,
    bundles: Bundles,
    /// The fold code tables of the accumulating kernels; the code holds
    /// their addresses.
    _tables: Vec<Box<[u32]>>,
    outputs: Vec<u32>,
    layout: Layout,
    n_inputs: usize,
    n_ops: usize,
}

/// A function body compiled natively, behind the bundle interface the
/// tape calls bodies through: emitted once, called per instance. A batch
/// of instances runs on the rayon pool when it is worth a fork; either way
/// the result is the serial loop's, bit for bit.
struct NativeBody {
    tape: NativeTape,
    n_out: usize,
}

/// Ops per batched call below which the loop stays on the calling thread.
const PAR_MIN_OPS: usize = 1 << 16;
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

impl ExternBundle for NativeBody {
    fn n_outputs(&self) -> usize {
        self.n_out
    }
    fn call(&self, args: &[f64], out: &mut [f64]) {
        // A pool rather than one buffer: a body that calls a body nests.
        thread_local! {
            static POOL: std::cell::RefCell<Vec<Vec<f64>>> = Default::default();
        }
        let mut work = POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
        self.run_groups(&mut work, args, args.len(), out, 0..1);
        POOL.with(|p| p.borrow_mut().push(work));
    }
    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        if n_groups * self.tape.n_ops < PAR_MIN_OPS || n_groups < 2 {
            thread_local! {
                static POOL: std::cell::RefCell<Vec<Vec<f64>>> = Default::default();
            }
            let mut work = POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
            self.run_groups(&mut work, args, n_args, out, 0..n_groups);
            POOL.with(|p| p.borrow_mut().push(work));
            return;
        }
        // Blocks of instances per task, so a thread amortises its buffer
        // and the scheduler's hand-offs over many bodies.
        let block = (n_groups / (rayon::current_num_threads() * 4)).clamp(1, 4096);
        let n_out = self.n_out.max(1);
        out.par_chunks_mut(block * n_out)
            .enumerate()
            .for_each_init(Vec::new, |work, (b, dst)| {
                let g0 = b * block;
                let g1 = (g0 + block).min(n_groups);
                let ins = &args[g0 * n_args..g1 * n_args];
                self.run_groups(work, ins, n_args, dst, 0..g1 - g0);
            });
    }
}

/// The work array: the tape's slots, then the bundle scratch, then the
/// gather area for host calls.
#[derive(Clone, Copy)]
struct Layout {
    /// First element of the gather area (after the slots).
    gather: usize,
    total: usize,
}

/// One executable chunk.
struct Code {
    /// Kept alive for the code it holds; `func` points into it.
    _map: Mapping,
    func: ChunkFn,
}
// The mapping is immutable after `Mapping::new`, so calling the code from
// any thread is sound and the chunks can be built on a rayon pool.
unsafe impl Send for Code {}
unsafe impl Sync for Code {}

impl NativeTape {
    pub fn compile(tape: &Tape) -> Result<NativeTape, JitError> {
        Self::compile_with(tape, CHUNK_OPS)
    }

    /// Compile with `chunk_ops` ops per emitted function.
    pub fn compile_with(tape: &Tape, chunk_ops: usize) -> Result<NativeTape, JitError> {
        Self::compile_live(tape, chunk_ops, &[])
    }

    /// Compile with `chunk_ops` ops per emitted function, keeping the slots
    /// `live` written to the work array at the end of the program, as the
    /// outputs are: what a consumer that reads a specialized tape's
    /// prolog guards from `work` after [`eval_prolog`](Self::eval_prolog)
    /// passes ([`rsdag::SpecializedTape::prolog_guards`]).
    pub fn compile_live(
        tape: &Tape,
        chunk_ops: usize,
        live: &[u32],
    ) -> Result<NativeTape, JitError> {
        if !cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            return Err(JitError::Unsupported);
        }
        let mut rec = Recorder::default();
        tape.lower(&mut rec);
        // Function bodies that are tapes become native bodies of their own.
        let bundles: Result<Bundles, JitError> = rec
            .bundles
            .iter()
            .map(|b| match b.body() {
                Some(body) => Ok(Arc::new(NativeBody {
                    tape: NativeTape::compile_with(body, chunk_ops)?,
                    n_out: b.n_outputs(),
                }) as Arc<dyn ExternBundle>),
                None => Ok(b.clone()),
            })
            .collect();
        rec.bundles = bundles?;
        // The fold code tables, boxed so their addresses hold for the
        // tape's life; the ops carry the addresses.
        let mut tables: Vec<Box<[u32]>> = Vec::new();
        for op in rec.ops.iter_mut() {
            if let ROp::Gemv {
                acc: Some((_, codes, table)),
                ..
            }
            | ROp::Gemm {
                acc: Some((_, codes, table)),
                ..
            } = op
            {
                let b: Box<[u32]> = codes.clone().into_boxed_slice();
                *table = b.as_ptr() as usize;
                tables.push(b);
            }
        }
        // Inputs the code reads: tagged operands, dense runs, outputs.
        let mut n_inputs = 0usize;
        for op in &rec.ops {
            let mut top = 0usize;
            op.for_each_operand(|k| {
                if let Some(i) = input_index(k) {
                    top = top.max(i as usize + 1);
                }
            });
            n_inputs = n_inputs.max(top);
            let runs: Vec<(&Dense, u32)> = match op {
                ROp::Gemv {
                    a, x, m, n, acc, ..
                } => {
                    let mut v = vec![(a, m * n), (x, *n)];
                    if let Some((Some(c), _, _)) = acc {
                        v.push((c, *m));
                    }
                    v
                }
                ROp::Gemm {
                    a, b, m, k, n, acc, ..
                } => {
                    let mut v = vec![(a, m * k), (b, n * k)];
                    if let Some((Some(c), _, _)) = acc {
                        v.push((c, m * n));
                    }
                    v
                }
                ROp::Solve { a, b, n, .. } => vec![(a, n * n), (b, *n)],
                ROp::SolveMany { a, b, n, k, .. } => vec![(a, n * n), (b, n * k)],
                _ => Vec::new(),
            };
            for (d, len) in runs {
                if let Dense::Inputs(k) = d {
                    n_inputs = n_inputs.max(*k as usize + len as usize);
                }
            }
        }
        for &o in tape.outputs() {
            if let Some(i) = input_index(o) {
                n_inputs = n_inputs.max(i as usize + 1);
            }
        }
        let gather_len = rec.ops.iter().map(ROp::gather_len).max().unwrap_or(0);
        let n_work = tape.n_slots();
        // The last op reading each slot; outputs are read after the program.
        let mut last_use = vec![0u32; n_work.max(1)];
        for (i, op) in rec.ops.iter().enumerate() {
            op.for_each_read(|s| last_use[s as usize] = i as u32);
        }
        for &o in tape.outputs().iter().chain(live) {
            if input_index(o).is_none() {
                last_use[o as usize] = u32::MAX;
            }
        }
        let layout = Layout {
            gather: n_work,
            total: (n_work + gather_len).max(1),
        };
        // Chunk the prolog and main phases separately so no chunk straddles
        // the split; the recorded stream is 1:1 with the tape's ops.
        let chunk_ops = chunk_ops.max(1);
        let split = tape.prolog_len().min(rec.ops.len());
        let (pro, main) = rec.ops.split_at(split);
        let jobs: Vec<&[ROp]> = pro
            .chunks(chunk_ops)
            .chain(main.chunks(chunk_ops))
            .collect();
        let prolog_chunks = pro.chunks(chunk_ops).count();
        let starts: Vec<usize> = jobs
            .iter()
            .scan(0, |acc, ops| {
                let s = *acc;
                *acc += ops.len();
                Some(s)
            })
            .collect();
        let chunks: Result<Vec<Code>, JitError> = jobs
            .par_iter()
            .zip(&starts)
            .map(|(ops, &start)| emit_chunk(ops, start, layout, &last_use))
            .collect();
        Ok(NativeTape {
            chunks: chunks?,
            prolog_chunks,
            bundles: rec.bundles,
            _tables: tables,
            outputs: tape.outputs().to_vec(),
            layout,
            n_inputs,
            n_ops: tape.n_ops(),
        })
    }

    /// Number of emitted functions (diagnostics).
    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
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
            work.len() >= self.layout.total && inputs.len() >= self.n_inputs,
            "buffers not prepared by this tape"
        );
        let (wp, ip, bp) = (
            work.as_mut_ptr(),
            inputs.as_ptr(),
            &self.bundles as *const Bundles,
        );
        for c in &self.chunks[range] {
            (c.func)(wp, ip, bp);
        }
    }

    fn collect(&self, inputs: &[f64], work: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.extend(self.outputs.iter().map(|&s| match input_index(s) {
            Some(i) => inputs.get(i as usize).copied().unwrap_or(f64::NAN),
            None => work[s as usize],
        }));
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

    /// Evaluate many instances at once: `inputs` holds `n` input vectors of
    /// `stride` values back to back, `out` receives the `n` output vectors
    /// back to back. Instances share nothing, so they run on the rayon pool
    /// with a work buffer per thread; this is what a batch of identical
    /// devices, a parameter sweep or an ensemble amounts to.
    pub fn eval_many(&self, inputs: &[f64], stride: usize, out: &mut Vec<f64>) {
        let n = inputs.len().checked_div(stride).unwrap_or(0);
        let n_out = self.outputs.len();
        out.clear();
        out.resize(n * n_out, 0.0);
        out.par_chunks_mut(n_out.max(1))
            .zip(inputs.par_chunks(stride.max(1)))
            .for_each_init(
                || (Vec::new(), Vec::new()),
                |(work, o), (dst, ins)| {
                    self.eval(ins, work, o);
                    dst.copy_from_slice(o);
                },
            );
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
    /// Global index of the last op reading each slot.
    last_use: &'a [u32],
    /// Global index of the op being emitted.
    pos: u32,
    /// Global index of the next op that calls a host routine (`u32::MAX`
    /// when the chunk has none left).
    next_call: u32,
    /// Slot held by each cache register (by cache index).
    held: Vec<Option<u32>>,
    /// Whether the register's value is newer than the slot in memory.
    dirty: Vec<bool>,
    /// Cache index holding each slot.
    at: FxHashMap<u32, usize>,
    /// Cache index of each register number.
    index: [u8; 32],
    /// Round-robin victim pointers of the callee-saved and caller-saved pools.
    next: [usize; 2],
    pinned: Vec<bool>,
}

impl<'a, I: Isa> Emitter<'a, I> {
    fn new(layout: Layout, last_use: &'a [u32], hot: &[*const ()]) -> Emitter<'a, I> {
        let mut index = [0u8; 32];
        for (i, &r) in I::CACHE.iter().enumerate() {
            index[r as usize] = i as u8;
        }
        Emitter {
            isa: I::new(hot),
            layout,
            last_use,
            pos: 0,
            next_call: u32::MAX,
            held: vec![None; I::CACHE.len()],
            dirty: vec![false; I::CACHE.len()],
            at: Default::default(),
            index,
            next: [0, 0],
            pinned: vec![false; I::CACHE.len()],
        }
    }

    /// Forget what cache index `i` holds, writing it back first if memory
    /// does not have it and some later op reads it. The current op counts
    /// as later: an operand it has not fetched yet may be what is evicted
    /// to make room for another.
    fn drop_index(&mut self, i: usize) {
        if let Some(s) = self.held[i].take() {
            if self.dirty[i] && self.last_use[s as usize] >= self.pos {
                self.isa.store(I::CACHE[i], Base::Work, s as usize * 8);
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
    /// A register for the value of `slot`: preferably callee-saved when a
    /// host call comes before the value's last use, so the call does not
    /// cost it a store and a reload; preferably caller-saved otherwise.
    fn fresh_for(&mut self, slot: u32) -> u8 {
        let keep = I::SAVED > 0 && self.next_call <= self.last_use[slot as usize];
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
                    None => self.last_use[s as usize] < self.pos,
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
                self.isa.load(r, Base::Inputs, k as usize * 8);
                self.bind(r, slot);
                r
            }
            None => {
                let r = self.fresh_for(slot);
                self.isa.load(r, Base::Work, slot as usize * 8);
                self.bind(r, slot);
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
    fn bind(&mut self, r: u8, slot: u32) {
        // A slot rebound to a new value: the old one is dead by the tape's
        // construction, so it is dropped without a write-back.
        if let Some(i) = self.at.remove(&slot) {
            self.held[i] = None;
            self.dirty[i] = false;
        }
        let i = self.index[r as usize] as usize;
        self.drop_index(i);
        self.held[i] = Some(slot);
        self.at.insert(slot, i);
    }
    /// `r` is the value of `slot` now; memory will get it when it must.
    fn put(&mut self, slot: u32, r: u8) {
        self.bind(r, slot);
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
    /// A host call whose result is the value of `dst`.
    fn call_into(&mut self, dst: u32, addr: *const (), args: &[Arg]) {
        self.call(addr, args);
        let r = self.fresh_for(dst);
        self.isa.mov(r, I::RESULT);
        self.put(dst, r);
    }
    /// Copy `slots` into the gather area; its byte offset.
    fn gather(&mut self, slots: &[u32]) -> usize {
        self.gather_at(slots, 0)
    }
    /// Copy `slots` into the gather area from element `at` on; the byte
    /// offset of the copy.
    fn gather_at(&mut self, slots: &[u32], at: usize) -> usize {
        let base = (self.layout.gather + at) * 8;
        for (k, &s) in slots.iter().enumerate() {
            let r = self.get(s);
            self.isa.store(r, Base::Work, base + k * 8);
            self.release_except(&[]);
        }
        base
    }

    /// A kernel's two dense operands as host arguments: an input run is
    /// its address, gathered slots are packed into the gather area in
    /// order, so the area holds exactly the slots the ops gather
    /// ([`ROp::gather_len`]).
    fn dense_args(&mut self, a: &Dense, b: &Dense) -> (IArg, IArg) {
        let mut at = 0usize;
        let a = self.dense_arg(a, &mut at);
        let b = self.dense_arg(b, &mut at);
        (a, b)
    }

    /// A dense operand's address: in place (inputs, a consecutive run of
    /// work slots), or gathered into the gather area from `*at`.
    fn dense_arg(&mut self, d: &Dense, at: &mut usize) -> IArg {
        let mut arg = |this: &mut Self, d: &Dense| match d {
            Dense::Inputs(k) => IArg::InputAddr(*k as usize * 8),
            Dense::Slots(s) => {
                let consecutive =
                    !s.is_empty() && s.iter().enumerate().all(|(j, &x)| x == s[0] + j as u32);
                if consecutive {
                    // Read in place: whatever of the run the register cache
                    // still owes to memory is written back first.
                    for &slot in s {
                        if let Some(&i) = this.at.get(&slot) {
                            this.drop_index(i);
                        }
                    }
                    return IArg::WorkAddr(s[0] as usize * 8);
                }
                let p = IArg::WorkAddr(this.gather_at(s, *at));
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
                let (x, y) = (self.get(a), self.get(b));
                let m = self.fresh();
                self.isa.arith(Arith::Mul, m, x, y);
                let z = self.get(c);
                let r = self.fresh_for(dst);
                self.isa.arith(Arith::Add, r, m, z);
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
            ROp::Call(dst, idx, ref args, n_out) => {
                let at = self.gather(args);
                let args = [
                    Arg::I(IArg::Bundles),
                    Arg::I(IArg::Imm(idx as u64)),
                    Arg::I(IArg::WorkAddr(at)),
                    Arg::I(IArg::Imm(args.len() as u64)),
                    Arg::I(IArg::WorkAddr(dst as usize * 8)),
                ];
                self.call(host::h_bundle as *const (), &args);
                self.invalidate(dst, n_out);
            }
            ROp::CallBatch(dst, idx, ref args, n_groups, n_args, n_out) => {
                let at = self.gather(args);
                let args = [
                    Arg::I(IArg::Bundles),
                    Arg::I(IArg::Imm(idx as u64)),
                    Arg::I(IArg::WorkAddr(at)),
                    Arg::I(IArg::Imm(n_groups as u64)),
                    Arg::I(IArg::Imm(n_args as u64)),
                    Arg::I(IArg::WorkAddr(dst as usize * 8)),
                ];
                self.call(host::h_bundle_batch as *const (), &args);
                self.invalidate(dst, n_groups * n_out);
            }
            ROp::Gemv {
                dst,
                ref a,
                ref x,
                m,
                n,
                ref acc,
            } => {
                let mut at = 0usize;
                let a_arg = self.dense_arg(a, &mut at);
                let x_arg = self.dense_arg(x, &mut at);
                match acc {
                    None => {
                        let args = [
                            Arg::I(a_arg),
                            Arg::I(x_arg),
                            Arg::I(IArg::Imm(m as u64)),
                            Arg::I(IArg::Imm(n as u64)),
                            Arg::I(IArg::WorkAddr(dst as usize * 8)),
                        ];
                        self.call(host::h_gemv as *const (), &args);
                    }
                    Some((c, _, table)) => {
                        let c_arg = match c {
                            Some(c) => self.dense_arg(c, &mut at),
                            None => IArg::Imm(0),
                        };
                        let args = [
                            Arg::I(a_arg),
                            Arg::I(x_arg),
                            Arg::I(c_arg),
                            Arg::I(IArg::Imm(*table as u64)),
                            Arg::I(IArg::Imm(m as u64)),
                            Arg::I(IArg::Imm(n as u64)),
                            Arg::I(IArg::WorkAddr(dst as usize * 8)),
                        ];
                        self.call(host::h_gemv_acc as *const (), &args);
                    }
                }
                self.invalidate(dst, m);
            }
            ROp::Gemm {
                dst,
                ref a,
                ref b,
                m,
                k,
                n,
                ref acc,
            } => {
                let mut at = 0usize;
                let a_arg = self.dense_arg(a, &mut at);
                let b_arg = self.dense_arg(b, &mut at);
                match acc {
                    None => {
                        let args = [
                            Arg::I(a_arg),
                            Arg::I(b_arg),
                            Arg::I(IArg::Imm(m as u64)),
                            Arg::I(IArg::Imm(k as u64)),
                            Arg::I(IArg::Imm(n as u64)),
                            Arg::I(IArg::WorkAddr(dst as usize * 8)),
                        ];
                        self.call(host::h_gemm as *const (), &args);
                    }
                    Some((c, _, table)) => {
                        let c_arg = match c {
                            Some(c) => self.dense_arg(c, &mut at),
                            None => IArg::Imm(0),
                        };
                        let args = [
                            Arg::I(a_arg),
                            Arg::I(b_arg),
                            Arg::I(c_arg),
                            Arg::I(IArg::Imm(*table as u64)),
                            Arg::I(IArg::Imm(m as u64)),
                            Arg::I(IArg::Imm(k as u64)),
                            Arg::I(IArg::Imm(n as u64)),
                            Arg::I(IArg::WorkAddr(dst as usize * 8)),
                        ];
                        self.call(host::h_gemm_acc as *const (), &args);
                    }
                }
                self.invalidate(dst, m * n);
            }
            ROp::SolveMany {
                dst,
                ref a,
                ref b,
                n,
                k,
            } => {
                let (a_arg, b_arg) = self.dense_args(a, b);
                let args = [
                    Arg::I(a_arg),
                    Arg::I(b_arg),
                    Arg::I(IArg::Imm(n as u64)),
                    Arg::I(IArg::Imm(k as u64)),
                    Arg::I(IArg::WorkAddr(dst as usize * 8)),
                ];
                self.call(host::h_solve_many as *const (), &args);
                self.invalidate(dst, n * k);
            }
            ROp::Solve {
                dst,
                ref a,
                ref b,
                n,
            } => {
                let (a_arg, b_arg) = self.dense_args(a, b);
                let args = [
                    Arg::I(a_arg),
                    Arg::I(b_arg),
                    Arg::I(IArg::Imm(n as u64)),
                    Arg::I(IArg::WorkAddr(dst as usize * 8)),
                ];
                self.call(host::h_solve as *const (), &args);
                self.invalidate(dst, n);
            }
        }
    }

    fn bin2(&mut self, op: Arith, dst: u32, a: u32, b: u32) {
        let (x, y) = (self.get(a), self.get(b));
        let r = self.fresh_for(dst);
        self.isa.arith(op, r, x, y);
        self.put(dst, r);
    }

    fn unary(&mut self, dst: u32, uop: UnaryOp, a: u32) {
        let x = self.get(a);
        let r = self.fresh_for(dst);
        let inline = match uop {
            UnaryOp::Sqrt => {
                // x > 0 ? sqrt(x) : 0, the reference's guard.
                let zero = self.fconst(0.0);
                let s = self.fresh();
                self.isa.sqrt(s, x);
                self.isa.cmp_select(CmpOp::Gt, x, zero, s, zero, r);
                true
            }
            UnaryOp::Floor => self.isa.round(Round::Floor, r, x),
            UnaryOp::Ceil => self.isa.round(Round::Ceil, r, x),
            UnaryOp::Trunc => self.isa.round(Round::Trunc, r, x),
            UnaryOp::Abs => {
                self.isa.abs(r, x);
                true
            }
            UnaryOp::Sign => {
                let zero = self.fconst(0.0);
                let one = self.fconst(1.0);
                let minus = self.fconst(-1.0);
                let t = self.fresh();
                self.isa.cmp_select(CmpOp::Lt, x, zero, minus, x, t);
                self.isa.cmp_select(CmpOp::Gt, x, zero, one, t, r);
                true
            }
            _ => false,
        };
        if inline {
            self.put(dst, r);
        } else {
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

    /// A reduction (or, with `b`, a dot) in the reference order: four
    /// accumulators, merged as `(a0 + a1) + (a2 + a3)`, then the tail.
    /// Accumulators stay pinned across the terms; a term's registers are
    /// released once it is folded in, so a long list needs seven registers,
    /// not one per operand.
    fn fold(&mut self, op: Arith, ident: f64, a: &[u32], b: Option<&[u32]>) -> u8 {
        let n = a.len();
        let acc: [u8; 4] = std::array::from_fn(|_| self.fconst(ident));
        let ch = n / 4;
        for c in 0..ch {
            for (k, &ak) in acc.iter().enumerate() {
                let t = self.term(a, b, 4 * c + k);
                self.isa.arith(op, ak, ak, t);
                self.release_except(&acc);
            }
        }
        let l = self.fresh();
        self.isa.arith(op, l, acc[0], acc[1]);
        let r = self.fresh();
        self.isa.arith(op, r, acc[2], acc[3]);
        let mut s = self.fresh();
        self.isa.arith(op, s, l, r);
        for k in ch * 4..n {
            let t = self.term(a, b, k);
            let s2 = self.fresh();
            self.isa.arith(op, s2, s, t);
            s = s2;
            self.release_except(&[s]);
        }
        s
    }

    /// Term `k` of a fold: the operand, or the product for a dot.
    fn term(&mut self, a: &[u32], b: Option<&[u32]>, k: usize) -> u8 {
        match b {
            None => self.get(a[k]),
            Some(bb) => {
                let (x, y) = (self.get(a[k]), self.get(bb[k]));
                let p = self.fresh();
                self.isa.arith(Arith::Mul, p, x, y);
                p
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

fn emit_chunk(
    ops: &[ROp],
    start: usize,
    layout: Layout,
    last_use: &[u32],
) -> Result<Code, JitError> {
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
    let mut e: Emitter<Arch> = Emitter::new(layout, last_use, &hot);
    e.isa.prologue();
    for (k, op) in ops.iter().enumerate() {
        e.pos = (start + k) as u32;
        e.next_call = next_call[k];
        e.op(op);
    }
    e.flush();
    e.isa.epilogue();
    let map = Mapping::new(&e.isa.finish())?;
    let func: ChunkFn = unsafe { std::mem::transmute(map.ptr) };
    Ok(Code { _map: map, func })
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
    fn new(bytes: &[u8]) -> Result<Mapping, JitError> {
        let len = bytes.len().max(1);
        unsafe {
            #[cfg(target_os = "macos")]
            let flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT;
            #[cfg(not(target_os = "macos"))]
            let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                flags,
                -1,
                0,
            );
            if ptr == libc::MAP_FAILED {
                return Err(JitError::Codegen("mmap of executable memory failed".into()));
            }
            let ptr = ptr as *mut u8;
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
                libc::mprotect(
                    ptr as *mut libc::c_void,
                    len,
                    libc::PROT_READ | libc::PROT_EXEC,
                );
                __clear_cache(
                    ptr as *mut libc::c_char,
                    ptr.add(bytes.len()) as *mut libc::c_char,
                );
            }
            Ok(Mapping { ptr, len })
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
    fn GetCurrentProcess() -> isize;
    fn FlushInstructionCache(process: isize, addr: *const u8, size: usize) -> i32;
}

#[cfg(windows)]
impl Mapping {
    fn new(bytes: &[u8]) -> Result<Mapping, JitError> {
        const MEM_COMMIT_RESERVE: u32 = 0x1000 | 0x2000;
        const PAGE_EXECUTE_READWRITE: u32 = 0x40;
        let len = bytes.len().max(1);
        unsafe {
            let ptr = VirtualAlloc(
                std::ptr::null_mut(),
                len,
                MEM_COMMIT_RESERVE,
                PAGE_EXECUTE_READWRITE,
            );
            if ptr.is_null() {
                return Err(JitError::Codegen(
                    "VirtualAlloc of executable memory failed".into(),
                ));
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            FlushInstructionCache(GetCurrentProcess(), ptr, bytes.len());
            Ok(Mapping { ptr, len })
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
