//! Choice specialization: shorten a tape against a recorded `Select` trace,
//! keeping guard outputs that detect a region flip. See [`Tape::specialize`].

use super::{input_index, Accum, Op, Src, Tape, INPUT};

/// A value of the source tape: an input, or output `off` of op `i`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Val {
    Input(u32),
    Op(u32, u32),
}

impl Tape {
    /// A shortened tape for the region a choice trace describes: `choices`
    /// is one byte per `Select` as recorded by [`Tape::eval_with`] with a
    /// `Vec<u8>` sink (`len == n_selects()`); `pin[k]` pins select `k` to
    /// its traced arm, guarded, and an unpinned select stays a real `Select`
    /// in the shortened tape, both arms live, no guard. A guarded tape
    /// reports through its checks whether the region still holds.
    pub fn specialize(&self, choices: &[u8], pin: &[bool]) -> SpecializedTape {
        assert_eq!(
            choices.len(),
            self.n_selects,
            "choice trace length mismatch"
        );
        assert_eq!(pin.len(), self.n_selects, "pin mask length mismatch");
        let m = self.ops.len();
        let width = |op: &Op| -> u32 {
            match *op {
                Op::Call { n_out, .. } => n_out,
                Op::CallBatch {
                    n_groups, n_out, ..
                } => n_groups * n_out,
                Op::Gemv { m, .. } => m,
                Op::Gemm { m, n, .. } => m * n,
                Op::Solve { n, .. } => n,
                Op::SolveMany { n, k, .. } => n * k,
                _ => 1,
            }
        };
        let pool = |start: u32, len: u32| &self.arg_pool[start as usize..(start + len) as usize];
        // The slot operands of an op, in read order; a kernel's in-place
        // input operands are not slots.
        let operands = |i: usize| -> Vec<u32> {
            let mut v: Vec<u32> = Vec::new();
            let src = |s: Src, len: u32, v: &mut Vec<u32>| {
                if let Src::Pool(start) = s {
                    v.extend_from_slice(pool(start, len));
                }
            };
            match self.ops[i] {
                Op::Const(_) => {}
                Op::Add(a, b)
                | Op::Mul(a, b)
                | Op::Sub(a, b)
                | Op::Cmp(_, a, b)
                | Op::Binary(_, a, b) => v.extend([a, b]),
                Op::MulAdd(a, b, c) | Op::Select(a, b, c) => v.extend([a, b, c]),
                Op::Neg(a) | Op::Powi(a, _) | Op::Unary(_, a) => v.push(a),
                Op::Reduce(_, s, l) => v.extend_from_slice(pool(s, l)),
                Op::Dot(s, l) => v.extend_from_slice(pool(s, 2 * l)),
                Op::Call { start, n_args, .. } => v.extend_from_slice(pool(start, n_args)),
                Op::CallBatch {
                    start,
                    n_groups,
                    n_args,
                    ..
                } => v.extend_from_slice(pool(start, n_groups * n_args)),
                Op::Gemv {
                    a,
                    x,
                    m: rows,
                    n,
                    acc,
                } => {
                    src(a, rows * n, &mut v);
                    src(x, n, &mut v);
                    if let Some(Accum { c: Some(c), .. }) = acc {
                        src(c, rows, &mut v);
                    }
                }
                Op::Gemm {
                    a,
                    b,
                    m: rows,
                    k,
                    n,
                    acc,
                } => {
                    src(a, rows * k, &mut v);
                    src(b, n * k, &mut v);
                    if let Some(Accum { c: Some(c), .. }) = acc {
                        src(c, rows * n, &mut v);
                    }
                }
                Op::Solve { a, b, n } => {
                    src(a, n * n, &mut v);
                    src(b, n, &mut v);
                }
                Op::SolveMany { a, b, n, k } => {
                    src(a, n * n, &mut v);
                    src(b, n * k, &mut v);
                }
            }
            v
        };

        // Forward pass: what value each operand named when it was read.
        // Slots are reused, so this is only recoverable in execution order:
        // `prod[slot]` is the op that produced the slot's current value. A
        // pinned `Select` vacates the value chain: its slot stands for the
        // arm it took, resolved when the select is met, so `vsrc[i]` is the
        // value op `i`'s slot carries.
        let mut prod = vec![u32::MAX; self.n_work];
        let mut vsrc: Vec<Val> = Vec::with_capacity(m);
        // Per op, the values of its slot operands in read order, and the
        // ops that had produced those slots (a pinned select among them:
        // the value chain skips it, the liveness chain runs through it).
        let mut reads: Vec<Vec<Val>> = Vec::with_capacity(m);
        let mut raw: Vec<Vec<u32>> = Vec::with_capacity(m);
        // Per pinned select: (condition value, condition producer, taken
        // arm's producer, traced choice).
        let mut sel: Vec<(Val, u32, u32, u8)> = Vec::with_capacity(self.n_selects);
        let mut sel_at = vec![u32::MAX; m];
        let mut n_sel_seen = 0usize;
        for i in 0..m {
            let val_of = |k: u32| -> Val {
                match input_index(k) {
                    Some(j) => Val::Input(j),
                    None => {
                        let j = prod[k as usize];
                        match vsrc[j as usize] {
                            Val::Op(o, 0) if o == j => Val::Op(j, k - self.dst[j as usize]),
                            alias => alias,
                        }
                    }
                }
            };
            let slots = operands(i);
            let vals: Vec<Val> = slots.iter().map(|&k| val_of(k)).collect();
            let prods: Vec<u32> = slots
                .iter()
                .filter_map(|&k| match input_index(k) {
                    Some(_) => None,
                    None => Some(prod[k as usize]),
                })
                .collect();
            let mut alias: Option<Val> = None;
            if let Op::Select(c, t, e) = self.ops[i] {
                let k = n_sel_seen;
                n_sel_seen += 1;
                if pin[k] {
                    let arm = if choices[k] != 0 { vals[1] } else { vals[2] };
                    let raw_of = |k: u32| match input_index(k) {
                        Some(_) => u32::MAX,
                        None => prod[k as usize],
                    };
                    let arm_raw = raw_of(if choices[k] != 0 { t } else { e });
                    sel_at[i] = sel.len() as u32;
                    sel.push((vals[0], raw_of(c), arm_raw, choices[k]));
                    alias = Some(arm);
                }
            }
            vsrc.push(alias.unwrap_or(Val::Op(i as u32, 0)));
            reads.push(vals);
            raw.push(prods);
            let w = width(&self.ops[i]);
            for k in 0..w {
                prod[(self.dst[i] + k) as usize] = i as u32;
            }
        }
        let resolve = |k: u32| -> Val {
            match input_index(k) {
                Some(j) => Val::Input(j),
                None => {
                    let j = prod[k as usize];
                    match vsrc[j as usize] {
                        Val::Op(o, 0) if o == j => Val::Op(j, k - self.dst[j as usize]),
                        alias => alias,
                    }
                }
            }
        };
        let op_of = |v: Val| -> Option<usize> {
            match v {
                Val::Op(i, _) => Some(i as usize),
                Val::Input(_) => None,
            }
        };
        // Liveness runs over the ops that produced what an op read, pinned
        // selects included: a pinned select keeps its condition (the guard)
        // and its taken arm. Slot lifetimes run over the resolved values,
        // which skip the pinned selects.
        let raw_producers = |i: usize| -> Vec<usize> {
            match sel_at[i] {
                u32::MAX => raw[i].iter().map(|&j| j as usize).collect(),
                s => {
                    let (_, cond, arm, _) = sel[s as usize];
                    [cond, arm]
                        .into_iter()
                        .filter(|&j| j != u32::MAX)
                        .map(|j| j as usize)
                        .collect()
                }
            }
        };
        let producers =
            |i: usize| -> Vec<usize> { reads[i].iter().filter_map(|&v| op_of(v)).collect() };

        // Backward liveness from the outputs.
        let out_old: Vec<Val> = self.outputs.iter().map(|&k| resolve(k)).collect();
        let mut live = vec![false; m];
        let mut stack: Vec<usize> = self
            .outputs
            .iter()
            .filter_map(|&k| match input_index(k) {
                Some(_) => None,
                None => Some(prod[k as usize] as usize),
            })
            .collect();
        while let Some(i) = stack.pop() {
            if live[i] {
                continue;
            }
            live[i] = true;
            stack.extend(raw_producers(i));
        }

        // Guard outputs: each live pinned select's condition, deduped
        // (hash-consing makes shared conditions common; two selects sharing
        // one traced the same truth).
        let mut guards: Vec<Val> = Vec::new();
        let mut expected: Vec<u8> = Vec::new();
        for i in 0..m {
            if !live[i] || sel_at[i] == u32::MAX {
                continue;
            }
            let (cond, _, _, choice) = sel[sel_at[i] as usize];
            if !guards.contains(&cond) {
                guards.push(cond);
                expected.push(choice);
            }
        }

        // Emitted ops: live and not a pinned select. Last use of each
        // producer's block over them; outputs, guards and, under the
        // inherited prolog split, prolog values read by the main phase are
        // pinned.
        let emitted = |i: usize| live[i] && sel_at[i] == u32::MAX;
        let mut last = vec![0usize; m];
        let mut pinned = vec![false; m];
        for i in 0..m {
            if emitted(i) {
                for j in producers(i) {
                    last[j] = i;
                }
            }
        }
        for &v in out_old.iter().chain(guards.iter()) {
            if let Some(i) = op_of(v) {
                pinned[i] = true;
                last[i] = usize::MAX;
            }
        }
        if self.prolog_ops > 0 {
            for i in self.prolog_ops..m {
                if !emitted(i) {
                    continue;
                }
                for j in producers(i) {
                    if j < self.prolog_ops {
                        pinned[j] = true;
                        last[j] = usize::MAX;
                    }
                }
            }
        }

        // Emit the surviving subsequence with fresh slots (the same free-list
        // scheme as `compile`): a value maps to the new base of its producer
        // plus its offset in the block.
        let mut ops: Vec<Op> = Vec::new();
        let mut dst: Vec<u32> = Vec::new();
        let mut arg_pool: Vec<u32> = Vec::new();
        let mut new_base = vec![u32::MAX; m];
        let mut free: Vec<u32> = Vec::new();
        let mut next: u32 = 0;
        let mut max_args = 0usize;
        let mut spec_prolog_ops = 0usize;
        let map_val = |v: Val, new_base: &[u32]| -> u32 {
            match v {
                Val::Input(j) => j | INPUT,
                Val::Op(i, off) => new_base[i as usize] + off,
            }
        };
        for i in 0..m {
            if !emitted(i) {
                continue;
            }
            if i < self.prolog_ops {
                spec_prolog_ops += 1;
            }
            let vals: Vec<u32> = reads[i].iter().map(|&v| map_val(v, &new_base)).collect();
            let mut at = 0usize;
            let mut take = |n: usize| -> Vec<u32> {
                let v = vals[at..at + n].to_vec();
                at += n;
                v
            };
            let gather = |ks: &[u32], arg_pool: &mut Vec<u32>, max_args: &mut usize| -> u32 {
                let start = arg_pool.len() as u32;
                arg_pool.extend_from_slice(ks);
                *max_args = (*max_args).max(ks.len());
                start
            };
            let op = match self.ops[i] {
                Op::Const(v) => Op::Const(v),
                Op::Add(..) => {
                    let o = take(2);
                    Op::Add(o[0], o[1])
                }
                Op::Mul(..) => {
                    let o = take(2);
                    Op::Mul(o[0], o[1])
                }
                Op::MulAdd(..) => {
                    let o = take(3);
                    Op::MulAdd(o[0], o[1], o[2])
                }
                Op::Sub(..) => {
                    let o = take(2);
                    Op::Sub(o[0], o[1])
                }
                Op::Neg(..) => Op::Neg(take(1)[0]),
                Op::Powi(_, n) => Op::Powi(take(1)[0], n),
                Op::Unary(op, _) => Op::Unary(op, take(1)[0]),
                Op::Cmp(op, ..) => {
                    let o = take(2);
                    Op::Cmp(op, o[0], o[1])
                }
                Op::Binary(op, ..) => {
                    let o = take(2);
                    Op::Binary(op, o[0], o[1])
                }
                Op::Select(..) => {
                    let o = take(3);
                    Op::Select(o[0], o[1], o[2])
                }
                Op::Reduce(op, _, l) => {
                    let o = take(l as usize);
                    Op::Reduce(op, gather(&o, &mut arg_pool, &mut max_args), l)
                }
                Op::Dot(_, l) => {
                    let o = take(2 * l as usize);
                    Op::Dot(gather(&o, &mut arg_pool, &mut max_args), l)
                }
                Op::Call {
                    bundle,
                    n_args,
                    n_out,
                    ..
                } => {
                    let o = take(n_args as usize);
                    Op::Call {
                        bundle,
                        start: gather(&o, &mut arg_pool, &mut max_args),
                        n_args,
                        n_out,
                    }
                }
                Op::CallBatch {
                    bundle,
                    n_groups,
                    n_args,
                    n_out,
                    ..
                } => {
                    let o = take((n_groups * n_args) as usize);
                    Op::CallBatch {
                        bundle,
                        start: gather(&o, &mut arg_pool, &mut max_args),
                        n_groups,
                        n_args,
                        n_out,
                    }
                }
                Op::Gemv {
                    a,
                    x,
                    m: rows,
                    n,
                    acc,
                } => {
                    let a = match a {
                        Src::Inputs(k) => Src::Inputs(k),
                        Src::Pool(_) => {
                            let o = take((rows * n) as usize);
                            Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                        }
                    };
                    let x = match x {
                        Src::Inputs(k) => Src::Inputs(k),
                        Src::Pool(_) => {
                            let o = take(n as usize);
                            Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                        }
                    };
                    let acc = acc.map(|Accum { c, codes }| {
                        let c = c.map(|c| match c {
                            Src::Inputs(k) => Src::Inputs(k),
                            Src::Pool(_) => {
                                let o = take(rows as usize);
                                Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                            }
                        });
                        let start = arg_pool.len() as u32;
                        arg_pool.extend_from_slice(pool(codes, rows));
                        Accum { c, codes: start }
                    });
                    max_args = max_args.max((rows * n + n + rows) as usize);
                    Op::Gemv {
                        a,
                        x,
                        m: rows,
                        n,
                        acc,
                    }
                }
                Op::Gemm {
                    a,
                    b,
                    m: rows,
                    k,
                    n,
                    acc,
                } => {
                    let a = match a {
                        Src::Inputs(i) => Src::Inputs(i),
                        Src::Pool(_) => {
                            let o = take((rows * k) as usize);
                            Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                        }
                    };
                    let b = match b {
                        Src::Inputs(i) => Src::Inputs(i),
                        Src::Pool(_) => {
                            let o = take((n * k) as usize);
                            Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                        }
                    };
                    let acc = acc.map(|Accum { c, codes }| {
                        let c = c.map(|c| match c {
                            Src::Inputs(i) => Src::Inputs(i),
                            Src::Pool(_) => {
                                let o = take((rows * n) as usize);
                                Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                            }
                        });
                        let start = arg_pool.len() as u32;
                        arg_pool.extend_from_slice(pool(codes, rows * n));
                        Accum { c, codes: start }
                    });
                    max_args = max_args.max((rows * k + n * k + rows * n) as usize);
                    Op::Gemm {
                        a,
                        b,
                        m: rows,
                        k,
                        n,
                        acc,
                    }
                }
                Op::Solve { a, b, n } => {
                    let a = match a {
                        Src::Inputs(k) => Src::Inputs(k),
                        Src::Pool(_) => {
                            let o = take((n * n) as usize);
                            Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                        }
                    };
                    let b = match b {
                        Src::Inputs(k) => Src::Inputs(k),
                        Src::Pool(_) => {
                            let o = take(n as usize);
                            Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                        }
                    };
                    max_args = max_args.max((n * n + n) as usize);
                    Op::Solve { a, b, n }
                }
                Op::SolveMany { a, b, n, k } => {
                    let a = match a {
                        Src::Inputs(k) => Src::Inputs(k),
                        Src::Pool(_) => {
                            let o = take((n * n) as usize);
                            Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                        }
                    };
                    let b = match b {
                        Src::Inputs(k) => Src::Inputs(k),
                        Src::Pool(_) => {
                            let o = take((n * k) as usize);
                            Src::Pool(gather(&o, &mut arg_pool, &mut max_args))
                        }
                    };
                    max_args = max_args.max((n * n + n * k) as usize);
                    Op::SolveMany { a, b, n, k }
                }
            };
            // Free the blocks that die at this step, then take a slot or a
            // block.
            let mut dying: Vec<usize> = producers(i)
                .into_iter()
                .filter(|&j| last[j] == i && !pinned[j])
                .collect();
            dying.sort_unstable();
            dying.dedup();
            for j in dying {
                let w = width(&self.ops[j]);
                free.extend(new_base[j]..new_base[j] + w);
            }
            let w = width(&self.ops[i]);
            let d = if w == 1 {
                free.pop().unwrap_or_else(|| {
                    let s = next;
                    next += 1;
                    s
                })
            } else {
                let s = next;
                next += w;
                s
            };
            new_base[i] = d;
            ops.push(op);
            dst.push(d);
        }

        let n_real = out_old.len();
        let outputs: Vec<u32> = out_old
            .iter()
            .chain(guards.iter())
            .map(|&v| map_val(v, &new_base))
            .collect();
        // Guards whose condition lives in the prolog, or is an input: checked
        // once after a prolog pass (pinned, so valid across every later main
        // pass).
        let prolog_guards: Vec<(u32, u8)> = guards
            .iter()
            .zip(&expected)
            .filter(|(&v, _)| match op_of(v) {
                None => true,
                Some(i) => i < self.prolog_ops,
            })
            .map(|(&v, &e)| (map_val(v, &new_base), e))
            .collect();
        let n_selects_out = ops.iter().filter(|o| matches!(o, Op::Select(..))).count();
        SpecializedTape {
            tape: Tape {
                ops,
                dst,
                n_selects: n_selects_out,
                arg_pool,
                outputs,
                n_work: next as usize,
                max_args,
                bundles: self.bundles.clone(),
                prolog_ops: spec_prolog_ops,
            },
            n_real,
            expected,
            prolog_guards,
        }
    }
}

/// A [`Tape`] shortened against a choice trace ([`Tape::specialize`]): the
/// pinned `Select`s gone, their untaken arms removed, plus guard outputs
/// that re-validate the trace at every evaluation.
pub struct SpecializedTape {
    tape: Tape,
    /// The first `n_real` outputs are the original tape's; the rest are guards.
    n_real: usize,
    /// Expected truth (`1`/`0`) of each guard output.
    expected: Vec<u8>,
    /// Guards whose condition lives in the inherited prolog (or is an
    /// input), as `(operand, expected)`, checked right after a prolog pass.
    prolog_guards: Vec<(u32, u8)>,
}

impl SpecializedTape {
    /// Instruction count of the shortened tape (vs. [`Tape::n_ops`]).
    pub fn n_ops(&self) -> usize {
        self.tape.ops.len()
    }

    /// The shortened tape itself, for other backends (its outputs are the
    /// real outputs followed by the guards).
    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// Number of real (non-guard) outputs.
    pub fn n_real(&self) -> usize {
        self.n_real
    }

    /// Expected truth (`1`/`0`) of each guard output, for a caller that
    /// evaluates [`tape`](Self::tape) through another backend and re-implements
    /// the [`eval_checked`](Self::eval_checked) guard test.
    pub fn expected(&self) -> &[u8] {
        &self.expected
    }

    /// Evaluate the shortened tape. Returns `true` if every pinned choice still
    /// holds, in which case `out` is bit-exact against the full tape. On
    /// `false` a region flipped and `out` is NOT valid: re-trace on the full
    /// tape ([`Tape::eval_with`]) and respecialize.
    pub fn eval_checked(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) -> bool {
        self.tape.eval(inputs, work, out);
        self.check_outputs(out)
    }

    /// Evaluate the inherited prolog prefix into `work` and check the guards
    /// that live in it. `false` means a *parameter* change flipped a pinned
    /// region: respecialize before running any main pass.
    pub fn eval_prolog_checked(&self, inputs: &[f64], work: &mut Vec<f64>) -> bool {
        self.tape.eval_prolog(inputs, work);
        self.check_prolog_guards(inputs, work)
    }

    /// The prolog-resident guards as `(operand, expected)`, for a backend
    /// that re-implements [`check_prolog_guards`](Self::check_prolog_guards).
    pub fn prolog_guards(&self) -> &[(u32, u8)] {
        &self.prolog_guards
    }

    /// Check the prolog-resident guards against a `work` buffer some backend
    /// filled with this tape's prolog pass.
    pub fn check_prolog_guards(&self, inputs: &[f64], work: &[f64]) -> bool {
        self.prolog_guards.iter().all(|&(k, e)| {
            let v = match input_index(k) {
                Some(i) => inputs.get(i as usize).copied().unwrap_or(f64::NAN),
                None => work[k as usize],
            };
            (v != 0.0) == (e != 0)
        })
    }

    /// Evaluate the main phase over a buffer prepared by
    /// [`eval_prolog_checked`](Self::eval_prolog_checked), guard-checked like
    /// [`eval_checked`](Self::eval_checked).
    pub fn eval_main_checked(&self, inputs: &[f64], work: &mut [f64], out: &mut Vec<f64>) -> bool {
        self.tape.eval_main(inputs, work, out);
        self.check_outputs(out)
    }

    /// Verify guard outputs produced by another backend's evaluation of
    /// [`tape`](Self::tape) (`out` = real outputs ++ guards); truncates
    /// `out` to the real outputs.
    pub fn check_outputs(&self, out: &mut Vec<f64>) -> bool {
        let ok = out[self.n_real..]
            .iter()
            .zip(&self.expected)
            .all(|(&v, &e)| (v != 0.0) == (e != 0));
        out.truncate(self.n_real);
        ok
    }
}
