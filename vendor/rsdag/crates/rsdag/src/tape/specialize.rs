//! Choice specialization: shorten a tape against a recorded `Select` trace,
//! keeping guard outputs that detect a region flip. See [`Tape::specialize`].

use rustc_hash::FxHashSet;

use super::compile::{Kind, Ref};
use super::{input_index, Tape};

impl Tape {
    /// A shortened tape for the region a choice trace describes: `choices`
    /// is one byte per `Select` as recorded by [`Tape::eval_with`] with a
    /// `Vec<u8>` sink (`len == n_selects()`); `pin[k]` pins select `k` to
    /// its traced arm, guarded, and an unpinned select stays a real `Select`
    /// in the shortened tape, both arms live, no guard. A guarded tape
    /// reports through its checks whether the region still holds.
    ///
    /// The tape's program is transformed and compiled again: a pinned
    /// select becomes the arm it took, the condition of every one the
    /// outputs still reach becomes a guard output, and what nothing reads
    /// any more is dropped before the program is scheduled and given slots
    /// like a fresh compilation (the prolog split kept).
    pub fn specialize(&self, choices: &[u8], pin: &[bool]) -> SpecializedTape {
        assert_eq!(
            choices.len(),
            self.n_selects,
            "choice trace length mismatch"
        );
        assert_eq!(pin.len(), self.n_selects, "pin mask length mismatch");
        let mut p = self.lift();
        let m = p.insts.len();
        // Per pinned select, its condition and its taken arm, the selects
        // in stream order as the trace has them.
        let mut pinned: Vec<Option<(Ref, Ref, u8)>> = vec![None; m];
        let mut k = 0usize;
        for (i, slot) in pinned.iter_mut().enumerate() {
            if matches!(p.insts[i].kind, Kind::Select) {
                if pin[k] {
                    let ins = p.ins(i);
                    let arm = if choices[k] != 0 { ins[1] } else { ins[2] };
                    *slot = Some((ins[0], arm, choices[k]));
                }
                k += 1;
            }
        }
        // A pinned select is its arm, through chains of them.
        let resolve = |r: Ref| -> Ref {
            let mut r = r;
            while let Ref::Value(i, _) = r {
                match pinned[i as usize] {
                    Some((_, arm, _)) => r = arm,
                    None => break,
                }
            }
            r
        };
        // The selects the outputs reach, a pinned one reading only its
        // condition and its arm: their conditions are the guards.
        let mut live = vec![false; m];
        let mut stack: Vec<Ref> = p.roots.clone();
        while let Some(r) = stack.pop() {
            let Ref::Value(i, _) = r else { continue };
            if std::mem::replace(&mut live[i as usize], true) {
                continue;
            }
            match pinned[i as usize] {
                Some((cond, arm, _)) => stack.extend([cond, arm]),
                None => stack.extend_from_slice(p.ins(i as usize)),
            }
        }
        let mut seen: FxHashSet<Ref> = FxHashSet::default();
        let mut guards: Vec<Ref> = Vec::new();
        let mut expected: Vec<u8> = Vec::new();
        for i in 0..m {
            if let (true, Some((cond, _, choice))) = (live[i], pinned[i]) {
                // Hash-consing makes shared conditions common; two selects
                // sharing one traced the same truth.
                let cond = resolve(cond);
                if seen.insert(cond) {
                    guards.push(cond);
                    expected.push(choice);
                }
            }
        }
        let n_real = p.roots.len();
        p.roots.extend(guards);
        for r in p.pool.iter_mut().chain(p.roots.iter_mut()) {
            *r = resolve(*r);
        }
        p.retain_reachable();
        let order = p.schedule();
        let mut tape = p.emit(&order, self.prolog_ops > 0);
        tape.n_inputs = self.n_inputs;
        // Guards on an input or a prolog value: checked once after a prolog
        // pass (roots are pinned, so valid across every later main pass).
        let prolog_guards: Vec<(u32, u8)> = (0..expected.len())
            .filter(|&g| match p.roots[n_real + g] {
                Ref::Input(_) => true,
                Ref::Value(i, _) => p.insts[i as usize].pure && tape.prolog_ops > 0,
            })
            .map(|g| (tape.outputs[n_real + g], expected[g]))
            .collect();
        SpecializedTape {
            tape,
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

/// The `Select`s of a tape split over its parameters whose condition is
/// parameter-pure (a prolog value, or a pure input): the choices a
/// parameter binding makes once for every evaluation after it (see
/// [`Tape::param_selects`]).
pub struct ParamSelects {
    /// The distinct conditions, one output each, over the tape's inputs.
    conds: Tape,
    /// Per decided select, its instruction in the tape's program and the
    /// output of `conds` that is its condition.
    of: Vec<(u32, u32)>,
}

impl ParamSelects {
    /// Per instruction of `p` (the tape's program), the arm `pattern`
    /// decides it to, for the selects this decides.
    fn arms(&self, p: &super::compile::Program, pattern: &[bool]) -> Vec<Option<Ref>> {
        let mut arm: Vec<Option<Ref>> = vec![None; p.insts.len()];
        for &(i, c) in &self.of {
            let ins = p.ins(i as usize);
            arm[i as usize] = Some(if pattern[c as usize] { ins[1] } else { ins[2] });
        }
        arm
    }

    /// The number of distinct conditions, the length of a pattern.
    pub fn n_conds(&self) -> usize {
        self.conds.outputs.len()
    }

    /// The pattern of the conditions at the inputs `args` (the tape's
    /// inputs; only the pure ones are read): one truth per condition.
    pub fn pattern(&self, args: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) -> Vec<bool> {
        self.conds.eval(args, work, out);
        out.iter().map(|&c| c != 0.0).collect()
    }

    /// The tape of the conditions, an output each, over the tape's inputs:
    /// what a backend compiles to compute [`pattern`](Self::pattern).
    pub fn tape(&self) -> &Tape {
        &self.conds
    }
}

impl Tape {
    /// Whether the tape has a select a parameter binding decides: any of the
    /// prolog's, or one of the main phase whose condition is a prolog value
    /// or a pure input (`pure_inputs`). A scan of the ops, for deciding
    /// cheaply whether [`param_selects`](Self::param_selects) has anything
    /// to find.
    pub fn has_param_selects(&self, pure_inputs: &[bool]) -> bool {
        self.ops.iter().enumerate().any(|(k, op)| match *op {
            super::Op::Select(_, _, _) if k < self.prolog_ops => true,
            super::Op::Select(c, _, _) => match input_index(c) {
                Some(i) => pure_inputs.get(i as usize).copied().unwrap_or(false),
                None => (c as usize) < self.state_len,
            },
            _ => false,
        })
    }

    /// The selects a parameter binding decides: those whose condition is
    /// computed in the prolog or is a pure input (`pure_inputs`, one flag
    /// per input), in the prolog or the main phase. `None` for a tape
    /// without a prolog split or without such a select.
    pub fn param_selects(&self, pure_inputs: &[bool]) -> Option<ParamSelects> {
        if self.prolog_ops == 0 && !pure_inputs.iter().any(|&p| p) {
            return None;
        }
        let mut p = self.lift();
        let pure = |r: Ref, p: &super::compile::Program| match r {
            Ref::Input(k) => pure_inputs.get(k as usize).copied().unwrap_or(false),
            Ref::Value(j, _) => p.insts[j as usize].pure,
        };
        let mut conds: Vec<Ref> = Vec::new();
        let mut of = Vec::new();
        for i in 0..p.insts.len() {
            let inst = &p.insts[i];
            if !matches!(inst.kind, Kind::Select) {
                continue;
            }
            let c = p.ins(i)[0];
            if !pure(c, &p) {
                continue;
            }
            let k = match conds.iter().position(|&d| d == c) {
                Some(k) => k,
                None => {
                    conds.push(c);
                    conds.len() - 1
                }
            };
            of.push((i as u32, k as u32));
        }
        if of.is_empty() {
            return None;
        }
        p.roots = conds;
        p.retain_reachable();
        let order = p.schedule();
        let mut tape = p.emit(&order, false);
        tape.n_inputs = self.n_inputs;
        Some(ParamSelects { conds: tape, of })
    }

    /// The instructions, undecided and decided by each of `patterns` (see
    /// [`decide`](Self::decide)), counted on the tape's program without
    /// compiling it again: what deciding would remove.
    pub fn ops_decided(&self, ps: &ParamSelects, patterns: &[Vec<bool>]) -> (usize, Vec<usize>) {
        let p = self.lift();
        let count = |pattern: Option<&[bool]>| -> usize {
            let arm = match pattern {
                Some(pattern) => ps.arms(&p, pattern),
                None => vec![None; p.insts.len()],
            };
            let mut seen = vec![false; p.insts.len()];
            let mut stack: Vec<Ref> = p
                .roots
                .iter()
                .map(|&r| decided(&p, &arm, r, true))
                .collect();
            let mut n = 0;
            while let Some(r) = stack.pop() {
                let Ref::Value(i, _) = r else { continue };
                let i = i as usize;
                if std::mem::replace(&mut seen[i], true) {
                    continue;
                }
                n += 1;
                let main = !p.insts[i].pure;
                stack.extend(p.ins(i).iter().map(|&r| decided(&p, &arm, r, main)));
            }
            n
        };
        let decided = patterns.iter().map(|q| count(Some(q))).collect();
        (count(None), decided)
    }

    /// This tape with the selects of `ps` decided by `pattern` (one truth
    /// per condition, see [`ParamSelects::pattern`]):
    /// each the arm its condition picks, what only the other arms read
    /// dropped. Bit for bit the tape's outputs wherever the conditions take
    /// `pattern`; the prolog split kept, its state laid out anew, the main
    /// phase reading no more than the tape's (see [`decided`]). No guards:
    /// the caller computes the pattern.
    pub fn decide(&self, ps: &ParamSelects, pattern: &[bool]) -> Tape {
        let mut p = self.lift();
        assert_eq!(pattern.len(), ps.n_conds(), "pattern length mismatch");
        let arm = ps.arms(&p, pattern);
        // Each operand as the instruction reading it sees it decided.
        let mut main = vec![false; p.pool.len()];
        for inst in &p.insts {
            let (s, l) = inst.ins;
            main[s as usize..(s + l) as usize].fill(!inst.pure);
        }
        let pool: Vec<Ref> = (0..p.pool.len())
            .map(|k| decided(&p, &arm, p.pool[k], main[k]))
            .collect();
        let roots: Vec<Ref> = p
            .roots
            .iter()
            .map(|&r| decided(&p, &arm, r, true))
            .collect();
        (p.pool, p.roots) = (pool, roots);
        p.retain_reachable();
        let order = p.schedule();
        let mut tape = p.emit(&order, self.prolog_ops > 0);
        tape.n_inputs = self.n_inputs;
        tape
    }
}

/// The operand `r` of the program `p`, read by the main phase (`main`) or
/// the prolog, with the selects `arm` decides resolved to their arms. A
/// prolog select whose arm is an input stays, for the main phase: its value
/// is in the state, where its arm would be a parameter the main phase reads
/// itself, passed on every evaluation. Deciding it saves the prolog one op
/// per binding, nothing per evaluation.
fn decided(p: &super::compile::Program, arm: &[Option<Ref>], r: Ref, main: bool) -> Ref {
    let mut r = r;
    while let Ref::Value(i, _) = r {
        match arm[i as usize] {
            Some(Ref::Input(_)) if main && p.insts[i as usize].pure => break,
            Some(a) => r = a,
            None => break,
        }
    }
    r
}
