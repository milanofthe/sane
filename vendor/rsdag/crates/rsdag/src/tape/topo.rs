//! The dynamic topological order of the accumulator fusion (see
//! `Program::fuse_accumulators`): whether a kernel can read a value
//! without a cycle, kept answerable as kernels take accumulators.

use std::collections::BTreeSet;

use rustc_hash::FxHashMap as HashMap;

use super::compile::{Kind, Program, Ref};

/// A topological order of a program's instructions that the fold pass
/// keeps valid as kernels take accumulators, by two-way search (Haeupler,
/// Kavitha, Mathew, Sen and Tarjan). Whether accumulator `a` depends on
/// kernel `k` is only a question when `a` comes after `k`; then the
/// readers of `k` searched forward up to `a`, and what `a` reads searched
/// backward down to `k`, step in turn. Either finding the other end is a
/// cycle. The side that runs out first is closed under its direction
/// between the two and moves, the backward side right before `k` or the
/// forward side right after `a`, so a question costs the smaller side.
/// Instructions added after the order was taken (the placeholder) are
/// constants and take part in no edge.
pub(super) struct Topo {
    order: Order,
    /// Readers of each instruction (`readers[start[i]..start[i + 1]]`),
    /// and the ones the pass added.
    start: Vec<u32>,
    readers: Vec<u32>,
    added: HashMap<u32, Vec<u32>>,
    seen: Vec<u32>,
    stamp: u32,
    stack_f: Vec<(u32, u32)>,
    stack_b: Vec<(u32, u32)>,
    fwd: Vec<u32>,
    back: Vec<u32>,
}

impl Topo {
    /// The order of `p` with kernel `k` also reading `taken`, the
    /// accumulators it took before the order was needed.
    pub(super) fn new(p: &Program, hints: &[(u32, u32)], k: u32, taken: &[u32]) -> Topo {
        let m = p.insts.len();
        // Every edge `(j, i)`, instruction `i` reading `j`, once.
        let edges = |f: &mut dyn FnMut(u32, u32)| {
            let mut last = vec![u32::MAX; m];
            for i in 0..m {
                for r in p.ins(i) {
                    if let Ref::Value(j, _) = *r {
                        if last[j as usize] != i as u32 {
                            last[j as usize] = i as u32;
                            f(j, i as u32);
                        }
                    }
                }
            }
            for &a in taken {
                if last[a as usize] != k {
                    last[a as usize] = k;
                    f(a, k);
                }
            }
        };
        let mut count = vec![0u32; m + 1];
        edges(&mut |j, _| count[j as usize + 1] += 1);
        for i in 0..m {
            count[i + 1] += count[i];
        }
        let mut fill = count.clone();
        let mut readers = vec![0u32; count[m] as usize];
        edges(&mut |j, i| {
            readers[fill[j as usize] as usize] = i;
            fill[j as usize] += 1;
        });
        // The starting order: a kernel after the accumulators `hints`
        // offers it (`(operand, kernel)`, sorted) where the program allows,
        // and otherwise as late as its readers allow. The lowering places a
        // kernel with its first consumer, before the accumulators of its
        // other outputs and before the kernels those depend on, so most
        // accumulators would otherwise come after their kernel. When
        // nothing else can go, the kernel with the fewest hints open does
        // (a hint that depends on its kernel never closes).
        let kernel = |i: usize| matches!(p.insts[i].kind, Kind::Gemv { .. } | Kind::Gemm { .. });
        let mut pending = vec![0u32; m];
        for &r in &readers {
            pending[r as usize] += 1;
        }
        let mut hinted = vec![0u32; m];
        for &(_, k) in hints {
            hinted[k as usize] += 1;
        }
        let hints_of = |x: u32| {
            let lo = hints.partition_point(|&(y, _)| y < x);
            let hi = hints.partition_point(|&(y, _)| y <= x);
            &hints[lo..hi]
        };
        let mut ready: Vec<u32> = Vec::new();
        let mut ready_kernels: BTreeSet<u32> = BTreeSet::new();
        // Kernels whose operands are there, by their hints still open.
        let mut blocked: BTreeSet<(u32, u32)> = BTreeSet::new();
        let release = |r: u32,
                       hinted: &[u32],
                       ready: &mut Vec<u32>,
                       ready_kernels: &mut BTreeSet<u32>,
                       blocked: &mut BTreeSet<(u32, u32)>| {
            if !kernel(r as usize) {
                ready.push(r);
            } else if hinted[r as usize] == 0 {
                ready_kernels.insert(r);
            } else {
                blocked.insert((hinted[r as usize], r));
            }
        };
        for i in (0..m as u32).rev() {
            if pending[i as usize] == 0 {
                release(i, &hinted, &mut ready, &mut ready_kernels, &mut blocked);
            }
        }
        let mut at: Vec<u32> = Vec::with_capacity(m);
        while let Some(i) = ready
            .pop()
            .or_else(|| ready_kernels.pop_first())
            .or_else(|| blocked.pop_first().map(|(_, k)| k))
        {
            at.push(i);
            for &(_, k) in hints_of(i) {
                let h = hinted[k as usize];
                hinted[k as usize] = h - 1;
                if blocked.remove(&(h, k)) {
                    if h == 1 {
                        ready_kernels.insert(k);
                    } else {
                        blocked.insert((h - 1, k));
                    }
                }
            }
            for &r in &readers[count[i as usize] as usize..count[i as usize + 1] as usize] {
                pending[r as usize] -= 1;
                if pending[r as usize] == 0 {
                    release(r, &hinted, &mut ready, &mut ready_kernels, &mut blocked);
                }
            }
        }
        debug_assert_eq!(at.len(), m, "the lowered program is acyclic");
        Topo {
            order: Order::new(&at),
            start: count,
            readers,
            added: HashMap::default(),
            seen: vec![0; m],
            stamp: 0,
            stack_f: Vec::new(),
            stack_b: Vec::new(),
            fwd: Vec::new(),
            back: Vec::new(),
        }
    }

    /// Lets kernel `k` read instruction `a` unless `a` depends on `k`;
    /// whether it did.
    pub(super) fn read(&mut self, p: &Program, k: u32, a: u32) -> bool {
        let n = self.seen.len() as u32;
        if a >= n {
            return true;
        }
        let label = &self.order.label;
        let (lb, ub) = (label[k as usize], label[a as usize]);
        if ub < lb {
            self.added.entry(a).or_default().push(k);
            return true;
        }
        // Forward marks are `sf`, backward ones `sb`. The sides take one
        // edge each in turn (a kernel's readers can be thousands), each a
        // stack of nodes with the next edge to look at.
        if self.stamp > u32::MAX - 4 {
            self.seen.fill(0);
            self.stamp = 0;
        }
        self.stamp += 2;
        let (sf, sb) = (self.stamp, self.stamp + 1);
        self.stack_f.clear();
        self.stack_b.clear();
        self.fwd.clear();
        self.back.clear();
        self.stack_f.push((k, 0));
        self.fwd.push(k);
        self.seen[k as usize] = sf;
        self.stack_b.push((a, 0));
        self.back.push(a);
        self.seen[a as usize] = sb;
        let backward = loop {
            // One forward edge.
            let Some(&(w, e)) = self.stack_f.last() else {
                break false;
            };
            let (s0, s1) = (self.start[w as usize], self.start[w as usize + 1]);
            let deg = s1 - s0;
            let r = if e < deg {
                Some(self.readers[(s0 + e) as usize])
            } else {
                self.added
                    .get(&w)
                    .and_then(|v| v.get((e - deg) as usize))
                    .copied()
            };
            match r {
                None => {
                    self.stack_f.pop();
                }
                Some(r) => {
                    self.stack_f.last_mut().unwrap().1 += 1;
                    if r == a {
                        return false;
                    }
                    if self.seen[r as usize] != sf && label[r as usize] < ub {
                        self.seen[r as usize] = sf;
                        self.fwd.push(r);
                        self.stack_f.push((r, 0));
                    }
                }
            }
            // One backward edge.
            let Some(&(w, e)) = self.stack_b.last() else {
                break true;
            };
            match p.ins(w as usize).get(e as usize) {
                None => {
                    self.stack_b.pop();
                }
                Some(&r) => {
                    self.stack_b.last_mut().unwrap().1 += 1;
                    if let Ref::Value(j, _) = r {
                        if j == k {
                            return false;
                        }
                        if j < n && self.seen[j as usize] != sb && label[j as usize] > lb {
                            self.seen[j as usize] = sb;
                            self.back.push(j);
                            self.stack_b.push((j, 0));
                        }
                    }
                }
            }
        };
        let order = &mut self.order;
        let set = if backward {
            &mut self.back
        } else {
            &mut self.fwd
        };
        set.sort_unstable_by_key(|&w| order.label[w as usize]);
        for &w in set.iter() {
            order.unlink(w);
        }
        let mut at = if backward { order.prev[k as usize] } else { a };
        for &w in set.iter() {
            order.insert_after(at, w);
            at = w;
        }
        self.added.entry(a).or_default().push(k);
        true
    }
}

/// A list whose order is compared by labels (the order maintenance of
/// Bender, Cole, Demaine, Farach-Colton and Zito): an element moved in
/// takes the middle of the gap it lands in, and a gap too narrow relabels
/// the smallest aligned label range around it that is sparse enough, its
/// allowance shrinking with its size, amortized O(log n) labels a move.
struct Order {
    label: Vec<u64>,
    prev: Vec<u32>,
    next: Vec<u32>,
}

impl Order {
    /// The elements `0..n` in `order`, between a head (`n`, label 0) and a
    /// tail (`n + 1`, the largest label).
    fn new(order: &[u32]) -> Order {
        let n = order.len();
        let (head, tail) = (n as u32, n as u32 + 1);
        let mut label = vec![0u64; n + 2];
        let mut prev = vec![0u32; n + 2];
        let mut next = vec![0u32; n + 2];
        let step = u64::MAX / (n as u64 + 2);
        let mut last = head;
        for (q, &w) in order.iter().enumerate() {
            label[w as usize] = (q as u64 + 1) * step;
            prev[w as usize] = last;
            next[last as usize] = w;
            last = w;
        }
        next[last as usize] = tail;
        prev[tail as usize] = last;
        label[tail as usize] = u64::MAX;
        Order { label, prev, next }
    }

    fn unlink(&mut self, w: u32) {
        let (p, q) = (self.prev[w as usize], self.next[w as usize]);
        self.next[p as usize] = q;
        self.prev[q as usize] = p;
    }

    /// Links `w` right after `x`.
    fn insert_after(&mut self, x: u32, w: u32) {
        let y = self.next[x as usize];
        self.prev[w as usize] = x;
        self.next[w as usize] = y;
        self.next[x as usize] = w;
        self.prev[y as usize] = w;
        let (lo, hi) = (self.label[x as usize], self.label[y as usize]);
        if hi - lo >= 2 {
            self.label[w as usize] = lo + (hi - lo) / 2;
            return;
        }
        let head = self.label.len() as u32 - 2;
        let tail = head + 1;
        for i in 1..=64u32 {
            let size = 1u128 << i;
            let base = lo as u128 & !(size - 1);
            let end = base + size;
            let mut first = w;
            let mut c = 1u128;
            let mut v = x;
            while v != head && self.label[v as usize] as u128 >= base {
                first = v;
                c += 1;
                v = self.prev[v as usize];
            }
            let mut v = y;
            while v != tail && (self.label[v as usize] as u128) < end {
                c += 1;
                v = self.next[v as usize];
            }
            // Sparse enough: fewer than (2 / 1.5)^i elements in 2^i labels.
            if i == 64 || (c as f64) < (4.0f64 / 3.0).powi(i as i32) {
                let from = base.max(1);
                let to = end.min(u64::MAX as u128);
                let gap = (to - from) / (c + 1);
                let mut v = first;
                for j in 0..c {
                    self.label[v as usize] = (from + gap * (j + 1)) as u64;
                    v = self.next[v as usize];
                }
                return;
            }
        }
    }
}
