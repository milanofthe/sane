//! Symbolic determinants.

use rustc_hash::FxHashMap as HashMap;

use crate::field::Field;
use crate::graph::Graph;
use crate::node::ExprId;

/// Determinant of a square matrix of expressions by Laplace expansion along
/// the first row, skipping structurally zero entries (the sparse case a
/// circuit matrix is). A minor is the rows below a point over the columns
/// still free, so each is expanded once however many expansions reach it:
/// `O(n 2^n)` at worst rather than `n!`, and use [`count_det_terms`] to
/// bound the term count first. At most 128 rows.
pub fn determinant<K: Field>(g: &mut Graph<K>, m: &[Vec<ExprId>]) -> ExprId {
    let n = m.len();
    assert!(n <= 128, "a Laplace expansion of at most 128 rows");
    let all = if n == 128 { !0 } else { (1u128 << n) - 1 };
    minor(g, m, all, &mut HashMap::default())
}

/// The determinant of the rows below `n - |free|` over the columns `free`.
fn minor<K: Field>(
    g: &mut Graph<K>,
    m: &[Vec<ExprId>],
    free: u128,
    memo: &mut HashMap<u128, ExprId>,
) -> ExprId {
    if let Some(&d) = memo.get(&free) {
        return d;
    }
    let row = &m[m.len() - free.count_ones() as usize];
    let cols: Vec<usize> = (0..m.len()).filter(|&j| free & (1 << j) != 0).collect();
    let d = match cols.len() {
        0 => g.one(),
        1 => row[cols[0]],
        2 => {
            let next = &m[m.len() - 1];
            let ad = g.mul(row[cols[0]], next[cols[1]]);
            let bc = g.mul(row[cols[1]], next[cols[0]]);
            g.sub(ad, bc)
        }
        _ => {
            let mut acc = g.zero();
            for (k, &j) in cols.iter().enumerate() {
                let entry = row[j];
                if g.is_zero(entry) {
                    continue;
                }
                let sub = minor(g, m, free & !(1 << j), memo);
                let mut term = g.mul(entry, sub);
                if k % 2 == 1 {
                    term = g.neg(term);
                }
                acc = g.add(acc, term);
            }
            acc
        }
    };
    memo.insert(free, d);
    d
}

/// Number of nonzero terms a determinant expansion of a matrix with the
/// given nonzero `pattern` produces, saturating at `cap` (matrices above
/// 100 rows report `cap`).
pub fn count_det_terms(pattern: &[Vec<bool>], cap: u64) -> u64 {
    let n = pattern.len();
    if n > 100 || cap == 0 {
        return cap;
    }
    let rows: Vec<u128> = pattern
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .filter(|&(_, &b)| b)
                .fold(0u128, |m, (j, _)| m | (1u128 << j))
        })
        .collect();
    // The terms of the rows from `depth` on over the columns `avail`, once
    // per column set.
    fn count(
        rows: &[u128],
        depth: usize,
        avail: u128,
        cap: u64,
        memo: &mut HashMap<u128, u64>,
    ) -> u64 {
        if depth == rows.len() {
            return 1;
        }
        if let Some(&c) = memo.get(&avail) {
            return c;
        }
        let mut acc: u64 = 0;
        let mut bits = rows[depth] & avail;
        while bits != 0 && acc < cap {
            let j = bits.trailing_zeros();
            bits &= bits - 1;
            acc = acc.saturating_add(count(rows, depth + 1, avail & !(1u128 << j), cap, memo));
        }
        let acc = acc.min(cap);
        memo.insert(avail, acc);
        acc
    }
    count(&rows, 0, !0u128, cap, &mut HashMap::default())
}
