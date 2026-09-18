//! What a factorization will cost, before a single node is built: the
//! elimination tree of the symmetrized pattern in a given order, and from
//! it the column counts of the factor (each row's structure is a path
//! climb in the tree, Liu's row subtrees), so fill and flops are known in
//! about the time it takes to read the pattern. A consumer decides on that
//! number whether a subsystem's solve belongs in the graph or in a sparse
//! solver: the graph wins below a few hundred ops per unknown.

use super::Pattern;

/// Predicted cost of a static LU on `pattern` in `order`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cost {
    /// Entries of `L` and `U` beyond the pattern.
    pub fill: usize,
    /// Multiply-adds of the factorization, summed over both triangles.
    pub flops: usize,
}

/// Fill and flops of factoring the symmetrized pattern in `order`
/// (`order[k]` is the vertex eliminated at step `k`). Exact for a
/// symmetric pattern; an upper bound for an unsymmetric one, since a
/// static LU fills within the symmetric envelope.
pub fn cost(pattern: &Pattern, order: &[usize]) -> Cost {
    let n = pattern.len();
    let mut pos = vec![0usize; n];
    for (k, &i) in order.iter().enumerate() {
        pos[i] = k;
    }
    // Lower entries in permuted coordinates: for each k, the j < k with
    // A'(k, j) or A'(j, k).
    let mut lower: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, row) in pattern.iter().enumerate() {
        for &j in row {
            let (a, b) = (pos[i], pos[j]);
            match a.cmp(&b) {
                std::cmp::Ordering::Greater => lower[a].push(b),
                std::cmp::Ordering::Less => lower[b].push(a),
                std::cmp::Ordering::Equal => {}
            }
        }
    }
    let mut nnz_lower = 0usize;
    for l in lower.iter_mut() {
        l.sort_unstable();
        l.dedup();
        nnz_lower += l.len();
    }
    // Elimination tree with path compression (Liu).
    let mut parent = vec![usize::MAX; n];
    let mut ancestor = vec![usize::MAX; n];
    for k in 0..n {
        for &j in &lower[k] {
            let mut r = j;
            while r != usize::MAX && r != k {
                let next = ancestor[r];
                ancestor[r] = k;
                if next == usize::MAX {
                    parent[r] = k;
                    break;
                }
                r = next;
            }
        }
    }
    // Column counts of L: row k's structure is the union of the paths from
    // its lower entries up to k.
    let mut count = vec![1usize; n];
    let mut mark = vec![usize::MAX; n];
    for k in 0..n {
        mark[k] = k;
        for &j in &lower[k] {
            let mut r = j;
            while mark[r] != k {
                mark[r] = k;
                count[r] += 1;
                r = parent[r];
            }
        }
    }
    let nnz_l: usize = count.iter().sum::<usize>() - n;
    let flops: usize = count.iter().map(|&c| (c - 1) * (c - 1)).sum();
    Cost {
        fill: 2 * nnz_l.saturating_sub(nnz_lower),
        flops: 2 * flops,
    }
}
