//! Block triangular form: a maximum transversal pairs every column with a
//! row holding a structural nonzero (augmenting paths, MC21-style), then
//! Tarjan's strongly connected components on the matched matrix give a
//! permutation to block upper triangular form. Every entry outside the
//! diagonal blocks lies above them, each block is irreducible, and a block
//! never fills into another: the solve is one small static LU per block
//! and a substitution across them. The transversal is also the structural
//! guarantee a static pivot needs, a zero-free diagonal. An incomplete
//! matching proves the matrix structurally singular.

use super::Pattern;

/// The permutation to block upper triangular form.
#[derive(Clone, Debug)]
pub struct Btf {
    /// Permuted row `k` is original row `row_perm[k]`.
    pub row_perm: Vec<usize>,
    /// Permuted column `k` is original column `col_perm[k]`.
    pub col_perm: Vec<usize>,
    /// Block `b` spans permuted indices `blocks[b]..blocks[b + 1]`.
    pub blocks: Vec<usize>,
}

impl Btf {
    pub fn n_blocks(&self) -> usize {
        self.blocks.len() - 1
    }
    pub fn block(&self, b: usize) -> std::ops::Range<usize> {
        self.blocks[b]..self.blocks[b + 1]
    }
}

/// A row for every column: `row_of[j]` holds a structural nonzero in
/// column `j`, all distinct. `None` when no such matching exists, which
/// is exactly a structurally singular matrix.
pub fn max_transversal(pattern: &Pattern) -> Option<Vec<usize>> {
    let n = pattern.len();
    let mut row_of: Vec<usize> = vec![usize::MAX; n];
    let mut col_of: Vec<usize> = vec![usize::MAX; n];
    // The diagonal first: cheap, and the common case for a circuit.
    for (i, row) in pattern.iter().enumerate() {
        if row.contains(&i) {
            row_of[i] = i;
            col_of[i] = i;
        }
    }
    // Then an augmenting path from every unmatched row, depth first with
    // an explicit stack over (row, next column index to try).
    let mut stamp = vec![usize::MAX; n];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for start in 0..n {
        if col_of[start] != usize::MAX {
            continue;
        }
        stack.clear();
        stack.push((start, 0));
        stamp[start] = start;
        let mut found = false;
        'search: while let Some(&mut (r, ref mut next)) = stack.last_mut() {
            while *next < pattern[r].len() {
                let j = pattern[r][*next];
                *next += 1;
                let owner = row_of[j];
                if owner == usize::MAX {
                    // Free column: flip the path.
                    let mut jj = j;
                    for &(rr, nx) in stack.iter().rev() {
                        let prev = col_of[rr];
                        row_of[jj] = rr;
                        col_of[rr] = jj;
                        let _ = nx;
                        jj = prev;
                    }
                    found = true;
                    break 'search;
                }
                if stamp[owner] != start {
                    stamp[owner] = start;
                    stack.push((owner, 0));
                    continue 'search;
                }
            }
            stack.pop();
        }
        if !found {
            return None;
        }
    }
    Some(row_of)
}

/// The block upper triangular form of a pattern, or `None` when it is
/// structurally singular.
pub fn block_triangular(pattern: &Pattern) -> Option<Btf> {
    let n = pattern.len();
    let row_of = max_transversal(pattern)?;
    // The matched matrix as a directed graph on columns: j reaches k when
    // the row matched to j has an entry in column k.
    let succ = |j: usize| pattern[row_of[j]].iter().copied().filter(move |&k| k != j);
    // Tarjan, iterative. Components come out sinks first (reverse
    // topological order of the condensation); numbering them in that
    // order would make the form block lower triangular, so the block
    // order is reversed at the end.
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut tarjan: Vec<usize> = Vec::new();
    let mut comps: Vec<Vec<usize>> = Vec::new();
    let mut next_index = 0usize;
    let mut call: Vec<(usize, usize)> = Vec::new(); // (node, next successor position)
    let succ_list: Vec<Vec<usize>> = (0..n).map(|j| succ(j).collect()).collect();
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        call.push((root, 0));
        index[root] = next_index;
        low[root] = next_index;
        next_index += 1;
        tarjan.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut pos)) = call.last_mut() {
            if *pos < succ_list[v].len() {
                let w = succ_list[v][*pos];
                *pos += 1;
                if index[w] == usize::MAX {
                    index[w] = next_index;
                    low[w] = next_index;
                    next_index += 1;
                    tarjan.push(w);
                    on_stack[w] = true;
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                call.pop();
                if let Some(&(u, _)) = call.last() {
                    low[u] = low[u].min(low[v]);
                }
                if low[v] == index[v] {
                    let mut comp = Vec::new();
                    loop {
                        let w = tarjan.pop().expect("the component root is on the stack");
                        on_stack[w] = false;
                        comp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    comp.sort_unstable();
                    comps.push(comp);
                }
            }
        }
    }
    comps.reverse();
    let mut col_perm = Vec::with_capacity(n);
    let mut blocks = vec![0];
    for comp in &comps {
        col_perm.extend_from_slice(comp);
        blocks.push(col_perm.len());
    }
    let row_perm = col_perm.iter().map(|&j| row_of[j]).collect();
    Some(Btf {
        row_perm,
        col_perm,
        blocks,
    })
}
