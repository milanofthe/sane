//! The planned scalar elimination run over panels: consecutive steps
//! along the elimination forest (postordered) form a panel while the
//! explicit zeros stay within an allowance, and the elimination proceeds
//! over the panel partition as a block elimination of variable block
//! sizes ([`super::solve_block_planned_sizes`]): the fill of the scalar
//! ordering, the flops in the dense kernels. The pivot rows are the
//! plan's; within a panel the dense kernel pivots, across panels the
//! block guard applies, so [`super::Plan::repivot`] serves this program as
//! it serves the scalar one.

use super::block::{solve_block_planned_sizes, Block, BlockRows};
use super::num::Num;
use super::{Btf, Pattern, Plan, Solved};
use crate::field::Field;
use crate::graph::Graph;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// Panels grow to at most this width.
pub const MAX_WIDTH: usize = 64;

/// The share of explicit zeros a panel of `width` columns may carry in
/// its rows below the diagonal block: lenient for narrow panels (a few
/// zero flops buy a kernel), strict for wide ones.
fn allowed_zeros(width: usize) -> f64 {
    match width {
        0..=4 => 0.8,
        5..=16 => 0.3,
        17..=32 => 0.1,
        _ => 0.05,
    }
}

/// The panel partition of a planned elimination.
#[derive(Clone, Debug)]
pub struct Supernodes {
    /// The original row that pivots at step `s`.
    row_of: Vec<usize>,
    /// The original column eliminated at step `s`.
    col_of: Vec<usize>,
    /// Panel `p` spans steps `bounds[p]..bounds[p + 1]`.
    bounds: Vec<usize>,
    /// Block `b` of the block triangular form spans panels
    /// `blocks[b]..blocks[b + 1]`.
    blocks: Vec<usize>,
}

impl Supernodes {
    pub fn n_panels(&self) -> usize {
        self.bounds.len() - 1
    }

    /// The panels' widths.
    pub fn widths(&self) -> Vec<usize> {
        self.bounds.windows(2).map(|w| w[1] - w[0]).collect()
    }

    pub fn max_width(&self) -> usize {
        self.widths().into_iter().max().unwrap_or(0)
    }

    /// The share of unknowns in panels wider than one: what the kernels
    /// cover.
    pub fn panel_share(&self) -> f64 {
        let n = self.row_of.len().max(1);
        self.widths().iter().filter(|&&w| w > 1).sum::<usize>() as f64 / n as f64
    }

    /// The order in which the program reads the right-hand side in place:
    /// for each unknown (original row) the step whose panel it belongs to,
    /// so a right-hand side laid out by step is read block by block.
    pub fn rhs_order(&self) -> Vec<usize> {
        let n = self.row_of.len();
        let mut step_of_row = vec![0usize; n];
        for s in 0..n {
            step_of_row[self.row_of[s]] = s;
        }
        step_of_row
    }

    /// The order in which the program reads the entries in place: for each
    /// entry `k` of `entries` (original coordinates) its position in a
    /// layout that goes block by block over the panels, a block right of
    /// its pivot panel column-major and every other block row-major. A
    /// consumer whose inputs follow this order feeds the kernels without
    /// a gather.
    pub fn value_order(&self, entries: &[(usize, usize)]) -> Vec<usize> {
        let n = self.row_of.len();
        let mut step_of_row = vec![0usize; n];
        let mut step_of_col = vec![0usize; n];
        for s in 0..n {
            step_of_row[self.row_of[s]] = s;
            step_of_col[self.col_of[s]] = s;
        }
        let np = self.n_panels();
        let mut panel_of = vec![0usize; n];
        for p in 0..np {
            for s in self.bounds[p]..self.bounds[p + 1] {
                panel_of[s] = p;
            }
        }
        // Sort key per entry: (block row, block column, within-block index).
        let mut keyed: Vec<((usize, usize, usize), usize)> = entries
            .iter()
            .enumerate()
            .map(|(k, &(i, j))| {
                let (rs, cs) = (step_of_row[i], step_of_col[j]);
                let (pr, pc) = (panel_of[rs], panel_of[cs]);
                let (r, c) = (rs - self.bounds[pr], cs - self.bounds[pc]);
                let (sr, sc) = (
                    self.bounds[pr + 1] - self.bounds[pr],
                    self.bounds[pc + 1] - self.bounds[pc],
                );
                let within = if pc > pr { c * sr + r } else { r * sc + c };
                ((pr, pc, within), k)
            })
            .collect();
        keyed.sort_unstable();
        let mut order = vec![0usize; entries.len()];
        for (pos, (_, k)) in keyed.into_iter().enumerate() {
            order[k] = pos;
        }
        order
    }
}

/// The symbolic elimination of a pattern in step coordinates (`entries`
/// as `(row step, column step)`): the structure of every column of `L`
/// below the diagonal, after fill, and of every row of `U`.
fn symbolic(m: usize, entries: &[(usize, usize)]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut present: HashSet<(usize, usize)> = HashSet::default();
    let mut in_row: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut in_col: Vec<Vec<usize>> = vec![Vec::new(); m];
    for &(rs, cs) in entries {
        if present.insert((rs, cs)) {
            in_row[rs].push(cs);
            in_col[cs].push(rs);
        }
    }
    let mut colstruct: Vec<Vec<usize>> = Vec::with_capacity(m);
    let mut rowstruct: Vec<Vec<usize>> = Vec::with_capacity(m);
    for k in 0..m {
        let mut row_k: Vec<usize> = in_row[k].iter().copied().filter(|&c| c > k).collect();
        let mut col_k: Vec<usize> = in_col[k].iter().copied().filter(|&r| r > k).collect();
        row_k.sort_unstable();
        row_k.dedup();
        col_k.sort_unstable();
        col_k.dedup();
        for &i in &col_k {
            for &j in &row_k {
                if present.insert((i, j)) {
                    in_row[i].push(j);
                    in_col[j].push(i);
                }
            }
        }
        colstruct.push(col_k);
        rowstruct.push(row_k);
    }
    (colstruct, rowstruct)
}

/// The postorder of the elimination forest whose parent of `k` is the
/// first row below the diagonal in column `k` of `L`: children before
/// parents, so the chains a panel grows along are adjacent.
fn postorder(colstruct: &[Vec<usize>]) -> Vec<usize> {
    let m = colstruct.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut roots = Vec::new();
    for k in 0..m {
        match colstruct[k].first() {
            Some(&p) => children[p].push(k),
            None => roots.push(k),
        }
    }
    let mut order = Vec::with_capacity(m);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for &r in &roots {
        stack.push((r, 0));
        while let Some(&mut (node, ref mut next)) = stack.last_mut() {
            if *next < children[node].len() {
                let c = children[node][*next];
                *next += 1;
                stack.push((c, 0));
            } else {
                order.push(node);
                stack.pop();
            }
        }
    }
    debug_assert_eq!(order.len(), m);
    order
}

/// Sorted union of two sorted lists, without `except`.
fn union_without(a: &[usize], b: &[usize], except: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let v = match (a.get(i), b.get(j)) {
            (Some(&x), Some(&y)) if x == y => {
                i += 1;
                j += 1;
                x
            }
            (Some(&x), Some(&y)) if x < y => {
                i += 1;
                x
            }
            (Some(_), Some(&y)) => {
                j += 1;
                y
            }
            (Some(&x), None) => {
                i += 1;
                x
            }
            (None, Some(&y)) => {
                j += 1;
                y
            }
            (None, None) => unreachable!(),
        };
        if v != except {
            out.push(v);
        }
    }
    out
}

/// The panel partition of `plan` over `pattern`.
pub fn supernodes(pattern: &Pattern, plan: &Plan) -> Supernodes {
    let btf = &plan.btf;
    let n = pattern.len();
    let mut col_pos = vec![0usize; n];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        col_pos[j] = k;
    }
    let mut row_of = vec![0usize; n];
    let mut col_of = vec![0usize; n];
    let mut bounds = vec![0usize];
    let mut blocks = vec![0usize];
    for b in 0..btf.n_blocks() {
        let range = btf.block(b);
        let (lo, m) = (range.start, range.len());
        let order = &plan.orders[b];
        let pivots = &plan.pivots[b];
        // Step of each local column and row under the plan's order.
        let mut cstep = vec![0usize; m];
        let mut rstep = vec![0usize; m];
        for k in 0..m {
            cstep[order[k]] = k;
            rstep[pivots[k]] = k;
        }
        // The block's own entries in step coordinates.
        let mut entries: Vec<(usize, usize)> = Vec::new();
        for lr in 0..m {
            let i = btf.row_perm[lo + lr];
            for &j in &pattern[i] {
                let c = col_pos[j];
                if range.contains(&c) {
                    entries.push((rstep[lr], cstep[c - lo]));
                }
            }
        }
        // The elimination forest of that order, postordered: the steps
        // renumbered so that a column's parent follows its subtree.
        let (colstruct, _) = symbolic(m, &entries);
        let post = postorder(&colstruct);
        let mut new_of = vec![0usize; m];
        for (k, &old) in post.iter().enumerate() {
            new_of[old] = k;
        }
        for (k, &old) in post.iter().enumerate() {
            row_of[lo + k] = btf.row_perm[lo + pivots[old]];
            col_of[lo + k] = btf.col_perm[lo + order[old]];
        }
        let entries: Vec<(usize, usize)> = entries
            .iter()
            .map(|&(r, c)| (new_of[r], new_of[c]))
            .collect();
        let (colstruct, _) = symbolic(m, &entries);
        // Panels along the postorder: step k + 1 joins the panel ending at
        // k when it is the parent of the panel's rows (the panel's L
        // structure reaches it) and the explicit zeros of the merged panel's
        // rows stay within the allowance for its width.
        let mut start = 0usize;
        // The panel's rows below it (union of its columns' structures,
        // panel members excluded) and its nonzeros there.
        let mut rows_p: Vec<usize> = Vec::new();
        let mut nz_p = 0usize;
        for k in 0..m {
            if k == start {
                rows_p = colstruct[k].clone();
                nz_p = rows_p.len();
            }
            let last = k + 1 == m;
            let joins = !last && {
                let next = k + 1;
                let width = next - start + 1;
                let connected = rows_p.first() == Some(&next);
                connected && width <= MAX_WIDTH && {
                    let rows_new = union_without(&rows_p, &colstruct[next], next);
                    // Columns of the panel that reach `next` leave the rows
                    // below (they enter the diagonal block).
                    let reach = (start..=k)
                        .filter(|&c| colstruct[c].binary_search(&next).is_ok())
                        .count();
                    let nz_new = nz_p - reach + colstruct[next].len();
                    let cells = width * rows_new.len();
                    let zeros = cells - nz_new.min(cells);
                    let ok = cells == 0 || (zeros as f64) <= allowed_zeros(width) * cells as f64;
                    if ok {
                        rows_p = rows_new;
                        nz_p = nz_new;
                    }
                    ok
                }
            };
            if !joins {
                bounds.push(lo + k + 1);
                start = k + 1;
            }
        }
        blocks.push(bounds.len() - 1);
    }
    Supernodes {
        row_of,
        col_of,
        bounds,
        blocks,
    }
}

/// Solve `A x = b` along `plan` over the panels `sn` (from
/// [`supernodes`] of the same pattern and plan): the rows are the scalar
/// entries in original coordinates, `b` and the result too.
pub fn solve_supernodal_planned<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &[Vec<(usize, N)>],
    plan: &Plan,
    sn: &Supernodes,
    b: &[N],
) -> Solved<N> {
    let n = m.len();
    assert_eq!(b.len(), n);
    assert_eq!(sn.row_of.len(), n, "the partition of this system");
    let np = sn.n_panels();
    let mut step_of_row = vec![0usize; n];
    let mut step_of_col = vec![0usize; n];
    for s in 0..n {
        step_of_row[sn.row_of[s]] = s;
        step_of_col[sn.col_of[s]] = s;
    }
    let mut panel_of = vec![0usize; n];
    for p in 0..np {
        for s in sn.bounds[p]..sn.bounds[p + 1] {
            panel_of[s] = p;
        }
    }
    let sizes: Vec<usize> = sn.widths();
    // Entries per block position, in panel-local coordinates.
    let mut entries: HashMap<(usize, usize), Vec<(usize, usize, N)>> = HashMap::default();
    for (i, row) in m.iter().enumerate() {
        let rs = step_of_row[i];
        let (pr, r) = (panel_of[rs], rs - sn.bounds[panel_of[rs]]);
        for &(j, e) in row {
            let cs = step_of_col[j];
            let (pc, c) = (panel_of[cs], cs - sn.bounds[panel_of[cs]]);
            entries.entry((pr, pc)).or_default().push((r, c, e));
        }
    }
    let zero = N::zero(g);
    let mut rows: BlockRows<N> = vec![Vec::new(); np];
    let mut keys: Vec<(usize, usize)> = entries.keys().copied().collect();
    keys.sort_unstable();
    for (pr, pc) in keys {
        let list = &entries[&(pr, pc)];
        let (sr, sc) = (sizes[pr], sizes[pc]);
        let diagonal = sr == sc && list.iter().all(|&(r, c, _)| r == c);
        let block = if diagonal {
            let mut d = vec![zero; sr];
            for &(r, _, e) in list {
                d[r] = e;
            }
            Block::Diag(d)
        } else {
            let mut v = vec![zero; sr * sc];
            for &(r, c, e) in list {
                v[r * sc + c] = e;
            }
            Block::Dense(v)
        };
        rows[pr].push((pc, block));
    }
    // The panels in elimination order: the block triangular form's blocks
    // over the panels, identity permutations, identity orders.
    let bplan = Plan {
        btf: Btf {
            row_perm: (0..np).collect(),
            col_perm: (0..np).collect(),
            blocks: sn.blocks.clone(),
        },
        orders: (0..sn.blocks.len() - 1)
            .map(|b| (0..sn.blocks[b + 1] - sn.blocks[b]).collect())
            .collect(),
        pivots: (0..sn.blocks.len() - 1)
            .map(|b| (0..sn.blocks[b + 1] - sn.blocks[b]).collect())
            .collect(),
        cost: plan.cost,
    };
    let rhs: Vec<N> = (0..n).map(|s| b[sn.row_of[s]]).collect();
    let solved = solve_block_planned_sizes(g, &rows, &sizes, &bplan, &rhs);
    let mut x = vec![zero; n];
    for s in 0..n {
        x[sn.col_of[s]] = solved.x[s];
    }
    Solved {
        x,
        pivots_ok: solved.pivots_ok,
        fill: solved.fill,
    }
}
