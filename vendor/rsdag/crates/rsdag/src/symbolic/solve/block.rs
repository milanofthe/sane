//! The solve of a block-sparse system as graph ops: a sparse pattern of
//! blocks (a harmonic-balance Jacobian, a multi-port coupling, the panels
//! of a supernodal factorization), eliminated block by block along the
//! block pattern's [`Plan`]. Block row `i` and block column `i` have
//! `sizes[i]` scalars; a block is dense or diagonal ([`Block`]). A dense
//! pivot block is inverted through the dense [`Graph::solve_dense`] kernel
//! (its right-hand sides fuse into one factorization), a diagonal one by
//! reciprocals, and every block update is a set of dot products over whole
//! block rows and columns, which the tape compiler fuses into `Gemm`
//! kernels. The scalar is a [`Num`]: real, or complex as pairs of real
//! expressions.
//!
//! The pivot blocks are the block pattern's transversal; the dense kernel
//! pivots within a block. Across blocks the pivot rows are static, and
//! guarded as in the scalar solve: a pivot block whose column has entries
//! below it checks that its own largest entry in that column is at least
//! [`super::PIVOT_TOLERANCE`] times the largest below ([`Solved::pivots_ok`]).

use super::num::Num;
use super::{Plan, Solved};
use crate::field::Field;
use crate::graph::Graph;
use crate::node::{CmpOp, ExprId, ReduceOp};
use rustc_hash::FxHashMap as HashMap;

/// One block of a block-sparse matrix: dense (`rows * cols` entries,
/// row-major) or diagonal (`rows` entries, a square block).
#[derive(Clone, Debug)]
pub enum Block<N = ExprId> {
    Dense(Vec<N>),
    Diag(Vec<N>),
}

/// A block-sparse matrix: one list per block row of `(block column, block)`.
pub type BlockRows<N = ExprId> = Vec<Vec<(usize, Block<N>)>>;

/// Blocks at least this size take their updates one product at a time,
/// each a kernel folding into the previous result in place; smaller
/// blocks take every update in one dot per entry.
const CHAIN_MIN: usize = 8;

/// The block pattern of `m` (the input of [`super::plan`]).
pub fn block_pattern<N>(m: &BlockRows<N>) -> super::Pattern {
    m.iter()
        .map(|row| row.iter().map(|&(j, _)| j).collect())
        .collect()
}

/// A block with its shape and storage order. A dense block is row-major
/// (`cm` false) or column-major: the blocks of `U` (right of a pivot) and
/// the pivot inverses are kept column-major, so that a product's right
/// factor, read by columns, is a consecutive run the kernel takes in
/// place; `L` blocks and pivots are row-major for the same reason.
#[derive(Clone, Debug)]
struct Blk<N> {
    r: usize,
    c: usize,
    d: Block<N>,
    cm: bool,
}

impl<N: Num> Blk<N> {
    fn from(block: &Block<N>, r: usize, c: usize) -> Self {
        match block {
            Block::Dense(v) => assert_eq!(v.len(), r * c, "a dense block of {r} by {c}"),
            Block::Diag(v) => {
                assert_eq!(r, c, "a diagonal block is square");
                assert_eq!(v.len(), r, "a diagonal block of {r}");
            }
        }
        Blk {
            r,
            c,
            d: block.clone(),
            cm: false,
        }
    }

    fn is_diag(&self) -> bool {
        matches!(self.d, Block::Diag(_))
    }

    #[inline]
    fn index(&self, r: usize, c: usize) -> usize {
        if self.cm {
            c * self.r + r
        } else {
            r * self.c + c
        }
    }

    /// Entry `(r, c)`, if structurally present.
    fn at(&self, r: usize, c: usize) -> Option<N> {
        match &self.d {
            Block::Dense(v) => Some(v[self.index(r, c)]),
            Block::Diag(v) => (r == c).then(|| v[r]),
        }
    }

    /// Row `r` as a list (a diagonal block's row with its zeros).
    fn row(&self, r: usize, zero: N) -> Vec<N> {
        match &self.d {
            Block::Dense(v) => (0..self.c).map(|c| v[self.index(r, c)]).collect(),
            Block::Diag(d) => (0..self.c)
                .map(|c| if c == r { d[r] } else { zero })
                .collect(),
        }
    }

    /// Column `c` as a list.
    fn col(&self, c: usize, zero: N) -> Vec<N> {
        match &self.d {
            Block::Dense(v) => (0..self.r).map(|r| v[self.index(r, c)]).collect(),
            Block::Diag(d) => (0..self.r)
                .map(|r| if r == c { d[c] } else { zero })
                .collect(),
        }
    }

    /// Entry `(r, c)` of `self * u` as a dot. A diagonal block against a
    /// dense one is taken dense (its zeros in the dot): the dots then fuse
    /// into one kernel, where the products of the diagonal alone would be
    /// scalar ops, slower than the kernel. With `cm`, the dot's lists are
    /// swapped (the same fold, term by term), so the kernel the dots fuse
    /// into writes the product column-major.
    fn product_entry<K: Field>(
        &self,
        g: &mut Graph<K>,
        u: &Blk<N>,
        r: usize,
        c: usize,
        cm: bool,
        zero: N,
    ) -> N {
        debug_assert_eq!(self.c, u.r, "conformable blocks");
        if let (Block::Diag(l), Block::Diag(ud)) = (&self.d, &u.d) {
            return if r == c { N::mul(g, l[r], ud[r]) } else { zero };
        }
        let (ls, us) = (self.row(r, zero), u.col(c, zero));
        if cm {
            N::dot(g, us, ls)
        } else {
            N::dot(g, ls, us)
        }
    }

    /// The terms of entry `r` of `self * y`, appended to `ls, ys`.
    fn apply_terms(&self, y: &[N], r: usize, ls: &mut Vec<N>, ys: &mut Vec<N>) {
        debug_assert_eq!(y.len(), self.c);
        match &self.d {
            Block::Dense(_) => {
                let zero = y[0];
                ls.extend(self.row(r, zero));
                ys.extend_from_slice(y);
            }
            Block::Diag(l) => {
                ls.push(l[r]);
                ys.push(y[r]);
            }
        }
    }

    /// `self * u` as a block: diagonal when both are, else dense in the
    /// order `cm`.
    fn product<K: Field>(&self, g: &mut Graph<K>, u: &Blk<N>, cm: bool) -> Blk<N> {
        if let (Block::Diag(l), Block::Diag(d)) = (&self.d, &u.d) {
            let v = (0..self.r).map(|r| N::mul(g, l[r], d[r])).collect();
            return Blk {
                r: self.r,
                c: u.c,
                d: Block::Diag(v),
                cm: false,
            };
        }
        let zero = N::zero(g);
        let (rows, cols) = (self.r, u.c);
        let mut out = Vec::with_capacity(rows * cols);
        if cm {
            for c in 0..cols {
                for r in 0..rows {
                    out.push(self.product_entry(g, u, r, c, true, zero));
                }
            }
        } else {
            for r in 0..rows {
                for c in 0..cols {
                    out.push(self.product_entry(g, u, r, c, false, zero));
                }
            }
        }
        Blk {
            r: rows,
            c: cols,
            d: Block::Dense(out),
            cm,
        }
    }

    /// The inverse of a square block: reciprocals of a diagonal one, the
    /// dense kernel's solves against the unit vectors of a dense one,
    /// column-major (the solves' layout).
    fn inverse<K: Field>(&self, g: &mut Graph<K>) -> Blk<N> {
        let s = self.r;
        let zero = N::zero(g);
        let (d, cm) = match &self.d {
            Block::Diag(d) => (
                Block::Diag(d.iter().map(|&v| N::recip(g, v)).collect()),
                false,
            ),
            Block::Dense(_) => {
                let a: Vec<N> = (0..s).flat_map(|r| self.row(r, zero)).collect();
                let cols = N::inverse_columns(g, &a, s);
                (Block::Dense(cols.into_iter().flatten().collect()), true)
            }
        };
        Blk { r: s, c: s, d, cm }
    }

    /// The largest size of column `c`'s entries.
    fn column_size<K: Field>(&self, g: &mut Graph<K>, c: usize, sizes: &mut Vec<ExprId>) {
        match &self.d {
            Block::Dense(v) => {
                for r in 0..self.r {
                    sizes.push(N::size(g, v[self.index(r, c)]));
                }
            }
            Block::Diag(d) => sizes.push(N::size(g, d[c])),
        }
    }
}

/// `y - L_1 v_1 - L_2 v_2 - ...`, one product per pair subtracted from the
/// running vector: each product is one kernel whose accumulator is the
/// previous kernel's output, read in place; a concatenated dot over every
/// pair would gather its operands.
fn subtract_products<K: Field, N: Num>(
    g: &mut Graph<K>,
    y: &[N],
    pairs: &[(&Blk<N>, &[N])],
) -> Vec<N> {
    let mut acc: Vec<N> = y.to_vec();
    for (l, v) in pairs {
        acc = (0..acc.len())
            .map(|r| {
                let (mut ls, mut vs) = (Vec::new(), Vec::new());
                l.apply_terms(v, r, &mut ls, &mut vs);
                let d = N::dot(g, ls, vs);
                N::sub(g, acc[r], d)
            })
            .collect();
    }
    acc
}

fn max_of<K: Field>(g: &mut Graph<K>, mut v: Vec<ExprId>) -> ExprId {
    if v.len() == 1 {
        v.pop().unwrap()
    } else {
        g.reduce(ReduceOp::Max, v)
    }
}

/// Solve `A x = rhs` for a block-sparse `A` of `b` by `b` blocks along
/// `plan` (planned over [`block_pattern`]); `rhs` has `b` entries per block
/// row, `x` comes back the same way. `fill` counts blocks.
pub fn solve_block_planned<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &BlockRows<N>,
    b: usize,
    plan: &Plan,
    rhs: &[N],
) -> Solved<N> {
    let sizes = vec![b; m.len()];
    solve_block_planned_sizes(g, m, &sizes, plan, rhs)
}

/// [`solve_block_planned`] with block row and column `i` of `sizes[i]`
/// scalars: block `(i, j)` is `sizes[i]` by `sizes[j]`, the right-hand
/// side and the solution are the blocks' entries back to back.
pub fn solve_block_planned_sizes<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &BlockRows<N>,
    sizes: &[usize],
    plan: &Plan,
    rhs: &[N],
) -> Solved<N> {
    let nb = m.len();
    assert_eq!(sizes.len(), nb, "a size per block row");
    let total: usize = sizes.iter().sum();
    assert_eq!(rhs.len(), total, "the right-hand side over every block row");
    let mut starts = Vec::with_capacity(nb + 1);
    let mut at = 0;
    for &s in sizes {
        starts.push(at);
        at += s;
    }
    starts.push(at);
    let btf = &plan.btf;
    let mut col_pos = vec![0usize; nb];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        col_pos[j] = k;
    }
    // Permuted block rows: (permuted block column, block with its shape).
    let rows: Vec<Vec<(usize, Blk<N>)>> = btf
        .row_perm
        .iter()
        .map(|&i| {
            let mut r: Vec<(usize, Blk<N>)> = m[i]
                .iter()
                .map(|(j, e)| (col_pos[*j], Blk::from(e, sizes[i], sizes[*j])))
                .collect();
            r.sort_by_key(|&(c, _)| c);
            r
        })
        .collect();
    let zero = N::zero(g);
    // Solutions per permuted block column.
    let mut x: Vec<Vec<N>> = (0..nb)
        .map(|k| vec![zero; sizes[btf.col_perm[k]]])
        .collect();
    let mut fill = 0usize;
    let mut guards: Vec<ExprId> = Vec::new();
    for blk in (0..btf.n_blocks()).rev() {
        let range = btf.block(blk);
        let (lo, hi) = (range.start, range.end);
        let nl = hi - lo;
        let mut local: Vec<Vec<(usize, &Blk<N>)>> = Vec::with_capacity(nl);
        let mut local_rhs: Vec<Vec<N>> = Vec::with_capacity(nl);
        let local_sizes: Vec<usize> = (lo..hi).map(|k| sizes[btf.row_perm[k]]).collect();
        for k in lo..hi {
            let mut row = Vec::new();
            let mut couplings: Vec<(&Blk<N>, &Vec<N>)> = Vec::new();
            for (c, e) in &rows[k] {
                if *c >= hi {
                    couplings.push((e, &x[*c]));
                } else if *c >= lo {
                    row.push((c - lo, e));
                }
            }
            local.push(row);
            let i = btf.row_perm[k];
            let bk = &rhs[starts[i]..starts[i + 1]];
            // rhs_k = b_k - sum over solved blocks of A_kc x_c.
            let pairs: Vec<(&Blk<N>, &[N])> = couplings
                .iter()
                .map(|(e, xc)| (*e, xc.as_slice()))
                .collect();
            let r = subtract_products(g, bk, &pairs);
            local_rhs.push(r);
        }
        let (sol, gs, f) = eliminate_blocks(g, &local, &local_sizes, &plan.orders[blk], &local_rhs);
        fill += f;
        guards.extend(gs);
        for (k, v) in (lo..hi).zip(sol) {
            x[k] = v;
        }
    }
    let mut out = vec![zero; total];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        out[starts[j]..starts[j + 1]].copy_from_slice(&x[k]);
    }
    let pivots_ok = match guards.len() {
        0 => g.one(),
        1 => guards[0],
        _ => g.reduce(ReduceOp::Min, guards),
    };
    Solved {
        x: out,
        pivots_ok,
        fill,
    }
}

/// One irreducible block of the block pattern: Crout elimination over
/// blocks in `order`, the right-hand side carried along. Returns the
/// solution per block, the guards, and the fill in blocks.
fn eliminate_blocks<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &[Vec<(usize, &Blk<N>)>],
    sizes: &[usize],
    order: &[usize],
    rhs: &[Vec<N>],
) -> (Vec<Vec<N>>, Vec<ExprId>, usize) {
    let n = m.len();
    assert_eq!(order.len(), n);
    let mut pos = vec![0usize; n];
    for (k, &j) in order.iter().enumerate() {
        pos[j] = k;
    }
    // Both rows and columns permuted by the order: the pivot blocks are the
    // diagonal of the permuted pattern (the transversal). Sizes follow.
    let psize: Vec<usize> = (0..n).map(|k| sizes[order[k]]).collect();
    let mut orig: HashMap<(usize, usize), Blk<N>> = HashMap::default();
    let mut in_row: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_col: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut y: Vec<Vec<N>> = vec![Vec::new(); n];
    for (i, row) in m.iter().enumerate() {
        for &(j, e) in row {
            let (r, c) = (pos[i], pos[j]);
            if orig.insert((r, c), e.clone()).is_none() {
                in_row[r].push(c);
                in_col[c].push(r);
            }
        }
        y[pos[i]] = rhs[i].clone();
    }
    // Pending updates per block position: the (L, U) block pairs whose
    // product is subtracted, in elimination order; per right-hand side
    // row the (L, y) pairs.
    let mut pending: HashMap<(usize, usize), Vec<(Blk<N>, Blk<N>)>> = HashMap::default();
    let mut pending_y: Vec<Vec<(Blk<N>, Vec<N>)>> = vec![Vec::new(); n];
    // The block at `at` after its pending updates: diagonal when the base
    // and every update are, dense otherwise.
    fn finalize<K: Field, N: Num>(
        g: &mut Graph<K>,
        orig: &mut HashMap<(usize, usize), Blk<N>>,
        pending: &mut HashMap<(usize, usize), Vec<(Blk<N>, Blk<N>)>>,
        at: (usize, usize),
        shape: (usize, usize),
    ) -> Blk<N> {
        let base = orig.get(&at).cloned();
        let Some(updates) = pending.remove(&at) else {
            return base.expect("an occupied block position has entries");
        };
        let (r, c) = shape;
        // A block right of its pivot (of `U`) is kept column-major.
        let cm = at.1 > at.0;
        let zero = N::zero(g);
        let diag = r == c
            && base.as_ref().is_none_or(|k| k.is_diag())
            && updates.iter().all(|(l, u)| l.is_diag() && u.is_diag());
        let out = if diag {
            let d = (0..r)
                .map(|q| {
                    let mut acc = base.as_ref().and_then(|k| k.at(q, q));
                    for (l, u) in &updates {
                        let d = l.product_entry(g, u, q, q, false, zero);
                        acc = Some(match acc {
                            Some(o) => N::sub(g, o, d),
                            None => N::neg(g, d),
                        });
                    }
                    acc.expect("an update")
                })
                .collect();
            Blk {
                r,
                c,
                d: Block::Diag(d),
                cm: false,
            }
        } else {
            // One product per update, subtracted from the running block:
            // each is one kernel with the previous result as its
            // accumulator, read in place. Entries in the block's order.
            let order: Vec<(usize, usize)> = if cm {
                (0..c)
                    .flat_map(|cc| (0..r).map(move |rr| (rr, cc)))
                    .collect()
            } else {
                (0..r)
                    .flat_map(|rr| (0..c).map(move |cc| (rr, cc)))
                    .collect()
            };
            let mut acc: Option<Vec<N>> = base.as_ref().map(|k| {
                order
                    .iter()
                    .map(|&(rr, cc)| k.at(rr, cc).unwrap_or(zero))
                    .collect()
            });
            if r.min(c) >= CHAIN_MIN {
                for (l, u) in &updates {
                    let out: Vec<N> = order
                        .iter()
                        .enumerate()
                        .map(|(q, &(rr, cc))| {
                            let d = l.product_entry(g, u, rr, cc, cm, zero);
                            match &acc {
                                Some(a) => N::sub(g, a[q], d),
                                None => N::neg(g, d),
                            }
                        })
                        .collect();
                    acc = Some(out);
                }
            } else {
                // Small blocks: every update in one dot per entry.
                let out: Vec<N> = order
                    .iter()
                    .enumerate()
                    .map(|(q, &(rr, cc))| {
                        let (mut ls, mut us) = (Vec::new(), Vec::new());
                        for (l, u) in &updates {
                            ls.extend(l.row(rr, zero));
                            us.extend(u.col(cc, zero));
                        }
                        let d = if cm {
                            N::dot(g, us, ls)
                        } else {
                            N::dot(g, ls, us)
                        };
                        match &acc {
                            Some(a) => N::sub(g, a[q], d),
                            None => N::neg(g, d),
                        }
                    })
                    .collect();
                acc = Some(out);
            }
            Blk {
                r,
                c,
                d: Block::Dense(acc.expect("an update")),
                cm,
            }
        };
        orig.insert(at, out.clone());
        out
    }
    let mut fill = 0usize;
    let mut guards: Vec<ExprId> = Vec::new();
    let mut upper: Vec<Vec<(usize, Blk<N>)>> = vec![Vec::new(); n];
    let mut inv: Vec<Option<Blk<N>>> = vec![None; n];
    let tol = N::tolerance(g);
    for k in 0..n {
        assert!(
            orig.contains_key(&(k, k)) || pending.contains_key(&(k, k)),
            "a structurally nonzero pivot block at {k}"
        );
        let s = psize[k];
        let pivot = finalize(g, &mut orig, &mut pending, (k, k), (s, s));
        // The right-hand side row of this step, after its updates.
        let yk = {
            let updates = std::mem::take(&mut pending_y[k]);
            let base = std::mem::take(&mut y[k]);
            let pairs: Vec<(&Blk<N>, &[N])> =
                updates.iter().map(|(l, v)| (l, v.as_slice())).collect();
            subtract_products(g, &base, &pairs)
        };
        let mut row_k: Vec<usize> = in_row[k].iter().copied().filter(|&c| c > k).collect();
        let mut col_k: Vec<usize> = in_col[k].iter().copied().filter(|&r| r > k).collect();
        row_k.sort_unstable();
        row_k.dedup();
        col_k.sort_unstable();
        col_k.dedup();
        let us: Vec<Blk<N>> = row_k
            .iter()
            .map(|&j| finalize(g, &mut orig, &mut pending, (k, j), (s, psize[j])))
            .collect();
        for (&j, u) in row_k.iter().zip(&us) {
            upper[k].push((j, u.clone()));
        }
        let ws: Vec<Blk<N>> = col_k
            .iter()
            .map(|&i| finalize(g, &mut orig, &mut pending, (i, k), (psize[i], s)))
            .collect();
        if !ws.is_empty() {
            // The guard: in every column of the panel, the pivot block's
            // largest entry against the largest below it.
            for c in 0..s {
                let mut ps = Vec::new();
                pivot.column_size(g, c, &mut ps);
                let mut below = Vec::new();
                for w in &ws {
                    w.column_size(g, c, &mut below);
                }
                let pm = max_of(g, ps);
                let bm = max_of(g, below);
                let bound = g.mul(tol, bm);
                guards.push(g.cmp(CmpOp::Ge, pm, bound));
            }
        }
        let pinv = pivot.inverse(g);
        // L_ik = W A_kk^-1, row-major.
        let ls: Vec<Blk<N>> = ws.iter().map(|w| w.product(g, &pinv, false)).collect();
        for (&i, l) in col_k.iter().zip(&ls) {
            for (&j, u) in row_k.iter().zip(&us) {
                let entry = pending.entry((i, j)).or_default();
                if entry.is_empty() && !orig.contains_key(&(i, j)) {
                    fill += 1;
                    in_row[i].push(j);
                    in_col[j].push(i);
                }
                entry.push((l.clone(), u.clone()));
            }
            pending_y[i].push((l.clone(), yk.clone()));
        }
        y[k] = yk;
        inv[k] = Some(pinv);
    }
    // Back substitution over blocks: x_i = A_ii^-1 (y_i - sum U_ij x_j).
    let mut x: Vec<Vec<N>> = vec![Vec::new(); n];
    for i in (0..n).rev() {
        let pairs: Vec<(&Blk<N>, &[N])> = upper[i]
            .iter()
            .map(|(j, u)| (u, x[*j].as_slice()))
            .collect();
        let r = subtract_products(g, &y[i], &pairs);
        let pinv = inv[i].as_ref().unwrap();
        let s = psize[i];
        let zero = N::zero(g);
        x[i] = match &pinv.d {
            Block::Diag(d) => (0..s).map(|q| N::mul(g, d[q], r[q])).collect(),
            Block::Dense(_) => (0..s)
                .map(|q| N::dot(g, pinv.row(q, zero), r.clone()))
                .collect(),
        };
    }
    let mut out = vec![Vec::new(); n];
    for (k, &j) in order.iter().enumerate() {
        out[j] = std::mem::take(&mut x[k]);
    }
    (out, guards, fill)
}
