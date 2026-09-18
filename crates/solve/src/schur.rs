//! Linear/nonlinear Schur-complement partitioning: the constant linear block is
//! factored once per Newton solve and reused; each iteration solves only a small
//! dense system over the device-touched unknowns. See [`Partition`].

use crate::{sparse, CompiledDc};

/// Partition of the unknowns into a large *linear* block `L` (whose Jacobian
/// contribution is constant across Newton iterations) and a small *nonlinear*
/// block `V` (unknowns touched by a device, whose entries vary with `x`). Lets
/// the constant `L`-block be factored once per Newton solve and reused, with
/// only a small `|V| x |V|` Schur-complement system solved each iteration --
/// the key to large mostly-linear systems (e.g. RC parasitic networks with a
/// few embedded devices).
pub(crate) struct Partition {
    /// Per jx entry: does it vary with `x` (a nonlinear/device entry)?
    pub(crate) var_entry: Vec<bool>,
    /// Unknown indices in the linear block, in original order.
    pub(crate) lin: Vec<usize>,
    /// Unknown indices in the nonlinear block, in original order.
    pub(crate) nonlin: Vec<usize>,
}

/// Cached factorization of the constant linear block for a *fixed* `gmin`,
/// built once per Newton solve and reused across its iterations.
pub(crate) struct LinCache {
    nl: usize,
    nv: usize,
    /// Factorization of `A = J[L,L] + gmin*I_L` (constant).
    lu_a: sparse::TripletLu,
    /// Constant coupling block `B = J[L,V]` as `(l_row, v_col, val)`.
    b: Vec<(usize, usize, f64)>,
    /// Constant coupling block `C = J[V,L]` as `(v_row, l_col, val)`.
    c: Vec<(usize, usize, f64)>,
    /// Constant part of the Schur complement: `(D_const + gmin*I_V) - C A^{-1} B`.
    /// `nv` by `nv`, row-major.
    s_base: Vec<f64>,
    /// Variable `V x V` entries: `(jx index, v_row, v_col)`, added each iteration.
    var: Vec<(usize, usize, usize)>,
}

impl CompiledDc {
    /// `(|L|, |V|)`: sizes of the linear and nonlinear blocks, or `None` if the
    /// system is not partitioned (used for diagnostics / tests).
    pub fn partition_sizes(&self) -> Option<(usize, usize)> {
        self.partition
            .as_ref()
            .map(|p| (p.lin.len(), p.nonlin.len()))
    }

    /// Build the cached factorization of the constant linear block `A = J[L,L] +
    /// gmin*I` (valid for this fixed `gmin`), the constant coupling blocks `B`,
    /// `C`, and the constant part of the Schur complement `S0 = (D_const +
    /// gmin*I_V) - C A^{-1} B`. `jac` are the current (structurally complete)
    /// `dF/dx` nonzeros; only the constant entries are read here. Returns `None`
    /// if `A` is singular (caller falls back to the plain solve).
    pub(crate) fn build_lin_cache(
        &self,
        part: &Partition,
        jac: &[f64],
        gmin: f64,
    ) -> Option<LinCache> {
        let n = self.n;
        let (nl, nv) = (part.lin.len(), part.nonlin.len());
        let mut l_of = vec![usize::MAX; n];
        let mut v_of = vec![usize::MAX; n];
        for (loc, &orig) in part.lin.iter().enumerate() {
            l_of[orig] = loc;
        }
        for (loc, &orig) in part.nonlin.iter().enumerate() {
            v_of[orig] = loc;
        }

        let (mut a_r, mut a_c, mut a_v) = (Vec::new(), Vec::new(), Vec::new());
        let mut b: Vec<(usize, usize, f64)> = Vec::new();
        let mut c: Vec<(usize, usize, f64)> = Vec::new();
        let mut d_const = vec![0.0f64; nv * nv];
        let mut var: Vec<(usize, usize, usize)> = Vec::new();
        for k in 0..jac.len() {
            let (r, col, val) = (self.jx_rows[k], self.jx_cols[k], jac[k]);
            if part.var_entry[k] {
                // Variable entries are confined to the V x V block.
                var.push((k, v_of[r], v_of[col]));
                continue;
            }
            match (l_of[r] != usize::MAX, l_of[col] != usize::MAX) {
                (true, true) => {
                    a_r.push(l_of[r]);
                    a_c.push(l_of[col]);
                    a_v.push(val);
                }
                (true, false) => b.push((l_of[r], v_of[col], val)),
                (false, true) => c.push((v_of[r], l_of[col], val)),
                (false, false) => d_const[v_of[r] * nv + v_of[col]] += val,
            }
        }
        // gmin on every diagonal (constant for this newton call).
        for i in 0..nl {
            a_r.push(i);
            a_c.push(i);
            a_v.push(gmin);
        }
        for j in 0..nv {
            d_const[j * nv + j] += gmin;
        }

        let lu_a = sparse::factor_triplets_both(nl, &a_r, &a_c, &a_v)?;

        // M = C A^{-1} B, column by column (no |L| x |V| dense intermediate).
        let mut s_base = d_const;
        let mut bcol = vec![0.0; nl];
        for j in 0..nv {
            for v in bcol.iter_mut() {
                *v = 0.0;
            }
            for &(lr, vc, val) in &b {
                if vc == j {
                    bcol[lr] += val;
                }
            }
            let t = lu_a.solve(&bcol)?; // A^{-1} B[:,j]
            for &(vr, lc, val) in &c {
                s_base[vr * nv + j] -= val * t[lc]; // S0 = D_const - M
            }
        }

        Some(LinCache {
            nl,
            nv,
            lu_a,
            b,
            c,
            s_base,
            var,
        })
    }

    /// Solve `(J + gmin*I) dx = rhs` via the Schur complement using the cached
    /// linear-block factorization: only a small `|V| x |V|` dense system plus two
    /// reuses of the cached `A` factorization, instead of refactorizing the whole
    /// matrix. `jac` are this iteration's `dF/dx` nonzeros (only the variable
    /// ones are read). Returns `dx` (length `n`).
    pub(crate) fn solve_partitioned(
        &self,
        cache: &LinCache,
        part: &Partition,
        jac: &[f64],
        rhs: &[f64],
    ) -> Option<Vec<f64>> {
        let (nl, nv) = (cache.nl, cache.nv);

        // r partitioned. u = A^{-1} r_L.
        let mut r_l = vec![0.0; nl];
        for (loc, &orig) in part.lin.iter().enumerate() {
            r_l[loc] = rhs[orig];
        }
        let u = cache.lu_a.solve(&r_l)?;

        // rhs_S = r_V - C u.
        let mut rhs_s = vec![0.0f64; nv];
        for (loc, &orig) in part.nonlin.iter().enumerate() {
            rhs_s[loc] = rhs[orig];
        }
        for &(vr, lc, val) in &cache.c {
            rhs_s[vr] -= val * u[lc];
        }

        // S = S0 + variable V x V entries; dx_V = S^{-1} rhs_S.
        let mut s = cache.s_base.clone();
        for &(k, vr, vc) in &cache.var {
            s[vr * nv + vc] += jac[k];
        }
        let mut dx_v = vec![0.0f64; nv];
        rsdag::semantics::solve(&s, &rhs_s, nv, &mut dx_v);
        if dx_v.iter().any(|v| !v.is_finite()) {
            return None;
        }

        // dx_L = u - A^{-1} (B dx_V).
        let mut bdv = vec![0.0; nl];
        for &(lr, vc, val) in &cache.b {
            bdv[lr] += val * dx_v[vc];
        }
        let w = cache.lu_a.solve(&bdv)?;

        let mut dx = vec![0.0; self.n];
        for (loc, &orig) in part.lin.iter().enumerate() {
            dx[orig] = u[loc] - w[loc];
        }
        for (loc, &orig) in part.nonlin.iter().enumerate() {
            dx[orig] = dx_v[loc];
        }
        // A singular Schur complement yields Inf/NaN here; signal failure so the
        // caller falls back to the full sparse LU rather than stepping with a
        // garbage update.
        if dx.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(dx)
    }
}
