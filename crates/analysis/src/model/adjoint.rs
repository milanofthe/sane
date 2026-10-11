//! The adjoint of a small-signal quantity, one way for all of them.
//!
//! The AC response, the noise, a pole and a zero each differentiate by every
//! parameter as a scalar functional of the operating point's Jacobians:
//! `Phi = l^T (a_G G + a_C C) r + a_b l^T b`, with `l`, `r` an adjoint and a
//! solution of the analysis and `b = dI/d(input)` the input coupling (plus,
//! for the noise, the sources' levels and injections). Its partial gradient
//! by the states and the parameters is one evaluation of a compiled pull
//! ([`sane_solve::CompiledDc::ensure_pull`]) per real and imaginary part,
//! the entries' weights the products of `l` and `r`; the operating point's
//! shift adds `-mu^T dI/dp` with `G^T mu = dPhi/dx`, one transpose solve on
//! the DC Jacobian factored once per call.

use std::sync::Arc;

use num_complex::Complex64;
use rsdag::ExprId;
use sane_solve::{TripletLu, Triplets};

use crate::model::{Model, ModelError};

/// The entries of a pull over the pencil: the positions of `G`'s and `C`'s
/// entries, then the input coupling's rows when it has one, then entries of
/// its own (the noise's levels and injections).
pub(crate) struct PullLayout {
    pub key: String,
    pub g_pos: Vec<(usize, usize)>,
    pub c_pos: Vec<(usize, usize)>,
    /// `b`'s entries, one per row, when the pull has an input.
    pub nb: usize,
    pub extra: usize,
}

impl PullLayout {
    fn len(&self) -> usize {
        self.g_pos.len() + self.c_pos.len() + self.nb + self.extra
    }
}

/// The weights of `Phi = l^T (a_G G + a_C C) r + a_b l^T b + sum_k e_k extra_k`
/// over a [`PullLayout`]'s entries, complex.
pub(crate) struct Bilinear<'a> {
    pub l: &'a [Complex64],
    pub r: &'a [Complex64],
    pub a_g: Complex64,
    pub a_c: Complex64,
    pub a_b: Complex64,
    pub extra: &'a [Complex64],
}

impl Bilinear<'_> {
    fn weights(&self, layout: &PullLayout) -> Vec<Complex64> {
        let (l, r) = (self.l, self.r);
        let mut w = Vec::with_capacity(layout.len());
        w.extend((layout.g_pos.iter()).map(|&(i, j)| self.a_g * l[i] * r[j]));
        w.extend((layout.c_pos.iter()).map(|&(i, j)| self.a_c * l[i] * r[j]));
        w.extend((0..layout.nb).map(|i| self.a_b * l[i]));
        w.extend((0..layout.extra).map(|k| self.extra.get(k).copied().unwrap_or_default()));
        w
    }
}

/// An operating point's adjoint context: its Jacobian `G` (with the gmin of
/// the DC solve) factored once, and `dI/dp`.
pub(crate) struct AdjointPoint {
    pub x: Vec<f64>,
    pub p: Vec<f64>,
    pub n: usize,
    dc: TripletLu<Complex64>,
    pj: Triplets,
    pub pnames: Vec<String>,
}

impl AdjointPoint {
    /// `G^{-T} b` on the DC Jacobian.
    pub fn solve_dc_transpose(&self, b: &[Complex64]) -> Result<Vec<Complex64>, ModelError> {
        (self.dc.solve_transpose(b))
            .ok_or_else(|| ModelError::Numeric("singular DC Jacobian".into()))
    }
}

impl Model {
    /// The adjoint context at `(x, p)` (see [`AdjointPoint`]).
    pub(crate) fn adjoint_point(&self, x: &[f64], p: &[f64]) -> Result<AdjointPoint, ModelError> {
        self.inner.bound(p)?;
        let g = self.cdc().system_triplets_dc(x, p);
        let gv: Vec<Complex64> = g.vals.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let dc = sane_solve::factor_triplets(self.dim(), &g.rows, &g.cols, &gv)
            .ok_or_else(|| ModelError::Numeric("singular DC Jacobian".into()))?;
        let pnames = self.cdc().param_names().to_vec();
        let (pj, _) = self.cdc().jacobian_p_sparse(x, p, 0.0);
        Ok(AdjointPoint {
            x: x.to_vec(),
            p: p.to_vec(),
            n: self.dim(),
            dc,
            pj,
            pnames,
        })
    }

    /// The pull over the pencil `G`, `C` (and `input`'s coupling `b`, and
    /// the expressions `extra` makes, under `tag`), built on first use.
    pub(crate) fn pencil_pull(
        &self,
        input: Option<&str>,
        tag: Option<(&str, &dyn Fn(&mut sane_core::Graph) -> Vec<ExprId>)>,
    ) -> Result<Arc<PullLayout>, ModelError> {
        let key = format!(
            "pencil|{}|{}",
            input.unwrap_or(""),
            tag.as_ref().map_or("", |t| t.0)
        );
        if let Some(l) = self.pull_layouts().lock().unwrap().get(&key) {
            return Ok(l.clone());
        }
        let arc = self.context_arc();
        let mut cg = arc.lock().unwrap();
        let c = &mut *cg;
        let dae = self.dae();
        let ((gr, gc, ge), (cr, cc, ce)) = dae.jacobian_iq_coo(c);
        let mut entries: Vec<ExprId> = ge.iter().chain(&ce).copied().collect();
        let nb = match input {
            Some(input) => {
                let syms = self.inner.drive(input)?;
                let rest = dae.at_rest(c).0.clone();
                entries.extend(rest.iter().map(|&r| crate::d_drive(c, r, &syms)));
                rest.len()
            }
            None => 0,
        };
        let extra = match &tag {
            Some((_, make)) => {
                let more = make(c);
                let k = more.len();
                entries.extend(more);
                k
            }
            None => 0,
        };
        drop(cg);
        self.cdc().ensure_pull(&key, |_, _| entries);
        let layout = Arc::new(PullLayout {
            key: key.clone(),
            g_pos: gr.into_iter().zip(gc).collect(),
            c_pos: cr.into_iter().zip(cc).collect(),
            nb,
            extra,
        });
        self.pull_layouts()
            .lock()
            .unwrap()
            .insert(key, layout.clone());
        Ok(layout)
    }

    /// `(dPhi/dx, dPhi/dp)` of the bilinear functional `phi` over `layout`
    /// at `at`, the parameter part partial (the states held).
    pub(crate) fn pull(
        &self,
        at: &AdjointPoint,
        layout: &PullLayout,
        phi: &Bilinear<'_>,
    ) -> Result<(Vec<Complex64>, Vec<Complex64>), ModelError> {
        let w = phi.weights(layout);
        let part = |f: fn(&Complex64) -> f64| -> Result<(Vec<f64>, Vec<f64>), ModelError> {
            let wr: Vec<f64> = w.iter().map(f).collect();
            (self.cdc().pull(&layout.key, &at.x, &at.p, &wr))
                .ok_or_else(|| ModelError::Numeric("pull not built".into()))
        };
        let (xr, pr) = part(|c| c.re)?;
        let (xi, pi) = part(|c| c.im)?;
        let join = |a: Vec<f64>, b: Vec<f64>| -> Vec<Complex64> {
            a.into_iter()
                .zip(b)
                .map(|(re, im)| Complex64::new(re, im))
                .collect()
        };
        Ok((join(xr, xi), join(pr, pi)))
    }

    /// The total derivative by every parameter of a functional whose partial
    /// gradients are `(dx, dp)`: the operating point moves with the
    /// parameter, `dPhi/dp - mu^T dI/dp` with `G^T mu = dPhi/dx`.
    pub(crate) fn total(
        &self,
        at: &AdjointPoint,
        dx: &[Complex64],
        mut dp: Vec<Complex64>,
    ) -> Result<Vec<Complex64>, ModelError> {
        let mu = at.solve_dc_transpose(dx)?;
        for (r, c, &v) in at.pj.iter() {
            dp[c] -= mu[r] * v;
        }
        Ok(dp)
    }
}
