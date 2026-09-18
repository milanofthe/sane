//! The public, physical-domain **rational macromodel** produced by the vector fit — a common
//! pole set with per-entry residues plus constant `D` and proportional `sE` terms — for a
//! `p`-port admittance response `Y(s)`.
//!
//! The engine ([`super::fit_auto`]) fits on a *normalised* imaginary axis `s_n = j·f/fmax` for
//! conditioning; this struct stores the **denormalised, physical** model so that
//! [`RationalModel::eval`] at `s = j·2π·f` reconstructs `Y(f)` in Siemens directly, and the
//! Verilog-A exporter can emit physical-time `ddt` state equations without a hidden scaling.
//!
//! Physical mapping from the normalised fit (`K = 2π·fmax`, `s_n = s/K`):
//! `p_phys = K·p_n`, `R_phys = K·R_n`, `D_phys = D_n`, `E_phys = E_n/K`.

use std::f64::consts::TAU;

use super::model::VfModel;
use super::C;

/// A fitted `p`-port rational admittance macromodel `Y(s) = D + sE + Σ_k R_k/(s − p_k)`, with a
/// **common pole set** shared by all `p²` matrix entries, in **physical** units (`s = j·2π·f`,
/// `Y` in Siemens, poles in rad/s).
///
/// Poles are stored split into strictly-real poles and complex-conjugate *pair representatives*
/// (`Im > 0`; the conjugate is implied) — the form both the real state-space realisation
/// (passivity) and the Verilog-A 2nd-order sections consume. Matrix entries are flattened
/// row-major: entry `e = i·p + j` is `Y_ij` (observed port `i`, driven port `j`).
#[derive(Clone, Debug)]
pub struct RationalModel {
    /// Number of ports `p`.
    pub n_ports: usize,
    /// Strictly-real poles [rad/s] (negative for a stable model).
    pub real_poles: Vec<f64>,
    /// Complex-pole pair representatives [rad/s] (`Re < 0`, `Im > 0`); each implies its conjugate.
    pub cpx_poles: Vec<C>,
    /// Real-pole residues, `res_real[e][k]` for entry `e`, pole `real_poles[k]` [S·rad/s].
    pub res_real: Vec<Vec<f64>>,
    /// Complex-pole residues, `res_cpx[e][k]` for entry `e`, pair `cpx_poles[k]` [S·rad/s].
    pub res_cpx: Vec<Vec<C>>,
    /// Constant term `D` per entry [S].
    pub d: Vec<f64>,
    /// Proportional term `E` per entry (`sE` feedthrough) [S·s].
    pub e: Vec<f64>,
    /// √s skin-effect branch coefficient per entry [S/√(rad/s)]; all-zero for a
    /// purely rational model. Captures the Kramers-Kronig-consistent √f surface-
    /// impedance pair that is outside the rational class (issue rapidmom#43).
    /// Present in [`eval`](Self::eval) and the passivity grid scan; the
    /// Verilog-A export rejects models with a significant √s part (not
    /// realisable in the ddt-only grammar) and the Hamiltonian crossing
    /// localisation ignores it (the dense scan still covers it).
    pub sq: Vec<f64>,
    /// Port names in network-matrix order.
    pub port_names: Vec<String>,
    /// Per-port reference impedances [Ω].
    pub z0: Vec<f64>,
    /// Fitted frequency band [Hz].
    pub fmin: f64,
    pub fmax: f64,
    /// Max relative fit error achieved over the fit samples (per-entry, magnitude-weighted).
    pub fit_error: f64,
    /// The fit sample frequencies [Hz] (anchors, or the tabulated grid) — retained so passivity
    /// enforcement and the export gate can report the true fit error.
    pub fit_freqs: Vec<f64>,
    /// The fit sample data: `fit_data[k]` is the flattened `p²` admittance at `fit_freqs[k]` [S].
    pub fit_data: Vec<Vec<C>>,
}

impl RationalModel {
    /// Number of poles (real + 2×complex pairs) — the model order.
    pub fn n_poles(&self) -> usize {
        self.real_poles.len() + 2 * self.cpx_poles.len()
    }

    /// Assess passivity (Hamiltonian localisation + dense `λ_min(Φ)` scan).
    pub fn passivity(&self) -> super::passivity::PassivityReport {
        super::passivity::check(self)
    }

    /// Enforce passivity in place by iterated minimum-norm residue perturbation (Gustavsen FRP).
    /// Returns the before/after margins and fit errors.
    pub fn enforce_passivity(&mut self, max_iter: usize) -> super::passivity::EnforceReport {
        super::passivity::enforce(self, max_iter)
    }

    /// Render the deterministic Verilog-A source (see [`crate::export`]).
    pub fn to_verilog_a(
        &self,
        name: &str,
        date: &str,
        force: bool,
        threshold: f64,
    ) -> Result<String, crate::export::ExportError> {
        crate::export::export_verilog_a(self, name, date, force, threshold)
    }

    /// Evaluate matrix entry `e = i·p + j` at frequency `f` [Hz]: `Y_ij(j·2π·f)` [S].
    pub fn eval_entry(&self, e: usize, f: f64) -> C {
        let s = C::new(0.0, TAU * f);
        let mut v = C::new(self.d[e], 0.0)
            + s * C::new(self.e[e], 0.0)
            + s.sqrt() * C::new(self.sq[e], 0.0);
        for (k, &pr) in self.real_poles.iter().enumerate() {
            v += C::new(self.res_real[e][k], 0.0) / (s - C::new(pr, 0.0));
        }
        for (k, &pc) in self.cpx_poles.iter().enumerate() {
            let r = self.res_cpx[e][k];
            v += r / (s - pc) + r.conj() / (s - pc.conj());
        }
        v
    }

    /// Evaluate the full `p×p` admittance at `f` [Hz], flattened row-major (`e = i·p + j`).
    pub fn eval(&self, f: f64) -> Vec<C> {
        (0..self.n_ports * self.n_ports)
            .map(|e| self.eval_entry(e, f))
            .collect()
    }

    /// Max relative error of the current model vs the stored fit samples (per entry, floored by
    /// `1e-4·max|entry|` so an in-band null cannot blow the ratio up). Used to report the effect of
    /// passivity enforcement on fit quality.
    pub fn sample_error(&self) -> f64 {
        let p2 = self.n_ports * self.n_ports;
        let mut emax = vec![0.0f64; p2];
        for row in &self.fit_data {
            for e in 0..p2 {
                emax[e] = emax[e].max(row[e].norm());
            }
        }
        let mut worst = 0.0f64;
        for (k, &f) in self.fit_freqs.iter().enumerate() {
            let got = self.eval(f);
            for e in 0..p2 {
                let floor = (1e-4 * emax[e]).max(1e-300);
                worst = worst.max(
                    (got[e] - self.fit_data[k][e]).norm() / self.fit_data[k][e].norm().max(floor),
                );
            }
        }
        worst
    }

    /// Build a physical model from a normalised [`VfModel`] fit (`s_n = j·f/fmax`). `fmin_hz`/
    /// `fmax_hz` are the fitted band; the fit samples (`fit_freqs`/`fit_data`, physical) are stored.
    /// Build from a raw engine [`VfModel`](crate::VfModel) plus the fit context —
    /// the bridge an embedding solver (e.g. a ROM frequency sweep) uses.
    pub fn from_vf(
        vf: &VfModel,
        fmin_hz: f64,
        fmax_hz: f64,
        n_ports: usize,
        port_names: Vec<String>,
        z0: Vec<f64>,
        fit_freqs: Vec<f64>,
        fit_data: Vec<Vec<C>>,
    ) -> Self {
        let k = TAU * fmax_hz; // normalisation constant s_n = s / K
        let p2 = n_ports * n_ports;
        // Split the expanded pole set into real poles and c.c.-pair representatives (Im > 0).
        // `VfModel::model` emits reals first, then (p, conj(p)) pairs, with residues aligned.
        let poles = &vf.poles;
        let eps = poles.iter().map(|p| p.norm()).fold(0.0, f64::max) * 1e-12;
        let real_idx: Vec<usize> = (0..poles.len())
            .filter(|&i| poles[i].im.abs() <= eps)
            .collect();
        let cpx_idx: Vec<usize> = (0..poles.len()).filter(|&i| poles[i].im > eps).collect();

        let real_poles: Vec<f64> = real_idx.iter().map(|&i| poles[i].re * k).collect();
        let cpx_poles: Vec<C> = cpx_idx.iter().map(|&i| poles[i] * k).collect();

        let mut res_real = vec![Vec::with_capacity(real_idx.len()); p2];
        let mut res_cpx = vec![Vec::with_capacity(cpx_idx.len()); p2];
        for e in 0..p2 {
            for &i in &real_idx {
                res_real[e].push(vf.res[e][i].re * k);
            }
            for &i in &cpx_idx {
                res_cpx[e].push(vf.res[e][i] * k);
            }
        }
        let d: Vec<f64> = (0..p2).map(|e| vf.cst[e].re).collect();
        let e_prop: Vec<f64> = (0..p2).map(|e| vf.dif[e].re / k).collect();
        // √(s) = √(K·s_n) = √K·√s_n ⇒ physical coefficient = normalised / √K.
        let sq: Vec<f64> = (0..p2).map(|e| vf.sqt[e].re / k.sqrt()).collect();

        let mut m = RationalModel {
            n_ports,
            real_poles,
            cpx_poles,
            res_real,
            res_cpx,
            d,
            e: e_prop,
            sq,
            port_names,
            z0,
            fmin: fmin_hz,
            fmax: fmax_hz,
            fit_error: 0.0,
            fit_freqs,
            fit_data,
        };
        m.fit_error = m.sample_error();
        m
    }
}

/// Fit tabulated `p`-port admittance data `Y(f)` with the relaxed vector-fitting engine and return
/// the physical [`RationalModel`]. `data[k]` is the flattened `p²` admittance at `freqs[k]` [Hz].
/// `max_poles` caps the model order; `tol` is the relative convergence target.
///
/// The engine fits on the normalised axis `s_n = j·f/fmax`; the returned model is denormalised to
/// physical units.
pub fn fit(
    freqs: &[f64],
    data: &[Vec<C>],
    n_ports: usize,
    max_poles: usize,
    tol: f64,
    port_names: Vec<String>,
    z0: Vec<f64>,
) -> RationalModel {
    let fmax = freqs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let fmin = freqs.iter().cloned().fold(f64::INFINITY, f64::min);
    let s: Vec<C> = freqs.iter().map(|&f| C::new(0.0, f / fmax)).collect();
    let vf = super::fit_auto(&s, data, tol, max_poles);
    RationalModel::from_vf(
        &vf,
        fmin,
        fmax,
        n_ports,
        port_names,
        z0,
        freqs.to_vec(),
        data.to_vec(),
    )
}

/// Fit with a **fixed** initial order (`n_cpx` complex-conjugate pairs + `n_real` real poles), no
/// automatic order selection. The total pole count `n_real + 2·n_cpx` is preserved through
/// relocation, so fitting every parameter anchor with the SAME `(n_cpx, n_real)` yields models of a
/// COMMON order — the prerequisite for a determined, spare-DOF-free common-basis conversion (a
/// mixed-order set, as `fit_auto`'s parsimony can produce across anchors, makes the parametric
/// coefficient interpolation non-unique and non-smooth). Used by the parametric fit.
pub fn fit_fixed(
    freqs: &[f64],
    data: &[Vec<C>],
    n_ports: usize,
    n_cpx: usize,
    n_real: usize,
    tol: f64,
    port_names: Vec<String>,
    z0: Vec<f64>,
) -> RationalModel {
    let fmax = freqs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let fmin = freqs.iter().cloned().fold(f64::INFINITY, f64::min);
    let s: Vec<C> = freqs.iter().map(|&f| C::new(0.0, f / fmax)).collect();
    let vf = super::fit_order(&s, data, n_cpx, n_real, tol, 12);
    RationalModel::from_vf(
        &vf,
        fmin,
        fmax,
        n_ports,
        port_names,
        z0,
        freqs.to_vec(),
        data.to_vec(),
    )
}
