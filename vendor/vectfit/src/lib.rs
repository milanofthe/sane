//! Fast Relaxed Vector Fitting (Gustavsen-Semlyen 1999; relaxed pole relocation, Gustavsen
//! 2006; multiport, Deschrijver 2008) with automatic order selection by adding-and-skimming
//! (Grivet-Talocia-Bandinu 2006) — a set-valued macromodel of a matrix response on the jω axis
//! with COMMON poles. Ported (compact, clean-room from the algorithm) from the author's
//! `vectorfitting` Python package.
//!
//! Self-contained: depends only on `faer` (dense LS/eigensolve) and `num_complex`, so this module
//! is extractable as a standalone crate. [`fit`] is the fixed-order primitive; [`fit_auto`] grows
//! and skims the pole set to the minimal one that meets a target error.
//!
//! Per relocation step: identify the weighting σ with a RELAXED constant `d_relax` (so the pole
//! relocation is not biased by the σ(∞)=1 normalisation), get the new poles as the zeros of σ
//! (eigenvalues of `A − b·rᵀ/d_relax`), reflect unstable poles, then fit the residues. Real and
//! complex-conjugate poles are tracked separately. The σ LS is the small full system (our
//! port/anchor counts are modest, so the block-QR "fast" trick of the reference buys little).

use num_complex::Complex;

pub type C = Complex<f64>;

/// Pin faer's process-global parallelism to sequential, once. The LS panels
/// here are small (2·ns × order), so faer-internal parallelism buys nothing —
/// but its pool-width-dependent reductions would break the bit-identical
/// guarantee of the (entry-parallel) fit. Self-contained (no dependency on
/// the host crate) so the module stays extractable as a standalone crate.
pub(crate) fn pin_faer_sequential() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| faer::set_global_parallelism(faer::Par::Seq));
}

/// Stage logging gate, mirroring the host crate's `RAPIDMOM_LOG` convention but
/// read directly from the environment (once) so the crate stays standalone.
/// `[vf]` lines go to stderr — the same stream the host's default sink uses.
pub(crate) fn log_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("RAPIDMOM_LOG")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

macro_rules! vflog {
    ($($arg:tt)*) => {
        if $crate::log_on() {
            eprintln!($($arg)*);
        }
    };
}
pub(crate) use vflog;

mod engine;
pub mod export;
mod model;
pub mod passivity;
pub mod ratmodel;

pub use engine::fit as fit_order;
pub use engine::{fit_auto, fit_auto_with}; // production entry points; return `model::VfModel`
pub use model::VfModel;

#[cfg(test)]
mod tests {
    use super::engine::fit;
    use super::*;

    /// Recovers a known stable set-valued rational (shared poles) to high accuracy.
    #[test]
    fn vf_recovers_known_rational() {
        let p = [C::new(-0.05, 0.7), C::new(-0.2, 2.3)];
        let r = [
            [C::new(0.4, 0.1), C::new(-0.2, 0.3)],
            [C::new(0.5, -0.2), C::new(0.1, 0.05)],
        ];
        let cst = [C::new(0.3, 0.0), C::new(-0.1, 0.0)];
        let s: Vec<C> = (0..60)
            .map(|i| C::new(0.0, 0.05 + 3.0 * i as f64 / 59.0))
            .collect();
        let data: Vec<Vec<C>> = s
            .iter()
            .map(|&z| {
                (0..2)
                    .map(|e| {
                        let mut v = cst[e];
                        for (pp, rr) in p.iter().zip(&r) {
                            v += rr[e] / (z - pp) + rr[e].conj() / (z - pp.conj());
                        }
                        v
                    })
                    .collect()
            })
            .collect();
        let m = fit(&s, &data, 2, 0, 1e-6, 8);
        let mut maxerr = 0.0f64;
        for (k, &z) in s.iter().enumerate() {
            let got = m.eval(z);
            for e in 0..2 {
                maxerr = maxerr.max((got[e] - data[k][e]).norm());
            }
        }
        assert!(maxerr < 1e-4, "VF recovery error {maxerr:e}");
        assert!(m.n_support() <= 5, "VF order {}", m.n_support());
    }

    /// `fit_auto` recovers a known shared-pole set-valued response and stays parsimonious: it
    /// grows the order only until the error elbow and skims spurious poles, so it does not
    /// over-fit into extra (noise-fitting) poles.
    #[test]
    fn fit_auto_recovers_setvalued_and_is_parsimonious() {
        let p = [C::new(-0.05, 0.7), C::new(-0.05, -0.7)];
        let r0 = [C::new(0.4, 0.1), C::new(0.4, -0.1)];
        let r1 = [C::new(-0.2, 0.3), C::new(-0.2, -0.3)];
        let s: Vec<C> = (0..50)
            .map(|i| C::new(0.0, 0.05 + 1.95 * i as f64 / 49.0))
            .collect();
        let data: Vec<Vec<C>> = s
            .iter()
            .map(|&z| {
                let e0 = C::new(0.2, 0.0) + r0[0] / (z - p[0]) + r0[1] / (z - p[1]);
                let e1 = C::new(-0.1, 0.0) + r1[0] / (z - p[0]) + r1[1] / (z - p[1]);
                vec![e0, e1]
            })
            .collect();
        let m = fit_auto(&s, &data, 1e-3, 12);
        let mut maxerr = 0.0f64;
        for (k, &z) in s.iter().enumerate() {
            let got = m.eval(z);
            for e in 0..2 {
                maxerr = maxerr.max((got[e] - data[k][e]).norm());
            }
        }
        assert!(maxerr < 1e-3, "set-valued recovery error {maxerr:e}");
        assert!(
            m.n_support() <= 4,
            "over-parametrised: order {}",
            m.n_support()
        );
    }
}
