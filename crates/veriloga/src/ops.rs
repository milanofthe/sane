//! The analog operators of Verilog-A: the calls the lowering builds
//! structure for (a charge, a state, a delay, a noise generator) instead of
//! an expression, and the filters it does not support. Named once here; the
//! elementary functions are [`sane_core::MathFn`].

/// An analog operator, by what the lowering makes of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnalogOp {
    /// `ddt(q)`: a charge.
    Ddt,
    /// `ddx(f, V(n))`: a partial derivative.
    Ddx,
    /// `idt(u[, ic])`: a state integrating `u`.
    Idt,
    /// `idtmod(u, ic, modulus[, offset])`: an integral wrapped into a range.
    Idtmod,
    /// `absdelay(u, td)`: a transport delay.
    Absdelay,
    /// `laplace_nd(u, num, den)`: a rational filter in `s`.
    LaplaceNd,
    /// `white_noise(psd)`.
    WhiteNoise,
    /// `flicker_noise(psd, exp)`.
    FlickerNoise,
    /// `noise_table(...)`, `noise_table_log(...)`: a tabulated spectrum.
    NoiseTable,
    /// `analysis(...)`: which analysis runs; zero for SANE's one model of
    /// every analysis.
    Analysis,
    /// `limexp(x)`: the exponential, its growth limited.
    Limexp,
    /// `$limit`-style `limit(x, ...)`: a Newton-path aid, the identity on
    /// the converged result.
    Limit,
    /// A time-domain filter (`transition`, `slew`, `last_crossing`).
    TimeFilter,
    /// A Laplace or Z filter other than `laplace_nd`.
    ZFilter,
}

impl AnalogOp {
    /// The operator a call names.
    pub(crate) fn from_name(name: &str) -> Option<AnalogOp> {
        Some(match name {
            "ddt" => AnalogOp::Ddt,
            "ddx" => AnalogOp::Ddx,
            "idt" => AnalogOp::Idt,
            "idtmod" => AnalogOp::Idtmod,
            "absdelay" => AnalogOp::Absdelay,
            "laplace_nd" => AnalogOp::LaplaceNd,
            "white_noise" => AnalogOp::WhiteNoise,
            "flicker_noise" => AnalogOp::FlickerNoise,
            "noise_table" | "noise_table_log" => AnalogOp::NoiseTable,
            "analysis" => AnalogOp::Analysis,
            "limexp" => AnalogOp::Limexp,
            "limit" => AnalogOp::Limit,
            "transition" | "slew" | "last_crossing" => AnalogOp::TimeFilter,
            "laplace_zd" | "laplace_np" | "laplace_zp" | "zi_nd" | "zi_zd" | "zi_np" | "zi_zp" => {
                AnalogOp::ZFilter
            }
            _ => return None,
        })
    }
}
