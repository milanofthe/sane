//! What goes wrong building or analysing a [`Model`](super::Model).

/// Error from building or analysing a [`Model`](super::Model).
#[derive(Debug, Clone)]
pub enum ModelError {
    /// The netlist does not parse, where (see [`sane_netlist::ParseError`]).
    Parse(sane_netlist::ParseError),
    /// The circuit does not set up: a device's model does not lower, an
    /// element does not assemble.
    Build(String),
    /// The DC operating point did not converge.
    NoConverge,
    /// A reference is not a known node / unknown / branch current.
    UnknownRef(String),
    /// A name is not a parameter.
    UnknownParam(String),
    /// A numeric step failed (singular matrix, AFT grid too coarse, ...).
    Numeric(String),
    /// The parameters leave the model invalid: an assertion of a device does
    /// not hold (a loop unrolled only so far, an `$error` on these values).
    Invalid(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Parse(e) => write!(f, "parse error: {e}"),
            ModelError::Build(m) => write!(f, "{m}"),
            ModelError::NoConverge => write!(f, "DC operating point did not converge"),
            ModelError::UnknownRef(r) => {
                write!(f, "'{r}' is not a known node, unknown or branch current")
            }
            ModelError::UnknownParam(p) => write!(f, "'{p}' is not a parameter"),
            ModelError::Numeric(m) => write!(f, "{m}"),
            ModelError::Invalid(m) => write!(f, "invalid parameters: {m}"),
        }
    }
}

impl std::error::Error for ModelError {}
