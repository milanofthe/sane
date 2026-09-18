//! The constant field of a graph.
//!
//! A graph is generic over the type its constants are stored and folded in:
//! exact rationals for symbolic work (a transfer function must not lose
//! precision while it is built), plain `f64` for numeric-only frontends such
//! as a Python tracer. The smart constructors only use what this trait
//! provides, so every construction rule holds for every field.
//!
//! Transcendental folding (`exp` of a constant, and the like) is not part of
//! the field: it exists only for floating fields, through the same reference
//! math the execution backends use, and lives in the optimizer.

use std::hash::Hash;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

/// A field with exact equality and hashing, the constant type of a graph.
pub trait Field: Clone + PartialEq + Eq + Hash + std::fmt::Debug + Send + Sync + 'static {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_i64(n: i64) -> Self;
    /// `num / den`; `den` must not be zero.
    fn from_ratio(num: i64, den: i64) -> Self;
    /// The exact value of an `f64`, or `None` if the field cannot hold it
    /// (a rational cannot hold infinities or NaN).
    fn from_f64(x: f64) -> Option<Self>;
    /// The nearest `f64` (the value every execution type starts from).
    fn to_f64(&self) -> f64;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    /// `self^n` for an integer `n`, `None` for a negative power of zero.
    fn powi(&self, n: i64) -> Option<Self>;
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    /// Whether the field is exact (no rounding in `add`, `mul`, `powi`).
    /// An exact field never folds transcendental functions.
    fn is_exact() -> bool;
    /// Ordering of two constants (`None` only for NaN in a floating field).
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering>;
    /// Exact textual form for the printer (`3/2`, `-7`, `0.1`).
    fn render(&self) -> String;
}

impl Field for BigRational {
    fn zero() -> Self {
        <BigRational as Zero>::zero()
    }
    fn one() -> Self {
        <BigRational as One>::one()
    }
    fn from_i64(n: i64) -> Self {
        BigRational::from_integer(BigInt::from(n))
    }
    fn from_ratio(num: i64, den: i64) -> Self {
        BigRational::new(BigInt::from(num), BigInt::from(den))
    }
    fn from_f64(x: f64) -> Option<Self> {
        BigRational::from_float(x)
    }
    fn to_f64(&self) -> f64 {
        ToPrimitive::to_f64(self).unwrap_or(f64::NAN)
    }
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    fn neg(&self) -> Self {
        -self
    }
    fn powi(&self, n: i64) -> Option<Self> {
        if Zero::is_zero(self) && n < 0 {
            return None;
        }
        Some(ratio_powi(self, n))
    }
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }
    fn is_one(&self) -> bool {
        One::is_one(self)
    }
    fn is_exact() -> bool {
        true
    }
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
    fn render(&self) -> String {
        if One::is_one(self.denom()) {
            self.numer().to_string()
        } else {
            format!("{}/{}", self.numer(), self.denom())
        }
    }
}

/// Exact integer power of a rational by repeated squaring (negative
/// exponents invert; the caller excludes zero to a negative power).
pub fn ratio_powi(base: &BigRational, n: i64) -> BigRational {
    if n == 0 {
        return <BigRational as One>::one();
    }
    let (b, mut e) = if n < 0 {
        (base.recip(), (-n) as u64)
    } else {
        (base.clone(), n as u64)
    };
    let mut acc = <BigRational as One>::one();
    let mut sq = b;
    while e > 0 {
        if e & 1 == 1 {
            acc *= &sq;
        }
        e >>= 1;
        if e > 0 {
            sq = &sq * &sq;
        }
    }
    acc
}

/// `f64` as a field: hashed and compared by bit pattern, with `-0.0`
/// canonicalized to `0.0` (the rational field does not distinguish them
/// either) and every NaN to one canonical NaN. Infinities are kept.
#[derive(Clone, Copy, Debug)]
pub struct F64(f64);

impl F64 {
    #[inline]
    pub fn new(x: f64) -> Self {
        if x.is_nan() {
            F64(f64::NAN)
        } else if x == 0.0 {
            F64(0.0)
        } else {
            F64(x)
        }
    }
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl PartialEq for F64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for F64 {}
impl Hash for F64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state)
    }
}

/// `F64` serializes as its IEEE bit pattern, not as a number.
///
/// A text format is not a safe container for a `f64`: `serde_json` writes
/// the shortest representation that round-trips but its parser returns a
/// neighbouring double for some of them, and JSON cannot represent an
/// infinity or a NaN at all. A module is the program, so a constant that
/// comes back one ulp away is a different program. The bit pattern is exact
/// in every format.
#[cfg(feature = "serde")]
impl serde::Serialize for F64 {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_u64(self.0.to_bits())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for F64 {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<F64, D::Error> {
        let bits = <u64 as serde::Deserialize>::deserialize(de)?;
        Ok(F64(f64::from_bits(bits)))
    }
}

impl Field for F64 {
    fn zero() -> Self {
        F64(0.0)
    }
    fn one() -> Self {
        F64(1.0)
    }
    fn from_i64(n: i64) -> Self {
        F64::new(n as f64)
    }
    fn from_ratio(num: i64, den: i64) -> Self {
        F64::new(num as f64 / den as f64)
    }
    fn from_f64(x: f64) -> Option<Self> {
        Some(F64::new(x))
    }
    fn to_f64(&self) -> f64 {
        self.0
    }
    fn add(&self, other: &Self) -> Self {
        F64::new(self.0 + other.0)
    }
    fn mul(&self, other: &Self) -> Self {
        F64::new(self.0 * other.0)
    }
    fn neg(&self) -> Self {
        F64::new(-self.0)
    }
    fn powi(&self, n: i64) -> Option<Self> {
        if self.0 == 0.0 && n < 0 {
            return None;
        }
        Some(F64::new(self.0.powi(n as i32)))
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
    fn is_one(&self) -> bool {
        self.0 == 1.0
    }
    fn is_exact() -> bool {
        false
    }
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
    fn render(&self) -> String {
        format!("{:?}", self.0)
    }
}
