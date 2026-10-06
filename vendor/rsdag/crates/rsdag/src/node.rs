use crate::func::OutputId;

/// Index of an interned expression node inside a [`crate::Graph`].
///
/// Cheap to copy and compare; identical subexpressions share one `ExprId`
/// thanks to hash-consing, so structural equality is `O(1)`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ExprId(pub u32);

/// Index of an interned exact rational constant in a [`crate::Graph`]'s
/// constant table. Two equal rationals share one id, so a constant node is a
/// 4-byte handle and comparing constants is an integer compare.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ConstId(pub u32);

/// An interned operand list: a `(start, len)` window into the context's shared
/// argument pool. Lists are deduplicated by content, so two structurally equal
/// variadic nodes carry the *same* `ArgList` and hash-cons to one node. Resolve
/// to a slice with [`crate::Graph::args`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ArgList {
    pub start: u32,
    pub len: u32,
}

impl ArgList {
    pub fn len(self) -> usize {
        self.len as usize
    }
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Index of a free symbol (component value, `gm`, the Laplace variable `s`, ...).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SymbolId(pub u32);

/// Comparison operators; a `Cmp` node evaluates to `1.0` (true) or `0.0`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

impl CmpOp {
    /// The operator as written, `>`, `>=`, ...
    pub fn symbol(self) -> &'static str {
        match self {
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
        }
    }
}

/// Associative reduction over a variadic operand list. Folds a flat list of
/// terms in one node, shrinking the tape (KCL current sums become one `Reduce`
/// instead of an Add-tree) and exposing a vectorizable loop.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
}

impl ReduceOp {
    /// Name in printed expressions.
    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Product => "prod",
            ReduceOp::Min => "min",
            ReduceOp::Max => "max",
        }
    }

    /// Combine an accumulator with the next element (left fold).
    pub fn combine(self, acc: f64, x: f64) -> f64 {
        match self {
            ReduceOp::Sum => acc + x,
            ReduceOp::Product => acc * x,
            ReduceOp::Min => acc.min(x),
            ReduceOp::Max => acc.max(x),
        }
    }
}

/// Transcendental / elementary unary functions, needed by nonlinear device
/// constitutive equations (diode `exp`, EKV/`tanh`, ...).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum UnaryOp {
    Exp,
    Ln,
    Sqrt,
    Sin,
    Cos,
    Sinh,
    Cosh,
    Tanh,
    Atan,
    Floor,
    // The floating-point extension (reference implementations from `libm`,
    // so the bits do not depend on the platform's C library).
    Tan,
    Log10,
    Log2,
    Log1p,
    Expm1,
    Cbrt,
    Abs,
    /// `-1`, `0`, `1` (`+-0.0` and NaN pass through, numpy's `sign`).
    Sign,
    Ceil,
    Round,
    Trunc,
    Asin,
    Acos,
    Asinh,
    Acosh,
    Atanh,
    Erf,
    Erfc,
    Lgamma,
    Tgamma,
    Digamma,
    Trigamma,
    /// Counter-based uniform noise in `[0, 1)` keyed by the argument's bits:
    /// a pure function, so it traces, replays and batches like any other op.
    RandUniform,
}

/// Binary operations beyond the ring (`Add`, `Mul`, `Neg`, `Pow` are their
/// own node kinds for the canonical ordering and the reduction fusion; `Sub`
/// and `Div` are `add(neg)` and `mul(recip)`).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum BinOp {
    /// `a^b` for real exponents (`Pow` covers integer exponents).
    Powf,
    /// `fmod(a, b)`: the remainder with the sign of `a`.
    Mod,
    /// `atan2(a, b)`.
    Atan2,
    /// `sqrt(a^2 + b^2)` without overflow.
    Hypot,
}

#[derive(Clone, Copy, Debug)]
pub struct UnarySpec {
    /// The variant this row describes; `UNARY_OPS[op as usize].op == op`.
    pub op: UnaryOp,
    /// Name in printed expressions and in the Python frontend.
    pub name: &'static str,
    /// Differentiable where it is defined, and differentiated by rsdag.
    /// The rough ones (`floor`, `sign`, the roundings, the noise source)
    /// have a zero or undefined derivative, `trigamma` one rsdag does not
    /// carry; they are excluded from smooth generated programs.
    pub smooth: bool,
}

/// The unary vocabulary, in the order of the enum: `UNARY_OPS[op as usize]`
/// is `op`'s row (checked by a test).
pub const UNARY_OPS: &[UnarySpec] = &[
    UnarySpec {
        op: UnaryOp::Exp,
        name: "exp",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Ln,
        name: "ln",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Sqrt,
        name: "sqrt",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Sin,
        name: "sin",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Cos,
        name: "cos",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Sinh,
        name: "sinh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Cosh,
        name: "cosh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Tanh,
        name: "tanh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Atan,
        name: "atan",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Floor,
        name: "floor",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Tan,
        name: "tan",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Log10,
        name: "log10",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Log2,
        name: "log2",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Log1p,
        name: "log1p",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Expm1,
        name: "expm1",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Cbrt,
        name: "cbrt",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Abs,
        name: "abs",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Sign,
        name: "sign",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Ceil,
        name: "ceil",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Round,
        name: "round",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Trunc,
        name: "trunc",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Asin,
        name: "asin",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Acos,
        name: "acos",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Asinh,
        name: "asinh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Acosh,
        name: "acosh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Atanh,
        name: "atanh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Erf,
        name: "erf",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Erfc,
        name: "erfc",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Lgamma,
        name: "lgamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Tgamma,
        name: "tgamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Digamma,
        name: "digamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Trigamma,
        name: "trigamma",
        // Its derivative (polygamma of order 2) is not implemented.
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::RandUniform,
        name: "rand_uniform",
        smooth: false,
    },
];

impl UnaryOp {
    /// This op's row of [`UNARY_OPS`].
    #[inline]
    pub fn spec(self) -> &'static UnarySpec {
        &UNARY_OPS[self as usize]
    }
    /// Name in printed expressions.
    #[inline]
    pub fn name(self) -> &'static str {
        self.spec().name
    }
    /// Differentiable everywhere it is defined.
    #[inline]
    pub fn is_smooth(self) -> bool {
        self.spec().smooth
    }
    /// A stable integer code, for backends that pass the op to a host
    /// routine as a value (the JIT's trampolines).
    #[inline]
    pub fn code(self) -> u32 {
        self as u32
    }
    /// The op for a [`UnaryOp::code`]; panics on an unknown code.
    #[inline]
    pub fn from_code(code: u32) -> UnaryOp {
        UNARY_OPS[code as usize].op
    }
    /// The op a frontend's function name denotes, by [`UnarySpec::name`].
    pub fn from_name(name: &str) -> Option<UnaryOp> {
        UNARY_OPS
            .iter()
            .find(|spec| spec.name == name)
            .map(|spec| spec.op)
    }
}

/// As [`UnarySpec`], for the binary ops beyond the ring.
#[derive(Clone, Copy, Debug)]
pub struct BinarySpec {
    pub op: BinOp,
    pub name: &'static str,
    pub smooth: bool,
}

/// The binary vocabulary, in the order of the enum.
pub const BINARY_OPS: &[BinarySpec] = &[
    BinarySpec {
        op: BinOp::Powf,
        name: "powf",
        smooth: true,
    },
    BinarySpec {
        op: BinOp::Mod,
        name: "mod",
        smooth: false,
    },
    BinarySpec {
        op: BinOp::Atan2,
        name: "atan2",
        smooth: true,
    },
    BinarySpec {
        op: BinOp::Hypot,
        name: "hypot",
        smooth: true,
    },
];

impl BinOp {
    #[inline]
    pub fn spec(self) -> &'static BinarySpec {
        &BINARY_OPS[self as usize]
    }
    #[inline]
    pub fn name(self) -> &'static str {
        self.spec().name
    }
    #[inline]
    pub fn is_smooth(self) -> bool {
        self.spec().smooth
    }
    #[inline]
    pub fn code(self) -> u32 {
        self as u32
    }
    #[inline]
    pub fn from_code(code: u32) -> BinOp {
        BINARY_OPS[code as usize].op
    }
    /// The op a frontend's function name denotes, by [`BinarySpec::name`].
    pub fn from_name(name: &str) -> Option<BinOp> {
        BINARY_OPS
            .iter()
            .find(|spec| spec.name == name)
            .map(|spec| spec.op)
    }
}

/// A node in the symbolic DAG.
///
/// Leaves are exact rational constants or free symbols. Inner nodes are the
/// algebraic operations that show up when stamping and solving circuit
/// equations, plus the elementary functions used by nonlinear device models.
///
/// A node is a 16-byte `Copy` value: every payload is a small integer handle
/// (an `ExprId`, a `ConstId` into the constant table, an `ArgList` into the
/// argument pool). Interning therefore hashes and stores 16 bytes, never a
/// heap allocation, and the arena is one dense array the caches like -- the
/// build passes (differentiation, substitution, tape compilation) are bound by
/// exactly this per-node cost.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Node {
    /// Exact rational constant, by id into the context's constant table (kept
    /// reduced, sign in the numerator). Resolve with [`crate::Graph::const_val`].
    Const(ConstId),
    /// Free symbol, referenced by id into the context symbol table.
    Symbol(SymbolId),
    /// Binary addition. Operands ordered by id to maximise sharing.
    Add(ExprId, ExprId),
    /// Binary multiplication. Operands ordered by id to maximise sharing.
    Mul(ExprId, ExprId),
    /// Unary negation.
    Neg(ExprId),
    /// Integer power (covers reciprocals via negative exponents). The
    /// exponent fits `i32` ([`Graph::pow_i`](crate::Graph::pow_i) sees to it).
    Pow(ExprId, i64),
    /// Elementary unary function.
    Unary(UnaryOp, ExprId),
    /// Binary function beyond the ring (`Powf`, `Mod`, `Atan2`, `Hypot`).
    Binary(BinOp, ExprId, ExprId),
    /// Comparison, yielding `1.0`/`0.0`. Used to build region conditions.
    Cmp(CmpOp, ExprId, ExprId),
    /// Conditional: `cond != 0 ? then : else_`. For region-based device models.
    Select(ExprId, ExprId, ExprId),
    /// Associative reduction over a flat operand list (KCL sums, products).
    /// One fused node instead of a balanced/again-binary tree.
    Reduce(ReduceOp, ArgList),
    /// Inner product `Σ_i a[i]*b[i]` over two equal-length operand lists,
    /// stored as ONE list `[a_0..a_n, b_0..b_n]` (resolve the halves with
    /// [`crate::Graph::dot_args`]). The fused form of a sum of pairwise
    /// products (matrix-vector rows).
    Dot(ArgList),
    /// Component `i` of `x` solving the dense system `A x = b`, the list
    /// being `A` row-major (`n * n`) followed by `b` (`n`), with `n` from
    /// the length. The `n` components share the list and evaluate as one
    /// pivoting kernel (see [`crate::semantics::solve_t`]); the graph
    /// differentiates through it by the identities of the inverse.
    Solve(ArgList, u32),
    /// A call: output `OutputId` of a function (see [`crate::func`]) applied
    /// to the argument list. One compact model instantiated many times is one
    /// function and many calls; differentiation references the function's
    /// derivative outputs by the chain rule.
    Call(OutputId, ArgList),
}

const _: () = assert!(std::mem::size_of::<Node>() == 16);

impl Node {
    /// The operands this node reads, its list resolved in `pool` (the
    /// argument pool of the graph or module it belongs to).
    #[inline]
    pub fn operands<'a>(&self, pool: &'a [ExprId]) -> Operands<'a> {
        let z = ExprId(0);
        match *self {
            Node::Const(_) | Node::Symbol(_) => Operands::Inline { buf: [z; 3], n: 0 },
            Node::Add(a, b) | Node::Mul(a, b) | Node::Cmp(_, a, b) | Node::Binary(_, a, b) => {
                Operands::Inline {
                    buf: [a, b, z],
                    n: 2,
                }
            }
            Node::Neg(a) | Node::Pow(a, _) | Node::Unary(_, a) => Operands::Inline {
                buf: [a, z, z],
                n: 1,
            },
            Node::Select(c, t, e) => Operands::Inline {
                buf: [c, t, e],
                n: 3,
            },
            Node::Reduce(_, l) | Node::Dot(l) | Node::Call(_, l) | Node::Solve(l, _) => {
                Operands::Slice(&pool[l.start as usize..(l.start + l.len) as usize])
            }
        }
    }

    /// This node over other operands: `ops` in [`operands`](Self::operands)
    /// order, a variadic node taking `list` (their interned window) instead.
    pub(crate) fn with_operands(self, ops: &[ExprId], list: ArgList) -> Node {
        match self {
            Node::Const(_) | Node::Symbol(_) => self,
            Node::Add(..) => Node::Add(ops[0], ops[1]),
            Node::Mul(..) => Node::Mul(ops[0], ops[1]),
            Node::Neg(_) => Node::Neg(ops[0]),
            Node::Pow(_, n) => Node::Pow(ops[0], n),
            Node::Unary(op, _) => Node::Unary(op, ops[0]),
            Node::Binary(op, ..) => Node::Binary(op, ops[0], ops[1]),
            Node::Cmp(op, ..) => Node::Cmp(op, ops[0], ops[1]),
            Node::Select(..) => Node::Select(ops[0], ops[1], ops[2]),
            Node::Reduce(op, _) => Node::Reduce(op, list),
            Node::Dot(_) => Node::Dot(list),
            Node::Solve(_, i) => Node::Solve(list, i),
            Node::Call(o, _) => Node::Call(o, list),
        }
    }

    /// Whether the operands are an interned list rather than inline.
    #[inline]
    pub(crate) fn is_variadic(&self) -> bool {
        matches!(
            self,
            Node::Reduce(..) | Node::Dot(_) | Node::Solve(..) | Node::Call(..)
        )
    }
}

/// The operands of a node, borrowed without allocation: inline for the
/// fixed-arity variants, a pool slice for the variadic ones. Derefs to
/// `&[ExprId]`.
pub enum Operands<'a> {
    Inline {
        buf: [ExprId; 3],
        n: u8,
    },
    Slice(&'a [ExprId]),
    /// A bound call's: its arguments, then its context's expressions.
    Owned(Vec<ExprId>),
}

impl std::ops::Deref for Operands<'_> {
    type Target = [ExprId];
    fn deref(&self) -> &[ExprId] {
        match self {
            Operands::Inline { buf, n } => &buf[..*n as usize],
            Operands::Slice(s) => s,
            Operands::Owned(v) => v,
        }
    }
}

/// The arithmetic of structural fingerprints (see
/// [`Graph::fingerprint`](crate::Graph::fingerprint)): a fixed mixing
/// function and a fixed byte hasher, so a fingerprint is the same on every
/// platform, in every run and under every version of the hash crates.
pub(crate) mod shape {
    use std::hash::{Hash, Hasher};

    /// One tag per node kind, so equal payloads of different kinds differ.
    #[derive(Clone, Copy)]
    pub(crate) enum Tag {
        Const = 1,
        Symbol,
        Add,
        Mul,
        Neg,
        Pow,
        Unary,
        Binary,
        Cmp,
        Select,
        Reduce,
        Dot,
        Solve,
        Call,
    }

    /// Fold `x` into `h` (a splitmix64 finalizer over the sum).
    #[inline]
    pub(crate) fn mix(h: u64, x: u64) -> u64 {
        let mut z = h
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(x)
            .wrapping_add(0x6A09_E667_F3BC_C909);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value's `Hash` fed through [`mix`], byte for byte.
    pub(crate) fn of_hash<T: Hash + ?Sized>(v: &T) -> u64 {
        let mut h = Stable(0);
        v.hash(&mut h);
        h.0
    }

    struct Stable(u64);

    /// Integers in little-endian and `usize` as 64 bits (a length prefix
    /// hashed on wasm32 must match the one hashed on a 64-bit host).
    impl Hasher for Stable {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write_u16(&mut self, i: u16) {
            self.write(&i.to_le_bytes());
        }
        fn write_u32(&mut self, i: u32) {
            self.write(&i.to_le_bytes());
        }
        fn write_u64(&mut self, i: u64) {
            self.write(&i.to_le_bytes());
        }
        fn write_u128(&mut self, i: u128) {
            self.write(&i.to_le_bytes());
        }
        fn write_usize(&mut self, i: usize) {
            self.write_u64(i as u64);
        }
        fn write(&mut self, bytes: &[u8]) {
            for chunk in bytes.chunks(8) {
                let mut w = [0u8; 8];
                w[..chunk.len()].copy_from_slice(chunk);
                self.0 = mix(self.0, u64::from_le_bytes(w) ^ ((chunk.len() as u64) << 56));
            }
        }
    }
}
