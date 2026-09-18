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
    /// The identity element (value of an empty reduction).
    pub fn identity(self) -> f64 {
        match self {
            ReduceOp::Sum => 0.0,
            ReduceOp::Product => 1.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
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
    /// Name in printed expressions and in the Python and C frontends.
    pub name: &'static str,
    /// The callee in generated C: a `libm` name where the semantics agree,
    /// an `rsdag_` helper where rsdag guards or defines the op itself.
    pub c_fn: &'static str,
    /// Differentiable everywhere it is defined. The rough ones (`floor`,
    /// `sign`, the roundings, the noise source) have a zero or undefined
    /// derivative and are excluded from smooth generated programs.
    pub smooth: bool,
}

/// The unary vocabulary, in the order of the enum: `UNARY_OPS[op as usize]`
/// is `op`'s row (checked by a test).
pub const UNARY_OPS: &[UnarySpec] = &[
    UnarySpec {
        op: UnaryOp::Exp,
        name: "exp",
        c_fn: "rsdag_exp",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Ln,
        name: "ln",
        c_fn: "rsdag_ln",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Sqrt,
        name: "sqrt",
        c_fn: "rsdag_sqrt",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Sin,
        name: "sin",
        c_fn: "sin",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Cos,
        name: "cos",
        c_fn: "cos",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Sinh,
        name: "sinh",
        c_fn: "sinh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Cosh,
        name: "cosh",
        c_fn: "cosh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Tanh,
        name: "tanh",
        c_fn: "tanh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Atan,
        name: "atan",
        c_fn: "atan",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Floor,
        name: "floor",
        c_fn: "floor",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Tan,
        name: "tan",
        c_fn: "tan",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Log10,
        name: "log10",
        c_fn: "log10",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Log2,
        name: "log2",
        c_fn: "log2",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Log1p,
        name: "log1p",
        c_fn: "log1p",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Expm1,
        name: "expm1",
        c_fn: "expm1",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Cbrt,
        name: "cbrt",
        c_fn: "cbrt",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Abs,
        name: "abs",
        c_fn: "fabs",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Sign,
        name: "sign",
        c_fn: "rsdag_sign",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Ceil,
        name: "ceil",
        c_fn: "ceil",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Round,
        name: "round",
        c_fn: "round",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Trunc,
        name: "trunc",
        c_fn: "trunc",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Asin,
        name: "asin",
        c_fn: "asin",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Acos,
        name: "acos",
        c_fn: "acos",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Asinh,
        name: "asinh",
        c_fn: "asinh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Acosh,
        name: "acosh",
        c_fn: "acosh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Atanh,
        name: "atanh",
        c_fn: "atanh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Erf,
        name: "erf",
        c_fn: "erf",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Erfc,
        name: "erfc",
        c_fn: "erfc",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Lgamma,
        name: "lgamma",
        c_fn: "lgamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Tgamma,
        name: "tgamma",
        c_fn: "tgamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Digamma,
        name: "digamma",
        c_fn: "rsdag_digamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Trigamma,
        name: "trigamma",
        c_fn: "rsdag_trigamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::RandUniform,
        name: "rand_uniform",
        c_fn: "rsdag_rand_uniform",
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
    /// The callee to emit in generated C.
    #[inline]
    pub fn c_fn(self) -> &'static str {
        self.spec().c_fn
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
    pub c_fn: &'static str,
    pub smooth: bool,
}

/// The binary vocabulary, in the order of the enum.
pub const BINARY_OPS: &[BinarySpec] = &[
    BinarySpec {
        op: BinOp::Powf,
        name: "powf",
        c_fn: "pow",
        smooth: true,
    },
    BinarySpec {
        op: BinOp::Mod,
        name: "mod",
        c_fn: "fmod",
        smooth: false,
    },
    BinarySpec {
        op: BinOp::Atan2,
        name: "atan2",
        c_fn: "atan2",
        smooth: true,
    },
    BinarySpec {
        op: BinOp::Hypot,
        name: "hypot",
        c_fn: "hypot",
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
    pub fn c_fn(self) -> &'static str {
        self.spec().c_fn
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
    /// Integer power (covers reciprocals via negative exponents).
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

/// The operands of a node, borrowed without allocation: inline for the
/// fixed-arity variants, a pool slice for the variadic ones. Derefs to
/// `&[ExprId]`.
pub enum Operands<'a> {
    Inline { buf: [ExprId; 3], n: u8 },
    Slice(&'a [ExprId]),
}

impl std::ops::Deref for Operands<'_> {
    type Target = [ExprId];
    fn deref(&self) -> &[ExprId] {
        match self {
            Operands::Inline { buf, n } => &buf[..*n as usize],
            Operands::Slice(s) => s,
        }
    }
}
