//! Abstract syntax tree for the Verilog-A analog subset.
//!
//! Spans are carried on expressions and statements that can fail elaboration so
//! diagnostics point at the right source. Numeric values are `f64` throughout
//! (Verilog-A reals); integer-ness is tracked only where it matters (parameter
//! types, literal flags).

use crate::error::Span;

/// A parsed Verilog-A module (one `module ... endmodule`).
#[derive(Clone, Debug)]
pub struct Module {
    pub name: String,
    /// Terminal names in declaration order (the instance terminal order).
    pub ports: Vec<String>,
    /// Electrical net names declared in the module (ports + internal nodes), in
    /// declaration order. Internal nodes are those not also in `ports`.
    pub nets: Vec<String>,
    /// Declared parameters, in source order.
    pub params: Vec<ParamDecl>,
    /// `parameter string name = "lit";` declarations: name -> default literal.
    /// String parameters are compile-time-only (they select model variants via
    /// `==`/`!=` comparisons that fold during lowering); they mint no symbol.
    pub string_params: Vec<(String, String)>,
    /// `aliasparam alias = target;` declarations, in source order. `target` is a
    /// parameter name, or a system function carried with its `$` prefix (the
    /// LRM allows e.g. `aliasparam mult = $mfactor;`).
    pub aliases: Vec<AliasDecl>,
    /// `real`/`integer` local variables declared in the module.
    pub vars: Vec<VarDecl>,
    /// Named branches (`branch (a,b) br;`).
    pub branches: Vec<BranchDecl>,
    /// `analog function` definitions.
    pub functions: Vec<AnalogFunction>,
    /// Statements of the `analog` block (one analog block, possibly a `begin`).
    pub analog: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VarType {
    Real,
    Integer,
}

/// The value of one `(* name = value *)` attribute.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrVal {
    /// Bare attribute name, no value (`(* flag *)`).
    Flag,
    /// String (or identifier) value.
    Str(String),
    Num(f64),
}

#[derive(Clone, Debug)]
pub struct ParamDecl {
    pub name: String,
    pub ty: VarType,
    pub default: Expr,
    /// Range constraints (`from`/`exclude`); empty if unconstrained.
    pub ranges: Vec<RangeConstraint>,
    /// `(* ... *)` attributes preceding the declaration (`type="instance"`,
    /// `units`, `desc`, ...).
    pub attrs: Vec<(String, AttrVal)>,
    pub span: Span,
}

/// `aliasparam alias = target;` -- an alternate deck-facing name for a
/// parameter (or a system function like `$mfactor`). Setting the alias sets the
/// target; the alias itself is not a parameter.
#[derive(Clone, Debug)]
pub struct AliasDecl {
    pub alias: String,
    pub target: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct VarDecl {
    pub name: String,
    pub ty: VarType,
    /// `(* ... *)` attributes preceding the declaration. A `desc`/`units`
    /// annotation marks an operating-point variable (the compact-model OPP
    /// idiom) that analyses can report after a solve.
    pub attrs: Vec<(String, AttrVal)>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct BranchDecl {
    pub name: String,
    pub hi: String,
    pub lo: String,
}

/// A `from`/`exclude` range on a parameter.
#[derive(Clone, Debug)]
pub struct RangeConstraint {
    pub include: bool, // true = `from`, false = `exclude`
    pub lo: Bound,
    pub hi: Bound,
}

#[derive(Clone, Debug)]
pub enum Bound {
    /// Inclusive `[`/`]` end at this value.
    Inclusive(Expr),
    /// Exclusive `(`/`)` end at this value.
    Exclusive(Expr),
    /// `inf` / `-inf`.
    Inf(bool), // true = +inf
}

#[derive(Clone, Debug)]
pub struct AnalogFunction {
    pub name: String,
    pub ty: VarType,
    pub args: Vec<String>,
    /// Argument names declared `output`/`inout` (written back to the caller).
    pub outputs: Vec<String>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

/// A nature access function. Electrical potential `V(...)` and flow `I(...)` are
/// first-class (the only disciplines SANE lowers today); other-discipline
/// accesses (thermal `Pwr`/`Temp`, `Q`, `Phi`, ...) are carried by name so the
/// parser accepts real models, and the lowering can defer them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Access {
    V,
    I,
    Other(String),
}

#[derive(Clone, Debug)]
pub enum Stmt {
    /// `begin ... end` (optionally named).
    Block(Vec<Stmt>),
    /// `lhs = rhs;` (procedural assignment to a variable).
    Assign { lhs: String, rhs: Expr, span: Span },
    /// `V(a,b) <+ rhs;` or `I(a,b) <+ rhs;`
    Contribution {
        access: Access,
        hi: String,
        lo: Option<String>,
        rhs: Expr,
        span: Span,
    },
    /// Indirect (implicit) contribution `V(a,b) : lhs == rhs ;` — drive the
    /// branch so the implicit equation `lhs == rhs` holds.
    Indirect {
        access: Access,
        hi: String,
        lo: Option<String>,
        lhs: Expr,
        rhs: Expr,
        span: Span,
    },
    /// `if (cond) then [else else_]`
    If {
        cond: Expr,
        then: Box<Stmt>,
        els: Option<Box<Stmt>>,
    },
    /// `case (sel) items... [default: ...] endcase`
    Case {
        sel: Expr,
        items: Vec<(Vec<Expr>, Stmt)>,
        default: Option<Box<Stmt>>,
    },
    /// `for (init; cond; step) body` (constant bounds; unrolled at elaboration).
    For {
        init: Box<Stmt>,
        cond: Expr,
        step: Box<Stmt>,
        body: Box<Stmt>,
    },
    /// `while (cond) body` (bounded fixed-point loop; unrolled to its static
    /// counter bound at lowering).
    While { cond: Expr, body: Box<Stmt> },
    /// `@(initial_step) ...` one-time initialization.
    InitialStep(Box<Stmt>),
    /// An analog event control `@(control(args...)) body`. `cross`/`above`
    /// declare a switching surface the transient integrator lands on (their
    /// arguments are parsed); every other control (`timer`, `final_step`) is
    /// carried to lowering, which rejects it with a diagnostic instead of
    /// running the body unconditionally with its guard stripped (issue #42).
    Event {
        control: String,
        args: Vec<Expr>,
        body: Box<Stmt>,
        span: Span,
    },
    /// A system task call ignored for simulation (`$strobe`, `$display`).
    IgnoredCall,
    /// A diagnostic system task whose arguments are preserved so lowering can
    /// surface it: `$warning`/`$error`/`$fatal`/`$finish(...)` (issue #41). A
    /// `$warning` becomes a captured warning; a compile-time-reached
    /// `$error`/`$fatal` is a hard load failure; a runtime-guarded one becomes a
    /// load-time notice that the assertion cannot be enforced.
    SysTask {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    /// A bare call statement `name(args);` — an analog-function call (inlined for
    /// its `output` side effects) or an ignored task.
    Call {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    /// An empty statement (`;`).
    Empty,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Num(f64),
    /// String literal (only valid as a system-function argument, e.g. the option
    /// name in `$simparam("gmin")` or text in `$strobe`).
    Str(String),
    /// Array / concatenation literal `{e0, e1, ...}` (e.g. Laplace filter
    /// coefficient vectors).
    Array(Vec<Expr>),
    /// A bare identifier: a parameter, variable, or named branch.
    Ident(String, Span),
    /// `V(node)` / `V(a,b)` / `I(...)` / `I(branch)`.
    Access {
        access: Access,
        hi: String,
        lo: Option<String>,
        span: Span,
    },
    /// System function: `$temperature`, `$vt`, `$abstime`, ...
    SysFn {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    /// `ddt(x)`, `idt(x[,ic])`, `limexp(x)`, builtin math, analog function call.
    Call {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    Unary {
        op: UnOp,
        arg: Box<Expr>,
        span: Span,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    /// `cond ? then : else`.
    Ternary {
        cond: Box<Expr>,
        then: Box<Expr>,
        els: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Num(_) | Expr::Str(_) | Expr::Array(_) => Span::default(),
            Expr::Ident(_, s)
            | Expr::Access { span: s, .. }
            | Expr::SysFn { span: s, .. }
            | Expr::Call { span: s, .. }
            | Expr::Unary { span: s, .. }
            | Expr::Binary { span: s, .. }
            | Expr::Ternary { span: s, .. } => *s,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}
