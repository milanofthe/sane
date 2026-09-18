//! Circuit intermediate representation and symbolic determinant for SANE.
//!
//! This crate owns the topology-level data model that the whole pipeline shares
//! -- [`Circuit`], [`Element`], [`Kind`], [`SourceFn`], [`Coupling`] and the
//! behavioral-source tree ([`BExpr`] / [`BehavioralSource`]) -- together with
//! [`value_symbol_name`], the reserved-namespace rename that keeps element value
//! symbols from colliding with the solver's internally minted unknowns.
//!
//! Node `0` is ground throughout. Numeric analyses build their systems from a
//! [`Circuit`] in the `dae` crate (time-domain residual `F(x, x', t) = 0`) and
//! in `analysis` (sparse complex AC); closed-form transfer functions are
//! obtained by Cramer's rule on the symbolic small-signal matrix, with
//! `rsdag::determinant` as the kernel.

/// Topological index-2 detection (CV loops / LI cutsets).
pub mod index2;

mod source;
pub use source::SourceFn;

/// Symbol name for an element's value parameter, kept out of the namespace
/// SANE reserves for the solver's own unknowns.
///
/// An element's name doubles as the symbol of its defining value (`R1` is the
/// resistance, `V1` the source voltage, ...). Node voltages, however, are minted
/// internally as `v{k}` (and their derivatives `vdot{k}`), branch currents as
/// `i_{elem}`, inductor-current derivatives as `idot_{elem}` and internal-node
/// derivatives as `vdot_{label}`; time is `t`. Symbols are interned by name, so
/// an element whose name lands in that reserved space would be hash-consed onto
/// an unknown rather than staying a free parameter. The classic case is a power
/// grid with a voltage source literally named `v91`: it would share the symbol
/// of node 91's voltage, so its DC value silently vanishes from the parameter
/// set and the source stops constraining its node.
///
/// This maps any reserved-looking element name to a private, collision-free
/// spelling (a leading `_`, which no SPICE element name can have, since element
/// names start with their type letter); all ordinary names pass through
/// unchanged. It MUST be applied consistently wherever an element value symbol
/// is created and wherever its numeric value is bound by name, so the parameter
/// symbol and its value key always agree.
pub fn value_symbol_name(name: &str) -> String {
    if is_reserved_unknown_name(name) {
        format!("_{name}")
    } else {
        name.to_string()
    }
}

/// Whether `name` collides with SANE's internally generated unknown / time
/// symbol namespace (see [`value_symbol_name`]).
fn is_reserved_unknown_name(name: &str) -> bool {
    if name == "t" {
        return true;
    }
    // `v{digits}` (node voltage) or `vdot{digits}` (its derivative).
    let trailing_digits = |stem: &str| {
        name.strip_prefix(stem)
            .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
    };
    if trailing_digits("vdot") || trailing_digits("v") {
        return true;
    }
    // Prefixed branch-current / derivative unknowns.
    name.starts_with("i_") || name.starts_with("idot_") || name.starts_with("vdot_")
}

/// Element kind. The element name doubles as its parameter symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    /// Voltage-controlled voltage source (`E`): `V(a,b) = gain * V(nc+, nc-)`.
    Vcvs,
    /// Voltage-controlled current source (`G`): `I(a->b) = gm * V(nc+, nc-)`.
    Vccs,
    /// Current-controlled current source (`F`): `I(a->b) = gain * I(Vctrl)`.
    Cccs,
    /// Current-controlled voltage source (`H`): `V(a,b) = gain * I(Vctrl)`.
    Ccvs,
}

/// An element between output nodes `a` (+) and `b` (-). Voltage-controlled
/// sources carry their controlling node pair in `ctrl`; current-controlled
/// sources carry the name of the controlling (voltage-defined) element in
/// `ctrl_elem`. Independent sources may carry a time-domain `source` shape.
#[derive(Clone, Debug)]
pub struct Element {
    pub kind: Kind,
    pub name: String,
    pub a: usize,
    pub b: usize,
    pub ctrl: Option<(usize, usize)>,
    pub ctrl_elem: Option<String>,
    pub source: Option<SourceFn>,
}

/// A mutual-inductance coupling (`K`) between two inductors, with coupling
/// coefficient named after the `K` element. Not a node element.
#[derive(Clone, Debug)]
pub struct Coupling {
    pub name: String,
    pub l1: String,
    pub l2: String,
}

/// A behavioral (`B`) source expression tree over node voltages, branch currents
/// and parameters. The topology layer keeps it as data (node indices, element
/// names); the DAE layer translates it into the symbolic graph. `NodeV(0)` is
/// ground (zero).
#[derive(Clone, Debug)]
pub enum BExpr {
    Const(f64),
    /// Node voltage `V(node)`.
    NodeV(usize),
    /// Branch current `I(element)` of a voltage-defined element.
    BranchI(String),
    /// A named parameter (stays symbolic).
    Param(String),
    Neg(Box<BExpr>),
    /// Binary operator, one of `+ - * / ^`.
    Bin(char, Box<BExpr>, Box<BExpr>),
    /// Function call (`sin`, `exp`, `sqrt`, `tanh`, `pow`, ...).
    Call(String, Vec<BExpr>),
}

impl BExpr {
    /// Largest node index referenced (so the circuit can size its node set).
    fn max_node(&self) -> usize {
        match self {
            BExpr::NodeV(n) => *n,
            BExpr::Neg(a) => a.max_node(),
            BExpr::Bin(_, a, b) => a.max_node().max(b.max_node()),
            BExpr::Call(_, args) => args.iter().map(|a| a.max_node()).max().unwrap_or(0),
            _ => 0,
        }
    }
}

/// Behavioral source kind: a voltage `V=expr` (constraint `V(a,b)=expr`) or a
/// current `I=expr` (current `a -> b`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BKind {
    V,
    I,
}

/// A behavioral source (`B`): an arbitrary expression driving a voltage or a
/// current between nodes `a` (+) and `b` (-).
#[derive(Clone, Debug)]
pub struct BehavioralSource {
    pub name: String,
    pub a: usize,
    pub b: usize,
    pub kind: BKind,
    pub expr: BExpr,
}

/// A circuit: a set of elements over integer nodes (0 = ground), plus mutual
/// couplings between inductors.
#[derive(Default, Debug)]
pub struct Circuit {
    elements: Vec<Element>,
    couplings: Vec<Coupling>,
    behavioral: Vec<BehavioralSource>,
    max_node: usize,
}

impl Circuit {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, kind: Kind, name: &str, a: usize, b: usize, ctrl: Option<(usize, usize)>) {
        self.max_node = self.max_node.max(a).max(b);
        if let Some((p, q)) = ctrl {
            self.max_node = self.max_node.max(p).max(q);
        }
        self.elements.push(Element {
            kind,
            name: name.to_string(),
            a,
            b,
            ctrl,
            ctrl_elem: None,
            source: None,
        });
    }

    /// Push a current-controlled source (`F`/`H`) controlled by element `ctrl`.
    fn push_cc(&mut self, kind: Kind, name: &str, a: usize, b: usize, ctrl: &str) {
        self.max_node = self.max_node.max(a).max(b);
        self.elements.push(Element {
            kind,
            name: name.to_string(),
            a,
            b,
            ctrl: None,
            ctrl_elem: Some(ctrl.to_string()),
            source: None,
        });
    }

    /// Attach a time-domain source shape to the most recently added element.
    pub fn set_source(&mut self, src: SourceFn) -> &mut Self {
        if let Some(e) = self.elements.last_mut() {
            e.source = Some(src);
        }
        self
    }

    pub fn resistor(&mut self, name: &str, a: usize, b: usize) -> &mut Self {
        self.push(Kind::Resistor, name, a, b, None);
        self
    }
    pub fn capacitor(&mut self, name: &str, a: usize, b: usize) -> &mut Self {
        self.push(Kind::Capacitor, name, a, b, None);
        self
    }
    pub fn inductor(&mut self, name: &str, a: usize, b: usize) -> &mut Self {
        self.push(Kind::Inductor, name, a, b, None);
        self
    }
    pub fn voltage_source(&mut self, name: &str, a: usize, b: usize) -> &mut Self {
        self.push(Kind::VoltageSource, name, a, b, None);
        self
    }
    pub fn current_source(&mut self, name: &str, a: usize, b: usize) -> &mut Self {
        self.push(Kind::CurrentSource, name, a, b, None);
        self
    }
    /// VCVS `E`: output `(np, nm)`, controlled by `V(ncp, ncm)`.
    pub fn vcvs(&mut self, name: &str, np: usize, nm: usize, ncp: usize, ncm: usize) -> &mut Self {
        self.push(Kind::Vcvs, name, np, nm, Some((ncp, ncm)));
        self
    }
    /// VCCS `G`: output `(np, nm)`, controlled by `V(ncp, ncm)`.
    pub fn vccs(&mut self, name: &str, np: usize, nm: usize, ncp: usize, ncm: usize) -> &mut Self {
        self.push(Kind::Vccs, name, np, nm, Some((ncp, ncm)));
        self
    }
    /// CCCS `F`: output `(np, nm)`, current = gain * I through element `ctrl`.
    pub fn cccs(&mut self, name: &str, np: usize, nm: usize, ctrl: &str) -> &mut Self {
        self.push_cc(Kind::Cccs, name, np, nm, ctrl);
        self
    }
    /// CCVS `H`: output `(np, nm)`, voltage = gain * I through element `ctrl`.
    pub fn ccvs(&mut self, name: &str, np: usize, nm: usize, ctrl: &str) -> &mut Self {
        self.push_cc(Kind::Ccvs, name, np, nm, ctrl);
        self
    }
    /// Mutual inductance `K` between inductors `l1` and `l2`.
    pub fn mutual(&mut self, name: &str, l1: &str, l2: &str) -> &mut Self {
        self.couplings.push(Coupling {
            name: name.to_string(),
            l1: l1.to_string(),
            l2: l2.to_string(),
        });
        self
    }

    /// Mutual-inductance couplings.
    pub fn couplings(&self) -> &[Coupling] {
        &self.couplings
    }

    /// Add a behavioral (`B`) source between nodes `a` (+) and `b` (-).
    pub fn behavioral_source(&mut self, name: &str, a: usize, b: usize, kind: BKind, expr: BExpr) {
        self.max_node = self.max_node.max(a).max(b).max(expr.max_node());
        self.behavioral.push(BehavioralSource {
            name: name.to_string(),
            a,
            b,
            kind,
            expr,
        });
    }

    /// The behavioral (`B`) sources, in insertion order.
    pub fn behavioral(&self) -> &[BehavioralSource] {
        &self.behavioral
    }

    /// Number of non-ground nodes (public accessor for downstream assemblers).
    pub fn node_count(&self) -> usize {
        self.max_node
    }

    /// Resistors as `(name, node_a, node_b)`, for noise analysis (each resistor
    /// is a thermal-noise current source between its nodes).
    pub fn resistors(&self) -> Vec<(String, usize, usize)> {
        self.elements
            .iter()
            .filter(|e| e.kind == Kind::Resistor)
            .map(|e| (e.name.clone(), e.a, e.b))
            .collect()
    }

    /// The elements of the circuit, in insertion order.
    pub fn elements(&self) -> &[Element] {
        &self.elements
    }
}
