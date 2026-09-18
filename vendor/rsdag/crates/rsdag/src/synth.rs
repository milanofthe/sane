//! Synthetic programs: one seeded generator of random functions over the
//! whole op vocabulary.
//!
//! Every guarantee rsdag makes is a statement over arbitrary programs -- the
//! backends agree bit for bit, a specialization evaluates like the choice it
//! froze, a derivative matches finite differences -- so the tests that check
//! them and the benchmarks that price them want the same thing: programs
//! nobody wrote by hand, reproducible from a seed. Keeping one generator
//! means a new op is covered everywhere the moment it is drawable here, and
//! that the benchmark measures the same population the parity suite checks.
//!
//! ```
//! use rsdag::{synth::{Spec, build, inputs}, Graph, Tape, F64};
//! let mut g: Graph<F64> = Graph::new();
//! let mut spec = Spec::new(7);          // seed
//! spec.steps = 200;
//! let (outs, syms) = build(&mut g, &mut spec);
//! let tape = Tape::compile(&g, &outs, &syms);
//! let (mut w, mut out) = (Vec::new(), Vec::new());
//! tape.eval(&inputs(&mut spec.rng(), syms.len()), &mut w, &mut out);
//! ```

use crate::field::Field;
use crate::graph::Graph;
use crate::node::{BinOp, CmpOp, ExprId, ReduceOp, SymbolId, UnaryOp};

/// Deterministic xorshift64*, so a seed reproduces a program exactly on
/// every platform (a test that fails names a seed, and the seed is the
/// repro).
#[derive(Clone, Copy, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // A zero state is a fixed point of xorshift; move it off.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// A value in `[-2, 2)`, the range the domain-safe forms below assume.
    pub fn val(&mut self) -> f64 {
        (self.next_u64() % 4000) as f64 / 1000.0 - 2.0
    }
    /// A value in `[0.5, 2.5)`, for arguments that must stay positive.
    pub fn pos(&mut self) -> f64 {
        (self.next_u64() % 2000) as f64 / 1000.0 + 0.5
    }
    pub fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

/// Which forms a program may contain. The three levels are the three
/// guarantees the backends give: the ring is bit-exact everywhere including
/// generated C, the elementary functions are bit-exact between the paths
/// that share the host's math routines (interpreter, JIT) and within a
/// tolerance against another `libm`, and the full vocabulary is everything
/// rsdag can lower.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vocabulary {
    /// Ring, integer powers, comparisons, selects, reductions and dots:
    /// every backend must reproduce these bit for bit.
    Ring,
    /// The ring plus the elementary functions (`exp`, `ln`, `sqrt`, the
    /// trigonometric and hyperbolic ones, `floor`, `abs`).
    Elementary,
    /// Everything, including `erf`, `lgamma`, `digamma`, `atan2`, `hypot`,
    /// `powf`, the roundings and the random source.
    Full,
}

/// What to draw. The defaults are the general case; a caller narrows them to
/// the property it is checking -- `smooth` for a derivative check,
/// `Vocabulary::Ring` for a bit-exactness check.
#[derive(Clone, Debug)]
pub struct Spec {
    /// Number of nodes drawn (the graph is smaller after hash-consing).
    pub steps: usize,
    /// Free symbols the program reads.
    pub n_params: usize,
    /// Root expressions ("outputs") to return.
    pub n_outputs: usize,
    /// Only differentiable forms: no `Cmp`, `Select`, `Floor`, `Min`/`Max`,
    /// none of the rough extension ops.
    pub smooth: bool,
    /// Which forms may be drawn.
    pub vocab: Vocabulary,
    /// Percent of drawn ops that are variadic (`Reduce`, `Dot`).
    pub list_percent: usize,
    /// How wide the generated programs are.
    ///
    /// Operands are drawn from the last `width` nodes built, so 1 is a
    /// chain and a number past the program size is a uniform draw over
    /// everything so far. This is the knob that decides whether a corpus
    /// looks like real work: an assembled residual or Jacobian is very
    /// wide -- 8196 nodes at depth 7, one level per row -- while a uniform
    /// draw over a growing pool builds deep chains that no consumer
    /// produces. The fuzzers want both; a benchmark wants the wide end.
    pub width: usize,
    /// Percent of drawn ops that are a `Select` over a fresh comparison.
    /// The knob a region or specialization test turns up: those need
    /// programs whose guards actually flip as the inputs move.
    pub select_percent: usize,
    /// Longest operand list drawn for a variadic op.
    pub max_list: usize,
    seed: u64,
    rng: Rng,
}

impl Spec {
    pub fn new(seed: u64) -> Spec {
        Spec {
            steps: 100,
            n_params: 4,
            n_outputs: 1,
            smooth: false,
            vocab: Vocabulary::Full,
            width: RECENT,
            list_percent: 15,
            select_percent: 0,
            max_list: 4,
            seed,
            rng: Rng::new(seed),
        }
    }
    /// Only differentiable forms.
    pub fn smooth(mut self) -> Spec {
        self.smooth = true;
        self
    }
    pub fn vocab(mut self, v: Vocabulary) -> Spec {
        self.vocab = v;
        self
    }
    /// Draw operand lists up to `n` long. Above the backends' SIMD
    /// threshold a reduction takes a different code path, so a bit-exactness
    /// check wants lists on both sides of it.
    pub fn max_list(mut self, n: usize) -> Spec {
        self.max_list = n;
        self
    }
    /// How far back operands are drawn (see [`Spec::width`]).
    pub fn width(mut self, n: usize) -> Spec {
        self.width = n.max(1);
        self
    }
    /// Draw a guarded `Select` for this percentage of the nodes.
    pub fn selects(mut self, percent: usize) -> Spec {
        self.select_percent = percent;
        self
    }
    pub fn steps(mut self, n: usize) -> Spec {
        self.steps = n;
        self
    }
    pub fn params(mut self, n: usize) -> Spec {
        self.n_params = n;
        self
    }
    pub fn outputs(mut self, n: usize) -> Spec {
        self.n_outputs = n;
        self
    }
    /// A fresh generator on this spec's seed, so inputs drawn for a program
    /// are reproducible independently of how many nodes the build consumed.
    pub fn rng(&self) -> Rng {
        Rng::new(self.seed.wrapping_add(0xA5A5_A5A5))
    }
}

/// Random inputs in `[-2, 2)`, the range the generated programs are built to
/// stay finite over.
pub fn inputs(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.val()).collect()
}

/// Ring forms (see [`ring_op`]): smooth prefix, then comparisons, selects
/// and the ordered reductions.
const RING_SMOOTH: usize = 8;
const RING_ROUGH: usize = 4;
/// Elementary forms (see [`elementary_op`]), `floor` and `abs` last.
const ELEM_SMOOTH: usize = 10;
const ELEM_ROUGH: usize = 2;
/// Extension forms (see [`extended_op`]), smooth ones first.
const EXT_SMOOTH: usize = 16;
const EXT_ROUGH: usize = 7;

/// Build one program: `n_outputs` root expressions over `n_params` free
/// symbols. Returns the roots and the symbols in input order.
pub fn build<K: Field>(g: &mut Graph<K>, spec: &mut Spec) -> (Vec<ExprId>, Vec<SymbolId>) {
    let syms: Vec<ExprId> = (0..spec.n_params)
        .map(|i| g.sym(&format!("p{i}")))
        .collect();
    let sym_ids: Vec<SymbolId> = syms.iter().map(|&e| symbol_of(g, e)).collect();
    (build_roots(g, spec, &syms), sym_ids)
}

/// Draw `spec.steps` nodes over caller-provided symbols and return the last
/// one, for a caller that owns its symbol set.
pub fn build_over<K: Field>(g: &mut Graph<K>, spec: &mut Spec, syms: &[ExprId]) -> ExprId {
    let saved = std::mem::replace(&mut spec.n_outputs, 1);
    let roots = build_roots(g, spec, syms);
    spec.n_outputs = saved;
    roots[0]
}

/// The draw loop shared by both entry points.
fn build_roots<K: Field>(g: &mut Graph<K>, spec: &mut Spec, syms: &[ExprId]) -> Vec<ExprId> {
    let mut pool = syms.to_vec();
    for _ in 0..2 {
        let v = spec.rng.val();
        let k = g.konst_f64(v);
        pool.push(k);
    }
    let mut roots = Vec::with_capacity(spec.n_outputs);
    let per_root = spec.steps / spec.n_outputs.max(1);
    for r in 0..spec.n_outputs {
        let steps = if r + 1 == spec.n_outputs {
            spec.steps - per_root * r
        } else {
            per_root
        };
        for _ in 0..steps {
            let e = draw(g, spec, &pool, syms);
            pool.push(e);
        }
        roots.push(*pool.last().unwrap());
    }
    roots
}

/// The symbol behind a symbol node (the generator only ever passes its own).
fn symbol_of<K: Field>(g: &Graph<K>, e: ExprId) -> SymbolId {
    match g.node(e) {
        crate::node::Node::Symbol(s) => *s,
        _ => unreachable!("synth passes only symbol nodes"),
    }
}

/// Pick an operand, biased toward the most recently built nodes.
///
/// Drawing uniformly from the whole pool builds a soup of small shared
/// subexpressions that hash-consing collapses, so a program of 4000 draws
/// ends up a few hundred ops deep in nothing. Real code builds chains: a
/// value is consumed by what comes right after it. Two thirds of the
/// operands therefore come from the last [`RECENT`] entries, which makes
/// the generated programs both larger and deeper for the same draw count.
const RECENT: usize = 12;

fn pick(rng: &mut Rng, pool: &[ExprId], width: usize) -> ExprId {
    let n = pool.len();
    if n > width && rng.chance(66) {
        pool[n - 1 - rng.below(width)]
    } else {
        pool[rng.below(n)]
    }
}

/// Draw and build one node over the current pool.
fn draw<K: Field>(g: &mut Graph<K>, spec: &mut Spec, pool: &[ExprId], syms: &[ExprId]) -> ExprId {
    let spec_width = spec.width;
    let rng = &mut spec.rng;
    let a = pick(rng, pool, spec_width);
    let b = pick(rng, pool, spec_width);
    let c = pick(rng, pool, spec_width);
    if !spec.smooth && rng.chance(spec.select_percent) {
        // A select over a fresh comparison of two pool values: a guard that
        // moves when the inputs move, which is what a region test needs.
        let cmp = [CmpOp::Gt, CmpOp::Le, CmpOp::Lt][rng.below(3)];
        let cond = g.cmp(cmp, a, b);
        return g.select(cond, b, c);
    }
    if rng.chance(spec.list_percent) {
        let len = 2 + rng.below(spec.max_list.max(3) - 1);
        let list: Vec<ExprId> = (0..len).map(|_| pick(rng, pool, spec_width)).collect();
        let kind = if spec.smooth {
            rng.below(3)
        } else {
            rng.below(5)
        };
        return match kind {
            0 => g.reduce(ReduceOp::Sum, list),
            1 => g.reduce(ReduceOp::Product, list),
            2 => {
                let rhs: Vec<ExprId> = (0..list.len())
                    .map(|_| pick(rng, pool, spec_width))
                    .collect();
                g.dot(list, rhs)
            }
            3 => g.reduce(ReduceOp::Min, list),
            _ => g.reduce(ReduceOp::Max, list),
        };
    }
    // Index space: the ring forms, then the elementary ones, then the
    // extension forms, each block split into its smooth prefix and its
    // rough suffix so `smooth` is a prefix restriction at every level.
    let (ring, elem) = (
        if spec.smooth {
            RING_SMOOTH
        } else {
            RING_SMOOTH + RING_ROUGH
        },
        if spec.smooth {
            ELEM_SMOOTH
        } else {
            ELEM_SMOOTH + ELEM_ROUGH
        },
    );
    let ext = match spec.vocab {
        Vocabulary::Full if spec.smooth => EXT_SMOOTH,
        Vocabulary::Full => EXT_SMOOTH + EXT_ROUGH,
        _ => 0,
    };
    let elem = if spec.vocab == Vocabulary::Ring {
        0
    } else {
        elem
    };
    let i = rng.below(ring + elem + ext);
    if i >= ring + elem {
        return extended_op(g, i - ring - elem, a, b);
    }
    if i >= ring {
        return elementary_op(g, i - ring, a, b, syms);
    }
    ring_op(g, rng, i, a, b, c)
}

/// The ring forms: what every backend, generated C included, reproduces bit
/// for bit. The smooth ones first.
fn ring_op<K: Field>(
    g: &mut Graph<K>,
    rng: &mut Rng,
    k: usize,
    a: ExprId,
    b: ExprId,
    c: ExprId,
) -> ExprId {
    let pow = rng.below(6) as i64 - 2;
    let cmp = [CmpOp::Gt, CmpOp::Le, CmpOp::Lt][rng.below(3)];
    match k {
        0 => g.add(a, b),
        1 => g.sub(a, b),
        2 => g.mul(a, b),
        3 => g.neg(a),
        4 => {
            // `0^-n` is undefined over the exact rationals and the smart
            // constructor rejects it; steer a structural zero to a safe
            // exponent instead of losing the draw.
            let n = if pow < 0 && g.is_zero(a) { 2 } else { pow };
            g.pow_i(a, n)
        }
        5 => {
            let s = g.mul(a, b);
            g.add(s, c)
        }
        6 => g.reduce(ReduceOp::Sum, vec![a, b, c]),
        7 => g.dot(vec![a, b], vec![b, c]),
        8 => g.cmp(cmp, a, b),
        9 => g.select(a, b, c),
        10 => g.reduce(ReduceOp::Min, vec![a, b]),
        _ => g.reduce(ReduceOp::Max, vec![a, b, c]),
    }
}

/// The elementary functions, smooth ones first. `ln` and `sqrt` of a
/// structural zero would differentiate to `1/0`, so those arguments are
/// steered to a symbol.
fn elementary_op<K: Field>(
    g: &mut Graph<K>,
    k: usize,
    a: ExprId,
    b: ExprId,
    syms: &[ExprId],
) -> ExprId {
    match k {
        0 => g.exp(a),
        1 => {
            let arg = if g.is_zero(a) { syms[0] } else { a };
            g.ln(arg)
        }
        2 => {
            let arg = if g.is_zero(a) { syms[0] } else { a };
            g.sqrt(arg)
        }
        3 => g.sin(a),
        4 => g.cos(a),
        5 => g.sinh(a),
        6 => g.cosh(a),
        7 => g.tanh(a),
        8 => g.unary(UnaryOp::Atan, a),
        9 => {
            let s = g.sin(a);
            g.mul(s, b)
        }
        10 => g.floor(a),
        _ => g.unary(UnaryOp::Abs, a),
    }
}

/// Extension form `k` over `a` and `b`, each argument shifted or bounded
/// into the function's domain so the value stays finite; the smooth forms
/// come first so a derivative check can draw from the prefix alone.
fn extended_op<K: Field>(g: &mut Graph<K>, k: usize, a: ExprId, b: ExprId) -> ExprId {
    let one = g.one();
    let a2 = g.mul(a, a);
    let pos = g.add(a2, one); // >= 1
    let half = g.ratio(1, 2);
    let s = g.sin(a);
    let bounded = g.mul(half, s); // |.| <= 1/2
    match k {
        0 => {
            let t = g.tanh(a);
            g.unary(UnaryOp::Tan, t)
        }
        1 => g.unary(UnaryOp::Asinh, a),
        2 => g.unary(UnaryOp::Expm1, a),
        3 => g.unary(UnaryOp::Log1p, a2),
        4 => g.unary(UnaryOp::Erf, a),
        5 => g.unary(UnaryOp::Erfc, a),
        6 => g.binary(BinOp::Atan2, a, pos),
        7 => g.binary(BinOp::Hypot, pos, b),
        8 => {
            let t = g.tanh(b);
            g.binary(BinOp::Powf, pos, t)
        }
        9 => g.unary(UnaryOp::Cbrt, pos),
        10 => g.unary(UnaryOp::Lgamma, pos),
        11 => g.unary(UnaryOp::Digamma, pos),
        12 => g.unary(UnaryOp::Asin, bounded),
        13 => g.unary(UnaryOp::Atanh, bounded),
        14 => g.unary(UnaryOp::Log10, pos),
        15 => {
            let two = g.konst_int(2);
            let p2 = g.add(a2, two);
            g.unary(UnaryOp::Acosh, p2)
        }
        16 => g.unary(UnaryOp::Abs, a),
        17 => g.unary(UnaryOp::Sign, a),
        18 => g.unary(UnaryOp::Ceil, a),
        19 => g.unary(UnaryOp::Round, a),
        20 => g.unary(UnaryOp::Trunc, a),
        21 => {
            let m = g.ratio(3, 2);
            g.binary(BinOp::Mod, a, m)
        }
        _ => g.unary(UnaryOp::RandUniform, a),
    }
}

/// One generated program with everything a parity check needs: the graph,
/// the compiled tape, input rows, and the arena sweep as the reference.
///
/// The arena is the reference on purpose: it evaluates the graph directly,
/// so a `Case` checks the tape as well as whatever backend the caller adds.
pub struct Case {
    pub seed: u64,
    pub spec: Spec,
    pub graph: Graph<crate::F64>,
    pub roots: Vec<ExprId>,
    pub syms: Vec<SymbolId>,
    /// Input rows to evaluate over.
    pub rows: Vec<Vec<f64>>,
    pub tape: crate::Tape,
}

impl Case {
    /// The arena sweep for every row: what every other path must reproduce.
    pub fn reference(&self) -> Vec<Vec<f64>> {
        self.rows
            .iter()
            .map(|row| {
                let env: std::collections::HashMap<SymbolId, f64> =
                    self.syms.iter().copied().zip(row.iter().copied()).collect();
                crate::eval(&self.graph, &self.roots, &env)
            })
            .collect()
    }

    /// Check a path against the reference bit for bit (NaN counts as equal
    /// to NaN: the bit pattern of a NaN is not part of any guarantee).
    pub fn expect_bits(&self, path: &str, f: impl FnMut(&[f64]) -> Vec<f64>) {
        self.compare(path, 0.0, f)
    }

    /// Check a path against the reference within a relative tolerance, for
    /// the paths that call a different `libm` than the interpreter.
    pub fn expect_close(&self, path: &str, tol: f64, f: impl FnMut(&[f64]) -> Vec<f64>) {
        self.compare(path, tol, f)
    }

    fn compare(&self, path: &str, tol: f64, mut f: impl FnMut(&[f64]) -> Vec<f64>) {
        for (row, want) in self.rows.iter().zip(self.reference()) {
            let got = f(row);
            assert_eq!(
                got.len(),
                want.len(),
                "{path}: seed {} returned {} values, expected {}",
                self.seed,
                got.len(),
                want.len()
            );
            for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let ok = if g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()) {
                    true
                } else if tol > 0.0 {
                    (g - w).abs() <= tol * (1.0 + w.abs())
                } else {
                    false
                };
                assert!(
                    ok,
                    "{path}: seed {}, output {k}, inputs {row:?}\n  reference {w:?} ({:016x})\n  got       {g:?} ({:016x})\n{}",
                    self.seed,
                    w.to_bits(),
                    g.to_bits(),
                    self.tape.dump(),
                );
            }
        }
    }
}

/// A corpus of generated programs: `spec_for(seed)` decides what each one
/// draws, so a caller varies size, vocabulary and smoothness across the
/// seeds instead of checking one shape many times.
pub fn cases(
    seeds: std::ops::Range<u64>,
    spec_for: impl Fn(u64) -> Spec,
) -> impl Iterator<Item = Case> {
    seeds.map(move |seed| {
        let mut spec = spec_for(seed);
        let mut graph: Graph<crate::F64> = Graph::new();
        let (roots, syms) = build(&mut graph, &mut spec);
        let tape = crate::Tape::compile(&graph, &roots, &syms);
        let mut rng = spec.rng();
        let rows = (0..4).map(|_| inputs(&mut rng, syms.len())).collect();
        Case {
            seed,
            spec,
            graph,
            roots,
            syms,
            rows,
            tape,
        }
    })
}
