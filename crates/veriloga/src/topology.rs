//! The topology slice of an analog block: what the lowering's topology pass
//! needs to find the branches a module shorts and switches (see
//! [`crate::lower::lower_analog`]), the potential contributions, the
//! conditions around them and every assignment their values depend on.
//!
//! The slice is closed under dependence: a statement is in it when it
//! contributes a potential or writes a variable the slice reads, and what it
//! reads joins the slice, the conditions it runs under included (an
//! assignment inside an `if` reads the condition). A compound statement is in
//! it when a statement inside is. The rest the topology pass skips: lowering
//! it would change nothing the pass reports.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::ast::{Expr, Stmt};
use crate::elaborate::ElaboratedModule;
use crate::lower::is_potential;

/// The statements of a module's analog block outside its topology slice, by
/// address, once found (see [`outside_slice`]). A clone starts empty: its
/// statements live elsewhere.
#[derive(Debug, Default)]
pub(crate) struct SliceCache(std::sync::OnceLock<HashSet<usize>>);

impl Clone for SliceCache {
    fn clone(&self) -> Self {
        SliceCache::default()
    }
}

/// The statements of `em`'s analog block outside its topology slice, by
/// address: found once per module (the slice reads no parameter value).
pub(crate) fn outside_slice(em: &ElaboratedModule) -> &HashSet<usize> {
    em.slice.0.get_or_init(|| find(em))
}

fn find(em: &ElaboratedModule) -> HashSet<usize> {
    let mut leaves = Leaves {
        em,
        names: HashMap::default(),
        conds: Vec::new(),
        list: Vec::new(),
    };
    for s in &em.analog {
        leaves.collect(s);
    }
    // The fixpoint, backwards as the dependences run: a chain of
    // assignments joins in one sweep, a loop carrying one around in another.
    let mut read = vec![false; leaves.names.len()];
    let mut kept = vec![false; leaves.list.len()];
    let mut grew = true;
    while std::mem::take(&mut grew) {
        for (k, leaf) in leaves.list.iter().enumerate().rev() {
            if kept[k] || !(leaf.potential || leaf.writes.iter().any(|&w| read[w as usize])) {
                continue;
            }
            kept[k] = true;
            for &r in &leaf.reads {
                grew |= !std::mem::replace(&mut read[r as usize], true);
            }
        }
    }
    let kept: HashSet<usize> = (leaves.list.iter().zip(&kept))
        .filter(|(_, &k)| k)
        .map(|(leaf, _)| leaf.at)
        .collect();
    let mut skip = HashSet::default();
    for s in &em.analog {
        skipped(s, &kept, &mut skip);
    }
    skip
}

/// A statement without statements inside, and what the slice asks of it.
struct Leaf {
    at: usize,
    /// Contributes a potential: in the slice whatever it reads.
    potential: bool,
    /// The variables it may write: an assignment's target, every identifier
    /// passed to an analog function (an `output` argument is one).
    writes: Vec<u32>,
    /// The variables it reads, the conditions it runs under included.
    reads: Vec<u32>,
}

/// The leaves of an analog block in order, their variables interned.
struct Leaves<'a> {
    em: &'a ElaboratedModule,
    names: HashMap<&'a str, u32>,
    /// The variables of the conditions around the statement at hand.
    conds: Vec<Vec<u32>>,
    list: Vec<Leaf>,
}

impl<'a> Leaves<'a> {
    fn name(&mut self, n: &'a str) -> u32 {
        let k = self.names.len() as u32;
        *self.names.entry(n).or_insert(k)
    }

    /// The identifiers `es` read, at any depth.
    fn idents(&mut self, es: impl IntoIterator<Item = &'a Expr>) -> Vec<u32> {
        let mut out = Vec::new();
        for e in es {
            e.visit(&mut |e| {
                if let Expr::Ident(n, _) = e {
                    out.push(n.as_str());
                }
            });
        }
        out.into_iter().map(|n| self.name(n)).collect()
    }

    /// `s` under a condition reading `on`.
    fn under(&mut self, on: Vec<u32>, s: &'a Stmt) {
        self.conds.push(on);
        self.collect(s);
        self.conds.pop();
    }

    fn collect(&mut self, s: &'a Stmt) {
        match s {
            Stmt::Block(ss) => ss.iter().for_each(|s| self.collect(s)),
            Stmt::If { cond, then, els } => {
                let on = self.idents([cond]);
                self.conds.push(on);
                self.collect(then);
                if let Some(e) = els {
                    self.collect(e);
                }
                self.conds.pop();
            }
            Stmt::Case {
                sel,
                items,
                default,
            } => {
                let on = self.idents(std::iter::once(sel).chain(items.iter().flat_map(|(l, _)| l)));
                self.conds.push(on);
                items.iter().for_each(|(_, b)| self.collect(b));
                if let Some(d) = default {
                    self.collect(d);
                }
                self.conds.pop();
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
            } => {
                self.collect(init);
                let on = self.idents([cond]);
                self.conds.push(on);
                self.collect(body);
                self.collect(step);
                self.conds.pop();
            }
            Stmt::While { cond, body } => {
                let on = self.idents([cond]);
                self.under(on, body);
            }
            Stmt::InitialStep(body) => self.collect(body),
            Stmt::Event { args, body, .. } => {
                let on = self.idents(args);
                self.under(on, body);
            }
            leaf => {
                let mut exprs: Vec<&'a Expr> = Vec::new();
                leaf.for_each_expr(&mut |e| exprs.push(e));
                let mut writes: Vec<&'a str> = Vec::new();
                match leaf {
                    Stmt::Assign { lhs, .. } => writes.push(lhs),
                    Stmt::Call { name, args, .. } if self.em.functions.contains_key(name) => {
                        writes.extend(args.iter().filter_map(ident))
                    }
                    _ => {}
                }
                for e in &exprs {
                    e.visit(&mut |e| {
                        if let Expr::Call { name, args, .. } = e {
                            if self.em.functions.contains_key(name) {
                                writes.extend(args.iter().filter_map(ident));
                            }
                        }
                    });
                }
                let writes = writes.into_iter().map(|w| self.name(w)).collect();
                let mut reads = self.idents(exprs);
                reads.extend(self.conds.iter().flatten());
                self.list.push(Leaf {
                    at: at(leaf),
                    potential: matches!(leaf, Stmt::Contribution { access, .. } if is_potential(access)),
                    writes,
                    reads,
                });
            }
        }
    }
}

/// A statement's address, its identity while its block lives.
pub(crate) fn at(s: &Stmt) -> usize {
    s as *const Stmt as usize
}

/// An identifier argument's name (an argument a function may write).
fn ident(a: &Expr) -> Option<&str> {
    match a {
        Expr::Ident(n, _) => Some(n),
        _ => None,
    }
}

/// Add the statements of `s` outside the slice (no leaf inside is `kept`)
/// to `skip`, the outermost of each run; whether `s` is in the slice.
fn skipped(s: &Stmt, kept: &HashSet<usize>, skip: &mut HashSet<usize>) -> bool {
    let mut any = |s: &Stmt| skipped(s, kept, skip);
    let inside = match s {
        Stmt::Block(ss) => ss.iter().fold(false, |k, s| any(s) | k),
        Stmt::If { then, els, .. } => any(then) | els.as_deref().is_some_and(&mut any),
        Stmt::Case { items, default, .. } => {
            items.iter().fold(false, |k, (_, b)| any(b) | k) | default.as_deref().is_some_and(any)
        }
        Stmt::For {
            init, step, body, ..
        } => any(init) | any(step) | any(body),
        Stmt::While { body, .. } | Stmt::InitialStep(body) | Stmt::Event { body, .. } => any(body),
        leaf => kept.contains(&at(leaf)),
    };
    if !inside {
        skip.insert(at(s));
    }
    inside
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
    use sane_core::Graph;
    use sane_device::Lowerer;

    use crate::ast::VarType;
    use crate::elaborate::ElaboratedModule;
    use crate::lower::topology_pass;
    use crate::{builtin_module, elaborate, parse_modules};

    /// The corpus models checked in with the benchmarks; none where the
    /// tree has no benchmarks (a source export without them).
    fn corpus() -> Vec<Arc<ElaboratedModule>> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/corpus/models");
        if !root.is_dir() {
            println!("corpus: SKIP (no benchmarks/corpus in this tree)");
            return Vec::new();
        }
        let files = [
            "psp103/vacode/psp103.va",
            "psp103/vacode/psp103t.va",
            "psp103/vacode/psp103_nqs.va",
            "bsim4/bsim4.va",
            "ekv/vacode/ekv26.va",
            "hicum0/vacode/hicumL0_v2p1p0.va",
            "vbic/vacode/vbic_1p3.va",
            "vbic/vacode/vbic_4T_et_cf.va",
        ];
        let mut out = Vec::new();
        for f in files {
            let path = root.join(f);
            let src = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{f}"));
            let dir = path.parent().unwrap().to_path_buf();
            for m in parse_modules(&src, f, &[dir]).unwrap_or_else(|d| panic!("{f}: {d:?}")) {
                out.push(Arc::new(
                    elaborate(&m).unwrap_or_else(|d| panic!("{f}: {d:?}")),
                ));
            }
        }
        out
    }

    /// The topology pass's findings for `em` at `values`, `given` set, the
    /// statements `skip` names left out.
    #[allow(clippy::type_complexity)]
    fn topology(
        em: &ElaboratedModule,
        values: &HashMap<String, f64>,
        given: &HashSet<String>,
        skip: &HashSet<usize>,
    ) -> Result<(Vec<(String, String)>, Vec<(String, String)>), String> {
        let mut ctx = Graph::new();
        let tv: Vec<_> = (0..em.ports.len())
            .map(|k| ctx.sym(&format!("t{k}")))
            .collect();
        let mut lo = Lowerer::new(&mut ctx);
        topology_pass(em, "x", given, values, 1.0, &mut lo, &tv, skip)
    }

    /// A branch switched by the state through a chain of variables, a
    /// short decided by a parameter through another, beside flow
    /// contributions the slice leaves out.
    fn switching() -> Arc<ElaboratedModule> {
        let src = r#"
`include "disciplines.vams"
module sw(a, b, c);
  inout a, b, c; electrical a, b, c, m;
  parameter real R = 1k;
  parameter real Rs = 0;
  real v, on, g, rs2, w;
  analog begin
    v = V(a, b);
    on = v - 0.5;
    g = 1 / R;
    w = g * v * v;
    rs2 = 2 * Rs;
    if (rs2 > 0)
      I(c, m) <+ V(c, m) / rs2;
    else
      V(c, m) <+ 0;
    if (on > 0)
      V(a, b) <+ 0.1;
    else
      I(a, b) <+ g * v + w;
    I(m, b) <+ V(m, b) / R;
  end
endmodule
"#;
        let m = &parse_modules(src, "sw.va", &[]).expect("parses")[0];
        Arc::new(elaborate(m).expect("elaborates"))
    }

    /// The topology pass over the slice finds what it finds over the whole
    /// block: for the built-in and the corpus models, at the defaults and
    /// with every real parameter set off its default.
    #[test]
    fn the_slice_finds_the_whole_blocks_topology() {
        let builtins = [
            "sane_diode",
            "sane_mos",
            "sane_bjt",
            "sane_jfet",
            "sane_mesfet",
            "sane_vswitch",
            "sane_transformer",
            "sane_tline",
        ];
        let mut modules: Vec<Arc<ElaboratedModule>> = builtins
            .iter()
            .map(|n| builtin_module(n).expect("a builtin"))
            .collect();
        modules.extend(corpus());
        let sw = switching();
        let defaults = sw
            .params
            .iter()
            .map(|p| (p.name.clone(), p.default))
            .collect();
        let (shorts, switched) =
            topology(&sw, &defaults, &HashSet::default(), &HashSet::default()).expect("lowers");
        assert_eq!(
            (shorts.len(), switched.len()),
            (1, 1),
            "a short and a switch to find"
        );
        modules.push(sw);
        for em in &modules {
            let defaults: HashMap<String, f64> = em
                .params
                .iter()
                .map(|p| (p.name.clone(), p.default))
                .collect();
            let off: HashMap<String, f64> = (em.params.iter())
                .map(|p| match p.ty {
                    VarType::Integer => (p.name.clone(), p.default),
                    _ => (p.name.clone(), p.default * 1.5 + 0.1),
                })
                .collect();
            let all: HashSet<String> = em.params.iter().map(|p| p.name.clone()).collect();
            for (values, given) in [(&defaults, HashSet::default()), (&off, all)] {
                let whole = topology(em, values, &given, &HashSet::default());
                let sliced = topology(em, values, &given, &super::outside_slice(em));
                match (whole, sliced) {
                    (Ok(w), Ok(s)) => assert_eq!(w, s, "{}", em.name),
                    (Err(_), _) => {} // the module does not lower at this binding
                    (Ok(_), Err(e)) => panic!("{}: the slice fails: {e}", em.name),
                }
            }
        }
    }
}
