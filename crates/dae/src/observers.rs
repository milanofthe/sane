//! What a circuit's devices export besides their residuals, for the analyses
//! that look at a solved point: noise sources and operating-point variables.
//!
//! They are kept as the circuit is assembled, a tree: a part's own, in its
//! frame, and per subcircuit instance its body's, shared by every instance
//! of the body, with the instance's binding of the body's nodes. A circuit
//! of thousands of compact models exports millions of them, most of which
//! nobody asks for; they are carried into the top frame when asked for, the
//! body's expressions with the instance's names substituted, once.

use std::sync::{Arc, OnceLock};

use rsdag::{ExprId, SymbolId};
use rustc_hash::FxHashMap as HashMap;
use sane_core::Graph;
use sane_device::{NoiseSource, OpVar};

use crate::hierarchy::rename;

/// The noise sources and op-vars of a part: its own and its instances'.
#[derive(Clone, Default)]
pub struct Observers {
    pub noise: Vec<NoiseSource>,
    pub op_vars: Vec<OpVar>,
    instances: Vec<Observed>,
    /// Every one in this frame, once asked for.
    flat: OnceLock<Arc<Flat>>,
}

/// Noise sources and op-vars in one frame.
#[derive(Clone, Default)]
pub struct Flat {
    pub noise: Vec<NoiseSource>,
    pub op_vars: Vec<OpVar>,
}

/// A subcircuit instance's observers: its body's, and how the instance
/// binds the body's formal nodes.
#[derive(Clone)]
struct Observed {
    name: String,
    binding: Vec<(SymbolId, ExprId)>,
    body: Arc<(String, Observers)>,
}

impl Observers {
    /// Only the given ones, in this frame.
    pub fn flat(noise: Vec<NoiseSource>, op_vars: Vec<OpVar>) -> Self {
        Observers {
            noise,
            op_vars,
            ..Self::default()
        }
    }

    /// Whether neither this part nor any instance in it exports one.
    pub fn is_empty(&self) -> bool {
        self.noise.is_empty()
            && self.op_vars.is_empty()
            && self.instances.iter().all(|i| i.body.1.is_empty())
    }

    /// The observers of an instance `name` of a body in namespace `ns`, its
    /// formal nodes bound by `binding`.
    pub(crate) fn place(
        &mut self,
        name: &str,
        binding: Vec<(SymbolId, ExprId)>,
        body: &Arc<(String, Observers)>,
    ) {
        if !body.1.is_empty() {
            self.instances.push(Observed {
                name: name.to_string(),
                binding,
                body: body.clone(),
            });
        }
    }

    /// Every one in this frame: the instances' carried into it.
    pub fn flatten(&self, ctx: &mut Graph) -> Arc<Flat> {
        if let Some(f) = self.flat.get() {
            return f.clone();
        }
        let mut memo = HashMap::default();
        let f = flatten(self, ctx, &mut memo);
        self.flat.get_or_init(|| f).clone()
    }
}

/// [`Observers::flatten`], each body once.
fn flatten(
    obs: &Observers,
    ctx: &mut Graph,
    memo: &mut HashMap<*const (String, Observers), Arc<Flat>>,
) -> Arc<Flat> {
    if obs.instances.is_empty() {
        return Arc::new(Flat {
            noise: obs.noise.clone(),
            op_vars: obs.op_vars.clone(),
        });
    }
    let mut out = Flat {
        noise: obs.noise.clone(),
        op_vars: obs.op_vars.clone(),
    };
    for inst in &obs.instances {
        let key = Arc::as_ptr(&inst.body);
        let body = match memo.get(&key) {
            Some(b) => b.clone(),
            None => {
                let b = flatten(&inst.body.1, ctx, memo);
                memo.insert(key, b.clone());
                b
            }
        };
        inst.carry(ctx, &body, &mut out);
    }
    Arc::new(out)
}

impl Observed {
    /// The body's `body` (in its frame) into the parent's `out`: its formal
    /// nodes bound, its own names the instance's.
    fn carry(&self, ctx: &mut Graph, body: &Flat, out: &mut Flat) {
        let ns = &self.body.0;
        let exprs: Vec<ExprId> = body
            .noise
            .iter()
            .flat_map(|n| n.exprs())
            .chain(body.op_vars.iter().map(|o| o.value))
            .collect();
        let mut map: HashMap<SymbolId, ExprId> = self.binding.iter().copied().collect();
        let mut bind = |ctx: &mut Graph, s: SymbolId| -> ExprId {
            if let Some(&e) = map.get(&s) {
                return e;
            }
            let name = rename(ctx.symbol_name(s), &self.name, ns);
            let e = ctx.sym(&name);
            map.insert(s, e);
            e
        };
        for s in ctx.free_symbols_in(&exprs) {
            if ctx.symbol_name(s).contains(ns.as_str()) {
                bind(ctx, s);
            }
        }
        let mut sym = |ctx: &mut Graph, s: Option<SymbolId>| {
            let e = bind(ctx, s?);
            match *ctx.node(e) {
                rsdag::Node::Symbol(s) => Some(s),
                _ => None,
            }
        };
        let hilo: Vec<(Option<SymbolId>, Option<SymbolId>)> = body
            .noise
            .iter()
            .map(|n| (sym(ctx, n.hi), sym(ctx, n.lo)))
            .collect();
        let mut vals = rsdag::substitute(ctx, &exprs, &map).into_iter();
        for (n, (hi, lo)) in body.noise.iter().zip(hilo) {
            out.noise.push(n.with_exprs(hi, lo, &mut vals));
        }
        for (o, value) in body.op_vars.iter().zip(vals) {
            out.op_vars.push(OpVar {
                name: rename(&o.name, &self.name, ns),
                value,
                ..o.clone()
            });
        }
    }
}
