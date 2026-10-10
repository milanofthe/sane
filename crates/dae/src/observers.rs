//! What a circuit's devices export besides their residuals, for the analyses
//! that look at a solved point: noise sources and operating-point variables.
//!
//! They are kept as the circuit is assembled, a tree: a part's own, in its
//! frame, and per subcircuit instance its body's, shared by every instance
//! of the body, with what the instance binds the body's symbols to. A body's
//! own are the outputs of one function over the body's symbols (the noise
//! levels and the op-vars, by role). Asked for, the tree resolves into
//! sites: per instance whose body has its own, that function with its
//! arguments in the top frame. A site's observers are calls of it, so no
//! body's expressions are copied per instance, and what they read is the
//! site's arguments and its function's globals. A circuit of thousands of
//! compact models exports millions of op-vars, which only an op-var query
//! reads; the noise sources and the op-vars are made apart, each once
//! asked for.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use rsdag::{ExprId, FuncId, Node, OutputRole, SymbolId};
use rustc_hash::FxHashMap as HashMap;
use sane_core::Graph;
use sane_device::{NoiseSource, OpVar};

use sane_circuit::rename;

/// The noise sources and op-vars of a part: its own and its instances'.
#[derive(Clone, Default)]
pub struct Observers {
    pub noise: Vec<NoiseSource>,
    pub op_vars: Vec<OpVar>,
    instances: Vec<Observed>,
    /// The instances' as sites in this frame, once asked for.
    sites: OnceLock<Arc<[Site]>>,
    /// Every noise source in this frame, once asked for.
    all_noise: OnceLock<Arc<[NoiseSource]>>,
    /// Every op-var in this frame, once asked for.
    all_op_vars: OnceLock<Arc<[OpVar]>>,
}

/// A subcircuit body's observers, shared by its instances.
pub(crate) struct Body {
    /// The body's namespace (`__name__.`).
    ns: String,
    own: Option<Own>,
    /// Its instances', in its frame.
    instances: Vec<Observed>,
}

/// A body's own observers, in its frame, and the function they are
/// outputs of: from `first` on, the noise sources' expressions source by
/// source, then the op-vars' values (see [`Observers::exprs`]).
struct Own {
    func: FuncId,
    leaves: Vec<SymbolId>,
    first: u32,
    noise: Vec<NoiseSource>,
    /// How many outputs the noise sources take, before the op-vars'.
    n_noise: u32,
    op_vars: Vec<OpVar>,
}

/// A subcircuit instance's observers: its body's, and what the instance
/// binds the body's symbols to in its parent's frame (the formal nodes,
/// and the ones it found assembling its calls).
#[derive(Clone)]
struct Observed {
    name: String,
    binding: Arc<HashMap<SymbolId, ExprId>>,
    body: Arc<Body>,
}

/// A body's own observers placed in a frame: the function a call of them
/// runs (the body's, or a copy a rewrite bound), its arguments there, the
/// noise generators there, and what the body's namespace reads as there
/// (`x1.x5.`).
#[derive(Clone)]
struct Site {
    body: Arc<Body>,
    func: FuncId,
    args: Vec<ExprId>,
    inputs: Vec<SymbolId>,
    prefix: String,
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
            && self.instances.is_empty()
            && self.sites.get().is_none_or(|s| s.is_empty())
    }

    /// The observers of an instance `name` of a body, the body's symbols
    /// bound by `binding` (the formal nodes at least).
    pub(crate) fn place(
        &mut self,
        name: &str,
        binding: HashMap<SymbolId, ExprId>,
        body: &Arc<Body>,
    ) {
        if body.own.is_some() || !body.instances.is_empty() {
            self.instances.push(Observed {
                name: name.to_string(),
                binding: Arc::new(binding),
                body: body.clone(),
            });
        }
    }

    /// Its own expressions: the noise sources' source by source, then the
    /// op-vars' values.
    pub(crate) fn exprs(&self) -> Vec<ExprId> {
        (self.noise.iter().flat_map(|n| n.exprs()))
            .chain(self.op_vars.iter().map(|v| v.value))
            .collect()
    }

    /// These, a subcircuit body's in its namespace `ns`: its own
    /// [`exprs`](Self::exprs) outputs of `func` (over `leaves`) from `first`
    /// on, given their roles there. `None` for none of its own.
    pub(crate) fn into_body(
        self,
        ctx: &mut Graph,
        ns: &str,
        func: Option<(FuncId, Vec<SymbolId>, u32)>,
    ) -> Arc<Body> {
        let own = func.map(|(func, leaves, first)| {
            let mut out = first;
            for (id, n) in self.noise.iter().enumerate() {
                for elem in 0..n.exprs().len() as u32 {
                    let id = id as u32;
                    ctx.set_output_role(func, out, OutputRole::NoiseLevel { id, elem });
                    out += 1;
                }
            }
            let n_noise = out - first;
            for id in 0..self.op_vars.len() as u32 {
                ctx.set_output_role(func, out, OutputRole::Observer { id });
                out += 1;
            }
            Own {
                func,
                leaves,
                first,
                noise: self.noise,
                n_noise,
                op_vars: self.op_vars,
            }
        });
        Arc::new(Body {
            ns: ns.to_string(),
            own,
            instances: self.instances,
        })
    }

    /// Every noise source in this frame: its own, then the sites' in order.
    pub fn noise(&self, ctx: &mut Graph) -> Arc<[NoiseSource]> {
        if let Some(n) = self.all_noise.get() {
            return n.clone();
        }
        let mut out = self.noise.clone();
        for site in self.sites(ctx).iter() {
            let own = site.own();
            let exprs = own.noise.iter().flat_map(|n| n.exprs());
            let mut vals = site.values(ctx, own.first, exprs).into_iter();
            for (n, &input) in own.noise.iter().zip(&site.inputs) {
                out.push(n.with_exprs(input, &mut vals));
            }
        }
        self.all_noise.get_or_init(|| out.into()).clone()
    }

    /// The noise generators, in the order of [`noise`](Self::noise),
    /// without the sources' levels.
    pub fn generators(&self, ctx: &mut Graph) -> Vec<SymbolId> {
        let own = self.noise.iter().map(|n| n.input);
        own.chain(
            self.sites(ctx)
                .iter()
                .flat_map(|s| s.inputs.iter().copied()),
        )
        .collect()
    }

    /// Every op-var in this frame: its own, then the sites' in order.
    pub fn op_vars(&self, ctx: &mut Graph) -> Arc<[OpVar]> {
        if let Some(v) = self.all_op_vars.get() {
            return v.clone();
        }
        let mut out = self.op_vars.clone();
        for site in self.sites(ctx).iter() {
            let own = site.own();
            let exprs = own.op_vars.iter().map(|v| v.value);
            let vals = site.values(ctx, own.first + own.n_noise, exprs);
            for (v, value) in own.op_vars.iter().zip(vals) {
                out.push(OpVar {
                    name: replace_ns(&v.name, &site.body.ns, &site.prefix),
                    short: v.short.clone(),
                    desc: v.desc.clone(),
                    units: v.units.clone(),
                    value,
                });
            }
        }
        self.all_op_vars.get_or_init(|| out.into()).clone()
    }

    /// Expressions whose free symbols are what the observers read: their
    /// own, and per site its arguments and its function's globals.
    pub fn reads(&self, ctx: &mut Graph) -> Vec<ExprId> {
        let mut out: Vec<ExprId> = (self.noise.iter().flat_map(|n| n.exprs()))
            .chain(self.op_vars.iter().map(|v| v.value))
            .collect();
        let mut funcs = HashSet::new();
        for site in self.sites(ctx).iter() {
            out.extend_from_slice(&site.args);
            if funcs.insert(site.func) {
                out.extend(ctx.globals(site.func).iter().copied());
            }
        }
        out
    }

    /// These with `subst` (a symbol of this frame to what it is now)
    /// carried in, as [`rsdag::substitute`] carries it into a call: their
    /// own expressions substituted, and per site its arguments, its
    /// function rebound where the substitution reaches into its body. The
    /// noise generators stay.
    pub(crate) fn rewrite(&self, ctx: &mut Graph, subst: &HashMap<SymbolId, ExprId>) -> Observers {
        if subst.is_empty() {
            return self.clone();
        }
        let sites = self.sites(ctx);
        let roots: Vec<ExprId> = (self.noise.iter().flat_map(|n| n.exprs()))
            .chain(self.op_vars.iter().map(|v| v.value))
            .chain(sites.iter().flat_map(|s| s.args.iter().copied()))
            .collect();
        let mut new = rsdag::substitute(ctx, &roots, subst).into_iter();
        let noise = (self.noise.iter())
            .map(|n| n.with_exprs(n.input, &mut new))
            .collect();
        let mut next = || new.next().expect("one per expression carried");
        let op_vars = (self.op_vars.iter())
            .map(|v| OpVar {
                value: next(),
                ..v.clone()
            })
            .collect();
        let mut copies: HashMap<FuncId, FuncId> = HashMap::default();
        let sites: Vec<Site> = (sites.iter())
            .map(|s| {
                let func = *copies
                    .entry(s.func)
                    .or_insert_with(|| ctx.rebound(s.func, subst).unwrap_or(s.func));
                Site {
                    func,
                    args: s.args.iter().map(|_| next()).collect(),
                    ..s.clone()
                }
            })
            .collect();
        Observers {
            noise,
            op_vars,
            sites: OnceLock::from(Arc::from(sites)),
            ..Self::default()
        }
    }

    /// The instances' bodies with their own, as sites in this frame.
    fn sites(&self, ctx: &mut Graph) -> Arc<[Site]> {
        if let Some(s) = self.sites.get() {
            return s.clone();
        }
        let mut out = Vec::new();
        let mut frames = Frames::default();
        for inst in &self.instances {
            frames.visit(ctx, inst, &mut out);
        }
        self.sites.get_or_init(|| out.into()).clone()
    }
}

impl Site {
    fn own(&self) -> &Own {
        self.body.own.as_ref().expect("a site's body has its own")
    }

    /// The body's own expressions `exprs`, its function's outputs from
    /// `first` on, at this site: a constant as itself, else a call.
    fn values(
        &self,
        ctx: &mut Graph,
        first: u32,
        exprs: impl Iterator<Item = ExprId>,
    ) -> Vec<ExprId> {
        let exprs: Vec<(ExprId, bool)> = exprs.map(|e| (e, ctx.const_f64(e).is_some())).collect();
        let outs: Vec<u32> = (exprs.iter().enumerate())
            .filter(|(_, &(_, konst))| !konst)
            .map(|(k, _)| first + k as u32)
            .collect();
        let mut calls = ctx.calls(self.func, &outs, &self.args).into_iter();
        (exprs.into_iter())
            .map(|(e, konst)| {
                if konst {
                    e
                } else {
                    calls.next().expect("one per output")
                }
            })
            .collect()
    }
}

/// The instances from the top frame down to the body being visited, each
/// with what its body's namespace reads as in the top frame and the body's
/// symbols found there so far.
#[derive(Default)]
struct Frames<'a> {
    stack: Vec<Frame<'a>>,
}

struct Frame<'a> {
    inst: &'a Observed,
    prefix: String,
    found: HashMap<SymbolId, ExprId>,
}

impl<'a> Frames<'a> {
    /// The sites of `inst` and of the instances in its body, in order.
    fn visit(&mut self, ctx: &mut Graph, inst: &'a Observed, out: &mut Vec<Site>) {
        let name = match self.stack.last() {
            None => inst.name.clone(),
            Some(f) => replace_ns(&inst.name, &f.inst.body.ns, &f.prefix),
        };
        self.stack.push(Frame {
            inst,
            prefix: format!("{name}."),
            found: HashMap::default(),
        });
        let body = &inst.body;
        if let Some(own) = &body.own {
            let level = self.stack.len() - 1;
            let mut args = Vec::with_capacity(own.leaves.len());
            for &s in &own.leaves {
                args.push(self.find(ctx, level, s));
            }
            let mut inputs = Vec::with_capacity(own.noise.len());
            for n in &own.noise {
                let e = self.find(ctx, level, n.input);
                inputs.push(match *ctx.node(e) {
                    Node::Symbol(s) => s,
                    _ => unreachable!("a noise generator binds to a symbol"),
                });
            }
            out.push(Site {
                body: body.clone(),
                func: own.func,
                args,
                inputs,
                prefix: self.stack[level].prefix.clone(),
            });
        }
        for nested in &body.instances {
            self.visit(ctx, nested, out);
        }
        self.stack.pop();
    }

    /// Symbol `s` of the body at `level` in the top frame: in the frame of
    /// the instance's parent what the instance binds it to, else its name
    /// in the instance's namespace; that in turn found there.
    fn find(&mut self, ctx: &mut Graph, level: usize, s: SymbolId) -> ExprId {
        if let Some(&e) = self.stack[level].found.get(&s) {
            return e;
        }
        let inst = self.stack[level].inst;
        let e = match inst.binding.get(&s) {
            Some(&e) => e,
            None if ctx.symbol_name(s).contains(inst.body.ns.as_str()) => {
                let name = rename(ctx.symbol_name(s), &inst.name, &inst.body.ns);
                ctx.sym(&name)
            }
            None => ctx.symbol_expr(s),
        };
        let e = match level {
            0 => e,
            _ => self.carry(ctx, level - 1, e),
        };
        self.stack[level].found.insert(s, e);
        e
    }

    /// Expression `e` in the frame of the body at `level` in the top frame.
    fn carry(&mut self, ctx: &mut Graph, level: usize, e: ExprId) -> ExprId {
        match *ctx.node(e) {
            Node::Symbol(s) => self.find(ctx, level, s),
            Node::Const(_) => e,
            _ => {
                let map: HashMap<SymbolId, ExprId> = (ctx.free_symbols_in(&[e]).into_iter())
                    .map(|s| (s, self.find(ctx, level, s)))
                    .collect();
                rsdag::substitute(ctx, &[e], &map)[0]
            }
        }
    }
}

/// `name` of a body in namespace `ns` with the namespace read as `prefix`
/// (`x1.x5.`): [`rename`] through every instance down to the body at once.
fn replace_ns(name: &str, ns: &str, prefix: &str) -> String {
    match name.find(ns) {
        Some(i) if !ns.is_empty() => format!("{}{prefix}{}", &name[..i], &name[i + ns.len()..]),
        _ => name.to_string(),
    }
}
