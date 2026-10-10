//! A DAE rewritten: the one way a transform makes a DAE of another.
//!
//! A transform -- folding parameters, merging nodes, eliminating them,
//! pruning branches, linearizing -- gives new rows over the unknowns it keeps
//! and says what each symbol is now. [`Dae::rewrite`] carries everything else
//! the DAE holds through that one substitution: the switching surfaces, the
//! transport delays (the signal each delays and its delay), the noise
//! sources and operating-point variables, the assertions, the Newton aids
//! (onto the unknowns that remain), the node-sets, the source shapes. So no
//! transform drops what it does not touch.

use std::collections::HashSet;

use rsdag::{ExprId, Node, SymbolId};
use rustc_hash::FxHashMap;
use sane_core::Graph;

use crate::{Dae, DelaySpec, EventSpec};

/// A rewrite of a DAE (see [`Dae::rewrite`]).
pub(crate) struct Rewrite {
    /// The rows anew, over the unknowns `keep`; `None` for the rows through
    /// the substitution.
    pub rows: Option<(Vec<ExprId>, Vec<ExprId>)>,
    /// The old index of each unknown kept, in the new order.
    pub keep: Vec<usize>,
    /// How many of the kept unknowns lead as node voltages.
    pub n_nodes: usize,
    /// What a symbol is now: a merged or eliminated node's voltage, a
    /// folded parameter, an unknown frozen at its operating point.
    pub subst: FxHashMap<SymbolId, ExprId>,
}

impl Dae {
    /// This DAE rewritten as `rw` says (see the module docs).
    pub(crate) fn rewrite(&self, ctx: &mut Graph, rw: Rewrite) -> Dae {
        let sub = |ctx: &mut Graph, roots: Vec<ExprId>| {
            if rw.subst.is_empty() {
                roots
            } else {
                rsdag::substitute(ctx, &roots, &rw.subst)
            }
        };
        let (currents, charges) = match rw.rows {
            Some(rows) => rows,
            None => {
                let n = self.currents.len();
                let mut rows = sub(
                    ctx,
                    self.currents.iter().chain(&self.charges).copied().collect(),
                );
                let charges = rows.split_off(n);
                (rows, charges)
            }
        };

        // Everything else the DAE carries, in one pass.
        let label_keys: Vec<ExprId> = self.labels.keys().copied().collect();
        let roots: Vec<ExprId> = (self.events.iter().map(|e| e.g))
            .chain(self.delays.iter().flat_map(|d| [d.src, d.tau]))
            .chain(
                self.assertions
                    .iter()
                    .chain(&self.structure)
                    .map(|a| a.holds),
            )
            .chain(self.limits.iter().filter_map(|l| l.when))
            .chain(label_keys.iter().copied())
            .collect();
        let mut new = sub(ctx, roots).into_iter();
        let mut next = || new.next().expect("one per expression carried");
        let events = (self.events.iter())
            .map(|e| EventSpec {
                g: next(),
                ..e.clone()
            })
            .collect();
        let delays = (self.delays.iter())
            .map(|d| DelaySpec {
                src: next(),
                tau: next(),
                hist: d.hist,
            })
            .collect();
        let mut carried = |list: &[sane_device::Assertion]| -> Vec<sane_device::Assertion> {
            (list.iter())
                .map(|a| sane_device::Assertion {
                    holds: next(),
                    message: a.message.clone(),
                })
                .collect()
        };
        let assertions = carried(&self.assertions);
        let structure = carried(&self.structure);
        let whens: Vec<Option<ExprId>> =
            self.limits.iter().map(|l| l.when.map(|_| next())).collect();
        let labels = (label_keys.iter())
            .map(|k| (next(), self.labels[k].clone()))
            .collect();

        // The unknowns kept; what an unknown is now, where the Newton aids
        // name one: itself, the unknown it was merged onto, ground, or gone
        // (eliminated into an expression).
        let x: Vec<SymbolId> = rw.keep.iter().map(|&i| self.x[i]).collect();
        let unknowns: Vec<String> = rw.keep.iter().map(|&i| self.unknowns[i].clone()).collect();
        let kept: HashSet<SymbolId> = x.iter().copied().collect();
        let onto = |s: SymbolId| -> Option<Option<SymbolId>> {
            let e = match rw.subst.get(&s) {
                None => return kept.contains(&s).then_some(Some(s)),
                Some(&e) => e,
            };
            match *ctx.node(e) {
                Node::Symbol(t) if kept.contains(&t) => Some(Some(t)),
                _ if ctx.const_f64(e) == Some(0.0) => Some(None),
                _ => None,
            }
        };
        let limits = (self.limits.iter().zip(whens))
            .filter_map(|(l, when)| {
                let side = |s: Option<SymbolId>| match s {
                    None => Some(None),
                    Some(s) => onto(s),
                };
                Some(sane_device::FragmentLimit {
                    hi: side(l.hi)?,
                    lo: side(l.lo)?,
                    kind: l.kind,
                    when,
                })
            })
            .collect();
        let companion = (self.companion.iter())
            .filter_map(|&(r, c, g)| Some((onto(r)??, onto(c)??, g)))
            .collect();
        // a node merged onto another, or onto ground, is an alias of it
        let mut aliases = self.aliases.clone();
        let mut gone = vec![true; self.dim()];
        for &i in &rw.keep {
            gone[i] = false;
        }
        let name_of: FxHashMap<SymbolId, &String> = x.iter().copied().zip(&unknowns).collect();
        for (i, name) in self.unknowns.iter().enumerate() {
            if !gone[i] {
                continue;
            }
            if let Some(onto) = onto(self.x[i]) {
                aliases.push((name.clone(), onto.map(|s| name_of[&s].clone())));
            }
        }
        let names: HashSet<&String> = unknowns.iter().collect();
        let dc_seeds = (self.dc_seeds.iter())
            .filter(|(name, _)| names.contains(name))
            .cloned()
            .collect();
        Dae {
            currents,
            charges,
            kinds: rw.keep.iter().map(|&i| self.kinds[i]).collect(),
            unknowns,
            x,
            t: self.t,
            n_nodes: rw.n_nodes,
            param_defaults: self.param_defaults.clone(),
            events,
            delays,
            companion,
            observers: self.observers.rewrite(ctx, &rw.subst),
            dc_seeds,
            limits,
            sources: self.sources.clone(),
            source_names: self.source_names.clone(),
            assertions,
            structure,
            aliases,
            labels,
            injection: Default::default(),
            rest: Default::default(),
        }
    }
}
