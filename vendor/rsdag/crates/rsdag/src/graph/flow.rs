//! The one structural analysis: a value per node of a cone, a leaf's from
//! the caller, every other node's the join of its operands', through a call
//! the join of the arguments its output reads. Support, dependence,
//! feedthrough and the sparsity of a Jacobian are this with a different
//! leaf and a different way through calls.
//!
//! What an output of a function reads is found once per function, per way
//! through and per set of its parameters that carry a value (the moving
//! ones): a call that passes a value in a few arguments of hundreds asks
//! only about those few.

use std::cell::RefCell;
use std::rc::Rc;

use super::*;

thread_local! {
    /// Position tables for the cones of this thread's analyses, reused.
    static MEMOS: RefCell<Vec<Memo>> = const { RefCell::new(Vec::new()) };
}

/// The operands a value passes through, and how a call passes it on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Through {
    /// Every operand; of a call, the arguments its output reads.
    Reads,
    /// The operands a derivative reads (not a comparison's, not a
    /// selector's condition); of a call, the arguments its output's
    /// derivative carries.
    Carries,
}

/// A join-semilattice value of an analysis.
pub(crate) trait Join: Clone {
    fn bottom() -> Self;
    fn is_bottom(&self) -> bool;
    fn join(&mut self, other: &Self);
    /// The join of `parts`, at once (a set merges them in one pass, not
    /// one operand after the other).
    fn join_all<'a>(parts: impl Iterator<Item = &'a Self>) -> Self
    where
        Self: 'a,
    {
        let mut v = Self::bottom();
        for p in parts {
            v.join(p);
        }
        v
    }
}

impl Join for bool {
    fn bottom() -> bool {
        false
    }
    fn is_bottom(&self) -> bool {
        !*self
    }
    fn join(&mut self, other: &bool) {
        *self |= *other;
    }
}

/// A set of small integers: indices of a universe of at most
/// [`BITS_MAX`] as a bitset (a union an `or` over a few words), of a wider
/// one sorted. A node with one contributing operand shares that operand's.
#[derive(Clone, Default, PartialEq, Debug)]
pub(crate) struct Set(Option<Ids>);

#[derive(Clone, PartialEq, Debug)]
enum Ids {
    Bits(Rc<[u64]>),
    Sorted(Rc<[u32]>),
}

/// The widest universe a [`Set`] keeps as a bitset.
pub(crate) const BITS_MAX: usize = 4096;

impl Set {
    /// `{k}`, of a universe of `n`.
    pub(crate) fn one(k: u32, n: usize) -> Set {
        if n <= BITS_MAX {
            let mut w = vec![0u64; k as usize / 64 + 1];
            w[k as usize / 64] = 1 << (k % 64);
            Set(Some(Ids::Bits(w.into())))
        } else {
            Set(Some(Ids::Sorted(Rc::from([k]))))
        }
    }

    /// Whether `k` is a member.
    pub(crate) fn contains(&self, k: u32) -> bool {
        match &self.0 {
            None => false,
            Some(Ids::Bits(w)) => w
                .get(k as usize / 64)
                .is_some_and(|&b| b >> (k % 64) & 1 == 1),
            Some(Ids::Sorted(v)) => v.binary_search(&k).is_ok(),
        }
    }

    /// The members, ascending.
    pub(crate) fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        let (bits, sorted): (&[u64], &[u32]) = match &self.0 {
            None => (&[], &[]),
            Some(Ids::Bits(w)) => (w, &[]),
            Some(Ids::Sorted(v)) => (&[], v),
        };
        let from_bits = bits.iter().enumerate().flat_map(|(i, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                (w != 0).then(|| {
                    let b = w.trailing_zeros();
                    w &= w - 1;
                    i as u32 * 64 + b
                })
            })
        });
        from_bits.chain(sorted.iter().copied())
    }
}

impl Join for Set {
    fn join_all<'a>(parts: impl Iterator<Item = &'a Set>) -> Set {
        let mut first: Option<&Set> = None;
        let (mut bits, mut sorted): (Vec<u64>, Vec<u32>) = (Vec::new(), Vec::new());
        let add = |p: &Set, bits: &mut Vec<u64>, sorted: &mut Vec<u32>| match &p.0 {
            Some(Ids::Bits(w)) => {
                if bits.len() < w.len() {
                    bits.resize(w.len(), 0);
                }
                bits.iter_mut().zip(w.iter()).for_each(|(a, &b)| *a |= b);
            }
            Some(Ids::Sorted(v)) => sorted.extend_from_slice(v),
            None => {}
        };
        let mut many = false;
        for p in parts.filter(|p| !p.is_bottom()) {
            match first {
                None => first = Some(p),
                Some(f) => {
                    if !many {
                        add(f, &mut bits, &mut sorted);
                        many = true;
                    }
                    add(p, &mut bits, &mut sorted);
                }
            }
        }
        if !many {
            // none, or one set: shared as it is
            return first.cloned().unwrap_or_default();
        }
        if sorted.is_empty() {
            return Set(Some(Ids::Bits(bits.into())));
        }
        // one flow takes one kind; mixed, the bits join the list
        sorted.extend(Set(Some(Ids::Bits(bits.into()))).iter());
        sorted.sort_unstable();
        sorted.dedup();
        Set(Some(Ids::Sorted(sorted.into())))
    }
    fn bottom() -> Set {
        Set(None)
    }
    fn is_bottom(&self) -> bool {
        self.0.is_none()
    }
    fn join(&mut self, other: &Set) {
        if other.is_bottom() {
            return;
        }
        *self = Set::join_all([&*self, other].into_iter());
    }
}

/// Of a comparison its operands, of a selector its condition, carry no
/// derivative: the leading operands [`Through::Carries`] skips.
fn inert(node: &Node) -> usize {
    match node {
        Node::Cmp(..) => 2,
        Node::Select(..) => 1,
        _ => 0,
    }
}

/// Values over a cone (see [`Graph::flow`]).
pub(crate) struct Flow<V> {
    at: Memo,
    vals: Vec<V>,
    /// The nodes of the cone, in the order of `vals`.
    nodes: Vec<ExprId>,
}

impl<V> Drop for Flow<V> {
    fn drop(&mut self) {
        let at = std::mem::take(&mut self.at);
        MEMOS.with(|m| m.borrow_mut().push(at));
    }
}

impl<V> Flow<V> {
    /// The value of a node of the cone.
    pub(crate) fn get(&self, e: ExprId) -> &V {
        &self.vals[self.at.get(e).expect("a node of the cone").0 as usize]
    }

    /// Every node of the cone with its value.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (ExprId, &V)> {
        self.nodes.iter().copied().zip(&self.vals)
    }
}

impl<K: Field> Graph<K> {
    /// The nodes under `roots`, every operand and every argument list once,
    /// in ascending id order (a topological one: a hash-consed node is
    /// newer than its operands), with each node's position in `at`.
    pub(crate) fn cone(&self, roots: &[ExprId], at: &mut Memo) -> Vec<ExprId> {
        let mut out = self.reach(roots, at, true);
        out.sort_unstable();
        for (k, &e) in out.iter().enumerate() {
            at.set(e, ExprId(k as u32));
        }
        out
    }

    /// The nodes under `roots`, every operand and every argument list once,
    /// in the order the walk meets them, marked in `at`; with `globals`, a
    /// call's globals as its operands too (what it reads, not only what it
    /// mentions).
    fn reach(&self, roots: &[ExprId], at: &mut Memo, globals: bool) -> Vec<ExprId> {
        at.begin(self.len());
        let mut lists: FxHashSet<ArgList> = FxHashSet::default();
        let mut funcs: FxHashSet<FuncId> = FxHashSet::default();
        let mut contexts: FxHashSet<u32> = FxHashSet::default();
        let mut stack: Vec<ExprId> = roots.to_vec();
        let mut out = Vec::new();
        while let Some(e) = stack.pop() {
            if at.get(e).is_some() {
                continue;
            }
            at.set(e, e);
            out.push(e);
            match *self.node(e) {
                Node::Call(o, l) => {
                    if lists.insert(l) {
                        stack.extend_from_slice(self.args(l));
                    }
                    let f = self.output(o).0;
                    if globals && funcs.insert(f) {
                        stack.extend_from_slice(&self.globals(f));
                    }
                    let c = self.output_ctx[o.0 as usize];
                    if c != NO_CONTEXT && contexts.insert(c) {
                        stack.extend_from_slice(self.args(self.contexts[c as usize].exprs));
                    }
                }
                _ => stack.extend_from_slice(&self.operands(e)),
            }
        }
        out
    }

    /// The symbol nodes `f`'s body reads that are not its parameters, its
    /// calls' included, ascending: the globals every call of `f` reads
    /// besides its arguments. A call's operand `p` is its argument `p`, and
    /// past the arguments global `p - arity` (see
    /// [`operand_symbol`](Self::operand_symbol)).
    pub fn globals(&self, f: FuncId) -> Arc<[ExprId]> {
        self.funcs[f.0 as usize].globals_in(self)
    }

    /// The symbol a call of `f` binds by its operand `p`: parameter `p`, or
    /// past the parameters a global (see [`globals`](Self::globals)).
    pub fn operand_symbol(&self, f: FuncId, p: u32) -> SymbolId {
        let params = self.funcs[f.0 as usize].params();
        match params.get(p as usize) {
            Some(&s) => s,
            None => match *self.node(self.globals(f)[p as usize - params.len()]) {
                Node::Symbol(s) => s,
                _ => unreachable!("a global is a symbol"),
            },
        }
    }

    /// Operand `p` of a call of output `o` over the list `l`: in its
    /// function's parameter order its argument or, where its context binds
    /// the parameter, the bound expression; past the parameters a global.
    pub fn call_operand(&self, o: OutputId, l: ArgList, p: u32) -> ExprId {
        let f = self.output(o).0;
        let n = self.funcs[f.0 as usize].params().len();
        if p as usize >= n {
            return self.globals_of(f)[p as usize - n];
        }
        let args = self.args(l);
        match self.output_ctx[o.0 as usize] {
            NO_CONTEXT => args[p as usize],
            c => {
                let c = &self.contexts[c as usize];
                match c.slot[p as usize] {
                    s if s & BOUND == 0 => args[s as usize],
                    s => self.args(c.exprs)[(s & !BOUND) as usize],
                }
            }
        }
    }

    /// [`globals`](Self::globals), borrowed.
    pub(crate) fn globals_of(&self, f: FuncId) -> &[ExprId] {
        let func = &self.funcs[f.0 as usize];
        if func.globals.get().is_none() {
            func.globals_in(self);
        }
        func.globals.get().expect("just found")
    }

    /// The nodes under `roots`, ascending; with `globals` the globals the
    /// calls read (see [`cone`](Self::cone)).
    pub(crate) fn cone_sorted(&self, roots: &[ExprId], globals: bool) -> Vec<ExprId> {
        let mut cone = self.cone_nodes(roots, globals);
        cone.sort_unstable();
        cone
    }

    /// The nodes under `roots`, unordered; with `globals` the globals the
    /// calls read (see [`cone`](Self::cone)).
    pub(crate) fn cone_nodes(&self, roots: &[ExprId], globals: bool) -> Vec<ExprId> {
        let mut at = MEMOS.with(|m| m.borrow_mut().pop()).unwrap_or_default();
        let cone = self.reach(roots, &mut at, globals);
        MEMOS.with(|m| m.borrow_mut().push(at));
        cone
    }

    /// Values over the cone of `roots`: a constant's and a symbol's from
    /// `leaf`, every other node's the join of its operands' as `through`
    /// passes them, a call's the join of the arguments its output reads
    /// (see [`reads`](Self::reads)).
    pub(crate) fn flow<V: Join>(
        &self,
        roots: &[ExprId],
        through: Through,
        leaf: impl Fn(&Node) -> V,
    ) -> Flow<V> {
        let mut at = MEMOS.with(|m| m.borrow_mut().pop()).unwrap_or_default();
        let cone = self.cone(roots, &mut at);
        let mut vals: Vec<V> = Vec::with_capacity(cone.len());
        // per instance, what each output reads among its moving arguments
        let mut sites: HashMap<(FuncId, u32, ArgList), Arc<[Arc<[u32]>]>> = HashMap::default();
        for &e in &cone {
            let val = |c: &ExprId| &vals[at.get(*c).expect("in the cone").0 as usize];
            let node = *self.node(e);
            let v = match node {
                Node::Const(_) | Node::Symbol(_) => leaf(&node),
                Node::Call(o, l) => {
                    let (f, k) = self.output(o);
                    let (args, globals) = (self.full_args(o, l), self.globals(f));
                    let operand = |p: u32| match args.get(p as usize) {
                        Some(a) => a,
                        None => &globals[p as usize - args.len()],
                    };
                    let site = (f, self.output_ctx[o.0 as usize], l);
                    let reads = match sites.get(&site) {
                        Some(r) => r.clone(),
                        None => {
                            let moving: Vec<u32> = (0..(args.len() + globals.len()) as u32)
                                .filter(|&p| !val(operand(p)).is_bottom())
                                .collect();
                            let r = self.reads(f, through, &moving);
                            sites.insert(site, r.clone());
                            r
                        }
                    };
                    let read = reads.get(k as usize).map_or(&[][..], |r| &r[..]);
                    V::join_all(read.iter().map(|&p| val(operand(p))))
                }
                _ => {
                    let ops = self.operands(e);
                    let skip = if through == Through::Carries {
                        inert(&node)
                    } else {
                        0
                    };
                    V::join_all(ops[skip..].iter().map(val))
                }
            };
            vals.push(v);
        }
        Flow {
            at,
            vals,
            nodes: cone,
        }
    }

    /// Per output of `f`, the parameters among `moving` (indices, ascending)
    /// it reads through `through`, ascending. An extern output reads every
    /// one, a zero output none. Kept per function, way through and moving
    /// set; over all parameters with [`Through::Carries`] it is
    /// [`output_support`](Self::output_support).
    pub(crate) fn reads(&self, f: FuncId, through: Through, moving: &[u32]) -> Arc<[Arc<[u32]>]> {
        let func = &self.funcs[f.0 as usize];
        let outputs = func.outputs();
        let key = (through, Box::<[u32]>::from(moving));
        // outputs are only ever appended (derivatives): the known ones stay
        let known = func.cached_reads(&key);
        let done = known.as_ref().map_or(0, |r| r.len());
        if done == outputs.len() {
            return known.expect("all outputs known");
        }
        let index: HashMap<SymbolId, u32> = moving
            .iter()
            .map(|&p| (self.operand_symbol(f, p), p))
            .collect();
        let universe = moving.iter().max().map_or(0, |&p| p as usize + 1);
        let exprs: Vec<ExprId> = outputs[done..]
            .iter()
            .filter_map(|o| match *o {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        let found = (!moving.is_empty() && !exprs.is_empty()).then(|| {
            self.flow(&exprs, through, |n| match *n {
                Node::Symbol(s) => index
                    .get(&s)
                    .map_or(Set::bottom(), |&p| Set::one(p, universe)),
                _ => Set::bottom(),
            })
        });
        let fresh = outputs[done..].iter().map(|o| match *o {
            Output::Zero => Arc::from([]),
            Output::Slot(_) => Arc::from(moving),
            Output::Expr(e) => found
                .as_ref()
                .map_or(Arc::from([]), |fl| fl.get(e).iter().collect()),
        });
        let r: Arc<[Arc<[u32]>]> = known
            .iter()
            .flat_map(|k| k.iter().cloned())
            .chain(fresh)
            .collect();
        func.cache_reads(key, r.clone());
        r
    }
}
