//! Topological index detection for charge/flux-oriented MNA.
//!
//! Charge-oriented MNA is an index-1 DAE for most circuits, but two topologies
//! push it to index 2 (Estévez-Schwarz & Tischendorf):
//!
//! * a **CV loop** -- a loop of capacitors containing at least one voltage
//!   source. The loop pins the capacitor voltages algebraically, so the branch
//!   current follows the *derivative* of the source: `i = C·dv/dt`.
//! * an **LI cutset** -- a cutset of inductors containing at least one current
//!   source. Dual statement: the node voltage follows `v = L·di/dt`.
//!
//! Both are ordinary things to draw (a decoupling capacitor straight across an
//! ideal supply is a CV loop), and both change what the integrator can do: the
//! hidden constraint carries no local truncation error, so the step-size
//! controller sees nothing to control and strides far past the accuracy the
//! user asked for. The detector exists so the engine can say which elements are
//! responsible and bound its step accordingly, instead of returning a confident
//! wrong answer.
//!
//! Detection is the standard spanning-forest argument, and it is what a SPICE
//! engine can afford: near-linear in the element count, no numerics involved.

use std::collections::HashMap;

use crate::{Circuit, Kind};

/// One offending topology: the elements that form it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Index2Path {
    /// The source that makes it index 2 (voltage source in a CV loop, current
    /// source in an LI cutset).
    pub source: String,
    /// The storage elements completing it (capacitors / inductors), in path
    /// order for a loop, arbitrary order for a cutset.
    pub storage: Vec<String>,
}

impl Index2Path {
    /// `"V1-C1-C2"`, for a diagnostic line.
    pub fn describe(&self) -> String {
        std::iter::once(self.source.as_str())
            .chain(self.storage.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join("-")
    }
}

/// What the topology says about the DAE index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Index2Report {
    /// Loops of capacitors closed by a voltage source.
    pub cv_loops: Vec<Index2Path>,
    /// Cutsets of inductors bridged by a current source.
    pub li_cutsets: Vec<Index2Path>,
}

impl Index2Report {
    pub fn is_index2(&self) -> bool {
        !self.cv_loops.is_empty() || !self.li_cutsets.is_empty()
    }
    /// A one-line summary naming the elements, for a warning.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.cv_loops.is_empty() {
            parts.push(format!(
                "capacitor/voltage-source loop ({})",
                join_paths(&self.cv_loops)
            ));
        }
        if !self.li_cutsets.is_empty() {
            parts.push(format!(
                "inductor/current-source cutset ({})",
                join_paths(&self.li_cutsets)
            ));
        }
        parts.join(", ")
    }
}

fn join_paths(ps: &[Index2Path]) -> String {
    ps.iter()
        .map(|p| p.describe())
        .collect::<Vec<_>>()
        .join("; ")
}

/// Voltage-defining elements: their branch voltage is prescribed, so they close
/// a capacitor loop the same way an independent source does.
fn defines_voltage(k: Kind) -> bool {
    matches!(k, Kind::VoltageSource | Kind::Vcvs | Kind::Ccvs)
}
/// Current-defining elements: dual of the above for cutsets.
fn defines_current(k: Kind) -> bool {
    matches!(k, Kind::CurrentSource | Kind::Vccs | Kind::Cccs)
}

/// Disjoint-set over node indices.
struct Forest {
    parent: Vec<usize>,
}
impl Forest {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, mut a: usize) -> usize {
        while self.parent[a] != a {
            self.parent[a] = self.parent[self.parent[a]];
            a = self.parent[a];
        }
        a
    }
    /// Returns false when the two were already connected (the edge closes a loop).
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.parent[ra] = rb;
        true
    }
}

/// Detect the index-2 topologies of a circuit.
///
/// `device_terminals` carries the nonlinear devices (transistors, diodes): they
/// live outside the linear element list but are still branches, so they break
/// cutsets. Omitting them makes every node reachable only through a transistor
/// look isolated -- which reports a phantom cutset on essentially every analog
/// deck. They never *create* one of these topologies: a device is neither an
/// ideal capacitor nor an ideal source.
pub fn detect(circuit: &Circuit, device_terminals: &[Vec<usize>]) -> Index2Report {
    // a node that appears only on a device terminal never raised the element
    // graph's node count, so size the forests over both
    let n = device_terminals
        .iter()
        .flatten()
        .copied()
        .max()
        .unwrap_or(0)
        .max(circuit.node_count())
        + 1;
    let elems = circuit.elements();
    let mut report = Index2Report::default();

    // --- CV loops: grow a forest of capacitors, then close it with sources ---
    // A voltage source whose endpoints the capacitors already connect closes a
    // loop; the capacitors on the path between them are its other members. A
    // source that does NOT close one joins the forest, so a second source in
    // series with it is judged against the union (two sources plus capacitors
    // still form one loop).
    let mut cap_adj: HashMap<usize, Vec<(usize, &str)>> = HashMap::new();
    let mut forest = Forest::new(n);
    for e in elems.iter().filter(|e| e.kind == Kind::Capacitor) {
        cap_adj.entry(e.a).or_default().push((e.b, &e.name));
        cap_adj.entry(e.b).or_default().push((e.a, &e.name));
        forest.union(e.a, e.b);
    }
    // sources joining the forest also become traversable path edges
    let mut src_adj: HashMap<usize, Vec<(usize, &str)>> = HashMap::new();
    for e in elems.iter().filter(|e| defines_voltage(e.kind)) {
        if forest.union(e.a, e.b) {
            src_adj.entry(e.a).or_default().push((e.b, &e.name));
            src_adj.entry(e.b).or_default().push((e.a, &e.name));
            continue;
        }
        // closes a loop: name the storage elements on the path
        let mut adj = cap_adj.clone();
        for (k, v) in &src_adj {
            adj.entry(*k).or_default().extend(v.iter().copied());
        }
        let storage: Vec<String> = path_between(&adj, e.a, e.b)
            .into_iter()
            .filter(|nm| {
                elems
                    .iter()
                    .any(|c| c.kind == Kind::Capacitor && c.name == *nm)
            })
            .collect();
        // a loop of sources with no capacitor in it is an over-determined
        // deck, not an index-2 one -- a different diagnosis, not ours
        if !storage.is_empty() {
            report.cv_loops.push(Index2Path {
                source: e.name.clone(),
                storage,
            });
        }
    }

    // --- LI cutsets: contract everything that is NOT an inductor or a current
    // source; an L/I element whose endpoints land in different components is a
    // bridge, and bridges sharing the same component pair form one cutset. ---
    let mut rest = Forest::new(n);
    for e in elems
        .iter()
        .filter(|e| e.kind != Kind::Inductor && !defines_current(e.kind))
    {
        rest.union(e.a, e.b);
    }
    // every device terminal pair conducts
    for t in device_terminals {
        for w in t.windows(2) {
            rest.union(w[0], w[1]);
        }
    }
    let mut groups: HashMap<(usize, usize), (Vec<String>, Vec<String>)> = HashMap::new();
    for e in elems
        .iter()
        .filter(|e| e.kind == Kind::Inductor || defines_current(e.kind))
    {
        let (ra, rb) = (rest.find(e.a), rest.find(e.b));
        if ra == rb {
            continue; // shorted by the rest of the circuit: no cutset
        }
        let key = (ra.min(rb), ra.max(rb));
        let slot = groups.entry(key).or_default();
        if e.kind == Kind::Inductor {
            slot.1.push(e.name.clone());
        } else {
            slot.0.push(e.name.clone());
        }
    }
    let mut cutsets: Vec<_> = groups.into_iter().collect();
    cutsets.sort_by_key(|(k, _)| *k); // deterministic order
    for (_, (sources, inductors)) in cutsets {
        // index 2 needs BOTH: an inductor whose current is prescribed, and the
        // source prescribing it. A cutset of current sources alone is a
        // floating-node problem; inductors alone are index 1.
        if inductors.is_empty() {
            continue;
        }
        if let Some(src) = sources.first() {
            report.li_cutsets.push(Index2Path {
                source: src.clone(),
                storage: inductors,
            });
        }
    }
    report
}

/// Element names along a path between two nodes (BFS over the given adjacency).
/// Empty when they are not connected.
fn path_between(adj: &HashMap<usize, Vec<(usize, &str)>>, from: usize, to: usize) -> Vec<String> {
    let mut prev: HashMap<usize, (usize, String)> = HashMap::new();
    let mut queue = std::collections::VecDeque::from([from]);
    let mut seen = std::collections::HashSet::from([from]);
    while let Some(u) = queue.pop_front() {
        if u == to {
            break;
        }
        for (v, nm) in adj.get(&u).into_iter().flatten() {
            if seen.insert(*v) {
                prev.insert(*v, (u, nm.to_string()));
                queue.push_back(*v);
            }
        }
    }
    let mut out = Vec::new();
    let mut cur = to;
    while let Some((p, nm)) = prev.get(&cur) {
        out.push(nm.clone());
        cur = *p;
        if cur == from {
            break;
        }
    }
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cv_circuit() -> Circuit {
        // PSIM's ex_id2_1: V1-C1-C2 close a loop
        let mut c = Circuit::new();
        c.voltage_source("V1", 1, 0);
        c.resistor("R1", 1, 0);
        c.resistor("R2", 2, 0);
        c.capacitor("C1", 1, 2);
        c.capacitor("C2", 2, 0);
        c
    }

    #[test]
    fn detects_a_capacitor_source_loop() {
        let r = detect(&cv_circuit(), &[]);
        assert!(r.is_index2());
        assert_eq!(r.cv_loops.len(), 1);
        assert_eq!(r.cv_loops[0].source, "V1");
        let mut st = r.cv_loops[0].storage.clone();
        st.sort();
        assert_eq!(st, vec!["C1", "C2"]);
        assert!(r.li_cutsets.is_empty());
    }

    #[test]
    fn a_capacitor_straight_across_a_source_is_a_loop() {
        let mut c = Circuit::new();
        c.voltage_source("V1", 1, 0);
        c.resistor("R1", 1, 0);
        c.capacitor("C1", 1, 0);
        let r = detect(&c, &[]);
        assert_eq!(r.cv_loops.len(), 1);
        assert_eq!(r.cv_loops[0].source, "V1");
        assert_eq!(r.cv_loops[0].storage, vec!["C1"]);
    }

    #[test]
    fn a_resistor_in_the_loop_breaks_it() {
        // the same circuit with the loop opened by a series resistor is index 1
        let mut c = Circuit::new();
        c.voltage_source("V1", 1, 0);
        c.resistor("R1", 1, 0);
        c.resistor("R2", 2, 0);
        c.capacitor("C1", 1, 3);
        c.resistor("Rx", 3, 2);
        c.capacitor("C2", 2, 0);
        assert!(!detect(&c, &[]).is_index2());
    }

    #[test]
    fn detects_an_inductor_current_source_cutset() {
        // I1 drives a node reachable only through L1: {I1, L1} is a cutset
        let mut c = Circuit::new();
        c.current_source("I1", 0, 1);
        c.inductor("L1", 1, 2);
        c.resistor("R2", 2, 0);
        let r = detect(&c, &[]);
        assert_eq!(r.li_cutsets.len(), 1);
        assert_eq!(r.li_cutsets[0].source, "I1");
        assert_eq!(r.li_cutsets[0].storage, vec!["L1"]);
    }

    #[test]
    fn a_resistor_across_the_cutset_breaks_it() {
        let mut c = Circuit::new();
        c.current_source("I1", 0, 1);
        c.inductor("L1", 1, 2);
        c.resistor("R1", 1, 0); // gives the source a path around the inductor
        c.resistor("R2", 2, 0);
        assert!(!detect(&c, &[]).is_index2());
    }

    #[test]
    fn a_device_breaks_a_cutset() {
        // A genuine L/I cutset -- until a transistor bridges it. Devices live
        // outside the linear element list, so ignoring them makes every node
        // reachable only through a transistor look isolated: the false positive
        // that fired on the op-amp decks in the corpus.
        let mut c = Circuit::new();
        c.current_source("I0", 0, 1);
        c.inductor("L1", 1, 2);
        c.resistor("R2", 2, 0);
        assert_eq!(
            detect(&c, &[]).li_cutsets.len(),
            1,
            "without a bypass it is a cutset"
        );
        let dev = vec![vec![1usize, 0]]; // a device conducting from node 1 to ground
        assert!(
            detect(&c, &dev).li_cutsets.is_empty(),
            "the device gives the source a path around the inductor"
        );
    }

    #[test]
    fn a_current_source_cutset_without_an_inductor_is_not_index_2() {
        // Only current sources bridge the two halves: that is a floating-node
        // deck, a different diagnosis.
        let mut c = Circuit::new();
        c.current_source("I1", 0, 1);
        c.resistor("R1", 2, 0);
        assert!(detect(&c, &[]).li_cutsets.is_empty());
    }

    #[test]
    fn a_plain_rc_lowpass_is_index_1() {
        let mut c = Circuit::new();
        c.voltage_source("V1", 1, 0);
        c.resistor("R1", 1, 2);
        c.capacitor("C1", 2, 0);
        assert!(!detect(&c, &[]).is_index2());
    }
}
