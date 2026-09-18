//! A minimum degree ordering on the quotient graph, the representation
//! AMD introduced: an eliminated vertex becomes an element standing for
//! the clique its elimination created, so the fill is never stored, and
//! the elements a pivot touches are absorbed into the new one. Degrees
//! are exact external degrees counted over the quotient graph (AMD proper
//! approximates them; the quotient graph and absorption are what make it
//! fast, and exact degrees never order worse). Variables far denser than
//! the rest, a global net, are deferred to the end.

use rustc_hash::FxHashSet as HashSet;
use std::collections::BTreeSet;

/// A fill-reducing elimination order over a symmetric adjacency (`adj[i]`
/// lists the neighbours of `i`, both directions present). `order[k]` is
/// the vertex eliminated at step `k`.
pub fn amd(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    // Variable-to-variable and variable-to-element adjacency; an element's
    // members. An eliminated vertex reuses its id as the element's.
    let mut vars: Vec<HashSet<usize>> = adj
        .iter()
        .enumerate()
        .map(|(i, nb)| nb.iter().copied().filter(|&j| j != i).collect())
        .collect();
    let mut elems: Vec<HashSet<usize>> = vec![HashSet::default(); n];
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut degree: Vec<usize> = vars.iter().map(HashSet::len).collect();
    let dense = ((10.0 * (n as f64).sqrt()) as usize).max(16);
    let mut queue: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut deferred: Vec<usize> = Vec::new();
    for i in 0..n {
        if degree[i] > dense && n > 2 * dense {
            deferred.push(i);
        } else {
            queue.insert((degree[i], i));
        }
    }
    let mut mark = vec![usize::MAX; n];
    let mut next_stamp = n;
    let mut order = Vec::with_capacity(n);
    let eliminate = |p: usize,
                     vars: &mut Vec<HashSet<usize>>,
                     elems: &mut Vec<HashSet<usize>>,
                     members: &mut Vec<Vec<usize>>,
                     degree: &mut Vec<usize>,
                     queue: &mut BTreeSet<(usize, usize)>,
                     mark: &mut Vec<usize>,
                     stamp: &mut usize| {
        // The new element: p's variables and the members of p's elements.
        mark[p] = p;
        let mut new: Vec<usize> = Vec::new();
        for &v in &vars[p] {
            if mark[v] != p {
                mark[v] = p;
                new.push(v);
            }
        }
        let absorbed: Vec<usize> = elems[p].iter().copied().collect();
        for &e in &absorbed {
            for &v in &members[e] {
                if mark[v] != p {
                    mark[v] = p;
                    new.push(v);
                }
            }
        }
        new.sort_unstable();
        for &e in &absorbed {
            for &v in std::mem::take(&mut members[e]).iter() {
                elems[v].remove(&e);
            }
        }
        for &v in &new {
            vars[v].remove(&p);
            elems[v].insert(p);
            // A variable reached through the element need not be listed
            // twice: prune it from the variable adjacency.
            vars[v].retain(|&w| mark[w] != p);
        }
        members[p] = new.clone();
        vars[p].clear();
        elems[p].clear();
        // Exact external degrees of the touched variables.
        for &v in &new {
            queue.remove(&(degree[v], v));
            // A stamp unique to this count; vertex ids stay below `n`.
            *stamp += 1;
            let stamp = *stamp;
            mark[v] = stamp;
            let mut d = 0;
            for &w in &vars[v] {
                if mark[w] != stamp {
                    mark[w] = stamp;
                    d += 1;
                }
            }
            for &e in &elems[v] {
                for &w in &members[e] {
                    if w != v && mark[w] != stamp {
                        mark[w] = stamp;
                        d += 1;
                    }
                }
            }
            degree[v] = d;
            queue.insert((d, v));
        }
    };
    while let Some(&(d, p)) = queue.iter().next() {
        queue.remove(&(d, p));
        order.push(p);
        eliminate(
            p,
            &mut vars,
            &mut elems,
            &mut members,
            &mut degree,
            &mut queue,
            &mut mark,
            &mut next_stamp,
        );
    }
    // The deferred hubs, by their degree at the end.
    let mut rest: Vec<(usize, usize)> = deferred.iter().map(|&v| (degree[v], v)).collect();
    rest.sort_unstable();
    for (_, p) in rest {
        order.push(p);
        eliminate(
            p,
            &mut vars,
            &mut elems,
            &mut members,
            &mut degree,
            &mut queue,
            &mut mark,
            &mut next_stamp,
        );
    }
    order
}
