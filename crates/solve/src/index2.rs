//! The index-2 unknowns of `d/dt Q(x) + I(x, t) = 0`, from the structure of
//! `C = dQ/dx` and `G = dI/dx`.
//!
//! An unknown no charge depends on is algebraic, and so is a row no charge
//! enters. The algebraic rows determine the algebraic unknowns where `G`
//! couples them one to one (a maximum matching between the two). An
//! algebraic unknown no maximum matching needs -- unmatched, or reachable
//! from an unmatched one along alternating paths (the column-surplus part of
//! the Dulmage-Mendelsohn decomposition) -- is fixed by no algebraic row but
//! by the derivative of a constraint: the current through a loop of
//! capacitors and voltage-defined branches, the voltage across a cutset of
//! inductors and current-defined branches. Its value is a rate of the other
//! unknowns, and an integration's error in it is one order lower than in
//! them: an error estimate that controls it keeps its value as the step
//! shrinks on a kink of that rate (Hairer & Wanner, Solving ODEs II,
//! VII.8, exclude or scale those components). Structural: a numerically
//! singular coupling is not seen, a structurally present one counts.

/// Per unknown, whether it is index-2, from the `(row, column)` patterns of
/// `G` and `C` over `n` unknowns.
pub(crate) fn index2_unknowns(
    n: usize,
    g: (&[usize], &[usize]),
    c: (&[usize], &[usize]),
) -> Vec<bool> {
    const NONE: usize = usize::MAX;
    let (mut diff_row, mut diff_col) = (vec![false; n], vec![false; n]);
    for (&r, &k) in c.0.iter().zip(c.1) {
        diff_row[r] = true;
        diff_col[k] = true;
    }
    // per algebraic unknown, the algebraic rows `G` couples it to
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (&r, &k) in g.0.iter().zip(g.1) {
        if !diff_row[r] && !diff_col[k] && !adj[k].contains(&r) {
            adj[k].push(r);
        }
    }
    let (mut row_match, mut col_match) = (vec![NONE; n], vec![NONE; n]);
    for k in (0..n).filter(|&k| !diff_col[k]) {
        if let Some(&r) = adj[k].iter().find(|&&r| row_match[r] == NONE) {
            row_match[r] = k;
            col_match[k] = r;
        }
    }
    // augmenting paths, breadth first from each unmatched unknown
    let (mut seen, mut from) = (vec![NONE; n], vec![NONE; n]);
    let mut queue = Vec::new();
    for k0 in 0..n {
        if diff_col[k0] || col_match[k0] != NONE {
            continue;
        }
        queue.clear();
        queue.push(k0);
        let (mut head, mut free) = (0, NONE);
        'search: while head < queue.len() {
            let k = queue[head];
            head += 1;
            for &r in &adj[k] {
                if seen[r] == k0 {
                    continue;
                }
                seen[r] = k0;
                from[r] = k;
                if row_match[r] == NONE {
                    free = r;
                    break 'search;
                }
                queue.push(row_match[r]);
            }
        }
        let mut r = free;
        while r != NONE {
            let k = from[r];
            let next = col_match[k];
            col_match[k] = r;
            row_match[r] = k;
            r = next;
        }
    }
    // the unmatched unknowns and every one an alternating path reaches
    let mut index2 = vec![false; n];
    let mut stack: Vec<usize> = (0..n)
        .filter(|&k| !diff_col[k] && col_match[k] == NONE)
        .collect();
    for &k in &stack {
        index2[k] = true;
    }
    while let Some(k) = stack.pop() {
        for &r in &adj[k] {
            let j = row_match[r];
            if j != NONE && !index2[j] {
                index2[j] = true;
                stack.push(j);
            }
        }
    }
    index2
}

#[cfg(test)]
mod tests {
    use super::index2_unknowns;

    fn run(n: usize, g: &[(usize, usize)], c: &[(usize, usize)]) -> Vec<bool> {
        let split =
            |e: &[(usize, usize)]| -> (Vec<usize>, Vec<usize>) { e.iter().copied().unzip() };
        let (g, c) = (split(g), split(c));
        index2_unknowns(n, (&g.0, &g.1), (&c.0, &c.1))
    }

    /// `V1 a 0; C1 a 0`: unknowns `v_a, i_V1`; rows `a` (charge, current),
    /// `V1` (`v_a - V(t)`). The source's current is the loop's rate.
    #[test]
    fn a_capacitor_voltage_source_loop_makes_its_current_index_2() {
        let g = [(0, 1), (1, 0)];
        let c = [(0, 0)];
        assert_eq!(run(2, &g, &c), [false, true]);
    }

    /// `V1 a 0; R1 a 0`: the node and the current are index-1 algebraic.
    #[test]
    fn a_source_into_a_resistor_is_index_1() {
        let g = [(0, 0), (0, 1), (1, 0)];
        assert_eq!(run(2, &g, &[]), [false, false]);
    }

    /// `I1 0 m; L1 m 0`: unknowns `v_m, i_L1`; rows `m` (`i_L1 - I(t)`), `L1`
    /// (flux, `-v_m`). The node voltage is the cutset's rate.
    #[test]
    fn an_inductor_current_source_cutset_makes_its_voltage_index_2() {
        let g = [(0, 1), (1, 0)];
        let c = [(1, 1)];
        assert_eq!(run(2, &g, &c), [true, false]);
    }

    /// Two sources in parallel with a capacitor: which current carries the
    /// rate is not fixed, both count.
    #[test]
    fn an_ambiguous_loop_marks_every_current_it_could_be() {
        // unknowns v_a, i1, i2; rows a (C, i1, i2), V1 (v_a), V2 (v_a)
        let g = [(0, 1), (0, 2), (1, 0), (2, 0)];
        let c = [(0, 0)];
        assert_eq!(run(3, &g, &c), [false, true, true]);
    }

    /// An augmenting path is needed where the greedy matching takes the
    /// wrong row first.
    #[test]
    fn the_matching_augments_past_a_greedy_choice() {
        // unknowns 0, 1 algebraic; rows 0 (both), 1 (unknown 0 only)
        let g = [(0, 0), (0, 1), (1, 0)];
        assert_eq!(run(2, &g, &[]), [false, false]);
    }
}
