//! Every netlist fixture must parse and yield a non-empty DAE. This is broad
//! regression coverage for the parser + DAE assembly across circuit topologies.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use rsdag::{ExprId, Node};
use sane_core::Graph;
use sane_dae::assemble_dae;
use sane_netlist::parse;

/// Count the unique DAG nodes reachable from `roots` (shared nodes once).
fn count_nodes(ctx: &Graph, roots: &[ExprId]) -> usize {
    let mut seen = HashSet::new();
    let mut stack = roots.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match ctx.node(id) {
            Node::Const(_) | Node::Symbol(_) => {}
            Node::Add(a, b) | Node::Mul(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Node::Neg(a) | Node::Pow(a, _) | Node::Unary(_, a) => stack.push(*a),
            Node::Cmp(_, a, b) | Node::Binary(_, a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Node::Select(c, t, e) => {
                stack.push(*c);
                stack.push(*t);
                stack.push(*e);
            }
            Node::Reduce(_, l) | Node::Dot(l) | Node::Call(_, l) | Node::Solve(l, _) => {
                stack.extend_from_slice(ctx.args(*l))
            }
        }
    }
    seen.len()
}

#[test]
fn all_fixtures_parse_and_extract() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut count = 0;
    for entry in fs::read_dir(&dir).expect("fixtures dir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("cir") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = fs::read_to_string(&path).unwrap();

        let parsed = parse(&text).unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
        let mut ctx = Graph::new();
        let dae = assemble_dae(&mut ctx, &parsed.circuit, &parsed.devices);
        assert!(dae.dim() > 0, "{name}: empty DAE");

        let res_nodes = count_nodes(&ctx, &dae.residuals);
        let jac = dae.jacobian_x(&mut ctx);
        let flat: Vec<ExprId> = jac.iter().flatten().copied().collect();
        let jac_nnz = flat.iter().filter(|&&e| !ctx.is_zero(e)).count();
        let jac_nodes = count_nodes(&ctx, &flat);

        // True dynamic states: columns of dF/dx' that are not structurally zero.
        let jxd = dae.jacobian_xdot(&mut ctx);
        let states = (0..dae.dim())
            .filter(|&j| jxd.iter().any(|row| !ctx.is_zero(row[j])))
            .count();

        println!(
            "{name:24} dim={:2} states={:2} params={:2} | residual nodes={:3} | jac {}x{} nnz={:2} nodes={:3}",
            dae.dim(),
            states,
            dae.params(&ctx).len(),
            res_nodes,
            dae.dim(),
            dae.dim(),
            jac_nnz,
            jac_nodes,
        );
        count += 1;
    }
    assert!(count >= 15, "expected >= 15 fixtures, found {count}");
}
