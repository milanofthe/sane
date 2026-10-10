//! Every netlist fixture must parse and yield a non-empty DAE. This is broad
//! regression coverage for the parser + DAE assembly across circuit topologies.

use std::fs;
use std::path::PathBuf;

use rsdag::ExprId;
use sane_core::Graph;
use sane_netlist::parse;

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
        let dae = sane_dae::assemble(&mut ctx, &parsed).unwrap();
        assert!(dae.dim() > 0, "{name}: empty DAE");

        let rows: Vec<ExprId> = dae.currents.iter().chain(&dae.charges).copied().collect();
        let res_nodes = rsdag::dot::reachable(&ctx, &rows).len();
        let ((_, _, g), (_, cols, _)) = dae.jacobian_iq_coo(&mut ctx);
        let jac_nnz = g.iter().filter(|&&e| !ctx.is_zero(e)).count();
        let jac_nodes = rsdag::dot::reachable(&ctx, &g).len();

        // True dynamic states: the columns of dQ/dx.
        let states = cols.iter().collect::<std::collections::BTreeSet<_>>().len();

        println!(
            "{name:24} dim={:2} states={:2} params={:2} | row nodes={:3} | jac {}x{} nnz={:2} nodes={:3}",
            dae.dim(),
            states,
            dae.params(&mut ctx).len(),
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
