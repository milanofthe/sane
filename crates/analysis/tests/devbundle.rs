//! Instance batching: every instance of a template is a call into one shared
//! function whose compiled body the solver evaluates per instance. Three identical, independent branches therefore
//! cross-validate the bundle against the scalar path within one solve: any
//! bundle defect shows up as a branch-1 vs branch-2/3 mismatch.

use sane_analysis::Model;
use sane_solve::TransientMethod;

const DECK_HEAD: &str = "\
.veriloga
module vadiode(a, c);
  inout a, c; electrical a, c;
  parameter real Is = 1e-14;
  parameter real N  = 1.0;
  parameter real C0 = 1e-12;
  analog begin
    I(a, c) <+ Is * (limexp(V(a,c)/(N*0.025852)) - 1.0);
    I(a, c) <+ ddt(C0 * V(a,c));
  end
endmodule
.endveriloga
";

fn three_branch_deck(drive: &str) -> String {
    let mut d = format!("* bundle parity\n{DECK_HEAD}");
    for k in 1..=3 {
        d.push_str(&format!(
            "V{k} s{k} 0 {drive}\nR{k} s{k} d{k} 1k\nN{k} d{k} 0 vadiode\n"
        ));
    }
    d.push_str(".end\n");
    d
}

/// The assembled model lowers the three instances as calls into ONE function
/// of the graph (the residuals reference its outputs with three distinct
/// argument lists).
#[test]
fn instances_call_one_function() {
    let model = Model::from_netlist(&three_branch_deck("2.0")).expect("model");
    let dae = model.dae();
    let ctx = model.context_arc();
    let ctx = ctx.lock().unwrap();
    // The graph also holds the DAE's own signature function (roles, guards),
    // which nothing calls; what matters here is that the three instances call
    // one and the same device function.
    let called: std::collections::BTreeSet<_> = ctx
        .free_calls_in(&dae.residuals)
        .into_iter()
        .map(|o| ctx.output(o).0)
        .collect();
    assert_eq!(called.len(), 1, "one template function");
    let calls = ctx.free_calls_in(&dae.residuals);
    assert!(!calls.is_empty(), "residuals call the function");
    let mut arg_lists = std::collections::BTreeSet::new();
    let mut stack = dae.residuals.clone();
    let mut seen = std::collections::HashSet::new();
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        if let rsdag::Node::Call(_, l) = *ctx.node(e) {
            arg_lists.insert(l);
        }
        stack.extend_from_slice(&ctx.operands(e));
    }
    assert_eq!(arg_lists.len(), 3, "three instances, three argument lists");
}

/// DC: branch 1 (scalar representative) and branches 2/3 (bundled clones) are
/// identical circuits, so their node voltages must agree to solver tolerance.
#[test]
fn dc_scalar_vs_bundled_branches_agree() {
    let model = Model::from_netlist(&three_branch_deck("2.0")).expect("model");
    let op = model.operating_point(&[]).expect("dc");
    let v = |n: &str| op.vector()[model.resolve(n).expect(n)];
    let (d1, d2, d3) = (v("d1"), v("d2"), v("d3"));
    assert!(d1 > 0.4 && d1 < 0.9, "diode drop plausible: {d1}");
    assert!((d1 - d2).abs() < 1e-10, "d1={d1} d2={d2}");
    assert!((d1 - d3).abs() < 1e-10, "d1={d1} d3={d3}");
}

/// Transient: the bundled branches track the scalar branch through a sine
/// drive (charge storage included, so the Jacobian xdot path is exercised).
#[test]
fn transient_scalar_vs_bundled_branches_agree() {
    let model = Model::from_netlist(&three_branch_deck("SIN(0.6 0.3 1Meg)")).expect("model");
    let t: Vec<f64> = (0..200).map(|k| 2e-6 * k as f64 / 199.0).collect();
    let traj = model
        .transient(TransientMethod::Esdirk32, &[], &t, 1e-6, 1e-9)
        .expect("transient");
    let (i1, i2) = (model.resolve("d1").unwrap(), model.resolve("d2").unwrap());
    for (k, row) in traj.rows().iter().enumerate() {
        assert!(
            (row[i1] - row[i2]).abs() < 1e-9,
            "t[{k}]: d1={} d2={}",
            row[i1],
            row[i2]
        );
    }
}

/// Sensitivity w.r.t. a BUNDLED instance's own parameter (the parameter enters
/// the bundle as an argument; the dF/dp path materialises partial markers over
/// it): dV(d2)/d(N2.Is) must equal the scalar branch's dV(d1)/d(N1.Is).
#[test]
fn sensitivity_wrt_bundled_device_param_matches_scalar() {
    let model = Model::from_netlist(&three_branch_deck("2.0")).expect("model");
    let op = model.operating_point(&[]).expect("dc");
    let x = op.vector().to_vec();
    let p = model.pvec(&[]);
    let canon = |node: &str| model.unknowns()[model.resolve(node).unwrap()].clone();
    let (n1, g1) = model
        .sensitivity(&canon("d1"), x.clone(), p.clone(), 0.0)
        .expect("sens d1");
    let (n2, g2) = model.sensitivity(&canon("d2"), x, p, 0.0).expect("sens d2");
    let pick = |names: &[String], vals: &[f64], key: &str| -> f64 {
        vals[names
            .iter()
            .position(|n| n == key)
            .unwrap_or_else(|| panic!("param {key}"))]
    };
    let s1 = pick(&n1, &g1, "N1.Is");
    let s2 = pick(&n2, &g2, "N2.Is");
    assert!(s1.abs() > 1e3, "sensitivity nonzero: {s1}");
    assert!(((s1 - s2) / s1).abs() < 1e-6, "scalar {s1} vs bundled {s2}");
}

/// A module whose `TYPE` decides its structure through a variable (the
/// decision reads `k`, which came from `TYPE`) and whose `Is` does not.
const DECK_SWITCHED: &str = "\
.veriloga
module swdiode(a, c);
  inout a, c; electrical a, c;
  parameter real Is = 1e-14;
  parameter real TYPE = 1;
  real k;
  analog begin
    k = TYPE * 2;
    if (k > 0)
      I(a, c) <+ Is * (limexp(V(a,c)/0.025852) - 1.0);
    else
      I(a, c) <+ V(a, c) / 1e3;
  end
endmodule
.endveriloga
";

fn switched_deck(instances: &[(usize, &str)]) -> String {
    let mut d = format!("* shared templates\n{DECK_SWITCHED}");
    for (k, params) in instances {
        d.push_str(&format!(
            "V{k} s{k} 0 2.0\nR{k} s{k} d{k} 1k\nN{k} d{k} 0 swdiode {params}\n"
        ));
    }
    d.push_str(".end\n");
    d
}

fn called_functions(model: &Model) -> usize {
    let dae = model.dae();
    let ctx = model.context_arc();
    let ctx = ctx.lock().unwrap();
    ctx.free_calls_in(&dae.residuals)
        .into_iter()
        .map(|o| ctx.output(o).0)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

/// Instances that differ only in a parameter no structural decision reads
/// share one template, one function; a parameter a decision reads, even
/// through a variable, splits them. Every node voltage is the one of the
/// instance alone in its own deck, where nothing is shared.
#[test]
fn templates_are_shared_across_non_structural_parameters() {
    let instances = [
        (1, "Is=1e-14"),
        (2, "Is=3e-14"),
        (3, "Is=1e-13"),
        (4, "TYPE=-1"),
        (5, "TYPE=-1 Is=5e-14"),
    ];
    let model = Model::from_netlist(&switched_deck(&instances)).expect("model");
    assert_eq!(called_functions(&model), 2, "one function per structure");
    let op = model.operating_point(&[]).expect("dc");
    for &(k, params) in &instances {
        let alone = Model::from_netlist(&switched_deck(&[(k, params)])).expect("alone");
        let op_alone = alone.operating_point(&[]).expect("dc alone");
        let node = format!("d{k}");
        let (shared, single) = (
            op.vector()[model.resolve(&node).unwrap()],
            op_alone.vector()[alone.resolve(&node).unwrap()],
        );
        assert!(
            (shared - single).abs() < 1e-10,
            "{node} ({params}): shared {shared}, alone {single}"
        );
    }
}
