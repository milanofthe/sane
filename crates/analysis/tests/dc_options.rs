//! How operating points are solved: the deck's `.option`s, a model's
//! settings, kept by the points taken under them, and the explicit warm
//! start of one point from another.

use sane_analysis::{DcOptions, Model};

const CLAMP: &str = "\
V1 in 0 2
R1 in out 1k
D1 out 0 dmod
.model dmod D(Is=1e-14 N=1)
";

#[test]
fn a_decks_options_set_the_solve() {
    let m = Model::from_netlist(&format!(
        "{CLAMP}.options reltol=1e-3 vntol=1e-6 itl1=50\n.end"
    ))
    .unwrap();
    let o = m.dc_options();
    assert_eq!((o.reltol, o.vntol, o.max_iter), (1e-3, 1e-6, 50));
    assert_eq!(o.abstol, DcOptions::default().abstol);
    let loose = m
        .at(&[])
        .unwrap()
        .operating_point()
        .unwrap()
        .get("out")
        .unwrap();
    let tight = Model::from_netlist(&format!("{CLAMP}.end")).unwrap();
    let tight = tight
        .at(&[])
        .unwrap()
        .operating_point()
        .unwrap()
        .get("out")
        .unwrap();
    assert!((loose - tight).abs() < 1e-4, "{loose} vs {tight}");
}

#[test]
fn a_point_keeps_the_options_it_was_taken_with() {
    let m = Model::from_netlist(&format!("{CLAMP}.end")).unwrap();
    let before = m.at(&[]).unwrap();
    let loose = DcOptions {
        reltol: 1e-3,
        max_iter: 20,
        ..DcOptions::default()
    };
    m.set_dc_options(loose);
    assert_eq!(m.at(&[]).unwrap().dc_options(), loose);
    assert_eq!(before.dc_options(), DcOptions::default());
}

/// A latch: V(a) = g(V(b)), V(b) = g(V(a)) with a steep decreasing g, the
/// symmetric root an unstable saddle a cold solve lands on; a bias current
/// into `a` tips it onto one branch.
const LATCH: &str = "R1 a 0 1
R2 b 0 1
I1 0 a {ib}
.param ib=0
B1 0 a I=0.5 - 0.3183098862*atan(10*(V(b)-0.5))
B2 0 b I=0.5 - 0.3183098862*atan(10*(V(a)-0.5))
.end
";

#[test]
fn a_warm_start_stays_on_the_branch_it_starts_on() {
    let m = Model::from_netlist(LATCH).unwrap();
    let ib = m
        .params()
        .iter()
        .find(|p| p.eq_ignore_ascii_case("I1"))
        .unwrap()
        .clone();
    let tipped = m.at(&[(ib.as_str(), 0.3)]).unwrap();
    let hi = tipped.operating_point().unwrap();
    assert!(
        hi.get("a").unwrap() > 0.6,
        "the bias tips it high: {:?}",
        hi.get("a")
    );
    let cold = m
        .at(&[(ib.as_str(), 0.0)])
        .unwrap()
        .operating_point()
        .unwrap();
    assert!(
        (cold.get("a").unwrap() - 0.5).abs() < 1e-6,
        "cold: the saddle"
    );
    let warm = (m.at(&[(ib.as_str(), 0.0)]).unwrap())
        .near(&tipped)
        .unwrap();
    let op = warm.operating_point().unwrap();
    let (a, b) = (op.get("a").unwrap(), op.get("b").unwrap());
    assert!(a > 0.6 && b < 0.4, "warm: the high branch, a={a} b={b}");
}
