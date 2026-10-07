//! A transient at the parameters of an operating point starts from that
//! point: one DC solve serves both, and the trajectory begins exactly where
//! the operating point is.

use sane_analysis::Model;
use sane_solve::TransientMethod;

const DECK: &str = "\
* diode clamp
V1 in 0 2
R1 in out 1k
D1 out 0 dmod
C1 out 0 1n
.model dmod D(Is=1e-14 N=1)
.end
";

#[test]
fn a_transient_starts_at_the_operating_point() {
    let model = Model::from_netlist(DECK).expect("model");
    let op = model.operating_point(&[]).expect("dc");
    let t: Vec<f64> = (0..=10).map(|k| k as f64 * 1e-7).collect();
    let rows = model
        .transient(TransientMethod::Esdirk32, &[], &t, 1e-4, 1e-7)
        .expect("transient")
        .rows()
        .to_vec();
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&rows[0]), bits(op.vector()), "starts at the operating point");
    // And a transient at other parameters starts at their own point.
    let other = model
        .transient(TransientMethod::Esdirk32, &[("V1", 3.0)], &t, 1e-4, 1e-7)
        .expect("transient")
        .rows()
        .to_vec();
    let op3 = model.operating_point(&[("V1", 3.0)]).expect("dc");
    assert_eq!(bits(&other[0]), bits(op3.vector()));
}
