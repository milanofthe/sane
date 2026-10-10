"""The DC convergence aids: `.nodeset` symmetry breaking (the node set
belongs to the circuit, so a deck's `.nodeset` selects which root its
operating point is), the per-component reltol/abstol/vntol criterion (a
deck's `.option`s or the model's settings) and the explicit warm start.

The fixture is a high-impedance cross-coupled-inverter latch (see
crates/netlist/tests/nodeset.rs): V(a)=g(V(b)), V(b)=g(V(a)) with a steep
decreasing sigmoid, so the symmetric root a=b=0.5 is an unstable saddle.

    python -m pytest crates/py/tests/test_nodeset_convergence.py
"""
import sane

LATCH = (
    "R1 a 0 1\n"
    "R2 b 0 1\n"
    "B1 0 a I=0.5 - 0.3183098862*atan(10*(V(b)-0.5))\n"
    "B2 0 b I=0.5 - 0.3183098862*atan(10*(V(a)-0.5))\n"
)


def _op(nodeset=""):
    return sane.Model.from_netlist(LATCH + nodeset + ".end\n").at().operating_point()


def test_cold_solve_is_symmetric():
    op = _op()
    assert abs(op["a"] - 0.5) < 1e-6
    assert abs(op["b"] - 0.5) < 1e-6


def test_nodeset_selects_branch():
    hi = _op(".nodeset V(a)=1\n")
    assert hi["a"] > 0.6 and hi["b"] < 0.4, (hi["a"], hi["b"])

    lo = _op(".nodeset V(a)=0\n")
    assert lo["a"] < 0.4 and lo["b"] > 0.6, (lo["a"], lo["b"])

    # Mirror branches.
    assert abs(hi["a"] - lo["b"]) < 1e-3


def test_convergence_settings():
    tight = _op()
    m = sane.Model.from_netlist(LATCH + ".options reltol=1e-3 vntol=1e-6\n.end\n")
    assert m.dc_options["reltol"] == 1e-3 and m.dc_options["vntol"] == 1e-6
    loose = m.at().operating_point()
    assert abs(tight["a"] - loose["a"]) < 1e-3
    # a point keeps the settings it was taken with
    before = m.at()
    m.set_dc_options(reltol=1e-6)
    assert before.dc_options["reltol"] == 1e-3
    assert m.at().dc_options["reltol"] == 1e-6


def test_warm_start_keeps_the_branch():
    # a bias current into `a` tips the latch high; solved from there, the
    # unbiased latch stays high where a cold solve lands on the saddle
    m = sane.Model.from_netlist(LATCH + "I1 0 a 0\n.end\n")
    tipped = m.at(I1=0.3)
    assert tipped.operating_point()["a"] > 0.6
    assert abs(m.at().operating_point()["a"] - 0.5) < 1e-6
    warm = m.at().near(tipped).operating_point()
    assert warm["a"] > 0.6 and warm["b"] < 0.4
