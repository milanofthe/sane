"""Python API for the two DC convergence aids: `.nodeset` symmetry breaking
and the per-component reltol/abstol/vntol criterion.

The fixture is a high-impedance cross-coupled-inverter latch (see
crates/netlist/tests/nodeset.rs): V(a)=g(V(b)), V(b)=g(V(a)) with a steep
decreasing sigmoid, so the symmetric root a=b=0.5 is an unstable saddle.

Run after `maturin develop -m crates/py/Cargo.toml`:
    python -m pytest crates/py/tests/test_nodeset_convergence.py
"""
import sane

LATCH = (
    "R1 a 0 1\n"
    "R2 b 0 1\n"
    "B1 0 a I=0.5 - 0.3183098862*atan(10*(V(b)-0.5))\n"
    "B2 0 b I=0.5 - 0.3183098862*atan(10*(V(a)-0.5))\n"
    ".end\n"
)


def _dae():
    return sane.parse(LATCH).extract()


def test_cold_solve_is_symmetric():
    op = _dae().operating_point()
    assert abs(op["a"] - 0.5) < 1e-6
    assert abs(op["b"] - 0.5) < 1e-6


def test_nodeset_selects_branch():
    dae = _dae()
    hi = dae.operating_point(nodeset={"a": 1.0})
    assert hi["a"] > 0.6 and hi["b"] < 0.4, (hi["a"], hi["b"])

    lo = dae.operating_point(nodeset={"a": 0.0})
    assert lo["a"] < 0.4 and lo["b"] > 0.6, (lo["a"], lo["b"])

    # Mirror branches.
    assert abs(hi["a"] - lo["b"]) < 1e-3


def test_convergence_knobs():
    dae = _dae()
    # Loose SPICE-like reltol still converges to the same (symmetric) root as the
    # tight default; both are valid stopping rules.
    tight = dae.operating_point()
    loose = dae.operating_point(reltol=1e-3, abstol=1e-12, vntol=1e-6)
    assert abs(tight["a"] - loose["a"]) < 1e-3

    # nodeset composes with the per-component criterion.
    hi = dae.operating_point(nodeset={"a": 1.0}, reltol=1e-6, vntol=1e-9)
    assert hi["a"] > 0.6 and hi["b"] < 0.4


if __name__ == "__main__":
    test_cold_solve_is_symmetric()
    test_nodeset_selects_branch()
    test_convergence_knobs()
    print("OK")
