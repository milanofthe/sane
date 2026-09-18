"""The embeddable `Sim` facade: orchestration/store/state live in Rust; this
exercises the thin Python handles and checks they agree with the legacy DAE path.

    python -m pytest crates/py/tests/test_sim.py
"""
import pytest

import sane

DIV = "V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end"


def test_operating_point_store_and_override():
    sim = sane._core.Model.from_netlist(DIV)
    op = sim.operating_point()
    assert op["out"] == pytest.approx(2.5)

    # persistent mutation
    sim.set("R2", 3e3)
    assert sim.get("R2") == pytest.approx(3e3)
    assert sim.operating_point()["out"] == pytest.approx(5 * 3e3 / 4e3)

    # non-destructive per-call override
    assert sim.operating_point({"R2": 1e3})["out"] == pytest.approx(2.5)
    assert sim.get("R2") == pytest.approx(3e3)

    sim.reset()
    assert sim.get("R2") == pytest.approx(1e3)


def test_sensitivity_off_point():
    sim = sane._core.Model.from_netlist(DIV)
    op = sim.operating_point()
    s = op.sensitivity("out")
    assert len(s.names) == len(s.grad)
    assert s.value == pytest.approx(2.5)
    i = s.names.index("R2")
    assert s.grad[i] > 0.0


def test_transient_and_noise():
    sim = sane.Circuit.parse("V1 in 0 1\nR1 in out 1k\nC1 out 0 1u").extract()
    t = [k * 5e-3 / 20 for k in range(21)]
    traj = sim.transient(t)
    assert traj["out"][-1] > 0.9
    ns = sim.noise("out", 1.0, 1e6, points=10)
    assert len(ns.freqs) == len(ns.noise)


def test_matches_legacy_dae():
    sim = sane._core.Model.from_netlist(DIV)
    dae = sane.parse(DIV).extract()
    a = sim.operating_point()["out"]
    b = dae.operating_point()["out"]
    assert a == pytest.approx(b)


def test_hierarchy():
    sim = sane._core.Model.from_netlist(
        ".subckt rcdiv a b\nR1 a m 1k\nR2 m b 2k\n.ends\nV1 in 0 5\nX1 in 0 rcdiv\n.end"
    )
    assert sim.is_group("X1")
    assert sim.get("X1.R2") == pytest.approx(2e3)
    assert "R1" in sim.children("X1")


def test_ac_poles_state_space():
    sim = sane._core.Model.from_netlist("V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end")
    ac = sim.ac("V1", "out", 1.0, 1e5, 50)
    assert len(ac.freqs) == 50 and ac.mag_db[0] > ac.mag_db[-1]
    pz = sim.poles_zeros("V1", "out")
    assert pz.poles()[0][0] == pytest.approx(-1.0 / (1e3 * 1e-6), abs=1.0)
    ss = sim.state_space("V1", "out")
    assert len(ss.e) == sim.dim()


def test_temp_sweep():
    sim = sane._core.Model.from_netlist(DIV)
    ts = sim.temp_sweep("out", 0.0, 100.0, 5)
    assert len(ts.temps) == len(ts.values)


def test_harmonic_balance():
    sim = sane._core.Model.from_netlist(
        "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 dm\n.model dm D(Is=1e-14 N=1 Vt=0.025852)\n.end"
    )
    hb = sim.harmonic_balance(1000.0, harmonics=5)
    assert hb.converged
    assert len(hb.magnitude("mid")) == 6


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
    print("OK")
