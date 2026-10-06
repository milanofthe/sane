"""Smoke test for the SANE Python bindings.

Numeric interface: residual / Jacobian are evaluated in Rust; Python just drives
an external solver (scipy). Symbolic interface: equations / LaTeX / AC transfer.

Run after `maturin develop -m crates/py/Cargo.toml`:
    python crates/py/tests/smoke.py
"""

import numpy as np
from scipy.optimize import fsolve

import sane


def build_p(dae, overrides=None):
    """Parameter vector in dae.params() order, from netlist values + overrides."""
    vals = dict(dae.values())
    if overrides:
        vals.update(overrides)
    return [float(vals.get(name, 0.0)) for name in dae.params()]


def solve_dc(dae, p, x0=None):
    """DC operating point: solve I(x, p, 0) = 0 via the Rust currents and G."""
    n = dae.dim()
    fun = lambda x: np.array(dae.currents(list(x), p, 0.0))
    jac = lambda x: np.array(dae.jacobian_i_x(list(x), p, 0.0))
    x0 = np.zeros(n) if x0 is None else x0
    return fsolve(fun, x0, fprime=jac)


def test_rc_dc():
    c = sane.Circuit()
    c.voltage_source("V1", 1, 0)
    c.resistor("R", 1, 2)
    c.capacitor("C", 2, 0)
    dae = c.extract_dae()
    assert dae.unknowns() == ["v1", "v2", "i_V1"]
    p = build_p(dae, {"V1": 5.0, "R": 1000.0, "C": 1e-6})
    sol = solve_dc(dae, p)
    # At DC the capacitor is open: v1 = v2 = 5, i_V1 = 0.
    assert np.allclose(sol, [5.0, 5.0, 0.0], atol=1e-9), sol
    print("RC DC:", dict(zip(dae.unknowns(), np.round(sol, 6))))


def test_diode_dc():
    c = sane.parse(
        "V1 in 0 0.72\nR1 in out 1k\nD1 out 0 dm\n.model dm D(Is=1e-14 N=1 Vt=0.025852)\n.end"
    )
    dae = c.extract_dae()
    p = build_p(dae)
    sol = solve_dc(dae, p)
    res = np.array(dae.currents(list(sol), p, 0.0))
    assert np.max(np.abs(res)) < 1e-9, res
    print("diode DC:", dict(zip(dae.unknowns(), np.round(sol, 6))))


if __name__ == "__main__":
    test_rc_dc()
    test_diode_dc()
    print("OK")
