"""Small-signal linearisation round trip.

`Model.linearize()` turns the nonlinear DAE into the linear DAE
`G dx + d/dt (C dx) = 0`, with the operating-point bias frozen into `name#op`
parameters. The contract: bound to the operating point, the linearised model's
`G` and `C` equal the original's there, whatever its own state.

    python -m pytest crates/py/tests/test_smallsignal.py
"""

import numpy as np

import sane

RLC = "I1 0 1 1\nR1 1 0 100\nL1 1 0 1m\nC1 1 0 1u\n.end"
MOSFET = (
    "VDD vdd 0 5\nVG g 0 2\nRD vdd d 5k\nM1 d g 0 0 NM\n"
    ".model NM NMOS(Kp=200u W=10 L=1 Vto=0.7 Lambda=0.02)\n.end"
)


def _gc(model, x, p):
    return model.jacobian_i_x(x, p), model.jacobian_q_x(x, p)


def _roundtrip(deck):
    model = sane.Model.from_netlist(deck)
    x = model.at().operating_point().x
    g0, c0 = _gc(model, x, model.param_vector())
    lin = model.linearize()
    assert lin.dim == model.dim and lin.unknowns == model.unknowns
    values = {k: v for k, v in model.values().items() if lin.is_param(k)}
    values.update({f"{u}#op": x[k] for k, u in enumerate(model.unknowns)})
    p = lin.param_vector(values)
    # linear: the same matrices at the operating point and anywhere else
    for state in (np.zeros_like(x), x + 0.37):
        g1, c1 = _gc(lin, state, p)
        np.testing.assert_allclose(g1, g0, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(c1, c0, rtol=1e-12, atol=1e-21)


def test_rlc_roundtrip():
    _roundtrip(RLC)


def test_mosfet_roundtrip():
    _roundtrip(MOSFET)
