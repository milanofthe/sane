"""The raw numeric interface (`Model.core`): currents, charges, the residual
and the sparse Jacobians agree with each other and with finite differences.

Run after `maturin develop -m crates/py/Cargo.toml`:
    python -m pytest crates/py/tests/test_raw_api.py
"""

import numpy as np

import sane

DECK = (
    "VDD vdd 0 5\nRD vdd d 2k\nD1 d k DM\nRK k 0 1k\nCK k 0 1n\n"
    ".model DM D(Is=1e-14 Cjo=2p Vj=0.7 M=0.5)\n.end"
)


def _dense(coo, shape):
    m = np.zeros(shape)
    for r, c, v in zip(*coo):
        m[r, c] += v
    return m


def test_raw_numeric_interface_is_consistent():
    model = sane.Circuit.parse(DECK).extract()
    d = model.core
    n = model.dim
    p = [float(model.values.get(k, 0.0)) for k in model.params]
    x = list(np.asarray(model.operating_point().vector) + 0.01)
    xdot = list(np.linspace(-1.0, 1.0, n))

    i = np.array(d.currents(x, p, 0.0))
    c = np.array(d.jacobian_q_x(x, p, 0.0))
    np.testing.assert_allclose(_dense(d.jacobian_q_x_sparse(x, p, 0.0), (n, n)), c)
    np.testing.assert_allclose(d.residual(x, xdot, p, 0.0), i + c @ xdot, rtol=1e-12)

    # dQ/dx and dQ/dp, dI/dp against central differences
    h = 1e-6
    for k in range(n):
        xp, xm = list(x), list(x)
        xp[k] += h
        xm[k] -= h
        fd = (np.array(d.charges(xp, p, 0.0)) - np.array(d.charges(xm, p, 0.0))) / (2 * h)
        np.testing.assert_allclose(fd, c[:, k], rtol=1e-5, atol=1e-18)
    gi = _dense(d.jacobian_i_p_sparse(x, p, 0.0), (n, len(p)))
    gq = _dense(d.jacobian_q_p_sparse(x, p, 0.0), (n, len(p)))
    for k in range(len(p)):
        s = h * max(1.0, abs(p[k]))
        pp, pm = list(p), list(p)
        pp[k] += s
        pm[k] -= s
        fi = (np.array(d.currents(x, pp, 0.0)) - np.array(d.currents(x, pm, 0.0))) / (2 * s)
        fq = (np.array(d.charges(x, pp, 0.0)) - np.array(d.charges(x, pm, 0.0))) / (2 * s)
        np.testing.assert_allclose(gi[:, k], fi, rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(gq[:, k], fq, rtol=1e-4, atol=1e-21)
