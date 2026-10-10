"""Derivatives read off the results at a point -- sensitivities and Hessians
of every analysis -- validated against finite differences.

The derivatives are exact autodiff (adjoints, second-order adjoints,
eigenvalue perturbation); the finite differences here only validate them.

    python -m pytest crates/py/tests/test_derivatives.py
"""

import numpy as np

import sane

# A nonlinear resistive divider with a diode load: a real operating-point metric
# with parameters that genuinely couple (R1, R2, the source, the diode).
DECK = "V1 in 0 2\nR1 in out 1k\nR2 out 0 2k\nD1 out 0 DM\n.model DM D(Is=1e-14 N=1)\n.end"
RC = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.end"
DIODE_RC = "V1 in 0 2\nR1 in out 1k\nC1 out 0 1u\nD1 out 0 DM\n.model DM D(Is=1e-14 N=1)\n.end"
MOS_AMP = (
    "VDD vdd 0 5\nVG g 0 1.2\nRD vdd d 5k\nM1 d g 0 0 NM\n"
    ".model NM NMOS(Kp=200u W=10 L=1 Vto=0.7 Lambda=0.02)\n.end"
)
HB_DECK = (
    "V1 in 0 SIN(0.6 0.15 1000)\n"
    "R1 in mid 1k\n"
    "D1 mid 0 DMOD\n"
    "C1 mid 0 100n\n"
    ".model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end"
)


def _model(deck):
    return sane.Model.from_netlist(deck)


def _by_name(params, values):
    return dict(zip(params, values))


def _fd(fn, model, pn, rel=1e-6):
    """Central difference of `fn(point)` by the parameter `pn`."""
    p0 = model[pn]
    h = abs(p0) * rel
    return (fn(model.at({pn: p0 + h})) - fn(model.at({pn: p0 - h}))) / (2 * h)


def test_dc_sensitivity_matches_fd():
    m = _model(DECK)
    s = m.at().operating_point().sensitivity("out")
    g = _by_name(s.params, s.grad[0])
    for pn in ("R1", "R2", "V1"):
        fd = _fd(lambda pt: pt.operating_point()["out"], m, pn)
        assert abs(g[pn] - fd) <= 1e-6 * (1.0 + abs(fd)), f"{pn}: AD {g[pn]} vs FD {fd}"


def test_dc_hessian_matches_fd_of_gradient():
    m = _model(DECK)
    knobs = ["R1", "R2", "V1"]
    hs = m.at().operating_point().hessian("out", knobs)
    H = hs.h[0]
    assert H.shape == (3, 3)
    assert np.allclose(H, H.T, atol=1e-8), "Hessian must be symmetric"
    for j, pj in enumerate(hs.params):
        def grad(pt):
            s = pt.operating_point().sensitivity("out", hs.params)
            return s.grad[0].copy()
        fd = _fd(grad, m, pj, rel=1e-5)
        for i in range(len(knobs)):
            assert abs(H[i, j] - fd[i]) <= 1e-4 * (1.0 + abs(fd[i])), (
                f"H[{hs.params[i]},{pj}] AD {H[i, j]} vs FD {fd[i]}"
            )


def test_ac_complex_gradient_matches_fd():
    """dH/dp by the AC adjoint (the operating point's shift included)
    matches finite differences of the response."""
    f = [1000.0]
    for deck, params, inp, out in (
        (RC, ["R1", "C1"], "V1", "out"),
        (DIODE_RC, ["R1", "C1", "dm.Is"], "V1", "out"),
        (MOS_AMP, ["nm.Kp", "RD"], "VG", "d"),
    ):
        m = _model(deck)
        s = m.at().ac(inp, out, f).sensitivity(params)
        g = _by_name(s.params, s.grad[0, 0])
        for pn in params:
            fd = _fd(lambda pt: pt.ac(inp, out, f).h[0, 0], m, pn)
            assert abs(g[pn] - fd) <= 1e-4 * (1.0 + abs(fd)), f"{pn}: AD {g[pn]} vs FD {fd}"


def test_ac_vjp_matches_the_sensitivity():
    """The weighted adjoint of a real loss equals the contraction of the
    per-parameter sensitivities."""
    m = _model(DIODE_RC)
    freqs = [10.0, 1e3, 1e5]
    ac = m.at().ac("V1", "out", freqs)
    rng = np.random.default_rng(3)
    cot = rng.standard_normal((1, 3)) + 1j * rng.standard_normal((1, 3))
    g = ac.vjp(cot)
    s = ac.sensitivity()
    want = np.real(np.einsum("ik,ikj->j", np.conj(cot), s.grad))
    got = _by_name(g.params, g.grad)
    for j, pn in enumerate(s.params):
        assert abs(got[pn] - want[j]) <= 1e-9 * (1.0 + abs(want[j])), pn


def test_ac_hessian_matches_fd_of_gradient():
    """The AC Hessian (second-order adjoint on the combined DC+AC system) is
    symmetric and matches finite differences of the exact gradient."""
    f = [1000.0]
    for deck in (RC, DIODE_RC):
        m = _model(deck)
        hs = m.at().ac("V1", "out", f).hessian(["R1", "C1"])
        H = hs.h[0, 0]
        assert np.allclose(H, H.T, atol=1e-9 * np.max(np.abs(H)))
        for j, pj in enumerate(hs.params):
            fd = _fd(lambda pt: pt.ac("V1", "out", f).sensitivity(hs.params).grad[0, 0].copy(), m, pj)
            for i in range(len(hs.params)):
                assert abs(H[i, j] - fd[i]) <= 1e-4 * (1.0 + abs(fd[i])), (i, j, H[i, j], fd[i])


def test_transient_sensitivity_matches_fd():
    """d output(t*)/dp by the forward sensitivities matches finite
    differences of the response at t*."""
    deck = "V1 in 0 SIN(0 1 1000)\nR1 in out 1k\nC1 out 0 1u\n.end"
    m = _model(deck)
    t = np.linspace(0, 2e-3, 21)
    s = m.at().transient(t).sensitivity("out", ["R1", "C1"])
    g = _by_name(s.params, s.grad[0, -1])
    for pn in ("R1", "C1"):
        fd = _fd(lambda pt: pt.transient(t).signal("out")[-1], m, pn, rel=1e-5)
        assert abs(g[pn] - fd) <= 1e-3 * (1.0 + abs(fd)), f"{pn}: fwd {g[pn]} vs FD {fd}"


def _nearest(roots, r):
    return roots[np.argmin(np.abs(roots - r))]


def test_pole_sensitivity_matches_fd():
    """ds/dp by eigenvalue perturbation (the operating point's shift
    included) matches finite differences of the poles."""
    m = _model("V1 in 0 1\nR1 in 1 50\nL1 1 0 1m\nC1 1 0 1u\n.end")
    rs = m.at().poles().sensitivity(["R1", "L1", "C1"])
    for r, pole in enumerate(rs.roots):
        g = _by_name(rs.params, rs.grad[r])
        for pn in ("R1", "L1", "C1"):
            fd = _fd(lambda pt: _nearest(np.asarray(pt.poles().poles), pole), m, pn)
            assert abs(g[pn] - fd) <= 1e-6 * (1.0 + abs(fd)), f"{pole} d/d{pn}: {g[pn]} vs {fd}"


def test_zero_sensitivity_matches_fd():
    """The transmission zeros' sensitivities (Rosenbrock pencil
    perturbation) match finite differences of the zeros."""
    m = _model("V1 in 0 1\nR1 in out 2k\nC1 in m 1u\nC2 m out 1u\nR2 m 0 1k\n.end")
    rs = m.at().zeros("V1", "out").sensitivity(["R1", "C1", "R2"])[0]
    for r, zero in enumerate(rs.roots):
        g = _by_name(rs.params, rs.grad[r])
        for pn in ("R1", "C1", "R2"):
            fd = _fd(lambda pt: _nearest(np.asarray(pt.zeros("V1", "out").of("out")), zero), m, pn)
            assert abs(g[pn] - fd) <= 1e-6 * (1.0 + abs(fd)), f"{zero} d/d{pn}: {g[pn]} vs {fd}"


def test_noise_sensitivity_matches_fd():
    """dS/dp of the output noise PSD (noise adjoint and the operating
    point's shift) matches finite differences of the PSD."""
    m = _model("V1 in 0 0\nR1 in out 1k\nR2 out 0 2k\n.end")
    f = [1e3]
    s = m.at().noise("out", f).sensitivity(["R1", "R2"])
    g = _by_name(s.params, s.grad[0, 0])
    for pn in ("R1", "R2"):
        fd = _fd(lambda pt: pt.noise("out", f).psd[0, 0], m, pn)
        assert abs(g[pn] - fd) <= 1e-6 * (1.0 + abs(fd)), f"dS/d{pn}: {g[pn]} vs {fd}"


def test_hb_sensitivity_matches_fd():
    """dX_k/dp by the implicit-function adjoint on the two-sided HB
    Jacobian matches finite differences of the converged coefficients."""
    m = _model(HB_DECK)
    K = 6
    hb = m.at().harmonic_balance(f0=1000.0, harmonics=K)
    knobs = ["R1", "C1", "dmod.Is"]
    s = hb.sensitivity("mid", knobs)
    for k in (1, 2):
        g = _by_name(s.params, s.grad[0, k])
        for pn in knobs:
            fd = _fd(lambda pt: pt.harmonic_balance(f0=1000.0, harmonics=K).spectrum("mid")[k], m, pn)
            assert abs(g[pn] - fd) <= 1e-4 * (1.0 + abs(fd)), f"dX[{k}]/d{pn}: AD {g[pn]} vs FD {fd}"


def _hb_hessian_vs_fd(deck, knobs):
    m = _model(deck)
    K, k = 6, 1
    hb = m.at().harmonic_balance(f0=1000.0, harmonics=K)
    hs = hb.hessian("mid", k, knobs)
    H = hs.h
    assert np.allclose(H, H.T, atol=1e-6 * np.max(np.abs(H)))
    Hfd = np.zeros_like(H)
    for j, pj in enumerate(hs.params):
        def grad(pt):
            r = pt.harmonic_balance(f0=1000.0, harmonics=K)
            return r.sensitivity("mid", hs.params).grad[0, k].copy()
        Hfd[:, j] = _fd(grad, m, pj)
    Hfd = 0.5 * (Hfd + Hfd.T)
    assert np.max(np.abs(H - Hfd)) <= 1e-4 * (1.0 + np.max(np.abs(Hfd)))


def test_hb_hessian_matches_fd():
    """The HB coefficient Hessian (second-order adjoint, device second
    derivatives through the AFT) matches finite differences of the exact
    gradient."""
    _hb_hessian_vs_fd(HB_DECK, ["R1", "C1", "dmod.Is"])


def test_hb_hessian_nonlinear_charge_matches_fd():
    """Exact also with nonlinear charge storage: a junction capacitance
    makes dF/dx' depend on x, so the rate Hessian blocks contribute; the
    charge parameters Cj0/Vj act only through them."""
    deck = (
        "V1 in 0 SIN(0.6 0.25 1000)\n"
        "R1 in mid 1k\n"
        "D1 mid 0 DM\n"
        "C1 mid 0 50n\n"
        ".model DM D(Is=1e-14 N=1 Cj0=20n Vj=0.7 M=0.5)\n.end"
    )
    _hb_hessian_vs_fd(deck, ["R1", "C1", "dm.Is", "dm.Cj0", "dm.Vj"])


def test_sensitivity_ranking():
    """The ranking orders by magnitude and the roll-up groups by device."""
    m = _model(DECK)
    s = m.at().operating_point().sensitivity("out")
    ranked = s.ranked("out")
    mags = [abs(v) for _, v in ranked]
    assert mags == sorted(mags, reverse=True)
    rel = s.relative()[0]
    by = _by_name(s.params, rel)
    for name, v in ranked:
        assert v == by[name]
    groups = dict(s.rollup("out"))
    assert "dm" in groups and groups["dm"] >= abs(by["dm.Is"])
