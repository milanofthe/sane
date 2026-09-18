"""Unified derivative API: sensitivities and sparse Hessians read off an analysis
result object, validated against finite differences.

Phase 1 covers the DC operating point: ``op.sensitivity(output)`` is the gradient
over every parameter (one adjoint solve, the parameter Jacobian is sparse), and
``op.hessian(output, subset)`` is the sparse second-order-adjoint Hessian over the
identified knob subset. Both are exact autodiff; finite differences here only
*validate* them.

Run after `maturin develop -m crates/py/Cargo.toml`:
    python -m pytest crates/py/tests/test_derivatives.py
"""

import numpy as np

import sane

# A nonlinear resistive divider with a diode load: a real operating-point metric
# with parameters that genuinely couple (R1, R2, the source, the diode).
DECK = "V1 in 0 2\nR1 in out 1k\nR2 out 0 2k\nD1 out 0 DM\n.model DM D(Is=1e-14 N=1)\n.end"


def _grad(sens):
    return dict(zip(sens.params, sens.gradient))


def test_dc_sensitivity_matches_fd():
    dae = sane.Circuit.parse(DECK).extract()
    op = dae.operating_point()
    g = _grad(op.sensitivity("out"))

    base = dae.values
    for pn in ("R1", "R2", "V1"):
        p0 = base[pn]
        h = abs(p0) * 1e-6
        yp = dae.operating_point(values={pn: p0 + h})["out"]
        ym = dae.operating_point(values={pn: p0 - h})["out"]
        fd = (yp - ym) / (2 * h)
        assert abs(g[pn] - fd) <= 1e-6 * (1.0 + abs(fd)), f"{pn}: AD {g[pn]} vs FD {fd}"


def test_dc_hessian_matches_fd_of_gradient():
    dae = sane.Circuit.parse(DECK).extract()
    op = dae.operating_point()
    knobs = ["R1", "R2", "V1"]
    H = op.hessian("out", knobs)
    assert H.shape == (3, 3)
    assert np.allclose(H, H.T, atol=1e-8), "Hessian must be symmetric"

    # FD of the adjoint gradient w.r.t. each knob reproduces the Hessian columns.
    base = dae.values
    for j, pj in enumerate(knobs):
        p0 = base[pj]
        h = abs(p0) * 1e-5
        gp = _grad(dae.operating_point(values={pj: p0 + h}).sensitivity("out"))
        gm = _grad(dae.operating_point(values={pj: p0 - h}).sensitivity("out"))
        for i, pi in enumerate(knobs):
            fd = (gp[pi] - gm[pi]) / (2 * h)
            assert abs(H[i, j] - fd) <= 1e-4 * (1.0 + abs(fd)), (
                f"H[{pi},{pj}] AD {H[i, j]} vs FD {fd}"
            )


RC = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.end"
DIODE_RC = "V1 in 0 2\nR1 in out 1k\nC1 out 0 1u\nD1 out 0 DM\n.model DM D(Is=1e-14 N=1)\n.end"


def _ac_cgrad(dae, f, x, p):
    rows = dae._d.ac_gradient("V1", dae._idx("out"), list(x), list(p), float(f))
    return {r[0]: complex(r[1], r[2]) for r in rows}


def test_ac_complex_gradient_matches_fd():
    """The AC adjoint gradient dH/dp (all params, including the operating-point
    shift) matches finite differences of the numeric response."""
    f = 1000.0
    for deck, params in ((RC, ["R1", "C1"]), (DIODE_RC, ["R1", "C1", "D1.Is"])):
        dae = sane.Circuit.parse(deck).extract()
        res = dae.ac("V1", "out", [f])
        g = _ac_cgrad(dae, f, res._x, res._p)
        base = dae.values
        for pn in params:
            p0 = base[pn]
            h = abs(p0) * 1e-6
            hp = dae.ac("V1", "out", [f], values={pn: p0 + h}).value[0]
            hm = dae.ac("V1", "out", [f], values={pn: p0 - h}).value[0]
            fd = (hp - hm) / (2 * h)
            assert abs(g[pn] - fd) <= 1e-4 * (1.0 + abs(fd)), f"{pn}: AD {g[pn]} vs FD {fd}"


MOS_AMP = (
    "VDD vdd 0 5\nVG g 0 1.2\nRD vdd d 5k\nM1 d g 0 0 NM\n"
    ".model NM NMOS(Kp=200u W=10 L=1 Vto=0.7 Lambda=0.02)\n.end"
)


def test_ac_sensitivity_nonlinear_param_matches_fd():
    """Regression for issue #38: the per-parameter AC sensitivity ``dH/dp`` via
    ``Model.ac_sensitivity`` (``ac_derivatives`` under the hood) must include the
    operating-point-shift term ``dx/dp``. For a NONLINEAR device parameter the
    small-signal conductances depend on bias, so that term is typically dominant.

    Before the fix this path never called ``ensure_param_jac``: ``dF/dp`` was
    unbuilt, ``state_sensitivity`` returned an all-zeros ``dx/dp``, and the
    chain-rule term ``(dG/dx)(dx/dp)`` was silently dropped -- the diode ``Is`` and
    MOSFET ``Kp`` sensitivities came out wrong (linear R/C params stayed exact,
    hiding the bug). FD validates.
    """
    f = 1000.0
    # Diode Is: nonlinear conductance g = Is/Vt * exp(Vd/Vt) depends on the bias.
    dae = sane.Circuit.parse(DIODE_RC).extract()
    base = dae.values
    for pn in ("D1.Is", "R1", "C1"):
        ad = dae.ac_sensitivity("V1", pn, "out", [f])[0]
        p0 = base[pn]
        h = abs(p0) * 1e-6
        hp = dae.ac("V1", "out", [f], values={pn: p0 + h}).value[0]
        hm = dae.ac("V1", "out", [f], values={pn: p0 - h}).value[0]
        fd = (hp - hm) / (2 * h)
        assert abs(ad - fd) <= 1e-4 * (1.0 + abs(fd)), f"diode {pn}: AC-sens {ad} vs FD {fd}"

    # MOSFET Kp: transconductance gm = sqrt(2 Kp W/L Id) depends on the bias too.
    amp = sane.Circuit.parse(MOS_AMP).extract()
    base = amp.values
    for pn in ("M1.Kp", "RD"):
        ad = amp.ac_sensitivity("VG", pn, "d", [f])[0]
        p0 = base[pn]
        h = abs(p0) * 1e-6
        hp = amp.ac("VG", "d", [f], values={pn: p0 + h}).value[0]
        hm = amp.ac("VG", "d", [f], values={pn: p0 - h}).value[0]
        fd = (hp - hm) / (2 * h)
        assert abs(ad - fd) <= 1e-4 * (1.0 + abs(fd)), f"mosfet {pn}: AC-sens {ad} vs FD {fd}"


def test_ac_metric_sensitivity_matches_fd():
    """The projected magnitude sensitivity matches FD of |H|."""
    f = 1000.0
    dae = sane.Circuit.parse(RC).extract()
    res = dae.ac("V1", "out", [f])
    g = dict(zip(res.sensitivity(f, "mag").params, res.sensitivity(f, "mag").gradient))
    base = dae.values
    for pn in ("R1", "C1"):
        p0 = base[pn]
        h = abs(p0) * 1e-6
        mp = abs(dae.ac("V1", "out", [f], values={pn: p0 + h}).value[0])
        mm = abs(dae.ac("V1", "out", [f], values={pn: p0 - h}).value[0])
        fd = (mp - mm) / (2 * h)
        assert abs(g[pn] - fd) <= 1e-5 * (1.0 + abs(fd)), f"|H| d/d{pn}: {g[pn]} vs {fd}"


def test_ac_hessian_analytic_matches_fd_of_gradient():
    """The analytic AC Hessian (second-order adjoint on the combined DC+AC system)
    is symmetric and matches finite differences of the EXACT analytic gradient
    (the FD here is only the validation reference, not how the Hessian is built)."""
    f = 1000.0
    knobs = ["R1", "C1"]
    for deck in (RC, DIODE_RC):
        dae = sane.Circuit.parse(deck).extract()
        H = dae.ac("V1", "out", [f]).hessian(f, knobs, "mag")
        assert H.shape == (2, 2)
        assert np.allclose(H, H.T, atol=1e-9)

        def gmag(over):
            s = dae.ac("V1", "out", [f], values=over).sensitivity(f, "mag")
            return dict(zip(s.params, s.gradient))

        base = dae.values
        hfd = np.zeros((2, 2))
        for j, pj in enumerate(knobs):
            p0 = base[pj]
            h = abs(p0) * 1e-6
            gp, gm = gmag({pj: p0 + h}), gmag({pj: p0 - h})
            for i, pi in enumerate(knobs):
                hfd[i, j] = (gp[pi] - gm[pi]) / (2 * h)
        hfd = 0.5 * (hfd + hfd.T)
        scale = 1.0 + np.max(np.abs(hfd))
        assert np.max(np.abs(H - hfd)) <= 1e-4 * scale


def test_transient_sensitivity_matches_fd():
    """The forward transient sensitivity d(output(t*))/dp matches finite
    differences of the transient response value at t*."""
    deck = "V1 in 0 SIN(0 1 1000)\nR1 in out 1k\nC1 out 0 1u\n.end"
    dae = sane.Circuit.parse(deck).extract()
    t = np.linspace(0, 2e-3, 21)
    tr = dae.transient(t)
    s = tr.sensitivity("out", wrt=["R1", "C1"])
    g = dict(zip(s.params, s.gradient))
    base = dae.values
    for pn in ("R1", "C1"):
        p0 = base[pn]
        h = abs(p0) * 1e-5
        yp = dae.transient(t, values={pn: p0 + h})["out"][-1]
        ym = dae.transient(t, values={pn: p0 - h})["out"][-1]
        fd = (yp - ym) / (2 * h)
        assert abs(g[pn] - fd) <= 1e-3 * (1.0 + abs(fd)), f"{pn}: fwd {g[pn]} vs FD {fd}"


def test_pole_sensitivity_matches_fd():
    """Analytic all-parameter pole sensitivity ds/dp (eigenvalue perturbation +
    adjoint operating-point shift) matches finite differences of the pole location."""
    deck = "V1 in 0 1\nR1 in 1 50\nL1 1 0 1m\nC1 1 0 1u\n.end"
    dae = sane.Circuit.parse(deck).extract()
    ss = dae.small_signal("V1", "1")
    base = dae.values

    def poles_at(over):
        return dae.small_signal("V1", "1", values=over).poles()

    for pole, sens in ss.pole_sensitivity():
        cg = dict(zip(sens.params, sens.complex))
        for pn in ("R1", "L1", "C1"):
            p0 = base[pn]
            h = abs(p0) * 1e-6
            pp = poles_at({pn: p0 + h})
            pm = poles_at({pn: p0 - h})
            sp = pp[np.argmin(np.abs(pp - pole))]
            sm = pm[np.argmin(np.abs(pm - pole))]
            fd = (sp - sm) / (2 * h)
            assert abs(cg[pn] - fd) <= 1e-6 * (1.0 + abs(fd)), f"{pole} d/d{pn}: {cg[pn]} vs {fd}"


def test_zero_sensitivity_matches_fd():
    """Analytic all-parameter transmission-zero sensitivity (Rosenbrock pencil
    perturbation + adjoint shift) matches finite differences of the zero location."""
    deck = "V1 in 0 1\nR1 in out 2k\nC1 in m 1u\nC2 m out 1u\nR2 m 0 1k\n.end"
    dae = sane.Circuit.parse(deck).extract()
    ss = dae.small_signal("V1", "out")
    base = dae.values

    def zeros_at(over):
        return dae.small_signal("V1", "out", values=over).zeros()

    for zero, sens in ss.zero_sensitivity():
        cg = dict(zip(sens.params, sens.complex))
        for pn in ("R1", "C1", "R2"):
            p0 = base[pn]
            h = abs(p0) * 1e-6
            zp = zeros_at({pn: p0 + h})
            zm = zeros_at({pn: p0 - h})
            sp = zp[np.argmin(np.abs(zp - zero))]
            sm = zm[np.argmin(np.abs(zm - zero))]
            fd = (sp - sm) / (2 * h)
            assert abs(cg[pn] - fd) <= 1e-6 * (1.0 + abs(fd)), f"{zero} d/d{pn}: {cg[pn]} vs {fd}"


def test_noise_sensitivity_matches_fd():
    """Analytic output-noise PSD sensitivity dS_v/dp (noise adjoint + extra solve
    + DC adjoint shift) matches finite differences of the noise PSD."""
    deck = "V1 in 0 0\nR1 in out 1k\nR2 out 0 2k\n.end"
    dae = sane.Circuit.parse(deck).extract()
    f = 1e3
    s = dae.noise("out", f, f * 1.0001, 2).sensitivity(f)
    g = dict(zip(s.params, s.gradient))

    def psd_at(over):
        return dae.noise("out", f, f * 1.0001, 2, values=over).noise[0] ** 2

    base = dae.values
    for pn in ("R1", "R2"):
        p0 = base[pn]
        h = abs(p0) * 1e-6
        fd = (psd_at({pn: p0 + h}) - psd_at({pn: p0 - h})) / (2 * h)
        assert abs(g[pn] - fd) <= 1e-6 * (1.0 + abs(fd)), f"dS_v/d{pn}: {g[pn]} vs {fd}"


def test_hb_sensitivity_matches_fd():
    """Analytic harmonic-balance coefficient sensitivity dX_k/dp (implicit-
    function adjoint on the exact two-sided Toeplitz Jacobian, with dF/dp routed
    through the AFT) matches finite differences of the converged HB coefficient,
    for both the complex coefficient and its magnitude. FD here only validates."""
    deck = (
        "V1 in 0 SIN(0.6 0.15 1000)\n"
        "R1 in mid 1k\n"
        "D1 mid 0 DMOD\n"
        "C1 mid 0 100n\n"
        ".model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end"
    )
    dae = sane.Circuit.parse(deck).extract()
    f0, K = 1000.0, 6
    hb = dae.harmonic_balance(f0, harmonics=K)
    assert hb.converged
    base = dae.values

    for k in (1, 2):
        sc = hb.sensitivity("mid", k, metric="coeff")
        cg = dict(zip(sc.params, sc.complex))
        sm = hb.sensitivity("mid", k, metric="mag")
        gm = dict(zip(sm.params, sm.gradient))
        for pn in ("R1", "C1", "D1.Is"):
            p0 = base[pn]
            h = abs(p0) * 1e-6
            xp = dae.harmonic_balance(f0, harmonics=K, values={pn: p0 + h}).harmonic("mid", k)
            xm = dae.harmonic_balance(f0, harmonics=K, values={pn: p0 - h}).harmonic("mid", k)
            fd = (xp - xm) / (2 * h)
            assert abs(cg[pn] - fd) <= 1e-4 * (1.0 + abs(fd)), (
                f"dX[{k}]/d{pn}: AD {cg[pn]} vs FD {fd}"
            )
            fd_mag = (abs(xp) - abs(xm)) / (2 * h)
            assert abs(gm[pn] - fd_mag) <= 1e-4 * (1.0 + abs(fd_mag)), (
                f"d|X[{k}]|/d{pn}: AD {gm[pn]} vs FD {fd_mag}"
            )


def test_hb_hessian_matches_fd():
    """Analytic HB coefficient Hessian (second-order adjoint on the exact
    two-sided Jacobian, device second derivatives contracted through the AFT)
    is symmetric and matches finite differences of the EXACT analytic gradient,
    for both the complex coefficient and its magnitude. FD validates only."""
    deck = (
        "V1 in 0 SIN(0.6 0.15 1000)\n"
        "R1 in mid 1k\n"
        "D1 mid 0 DMOD\n"
        "C1 mid 0 100n\n"
        ".model DMOD D(Is=1e-14 N=1 Vt=0.02585)\n.end"
    )
    dae = sane.Circuit.parse(deck).extract()
    f0, K, k = 1000.0, 6, 1
    knobs = ["R1", "C1", "D1.Is"]
    hb = dae.harmonic_balance(f0, harmonics=K)
    assert hb.converged
    base = dae.values
    ns = len(knobs)

    # Complex coefficient Hessian d^2 X_k / dp_a dp_b.
    Hc = hb.hessian("mid", k, knobs, metric="coeff")
    assert Hc.shape == (ns, ns)
    assert np.allclose(Hc, Hc.T, atol=1e-6 * np.max(np.abs(Hc)))

    def cgrad(over):
        s = dae.harmonic_balance(f0, harmonics=K, values=over).sensitivity("mid", k, "coeff")
        return dict(zip(s.params, s.complex))

    Hfd = np.zeros((ns, ns), dtype=complex)
    for j, pj in enumerate(knobs):
        p0 = base[pj]
        h = abs(p0) * 1e-6
        gp, gm = cgrad({pj: p0 + h}), cgrad({pj: p0 - h})
        for i, pi in enumerate(knobs):
            Hfd[i, j] = (gp[pi] - gm[pi]) / (2 * h)
    Hfd = 0.5 * (Hfd + Hfd.T)
    assert np.max(np.abs(Hc - Hfd)) <= 1e-4 * (1.0 + np.max(np.abs(Hfd)))

    # Magnitude Hessian d^2 |X_k| / dp_a dp_b.
    Hm = hb.hessian("mid", k, knobs, metric="mag")
    assert np.allclose(Hm, Hm.T, atol=1e-6 * (1.0 + np.max(np.abs(Hm))))

    def mgrad(over):
        s = dae.harmonic_balance(f0, harmonics=K, values=over).sensitivity("mid", k, "mag")
        return dict(zip(s.params, s.gradient))

    Hmfd = np.zeros((ns, ns))
    for j, pj in enumerate(knobs):
        p0 = base[pj]
        h = abs(p0) * 1e-6
        gp, gm = mgrad({pj: p0 + h}), mgrad({pj: p0 - h})
        for i, pi in enumerate(knobs):
            Hmfd[i, j] = (gp[pi] - gm[pi]) / (2 * h)
    Hmfd = 0.5 * (Hmfd + Hmfd.T)
    assert np.max(np.abs(Hm - Hmfd)) <= 1e-4 * (1.0 + np.max(np.abs(Hmfd)))


def test_hb_hessian_nonlinear_charge_matches_fd():
    """The HB Hessian stays exact with NONLINEAR charge storage: a diode with
    junction capacitance (Cj0/Vj/M) makes dF/dx' depend on x, so the rate (x')
    Hessian blocks (L_x'x', L_x x', L_x'p) must contribute. Knobs include the
    charge parameters Cj0/Vj, which act only through that path -- their Hessian
    entries would be wrong if the rate blocks were dropped. FD validates."""
    deck = (
        "V1 in 0 SIN(0.6 0.25 1000)\n"
        "R1 in mid 1k\n"
        "D1 mid 0 DM\n"
        "C1 mid 0 50n\n"
        ".model DM D(Is=1e-14 N=1 Cj0=20n Vj=0.7 M=0.5)\n.end"
    )
    dae = sane.Circuit.parse(deck).extract()
    f0, K, k = 1000.0, 6, 1
    knobs = ["R1", "C1", "D1.Is", "D1.Cj0", "D1.Vj"]
    hb = dae.harmonic_balance(f0, harmonics=K)
    assert hb.converged
    base = dae.values
    ns = len(knobs)

    Hc = hb.hessian("mid", k, knobs, metric="coeff")
    assert np.allclose(Hc, Hc.T, atol=1e-6 * np.max(np.abs(Hc)))

    def cgrad(over):
        s = dae.harmonic_balance(f0, harmonics=K, values=over).sensitivity("mid", k, "coeff")
        return dict(zip(s.params, s.complex))

    Hfd = np.zeros((ns, ns), dtype=complex)
    for j, pj in enumerate(knobs):
        p0 = base[pj]
        h = abs(p0) * 1e-6
        gp, gm = cgrad({pj: p0 + h}), cgrad({pj: p0 - h})
        for i, pi in enumerate(knobs):
            Hfd[i, j] = (gp[pi] - gm[pi]) / (2 * h)
    Hfd = 0.5 * (Hfd + Hfd.T)
    assert np.max(np.abs(Hc - Hfd)) <= 1e-4 * (1.0 + np.max(np.abs(Hfd)))


def test_operating_point_and_dae_sensitivity_agree():
    """The result-object path and the convenience ``dae.sensitivity`` are one
    source of truth: identical gradients."""
    dae = sane.Circuit.parse(DECK).extract()
    g_op = _grad(dae.operating_point().sensitivity("out"))
    g_dae = _grad(dae.sensitivity("out"))
    assert set(g_op) == set(g_dae)
    for k in g_op:
        assert abs(g_op[k] - g_dae[k]) <= 1e-12 * (1.0 + abs(g_dae[k]))
