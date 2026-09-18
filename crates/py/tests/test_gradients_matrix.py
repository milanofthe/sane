"""Gradient-vs-FD CI matrix across the differentiable analysis functions:
DC, AC (mag), transient and harmonic balance, on linear and nonlinear
circuits. Every `*_fn` VJP must match central finite differences of its own
forward to truncation accuracy. (The transient FD matrix over EVERY engine
parameter lives in test_transient_adjoint.py; this file covers the facade.)"""

import numpy as np
import pytest

import sane

RC = "V1 in 0 0 SIN(0 1 1k)\nR1 in out 1k\nC1 out 0 100n"
CLIPPER = (
    ".model Dmod D(Is=1e-12 N=1)\n"
    "V1 in 0 0.3 SIN(0.3 1 1k)\nR1 in out 1k\nD1 out 0 Dmod\nC1 out 0 200n"
)
CE = (
    ".model QN NPN(Is=1e-15 betaF=100)\n"
    "V1 in 0 0 SIN(0 10m 1k)\nV2 vcc 0 12\n"
    "C1 in nb 1u\nR1 vcc nb 100k\nR2 nb 0 22k\n"
    "Q1 out nb ne QN\nR3 vcc out 4.7k\nR4 ne 0 1k\nC2 ne 0 100u"
)


def _fd_check(f, p0, cot, rel=1e-3, tol=5e-4):
    """Central FD of L(p) = cot . f(p) against f.vjp(cot), per parameter."""
    p0 = np.asarray(p0, dtype=float)
    y = f(p0)
    g = f.vjp(cot)
    assert g.shape == (len(p0),)
    for j in range(len(p0)):
        h = abs(p0[j]) * rel
        pp, pm = p0.copy(), p0.copy()
        pp[j] += h
        pm[j] -= h
        fd = (np.dot(cot, np.atleast_1d(f(pp))) - np.dot(cot, np.atleast_1d(f(pm)))) / (2 * h)
        scale = max(abs(fd), abs(g[j]), 1e-12)
        assert abs(g[j] - fd) <= tol * scale, (
            f"{f.wrt[j]}: vjp {g[j]} vs FD {fd} (rel {(abs(g[j] - fd) / scale):.2e})"
        )
    return y, g


# --- DC --------------------------------------------------------------------


@pytest.mark.parametrize(
    "deck,out,wrt,p0",
    [
        (CLIPPER, "out", ["R1", "V1.sin_off"], [1e3, 0.3]),
        (CE, "out", ["R3", "R4", "R1"], [4.7e3, 1e3, 100e3]),
    ],
    ids=["clipper", "ce"],
)
def test_dc_fn_matches_fd(deck, out, wrt, p0):
    model = sane.Circuit.parse(deck).extract()
    f = model.dc_fn(out, wrt=wrt)
    _fd_check(f, p0, np.array([1.0]))


def test_dc_fn_warm_start_same_result():
    model = sane.Circuit.parse(CE).extract()
    warm = model.dc_fn("out", wrt=["R3"], warm=True)
    cold = model.dc_fn("out", wrt=["R3"], warm=False)
    a = warm(np.array([4.7e3]))
    assert warm._x_warm is not None  # warm state armed
    b = warm(np.array([4.8e3]))  # second solve seeds from the first
    c = cold(np.array([4.8e3]))
    np.testing.assert_allclose(b, c, rtol=1e-9)
    assert a != b


# --- AC (magnitude) --------------------------------------------------------


@pytest.mark.parametrize(
    "deck,inp,out,wrt,p0",
    [
        (RC, "V1", "out", ["R1", "C1"], [1e3, 100e-9]),
        (CE, "V1", "out", ["R3", "R4", "C2"], [4.7e3, 1e3, 100e-6]),
    ],
    ids=["rc", "ce"],
)
def test_ac_fn_mag_matches_fd(deck, inp, out, wrt, p0):
    model = sane.Circuit.parse(deck).extract()
    freqs = np.array([100.0, 1e3, 10e3])
    f = model.ac_fn(inp, out, freqs, wrt=wrt, metric="mag")
    cot = 0.5 + np.sin(0.7 * np.arange(len(freqs)))
    _fd_check(f, p0, cot)


def test_ac_fn_complex_vjp_matches_fd():
    model = sane.Circuit.parse(RC).extract()
    freqs = np.array([1e3])
    f = model.ac_fn("V1", "out", freqs, wrt=["R1", "C1"], metric="complex")
    p0 = np.array([1e3, 100e-9])
    cot = np.array([0.7 + 0.3j])  # dL/dRe + j dL/dIm
    f(p0)
    g = f.vjp(cot)

    def loss(p):
        h = f(p)[0]
        return 0.7 * h.real + 0.3 * h.imag

    for j in range(2):
        h = p0[j] * 1e-3
        pp, pm = p0.copy(), p0.copy()
        pp[j] += h
        pm[j] -= h
        fd = (loss(pp) - loss(pm)) / (2 * h)
        assert abs(g[j] - fd) <= 5e-4 * max(abs(fd), abs(g[j]))


# --- harmonic balance ------------------------------------------------------


def test_hb_fn_mag_matches_fd():
    model = sane.Circuit.parse(CLIPPER).extract()
    f = model.hb_fn("out", f0=1e3, harmonics=6, wrt=["R1", "C1", "V1.sin_amp"])
    p0 = np.array([1e3, 200e-9, 1.0])
    cot = 0.5 + np.sin(0.7 * np.arange(f.out_len))
    _fd_check(f, p0, cot, rel=1e-4, tol=2e-3)


# --- poles/zeros -----------------------------------------------------------


def test_pz_fn_matches_fd():
    model = sane.Circuit.parse(CE).extract()
    # R2/C2 shape the input and emitter poles directly (R3 drives no pole:
    # the CE model has no collector capacitance)
    f = model.pz_fn("V1", wrt=["R2", "C2"])
    p0 = np.array([22e3, 100e-6])
    s = f(p0)
    cot = (0.3 + 0.7j) * np.ones(len(s))  # dL/dRe + j dL/dIm per pole
    g = f.vjp(cot)

    def loss(p):
        poles = f(p)
        return float(np.sum(0.3 * poles.real + 0.7 * poles.imag))

    for j in range(len(p0)):
        h = p0[j] * 1e-4
        pp, pm = p0.copy(), p0.copy()
        pp[j] += h
        pm[j] -= h
        fd = (loss(pp) - loss(pm)) / (2 * h)
        # absolute floor: FD of eigenvalues carries eigensolver/DC noise
        # relative to the pole magnitudes
        noise = 1e-8 * float(np.max(np.abs(s)))
        scale = max(abs(fd), abs(g[j]), 1e-9)
        assert abs(g[j] - fd) <= 1e-3 * scale + noise, (
            f"{f.wrt[j]}: vjp {g[j]} vs FD {fd}"
        )


# --- runtime budgets: order-of-magnitude tripwires, generously bounded ------


def test_runtime_budgets():
    import time

    n = 30
    lines = ["V1 n0 0 0 SIN(0 1 1k)"]
    for i in range(1, n + 1):
        lines.append(f"R{i} n{i-1} n{i} 1k")
        lines.append(f"C{i} n{i} 0 100n")
    model = sane.Circuit.parse("\n".join(lines)).extract()
    t = np.linspace(0, 2e-3, 60)
    f = model.transient_fn("n3", t, wrt=["R1", "C1"])
    f(np.array([1e3, 100e-9]))  # warm/JIT-free baseline
    t0 = time.perf_counter()
    f(np.array([1.01e3, 100e-9]))
    f.vjp(np.ones(len(t)))
    assert time.perf_counter() - t0 < 2.0, "transient forward+vjp blew its budget"

    model2 = sane.Circuit.parse(CLIPPER).extract()
    fh = model2.hb_fn("out", f0=1e3, harmonics=6, wrt=["R1"])
    fh(np.array([1e3]))
    t0 = time.perf_counter()
    fh(np.array([1.01e3]))
    fh.vjp(np.ones(fh.out_len))
    assert time.perf_counter() - t0 < 2.0, "hb forward+vjp blew its budget"


# --- transient (facade smoke; the exhaustive matrix is in
#     test_transient_adjoint.py) ------------------------------------------


def test_transient_fn_warm_start_same_result():
    model = sane.Circuit.parse(CLIPPER).extract()
    t = np.linspace(0, 1e-3, 50)
    warm = model.transient_fn("out", t, wrt=["R1"], warm=True)
    cold = model.transient_fn("out", t, wrt=["R1"], warm=False)
    warm(np.array([1e3]))
    yw = warm(np.array([1.1e3]))
    yc = cold(np.array([1.1e3]))
    np.testing.assert_allclose(yw, yc, rtol=1e-6, atol=1e-9)
    gw = warm.vjp(np.ones(len(t)))
    gc = cold.vjp(np.ones(len(t)))
    np.testing.assert_allclose(gw, gc, rtol=1e-5)


# --- interop over the facade ----------------------------------------------


def test_as_jax_over_dcish_transient():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    model = sane.Circuit.parse(CE).extract()
    freqs = np.array([1e3, 10e3])
    f = model.ac_fn("V1", "out", freqs, wrt=["R3", "R4"], metric="mag")
    g = sane.interop.as_jax(f)
    p0 = jnp.array([4.7e3, 1e3])
    loss = lambda p: jnp.sum(g(p))  # noqa: E731
    got = np.asarray(jax.grad(loss)(p0))
    f(np.asarray(p0))
    want = f.vjp(np.ones(len(freqs)))
    np.testing.assert_allclose(got, want, rtol=1e-10)
