"""S-parameter analysis (`Model.sp` / `Model.sp_fn`) and Touchstone I/O.

Port convention under test: an ideal V source in series with a z0 resistor,
the port node on the network side. References: the closed form of a shunt
resistor between two matched ports, unitarity and reciprocity of a lossless
LC ladder, a Touchstone write/read roundtrip in all three formats, and the
SP adjoint against central finite differences."""

import numpy as np
import pytest

import sane

SHUNT = """
VP1 t1 0 DC 0 AC 1
RZ1 t1 n1 50
VP2 t2 0 DC 0 AC 1
RZ2 t2 n1 50
Rsh n1 0 25
"""

# 3rd-order 0.05 uF/1.6 uH ladder: lossless, fc ~ 100 MHz scale
LC = """
VP1 t1 0 DC 0 AC 1
RZ1 t1 in 50
L1 in mid 1.591549u
C1 mid 0 636.6198p
L2 mid out 1.591549u
VP2 t2 0 DC 0 AC 1
RZ2 t2 out 50
"""

PORTS_SHUNT = [("VP1", "n1"), ("VP2", "n1")]
PORTS_LC = [("VP1", "in"), ("VP2", "out")]


def test_shunt_resistor_closed_form():
    # S11 = -Z0/(2R+Z0), S21 = 2R/(2R+Z0); R = 25, Z0 = 50 -> -0.5 / +0.5
    m = sane.parse(SHUNT).extract()
    sp = m.sp([1e3, 1e6], ports=PORTS_SHUNT)
    assert np.allclose(sp[1, 1], -0.5, atol=1e-9)
    assert np.allclose(sp[2, 1], 0.5, atol=1e-9)
    assert np.allclose(sp[2, 2], -0.5, atol=1e-9)
    assert np.allclose(sp[1, 2], 0.5, atol=1e-9)


def test_lossless_unitarity_and_reciprocity():
    m = sane.parse(LC).extract()
    sp = m.sp(np.geomspace(1e6, 200e6, 31), ports=PORTS_LC)
    power = np.abs(sp[1, 1]) ** 2 + np.abs(sp[2, 1]) ** 2
    assert np.max(np.abs(power - 1.0)) < 1e-8
    assert np.max(np.abs(sp[2, 1] - sp[1, 2])) < 1e-12


@pytest.mark.parametrize("fmt", ["RI", "MA", "DB"])
def test_touchstone_roundtrip(fmt, tmp_path):
    m = sane.parse(LC).extract()
    sp = m.sp(np.geomspace(1e6, 200e6, 11), ports=PORTS_LC)
    path = tmp_path / f"lc_{fmt}.s2p"
    sp.to_touchstone(path, fmt=fmt)
    back = sane.read_touchstone(path)
    assert back.nports == 2
    assert np.max(np.abs(back.freqs / sp.freqs - 1.0)) < 1e-9
    assert np.max(np.abs(back.s - sp.s)) < 1e-8
    assert back.z0[0] == 50.0


def test_vectfit_verilog_a_roundtrip(tmp_path):
    # sp -> S-to-Y -> vector fit -> Verilog-A -> N instance -> sp again:
    # the macromodel must reproduce the reference S over the fitted band,
    # and auto-order must find the network's true order (3 poles)
    m = sane.parse(LC).extract()
    freqs = np.geomspace(1e6, 300e6, 60)
    ref = m.sp(freqs, ports=PORTS_LC)
    va, fit_err, n_poles = ref.to_verilog_a("lcmacro", port_names=["a", "b"])
    assert fit_err < 1e-6
    assert n_poles == 3
    va_path = tmp_path / "lcmacro.va"
    va_path.write_text(va)
    deck = (
        f'.veriloga "{va_path}"\n'
        "VP1 t1 0 DC 0 AC 1\nRZ1 t1 in 50\n"
        "VP2 t2 0 DC 0 AC 1\nRZ2 t2 out 50\n"
        "N1 in out 0 lcmacro\n"
    )
    m2 = sane.parse(deck).extract()
    fit = m2.sp(freqs, ports=PORTS_LC)
    assert np.max(np.abs(fit.s - ref.s)) < 1e-6


def test_sp_fn_gradient_vs_fd():
    m = sane.parse(LC).extract()
    f = m.sp_fn(np.geomspace(5e6, 100e6, 5), ports=PORTS_LC, wrt=["L1", "C1", "L2"])
    p0 = np.array([1.591549e-6, 636.6198e-12, 1.591549e-6])
    s0 = f(p0)
    rng = np.random.default_rng(7)
    cot = rng.standard_normal(s0.shape) + 1j * rng.standard_normal(s0.shape)

    def loss(s):
        return float(np.sum(s.real * cot.real + s.imag * cot.imag))

    g = f.vjp(cot)
    for i in range(3):
        h = 1e-6 * p0[i]
        pp = p0.copy()
        pm = p0.copy()
        pp[i] += h
        pm[i] -= h
        fd = (loss(f(pp)) - loss(f(pm))) / (2 * h)
        assert abs(g[i] - fd) / max(abs(fd), 1e-12) < 1e-6


P_LC = """
P1 in 0 Z0=50
L1 in mid 1.591549u
C1 mid 0 636.6198p
L2 mid out 1.591549u
P2 out 0 Z0=50
"""


def test_p_element_ports():
    # deck-defined ports (`P` elements) must reproduce the explicit V+R form
    ref = sane.parse(LC).extract().sp(np.geomspace(1e6, 200e6, 21), ports=PORTS_LC)
    m = sane.parse(P_LC).extract()
    assert m._deck_ports == [("P1", "in", 50.0), ("P2", "out", 50.0)]
    sp = m.sp(np.geomspace(1e6, 200e6, 21))
    assert np.max(np.abs(sp.s - ref.s)) < 1e-9
    assert list(sp.z0) == [50.0, 50.0]
