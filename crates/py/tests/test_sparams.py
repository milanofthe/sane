"""S-parameter analysis (`Point.s_parameters`, `SpFunction`) and Touchstone
I/O.

The ports are the deck's `P` elements (an ideal source behind its z0).
References: the closed form of a shunt resistor between two matched ports,
unitarity and reciprocity of a lossless LC ladder, a Touchstone write/read
roundtrip in all three formats, and the SP adjoint against central finite
differences."""

import numpy as np
import pytest

import sane

SHUNT = """
P1 n1 0 Z0=50
P2 n1 0 Z0=50
Rsh n1 0 25
"""

# 3rd-order 0.05 uF/1.6 uH ladder: lossless, fc ~ 100 MHz scale
LC = """
P1 in 0 Z0=50
L1 in mid 1.591549u
C1 mid 0 636.6198p
L2 mid out 1.591549u
P2 out 0 Z0=50
"""


def _sp(deck, freqs):
    return sane.Model.from_netlist(deck).at().s_parameters(freqs)


def test_shunt_resistor_closed_form():
    # S11 = -Z0/(2R+Z0), S21 = 2R/(2R+Z0); R = 25, Z0 = 50 -> -0.5 / +0.5
    s = _sp(SHUNT, [1e3, 1e6]).s
    assert np.allclose(s[:, 0, 0], -0.5, atol=1e-9)
    assert np.allclose(s[:, 1, 0], 0.5, atol=1e-9)
    assert np.allclose(s[:, 1, 1], -0.5, atol=1e-9)
    assert np.allclose(s[:, 0, 1], 0.5, atol=1e-9)


def test_lossless_unitarity_and_reciprocity():
    s = _sp(LC, np.geomspace(1e6, 200e6, 31)).s
    power = np.abs(s[:, 0, 0]) ** 2 + np.abs(s[:, 1, 0]) ** 2
    assert np.max(np.abs(power - 1.0)) < 1e-8
    assert np.max(np.abs(s[:, 1, 0] - s[:, 0, 1])) < 1e-12


def test_ports_are_the_decks():
    m = sane.Model.from_netlist(LC)
    assert m.ports == [("P1", "in", 50.0), ("P2", "out", 50.0)]
    sp = m.at().s_parameters([1e6])
    assert sp.port_names == ["P1", "P2"]
    assert list(sp.z0) == [50.0, 50.0]


@pytest.mark.parametrize("fmt", ["RI", "MA", "DB"])
def test_touchstone_roundtrip(fmt, tmp_path):
    sp = _sp(LC, np.geomspace(1e6, 200e6, 11))
    data = sane.SParams(sp.freqs, sp.s, sp.z0)
    path = tmp_path / f"lc_{fmt}.s2p"
    data.to_touchstone(path, fmt=fmt)
    back = sane.read_touchstone(path)
    assert back.nports == 2
    assert np.max(np.abs(back.freqs / sp.freqs - 1.0)) < 1e-9
    assert np.max(np.abs(back.s - sp.s)) < 1e-8
    assert back.z0[0] == 50.0


def test_vectfit_verilog_a_roundtrip(tmp_path):
    # S -> S-to-Y -> vector fit -> Verilog-A -> N instance -> S again: the
    # macromodel must reproduce the reference S over the fitted band, and
    # auto-order must find the network's true order (3 poles)
    freqs = np.geomspace(1e6, 300e6, 60)
    ref = _sp(LC, freqs)
    va, fit_err, n_poles = sane.SParams(ref.freqs, ref.s, ref.z0).to_verilog_a(
        "lcmacro", port_names=["a", "b"]
    )
    assert fit_err < 1e-6
    assert n_poles == 3
    va_path = tmp_path / "lcmacro.va"
    va_path.write_text(va)
    deck = f'.veriloga "{va_path}"\nP1 in 0 Z0=50\nP2 out 0 Z0=50\nN1 in out 0 lcmacro\n'
    fit = _sp(deck, freqs)
    assert np.max(np.abs(fit.s - ref.s)) < 1e-6


def test_sp_function_gradient_vs_fd():
    m = sane.Model.from_netlist(LC)
    f = sane.SpFunction(m, np.geomspace(5e6, 100e6, 5), wrt=["L1", "C1", "L2"])
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
