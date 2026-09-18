"""Regression for issue #39: a singular AC system must surface as NaN + a
catchable warning, never as a silent flat/zero response (~-600 dB) that
masquerades as a size limit or a real spectral null.

Run after ``maturin develop -m crates/py/Cargo.toml``:
    python -m pytest crates/py/tests/test_ac_singular.py
"""

import warnings

import numpy as np
import pytest

import sane
from sane import SaneNumericalWarning

# A lossless parallel LC tank driven by a current source. At the resonance
# w0 = 1/sqrt(LC) the parallel admittance jwC + 1/(jwL) is exactly zero, so the
# nodal matrix G + jwC is singular and the AC solve has no solution. With L=C=1,
# w0 = 1 rad/s, i.e. f0 = 1/(2*pi) Hz; 2*pi*(1/(2*pi)) == 1.0 exactly in IEEE754
# (SANE's Rust uses the same PI constant and the same op sequence), so the
# singular frequency is hit deterministically. Before the fix this frequency
# silently returned 0.0 -- a notch to -inf dB indistinguishable from a real null;
# now it is NaN + a SaneNumericalWarning.
TANK = "I1 0 1 1\nL1 1 0 1\nC1 1 0 1\n.end"
F0 = 1.0 / (2.0 * np.pi)
HEALTHY = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.end"


def test_singular_ac_frequency_warns_and_is_nan():
    dae = sane.Circuit.parse(TANK).extract()
    freqs = np.array([F0 * 0.5, F0, F0 * 2.0])
    with pytest.warns(SaneNumericalWarning, match="singular"):
        res = dae.ac("I1", "1", freqs)
    assert np.isnan(res.value[1]), "the resonance (singular) frequency must be NaN, not 0.0"
    assert np.isfinite(res.value[0]) and np.isfinite(res.value[2]), "off-resonance stays finite"


def test_healthy_ac_does_not_warn():
    dae = sane.Circuit.parse(HEALTHY).extract()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SaneNumericalWarning)
        res = dae.ac("V1", "out", np.logspace(1, 6, 40))
    assert np.isfinite(res.value).all()
