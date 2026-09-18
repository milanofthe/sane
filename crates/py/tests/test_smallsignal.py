"""Small-signal linearisation: the canonical KS graph and the AC round-trip.

`DAE.linearize()` turns the nonlinear DAE into the linear mass-matrix DAE
`G dx + C dx' = 0`, with the operating-point bias frozen into `name#op` symbols.
With `canonical=True` each stamp is one canonical element, so the small-signal
element graph lives directly in the returned DAE. The contract: the linearised
DAE's `A(s)` equals the original's, evaluated at the same operating point.

Run after `maturin develop -m crates/py/Cargo.toml`:
    python -m pytest crates/py/tests/test_smallsignal.py
"""

import random

import sane

RLC = "I1 0 1 1\nR1 1 0 100\nL1 1 0 1m\nC1 1 0 1u\n.end"
MOSFET = (
    "VDD vdd 0 5\nVG g 0 2\nRD vdd d 5k\nM1 d g 0 0 NM\n"
    ".model NM NMOS(Kp=200u W=10 L=1 Vto=0.7 Lambda=0.02)\n.end"
)


def _free_symbols(expr):
    fs = expr.free_symbols
    return fs() if callable(fs) else fs


def _aligned_env(matrices, seed):
    """A numeric binding for every symbol of the matrices, with each `X#op`
    operating-point symbol pinned to the same value as its base `X` -- so the
    frozen and unfrozen matrices are evaluated at the same point. Symbols are kept
    on the thermal-voltage scale so device exponentials stay finite; `$temp` is
    physical and `s = j*omega`."""
    rng = random.Random(seed)
    names = set()
    for a in matrices:
        for row in a:
            for e in row:
                names |= set(_free_symbols(e))
    base_val = {}

    def val_for(base):
        if base == "$temp" or base.lower().endswith("tnom"):
            return 300.15
        if base == "s":
            return 1j * 2 * 3.14159 * 1e3
        if base not in base_val:
            base_val[base] = 0.02 + 0.03 * rng.random()
        return base_val[base]

    env = {}
    for nm in names:
        base = nm[:-3] if nm.endswith("#op") else nm
        env[nm] = val_for(base)
    return env


def _max_matrix_diff(dae):
    lin = dae.linearize(canonical=True)
    a0, a1 = dae.system_matrix(), lin.system_matrix()
    assert len(a0) == len(a1) == dae.dim
    worst = 0.0
    for seed in (1, 7, 31):
        env = _aligned_env([a0, a1], seed)
        for r0, r1 in zip(a0, a1):
            for e0, e1 in zip(r0, r1):
                worst = max(worst, abs(e0.eval_complex(**env) - e1.eval_complex(**env)))
    return worst


def test_rlc_roundtrip():
    dae = sane.Circuit.parse(RLC).extract()
    assert _max_matrix_diff(dae) < 1e-9


def test_mosfet_roundtrip():
    dae = sane.Circuit.parse(MOSFET).extract()
    assert _max_matrix_diff(dae) < 1e-9


def test_canonical_stamps_are_linear():
    """Every residual of the canonical small-signal DAE is linear: its second
    derivative w.r.t. each unknown vanishes."""
    dae = sane.Circuit.parse(MOSFET).extract()
    lin = dae.linearize(canonical=True)
    ctx = lin.symbolic_context
    res = lin.residuals
    xs = [ctx.sym(u) for u in lin.unknowns]
    for f in res:
        for x in xs:
            assert f.diff(x).diff(x).is_zero()


def test_linearize_preserves_dimension():
    dae = sane.Circuit.parse(MOSFET).extract()
    assert dae.linearize().dim == dae.dim
    assert dae.linearize(canonical=True).unknowns == dae.unknowns
