"""Cross-validate SANE's DAE export against ngspice DC operating points.

For each deck: solve F(x, xdot=0, p_default, t=0) = 0 with scipy (the exported
residual + Jacobian), and compare node voltages to `ngspice -b` `.op`.

Run after `maturin develop -m crates/py/Cargo.toml`:
    python crates/py/tests/validate_ngspice.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.optimize import fsolve

import sane


def find_ngspice():
    # Prefer the console build: on Windows `ngspice.exe` is the GUI build and
    # writes nothing to stdout; `ngspice_con.exe` is the batch/console one.
    for name in ("ngspice_con", "ngspice"):
        p = shutil.which(name)
        if p:
            return p
        for cand in (
            os.path.join(sys.prefix, "Library", "bin", f"{name}.exe"),  # win conda
            os.path.join(sys.prefix, "bin", name),
        ):
            if os.path.exists(cand):
                return cand
    raise RuntimeError("ngspice not found")


def sane_dc(deck):
    """Node-name -> DC voltage, solved via SANE's Rust-evaluated residual +
    Jacobian (no code generation)."""
    c = sane._core.parse(deck)
    names = c.node_names()
    dae = c.extract_dae()
    unk = dae.unknowns()

    vals = dae.values()
    p = [float(vals.get(name, 0.0)) for name in dae.params()]
    assert all(name in vals for name in dae.params()), ("unbound params", dae.params())

    n = dae.dim()
    z = [0.0] * n
    sol = fsolve(
        lambda x: np.array(dae.residual(list(x), z, p, 0.0)),
        np.zeros(n),
        fprime=lambda x: np.array(dae.jacobian_x(list(x), z, p, 0.0)),
    )
    volts = {}
    for k, name in enumerate(names):
        if k == 0:
            continue
        key = f"v{k}"
        if key in unk:
            volts[name] = sol[unk.index(key)]
    return volts


def ngspice_dc(ng, deck, nodes):
    """Node-name -> DC voltage from ngspice `.op`."""
    prints = " ".join(f"v({n})" for n in nodes)
    text = f"* sane validation\n{deck}\n.control\nop\nprint {prints}\n.endc\n.end\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deck.cir")
        with open(path, "w") as f:
            f.write(text)
        out = subprocess.run([ng, "-b", path], capture_output=True, text=True).stdout
    volts = {}
    for m in re.finditer(r"v\((\w+)\)\s*=\s*([-+0-9.eE]+)", out):
        volts[m.group(1)] = float(m.group(2))
    return volts


DECKS = {
    "voltage_divider": ("V1 in 0 10\nR1 in mid 10k\nR2 mid 0 10k", ["in", "mid"]),
    "rc_dc": ("V1 in 0 5\nR1 in out 1k\nC1 out 0 1u", ["in", "out"]),
    "rl_dc": ("V1 in 0 5\nR1 in out 1k\nL1 out 0 1m", ["in", "out"]),
    "ladder": (
        "V1 1 0 12\nR1 1 2 1k\nR2 2 0 2k\nR3 2 3 1k\nR4 3 0 1k",
        ["1", "2", "3"],
    ),
    "vcvs_amp": (
        "V1 in 0 1\nR1 in n 10k\nR2 n out 20k\nE1 out 0 0 n 100000",
        ["in", "n", "out"],
    ),
    "cccs_mirror": (
        "V1 1 0 5\nR1 1 2 1k\nVsense 2 0 0\nF1 3 0 Vsense 2\nRL 3 0 500",
        ["1", "3"],
    ),
    "diode_rect": (
        "V1 in 0 0.72\nR1 in out 1k\nD1 out 0 dm\n.model dm D(Is=1e-14 N=1 Vt=0.025852)",
        ["in", "out"],
    ),
}


def main():
    ng = find_ngspice()
    print(f"ngspice: {ng}\n")
    worst = 0.0
    for name, (deck, nodes) in DECKS.items():
        oh = sane_dc(deck)
        sp = ngspice_dc(ng, deck, nodes)
        print(f"{name}:")
        for node in nodes:
            a = oh.get(node, float("nan"))
            b = sp.get(node, float("nan"))
            err = abs(a - b)
            if not np.isfinite(err):
                raise AssertionError(f"{name} v({node}): missing value (sane={a}, ngspice={b})")
            worst = max(worst, err)
            print(f"    v({node}): sane={a:+.6f}  ngspice={b:+.6f}  |diff|={err:.2e}")
    print(f"\nworst |diff| = {worst:.3e}")
    assert worst < 1e-3, f"validation exceeded tolerance: {worst}"
    print("OK")


if __name__ == "__main__":
    main()
