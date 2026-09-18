"""Cross-check SANE harmonic balance against Xyce (.HB), the open-source
reference that actually has harmonic balance (ngspice does not).

Xyce is not on conda/winget; install the Sandia Windows build from
https://xyce.sandia.gov/downloads/ (free, registration required), then either
put `Xyce.exe` on PATH or set the XYCE env var to its full path. This script
skips gracefully if Xyce is not found.

For each circuit it runs SANE's solve_hb and Xyce's .HB on the same netlist and
compares the harmonic magnitudes at one node, plus wall-clock.

    python crates/py/tests/bench_hb_xyce.py
"""

import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

import sane


def find_xyce():
    env = os.environ.get("XYCE")
    if env and os.path.exists(env):
        return env
    for name in ("Xyce", "xyce", "Xyce.exe"):
        p = shutil.which(name)
        if p:
            return p
    import glob
    for pat in (r"C:\Program Files\Xyce*\bin\Xyce.exe",
                r"C:\Program Files (x86)\Xyce*\bin\Xyce.exe"):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


# (name, deck-body without .end, f0, output node). Diode circuits where both
# engines converge cleanly -- the apples-to-apples comparison.
CIRCUITS = [
    ("biased_diode_rc",
     "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 DMOD\n"
     "C1 mid 0 100n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)", 1000.0, "mid"),
    ("diode_mixer_bias",
     "V1 in 0 SIN(0.5 0.25 2000)\nR1 in mid 2k\nD1 mid 0 DMOD\n"
     "C1 mid 0 47n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)", 2000.0, "mid"),
]
HARMONICS = 8


def sane_hb(body, f0, node):
    c = sane._core.parse(body + "\n.end")
    dae = c.extract_dae()
    p = [dae.values().get(n, 0.0) for n in dae.params()]
    x_dc = dae.solve_dc(p)
    t0 = time.perf_counter()
    spectra, conv, iters, rnorm, _, _ = dae.solve_hb(p, f0, harmonics=HARMONICS, x0=x_dc)
    dt = time.perf_counter() - t0
    names = c.node_names()
    oi = dae.unknowns().index(f"v{names.index(node)}")
    mags = {k: abs(complex(*spectra[oi][k])) for k in range(HARMONICS + 1)}
    return mags, conv, iters, dt


def xyce_hb(xyce, body, f0, node):
    """Run Xyce .HB and parse the .HB.FD.prn (index, freq, Re, Im per variable)."""
    deck = (f"* xyce hb\n{body}\n.options hbint numfreq={HARMONICS}\n.hb {f0}\n"
            f".print hb vr({node}) vi({node})\n.end\n")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deck.cir")
        open(path, "w").write(deck)
        t0 = time.perf_counter()
        r = subprocess.run([xyce, path], capture_output=True, text=True, cwd=d)
        wall = time.perf_counter() - t0
        # Xyce's own internal timing, for a fair solve-vs-solve comparison (the
        # subprocess wall-clock is dominated by process start + file I/O).
        msolve = re.search(r"Solvers Run Time:\s*([\d.eE+-]+)", r.stdout)
        melapsed = re.search(r"Total Elapsed Run Time:\s*([\d.eE+-]+)", r.stdout)
        solve = float(msolve.group(1)) if msolve else float("nan")
        elapsed = float(melapsed.group(1)) if melapsed else float("nan")
        prn = next((os.path.join(d, f) for f in os.listdir(d)
                    if f.endswith(".FD.prn") or f.endswith(".HB.FD.prn")), None)
        if prn is None:
            raise RuntimeError(f"no FD.prn (stderr: {r.stderr[:200]})")
        mags = {}
        for line in open(prn):
            t = line.split()
            if len(t) < 4 or not re.match(r"^-?\d", t[0]):
                continue
            freq, re_v, im_v = float(t[1]), float(t[-2]), float(t[-1])
            k = round(freq / f0)
            if k >= 0:  # positive-frequency coefficients (same convention as SANE)
                mags[k] = abs(complex(re_v, im_v))
        return mags, wall, solve, elapsed


def diode_ladder(n, bias=0.7, amp=0.1, f0=1000.0):
    L = [f"V1 1 0 SIN({bias} {amp} {f0})", ".model dm D(Is=1e-14 N=1 Vt=0.025852)"]
    for i in range(1, n + 1):
        L += [f"R{i} {i} {i+1} 1k", f"D{i} {i+1} 0 dm", f"C{i} {i+1} 0 10n"]
    return "\n".join(L)


def compare_one(name, body, f0, node, xyce):
    s_mag, conv, iters, t_s = sane_hb(body, f0, node)
    try:
        x_mag, wall, solve, elapsed = xyce_hb(xyce, body, f0, node)
    except Exception as e:
        print(f"{name:<20}  Xyce failed: {str(e)[:60]}")
        return
    cols = []
    for k in range(1, 4):
        a, b = s_mag.get(k, 0.0), x_mag.get(k, float("nan"))
        rel = abs(a - b) / b if b else float("nan")
        cols.append(f"H{k}:{rel:5.1%}")
    sp = solve / t_s if t_s else float("nan")
    print(f"{name:<20} {node:>5} {t_s*1e3:>7.1f}m {solve*1e3:>7.1f}m {sp:>6.1f}x  "
          + "  ".join(cols))


def main():
    xyce = find_xyce()
    if xyce is None:
        print("Xyce not found. Install from https://xyce.sandia.gov/downloads/ ")
        print("then put Xyce.exe on PATH or set XYCE=<full path>. Skipping.")
        return
    print(f"Xyce: {xyce}\n")
    print("t_SANE = solve_hb (in-process); t_Xyce = Xyce's internal solver time")
    print("(process start + file I/O excluded). Harmonic columns are SANE-vs-Xyce rel error.\n")
    print(f"{'circuit':<20} {'node':>5} {'t_SANE':>8} {'t_Xyce':>8} {'speedup':>6}  rel error")
    for name, body, f0, node in CIRCUITS:
        compare_one(name, body, f0, node, xyce)
    for n in (16, 64, 256, 512):
        compare_one(f"diode_ladder_{n}", diode_ladder(n), 1000.0, "2", xyce)


if __name__ == "__main__":
    main()
