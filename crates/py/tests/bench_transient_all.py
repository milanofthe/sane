"""Comprehensive transient benchmark over ALL real fixture netlists: run each
through SANE's BDF transient solve and ngspice `.tran`, comparing the full node
trajectory (max abs error over all nodes, interpolated to common time points)
and the wall-clock solve time.

Per circuit the time span is derived automatically: a few periods of the lowest
source frequency (SIN/PULSE/EXP), else a few of the largest R*C / L/R time
constant, else a short default (purely resistive -> flat DC). Oscillators are
flagged: their phase diverges from ngspice over many cycles, so a point-wise
trajectory match is not meaningful there.

Each fixture is measured in its OWN subprocess: the cases are independent, and
isolation keeps one circuit's run from perturbing another's (and is the correct
way to benchmark independent cases anyway).

    python crates/py/tests/bench_transient_all.py
"""

import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time


def find_ngspice():
    for name in ("ngspice_con", "ngspice"):
        p = shutil.which(name)
        if p:
            return p
        cand = os.path.join(sys.prefix, "Library", "bin", f"{name}.exe")
        if os.path.exists(cand):
            return cand
    raise RuntimeError("ngspice not found")


FIX = os.path.join(os.path.dirname(__file__), "..", "..", "netlist", "tests", "fixtures")
OSCILLATORS = {"colpitts_oscillator.cir"}


def derive_tstop(dae):
    """A sensible transient span from the circuit's own time scales."""
    vals = dae.values()
    freqs = []
    for k, v in vals.items():
        if k.endswith(".sin_w") and v > 0:
            freqs.append(v / (2 * math.pi))
        if k.endswith(".pulse_per") and v > 0:
            freqs.append(1.0 / v)
    if freqs:
        return 3.0 / min(freqs)
    rs = [v for k, v in vals.items() if re.match(r"(?i)^R[^.]*$", k) and v > 0]
    cs = [v for k, v in vals.items() if re.match(r"(?i)^C[^.]*$", k) and v > 0]
    ls = [v for k, v in vals.items() if re.match(r"(?i)^L[^.]*$", k) and v > 0]
    taus = []
    if rs and cs:
        taus.append(max(rs) * max(cs))
    if rs and ls:
        taus.append(max(ls) / min(rs))
    if taus:
        return 5.0 * max(taus)
    return 1e-3


def ng_tran(ng, deck, tstep, tstop, nodes):
    prints = "\n".join(f"print v({n})" for n in nodes if n != "0")
    text = f"* t\n{deck}\n.control\ntran {tstep} {tstop}\n{prints}\n.endc\n.end\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deck.cir")
        with open(path, "w") as f:
            f.write(text)
        t0 = time.perf_counter()
        out = subprocess.run([ng, "-b", path], capture_output=True, text=True).stdout
        wall = time.perf_counter() - t0
    series, cur = {}, None
    for ln in out.splitlines():
        mh = re.search(r"v\(([^)]+)\)", ln, re.IGNORECASE)
        if mh and "=" not in ln and "Index" in ln:
            cur = mh.group(1).lower()
            series[cur] = []
            continue
        m = re.match(r"\s*\d+\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$", ln)
        if m and cur is not None:
            series[cur].append((float(m.group(1)), float(m.group(2))))
    return wall, series


def run_one(path):
    """Worker: measure one fixture, print a single result line. Runs isolated."""
    import numpy as np
    import sane

    name = os.path.basename(path)
    ng = find_ngspice()
    deck = open(path).read()
    deck_ng = re.sub(r"(?im)^\s*\.end\s*$", "", deck)
    npts = 201
    try:
        c = sane._core.parse(deck)
        dae = c.extract_dae()
        names = c.node_names()
        unknowns = dae.unknowns()
        p = [dae.values().get(n, 0.0) for n in dae.params()]
        tstop = derive_tstop(dae)
        ts = list(np.linspace(0.0, tstop, npts))
        dae.solve_transient(p, ts)  # warm
        t0 = time.perf_counter()
        traj = dae.solve_transient(p, ts)
        mwall = time.perf_counter() - t0
    except Exception as e:
        print(f"{name}|FAIL|{str(e)[:40]}")
        return

    ngwall, series = ng_tran(ng, deck_ng, tstop / 1000.0, tstop, names)
    maxdv = 0.0
    for k, nm in enumerate(names):
        key, low = f"v{k}", nm.lower()
        if nm == "0" or key not in unknowns or low not in series or not series[low]:
            continue
        ngt = np.array([r[0] for r in series[low]])
        ngv = np.array([r[1] for r in series[low]])
        mv = np.array([row[unknowns.index(key)] for row in traj])
        maxdv = max(maxdv, float(np.max(np.abs(mv - np.interp(ts, ngt, ngv)))))

    if name in OSCILLATORS:
        status = "OSC"
    elif maxdv < 1e-3:
        status = "OK"
    elif maxdv < 1e-1:
        status = "WEAK"
    else:
        status = "MISMATCH"
    factor = ngwall / mwall if mwall > 0 else float("inf")
    print(f"{name}|{status}|{dae.dim()}|{tstop*1e3:.2f}|{mwall*1e3:.2f}|"
          f"{ngwall*1e3:.2f}|{factor:.1f}|{maxdv:.2e}")


def main():
    print(f"{'fixture':>24} {'dim':>4} {'tstop':>9} {'sane':>9} {'ngspice':>9} "
          f"{'factor':>7} {'max|dv|':>9}  status")
    for path in sorted(glob.glob(os.path.join(FIX, "*.cir"))):
        # Isolate each fixture in its own process for clean, independent results.
        r = subprocess.run([sys.executable, __file__, path], capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if "|" in l), None)
        if not line:
            print(f"{os.path.basename(path):>24}  (no result)")
            continue
        parts = line.split("|")
        name, status = parts[0], parts[1]
        if status == "FAIL":
            print(f"{name:>24} {'-':>4} {'-':>9} {'-':>9} {'-':>9} {'-':>7} {'-':>9}  FAIL {parts[2]}")
        else:
            _, _, dim, tstop, mw, ngw, fac, dv = parts
            print(f"{name:>24} {dim:>4} {float(tstop):>7.2f}ms {float(mw):>7.2f}ms "
                  f"{float(ngw):>7.2f}ms {float(fac):>6.1f}x {float(dv):>9.2e}  {status}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
    else:
        main()
