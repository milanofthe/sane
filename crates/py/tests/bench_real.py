"""Performance benchmark on REAL / nonlinear circuits: SANE's Rust DC solve
(damped Newton + sparse LU + tape eval) vs ngspice.

Two parts:
  1. A scaling nonlinear diode ladder (N stages -> N diodes, sparse, real Newton
     iterations), to see how we scale toward large circuits.
  2. The real fixture netlists (real topologies), SANE vs ngspice per `.op`.

Run after `maturin develop --release -m crates/py/Cargo.toml`:
    python crates/py/tests/bench_real.py
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import sane


def find_ngspice():
    for name in ("ngspice_con", "ngspice"):
        p = shutil.which(name)
        if p:
            return p
        for cand in (
            os.path.join(sys.prefix, "Library", "bin", f"{name}.exe"),
            os.path.join(sys.prefix, "bin", name),
        ):
            if os.path.exists(cand):
                return cand
    raise RuntimeError("ngspice not found")


NG = find_ngspice()


def diode_ladder(n):
    """N series R rungs, each node clamped to ground by a diode. Nonlinear."""
    lines = ["V1 1 0 2", ".model dm D(Is=1e-14 N=1 Vt=0.025852)"]
    for i in range(1, n + 1):
        lines.append(f"R{i} {i} {i + 1} 1k")
        lines.append(f"D{i} {i + 1} 0 dm")
    return "\n".join(lines)


def sane_solve(deck, repeats):
    c = sane._core.parse(deck + "\n.end")
    t0 = time.perf_counter()
    dae = c.extract_dae()
    extract = time.perf_counter() - t0
    p = [dae.values().get(name, 0.0) for name in dae.params()]
    x = dae.solve_dc(p)
    t0 = time.perf_counter()
    for _ in range(repeats):
        x = dae.solve_dc(p)
    solve = (time.perf_counter() - t0) / repeats
    return dae, x, extract, solve


def ngspice_op(deck, node, repeats):
    text = f"* bench\n{deck}\n.control\nrepeat {repeats}\nop\nend\nprint v({node})\n.endc\n.end\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deck.cir")
        with open(path, "w") as f:
            f.write(text)
        t0 = time.perf_counter()
        out = subprocess.run([NG, "-b", path], capture_output=True, text=True).stdout
        wall = time.perf_counter() - t0
    m = re.search(rf"v\({node}\)\s*=\s*([-+0-9.eE]+)", out, re.IGNORECASE)
    return wall / repeats, (float(m.group(1)) if m else float("nan"))


def bench_scaling():
    print("== nonlinear diode ladder ==")
    print(f"{'N':>6} {'dim':>6} {'extract':>10} {'sane':>10} {'ngspice op':>11} {'speedup':>8} {'match':>10}")
    for n in [10, 50, 100, 300, 1000, 3000]:
        deck = diode_ladder(n)
        reps = max(5, int(200_000 / n))
        dae, x, ex, sv = sane_solve(deck, reps)
        # node n+1 is the last ladder node; its unknown index via node_names
        c = sane._core.parse(deck + "\n.end")
        names = c.node_names()
        last = f"v{names.index(str(n + 1))}"
        v_m = x[dae.unknowns().index(last)]
        ng, v_sp = ngspice_op(deck, n + 1, min(reps, 50))
        print(f"{n:>6} {dae.dim():>6} {ex * 1e3:>8.1f}ms {sv * 1e6:>8.1f}us {ng * 1e6:>9.1f}us "
              f"{ng / sv:>7.1f}x {abs(v_m - v_sp):>10.2e}")


def bench_fixtures():
    print("\n== real fixtures ==")
    fix_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "netlist", "tests", "fixtures"
    )
    print(f"{'fixture':>22} {'dim':>5} {'sane':>10} {'ngspice':>10} {'status':>10}")
    for path in sorted(glob.glob(os.path.join(fix_dir, "*.cir"))):
        name = os.path.basename(path)
        deck = open(path).read()
        try:
            dae, x, _ex, sv = sane_solve(deck, 50)
        except Exception as e:
            print(f"{name:>22} {'-':>5} {'-':>10} {'-':>10} {'no-conv':>10}")
            continue
        # pick first non-ground node to compare
        c = sane._core.parse(deck)
        names = c.node_names()
        node = names[1] if len(names) > 1 else None
        key = f"v{1}"
        v_m = x[dae.unknowns().index(key)] if key in dae.unknowns() else float("nan")
        # Strip only the standalone `.end` line (not `.ends` of a subckt).
        deck_ng = re.sub(r"(?im)^\s*\.end\s*$", "", deck)
        ng, v_sp = ngspice_op(deck_ng, node, 30) if node else (float("nan"), float("nan"))
        status = "ok" if abs(v_m - v_sp) < 1e-3 else f"d={abs(v_m - v_sp):.1e}"
        print(f"{name:>22} {dae.dim():>5} {sv * 1e6:>8.1f}us {ng * 1e6:>8.1f}us {status:>10}")


if __name__ == "__main__":
    print(f"ngspice: {NG}\n")
    bench_scaling()
    bench_fixtures()
