"""Performance benchmark: SANE's Rust DC solve (Newton + sparse LU) vs ngspice.

Scaling test on a resistor ladder (N nodes, sparse tridiagonal Jacobian). Times
SANE's `solve_dc` (per solve, extraction amortised) against ngspice's `.op`
(amortised over `repeat` to factor out process/parse startup).

Run after `maturin develop --release -m crates/py/Cargo.toml`:
    python crates/py/tests/bench_ngspice.py
"""

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


def ladder(n):
    """Linear resistor ladder: node 1 driven, rungs to ground. N nodes."""
    lines = ["V1 1 0 5"]
    for i in range(1, n):
        lines.append(f"R{i} {i} {i + 1} 1k")
        lines.append(f"Rg{i} {i} 0 10k")
    lines.append(f"Rg{n} {n} 0 10k")
    return "\n".join(lines)


def sane_solve_time(deck, repeats):
    c = sane._core.parse(deck + "\n.end")
    t0 = time.perf_counter()
    dae = c.extract_dae()
    extract = time.perf_counter() - t0
    p = [dae.values().get(name, 0.0) for name in dae.params()]
    # warm up + solve
    x = dae.solve_dc(p)
    t0 = time.perf_counter()
    for _ in range(repeats):
        x = dae.solve_dc(p)
    solve = (time.perf_counter() - t0) / repeats
    names = c.node_names()
    unk = dae.unknowns()
    v_last = x[unk.index(f"v{len(names) - 1}")]  # voltage at the last node
    return extract, solve, v_last


def ngspice_op_time(ng, deck, n, repeats):
    text = (
        f"* bench\n{deck}\n.control\nrepeat {repeats}\nop\nend\nprint v({n})\n.endc\n.end\n"
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deck.cir")
        with open(path, "w") as f:
            f.write(text)
        t0 = time.perf_counter()
        out = subprocess.run([ng, "-b", path], capture_output=True, text=True).stdout
        wall = time.perf_counter() - t0
    m = re.search(rf"v\({n}\)\s*=\s*([-+0-9.eE]+)", out)
    v_last = float(m.group(1)) if m else float("nan")
    return wall / repeats, v_last


def main():
    ng = find_ngspice()
    print(f"ngspice: {ng}\n")
    print(f"{'N':>6} {'sane extract':>15} {'sane solve':>14} {'ngspice op':>12} "
          f"{'speedup':>8}  {'v_last match':>12}")
    for n in [10, 50, 100, 300, 1000]:
        deck = ladder(n)
        reps = max(20, int(2_000_000 / (n * n)))
        ex, sv, v_oh = sane_solve_time(deck, reps)
        ng_op, v_sp = ngspice_op_time(ng, deck, n, min(reps, 200))
        speedup = ng_op / sv if sv > 0 else float("inf")
        match = abs(v_oh - v_sp)
        print(f"{n:>6} {ex * 1e3:>13.2f}ms {sv * 1e6:>12.1f}us {ng_op * 1e6:>10.1f}us "
              f"{speedup:>7.1f}x  {match:>12.2e}")


if __name__ == "__main__":
    main()
