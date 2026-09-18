"""Honest large-transistor benchmark: a cascade of N nonlinear NMOS common-
source stages (R load from VDD, gate driven by the previous stage's drain).
Every stage is a real transistor in saturation/triode, so this exercises gmin
homotopy and Newton convergence on a genuinely nonlinear network -- unlike the
diode ladder (banded, trivial) or the textbook fixtures (3-8 elements).

We compare the *full node vector* against ngspice (max abs error over all
nodes), not a single supply node. Run after
    maturin develop --release -m crates/py/Cargo.toml
    python crates/py/tests/bench_transistor.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import sane


def find_ngspice():
    for name in ("ngspice_con", "ngspice"):
        p = shutil.which(name)
        if p:
            return p
        cand = os.path.join(sys.prefix, "Library", "bin", f"{name}.exe")
        if os.path.exists(cand):
            return cand
    raise RuntimeError("ngspice not found")


NG = find_ngspice()

MODEL = ".model nm NMOS(Kp=120u W=4 L=1 Vth=0.7 lambda=0.02)"


def cs_chain(n):
    """N NMOS common-source stages. Node 1 = VDD, node 2 = input bias.
    Stage k: load Rk (VDD->drain dk), Mk (drain dk, gate = prev drain, source 0).
    Drains are nodes 3..3+n-1."""
    lines = ["VDD 1 0 DC 5", "VIN 2 0 DC 1.2", MODEL]
    for k in range(n):
        d = 3 + k
        g = 2 if k == 0 else 3 + (k - 1)
        lines.append(f"R{k} 1 {d} 10k")
        lines.append(f"M{k} {d} {g} 0 0 nm")
    return "\n".join(lines), n  # n transistors


def sane_solve(deck, repeats):
    c = sane._core.parse(deck + "\n.end")
    t0 = time.perf_counter()
    dae = c.extract_dae()
    extract = time.perf_counter() - t0
    p = [dae.values().get(name, 0.0) for name in dae.params()]
    x = dae.solve_dc(p)  # raises if not converged
    t0 = time.perf_counter()
    for _ in range(repeats):
        x = dae.solve_dc(p)
    solve = (time.perf_counter() - t0) / repeats
    return c, dae, x, extract, solve


def ngspice_all_nodes(deck, node_names, repeats):
    prints = "\n".join(f"print v({nm})" for nm in node_names if nm != "0")
    text = (
        f"* bench\n{deck}\n.control\nrepeat {repeats}\nop\nend\n{prints}\n.endc\n.end\n"
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deck.cir")
        with open(path, "w") as f:
            f.write(text)
        t0 = time.perf_counter()
        out = subprocess.run([NG, "-b", path], capture_output=True, text=True).stdout
        wall = time.perf_counter() - t0
    vals = {}
    for nm in node_names:
        if nm == "0":
            continue
        m = re.search(rf"v\({re.escape(nm)}\)\s*=\s*([-+0-9.eE]+)", out, re.IGNORECASE)
        if m:
            vals[nm] = float(m.group(1))
    return wall / repeats, vals


def main():
    print(f"ngspice: {NG}\n")
    print("== NMOS common-source cascade (real nonlinear transistors) ==")
    print(f"{'#FET':>5} {'dim':>6} {'extract':>10} {'sane':>11} "
          f"{'ngspice op':>12} {'speedup':>8} {'max|dv|':>10}")
    for n in [1, 5, 20, 50, 100, 200, 500]:
        deck, n_fet = cs_chain(n)
        reps = max(3, int(50_000 / max(n, 1)))
        try:
            c, dae, x, ex, sv = sane_solve(deck, reps)
        except Exception as e:
            print(f"{n_fet:>5} {'-':>6} {'-':>10} {'-':>11} {'-':>12} "
                  f"{'NO-CONV':>8} {str(e)[:20]:>10}")
            continue
        names = c.node_names()
        ng, vsp = ngspice_all_nodes(deck, names, min(reps, 20))
        # compare all node voltages v{k} = node names[k]
        maxdv = 0.0
        unknowns = dae.unknowns()
        for k, nm in enumerate(names):
            if nm == "0" or nm not in vsp:
                continue
            key = f"v{k}"
            if key in unknowns:
                vo = x[unknowns.index(key)]
                maxdv = max(maxdv, abs(vo - vsp[nm]))
        print(f"{n_fet:>5} {dae.dim():>6} {ex * 1e3:>8.1f}ms {sv * 1e6:>9.1f}us "
              f"{ng * 1e6:>10.1f}us {ng / sv:>7.1f}x {maxdv:>10.2e}")


if __name__ == "__main__":
    main()
