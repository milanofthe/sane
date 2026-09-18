"""Standing SANE-vs-ngspice benchmark over every available circuit.

Discovers all `.cir` decks (the shipped fixtures plus any imported into
`bench_circuits/`), solves the DC operating point in both SANE and ngspice, and
reports per-node accuracy and timing. The honest, always-on cross-check of
correctness and speed on real decks.

    python crates/py/tests/bench_vs_ngspice.py

Note: ngspice times are end-to-end (they include ~tens of ms of process
startup), so for small decks they overstate the solver cost.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
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
    return None


NG = find_ngspice()
HERE = os.path.dirname(__file__)
CIRCUIT_DIRS = [
    os.path.join(HERE, "..", "..", "netlist", "tests", "fixtures"),
    os.path.join(HERE, "bench_circuits"),
]


def ngspice_dc(deck, node_names):
    """ngspice DC node voltages, and the wall time of the run."""
    prints = "\n".join(f"print v({nm})" for nm in node_names if nm != "0")
    text = f"* dc\n{deck}\n.control\nop\n{prints}\n.endc\n.end\n"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deck.cir")
        with open(path, "w") as f:
            f.write(text)
        t0 = time.perf_counter()
        out = subprocess.run([NG, "-b", path], capture_output=True, text=True).stdout
        dt = time.perf_counter() - t0
    vals = {}
    for nm in node_names:
        if nm == "0":
            continue
        m = re.search(rf"v\({re.escape(nm)}\)\s*=\s*([-+0-9.eE]+)", out, re.IGNORECASE)
        if m:
            vals[nm] = float(m.group(1))
    return vals, dt


def sane_dc(deck):
    """SANE DC: returns (operating_point, node_names, extract_ms, solve_ms)."""
    t0 = time.perf_counter()
    ckt = sane.Circuit.parse(deck)
    dae = ckt.extract()
    t_ex = time.perf_counter() - t0
    t0 = time.perf_counter()
    op = dae.operating_point()
    t_so = time.perf_counter() - t0
    return op, [n for n in ckt.node_names if n != "0"], t_ex * 1e3, t_so * 1e3


def main():
    if NG is None:
        print("ngspice not found; install it for the cross-check.")
        return
    print(f"ngspice: {NG}\n")
    print(f"{'circuit':>26} {'dim':>4} {'cmp':>4} {'max|dv|':>10} "
          f"{'M_ext':>7} {'M_dc':>7} {'ng':>7} {'status':>9}")

    decks = []
    for d in CIRCUIT_DIRS:
        decks += sorted(glob.glob(os.path.join(d, "*.cir")))

    n_ok = n_weak = n_fail = 0
    for path in decks:
        name = os.path.basename(path)
        deck = open(path).read()
        deck_ng = re.sub(r"(?im)^\s*\.end\s*$", "", deck)
        try:
            op, nodes, t_ex, t_so = sane_dc(deck)
        except Exception as e:
            print(f"{name:>26} {'-':>4} {'-':>4} {'-':>10} {'-':>7} {'-':>7} {'-':>7} "
                  f"{'NO-CONV':>9}  {str(e)[:24]}")
            n_fail += 1
            continue
        vsp, t_ng = ngspice_dc(deck_ng, nodes)
        maxdv, cmp = 0.0, 0
        for nm in nodes:
            if nm in vsp:
                try:
                    maxdv = max(maxdv, abs(op[nm] - vsp[nm]))
                    cmp += 1
                except (KeyError, IndexError):
                    pass
        if cmp == 0:
            # SANE solved it but ngspice produced no comparable nodes -- it
            # rejected the deck (e.g. PSpice-only model params it does not know).
            status = "NO-CMP"
        elif maxdv < 1e-3:
            status = "OK"
        elif maxdv < 1e-1:
            status = "WEAK"
        else:
            status = "MISMATCH"
        n_ok += status == "OK"
        n_weak += status == "WEAK"
        n_fail += status == "MISMATCH"
        print(f"{name:>26} {op.vector.size:>4} {cmp:>4} {maxdv:>10.2e} "
              f"{t_ex:>6.1f} {t_so:>6.2f} {t_ng*1e3:>6.0f} {status:>9}")

    n_nocmp = len(decks) - n_ok - n_weak - n_fail
    print(f"\n{len(decks)} circuits: {n_ok} OK, {n_weak} WEAK, {n_fail} FAIL/MISMATCH, "
          f"{n_nocmp} NO-CMP (ngspice rejected / no convergence)")


if __name__ == "__main__":
    main()
