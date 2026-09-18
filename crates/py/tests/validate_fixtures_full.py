"""Deep validation of the real fixture netlists: solve the DC operating point in
SANE and compare EVERY node voltage against ngspice (max abs error over all
nodes), not just one node. This is the honest cross-check of correctness on the
real decks we ship.

    python crates/py/tests/validate_fixtures_full.py
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

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
FIX = os.path.join(os.path.dirname(__file__), "..", "..", "netlist", "tests", "fixtures")


def ngspice_nodes(deck, node_names):
    prints = "\n".join(f"print v({nm})" for nm in node_names if nm != "0")
    text = f"* v\n{deck}\n.control\nop\n{prints}\n.endc\n.end\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deck.cir")
        with open(path, "w") as f:
            f.write(text)
        out = subprocess.run([NG, "-b", path], capture_output=True, text=True).stdout
    vals = {}
    for nm in node_names:
        if nm == "0":
            continue
        m = re.search(rf"v\({re.escape(nm)}\)\s*=\s*([-+0-9.eE]+)", out, re.IGNORECASE)
        if m:
            vals[nm] = float(m.group(1))
    return vals


def main():
    print(f"ngspice: {NG}\n")
    print(f"{'fixture':>24} {'dim':>4} {'#nodes':>6} {'cmp':>4} {'max|dv|':>10}  status")
    for path in sorted(glob.glob(os.path.join(FIX, "*.cir"))):
        name = os.path.basename(path)
        deck = open(path).read()
        deck_ng = re.sub(r"(?im)^\s*\.end\s*$", "", deck)
        try:
            c = sane._core.parse(deck)
            dae = c.extract_dae()
            p = [dae.values().get(n, 0.0) for n in dae.params()]
            x = dae.solve_dc(p)
        except Exception as e:
            print(f"{name:>24} {'-':>4} {'-':>6} {'-':>4} {'-':>10}  NO-CONV {str(e)[:30]}")
            continue
        names = c.node_names()
        unknowns = dae.unknowns()
        vsp = ngspice_nodes(deck_ng, names)
        maxdv, cmp = 0.0, 0
        for k, nm in enumerate(names):
            if nm == "0" or nm not in vsp:
                continue
            key = f"v{k}"
            if key in unknowns:
                vo = x[unknowns.index(key)]
                maxdv = max(maxdv, abs(vo - vsp[nm]))
                cmp += 1
        status = "OK" if maxdv < 1e-3 else ("WEAK" if maxdv < 1e-1 else "MISMATCH")
        print(f"{name:>24} {dae.dim():>4} {len(names) - 1:>6} {cmp:>4} {maxdv:>10.2e}  {status}")


if __name__ == "__main__":
    main()
