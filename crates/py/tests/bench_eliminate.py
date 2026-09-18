"""Exact node elimination across every benchmark circuit: eliminate all internal
resistive nodes on the graph and check the reduced model preserves the DC
operating point and the small-signal AC response at every surviving node (the
elimination is exact, so the error must be at machine precision).

    python crates/py/tests/bench_eliminate.py
"""

import glob
import os

import numpy as np

import sane

HERE = os.path.dirname(__file__)
CIRCUIT_DIRS = [
    os.path.join(HERE, "..", "..", "netlist", "tests", "fixtures"),
    os.path.join(HERE, "bench_circuits"),
]


def surviving_nodes(red):
    out = []
    for n in red.node_names:
        if n == "0":
            continue
        try:
            red.unknown_index(n)
            out.append(n)
        except Exception:
            pass
    return out


def main():
    decks = []
    for d in CIRCUIT_DIRS:
        decks += sorted(glob.glob(os.path.join(d, "*.cir")))

    freqs = np.geomspace(1e1, 1e8, 12)
    print(f"{'circuit':>24} {'dim':>4} {'elim':>5} {'dDC':>9} {'dAC_dB':>9}")
    worst_dc = worst_ac = 0.0
    for path in decks:
        name = os.path.basename(path)
        deck = open(path).read()
        try:
            dae = sane.Circuit.parse(deck).extract()
            op = dae.operating_point()
            red = dae.eliminate()          # keep nothing
            op2 = red.operating_point()

            dd = 0.0
            for n in surviving_nodes(red):
                try:
                    dd = max(dd, abs(op[n] - op2[n]))
                except Exception:
                    pass

            da = float("nan")
            srcs = [p for p in dae.params if p.startswith("V")]
            surv = surviving_nodes(red)
            if srcs and surv:
                tgt = surv[-1]
                try:
                    Hf = dae.small_signal(srcs[0], tgt).response(freqs)
                    Hr = red.small_signal(srcs[0], tgt).response(freqs)
                    da = np.max(np.abs(20 * np.log10(np.abs(Hr) + 1e-300)
                                       - 20 * np.log10(np.abs(Hf) + 1e-300)))
                except Exception:
                    da = float("nan")

            print(f"{name:>24} {dae.dim:>4} {len(red.eliminated):>5} {dd:>9.1e} {da:>9.1e}")
            worst_dc = max(worst_dc, dd)
            if da == da:
                worst_ac = max(worst_ac, da)
        except Exception as e:
            print(f"{name:>24}  ERR {e}")

    print(f"\nworst dDC = {worst_dc:.2e}   worst dAC = {worst_ac:.2e} dB")


if __name__ == "__main__":
    main()
