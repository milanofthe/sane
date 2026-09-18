"""Graph reduction across every benchmark circuit: reduce each on the graph and
check the reduced model preserves the DC operating point, reporting how much was
pruned and the cost. A robustness + value sweep of `dae.reduce` over all the
decks we have.

    python crates/py/tests/bench_reduction.py
"""

import glob
import os
import time

import numpy as np

import sane

HERE = os.path.dirname(__file__)
CIRCUIT_DIRS = [
    os.path.join(HERE, "..", "..", "netlist", "tests", "fixtures"),
    os.path.join(HERE, "bench_circuits"),
]


def n_terms(dae):
    return sum(str(r).count(",") + 1 if str(r).startswith("sum(") else 1
               for r in dae.residuals)


def main():
    decks = []
    for d in CIRCUIT_DIRS:
        decks += sorted(glob.glob(os.path.join(d, "*.cir")))

    print(f"{'circuit':>26} {'dim':>4} {'dim_r':>5} {'terms':>11} {'open':>5} {'short':>6} "
          f"{'max|dV|':>9} {'reduce':>7}")
    rel_tol = 1e-4
    freqs = np.geomspace(1.0, 1e9, 8)
    tot_full = tot_red = 0
    for path in decks:
        name = os.path.basename(path)
        deck = open(path).read()
        try:
            t0 = time.perf_counter()
            dae = sane.Circuit.parse(deck).extract()
            t_ext = (time.perf_counter() - t0) * 1e3
            op = dae.operating_point()

            t0 = time.perf_counter()
            red = dae.reduce(rel_tol=rel_tol, freqs=freqs)
            t_red = (time.perf_counter() - t0) * 1e3
            op_r = red.operating_point()
        except Exception as e:
            print(f"{name:>26} {'-':>4} {'-':>11} {'-':>7} {'-':>9}  {str(e)[:24]}")
            continue

        nf, nr = n_terms(dae), n_terms(red)
        tot_full += nf
        tot_red += nr
        n_open = sum(1 for _, op in red.transforms if op == "open")
        n_short = sum(1 for _, op in red.transforms if op == "short")
        # DC operating point must be preserved at every surviving node (a shorted
        # node merges into another, so it may no longer exist by name).
        maxdv = 0.0
        for nd in dae.node_names:
            if nd != "0":
                try:
                    maxdv = max(maxdv, abs(op[nd] - op_r[nd]))
                except (KeyError, IndexError):
                    pass
        print(f"{name:>26} {dae.dim:>4} {red.dim:>5} {nf:>5}->{nr:<4} {n_open:>5} {n_short:>6} "
              f"{maxdv:>9.2e} {t_red:>6.1f}")

    print(f"\ntotal terms across all circuits: {tot_full} -> {tot_red} "
          f"({100*(1-tot_red/tot_full):.0f}% pruned at rel_tol={rel_tol:.0e})")


if __name__ == "__main__":
    main()
