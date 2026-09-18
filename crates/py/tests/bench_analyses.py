"""Per-analysis performance across every real benchmark circuit.

For each fixture, time every analysis SANE can do -- parse, extract, DC operating
point, transient, small-signal (poles / AC / zeros), symbolic AC transfer, exact
first- and second-order sensitivity (adjoint / Hessian), pole & zero sensitivity,
and graph reduction / node elimination -- so the cost of the whole capability set
is visible on realistic circuits, not just synthetic scaling decks.

    python crates/py/tests/bench_analyses.py
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
FREQS = np.geomspace(1.0, 1e9, 50)
TVEC = np.linspace(0, 1e-6, 50)


def timed(fn, repeats=1, warmup=False):
    """Min wall-clock over `repeats`, in ms; None if the analysis raises. With
    `warmup`, one untimed call first -- so analyses that lazily build a reusable
    tape (sensitivity / Hessian) report their amortized per-call (sweep) cost,
    not the one-time build."""
    if warmup:
        try:
            fn()
        except Exception:
            return None
    best = None
    for _ in range(repeats):
        try:
            t0 = time.perf_counter()
            fn()
            dt = (time.perf_counter() - t0) * 1e3
        except Exception:
            return None
        best = dt if best is None else min(best, dt)
    return best


def first_source(dae):
    for p in dae.params:
        if p[0] in "Vv":
            return p
    for p in dae.params:
        if p[0] in "Ii":
            return p
    return None


def probe_node(dae):
    for n in reversed(dae.node_names):
        if n != "0":
            return n
    return None


def main():
    decks = []
    for d in CIRCUIT_DIRS:
        decks += sorted(glob.glob(os.path.join(d, "*.cir")))

    cols = ["parse", "extract", "op", "tran", "poles", "ac", "ac_sym",
            "sens", "hess", "psens", "zeros", "reduce", "elim"]
    hdr = f"{'circuit':>22} {'dim':>4} " + " ".join(f"{c:>7}" for c in cols)
    print(hdr)
    print("-" * len(hdr))

    totals = {c: 0.0 for c in cols}
    counts = {c: 0 for c in cols}

    for path in decks:
        name = os.path.basename(path)[:22]
        deck = open(path).read()
        row = {}

        ckt = None
        def do_parse():
            nonlocal ckt
            ckt = sane.Circuit.parse(deck)
        row["parse"] = timed(do_parse, repeats=3)
        if ckt is None:
            print(f"{name:>22}  parse failed")
            continue

        dae = None
        def do_extract():
            nonlocal dae
            dae = ckt.extract()
        row["extract"] = timed(do_extract, repeats=2)
        if dae is None:
            continue

        src = first_source(dae)
        out = probe_node(dae)
        params = dae.params[:3]

        row["op"] = timed(lambda: dae.operating_point(), repeats=2)
        row["tran"] = timed(lambda: dae.transient(TVEC))
        if src and out:
            ss = [None]
            def do_ss():
                ss[0] = dae.small_signal(src, out)
            row["poles"] = timed(lambda: dae.small_signal(src, out).poles(), repeats=2)
            row["ac"] = timed(lambda: dae.small_signal(src, out).response(FREQS), repeats=2)
            # Symbolic AC (Cramer/determinant) blows up quasi-exponentially in
            # the symbolic expression size, so only attempt it on small systems.
            if dae.dim <= 12:
                row["ac_sym"] = timed(lambda: dae.ac_transfer(src, out, FREQS))
            # warmup: these lazily build a reusable tape on first call; report
            # the amortized per-call (parameter-sweep) cost.
            row["sens"] = timed(lambda: dae.sensitivity(out), repeats=2, warmup=True)
            row["hess"] = timed(lambda: dae.hessian(out, params), repeats=2, warmup=True)
            row["psens"] = timed(lambda: dae.pole_sensitivity(src, params[0]), repeats=2)
            row["zeros"] = timed(lambda: dae.small_signal(src, out).zeros(), repeats=2)
        row["reduce"] = timed(lambda: dae.reduce(rel_tol=1e-3))
        row["elim"] = timed(lambda: dae.eliminate())

        cells = []
        for c in cols:
            v = row.get(c)
            if v is None:
                cells.append(f"{'-':>7}")
            else:
                cells.append(f"{v:>7.1f}")
                totals[c] += v
                counts[c] += 1
        print(f"{name:>22} {dae.dim:>4} " + " ".join(cells))

    print("-" * len(hdr))
    avg = []
    for c in cols:
        avg.append(f"{totals[c]/counts[c]:>7.1f}" if counts[c] else f"{'-':>7}")
    print(f"{'mean (ms)':>22} {'':>4} " + " ".join(avg))
    print("\nlegend: op=DC operating point, tran=transient(50 steps), "
          "poles/ac/zeros=small-signal, ac_sym=symbolic AC (Cramer),")
    print("        sens=adjoint dy/dp, hess=2nd-order (3 params), "
          "psens=pole sensitivity, reduce/elim=graph reduction. times in ms.")


if __name__ == "__main__":
    main()
