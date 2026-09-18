"""Extract-pipeline scaling study over synthetic circuits.

Sweeps each generator family (analog_array, rc_mesh, inverter_chain) over growing
size, timing the whole pipeline -- parse (Python/PyO3) plus the native per-stage
extract profile (assemble, sparse Jacobians, tape compilation, symbolic LU) -- and
fits the empirical complexity exponent (slope of log time vs log element count)
for the total and the dominant stages. This surfaces any hidden super-linearity
on the road to very large netlists.

    python crates/py/tests/bench_extract_scaling.py
    python crates/py/tests/bench_extract_scaling.py --html report.html

CPU-friendly: each family stops once a single extract exceeds the time budget.
"""

import argparse
import math
import sys
import time
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402

import sane  # noqa: E402

TIME_BUDGET_S = 4.0          # stop a family once one extract exceeds this
REPEATS = 2                  # take the min over this many runs (timer noise)


def n_elements(deck):
    return sum(1 for l in deck.splitlines() if l.strip() and l.strip()[0] not in "*.+")


def measure(deck):
    """Return (parse_ms, extract_total_ms, profile_dict) as the min over repeats."""
    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        ckt = sane.Circuit.parse(deck)
        t1 = time.perf_counter()
        dae = ckt.extract()
        prof = dict(dae.profile)
        parse_ms = (t1 - t0) * 1e3
        extract_ms = sum(prof.values())
        if best is None or extract_ms < best[1]:
            best = (parse_ms, extract_ms, prof, dae.dim)
    return best


def sizes_for(family):
    if family == "rc_mesh":
        return [(s, s * s) for s in (4, 6, 8, 12, 16, 24, 32, 48, 64)]
    return [(s, s) for s in (16, 32, 64, 128, 256, 512, 1024, 2048)]


def fit_exponent(xs, ys):
    """Slope of log(y) vs log(x): the empirical complexity exponent."""
    lx = np.log(np.array(xs, float))
    ly = np.log(np.array(ys, float))
    if len(lx) < 2:
        return float("nan")
    return float(np.polyfit(lx, ly, 1)[0])


def run_family(name, gen):
    rows = []
    for knob, _ in sizes_for(name):
        deck, ncells = gen(knob)
        nel = n_elements(deck)
        parse_ms, extract_ms, prof, dim = measure(deck)
        rows.append((nel, dim, parse_ms, extract_ms, prof))
        if extract_ms / 1e3 > TIME_BUDGET_S:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=None, help="write a Plotly HTML report")
    args = ap.parse_args()

    all_results = {}
    for name, gen in synth.GENERATORS.items():
        print(f"\n=== {name} ===")
        print(f"{'elements':>9} {'dim':>6} {'parse':>9} {'extract':>9} {'top stage':>22}")
        rows = run_family(name, gen)
        for nel, dim, parse_ms, extract_ms, prof in rows:
            top = max(prof.items(), key=lambda kv: kv[1])
            print(f"{nel:>9} {dim:>6} {parse_ms:>8.2f}m {extract_ms:>8.2f}m "
                  f"{top[0]:>14} {top[1]:>6.1f}m")
        all_results[name] = rows

        # Empirical exponents.
        nels = [r[0] for r in rows]
        ext = [r[3] for r in rows]
        par = [r[2] for r in rows]
        print(f"  exponent  parse ~ O(n^{fit_exponent(nels, par):.2f})   "
              f"extract ~ O(n^{fit_exponent(nels, ext):.2f})")
        # Per-stage exponents for the stages that matter at the largest size.
        big = rows[-1][4]
        stage_keys = sorted(big, key=lambda k: big[k], reverse=True)[:4]
        for sk in stage_keys:
            ys = [r[4].get(sk, 0.0) or 1e-9 for r in rows]
            print(f"      {sk:<22} ~ O(n^{fit_exponent(nels, ys):.2f})  "
                  f"(largest {big[sk]:.1f} ms)")

    if args.html:
        write_html(all_results, args.html)
        print(f"\nwrote {args.html}")


def write_html(results, path):
    import json
    traces = []
    for name, rows in results.items():
        xs = [r[0] for r in rows]
        traces.append({"x": xs, "y": [r[3] for r in rows], "name": f"{name} extract",
                       "mode": "lines+markers"})
        traces.append({"x": xs, "y": [r[2] for r in rows], "name": f"{name} parse",
                       "mode": "lines+markers", "line": {"dash": "dot"}})
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<title>SANE extract scaling</title></head><body>
<div id="p" style="width:100%;height:90vh"></div><script>
Plotly.newPlot('p', {json.dumps(traces)}, {{
  title:'SANE extract pipeline scaling',
  xaxis:{{title:'elements', type:'log'}}, yaxis:{{title:'time (ms)', type:'log'}}
}});</script></body></html>"""
    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
