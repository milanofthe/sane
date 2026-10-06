"""Operating-point-guided graph reduction: shrink the extracted system itself.

Each KCL residual is a sum of branch currents; at the bias point each branch has
a conductance and a capacitance magnitude. `dae.reduce(rel_tol)` drops the
negligible branches directly on the graph -- consistently across the nodes they
couple, no netlist surgery, no re-extraction. The reduced DAE keeps the same
unknowns (same interface) but has fewer terms, so its Jacobian, tapes, and every
downstream analysis are smaller and faster -- a parametric reduced-order model.

    python crates/py/tests/graph_reduction_demo.py
"""

import time

import numpy as np

import sane


def build_parasitic_rc(stages, seed=0):
    """An RC line (the 'real' circuit) buried in random small parasitics: leak
    resistors to ground and tiny strays between nodes."""
    rng = np.random.default_rng(seed)
    lines = ["V1 1 0 AC 1"]
    for k in range(1, stages + 1):
        lines.append(f"R{k} {k} {k+1} 1k")
        lines.append(f"C{k} {k+1} 0 10n")
    # parasitics: high-value leaks and femtofarad strays
    pid = 0
    for k in range(1, stages + 2):
        lines.append(f"Rleak{pid} {k} 0 {rng.uniform(20, 200):.0f}meg"); pid += 1
        if k + 2 <= stages + 1:
            lines.append(f"Cx{pid} {k} {k+2} {rng.uniform(0.1, 5):.2f}f"); pid += 1
    lines.append(f"Rload {stages+1} 0 10k")
    return "\n".join(lines), stages + 1


def main():
    deck, out = build_parasitic_rc(stages=8)
    dae = sane.Circuit.parse(deck).extract()

    print("== operating-point-guided graph reduction ==")
    freqs = np.geomspace(1.0, 1e6, 60)
    Hf = dae.small_signal("V1", str(out)).response(freqs)

    def teval(d, p, x, reps=3000):
        """Residual + Jacobian evaluation time (the per-Newton-step cost)."""
        d.core.jacobian_i_x_sparse(x, p, 0.0)
        t0 = time.perf_counter()
        for _ in range(reps):
            d.core.jacobian_i_x_sparse(x, p, 0.0)
        return (time.perf_counter() - t0) / reps

    p = [dae.values.get(n, 0.0) for n in dae.params]
    x = dae.core.solve_dc(p, None, 1e-10, 100)
    t_full = teval(dae, p, x)

    print(f"{'rel_tol':>9} {'removed':>7} {'max|dGain|':>12} {'eval speedup':>13}")
    print(f"{'full':>9} {0:>7} {'-':>12} {'1.00x':>13}")
    for rel_tol in (1e-4, 1e-3, 1e-2):
        # judge negligibility over the band we actually care about
        red = dae.reduce(rel_tol=rel_tol, freqs=freqs)
        Hr = red.small_signal("V1", str(out)).response(freqs)
        err = np.max(np.abs(20 * np.log10(np.abs(Hr)) - 20 * np.log10(np.abs(Hf))))
        pr = [red.values.get(n, 0.0) for n in red.params]
        xr = red.core.solve_dc(pr, None, 1e-10, 100)
        sp = t_full / teval(red, pr, xr)
        print(f"{rel_tol:>9.0e} {len(red.transforms):>7} {err:>11.3f}  {sp:>11.2f}x")

    print("\nthe reduced model keeps the same unknowns (same interface), so it"
          "\ndrops straight into transient / AC / sensitivity -- just faster.")


if __name__ == "__main__":
    main()
