"""What SANE is actually for: rank every component by how strongly it controls
a behavior -- which ones are negligible (prune them) and which are the tuning
knobs. Computed EXACTLY and CHEAPLY via the adjoint method on SANE's symbolic
Jacobians (one extra linear solve for ALL components), not N finite-difference
re-simulations.

This is the same analysis the README leads with, now through SANE's ergonomic
Python API: build, extract, ``sensitivity`` -- the adjoint solve, the relative
ranking and the device roll-up are all handled for you and labeled by name.

    python crates/py/tests/sensitivity_demo.py
"""

import sane


# BJT common-emitter with emitter degeneration + a deliberately negligible
# 10 Mohm parasitic Rp from the collector to ground. Which components set the
# collector bias (the knobs), and which can we drop (Rp)?
CE = """
    Vcc 1 0 12
    R1 1 2 47k
    R2 2 0 10k
    Rc 1 3 4.7k
    Re 4 0 1k
    Rp 3 0 10meg
    Q1 3 2 4 qm
    .model qm NPN(Is=1e-15 Bf=150 VAf=80)
"""


def main():
    dae = sane.Circuit.parse(CE).extract()

    out = "3"  # collector node
    y = dae.operating_point()[out]
    sens = dae.sensitivity(out)

    print(f"== V_collector = {y:+.4f} V  (sensitivity ranking) ==")
    print(f"{'component':>10} {'rel.sens (adjoint)':>18} {'rel.sens (FD)':>14}")

    # Exact relative sensitivity for every parameter, ranked, with a
    # finite-difference cross-check (re-solve DC with a perturbed value).
    for name, s in sens.ranked(relative=True, threshold=1e-9):
        p0 = dae.values.get(name, 0.0)
        if p0:
            h = 1e-6 * abs(p0)
            yk = dae.operating_point(values={name: p0 + h})[out]
            fd = (yk - y) / h * p0 / y
        else:
            fd = float("nan")
        print(f"{name:>10} {s:>18.4f} {fd:>14.4f}")

    # The negligible parasitic falls to the bottom of the ranking on its own.
    prunable = [n for n, s in sens.ranked(relative=True) if abs(s) < 1e-3]
    print(f"\nnegligible / prunable: {', '.join(prunable)}")


if __name__ == "__main__":
    main()
