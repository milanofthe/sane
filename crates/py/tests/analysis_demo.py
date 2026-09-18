"""The final vision: from a circuit, tell the designer exactly which components
matter, hierarchically, and how the dominant tuning knobs interact -- all from
SANE's symbolic engine, exactly, through the ergonomic Python API.

1. First-order: exact component sensitivity of a chosen metric w.r.t. EVERY
   parameter (adjoint, one solve). Ranked, with hierarchical roll-up
   (parameter -> device -> subsystem) using the dotted parameter names --
   ``sensitivity().ranked()`` and ``.rollup()``.
2. Second-order: the Hessian on the dominant parameter subset -- the
   *interactions* between tuning knobs (off-diagonal) and the range over which
   the linear sensitivity holds (diagonal). Exact via the second-order adjoint
   (``hessian``), validated here against a full finite-difference Hessian.

    python crates/py/tests/analysis_demo.py
"""

import numpy as np

import sane

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

    y0 = dae.operating_point()[out]
    print(f"metric  V_collector = {y0:+.4f} V\n")

    # --- 1. first-order: exact sensitivity, ranked + hierarchical roll-up ---
    sens = dae.sensitivity(out)
    print("first-order sensitivity (relative), ranked:")
    for name, s in sens.ranked(relative=True, threshold=1e-6):
        print(f"  {name:>10}  {s:+.4f}")
    prunable = [n for n, s in sens.ranked(relative=True) if abs(s) <= 1e-3]
    print("  (negligible / prunable: " + ", ".join(prunable) + ")")

    print("\n  component roll-up (importance = L2 of relative leaf sensitivities):")
    for comp, imp in sens.rollup(relative=True)[:6]:
        print(f"    {comp:>8}  {imp:.4f}")

    # --- 2. second-order: Hessian on the dominant subset (the tuning knobs) ---
    sub = [n for n, s in sens.ranked(relative=True) if abs(s) > 1e-1][:4]
    Hs = dae.hessian(out, sub)
    p0 = {n: dae.values.get(n, 0.0) for n in sub}

    print(f"\nsecond-order interactions among tuning knobs {sub}")
    print("  (relative Hessian; diagonal = curvature, off-diagonal = coupling)")
    print("            " + "".join(f"{nm:>10}" for nm in sub))
    for a, na in enumerate(sub):
        row = "".join(f"{Hs[a, b] * p0[na] * p0[nb] / y0:>10.3f}" for b, nb in enumerate(sub))
        print(f"  {na:>10}{row}")

    # --- validate the EXACT (AD) Hessian against a full FD Hessian oracle ---
    def y(overrides):
        return dae.operating_point(values=overrides)[out]

    base = {n: dae.values.get(n, 0.0) for n in sub}
    h = 1e-4
    maxerr = 0.0
    for a, na in enumerate(sub):
        for b, nb in enumerate(sub):
            ha, hb = h * abs(base[na]), h * abs(base[nb])
            def shifted(da, db):
                o = dict(base); o[na] = o[na] + da; o[nb] = o[nb] + db
                return o
            fd = (y(shifted(ha, hb)) - y(shifted(ha, -hb))
                  - y(shifted(-ha, hb)) + y(shifted(-ha, -hb))) / (4 * ha * hb)
            maxerr = max(maxerr, abs(fd - Hs[a, b]) / (abs(fd) + 1e-12))
    print(f"\n  exact (AD) Hessian vs full FD Hessian oracle: max rel error = {maxerr:.2e}")


if __name__ == "__main__":
    main()
