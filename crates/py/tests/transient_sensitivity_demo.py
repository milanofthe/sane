"""Exact forward transient sensitivity: dx(t)/dp for each parameter, obtained by
symbolically augmenting the DAE with the (exact) sensitivity equations
`G s + C s' + dF/dp = 0` and integrating the joint system. No finite differences
in the method; reactive parameters (C, L) are handled correctly because dF/dp
keeps the x'-coupling symbolically.

The sensitivity equations are exact; the time integration carries the usual
solver tolerance, so the result converges to a finite-difference oracle as the
tolerance tightens (shown below).

Setup and the FD oracle go through the ergonomic API; the augmented integration
uses the raw ``dae.core.transient_sensitivity`` (an advanced primitive).

    python crates/py/tests/transient_sensitivity_demo.py
"""

import numpy as np

import sane

DECK = "V1 1 0 SIN(0 1 1k)\nR1 1 2 1k\nC1 2 0 1u"  # RC low-pass, sine-driven


def main():
    dae = sane.Circuit.parse(DECK).extract()
    p = [dae.values.get(n, 0.0) for n in dae.params]
    ts = np.linspace(0, 2e-3, 41)
    v2 = dae.unknown_index("2")

    def fd(name, rt, at):  # central-difference oracle on the transient
        k = dae.params.index(name)
        h = 1e-2 * abs(p[k])
        run = lambda dp: dae.transient(ts, values={name: p[k] + dp}, rtol=rt, atol=at)["v2"]
        return (run(+h) - run(-h)) / (2 * h)

    print("forward transient sensitivity  dv2(t)/dp  (exact AD, vs FD oracle)")
    print("  reactive C1 included; convergence to FD as the tolerance tightens:\n")
    print(f"  {'rtol':>8} {'dv2/dR1 err':>14} {'dv2/dC1 err':>14}")
    for rt in [1e-4, 1e-5, 3e-6]:
        at = rt * 1e-3
        n, traj = dae.core.transient_sensitivity(["R1", "C1"], list(ts), rt, at)
        errs = []
        for kidx, name in [(0, "R1"), (1, "C1")]:
            ad = np.array([traj[ti][n + kidx * n + v2] for ti in range(len(ts))])
            f = fd(name, rt, at)
            m = np.abs(f) > 0.2 * np.max(np.abs(f))
            errs.append(np.max(np.abs(ad[m] - f[m]) / np.abs(f[m])))
        print(f"  {rt:>8.0e} {errs[0]:>14.2e} {errs[1]:>14.2e}")

    # The actual deliverable: the time-domain sensitivity waveforms.
    n, traj = dae.core.transient_sensitivity(["R1", "C1"], list(ts), 3e-6, 3e-9)
    sR = np.array([traj[ti][n + 0 * n + v2] for ti in range(len(ts))])
    sC = np.array([traj[ti][n + 1 * n + v2] for ti in range(len(ts))])
    print(f"\n  peak |dv2/dR1| = {np.max(np.abs(sR)):.3e} V/Ohm,"
          f"  peak |dv2/dC1| = {np.max(np.abs(sC)):.3e} V/F")


if __name__ == "__main__":
    main()
