"""Pole / zero analysis with exact sensitivities, all numeric.

Completes the small-signal foundation: poles (where the response blows up) and
transmission zeros (where it vanishes), plus the exact first-order movement of
each under any parameter -- the information a designer reads to place a pole or
notch. Everything is a numeric eigenproblem on the operating-point-linearized
G / C / B matrices; the symbolic transfer function is never formed.

    python crates/py/tests/pole_zero_demo.py
"""

import numpy as np

import sane


def main():
    np.set_printoptions(precision=4, suppress=True)

    # A lead network: R1 || C1 feeding R2. H(s) = R2 (1 + s R1 C1) / (...),
    # so a finite zero at -1/(R1 C1) and a pole at -(R1+R2)/(R1 R2 C1).
    deck = "V1 1 0 AC 1\nR1 1 2 1k\nC1 1 2 100n\nR2 2 0 2k"
    dae = sane.Circuit.parse(deck).extract()
    ss = dae.small_signal("V1", "2")

    poles = ss.poles()
    zeros = ss.zeros()
    print("== lead network (R1||C1 - R2) ==")
    print(f"  poles  = {poles} rad/s")
    print(f"  zeros  = {zeros} rad/s   (analytic zero = {-1/(1e3*1e-7):.0f})")

    # Self-check: the transfer vanishes at the zero's location s = z (which may
    # be off the jw axis): H(z) = e_out^T (z C + G)^{-1} B ~= 0.
    zc = zeros[0]
    hz = np.linalg.solve(ss.G + zc * ss.C, ss.B)[ss.out_index]
    print(f"  |H(s = zero)| = {abs(hz):.3e}  (~0, transfer blocked)\n")

    # Exact pole/zero sensitivity vs a finite-difference oracle.
    print("  exact d/dp of pole and zero (relative), vs finite differences:")
    print(f"    {'param':>6} {'dpole/dp (rel)':>16} {'dzero/dp (rel)':>16}")
    for prm in ["R1", "C1", "R2"]:
        p0 = dae.values[prm]
        psens = dae.pole_sensitivity("V1", prm)
        zsens = dae.zero_sensitivity("V1", "2", prm)
        # relative movement of the dominant pole / zero
        dp_rel = psens[0][1] * p0 / psens[0][0]
        dz_rel = zsens[0][1] * p0 / zsens[0][0] if zsens else 0.0

        # FD oracle.
        h = 1e-6 * p0
        pp = dae.small_signal("V1", "2", values={prm: p0 + h})
        fd_p = (pp.poles()[0] - poles[0]) / h * p0 / poles[0]
        zp = pp.zeros()
        fd_z = (zp[0] - zeros[0]) / h * p0 / zeros[0] if len(zp) else 0.0
        print(f"    {prm:>6} {dp_rel.real:>9.4f} (FD {fd_p.real:+.4f}) "
              f"{dz_rel.real:>9.4f} (FD {fd_z.real:+.4f})")

    # A series RLC read across the inductor is a band-stop: a complex zero pair
    # at the LC resonance, distinct from the (damped) pole pair.
    print("\n== series RLC, output across L (band-stop) ==")
    rlc = "V1 1 0 AC 1\nR1 1 2 50\nL1 2 3 1m\nC1 3 0 10n"
    # output = voltage across L = v2 - v3; approximate by the node with the zero
    ss2 = sane.Circuit.parse(rlc).extract().small_signal("V1", "2")
    print(f"  poles = {ss2.poles()} rad/s")
    print(f"  zeros = {ss2.zeros()} rad/s")


if __name__ == "__main__":
    main()
