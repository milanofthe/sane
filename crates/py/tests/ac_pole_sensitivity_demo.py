"""Exact AC and pole/zero sensitivity via AD (no finite differences in the
method; FD is used only as an independent validation oracle).

Small-signal model at the operating point: A(s) = G + s*C, response
H(s) = e_out^T A(s)^{-1} B. SANE provides the EXACT total derivatives dG/dp,
dC/dp, dB/dp (including the operating-point shift) by AD. From those:

  AC:    dH/dp = e_out^T A^{-1} ( dB - (dG + s dC) A^{-1} B )
  pole:  dlam/dp = -u^T (dG + lam dC) v / (u^T C v)       (lam a generalized eig of (-G,C),
                                                       v/u its right/left eigenvectors)

Setup goes through the ergonomic API (``Circuit.parse(...).extract().small_signal``);
the exact AD derivatives use the raw ``dae.core.ac_derivatives`` (an advanced,
not-yet-wrapped primitive).

    python crates/py/tests/ac_pole_sensitivity_demo.py
"""

import numpy as np
import scipy.linalg as sla

import sane


def model(deck, input_src, out_node):
    """Operating-point small-signal model through the ergonomic API."""
    dae = sane.Circuit.parse(deck).extract()
    ss = dae.small_signal(input_src, out_node)
    p = [dae.values.get(n, 0.0) for n in dae.params]
    x0 = dae.operating_point().vector.tolist()
    return dae, p, x0, ss.out_index, ss.G, ss.C, ss.B


def main():
    np.set_printoptions(precision=4, suppress=True)

    # --- AC sensitivity: RC low-pass, sensitivity of |H| at the corner ---
    print("== AC sensitivity (RC low-pass, |H| at f=1591 Hz corner) ==")
    rc = "V1 1 0 AC 1\nR1 1 2 1k\nC1 2 0 100n"
    dae, p, x0, out, G, C, B = model(rc, "V1", "2")
    f = 1591.5
    s = 2j * np.pi * f
    A = G + s * C
    dx = np.linalg.solve(A, B)
    H = dx[out]
    e = np.zeros(len(G)); e[out] = 1.0
    print(f"  H = {H:.4f}   |H| = {abs(H):.4f}")
    for name in ["R1", "C1"]:
        dG, dC, dB = (np.array(m) for m in dae.core.ac_derivatives("V1", name, x0, p, 0.0))
        dA = dG + s * dC
        dH = e @ np.linalg.solve(A, np.array(dB) - dA @ dx)  # exact (AD)
        # FD oracle: perturb the parameter, re-linearize, recompute H.
        k = dae.params.index(name); h = 1e-6 * abs(p[k])
        ssp = dae.small_signal("V1", "2", values={name: p[k] + h})
        H1 = np.linalg.solve(ssp.G + s * ssp.C, ssp.B)[out]
        dH_fd = (H1 - H) / h
        rel = (dH * p[k] / H).real  # relative sens of |H| (Re of dlnH/dlnp)
        print(f"  {name:>4}: dlnH per d{name}/{name} = {rel:+.4f}   "
              f"(AD dH={dH:.3e}, FD dH={dH_fd:.3e}, rel.err {abs(dH-dH_fd)/(abs(dH_fd)+1e-30):.1e})")

    # --- pole sensitivity: series RLC, how the pole pair moves per component ---
    print("\n== pole sensitivity (series RLC) ==")
    rlc = "V1 1 0 AC 1\nR1 1 2 50\nL1 2 3 1m\nC1 3 0 10n"
    dae, p, x0, out, G, C, B = model(rlc, "V1", "3")
    # poles = finite generalized eigenvalues of (-G, C); right v, left u.
    w, vr = sla.eig(-G, C)
    wt, ur = sla.eig((-G).T, C.T)
    finite = np.isfinite(w)
    lam = w[finite][np.argmax(np.abs(w[finite].imag))]  # the resonant pole
    vi = np.where(finite)[0][np.argmax(np.abs(w[finite].imag))]
    v = vr[:, vi]
    ui = np.argmin(np.abs(wt - lam))
    uvec = ur[:, ui]
    print(f"  resonant pole lam = {lam:.3e} rad/s  (f0 ~ {abs(lam)/2/np.pi:.0f} Hz)")
    for name in ["R1", "L1", "C1"]:
        dG, dC, _ = (np.array(m) for m in dae.core.ac_derivatives("V1", name, x0, p, 0.0))
        dlam = -(uvec @ (dG + lam * dC) @ v) / (uvec @ C @ v)  # exact (AD)
        # FD oracle.
        k = dae.params.index(name); h = 1e-6 * abs(p[k])
        ssp = dae.small_signal("V1", "3", values={name: p[k] + h})
        w1 = sla.eig(-ssp.G, ssp.C, right=False)
        lam1 = w1[np.isfinite(w1)][np.argmin(np.abs(w1[np.isfinite(w1)] - lam))]
        dlam_fd = (lam1 - lam) / h
        rel = dlam * p[k] / lam
        print(f"  {name:>4}: dlam/dlam per d{name}/{name} = {rel:+.4f}   "
              f"(AD dlam={dlam:.3e}, FD dlam={dlam_fd:.3e}, rel.err "
              f"{abs(dlam-dlam_fd)/(abs(dlam_fd)+1e-30):.1e})")


if __name__ == "__main__":
    main()
