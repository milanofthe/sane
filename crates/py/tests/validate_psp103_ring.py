"""End-to-end validation: a CMOS ring oscillator on a native Verilog-A compact
model -- the SANE counterpart to circulax's `ring_oscillator_osdi` example.

Unlike circulax (which loads a compiled PSP103 `.osdi` binary as a black box),
SANE lowers the Verilog-A device *natively* onto its symbolic DAG (no OSDI, no C
codegen), so the whole analysis suite runs on it unchanged. We:

  1. load the model via `.veriloga` and instantiate N/P transistors (TYPE=+/-1),
  2. solve the (metastable) DC operating point,
  3. take exact small-signal poles -- a right-half-plane pair proves the ring is
     an oscillator and predicts its frequency (impossible on an OSDI black box),
  4. run a BDF transient from a symmetry-broken initial state,
  5. extract the oscillation frequency by zero-crossing + FFT.

Status (validated 2026-06):
  * SANE_RING_MODEL=EKV  -> converges, oscillates ~0.92 GHz: full parity proven.
  * SANE_RING_MODEL=PSP103 (the exact circulax model) lowers symbolically (845
    params, 16 residuals) but its internal surface-potential nodes do NOT yet
    converge in the DC Newton -- even a single pinned device fails. Same for
    PSP102. This is the one real parity gap vs circulax; tracked separately.

Run:  SANE_RING_MODEL=EKV .venv/Scripts/python crates/py/tests/validate_psp103_ring.py
"""
import os
import sys

import numpy as np

import sane

_CORPUS = os.environ.get(
    "SANE_VA_CORPUS",
    "C:/Repositories/TEMP/OpenVAF-Reloaded/integration_tests",
).replace("\\", "/")

# The compact model under test. EKV converges end-to-end in SANE today; the PSP
# surface-potential family (psp102/psp103) lowers symbolically but its internal
# nodes do not yet converge in the DC Newton -- see module docstring / report.
MODEL = os.environ.get("SANE_RING_MODEL", "EKV").upper()
_MODELS = {
    "EKV": (f"{_CORPUS}/EKV/ekv.va", "ekv_va"),
    "PSP103": (f"{_CORPUS}/PSP103/psp103.va", "PSP103VA"),
    "PSP102": (f"{_CORPUS}/PSP102/psp102.va", "PSP102VA"),
}
VA_PATH, VA_MODULE = _MODELS[MODEL]

VDD = 3.0
STAGES = 5
WN, LN = "1u", "1u"
WP, LP = "2u", "1u"
CL = "5f"


def build_ring(stages=STAGES):
    """A `stages`-stage CMOS ring oscillator on the chosen compact model.

    A brief startup current pulse into node n0 (like circulax's kick injector)
    breaks the symmetry *dynamically* from the consistent DC operating point, so
    the implicit integrator never has to start from an inconsistent state."""
    lines = [f'.veriloga "{VA_PATH}"', f"Vdd vdd 0 {VDD}"]
    for k in range(stages):
        inn = f"n{k}"
        out = f"n{(k + 1) % stages}"
        # N<name> D G S B  <module>  <params>   (TYPE=+1 NMOS, -1 PMOS)
        lines.append(f"NN{k} {out} {inn} 0 0 {VA_MODULE} TYPE=1 W={WN} L={LN}")
        lines.append(f"NP{k} {out} {inn} vdd vdd {VA_MODULE} TYPE=-1 W={WP} L={LP}")
        lines.append(f"Cl{k} {out} 0 {CL}")
    # Startup kick: 100 uA into n0 for the first 50 ps (DC value 0, so it leaves
    # the operating point untouched and only fires in the transient).
    lines.append("Ikick 0 n0 PULSE(0 100u 0 5p 5p 50p 1)")
    lines.append(".end")
    return "\n".join(lines)


def osc_frequency(t, y):
    """Frequency from rising-edge zero-crossings about the signal mean, with an
    FFT cross-check. Returns (f_zero_crossing, f_fft)."""
    y = np.asarray(y) - np.mean(y)
    # rising zero crossings via linear interpolation
    s = np.signbit(y)
    idx = np.where(~s[1:] & s[:-1])[0]  # - -> + transitions
    cross = []
    for i in idx:
        y0, y1 = y[i], y[i + 1]
        cross.append(t[i] + (t[i + 1] - t[i]) * (-y0) / (y1 - y0))
    f_zc = 1.0 / np.mean(np.diff(cross)) if len(cross) > 2 else float("nan")
    # FFT (uniform resample)
    tu = np.linspace(t[0], t[-1], len(t))
    yu = np.interp(tu, t, y)
    spec = np.abs(np.fft.rfft(yu * np.hanning(len(yu))))
    freqs = np.fft.rfftfreq(len(yu), tu[1] - tu[0])
    f_fft = freqs[1 + np.argmax(spec[1:])]
    return f_zc, f_fft


def main():
    print(f"Compact model: {MODEL}  ({VA_PATH})")
    print(f"Building {STAGES}-stage {MODEL} ring oscillator (VDD={VDD} V)...")
    dae = sane.Circuit.parse(build_ring()).extract()
    print(f"  extracted: dim={dae.dim}, params={len(dae.params)}, nnz={dae.nnz}")

    names = [dae.unknown_name(i) for i in range(dae.dim)]
    # The node-voltage unknowns (named v1, v2, ... by the extractor).
    v_idx = [i for i, n in enumerate(names) if n.startswith("v")]
    print("DC operating point (a ring has no stable DC -- may not converge)...")
    try:
        op = dae.operating_point()
        x0 = np.array([float(op[n]) for n in names])
        print("  converged (metastable point), node v's:",
              [round(x0[i], 4) for i in v_idx])
    except Exception as e:
        print(f"  did not converge ({str(e)[:48]}) -> start from mid-rail.")
        x0 = np.full(len(names), VDD / 2.0)

    # Architectural bonus over an OSDI black box: because the compact model is
    # lowered onto the symbolic DAG, we can take exact small-signal poles about
    # the metastable point. A right-half-plane complex pair PROVES the ring will
    # oscillate and predicts the frequency -- before any transient is run.
    try:
        poles = np.asarray(dae.small_signal("n0", "n2").poles())
        rhp = poles[poles.real > 0]
        if len(rhp):
            f_pred = abs(rhp[np.argmax(rhp.real)].imag) / (2 * np.pi)
            print(f"  small-signal: {len(rhp)} RHP pole(s) -> unstable, "
                  f"predicted f_osc ~ {f_pred / 1e9:.3f} GHz")
    except Exception as e:
        print(f"  small_signal skipped ({str(e)[:40]})")

    # The metastable point is an *unstable* equilibrium (RHP poles above). We do
    # NOT perturb x0 (an inconsistent state stalls the first implicit step's
    # stage Newton); instead the in-circuit startup current pulse kicks the ring
    # off the equilibrium dynamically, and its own instability grows the swing.
    # The integrator starts from the consistent DC operating point.
    del x0

    # Resolve a handful of stage delays; pick a horizon long enough for many
    # periods once it starts. dt_max keeps the integrator from over-striding.
    t_end = 50e-9
    t = np.linspace(0, t_end, 6000)
    print(f"Transient 0..{t_end * 1e9:.1f} ns (ESDIRK32, in-circuit kick)...")
    traj = dae.transient(t, rtol=1e-5, atol=1e-9, dt_max=2e-11)

    # Pick the ring node with the largest back-half swing to characterise.
    settle = len(t) // 2  # analyse the back half (post-startup)
    ring_nodes = [f"n{k}" for k in range(STAGES)]
    sigs = {nm: np.asarray(traj[nm]) for nm in ring_nodes}
    node = max(ring_nodes, key=lambda nm: np.ptp(sigs[nm][settle:]))
    v = sigs[node]
    swing = float(np.ptp(v[settle:]))
    print(f"  V({node}) swing (back half): {swing * 1e3:.1f} mV "
          f"[min {v[settle:].min():.3f}, max {v[settle:].max():.3f}]")

    if not np.isfinite(swing) or swing < 0.1 * VDD:
        print("  -> NO sustained oscillation (swing < 10% VDD or non-finite). "
              "For PSP102/PSP103 this is the known DC-convergence gap; "
              "for EKV-class models tune VDD/W/L/CL or stage count.")
        return 1

    f_zc, f_fft = osc_frequency(t[settle:], v[settle:])
    print(f"  f_osc (zero-crossing) = {f_zc / 1e9:.3f} GHz")
    print(f"  f_osc (FFT)           = {f_fft / 1e9:.3f} GHz")
    print(f"  stage delay ~ {1e12 / (2 * STAGES * f_zc):.1f} ps")
    print(f"OK: {MODEL} ring oscillator sustains oscillation end-to-end in SANE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
