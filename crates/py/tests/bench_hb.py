"""Harmonic-balance benchmark on real nonlinear circuits.

ngspice has no harmonic balance, so the reference for accuracy is SANE's own
settled transient simulation, FFT'd over the last period -- the time-domain
truth HB must reproduce. We then report scaling: HB wall-clock vs circuit size
(a parametric diode ladder) and vs harmonic count K.

(External HB cross-check: Xyce -- open source, Sandia -- supports single and
multi-tone HB via .HB; SPICE OPUS has a shooting PSS. Not wired up here yet.)

Run after `maturin develop --release -m crates/py/Cargo.toml`:
    python crates/py/tests/bench_hb.py
"""

import os
import time

import numpy as np

import sane

FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "paper", "benchmarks", "circuits", "fixtures"
)


def build(deck):
    c = sane._core.parse(deck + "\n.end")
    dae = c.extract_dae()
    p = [dae.values().get(name, 0.0) for name in dae.params()]
    return dae, p


def voltage_indices(unknowns):
    return [i for i, u in enumerate(unknowns) if u.startswith("v") and u[1:].isdigit()]


def transient_fft(dae, p, f0, n_periods=40, mfft=128):
    """Settle the transient and FFT the last period -> physical coefficients
    X_k = (1/M) sum_m x_m e^{-j k w0 t_m}, one column per unknown."""
    period = 1.0 / f0
    t_eval = [j * period / mfft for j in range(n_periods * mfft)]
    t0 = time.perf_counter()
    traj = np.array(dae.solve_transient(p, t_eval, rtol=1e-6, atol=1e-9))
    dt = time.perf_counter() - t0
    last = traj[(n_periods - 1) * mfft : n_periods * mfft, :]
    return np.fft.rfft(last, axis=0) / mfft, dt


def hb_run(dae, p, f0, harmonics, x0=None, repeats=1):
    """Run HB `repeats` times (best total wall-clock) from a precomputed DC
    start. Returns spectra plus the best (setup_ms, solve_ms) split and total."""
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = dae.solve_hb(p, f0, harmonics=harmonics, x0=x0)
        best = min(best, time.perf_counter() - t0)
    spectra, conv, iters, rnorm, setup_ms, solve_ms = out
    arr = np.array([[complex(re, im) for (re, im) in row] for row in spectra])
    return arr, conv, iters, rnorm, best, setup_ms, solve_ms


# ----------------------------------------------------------------------------
# 1. Accuracy on real circuits vs settled transient FFT.
# ----------------------------------------------------------------------------
ACCURACY = [
    ("biased_diode_rc",
     "V1 in 0 SIN(0.6 0.15 1000)\nR1 in mid 1k\nD1 mid 0 DMOD\n"
     "C1 mid 0 100n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)", 1000.0),
    ("diode_mixer_bias",
     "V1 in 0 SIN(0.5 0.25 2000)\nR1 in mid 2k\nD1 mid 0 DMOD\n"
     "C1 mid 0 47n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)", 2000.0),
    ("antiparallel_clip",
     "V1 in 0 SIN(0 0.9 1000)\nR1 in out 1k\nD1 out 0 DMOD\nD2 0 out DMOD\n"
     "C1 out 0 100n\n.model DMOD D(Is=1e-14 N=1 Vt=0.02585)", 1000.0),
    # An active device (MOSFET, level-1 square law): biased into saturation so
    # the operating point stays within one region -- the piecewise model is then
    # locally smooth and HB converges fast.
    ("mosfet_cs_amp",
     "M1 d g 0 0 NMOS1\nRD vdd d 5k\nVdd vdd 0 DC 5\nVg g 0 SIN(1.0 0.05 1000)\n"
     "Cout d 0 1n\n.model NMOS1 NMOS(Kp=20u W=10 L=1 Vto=0.5)", 1000.0),
]


def accuracy_table():
    print("== accuracy vs settled transient FFT (K=8) ==")
    print(f"{'circuit':<18} {'conv':>4} {'it':>3} {'||R||':>8} "
          f"{'t_HB':>7} {'t_tran':>8} {'speedup':>7}  node  H1/H2/H3/H4/H5 (HB|tran|rel)")
    for name, deck, f0 in ACCURACY:
        dae, p = build(deck)
        unknowns = dae.unknowns()
        vidx = voltage_indices(unknowns)
        x_dc = dae.solve_dc(p)
        hb, conv, iters, rnorm, t_hb, _, _ = hb_run(dae, p, f0, 8, x0=x_dc, repeats=5)
        try:
            coeffs, t_tr = transient_fft(dae, p, f0)
        except Exception as e:
            print(f"{name:<18}  transient failed: {str(e)[:60]}")
            continue
        kmax = coeffs.shape[0]
        oi = max(vidx, key=lambda i: sum(abs(coeffs[k, i]) for k in range(2, kmax)))
        floor = 5e-4 * abs(coeffs[1, oi])
        cols = []
        for k in range(1, 6):
            a, b = abs(hb[oi, k]), abs(coeffs[k, oi])
            tag = f"{abs(a - b) / b:4.1%}" if b > floor else " neg"
            cols.append(f"H{k}:{a:.1e}|{b:.1e}|{tag}")
        sp = t_tr / t_hb
        print(f"{name:<18} {str(conv):>4} {iters:>3} {rnorm:>8.1e} "
              f"{t_hb*1e3:>6.1f}m {t_tr*1e3:>7.1f}m {sp:>6.1f}x  {unknowns[oi]:>4}  "
              + "  ".join(cols))


# ----------------------------------------------------------------------------
# 2. Scaling with circuit size: a biased RC diode ladder, N stages.
# ----------------------------------------------------------------------------
def diode_ladder(n, bias=0.7, amp=0.1, f0=1000.0):
    lines = [f"V1 1 0 SIN({bias} {amp} {f0})", ".model dm D(Is=1e-14 N=1 Vt=0.025852)"]
    for i in range(1, n + 1):
        lines += [f"R{i} {i} {i+1} 1k", f"D{i} {i+1} 0 dm", f"C{i} {i+1} 0 10n"]
    return "\n".join(lines)


def scaling_size():
    print("\n== scaling with circuit size (diode ladder, K=8) ==")
    print(f"{'N':>5} {'n_unk':>6} {'hb_dim':>7} {'it':>3} "
          f"{'||R||':>8} {'setup':>7} {'solve':>8} {'ms/it':>7} {'t_dc':>7}")
    for n in (16, 64, 256, 512, 1024, 2048):
        dae, p = build(diode_ladder(n))
        nunk = dae.dim()
        t0 = time.perf_counter()
        x_dc = dae.solve_dc(p)
        t_dc = time.perf_counter() - t0
        hb, conv, iters, rnorm, t_hb, setup, solve = hb_run(dae, p, 1000.0, 8, x0=x_dc, repeats=2)
        print(f"{n:>5} {nunk:>6} {nunk*9:>7} {iters:>3} {rnorm:>8.1e} "
              f"{setup:>6.1f}m {solve:>7.1f}m {solve/max(iters,1):>6.2f}m {t_dc*1e3:>6.2f}m"
              + ("" if conv else "  NOT CONVERGED"))


# ----------------------------------------------------------------------------
# 3. Scaling with harmonic count K, fixed mid-size circuit.
# ----------------------------------------------------------------------------
def scaling_harmonics():
    print("\n== scaling with harmonic count (diode ladder N=16) ==")
    dae, p = build(diode_ladder(16))
    nunk = dae.dim()
    x_dc = dae.solve_dc(p)
    print(f"{'K':>4} {'hb_dim':>7} {'it':>3} {'||R||':>8} {'setup':>7} {'solve':>8} {'ms/it':>7}")
    for k in (2, 4, 8, 16, 32, 64):
        hb, conv, iters, rnorm, t_hb, setup, solve = hb_run(dae, p, 1000.0, k, x0=x_dc, repeats=3)
        print(f"{k:>4} {nunk*(k+1):>7} {iters:>3} {rnorm:>8.1e} "
              f"{setup:>6.1f}m {solve:>7.1f}m {solve/max(iters,1):>6.2f}m"
              + ("" if conv else "  NOT CONVERGED"))


# ----------------------------------------------------------------------------
# 4. Real transistor-level topologies (the fixtures), with a SIN drive injected.
#    The closed-loop uA741 is robust; the open-loop OTA / 3-stage opamp need a
#    small drive to keep the output in the linear region (large swing -> region
#    crossings -> HB needs continuation, not yet wired up).
# ----------------------------------------------------------------------------
REAL_LARGE = [
    ("ua741_inverting", "ua741_inverting.cir", "Vin 100 0 DC 0V",
     "Vin 100 0 SIN(0 0.1 1000)", 1000.0),
    ("cmos_diffpair_ota", "cmos_diffpair_ota.cir", "Vd 101 0 DC 0V",
     "Vd 101 0 SIN(0 0.01 1000)", 1000.0),
    ("multistage_bjt_opamp", "multistage_bjt_opamp.cir", "Vd 101 0 DC 0V",
     "Vd 101 0 SIN(0 1e-4 1000)", 1000.0),
]


def real_large_table():
    print("\n== real transistor-level topologies (K=8, SIN drive) ==")
    print(f"{'circuit':<22} {'devices':>7} {'n_unk':>6} {'hb_dim':>7} {'conv':>4} {'it':>3} "
          f"{'||R||':>8} {'t_dc':>7} {'setup':>7} {'solve':>8}  max_HD_err")
    for name, fname, old, new, f0 in REAL_LARGE:
        deck = open(os.path.join(FIXTURES, fname)).read().replace(old, new)
        ndev = sum(1 for ln in deck.splitlines()
                   if ln[:1] in "QMDJ" and not ln.startswith("."))
        dae, p = build(deck.replace(".end", "").replace(".OP", "").replace(".op", ""))
        unknowns = dae.unknowns()
        vidx = voltage_indices(unknowns)
        t0 = time.perf_counter()
        x_dc = dae.solve_dc(p)
        t_dc = time.perf_counter() - t0
        hb, conv, iters, rnorm, _, setup, solve = hb_run(dae, p, f0, 8, x0=x_dc, repeats=3)
        # Accuracy vs transient where it settles (best-effort; opamps are stiff).
        hd = "n/a"
        try:
            coeffs, _ = transient_fft(dae, p, f0, n_periods=20, mfft=64)
            kmax = coeffs.shape[0]
            oi = max(vidx, key=lambda i: sum(abs(coeffs[k, i]) for k in range(2, kmax)))
            floor = 5e-4 * abs(coeffs[1, oi])
            errs = [abs(abs(hb[oi, k]) - abs(coeffs[k, oi])) / abs(coeffs[k, oi])
                    for k in range(1, min(6, kmax)) if abs(coeffs[k, oi]) > floor]
            hd = f"{max(errs):.1%}" if errs else "~0"
        except Exception:
            hd = "tran-fail"
        print(f"{name:<22} {ndev:>7} {dae.dim():>6} {dae.dim()*9:>7} {str(conv):>4} {iters:>3} "
              f"{rnorm:>8.1e} {t_dc*1e3:>6.1f}m {setup:>6.1f}m {solve:>7.1f}m  {hd}")


if __name__ == "__main__":
    accuracy_table()
    real_large_table()
    scaling_size()
    scaling_harmonics()
