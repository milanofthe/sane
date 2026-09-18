"""Demonstration: extract an operating-point-linearized compact model (LTI
state-space / transfer function) from a circuit, then validate poles and the AC
response against ngspice `.ac` -- through SANE's ergonomic Python API.

The whole linearization is one call:

    ss = dae.small_signal(input="V1", output="2")

which solves the DC bias, assembles G = dF/dx, C = dF/dx' and the exact input
coupling B = -dF/d(input) at the operating point, and exposes:

    ss.poles()           finite generalized eigenvalues of (-G, C)   [rad/s]
    ss.response(freqs)   H(j2*pi*f) = e_out^T (G + jw C)^-1 B
    ss.bode(freqs)       magnitude [dB] and phase [deg]

    python crates/py/tests/small_signal_demo.py
"""

import os
import re
import subprocess
import sys
import tempfile

import numpy as np

import sane


def find_ngspice():
    cand = os.path.join(sys.prefix, "Library", "bin", "ngspice_con.exe")
    return cand if os.path.exists(cand) else "ngspice"


NG = find_ngspice()


def ngspice_ac(deck, output_node, fstart, fstop, n):
    text = (f"* ac\n{deck}\n.control\nac dec {n} {fstart} {fstop}\n"
            f"print vdb({output_node}) vp({output_node})\n.endc\n.end\n")
    d = tempfile.mkdtemp()
    path = os.path.join(d, "ac.cir")
    open(path, "w").write(text)
    out = subprocess.run([NG, "-b", path], capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        m = re.match(r"\s*\d+\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$", ln)
        if m:
            rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return rows


def main():
    print("== operating-point-linearized compact models from circuit analysis ==\n")

    # 1) RC low-pass: single real pole at -1/(RC).
    rc = "V1 1 0 AC 1\nR1 1 2 1k\nC1 2 0 100n"
    ss = sane.Circuit.parse(rc).extract().small_signal("V1", "2")
    poles = ss.poles()
    fp = poles[np.argmin(np.abs(poles))] / (2 * np.pi)
    print(f"RC low-pass: pole = {poles} rad/s  (f_pole = {fp.real:.1f} Hz, "
          f"analytic = {-1/(2*np.pi*1e3*100e-9):.1f} Hz)")

    # AC response vs ngspice.
    freqs = np.logspace(2, 6, 9)
    mydb, _ = ss.bode(freqs)
    rows = ngspice_ac(rc, "2", 100, 1e6, 2)
    if rows:
        ngf = np.array([r[0] for r in rows])
        ngdb = np.array([r[1] for r in rows])
        ngi = np.interp(freqs, ngf, ngdb)
        print(f"  |H| vs ngspice .ac: max |dB error| = {np.max(np.abs(mydb - ngi)):.3f} dB\n")

    # 2) Series RLC: complex pole pair at -R/2L +/- j*sqrt(1/LC - (R/2L)^2).
    rlc = "V1 1 0 AC 1\nR1 1 2 50\nL1 2 3 1m\nC1 3 0 10n"
    poles = sane.Circuit.parse(rlc).extract().small_signal("V1", "3").poles()
    f0 = 1 / (2 * np.pi * np.sqrt(1e-3 * 1e-8))
    print(f"series RLC: poles = {poles} rad/s")
    print(f"  resonance f0 = {np.max(np.abs(poles))/(2*np.pi):.0f} Hz "
          f"(analytic {f0:.0f} Hz)\n")

    # 3) BJT common-emitter amp: small-signal at the bias point (nonlinear DC,
    #    linearized) -> midband gain + dominant pole. This is the real value:
    #    a compact LTI model of a NONLINEAR circuit at its operating point.
    ce = ("Vcc 1 0 12\nVin 4 0 DC 0.7 AC 1\n"
          "R1 1 2 4.7k\nRb 4 3 100k\nCb 3 0 1u\n"
          "Q1 2 3 0 qm\n.model qm NPN(Is=1e-15 Bf=150 VAf=80)")
    ss = sane.Circuit.parse(ce).extract().small_signal("Vin", "2")
    gain_db, _ = ss.bode([1e3])
    print(f"BJT CE amp: midband gain @1kHz = {gain_db[0]:.2f} dB, "
          f"poles = {ss.poles()} rad/s")


if __name__ == "__main__":
    main()
