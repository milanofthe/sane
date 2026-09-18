"""S-parameter results and Touchstone v1 file I/O.

The port convention used across SANE's SP analysis: a port is an ideal
voltage source in series with its reference impedance ``z0`` (a plain
resistor in the netlist), and the port node is the terminal BEHIND that
resistor, on the network side. With that Thevenin form the scattering
matrix follows from node voltages alone:

    S_ij = 2 * sqrt(z0_j / z0_i) * V_i|_(Vs_j = 1)  -  delta_ij

so each column j is one AC transfer sweep with source j driving and every
other port source dead (the AC stimulus is per-input by construction).
"""

import numpy as np

__all__ = [
    "SParams",
    "read_touchstone",
    "write_touchstone",
    "s_to_y",
    "y_to_s",
    "fit_verilog_a",
]

_UNITS = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}


class SParams:
    """An S-parameter sweep: ``s[k, i, j]`` over ``freqs[k]``, with the
    per-port reference impedances ``z0``."""

    def __init__(self, freqs, s, z0):
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        self.s = np.asarray(s, dtype=complex)
        nports = self.s.shape[-1]
        self.z0 = np.broadcast_to(np.asarray(z0, dtype=float), (nports,)).copy()
        if self.s.shape != (len(self.freqs), nports, nports):
            raise ValueError(f"S must be (nf, n, n); got {self.s.shape}")

    @property
    def nports(self):
        return self.s.shape[-1]

    def __getitem__(self, ij):
        """``sp[2, 1]`` -> the S21 sweep (1-based port indices)."""
        i, j = ij
        return self.s[:, i - 1, j - 1]

    def db(self, i, j):
        """``|S_ij|`` in dB (1-based port indices)."""
        return 20.0 * np.log10(np.maximum(np.abs(self[i, j]), 1e-300))

    def to_touchstone(self, path, fmt="RI"):
        write_touchstone(path, self.freqs, self.s, z0=float(self.z0[0]), fmt=fmt)

    def to_verilog_a(self, name, port_names=None, **kw):
        """Rational macromodel of this sweep as Verilog-A source (see
        :func:`fit_verilog_a`)."""
        return fit_verilog_a(self.freqs, self.s, z0=self.z0, name=name,
                             port_names=port_names, **kw)

    def __repr__(self):
        f = self.freqs
        return (f"SParams({self.nports} ports, {len(f)} points, "
                f"{f[0]:.4g}..{f[-1]:.4g} Hz, z0={self.z0.tolist()})")


def _pairs_to_complex(fmt, a, b):
    if fmt == "RI":
        return a + 1j * b
    if fmt == "MA":
        return a * np.exp(1j * np.deg2rad(b))
    if fmt == "DB":
        return 10.0 ** (a / 20.0) * np.exp(1j * np.deg2rad(b))
    raise ValueError(f"unknown touchstone format '{fmt}'")


def read_touchstone(path):
    """Parse a Touchstone v1 file (``.sNp``); the port count comes from the
    extension. Returns an :class:`SParams`. Handles the spec's 2-port column
    order (S11 S21 S12 S22) and the row-major order for other port counts;
    continuation lines are folded by reading a flat number stream."""
    path = str(path)
    m = path.lower().rsplit(".s", 1)
    if len(m) != 2 or not m[1][:-1].isdigit() or not m[1].endswith("p"):
        raise ValueError(f"not a .sNp touchstone path: {path}")
    n = int(m[1][:-1])
    unit, fmt, z0 = 1e9, "MA", 50.0  # spec defaults: GHz, MA, 50 ohms
    nums = []
    with open(path) as fh:
        for raw in fh:
            line = raw.split("!", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                tok = line[1:].upper().split()
                for k, t in enumerate(tok):
                    if t in _UNITS:
                        unit = _UNITS[t]
                    elif t in ("RI", "MA", "DB"):
                        fmt = t
                    elif t == "R" and k + 1 < len(tok):
                        z0 = float(tok[k + 1])
                continue
            nums.extend(float(t) for t in line.split())
    per = 1 + 2 * n * n
    if len(nums) % per:
        raise ValueError(f"touchstone data not a multiple of {per} values")
    rows = np.asarray(nums).reshape(-1, per)
    freqs = rows[:, 0] * unit
    a = rows[:, 1::2]
    b = rows[:, 2::2]
    s = _pairs_to_complex(fmt, a, b).reshape(-1, n, n)
    if n == 2:
        # v1 stores 2-ports as S11 S21 S12 S22, i.e. column-major
        s = s.transpose(0, 2, 1)
    return SParams(freqs, s, z0)


def s_to_y(s, z0=50.0):
    """Scattering to admittance matrices, per frequency:
    ``Y = sqrt(Y0) (I - S) (I + S)^-1 sqrt(Y0)`` with ``Y0 = diag(1/z0)``."""
    s = np.asarray(s, dtype=complex)
    n = s.shape[-1]
    z0 = np.broadcast_to(np.asarray(z0, dtype=float), (n,))
    ry = np.diag(1.0 / np.sqrt(z0))
    eye = np.eye(n)
    return np.stack([ry @ (eye - sk) @ np.linalg.inv(eye + sk) @ ry for sk in s])


def y_to_s(y, z0=50.0):
    """Admittance to scattering matrices, per frequency (inverse of
    :func:`s_to_y`): ``S = (I - Z) (I + Z)^-1`` with ``Z = sqrt(z0) Y sqrt(z0)``."""
    y = np.asarray(y, dtype=complex)
    n = y.shape[-1]
    z0 = np.broadcast_to(np.asarray(z0, dtype=float), (n,))
    rz = np.diag(np.sqrt(z0))
    eye = np.eye(n)
    out = []
    for yk in y:
        zn = rz @ yk @ rz
        out.append((eye - zn) @ np.linalg.inv(eye + zn))
    return np.stack(out)


def fit_verilog_a(freqs, s, z0=50.0, name="sparam_macro", port_names=None,
                  tol=1e-4, max_poles=40, enforce_passivity=True,
                  threshold=1e-2, force=False):
    """Rational macromodel of an S-parameter sweep as Verilog-A source.

    The sweep is converted to admittances, fitted with relaxed vector
    fitting (common poles, automatic order), passivity-checked, and emitted
    as a deterministic Verilog-A module in SANE's AD-friendly grammar
    subset (``ddt`` + linear contributions only, so the macromodel is smooth
    for gradients). Load the result with a ``.veriloga`` directive and
    instantiate it as ``N<name> <nodes...> gnd <module>`` (the module carries
    an explicit ground terminal).

    Returns ``(va_source, fit_error, n_poles)``.
    """
    from . import _core

    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    y = s_to_y(s, z0)
    nf, n = y.shape[0], y.shape[-1]
    yf = y.reshape(nf, n * n)
    z0s = np.broadcast_to(np.asarray(z0, dtype=float), (n,))
    return _core.vectfit_verilog_a(
        list(freqs), yf.real.tolist(), yf.imag.tolist(), n, name,
        port_names=list(port_names) if port_names else None,
        z0=float(z0s[0]), tol=tol, max_poles=max_poles,
        enforce_passivity=enforce_passivity, threshold=threshold, force=force,
    )


def write_touchstone(path, freqs, s, z0=50.0, fmt="RI"):
    """Write a Touchstone v1 file in ``RI``, ``MA`` or ``DB`` format (Hz
    frequency unit, common reference impedance ``z0``)."""
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    s = np.asarray(s, dtype=complex)
    n = s.shape[-1]
    if fmt not in ("RI", "MA", "DB"):
        raise ValueError(f"unknown touchstone format '{fmt}'")
    data = s.transpose(0, 2, 1) if n == 2 else s  # spec 2-port column order
    lines = [f"# HZ S {fmt} R {z0:g}"]
    for k, f in enumerate(freqs):
        vals = []
        for entry in data[k].reshape(-1):
            if fmt == "RI":
                vals += [entry.real, entry.imag]
            elif fmt == "MA":
                vals += [abs(entry), np.rad2deg(np.angle(entry))]
            else:
                vals += [20.0 * np.log10(max(abs(entry), 1e-300)),
                         np.rad2deg(np.angle(entry))]
        lines.append(" ".join([f"{f:.10g}"] + [f"{v:.10g}" for v in vals]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
