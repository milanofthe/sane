"""SANE: circuit analysis with exact sensitivities.

SANE extracts the differential-algebraic system ``I(x, t) + d/dt Q(x) = 0``
from a circuit as an expression graph and analyzes it exactly: DC operating
point and sweeps, transient response, small-signal AC transfer and
S-parameters, poles / zeros, noise, state space, reduced models and harmonic
balance -- and, for every one of them, exact parameter sensitivities (first
and second order, hierarchically) by automatic differentiation of the graph.

Everything runs in Rust; this package is a thin layer over it. A model is
taken to a binding of its parameters, the point every analysis runs at;
results carry their data as numpy arrays (over the engine's own memory, no
copies) and their derivatives as methods:

.. code-block:: python

    import numpy as np
    import sane

    model = sane.Model.from_netlist('''
        V1 in 0 5
        R1 in out 1k
        C1 out 0 1u
    ''')
    pt = model.at(R1=2e3)                       # the point of a binding
    op = pt.operating_point()
    print(op["out"], op.sensitivity("out").grad)

    ac = pt.ac("V1", ["out"], np.geomspace(1, 1e6, 61))
    print(ac.mag_db("out"), ac.sensitivity().grad.shape)

    tr = pt.transient(np.linspace(0, 5e-3, 200))
    print(tr.signal("out")[-1])

Circuits come from SPICE netlists (``sane.parse``) or are built element by
element (:class:`Circuit`), Verilog-A modules included; ``sane.Model(circuit)``
sets up the model of either.
"""


from . import _core
from ._core import (
    Circuit,
    Waveform,
    Model,
    Point,
    ParamGroup,
    OperatingPoint,
    Regularization,
    Sensitivity,
    Hessian,
    Gradient,
    DcSweep,
    Trajectory,
    TrajectorySensitivity,
    AcResponse,
    AcSensitivity,
    AcHessian,
    SParameters,
    SpSensitivity,
    NoiseSpectrum,
    NoiseSensitivity,
    Poles,
    Zeros,
    RootSensitivity,
    StateSpace,
    ReducedModel,
    HarmonicBalance,
    HbSensitivity,
    HbHessian,
)
from .differentiable import (
    ParamFunction,
    TransientFunction,
    DcFunction,
    AcFunction,
    SpFunction,
    HbFunction,
    PzFunction,
)
from .rf import SParams, read_touchstone, write_touchstone, s_to_y, y_to_s, fit_verilog_a
from . import interop
from .warnings import (
    SaneWarning,
    SaneConvergenceWarning,
    SaneNumericalWarning,
)


# CONVENIENCE ===========================================================================

def parse(netlist):
    """The :class:`Circuit` of a SPICE-like netlist (:meth:`Circuit.parse`)."""
    return Circuit.parse(netlist)


from .netlist import reduced_netlist


def set_parallelism(threads):
    """Set the number of threads for the parallel work: the sweeps (AC and
    noise over frequency, harmonic-balance device sampling) and, in every
    solve (operating point, transient, harmonic balance), the device
    instances of each evaluation. Results do not depend on it. The linear
    solves themselves are sequential.

    Takes effect before the first analysis; the default is 4 (or the
    ``SANE_THREADS`` environment variable).

    Parameters
    ----------
    threads : int
        ``n`` for ``n`` threads, ``1`` for none in parallel, ``0`` for the
        default
    """
    _core.set_parallelism(threads)


def set_log_level(level="info"):
    """Enable the engine's native logging (fastsim style) at the given level.

    Turns on stdout/stderr progress logging for long-running solves: the DC
    homotopy fallbacks (gmin / source stepping), the transient progress bar with
    ETA, and the AC / sweep / optimize progress. Lines are
    formatted ``HH:MM:SS - LEVEL - message`` (INFO/WARNING to stdout, ERROR to
    stderr), mirroring Python's :mod:`logging`.

    Parameters
    ----------
    level : str
        ``"debug"``, ``"info"``, ``"warning"``, ``"error"`` or ``"off"``
        (case-insensitive). Defaults to ``"info"``.
    """
    _core.set_log_level(level)


def profile_begin():
    """Begin collecting the engine's internal per-stage timings.

    Captures every instrumented stage that flows through the native logger
    (``log_stage!`` / ``time_stage!`` / ``log::scope``) -- e.g. ``dc/newton``,
    ``ac/eval_b``, ``tran/irk_step``, and the extract/compile phases -- into a
    process-global sink, independent of the log level. Pair with
    :func:`profile_take`::

        sane.profile_begin()
        op = model.at().operating_point()
        breakdown = sane.profile_take()   # [(stage, ms), ...]
    """
    _core.profile_begin()


def profile_take():
    """Stop collecting and return the ``(stage, milliseconds)`` breakdown.

    Summed per distinct stage name, in first-seen order (a stage hit many times,
    e.g. ``dc/newton`` across continuation steps, collapses to one total row).
    Empty if collection was not active. See :func:`profile_begin`.
    """
    return list(_core.profile_take())


# VERSION ===============================================================================

try:
    from importlib.metadata import version as _version

    __version__ = _version("sane")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = [
    "Circuit",
    "Waveform",
    "Model",
    "Point",
    "ParamGroup",
    "OperatingPoint",
    "Regularization",
    "Sensitivity",
    "Hessian",
    "Gradient",
    "DcSweep",
    "Trajectory",
    "TrajectorySensitivity",
    "AcResponse",
    "AcSensitivity",
    "AcHessian",
    "SParameters",
    "SpSensitivity",
    "NoiseSpectrum",
    "NoiseSensitivity",
    "Poles",
    "Zeros",
    "RootSensitivity",
    "StateSpace",
    "ReducedModel",
    "HarmonicBalance",
    "HbSensitivity",
    "HbHessian",
    # differentiable functions + optimizer interop
    "ParamFunction",
    "TransientFunction",
    "DcFunction",
    "AcFunction",
    "SpFunction",
    "HbFunction",
    "PzFunction",
    "interop",
    # RF data
    "SParams",
    "read_touchstone",
    "write_touchstone",
    "s_to_y",
    "y_to_s",
    "fit_verilog_a",
    "parse",
    "set_parallelism",
    "set_log_level",
    "profile_begin",
    "profile_take",
    "reduced_netlist",
    # diagnostics
    "SaneWarning",
    "SaneConvergenceWarning",
    "SaneNumericalWarning",
    "_core",
]
