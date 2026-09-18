#########################################################################################
##
##                                      SANE
##                                  (__init__.py)
##
##            Symbolic + numeric circuit analysis: exact component sensitivity
##         (DC / transient / AC / pole-zero / noise / harmonic balance) and
##          nonlinear DAE extraction, on a hash-consed symbolic engine in Rust.
##
#########################################################################################
"""SANE: symbolic and numeric circuit analysis.

SANE extracts the symbolic differential-algebraic system ``F(x, x', t) = 0``
from a circuit and analyzes it exactly: DC operating point, transient response,
small-signal AC transfer, poles / zeros, noise PSD and harmonic balance, and --
its distinguishing capability -- exact component sensitivity (which component
matters how much, to first and second order, hierarchically) for every one of
those analyses, all via automatic differentiation of the symbolic DAG.

Every analysis returns a result object with the same derivative workflow:
``result.sensitivity(...)`` is the exact gradient over **all** parameters (one
adjoint solve -- the parameter Jacobian is sparse, so the full gradient is
cheap); rank it to find the influential knobs, then ``result.hessian(...,
wrt=knobs)`` is the exact sparse second-order-adjoint Hessian over that subset.
No finite differences anywhere -- the gradients and Hessians are autodiff of the
one symbolic DAG.

Circuits come from SPICE netlists or are built programmatically; Verilog-A
models drop into a deck (a ``.veriloga ... .endveriloga`` block placed with an
``N`` element) and are compiled natively onto the same symbolic DAG, so they
run through every analysis like a built-in device.

The ergonomic surface is two classes:

- :class:`~sane.circuit.Circuit` -- build from a netlist or programmatically
- :class:`~sane.model.Model` -- the circuit as an analyzable symbolic graph

Example
-------

.. code-block:: python

    import numpy as np
    import sane

    model = sane.Circuit.parse('''
        V1 in 0 5
        R1 in out 1k
        C1 out 0 1u
    ''').extract()                       # or sane.Model.from_netlist(...)

    op = model.operating_point()         # DC bias, labeled by node
    print(op["out"])

    ss = model.small_signal("V1", "out") # linearize at the bias
    print(ss.poles())                    # the RC pole

    traj = model.transient(np.linspace(0, 5e-3, 200))
    print(traj["out"][-1])

SANE's hash-consed symbolic core is also exposed directly: :func:`symbols`
builds :class:`~sane.symbolic.Expr` that support the Python operators,
``.diff`` / ``.simplify`` / ``.eval`` and :func:`compile_tape`. The extracted
:class:`~sane.model.Model` shares that context, so ``model.residuals``,
``model.system_matrix()`` and ``model.transfer_function(...)`` come back as
manipulable expressions.

The raw, positional compiled API remains available as :mod:`sane._core`.
"""

# IMPORTS ===============================================================================

from . import _core
from .circuit import Circuit, GROUND_ALIASES
# The one analyzable object: a circuit as a symbolic graph you run analyses on and
# transform. The orchestration, parameter store and solved state live in Rust
# (``sane_analysis::Model``, exposed as ``_core.Model``); this Python ``Model`` is a
# thin, name-ergonomic wrapper. The identical API is available natively in Rust for
# embedding without Python.
from .model import Model
from .analysis import (
    OperatingPoint,
    Trajectory,
    Sensitivity,
    AcResponse,
    SmallSignal,
    NoiseSpectrum,
    StateSpace,
    TempSweep,
    ReducedModel,
    HarmonicBalance,
)
from .differentiable import ParamFunction, TransientFunction, DcFunction, AcFunction, HbFunction, PzFunction, SpFunction
from .rf import SParams, read_touchstone, write_touchstone, s_to_y, y_to_s, fit_verilog_a
from . import interop
from .symbolic import (
    Context,
    Expr,
    Tape,
    symbols,
    jacobian,
    sparsity,
    compile_tape,
)
from .warnings import (
    SaneWarning,
    SaneConvergenceWarning,
    SaneNumericalWarning,
)


# CONVENIENCE ===========================================================================

def parse(netlist):
    """Parse a SPICE-like netlist into a :class:`~sane.circuit.Circuit`.

    Shorthand for :meth:`sane.circuit.Circuit.parse`.

    Parameters
    ----------
    netlist : str
        the netlist text; a trailing ``.end`` is optional

    Returns
    -------
    sane.circuit.Circuit
    """
    return Circuit.parse(netlist)


from .netlist import reduced_netlist


def set_parallelism(threads):
    """Set the sparse-LU parallelism: ``1`` sequential, ``0`` all rayon threads,
    ``n`` exactly ``n`` threads.

    The solver runs sequentially by default. Measured, faer's parallel sparse LU
    is neutral-to-harmful on the narrow elimination trees of 2D parasitic meshes,
    so this is an opt-in knob -- most useful for pinning ``1`` (sequential)
    inside an outer parallel sweep (frequency / tolerance), which is where real
    throughput parallelism lives.

    Parameters
    ----------
    threads : int
        ``1`` sequential, ``0`` all available threads, ``n`` for ``n`` threads
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
        op = model.operating_point()
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
    "Model",
    "OperatingPoint",
    "Trajectory",
    "Sensitivity",
    "AcResponse",
    "SmallSignal",
    "NoiseSpectrum",
    "StateSpace",
    "TempSweep",
    "ReducedModel",
    "HarmonicBalance",
    "GROUND_ALIASES",
    # differentiable functions + optimizer interop
    "ParamFunction",
    "TransientFunction",
    "DcFunction",
    "AcFunction",
    "HbFunction",
    "PzFunction",
    "interop",
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
    # symbolic engine
    "Context",
    "Expr",
    "Tape",
    "symbols",
    "jacobian",
    "sparsity",
    "compile_tape",
    "_core",
]
