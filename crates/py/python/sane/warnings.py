"""SANE's Python-level warning categories and the bridge that raises them.

Correctness-affecting diagnostics from the engine must be *unconditional and
catchable*: a result that is converged-but-regularized, a singular AC frequency,
or an out-of-range device parameter has to reach the caller regardless of the
native log level (``sane.set_log_level``), which is disabled by default. These
categories ride the standard :mod:`warnings` machinery, so a script can catch,
filter, or escalate them with :func:`warnings.catch_warnings` /
:func:`warnings.simplefilter` (and :func:`pytest.warns` in tests) exactly like
any other :class:`UserWarning`.

The pattern mirrors rapidmom's ``RapidmomConvergenceWarning`` bridge.
"""

import warnings as _warnings

__all__ = [
    "SaneWarning",
    "SaneConvergenceWarning",
    "SaneNumericalWarning",
]


class SaneWarning(UserWarning):
    """Base class for every SANE diagnostic surfaced to the Python layer."""


class SaneConvergenceWarning(SaneWarning):
    """A solve returned a usable-but-suspect result the caller should know about.

    Emitted when the engine reports a converged operating point that is actually
    regularized (held only by a raised gmin shunt), or a device parameter that
    was clamped to its valid range. The returned data is still usable but may be
    physically inaccurate.
    """


class SaneNumericalWarning(SaneWarning):
    """A numerical breakdown left part of a result undefined (NaN).

    Emitted when a small-signal AC system ``G + jwC`` is singular at one or more
    frequencies (a floating subnet, an ideal VCVS/inductor loop, or a bad DC
    operating point): those frequencies come back as NaN rather than a silent
    ``0.0`` that would read as a flat ~-600 dB response.
    """


def warn(message, category=SaneWarning, stacklevel=2):
    """Emit ``message`` as ``category`` on the standard :mod:`warnings` channel,
    unconditionally (independent of ``sane.set_log_level``). ``stacklevel`` points
    the warning at the caller of the analysis method, not this helper."""
    _warnings.warn(message, category, stacklevel=stacklevel + 1)
