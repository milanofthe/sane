"""Regression for issue #41: Verilog-A dollar-diagnostic system tasks must not be
silent no-ops.

- ``$error``/``$fatal`` reached unconditionally (or under a parameter guard that
  folds true for the instance) is a hard load failure carrying the message.
- ``$warning`` and an unresolved ``\\`include`` surface as catchable
  ``SaneConvergenceWarning`` at parse/extract, independent of the log level.

Run after ``maturin develop --release -m crates/py/Cargo.toml``:
    python -m pytest crates/py/tests/test_va_diagnostics.py
"""

import warnings

import pytest

import sane
from sane import SaneConvergenceWarning

# `$error` under an always-true parameter guard: the author flags this config
# invalid, so the model must refuse to load with the message + module + line.
VA_ERROR_GUARD = """.veriloga
module errmod(a, c);
  inout a, c;
  electrical a, c;
  parameter real bad = 1;
  analog begin
    if (bad > 0) $error("bad must be non-positive");
    I(a,c) <+ V(a,c);
  end
endmodule
.endveriloga
V1 1 0 1
N1 1 0 errmod
.end"""

# `$warning` fires unconditionally: a catchable warning, not a dropped no-op.
VA_WARNING = """.veriloga
module warnmod(a, c);
  inout a, c;
  electrical a, c;
  parameter real R = 1000;
  analog begin
    $warning("warnmod is a preview model");
    I(a,c) <+ V(a,c)/R;
  end
endmodule
.endveriloga
V1 1 0 1
N1 1 0 warnmod
.end"""

# An unresolved `include (a simulator-private header) must warn, not vanish.
VA_UNRESOLVED_INCLUDE = """.veriloga
`include "no_such_header.vams"
module incmod(a, c);
  inout a, c;
  electrical a, c;
  parameter real R = 1000;
  analog I(a,c) <+ V(a,c)/R;
endmodule
.endveriloga
V1 1 0 1
N1 1 0 incmod
.end"""


def test_error_in_true_guard_fails_to_load():
    with pytest.raises(ValueError, match="bad must be non-positive"):
        sane.Circuit.parse(VA_ERROR_GUARD).extract()


def test_warning_surfaces_as_convergence_warning():
    with pytest.warns(SaneConvergenceWarning, match="preview model"):
        sane.Circuit.parse(VA_WARNING).extract()


def test_unresolved_include_warns():
    with pytest.warns(SaneConvergenceWarning, match="include"):
        sane.Circuit.parse(VA_UNRESOLVED_INCLUDE).extract()


def test_healthy_va_model_does_not_warn():
    healthy = VA_WARNING.replace('    $warning("warnmod is a preview model");\n', "")
    with warnings.catch_warnings():
        warnings.simplefilter("error", SaneConvergenceWarning)
        sane.Circuit.parse(healthy).extract()
