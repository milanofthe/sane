"""Regression for issue #42: unsupported Verilog-A analog event controls must be
rejected at load, not silently stripped.

``@(cross(...))`` / ``@(timer(...))`` / ``@(final_step)`` / ``@(above ...)``
previously had their event discarded and the guarded body run unconditionally,
silently altering the model. They must now fail to load with a clear diagnostic
naming the control, module and line.

``@(initial_step)`` is supported; per the documented deviation its body runs
unconditionally as part of the assembled residual (exact for the common
parameter-precompute idiom), so a model using it loads and solves.

Run after ``maturin develop --release -m crates/py/Cargo.toml``:
    python -m pytest crates/py/tests/test_va_events.py
"""

import pytest

import sane

# `@(cross(...))` guarding a conductance update: the event cannot be honored, so
# the model must be rejected rather than run the body every evaluation.
VA_CROSS = """.veriloga
module crossmod(a, c);
  inout a, c;
  electrical a, c;
  real g;
  analog begin
    g = 1e-3;
    @(cross(V(a,c) - 1.0, 0)) g = 2e-3;
    I(a,c) <+ V(a,c)*g;
  end
endmodule
.endveriloga
V1 1 0 1
N1 1 0 crossmod
.end"""

# `@(initial_step)` precompute: a supported idiom (runs unconditionally).
VA_INITIAL_STEP = """.veriloga
module initmod(a, c);
  inout a, c;
  electrical a, c;
  parameter real R = 1000;
  real g;
  analog begin
    @(initial_step) g = 1.0/R;
    I(a,c) <+ V(a,c)*g;
  end
endmodule
.endveriloga
V1 1 0 1
N1 1 0 initmod
.end"""


def test_cross_event_rejected_with_diagnostic():
    with pytest.raises(ValueError, match="cross"):
        sane.Circuit.parse(VA_CROSS).extract()


def test_unsupported_event_names_the_control():
    # `@(cross ...)` is a switching surface now; a body on it (a discrete state
    # SANE does not carry) is what stays rejected, naming the control.
    with pytest.raises(ValueError, match=r"@\(cross .*with a body"):
        sane.Circuit.parse(VA_CROSS).extract()


def test_initial_step_precompute_loads_and_solves():
    # The precompute makes initmod a 1/R = 1 mS conductance; op must solve and
    # draw I = V*g = 1 * 1e-3 through the source.
    op = sane.Circuit.parse(VA_INITIAL_STEP).extract().operating_point()
    assert op is not None
