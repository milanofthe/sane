"""Regression for issue #43: two Verilog-A lowering semantics gaps.

1. ``idt(u, ic)`` parses its initial condition but SANE's DC/transient-shared DAE
   cannot apply it (the transient seeds from the DC point). A non-zero ``ic`` must
   raise a catchable ``SaneConvergenceWarning`` instead of being silently dropped.

2. A runtime-guarded divide inside a conditional (``if (x != 0) y = 1/x;``) is
   lowered eagerly in both arms; at ``x = 0`` the not-taken ``1/x`` must not inject
   a NaN/Inf. Both the residual AND the state Jacobian must stay finite at ``x = 0``.

Run after ``maturin develop --release -m crates/py/Cargo.toml``:
    python -m pytest crates/py/tests/test_va_lowering.py
"""

import math

import pytest

import sane
from sane import SaneConvergenceWarning

# idt with a non-default (non-zero) initial condition.
VA_IDT_IC = """.veriloga
module idtdev(a, c);
  inout a, c;
  electrical a, c;
  parameter real G = 1e-3;
  analog begin
    I(a,c) <+ G*V(a,c) + idt(V(a,c), 0.5);
  end
endmodule
.endveriloga
V1 1 0 1
N1 1 0 idtdev
.end"""

# A runtime-guarded 1/x. Forced to x = V(a,c) = 0 at the operating point.
VA_GUARDED_DIV = """.veriloga
module divdev(a, c);
  inout a, c;
  electrical a, c;
  real y;
  analog begin
    if (V(a,c) != 0.0)
      y = 1.0/V(a,c);
    else
      y = 0.0;
    I(a,c) <+ V(a,c)*1e-3 + y;
  end
endmodule
.endveriloga
V1 1 0 0
N1 1 0 divdev
.end"""


def test_idt_constant_ic_does_not_warn():
    # A constant ic is routed as the state's DC Newton seed (nodeset
    # semantics) rather than dropped, so it no longer warns.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", SaneConvergenceWarning)
        sane.Circuit.parse(VA_IDT_IC).extract()


def test_idt_nonconstant_ic_warns():
    # A non-constant ic cannot seed a numeric solve: surfaced as a warning.
    varying = VA_IDT_IC.replace("idt(V(a,c), 0.5)", "idt(V(a,c), V(a,c))")
    with pytest.warns(SaneConvergenceWarning, match="not a constant"):
        sane.Circuit.parse(varying).extract()


def test_idt_zero_ic_does_not_warn():
    zero_ic = VA_IDT_IC.replace("idt(V(a,c), 0.5)", "idt(V(a,c), 0.0)")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", SaneConvergenceWarning)
        sane.Circuit.parse(zero_ic).extract()


def _eval_at_zero(expr):
    """Evaluate a residual/Jacobian entry with every free symbol bound to 0
    (x = 0, xdot = 0, t = 0). ``Expr.eval`` raises on NaN; an Inf comes back and
    is caught by ``math.isfinite``."""
    binding = {name: 0.0 for name in expr.free_symbols}
    return expr.eval(**binding)


def test_guarded_divide_finite_residual_and_jacobian_at_zero():
    dae = sane.Circuit.parse(VA_GUARDED_DIV).extract()

    # Residual F(x=0) and the full state Jacobian dF/dx at x=0 must be finite.
    for f in dae._d.residuals():
        assert math.isfinite(_eval_at_zero(f)), "residual is non-finite at x=0"
    for row in dae._d.jacobian_x_symbolic():
        for entry in row:
            assert math.isfinite(_eval_at_zero(entry)), "Jacobian entry non-finite at x=0"

    # And the DC solve (whose first Newton evaluation is at the x=0 guess) must
    # converge to a finite solution vector.
    op = dae.operating_point()
    assert all(math.isfinite(v) for v in op.vector)
