"""Regression for issue #54: correctness-affecting DC diagnostics must be both
machine-readable and unconditionally catchable, independent of the native log
level (which is disabled by default).

- A gmin-regularized operating point returns converged=True but is physically
  suspect: it must carry a ``regularized_at_gmin`` flag AND raise a
  ``SaneConvergenceWarning``. Two things earn the flag -- the solve never
  reached the gmin floor, or it reached the floor and the floor is what sets an
  unknown's value -- and the second one converges cleanly, so only an explicit
  check finds it.
- An out-of-range Verilog-A device parameter must raise a
  ``SaneConvergenceWarning`` at parse/extract instead of only logging on the
  default-disabled channel.

Run after ``maturin develop -m crates/py/Cargo.toml``:
    python -m pytest crates/py/tests/test_convergence_warnings.py
"""

import warnings

import pytest

import sane
from sane import SaneConvergenceWarning

# A current source into a resistance three decades above the gmin shunt: the
# circuit says 1 GV, the shunt says 999 kV, and the shunt wins. Newton converges
# on the first iterate, so nothing but the dominance check notices.
REGULARIZED = "I1 0 1 1u\nR1 1 0 1e15\n.end"
HEALTHY = "V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end"
# Nodes that carry no DC current at all: an island behind a capacitor rests at
# zero volts, where the shunt current is zero too. Being open is not the same as
# being wrong, and neither may raise the flag.
FLOATING = "V1 in 0 5\nR1 in a 1k\nC1 a b 1u\nR2 b 0 1e6\n.end"
ISLAND = "V1 in 0 5\nR1 in a 1k\nC1 a b 1u\n.end"

# A Verilog-A resistor declaring R in (0, inf), instantiated with R = -5.
VA_OUT_OF_RANGE = """.veriloga
module vres(a, c);
  inout a, c;
  electrical a, c;
  parameter real R = 1000 from (0:inf);
  analog I(a,c) <+ V(a,c)/R;
endmodule
.endveriloga
V1 1 0 1
N1 1 0 vres R=-5
.end"""


def test_gmin_regularized_warns_and_flags():
    dae = sane.Circuit.parse(REGULARIZED).extract()
    with pytest.warns(SaneConvergenceWarning, match="gmin-regularized"):
        op = dae.operating_point()
    assert op.regularized_at_gmin is not None, "the flag must record the gmin in question"
    assert op.regularized_at_gmin > 0


def test_gmin_dominance_names_the_node_and_its_shift():
    """The warning has to be actionable: which node, and how wrong it is."""
    dae = sane.Circuit.parse(REGULARIZED).extract()
    with pytest.warns(SaneConvergenceWarning, match=r"shift node '1' by 99\.9%"):
        dae.operating_point()
    idx, shift = dae._d.gmin_dominance()
    assert idx == 0
    # g/(g + 1/R): the exact first-order error gmin imprints on this node
    assert shift == pytest.approx(0.999, abs=1e-3)


@pytest.mark.parametrize("deck", [HEALTHY, FLOATING, ISLAND])
def test_currentless_nodes_are_not_flagged(deck):
    """The obvious false positive: a node the shunt "dominates" only because
    nothing flows there at all."""
    dae = sane.Circuit.parse(deck).extract()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SaneConvergenceWarning)
        op = dae.operating_point()
    assert op.regularized_at_gmin is None
    assert dae._d.gmin_dominance() is None


@pytest.mark.parametrize(
    "r,flagged",
    [(1e9, False), (1e11, False), (1e13, True), (1e15, True)],
    ids=["1e9", "1e11", "1e13", "1e15"],
)
def test_the_threshold_tracks_the_error(r, flagged):
    """1 uA into R, shunted by gmin = 1e-12: the reported voltage is off by the
    ratio of the two conductances. 1e9 errs by 0.1%, 1e11 by 9%, 1e13 by 10x,
    1e15 by 1000x -- the flag has to turn on where the error does."""
    dae = sane.Circuit.parse(f"I1 0 1 1u\nR1 1 0 {r:g}\n.end").extract()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SaneConvergenceWarning)
        op = dae.operating_point()
    assert (op.regularized_at_gmin is not None) is flagged


def test_healthy_op_no_warning_and_flag_none():
    dae = sane.Circuit.parse(HEALTHY).extract()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SaneConvergenceWarning)
        op = dae.operating_point()
    assert op.regularized_at_gmin is None


def test_gmin_warning_is_unconditional_of_log_level():
    # The warning must fire even with the native logger disabled (the default).
    sane.set_log_level("off")
    dae = sane.Circuit.parse(REGULARIZED).extract()
    with pytest.warns(SaneConvergenceWarning):
        dae.operating_point()


def test_va_out_of_range_param_warns():
    with pytest.warns(SaneConvergenceWarning, match="range"):
        sane.Circuit.parse(VA_OUT_OF_RANGE).extract()
