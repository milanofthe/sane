"""Hierarchical parameter API: read/mutate leaves as attributes of the DAE,
flat dotted indexing, dict behaviour, reset, and that mutation drives the solve.

    python -m pytest crates/py/tests/test_params.py
"""
import math

import pytest

import sane
from sane.model import ParamNode

# A subcircuit (rc divider) instantiated as X1, plus a nested subcircuit, so the
# parameter tree has depth: V1, X1.R1, X1.R2, XN.XI.R3.
NET = (
    ".subckt rcdiv a b\n"
    "R1 a mid 1k\n"
    "R2 mid b 2k\n"
    ".ends\n"
    ".subckt inner a b\n"
    "R3 a b 1k\n"
    ".ends\n"
    ".subckt outer p q\n"
    "XI p q inner\n"
    ".ends\n"
    "V1 in 0 5\n"
    "X1 in 0 rcdiv\n"
    "XN in 0 outer\n"
    ".end\n"
)


def _dae():
    return sane.parse(NET).extract()


def test_read_leaves_hierarchically():
    dae = _dae()
    assert dae.X1.R1 == pytest.approx(1e3)
    assert dae.X1.R2 == pytest.approx(2e3)
    assert dae.V1 == pytest.approx(5.0)
    # Nested group chains to a leaf.
    assert dae.XN.XI.R3 == pytest.approx(1e3)
    # A group is a navigable node, not a value.
    assert isinstance(dae.X1, ParamNode)
    assert isinstance(dae.XN.XI, ParamNode)


def test_leaf_is_a_plain_float():
    dae = _dae()
    r = dae.X1.R1
    assert isinstance(r, float)
    assert r * 2 == pytest.approx(2e3)


def test_mutation_drives_the_solve():
    dae = _dae()
    # Divider mid node: v = 5 * R2/(R1+R2) = 5*2/3.
    op0 = dae.operating_point()
    assert op0["X1.mid"] == pytest.approx(5 * 2e3 / 3e3, rel=1e-6)
    # Make the legs equal -> mid at half supply.
    dae.X1.R1 = 2e3
    assert dae.X1.R1 == pytest.approx(2e3)
    op1 = dae.operating_point()
    assert op1["X1.mid"] == pytest.approx(2.5, rel=1e-6)
    # reset() restores the netlist default and the original solution.
    dae.reset()
    assert dae.X1.R1 == pytest.approx(1e3)
    op2 = dae.operating_point()
    assert op2["X1.mid"] == pytest.approx(5 * 2e3 / 3e3, rel=1e-6)


def test_dotted_indexing():
    dae = _dae()
    assert dae["X1.R2"] == pytest.approx(2e3)
    dae["X1.R2"] = 3e3
    assert dae.X1.R2 == pytest.approx(3e3)
    assert dae["XN.XI.R3"] == pytest.approx(1e3)
    # node-level indexing too
    assert dae.X1["R1"] == pytest.approx(1e3)
    dae.X1["R1"] = 9e3
    assert dae.X1.R1 == pytest.approx(9e3)


def test_dict_behaviour():
    dae = _dae()
    assert "X1.R1" in dae
    assert "nope.R9" not in dae
    assert len(dae) == len(dae.params)
    assert set(dae.params) == set(iter(dae))
    dae.update({"X1.R1": 4e3}, V1=10.0)
    assert dae.X1.R1 == pytest.approx(4e3) and dae.V1 == pytest.approx(10.0)


def test_tab_completion_segments():
    dae = _dae()
    top = set(dir(dae))
    assert {"X1", "XN", "V1"} <= top
    assert "R3" in dir(dae.XN.XI)


def test_errors():
    dae = _dae()
    with pytest.raises(AttributeError):
        _ = dae.nope
    with pytest.raises(AttributeError):
        dae.X1 = 5.0  # a group is not a leaf
    with pytest.raises(KeyError):
        _ = dae["nope.R9"]
    with pytest.raises(KeyError):
        dae.update({"bogus": 1.0})


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK")
