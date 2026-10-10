"""Folding parameters: `fold` names the parameters that become constants,
`keep` the only ones that stay symbolic. The solution does not change, the
parameter set does.

    python -m pytest crates/py/tests/test_fold.py
"""
import pytest

import sane

DECK = "V1 in 0 2\nR1 in out 1k\nR2 out 0 2k\nD1 out 0 DM\n.model DM D(Is=1e-14 N=1)\n.end"


def test_keep_is_the_complement_of_fold():
    m = sane.Model.from_netlist(DECK)
    kept = m.keep("R1", "V1")
    assert sorted(kept.params) == ["R1", "V1"]
    folded = m.fold([p for p in m.params if p not in ("R1", "V1")])
    assert sorted(folded.params) == sorted(kept.params)
    out = m.at().operating_point()["out"]
    assert kept.at().operating_point()["out"] == pytest.approx(out, rel=1e-12)


def test_a_card_parameter_is_kept_for_every_instance_of_the_card():
    deck = DECK.replace("D1 out 0 DM", "D1 out 0 DM\nD2 out 0 DM")
    m = sane.Model.from_netlist(deck).keep("dm.Is", "V1")
    assert sorted(m.params) == ["V1", "dm.Is"]
    s = m.at().operating_point().sensitivity("out")
    assert s.get("out", "dm.Is") != 0.0
