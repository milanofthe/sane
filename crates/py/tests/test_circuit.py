"""A circuit built by name in Python is the circuit the netlist states."""

import numpy as np
import pytest

import sane


def test_a_built_divider_is_the_parsed_one():
    c = sane.Circuit()
    c.voltage_source("V1", "in", "0", 5.0).resistor("R1", "in", "out", 1e3)
    c.resistor("R2", "out", "gnd", 3e3)
    built = sane.Model(c)
    parsed = sane.Model.from_netlist("V1 in 0 5\nR1 in out 1k\nR2 out 0 3k")
    assert built.node_names == parsed.node_names
    assert built.at().operating_point()["out"] == pytest.approx(3.75, rel=1e-8)
    assert built.at(R1=3e3).operating_point()["out"] == pytest.approx(2.5, rel=1e-8)
    assert c.values["R2"] == 3e3
    assert [e["nodes"] for e in c.elements][1] == ("in", "out")


def test_devices_take_their_parameters_by_keyword():
    c = sane.Circuit()
    c.voltage_source("V1", "in", "0", 1.0).resistor("R1", "in", "a", 1e3)
    c.diode("D1", "a", "0", Is=2e-14, N=1.1)
    parsed = sane.Model.from_netlist("V1 in 0 1\nR1 in a 1k\nD1 a 0 dm\n.model dm D(Is=2e-14 N=1.1)")
    got = sane.Model(c).at().operating_point()["a"]
    assert got == pytest.approx(parsed.at().operating_point()["a"], rel=1e-9)
    with pytest.raises(ValueError):
        sane.Circuit().diode("D1", "a", "0", bogus=1.0)


def test_a_subcircuit_is_placed_by_instance():
    half = sane.Circuit.subckt("half", ["in", "out"])
    half.resistor("R1", "in", "mid", 1e3).resistor("R2", "mid", "out", 1e3)
    half.resistor("R3", "out", "0", 2e3)
    c = sane.Circuit()
    c.voltage_source("V1", "a", "0", 8.0)
    c.instance("X1", half, ["a", "b"]).instance("X2", half, ["b", "c"])
    op = sane.Model(c).at().operating_point()
    assert op["b"] == pytest.approx(3.2, rel=1e-8)
    assert op["X2.mid"] == pytest.approx(2.4, rel=1e-8)
    with pytest.raises(ValueError):
        sane.Circuit().instance("X1", half, ["a"])


def test_a_waveform_drives_the_transient():
    c = sane.Circuit()
    c.voltage_source("V1", "in", "0", sane.Waveform.sin(0.5, 1.0, 1e3))
    c.resistor("R1", "in", "out", 1e3).capacitor("C1", "out", "0", 1e-7)
    parsed = sane.Model.from_netlist("V1 in 0 SIN(0.5 1 1k)\nR1 in out 1k\nC1 out 0 100n")
    t = np.linspace(0.0, 2e-3, 41)
    got = sane.Model(c).at().transient(t).signal("out")
    want = parsed.at().transient(t).signal("out")
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


def test_a_model_hands_its_circuit_back():
    c = sane.Circuit()
    c.voltage_source("V1", "in", "0", 2.0).resistor("R1", "in", "out", 1e3)
    first = sane.Model(c)
    more = first.circuit
    more.resistor("R2", "out", "0", 1e3)
    assert sane.Model(more).at().operating_point()["out"] == pytest.approx(1.0, rel=1e-8)
    # the first model's circuit is untouched
    assert first.at().operating_point()["out"] == pytest.approx(2.0, rel=1e-8)
    assert len(first.circuit.elements) == 2
