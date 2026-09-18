"""Index-2 topologies: capacitor/voltage-source loops and inductor/current-source
cutsets.

Charge-oriented MNA is index 1 for almost every deck, but two topologies raise
it to index 2 -- and both are ordinary things to draw. A decoupling capacitor
straight across an ideal supply is one of them.

What the engine promises here:

* a CV loop **integrates** (its branch current jumps to ``C*dv/dt`` at the first
  step, which the Newton-step clamp used to throttle into a step-size underflow),
* every index-2 deck is **detected and named**, so the caller can reach for
  ``dt_max`` rather than trust a coarse trace,
* an LI cutset -- the dual topology -- integrates as well, and both are exact
  once the caller bounds the step with ``dt_max``.
"""

import warnings

import numpy as np
import pytest

import sane

T = np.linspace(0, 0.1, 201)
# every deck below is driven by sin(100 t), so v(n1) has a closed form wherever
# the source pins it directly
REF = np.sin(100 * T)

CV_LOOP = """V1 n1 0 SIN(0 1 15.9155)
R1 n1 0 1
R2 n2 0 1
C1 n1 n2 1
C2 n2 0 1
.end
"""
CAP_ACROSS_SOURCE = """V1 n1 0 SIN(0 1 15.9155)
R1 n1 0 1
C1 n1 0 1
.end
"""
LI_CUTSET = """I1 0 n1 SIN(0 1 15.9155)
L1 n1 n2 1
R2 n2 0 1
.end
"""
INDEX1_REFERENCE = """V1 n1 0 SIN(0 1 15.9155)
R1 n1 0 1
R2 n2 0 1
C1 n1 nx 1
Rx nx n2 1e-3
C2 n2 0 1
.end
"""


def build(deck):
    """Model plus the index-2 warning it raised (None when it raised none)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sane.Model.from_netlist(deck)
        msg = next((str(w.message) for w in caught if "index-2" in str(w.message)), None)
    return model, msg


@pytest.mark.parametrize(
    "deck,source,storage",
    [(CV_LOOP, "V1", {"C1", "C2"}), (CAP_ACROSS_SOURCE, "V1", {"C1"})],
)
def test_capacitor_source_loops_are_detected_and_named(deck, source, storage):
    model, msg = build(deck)
    loops = model._d.index2()["cv_loops"]
    assert len(loops) == 1, loops
    assert loops[0][0] == source
    assert set(loops[0][1:]) == storage
    assert msg is not None and source in msg


def test_inductor_source_cutsets_are_detected_and_named():
    model, msg = build(LI_CUTSET)
    cutsets = model._d.index2()["li_cutsets"]
    assert cutsets == [["I1", "L1"]]
    assert msg is not None and "cutset" in msg


@pytest.mark.parametrize("deck", [INDEX1_REFERENCE, "V1 n1 0 SIN(0 1 15.9155)\nR1 n1 n2 1\nC1 n2 0 1\n.end\n"])
def test_ordinary_decks_are_index_1_and_silent(deck):
    model, msg = build(deck)
    report = model._d.index2()
    assert report == {"cv_loops": [], "li_cutsets": []}
    assert msg is None


@pytest.mark.parametrize("deck", [CV_LOOP, CAP_ACROSS_SOURCE])
def test_a_cv_loop_integrates(deck):
    """The regression this exists for: the branch current of a loop capacitor
    steps to `C*dv/dt` -- tens of amperes -- at once. A per-unknown clamp of 1
    (a *voltage* heuristic) could not deliver that within the iteration budget,
    and shrinking the step only made the required current larger: underflow."""
    model, _ = build(deck)
    traj = model.transient(T)
    v1 = np.asarray(traj.to_dict()["v1"]).ravel()
    # It runs and stays finite -- that is the regression. Accuracy is NOT
    # asserted here: with the step left free the controller has no truncation
    # error to see on the pinned branch and strides far, which is exactly what
    # the warning is about (see the dt_max test for the accurate run).
    assert v1.shape == T.shape
    assert np.all(np.isfinite(v1))
    assert np.max(np.abs(v1)) < 10 * np.max(np.abs(REF))


@pytest.mark.parametrize("deck", [CV_LOOP, CAP_ACROSS_SOURCE])
def test_dt_max_restores_index_2_accuracy(deck):
    """Bounding the step is the caller's lever (the engine only warns): with it,
    the pinned node is exact again."""
    model, _ = build(deck)
    v1 = np.asarray(model.transient(T, dt_max=5e-4).to_dict()["v1"]).ravel()
    assert np.max(np.abs(v1 - REF)) < 1e-4


def test_an_li_cutset_integrates():
    """The dual of the CV loop: the inductor current is pinned by the source, so
    the node voltage follows `L*di/dt`. It integrates for the same reason -- no
    solver-side step clamp stands in the way -- and `dt_max` makes it exact."""
    model, _ = build(LI_CUTSET)
    v2 = np.asarray(model.transient(T, dt_max=5e-4).to_dict()["v2"]).ravel()
    # KCL pins i(L1) = I1(t), so v(n2) = R2 * i = sin(100 t)
    assert np.max(np.abs(v2 - REF)) < 1e-4


def test_breaking_the_loop_removes_the_diagnosis():
    """A series resistance in the loop is the parasitic that exists in reality;
    with it the deck is index 1 and integrates to full accuracy."""
    model, msg = build(INDEX1_REFERENCE)
    assert msg is None
    v1 = np.asarray(model.transient(T).to_dict()["v1"]).ravel()
    assert np.max(np.abs(v1 - REF)) < 1e-3
