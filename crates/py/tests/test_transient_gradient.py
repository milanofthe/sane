"""Transient gradients through the Python API: the forward sensitivities'
VJP against finite differences, the differentiable TransientFunction, and
the JAX/torch interop wrappers (skipped when the framework is not
installed)."""

import numpy as np
import pytest

import sane

RC = "V1 in 0 0 SIN(0 1 1k)\nR1 in out 1k\nC1 out 0 100n"
CLIPPER = (
    ".model Dmod D(Is=1e-12 N=1)\n"
    "V1 in 0 0.3 SIN(0.3 1 1k)\nR1 in out 1k\nD1 out 0 Dmod\nC1 out 0 200n"
)
# junction and diffusion charge: a state-dependent C and parameters that
# enter only through the charge (Cjo, Tt)
VARACTOR = (
    ".model Dmod D(Is=1e-12 N=1 Cjo=50n Vj=0.7 M=0.5 Tt=20u)\n"
    "V1 in 0 0.3 SIN(0.3 1 1k)\nR1 in out 1k\nD1 out 0 Dmod\nC1 out 0 20n"
)
CE = (
    ".model QN NPN(Is=1e-15 betaF=100)\n"
    "V1 in 0 0 SIN(0 10m 1k)\nV2 vcc 0 12\n"
    "C1 in nb 1u\nR1 vcc nb 100k\nR2 nb 0 22k\n"
    "Q1 out nb ne QN\nR3 vcc out 4.7k\nR4 ne 0 1k\nC2 ne 0 100u"
)
OTA = (
    ".model NM NMOS(Kp=120u W=2 L=1 Vto=0.7)\n"
    ".model PM PMOS(Kp=40u W=4 L=1 Vto=-0.7)\n"
    "V1 inp 0 2 SIN(2 50m 10k)\nV2 vdd 0 5\n"
    "M1 d1 inp t t NM\nM2 d2 inn t t NM\n"
    "M3 d1 d2 vdd vdd PM\nM4 d2 d2 vdd vdd PM\n"
    "M5 t bias 0 0 NM\nM8 bias bias 0 0 NM\nR3 vdd bias 75k\n"
    "M6 out d1 vdd vdd PM\nR4 out 0 20k\nC2 d1 out 10p\n"
    "R2 inn out 100k\nR1 inn fb 10k\nC3 fb 0 10u"
)
# tight enough that the step sequence's own dependence on the parameters
# (the adaptive control) stays below the comparison
TIGHT = dict(rtol=1e-9, atol=1e-12)


def _grid(npts=80, t_end=2e-3):
    return np.linspace(0.0, t_end, npts)


def _weights(npts):
    return 0.5 + np.sin(0.7 * np.arange(npts))


def _gradient(model, t, output, w, wrt=None):
    """d(w @ output)/dp by the parameters under ``wrt``."""
    tr = model.at().transient(t, **TIGHT)
    return tr.vjp([output], np.asarray(w).reshape(1, -1), wrt=wrt).to_dict()


def _signal(model, t, output, values=None):
    return model.at(values or {}).transient(t, **TIGHT).signal(output)


@pytest.mark.parametrize(
    "deck,t_end",
    [(RC, 2e-3), (CLIPPER, 2e-3), (VARACTOR, 2e-3), (CE, 1e-3), (OTA, 2e-4)],
    ids=["rc", "clipper", "varactor", "ce", "ota"],
)
def test_gradient_matches_fd(deck, t_end):
    model = sane.Model.from_netlist(deck)
    t = _grid(t_end=t_end)
    w = _weights(len(t))
    bound = model.values()
    # parameters with a value: a finite difference has a scale to step on
    wrt = [p for p in model.params if bound.get(p, 0.0) != 0.0]
    grad = _gradient(model, t, "out", w, wrt)

    def objective(values):
        return float(w @ _signal(model, t, "out", values))

    # the solve resolves a gradient only down to its tolerances, on each
    # parameter's own scale (dL/dlog p, not the raw dL/dp an Is spans): the
    # tolerance below is floored there instead of asserting into noise
    gmax = max(abs(v * bound[k]) for k, v in grad.items())
    for name, g in grad.items():
        base = bound[name]
        errs = []
        for rel in (1e-3, 1e-5):
            h = abs(base) * rel
            try:
                fd = (objective({name: base + h}) - objective({name: base - h})) / (2 * h)
            except ValueError:
                # the perturbation left the physical domain
                continue
            tol = 1e-4 * max(abs(g), abs(fd)) + 1e-6 * gmax / abs(base)
            errs.append((abs(g - fd) / tol, fd))
            if errs[-1][0] <= 1.0:
                break
        if errs:
            err, fd = min(errs)
            assert err <= 1.0, f"{name}: gradient {g} vs FD {fd}"


def test_vjp_is_the_sensitivities_contracted():
    model = sane.Model.from_netlist(CE)
    t = _grid(t_end=1e-3)
    w = _weights(len(t))
    tr = model.at().transient(t)
    names = ["R1", "R3", "qn.Is"]
    g = tr.vjp(["out"], w.reshape(1, -1), wrt=names).to_dict()
    s = tr.sensitivity("out", names)
    for j, name in enumerate(s.params):
        np.testing.assert_allclose(g[name], w @ s.grad[0, :, j], rtol=1e-12)


def test_transient_fn_forward_and_vjp():
    model = sane.Model.from_netlist(RC)
    t = _grid()
    w = _weights(len(t))
    f = sane.TransientFunction(model, "out", t, wrt=["R1", "C1"], **TIGHT)

    p0 = np.array([1e3, 100e-9])
    y = f(p0)
    assert y.shape == t.shape
    # forward equals the plain run at the same values
    ref = _signal(model, t, "out", {"R1": 1e3, "C1": 100e-9})
    np.testing.assert_allclose(y, ref, rtol=0, atol=1e-12)

    g = f.vjp(w)
    for j, (name, base) in enumerate(zip(f.wrt, p0)):
        h = base * 1e-5
        pp, pm = p0.copy(), p0.copy()
        pp[j], pm[j] = base + h, base - h
        fd = (w @ f(pp) - w @ f(pm)) / (2 * h)
        scale = max(abs(fd), abs(g[j]), 1e-9)
        assert abs(g[j] - fd) <= 1e-4 * scale, f"{name}: vjp {g[j]} vs FD {fd}"


def test_gradient_by_a_subset_of_a_ladder():
    # a ladder with two parameters per stage: the gradient by a few of them
    n = 30
    lines = ["V1 n0 0 0 SIN(0 1 1k)"]
    for i in range(1, n + 1):
        lines.append(f"R{i} n{i-1} n{i} 1k")
        lines.append(f"C{i} n{i} 0 100n")
    model = sane.Model.from_netlist("\n".join(lines))
    t = _grid(60)
    w = _weights(len(t))
    names = ["R1", "C2", "R3"]
    grad = _gradient(model, t, "n3", w, names)
    assert sorted(grad) == sorted(names)

    def objective(values):
        return float(w @ _signal(model, t, "n3", values))

    for name in names:
        base = model[name]
        h = base * 1e-5
        fd = (objective({name: base + h}) - objective({name: base - h})) / (2 * h)
        g = grad[name]
        scale = max(abs(fd), abs(g), 1e-9)
        assert abs(g - fd) <= 1e-4 * scale, f"{name}: gradient {g} vs FD {fd}"


def test_as_jax_grad():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    model = sane.Model.from_netlist(RC)
    t = _grid(60)
    w = jnp.asarray(_weights(len(t)))
    f = sane.TransientFunction(model, "out", t, wrt=["R1", "C1"])
    g = sane.interop.as_jax(f)

    loss = lambda p: jnp.sum(w * g(p))  # noqa: E731
    p0 = jnp.array([1e3, 100e-9])
    got = np.asarray(jax.grad(loss)(p0))
    f(np.asarray(p0))
    want = f.vjp(np.asarray(w))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_as_torch_backward():
    torch = pytest.importorskip("torch")

    model = sane.Model.from_netlist(RC)
    t = _grid(60)
    w_np = _weights(len(t))
    f = sane.TransientFunction(model, "out", t, wrt=["R1", "C1"])
    F = sane.interop.as_torch(f)

    p = torch.tensor([1e3, 100e-9], dtype=torch.float64, requires_grad=True)
    w = torch.as_tensor(w_np)
    loss = (w * F(p)).sum()
    loss.backward()
    f(np.array([1e3, 100e-9]))
    want = f.vjp(w_np)
    np.testing.assert_allclose(p.grad.numpy(), want, rtol=1e-12, atol=0)
