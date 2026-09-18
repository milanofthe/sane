"""Discrete transient adjoint through the Python API: FD validation matrix,
the differentiable TransientFunction, and the JAX/torch interop wrappers
(skipped when the framework is not installed)."""

import numpy as np
import pytest

import sane

RC = "V1 in 0 0 SIN(0 1 1k)\nR1 in out 1k\nC1 out 0 100n"
CLIPPER = (
    ".model Dmod D(Is=1e-12 N=1)\n"
    "V1 in 0 0.3 SIN(0.3 1 1k)\nR1 in out 1k\nD1 out 0 Dmod\nC1 out 0 200n"
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


def _grid(npts=80, t_end=2e-3):
    return np.linspace(0.0, t_end, npts)


def _weights(npts):
    return 0.5 + np.sin(0.7 * np.arange(npts))


@pytest.mark.parametrize(
    "deck,t_end",
    [(RC, 2e-3), (CLIPPER, 2e-3), (CE, 1e-3), (OTA, 2e-4)],
    ids=["rc", "clipper", "ce", "ota"],
)
def test_adjoint_matches_fd(deck, t_end):
    model = sane.Circuit.parse(deck).extract()
    t = _grid(t_end=t_end)
    w = _weights(len(t))

    grad = model.transient_adjoint(t, {"out": w})

    def objective(values):
        return float(w @ model.transient_grid(t, values=values)["out"])

    bound = model.values
    # near-zero gradients sit below what central FD can resolve against the
    # Newton termination noise of the forward solves: skip them relative to
    # the dominant gradient instead of asserting into noise
    gmax = max(abs(v) for v in grad.values())
    for name, g in grad.items():
        base = bound.get(name, 0.0)
        # 1e-3 relative: large enough that the FD difference clears the Newton
        # termination noise of the two forward solves, small enough that the
        # O(h^2) truncation stays inside the assertion tolerance
        if base == 0.0:
            # No finite difference can reference a parameter that defaults to
            # zero: there is no step that is both small on the parameter's own
            # (unknown) physical scale -- a leakage current lives near 1e-15, a
            # channel-length factor near 1e-2 -- and large enough to clear the
            # Newton termination noise of two forward solves. Stepping down also
            # leaves the physical domain, so a central difference reports a
            # branch jump whose numerator does not shrink with h. Forward-mode
            # AD covers these instead (test below); it needs no perturbation.
            continue
        h = abs(base) * 1e-3
        try:
            fd = (objective({name: base + h}) - objective({name: base - h})) / (2 * h)
        except ValueError:
            # the perturbation left the physical domain (e.g. a negative
            # saturation current from a zero-defaulted leakage parameter)
            continue
        if max(abs(g), abs(fd)) < 1e-6 * gmax:
            continue
        scale = max(abs(fd), abs(g), 1e-9)
        assert abs(g - fd) <= 2e-4 * scale + 1e-12, f"{name}: adjoint {g} vs FD {fd}"


@pytest.mark.parametrize(
    "deck,t_end,npts",
    [(RC, 2e-3, 80), (CE, 1e-3, 80), (OTA, 2e-4, 640)],
    ids=["rc", "ce", "ota"],
)
def test_adjoint_matches_forward_mode(deck, t_end, npts):
    """Cross-check the two AD modes against each other.

    The adjoint sweeps backwards and yields every parameter at once; the forward
    mode integrates an augmented sensitivity DAE per parameter. Different code,
    same quantity: ``d(w @ out)/dp = w @ (dx_out/dp)``. It is also the only
    reference that reaches parameters sitting at zero, where a finite difference
    has no usable step (see the skip above).

    Both paths are ESDIRK32 now, so what is left between them is step control:
    the adjoint runs one step per grid interval, the forward sensitivities step
    adaptively and interpolate. That difference is third order, so it shows up
    only where the grid does not resolve the dynamics -- the OTA's 10 pF
    compensation cap is the case in point, and it gets the grid that resolves
    it. Measured on that fixture as the grid refines: 6.3e-2, 3.7e-2, 1.4e-2,
    2.2e-3 for 80, 160, 320, 640 points, while every non-reactive parameter sits
    near 1e-5 throughout.
    """
    model = sane.Circuit.parse(deck).extract()
    t = _grid(npts=npts, t_end=t_end)
    w = _weights(len(t))
    out_key = f"v{model.node_names.index('out')}"

    grad = model.transient_adjoint(t, {"out": w})
    gmax = max(abs(v) for v in grad.values())
    names = [n for n, g in grad.items() if abs(g) > 1e-6 * gmax]
    assert names, "no parameter carries a gradient"

    fwd = model.transient_sensitivity(names, t)
    for name in names:
        ref = float(w @ np.asarray(fwd[name].to_dict()[out_key]).ravel())
        scale = max(abs(ref), abs(grad[name]), 1e-9)
        assert abs(grad[name] - ref) <= 5e-3 * scale, f"{name}: adjoint {grad[name]} vs forward {ref}"


def test_transient_fn_forward_and_vjp():
    model = sane.Circuit.parse(RC).extract()
    t = _grid()
    w = _weights(len(t))
    f = model.transient_fn("out", t, wrt=["R1", "C1"])

    p0 = np.array([1e3, 100e-9])
    y = f(p0)
    assert y.shape == t.shape
    # forward equals the plain fixed-grid run at the same values
    ref = model.transient_grid(t, values={"R1": 1e3, "C1": 100e-9})["out"]
    np.testing.assert_allclose(y, ref, rtol=0, atol=1e-12)

    g = f.vjp(w)
    for j, (name, base) in enumerate(zip(f.wrt, p0)):
        h = base * 1e-5
        pp, pm = p0.copy(), p0.copy()
        pp[j], pm[j] = base + h, base - h
        fd = (w @ f(pp) - w @ f(pm)) / (2 * h)
        scale = max(abs(fd), abs(g[j]), 1e-9)
        assert abs(g[j] - fd) <= 2e-4 * scale, f"{name}: vjp {g[j]} vs FD {fd}"


def test_adjoint_cost_independent_of_param_count():
    # ladder with ~2 params per stage: the adjoint returns ALL of them from
    # one sweep; sanity-check a handful against FD
    n = 30
    lines = ["V1 n0 0 0 SIN(0 1 1k)"]
    for i in range(1, n + 1):
        lines.append(f"R{i} n{i-1} n{i} 1k")
        lines.append(f"C{i} n{i} 0 100n")
    model = sane.Circuit.parse("\n".join(lines)).extract()
    t = _grid(60)
    w = _weights(len(t))
    grad = model.transient_adjoint(t, {"n3": w})
    assert len(grad) > 2 * n  # every parameter came back

    def objective(values):
        return float(w @ model.transient_grid(t, values=values)["n3"])

    for name in ["R1", "C2", "R3"]:
        base = model.values[name]
        h = base * 1e-5
        fd = (objective({name: base + h}) - objective({name: base - h})) / (2 * h)
        g = grad[name]
        scale = max(abs(fd), abs(g), 1e-9)
        assert abs(g - fd) <= 2e-4 * scale, f"{name}: adjoint {g} vs FD {fd}"


def test_as_jax_grad():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    model = sane.Circuit.parse(RC).extract()
    t = _grid(60)
    w = jnp.asarray(_weights(len(t)))
    f = model.transient_fn("out", t, wrt=["R1", "C1"])
    g = sane.interop.as_jax(f)

    loss = lambda p: jnp.sum(w * g(p))  # noqa: E731
    p0 = jnp.array([1e3, 100e-9])
    got = np.asarray(jax.grad(loss)(p0))
    f(np.asarray(p0))
    want = f.vjp(np.asarray(w))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_as_torch_backward():
    torch = pytest.importorskip("torch")

    model = sane.Circuit.parse(RC).extract()
    t = _grid(60)
    w_np = _weights(len(t))
    f = model.transient_fn("out", t, wrt=["R1", "C1"])
    F = sane.interop.as_torch(f)

    p = torch.tensor([1e3, 100e-9], dtype=torch.float64, requires_grad=True)
    w = torch.as_tensor(w_np)
    loss = (w * F(p)).sum()
    loss.backward()
    f(np.array([1e3, 100e-9]))
    want = f.vjp(w_np)
    np.testing.assert_allclose(p.grad.numpy(), want, rtol=1e-12, atol=0)
