#########################################################################################
##
##                          OPTIMIZER-FRAMEWORK INTEROP
##                                 (interop.py)
##
##          Wrap SANE's differentiable analysis functions into native
##          autograd nodes: `as_torch` -> torch.autograd.Function,
##          `as_jax` -> jax.custom_vjp. The engine runs the physics and
##          the exact adjoint; the framework composes the rest of the
##          loss graph. Imports of torch/jax are deferred so the core
##          package stays dependency-light.
##
#########################################################################################

import numpy as np


def _check_real(f):
    if getattr(f, "metric", "mag") == "complex":
        raise ValueError(
            "interop wrappers take real-output functions; use metric='mag' "
            "(complex responses have framework-specific cotangent conventions)"
        )


def as_torch(f):
    """Wrap a differentiable SANE function (:class:`~sane.differentiable.ParamFunction`)
    for PyTorch.

    Returns a callable ``F(p: Tensor) -> Tensor`` that participates in
    autograd: the forward runs the engine analysis, the backward its exact
    adjoint. The solver state is stashed per call, so several forwards may be
    in flight before their backwards run.

    Example
    -------
    ::

        f = model.transient_fn("out", t, wrt=["R1", "C1"])
        F = sane.interop.as_torch(f)
        p = torch.tensor([1e3, 100e-9], dtype=torch.float64, requires_grad=True)
        loss = ((F(p) - ref) ** 2).sum()
        loss.backward()               # p.grad via the engine adjoint
    """
    import torch

    _check_real(f)

    class _SaneFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p):
            y = f(p.detach().cpu().numpy())
            ctx.sane_stash = f._last_stash
            return torch.as_tensor(np.atleast_1d(y), dtype=p.dtype, device=p.device)

        @staticmethod
        def backward(ctx, grad_y):
            g = f.vjp(grad_y.detach().cpu().numpy().reshape(-1), stash=ctx.sane_stash)
            return torch.as_tensor(g, dtype=grad_y.dtype, device=grad_y.device)

    def wrapped(p):
        return _SaneFunction.apply(p)

    return wrapped


def as_jax(f):
    """Wrap a differentiable SANE function for JAX.

    Returns a ``jax.custom_vjp`` function ``g(p) -> response`` whose forward
    and backward run the engine through ``jax.pure_callback`` -- usable under
    ``jax.grad`` / ``jax.value_and_grad`` (and inside jit; the callback runs
    on the host). The backward re-runs the forward on the host to regenerate
    the solver state for exactly the requested parameters (with warm starting
    this re-solve is cheap), so the wrapper stays functionally pure.

    Example
    -------
    ::

        f = model.transient_fn("out", t, wrt=["R1", "C1"])
        g = sane.interop.as_jax(f)
        loss = lambda p: jnp.sum((g(p) - ref) ** 2)
        jax.grad(loss)(jnp.array([1e3, 100e-9]))
    """
    import jax
    import jax.numpy as jnp

    _check_real(f)
    out_len = getattr(f, "out_len", 1)
    y_shape = jax.ShapeDtypeStruct((out_len,), jnp.float64)
    g_shape = jax.ShapeDtypeStruct((f.n_wrt,), jnp.float64)

    def _forward_host(p):
        return np.atleast_1d(np.asarray(f(np.asarray(p)), dtype=np.float64))

    def _backward_host(p, gy):
        _, stash = f._forward(f.values_from(np.asarray(p)))
        return np.asarray(
            f._vjp(np.asarray(gy).reshape(-1), stash), dtype=np.float64
        )

    @jax.custom_vjp
    def g(p):
        return jax.pure_callback(_forward_host, y_shape, p)

    def g_fwd(p):
        return jax.pure_callback(_forward_host, y_shape, p), p

    def g_bwd(p, gy):
        return (jax.pure_callback(_backward_host, g_shape, p, gy),)

    g.defvjp(g_fwd, g_bwd)
    return g
