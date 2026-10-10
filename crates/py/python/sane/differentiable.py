#########################################################################################
##
##                        DIFFERENTIABLE ANALYSIS FUNCTIONS
##                               (differentiable.py)
##
##          Analyses as function objects an optimizer can call and
##          differentiate: parameters in, response out, exact gradients
##          through the engine's adjoints. One uniform shape across
##          DC / AC / S-parameters / transient / harmonic balance / poles --
##          and the surface the torch/JAX wrappers in `sane.interop` build on:
##
##              f = sane.DcFunction(model, "out", wrt=[...])
##              y = f(p)              # forward at parameter vector p
##              g = f.vjp(dL_dy)      # exact dL/dp of the computed response
##
#########################################################################################

import numpy as np


class ParamFunction:
    """Shared base: parameter handling and the VJP contract.

    Subclasses implement ``_forward(point) -> (y, stash)`` and
    ``_vjp(cotangent, stash) -> ndarray`` over the engine's adjoints. The
    parameters are a vector in ``wrt`` order (or a ``name -> value`` dict);
    everything not in ``wrt`` keeps its bound value. The result a forward
    computed is the stash its backward pass differentiates, so several
    forwards may be in flight.

    Warm starting: with ``warm=True`` (default) each forward solves its
    operating point from the previous forward's (:meth:`sane.Point.near`),
    a few Newton steps for an optimizer's small moves. A multi-stable circuit
    then stays on the branch it started on; ``warm=False`` solves every
    forward cold.
    """

    def __init__(self, model, wrt=None, warm=True):
        self.model = model
        if wrt is None:
            wrt = [p for p in model.params if "." not in p]
        self.wrt = list(wrt)
        missing = [p for p in self.wrt if not model.is_param(p)]
        if missing:
            raise ValueError(f"unknown parameters in wrt: {missing}")
        self.warm = bool(warm)
        self._last_point = None
        self._last_stash = None

    @property
    def n_wrt(self):
        """Number of parameter inputs (``len(wrt)``)."""
        return len(self.wrt)

    def values_from(self, p):
        """The ``name -> value`` binding of a parameter vector or dict."""
        if isinstance(p, dict):
            return {k: float(v) for k, v in p.items()}
        p = np.asarray(p, dtype=float).reshape(-1)
        if p.shape[0] != len(self.wrt):
            raise ValueError(f"expected {len(self.wrt)} parameters, got {p.shape[0]}")
        return {name: float(v) for name, v in zip(self.wrt, p)}

    def point(self, p):
        """The point of the parameters ``p``, warm-started (see above)."""
        pt = self.model.at(self.values_from(p))
        if self.warm:
            if self._last_point is not None:
                pt = pt.near(self._last_point)
            self._last_point = pt
        return pt

    def forward(self, p):
        """The response at ``p`` and the stash its :meth:`vjp` needs."""
        return self._forward(self.point(p))

    def __call__(self, p):
        y, stash = self.forward(p)
        self._last_stash = stash
        return y

    def vjp(self, cotangent, stash=None):
        """Pull the response cotangent ``dL/dy`` back to ``dL/dp`` (``wrt``
        order). Uses the most recent ``__call__`` unless a ``stash`` from an
        earlier forward is passed explicitly."""
        if stash is None:
            stash = self._last_stash
        if stash is None:
            raise RuntimeError("vjp before any forward call (and no stash given)")
        return self._vjp(np.asarray(cotangent), stash)

    def _select(self, names, grads):
        """The entries of a gradient over ``names`` in ``wrt`` order."""
        idx = {n: i for i, n in enumerate(names)}
        return np.array([grads[idx[name]] for name in self.wrt])


class TransientFunction(ParamFunction):
    """``f(p) -> waveform`` of ``output`` at the times ``t``; :meth:`vjp`
    contracts the cotangent with the forward sensitivities by ``wrt``.

    Example
    -------
    ::

        f = sane.TransientFunction(model, "out", t, wrt=["R1", "C1"])
        y = f([1e3, 100e-9])
        g = f.vjp(2.0 * (y - ref))    # d/dp of sum((y - ref)^2)
    """

    def __init__(self, model, output, t, wrt=None, warm=True, **options):
        super().__init__(model, wrt, warm)
        self.output = output
        self.t = np.asarray(t, dtype=float)
        # the transient's options (rtol, atol, dt_max, x0)
        self.options = options

    @property
    def out_len(self):
        return len(self.t)

    def _forward(self, point):
        tr = point.transient(self.t, **self.options)
        return tr.signal(self.output), tr

    def _vjp(self, cotangent, tr):
        c = np.asarray(cotangent, dtype=float).reshape(1, -1)
        g = tr.vjp([self.output], c, wrt=self.wrt)
        return self._select(g.params, g.grad)


class DcFunction(ParamFunction):
    """``f(p) -> float``: the DC value of ``output``; :meth:`vjp` is the DC
    adjoint (one transpose solve, every parameter).

    Example
    -------
    ::

        f = sane.DcFunction(model, "out", wrt=["R1", "R2"])
        y = f([4.7e3, 22e3])
        g = f.vjp()                   # dy/dp; vjp(c) scales by the cotangent c
    """

    def __init__(self, model, output, wrt=None, warm=True):
        super().__init__(model, wrt, warm)
        self.output = output

    def _forward(self, point):
        op = point.operating_point()
        return float(op[self.output]), op

    def _vjp(self, cotangent, op):
        s = op.sensitivity(self.output, self.wrt)
        return float(np.asarray(cotangent).reshape(-1)[0]) * self._select(s.params, s.grad[0])

    def vjp(self, cotangent=1.0, stash=None):
        return super().vjp(cotangent, stash)


def _complex_cotangent(cotangent, h, metric):
    """``dL/dRe h + 1j dL/dIm h`` for a cotangent of ``|h|`` (``metric
    "mag"``) or of ``h`` itself (``"complex"``)."""
    c = np.asarray(cotangent)
    if metric == "mag":
        mag = np.maximum(np.abs(h), 1e-300)
        return c.astype(float) * h / mag
    return c.astype(complex)


class AcFunction(ParamFunction):
    """``f(p) -> H`` over ``freqs`` for the transfer ``input -> output``;
    :meth:`vjp` is one weighted AC adjoint per frequency (the operating
    point's shift included).

    ``metric="mag"`` (default) returns ``|H|`` with a real cotangent;
    ``metric="complex"`` returns complex ``H`` -- its cotangent is
    ``dL/dRe(H) + 1j * dL/dIm(H)`` for a real-valued loss.
    """

    def __init__(self, model, input, output, freqs, wrt=None, metric="mag", warm=True):
        super().__init__(model, wrt, warm)
        if metric not in ("mag", "complex"):
            raise ValueError("metric must be 'mag' or 'complex'")
        self.input = input
        self.output = output
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        self.metric = metric

    @property
    def out_len(self):
        return len(self.freqs)

    def _forward(self, point):
        ac = point.ac(self.input, self.output, self.freqs)
        h = ac.of(self.output)
        return (np.abs(h) if self.metric == "mag" else h.copy()), ac

    def _vjp(self, cotangent, ac):
        cot = _complex_cotangent(cotangent, ac.of(self.output), self.metric)
        g = ac.vjp(cot.reshape(1, -1))
        return self._select(g.params, g.grad)


class SpFunction(ParamFunction):
    """``f(p) -> S`` of shape ``(nf, n, n)`` (complex) over the circuit's
    ports (deck ``P`` elements); :meth:`vjp` contracts a complex cotangent
    (``dL/dRe + 1j*dL/dIm`` per entry) through the AC adjoints.
    """

    metric = "complex"

    def __init__(self, model, freqs, wrt=None, warm=True):
        super().__init__(model, wrt, warm)
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=float))

    @property
    def out_len(self):
        return len(self.freqs) * len(self.model.ports) ** 2

    def _forward(self, point):
        sp = point.s_parameters(self.freqs)
        return sp.s.copy(), sp

    def _vjp(self, cotangent, sp):
        g = sp.vjp(np.asarray(cotangent, dtype=complex).reshape(sp.s.shape))
        return self._select(g.params, g.grad)


class HbFunction(ParamFunction):
    """``f(p) -> spectrum`` of ``output`` (harmonics ``k = 0..K``) by
    harmonic balance; :meth:`vjp` is the implicit-function adjoint on the
    HB Jacobian.

    ``metric="mag"`` (default) returns ``|X_k|``; ``metric="complex"`` the
    complex coefficients (cotangent convention as in :class:`AcFunction`).
    ``f0=None`` takes the fundamental of the deck's periodic source.
    """

    def __init__(self, model, output, f0=None, harmonics=8, wrt=None,
                 metric="mag", oversample=16, warm=True):
        super().__init__(model, wrt, warm)
        if metric not in ("mag", "complex"):
            raise ValueError("metric must be 'mag' or 'complex'")
        self.output = output
        self.f0 = f0
        self.harmonics = int(harmonics)
        self.metric = metric
        self.oversample = int(oversample)

    @property
    def out_len(self):
        return self.harmonics + 1

    def _forward(self, point):
        hb = point.harmonic_balance(f0=self.f0, harmonics=self.harmonics,
                                    oversample=self.oversample)
        x = hb.spectrum(self.output)
        return (np.abs(x) if self.metric == "mag" else x.copy()), hb

    def _vjp(self, cotangent, hb):
        s = hb.sensitivity(self.output, self.wrt)
        cot = _complex_cotangent(cotangent, hb.spectrum(self.output), self.metric)
        # dL/dp = sum_k Re(conj(cot_k) dX_k/dp)
        g = np.real(np.conj(cot) @ s.grad[0])
        return self._select(s.params, g)


class PzFunction(ParamFunction):
    """``f(p) -> poles`` (complex, sorted by real then imaginary part) of the
    small-signal system at the operating point; :meth:`vjp` uses the exact
    pole migration.

    The cotangent is complex per pole: ``dL/dRe(s_i) + 1j * dL/dIm(s_i)``
    for a real-valued loss.
    """

    metric = "complex"  # keeps the real-only interop wrappers honest

    def __init__(self, model, wrt=None, warm=True):
        super().__init__(model, wrt, warm)
        self._n_poles = None

    @property
    def out_len(self):
        if self._n_poles is None:
            raise RuntimeError("out_len known after the first forward call")
        return self._n_poles

    def _forward(self, point):
        poles = point.poles()
        s = np.sort_complex(np.asarray(poles.poles))
        self._n_poles = len(s)
        return s, (poles, s)

    def _vjp(self, cotangent, stash):
        poles, s = stash
        rs = poles.sensitivity(self.wrt)
        acc = np.zeros(len(rs.params))
        for r, root in enumerate(rs.roots):
            i = int(np.argmin(np.abs(s - root)))
            c = complex(cotangent[i])
            acc += c.real * rs.grad[r].real + c.imag * rs.grad[r].imag
        return self._select(rs.params, acc)
