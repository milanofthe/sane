#########################################################################################
##
##                        DIFFERENTIABLE ANALYSIS FUNCTIONS
##                               (differentiable.py)
##
##          Analyses as function objects an optimizer can call and
##          differentiate: parameters in, response out, exact gradients
##          through the engine's adjoints. One uniform shape across
##          DC / AC / transient / harmonic balance -- and the surface the
##          torch/JAX wrappers in `sane.interop` build on:
##
##              f = model.<analysis>_fn(..., wrt=[...])
##              y = f(p)              # forward at parameter vector p
##              g = f.vjp(dL_dy)      # exact dL/dp, one adjoint sweep
##
#########################################################################################

import numpy as np


class ParamFunction:
    """Shared base: parameter handling, warm-start state, VJP contract.

    Subclasses implement ``_forward(values) -> ndarray`` and
    ``_vjp(cotangent, stash) -> ndarray`` over the engine adjoints. The
    parameters are a vector in ``wrt`` order (or a ``name -> value`` dict);
    everything not in ``wrt`` keeps its bound value.

    Warm starting: with ``warm=True`` (default) each forward seeds its DC
    Newton with the previous operating point -- the solved point is the same
    fixed point, only reached in fewer iterations, so results and gradients
    are unaffected. State for the backward pass is stashed per call by the
    interop wrappers, so several forwards may be in flight.
    """

    def __init__(self, model, wrt=None, warm=True):
        self.model = model
        if wrt is None:
            wrt = [p for p in model.params if "." not in p]
        self.wrt = list(wrt)
        known = set(model.params)
        missing = [p for p in self.wrt if p not in known]
        if missing:
            raise ValueError(f"unknown parameters in wrt: {missing}")
        self.warm = bool(warm)
        self._x_warm = None
        self._last_stash = None

    @property
    def n_wrt(self):
        """Number of parameter inputs (``len(wrt)``)."""
        return len(self.wrt)

    def values_from(self, p):
        """The ``name -> value`` override dict for a parameter vector/dict."""
        if isinstance(p, dict):
            return {k: float(v) for k, v in p.items()}
        p = np.asarray(p, dtype=float).reshape(-1)
        if p.shape[0] != len(self.wrt):
            raise ValueError(f"expected {len(self.wrt)} parameters, got {p.shape[0]}")
        return {name: float(v) for name, v in zip(self.wrt, p)}

    def __call__(self, p):
        values = self.values_from(p)
        y, stash = self._forward(values)
        self._last_stash = stash
        return y

    def vjp(self, cotangent, stash=None):
        """Pull the response cotangent ``dL/dy`` back to ``dL/dp`` (``wrt``
        order). Uses the state of the most recent ``__call__`` unless a
        ``stash`` from an earlier forward is passed explicitly."""
        if stash is None:
            stash = self._last_stash
        if stash is None:
            raise RuntimeError("vjp before any forward call (and no stash given)")
        return self._vjp(np.asarray(cotangent), stash)

    def _select(self, names, grads):
        """Map an all-parameter gradient onto the ``wrt`` subset."""
        idx = {n: i for i, n in enumerate(names)}
        return np.array([grads[idx[name]] for name in self.wrt])


class TransientFunction(ParamFunction):
    """``f(p) -> waveform`` of ``output`` on the BE grid ``t``; :meth:`vjp`
    runs the discrete transient adjoint (all parameters, one backward sweep).

    Example
    -------
    ::

        f = model.transient_fn("out", t, wrt=["R1", "C1"])
        y = f([1e3, 100e-9])
        g = f.vjp(2.0 * (y - ref))    # d/dp of sum((y - ref)^2)
    """

    def __init__(self, model, output, t, wrt=None, warm=True):
        super().__init__(model, wrt, warm)
        self.output = output
        self.t = np.asarray(t, dtype=float)
        self._out_idx = model._idx(output)

    @property
    def out_len(self):
        return len(self.t)

    def _forward(self, values):
        guess = self._x_warm if self.warm else None
        traj = self.model.transient_grid(self.t, values=values, dc_guess=guess)
        m = np.asarray(traj.matrix)
        if self.warm:
            self._x_warm = list(m[0])
        return m[:, self._out_idx], values

    def _vjp(self, cotangent, values):
        grad = self.model.transient_adjoint(
            self.t,
            {self.output: cotangent},
            values=values,
            dc_guess=self._x_warm if self.warm else None,
        )
        return np.array([grad[name] for name in self.wrt])


class DcFunction(ParamFunction):
    """``f(p) -> float``: the DC value of ``output``; :meth:`vjp` is the DC
    adjoint (one transpose solve, every parameter).

    Example
    -------
    ::

        f = model.dc_fn("out", wrt=["R1", "R2"])
        y = f([4.7e3, 22e3])
        g = f.vjp()                   # dy/dp; vjp(c) scales by the cotangent c
    """

    def __init__(self, model, output, wrt=None, warm=True):
        super().__init__(model, wrt, warm)
        self.output = output
        self._out_idx = model._idx(output)
        # the engine's DC adjoint resolves the canonical unknown name
        self._unknown = model._unknowns[self._out_idx]

    def _forward(self, values):
        p = self.model._pvec(values)
        x = self.model._dc(p, self._x_warm if self.warm else None)
        if self.warm:
            self._x_warm = list(x)
        return float(x[self._out_idx]), (list(p), list(x))

    def _vjp(self, cotangent, stash):
        p, x = stash
        names, grads = self.model._d.sensitivity(self._unknown, x, p, 0.0)
        return float(np.asarray(cotangent).reshape(-1)[0]) * self._select(names, np.asarray(grads))

    def vjp(self, cotangent=1.0, stash=None):
        return super().vjp(cotangent, stash)


class AcFunction(ParamFunction):
    """``f(p) -> H`` over ``freqs`` for the transfer ``input -> output``;
    :meth:`vjp` runs one all-parameter AC adjoint per frequency (incl. the
    operating-point shift).

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
        self._out_idx = model._idx(output)

    @property
    def out_len(self):
        return len(self.freqs)

    def _forward(self, values):
        p = self.model._pvec(values)
        x = self.model._dc(p, self._x_warm if self.warm else None)
        if self.warm:
            self._x_warm = list(x)
        pairs = self.model._d.ac_response(
            self.input, self._out_idx, list(x), list(p), list(self.freqs)
        )
        h = np.array([re + 1j * im for re, im in pairs])
        y = np.abs(h) if self.metric == "mag" else h
        return y, (list(p), list(x), h)

    def _vjp(self, cotangent, stash):
        p, x, h = stash
        acc = np.zeros(len(self.wrt))
        for i, f in enumerate(self.freqs):
            rows = self.model._d.ac_gradient(self.input, self._out_idx, x, p, float(f))
            names = [r[0] for r in rows]
            dre = np.asarray([r[1] for r in rows])
            dim = np.asarray([r[2] for r in rows])
            if self.metric == "mag":
                mag = max(abs(h[i]), 1e-300)
                dmag = (h[i].real * dre + h[i].imag * dim) / mag
                acc += float(cotangent[i]) * self._select(names, dmag)
            else:
                c = complex(cotangent[i])
                acc += self._select(names, c.real * dre + c.imag * dim)
        return acc


class SpFunction(ParamFunction):
    """``f(p) -> S`` of shape ``(nf, n, n)`` (complex) for the ports
    ``[(source, node), ...]``; :meth:`vjp` contracts a complex cotangent
    (``dL/dRe + 1j*dL/dIm`` per entry) through the AC adjoints of every
    port-to-port transfer.

    Port convention (see :mod:`sane.rf`): each port is an ideal V source in
    series with a ``z0`` resistor in the deck; ``node`` is the terminal on
    the network side. Then ``S_ij = 2*sqrt(z0_j/z0_i)*H_ij - delta_ij`` with
    ``H_ij`` the plain AC node transfer from source j to node i.
    """

    def __init__(self, model, ports, freqs, z0=50.0, wrt=None, warm=True):
        super().__init__(model, wrt, warm)
        self.ports = [(str(s), str(n)) for s, n in ports]
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        n = len(self.ports)
        self.z0 = np.broadcast_to(np.asarray(z0, dtype=float), (n,)).copy()
        # native port spec: (drive source, out_idx, z0) per port
        self._spec = [
            (src, model._idx(node), float(zp))
            for (src, node), zp in zip(self.ports, self.z0)
        ]

    @property
    def out_len(self):
        return len(self.freqs) * len(self.ports) ** 2

    def _forward(self, values):
        p = self.model._pvec(values)
        x = self.model._dc(p, self._x_warm if self.warm else None)
        if self.warm:
            self._x_warm = list(x)
        n = len(self.ports)
        rows = self.model._d.sp_response(self._spec, list(x), list(p), list(self.freqs))
        s = np.array([[re + 1j * im for re, im in row] for row in rows])
        return s.reshape(len(self.freqs), n, n), (list(p), list(x))

    def _vjp(self, cotangent, stash):
        p, x = stash
        n = len(self.ports)
        cot = np.asarray(cotangent, dtype=complex).reshape(len(self.freqs), n * n)
        rows = self.model._d.sp_vjp(
            self._spec,
            x,
            p,
            list(self.freqs),
            [[(float(c.real), float(c.imag)) for c in row] for row in cot],
        )
        names = [r[0] for r in rows]
        return self._select(names, np.asarray([r[1] for r in rows]))


class HbFunction(ParamFunction):
    """``f(p) -> spectrum`` of ``output`` (harmonics ``k = 0..K``) by harmonic
    balance; :meth:`vjp` runs the implicit-function adjoint on the HB Jacobian
    (all parameters and harmonics from one linear solve).

    ``metric="mag"`` (default) returns ``|X_k|``; ``metric="complex"`` the
    complex coefficients (cotangent convention as in :class:`AcFunction`).
    ``f0=0`` infers the fundamental from the deck's periodic source.
    """

    def __init__(self, model, output, f0=0.0, harmonics=8, wrt=None,
                 metric="mag", oversample=16, warm=True):
        super().__init__(model, wrt, warm)
        if metric not in ("mag", "complex"):
            raise ValueError("metric must be 'mag' or 'complex'")
        self.output = output
        self.f0 = float(f0)
        self.harmonics = int(harmonics)
        self.metric = metric
        self.oversample = int(oversample)
        self._out_idx = model._idx(output)

    @property
    def out_len(self):
        return self.harmonics + 1

    def _forward(self, values):
        p = list(self.model._pvec(values))
        spectra, conv, _iters, res, _su, _so, f0_eff = self.model._d.solve_hb(
            p, self.f0, self.harmonics,
            self._x_warm if self.warm else None,
            self.oversample, 1e-10, 60, None, None,
        )
        if not conv:
            raise RuntimeError(f"harmonic balance did not converge (residual {res:.3e})")
        row = spectra[self._out_idx]
        x = np.array([re + 1j * im for re, im in row])
        if self.warm:
            # the DC harmonic of every unknown seeds the next solve's DC start
            self._x_warm = [s[0][0] for s in spectra]
        y = np.abs(x) if self.metric == "mag" else x
        return y, (p, spectra, f0_eff, x)

    def _vjp(self, cotangent, stash):
        p, spectra, f0_eff, xk = stash
        rows = self.model._d.hb_gradient(
            self._out_idx, spectra, p, f0_eff, self.harmonics, self.oversample, None
        )
        acc = np.zeros(len(self.wrt))
        for k, row in enumerate(rows):
            names = [r[0] for r in row]
            dre = np.asarray([r[1] for r in row])
            dim = np.asarray([r[2] for r in row])
            if self.metric == "mag":
                mag = max(abs(xk[k]), 1e-300)
                dmag = (xk[k].real * dre + xk[k].imag * dim) / mag
                acc += float(cotangent[k]) * self._select(names, dmag)
            else:
                c = complex(cotangent[k])
                acc += self._select(names, c.real * dre + c.imag * dim)
        return acc


class PzFunction(ParamFunction):
    """``f(p) -> poles`` (complex, sorted by real part then imaginary part) of
    the small-signal pencil at the DC operating point; :meth:`vjp` uses the
    exact pole-migration sensitivities.

    The cotangent is complex per pole: ``dL/dRe(s_i) + 1j * dL/dIm(s_i)`` for
    a real-valued loss. Unlike the other functions the backward cost is one
    engine call per ``wrt`` parameter (the pole sensitivities are computed
    per-parameter); keep ``wrt`` to the knob subset under optimization.
    Sensitivities are matched to the forward's poles by nearest value.
    """

    metric = "complex"  # keeps the real-only interop wrappers honest

    def __init__(self, model, input, wrt=None, warm=True):
        super().__init__(model, wrt, warm)
        self.input = input
        self._n_poles = None

    @property
    def out_len(self):
        if self._n_poles is None:
            raise RuntimeError("out_len known after the first forward call")
        return self._n_poles

    def _forward(self, values):
        p = self.model._pvec(values)
        x = self.model._dc(p, self._x_warm if self.warm else None)
        if self.warm:
            self._x_warm = list(x)
        pairs = self.model._d.poles(list(x), list(p))
        poles = np.sort_complex(np.array([re + 1j * im for re, im in pairs]))
        self._n_poles = len(poles)
        return poles, (list(p), list(x), poles)

    def _vjp(self, cotangent, stash):
        p, x, poles = stash
        acc = np.zeros(len(self.wrt))
        for j, name in enumerate(self.wrt):
            rows = self.model._d.pole_sensitivity(self.input, name, x, p)
            for (sre, sim), (dre, dim) in rows:
                i = int(np.argmin(np.abs(poles - (sre + 1j * sim))))
                c = complex(cotangent[i])
                acc[j] += c.real * dre + c.imag * dim
        return acc
