#########################################################################################
##
##                              ANALYSIS RESULT OBJECTS
##                                   (analysis.py)
##
##           Labeled containers for the outputs of SANE's analyses. Every
##         result is indexable by node / unknown / parameter name, so node
##           voltages, trajectories and sensitivities read by name, never by
##                          positional index into a raw vector.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .warnings import SaneNumericalWarning, warn as _sane_warn


# NODE / UNKNOWN RESOLUTION =============================================================

def label_unknown(index, unknowns, node_names):
    """Human-readable label for an unknown, the inverse of :func:`resolve_unknown`.

    ``"v3"`` becomes ``"node 'out'"``, ``"i_V1"`` becomes ``"i(V1)"``; anything
    else is returned unchanged.

    Parameters
    ----------
    index : int
        index into the unknown / state vector
    unknowns : list[str]
        the unknown names, in vector order
    node_names : list[str]
        node names indexed by internal node id

    Returns
    -------
    str
    """
    name = unknowns[index]
    if name.startswith("v") and name[1:].isdigit() and int(name[1:]) < len(node_names):
        return f"node '{node_names[int(name[1:])]}'"
    if name.startswith("i_"):
        return f"i({name[2:]})"
    return name


def resolve_unknown(ref, unknowns, node_names):
    """Resolve a reference to an index into the unknown vector.

    Accepts, in order of precedence: an exact unknown name (``"v3"``,
    ``"i_V1"``), a node name (``"out"`` -> the ``v{id}`` of that node), or an
    element name whose branch current is an unknown (``"V1"`` -> ``"i_V1"``).

    Parameters
    ----------
    ref : str | int
        the reference to resolve
    unknowns : list[str]
        the unknown names, in vector order
    node_names : list[str]
        node names indexed by internal node id

    Returns
    -------
    int
        index into the unknown / state vector
    """
    if isinstance(ref, int):
        return ref
    s = str(ref)
    if s in unknowns:
        return unknowns.index(s)
    if s in node_names:
        u = f"v{node_names.index(s)}"
        if u in unknowns:
            return unknowns.index(u)
    ic = f"i_{s}"
    if ic in unknowns:
        return unknowns.index(ic)
    raise KeyError(
        f"'{ref}' is not a known node, unknown or branch current "
        f"(unknowns: {unknowns})"
    )


# OPERATING POINT =======================================================================

class OperatingPoint:
    """A DC operating point: the solved state vector, labeled by node / unknown.

    Index by node name (``op["out"]``), unknown name (``op["v3"]``) or source
    name for its branch current (``op["V1"]``). When the point was produced by a
    bound :class:`~sane.model.Model`, derived analyses (:meth:`sensitivity`,
    :meth:`hessian`) read straight off the stored solution -- no DC re-solve.

    Attributes
    ----------
    vector : numpy.ndarray
        the raw state vector ``x``, in :attr:`unknowns` order
    unknowns : list[str]
        unknown names, in vector order
    node_names : list[str]
        node names indexed by internal node id

    Example
    -------

    .. code-block:: python

        op = model.operating_point()
        vout = op["out"]              # node voltage by name
        ib = op["V1"]                 # source branch current
        sens = op.sensitivity("out")  # exact dy/dp over every parameter
    """

    def __init__(self, x, unknowns, node_names, dae=None, p=None, regularized_at_gmin=None):
        self.vector = np.asarray(x, dtype=float)
        self.unknowns = list(unknowns)
        self.node_names = list(node_names)
        # back-reference so derivatives can be read off this solved point
        self._dae = dae
        self._p = p
        # Machine-readable regularization flag (issue #54): the gmin this point
        # held at if it is a gmin-regularized (physically suspect) solution that
        # never reached the true DC floor, else None. converged is still True.
        self.regularized_at_gmin = regularized_at_gmin

    def sensitivity(self, output):
        """First-order sensitivity ``dy/dp`` of the metric ``y = output`` over
        **every** parameter, by one adjoint solve at this operating point.

        Computed off the stored operating point (no DC re-solve). Rank the result
        to identify the influential knobs, then pass those to :meth:`hessian`.
        (The parameter Jacobian is sparse, so the full gradient is cheap.)

        Parameters
        ----------
        output : str | int
            the metric to differentiate -- a node name, unknown name or source
            branch current, resolved as in :func:`resolve_unknown`

        Returns
        -------
        sane.analysis.Sensitivity
            the exact gradient over every parameter, with the operating-point
            values and metric attached for :meth:`Sensitivity.relative`

        Notes
        -----
        Exact via the adjoint (transpose) solve -- no finite differences.
        """
        if self._dae is None:
            raise RuntimeError("operating point is not bound to a Model")
        return self._dae._sensitivity_at(output, self.vector, self._p)

    def hessian(self, output, wrt):
        """Second-order sensitivity (Hessian) of ``y = output`` w.r.t. the
        parameter subset ``wrt``.

        Use the knobs identified from :meth:`sensitivity`. Computed by the
        second-order adjoint with exact autodiff directional derivatives off the
        stored operating point.

        Parameters
        ----------
        output : str | int
            the metric to differentiate (resolved as in :func:`resolve_unknown`)
        wrt : list[str]
            the parameter names to form the Hessian over (the identified knobs)

        Returns
        -------
        numpy.ndarray
            the dense symmetric ``len(wrt) x len(wrt)`` matrix -- curvature on the
            diagonal, knob interactions off it

        Notes
        -----
        Exact via the second-order adjoint -- no finite differences.
        """
        if self._dae is None:
            raise RuntimeError("operating point is not bound to a Model")
        return self._dae._hessian_at(output, list(wrt), self.vector, self._p)

    def __getitem__(self, ref):
        return float(self.vector[resolve_unknown(ref, self.unknowns, self.node_names)])

    def get(self, ref, default=None):
        """Return the value at ``ref``, or ``default`` if it cannot be resolved.

        Parameters
        ----------
        ref : str | int
            node name, unknown name or source branch current (see
            :func:`resolve_unknown`)
        default : object, optional
            value returned when ``ref`` does not resolve (default ``None``)

        Returns
        -------
        float
            the solved value at ``ref``, else ``default``
        """
        try:
            return self[ref]
        except (KeyError, IndexError):
            return default

    def to_dict(self):
        """Return the operating point as an ``{unknown: value}`` dict.

        Returns
        -------
        dict[str, float]
            one entry per unknown, keyed by unknown name
        """
        return {u: float(v) for u, v in zip(self.unknowns, self.vector)}

    def __repr__(self):
        body = ", ".join(f"{u}={v:.4g}" for u, v in zip(self.unknowns, self.vector))
        return f"OperatingPoint({body})"


# TRANSIENT TRAJECTORY ==================================================================

class Trajectory:
    """A transient solution: state versus time, labeled by node / unknown.

    Index by name to get that signal's time series (``traj["out"]`` -> array
    over :attr:`t`). When the trajectory was produced by a bound
    :class:`~sane.model.Model`, :meth:`sensitivity` reads the forward transient
    sensitivity off the same run.

    Attributes
    ----------
    t : numpy.ndarray
        the time points, shape ``(T,)``
    matrix : numpy.ndarray
        the state at each time point, shape ``(T, n)`` in :attr:`unknowns` order
    unknowns : list[str]
        unknown names, in column order
    node_names : list[str]
        node names indexed by internal node id

    Example
    -------

    .. code-block:: python

        traj = model.transient(np.linspace(0, 1e-3, 1000))
        vout = traj["out"]                 # time series for a node
        traj.plot("out", "in", show=True)  # overlay signals vs time
        sens = traj.sensitivity("out")     # d out(t_final)/dp
    """

    def __init__(self, t, matrix, unknowns, node_names, dae=None, values=None, x0=None):
        self.t = np.asarray(t, dtype=float)
        self.matrix = np.asarray(matrix, dtype=float)
        self.unknowns = list(unknowns)
        self.node_names = list(node_names)
        # back-reference + the bindings that produced this trajectory, so
        # derivatives can be read off it (forward transient sensitivity)
        self._dae = dae
        self._values = values
        self._x0 = x0

    def _metric_index(self, t):
        return len(self.t) - 1 if t is None else int(np.argmin(np.abs(self.t - float(t))))

    def sensitivity(self, output, t=None, wrt=None, rtol=1e-4, atol=1e-7):
        """Forward transient sensitivity ``d output(t*)/dp`` of the metric (the
        output value at time ``t*``) over the parameters ``wrt``.

        Exact -- the Model is augmented with the sensitivity equations and
        integrated jointly (no finite differences). Unlike the DC/AC adjoint,
        transient sensitivity is forward, so its cost scales with ``len(wrt)``;
        pass the candidate knobs for large parameter sets rather than relying on
        the all-parameter default.

        Parameters
        ----------
        output : str | int
            the metric to differentiate (resolved as in :func:`resolve_unknown`)
        t : float, optional
            the evaluation time ``t*``; the nearest stored time point is used.
            Defaults to the final time.
        wrt : list[str], optional
            the parameter names to differentiate over; defaults to every
            parameter of the bound Model
        rtol : float
            relative tolerance for the joint sensitivity integration (default
            ``1e-4``)
        atol : float
            absolute tolerance for the joint sensitivity integration (default
            ``1e-7``)

        Returns
        -------
        sane.analysis.Sensitivity
            the gradient at ``t*`` over ``wrt``, labeled ``output(t=t*)``, with
            parameter values and metric attached
        """
        if self._dae is None:
            raise RuntimeError("trajectory is not bound to a Model")
        wrt = list(wrt) if wrt is not None else list(self._dae.params)
        idx = self._metric_index(t)
        ds = self._dae.transient_sensitivity(wrt, self.t, rtol, atol, values=self._values)
        grad = [float(ds[pn][output][idx]) for pn in wrt]
        base = self._dae.values
        p0 = [base.get(pn, 0.0) for pn in wrt]
        y0 = float(self[output][idx])
        return Sensitivity(wrt, grad, f"{output}(t={self.t[idx]:g})", p0=p0, y0=y0)

    def hessian(self, output, subset, t=None):
        """Not implemented: the transient Hessian is not yet available.

        A reliable second-order transient sensitivity needs the exact
        second-order forward sensitivity equations integrated jointly (no finite
        differences); differencing the forward gradient is numerically unstable
        through the time integration. Use :meth:`sensitivity` for the gradient;
        for curvature use the DC (:meth:`OperatingPoint.hessian`) or AC
        (:meth:`AcResponse.hessian`) Hessian, which are exact.

        Parameters
        ----------
        output : str | int
            the metric that would be differentiated
        subset : list[str]
            the parameter subset the Hessian would be formed over
        t : float, optional
            the evaluation time

        Raises
        ------
        NotImplementedError
            always -- the second-order forward sensitivity is not yet available
        """
        raise NotImplementedError(
            "transient Hessian requires exact second-order forward sensitivity "
            "(not yet implemented); finite-differencing the gradient is unstable"
        )

    def __getitem__(self, ref):
        return self.matrix[:, resolve_unknown(ref, self.unknowns, self.node_names)]

    def to_dict(self):
        """Return ``{unknown: time_series}`` for every unknown.

        Returns
        -------
        dict[str, numpy.ndarray]
            one ``(T,)`` time series per unknown, keyed by unknown name
        """
        return {u: self.matrix[:, i] for i, u in enumerate(self.unknowns)}

    def plot(self, *refs, ax=None, show=False):
        """Plot one or more signals against time (requires matplotlib).

        Parameters
        ----------
        *refs : str
            node / unknown names to plot; defaults to every unknown
        ax : matplotlib.axes.Axes, optional
            axis to draw on; a new one is created if omitted
        show : bool
            call ``pyplot.show()`` after drawing

        Returns
        -------
        matplotlib.axes.Axes
            the axis drawn on
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        refs = refs or tuple(self.unknowns)
        for ref in refs:
            ax.plot(self.t, self[ref], label=str(ref))
        ax.set_xlabel("time")
        ax.legend()
        if show:
            plt.show()
        return ax

    def __repr__(self):
        return f"Trajectory({len(self.t)} points, {len(self.unknowns)} states)"


# COMPONENT SENSITIVITY =================================================================

class Sensitivity:
    """Exact first-order component sensitivity ``dy/dp`` of a metric ``y``.

    The gradient w.r.t. every parameter, computed by the adjoint method (one
    transpose solve, all parameters at once). Parameter names carry the full
    hierarchy (``X1.Q3.betaF``), so the result rolls up to device / subsystem
    level exactly.

    Attributes
    ----------
    params : list[str]
        parameter names
    gradient : numpy.ndarray
        ``dy/dp`` for each parameter, in :attr:`params` order
    output : str
        the metric the sensitivity is taken of

    Example
    -------

    .. code-block:: python

        sens = model.operating_point().sensitivity("out")
        sens["R1.r"]          # dy/dp for one parameter
        sens.ranked()[:5]     # the five most influential knobs
        sens.rollup()         # importance aggregated to device level
    """

    def __init__(self, params, gradient, output, p0=None, y0=None):
        self.params = list(params)
        self.gradient = np.asarray(gradient, dtype=float)
        self.output = output
        self._p0 = None if p0 is None else np.asarray(p0, dtype=float)
        self._y0 = y0

    def __getitem__(self, name):
        try:
            return float(self.gradient[self.params.index(name)])
        except ValueError:
            raise KeyError(f"'{name}' is not a parameter of this sensitivity")

    def get(self, name, default=None):
        """Return the sensitivity to ``name``, or ``default`` if it is absent.

        Parameters
        ----------
        name : str
            a parameter name (the full hierarchical name, e.g. ``"X1.Q3.betaF"``)
        default : object, optional
            value returned when ``name`` is not a parameter here (default
            ``None``)

        Returns
        -------
        float
            ``dy/dp`` for ``name``, else ``default``
        """
        try:
            return self[name]
        except KeyError:
            return default

    def to_dict(self):
        """Return ``{param: dy/dp}``.

        Returns
        -------
        dict[str, float]
            the raw gradient keyed by parameter name
        """
        return {p: float(g) for p, g in zip(self.params, self.gradient)}

    def relative(self):
        """Return the dimensionless relative sensitivities ``(dy/dp) * p / y``.

        Requires the operating-point parameter values and metric value, which
        are attached when the sensitivity is computed from a bound circuit.

        Returns
        -------
        numpy.ndarray
            relative sensitivity per parameter (zero where ``p`` or ``y`` is 0)
        """
        if self._p0 is None or self._y0 is None:
            raise ValueError("relative() needs bound parameter values and metric")
        out = np.zeros_like(self.gradient)
        for k in range(len(self.params)):
            if self._p0[k] and self._y0:
                out[k] = self.gradient[k] * self._p0[k] / self._y0
        return out

    def ranked(self, relative=True, threshold=0.0):
        """Parameters ranked by importance (largest magnitude first).

        Parameters
        ----------
        relative : bool
            rank by relative sensitivity (dimensionless) instead of raw ``dy/dp``
        threshold : float
            drop entries whose magnitude is at or below this

        Returns
        -------
        list[tuple[str, float]]
            ``(param, sensitivity)`` pairs, most important first
        """
        vals = self.relative() if relative else self.gradient
        order = np.argsort(-np.abs(vals))
        return [(self.params[k], float(vals[k])) for k in order if abs(vals[k]) > threshold]

    def rollup(self, relative=True):
        """Aggregate leaf sensitivities to the device / subsystem level.

        Importance of a component is the L2 norm of its leaf-parameter
        sensitivities; the grouping key is the dotted parameter prefix
        (``X1.Q3.betaF`` -> ``X1.Q3``).

        Parameters
        ----------
        relative : bool
            aggregate relative sensitivities instead of raw ``dy/dp``

        Returns
        -------
        list[tuple[str, float]]
            ``(component, importance)`` pairs, most important first
        """
        vals = self.relative() if relative else self.gradient
        groups = {}
        for k, name in enumerate(self.params):
            comp = name.rsplit(".", 1)[0] if "." in name else name
            groups[comp] = groups.get(comp, 0.0) + float(vals[k]) ** 2
        items = [(c, float(np.sqrt(v))) for c, v in groups.items()]
        items.sort(key=lambda kv: -kv[1])
        return items

    def __repr__(self):
        return f"Sensitivity(output={self.output!r}, {len(self.params)} parameters)"


# AC RESPONSE ===========================================================================

class AcResponse:
    """Small-signal AC response of ``input -> output``, with derivatives read off
    the linearised system at the operating point.

    Attributes
    ----------
    freqs : numpy.ndarray
        the frequencies in Hz
    value : numpy.ndarray
        the complex transfer ``H(j2*pi*f)`` over :attr:`freqs`

    The workflow mirrors DC: :meth:`sensitivity` gives the gradient of a scalar
    metric (``mag`` / ``phase`` / ``real`` / ``imag``) over **every** parameter at
    one frequency, by the AC adjoint (two complex solves + one DC adjoint, exact);
    rank it to identify knobs, then pass those to :meth:`hessian`.

    Example
    -------

    .. code-block:: python

        ac = model.ac("V1", "out", np.logspace(1, 7, 200))
        H = ac.value                       # complex transfer over ac.freqs
        sens = ac.sensitivity(1e3)         # d|H|/dp at 1 kHz, every parameter
        hess = ac.hessian(1e3, ["R1.r", "C1.c"], metric="phase")
    """

    def __init__(self, dae, input, output, out_idx, x, p, freqs):
        self._dae = dae
        self._input = input
        self._output = output
        self._out_idx = out_idx
        self._x = np.asarray(x, dtype=float)
        self._p = list(p)
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        self.value = self._response(self.freqs)
        self._warn_on_singular()

    def _warn_on_singular(self):
        """Surface singular AC frequencies (NaN in :attr:`value`) as an
        unconditional, catchable warning instead of a silent flat sweep (#39)."""
        bad = ~np.isfinite(self.value)
        if not bad.any():
            return
        fs = np.atleast_1d(self.freqs)[bad]
        if bad.all():
            msg = (
                f"AC system G + jwC is singular at all {bad.size} frequencies "
                f"(input '{self._input}' -> output '{self._output}'): the response is "
                "NaN. This is a structural singularity -- check for floating nodes, "
                "an ideal VCVS/inductor loop, or a bad DC operating point."
            )
        else:
            shown = ", ".join(f"{f:g}" for f in fs[:5]) + (" ..." if fs.size > 5 else "")
            msg = (
                f"AC solve failed (singular G + jwC) at {fs.size} of {bad.size} "
                f"frequencies [{shown}] Hz for '{self._input}' -> '{self._output}'; "
                "the response is NaN there."
            )
        _sane_warn(msg, SaneNumericalWarning)

    def _response(self, freqs):
        pairs = self._dae._d.ac_response(
            self._input, self._out_idx, list(self._x), self._p, list(np.atleast_1d(freqs))
        )
        return np.array([re + 1j * im for re, im in pairs])

    def _grad_complex(self, f):
        rows = self._dae._d.ac_gradient(
            self._input, self._out_idx, list(self._x), self._p, float(f)
        )
        names = [r[0] for r in rows]
        dH = np.array([complex(r[1], r[2]) for r in rows])
        return names, dH

    @staticmethod
    def _project(metric, H, dH):
        """Project the complex gradient ``dH/dp`` onto a real scalar metric."""
        m = abs(H)
        if metric in ("mag", "gain"):
            return ((H.conjugate() * dH).real / m if m else dH.real * 0.0), m
        if metric == "phase":
            return (dH / H).imag, float(np.angle(H))
        if metric == "real":
            return dH.real, H.real
        if metric == "imag":
            return dH.imag, H.imag
        raise ValueError(f"unknown AC metric '{metric}' (use mag/phase/real/imag)")

    def sensitivity(self, f, metric="mag"):
        """Gradient of the AC metric at frequency ``f`` over every parameter, by
        one AC adjoint solve.

        Computed off the stored operating point (the operating-point shift is
        included exactly), so no DC re-solve is needed.

        Parameters
        ----------
        f : float
            the frequency [Hz] to evaluate the gradient at
        metric : {"mag", "phase", "real", "imag"}
            the scalar projection of the complex transfer ``H``: ``"mag"``
            (default, ``|H|``; ``"gain"`` is an alias), ``"phase"`` (radians),
            ``"real"`` or ``"imag"``

        Returns
        -------
        sane.analysis.Sensitivity
            the gradient over every parameter, labeled ``metric(H)@fHz``

        Notes
        -----
        Exact via the AC adjoint (two complex solves plus one DC adjoint) -- no
        finite differences.
        """
        names, dH = self._grad_complex(f)
        H = complex(np.atleast_1d(self._response(np.array([float(f)])))[0])
        g, y0 = self._project(metric, H, dH)
        return Sensitivity(names, list(g), f"{metric}(H)@{f:g}Hz", p0=self._p, y0=y0)

    def hessian(self, f, wrt, metric="mag"):
        """Exact analytic AC Hessian of the metric at frequency ``f`` w.r.t. the
        parameter subset ``wrt`` (the knobs).

        Computed by the second-order adjoint on the combined DC+AC system; the
        operating-point shift is included exactly.

        Parameters
        ----------
        f : float
            the frequency [Hz] to evaluate the Hessian at
        wrt : list[str]
            the parameter names to form the Hessian over (the identified knobs)
        metric : {"mag", "phase", "real", "imag"}
            the scalar projection of ``H``: ``"mag"`` (default, ``|H|``;
            ``"gain"`` is an alias), ``"phase"`` (radians), ``"real"`` or
            ``"imag"``

        Returns
        -------
        numpy.ndarray
            the dense symmetric ``len(wrt) x len(wrt)`` Hessian matrix

        Notes
        -----
        Exact analytic second derivative -- no finite differences.
        """
        wrt = list(wrt)
        vre, vim, g_re, g_im, h_re, h_im = self._dae._d.ac_hessian(
            self._input, self._out_idx, list(self._x), self._p, float(f), wrt
        )
        g_re = np.array(g_re)
        g_im = np.array(g_im)
        h_re = np.array(h_re)
        h_im = np.array(h_im)
        if metric == "real":
            H = h_re
        elif metric == "imag":
            H = h_im
        elif metric in ("mag", "gain"):
            m = np.hypot(vre, vim)
            mp = (vre * g_re + vim * g_im) / m
            H = (np.outer(g_re, g_re) + vre * h_re + np.outer(g_im, g_im) + vim * h_im) / m
            H = H - np.outer(mp, mp) / m
        elif metric == "phase":
            m2 = vre * vre + vim * vim
            num = vre * g_im - vim * g_re  # m2 * d(phase)
            a = np.outer(g_im, g_re) - np.outer(g_re, g_im) + vre * h_im - vim * h_re
            m2g = 2.0 * (vre * g_re + vim * g_im)
            H = a / m2 - np.outer(num, m2g) / m2**2
        else:
            raise ValueError(f"unknown AC metric '{metric}' (use mag/phase/real/imag)")
        return 0.5 * (H + H.T)

    def __repr__(self):
        return f"AcResponse({self._input}->{self._output}, {len(self.freqs)} freqs)"


# SMALL-SIGNAL MODEL ====================================================================

class SmallSignal:
    """An operating-point-linearized small-signal model of the circuit.

    The circuit is linearized at its DC operating point into the descriptor LTI
    system ``C x' = -G x + B u`` with ``y = x[output]``, where ``G = dF/dx``,
    ``C = dF/dx'`` and ``B = -dF/d(input)`` are the exact (autodiff) Jacobians
    at the bias point. From it you get the poles and the AC response.

    The poles, zeros and AC response are computed by the **native** engine
    (passing the compiled Model, the bias point and the parameter vector); no
    linear algebra runs in Python, so the results never drift from the engine's
    own pole-zero / AC analyses. The :attr:`G` / :attr:`C` / :attr:`B` matrices
    are exposed for inspection only.

    Attributes
    ----------
    G, C : numpy.ndarray
        the small-signal conductance and capacitance/mass matrices, ``(n, n)``
    B : numpy.ndarray
        the input-coupling column, ``(n,)``
    out_index : int
        index of the output unknown

    Example
    -------

    .. code-block:: python

        ss = model.small_signal("V1", "out")
        ss.poles()
        mag_db, phase = ss.bode(np.logspace(1, 5, 50))
        ps = ss.pole_sensitivity()    # (pole, Sensitivity) per finite pole
    """

    def __init__(self, G, C, B, out_index, unknowns, node_names, input_name, output,
                 core=None, x=None, p=None):
        self.G = np.asarray(G, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.out_index = out_index
        self.unknowns = list(unknowns)
        self.node_names = list(node_names)
        self.input_name = input_name
        self.output = output
        # Native-analysis handle: the compiled Model plus the bias / parameter
        # vectors the engine needs to recompute G/C/B itself.
        self._core = core
        self._x = list(x) if x is not None else None
        self._p = list(p) if p is not None else None

    @staticmethod
    def _sorted_complex(pairs):
        w = np.array([complex(re, im) for re, im in pairs], dtype=complex)
        return w[np.argsort(w.imag + 1j * w.real)]

    def _require_native(self):
        if self._core is None:
            raise RuntimeError(
                "this SmallSignal has no engine handle; build it via Model.small_signal()"
            )

    def poles(self):
        """The poles of the linearized system.

        The finite generalized eigenvalues of the pencil ``(G, C)``, computed by
        the native engine (the standard-eigenproblem reduction with a QZ
        fallback); identical to :func:`sane.pole_zero`'s poles.

        Returns
        -------
        numpy.ndarray
            complex poles in rad/s, sorted by imaginary then real part
        """
        self._require_native()
        return self._sorted_complex(self._core.poles(self._x, self._p))

    def pole_sensitivity(self):
        """Exact analytic sensitivity ``ds/dp`` of every finite pole w.r.t. every
        parameter (all parameters at once, including the operating-point shift),
        by first-order eigenvalue perturbation with the all-parameter adjoint for
        the bias shift -- the same trick as the AC gradient, no finite differences.

        Returns
        -------
        list[tuple[complex, sane.analysis.Sensitivity]]
            one ``(pole, sensitivity)`` per finite pole. The sensitivity's
            ``.gradient`` is ``|ds/dp|`` (rank it to find the parameters that move
            the pole most); the full complex ``ds/dp`` (real = damping, imaginary =
            frequency) is on its ``.complex`` attribute.

        Notes
        -----
        First-order perturbation assumes simple (well-separated) poles; a repeated
        or defective pole would need the degenerate treatment.
        """
        self._require_native()
        out = []
        for (sre, sim), items in self._core.pole_gradient(self._x, self._p):
            names = [it[0] for it in items]
            cg = np.array([complex(it[1], it[2]) for it in items])
            s = Sensitivity(names, list(np.abs(cg)), f"pole {complex(sre, sim):.4g}", p0=self._p)
            s.complex = cg
            out.append((complex(sre, sim), s))
        return out

    def zeros(self):
        """The transmission zeros of the linearized system (frequencies where the
        input-to-output transfer vanishes).

        The finite generalized eigenvalues of the Rosenbrock system-matrix pencil,
        computed by the native engine (same eigensolver as :meth:`poles`).

        Returns
        -------
        numpy.ndarray
            complex zeros in rad/s, sorted by imaginary then real part
        """
        self._require_native()
        return self._sorted_complex(
            self._core.zeros(self.input_name, self.out_index, self._x, self._p)
        )

    def zero_sensitivity(self):
        """Exact analytic sensitivity ``ds/dp`` of every finite transmission zero
        w.r.t. every parameter (all at once, including the operating-point shift),
        by eigenvalue perturbation on the Rosenbrock pencil with the adjoint shift
        -- no finite differences.

        Returns
        -------
        list[tuple[complex, sane.analysis.Sensitivity]]
            one ``(zero, sensitivity)`` per finite transmission zero. The
            sensitivity's ``.gradient`` is ``|ds/dp|`` (rank it to find the
            parameters that move the zero most); the full complex ``ds/dp``
            (real = damping, imaginary = frequency) is on its ``.complex``
            attribute.

        Notes
        -----
        First-order perturbation assumes simple (well-separated) zeros.
        """
        self._require_native()
        out = []
        rows = self._core.zero_gradient(self.input_name, self.out_index, self._x, self._p)
        for (sre, sim), items in rows:
            names = [it[0] for it in items]
            cg = np.array([complex(it[1], it[2]) for it in items])
            s = Sensitivity(names, list(np.abs(cg)), f"zero {complex(sre, sim):.4g}", p0=self._p)
            s.complex = cg
            out.append((complex(sre, sim), s))
        return out

    def response(self, freqs_hz):
        """The small-signal transfer ``H(j2*pi*f)`` from input to output.

        Solved natively as ``e_out^T (G + jw C)^{-1} (-dF/d(input))`` at each
        frequency (the engine's AC kernel; no Python-side linear algebra).

        Parameters
        ----------
        freqs_hz : array_like
            frequencies in Hz

        Returns
        -------
        numpy.ndarray
            complex transfer function, one value per frequency
        """
        self._require_native()
        freqs = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
        pairs = self._core.ac_response(
            self.input_name, self.out_index, self._x, self._p, list(freqs)
        )
        return np.array([complex(re, im) for re, im in pairs], dtype=complex)

    def bode(self, freqs_hz):
        """Magnitude (dB) and phase (degrees) of :meth:`response`.

        Parameters
        ----------
        freqs_hz : array_like
            frequencies in Hz

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(magnitude_db, phase_deg)`` -- ``20*log10|H|`` and ``arg H`` in
            degrees, one value per frequency
        """
        h = self.response(freqs_hz)
        return 20.0 * np.log10(np.abs(h)), np.degrees(np.angle(h))

    def __repr__(self):
        return (f"SmallSignal(input={self.input_name!r}, output={self.output!r}, "
                f"dim={self.G.shape[0]})")


# NOISE SPECTRUM ========================================================================

class NoiseSpectrum:
    """Output-referred noise voltage spectral density versus frequency.

    The square-root PSD ``sqrt(S_v(f))`` [V/sqrt(Hz)] at the output, summed over
    every noise source in the circuit (resistor thermal noise plus any Verilog-A
    ``white_noise`` / ``flicker_noise`` / ``noise_table`` source), each weighted
    by its transimpedance to the output.

    Attributes
    ----------
    freqs : numpy.ndarray
        the frequency points [Hz]
    noise : numpy.ndarray
        the output noise density [V/sqrt(Hz)] at each frequency
    output : str
        the output node / unknown the noise is referred to

    Example
    -------

    .. code-block:: python

        ns = model.noise("out", 1.0, 1e6)
        floor = ns.noise            # V/sqrt(Hz) over ns.freqs
        vrms = ns.integrated_rms()  # total RMS noise [V] over the band
        sens = ns.sensitivity(1e3)  # dS_v/dp at 1 kHz, every parameter
    """

    def __init__(self, freqs, noise, output, dae=None, x=None, p=None, out_idx=None):
        self.freqs = np.asarray(freqs, dtype=float)
        self.noise = np.asarray(noise, dtype=float)
        self.output = output
        self._dae = dae
        self._x = list(x) if x is not None else None
        self._p = list(p) if p is not None else None
        self._out_idx = out_idx

    def sensitivity(self, f):
        """Exact analytic sensitivity of the output-noise PSD ``S_v(f)`` w.r.t.
        every parameter at frequency ``f``.

        Computed all at once (the operating-point shift is included) by the noise
        adjoint plus one extra solve and one DC adjoint, for white and flicker
        sources alike.

        Parameters
        ----------
        f : float
            the frequency [Hz] to evaluate the gradient at

        Returns
        -------
        sane.analysis.Sensitivity
            gradient of the PSD ``dS_v/dp`` (note: of ``S_v = noise**2``, the
            power density, not the square-root density), labeled ``S_v@fHz``.
            Rank it to find the parameters that set the noise floor.

        Notes
        -----
        Exact via the noise adjoint -- no finite differences.
        """
        if self._dae is None:
            raise RuntimeError("noise spectrum is not bound to a Model")
        nval, items = self._dae._d.noise_gradient(self._out_idx, self._x, self._p, float(f))
        names = [it[0] for it in items]
        grad = [it[1] for it in items]
        return Sensitivity(names, grad, f"S_v@{f:g}Hz", p0=self._p, y0=nval)

    def integrated_rms(self):
        """Total RMS noise [V] integrated over the band.

        Integrates the PSD ``S_v = noise**2`` over :attr:`freqs` by the
        trapezoidal rule and takes the square root.

        Returns
        -------
        float
            the band RMS noise [V] (``0.0`` if fewer than two frequency points)
        """
        if self.freqs.size < 2:
            return 0.0
        # np.trapz was renamed to np.trapezoid in NumPy 2.0
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        return float(np.sqrt(trapezoid(self.noise ** 2, self.freqs)))

    def __repr__(self):
        return f"NoiseSpectrum(output={self.output!r}, {self.freqs.size} points)"


# DESCRIPTOR STATE SPACE ================================================================

class StateSpace:
    """A linearized descriptor state-space model at the operating point.

    ``E x' = A x + B u``, ``y = C x + D u``, with ``E = dF/dx'``, ``A = -dF/dx``,
    ``B = -dF/d(input)``, ``C`` the output selector and ``D = 0`` -- the exact
    (autodiff) small-signal realization. ``E`` may be singular (descriptor form)
    for circuits with algebraic constraints.

    Attributes
    ----------
    states : list[str]
        the state (unknown) labels, in row / column order
    E, A : numpy.ndarray
        the descriptor and system matrices, ``(n, n)``
    B, C : numpy.ndarray
        the input column and output row, ``(n,)``
    D : float
        the feedthrough term
    input, output : str
        the input source and output node / unknown

    Example
    -------

    .. code-block:: python

        sys = model.state_space("V1", "out")
        E, A, B, C = sys.E, sys.A, sys.B, sys.C
        labels = sys.states           # state (unknown) labels, in matrix order
    """

    def __init__(self, states, E, A, B, C, D, input_name, output):
        self.states = list(states)
        self.E = np.asarray(E, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = float(D)
        self.input = input_name
        self.output = output

    def __repr__(self):
        return (f"StateSpace(input={self.input!r}, output={self.output!r}, "
                f"dim={len(self.states)})")


# TEMPERATURE SWEEP =====================================================================

class TempSweep:
    """An output node value swept over temperature.

    Attributes
    ----------
    temps_c : numpy.ndarray
        the converged sweep temperatures [deg C]
    values : numpy.ndarray
        the output value at each temperature
    output : str
        the output node / unknown

    Example
    -------

    .. code-block:: python

        sweep = model.temp_sweep("out", -40.0, 125.0)
        sweep.temps_c        # converged sweep temperatures [deg C]
        sweep.values         # output value at each temperature
    """

    def __init__(self, temps_c, values, output):
        self.temps_c = np.asarray(temps_c, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.output = output

    def __repr__(self):
        return f"TempSweep(output={self.output!r}, {self.temps_c.size} points)"


# REDUCED MODEL =========================================================================

class ReducedModel:
    """A dominant-pole model-order reduction of an input-to-output transfer.

    Keeps the most dominant poles / zeros and fits the gain to the full DC gain;
    carries the full vs. reduced magnitude over the band for fidelity checking.

    Attributes
    ----------
    freqs : numpy.ndarray
        the frequency points [Hz]
    full_db, reduced_db : numpy.ndarray
        the full and reduced magnitude responses [dB]
    poles, zeros : numpy.ndarray
        the kept poles / zeros [rad/s] (complex)
    max_err_db : float
        the worst-case ``|full - reduced|`` over the band [dB]

    Example
    -------

    .. code-block:: python

        rm = model.model_reduce("V1", "out", order=2, fstart=1.0, fstop=1e6)
        rm.poles             # the kept dominant poles [rad/s]
        rm.max_err_db        # worst-case fidelity error over the band [dB]
    """

    def __init__(self, freqs, full_db, reduced_db, poles, zeros, max_err_db):
        self.freqs = np.asarray(freqs, dtype=float)
        self.full_db = np.asarray(full_db, dtype=float)
        self.reduced_db = np.asarray(reduced_db, dtype=float)
        self.poles = np.array([complex(re, im) for re, im in poles], dtype=complex)
        self.zeros = np.array([complex(re, im) for re, im in zeros], dtype=complex)
        self.max_err_db = float(max_err_db)

    def __repr__(self):
        return (f"ReducedModel({self.poles.size} poles, {self.zeros.size} zeros, "
                f"max_err={self.max_err_db:.3g} dB)")


# HARMONIC BALANCE =====================================================================

class HarmonicBalance:
    r"""Periodic steady-state spectrum of a circuit, labeled by node / unknown.

    Harmonic balance solves for the periodic steady state directly in the
    frequency domain. Each unknown is written as a truncated Fourier series at
    the drive's fundamental :math:`\omega_0`,

    .. math::

        x_i(t) = \sum_{k=0}^{K} X_{i,k}\, e^{\,j k \omega_0 t} + \text{c.c.},

    and the time-domain Model residual :math:`F(x, \dot x, t) = 0` is enforced on
    every harmonic simultaneously: :math:`\hat F_k(\mathbf{X}) = 0` for
    :math:`k = 0 \dots K`. The :math:`k`-th harmonic of :math:`\dot x_i` is
    :math:`j k \omega_0 X_{i,k}`, so reactive elements are exact and frequency
    aliasing is controlled by the sample count (chosen from the residual's
    polynomial degree). This container holds the converged coefficients
    :math:`X_{i,k}`.

    Index by name to get a node's full complex spectrum
    (``hb["out"]`` -> array of :math:`X_{0..K}`); the per-harmonic frequencies
    are :attr:`freqs` (``k * f0``).

    Attributes
    ----------
    spectra : numpy.ndarray
        complex coefficients :math:`X_{i,k}`, shape ``(n, K+1)`` in
        :attr:`unknowns` row order, harmonic ``0..K`` per column
    unknowns : list[str]
        unknown names, in row order
    node_names : list[str]
        node names indexed by internal node id
    f0 : float
        fundamental frequency [Hz]
    harmonics : int
        the highest harmonic ``K`` retained
    freqs : numpy.ndarray
        the harmonic frequencies ``[0, f0, 2*f0, ..., K*f0]`` [Hz]
    converged : bool
        whether the harmonic-balance Newton reached tolerance
    iters : int
        Newton iterations taken (summed over continuation steps, if used)
    residual_norm : float
        the final harmonic-residual max-norm
    setup_ms, solve_ms : float
        the block-Jacobian compile time and the Newton-solve time [ms]

    Example
    -------

    .. code-block:: python

        hb = model.harmonic_balance(f0=1e3, harmonics=8)
        spec = hb["out"]              # complex X_{0..K} for a node
        amp1 = 2 * abs(hb.harmonic("out", 1))   # fundamental amplitude
        d = hb.thd("out")            # total harmonic distortion
        sens = hb.sensitivity("out", 1)         # dX_1/dp, every parameter
    """

    def __init__(self, spectra, unknowns, node_names, f0, harmonics,
                 converged, iters, residual_norm, setup_ms, solve_ms,
                 dae=None, p=None, oversample=16, samples=None):
        self.spectra = np.asarray(spectra, dtype=complex)
        self.unknowns = list(unknowns)
        self.node_names = list(node_names)
        self.f0 = float(f0)
        self.harmonics = int(harmonics)
        self.freqs = np.arange(self.harmonics + 1, dtype=float) * self.f0
        self.converged = bool(converged)
        self.iters = int(iters)
        self.residual_norm = float(residual_norm)
        self.setup_ms = float(setup_ms)
        self.solve_ms = float(solve_ms)
        # Context for the analytic sensitivity (reconstructs the AFT grid).
        self._dae = dae
        self._p = None if p is None else list(p)
        self._oversample = int(oversample)
        self._samples = None if samples is None else int(samples)

    def _row(self, ref):
        return resolve_unknown(ref, self.unknowns, self.node_names)

    def __getitem__(self, ref):
        """The full complex Fourier spectrum :math:`X_{0..K}` at ``ref``."""
        return self.spectra[self._row(ref)]

    def harmonic(self, ref, k):
        r"""The complex coefficient :math:`X_{\text{ref},k}` of harmonic ``k``.

        ``k = 0`` is the DC (bias) term; ``k = 1`` the fundamental. Magnitudes
        are the coefficient magnitudes :math:`|X_k|`; the physical amplitude of
        the ``k``-th sinusoid (``k >= 1``) is :math:`2|X_k|`.

        Parameters
        ----------
        ref : str | int
            node / unknown to take the coefficient of (see
            :func:`resolve_unknown`)
        k : int
            harmonic index (``0`` = DC/bias, ``1`` = fundamental, ...)

        Returns
        -------
        complex
            the Fourier coefficient :math:`X_{\text{ref},k}`
        """
        return complex(self.spectra[self._row(ref)][int(k)])

    def dc(self, ref):
        """The DC (bias) component :math:`X_0` at ``ref`` (a real value).

        Parameters
        ----------
        ref : str | int
            node / unknown to read (see :func:`resolve_unknown`)

        Returns
        -------
        float
            the real part of the DC coefficient :math:`X_0`
        """
        return float(self.spectra[self._row(ref)][0].real)

    def magnitude(self, ref):
        r"""The coefficient magnitudes :math:`|X_{0..K}|` at ``ref``.

        Parameters
        ----------
        ref : str | int
            node / unknown to read (see :func:`resolve_unknown`)

        Returns
        -------
        numpy.ndarray
            the ``K+1`` coefficient magnitudes, harmonic ``0..K``
        """
        return np.abs(self.spectra[self._row(ref)])

    def phase(self, ref, deg=True):
        r"""The coefficient phases :math:`\arg X_{0..K}` at ``ref``.

        Parameters
        ----------
        ref : str | int
            node / unknown to read (see :func:`resolve_unknown`)
        deg : bool
            return degrees (default) or radians if ``False``

        Returns
        -------
        numpy.ndarray
            the ``K+1`` coefficient phases, harmonic ``0..K``
        """
        ph = np.angle(self.spectra[self._row(ref)])
        return np.degrees(ph) if deg else ph

    def thd(self, ref):
        r"""Total harmonic distortion at ``ref``:

        .. math::

            \text{THD} = \frac{\sqrt{\sum_{k \ge 2} |X_k|^2}}{|X_1|}.

        The constant factor between coefficient and amplitude (``2``) cancels,
        so this is the standard amplitude-ratio THD.

        Parameters
        ----------
        ref : str | int
            node / unknown to read (see :func:`resolve_unknown`)

        Returns
        -------
        float
            the THD ratio, or ``inf`` if the fundamental :math:`|X_1|` is zero
        """
        m = np.abs(self.spectra[self._row(ref)])
        if m.shape[0] < 2 or m[1] == 0.0:
            return float("inf")
        return float(np.sqrt(np.sum(m[2:] ** 2)) / m[1])

    def sensitivity(self, ref, k, metric="coeff"):
        r"""Exact analytic sensitivity of harmonic ``k`` at ``ref`` w.r.t. every
        parameter (all at once, including the operating-point shift) -- no finite
        differences.

        The steady state satisfies :math:`\hat F(\mathbf{X}, p) = 0`, so
        :math:`d\mathbf{X}/dp = -J^{-1}\,\partial \hat F/\partial p`. The
        parameter Jacobian :math:`\partial F/\partial p` is routed through the
        same alternating frequency-time scheme as the residual, and the exact
        harmonic-balance Jacobian (the two-sided block-Toeplitz, recovering the
        conjugate coupling the Newton solver folds away) is solved once per
        output harmonic by an adjoint (transpose) solve.

        Parameters
        ----------
        ref : str
            node / unknown to take the coefficient of
        k : int
            harmonic index (``0`` = DC/bias, ``1`` = fundamental, ...)
        metric : {"coeff", "mag"}
            ``"coeff"`` (default): the complex coefficient :math:`X_{\text{ref},k}`.
            The returned :attr:`Sensitivity.gradient` is the rankable magnitude
            :math:`|dX_k/dp|`; the full complex :math:`dX_k/dp` is on the
            ``.complex`` attribute. ``"mag"``: the coefficient magnitude
            :math:`|X_{\text{ref},k}|`; gradient is the real
            :math:`d|X_k|/dp = \mathrm{Re}(\overline{X_k}\, dX_k/dp)/|X_k|`.

        Returns
        -------
        sane.analysis.Sensitivity
        """
        if self._dae is None or self._p is None:
            raise ValueError("sensitivity needs a bound circuit (use dae.harmonic_balance)")
        if metric not in ("coeff", "mag"):
            raise ValueError("metric must be 'coeff' or 'mag'")
        out_idx = self._row(ref)
        k = int(k)
        if not (0 <= k <= self.harmonics):
            raise ValueError(f"harmonic {k} out of range 0..{self.harmonics}")
        pairs = [[(complex(c).real, complex(c).imag) for c in row] for row in self.spectra]
        rows = self._dae._d.hb_gradient(
            out_idx, pairs, list(self._p), self.f0,
            harmonics=self.harmonics, oversample=self._oversample,
            samples=self._samples,
        )
        items = rows[k]
        names = [it[0] for it in items]
        cg = np.array([complex(it[1], it[2]) for it in items])
        xk = complex(self.spectra[out_idx][k])
        if metric == "coeff":
            s = Sensitivity(names, list(np.abs(cg)),
                            f"X[{ref},{k}]", p0=self._p, y0=abs(xk))
            s.complex = cg
            return s
        # metric == "mag": d|X_k|/dp = Re(conj(X_k) dX_k/dp) / |X_k|.
        m = abs(xk)
        grad = (np.real(np.conj(xk) * cg) / m) if m > 0.0 else np.zeros_like(cg, dtype=float)
        return Sensitivity(names, list(grad), f"|X[{ref},{k}]|", p0=self._p, y0=m)

    def hessian(self, ref, k, wrt, metric="coeff"):
        r"""Exact second-order-adjoint Hessian of harmonic ``k`` at ``ref`` w.r.t.
        the parameter subset ``wrt`` (the knobs) -- no finite differences.

        The steady state is an algebraic system in the harmonic coefficients, so
        the Hessian is recovered by one extra (second-order-adjoint) solve on the
        already-factored harmonic-balance Jacobian: the adjoint
        :math:`J^T\lambda = e_{(\text{out},k)}` and the forward state
        sensitivities :math:`J s_a = -\partial \hat F/\partial p_a` are contracted
        with the device second derivatives, all routed through the same AFT.

        Parameters
        ----------
        ref : str
            node / unknown
        k : int
            harmonic index
        wrt : list[str]
            parameter names to form the Hessian over (the identified knobs)
        metric : {"coeff", "mag"}
            ``"coeff"`` (default): the complex Hessian
            :math:`d^2 X_{\text{ref},k}/dp_a dp_b`. ``"mag"``: the real Hessian
            of the coefficient magnitude :math:`|X_{\text{ref},k}|`.

        Returns
        -------
        numpy.ndarray
            dense symmetric ``len(wrt) x len(wrt)`` matrix -- complex for
            ``"coeff"``, real for ``"mag"``

        Notes
        -----
        Exact also for nonlinear charge storage: a harmonic-balance unknown drives
        both :math:`x` and :math:`\dot x`, so the contraction includes the rate
        (:math:`\dot x`) Hessian blocks.
        """
        if self._dae is None or self._p is None:
            raise ValueError("hessian needs a bound circuit (use dae.harmonic_balance)")
        if metric not in ("coeff", "mag"):
            raise ValueError("metric must be 'coeff' or 'mag'")
        out_idx = self._row(ref)
        k = int(k)
        if not (0 <= k <= self.harmonics):
            raise ValueError(f"harmonic {k} out of range 0..{self.harmonics}")
        wrt = list(wrt)
        pairs = [[(complex(c).real, complex(c).imag) for c in row] for row in self.spectra]
        hc = np.array(
            self._dae._d.hb_hessian(
                out_idx, k, wrt, pairs, list(self._p), self.f0,
                harmonics=self.harmonics, oversample=self._oversample,
                samples=self._samples,
            ),
            dtype=object,
        )
        hc = np.array([[complex(re, im) for (re, im) in row] for row in hc], dtype=complex)
        if metric == "coeff":
            return hc
        # metric == "mag": project onto |X_k| using the complex gradient.
        xk = complex(self.spectra[out_idx][k])
        m = abs(xk)
        ns = len(wrt)
        if m == 0.0:
            return np.zeros((ns, ns))
        sc = self.sensitivity(ref, k, metric="coeff")
        gmap = dict(zip(sc.params, sc.complex))
        xa = np.array([gmap[name] for name in wrt], dtype=complex)
        xb = np.conj(xk)
        re_xb_xa = np.real(xb * xa)  # Re(conj(X) dX/dp_a)
        hmag = np.zeros((ns, ns))
        for a in range(ns):
            for b in range(ns):
                term1 = np.real(np.conj(xa[b]) * xa[a] + xb * hc[a][b]) / m
                term2 = re_xb_xa[a] * re_xb_xa[b] / m**3
                hmag[a][b] = term1 - term2
        return hmag

    def to_dict(self):
        """Return ``{unknown: complex spectrum}`` for every unknown.

        Returns
        -------
        dict[str, numpy.ndarray]
            one complex ``(K+1,)`` spectrum per unknown, keyed by unknown name
        """
        return {u: self.spectra[i] for i, u in enumerate(self.unknowns)}

    def plot(self, *refs, ax=None, db=False, show=False):
        """Stem-plot the harmonic magnitude spectrum of one or more signals.

        Parameters
        ----------
        *refs : str
            node / unknown names to plot; defaults to every unknown
        ax : matplotlib.axes.Axes, optional
            axis to draw on; a new one is created if omitted
        db : bool
            plot magnitudes in dB (``20*log10``) instead of linear
        show : bool
            call ``pyplot.show()`` after drawing

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        refs = refs or tuple(self.unknowns)
        for ref in refs:
            m = self.magnitude(ref)
            y = 20.0 * np.log10(np.maximum(m, 1e-300)) if db else m
            ax.stem(self.freqs, y, label=str(ref))
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("magnitude [dB]" if db else "magnitude")
        ax.legend()
        if show:
            plt.show()
        return ax

    def __repr__(self):
        status = "converged" if self.converged else "NOT converged"
        return (f"HarmonicBalance(f0={self.f0:.4g} Hz, K={self.harmonics}, "
                f"{status}, {self.iters} iters, ||R||={self.residual_norm:.2e})")
