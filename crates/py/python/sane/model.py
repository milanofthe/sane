#########################################################################################
##
##                          CIRCUIT MODEL / ANALYSIS FRONTEND
##                                    (model.py)
##
##              The differential-algebraic system extracted from a circuit,
##           and the ergonomic entry point to every analysis: DC operating
##            point, transient, small-signal poles / AC, and exact component
##                       sensitivity -- all labeled by name.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from . import _core
from .analysis import (
    OperatingPoint,
    Trajectory,
    Sensitivity,
    AcResponse,
    SmallSignal,
    NoiseSpectrum,
    StateSpace,
    TempSweep,
    ReducedModel,
    HarmonicBalance,
    label_unknown,
    resolve_unknown,
)
from .warnings import SaneConvergenceWarning, warn as _sane_warn


# MODEL CLASS ===========================================================================

class ParamNode:
    """A node in a :class:`Model`'s hierarchical parameter tree.

    Each node is a subcircuit instance or device group (e.g. ``model.X1`` or
    ``model.X1.D1``). Reaching a leaf through attribute (or item) access returns
    its value; reaching a deeper group returns another :class:`ParamNode`, so
    chains compose. Assigning to a leaf writes through to the authoritative
    (Rust) parameter store::

        model.X1.R2 = 1e-3      # set the value of R2 inside subcircuit X1
        r = model.X1.R2         # read it back (a float)
        model.X1.D1.Is = 2e-15  # nested groups chain

    Proxies are created lazily and memoized by :meth:`Model._node`, so attribute
    chains in a loop allocate nothing. You rarely construct one directly; the
    :class:`Model` hands them out on attribute access.

    Parameters
    ----------
    dae : Model
        the owning model, whose parameter store the node reads and writes
    prefix : str
        the dotted path of this group (e.g. ``"X1.D1"``); child names are
        appended to it to form fully-qualified parameter names
    """

    __slots__ = ("_dae", "_prefix")

    def __init__(self, dae, prefix):
        object.__setattr__(self, "_dae", dae)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        dae = object.__getattribute__(self, "_dae")
        full = object.__getattribute__(self, "_prefix") + "." + name
        canon = dae._resolve_name(full)
        if canon is not None:
            return dae._d.get_param(canon)
        if full in dae._prefixes:
            return dae._node(full)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        dae = object.__getattribute__(self, "_dae")
        full = object.__getattribute__(self, "_prefix") + "." + name
        canon = dae._resolve_name(full)
        if canon is not None:
            dae._d.set_param(canon, float(value))
        elif full in dae._prefixes:
            raise AttributeError(f"'{full}' is a parameter group, not a leaf")
        else:
            raise AttributeError(f"'{full}' is not a parameter")

    def __getitem__(self, name):
        dae = object.__getattribute__(self, "_dae")
        full = object.__getattribute__(self, "_prefix") + "." + name
        canon = dae._resolve_name(full)
        if canon is None:
            raise KeyError(full)
        return dae._d.get_param(canon)

    def __setitem__(self, name, value):
        dae = object.__getattribute__(self, "_dae")
        full = object.__getattribute__(self, "_prefix") + "." + name
        canon = dae._resolve_name(full)
        if canon is None:
            raise KeyError(full)
        dae._d.set_param(canon, float(value))

    def _children(self):
        dae = object.__getattribute__(self, "_dae")
        pre = object.__getattribute__(self, "_prefix") + "."
        kids = set()
        for p in dae._params:
            if p.startswith(pre):
                kids.add(p[len(pre):].split(".")[0])
        for grp in dae._prefixes:
            if grp.startswith(pre):
                kids.add(grp[len(pre):].split(".")[0])
        return sorted(kids)

    def __dir__(self):
        return self._children()

    def __repr__(self):
        pre = object.__getattribute__(self, "_prefix")
        return f"<sane.ParamNode {pre!r}: {self._children()}>"


class Model:
    """A circuit as an analyzable symbolic graph: the differential-algebraic
    system ``F(x, x', t) = 0`` with its analytic Jacobians, and the front-end to
    every analysis on it.

    Build one from :meth:`sane.circuit.Circuit.extract` or
    :meth:`Model.from_netlist`. It owns (shares) the symbolic context and the
    precomputed analytic Jacobians, and binds the element/parameter values from
    the circuit, so every analysis takes and returns names, never raw vectors:

    - :meth:`operating_point` -- DC bias, a labeled :class:`~sane.analysis.OperatingPoint`
    - :meth:`transient` -- time-domain response, a :class:`~sane.analysis.Trajectory`
    - :meth:`small_signal` -- poles and AC response, a :class:`~sane.analysis.SmallSignal`
    - :meth:`harmonic_balance` -- periodic steady-state spectrum, a :class:`~sane.analysis.HarmonicBalance`
    - :meth:`sensitivity` -- exact component sensitivity, a :class:`~sane.analysis.Sensitivity`

    Parameters can be overridden per call with a ``values`` dict
    (``{symbol: value}``); anything not overridden keeps the value bound from the
    circuit.

    Example
    -------

    .. code-block:: python

        model = sane.Circuit.parse("V1 in 0 5\\nR1 in out 1k\\nC1 out 0 1u").extract()

        op = model.operating_point()           # DC bias
        ss = model.small_signal("V1", "out")   # linearize at the bias
        print(ss.poles())                      # the RC pole

        traj = model.transient(np.linspace(0, 5e-3, 200))
        print(traj["out"][-1])

    Attributes
    ----------
    unknowns : list[str]
        unknown names ``x``, in residual / column order
    params : list[str]
        parameter names, in the order the internal ``p`` vector expects
    node_names : list[str]
        node names indexed by internal node id (``0`` = ground)
    values : dict[str, float]
        the parameter values bound from the circuit
    dim : int
        the system dimension ``len(unknowns)``
    transforms : list[tuple[str, str]]
        graph transformations ``(element, operation)`` if this model came from
        :meth:`reduce` (``"open"`` / ``"short"``); empty otherwise
    eliminated : list[str]
        node-unknown names removed by :meth:`eliminate`; empty otherwise
    """

    def __init__(self, raw, node_names, values):
        self._d = raw
        self._names = list(node_names)
        self._unknowns = list(raw.unknowns())
        self._params = list(raw.params())
        self._params_set = set(self._params)
        # Prefix set for hierarchical navigation: "X1.D1.Is" registers the
        # intermediate groups "X1" and "X1.D1". O(1) leaf-vs-group decisions.
        prefixes = set()
        for p in self._params:
            parts = p.split(".")
            for i in range(1, len(parts)):
                prefixes.add(".".join(parts[:i]))
        self._prefixes = prefixes
        # Memoized ParamNode proxies, one per group (no per-access allocation in
        # loops like `for v in xs: model.X1.R2 = v`).
        self._node_cache = {}
        # Rust holds the authoritative parameter store. The bound values reach us
        # merged (raw extraction + circuit overrides); push the ones that are
        # parameters into Rust and snapshot them for reset().
        known = {}
        for k, v in dict(values).items():
            canon = self._resolve_name(k)
            if canon is not None:
                known[canon] = float(v)
        if known:
            self._d.set_params(known)
        self._defaults = known
        #: graph transformations `(element, operation)` ("open"/"short") if this
        #: model came from :meth:`reduce`, else empty.
        self.transforms = []
        #: node-unknown names eliminated by :meth:`eliminate`, else empty.
        self.eliminated = []

    @classmethod
    def from_netlist(cls, netlist):
        """Build a :class:`Model` directly from a SPICE-like netlist string.

        Convenience shorthand for ``sane.Circuit.parse(netlist).extract()``: it
        parses the netlist into a :class:`~sane.circuit.Circuit` and extracts the
        differential-algebraic system in one step.

        Parameters
        ----------
        netlist : str
            a SPICE-like netlist (one element per line, ``\\n``-separated), e.g.
            ``"V1 in 0 5\\nR1 in out 1k\\nC1 out 0 1u"``

        Returns
        -------
        Model
            the extracted model, ready to analyze

        Example
        -------

        .. code-block:: python

            model = sane.Model.from_netlist("V1 in 0 5\\nR1 in out 1k\\nC1 out 0 1u")
            print(model.operating_point()["out"])
        """
        from .circuit import Circuit

        return Circuit.parse(netlist).extract()

    # --- hierarchical parameter access -------------------------------------

    def _resolve_name(self, name):
        """Canonical parameter name for a user name (handling the reserved
        ``_`` rename), or ``None`` if it is not a parameter."""
        if name in self._params_set:
            return name
        r = _core.value_symbol_name(name)
        return r if r in self._params_set else None

    def _node(self, prefix):
        """Memoized :class:`ParamNode` proxy for a parameter group."""
        node = self._node_cache.get(prefix)
        if node is None:
            node = ParamNode(self, prefix)
            self._node_cache[prefix] = node
        return node

    def __getattr__(self, name):
        # Only reached when normal attribute lookup fails, so methods and
        # properties are never shadowed. Resolves `model.X1` (group) / `model.R1`
        # (leaf value).
        if name.startswith("_") or "_params_set" not in self.__dict__:
            raise AttributeError(name)
        canon = self._resolve_name(name)
        if canon is not None:
            return self._d.get_param(canon)
        if name in self._prefixes:
            return self._node(name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        # Internal attributes (underscore) and everything set before the
        # parameter index exists go straight through. A leaf parameter name
        # writes to the Rust store; a group name is a usage error.
        if name.startswith("_") or "_params_set" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        canon = self._resolve_name(name)
        if canon is not None:
            self._d.set_param(canon, float(value))
        elif name in self._prefixes:
            raise AttributeError(
                f"'{name}' is a parameter group, not a leaf; set a child like model.{name}.<leaf>"
            )
        else:
            object.__setattr__(self, name, value)

    def __getitem__(self, name):
        canon = self._resolve_name(name)
        if canon is None:
            raise KeyError(name)
        return self._d.get_param(canon)

    def __setitem__(self, name, value):
        canon = self._resolve_name(name)
        if canon is None:
            raise KeyError(name)
        self._d.set_param(canon, float(value))

    def get(self, name):
        """Read a parameter's bound value (alias for ``model[name]``).

        Parameters
        ----------
        name : str
            a parameter name; accepts both the canonical form and the
            user-facing form of reserved names (see :meth:`_resolve_name`)

        Returns
        -------
        float
            the value currently bound in the parameter store; raises
            ``KeyError`` if ``name`` is not a parameter
        """
        return self[name]

    def set(self, name, value):
        """Set a parameter's value (alias for ``model[name] = value``).

        Parameters
        ----------
        name : str
            a parameter name (canonical or user-facing form)
        value : float
            the new value; coerced to ``float`` and written through to the
            (Rust) parameter store. Raises ``KeyError`` if ``name`` is not a
            parameter
        """
        self[name] = value

    def __contains__(self, name):
        return self._resolve_name(name) is not None

    def __len__(self):
        return len(self._params)

    def __iter__(self):
        return iter(self._params)

    def update(self, mapping=None, **kw):
        """Bulk-set parameters from a mapping and/or keywords.

        Resolves and validates every key before writing, so the update is atomic:
        if any name is not a parameter the store is left untouched.

        Parameters
        ----------
        mapping : dict[str, float], optional
            ``{name: value}`` parameter assignments
        **kw : float
            further assignments as keywords (merged over ``mapping``)

        Returns
        -------
        None
            mutates the model in place; raises ``KeyError`` on the first unknown
            name (nothing is set if any key is bad)
        """
        merged = dict(mapping or {})
        merged.update(kw)
        known = {}
        for k, v in merged.items():
            canon = self._resolve_name(k)
            if canon is None:
                raise KeyError(k)
            known[canon] = float(v)
        self._d.set_params(known)

    def reset(self):
        """Restore every parameter to its netlist/construction default.

        Re-applies the values snapshotted at construction (the merged raw
        extraction plus circuit overrides), undoing any later assignments made
        through attribute, item, :meth:`set` or :meth:`update` access.

        Returns
        -------
        None
            mutates the model in place
        """
        self._d.set_params(self._defaults)

    # --- introspection -----------------------------------------------------

    @property
    def unknowns(self):
        """The unknown names ``x`` in residual / column order (list[str])."""
        return list(self._unknowns)

    @property
    def params(self):
        """The parameter names in the order the internal ``p`` vector expects
        (list[str])."""
        return list(self._params)

    @property
    def unbound_params(self):
        """Parameters that appear in the residual but have no bound value
        (``list[str]``). A numeric analysis uses ``0`` for each -- which can
        silently bias the result -- while a symbolic analysis keeps them free.
        Typically a foundry ``.param``/global the deck never defined. Bind them
        with :meth:`set` / a ``values=`` override, or define the missing
        ``.param``. The first numeric solve also emits a warning listing these."""
        return list(self._d.unbound_params())

    @property
    def node_names(self):
        """The node names indexed by internal node id, ``0`` = ground
        (list[str])."""
        return list(self._names)

    @property
    def values(self):
        """The bound parameter values as a ``name -> value`` dict (from the
        authoritative Rust store)."""
        return self._d.values()

    def __dir__(self):
        # Tab-completion: the normal members plus the top-level parameter and
        # group segments (so `model.<TAB>` offers `X1`, `R1`, ...).
        base = set(super().__dir__())
        for p in self._params:
            base.add(p.split(".")[0])
        for pre in self._prefixes:
            base.add(pre.split(".")[0])
        return sorted(base)

    @property
    def dim(self):
        """The system dimension ``len(unknowns)`` (int)."""
        return self._d.dim()

    @property
    def profile(self):
        """Per-stage extraction timings as a list of ``(stage, milliseconds)``,
        in execution order: ``assemble`` (graph/residual build) followed by the
        ``compile/...`` sub-stages (sparse Jacobians, tape compilation, symbolic
        LU, block classification). Empty for models produced by :meth:`reduce` /
        :meth:`eliminate`, which rebuild only the solver, not the graph."""
        return list(self._d.profile)

    # --- internal helpers --------------------------------------------------

    #: Newton tolerance (max-norm of the residual) for the internal DC solve
    #: that seeds every analysis linearized at the operating point.
    DC_TOL = 1e-10
    #: maximum Newton iterations for that internal DC solve.
    DC_MAX_ITER = 100

    def _dc(self, p, x0=None, tol=None, max_iter=None, nodeset=None,
            reltol=None, abstol=None, vntol=None):
        """Solve the DC operating point in Rust and return the raw state vector.

        The shared bias-point seed for the small-signal, sensitivity and
        spectral analyses, so they all linearize at the identical operating
        point. Uses the homotopy ladder (Newton -> gmin -> source stepping);
        raises ``ValueError`` from the engine if it cannot converge.

        ``nodeset`` ({node_ref: value}) runs the stiff-pin symmetry-breaking
        phase first (for bistable circuits). ``reltol``/``abstol``/``vntol`` set
        the per-component convergence criterion (``tol`` is the absolute floor
        when they are left ``None``).
        """
        ns = None
        if nodeset:
            ns = [(self._idx(ref), float(v)) for ref, v in nodeset.items()]
        x = self._solve_dc_raw(p, x0, tol, max_iter, ns, reltol, abstol, vntol)
        if x0 is None and self._d.regularized_at_gmin() is not None:
            # Settle fallback (mirrors the Rust Model::solve_dc_reg): a
            # cold-start Newton on a high-gain feedback loop can converge onto
            # a gmin-held rail solution while the true floor point lives in
            # another basin (seen on the LM741 internals). The real dynamics
            # escape it through the circuit's own capacitances -- integrate
            # briefly and re-solve Newton from the settled state; adopt the
            # result only if it reaches the true DC floor.
            try:
                traj = self._d.solve_transient(p, [0.0, 0.06], None, 1e-4, 1e-7, None)
                x2 = self._solve_dc_raw(p, traj[-1], tol, max_iter, ns,
                                        reltol, abstol, vntol)
                if self._d.regularized_at_gmin() is None:
                    x = x2
            except Exception:
                pass  # keep the regularized point (and its warning)
        return x

    def _solve_dc_raw(self, p, x0, tol, max_iter, ns, reltol, abstol, vntol):
        return self._d.solve_dc(
            p,
            list(x0) if x0 is not None else None,
            self.DC_TOL if tol is None else tol,
            self.DC_MAX_ITER if max_iter is None else max_iter,
            nodeset=ns,
            reltol=reltol,
            abstol=abstol,
            vntol=vntol,
        )

    def _pvec(self, values=None):
        """Assemble the parameter vector in ``params`` order from the bound
        values, overridden by ``values``. Built in Rust (the reserved-namespace
        rename of override keys is applied there), so the per-solve assembly is
        not a Python list comprehension over every parameter.

        On the first numeric use, warn about parameters that appear in the
        residual but are unbound (a numeric analysis uses ``0`` for them, which
        can silently bias the result -- e.g. a foundry ``.param``/global the deck
        never defines). Symbolic analyses do not build this vector, so they keep
        such parameters free and are never warned."""
        if not getattr(self, "_warned_unbound", False):
            object.__setattr__(self, "_warned_unbound", True)
            unbound = self._d.unbound_params()
            if values:
                unbound = [u for u in unbound if u not in values]
            if unbound:
                import warnings
                shown = ", ".join(unbound[:30])
                more = f", ... (+{len(unbound) - 30} more)" if len(unbound) > 30 else ""
                warnings.warn(
                    f"{len(unbound)} parameter(s) referenced in the circuit are "
                    f"unbound; numeric analyses use 0 for them, which may bias the "
                    f"result (set them via .set()/values=, or define the missing "
                    f".param): {shown}{more}",
                    stacklevel=2,
                )
        return self._d.param_vector(values)

    def _idx(self, ref):
        return resolve_unknown(ref, self._unknowns, self._names)

    def unknown_index(self, ref):
        """Resolve a node / unknown / branch-current reference to its index in
        the state vector.

        Parameters
        ----------
        ref : str | int
            a node name (``"out"``), unknown name (``"v3"``) or source name for
            its branch current (``"V1"``)

        Returns
        -------
        int
            index into :attr:`unknowns`
        """
        return self._idx(ref)

    def unknown_name(self, ref):
        """Resolve a reference to its canonical unknown name (e.g. node ``"out"``
        -> ``"v3"``). See :meth:`unknown_index`."""
        return self._unknowns[self._idx(ref)]

    # --- DC ----------------------------------------------------------------

    def operating_point(self, values=None, x0=None, tol=1e-10, max_iter=100,
                        nodeset=None, reltol=None, abstol=None, vntol=None):
        r"""Solve the DC operating point (Newton + sparse LU, in Rust).

        The bias point is the steady state with all time derivatives zero,
        i.e. the root of the algebraic system

        .. math::

            F(x,\, \dot x = 0,\, t = 0) = 0,

        found by damped Newton iteration with the exact (autodiff) Jacobian
        :math:`\partial F/\partial x`, factored by sparse LU. A homotopy ladder
        (plain Newton :math:`\to` gmin stepping :math:`\to` source stepping)
        makes nonlinear circuits converge from a cold start: gmin stepping adds
        a vanishing shunt conductance to every node and relaxes it to zero;
        source stepping ramps the independent sources up from zero.

        Parameters
        ----------
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the state; defaults to the homotopy cold start
        tol : float
            absolute residual floor of the per-component criterion (see
            ``reltol``/``abstol``/``vntol`` for full control)
        max_iter : int
            maximum Newton iterations
        nodeset : dict[str, float], optional
            ``{node_ref: voltage}`` targets for the stiff-pin symmetry-breaking
            phase; breaks the metastable root of a bistable circuit (latch,
            symmetric differential pair) onto the selected branch
        reltol, abstol, vntol : float, optional
            per-component convergence: ``reltol`` is the relative update
            tolerance, ``abstol`` the absolute current floor (KCL rows /
            branch-current unknowns), ``vntol`` the absolute voltage floor (KVL
            rows / node-voltage unknowns)

        Returns
        -------
        sane.analysis.OperatingPoint
            the labeled DC state; raises ``ValueError`` if it does not converge

        Example
        -------

        .. code-block:: python

            model = sane.Circuit.parse("V1 in 0 5\\nR1 in out 1k\\nR2 out 0 1k").extract()
            op = model.operating_point()
            print(op["out"])              # node voltage by name
            print(op.sensitivity("out"))  # reuse the solved point, no re-solve
        """
        p = self._pvec(values)
        x = self._dc(p, x0, tol, max_iter, nodeset=nodeset,
                     reltol=reltol, abstol=abstol, vntol=vntol)
        # Gmin regularization flag (issue #54): read straight after the solve, so
        # it reflects this operating point. A converged=True but gmin-regularized
        # point is physically suspect; surface it unconditionally (catchable),
        # independent of sane.set_log_level. Two things earn the flag, and they
        # want different words: the solve never reached the gmin floor, or it
        # reached it and the floor is what sets an unknown's value.
        reg = self._d.regularized_at_gmin()
        if reg is not None:
            dom = self._d.gmin_dominance()
            if dom is not None:
                idx, shift = dom
                where = label_unknown(idx, self._unknowns, self._names)
                detail = (
                    f"removing the gmin={reg:.1e} shunt to ground would shift {where} by "
                    f"{shift:.1%} (first order), so its voltage is the regularization's answer "
                    "rather than the circuit's (that node needs a real path to ground)"
                )
            else:
                detail = (
                    f"it held only at gmin={reg:.1e} and never reached the true DC floor "
                    "(a high-impedance node is unstable at the floor)"
                )
            _sane_warn(
                f"DC operating point is gmin-regularized: {detail}. It is reported as "
                "converged but is physically suspect. Check op.regularized_at_gmin.",
                SaneConvergenceWarning,
            )
        return OperatingPoint(x, self._unknowns, self._names, dae=self, p=p,
                              regularized_at_gmin=reg)

    # --- transient ---------------------------------------------------------

    def transient(self, t, values=None, x0=None, rtol=1e-4, atol=1e-7, dt_max=None):
        r"""Integrate the circuit in the time domain (in Rust).

        Integrates the nonlinear differential-algebraic system

        .. math::

            F\big(x(t),\, \dot x(t),\, t\big) = 0, \qquad x(0) = x_0,

        with the stiffly-accurate, L-stable **ESDIRK32** method (implicit, so
        stiff circuits are stable): each stage solves an implicit nonlinear system
        for the new state by Newton with the exact Jacobian
        :math:`\partial F/\partial x + \alpha\, \partial F/\partial \dot x`
        (:math:`\alpha` the stage coefficient). The initial state defaults to the
        DC operating point. Time-domain source waveforms (set with
        :meth:`~sane.circuit.Circuit.sine` etc.) are honored.

        Parameters
        ----------
        t : array_like
            the time points to report the solution at
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial state; defaults to the DC operating point
        rtol, atol : float
            relative and absolute integration tolerances
        dt_max : float, optional
            maximum internal time step. The adaptive integrator can otherwise
            take steps far larger than the period of a fast time-varying source
            when the circuit's own dynamics are slow (the source enters
            algebraic, not differential, equations, so it barely moves the
            error estimate), silently aliasing the forced response to a flat
            line. Cap the step at a fraction of the shortest source period (a
            handful of steps per period) to resolve it. ``None`` leaves the step
            fully adaptive.

        Returns
        -------
        sane.analysis.Trajectory
            the labeled state versus time

        Example
        -------

        .. code-block:: python

            model = sane.Circuit.parse("V1 in 0 5\\nR1 in out 1k\\nC1 out 0 1u").extract()
            traj = model.transient(np.linspace(0, 5e-3, 200))
            print(traj["out"][-1])        # settled output voltage
        """
        t = np.asarray(t, dtype=float)
        p = self._pvec(values)
        traj = self._d.solve_transient(
            p, list(t), list(x0) if x0 is not None else None, rtol, atol, dt_max
        )
        return Trajectory(t, traj, self._unknowns, self._names,
                          dae=self, values=values, x0=x0)

    def transient_events(self):
        """The switching events of the most recent transient.

        Devices declare switching surfaces (Verilog-A ``@(cross ...)``, the
        thresholds of the ``S``/``W`` switches); the integrator lands a step on
        every crossing. Returns a list of ``(name, t, direction)`` with the
        surface name ``instance#k``, the crossing time and ``+1`` when the
        surface expression rose through zero, ``-1`` when it fell.
        """
        return list(self._d.transient_events())

    # --- small signal ------------------------------------------------------

    def small_signal(self, input, output, values=None, x0=None):
        """Linearize the circuit at its DC operating point into a small-signal
        model (poles + AC response).

        Assembles ``G = dF/dx``, ``C = dF/dx'`` and the exact input coupling
        ``B = -dF/d(input)`` (all by autodiff) at the bias point.

        Parameters
        ----------
        input : str
            the stimulus source name (a parameter, e.g. ``"V1"``)
        output : str
            the output node / unknown
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve

        Returns
        -------
        sane.analysis.SmallSignal
            the linearized model; use ``.poles()`` and ``.response(freqs)``

        Example
        -------

        .. code-block:: python

            model = sane.Circuit.parse("V1 in 0 5\\nR1 in out 1k\\nC1 out 0 1u").extract()
            ss = model.small_signal("V1", "out")
            print(ss.poles())             # the RC pole, in rad/s
            print(ss.response(np.geomspace(1, 1e6, 50)))
        """
        p = self._pvec(values)
        x = self._dc(p, x0)
        z = [0.0] * self.dim
        G = np.array(self._d.jacobian_x(x, z, p, 0.0))
        C = np.array(self._d.jacobian_xdot(x, z, p, 0.0))
        B = -np.array(self._d.input_jacobian(input, x, z, p, 0.0))
        # The poles / zeros / response are computed by the native engine (passing
        # the compiled model, the bias `x` and the parameter vector `p`); the G/C/B
        # matrices are handed out only for inspection, never re-analyzed in Python.
        return SmallSignal(G, C, B, self._idx(output), self._unknowns,
                           self._names, input, output,
                           core=self._d, x=list(x), p=p)

    def ac_transfer(self, input, output, freqs_hz, values=None):
        """Symbolic small-signal transfer ``H(j2*pi*f)`` from ``input`` to
        ``output`` via Cramer's rule on the symbolic system (exact for linear
        circuits, no operating-point solve).

        Parameters
        ----------
        input : str
            source parameter name
        output : str
            output node / unknown
        freqs_hz : array_like
            frequencies in Hz
        values : dict[str, float], optional
            parameter (and, for nonlinear circuits, operating-point) bindings;
            defaults to the values bound from the circuit

        Returns
        -------
        numpy.ndarray
            the complex transfer function, or ``None`` if the output is unknown
        """
        merged = self.values
        if values:
            merged.update(values)
        out_u = self._unknowns[self._idx(output)]
        pairs = self._d.ac_transfer(input, out_u, merged, list(np.atleast_1d(freqs_hz)))
        if pairs is None:
            return None
        return np.array([re + 1j * im for re, im in pairs])

    def ac(self, input, output, freqs_hz, values=None, x0=None):
        """Small-signal AC response as a result object, linearised at the DC
        operating point.

        Unlike :meth:`ac_transfer` (which returns the raw complex array), this
        returns an :class:`~sane.analysis.AcResponse` whose ``.value`` is the
        transfer over ``freqs_hz`` and which carries the unified derivative API:
        ``.sensitivity(f, metric)`` (gradient over every parameter, one AC adjoint
        solve) and ``.hessian(f, subset, metric)`` (sparse, over the knob subset).

        Parameters
        ----------
        input : str
            stimulus source name
        output : str
            output node / unknown
        freqs_hz : array_like
            frequencies in Hz
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve

        Returns
        -------
        sane.analysis.AcResponse
        """
        p = self._pvec(values)
        x = self._dc(p, x0)
        freqs = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
        return AcResponse(self, input, output, self._idx(output), x, p, freqs)

    # --- harmonic balance --------------------------------------------------

    def harmonic_balance(self, f0=0.0, harmonics=8, values=None, x0=None,
                         continuation=None, tol=1e-10, max_iter=60,
                         oversample=16, samples=None):
        r"""Periodic steady state by single-tone harmonic balance (AFT, in Rust).

        Solves directly for the periodic steady-state response to a sinusoidal
        drive at ``f0``, in the frequency domain -- no transient settling. Each
        unknown is a truncated Fourier series at the fundamental
        :math:`\omega_0 = 2\pi f_0`,

        .. math::

            x_i(t) = \sum_{k=-K}^{K} X_{i,k}\, e^{\,j k \omega_0 t},
            \qquad X_{i,-k} = \overline{X_{i,k}},

        and the unknowns are the coefficients :math:`\mathbf{X} = \{X_{i,k}\}`,
        :math:`k = 0 \dots K`. Harmonic balance enforces the time-domain DAE
        residual on every harmonic at once,

        .. math::

            \hat F_k(\mathbf{X})
              = \mathcal{F}_k\!\big[\, F(x(t), \dot x(t), t) \,\big] = 0,
            \qquad k = 0 \dots K,

        where :math:`\mathcal{F}_k` is the :math:`k`-th Fourier coefficient and
        :math:`\dot x_i` has coefficients :math:`j k \omega_0 X_{i,k}` (so
        capacitors and inductors are treated exactly, no time stepping).

        It is solved by the **alternating frequency-time** (AFT) scheme: given
        the current :math:`\mathbf{X}`, synthesize the time samples
        :math:`x(t_m)` by an inverse real FFT, evaluate the nonlinear residual
        and its Jacobians sample-by-sample on the compiled model tapes, then
        forward-FFT back to the harmonic residual :math:`\hat F_k` and its
        block Jacobian. Newton on :math:`\hat F(\mathbf{X}) = 0` then has the
        block structure

        .. math::

            \frac{\partial \hat F_k}{\partial X_l}
              = G_{k-l} + j\, l\, \omega_0\, C_{k-l},

        i.e. linear (LTI) entries stay harmonic-diagonal while each nonlinear
        device contributes a dense Toeplitz coupling between harmonics
        (:math:`G`, :math:`C` are the Fourier coefficients of :math:`dF/dx`,
        :math:`dF/dx'` along the waveform). The sample count is chosen from the
        residual's exact polynomial degree (alias-free for a polynomial
        nonlinearity, oversampled otherwise).

        The periodic drive comes from a ``SIN`` source set on the circuit (its
        frequency should equal ``f0``). The start is the DC operating point
        (all higher harmonics zero); for large signal swings that cross device
        regions, source-stepping continuation ramps the drive amplitude from
        zero (see ``continuation``).

        Parameters
        ----------
        f0 : float
            the fundamental (drive) frequency [Hz], ``> 0``
        harmonics : int
            the number of harmonics ``K`` to retain (besides DC); the spectrum
            has ``K + 1`` coefficients per unknown (``0..K``)
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            a converged DC operating-point seed (length :attr:`dim`); by default
            the engine solves the DC point itself
        continuation : bool or None
            ``None`` (default) tries a direct Newton from DC and falls back to
            source-stepping continuation if it does not converge; ``True`` always
            uses continuation (robust, slower); ``False`` never does
        tol : float
            harmonic-residual convergence tolerance (max-norm)
        max_iter : int
            maximum Newton iterations (per continuation step, if used)
        oversample : int
            time-sample oversampling factor used when the residual is not a
            finite-degree polynomial (transcendental / rational device models)
        samples : int, optional
            force the number of time samples per period (overrides the
            degree-based choice); must be at least ``2*K + 1``

        Returns
        -------
        sane.analysis.HarmonicBalance
            the labeled steady-state spectrum; use ``.magnitude(node)``,
            ``.thd(node)``, ``.harmonic(node, k)`` and ``.plot()``

        Examples
        --------

        .. code-block:: python

            model = sane.Circuit.parse('''
                V1 in 0 SIN(0.6 0.15 1000)
                R1 in mid 1k
                D1 mid 0 DMOD
                C1 mid 0 100n
                .model DMOD D(Is=1e-14 N=1 Vt=0.02585)
            ''').extract()

            hb = model.harmonic_balance(1000.0, harmonics=8)
            print(hb.thd("mid"))            # total harmonic distortion
            print(hb.magnitude("mid"))      # |X_0|, |X_1|, ..., |X_8|
        """
        # f0 <= 0 infers the fundamental from a periodic source (a SIN's frequency,
        # a PULSE train's 1/period); the engine returns the value it actually used.
        p = self._pvec(values)
        x_seed = list(x0) if x0 is not None else None
        spectra, conv, iters, rnorm, setup_ms, solve_ms, f0 = self._d.solve_hb(
            p, float(f0), harmonics=int(harmonics), x0=x_seed,
            oversample=int(oversample), tol=float(tol), max_iter=int(max_iter),
            samples=None if samples is None else int(samples),
            continuation=continuation,
        )
        spec = [[complex(re, im) for (re, im) in row] for row in spectra]
        return HarmonicBalance(spec, self._unknowns, self._names, f0, harmonics,
                               conv, iters, rnorm, setup_ms, solve_ms,
                               dae=self, p=p, oversample=int(oversample),
                               samples=None if samples is None else int(samples))

    # --- sensitivity -------------------------------------------------------

    def sensitivity(self, output, values=None, x0=None, t=0.0):
        r"""Exact first-order sensitivity ``dy/dp`` of the metric ``y = output``
        w.r.t. every parameter, by the adjoint method (one transpose solve).

        For a metric :math:`y = c^\top x` at a solved point :math:`F(x,p) = 0`,
        implicit differentiation gives
        :math:`\partial x/\partial p = -(\partial F/\partial x)^{-1}\,
        \partial F/\partial p`, so

        .. math::

            \frac{dy}{dp_k} = -\,\lambda^\top \frac{\partial F}{\partial p_k},
            \qquad
            \Big(\frac{\partial F}{\partial x}\Big)^{\!\top}\!\lambda = c.

        The single adjoint solve for :math:`\lambda` yields the gradient w.r.t.
        **all** parameters at once (one transpose solve, not one per parameter),
        and every Jacobian is exact symbolic autodiff -- no finite differences.

        Parameters
        ----------
        output : str
            the metric: a node / unknown name
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve
        t : float
            evaluation time (default 0, the DC point)

        Returns
        -------
        sane.analysis.Sensitivity
            the labeled gradient; supports ``.ranked()`` and ``.rollup()``, and
            ``.relative()`` since the metric value and parameter values are bound

        Example
        -------

        .. code-block:: python

            model = sane.Circuit.parse("V1 in 0 5\\nR1 in out 1k\\nR2 out 0 1k").extract()
            s = model.sensitivity("out")
            print(s.ranked()[:3])         # the parameters that move v_out most
        """
        p = self._pvec(values)
        x = self._dc(p, x0)
        return self._sensitivity_at(output, x, p, t)

    def _sensitivity_at(self, output, x, p, t=0.0):
        """First-order sensitivity of ``output`` at an already-solved point
        ``(x, p)`` (shared by :meth:`sensitivity` and
        :meth:`~sane.analysis.OperatingPoint.sensitivity`)."""
        out_u = self._unknowns[self._idx(output)]
        x = list(x)
        p = list(p)
        y0 = x[self._idx(output)]
        names, grad = self._d.sensitivity(out_u, x, p, t)
        return Sensitivity(names, grad, output, p0=p, y0=y0)

    def hessian(self, output, wrt, values=None, x0=None, t=0.0):
        """Exact second-order sensitivity (Hessian) of ``y = output`` w.r.t. the
        parameter subset ``wrt``, by the second-order adjoint with exact
        autodiff directional derivatives (no finite differences).

        Parameters
        ----------
        output : str
            the metric node / unknown
        wrt : list[str]
            the parameter names to form the Hessian over (the identified knobs)
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve
        t : float
            evaluation time

        Returns
        -------
        numpy.ndarray
            the dense symmetric ``len(wrt) x len(wrt)`` Hessian; the
            diagonal is curvature, the off-diagonals are knob interactions
        """
        p = self._pvec(values)
        x = self._dc(p, x0)
        return self._hessian_at(output, wrt, x, p, t)

    def _hessian_at(self, output, wrt, x, p, t=0.0):
        """Sparse second-order sensitivity of ``output`` over ``wrt`` at an
        already-solved point ``(x, p)`` (shared by :meth:`hessian` and
        :meth:`~sane.analysis.OperatingPoint.hessian`)."""
        out_u = self._unknowns[self._idx(output)]
        return np.array(self._d.hessian(out_u, list(wrt), list(x), list(p), t))

    # --- advanced sensitivity ---------------------------------------------

    def transient_sensitivity(self, params, t, rtol=1e-4, atol=1e-7, values=None):
        """Exact forward transient sensitivity ``dx(t)/dp`` for each parameter.

        The model is symbolically augmented with the exact sensitivity equations
        and integrated jointly with the circuit, so reactive parameters (C, L)
        are handled correctly.

        Parameters
        ----------
        params : list[str]
            the parameters to compute sensitivities for
        t : array_like
            the time points
        rtol, atol : float
            integration tolerances
        values : dict[str, float], optional
            per-call parameter overrides (used to finite-difference the gradient
            into a transient Hessian); defaults to the circuit's bound values

        Returns
        -------
        dict[str, sane.analysis.Trajectory]
            ``{param: trajectory of dx/dp}``; index a trajectory by node name,
            e.g. ``result["R1"]["out"]`` is ``d v_out / d R1`` over time
        """
        params = list(params)
        t = np.asarray(t, dtype=float)
        n, traj = self._d.transient_sensitivity(params, list(t), rtol, atol, values)
        traj = np.asarray(traj, dtype=float)
        out = {}
        for i, name in enumerate(params):
            block = traj[:, n + i * n : n + (i + 1) * n]
            out[name] = Trajectory(t, block, self._unknowns, self._names)
        return out

    def transient_grid(self, t, values=None, x0=None, dc_guess=None):
        """Fixed-grid ESDIRK32 transient on exactly the grid ``t``.

        One implicit step per interval, no substepping and no error control:
        this is the forward pass :meth:`transient_adjoint` differentiates,
        exposed so the SAME discrete objective can be evaluated
        (finite-difference checks, line searches in an optimizer). Same
        third-order method as :meth:`transient`, which additionally adapts the
        step and is what you want for plain simulation.

        Parameters
        ----------
        t : array_like
            the (strictly increasing) time grid; ``t[0]`` is the initial time
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial state; defaults to the DC operating point

        Returns
        -------
        sane.analysis.Trajectory
        """
        t = np.asarray(t, dtype=float)
        p = self._pvec(values)
        traj = self._d.solve_transient_grid(
            list(p), list(t), list(x0) if x0 is not None else None,
            list(dc_guess) if dc_guess is not None else None,
        )
        return Trajectory(t, traj, self._unknowns, self._names,
                          dae=self, values=values, x0=x0)

    def transient_adjoint(self, t, cotangent, values=None, dc_guess=None):
        """Gradient of a trajectory objective w.r.t. EVERY parameter, by the
        discrete adjoint (VJP) of the fixed-grid ESDIRK32 transient on ``t``.

        Given the cotangents ``dL/dx_k`` along the :meth:`transient_grid`
        trajectory, one backward transposed solve per step yields ``dL/dp``
        for all parameters at once -- cost independent of the parameter count
        (the complement of :meth:`transient_sensitivity`, whose cost is linear
        in the parameters but yields whole sensitivity trajectories). The
        initial state is the DC operating point; its parameter dependence is
        included through one DC transpose solve.

        Parameters
        ----------
        t : array_like
            the time grid (same convention as :meth:`transient_grid`)
        cotangent : dict[str, array_like] or array_like
            either ``{unknown_ref: dL/d(that signal) over t}`` (sparse, the
            common case -- e.g. ``{"out": w}`` for ``L = sum_k w_k out(t_k)``)
            or a full ``(len(t), dim)`` array of ``dL/dx_k``
        values : dict[str, float], optional
            per-call parameter overrides

        Returns
        -------
        dict[str, float]
            ``param name -> dL/dp``

        Examples
        --------
        Gradient of the tracking error ``L = sum_k (out(t_k) - ref_k)^2``::

            traj = model.transient_grid(t)
            r = traj["out"] - ref
            grad = model.transient_adjoint(t, {"out": 2.0 * r})
        """
        t = np.asarray(t, dtype=float)
        n = len(self._unknowns)
        if isinstance(cotangent, dict):
            cot = np.zeros((len(t), n))
            for ref, wk in cotangent.items():
                cot[:, self._idx(ref)] = np.asarray(wk, dtype=float)
        else:
            cot = np.asarray(cotangent, dtype=float)
            if cot.shape != (len(t), n):
                raise ValueError(
                    f"cotangent shape {cot.shape} != ({len(t)}, {n})"
                )
        names, grad = self._d.transient_adjoint(
            list(t), [list(row) for row in cot], values,
            list(dc_guess) if dc_guess is not None else None,
        )
        return dict(zip(names, grad))

    def transient_fn(self, output, t, wrt=None, warm=True):
        """A differentiable transient as a function object: ``f(p) -> waveform``
        with :meth:`~sane.differentiable.TransientFunction.vjp` for gradients.

        The returned object is the optimizer-facing form of the discrete
        adjoint -- and what :func:`sane.interop.as_torch` /
        :func:`sane.interop.as_jax` wrap into native autograd nodes.

        Parameters
        ----------
        output : str
            the unknown whose waveform ``f`` returns
        t : array_like
            the BE time grid
        wrt : list[str], optional
            the parameters ``f`` takes as inputs (default: every top-level
            parameter, i.e. names without a dot)

        Returns
        -------
        sane.differentiable.TransientFunction
        """
        from .differentiable import TransientFunction

        return TransientFunction(self, output, t, wrt=wrt, warm=warm)

    def dc_fn(self, output, wrt=None, warm=True):
        """A differentiable DC solve: ``f(p) -> float`` (the DC value of
        ``output``) with ``f.vjp()`` -> exact ``dy/dp`` over ``wrt`` through
        the DC adjoint. ``warm=True`` seeds each Newton with the previous
        operating point. See :class:`sane.differentiable.DcFunction`."""
        from .differentiable import DcFunction

        return DcFunction(self, output, wrt=wrt, warm=warm)

    def ac_fn(self, input, output, freqs, wrt=None, metric="mag", warm=True):
        """A differentiable AC sweep: ``f(p) -> |H|`` (or complex ``H`` with
        ``metric="complex"``) over ``freqs`` with ``f.vjp(dL_dH)`` through one
        all-parameter AC adjoint per frequency, operating-point shift
        included. See :class:`sane.differentiable.AcFunction`."""
        from .differentiable import AcFunction

        return AcFunction(self, input, output, freqs, wrt=wrt, metric=metric, warm=warm)

    def hb_fn(self, output, f0=0.0, harmonics=8, wrt=None, metric="mag",
              oversample=16, warm=True):
        """A differentiable harmonic balance: ``f(p) -> |X_k|`` (or complex
        spectrum) of ``output`` with ``f.vjp(dL_dX)`` through the
        implicit-function adjoint on the HB Jacobian. ``f0=0`` infers the
        fundamental from the periodic source. See
        :class:`sane.differentiable.HbFunction`."""
        from .differentiable import HbFunction

        return HbFunction(self, output, f0=f0, harmonics=harmonics, wrt=wrt,
                          metric=metric, oversample=oversample, warm=warm)

    def pz_fn(self, input, wrt=None, warm=True):
        """Differentiable pole locations: ``f(p) -> complex poles`` (sorted)
        with ``f.vjp(dL_ds)`` through the exact pole-migration sensitivities
        (one engine call per ``wrt`` parameter). See
        :class:`sane.differentiable.PzFunction`."""
        from .differentiable import PzFunction

        return PzFunction(self, input, wrt=wrt, warm=warm)

    # --- S-parameters ------------------------------------------------------

    def _sp_ports(self, ports, z0):
        """Resolve the (ports, z0) pair: explicit arguments win; otherwise the
        deck's `P` elements (in deck order) define both."""
        if ports is None:
            deck = getattr(self, "_deck_ports", [])
            if not deck:
                raise ValueError(
                    "no ports: pass ports=[(source, node), ...] or put P "
                    "elements in the deck (P1 in 0 Z0=50)"
                )
            return [(n, node) for n, node, _ in deck], (
                [zp for _, _, zp in deck] if z0 is None else z0
            )
        return ports, (50.0 if z0 is None else z0)

    def sp(self, freqs_hz, ports=None, z0=None, values=None, x0=None):
        """Small-signal scattering parameters of the network.

        Ports come from the deck's ``P`` elements (``P1 in 0 Z0=50``, in deck
        order) unless given explicitly. Port convention: an ideal V source in
        series with its reference impedance ``z0``, the port ``node`` on the
        network side (exactly what a ``P`` element lowers to). With that
        Thevenin form ``S_ij = 2*sqrt(z0_j/z0_i)*V_i - delta_ij`` under unit
        drive of source j, so each column is one AC transfer sweep (every
        other port source is dead by construction of the AC stimulus).

        Parameters
        ----------
        freqs_hz : array_like
            frequencies in Hz
        ports : list of (source, node), optional
            explicit ports: driving V-source name and network-side node
            (default: the deck's ``P`` elements)
        z0 : float or array_like, optional
            reference impedance, common or per port (default: the ``P``
            elements' Z0, or 50 for explicit ports)
        values, x0 : optional
            per-call parameter overrides / DC initial guess

        Returns
        -------
        sane.rf.SParams
        """
        from .differentiable import SpFunction
        from .rf import SParams

        ports, z0 = self._sp_ports(ports, z0)
        f = SpFunction(self, ports, freqs_hz, z0=z0, wrt=[], warm=False)
        if x0 is not None:
            f._x_warm, f.warm = list(x0), True
        s = f(dict(values or {}))
        return SParams(f.freqs, s, f.z0)

    def sp_fn(self, freqs, ports=None, z0=None, wrt=None, warm=True):
        """Differentiable S-parameters: ``f(p) -> S`` of shape ``(nf, n, n)``
        (complex) with ``f.vjp(dL_dS)`` (complex cotangent, ``dL/dRe +
        1j*dL/dIm`` per entry) through the AC adjoints of every port pair.
        Ports resolve as in :meth:`sp`. See
        :class:`sane.differentiable.SpFunction`."""
        from .differentiable import SpFunction

        ports, z0 = self._sp_ports(ports, z0)
        return SpFunction(self, ports, freqs, z0=z0, wrt=wrt, warm=warm)

    def ac_sensitivity(self, input, param, output, freqs_hz, values=None, x0=None):
        """Exact AC sensitivity ``dH/dp(f)`` of the transfer ``input -> output``
        w.r.t. ``param``, including the operating-point shift, via autodiff.

        Parameters
        ----------
        input : str
            stimulus source name
        param : str
            the parameter to differentiate w.r.t.
        output : str
            output node / unknown
        freqs_hz : array_like
            frequencies in Hz
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve

        Returns
        -------
        numpy.ndarray
            complex ``dH/dp`` at each frequency
        """
        p = self._pvec(values)
        x = self._dc(p, x0)
        oi = self._idx(output)
        freqs = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
        pairs = self._d.ac_sensitivity(input, param, oi, x, p, list(freqs))
        return np.array([complex(re, im) for re, im in pairs], dtype=complex)

    def pole_sensitivity(self, input, param, values=None, x0=None):
        """Exact pole sensitivity ``dlambda/dp`` w.r.t. ``param`` for every pole
        of the small-signal model, including the operating-point shift.

        Parameters
        ----------
        input : str
            stimulus source name (selects the small-signal model's input)
        param : str
            the parameter to differentiate w.r.t.
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve

        Returns
        -------
        list[tuple[complex, complex]]
            ``(pole, dpole/dp)`` for each finite pole, in rad/s
        """
        p = self._pvec(values)
        x = self._dc(p, x0)
        pairs = self._d.pole_sensitivity(input, param, x, p)
        return [(complex(*pole), complex(*dpole)) for pole, dpole in pairs]

    def zero_sensitivity(self, input, output, param, values=None, x0=None):
        """Exact transmission-zero sensitivity ``dz/dp`` w.r.t. ``param`` for
        every finite zero of the ``input -> output`` transfer, including the
        operating-point shift.

        Same first-order pencil perturbation as :meth:`pole_sensitivity`, but on
        the augmented Rosenbrock pencil ``(M, N)`` whose finite eigenvalues are
        the zeros (see :meth:`sane.analysis.SmallSignal.zeros`):
        ``dz = uᵀ(dM - z·dN)v / (uᵀ N v)``.

        Parameters
        ----------
        input : str
            stimulus source name
        output : str
            output node / unknown
        param : str
            the parameter to differentiate w.r.t.
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve

        Returns
        -------
        list[tuple[complex, complex]]
            ``(zero, dzero/dp)`` for each finite zero, in rad/s
        """
        p = self._pvec(values)
        x = self._dc(p, x0)
        oi = self._idx(output)
        pairs = self._d.zero_sensitivity(input, oi, param, x, p)
        return [(complex(*z), complex(*dz)) for z, dz in pairs]

    # --- whole-circuit analyses -------------------------------------------

    def noise(self, output, fstart, fstop, points=50, values=None, x0=None):
        """Output-referred noise spectral density vs frequency (native adjoint
        sweep, in Rust).

        Sums every noise source in the circuit -- resistor thermal noise
        (``4kT/R``), the native semiconductor sources (diode and BJT shot
        ``2qI`` + flicker, MOSFET channel thermal ``(8/3)kT*gm`` + flicker), and
        any Verilog-A ``white_noise`` / ``flicker_noise`` / ``noise_table``
        source -- each weighted by its transimpedance to the output, computed by
        one adjoint solve per frequency at the DC operating point.

        Parameters
        ----------
        output : str
            the output node / unknown the noise is referred to
        fstart, fstop : float
            the frequency band [Hz] (``0 < fstart < fstop``)
        points : int
            number of log-spaced frequency points
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve

        Returns
        -------
        sane.analysis.NoiseSpectrum
            the labeled noise density ``sqrt(S_v(f))`` [V/sqrt(Hz)]
        """
        out_idx = self._idx(output)
        p = self._pvec(values)
        x = self._dc(p, x0)
        f, nv = self._d.noise_raw(out_idx, list(x), p, float(fstart), float(fstop), int(points))
        return NoiseSpectrum(f, nv, output, dae=self, x=list(x), p=p, out_idx=out_idx)

    def state_space(self, input, output, values=None, x0=None):
        """Linearized descriptor state-space ``(E, A, B, C, D)`` at the DC
        operating point.

        ``E x' = A x + B u``, ``y = C x + D u`` with ``E = dF/dx'``,
        ``A = -dF/dx``, ``B = -dF/d(input)`` (exact autodiff), ``C`` the output
        selector, ``D = 0``.

        Parameters
        ----------
        input : str
            the stimulus source name (a parameter, e.g. ``"V1"``)
        output : str
            the output node / unknown
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve

        Returns
        -------
        sane.analysis.StateSpace
            the labeled descriptor realization
        """
        out_idx = self._idx(output)
        p = self._pvec(values)
        x = self._dc(p, x0)
        states, e, a, b, c, d = self._d.state_space_raw(input, out_idx, list(x), p)
        return StateSpace(states, e, a, b, c, d, input, output)

    def temp_sweep(self, output, tstart, tstop, points=50, values=None):
        """Sweep an output node over temperature (native, in Rust).

        Sets the global temperature symbol ``$temp`` [K] and re-solves the
        operating point at each step (warm-started). The temperature model lives
        in the device equations: the thermal voltage ``V_T = k*T/q``, the
        saturation current ``Is(T) = Is*(T/Tnom)^XTI*exp((Eg/(N*k/q))(1/Tnom-1/T))``
        and the mobility ``~(T/Tnom)^-1.5`` all track ``$temp`` symbolically for
        the diode / BJT / MOSFET / JFET / MESFET, so the dependence is exact and
        consistent across analyses (and yields the exact ``d(metric)/dT``).

        Parameters
        ----------
        output : str
            the output node / unknown
        tstart, tstop : float
            the temperature band [deg C]
        points : int
            number of temperature points
        values : dict[str, float], optional
            per-call parameter overrides for the nominal point

        Returns
        -------
        sane.analysis.TempSweep
            the labeled output vs temperature (only converged points)
        """
        out_idx = self._idx(output)
        p0 = self._pvec(values)
        temps, vals = self._d.temp_sweep_raw(out_idx, p0, float(tstart), float(tstop), int(points))
        return TempSweep(temps, vals, output)

    def model_reduce(self, input, output, order, fstart, fstop, points=50,
                     values=None, x0=None):
        """Dominant-pole model-order reduction of the ``input -> output`` transfer
        (native, in Rust).

        Keeps the ``order`` most dominant poles (and matching zeros) of the
        small-signal pencil at the DC operating point, fits the gain to the full
        DC gain, and reports the full vs. reduced magnitude over the band for
        fidelity checking. (This is the spectral MOR; for exact graph-level
        pruning see :meth:`reduce` / :meth:`eliminate`.)

        Parameters
        ----------
        input : str
            the stimulus source name
        output : str
            the output node / unknown
        order : int
            the number of poles to keep (``>= 1``)
        fstart, fstop : float
            the comparison frequency band [Hz]
        points : int
            number of log-spaced frequency points
        values : dict[str, float], optional
            per-call parameter overrides
        x0 : array_like, optional
            initial guess for the DC solve

        Returns
        -------
        sane.analysis.ReducedModel
            the kept poles / zeros and the full / reduced magnitude responses
        """
        out_idx = self._idx(output)
        p = self._pvec(values)
        x = self._dc(p, x0)
        f, full, red, poles, zeros, err = self._d.model_reduce_raw(
            input, out_idx, list(x), p, int(order), float(fstart), float(fstop), int(points)
        )
        return ReducedModel(f, full, red, poles, zeros, err)

    # --- model reduction (on the graph) -----------------------------------

    def reduce(self, rel_tol=1e-3, freqs=None, values=None, x0=None):
        """Operating-point-guided graph transformation, directly on the extracted
        graph (no netlist surgery, no re-extraction). Two dual operations:

        - **open** a branch (drop its terms): negligible current and admittance
          at the bias point -- a high-impedance parasitic.
        - **short** a branch (merge its two nodes): its admittance dominates the
          node so the voltage drop is forced negligible -- effectively a wire;
          this also drops the dimension.

        Both are applied consistently across the coupled nodes. The result is a
        smaller system -- fewer terms (and nodes) -- that stays parametric and
        keeps the surviving interface.

        Parameters
        ----------
        rel_tol : float
            per-node relative threshold below which a branch is pruned
        freqs : array_like, optional
            the frequency band (Hz) over which a branch's admittance is judged
            against its node's; a branch must be negligible at every frequency
            (and at DC) to be dropped. Default: 1 Hz .. 1 GHz.
        values : dict[str, float], optional
            per-call parameter overrides for the linearization point
        x0 : array_like, optional
            initial guess for the operating-point solve

        Returns
        -------
        Model
            the reduced system (validate it against this one with the usual
            analyses). Its ``.transforms`` attribute is the list of
            ``(element, operation)`` applied (``"open"`` / ``"short"``) -- reusable
            to map the reduction back to netlist transformations.
        """
        if freqs is None:
            freqs = np.geomspace(1.0, 1e9, 6)
        # angular frequencies, plus DC (ω = 0) so pure-conductance relevance is
        # judged too.
        omegas = [0.0] + [2.0 * np.pi * float(f) for f in np.atleast_1d(freqs)]
        p = self._pvec(values)
        x = self._dc(p, x0)
        reduced, transforms = self._d.prune_graph(rel_tol, x, p, omegas)
        # One operation per element (a shorted branch's now-zero term may also be
        # reported by the open pass); short takes precedence.
        seen, deduped = set(), []
        for elem, op in transforms:
            if elem not in seen:
                seen.add(elem)
                deduped.append((elem, op))
        out = Model(reduced, self._names, self.values)
        out.transforms = deduped
        return out

    def eliminate(self, keep=None):
        """Exactly eliminate internal resistive nodes from the graph by Gaussian
        (Schur) elimination -- the series/star-mesh reduction that collapses a
        chain of series resistors into a single branch, losslessly and
        frequency-independently.

        A node is eliminable only when it is purely resistive-linear at this
        point: no incident capacitance, no source or inductor anchored there, and
        a constant (voltage-independent) self-conductance. Source and nonlinear
        device nodes are therefore protected automatically; pass ``keep`` to also
        protect resistive probe nodes you want to read out. Because the
        elimination is exact, the response between all surviving nodes is
        preserved identically (to machine precision).

        This composes with :meth:`reduce`: ``reduce`` opens the negligible
        parasitic capacitors, which turns the interior into resistive chains that
        ``eliminate`` then collapses exactly -- the parasitic-network collapse.

        Parameters
        ----------
        keep : list[str], optional
            node / unknown references to protect from elimination (e.g. the
            probe nodes whose voltage you need). Sources and device nodes are
            already protected.

        Returns
        -------
        Model
            the reduced system; its ``.eliminated`` attribute lists the
            eliminated node-unknown names, in elimination order
        """
        keep_names = [self.unknown_name(r) for r in (keep or [])]
        reduced, gone = self._d.eliminate_nodes(keep_names)
        out = Model(reduced, self._names, self.values)
        out.transforms = list(self.transforms)
        out.eliminated = gone
        return out

    def fold(self, *paths):
        """Fold parameters to their current values: each becomes a constant in a
        derived :class:`Model` (sharing this context). Its now-constant
        subexpressions collapse (smaller graph, faster evaluation) and it leaves
        the parameter set, so :meth:`sensitivity` / :meth:`hessian` no longer
        build a ``dF/dp`` column for it -- folding the parameters you do not tune
        turns an all-parameter sensitivity from seconds into milliseconds.

        A path is a single parameter (``"X1.R1"``, ``"N1.vth0"``) or a group
        prefix (``"X1"`` -> every ``X1.*``, recursively). In the same transform
        family as :meth:`reduce` / :meth:`eliminate` / :meth:`linearize`; the
        master model is unchanged.

        Parameters
        ----------
        *paths : str | iterable[str]
            parameter or group paths, given variadically (``model.fold("X1",
            "R1")``) or as a list (``model.fold(["X1", "R1"])``)

        Returns
        -------
        Model
            the folded model (fold the static PDK, keep the tuning knobs symbolic)

        Example
        -------

        .. code-block:: python

            tuned = model.fold("X1")            # freeze subcircuit X1's parameters
            s = tuned.sensitivity("out")        # sensitivity over the few remaining knobs
        """
        flat = []
        for p in paths:
            if isinstance(p, str):
                flat.append(p)
            else:
                flat.extend(str(x) for x in p)
        raw = self._d.fold(flat)
        return Model(raw, self._names, self.values)

    def linearize(self, canonical=False):
        """Linearise about the operating point into the **small-signal linear
        mass-matrix DAE** :math:`G\\,\\delta x + C\\,\\delta\\dot x = 0`, sharing
        this context.

        The residuals become linear forms; the operating-point bias is frozen into
        constant ``name#op`` symbols, so the coefficients :math:`G=\\partial
        F/\\partial x` and :math:`C=\\partial F/\\partial\\dot x` are constant and
        the matrix assembly :math:`A(s)=G+sC` lives in the graph exactly as the
        nonlinear residual assembly does. :meth:`system_matrix` on the result
        reproduces this model's :math:`A(s)` exactly.

        Parameters
        ----------
        canonical : bool
            if ``True``, split each stamp to a single canonical small-signal
            element (``coef * port``) -- the element graph then lives directly in
            the returned model's stamps (one element per stamp). Otherwise the stamps
            mirror the source elements.

        Returns
        -------
        Model
            the linear small-signal system, in the same context (render it with
            :meth:`to_dot`, read :meth:`system_matrix` / :meth:`transfer_function`).
        """
        raw = self._d.linearize(canonical)
        return Model(raw, self._names, self.values)

    # --- symbolic graph access --------------------------------------------

    @property
    def symbolic_context(self):
        """The shared symbolic :class:`~sane.symbolic.Context` of this model, so
        the residual / Jacobian :class:`~sane.symbolic.Expr` graph can be
        manipulated, differentiated and compiled."""
        return self._d.context

    @property
    def residuals(self):
        """The residual equations ``F(x, x', t)`` as manipulable
        :class:`~sane.symbolic.Expr` (one per row)."""
        return self._d.residuals()

    def to_dot(self, labels=None, highlight=None):
        """Render the **whole** model -- every residual :math:`F_i(x,\\dot x,t)=0`
        -- as one Graphviz DOT graph of the shared hash-consed expression DAG.

        Each residual becomes a red endpoint box (named by its unknown); a
        subexpression shared across equations (a node voltage that appears in
        several KCL rows, the global ``$temp``) is drawn once with several
        parents, so hash-consing is visible across the system, not just within
        one row. This is the honest picture of the system: a reactive circuit
        shows both its derivative terms (a capacitor's ``vdot`` and an
        inductor's ``idot`` live in different residuals), not the apparent order
        of any single row. Feed the string to ``dot -Tpdf``.

        Parameters
        ----------
        labels : list[str], optional
            one label per residual; defaults to ``F[<unknown>]``.
        highlight : list[Expr], optional
            keep only the nodes reachable from these expressions at full
            opacity; fade the rest to alpha ~0.2.

        Returns
        -------
        str
            the Graphviz DOT source (an empty ``"digraph G {}"`` if the system
            has no residuals)
        """
        res = self.residuals
        if not res:
            return "digraph G {}\n"
        names = labels if labels is not None else [f"F[{u}]" for u in self.unknowns]
        return res[0].to_dot(label=names[0], others=list(res[1:]),
                             labels=list(names[1:]), highlight=highlight)

    def jacobian_x_symbolic(self):
        """The symbolic Jacobian ``dF/dx``.

        Returns
        -------
        list[list[Expr]]
            the dense matrix of :class:`~sane.symbolic.Expr` on the shared
            context (row ``i`` is the gradient of residual ``i`` w.r.t. ``x``)
        """
        return self._d.jacobian_x_symbolic()

    def jacobian_xdot_symbolic(self):
        """The symbolic Jacobian ``dF/dx'`` (the reactive / mass matrix).

        Returns
        -------
        list[list[Expr]]
            the dense matrix of :class:`~sane.symbolic.Expr` on the shared
            context (row ``i`` is the gradient of residual ``i`` w.r.t. ``x'``)
        """
        return self._d.jacobian_xdot_symbolic()

    def system_matrix(self):
        """The symbolic small-signal system matrix ``A(s) = dF/dx + s*dF/dx'``.

        Returns
        -------
        list[list[Expr]]
            the dense matrix of :class:`~sane.symbolic.Expr` on the shared
            context, with ``s`` a free Laplace symbol
        """
        return self._d.small_signal_matrix()

    #: closed-form symbolic transfer functions above this estimated term count
    #: are refused; the determinant expansion scales up to ``n!`` and a large
    #: expression is neither computable in reasonable time nor useful.
    SYMBOLIC_TERM_BUDGET = 200_000

    def term_estimate(self, cap=None):
        """Estimate the number of pre-cancellation terms in the symbolic transfer
        function (the permanent of the small-signal matrix pattern, capped).

        Cheap and bounded; the closed-form transfer function scales up to
        ``n!``, so check this before requesting it on a large circuit.

        Parameters
        ----------
        cap : int, optional
            stop counting once this many terms are reached; defaults to
            :attr:`SYMBOLIC_TERM_BUDGET`

        Returns
        -------
        int
            the estimated term count, clamped to ``cap``
        """
        return self._d.transfer_term_estimate(int(cap or self.SYMBOLIC_TERM_BUDGET))

    def _guard_symbolic(self):
        est = self._d.transfer_term_estimate(self.SYMBOLIC_TERM_BUDGET)
        if est >= self.SYMBOLIC_TERM_BUDGET:
            raise ValueError(
                f"symbolic transfer function would have >= {self.SYMBOLIC_TERM_BUDGET} "
                f"terms (the determinant expansion scales up to n!); this circuit is "
                f"too large for a closed form. Use the numeric AC response instead, or "
                f"transfer_approx with a coarse tolerance once during-generation "
                f"pruning is available.")

    def transfer_function(self, input, output):
        """The symbolic transfer function ``H(s)`` from ``input`` to ``output``
        as a single manipulable :class:`~sane.symbolic.Expr` (or ``None``).

        Raises ``ValueError`` if the estimated term count exceeds
        :attr:`SYMBOLIC_TERM_BUDGET` (the closed form scales up to ``n!``); call
        :meth:`term_estimate` first on larger circuits.

        Parameters
        ----------
        input : str
            source parameter name
        output : str
            output node / unknown

        Returns
        -------
        sane.symbolic.Expr | None
            ``H(s)``; ``.simplify()`` reduces it to canonical rational form
        """
        self._guard_symbolic()
        return self._d.transfer_function(input, self._unknowns[self._idx(output)])

    def transfer_approx(self, input, output, tol=1e-3, freq=1e3):
        """Analog-Insydes-style symbolic approximation of ``H(s)``.

        Collects the transfer function as polynomials in ``s``, ranks every
        monomial by its magnitude at the DC operating point and ``freq`` (Hz),
        drops those below ``tol`` of the dominant term, and rebuilds a compact
        symbolic ``H(s)``. Returns ``(H_pruned, terms_total, terms_kept)`` or
        ``None``; call ``.simplify()`` on the expression for a compact form.

        Raises ``ValueError`` if the *full* transfer function (which is built
        before pruning) would exceed :attr:`SYMBOLIC_TERM_BUDGET` terms.
        """
        self._guard_symbolic()
        out_u = self._unknowns[self._idx(output)]
        return self._d.transfer_approx(input, out_u, tol, freq)

    def transfer_approx_at(self, input, output, freq=0.0, tol=1e-3, cap=4000):
        """Single-frequency symbolic approximation of ``H(s)`` (Analog-Insydes
        "simplification before/during generation").

        Solves ``H`` numerically at the operating point and ``freq`` (Hz; ``0``
        for the DC gain), drops the tableau entries whose removal (a
        Sherman-Morrison rank-1 update) keeps the relative error in ``H`` below
        ``tol``, then generates the closed form by symbolic Gaussian elimination
        with numeric pivoting and a relative drop tolerance, so the determinant
        comes out in compact factored form, free of cancellation. ``cap`` is
        accepted for API compatibility but unused. Returns
        ``(H, kept_entries, total_entries, expr_nodes, H_value)`` or ``None``,
        where ``expr_nodes`` is the DAG size of ``H`` (its readability) and
        ``H_value`` is the reduced transfer at ``freq`` (a ``complex``) for
        self-checking against the full numeric response.
        """
        out_u = self._unknowns[self._idx(output)]
        res = self._d.transfer_approx_at(input, out_u, freq, tol, cap)
        if res is None:
            return None
        h, kept, total, terms, re, im = res
        return h, kept, total, terms, complex(re, im)

    def transfer_approx_named_at(self, input, output, freq=0.0, tol=1e-3, cap=4000):
        """Named-stamp single-frequency symbolic approximation of ``H(s)`` (the
        SLiCAP / Analog-Insydes "symbolic MNA" form).

        Like :meth:`transfer_approx_at`, but the surviving tableau entries are
        replaced by named admittance stamps (``y{row}_{col}``, excitation column
        ``b{row}``), so ``H`` is a compact rational function in those named
        symbols, human-readable no matter how complicated each transistor's
        expanded contribution is. Returns
        ``(H, legend, stamps, total_entries, expr_nodes, H_value)`` or ``None``,
        where ``legend`` is a ``dict`` mapping each stamp name appearing in ``H``
        to its complex value, ``stamps`` is the number of distinct stamps used,
        ``expr_nodes`` is the DAG size of ``H``, and ``H_value`` is the reduced
        transfer at ``freq`` (a ``complex``) for self-checking.
        """
        out_u = self._unknowns[self._idx(output)]
        res = self._d.transfer_approx_named_at(input, out_u, freq, tol, cap)
        if res is None:
            return None
        h, legend, stamps, total, terms, re, im = res
        leg = {name: complex(lre, lim) for name, lre, lim in legend}
        return h, leg, stamps, total, terms, complex(re, im)

    @property
    def nnz(self):
        """Number of structural nonzeros in the sparse ``dF/dx`` pattern."""
        return self._d.nnz()

    def partition_sizes(self):
        """Schur partition sizes ``(linear_block, nonlinear_block)`` if the
        solver partitioned the Jacobian (large mostly-linear systems), else
        ``None``."""
        return self._d.partition_sizes()

    # --- symbolic strings -------------------------------------------------

    def latex(self):
        """The residual equations as a LaTeX ``aligned`` block.

        Returns
        -------
        str
            a LaTeX string, one ``F_i(x, x', t) = 0`` row per residual
        """
        return self._d.to_latex()

    def transfer_latex(self, input, output):
        """Symbolic small-signal transfer ``H(s)`` as LaTeX, or ``None`` if the
        output unknown is not found. Raises ``ValueError`` if the estimated term
        count exceeds :attr:`SYMBOLIC_TERM_BUDGET`."""
        self._guard_symbolic()
        out_u = self._unknowns[self._idx(output)]
        return self._d.ac_transfer_latex(input, out_u)

    # --- escape hatch ------------------------------------------------------

    @property
    def core(self):
        """The underlying compiled ``sane._core.Model`` (raw, positional API).

        An escape hatch to the native object that backs every analysis; prefer
        the labeled methods on :class:`Model` over calling it directly.

        Returns
        -------
        sane._core.Model
            the wrapped Rust handle (``self._d``)
        """
        return self._d

    def __repr__(self):
        return f"<sane.Model: dim={self.dim}, {len(self._params)} parameters>"
