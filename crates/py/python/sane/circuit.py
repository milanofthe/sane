#########################################################################################
##
##                              CIRCUIT BUILDER / FRONTEND
##                                    (circuit.py)
##
##              Ergonomic, named-node circuit construction on top of the
##           compiled `sane._core` extension. Build a circuit from a SPICE
##            netlist or programmatically, then `extract()` it into a Model.
##
#########################################################################################

# IMPORTS ===============================================================================

from . import _core
from .warnings import SaneConvergenceWarning, warn as _sane_warn


# GROUND NODE ALIASES ===================================================================

def _emit_captured_warnings(stacklevel=3):
    """Drain the engine's captured correctness warnings (e.g. out-of-range VA
    device parameters) and re-raise each unconditionally as a catchable
    :class:`~sane.warnings.SaneConvergenceWarning`, independent of the native
    log level (issue #54)."""
    for msg in _core.drain_warnings():
        _sane_warn(msg, SaneConvergenceWarning, stacklevel=stacklevel)

#: Node names that all map to the ground reference (internal node id ``0``).
GROUND_ALIASES = frozenset({"0", "gnd", "GND", "Gnd", "ground"})


# CIRCUIT CLASS =========================================================================

class Circuit:
    """A circuit, built either from a SPICE-like netlist or programmatically, and
    the entry point to SANE's symbolic analysis.

    Nodes are referenced by **name** (any string; the ground aliases in
    ``GROUND_ALIASES`` all map to the reference node). Element values are bound
    on construction and carried through to the extracted system, so you never
    assemble parameter vectors by hand -- pass a name/value, get back labeled
    results from the :class:`~sane.model.Model`.

    Two ways to build a circuit:

    1. From a netlist with :meth:`parse` (the full SPICE-parity front-end:
       ``.model`` cards, ``.param``, ``.subckt``, all device levels). This is the
       recommended path for anything with semiconductors.
    2. Programmatically with the element methods (:meth:`resistor`,
       :meth:`capacitor`, :meth:`voltage_source`, ...). Convenient for passive /
       linear networks; programmatic semiconductor models need their parameters
       passed explicitly (the netlist front-end is what supplies model defaults).

    Calling :meth:`extract` lowers the topology to the symbolic differential
    algebraic system ``F(x, x', t) = 0`` and returns a :class:`~sane.model.Model`,
    which carries every analysis (DC, transient, small-signal, sensitivity).

    Example
    -------

    A resistive divider from a netlist, solved for its DC operating point:

    .. code-block:: python

        import sane

        ckt = sane.Circuit.parse('''
            V1 in 0 5
            R1 in out 1k
            R2 out 0 1k
        ''')
        op = ckt.extract().operating_point()
        print(op["out"])     # 2.5

    The same divider built programmatically:

    .. code-block:: python

        ckt = sane.Circuit()
        ckt.voltage_source("V1", "in", "0", 5.0)
        ckt.resistor("R1", "in", "out", 1e3)
        ckt.resistor("R2", "out", "0", 1e3)
        op = ckt.extract().operating_point()

    Parameters
    ----------
    none
        Use :meth:`parse` for netlists, or construct an empty circuit and add
        elements with the builder methods.

    Attributes
    ----------
    node_names : list[str]
        node names indexed by internal node id; index ``0`` is ground
    values : dict[str, float]
        bound element/parameter values, keyed by symbol name (e.g. ``"R1"``,
        ``"D1.Is"``, ``"V1.sin_amp"``)

    Notes
    -----
    Every element-adding method (``resistor``, ``voltage_source``, ``diode``,
    ``bjt`` ...) returns ``self``, so construction calls can be chained.
    """

    def __init__(self):
        self._raw = _core.Circuit()
        self._names = ["0"]                 # node id -> name
        self._ids = {a: 0 for a in GROUND_ALIASES}
        self._values = {}                   # symbol name -> numeric value
        self._last = None                   # name of the most recently added element

    # --- construction from a netlist ---------------------------------------

    @classmethod
    def parse(cls, netlist):
        """Build a circuit by parsing a SPICE-like netlist.

        Runs the full front-end: the preprocessor (comments, line
        continuations, ``{expr}`` braces), ``.param`` expressions, ``.subckt``
        flattening, ``.model`` cards and device-level selection. Element values
        and model parameters found in the deck are bound automatically.

        **Verilog-A.** Inline a Verilog-A model in a
        ``.veriloga ... .endveriloga`` block and place it with an ``N``
        instance (``Nxxx node1 node2 ... modelname [param=value ...]``, in the
        module's port order). The analog block is compiled natively and lowered
        onto the same symbolic DAG, so the device runs through every analysis
        (DC, transient, AC, pole-zero, noise, sensitivity) exactly like a
        built-in model; instance parameters override the module defaults and are
        bound under the instance name (``"N1.Is"``). Constructs that cannot be
        lowered correctly raise a parse error rather than approximating.

        Real-world models that ship as files with companion headers load
        directly: ``.veriloga "path/to/model.va"`` reads each file and adds its
        own directory to the ``` `include ``` search path (the Accellera
        ``disciplines.vams`` / ``constants.vams`` are built in), so e.g. the EKV
        2.6, VBIC or HICUM compact models pull in their own includes and become
        usable by ``N`` instances under their module name. Relative paths
        resolve against the working directory.

        Parameters
        ----------
        netlist : str
            the netlist text; a trailing ``.end`` is optional

        Returns
        -------
        Circuit
            the parsed circuit, ready for :meth:`extract`

        Example
        -------

        .. code-block:: python

            ckt = sane.Circuit.parse('''
                Vcc 1 0 12
                R1 1 2 47k
                R2 2 0 10k
                Rc 1 3 4.7k
                Re 4 0 1k
                Q1 3 2 4 qm
                .model qm NPN(Is=1e-15 Bf=150 VAf=80)
            ''')

            # A Verilog-A model inline in the deck, instantiated with `N`:
            va = sane.Circuit.parse('''
                .veriloga
                module diode(a, c);
                  inout a, c;
                  electrical a, c;
                  parameter real Is = 1e-14;
                  analog I(a, c) <+ Is * (limexp(V(a,c)/0.025852) - 1.0);
                endmodule
                .endveriloga
                V1 1 0 0.7
                R1 1 2 1k
                N1 2 0 diode Is=2e-14
            ''')
        """
        raw = _core.parse(netlist if "\n.end" in netlist.lower() else netlist + "\n.end")
        _emit_captured_warnings()
        self = cls()
        self._raw = raw
        self._names = list(raw.node_names())
        self._ids = {a: 0 for a in GROUND_ALIASES}
        self._ids.update({n: i for i, n in enumerate(self._names)})
        return self

    # --- node bookkeeping --------------------------------------------------

    def _node(self, node):
        """Resolve a node reference (name or raw id) to an internal node id."""
        if isinstance(node, int):
            while len(self._names) <= node:
                self._names.append(str(len(self._names)))
            return node
        key = str(node)
        if key in self._ids:
            return self._ids[key]
        nid = len(self._names)
        self._ids[key] = nid
        self._names.append(key)
        return nid

    def _bind(self, name, value):
        # The value parameter is bound under a name kept clear of SANE's reserved
        # unknown namespace (so e.g. a source named "v91" does not collide with
        # node 91's voltage); the same mapping is applied during Model assembly.
        if value is not None:
            self._values[_core.value_symbol_name(name)] = float(value)
        self._last = name

    # --- passive / linear elements -----------------------------------------

    def resistor(self, name, n1, n2, value=None):
        """Add a resistor ``name`` between nodes ``n1`` and ``n2``.

        Parameters
        ----------
        name : str
            element name; also the symbol for its resistance (e.g. ``"R1"``)
        n1, n2 : str | int
            the two terminal node references: a node name (any string) or a raw
            integer node id; ``0``, ``"0"`` and the ``GROUND_ALIASES`` map to
            ground
        value : float, optional
            resistance in ohms; if omitted it stays a free symbol to be bound
            later via the analysis ``values`` argument

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.resistor(name, self._node(n1), self._node(n2))
        self._bind(name, value)
        return self

    def capacitor(self, name, n1, n2, value=None):
        """Add a capacitor ``name`` between nodes ``n1`` and ``n2``.

        Parameters
        ----------
        name : str
            element name; also the symbol for its capacitance (e.g. ``"C1"``)
        n1, n2 : str | int
            the two terminal node references; node ids are ints with ``0`` =
            ground (see :meth:`resistor`)
        value : float, optional
            capacitance in farads; if omitted it stays a free symbol to be
            bound later

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.capacitor(name, self._node(n1), self._node(n2))
        self._bind(name, value)
        return self

    def inductor(self, name, n1, n2, value=None):
        """Add an inductor ``name`` between nodes ``n1`` and ``n2``.

        Parameters
        ----------
        name : str
            element name; also the symbol for its inductance (e.g. ``"L1"``)
        n1, n2 : str | int
            the two terminal node references; node ids are ints with ``0`` =
            ground (see :meth:`resistor`)
        value : float, optional
            inductance in henries; if omitted it stays a free symbol to be
            bound later

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.inductor(name, self._node(n1), self._node(n2))
        self._bind(name, value)
        return self

    def voltage_source(self, name, n1, n2, dc=None):
        """Add an independent voltage source ``name`` from ``n1`` to ``n2``.

        Attach a time-domain waveform with :meth:`sine`, :meth:`pulse`,
        :meth:`exp` or :meth:`pwl` immediately afterwards.

        Parameters
        ----------
        name : str
            element name; also the symbol for its DC value (e.g. ``"V1"``)
        n1, n2 : str | int
            positive and negative terminal node references; node ids are ints
            with ``0`` = ground
        dc : float, optional
            DC source value in volts

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.voltage_source(name, self._node(n1), self._node(n2))
        self._bind(name, dc)
        return self

    def current_source(self, name, n1, n2, dc=None):
        """Add an independent current source ``name`` from ``n1`` to ``n2``.

        Positive current flows from ``n1`` through the source to ``n2``. Attach
        a time-domain waveform with :meth:`sine`, :meth:`pulse`, :meth:`exp` or
        :meth:`pwl` immediately afterwards.

        Parameters
        ----------
        name : str
            element name; also the symbol for its DC value (e.g. ``"I1"``)
        n1, n2 : str | int
            positive and negative terminal node references; node ids are ints
            with ``0`` = ground
        dc : float, optional
            DC source value in amperes

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.current_source(name, self._node(n1), self._node(n2))
        self._bind(name, dc)
        return self

    def vccs(self, name, out_p, out_n, ctrl_p, ctrl_n, gain=None):
        """Add a voltage-controlled current source (transconductance ``name``).

        Parameters
        ----------
        name : str
            element name; also the transconductance symbol (siemens),
            e.g. ``"G1"``
        out_p, out_n : str | int
            output (driven) node references (+, -); node ids are ints with
            ``0`` = ground
        ctrl_p, ctrl_n : str | int
            controlling (sensed) node references (+, -)
        gain : float, optional
            transconductance in siemens

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.vccs(name, self._node(out_p), self._node(out_n),
                       self._node(ctrl_p), self._node(ctrl_n))
        self._bind(name, gain)
        return self

    def vcvs(self, name, out_p, out_n, ctrl_p, ctrl_n, gain=None):
        """Add a voltage-controlled voltage source (voltage gain ``name``).

        Parameters
        ----------
        name : str
            element name; also the voltage-gain symbol (dimensionless),
            e.g. ``"E1"``
        out_p, out_n : str | int
            output (driven) node references (+, -); node ids are ints with
            ``0`` = ground
        ctrl_p, ctrl_n : str | int
            controlling (sensed) node references (+, -)
        gain : float, optional
            dimensionless voltage gain

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.vcvs(name, self._node(out_p), self._node(out_n),
                       self._node(ctrl_p), self._node(ctrl_n))
        self._bind(name, gain)
        return self

    def cccs(self, name, out_p, out_n, ctrl, gain=None):
        """Add a current-controlled current source (current gain ``name``).

        Parameters
        ----------
        name : str
            element name; also the current-gain symbol (dimensionless),
            e.g. ``"F1"``
        out_p, out_n : str | int
            output (driven) node references (+, -); node ids are ints with
            ``0`` = ground
        ctrl : str
            name of the voltage-defined element whose branch current is sensed
        gain : float, optional
            dimensionless current gain

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.cccs(name, self._node(out_p), self._node(out_n), ctrl)
        self._bind(name, gain)
        return self

    def ccvs(self, name, out_p, out_n, ctrl, gain=None):
        """Add a current-controlled voltage source (transresistance ``name``).

        Parameters
        ----------
        name : str
            element name; also the transresistance symbol (ohms), e.g. ``"H1"``
        out_p, out_n : str | int
            output (driven) node references (+, -); node ids are ints with
            ``0`` = ground
        ctrl : str
            name of the voltage-defined element whose branch current is sensed
        gain : float, optional
            transresistance in ohms

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.ccvs(name, self._node(out_p), self._node(out_n), ctrl)
        self._bind(name, gain)
        return self

    def mutual(self, name, l1, l2, k=None):
        """Couple two inductors with mutual coupling coefficient ``name``.

        The mutual inductance is ``M = k * sqrt(L1 * L2)``.

        Parameters
        ----------
        name : str
            coupling name; also the coupling-coefficient symbol ``k`` (0..1),
            e.g. ``"K1"``
        l1, l2 : str
            names of the two coupled inductors (e.g. ``"L1"``, ``"L2"``)
        k : float, optional
            coupling coefficient in ``0..1``

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.mutual(name, l1, l2)
        self._bind(name, k)
        return self

    # --- nonlinear device models -------------------------------------------

    def diode(self, name, anode, cathode, **params):
        """Add a junction diode ``name``.

        Parameters
        ----------
        name : str
            instance name (parameters are scoped as ``name.key``, e.g. ``"D1"``)
        anode, cathode : str | int
            the anode (+) and cathode (-) terminal node references; node ids are
            ints with ``0`` = ground
        **params : float
            model parameters, e.g. ``Is``, ``N``, ``Vt``. Programmatic devices
            do not inherit netlist model defaults, so supply what the model
            needs (or build via :meth:`parse` with a ``.model`` card).

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.diode(name, self._node(anode), self._node(cathode))
        self._bind_params(name, params)
        return self

    def mosfet(self, name, d, g, s, b=None, **params):
        """Add a MOSFET ``name`` (drain ``d``, gate ``g``, source ``s``).

        Parameters
        ----------
        name : str
            instance name (parameters scoped as ``name.key``, e.g. ``"M1"``)
        d, g, s : str | int
            drain, gate and source terminal node references, in that order;
            node ids are ints with ``0`` = ground
        b : str | int, optional
            bulk/body terminal; defaults to the source terminal when omitted
            (the common three-terminal connection).
        **params : float
            model parameters, e.g. ``Kp``, ``W``, ``L``, ``Vth``, ``lambda_``.
            ``W`` and ``L`` must both be nonzero (``k = Kp*W/L``). ``lambda_``
            is the Python-safe spelling of the SPICE ``lambda`` parameter.

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        body = s if b is None else b
        self._raw.mosfet(
            name, self._node(d), self._node(g), self._node(s), self._node(body)
        )
        self._bind_params(name, params)
        return self

    def bjt(self, name, c, b, e, **params):
        """Add a bipolar transistor ``name`` (collector ``c``, base ``b``,
        emitter ``e``).

        Parameters
        ----------
        name : str
            instance name (parameters scoped as ``name.key``, e.g. ``"Q1"``)
        c, b, e : str | int
            collector, base and emitter terminal node references, in that
            order; node ids are ints with ``0`` = ground
        **params : float
            model parameters, e.g. ``Is``, ``betaF``, ``betaR``, ``VAf``.

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.bjt(name, self._node(c), self._node(b), self._node(e))
        self._bind_params(name, params)
        return self

    def vswitch(self, name, n1, n2, ctrl_p, ctrl_n, **params):
        """Add a voltage-controlled switch ``name`` between ``n1`` and ``n2``,
        controlled by the voltage across ``ctrl_p``/``ctrl_n``.

        Parameters
        ----------
        name : str
            instance name (parameters scoped as ``name.key``, e.g. ``"S1"``)
        n1, n2 : str | int
            the switched terminal node references; node ids are ints with
            ``0`` = ground
        ctrl_p, ctrl_n : str | int
            the node references (+, -) whose voltage difference drives the
            switch
        **params : float
            model parameters (e.g. ``Vt``, ``Vh``, ``Ron``, ``Roff``)

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.vswitch(name, self._node(n1), self._node(n2),
                          self._node(ctrl_p), self._node(ctrl_n))
        self._bind_params(name, params)
        return self

    def cswitch(self, name, n1, n2, ctrl, **params):
        """Add a current-controlled switch ``name`` between ``n1`` and ``n2``,
        controlled by the branch current of element ``ctrl``.

        Parameters
        ----------
        name : str
            instance name (parameters scoped as ``name.key``, e.g. ``"W1"``)
        n1, n2 : str | int
            the switched terminal node references; node ids are ints with
            ``0`` = ground
        ctrl : str
            name of the voltage-defined element whose branch current drives it
        **params : float
            model parameters (e.g. ``It``, ``Ih``, ``Ron``, ``Roff``)

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.cswitch(name, self._node(n1), self._node(n2), ctrl)
        self._bind_params(name, params)
        return self

    def _bind_params(self, name, params):
        for key, value in params.items():
            # `lambda_` is the Python-safe spelling of the SPICE `lambda` param.
            spice_key = "lambda" if key == "lambda_" else key
            self._values[f"{name}.{spice_key}"] = float(value)
        self._last = name

    # --- time-domain source waveforms (attach to the last source) ----------

    def sine(self, offset=0.0, amplitude=1.0, freq=None, omega=None):
        """Give the most recently added source a sinusoidal waveform
        ``offset + amplitude * sin(omega * t)``.

        Parameters
        ----------
        offset : float
            DC offset
        amplitude : float
            sine amplitude
        freq : float, optional
            frequency in Hz (``omega = 2*pi*freq``)
        omega : float, optional
            angular frequency in rad/s; takes precedence over ``freq``

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        import math
        if omega is None:
            omega = 2.0 * math.pi * (freq or 0.0)
        self._raw.source_sin()
        self._values[f"{self._last}.sin_off"] = float(offset)
        self._values[f"{self._last}.sin_amp"] = float(amplitude)
        self._values[f"{self._last}.sin_w"] = float(omega)
        return self

    def pulse(self, v1, v2, delay=0.0, rise=0.0, fall=0.0, width=0.0, period=0.0):
        """Give the most recently added source a SPICE ``PULSE`` waveform.

        Parameters
        ----------
        v1, v2 : float
            initial and pulsed values
        delay : float
            delay before the first edge
        rise, fall : float
            rise and fall times
        width : float
            pulse width (time at ``v2``)
        period : float
            repetition period

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.source_pulse()
        n = self._last
        self._values.update({
            f"{n}.pulse_v1": float(v1), f"{n}.pulse_v2": float(v2),
            f"{n}.pulse_td": float(delay), f"{n}.pulse_tr": float(rise),
            f"{n}.pulse_tf": float(fall), f"{n}.pulse_pw": float(width),
            f"{n}.pulse_per": float(period),
        })
        return self

    def exp(self, v1, v2, td1=0.0, tau1=0.0, td2=0.0, tau2=0.0):
        """Give the most recently added source a SPICE ``EXP`` waveform
        (rising exponential at ``td1``/``tau1``, falling at ``td2``/``tau2``).

        Parameters
        ----------
        v1 : float
            initial value
        v2 : float
            pulsed value
        td1 : float
            rise delay (start of the rising exponential)
        tau1 : float
            rise time constant
        td2 : float
            fall delay (start of the falling exponential)
        tau2 : float
            fall time constant

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.source_exp()
        n = self._last
        self._values.update({
            f"{n}.exp_v1": float(v1), f"{n}.exp_v2": float(v2),
            f"{n}.exp_td1": float(td1), f"{n}.exp_tau1": float(tau1),
            f"{n}.exp_td2": float(td2), f"{n}.exp_tau2": float(tau2),
        })
        return self

    def pwl(self, points):
        """Give the most recently added source a piecewise-linear waveform.

        Parameters
        ----------
        points : list[tuple[float, float]]
            ``(time, value)`` breakpoints in increasing time order

        Returns
        -------
        Circuit
            ``self``, for chaining
        """
        self._raw.source_pwl(len(points))
        n = self._last
        for i, (t, v) in enumerate(points):
            self._values[f"{n}.pwl_t{i}"] = float(t)
            self._values[f"{n}.pwl_v{i}"] = float(v)
        return self

    # --- lowering ----------------------------------------------------------

    @property
    def node_names(self):
        """list[str]: node names indexed by internal node id (``0`` = ground)."""
        return list(self._names)

    @property
    def values(self):
        """dict[str, float]: bound element/parameter values, by symbol name
        (e.g. ``"R1"``, ``"D1.Is"``). Covers both construction paths: values from
        :meth:`parse` and any bound programmatically (the latter take precedence)."""
        merged = dict(self._raw.values())
        merged.update(self._values)
        return merged

    @property
    def elements(self):
        """list[dict]: the linear / controlled elements, each as
        ``{"name", "kind", "nodes", "control"}`` (node ids mapped back to names;
        ``control`` is the sensed element for F/H sources, else ``None``).
        Nonlinear devices are not included; see :attr:`device_count`."""
        names = self._names
        out = []
        for name, kind, a, b, ctrl in self._raw.elements():
            out.append({
                "name": name,
                "kind": kind,
                "nodes": (names[a] if a < len(names) else str(a),
                          names[b] if b < len(names) else str(b)),
                "control": ctrl,
            })
        return out

    @property
    def couplings(self):
        """list[dict]: inductive couplings, each as ``{"name", "inductors"}``."""
        return [{"name": n, "inductors": (l1, l2)} for n, l1, l2 in self._raw.couplings()]

    @property
    def device_count(self):
        """int: number of nonlinear device instances (D/M/Q/switches)."""
        return self._raw.device_count()

    def extract(self):
        """Extract the circuit as an analyzable :class:`~sane.model.Model`
        (the symbolic differential-algebraic system ``F(x, x', t) = 0`` with its
        analytic Jacobians).

        The returned object carries every analysis (DC operating point,
        transient, small-signal poles / AC response, and exact component
        sensitivity), all labeled by node and parameter name.

        Returns
        -------
        sane.model.Model
            the extracted system, with node names and bound values attached

        Example
        -------

        .. code-block:: python

            ckt = sane.Circuit.parse('''
                V1 in 0 5
                R1 in out 1k
                R2 out 0 1k
            ''')
            model = ckt.extract()
            op = model.operating_point()
            print(op["out"])     # 2.5
        """
        from .model import Model
        raw_model = self.extract_dae()
        _emit_captured_warnings()
        values = dict(raw_model.values())
        values.update(self._values)
        m = Model(raw_model, list(self._names), values)
        # power ports from deck `P` elements: (name, node, z0) per port, in
        # deck order; Model.sp() picks these up when no ports are passed
        m._deck_ports = list(self._raw.ports()) if hasattr(self._raw, "ports") else []
        return m

    def extract_dae(self):
        """Extract the raw compiled ``sane._core.Dae`` (positional API).

        The low-level escape hatch behind :meth:`extract`; prefer :meth:`extract`
        for the ergonomic, name-labeled :class:`~sane.model.Model`.

        Returns
        -------
        sane._core.Dae
            the raw compiled system (positional, unlabeled API)
        """
        return self._raw.extract_dae()

    def __repr__(self):
        return f"<sane.Circuit: {len(self._names) - 1} nodes, {len(self.values)} bound values>"
