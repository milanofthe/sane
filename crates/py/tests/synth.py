"""Synthetic circuit generators for extract-pipeline scaling studies.

Three structurally diverse families, each parameterized by a size knob so the
element / node count scales smoothly. They stress different axes of the extract
pipeline (a uniform RC ladder, by contrast, only exercises parser overhead):

- :func:`analog_array`   -- N replicated subcircuit cells (hierarchy flatten +
                            BJT expression cost), a cascade of gain stages.
- :func:`rc_mesh`        -- an m x m RC grid with a fraction of active drivers
                            (large linear parasitic bulk + nonlinear islands).
- :func:`inverter_chain` -- a CMOS inverter chain / ring (device-dense 1-D with
                            feedback), the digital-logic axis.

Each returns ``(deck, n_cells)``; the netlists target SANE's own device models
(BJT Ebers-Moll, MOSFET level 1), so they parse and extract natively.
"""


def analog_array(stages):
    """A cascade of `stages` identical BJT common-emitter gain cells, each a
    flattened subcircuit instance (stresses subckt expansion + the BJT, the
    heaviest symbolic device expression)."""
    lines = [
        ".subckt cell in out vcc",
        "Rc vcc out 4k",
        "Rb vcc in 100k",
        "Q1 out in 0 QN",
        ".model QN NPN(Is=1e-15 betaF=120 VAf=80)",
        ".ends",
        "Vcc 1 0 12",
        "Vin n0 0 AC 1 DC 0.65",
    ]
    # Cascade: each cell drives the next through a coupling cap.
    for k in range(stages):
        inn = f"n{k}"
        out = f"c{k}"
        nxt = f"n{k+1}"
        lines.append(f"X{k} {inn} {out} 1 cell")
        lines.append(f"Ccpl{k} {out} {nxt} 100n")
    lines.append(f"Rterm n{stages} 0 1meg")
    return "\n".join(lines), stages


def rc_mesh(side, driver_frac=0.1):
    """An `side` x `side` grid of resistors (horizontal + vertical) with a
    capacitor to ground at every node, plus a CMOS inverter driving a fraction
    `driver_frac` of the nodes (mostly-linear parasitic bulk with active
    islands). Node count ~ side^2."""
    def nd(i, j):
        return f"g{i}_{j}"

    lines = [
        "Vdd 1 0 1.8",
        "Vin din 0 AC 1 DC 0.9",
        ".model NM NMOS(Kp=200u Vth=0.4 W=2 L=1)",
        ".model PM PMOS(Kp=80u Vth=-0.4 W=4 L=1)",
    ]
    # Drive the top-left corner from the input through an inverter.
    rng = 1
    count = 0
    for i in range(side):
        for j in range(side):
            n = nd(i, j)
            lines.append(f"Cn{i}_{j} {n} 0 10f")
            if j + 1 < side:
                lines.append(f"Rh{i}_{j} {n} {nd(i, j+1)} 5")
            if i + 1 < side:
                lines.append(f"Rv{i}_{j} {n} {nd(i+1, j)} 5")
            # Active driver at a deterministic fraction of nodes.
            count += 1
            if driver_frac > 0 and (count * 997) % int(1 / driver_frac) == 0:
                lines.append(f"MN{i}_{j} {n} din 0 0 NM")
                lines.append(f"MP{i}_{j} {n} din 1 1 PM")
    # Tie the source node into the grid corner.
    lines.append(f"Rdrv din {nd(0, 0)} 50")
    return "\n".join(lines), side * side


def inverter_chain(stages, ring=False):
    """A chain of `stages` CMOS inverters, out_k -> in_{k+1}; if `ring`, the last
    output feeds the first (a ring oscillator). Device-dense, with feedback."""
    lines = [
        "Vdd 1 0 1.8",
        ".model NM NMOS(Kp=200u Vth=0.4 W=2 L=1)",
        ".model PM PMOS(Kp=80u Vth=-0.4 W=4 L=1)",
    ]
    if not ring:
        lines.append("Vin n0 0 AC 1 DC 0.9")
    for k in range(stages):
        inn = f"n{k}"
        out = f"n{k+1}" if (k + 1 < stages or not ring) else "n0"
        lines.append(f"MN{k} {out} {inn} 0 0 NM")
        lines.append(f"MP{k} {out} {inn} 1 1 PM")
        lines.append(f"Cl{k} {out} 0 5f")
    return "\n".join(lines), stages


GENERATORS = {
    "analog_array": analog_array,
    "rc_mesh": rc_mesh,
    "inverter_chain": inverter_chain,
}
