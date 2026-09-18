#########################################################################################
##
##                          NETLIST FROM GRAPH TRANSFORMATIONS
##                                   (netlist.py)
##
##            Project the graph reduction (open / short transformations on the
##         extracted DAE) back onto a netlist: a smaller deck with the same
##           behavior, the interpretable artifact of the reduction. Useful for
##                  re-importing a reduced post-extraction parasitic deck.
##
#########################################################################################

#: Token positions that hold node names, by element first letter (over-inclusive
#: where a type's node count varies, which only over-renames toward ground).
_NODE_POS = {
    "R": (1, 2), "C": (1, 2), "L": (1, 2), "D": (1, 2), "V": (1, 2), "I": (1, 2),
    "E": (1, 2, 3, 4), "G": (1, 2, 3, 4), "F": (1, 2), "H": (1, 2),
    "Q": (1, 2, 3), "M": (1, 2, 3, 4),
}

_GROUND = {"0", "gnd", "GND", "Gnd", "ground"}


def _node_positions(toks):
    t = toks[0][0].upper()
    if t == "X":  # subckt: all but the last token (the subckt name) are nodes
        return range(1, len(toks) - 1)
    if t == "K":
        return ()
    return _NODE_POS.get(t, (1, 2))


def reduced_netlist(netlist, transforms):
    """Apply graph transformations to a netlist, producing a smaller equivalent.

    Each ``("element", "open")`` deletes the element; each ``("element",
    "short")`` deletes it and merges its two nodes (a node rename everywhere).
    Ground is always kept as the merge survivor.

    Parameters
    ----------
    netlist : str
        the original deck
    transforms : list[tuple[str, str]]
        the ``(element, operation)`` list from :attr:`sane.model.Model.transforms`

    Returns
    -------
    str
        the reduced netlist
    """
    ops = {e for e, op in transforms if op == "open"}
    shorts = {e for e, op in transforms if op == "short"}

    # Union-find over node names; ground (and lower-named nodes) win as survivor.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # keep ground, else the smaller name, as the representative
        if rb in _GROUND or (ra not in _GROUND and rb < ra):
            ra, rb = rb, ra
        parent[rb] = ra

    kept = []
    for line in netlist.splitlines():
        s = line.strip()
        if not s or s[0] in "*.+":
            kept.append(line)
            continue
        toks = s.split()
        name = toks[0]
        if name in ops:
            continue
        if name in shorts:
            if len(toks) >= 3:
                union(toks[1], toks[2])
            continue
        kept.append(s)

    # Rename merged nodes everywhere.
    out = []
    for line in kept:
        s = line.strip()
        if not s or s[0] in "*.+":
            out.append(line)
            continue
        toks = s.split()
        for pos in _node_positions(toks):
            if pos < len(toks):
                toks[pos] = find(toks[pos])
        out.append(" ".join(toks))
    return "\n".join(out)
