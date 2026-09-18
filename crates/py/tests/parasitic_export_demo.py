"""From a giant post-extraction parasitic netlist to a small equivalent deck.

A parasitic extractor turns one wire into a long chain of tiny series resistors,
femtofarad caps to ground and high-value leaks. SANE reduces it *on the graph*
(short the wire segments -> merge nodes, the parasitic caps fuse in parallel;
open the leaks), then projects the transformations back onto a netlist. The
result is a small deck with the same terminal behavior -- re-importable.

    python crates/py/tests/parasitic_export_demo.py
"""

import numpy as np

import sane


def extracted_net(segments, seed=0):
    """A driver through an RC interconnect 'extracted' into `segments` tiny
    wire-resistor + parasitic-cap + leak stages, between input node `in` and the
    load. The intended circuit is just `Vin -> wire -> Cload`."""
    rng = np.random.default_rng(seed)
    lines = ["Vin 1 0 DC 1 AC 1", "Rdrv 1 2 50"]
    n = 2
    for _ in range(segments):
        lines.append(f"Rw{n} {n} {n+1} {rng.uniform(0.02, 0.08):.3f}")   # wire segment
        lines.append(f"Cp{n} {n+1} 0 {rng.uniform(1, 4):.2f}f")          # parasitic cap
        if n % 3 == 0:
            lines.append(f"Rlk{n} {n+1} 0 {rng.uniform(50, 300):.0f}meg")  # leak
        n += 1
    lines.append(f"Cload {n} 0 100f")
    lines.append(f"Rterm {n} 0 1meg")
    return "\n".join(lines), n


def count_elements(deck):
    return sum(1 for l in deck.splitlines() if l.strip() and l.strip()[0] not in "*.+")


def count_nodes(deck):
    nodes = set()
    for l in deck.splitlines():
        s = l.strip()
        if s and s[0] not in "*.+":
            t = s.split()
            nodes.update(t[1:3])
    return len(nodes)


def main():
    deck, out = extracted_net(segments=60)
    freqs = np.geomspace(1e3, 1e9, 30)

    dae = sane.Circuit.parse(deck).extract()
    red = dae.reduce(rel_tol=1e-3, freqs=freqs)
    new_deck = sane.reduced_netlist(deck, red.transforms)

    n_short = sum(1 for _, o in red.transforms if o == "short")
    n_open = sum(1 for _, o in red.transforms if o == "open")
    print("== parasitic netlist reduction ==")
    print(f"  original : {count_elements(deck):4d} elements, {count_nodes(deck):3d} nodes, dim {dae.dim}")
    print(f"  reduced  : {count_elements(new_deck):4d} elements, {count_nodes(new_deck):3d} nodes, dim {red.dim}")
    print(f"  transforms: {n_short} shorts (node merges) + {n_open} opens\n")

    # The regenerated deck re-extracts and matches the full terminal response.
    red2 = sane.Circuit.parse(new_deck).extract()
    surv = red2.node_names[-1] if red2.node_names[-1] != "0" else red2.node_names[-2]
    Hf = dae.small_signal("Vin", str(out)).response(freqs)
    Hr = red2.small_signal("Vin", surv).response(freqs)
    err = np.max(np.abs(20 * np.log10(np.abs(Hr)) - 20 * np.log10(np.abs(Hf))))
    print(f"  reduced-netlist vs full AC (1 kHz .. 1 GHz): max |dGain| = {err:.4f} dB")

    print("\n  reduced netlist:")
    for l in new_deck.splitlines():
        print("    " + l)


if __name__ == "__main__":
    main()
