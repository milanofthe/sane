"""SANE's symbolic engine and DAE graph manipulation, from Python.

Two halves:

1. A standalone symbolic-math layer: build expressions from symbols with the
   ordinary Python operators, differentiate, simplify, evaluate, and compile to
   a fast tape -- sympy-like, exact (hash-consed, exact rationals).
2. The extracted circuit DAE exposes its symbolic graph: residuals, Jacobians
   and the transfer function come back as manipulable expressions in a shared
   context, so you can differentiate / simplify / evaluate them in place.

    python crates/py/tests/symbolic_demo.py
"""

import numpy as np

import sane


def standalone():
    print("== standalone symbolic engine ==")
    x, y = sane.symbols("x y")

    f = 2 * x + (y ** 2).exp()
    print(f"  f            = {f}")
    print(f"  df/dx        = {f.diff(x)}")
    print(f"  df/dy        = {f.diff(y)}")
    print(f"  f(1, 0.5)    = {f.eval(x=1.0, y=0.5):.6f}  (= 2 + e^0.25)")

    # exact simplification: (z+1)/(z+1) -> 1
    z = sane.symbols("z")
    print(f"  (z+1)/(z+1)  = {((z + 1) / (z + 1)).simplify()}")

    # symbolic Jacobian + structural sparsity
    a, b = sane.symbols("a b")
    J = sane.jacobian([a * b, a + b], [a, b])
    print(f"  jacobian     = {[[str(e) for e in row] for row in J]}")
    print(f"  sparsity     = {sane.sparsity(J)}")

    # compile to a fast tape and evaluate at many points
    tape = sane.compile_tape([x * y, x + y], [x, y])
    print(f"  tape([3, 4]) = {tape.eval([3.0, 4.0])}")


def dae_graph():
    print("\n== DAE symbolic graph manipulation ==")
    dae = sane.Circuit.parse("V1 in 0 5\nR1 in out 1k\nC1 out 0 1u").extract()

    # the residual equations are manipulable expressions
    print("  residuals:")
    for r in dae.residuals:
        print(f"    {r}")

    # the transfer function comes back as an Expr -> simplify to canonical form
    H = dae.transfer_function("V1", "out")
    print(f"  H(s) raw      = {H}")
    print(f"  H(s) simplify = {H.simplify()}            (the RC low-pass)")

    # evaluate H(s) on the shared context at the -3 dB corner
    f3db = 1.0 / (2 * np.pi * 1e3 * 1e-6)
    h = H.eval_complex(s=2j * np.pi * f3db, R1=1e3, C1=1e-6)
    print(f"  |H(f_3dB)|    = {abs(h):.4f}                          (= 1/sqrt(2))")

    # differentiate a residual w.r.t. a node voltage, through the shared context
    v_out = dae.symbolic_context.sym("v2")
    print(f"  d(res[1])/d v_out = {dae.residuals[1].diff(v_out)}")

    # the symbolic system matrix A(s) = dF/dx + s dF/dx'
    print("  A(s):")
    for row in dae.system_matrix():
        print("    " + "  ".join(f"{str(e):>16}" for e in row))


def introspection():
    print("\n== circuit introspection ==")
    ckt = sane.Circuit.parse("V1 in 0 5\nR1 in out 1k\nC1 out 0 1u")
    for e in ckt.elements:
        print(f"  {e['name']:<3} {e['kind']:<14} {e['nodes']}")
    print(f"  nodes: {ckt.node_names}")


if __name__ == "__main__":
    standalone()
    dae_graph()
    introspection()
