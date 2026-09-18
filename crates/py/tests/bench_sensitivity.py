"""Per-call cost of the sensitivity machinery, to see what is amortized by the
one-time extract (reusable compiled tapes) and what currently rebuilds symbolic
work each call. Relevant to parameter optimization / tuning, where the DAE graph
is fixed and only leaf (parameter) values change.

    python crates/py/tests/bench_sensitivity.py
"""

import time

import numpy as np

import sane

CE = """
    Vcc 1 0 12
    R1 1 2 47k
    R2 2 0 10k
    Rc 1 3 4.7k
    Re 4 0 1k
    Rp 3 0 10meg
    Q1 3 2 4 qm
    .model qm NPN(Is=1e-15 Bf=150 VAf=80)
"""


def timeit(fn, reps):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    t0 = time.perf_counter()
    dae = sane.Circuit.parse(CE).extract()
    t_extract = time.perf_counter() - t0

    out = "3"
    p = [dae.values.get(n, 0.0) for n in dae.params]
    n = dae.dim
    z = [0.0] * n
    x = dae.core.solve_dc(p, None, 1e-10, 100)
    sub = [name for name, _ in dae.sensitivity(out).ranked(relative=False)[:4]]

    print(f"circuit: dim={n}, params={len(dae.params)}")
    print(f"extract (one-time)        : {t_extract*1e3:8.2f} ms\n")
    print("per-call (graph already built, only leaf values change):")

    reps = 200
    t_solve = timeit(lambda: dae.core.solve_dc(p, None, 1e-10, 100), reps)
    t_jac = timeit(lambda: dae.core.jacobian_x(x, z, p, 0.0), reps)
    t_grad = timeit(lambda: dae.sensitivity(out), reps)
    t_hess = timeit(lambda: dae.hessian(out, sub), 50)

    print(f"  solve_dc (operating pt) : {t_solve*1e3:8.3f} ms   ({t_extract/t_solve:6.0f} solves per extract)")
    print(f"  jacobian_x (JAC eval)   : {t_jac*1e3:8.3f} ms")
    print(f"  sensitivity (grad, all p): {t_grad*1e3:8.3f} ms   (adjoint: 1 solve, all {len(dae.params)} params)")
    print(f"  hessian ({len(sub)}x{len(sub)} subset)    : {t_hess*1e3:8.3f} ms   <-- compiled tape (was ~70x slower as symbolic rebuild)")

    # The compiled Hessian tape is a fixed leaf-value re-eval: repeated calls
    # are constant-time, no symbolic rebuild, no context growth.
    print("\nrepeated hessian calls (compiled tape -> constant per-call, no drift):")
    for i in range(1, 6):
        t = timeit(lambda: dae.hessian(out, sub), 20)
        print(f"  call block {i}: {t*1e3:7.3f} ms")

    # Gradient-based parameter sweep: pure tape re-eval, fully amortized.
    print("\ngradient-based parameter sweep (100 random parameter sets):")
    rng = np.random.default_rng(0)
    p0 = np.array(p)
    t0 = time.perf_counter()
    for _ in range(100):
        pv = p0 * rng.uniform(0.8, 1.2, size=p0.shape)
        vals = {nm: float(v) for nm, v in zip(dae.params, pv)}
        g = dae.sensitivity(out, values=vals)
    dt = (time.perf_counter() - t0)
    print(f"  100 (solve + full gradient) sweeps: {dt*1e3:.1f} ms total, {dt*10:.3f} ms each")


if __name__ == "__main__":
    main()
