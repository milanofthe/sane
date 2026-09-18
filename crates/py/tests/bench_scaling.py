"""Scaling benchmark: a k x k resistor mesh (n = k^2 nodes) driven by one
source. Dense-ish connectivity stresses the sparse LU. Reports extract and DC
solve time, the system dimension, the Jacobian nonzeros and the Schur partition.

    python crates/py/tests/bench_scaling.py
"""

import time

import sane


def mesh_netlist(k):
    """A k x k grid of unit resistors; node (i,j) -> name n{i}_{j}. A 1 V source
    drives the corner, the opposite corner is grounded via a resistor."""
    lines = ["V1 n0_0 0 1"]
    rid = 0
    for i in range(k):
        for j in range(k):
            if j + 1 < k:
                lines.append(f"R{rid} n{i}_{j} n{i}_{j+1} 1k"); rid += 1
            if i + 1 < k:
                lines.append(f"R{rid} n{i}_{j} n{i+1}_{j} 1k"); rid += 1
    # tie the far corner to ground so the system is non-singular
    lines.append(f"Rg n{k-1}_{k-1} 0 1k")
    return "\n".join(lines)


def time_solve(dae, p, reps):
    dae.core.solve_dc(p, None, 1e-10, 100)  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        dae.core.solve_dc(p, None, 1e-10, 100)
    return (time.perf_counter() - t0) / reps


def bench(k, reps=5):
    deck = mesh_netlist(k)
    t0 = time.perf_counter()
    dae = sane.Circuit.parse(deck).extract()
    t_extract = time.perf_counter() - t0
    p = [dae.values.get(n, 0.0) for n in dae.params]

    sane._core.set_parallelism(1)              # sequential
    t_seq = time_solve(dae, p, reps)
    sane._core.set_parallelism(0)              # all rayon threads
    t_par = time_solve(dae, p, reps)

    return dict(n=dae.dim, nnz=dae.nnz, extract=t_extract, seq=t_seq, par=t_par)


def main():
    import os
    print(f"cores: {os.cpu_count()}")
    print(f"{'n':>7} {'nnz':>8} {'extract[ms]':>12} {'seq[ms]':>10} {'par[ms]':>10} {'speedup':>8}")
    for k in [30, 45, 60, 80, 100, 120]:
        r = bench(k)
        sp = r["seq"] / r["par"] if r["par"] else 0.0
        print(f"{r['n']:>7} {r['nnz']:>8} {r['extract']*1e3:>12.1f} "
              f"{r['seq']*1e3:>10.2f} {r['par']*1e3:>10.2f} {sp:>7.2f}x")


if __name__ == "__main__":
    main()
