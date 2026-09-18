# sane

Symbolic circuit analysis with exact component sensitivities.

`sane` parses SPICE-like netlists (with a native Verilog-A compact-model
front-end), extracts a symbolic differential-algebraic model of the circuit, and
runs DC, AC, transient, pole/zero, harmonic-balance, noise and model-reduction
analyses. Every analysis exposes exact first- and second-order parameter
sensitivities (adjoint autodiff, no finite differences), so component tolerancing,
optimisation and cross-simulator validation all read off the same model.

```python
import sane

m = sane.Model.from_netlist("""
V1 in 0 5
R1 in out 1k
R2 out 0 1k
.end
""")
op = m.operating_point()
print(op.get("out"))          # 2.5 V
s = op.sensitivity("out")     # exact d v(out) / d p for every parameter
```

The compiled core is written in Rust (PyO3); the ergonomic, documented surface is
the pure-Python package overlaid on top. An optional native backend (rsdag's emitter) can be
enabled at build time (`maturin develop --features jit`, the default) and
disabled at runtime with `SANE_JIT=0`.

## License

Distributed under the PolyForm Noncommercial License 1.0.0
(`LicenseRef-PolyForm-Noncommercial-1.0.0`). See the repository `LICENSE` file for
the full terms. Commercial use requires a separate license.
