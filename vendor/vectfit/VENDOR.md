# Vendored vectfit

Vendored copy of the `vectfit` crate from rapidmom: Fast Relaxed Vector
Fitting (Gustavsen-Semlyen, relaxed relocation, multiport common poles,
automatic order by adding-and-skimming), passivity check/enforcement, and
deterministic Verilog-A export targeting SANE's AD-friendly grammar subset.
Used by `sane-py` for the S-parameter macromodel pipeline (Touchstone ->
rational fit -> Verilog-A -> `N` instance).

- Upstream: https://github.com/milanofthe/rapidmom (`crates/vectfit`)
- Vendored from: commit `f73410c`
- Contents: `src/` and `Cargo.toml` verbatim; the manifest's workspace
  inheritance (`edition.workspace` etc.) is replaced by standalone fields
  and the `[lints]` section dropped, nothing else changed.
- Local changes: none. Do not patch this tree; fix upstream and resync.

## Resync

```sh
rm -rf vendor/vectfit/src
cp -R ../rapidmom/crates/vectfit/src vendor/vectfit/src
```

Then update the commit hash above and run `cargo test -p vectfit -p sane-py`.
