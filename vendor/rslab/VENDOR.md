# Vendored rslab

Vendored copy of the rslab sparse direct solver, used by `sane-solve` for the
circuit-shaped KLU path (BTF + per-block Gilbert-Peierls LU with numeric-only
`refactor` and `solve_transpose`).

- Upstream: https://github.com/milanofthe/rslab
- Vendored from: commit `a0222fa` (v0.34.0 plus KLU single-block overhead, PR #56)
- Contents: `src/` (library only) plus the `crates/rslab-*` ordering crates,
  `LICENSE`, `NOTICE`, `LICENSE-THIRD-PARTY`. Upstream bins, benches,
  integration tests, xtask, python bindings and docs are stripped; the
  `matgen`/`matgen-download`/`tuning` optional features are dropped from the
  trimmed `Cargo.toml`.
- Local changes: none. Do not patch this tree; fix upstream and resync.

## Resync

```sh
cd rslab && git pull
cd ../sane
rm -rf vendor/rslab/src vendor/rslab/crates
cp -R ../rslab/src vendor/rslab/src && rm -rf vendor/rslab/src/bin
for c in rslab-ordering-core rslab-amd rslab-amf rslab-metis; do
  cp -R ../rslab/crates/$c vendor/rslab/crates/ && rm -rf vendor/rslab/crates/$c/tests
done
cp ../rslab/LICENSE ../rslab/NOTICE ../rslab/LICENSE-THIRD-PARTY vendor/rslab/
```

Then update the commit hash above, bump the `version` in
`vendor/rslab/Cargo.toml` to match upstream, and run `cargo test --workspace`.
