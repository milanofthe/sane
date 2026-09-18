# Vendored rsdag

Vendored copy of rsdag, the expression graph, differentiation, tape and
native backend that `sane-core` re-exports (see `crates/core/src/lib.rs`).

- Upstream: https://github.com/milanofthe/rsdag
- Vendored from: commit `20746e6`
- Contents: `crates/rsdag/src`, `crates/rsdag-jit/src`, their `Cargo.toml`,
  the workspace manifest (without the Python member) and `LICENSE`. Tests,
  examples, benchmarks, the Python crate and the CI scripts stay upstream.
- Local changes: none. Do not patch this tree; fix upstream and resync.

## Resync

```sh
scripts/vendor_rsdag.sh            # from ../rsdag
scripts/vendor_rsdag.sh path/to/rsdag
```

Then run `cargo test --workspace`.
