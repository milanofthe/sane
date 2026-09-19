# Vendored rsdag

Vendored copy of rsdag, the expression graph, differentiation, tape and
native backend that `sane-core` re-exports (see `crates/core/src/lib.rs`).

- Upstream: https://github.com/milanofthe/rsdag
- Vendored from: commit `a083321`
- Contents: `crates/rsdag/src`, `crates/rsdag-jit/src`, their `Cargo.toml`,
  the workspace manifest (without the Python member), `LICENSE` and `NOTICE`.
  Tests, examples, benchmarks, the Python crate and the CI scripts stay
  upstream.
- Local changes: none. Do not patch this tree; fix upstream and resync.

## License

Upstream, rsdag is AGPL-3.0-only; `LICENSE` is that text, kept as the record
of where this copy comes from. Milan Rother is rsdag's sole author and
licenses it on other terms as well (see `NOTICE`), and this vendored copy is
distributed as part of SANE under SANE's own PolyForm Noncommercial license,
like every other file in this repository. The AGPL's terms do not reach the
rest of SANE through it.

## Resync

```sh
scripts/vendor_rsdag.sh            # from ../rsdag
scripts/vendor_rsdag.sh path/to/rsdag
```

Then run `cargo test --workspace`.
