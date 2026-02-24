# Contributing

Contributions welcome. No CLA, no process overhead.

## Build from source

Requires Rust 1.80+.

```bash
git clone https://github.com/buildoak/tortuise.git
cd tortuise
cargo build --release
```

To enable the Metal GPU backend (macOS only, experimental):

```bash
cargo build --release --features metal
```

The Metal backend falls back to CPU automatically if initialization fails, so it's safe to enable.

## Running without a scene file

`--demo` is the fastest way to verify a change works end-to-end — no .ply or .splat file needed:

```bash
cargo run -- --demo
```

If `scenes/luigi.ply` is present in your working directory, the demo loads that. Otherwise it generates a procedural torus knot.

## Tests

```bash
cargo test
```

Tests live next to the modules they cover. If you add non-trivial logic, add a test.

## Lint and format

```bash
cargo clippy -- -D warnings
cargo fmt
```

Run both before opening a PR. Clippy warnings are blockers.

## What makes a good PR

- **Scope:** one thing per PR. Easier to review, easier to revert.
- **Demo-first:** if your change affects rendering, verify it looks right with `--demo` and a real scene.
- **No regressions on CPU path:** the Metal flag is experimental. The CPU path is the product — don't break it.
- **Perf-sensitive code:** the rasterizer and projection loop are hot. If you touch them, benchmark before and after (even informally — `--demo` FPS in the HUD is a good proxy).
- **Roadmap items welcome:** Kitty graphics protocol, SHARP integration, radix sort for depth ordering — all fair game. Open an issue first if the scope is large.

## License

MIT. Your contributions are MIT-licensed too.
