# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-24

### Fixed
- Camera init restored to proven `(0,0,5)` at origin; all AABB/scene-center heuristics removed
- Orbit mode always targets origin; WASD no longer shifts the orbit center
- Crate package excludes large scene files and binary assets

### Changed
- Demo image switched from GIF (9.3 MB) to animated WebP (4.4 MB)
- README tagline, intro, and "Why this exists" section rewritten for clarity
- Hardware table updated; ratatui attribution corrected

## [0.1.0] - 2026-02-14

### Added
- Metal GPU pipeline: single command buffer, tile-based radix sort, zero-copy readback
- CPU fallback renderer with packed framebuffer and color diffing
- Modal camera system: Free (WASD + R/F vertical) and Orbit modes; reset on Z
- Input thread with delta-time movement
- True-color detection for Ghostty, Kitty, WezTerm, iTerm2; 256-color fallback for Terminal.app
- Perceptual grayscale matching for 256-color environments
- `.ply` scene parser with checked arithmetic, file-size validation, and error context
- Clap CLI with `--flip-y` flag and graceful GPU-error degradation
- Procedural torus-knot scene as built-in fallback when no `.ply` is provided
- MSRV set to Rust 1.80
- MIT LICENSE, crates.io metadata, cross-platform dependencies
