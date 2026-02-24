# tortuise

Gaussian splats in your terminal.

<!-- ![tortuise demo](assets/demo.gif) -->
<!-- demo GIF coming soon -->

[![Crates.io](https://img.shields.io/crates/v/tortuise)](https://crates.io/crates/tortuise)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux-lightgrey)

## Why this exists

I've been fascinated by Gaussian splats since they first appeared -- that feeling when a point cloud suddenly snaps into a photorealistic scene is hard to shake. Then I binge-watched *Common Side Effects* over a weekend, caught a properly nasty cold, and needed something to decompress with that wasn't another episode. Somewhere between the cough syrup and episode six I realized: nobody had built a terminal viewer for 3DGS. Every viewer out there wants a browser or a GPU window. The itch was obvious. So I built one.

The name is a nod to the show and a hat-tip to the TUI ecosystem -- ratatouille begat ratatui, and ratatui begat tor-TUI-se. A tortoise carrying a 3D scene on its back.

Turns out 45K splats render just fine in Unicode half-block characters. Real scenes with 1M+ splats work too -- you just need a terminal that can keep up.

## Features

- **6 render modes** -- halfblock (default), braille, matrix, ASCII, block density, point cloud. Cycle with `M`.
- **Full 3D navigation** -- WASD movement, R/F for vertical, arrow keys for yaw and pitch. Smooth held-key input.
- **Two camera modes** -- Free (fly anywhere) and Orbit (auto-rotate around scene center). Switch with Space.
- **Loads .ply and .splat files** -- the standard 3DGS formats. Binary little-endian PLY with SH coefficients, 32-byte .splat records.
- **Built-in demo scene** -- loads `scenes/luigi.ply` when available; falls back to a procedural torus knot if not found. No files needed.
- **Intelligent terminal detection** -- truecolor for modern terminals, perceptual 256-color fallback for Terminal.app. Color mapping uses human-vision-weighted distance so the fallback looks as good as 256 colors can. Auto-detected, zero config.
- **Supersampling** -- 1x/2x/3x factor for higher fidelity at the cost of compute.
- **Cross-platform** -- macOS and Linux.

## Quick start

**Requires Rust 1.80+** (`rustup update` to upgrade)

```bash
# From source (recommended for now)
git clone https://github.com/buildoak/tortuise.git
cd tortuise
cargo install --path .

# Built-in demo (no scene file needed)
tortuise --demo

# Load a bundled scene
tortuise scenes/bonsai.splat

# Load any .ply or .splat file
tortuise your-scene.splat
tortuise your-scene.ply

# Some scenes need axis flips depending on capture coordinate system
tortuise --flip-y scene.ply
tortuise --flip-z scene.splat
```

### CLI options

```
tortuise [OPTIONS] [INPUT]

Arguments:
  [INPUT]    Path to a .ply or .splat scene file (runs demo if omitted)

Options:
  --demo              Run built-in demo scene
  --flip-y            Flip Y axis (some capture tools use Y-down)
  --flip-z            Flip Z axis
  --supersample <N>   Supersampling factor [default: 1]
  --cpu               Force CPU rendering
  -h, --help          Print help
  -V, --version       Print version
```

## Controls

### Free mode

| Key | Action |
|-----|--------|
| `W` / `A` / `S` / `D` | Move forward / left / back / right |
| `R` / `F` | Move up / down |
| Arrow keys | Yaw and pitch (look around) |
| `Space` | Switch to Orbit mode |
| `M` | Cycle render mode |
| `+` / `-` | Adjust movement speed |
| `Tab` | Toggle HUD |
| `Z` | Reset camera |
| `Q` / `Esc` | Quit |

### Orbit mode

| Key | Action |
|-----|--------|
| Arrow Up / Down | Adjust elevation |
| Arrow Left / Right | Nudge orbit angle |
| `Space` | Switch to Free mode |
| `+` / `-` | Adjust orbit speed |

## Supported terminals

**Truecolor (best experience):** Ghostty, iTerm2, Kitty, WezTerm, Alacritty

**256-color fallback:** Apple Terminal.app -- works, but reduced color fidelity. The perceptual color mapping does its best.

Auto-detected via `COLORTERM`, `TERM_PROGRAM`, and `TERM` environment variables. No configuration needed.

## Tested hardware

| Device | CPU | Scene | FPS |
|--------|-----|-------|-----|
| Mac Mini M4 | Apple M4 | luigi.ply (965K) | 120+ |
| MacBook Air M2 | Apple M2 | luigi.ply (965K) | ✓ |
| Jetson Orin Nano | ARM Cortex-A78AE | luigi.ply (965K) | ~30 |

## Where to get scenes

- `tortuise --demo` for an instant procedural scene -- no downloads required
- [Luma AI](https://lumalabs.ai/explore) -- large gallery of captured scenes, many export to .splat
- [Polycam](https://poly.cam/explore) -- photogrammetry captures, some with Gaussian splat export
- [nerfstudio](https://docs.nerf.studio/) -- train your own splats from video, exports to .ply
- Any standard 3DGS pipeline output in .ply or .splat format

Both formats are well-supported: PLY files with spherical harmonic coefficients (`f_dc_0/1/2`) or direct RGB, and the compact 32-byte .splat format used by most web viewers.

## How it works

The pipeline is straightforward: load splats, project them into screen space, depth-sort, splat onto a framebuffer, then convert to terminal characters. Each frame:

1. **Project** -- every Gaussian is transformed from world space through the camera view matrix. Frustum culling drops anything behind the near plane or outside the viewport. This step is parallelized with rayon.
2. **Sort** -- projected splats are depth-sorted back-to-front for correct alpha compositing.
3. **Rasterize** -- each splat is splatted onto an RGB framebuffer using its 2D covariance (scale + rotation). Front-to-back compositing with early alpha termination -- once a pixel is fully opaque, all remaining splats behind it are skipped. Per-splat saturation probes skip entire Gaussians when they land on already-saturated regions. At 1M+ splats, the back 80% are often invisible behind the front 20%.
4. **Encode** -- the framebuffer is converted to terminal output. In halfblock mode, each cell packs two vertical pixels using the `▄` character with separate foreground/background colors. Other modes use braille patterns, ASCII density ramps, or single characters.

The frame target is 8ms (~120fps). On truecolor terminals, colors are passed as 24-bit RGB. On 256-color terminals, a perceptual distance function maps each pixel to the closest ANSI color -- weighted toward green sensitivity, which is where human vision is sharpest.

## Roadmap

Things I want to build next -- and contribution opportunities if any of these scratch your itch too:

- **Kitty graphics protocol** -- pixel-perfect rendering via the terminal image protocol. Roughly 18x the resolution of half-block characters. This is the big one.
- **SHARP integration** -- image-to-splat-to-view pipeline. Single photo to 3D in your terminal.
- **Sample scene bundle** -- curated downloadable scenes so people can skip the "where do I find a .splat file" step.
- **GPU acceleration** -- a Metal compute backend exists behind a feature flag, but needs work at higher resolutions. Parked, not abandoned.
- **Performance** -- radix sort for depth ordering, SIMD-accelerated projection via glam, tighter memory layout.

## Built with

- [Rust](https://www.rust-lang.org/)
- [crossterm](https://github.com/crossterm-rs/crossterm) -- terminal control and input
- [rayon](https://github.com/rayon-rs/rayon) -- data parallelism for projection and rasterization
- [clap](https://github.com/clap-rs/clap) -- CLI argument parsing

The name lineage: ratatouille -> [ratatui](https://github.com/ratatui/ratatui) -> tortuise.

## License

MIT -- Nick Oak, 2026
