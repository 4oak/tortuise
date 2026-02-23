use super::{depth_attenuation, is_hud_overlay_row};
use crate::math::clamp_u8;
use crate::splat::{evaluate_2d_gaussian, ProjectedSplat};
use crossterm::{
    cursor, queue,
    style::{Color, Print, SetBackgroundColor, SetForegroundColor},
};
use rayon::prelude::*;
use std::io::{self, Write};

// --- Braille ---

fn braille_char(dots: u8) -> char {
    char::from_u32(0x2800 + dots as u32).unwrap_or(' ')
}

fn dot_bit(dx: usize, dy: usize) -> u8 {
    match (dx, dy) {
        (0, 0) => 0x01,
        (0, 1) => 0x02,
        (0, 2) => 0x04,
        (1, 0) => 0x08,
        (1, 1) => 0x10,
        (1, 2) => 0x20,
        (0, 3) => 0x40,
        (1, 3) => 0x80,
        _ => 0,
    }
}

pub fn render_braille(
    projected_splats: &[ProjectedSplat],
    term_cols: usize,
    term_rows: usize,
    proj_height: usize,
    stdout: &mut impl Write,
    show_hud: bool,
) -> io::Result<()> {
    let len = term_cols.saturating_mul(term_rows);
    let mut cell_dots = vec![0u8; len];
    let mut cell_depth = vec![f32::INFINITY; len];
    let mut cell_color = vec![[0u8; 3]; len];

    // Braille effective resolution: 2 subpixels per col, 4 subpixels per row.
    let braille_w = term_cols * 2;
    let braille_h = term_rows * 4;

    // Scaling factors from projection-pixel space to braille subpixel space.
    let bscale_x: f32 = braille_w as f32 / term_cols.max(1) as f32;
    let bscale_y: f32 = braille_h as f32 / proj_height.max(1) as f32;

    let inv_bscale_x = 1.0 / bscale_x;
    let inv_bscale_y = 1.0 / bscale_y;

    const BRAILLE_ALPHA_THRESHOLD: f32 = 0.04;

    // Parallel braille rasterization: split cell rows into bands
    let num_threads = rayon::current_num_threads();
    let band_rows = term_rows.div_ceil(num_threads);

    let dots_chunks: Vec<&mut [u8]> = cell_dots.chunks_mut(band_rows * term_cols).collect();
    let dep_chunks: Vec<&mut [f32]> = cell_depth.chunks_mut(band_rows * term_cols).collect();
    let col_chunks: Vec<&mut [[u8; 3]]> = cell_color.chunks_mut(band_rows * term_cols).collect();

    dots_chunks
        .into_par_iter()
        .zip(dep_chunks.into_par_iter())
        .zip(col_chunks.into_par_iter())
        .enumerate()
        .for_each(|(band_idx, ((dots, dep), col))| {
            let cell_row_start = band_idx * band_rows;
            let actual_cell_rows = dots.len() / term_cols.max(1);
            let cell_row_end = cell_row_start + actual_cell_rows;

            // Braille subpixel Y range for this band
            let by_start = cell_row_start * 4;
            let by_end = cell_row_end * 4;
            let by_start_f = by_start as f32;
            let by_end_f = by_end as f32;

            for splat in projected_splats {
                if !splat.screen_x.is_finite() || !splat.screen_y.is_finite() {
                    continue;
                }

                let inv_cov_a = splat.inv_cov_a;
                let inv_cov_b = splat.inv_cov_b;
                let inv_cov_c = splat.inv_cov_c;

                let center_bx = splat.screen_x * bscale_x;
                let center_by = splat.screen_y * bscale_y;
                let extent_bx = splat.radius_x * bscale_x;
                let extent_by = splat.radius_y * bscale_y;

                // Quick band overlap check
                let splat_min_by = center_by - extent_by;
                let splat_max_by = center_by + extent_by;
                if splat_max_by < by_start_f || splat_min_by >= by_end_f {
                    continue;
                }

                let min_bx = ((center_bx - extent_bx).floor() as isize).max(0) as usize;
                let max_bx = ((center_bx + extent_bx).ceil() as isize)
                    .min(braille_w as isize - 1)
                    .max(0) as usize;
                let min_by = (splat_min_by.floor() as isize).max(by_start as isize) as usize;
                let max_by = (splat_max_by.ceil() as isize)
                    .min(by_end as isize - 1)
                    .max(by_start as isize) as usize;

                if min_bx > max_bx || min_by > max_by {
                    continue;
                }

                let attenuation = depth_attenuation(splat.depth);
                let attenuated_color = [
                    clamp_u8(splat.color[0] as f32 * attenuation),
                    clamp_u8(splat.color[1] as f32 * attenuation),
                    clamp_u8(splat.color[2] as f32 * attenuation),
                ];

                for by in min_by..=max_by {
                    let cell_row = by / 4;
                    let dot_dy = by % 4;
                    let local_cell_row = cell_row - cell_row_start;
                    let row_base = local_cell_row * term_cols;

                    let dy_proj = (by as f32 + 0.5 - center_by) * inv_bscale_y;

                    for bx in min_bx..=max_bx {
                        let dx_proj = (bx as f32 + 0.5 - center_bx) * inv_bscale_x;

                        let gaussian =
                            evaluate_2d_gaussian(dx_proj, dy_proj, inv_cov_a, inv_cov_b, inv_cov_c);

                        if gaussian * splat.opacity < BRAILLE_ALPHA_THRESHOLD {
                            continue;
                        }

                        let cell_col = bx / 2;
                        let dot_dx = bx % 2;
                        let local_idx = row_base + cell_col;
                        if local_idx >= dots.len() {
                            continue;
                        }

                        dots[local_idx] |= dot_bit(dot_dx, dot_dy);

                        if splat.depth < dep[local_idx] {
                            dep[local_idx] = splat.depth;
                            col[local_idx] = attenuated_color;
                        }
                    }
                }
            }
        });

    let mut last_bg: Option<(u8, u8, u8)> = None;
    let mut last_fg: Option<(u8, u8, u8)> = None;
    let bg = (0, 0, 0);

    for row in 0..term_rows {
        if is_hud_overlay_row(show_hud, row, term_rows) {
            last_bg = None;
            last_fg = None;
            continue;
        }

        queue!(stdout, cursor::MoveTo(0, row as u16))?;
        for col in 0..term_cols {
            let idx = row * term_cols + col;
            let dots = cell_dots[idx];

            let (ch, fg) = if dots != 0 {
                let c = cell_color[idx];
                (braille_char(dots), (c[0], c[1], c[2]))
            } else {
                (' ', (0, 0, 0))
            };

            if last_bg != Some(bg) {
                queue!(
                    stdout,
                    SetBackgroundColor(Color::Rgb {
                        r: bg.0,
                        g: bg.1,
                        b: bg.2
                    })
                )?;
                last_bg = Some(bg);
            }
            if last_fg != Some(fg) {
                queue!(
                    stdout,
                    SetForegroundColor(Color::Rgb {
                        r: fg.0,
                        g: fg.1,
                        b: fg.2
                    })
                )?;
                last_fg = Some(fg);
            }
            queue!(stdout, Print(ch))?;
        }
    }

    Ok(())
}
