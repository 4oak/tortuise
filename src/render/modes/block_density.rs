use super::{depth_attenuation, is_hud_overlay_row, BLOCK_DENSITY_CHARS};
use crate::math::clamp_u8;
use crate::splat::ProjectedSplat;
use crossterm::{
    cursor, queue,
    style::{Color, Print, SetBackgroundColor, SetForegroundColor},
};
use rayon::prelude::*;
use std::io::{self, Write};

// --- Block Density ---

pub fn render_block_density(
    projected_splats: &[ProjectedSplat],
    term_cols: usize,
    term_rows: usize,
    proj_height: usize,
    stdout: &mut impl Write,
    show_hud: bool,
) -> io::Result<()> {
    let len = term_cols.saturating_mul(term_rows);
    let mut accumulated_opacity = vec![0.0_f32; len];
    let mut weighted_r = vec![0.0_f32; len];
    let mut weighted_g = vec![0.0_f32; len];
    let mut weighted_b = vec![0.0_f32; len];

    // Parallel accumulation: split rows into bands
    let num_threads = rayon::current_num_threads();
    let band_rows = term_rows.div_ceil(num_threads);

    let ao_chunks: Vec<&mut [f32]> = accumulated_opacity
        .chunks_mut(band_rows * term_cols)
        .collect();
    let wr_chunks: Vec<&mut [f32]> = weighted_r.chunks_mut(band_rows * term_cols).collect();
    let wg_chunks: Vec<&mut [f32]> = weighted_g.chunks_mut(band_rows * term_cols).collect();
    let wb_chunks: Vec<&mut [f32]> = weighted_b.chunks_mut(band_rows * term_cols).collect();

    ao_chunks
        .into_par_iter()
        .zip(wr_chunks.into_par_iter())
        .zip(wg_chunks.into_par_iter())
        .zip(wb_chunks.into_par_iter())
        .enumerate()
        .for_each(|(band_idx, (((ao, wr), wg), wb))| {
            let row_start = band_idx * band_rows;
            let actual_rows = ao.len() / term_cols.max(1);
            let row_end = row_start + actual_rows;

            for splat in projected_splats {
                if !splat.screen_x.is_finite() || !splat.screen_y.is_finite() {
                    continue;
                }
                let y_scale = proj_height as f32 / term_rows.max(1) as f32;
                let row = (splat.screen_y / y_scale).floor() as isize;
                if row < row_start as isize || row >= row_end as isize {
                    continue;
                }
                let col = splat.screen_x.floor() as isize;
                if col < 0 || col >= term_cols as isize {
                    continue;
                }

                let local_row = row as usize - row_start;
                let local_idx = local_row * term_cols + col as usize;

                let attenuation = depth_attenuation(splat.depth);
                let weight = splat.opacity * attenuation;
                if weight <= 0.0 {
                    continue;
                }

                ao[local_idx] += weight;
                wr[local_idx] += splat.color[0] as f32 * weight;
                wg[local_idx] += splat.color[1] as f32 * weight;
                wb[local_idx] += splat.color[2] as f32 * weight;
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
            let acc = accumulated_opacity[idx];

            let (ch, fg) = if acc > 0.0 {
                let density_char = if acc < 0.15 {
                    BLOCK_DENSITY_CHARS[0]
                } else if acc < 0.4 {
                    BLOCK_DENSITY_CHARS[1]
                } else if acc < 0.7 {
                    BLOCK_DENSITY_CHARS[2]
                } else {
                    BLOCK_DENSITY_CHARS[3]
                };
                let inv = 1.0 / acc;
                (
                    density_char,
                    (
                        clamp_u8(weighted_r[idx] * inv),
                        clamp_u8(weighted_g[idx] * inv),
                        clamp_u8(weighted_b[idx] * inv),
                    ),
                )
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
