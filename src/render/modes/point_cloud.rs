use super::{depth_attenuation, is_hud_overlay_row, POINT_CLOUD_CHARS};
use crate::math::clamp_u8;
use crate::render::make_color;
use crate::splat::ProjectedSplat;
use crossterm::{
    cursor, queue,
    style::{Print, SetBackgroundColor, SetForegroundColor},
};
use rayon::prelude::*;
use std::io::{self, Write};

// --- Point Cloud ---

pub fn render_point_cloud(
    projected_splats: &[ProjectedSplat],
    term_cols: usize,
    term_rows: usize,
    proj_height: usize,
    stdout: &mut impl Write,
    show_hud: bool,
    use_truecolor: bool,
) -> io::Result<()> {
    let len = term_cols.saturating_mul(term_rows);
    let mut depth_buffer = vec![f32::INFINITY; len];
    let mut cell_chars = vec![' '; len];
    let mut cell_fgs = vec![[0u8; 3]; len];
    let mut occupied = vec![false; len];

    // Parallel z-buffer: split rows into bands
    let num_threads = rayon::current_num_threads();
    let band_rows = term_rows.div_ceil(num_threads);

    let db_chunks: Vec<&mut [f32]> = depth_buffer.chunks_mut(band_rows * term_cols).collect();
    let cc_chunks: Vec<&mut [char]> = cell_chars.chunks_mut(band_rows * term_cols).collect();
    let cf_chunks: Vec<&mut [[u8; 3]]> = cell_fgs.chunks_mut(band_rows * term_cols).collect();
    let oc_chunks: Vec<&mut [bool]> = occupied.chunks_mut(band_rows * term_cols).collect();

    db_chunks
        .into_par_iter()
        .zip(cc_chunks.into_par_iter())
        .zip(cf_chunks.into_par_iter())
        .zip(oc_chunks.into_par_iter())
        .enumerate()
        .for_each(|(band_idx, (((db, cc), cf), oc))| {
            let row_start = band_idx * band_rows;
            let actual_rows = db.len() / term_cols.max(1);
            let row_end = row_start + actual_rows;

            for splat in projected_splats {
                let y_scale = proj_height as f32 / term_rows.max(1) as f32;
                let row = (splat.screen_y / y_scale).floor() as isize;
                if row < row_start as isize || row >= row_end as isize {
                    continue;
                }
                let col = splat.screen_x.floor() as isize;
                if col < 0 || col >= term_cols as isize {
                    continue;
                }
                if !splat.screen_x.is_finite() || !splat.screen_y.is_finite() {
                    continue;
                }

                let local_row = row as usize - row_start;
                let local_idx = local_row * term_cols + col as usize;
                if splat.depth >= db[local_idx] {
                    continue;
                }

                let attenuation = depth_attenuation(splat.depth);
                let brightness = (splat.opacity * attenuation).clamp(0.0, 1.0);
                let max_idx = (POINT_CLOUD_CHARS.len() - 1) as f32;
                let char_idx = (brightness * max_idx).clamp(0.0, max_idx) as usize;

                db[local_idx] = splat.depth;
                cc[local_idx] = POINT_CLOUD_CHARS[char_idx];
                cf[local_idx] = [
                    clamp_u8(splat.color[0] as f32 * attenuation),
                    clamp_u8(splat.color[1] as f32 * attenuation),
                    clamp_u8(splat.color[2] as f32 * attenuation),
                ];
                oc[local_idx] = true;
            }
        });

    let mut last_bg: Option<(u8, u8, u8)> = None;
    let mut last_fg: Option<(u8, u8, u8)> = None;
    let bg = (2, 2, 6);

    for row in 0..term_rows {
        if is_hud_overlay_row(show_hud, row, term_rows) {
            last_bg = None;
            last_fg = None;
            continue;
        }

        queue!(stdout, cursor::MoveTo(0, row as u16))?;
        for col in 0..term_cols {
            let idx = row * term_cols + col;
            let fg = if occupied[idx] {
                let c = cell_fgs[idx];
                (c[0], c[1], c[2])
            } else {
                (0, 0, 0)
            };

            if last_bg != Some(bg) {
                queue!(
                    stdout,
                    SetBackgroundColor(make_color(bg.0, bg.1, bg.2, use_truecolor))
                )?;
                last_bg = Some(bg);
            }
            if last_fg != Some(fg) {
                queue!(
                    stdout,
                    SetForegroundColor(make_color(fg.0, fg.1, fg.2, use_truecolor))
                )?;
                last_fg = Some(fg);
            }

            let ch = if occupied[idx] { cell_chars[idx] } else { ' ' };
            queue!(stdout, Print(ch))?;
        }
    }

    Ok(())
}
