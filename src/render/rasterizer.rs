use crate::math::clamp_u8;
use crate::splat::{
    evaluate_2d_gaussian, ProjectedSplat, MIN_GAUSSIAN_CONTRIBUTION, SATURATION_EPSILON,
};
use rayon::prelude::*;

// --- Rasterizer ---

pub fn blend_component(existing: u8, new: u8, weight: f32) -> u8 {
    clamp_u8(existing as f32 + new as f32 * weight)
}

pub fn rasterize_splats(
    projected_splats: &[ProjectedSplat],
    render_state: &mut super::RenderState,
    width: usize,
    height: usize,
) {
    if width == 0 || height == 0 || projected_splats.is_empty() {
        return;
    }

    let num_bands = rayon::current_num_threads();
    let band_height = height.div_ceil(num_bands);
    let actual_bands = height.div_ceil(band_height);

    // Phase 1: Pre-bin splat indices into bands.
    // Each splat goes into every band its bounding box overlaps.
    // Uses per-band Vec to avoid synchronization.
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); actual_bands];
    for (si, splat) in projected_splats.iter().enumerate() {
        let splat_min_y = (splat.screen_y - splat.radius_y).floor().max(0.0) as usize;
        let splat_max_y = (splat.screen_y + splat.radius_y)
            .ceil()
            .min(height.saturating_sub(1) as f32) as usize;

        let band_start = splat_min_y / band_height;
        let band_end = (splat_max_y / band_height).min(actual_bands - 1);
        for bin in &mut bins[band_start..=band_end] {
            bin.push(si);
        }
    }

    // Phase 2: Parallel rasterization -- each band processes only its binned splats.
    let fb_chunks: Vec<&mut [[u8; 3]]> = render_state
        .framebuffer
        .chunks_mut(band_height * width)
        .collect();
    let alpha_chunks: Vec<&mut [f32]> = render_state
        .alpha_buffer
        .chunks_mut(band_height * width)
        .collect();
    let depth_chunks: Vec<&mut [f32]> = render_state
        .depth_buffer
        .chunks_mut(band_height * width)
        .collect();

    fb_chunks
        .into_par_iter()
        .zip(alpha_chunks.into_par_iter())
        .zip(depth_chunks.into_par_iter())
        .zip(bins.par_iter())
        .enumerate()
        .for_each(|(band_idx, (((fb_band, alpha_band), depth_band), bin))| {
            let y_start = band_idx * band_height;
            let band_rows = fb_band.len() / width;
            let y_end = y_start + band_rows;

            for &si in bin {
                let splat = &projected_splats[si];

                let min_x = (splat.screen_x - splat.radius_x).floor().max(0.0) as usize;
                let max_x = (splat.screen_x + splat.radius_x)
                    .ceil()
                    .min((width.saturating_sub(1)) as f32) as usize;
                let min_y = (splat.screen_y - splat.radius_y)
                    .floor()
                    .max(y_start as f32) as usize;
                let max_y = (splat.screen_y + splat.radius_y)
                    .ceil()
                    .min((y_end - 1) as f32) as usize;

                if min_x > max_x || min_y > max_y {
                    continue;
                }

                let inv_cov_a = splat.inv_cov_a;
                let inv_cov_b = splat.inv_cov_b;
                let inv_cov_c = splat.inv_cov_c;

                for y in min_y..=max_y {
                    let local_y = y - y_start;
                    let row = local_y * width;
                    for x in min_x..=max_x {
                        let idx = row + x;
                        let existing_alpha = alpha_band[idx];
                        if existing_alpha >= SATURATION_EPSILON {
                            continue;
                        }

                        let dx = x as f32 + 0.5 - splat.screen_x;
                        let dy = y as f32 + 0.5 - splat.screen_y;
                        let gaussian =
                            evaluate_2d_gaussian(dx, dy, inv_cov_a, inv_cov_b, inv_cov_c);

                        if gaussian < MIN_GAUSSIAN_CONTRIBUTION {
                            continue;
                        }

                        let alpha = splat.opacity * gaussian;
                        if alpha <= 0.0 {
                            continue;
                        }

                        let weight = alpha * (1.0 - existing_alpha);
                        if weight < 1e-4 {
                            continue;
                        }

                        let pixel = &mut fb_band[idx];
                        pixel[0] = blend_component(pixel[0], splat.color[0], weight);
                        pixel[1] = blend_component(pixel[1], splat.color[1], weight);
                        pixel[2] = blend_component(pixel[2], splat.color[2], weight);

                        let new_alpha = (existing_alpha + weight).min(1.0);
                        alpha_band[idx] = new_alpha;
                        if new_alpha >= SATURATION_EPSILON {
                            depth_band[idx] = splat.depth;
                        }
                    }
                }
            }
        });
}
