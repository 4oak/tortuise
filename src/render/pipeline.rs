use rayon::prelude::*;
use crate::camera::Camera;
use crate::sort::sort_by_depth;
use crate::splat::{
    compute_2d_gaussian_extent, compute_3d_covariance,
    invert_2x2_covariance, project_covariance_to_2d, ProjectedSplat, Splat,
    MIN_SPLAT_RADIUS,
};

// --- Framebuffer ---

pub fn resize_render_state(render_state: &mut super::RenderState, width: usize, height: usize) {
    if render_state.width == width && render_state.height == height {
        return;
    }

    render_state.width = width;
    render_state.height = height;
    let len = width.saturating_mul(height);
    render_state.framebuffer.resize(len, [0, 0, 0]);
    render_state.alpha_buffer.resize(len, 0.0);
    render_state.depth_buffer.resize(len, f32::INFINITY);
}

pub fn clear_framebuffer(render_state: &mut super::RenderState) {
    render_state.framebuffer.fill([0, 0, 0]);
    render_state.alpha_buffer.fill(0.0);
    render_state.depth_buffer.fill(f32::INFINITY);
}

// --- Projection ---

pub fn project_and_cull_splats(
    splats: &[Splat],
    projected_splats: &mut Vec<ProjectedSplat>,
    camera: &Camera,
    screen_width: usize,
    screen_height: usize,
    visible_count: &mut usize,
) {
    let (fx, fy) = camera.focal_lengths(screen_width, screen_height);
    let half_w = screen_width as f32 * 0.5;
    let half_h = screen_height as f32 * 0.5;
    let sw = screen_width as f32;
    let sh = screen_height as f32;

    let result: Vec<ProjectedSplat> = splats
        .par_iter()
        .enumerate()
        .filter_map(|(i, splat)| {
            let view_pos = camera.world_to_view(splat.position);
            if view_pos.z < camera.near || view_pos.z > camera.far {
                return None;
            }

            let inv_z = 1.0 / view_pos.z.max(1e-5);
            let screen_x = half_w + view_pos.x * fx * inv_z;
            let screen_y = half_h - view_pos.y * fy * inv_z;

            if !screen_x.is_finite() || !screen_y.is_finite() {
                return None;
            }

            const BROAD_MARGIN: f32 = 120.0;
            if screen_x < -BROAD_MARGIN
                || screen_x > sw + BROAD_MARGIN
                || screen_y < -BROAD_MARGIN
                || screen_y > sh + BROAD_MARGIN
            {
                return None;
            }

            let cov_3d = compute_3d_covariance(splat.scale, splat.rotation);
            let (cov_a, cov_b, cov_c) =
                project_covariance_to_2d(cov_3d, camera, view_pos, fx, fy);

            if cov_a <= 0.0 || cov_c <= 0.0 {
                return None;
            }

            let (radius_x, radius_y) = compute_2d_gaussian_extent(cov_a, cov_b, cov_c);
            if radius_x < MIN_SPLAT_RADIUS || radius_y < MIN_SPLAT_RADIUS {
                return None;
            }

            if screen_x + radius_x < 0.0
                || screen_x - radius_x > sw
                || screen_y + radius_y < 0.0
                || screen_y - radius_y > sh
            {
                return None;
            }

            let (inv_cov_a, inv_cov_b, inv_cov_c) =
                invert_2x2_covariance(cov_a, cov_b, cov_c)?;

            Some(ProjectedSplat {
                screen_x,
                screen_y,
                depth: view_pos.z,
                radius_x,
                radius_y,
                color: splat.color,
                opacity: splat.opacity,
                inv_cov_a,
                inv_cov_b,
                inv_cov_c,
                original_index: i,
            })
        })
        .collect();

    *visible_count = result.len();
    *projected_splats = result;
}

pub fn cpu_project_and_sort(app_state: &mut super::AppState, width: usize, height: usize) {
    project_and_cull_splats(
        &app_state.splats,
        &mut app_state.projected_splats,
        &app_state.camera,
        width,
        height,
        &mut app_state.visible_splat_count,
    );
    sort_by_depth(&mut app_state.projected_splats);
}
