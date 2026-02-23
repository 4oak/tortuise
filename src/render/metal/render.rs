use super::error::MetalRenderError;
use super::pipeline::{read_shared_u32, set_bytes_u32, write_shared_struct};
use super::sort::{dispatch_1d, div_ceil_u32};
use super::sync::commit_and_wait_or_disable_gpu;
use super::types::{GpuCameraData, TileConfig, RADIX_BUCKETS, THREADS_PER_GROUP_1D, TILE_SIZE};
use super::MetalBackend;
use crate::camera::Camera;
use metal::{MTLSize, NSRange};
use std::{ffi::c_void, mem, time::Duration};
const GPU_WAIT_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug)]
pub struct RenderAttemptResult {
    pub overflow_flag: u32,
    pub total_overlaps: u32,
}

impl MetalBackend {
    pub fn render(
        &mut self,
        camera: &Camera,
        screen_width: usize,
        screen_height: usize,
        splat_count: usize,
    ) -> Result<(), MetalRenderError> {
        if self.gpu_disabled {
            return Err(MetalRenderError::GpuDisabled);
        }

        if !self.splats_uploaded {
            return Err("No splats uploaded to Metal backend".into());
        }

        if screen_width == 0 || screen_height == 0 {
            self.last_render_width = screen_width;
            self.last_render_height = screen_height;
            return Ok(());
        }

        if splat_count > self.max_splats {
            return Err("Too many splats for GPU buffers".into());
        }

        let screen_width_u32 = u32::try_from(screen_width)?;
        let screen_height_u32 = u32::try_from(screen_height)?;

        let tile_count_x = div_ceil_u32(screen_width_u32, TILE_SIZE).max(1);
        let tile_count_y = div_ceil_u32(screen_height_u32, TILE_SIZE).max(1);
        let num_tiles_u64 = u64::from(tile_count_x) * u64::from(tile_count_y);
        if num_tiles_u64 > 1023 {
            return Err("Tile count exceeds 10-bit tile_id encoding (max 1023 tiles)".into());
        }
        let num_tiles = usize::try_from(num_tiles_u64)?;

        self.ensure_framebuffer_capacity(screen_width, screen_height)?;
        if splat_count == 0 {
            self.clear_framebuffer(screen_width, screen_height);
            self.last_render_width = screen_width;
            self.last_render_height = screen_height;
            return Ok(());
        }
        self.ensure_tile_capacity(num_tiles)?;

        let mut attempt = 0u32;
        loop {
            let estimated_overlaps = if self.previous_total_overlaps > 0 {
                (self.previous_total_overlaps as usize)
                    .saturating_mul(2)
                    .max(splat_count.saturating_mul(4))
            } else {
                splat_count.saturating_mul(8)
            }
            .max(1);

            self.ensure_sort_capacity_with_headroom(estimated_overlaps, 2, 1)?;
            let result =
                run_single_render_attempt(self, camera, screen_width, screen_height, splat_count)?;

            self.previous_total_overlaps = result.total_overlaps;
            if result.overflow_flag == 0 {
                self.maybe_shrink_sort_capacity(result.total_overlaps as usize)?;
                break;
            }

            if attempt >= 1 {
                let growth_target = (result.total_overlaps as usize).saturating_mul(2).max(1);
                self.ensure_sort_capacity(growth_target)?;
                return Err(MetalRenderError::OverflowDeferred {
                    requested_capacity: growth_target,
                    overlaps: result.total_overlaps,
                });
            }

            let retry_target = (result.total_overlaps as usize).saturating_mul(2).max(1);
            self.ensure_sort_capacity(retry_target)?;
            attempt += 1;
        }

        self.last_render_width = screen_width;
        self.last_render_height = screen_height;
        Ok(())
    }
}

pub fn run_single_render_attempt(
    backend: &mut MetalBackend,
    camera: &Camera,
    screen_width: usize,
    screen_height: usize,
    splat_count: usize,
) -> Result<RenderAttemptResult, MetalRenderError> {
    let screen_width_u32 = u32::try_from(screen_width)?;
    let screen_height_u32 = u32::try_from(screen_height)?;
    let tile_count_x = div_ceil_u32(screen_width_u32, TILE_SIZE).max(1);
    let tile_count_y = div_ceil_u32(screen_height_u32, TILE_SIZE).max(1);
    let num_tiles_u64 = u64::from(tile_count_x) * u64::from(tile_count_y);
    let num_tiles = usize::try_from(num_tiles_u64)?;

    let sort_capacity_u32 = u32::try_from(backend.sort_capacity)?;
    backend.ensure_block_sums_capacity_for_count(num_tiles as u32)?;

    let (fx, fy) = camera.focal_lengths(screen_width, screen_height);
    let gpu_camera = GpuCameraData {
        pos_x: camera.position.x,
        pos_y: camera.position.y,
        pos_z: camera.position.z,
        right_x: camera.right.x,
        right_y: camera.right.y,
        right_z: camera.right.z,
        up_x: camera.up.x,
        up_y: camera.up.y,
        up_z: camera.up.z,
        forward_x: camera.forward.x,
        forward_y: camera.forward.y,
        forward_z: camera.forward.z,
        fx,
        fy,
        half_w: screen_width as f32 * 0.5,
        half_h: screen_height as f32 * 0.5,
        near_plane: camera.near,
        far_plane: camera.far,
    };

    let tile_config = TileConfig {
        tile_count_x,
        tile_count_y,
        screen_width: screen_width_u32,
        screen_height: screen_height_u32,
    };

    write_shared_struct(&backend.camera_buffer, &gpu_camera);
    write_shared_struct(&backend.tile_config_buffer, &tile_config);

    let tile_bytes = super::buffers::bytes_for_u32_elems(num_tiles)? as u64;
    let splat_count_u32 = u32::try_from(splat_count)?;
    let framebuffer_pixels = screen_width
        .checked_mul(screen_height)
        .ok_or_else(|| MetalRenderError::Other("framebuffer pixel count overflow".to_string()))?;
    let framebuffer_clear_bytes = framebuffer_pixels
        .checked_mul(mem::size_of::<u32>())
        .ok_or_else(|| MetalRenderError::Other("framebuffer clear size overflow".to_string()))?
        as u64;

    let stage_a = backend.command_queue.new_command_buffer();

    let blit = stage_a.new_blit_command_encoder();
    blit.fill_buffer(
        &backend.framebuffer,
        NSRange::new(0, framebuffer_clear_bytes),
        0,
    );
    blit.fill_buffer(&backend.tile_counts, NSRange::new(0, tile_bytes), 0);
    blit.fill_buffer(
        &backend.valid_count_buffer,
        NSRange::new(0, mem::size_of::<u32>() as u64),
        0,
    );
    blit.fill_buffer(
        &backend.total_overlaps_buffer,
        NSRange::new(0, mem::size_of::<u32>() as u64),
        0,
    );
    blit.fill_buffer(
        &backend.overflow_flag_buffer,
        NSRange::new(0, mem::size_of::<u32>() as u64),
        0,
    );
    blit.end_encoding();

    let encoder = stage_a.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&backend.project_splats_pipeline);
    encoder.set_buffer(0, Some(&backend.splat_buffer), 0);
    encoder.set_buffer(1, Some(&backend.projected_buffer), 0);
    encoder.set_buffer(2, Some(&backend.valid_count_buffer), 0);
    encoder.set_buffer(3, Some(&backend.camera_buffer), 0);
    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &splat_count_u32 as *const _ as *const c_void,
    );
    encoder.set_buffer(5, Some(&backend.tile_config_buffer), 0);
    dispatch_1d(encoder, splat_count_u32, THREADS_PER_GROUP_1D);
    encoder.end_encoding();

    let encoder = stage_a.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&backend.count_tile_overlaps_pipeline);
    encoder.set_buffer(0, Some(&backend.projected_buffer), 0);
    encoder.set_buffer(1, Some(&backend.tile_counts), 0);
    encoder.set_buffer(2, Some(&backend.total_overlaps_buffer), 0);
    encoder.set_buffer(3, Some(&backend.valid_count_buffer), 0);
    encoder.set_buffer(4, Some(&backend.tile_config_buffer), 0);
    dispatch_1d(encoder, splat_count_u32, THREADS_PER_GROUP_1D);
    encoder.end_encoding();

    let blit = stage_a.new_blit_command_encoder();
    blit.copy_from_buffer(
        &backend.tile_counts,
        0,
        &backend.tile_offsets,
        0,
        tile_bytes,
    );
    blit.copy_from_buffer(
        &backend.total_overlaps_buffer,
        0,
        &backend.tile_offsets,
        tile_bytes,
        mem::size_of::<u32>() as u64,
    );
    blit.fill_buffer(&backend.tile_counters, NSRange::new(0, tile_bytes), 0);
    blit.end_encoding();

    backend.encode_prefix_scan_in_place(stage_a, &backend.tile_offsets, 0, num_tiles as u32)?;
    commit_and_wait_or_disable_gpu(
        stage_a,
        "project_count_scan",
        GPU_WAIT_TIMEOUT,
        &mut backend.gpu_disabled,
    )?;

    let total_overlaps = read_shared_u32(&backend.total_overlaps_buffer);
    let dispatch_overlaps = sort_capacity_u32;

    if dispatch_overlaps > 0 {
        let sort_num_blocks = div_ceil_u32(dispatch_overlaps, THREADS_PER_GROUP_1D);
        let histogram_count = sort_num_blocks
            .checked_mul(RADIX_BUCKETS)
            .ok_or_else(|| MetalRenderError::Other("histogram count overflow".to_string()))?;
        backend.ensure_histogram_capacity(histogram_count as usize)?;
        backend.ensure_block_sums_capacity_for_count(histogram_count)?;
    }

    let stage_b = backend.command_queue.new_command_buffer();
    let mut keys_in_a = true;

    let encoder = stage_b.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&backend.emit_tile_keys_pipeline);
    encoder.set_buffer(0, Some(&backend.projected_buffer), 0);
    encoder.set_buffer(1, Some(&backend.tile_offsets), 0);
    encoder.set_buffer(2, Some(&backend.tile_counters), 0);
    encoder.set_buffer(3, Some(&backend.sort_keys_a), 0);
    encoder.set_buffer(4, Some(&backend.sort_values_a), 0);
    encoder.set_buffer(5, Some(&backend.valid_count_buffer), 0);
    encoder.set_buffer(6, Some(&backend.tile_config_buffer), 0);
    encoder.set_buffer(7, Some(&backend.overflow_flag_buffer), 0);
    set_bytes_u32(encoder, 8, sort_capacity_u32);
    dispatch_1d(encoder, splat_count_u32, THREADS_PER_GROUP_1D);
    encoder.end_encoding();

    if dispatch_overlaps > 0 {
        backend.run_radix_sort_passes(stage_b, dispatch_overlaps, &mut keys_in_a)?;

        let (sorted_keys, sorted_values) = if keys_in_a {
            (&backend.sort_keys_a, &backend.sort_values_a)
        } else {
            (&backend.sort_keys_b, &backend.sort_values_b)
        };

        let encoder = stage_b.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&backend.rasterize_tiles_pipeline);
        encoder.set_buffer(0, Some(&backend.projected_buffer), 0);
        encoder.set_buffer(1, Some(sorted_keys), 0);
        encoder.set_buffer(2, Some(sorted_values), 0);
        encoder.set_buffer(3, Some(&backend.tile_offsets), 0);
        encoder.set_buffer(4, Some(&backend.framebuffer), 0);
        encoder.set_buffer(5, Some(&backend.tile_config_buffer), 0);
        set_bytes_u32(encoder, 6, dispatch_overlaps);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(tile_count_x), u64::from(tile_count_y), 1),
            MTLSize::new(16, 16, 1),
        );
        encoder.end_encoding();
    }

    commit_and_wait_or_disable_gpu(
        stage_b,
        "sort_rasterize",
        GPU_WAIT_TIMEOUT,
        &mut backend.gpu_disabled,
    )?;

    let overflow_flag = read_shared_u32(&backend.overflow_flag_buffer);

    Ok(RenderAttemptResult {
        overflow_flag,
        total_overlaps,
    })
}
