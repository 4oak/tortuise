#[cfg(feature = "metal")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize, NSRange,
};
#[cfg(feature = "metal")]
use std::ffi::c_void;
#[cfg(feature = "metal")]
use std::{mem, ptr};

use crate::{
    camera::Camera,
    splat::Splat,
};

const TILE_SIZE: u32 = 16;
const THREADS_PER_GROUP_1D: u32 = 256;
const RADIX_BUCKETS: u32 = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuSplatData {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    pub rot_w: f32,
    pub rot_x: f32,
    pub rot_y: f32,
    pub rot_z: f32,
    pub opacity: f32,
    pub packed_color: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuCameraData {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub right_x: f32,
    pub right_y: f32,
    pub right_z: f32,
    pub up_x: f32,
    pub up_y: f32,
    pub up_z: f32,
    pub forward_x: f32,
    pub forward_y: f32,
    pub forward_z: f32,
    pub fx: f32,
    pub fy: f32,
    pub half_w: f32,
    pub half_h: f32,
    pub near_plane: f32,
    pub far_plane: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuProjectedSplat {
    pub screen_x: f32,
    pub screen_y: f32,
    pub depth: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub cov_a: f32,
    pub cov_b: f32,
    pub cov_c: f32,
    pub opacity: f32,
    pub packed_color: u32,
    pub original_index: u32,
    pub tile_min: u32,
    pub tile_max: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    pub screen_width: u32,
    pub screen_height: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RadixSortConfig {
    pub num_elements: u32,
    pub bit_offset: u32,
}

const _: [(); 48] = [(); std::mem::size_of::<GpuSplatData>()];
const _: [(); 72] = [(); std::mem::size_of::<GpuCameraData>()];
const _: [(); 52] = [(); std::mem::size_of::<GpuProjectedSplat>()];
const _: [(); 16] = [(); std::mem::size_of::<TileConfig>()];
const _: [(); 8] = [(); std::mem::size_of::<RadixSortConfig>()];

#[cfg(feature = "metal")]
#[allow(dead_code)]
pub struct MetalBackend {
    device: Device,
    command_queue: CommandQueue,

    project_splats_pipeline: ComputePipelineState,
    prefix_scan_blocks_pipeline: ComputePipelineState,
    prefix_scan_add_offsets_pipeline: ComputePipelineState,
    radix_sort_histogram_pipeline: ComputePipelineState,
    radix_sort_scatter_pipeline: ComputePipelineState,
    count_tile_overlaps_pipeline: ComputePipelineState,
    clamp_total_overlaps_pipeline: ComputePipelineState,
    emit_tile_keys_pipeline: ComputePipelineState,
    rasterize_tiles_pipeline: ComputePipelineState,

    splat_buffer: Buffer,
    camera_buffer: Buffer,
    valid_count_buffer: Buffer,
    total_overlaps_buffer: Buffer,
    tile_config_buffer: Buffer,
    framebuffer: Buffer,

    projected_buffer: Buffer,
    tile_counts: Buffer,
    tile_offsets: Buffer,
    tile_counters: Buffer,
    sort_keys_a: Buffer,
    sort_keys_b: Buffer,
    sort_values_a: Buffer,
    sort_values_b: Buffer,
    radix_histograms: Buffer,
    block_sums: Buffer,

    max_splats: usize,
    tile_capacity: usize,
    sort_capacity: usize,
    histogram_capacity: usize,
    block_sums_capacity: usize,
    framebuffer_capacity_pixels: usize,

    splats_uploaded: bool,
    previous_total_overlaps: u32,
    previous_valid_count: u32,
    overflow_flag_buffer: Buffer,
    last_render_width: usize,
    last_render_height: usize,
}

#[cfg(feature = "metal")]
impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("max_splats", &self.max_splats)
            .field("tile_capacity", &self.tile_capacity)
            .field("sort_capacity", &self.sort_capacity)
            .field(
                "framebuffer_capacity_pixels",
                &self.framebuffer_capacity_pixels,
            )
            .field("splats_uploaded", &self.splats_uploaded)
            .finish()
    }
}

#[cfg(feature = "metal")]
impl MetalBackend {
    pub fn new(max_splats: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::system_default().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "No Metal device found")
        })?;
        let command_queue = device.new_command_queue();

        let projection_library =
            compile_library(&device, include_str!("../../shaders/projection.metal"))?;
        let prefix_scan_library =
            compile_library(&device, include_str!("../../shaders/prefix_scan.metal"))?;
        let radix_sort_library =
            compile_library(&device, include_str!("../../shaders/radix_sort.metal"))?;
        let tile_ops_library =
            compile_library(&device, include_str!("../../shaders/tile_ops.metal"))?;
        let tile_rasterize_library =
            compile_library(&device, include_str!("../../shaders/tile_rasterize.metal"))?;

        let project_splats_pipeline =
            create_pipeline(&device, &projection_library, "project_splats")?;
        let prefix_scan_blocks_pipeline =
            create_pipeline(&device, &prefix_scan_library, "prefix_scan_blocks")?;
        let prefix_scan_add_offsets_pipeline =
            create_pipeline(&device, &prefix_scan_library, "prefix_scan_add_offsets")?;
        let radix_sort_histogram_pipeline =
            create_pipeline(&device, &radix_sort_library, "radix_sort_histogram")?;
        let radix_sort_scatter_pipeline =
            create_pipeline(&device, &radix_sort_library, "radix_sort_scatter")?;
        let count_tile_overlaps_pipeline =
            create_pipeline(&device, &tile_ops_library, "count_tile_overlaps")?;
        let clamp_total_overlaps_pipeline =
            create_pipeline(&device, &tile_ops_library, "clamp_total_overlaps")?;
        let emit_tile_keys_pipeline =
            create_pipeline(&device, &tile_ops_library, "emit_tile_keys")?;
        let rasterize_tiles_pipeline =
            create_pipeline(&device, &tile_rasterize_library, "rasterize_tiles")?;

        let splat_buffer = new_shared_buffer(
            &device,
            max_splats
                .checked_mul(mem::size_of::<GpuSplatData>())
                .ok_or_else(|| std::io::Error::other("splat buffer size overflow"))?,
        );
        let projected_buffer = new_private_buffer(
            &device,
            max_splats
                .checked_mul(mem::size_of::<GpuProjectedSplat>())
                .ok_or_else(|| std::io::Error::other("projected buffer size overflow"))?,
        );

        let camera_buffer = new_shared_buffer(&device, mem::size_of::<GpuCameraData>());
        let valid_count_buffer = new_shared_buffer(&device, mem::size_of::<u32>());
        let total_overlaps_buffer = new_shared_buffer(&device, mem::size_of::<u32>());
        let overflow_flag_buffer = new_shared_buffer(&device, mem::size_of::<u32>());
        let tile_config_buffer = new_shared_buffer(&device, mem::size_of::<TileConfig>());
        let framebuffer = new_shared_buffer(&device, mem::size_of::<u32>());
        let tile_counts = new_private_buffer(&device, mem::size_of::<u32>());
        let tile_offsets = new_private_buffer(&device, mem::size_of::<u32>() * 2);
        let tile_counters = new_private_buffer(&device, mem::size_of::<u32>());
        let sort_keys_a = new_private_buffer(&device, mem::size_of::<u32>());
        let sort_keys_b = new_private_buffer(&device, mem::size_of::<u32>());
        let sort_values_a = new_private_buffer(&device, mem::size_of::<u32>());
        let sort_values_b = new_private_buffer(&device, mem::size_of::<u32>());
        let radix_histograms = new_private_buffer(&device, mem::size_of::<u32>());
        let block_sums = new_private_buffer(&device, mem::size_of::<u32>());

        Ok(Self {
            device,
            command_queue,

            project_splats_pipeline,
            prefix_scan_blocks_pipeline,
            prefix_scan_add_offsets_pipeline,
            radix_sort_histogram_pipeline,
            radix_sort_scatter_pipeline,
            count_tile_overlaps_pipeline,
            clamp_total_overlaps_pipeline,
            emit_tile_keys_pipeline,
            rasterize_tiles_pipeline,

            splat_buffer,
            camera_buffer,
            valid_count_buffer,
            total_overlaps_buffer,
            tile_config_buffer,
            framebuffer,

            projected_buffer,
            tile_counts,
            tile_offsets,
            tile_counters,
            sort_keys_a,
            sort_keys_b,
            sort_values_a,
            sort_values_b,
            radix_histograms,
            block_sums,

            max_splats,
            tile_capacity: 1,
            sort_capacity: 1,
            histogram_capacity: 1,
            block_sums_capacity: 1,
            framebuffer_capacity_pixels: 1,

            splats_uploaded: false,
            previous_total_overlaps: 0,
            previous_valid_count: 0,
            overflow_flag_buffer,
            last_render_width: 0,
            last_render_height: 0,
        })
    }

    pub fn upload_splats(&mut self, splats: &[Splat]) -> Result<(), Box<dyn std::error::Error>> {
        if splats.len() > self.max_splats {
            return Err("Too many splats for GPU buffer".into());
        }

        let contents = self.splat_buffer.contents() as *mut GpuSplatData;
        for (i, splat) in splats.iter().enumerate() {
            let packed_color = (splat.color[0] as u32)
                | ((splat.color[1] as u32) << 8)
                | ((splat.color[2] as u32) << 16)
                | (255u32 << 24);

            let gpu = GpuSplatData {
                pos_x: splat.position.x,
                pos_y: splat.position.y,
                pos_z: splat.position.z,
                scale_x: splat.scale.x,
                scale_y: splat.scale.y,
                scale_z: splat.scale.z,
                rot_w: splat.rotation[0],
                rot_x: splat.rotation[1],
                rot_y: splat.rotation[2],
                rot_z: splat.rotation[3],
                opacity: splat.opacity,
                packed_color,
            };

            unsafe {
                *contents.add(i) = gpu;
            }
        }

        self.splats_uploaded = true;
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        self.splats_uploaded
    }

    pub fn render(
        &mut self,
        camera: &Camera,
        screen_width: usize,
        screen_height: usize,
        splat_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

        let screen_width_u32 =
            u32::try_from(screen_width).map_err(|_| "screen width exceeds u32")?;
        let screen_height_u32 =
            u32::try_from(screen_height).map_err(|_| "screen height exceeds u32")?;

        let tile_count_x = div_ceil_u32(screen_width_u32, TILE_SIZE).max(1);
        let tile_count_y = div_ceil_u32(screen_height_u32, TILE_SIZE).max(1);
        let num_tiles_u64 = u64::from(tile_count_x) * u64::from(tile_count_y);
        if num_tiles_u64 > 1023 {
            return Err("Tile count exceeds 10-bit tile_id encoding (max 1023 tiles)".into());
        }
        let num_tiles = usize::try_from(num_tiles_u64).map_err(|_| "tile count overflow")?;

        self.ensure_framebuffer_capacity(screen_width, screen_height)?;
        if splat_count == 0 {
            self.clear_framebuffer(screen_width, screen_height);
            self.last_render_width = screen_width;
            self.last_render_height = screen_height;
            return Ok(());
        }
        self.ensure_tile_capacity(num_tiles)?;

        // Overflow retry loop: if the sort buffer overflows, double capacity and re-render.
        // Limited to 3 attempts to avoid infinite loops.
        const MAX_OVERFLOW_RETRIES: u32 = 3;
        for attempt in 0..=MAX_OVERFLOW_RETRIES {
            let estimated_overlaps = if self.previous_total_overlaps > 0 {
                (self.previous_total_overlaps as usize * 3 / 2).max(splat_count.saturating_mul(4))
            } else {
                splat_count.saturating_mul(4)
            }
            .max(1);
            self.ensure_sort_capacity(estimated_overlaps)?;

            // Size histogram and dispatch based on sort_capacity (the actual
            // buffer size), NOT estimated_overlaps.  The GPU-side total_overlaps
            // counter is clamped to sort_capacity before the radix sort runs, so
            // the maximum number of sort blocks equals ceil(sort_capacity / 256).
            // Using the estimate instead caused out-of-bounds histogram writes
            // whenever actual overlaps exceeded the estimate, corrupting GPU
            // memory and freezing the device on subsequent frames.
            let sort_capacity_u32 =
                u32::try_from(self.sort_capacity).map_err(|_| "sort capacity exceeds u32")?;
            let sort_num_blocks =
                div_ceil_u32(sort_capacity_u32, THREADS_PER_GROUP_1D).max(1);
            let histogram_count = sort_num_blocks
                .checked_mul(RADIX_BUCKETS)
                .ok_or_else(|| std::io::Error::other("histogram count overflow"))?;
            self.ensure_histogram_capacity(histogram_count as usize)?;
            self.ensure_block_sums_capacity_for_count(num_tiles as u32)?;
            self.ensure_block_sums_capacity_for_count(histogram_count)?;

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

            write_shared_struct(&self.camera_buffer, &gpu_camera);
            write_shared_struct(&self.tile_config_buffer, &tile_config);

            let tile_bytes = bytes_for_u32_elems(num_tiles)? as u64;
            let splat_count_u32 =
                u32::try_from(splat_count).map_err(|_| "splat count exceeds u32")?;
            let framebuffer_clear_bytes =
                (self.framebuffer_capacity_pixels * mem::size_of::<u32>()) as u64;
            let histogram_bytes = bytes_for_u32_elems(histogram_count as usize)? as u64;

            let mut keys_in_a = true;
            {
                let command_buffer = self.command_queue.new_command_buffer();

                let blit = command_buffer.new_blit_command_encoder();
                blit.fill_buffer(
                    &self.framebuffer,
                    NSRange::new(0, framebuffer_clear_bytes),
                    0,
                );
                blit.fill_buffer(
                    &self.tile_counts,
                    NSRange::new(0, tile_bytes),
                    0,
                );
                blit.fill_buffer(
                    &self.valid_count_buffer,
                    NSRange::new(0, mem::size_of::<u32>() as u64),
                    0,
                );
                blit.fill_buffer(
                    &self.total_overlaps_buffer,
                    NSRange::new(0, mem::size_of::<u32>() as u64),
                    0,
                );
                blit.fill_buffer(
                    &self.overflow_flag_buffer,
                    NSRange::new(0, mem::size_of::<u32>() as u64),
                    0,
                );
                blit.end_encoding();

                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.project_splats_pipeline);
                encoder.set_buffer(0, Some(&self.splat_buffer), 0);
                encoder.set_buffer(1, Some(&self.projected_buffer), 0);
                encoder.set_buffer(2, Some(&self.valid_count_buffer), 0);
                encoder.set_buffer(3, Some(&self.camera_buffer), 0);
                encoder.set_bytes(
                    4,
                    mem::size_of::<u32>() as u64,
                    &splat_count_u32 as *const _ as *const c_void,
                );
                encoder.set_buffer(5, Some(&self.tile_config_buffer), 0);
                dispatch_1d(encoder, splat_count_u32, THREADS_PER_GROUP_1D);
                encoder.end_encoding();

                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.count_tile_overlaps_pipeline);
                encoder.set_buffer(0, Some(&self.projected_buffer), 0);
                encoder.set_buffer(1, Some(&self.tile_counts), 0);
                encoder.set_buffer(2, Some(&self.total_overlaps_buffer), 0);
                encoder.set_buffer(3, Some(&self.valid_count_buffer), 0);
                encoder.set_buffer(4, Some(&self.tile_config_buffer), 0);
                dispatch_1d(encoder, splat_count_u32, THREADS_PER_GROUP_1D);
                encoder.end_encoding();

                let blit = command_buffer.new_blit_command_encoder();
                blit.copy_from_buffer(
                    &self.tile_counts,
                    0,
                    &self.tile_offsets,
                    0,
                    tile_bytes,
                );
                blit.copy_from_buffer(
                    &self.total_overlaps_buffer,
                    0,
                    &self.tile_offsets,
                    tile_bytes,
                    mem::size_of::<u32>() as u64,
                );
                blit.fill_buffer(
                    &self.tile_counters,
                    NSRange::new(0, tile_bytes),
                    0,
                );
                blit.end_encoding();

                self.encode_prefix_scan_in_place(
                    command_buffer,
                    &self.tile_offsets,
                    0,
                    num_tiles as u32,
                )?;

                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.emit_tile_keys_pipeline);
                encoder.set_buffer(0, Some(&self.projected_buffer), 0);
                encoder.set_buffer(1, Some(&self.tile_offsets), 0);
                encoder.set_buffer(2, Some(&self.tile_counters), 0);
                encoder.set_buffer(3, Some(&self.sort_keys_a), 0);
                encoder.set_buffer(4, Some(&self.sort_values_a), 0);
                encoder.set_buffer(5, Some(&self.valid_count_buffer), 0);
                encoder.set_buffer(6, Some(&self.tile_config_buffer), 0);
                encoder.set_buffer(7, Some(&self.overflow_flag_buffer), 0);
                encoder.set_bytes(
                    8,
                    mem::size_of::<u32>() as u64,
                    &sort_capacity_u32 as *const _ as *const c_void,
                );
                dispatch_1d(encoder, splat_count_u32, THREADS_PER_GROUP_1D);
                encoder.end_encoding();

                // Clamp total_overlaps to sort_capacity so the radix sort's
                // internal num_blocks calculation never exceeds the histogram
                // buffer dimensions.  Without this, when actual overlaps exceed
                // sort_capacity the column-major histogram writes go out of
                // bounds, corrupting GPU memory and causing a device hang.
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.clamp_total_overlaps_pipeline);
                encoder.set_buffer(0, Some(&self.total_overlaps_buffer), 0);
                encoder.set_bytes(
                    1,
                    mem::size_of::<u32>() as u64,
                    &sort_capacity_u32 as *const _ as *const c_void,
                );
                encoder.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(1, 1, 1),
                );
                encoder.end_encoding();

                for bit_offset in [0u32, 8, 16, 24] {
                    let blit = command_buffer.new_blit_command_encoder();
                    blit.fill_buffer(
                        &self.radix_histograms,
                        NSRange::new(0, histogram_bytes),
                        0,
                    );
                    blit.end_encoding();

                    let (keys_in, values_in, keys_out, values_out) = if keys_in_a {
                        (
                            &self.sort_keys_a,
                            &self.sort_values_a,
                            &self.sort_keys_b,
                            &self.sort_values_b,
                        )
                    } else {
                        (
                            &self.sort_keys_b,
                            &self.sort_values_b,
                            &self.sort_keys_a,
                            &self.sort_values_a,
                        )
                    };

                    let encoder = command_buffer.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&self.radix_sort_histogram_pipeline);
                    encoder.set_buffer(0, Some(keys_in), 0);
                    encoder.set_buffer(1, Some(&self.radix_histograms), 0);
                    encoder.set_buffer(2, Some(&self.total_overlaps_buffer), 0);
                    encoder.set_bytes(
                        3,
                        mem::size_of::<u32>() as u64,
                        &bit_offset as *const _ as *const c_void,
                    );
                    dispatch_1d(encoder, sort_capacity_u32, THREADS_PER_GROUP_1D);
                    encoder.end_encoding();

                    self.encode_prefix_scan_in_place(
                        command_buffer,
                        &self.radix_histograms,
                        0,
                        histogram_count,
                    )?;

                    let encoder = command_buffer.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&self.radix_sort_scatter_pipeline);
                    encoder.set_buffer(0, Some(keys_in), 0);
                    encoder.set_buffer(1, Some(values_in), 0);
                    encoder.set_buffer(2, Some(keys_out), 0);
                    encoder.set_buffer(3, Some(values_out), 0);
                    encoder.set_buffer(4, Some(&self.radix_histograms), 0);
                    encoder.set_buffer(5, Some(&self.total_overlaps_buffer), 0);
                    encoder.set_bytes(
                        6,
                        mem::size_of::<u32>() as u64,
                        &bit_offset as *const _ as *const c_void,
                    );
                    dispatch_1d(encoder, sort_capacity_u32, THREADS_PER_GROUP_1D);
                    encoder.end_encoding();

                    keys_in_a = !keys_in_a;
                }

                let (sorted_keys, sorted_values) = if keys_in_a {
                    (&self.sort_keys_a, &self.sort_values_a)
                } else {
                    (&self.sort_keys_b, &self.sort_values_b)
                };

                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.rasterize_tiles_pipeline);
                encoder.set_buffer(0, Some(&self.projected_buffer), 0);
                encoder.set_buffer(1, Some(sorted_keys), 0);
                encoder.set_buffer(2, Some(sorted_values), 0);
                encoder.set_buffer(3, Some(&self.tile_offsets), 0);
                encoder.set_buffer(4, Some(&self.framebuffer), 0);
                encoder.set_buffer(5, Some(&self.tile_config_buffer), 0);
                // tile_offsets always reflects full overlap counts. During an
                // overflow retry attempt that can exceed the currently sized
                // sort buffers, so pass the capacity limit to clamp per-tile
                // ranges in the rasterizer before reading sort_values.
                encoder.set_bytes(
                    6,
                    mem::size_of::<u32>() as u64,
                    &sort_capacity_u32 as *const _ as *const c_void,
                );
                encoder.dispatch_thread_groups(
                    MTLSize::new(u64::from(tile_count_x), u64::from(tile_count_y), 1),
                    MTLSize::new(16, 16, 1),
                );
                encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();
            }

            self.previous_valid_count = read_shared_u32(&self.valid_count_buffer);
            self.previous_total_overlaps = read_shared_u32(&self.total_overlaps_buffer);

            let overflow = read_shared_u32(&self.overflow_flag_buffer);
            if overflow == 0 {
                // No overflow -- render succeeded.
                break;
            }

            if attempt == MAX_OVERFLOW_RETRIES {
                // Exhausted retries -- use whatever we got (may have artifacts).
                break;
            }

            // Double the sort capacity for the retry using the actual total overlaps.
            let new_capacity = if self.previous_total_overlaps > 0 {
                (self.previous_total_overlaps as usize).saturating_mul(2)
            } else {
                // Fallback: if total_overlaps reads as 0 somehow, use a larger default.
                self.sort_capacity.saturating_mul(4).max(splat_count.saturating_mul(16))
            };
            self.ensure_sort_capacity(new_capacity)?;
        }

        self.last_render_width = screen_width;
        self.last_render_height = screen_height;
        Ok(())
    }

    fn ensure_framebuffer_capacity(
        &mut self,
        screen_width: usize,
        screen_height: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pixels = screen_width
            .checked_mul(screen_height)
            .ok_or_else(|| std::io::Error::other("framebuffer pixel count overflow"))?;

        if pixels > self.framebuffer_capacity_pixels {
            self.framebuffer = new_shared_buffer(
                &self.device,
                pixels
                    .checked_mul(mem::size_of::<u32>())
                    .ok_or_else(|| std::io::Error::other("framebuffer size overflow"))?,
            );
            self.framebuffer_capacity_pixels = pixels;
        }

        Ok(())
    }

    fn ensure_tile_capacity(&mut self, num_tiles: usize) -> Result<(), Box<dyn std::error::Error>> {
        if num_tiles <= self.tile_capacity {
            return Ok(());
        }

        self.tile_counts = new_private_buffer(&self.device, bytes_for_u32_elems(num_tiles)?);
        self.tile_offsets = new_private_buffer(&self.device, bytes_for_u32_elems(num_tiles + 1)?);
        self.tile_counters = new_private_buffer(&self.device, bytes_for_u32_elems(num_tiles)?);
        self.tile_capacity = num_tiles;
        Ok(())
    }

    fn ensure_sort_capacity(&mut self, overlaps: usize) -> Result<(), Box<dyn std::error::Error>> {
        if overlaps <= self.sort_capacity {
            return Ok(());
        }

        let new_capacity = overlaps.next_power_of_two();
        let bytes = bytes_for_u32_elems(new_capacity)?;

        self.sort_keys_a = new_private_buffer(&self.device, bytes);
        self.sort_keys_b = new_private_buffer(&self.device, bytes);
        self.sort_values_a = new_private_buffer(&self.device, bytes);
        self.sort_values_b = new_private_buffer(&self.device, bytes);
        self.sort_capacity = new_capacity;
        Ok(())
    }

    fn ensure_histogram_capacity(
        &mut self,
        histogram_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if histogram_count <= self.histogram_capacity {
            return Ok(());
        }

        self.radix_histograms =
            new_private_buffer(&self.device, bytes_for_u32_elems(histogram_count)?);
        self.histogram_capacity = histogram_count;
        Ok(())
    }

    fn ensure_block_sums_capacity_for_count(
        &mut self,
        count: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let required = required_block_sum_elements(count as usize);
        if required <= self.block_sums_capacity {
            return Ok(());
        }

        self.block_sums = new_private_buffer(&self.device, bytes_for_u32_elems(required)?);
        self.block_sums_capacity = required;
        Ok(())
    }

    fn encode_prefix_scan_in_place(
        &self,
        command_buffer: &metal::CommandBufferRef,
        data_buffer: &Buffer,
        data_offset_bytes: u64,
        count: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if count == 0 {
            return Ok(());
        }

        self.encode_prefix_scan_recursive(command_buffer, data_buffer, data_offset_bytes, count, 0)
    }

    fn encode_prefix_scan_recursive(
        &self,
        command_buffer: &metal::CommandBufferRef,
        data_buffer: &Buffer,
        data_offset_bytes: u64,
        count: u32,
        scratch_offset_elems: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if count == 0 {
            return Ok(());
        }

        let num_blocks = div_ceil_u32(count, THREADS_PER_GROUP_1D);
        if num_blocks == 0 {
            return Ok(());
        }

        let block_sums_offset_bytes = scratch_offset_elems
            .checked_mul(mem::size_of::<u32>() as u64)
            .ok_or_else(|| std::io::Error::other("block sums offset overflow"))?;

        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.prefix_scan_blocks_pipeline);
            encoder.set_buffer(0, Some(data_buffer), data_offset_bytes);
            encoder.set_buffer(1, Some(&self.block_sums), block_sums_offset_bytes);
            encoder.set_bytes(
                2,
                mem::size_of::<u32>() as u64,
                &count as *const _ as *const c_void,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(u64::from(num_blocks), 1, 1),
                MTLSize::new(u64::from(THREADS_PER_GROUP_1D), 1, 1),
            );
            encoder.end_encoding();
        }

        if num_blocks > 1 {
            let next_scratch_offset = scratch_offset_elems
                .checked_add(u64::from(num_blocks))
                .ok_or_else(|| std::io::Error::other("block sums recursion overflow"))?;

            self.encode_prefix_scan_recursive(
                command_buffer,
                &self.block_sums,
                block_sums_offset_bytes,
                num_blocks,
                next_scratch_offset,
            )?;

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.prefix_scan_add_offsets_pipeline);
            encoder.set_buffer(0, Some(data_buffer), data_offset_bytes);
            encoder.set_buffer(1, Some(&self.block_sums), block_sums_offset_bytes);
            encoder.set_bytes(
                2,
                mem::size_of::<u32>() as u64,
                &count as *const _ as *const c_void,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(u64::from(num_blocks), 1, 1),
                MTLSize::new(u64::from(THREADS_PER_GROUP_1D), 1, 1),
            );
            encoder.end_encoding();
        }

        Ok(())
    }

    fn clear_framebuffer(&mut self, screen_width: usize, screen_height: usize) {
        if screen_width == 0 || screen_height == 0 {
            return;
        }

        let pixel_count = screen_width.saturating_mul(screen_height);
        let byte_count = pixel_count.saturating_mul(mem::size_of::<u32>());
        unsafe {
            ptr::write_bytes(self.framebuffer.contents() as *mut u8, 0, byte_count);
        }
    }

    pub fn framebuffer_slice(&self) -> &[u32] {
        let pixel_count = self.last_render_width.saturating_mul(self.last_render_height);
        if pixel_count == 0 {
            return &[];
        }

        let src = self.framebuffer.contents() as *const u32;
        unsafe { std::slice::from_raw_parts(src, pixel_count) }
    }
}

#[cfg(feature = "metal")]
fn compile_library(device: &Device, source: &str) -> Result<Library, Box<dyn std::error::Error>> {
    device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|e| std::io::Error::other(e).into())
}

#[cfg(feature = "metal")]
fn create_pipeline(
    device: &Device,
    library: &Library,
    function_name: &str,
) -> Result<ComputePipelineState, Box<dyn std::error::Error>> {
    let function = library
        .get_function(function_name, None)
        .map_err(std::io::Error::other)?;

    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| std::io::Error::other(e).into())
}

#[cfg(feature = "metal")]
fn new_shared_buffer(device: &Device, size_bytes: usize) -> Buffer {
    device.new_buffer(
        size_bytes.max(mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[cfg(feature = "metal")]
fn new_private_buffer(device: &Device, size_bytes: usize) -> Buffer {
    device.new_buffer(
        size_bytes.max(mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModePrivate,
    )
}

#[cfg(feature = "metal")]
fn dispatch_1d(encoder: &metal::ComputeCommandEncoderRef, count: u32, threads_per_group: u32) {
    if count == 0 {
        return;
    }

    let groups = u64::from(div_ceil_u32(count, threads_per_group));
    encoder.dispatch_thread_groups(
        MTLSize::new(groups, 1, 1),
        MTLSize::new(u64::from(threads_per_group), 1, 1),
    );
}

#[cfg(feature = "metal")]
fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

#[cfg(feature = "metal")]
fn bytes_for_u32_elems(count: usize) -> Result<usize, Box<dyn std::error::Error>> {
    count
        .checked_mul(mem::size_of::<u32>())
        .ok_or_else(|| std::io::Error::other("buffer size overflow").into())
}

#[cfg(feature = "metal")]
fn required_block_sum_elements(count: usize) -> usize {
    if count == 0 {
        return 1;
    }

    let mut total = 0usize;
    let mut blocks = count.div_ceil(THREADS_PER_GROUP_1D as usize);
    loop {
        total = total.saturating_add(blocks);
        if blocks <= 1 {
            break;
        }
        blocks = blocks.div_ceil(THREADS_PER_GROUP_1D as usize);
    }
    total.max(1)
}

#[cfg(feature = "metal")]
fn write_shared_struct<T: Copy>(buffer: &Buffer, value: &T) {
    unsafe {
        *(buffer.contents() as *mut T) = *value;
    }
}

#[cfg(feature = "metal")]
fn read_shared_u32(buffer: &Buffer) -> u32 {
    unsafe { *(buffer.contents() as *const u32) }
}

#[cfg(not(feature = "metal"))]
pub struct MetalBackend;

#[cfg(not(feature = "metal"))]
impl MetalBackend {
    pub fn new(_max_splats: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Metal backend not compiled".into())
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, Once, OnceLock};

    use rand::{Rng, SeedableRng};

    use crate::{
        camera::{look_at_origin, Camera},
        demo::generate_demo_splats,
        math::Vec3,
        render::{pipeline, rasterizer, RenderState},
        sort::sort_by_depth,
        splat::Splat,
    };

    static ENV_INIT: Once = Once::new();
    static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

    fn test_guard() -> MutexGuard<'static, ()> {
        let mutex = TEST_MUTEX.get_or_init(|| Mutex::new(()));
        // Recover from poison if a previous test panicked.
        match mutex.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn init_metal_validation_env() {
        ENV_INIT.call_once(|| {
            std::env::set_var("MTL_DEBUG_LAYER", "1");
            std::env::set_var("MTL_SHADER_VALIDATION", "1");
        });
    }

    fn setup_metal_test() -> Option<MutexGuard<'static, ()>> {
        let guard = test_guard();
        init_metal_validation_env();

        if metal::Device::system_default().is_none() {
            eprintln!("Skipping Metal test: no system-default Metal device.");
            return None;
        }

        Some(guard)
    }

    fn make_test_camera() -> Camera {
        let mut camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            -std::f32::consts::FRAC_PI_2,
            0.0,
        );
        look_at_origin(&mut camera);
        camera
    }

    fn make_render_state(width: usize, height: usize) -> RenderState {
        let len = width.saturating_mul(height);
        RenderState {
            framebuffer: vec![[0, 0, 0]; len],
            alpha_buffer: vec![0.0; len],
            depth_buffer: vec![f32::INFINITY; len],
            width,
            height,
        }
    }

    fn unpack_rgb(framebuffer: &[u32]) -> Vec<[u8; 3]> {
        framebuffer
            .iter()
            .map(|&p| [(p & 0xFF) as u8, ((p >> 8) & 0xFF) as u8, ((p >> 16) & 0xFF) as u8])
            .collect()
    }

    fn cpu_reference_framebuffer(splats: &[Splat], width: usize, height: usize) -> Vec<[u8; 3]> {
        let camera = make_test_camera();
        let mut projected = Vec::with_capacity(splats.len());
        let mut visible_count = 0usize;

        pipeline::project_and_cull_splats(
            splats,
            &mut projected,
            &camera,
            width,
            height,
            &mut visible_count,
        );
        sort_by_depth(&mut projected);

        let mut render_state = make_render_state(width, height);
        rasterizer::rasterize_splats(&projected, &mut render_state, width, height);
        render_state.framebuffer
    }

    fn make_center_red_splat() -> Splat {
        Splat {
            position: Vec3::new(0.0, 0.0, 0.0),
            color: [255, 0, 0],
            opacity: 1.0,
            scale: Vec3::new(0.5, 0.5, 0.5),
            rotation: [1.0, 0.0, 0.0, 0.0],
        }
    }

    fn generate_seeded_splats(count: usize, seed: u64) -> Vec<Splat> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut splats = Vec::with_capacity(count);

        for _ in 0..count {
            splats.push(Splat {
                position: Vec3::new(
                    rng.random_range(-1.6_f32..1.6_f32),
                    rng.random_range(-1.6_f32..1.6_f32),
                    rng.random_range(-1.5_f32..1.5_f32),
                ),
                color: [
                    rng.random_range(24_u8..=255_u8),
                    rng.random_range(24_u8..=255_u8),
                    rng.random_range(24_u8..=255_u8),
                ],
                opacity: rng.random_range(0.35_f32..0.95_f32),
                scale: Vec3::new(
                    rng.random_range(0.03_f32..0.12_f32),
                    rng.random_range(0.03_f32..0.12_f32),
                    rng.random_range(0.03_f32..0.12_f32),
                ),
                rotation: [1.0, 0.0, 0.0, 0.0],
            });
        }

        splats
    }

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<GpuSplatData>(), 48);
        assert_eq!(std::mem::size_of::<GpuCameraData>(), 72);
        assert_eq!(std::mem::size_of::<GpuProjectedSplat>(), 52);
        assert_eq!(std::mem::size_of::<TileConfig>(), 16);
        assert_eq!(std::mem::size_of::<RadixSortConfig>(), 8);
    }

    #[test]
    fn test_metal_backend_creation() {
        let _guard = match setup_metal_test() {
            Some(g) => g,
            None => return,
        };

        let backend = MetalBackend::new(16).expect("MetalBackend::new should succeed");
        assert!(!backend.is_ready(), "backend should not be ready before upload");
    }

    #[test]
    fn test_upload_splats() {
        let _guard = match setup_metal_test() {
            Some(g) => g,
            None => return,
        };

        let splats = generate_demo_splats();
        let mut backend =
            MetalBackend::new(splats.len()).expect("MetalBackend::new should succeed");
        backend
            .upload_splats(&splats)
            .expect("upload_splats should succeed for demo data");
        assert!(backend.is_ready(), "backend should report ready after upload");
    }

    #[test]
    fn test_render_empty_scene() {
        let _guard = match setup_metal_test() {
            Some(g) => g,
            None => return,
        };

        let camera = make_test_camera();
        let mut backend = MetalBackend::new(0).expect("MetalBackend::new should succeed");
        backend
            .upload_splats(&[])
            .expect("upload_splats should accept empty slice");

        backend
            .render(&camera, 64, 64, 0)
            .expect("render should succeed for empty scene");
        let framebuffer = backend.framebuffer_slice().to_vec();

        if !framebuffer.is_empty() {
            assert_eq!(framebuffer.len(), 64 * 64);
        }
        assert!(
            framebuffer.is_empty() || framebuffer.iter().all(|&p| p == 0),
            "empty scene should produce empty or fully zero framebuffer"
        );
    }

    #[test]
    fn test_render_single_splat() {
        let _guard = match setup_metal_test() {
            Some(g) => g,
            None => return,
        };

        let camera = make_test_camera();
        let splats = vec![make_center_red_splat()];

        let mut backend = MetalBackend::new(splats.len()).expect("MetalBackend::new should work");
        backend
            .upload_splats(&splats)
            .expect("upload_splats should succeed");

        backend
            .render(&camera, 64, 64, splats.len())
            .expect("render should succeed");
        let framebuffer = backend.framebuffer_slice().to_vec();
        assert_eq!(framebuffer.len(), 64 * 64);
        assert!(
            framebuffer.iter().any(|&p| p != 0),
            "single visible splat should render non-zero pixels"
        );

        let rgb = unpack_rgb(&framebuffer);
        let mut center_has_red = false;
        for y in 31..=33 {
            for x in 31..=33 {
                let idx = y * 64 + x;
                if rgb[idx][0] > 0 {
                    center_has_red = true;
                }
            }
        }
        assert!(
            center_has_red,
            "center neighborhood should contain red contribution"
        );

        let corner_indices = [0usize, 63usize, 63usize * 64, 64usize * 64 - 1];
        for &idx in &corner_indices {
            let px = rgb[idx];
            assert!(
                px[0] <= 2 && px[1] <= 2 && px[2] <= 2,
                "corner pixel should remain black/near-black, got {:?} at {}",
                px,
                idx
            );
        }
    }

    #[test]
    fn test_render_matches_cpu() {
        let _guard = match setup_metal_test() {
            Some(g) => g,
            None => return,
        };

        let width = 128usize;
        let height = 128usize;
        let camera = make_test_camera();
        let splats = generate_seeded_splats(50, 0xC0FFEE_u64);

        let mut backend = MetalBackend::new(splats.len()).expect("MetalBackend::new should work");
        backend
            .upload_splats(&splats)
            .expect("upload_splats should succeed");

        backend
            .render(&camera, width, height, splats.len())
            .expect("GPU render should succeed");
        let gpu_packed = backend.framebuffer_slice().to_vec();
        let gpu_rgb = unpack_rgb(&gpu_packed);
        let cpu_rgb = cpu_reference_framebuffer(&splats, width, height);

        assert_eq!(gpu_rgb.len(), width * height);
        assert_eq!(cpu_rgb.len(), width * height);

        // GPU uses 18-bit depth quantization for tile sorting and inverts covariance
        // per-pixel rather than pre-inverting. This causes ordering differences for
        // closely-spaced splats, which can produce larger pixel-level differences.
        // Tolerance: each channel within +/-8, up to 20% of pixels allowed to exceed.
        let tolerance = 8u8;
        let mut out_of_tolerance = 0usize;
        let mut first_mismatch: Option<(usize, [u8; 3], [u8; 3])> = None;

        for (i, (gpu_px, cpu_px)) in gpu_rgb.iter().zip(cpu_rgb.iter()).enumerate() {
            let within = gpu_px[0].abs_diff(cpu_px[0]) <= tolerance
                && gpu_px[1].abs_diff(cpu_px[1]) <= tolerance
                && gpu_px[2].abs_diff(cpu_px[2]) <= tolerance;
            if !within {
                out_of_tolerance += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, *gpu_px, *cpu_px));
                }
            }
        }

        let pixel_count = width * height;
        let allowed = (pixel_count as f32 * 0.20).ceil() as usize;
        assert!(
            out_of_tolerance <= allowed,
            "GPU/CPU mismatch: {} pixels exceed +/-{} tolerance (allowed {}). First mismatch: {:?}",
            out_of_tolerance,
            tolerance,
            allowed,
            first_mismatch
        );
    }

    #[test]
    fn test_render_demo_splats() {
        let _guard = match setup_metal_test() {
            Some(g) => g,
            None => return,
        };

        let camera = make_test_camera();
        let splats = generate_demo_splats();
        let mut backend =
            MetalBackend::new(splats.len()).expect("MetalBackend::new should succeed");
        backend
            .upload_splats(&splats)
            .expect("upload_splats should succeed");

        backend
            .render(&camera, 128, 128, splats.len())
            .expect("render should succeed");
        let framebuffer = backend.framebuffer_slice().to_vec();
        assert_eq!(framebuffer.len(), 128 * 128);
        assert!(
            framebuffer.iter().any(|&p| p != 0),
            "demo render should not be all black"
        );

        let first = framebuffer[0];
        assert!(
            framebuffer.iter().any(|&p| p != first),
            "demo render should not be a single flat color"
        );
    }

    #[test]
    fn test_resize_handling() {
        let _guard = match setup_metal_test() {
            Some(g) => g,
            None => return,
        };

        let camera = make_test_camera();
        let splats = vec![
            make_center_red_splat(),
            Splat {
                position: Vec3::new(0.8, -0.5, 0.4),
                color: [40, 220, 255],
                opacity: 0.85,
                scale: Vec3::new(0.25, 0.22, 0.20),
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
        ];

        let mut backend = MetalBackend::new(splats.len()).expect("MetalBackend::new should work");
        backend
            .upload_splats(&splats)
            .expect("upload_splats should succeed");

        backend
            .render(&camera, 64, 64, splats.len())
            .expect("64x64 render should succeed");
        let fb_64_a = backend.framebuffer_slice().to_vec();
        backend
            .render(&camera, 256, 256, splats.len())
            .expect("256x256 render should succeed");
        let fb_256 = backend.framebuffer_slice().to_vec();
        backend
            .render(&camera, 64, 64, splats.len())
            .expect("second 64x64 render should succeed");
        let fb_64_b = backend.framebuffer_slice().to_vec();

        assert_eq!(fb_64_a.len(), 64 * 64);
        assert_eq!(fb_256.len(), 256 * 256);
        assert_eq!(fb_64_b.len(), 64 * 64);

        assert!(fb_64_a.iter().any(|&p| p != 0), "first 64x64 output is empty");
        assert!(fb_256.iter().any(|&p| p != 0), "256x256 output is empty");
        assert!(
            fb_64_b.iter().any(|&p| p != 0),
            "second 64x64 output is empty"
        );
    }

    #[test]
    #[ignore]
    fn bench_render_throughput() {
        let _guard = match setup_metal_test() {
            Some(g) => g,
            None => {
                eprintln!("Skipping benchmark: no Metal device available");
                return;
            }
        };

        const SPLAT_COUNT: usize = 5000;
        const RESOLUTION: usize = 256;
        const WARMUP_FRAMES: usize = 10;
        const BENCHMARK_FRAMES: usize = 100;

        let splats = generate_seeded_splats(SPLAT_COUNT, 0xBEEF_5000);
        let camera = make_test_camera();

        let mut backend =
            MetalBackend::new(splats.len()).expect("MetalBackend::new should succeed");
        backend
            .upload_splats(&splats)
            .expect("upload_splats should succeed");

        // Warmup phase: let buffers stabilize and GPU caches warm
        for _ in 0..WARMUP_FRAMES {
            backend
                .render(&camera, RESOLUTION, RESOLUTION, splats.len())
                .expect("warmup render failed");
        }

        // Benchmark phase
        let start = std::time::Instant::now();
        for _ in 0..BENCHMARK_FRAMES {
            backend
                .render(&camera, RESOLUTION, RESOLUTION, splats.len())
                .expect("benchmark render failed");
        }
        let elapsed = start.elapsed();

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let avg_frame_ms = total_ms / BENCHMARK_FRAMES as f64;
        let fps = BENCHMARK_FRAMES as f64 / elapsed.as_secs_f64();

        eprintln!("=== Metal Render Benchmark ===");
        eprintln!("  Splats:      {}", SPLAT_COUNT);
        eprintln!("  Resolution:  {}x{}", RESOLUTION, RESOLUTION);
        eprintln!("  Frames:      {}", BENCHMARK_FRAMES);
        eprintln!("  Total time:  {:.2} ms", total_ms);
        eprintln!("  Avg frame:   {:.3} ms", avg_frame_ms);
        eprintln!("  Throughput:  {:.1} FPS", fps);
        eprintln!("==============================");
    }

    /// Verify that identical camera + splats produce bitwise-identical framebuffers
    /// across multiple renders.  This is the regression test for the sort-key
    /// determinism fix (10-bit tile + 18-bit depth + 4-bit tiebreaker).
    #[test]
    fn test_render_determinism() {
        let _guard = match setup_metal_test() {
            Some(g) => g,
            None => return,
        };

        let width = 128usize;
        let height = 128usize;
        let camera = make_test_camera();
        // Use enough splats to trigger depth collisions in the old 16-bit key.
        let splats = generate_seeded_splats(500, 0xD3AD_BEEF);

        let mut backend =
            MetalBackend::new(splats.len()).expect("MetalBackend::new should succeed");
        backend
            .upload_splats(&splats)
            .expect("upload_splats should succeed");

        // Render once to establish the reference framebuffer.
        backend
            .render(&camera, width, height, splats.len())
            .expect("first render should succeed");
        let reference = backend.framebuffer_slice().to_vec();

        // Render 9 more times and assert bitwise equality every frame.
        for frame in 1..10 {
            backend
                .render(&camera, width, height, splats.len())
                .expect("render should succeed");
            let current = backend.framebuffer_slice().to_vec();
            assert_eq!(
                reference, current,
                "frame {} differs from reference -- sort is non-deterministic",
                frame
            );
        }
    }
}
