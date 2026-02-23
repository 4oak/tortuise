mod buffers;
mod error;
mod pipeline;
mod render;
mod render_attempt;
mod sort;
mod sync;
#[cfg(test)]
mod tests;
mod types;

use metal::{Buffer, CommandQueue, ComputePipelineState, Device};

pub use error::MetalRenderError;

pub struct MetalBackend {
    pub(super) device: Device,
    pub(super) command_queue: CommandQueue,

    pub(super) project_splats_pipeline: ComputePipelineState,
    pub(super) prefix_scan_blocks_pipeline: ComputePipelineState,
    pub(super) prefix_scan_add_offsets_pipeline: ComputePipelineState,
    pub(super) radix_sort_histogram_pipeline: ComputePipelineState,
    pub(super) radix_sort_scatter_pipeline: ComputePipelineState,
    pub(super) count_tile_overlaps_pipeline: ComputePipelineState,
    pub(super) emit_tile_keys_pipeline: ComputePipelineState,
    pub(super) rasterize_tiles_pipeline: ComputePipelineState,

    pub(super) splat_buffer: Buffer,
    pub(super) camera_buffer: Buffer,
    pub(super) valid_count_buffer: Buffer,
    pub(super) total_overlaps_buffer: Buffer,
    pub(super) tile_config_buffer: Buffer,
    pub(super) framebuffer: Buffer,

    pub(super) projected_buffer: Buffer,
    pub(super) tile_counts: Buffer,
    pub(super) tile_offsets: Buffer,
    pub(super) tile_counters: Buffer,
    pub(super) sort_keys_a: Buffer,
    pub(super) sort_keys_b: Buffer,
    pub(super) sort_values_a: Buffer,
    pub(super) sort_values_b: Buffer,
    pub(super) radix_histograms: Buffer,
    pub(super) block_sums: Buffer,

    pub(super) max_splats: usize,
    pub(super) tile_capacity: usize,
    pub(super) sort_capacity: usize,
    pub(super) histogram_capacity: usize,
    pub(super) block_sums_capacity: usize,
    pub(super) framebuffer_capacity_pixels: usize,

    pub(super) splats_uploaded: bool,
    pub(super) previous_total_overlaps: u32,
    pub(super) overflow_flag_buffer: Buffer,
    pub(super) last_render_width: usize,
    pub(super) last_render_height: usize,
    pub(super) frames_below_threshold: u32,
    pub(super) gpu_disabled: bool,
}

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
            .field("gpu_disabled", &self.gpu_disabled)
            .finish()
    }
}
