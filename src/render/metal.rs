#[cfg(feature = "metal")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};
#[cfg(feature = "metal")]
use std::ffi::c_void;
#[cfg(feature = "metal")]
use std::mem;

use crate::{camera::Camera, math::Vec3, splat::{ProjectedSplat, Splat}};

// GPU struct definitions - must match MSL layout exactly
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
}

// Hard fail at compile time if Rust<->MSL layout drifts.
const _: [(); 48] = [(); std::mem::size_of::<GpuSplatData>()];
const _: [(); 72] = [(); std::mem::size_of::<GpuCameraData>()];
const _: [(); 44] = [(); std::mem::size_of::<GpuProjectedSplat>()];

#[cfg(feature = "metal")]
#[allow(dead_code)]
pub struct MetalBackend {
    device: Device,
    command_queue: CommandQueue,
    projection_pipeline: ComputePipelineState,

    // Buffers
    splat_buffer: Buffer,
    projected_buffer: Buffer,
    camera_buffer: Buffer,
    valid_count_buffer: Buffer,

    // Host-side data
    max_splats: usize,
    splats_uploaded: bool,
}

#[cfg(feature = "metal")]
impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("max_splats", &self.max_splats)
            .field("splats_uploaded", &self.splats_uploaded)
            .finish()
    }
}

#[cfg(feature = "metal")]
impl MetalBackend {
    pub fn new(
        max_splats: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let _ = Vec3::ZERO;

        let device = Device::system_default().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "No Metal device found")
        })?;
        let command_queue = device.new_command_queue();

        // Compile projection shader (only shader we need - sort is done on CPU)
        let projection_source = include_str!("../../shaders/projection.metal");

        let library: Library = device
            .new_library_with_source(projection_source, &CompileOptions::new())
            .map_err(std::io::Error::other)?;
        let projection_function = library
            .get_function("project_splats", None)
            .map_err(std::io::Error::other)?;
        let projection_pipeline = device
            .new_compute_pipeline_state_with_function(&projection_function)
            .map_err(std::io::Error::other)?;

        // Create buffers
        let splat_buffer = device.new_buffer(
            (max_splats * mem::size_of::<GpuSplatData>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let projected_buffer = device.new_buffer(
            (max_splats * mem::size_of::<GpuProjectedSplat>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let camera_buffer = device.new_buffer(
            mem::size_of::<GpuCameraData>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let valid_count_buffer = device.new_buffer(
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device,
            command_queue,
            projection_pipeline,
            splat_buffer,
            projected_buffer,
            camera_buffer,
            valid_count_buffer,
            max_splats,
            splats_uploaded: false,
        })
    }

    pub fn upload_splats(&mut self, splats: &[Splat]) -> Result<(), Box<dyn std::error::Error>> {
        if splats.len() > self.max_splats {
            return Err("Too many splats for GPU buffer".into());
        }

        let gpu_splats: Vec<GpuSplatData> = splats
            .iter()
            .map(|splat| {
                let packed_color = (splat.color[0] as u32)
                    | ((splat.color[1] as u32) << 8)
                    | ((splat.color[2] as u32) << 16)
                    | (255u32 << 24); // Full alpha

                GpuSplatData {
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
                }
            })
            .collect();

        let contents = self.splat_buffer.contents() as *mut GpuSplatData;
        unsafe {
            std::ptr::copy_nonoverlapping(gpu_splats.as_ptr(), contents, gpu_splats.len());
        }

        self.splats_uploaded = true;
        Ok(())
    }

    /// Returns true if splats have been uploaded to GPU.
    pub fn is_ready(&self) -> bool {
        self.splats_uploaded
    }

    /// GPU projection only - returns unsorted projected splats.
    /// Caller is responsible for sorting (CPU radix sort is faster than GPU bitonic
    /// sort due to the hundreds of dispatch calls bitonic sort requires).
    pub fn project(
        &mut self,
        camera: &Camera,
        screen_width: usize,
        screen_height: usize,
        splat_count: usize,
    ) -> Result<Vec<ProjectedSplat>, Box<dyn std::error::Error>> {
        if splat_count > self.max_splats {
            return Err("Too many splats for GPU buffer".into());
        }

        if splat_count == 0 {
            return Ok(Vec::new());
        }

        // Upload camera data
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

        let camera_contents = self.camera_buffer.contents() as *mut GpuCameraData;
        unsafe {
            *camera_contents = gpu_camera;
        }

        // Reset valid count
        let count_contents = self.valid_count_buffer.contents() as *mut u32;
        unsafe {
            *count_contents = 0;
        }

        // GPU Projection
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.projection_pipeline);
        encoder.set_buffer(0, Some(&self.splat_buffer), 0);
        encoder.set_buffer(1, Some(&self.projected_buffer), 0);
        encoder.set_buffer(2, Some(&self.valid_count_buffer), 0);
        encoder.set_buffer(3, Some(&self.camera_buffer), 0);
        let splat_count_u32 = u32::try_from(splat_count).map_err(|_| "Splat count exceeds u32")?;
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &splat_count_u32 as *const _ as *const c_void,
        );

        let threads_per_group = self.projection_pipeline.thread_execution_width();
        let thread_groups = (splat_count as u64).div_ceil(threads_per_group);

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read valid count
        let valid_count = unsafe { *(self.valid_count_buffer.contents() as *const u32) } as usize;

        if valid_count == 0 {
            return Ok(Vec::new());
        }

        // Read back projected results (unsorted)
        let projected_contents = self.projected_buffer.contents() as *const GpuProjectedSplat;
        let mut results = Vec::with_capacity(valid_count);

        for i in 0..valid_count {
            let gpu_splat = unsafe { *projected_contents.add(i) };
            let det = gpu_splat.cov_a * gpu_splat.cov_c - gpu_splat.cov_b * gpu_splat.cov_b;
            let (inv_a, inv_b, inv_c) = if det.abs() < 1e-8 {
                (0.0, 0.0, 0.0)
            } else {
                let inv_det = 1.0 / det;
                (gpu_splat.cov_c * inv_det, -gpu_splat.cov_b * inv_det, gpu_splat.cov_a * inv_det)
            };
            results.push(ProjectedSplat {
                screen_x: gpu_splat.screen_x,
                screen_y: gpu_splat.screen_y,
                depth: gpu_splat.depth,
                radius_x: gpu_splat.radius_x,
                radius_y: gpu_splat.radius_y,
                color: [
                    (gpu_splat.packed_color & 0xFF) as u8,
                    ((gpu_splat.packed_color >> 8) & 0xFF) as u8,
                    ((gpu_splat.packed_color >> 16) & 0xFF) as u8,
                ],
                opacity: gpu_splat.opacity,
                inv_cov_a: inv_a,
                inv_cov_b: inv_b,
                inv_cov_c: inv_c,
                original_index: gpu_splat.original_index as usize,
            });
        }

        Ok(results)
    }
}

#[cfg(not(feature = "metal"))]
pub struct MetalBackend;

#[cfg(not(feature = "metal"))]
impl MetalBackend {
    pub fn new(
        _max_splats: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Metal backend not compiled".into())
    }
}
