use super::{types, MetalBackend};
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
    let mut camera = Camera::new(Vec3::new(0.0, 0.0, 5.0), -std::f32::consts::FRAC_PI_2, 0.0);
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
        .map(|&p| {
            [
                (p & 0xFF) as u8,
                ((p >> 8) & 0xFF) as u8,
                ((p >> 16) & 0xFF) as u8,
            ]
        })
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
    assert_eq!(std::mem::size_of::<types::GpuSplatData>(), 48);
    assert_eq!(std::mem::size_of::<types::GpuCameraData>(), 72);
    assert_eq!(std::mem::size_of::<types::GpuProjectedSplat>(), 52);
    assert_eq!(std::mem::size_of::<types::TileConfig>(), 16);
}

#[test]
fn test_metal_backend_creation() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let backend = MetalBackend::new(16).expect("MetalBackend::new should succeed");
    assert!(
        !backend.is_ready(),
        "backend should not be ready before upload"
    );
}

#[test]
fn test_upload_splats() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let splats = generate_demo_splats();
    let mut backend = MetalBackend::new(splats.len()).expect("MetalBackend::new should succeed");
    backend
        .upload_splats(&splats)
        .expect("upload_splats should succeed for demo data");
    assert!(
        backend.is_ready(),
        "backend should report ready after upload"
    );
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
    assert!(framebuffer.is_empty() || framebuffer.iter().all(|&p| p == 0));
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

    let tolerance = 8u8;
    let mut out_of_tolerance = 0usize;
    for (gpu_px, cpu_px) in gpu_rgb.iter().zip(cpu_rgb.iter()) {
        let within = gpu_px[0].abs_diff(cpu_px[0]) <= tolerance
            && gpu_px[1].abs_diff(cpu_px[1]) <= tolerance
            && gpu_px[2].abs_diff(cpu_px[2]) <= tolerance;
        if !within {
            out_of_tolerance += 1;
        }
    }

    let pixel_count = width * height;
    let allowed = (pixel_count as f32 * 0.20).ceil() as usize;
    assert!(out_of_tolerance <= allowed);
}

#[test]
fn test_resize_handling() {
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
}
