pub mod frame;
pub mod hud;
#[cfg(feature = "metal")]
pub mod metal;
pub mod modes;
pub mod pipeline;
pub mod rasterizer;

use std::time::Instant;

use crate::camera::Camera;
use crate::splat::{ProjectedSplat, Splat};

pub type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

pub const HALF_BLOCK: char = '\u{2584}';
pub const FRAME_TARGET: std::time::Duration = std::time::Duration::from_millis(8);

#[derive(Debug)]
pub struct RenderState {
    pub framebuffer: Vec<[u8; 3]>,
    pub alpha_buffer: Vec<f32>,
    pub depth_buffer: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderMode {
    Halfblock,
    PointCloud,
    Matrix,
    BlockDensity,
    Braille,
    AsciiClassic,
}

impl RenderMode {
    pub fn next(self) -> Self {
        match self {
            Self::Halfblock => Self::PointCloud,
            Self::PointCloud => Self::Matrix,
            Self::Matrix => Self::BlockDensity,
            Self::BlockDensity => Self::Braille,
            Self::Braille => Self::AsciiClassic,
            Self::AsciiClassic => Self::Halfblock,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Halfblock => "Halfblock",
            Self::PointCloud => "PointCloud",
            Self::Matrix => "Matrix",
            Self::BlockDensity => "BlockDensity",
            Self::Braille => "Braille",
            Self::AsciiClassic => "AsciiClassic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "metal")]
    Metal,
}

impl Backend {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(feature = "metal")]
            Self::Metal => "Metal",
        }
    }
}

#[derive(Debug)]
pub struct AppState {
    pub camera: Camera,
    pub splats: Vec<Splat>,
    pub projected_splats: Vec<ProjectedSplat>,
    pub render_state: RenderState,
    pub show_hud: bool,
    pub auto_orbit: bool,
    pub move_speed: f32,
    pub frame_count: u64,
    pub last_frame_time: Instant,
    pub fps: f32,
    pub visible_splat_count: usize,
    pub orbit_angle: f32,
    pub orbit_radius: f32,
    pub orbit_height: f32,
    pub supersample_factor: u32,
    pub render_mode: RenderMode,
    pub backend: Backend,
    #[cfg(feature = "metal")]
    pub metal_backend: Option<crate::render::metal::MetalBackend>,
}
