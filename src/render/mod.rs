pub mod frame;
mod frame_halfblock;
pub mod hud;
#[cfg(feature = "metal")]
pub mod metal;
pub mod modes;
pub mod pipeline;
pub mod rasterizer;

use std::time::Instant;

use crate::camera::Camera;
use crate::splat::{ProjectedSplat, Splat};
use crossterm::style::Color;

pub fn rgb_to_ansi256(r: u8, g: u8, b: u8) -> u8 {
    if r == g && g == b {
        if r < 8 {
            return 16;
        }
        if r > 248 {
            return 231;
        }
        return 232 + ((r as f32 - 8.0) / 247.0 * 24.0) as u8;
    }
    let ri = (r as f32 / 255.0 * 5.0 + 0.5) as u8;
    let gi = (g as f32 / 255.0 * 5.0 + 0.5) as u8;
    let bi = (b as f32 / 255.0 * 5.0 + 0.5) as u8;
    16 + 36 * ri + 6 * gi + bi
}

pub fn make_color(r: u8, g: u8, b: u8, use_truecolor: bool) -> Color {
    if use_truecolor {
        Color::Rgb { r, g, b }
    } else {
        Color::AnsiValue(rgb_to_ansi256(r, g, b))
    }
}

pub type AppResult<T> = Result<T, Box<dyn std::error::Error>>;
pub type HalfblockCell = ([u8; 3], [u8; 3]);

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
    pub halfblock_cells: Vec<HalfblockCell>,
    pub hud_string_buf: String,
    pub input_state: crate::input::state::InputState,
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
    pub use_truecolor: bool,
    #[cfg(feature = "metal")]
    pub metal_backend: Option<crate::render::metal::MetalBackend>,
    #[cfg(feature = "metal")]
    pub last_gpu_error: Option<String>,
    #[cfg(feature = "metal")]
    pub gpu_fallback_active: bool,
}
