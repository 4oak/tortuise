use clap::Parser;
use crossterm::{
    cursor,
    event::{KeyboardEnhancementFlags, PushKeyboardEnhancementFlags},
    execute,
    terminal::{self, ClearType, EnterAlternateScreen},
};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

mod camera;
mod demo;
mod input;
mod math;
mod parser;
mod render;
mod sort;
mod splat;
mod terminal_setup;

use camera::Camera;
use math::Vec3;
use render::frame::run_app_loop;
use render::{AppState, Backend, CameraMode, RenderMode, RenderState};
use terminal_setup::{cleanup_terminal, install_panic_hook};

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Parser)]
#[command(name = "tortuise", version, about = "Terminal-native 3D Gaussian Splatting viewer")]
struct Cli {
    /// Path to a .ply or .splat scene file (runs demo if omitted)
    input: Option<PathBuf>,
    #[arg(long, help = "Force CPU rendering")]
    cpu: bool,
    #[cfg(feature = "metal")]
    #[arg(long, help = "Force Metal GPU rendering", conflicts_with = "cpu")]
    metal: bool,
    #[arg(long, help = "Flip Y axis")]
    flip_y: bool,
    #[arg(long, help = "Flip Z axis")]
    flip_z: bool,
    #[arg(long, help = "Run built-in demo scene", conflicts_with = "input")]
    demo: bool,
    #[arg(
        long,
        value_name = "N",
        default_value_t = 1,
        help = "Supersampling factor"
    )]
    supersample: u32,
}

fn find_luigi_ply() -> Option<PathBuf> {
    // 1. Check relative to cwd
    let cwd_candidate = PathBuf::from("scenes/luigi.ply");
    if cwd_candidate.exists() {
        return Some(cwd_candidate);
    }
    // 2. Check next to the executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let exe_candidate = exe_dir.join("scenes/luigi.ply");
            if exe_candidate.exists() {
                return Some(exe_candidate);
            }
        }
    }
    None
}

fn load_splats_from_cli(cli: &Cli) -> AppResult<Vec<splat::Splat>> {
    if cli.demo || cli.input.is_none() {
        // Try to load luigi.ply; fall back to procedural demo if not found
        if let Some(luigi_path) = find_luigi_ply() {
            let path_str = luigi_path.to_str().ok_or("luigi.ply path is non-UTF-8")?;
            return parser::ply::load_ply_file(path_str);
        }
        return Ok(demo::generate_demo_splats());
    }

    let path = match cli.input.as_ref() {
        Some(path) => path,
        None => return Ok(demo::generate_demo_splats()),
    };

    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let path_str = path.to_str().ok_or_else(|| {
        format!(
            "Input path contains non-UTF-8 characters: {}",
            path.display()
        )
    })?;

    match ext.as_str() {
        "ply" => parser::ply::load_ply_file(path_str),
        "splat" => parser::dot_splat::load_splat_file(path_str),
        _ => Err(format!(
            "Unsupported input '{}'. Use a .ply, .splat, or --demo",
            path.display()
        )
        .into()),
    }
}

fn main() -> AppResult<()> {
    install_panic_hook();
    let cli = Cli::parse();

    #[cfg(feature = "metal")]
    let mut backend = if cli.cpu {
        Backend::Cpu
    } else {
        Backend::Metal
    };
    #[cfg(not(feature = "metal"))]
    let backend = Backend::Cpu;

    let mut splats = load_splats_from_cli(&cli)?;
    if cli.flip_y || cli.flip_z {
        for splat in &mut splats {
            if cli.flip_y {
                splat.position.y = -splat.position.y;
            }
            if cli.flip_z {
                splat.position.z = -splat.position.z;
            }
        }
    }

    // Scene center: AABB center of all splats, used as orbit target and camera init.
    let scene_center = if splats.is_empty() {
        Vec3::ZERO
    } else {
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for s in &splats {
            min.x = min.x.min(s.position.x);
            min.y = min.y.min(s.position.y);
            min.z = min.z.min(s.position.z);
            max.x = max.x.max(s.position.x);
            max.y = max.y.max(s.position.y);
            max.z = max.z.max(s.position.z);
        }
        Vec3::new(
            (min.x + max.x) * 0.5,
            (min.y + max.y) * 0.5,
            (min.z + max.z) * 0.5,
        )
    };

    let use_truecolor = match std::env::var("COLORTERM") {
        Ok(val) => !val.is_empty() && (val == "truecolor" || val == "24bit"),
        Err(_) => match std::env::var("TERM_PROGRAM") {
            Ok(prog) => prog != "Apple_Terminal",
            Err(_) => match std::env::var("TERM") {
                Ok(term) => term.contains("ghostty") || term.contains("kitty") || term.contains("wezterm"),
                Err(_) => false,
            },
        },
    };

    let (cols, rows) = terminal::size().unwrap_or((120, 40));
    let width = cols.max(1) as usize;
    let height = rows.max(1) as usize * 2;

    let camera_start = Vec3::new(scene_center.x, scene_center.y, scene_center.z + 5.0);
    let mut camera = Camera::new(camera_start, -std::f32::consts::FRAC_PI_2, 0.0);
    camera::look_at_target(&mut camera, scene_center);

    #[cfg(feature = "metal")]
    let mut metal_backend = if backend == Backend::Metal {
        match render::metal::MetalBackend::new(splats.len()) {
            Ok(mut mb) => {
                mb.upload_splats(&splats)?;
                Some(mb)
            }
            Err(err) => {
                eprintln!(
                    "Warning: Metal initialization failed: {}. Falling back to CPU renderer.",
                    err
                );
                backend = Backend::Cpu;
                None
            }
        }
    } else {
        None
    };

    let mut app_state = AppState {
        camera,
        splats,
        projected_splats: Vec::with_capacity(32_768),
        render_state: RenderState {
            framebuffer: vec![[0, 0, 0]; width * height],
            alpha_buffer: vec![0.0; width * height],
            depth_buffer: vec![f32::INFINITY; width * height],
            width,
            height,
        },
        halfblock_cells: Vec::with_capacity(width * rows.max(1) as usize),
        hud_string_buf: String::with_capacity(512),
        input_state: input::state::InputState::default(),
        show_hud: true,
        camera_mode: CameraMode::Free,
        move_speed: 0.15,
        frame_count: 0,
        last_frame_time: Instant::now(),
        fps: 0.0,
        visible_splat_count: 0,
        orbit_angle: 0.0,
        orbit_radius: 5.0,
        orbit_height: 0.0,
        orbit_target: Vec3::ZERO,
        supersample_factor: cli.supersample.max(1),
        render_mode: RenderMode::Halfblock,
        backend,
        use_truecolor,
        scene_center,
        #[cfg(feature = "metal")]
        metal_backend: metal_backend.take(),
        #[cfg(feature = "metal")]
        last_gpu_error: None,
        #[cfg(feature = "metal")]
        gpu_fallback_active: false,
    };

    crossterm::terminal::enable_raw_mode()?;
    let input_rx = input::thread::spawn_input_thread();
    let mut stdout = BufWriter::with_capacity(1024 * 1024, io::stdout());

    execute!(
        stdout,
        EnterAlternateScreen,
        cursor::Hide,
        terminal::Clear(ClearType::All)
    )?;
    // Request key event kinds so key releases are observable for held-key movement.
    let _ = execute!(
        stdout,
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
        )
    );
    stdout.flush()?;

    let run_result = run_app_loop(&mut app_state, &input_rx, &mut stdout);
    #[cfg(feature = "metal")]
    let cleanup_result = cleanup_terminal(&mut stdout, app_state.last_gpu_error.as_deref());
    #[cfg(not(feature = "metal"))]
    let cleanup_result = cleanup_terminal(&mut stdout, None);

    run_result?;
    cleanup_result
}
