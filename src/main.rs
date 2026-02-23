use crossterm::{
    cursor, execute,
    terminal::{self, ClearType, EnterAlternateScreen},
};
use std::io::{self, BufWriter, Write};
use std::path::Path;
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
use render::frame::{run_app_loop, sync_orbit_from_camera};
use render::{AppState, Backend, RenderMode, RenderState};
use terminal_setup::{cleanup_terminal, install_panic_hook};

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

fn load_splats_from_args(args: &[String]) -> AppResult<Vec<splat::Splat>> {
    let mut input_arg: Option<&str> = None;
    for arg in args.iter().skip(1) {
        if arg == "--cpu" || arg == "--metal" || arg == "--flip-y" || arg == "--flip-z" {
            continue;
        }
        input_arg = Some(arg.as_str());
        break;
    }

    let Some(path) = input_arg else {
        return Ok(demo::generate_demo_splats());
    };

    if path == "--demo" {
        return Ok(demo::generate_demo_splats());
    }

    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "ply" => parser::ply::load_ply_file(path),
        "splat" => parser::dot_splat::load_splat_file(path),
        _ => Err("Unsupported input. Use a .ply, .splat, or --demo".into()),
    }
}

fn main() -> AppResult<()> {
    install_panic_hook();

    let args: Vec<String> = std::env::args().collect();

    let use_cpu = args.iter().any(|arg| arg == "--cpu");
    let _use_metal = args.iter().any(|arg| arg == "--metal"); // explicit flag for documentation
    let flip_y = args.iter().any(|arg| arg == "--flip-y");
    let flip_z = args.iter().any(|arg| arg == "--flip-z");
    let backend = if use_cpu {
        Backend::Cpu
    } else {
        #[cfg(feature = "metal")]
        {
            Backend::Metal
        }
        #[cfg(not(feature = "metal"))]
        {
            Backend::Cpu
        }
    };

    let mut splats = load_splats_from_args(&args)?;
    if flip_y || flip_z {
        for splat in splats.iter_mut() {
            if flip_y {
                splat.position.y = -splat.position.y;
            }
            if flip_z {
                splat.position.z = -splat.position.z;
            }
        }
    }

    let (cols, rows) = terminal::size().unwrap_or((120, 40));
    let width = cols.max(1) as usize;
    let height = rows.max(1) as usize * 2;

    let mut camera = Camera::new(Vec3::new(0.0, 0.0, 5.0), -std::f32::consts::FRAC_PI_2, 0.0);
    camera::look_at_origin(&mut camera);

    #[cfg(feature = "metal")]
    let mut metal_backend = if backend == Backend::Metal {
        Some(render::metal::MetalBackend::new(1_000_000)?)
    } else {
        None
    };

    #[cfg(feature = "metal")]
    if let Some(ref mut mb) = metal_backend {
        mb.upload_splats(&splats)?;
    }

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
        show_hud: true,
        auto_orbit: false,
        move_speed: 0.30,
        frame_count: 0,
        last_frame_time: Instant::now(),
        fps: 0.0,
        visible_splat_count: 0,
        orbit_angle: 0.0,
        orbit_radius: 5.0,
        orbit_height: 0.0,
        supersample_factor: 2,
        render_mode: RenderMode::Halfblock,
        backend,
        #[cfg(feature = "metal")]
        metal_backend,
    };
    sync_orbit_from_camera(&mut app_state);

    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = BufWriter::with_capacity(1024 * 1024, io::stdout());

    execute!(
        stdout,
        EnterAlternateScreen,
        cursor::Hide,
        terminal::Clear(ClearType::All)
    )?;
    stdout.flush()?;

    let run_result = run_app_loop(&mut app_state, &mut stdout);
    let cleanup_result = cleanup_terminal(&mut stdout);

    run_result?;
    cleanup_result
}
