use crossterm::{
    cursor, queue,
    style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor},
    terminal,
};
use std::io::{self, Write};
use std::time::Instant;

#[cfg(feature = "metal")]
use super::Backend;
use super::{AppResult, AppState, RenderMode, FRAME_TARGET, HALF_BLOCK};

/// Halfblock mode emits ~40 bytes/cell; cap at 30 FPS (~33 ms) to avoid
/// overwhelming the terminal's input buffer with ~400 KB/frame.
const HALFBLOCK_FRAME_TARGET: std::time::Duration = std::time::Duration::from_millis(33);

pub fn sync_orbit_from_camera(app_state: &mut AppState) {
    let radius_xz = (app_state.camera.position.x * app_state.camera.position.x
        + app_state.camera.position.z * app_state.camera.position.z)
        .sqrt()
        .max(0.1);
    app_state.orbit_radius = radius_xz;
    app_state.orbit_angle = app_state
        .camera
        .position
        .z
        .atan2(app_state.camera.position.x);
    app_state.orbit_height = app_state.camera.position.y;
}

fn update_auto_orbit(app_state: &mut AppState, delta_time: f32) {
    const ORBIT_SPEED: f32 = 0.55;
    app_state.orbit_angle += ORBIT_SPEED * delta_time;

    app_state.camera.position.x = app_state.orbit_radius * app_state.orbit_angle.cos();
    app_state.camera.position.z = app_state.orbit_radius * app_state.orbit_angle.sin();
    app_state.camera.position.y = app_state.orbit_height;

    crate::camera::look_at_origin(&mut app_state.camera);
}

pub fn render_frame(
    app_state: &mut AppState,
    terminal_size: (u16, u16),
    stdout: &mut impl Write,
) -> io::Result<()> {
    let cols = terminal_size.0.max(1);
    let rows = terminal_size.1.max(1);

    let term_cols = cols as usize;
    let term_rows = rows as usize;

    let ss = app_state.supersample_factor as usize;

    match app_state.render_mode {
        RenderMode::Halfblock => {
            let ss_width = term_cols * ss;
            let ss_height = term_rows * 2 * ss;

            super::pipeline::resize_render_state(&mut app_state.render_state, ss_width, ss_height);
            super::pipeline::clear_framebuffer(&mut app_state.render_state);

            #[cfg(feature = "metal")]
            let gpu_rendered = if app_state.backend == Backend::Metal {
                gpu_render_to_framebuffer(app_state, ss_width, ss_height)
            } else {
                false
            };
            #[cfg(not(feature = "metal"))]
            let gpu_rendered = false;

            if !gpu_rendered {
                super::pipeline::cpu_project_and_sort(app_state, ss_width, ss_height);
                super::rasterizer::rasterize_splats(
                    &app_state.projected_splats,
                    &mut app_state.render_state,
                    ss_width,
                    ss_height,
                );
            }

            let cells = if ss == 1 {
                // Fast path: 1x supersampling -- directly map pairs of pixel rows.
                let fb = &app_state.render_state.framebuffer;
                let mut out = vec![([0u8; 3], [0u8; 3]); term_cols * term_rows];
                for term_row in 0..term_rows {
                    let top_y = term_row * 2;
                    let bot_y = top_y + 1;
                    for x in 0..term_cols {
                        let top = fb[top_y * ss_width + x];
                        let bot = if bot_y < ss_height {
                            fb[bot_y * ss_width + x]
                        } else {
                            [0, 0, 0]
                        };
                        out[term_row * term_cols + x] = (top, bot);
                    }
                }
                out
            } else {
                super::modes::halfblock::downsample_to_terminal(
                    &app_state.render_state.framebuffer,
                    ss_width,
                    ss_height,
                    term_cols,
                    term_rows,
                    ss,
                )
            };

            let mut last_bg: Option<(u8, u8, u8)> = None;
            let mut last_fg: Option<(u8, u8, u8)> = None;

            for term_row in 0..term_rows {
                if super::modes::is_hud_overlay_row(app_state.show_hud, term_row, term_rows) {
                    last_bg = None;
                    last_fg = None;
                    continue;
                }

                queue!(stdout, cursor::MoveTo(0, term_row as u16))?;
                for x in 0..term_cols {
                    let (top, bottom) = cells[term_row * term_cols + x];
                    let bg = (top[0], top[1], top[2]);
                    let fg = (bottom[0], bottom[1], bottom[2]);

                    if last_bg != Some(bg) {
                        queue!(
                            stdout,
                            SetBackgroundColor(Color::Rgb {
                                r: bg.0,
                                g: bg.1,
                                b: bg.2
                            })
                        )?;
                        last_bg = Some(bg);
                    }
                    if last_fg != Some(fg) {
                        queue!(
                            stdout,
                            SetForegroundColor(Color::Rgb {
                                r: fg.0,
                                g: fg.1,
                                b: fg.2
                            })
                        )?;
                        last_fg = Some(fg);
                    }
                    queue!(stdout, Print(HALF_BLOCK))?;
                }
            }
        }
        RenderMode::PointCloud
        | RenderMode::Matrix
        | RenderMode::BlockDensity
        | RenderMode::Braille
        | RenderMode::AsciiClassic => {
            let proj_w = term_cols;
            let proj_h = term_rows * 2;

            // Non-halfblock modes work with projected splats directly;
            // the full GPU rasterize pipeline is only beneficial for halfblock.
            // CPU project+sort is used for all character-based modes.
            super::pipeline::cpu_project_and_sort(app_state, proj_w, proj_h);

            match app_state.render_mode {
                RenderMode::PointCloud => super::modes::point_cloud::render_point_cloud(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                )?,
                RenderMode::Matrix => super::modes::matrix::render_matrix(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                )?,
                RenderMode::BlockDensity => super::modes::block_density::render_block_density(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                )?,
                RenderMode::Braille => super::modes::braille::render_braille(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                )?,
                RenderMode::AsciiClassic => super::modes::ascii::render_ascii_classic(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                )?,
                _ => unreachable!(),
            }
        }
    }

    if app_state.show_hud {
        super::hud::draw_hud(app_state, cols, rows, ss, stdout)?;
    }

    queue!(stdout, ResetColor)?;
    stdout.flush()
}

/// Run the full GPU tile-based pipeline and write results into RenderState.
/// Returns `true` if the GPU path was used, `false` if it was unavailable.
#[cfg(feature = "metal")]
fn gpu_render_to_framebuffer(app_state: &mut AppState, width: usize, height: usize) -> bool {
    let is_ready = match app_state.metal_backend.as_ref() {
        Some(mb) => mb.is_ready(),
        None => return false,
    };
    if !is_ready {
        return false;
    }

    let render_result: Result<(), crate::render::metal::MetalRenderError> =
        match app_state.metal_backend.as_mut() {
            Some(mb) => mb.render(&app_state.camera, width, height, app_state.splats.len()),
            None => return false,
        };

    if let Err(err) = render_result {
        record_gpu_error(app_state, &err);
        if err.should_disable_gpu() {
            app_state.backend = Backend::Cpu;
            app_state.metal_backend = None;
            app_state.gpu_fallback_active = true;
            eprintln!("Metal disabled for remainder of session: {err}");
        }
        return false;
    }

    let packed = match app_state.metal_backend.as_ref() {
        Some(mb) => mb.framebuffer_slice(),
        None => return false,
    };

    // Unpack RGBA u32 into RenderState framebuffer ([u8; 3]) and alpha_buffer.
    let rs = &mut app_state.render_state;
    let pixel_count = width.saturating_mul(height);

    if packed.len() >= pixel_count {
        for (i, &p) in packed.iter().enumerate().take(pixel_count) {
            rs.framebuffer[i] = [
                (p & 0xFF) as u8,
                ((p >> 8) & 0xFF) as u8,
                ((p >> 16) & 0xFF) as u8,
            ];
            rs.alpha_buffer[i] = ((p >> 24) & 0xFF) as f32 / 255.0;
        }
    }

    // GPU pipeline handles projection+sort+rasterize in one shot.
    // We don't get a per-splat visible count from GPU render,
    // but report total splat count for the HUD.
    app_state.visible_splat_count = app_state.splats.len();
    true
}

#[cfg(feature = "metal")]
fn record_gpu_error(app_state: &mut AppState, err: &dyn std::error::Error) {
    app_state.last_gpu_error = Some(err.to_string());
}

pub fn run_app_loop(
    app_state: &mut AppState,
    input_rx: &crate::input::thread::InputReceiver,
    stdout: &mut io::BufWriter<io::Stdout>,
) -> AppResult<()> {
    loop {
        let frame_start = Instant::now();

        // Drain all pending input events -- never skip
        if crate::input::drain_input_events(app_state, input_rx)? {
            break;
        }

        let now = Instant::now();
        let delta_time = now
            .duration_since(app_state.last_frame_time)
            .as_secs_f32()
            .max(1e-6);
        app_state.last_frame_time = now;

        if app_state.auto_orbit {
            update_auto_orbit(app_state, delta_time);
        }
        crate::input::state::apply_movement_from_held_keys(app_state, delta_time);

        let terminal_size = terminal::size()?;
        render_frame(app_state, terminal_size, stdout)?;

        app_state.frame_count += 1;
        let instant_fps = 1.0 / delta_time;
        app_state.fps = if app_state.fps <= 0.01 {
            instant_fps
        } else {
            0.90 * app_state.fps + 0.10 * instant_fps
        };

        let spent = frame_start.elapsed();
        let target = if app_state.render_mode == RenderMode::Halfblock {
            HALFBLOCK_FRAME_TARGET
        } else {
            FRAME_TARGET
        };
        if spent < target {
            std::thread::sleep(target - spent);
        }
    }

    Ok(())
}
