use crossterm::{
    cursor,
    queue,
    style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor},
    terminal,
};
use std::io::{self, Write};
use std::time::Instant;

#[cfg(feature = "metal")]
use crate::sort::sort_by_depth;
#[cfg(feature = "metal")]
use super::Backend;
use super::{AppState, AppResult, RenderMode, HALF_BLOCK, FRAME_TARGET};

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
            {
                let mut used_metal = false;
                if app_state.backend == Backend::Metal {
                    if let Some(ref mut metal_backend) = app_state.metal_backend {
                        if metal_backend.is_ready() {
                            if let Ok(mut projected) = metal_backend.project(
                                &app_state.camera,
                                ss_width,
                                ss_height,
                                app_state.splats.len(),
                            ) {
                                // GPU projected, CPU sorts (radix sort is faster
                                // than GPU bitonic sort due to dispatch overhead)
                                sort_by_depth(&mut projected);
                                app_state.visible_splat_count = projected.len();
                                app_state.projected_splats = projected;
                                used_metal = true;
                            }
                        }
                    }
                }

                if !used_metal {
                    super::pipeline::cpu_project_and_sort(app_state, ss_width, ss_height);
                }
            }

            #[cfg(not(feature = "metal"))]
            {
                super::pipeline::cpu_project_and_sort(app_state, ss_width, ss_height);
            }

            // Always use CPU rasterizer (bounding-box per-splat is faster
            // than the GPU's naive per-pixel-all-splats approach)
            super::rasterizer::rasterize_splats(
                &app_state.projected_splats,
                &mut app_state.render_state,
                ss_width,
                ss_height,
            );

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

            #[cfg(feature = "metal")]
            {
                let mut used_metal = false;
                if app_state.backend == Backend::Metal {
                    if let Some(ref mut metal_backend) = app_state.metal_backend {
                        if metal_backend.is_ready() {
                            if let Ok(mut projected) = metal_backend.project(
                                &app_state.camera,
                                proj_w,
                                proj_h,
                                app_state.splats.len(),
                            ) {
                                sort_by_depth(&mut projected);
                                app_state.visible_splat_count = projected.len();
                                app_state.projected_splats = projected;
                                used_metal = true;
                            }
                        }
                    }
                }

                if !used_metal {
                    super::pipeline::cpu_project_and_sort(app_state, proj_w, proj_h);
                }
            }

            #[cfg(not(feature = "metal"))]
            {
                super::pipeline::cpu_project_and_sort(app_state, proj_w, proj_h);
            }

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

pub fn run_app_loop(app_state: &mut AppState, stdout: &mut io::BufWriter<io::Stdout>) -> AppResult<()> {
    loop {
        let frame_start = Instant::now();

        // Drain all pending input events -- never skip
        if crate::input::process_input_events(app_state)? {
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
        if spent < FRAME_TARGET {
            std::thread::sleep(FRAME_TARGET - spent);
        }
    }

    Ok(())
}
