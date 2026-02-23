use crate::camera;
use crate::render::frame::sync_orbit_from_camera;
use crate::render::{AppState, RenderMode};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use std::time::Duration;

pub type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

// --- Input ---

pub fn process_input_events(app_state: &mut AppState) -> AppResult<bool> {
    loop {
        if !event::poll(Duration::ZERO)? {
            break;
        }
        match event::read()? {
            Event::Key(key_event)
                if matches!(key_event.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
            {
                match key_event.code {
                    KeyCode::Esc => return Ok(true),
                    KeyCode::Up => camera::adjust_pitch(&mut app_state.camera, 0.05),
                    KeyCode::Down => camera::adjust_pitch(&mut app_state.camera, -0.05),
                    KeyCode::Left => camera::adjust_yaw(&mut app_state.camera, -0.05),
                    KeyCode::Right => camera::adjust_yaw(&mut app_state.camera, 0.05),
                    KeyCode::Tab => app_state.show_hud = !app_state.show_hud,
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        app_state.move_speed = (app_state.move_speed * 1.2).min(10.0);
                    }
                    KeyCode::Char('-') => {
                        app_state.move_speed = (app_state.move_speed / 1.2).max(0.01);
                    }
                    KeyCode::Char(' ') => {
                        app_state.auto_orbit = !app_state.auto_orbit;
                        if app_state.auto_orbit {
                            sync_orbit_from_camera(app_state);
                        }
                    }
                    KeyCode::Char(c) => match c.to_ascii_lowercase() {
                        'q' => return Ok(true),
                        'w' => camera::move_forward(&mut app_state.camera, app_state.move_speed),
                        's' => camera::move_forward(&mut app_state.camera, -app_state.move_speed),
                        'a' => camera::move_right(&mut app_state.camera, -app_state.move_speed),
                        'd' => camera::move_right(&mut app_state.camera, app_state.move_speed),
                        'm' => {
                            app_state.render_mode = app_state.render_mode.next();
                        }
                        'r' => {
                            camera::reset(&mut app_state.camera);
                            sync_orbit_from_camera(app_state);
                        }
                        '1' if app_state.render_mode == RenderMode::Halfblock => {
                            app_state.supersample_factor = 1;
                        }
                        '2' if app_state.render_mode == RenderMode::Halfblock => {
                            app_state.supersample_factor = 2;
                        }
                        '3' if app_state.render_mode == RenderMode::Halfblock => {
                            app_state.supersample_factor = 3;
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    Ok(false)
}
