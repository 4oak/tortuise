pub mod state;
pub mod thread;

use crate::camera;
use crate::render::frame::sync_orbit_from_camera;
use crate::render::{AppState, RenderMode};
use crossterm::event::{Event, KeyCode, KeyEventKind};
use std::sync::mpsc::{Receiver, TryRecvError};

pub type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn drain_input_events(
    app_state: &mut AppState,
    input_rx: &Receiver<crate::input::thread::InputMessage>,
) -> AppResult<bool> {
    loop {
        match input_rx.try_recv() {
            Ok(crate::input::thread::InputMessage::Event(event)) => {
                handle_input_event(app_state, event)?;
                if app_state.input_state.quit_requested {
                    return Ok(true);
                }
            }
            Ok(crate::input::thread::InputMessage::ReadError(err)) => {
                return Err(format!("Input thread read failed: {err}").into());
            }
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
        }
    }

    Ok(app_state.input_state.quit_requested)
}

pub fn handle_input_event(app_state: &mut AppState, event: Event) -> AppResult<()> {
    match event {
        Event::Key(key_event) => {
            match key_event.code {
                KeyCode::Char(c) => {
                    let lc = c.to_ascii_lowercase();
                    if matches!(
                        key_event.kind,
                        KeyEventKind::Press | KeyEventKind::Repeat | KeyEventKind::Release
                    ) {
                        match lc {
                            'w' => app_state.input_state.held.forward = key_event.kind != KeyEventKind::Release,
                            's' => app_state.input_state.held.back = key_event.kind != KeyEventKind::Release,
                            'a' => app_state.input_state.held.left = key_event.kind != KeyEventKind::Release,
                            'd' => app_state.input_state.held.right = key_event.kind != KeyEventKind::Release,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }

            if !matches!(key_event.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                return Ok(());
            }

            match key_event.code {
                KeyCode::Esc => app_state.input_state.quit_requested = true,
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
                    'q' => app_state.input_state.quit_requested = true,
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::math::Vec3;
    use crate::render::{AppState, Backend, RenderMode, RenderState};
    use std::sync::mpsc;
    use std::time::Instant;

    fn make_state() -> AppState {
        AppState {
            camera: Camera::new(
                Vec3::new(0.0, 0.0, 5.0),
                -std::f32::consts::FRAC_PI_2,
                0.0,
            ),
            splats: Vec::new(),
            projected_splats: Vec::new(),
            render_state: RenderState {
                framebuffer: vec![[0, 0, 0]; 4],
                alpha_buffer: vec![0.0; 4],
                depth_buffer: vec![f32::INFINITY; 4],
                width: 2,
                height: 2,
            },
            input_state: crate::input::state::InputState::default(),
            show_hud: true,
            auto_orbit: false,
            move_speed: 0.3,
            frame_count: 0,
            last_frame_time: Instant::now(),
            fps: 0.0,
            visible_splat_count: 0,
            orbit_angle: 0.0,
            orbit_radius: 5.0,
            orbit_height: 0.0,
            supersample_factor: 1,
            render_mode: RenderMode::Halfblock,
            backend: Backend::Cpu,
            #[cfg(feature = "metal")]
            metal_backend: None,
        }
    }

    #[test]
    fn held_keys_toggle_on_press_and_release() {
        let mut app = make_state();
        handle_input_event(
            &mut app,
            Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('w'),
                crossterm::event::KeyModifiers::NONE,
            )),
        )
        .expect("press should succeed");
        assert!(app.input_state.held.forward);

        let release = crossterm::event::KeyEvent {
            code: KeyCode::Char('w'),
            modifiers: crossterm::event::KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: crossterm::event::KeyEventState::NONE,
        };
        handle_input_event(&mut app, Event::Key(release)).expect("release should succeed");
        assert!(!app.input_state.held.forward);
    }

    #[test]
    fn drain_consumes_all_queued_events() {
        let (tx, rx) = mpsc::channel();
        tx.send(crate::input::thread::InputMessage::Event(Event::Key(
            crossterm::event::KeyEvent::new(KeyCode::Char('w'), crossterm::event::KeyModifiers::NONE),
        )))
        .expect("send w");
        tx.send(crate::input::thread::InputMessage::Event(Event::Key(
            crossterm::event::KeyEvent::new(KeyCode::Char('a'), crossterm::event::KeyModifiers::NONE),
        )))
        .expect("send a");

        let mut app = make_state();
        let quit = drain_input_events(&mut app, &rx).expect("drain should succeed");
        assert!(!quit);
        assert!(app.input_state.held.forward);
        assert!(app.input_state.held.left);
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
    }
}
