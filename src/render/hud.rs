use crossterm::{
    cursor, queue,
    style::{Color, Print, SetBackgroundColor, SetForegroundColor},
};
use std::io::{self, Write};

use super::{AppState, RenderMode};

pub fn truncate_to_width(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    for c in text.chars().take(width) {
        out.push(c);
    }
    out
}

pub fn draw_hud(
    app_state: &AppState,
    cols: u16,
    rows: u16,
    ss: usize,
    stdout: &mut impl Write,
) -> io::Result<()> {
    let width = cols as usize;
    let term_cols = cols as usize;
    let term_rows = rows as usize;
    let ss_text = if app_state.render_mode == RenderMode::Halfblock {
        format!(
            "{}x [{}x{}]",
            app_state.supersample_factor,
            term_cols * ss,
            term_rows * 2 * ss
        )
    } else {
        "N/A".to_string()
    };
    #[cfg(feature = "metal")]
    let gpu_status = if let Some(err) = app_state.last_gpu_error.as_deref() {
        format!("ERR:{err}")
    } else if app_state.gpu_fallback_active {
        "DISABLED".to_string()
    } else {
        "OK".to_string()
    };
    let hud = format!(
        "FPS:{:>5.1}  Splats:{}/{}  Pos:({:>6.2},{:>6.2},{:>6.2})  Speed:{:.2}  Orbit:{}  Mode:{}  Backend:{}  SS:{}  Cores:{}{}",
        app_state.fps,
        app_state.visible_splat_count,
        app_state.splats.len(),
        app_state.camera.position.x,
        app_state.camera.position.y,
        app_state.camera.position.z,
        app_state.move_speed,
        if app_state.auto_orbit { "ON" } else { "OFF" },
        app_state.render_mode.name(),
        app_state.backend.name(),
        ss_text,
        rayon::current_num_threads(),
        {
            #[cfg(feature = "metal")]
            {
                format!("  GPU:{gpu_status}")
            }
            #[cfg(not(feature = "metal"))]
            {
                String::new()
            }
        }
    );
    let hud_text = truncate_to_width(&hud, width);

    queue!(
        stdout,
        cursor::MoveTo(0, 0),
        SetBackgroundColor(Color::Rgb { r: 0, g: 0, b: 0 }),
        SetForegroundColor(Color::Rgb {
            r: 245,
            g: 245,
            b: 245
        }),
        Print(format!("{:<w$}", hud_text, w = width))
    )?;

    let controls = "WASD:Move  Arrows:Look  +/-:Speed  Space:Orbit  M:Mode  Tab:HUD  R:Reset  1/2/3:SS  Q/Esc:Quit";
    let controls_text = truncate_to_width(controls, width);

    queue!(
        stdout,
        cursor::MoveTo(0, rows - 1),
        SetBackgroundColor(Color::Rgb { r: 0, g: 0, b: 0 }),
        SetForegroundColor(Color::Rgb {
            r: 220,
            g: 220,
            b: 220
        }),
        Print(format!("{:<w$}", controls_text, w = width))
    )?;

    Ok(())
}
