use crossterm::{
    cursor, queue,
    style::{Color, Print, SetBackgroundColor, SetForegroundColor},
};
use std::fmt::Write as _;
use std::io::{self, Write};

use super::{AppState, RenderMode};

fn truncate_and_pad_in_place(text: &mut String, width: usize) {
    if width == 0 {
        text.clear();
        return;
    }

    let mut seen_chars = 0usize;
    let mut truncate_byte = None;
    for (idx, _) in text.char_indices() {
        if seen_chars == width {
            truncate_byte = Some(idx);
            break;
        }
        seen_chars += 1;
    }

    if let Some(idx) = truncate_byte {
        text.truncate(idx);
    } else {
        for _ in seen_chars..width {
            text.push(' ');
        }
    }
}

pub fn draw_hud(
    app_state: &mut AppState,
    cols: u16,
    rows: u16,
    ss: usize,
    stdout: &mut impl Write,
) -> io::Result<()> {
    let width = cols as usize;
    let term_cols = cols as usize;
    let term_rows = rows as usize;
    let hud = &mut app_state.hud_string_buf;
    hud.clear();
    write!(
        hud,
        "FPS:{:>5.1}  Splats:{}/{}  Pos:({:>6.2},{:>6.2},{:>6.2})  Speed:{:.2}  Orbit:{}  Mode:{}  Backend:{}  SS:",
        app_state.fps,
        app_state.visible_splat_count,
        app_state.splats.len(),
        app_state.camera.position.x,
        app_state.camera.position.y,
        app_state.camera.position.z,
        app_state.move_speed,
        if app_state.auto_orbit { "ON" } else { "OFF" },
        app_state.render_mode.name(),
        app_state.backend.name()
    )
    .map_err(|_| io::Error::other("failed to format HUD"))?;

    if app_state.render_mode == RenderMode::Halfblock {
        write!(
            hud,
            "{}x [{}x{}]",
            app_state.supersample_factor,
            term_cols * ss,
            term_rows * 2 * ss
        )
        .map_err(|_| io::Error::other("failed to format HUD"))?;
    } else {
        hud.push_str("N/A");
    }

    write!(hud, "  Cores:{}", rayon::current_num_threads())
        .map_err(|_| io::Error::other("failed to format HUD"))?;
    #[cfg(feature = "metal")]
    {
        hud.push_str("  GPU:");
        if let Some(err) = app_state.last_gpu_error.as_deref() {
            write!(hud, "ERR:{err}").map_err(|_| io::Error::other("failed to format HUD"))?;
        } else if app_state.gpu_fallback_active {
            hud.push_str("DISABLED");
        } else {
            hud.push_str("OK");
        }
    }
    truncate_and_pad_in_place(hud, width);

    queue!(
        stdout,
        cursor::MoveTo(0, 0),
        SetBackgroundColor(Color::Rgb { r: 0, g: 0, b: 0 }),
        SetForegroundColor(Color::Rgb {
            r: 245,
            g: 245,
            b: 245
        }),
        Print(hud.as_str())
    )?;

    let controls = "WASD:Move  Arrows:Look  +/-:Speed  Space:Orbit  M:Mode  Tab:HUD  R:Reset  1/2/3:SS  Q/Esc:Quit";
    hud.clear();
    hud.push_str(controls);
    truncate_and_pad_in_place(hud, width);

    queue!(
        stdout,
        cursor::MoveTo(0, rows - 1),
        SetBackgroundColor(Color::Rgb { r: 0, g: 0, b: 0 }),
        SetForegroundColor(Color::Rgb {
            r: 220,
            g: 220,
            b: 220
        }),
        Print(hud.as_str())
    )?;

    Ok(())
}
