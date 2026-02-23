use crossterm::{
    cursor, execute,
    style::ResetColor,
    terminal::{self, ClearType, LeaveAlternateScreen},
};
use std::io::{self, BufWriter, Write};
use std::panic;

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn install_panic_hook() {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        let raw_mode_result = terminal::disable_raw_mode();
        let mut stdout = io::stdout();
        let cleanup_result = execute!(
            stdout,
            ResetColor,
            cursor::Show,
            LeaveAlternateScreen,
            terminal::Clear(ClearType::All)
        );
        if raw_mode_result.is_err() || cleanup_result.is_err() {
            let mut stderr = io::stderr();
            let _ = writeln!(
                stderr,
                "panic cleanup fallback: writing terminal reset escape sequences"
            );
            let _ = stderr.write_all(b"\x1b[?1049l\x1b[?25h\x1b[0m");
            let _ = stderr.flush();
        }
        default_hook(panic_info);
    }));
}

pub fn cleanup_terminal(
    stdout: &mut BufWriter<io::Stdout>,
    last_gpu_error: Option<&str>,
) -> AppResult<()> {
    execute!(
        stdout,
        ResetColor,
        cursor::Show,
        LeaveAlternateScreen,
        terminal::Clear(ClearType::All)
    )?;
    stdout.flush()?;
    crossterm::terminal::disable_raw_mode()?;
    #[cfg(feature = "metal")]
    if let Some(err) = last_gpu_error {
        eprintln!("Last GPU error: {err}");
    }
    #[cfg(not(feature = "metal"))]
    let _ = last_gpu_error;
    Ok(())
}
