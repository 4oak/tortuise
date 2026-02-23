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
        let _ = terminal::disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(
            stdout,
            ResetColor,
            cursor::Show,
            LeaveAlternateScreen,
            terminal::Clear(ClearType::All)
        );
        default_hook(panic_info);
    }));
}

pub fn cleanup_terminal(stdout: &mut BufWriter<io::Stdout>) -> AppResult<()> {
    execute!(
        stdout,
        ResetColor,
        cursor::Show,
        LeaveAlternateScreen,
        terminal::Clear(ClearType::All)
    )?;
    stdout.flush()?;
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}
