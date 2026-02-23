pub mod ascii;
pub mod block_density;
pub mod braille;
pub mod halfblock;
pub mod matrix;
pub mod point_cloud;

pub const POINT_CLOUD_CHARS: &[char] = &[
    '.', '\u{00B7}', '\u{2218}', '\u{25CB}', '\u{25CF}', '\u{25C9}', '\u{2605}', '\u{2726}',
];
pub const MATRIX_CHARS: &[char] = &[
    '\u{FF8A}', '\u{FF90}', '\u{FF8B}', '\u{FF70}', '\u{FF73}', '\u{FF7C}', '\u{FF85}', '\u{FF93}',
    '\u{FF86}', '\u{FF7B}', '\u{FF9C}', '\u{FF82}', '\u{FF75}', '\u{FF98}', '0', '1', '2', '3',
    '4', '5', '6', '7', '8', '9', '\u{2211}', '\u{220F}', '\u{222B}', '\u{2202}', '\u{2207}',
];
pub const BLOCK_DENSITY_CHARS: &[char] = &['\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}'];
pub const ASCII_DENSITY_RAMP: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

pub fn depth_attenuation(depth: f32) -> f32 {
    1.0 / (1.0 + depth.max(0.0) * 0.15)
}

pub fn is_hud_overlay_row(show_hud: bool, row: usize, term_rows: usize) -> bool {
    show_hud && (row == 0 || row == term_rows.saturating_sub(1))
}
