// --- Downsample ---

pub fn downsample_to_terminal(
    fb: &[[u8; 3]],
    ss_width: usize,
    ss_height: usize,
    term_cols: usize,
    term_rows: usize,
    ss: usize,
) -> Vec<([u8; 3], [u8; 3])> {
    let mut out = vec![([0u8; 3], [0u8; 3]); term_cols * term_rows];

    for term_row in 0..term_rows {
        for term_col in 0..term_cols {
            // top pixel block: x in [col*ss .. (col+1)*ss], y in [row*2*ss .. (row*2+1)*ss]
            let x0 = term_col * ss;
            let x1 = ((term_col + 1) * ss).min(ss_width);
            let top_y0 = term_row * 2 * ss;
            let top_y1 = (term_row * 2 * ss + ss).min(ss_height);
            let bot_y0 = (term_row * 2 * ss + ss).min(ss_height);
            let bot_y1 = ((term_row + 1) * 2 * ss).min(ss_height);

            let mut tr = 0u32;
            let mut tg = 0u32;
            let mut tb = 0u32;
            let mut t_count = 0u32;

            for y in top_y0..top_y1 {
                for x in x0..x1 {
                    let p = fb[y * ss_width + x];
                    tr += p[0] as u32;
                    tg += p[1] as u32;
                    tb += p[2] as u32;
                    t_count += 1;
                }
            }

            let mut br = 0u32;
            let mut bg_g = 0u32;
            let mut bb = 0u32;
            let mut b_count = 0u32;

            for y in bot_y0..bot_y1 {
                for x in x0..x1 {
                    let p = fb[y * ss_width + x];
                    br += p[0] as u32;
                    bg_g += p[1] as u32;
                    bb += p[2] as u32;
                    b_count += 1;
                }
            }

            let bg_color = if t_count > 0 {
                [
                    (tr / t_count) as u8,
                    (tg / t_count) as u8,
                    (tb / t_count) as u8,
                ]
            } else {
                [0, 0, 0]
            };

            let fg_color = if b_count > 0 {
                [
                    (br / b_count) as u8,
                    (bg_g / b_count) as u8,
                    (bb / b_count) as u8,
                ]
            } else {
                [0, 0, 0]
            };

            out[term_row * term_cols + term_col] = (bg_color, fg_color);
        }
    }

    out
}
