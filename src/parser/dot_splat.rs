use std::fs;

use crate::math::{quat_normalize, Vec3};
use crate::splat::Splat;

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

fn read_vec3_f32(bytes: &[u8]) -> Vec3 {
    let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let z = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    Vec3::new(x, y, z)
}

fn decode_scale_value(v: f32) -> f32 {
    if v > 0.0 {
        v
    } else {
        v.exp().max(1e-4)
    }
}

pub fn load_splat_file(path: &str) -> AppResult<Vec<Splat>> {
    let data = fs::read(path)?;
    if data.len() < 32 {
        return Err("SPLAT parse error: file too small".into());
    }

    let mut splats = Vec::with_capacity(data.len() / 32);
    for chunk in data.chunks_exact(32) {
        let position = read_vec3_f32(&chunk[0..12]);
        let scale_raw = read_vec3_f32(&chunk[12..24]);
        let color = [chunk[24], chunk[25], chunk[26]];
        let opacity = (chunk[27] as f32 / 255.0).clamp(0.0, 1.0);

        let rotation = quat_normalize([
            chunk[28] as f32 / 127.5 - 1.0,
            chunk[29] as f32 / 127.5 - 1.0,
            chunk[30] as f32 / 127.5 - 1.0,
            chunk[31] as f32 / 127.5 - 1.0,
        ]);

        let scale = Vec3::new(
            decode_scale_value(scale_raw.x),
            decode_scale_value(scale_raw.y),
            decode_scale_value(scale_raw.z),
        );

        splats.push(Splat {
            position,
            color,
            opacity,
            scale,
            rotation,
        });
    }

    Ok(splats)
}
