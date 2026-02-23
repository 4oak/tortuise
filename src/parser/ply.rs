use std::fs;
use std::path::Path;

use crate::math::{clamp_u8, quat_normalize, sigmoid, Vec3};
use crate::splat::Splat;

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Copy)]
enum PlyType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl PlyType {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "char" | "int8" => Some(Self::Char),
            "uchar" | "uint8" => Some(Self::UChar),
            "short" | "int16" => Some(Self::Short),
            "ushort" | "uint16" => Some(Self::UShort),
            "int" | "int32" => Some(Self::Int),
            "uint" | "uint32" => Some(Self::UInt),
            "float" | "float32" => Some(Self::Float),
            "double" | "float64" => Some(Self::Double),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::Char | Self::UChar => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double => 8,
        }
    }

    fn read_as_f32(self, bytes: &[u8]) -> f32 {
        match self {
            Self::Char => i8::from_le_bytes([bytes[0]]) as f32,
            Self::UChar => u8::from_le_bytes([bytes[0]]) as f32,
            Self::Short => i16::from_le_bytes([bytes[0], bytes[1]]) as f32,
            Self::UShort => u16::from_le_bytes([bytes[0], bytes[1]]) as f32,
            Self::Int => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32,
            Self::UInt => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32,
            Self::Float => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Self::Double => {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes[0..8]);
                f64::from_le_bytes(arr) as f32
            }
        }
    }
}

#[derive(Debug, Clone)]
struct PlyProperty {
    name: String,
    ty: PlyType,
}

fn find_ply_header_end(data: &[u8]) -> Option<usize> {
    let marker = b"end_header";
    let pos = data.windows(marker.len()).position(|w| w == marker)?;
    let mut end = pos + marker.len();
    while end < data.len() && data[end] != b'\n' {
        end += 1;
    }
    if end < data.len() {
        end += 1;
    }
    Some(end)
}

pub fn load_ply_file(path: &str) -> AppResult<Vec<Splat>> {
    let data = fs::read(path)
        .map_err(|e| format!("failed to read '{}': {}", Path::new(path).display(), e))?;
    let header_end = find_ply_header_end(&data).ok_or("PLY parse error: missing end_header")?;
    let header_text = std::str::from_utf8(&data[..header_end])?;

    let mut is_binary_le = false;
    let mut vertex_count: usize = 0;
    let mut in_vertex_element = false;
    let mut vertex_props: Vec<PlyProperty> = Vec::new();

    for line in header_text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("comment") {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "ply" => {}
            "format" => {
                if parts.len() >= 2 && parts[1] == "binary_little_endian" {
                    is_binary_le = true;
                }
            }
            "element" => {
                if parts.len() >= 3 {
                    in_vertex_element = parts[1] == "vertex";
                    if in_vertex_element {
                        vertex_count = parts[2].parse::<usize>()?;
                    }
                }
            }
            "property" if in_vertex_element => {
                if parts.len() >= 3 && parts[1] == "list" {
                    return Err(
                        "PLY parse error: list properties in vertex element are unsupported".into(),
                    );
                }
                if parts.len() >= 3 {
                    let ty = PlyType::parse(parts[1]).ok_or_else(|| {
                        format!("PLY parse error: unsupported property type '{}'", parts[1])
                    })?;
                    vertex_props.push(PlyProperty {
                        name: parts[2].to_string(),
                        ty,
                    });
                }
            }
            _ => {}
        }
    }

    if !is_binary_le {
        return Err("PLY parse error: only binary_little_endian format is supported".into());
    }
    if vertex_count == 0 || vertex_props.is_empty() {
        return Err("PLY parse error: missing vertex element or properties".into());
    }

    let stride: usize = vertex_props.iter().try_fold(0usize, |acc, prop| {
        acc.checked_add(prop.ty.size())
            .ok_or("PLY parse error: size overflow computing vertex stride")
    })?;
    if stride == 0 {
        return Err("PLY parse error: invalid vertex stride".into());
    }

    let vertex_bytes = vertex_count
        .checked_mul(stride)
        .ok_or("PLY parse error: size overflow computing buffer size")?;
    let needed = header_end
        .checked_add(vertex_bytes)
        .ok_or("PLY parse error: size overflow computing buffer size")?;
    if data.len() < needed {
        return Err(format!(
            "PLY parse error: file truncated (need {needed} bytes, have {})",
            data.len()
        )
        .into());
    }

    let mut splats = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let vertex_offset = i
            .checked_mul(stride)
            .ok_or("PLY parse error: size overflow computing vertex offset")?;
        let base = header_end
            .checked_add(vertex_offset)
            .ok_or("PLY parse error: size overflow computing vertex offset")?;
        let end = base
            .checked_add(stride)
            .ok_or("PLY parse error: size overflow computing vertex offset")?;
        let chunk = data
            .get(base..end)
            .ok_or("PLY parse error: vertex data out of bounds")?;

        let mut p = Vec3::ZERO;
        let mut dc = [0.0_f32; 3];
        let mut rgb = [0.0_f32; 3];
        let mut have_dc = false;
        let mut have_rgb = false;
        let mut opacity_raw = 4.0_f32;
        let mut scale_raw = [-3.0_f32, -3.0_f32, -3.0_f32];
        let mut have_scale = false;
        let mut rotation = [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32];
        let mut have_rotation = false;

        let mut cursor: usize = 0;
        for prop in &vertex_props {
            let sz = prop.ty.size();
            let field_end = cursor
                .checked_add(sz)
                .ok_or("PLY parse error: size overflow computing property offset")?;
            let field = chunk
                .get(cursor..field_end)
                .ok_or("PLY parse error: property data out of bounds")?;
            let value = prop.ty.read_as_f32(field);
            cursor = field_end;

            match prop.name.as_str() {
                "x" => p.x = value,
                "y" => p.y = value,
                "z" => p.z = value,
                "f_dc_0" => {
                    dc[0] = value;
                    have_dc = true;
                }
                "f_dc_1" => {
                    dc[1] = value;
                    have_dc = true;
                }
                "f_dc_2" => {
                    dc[2] = value;
                    have_dc = true;
                }
                "red" | "r" => {
                    rgb[0] = value;
                    have_rgb = true;
                }
                "green" | "g" => {
                    rgb[1] = value;
                    have_rgb = true;
                }
                "blue" | "b" => {
                    rgb[2] = value;
                    have_rgb = true;
                }
                "opacity" => opacity_raw = value,
                "scale_0" => {
                    scale_raw[0] = value;
                    have_scale = true;
                }
                "scale_1" => {
                    scale_raw[1] = value;
                    have_scale = true;
                }
                "scale_2" => {
                    scale_raw[2] = value;
                    have_scale = true;
                }
                "rot_0" => {
                    rotation[0] = value;
                    have_rotation = true;
                }
                "rot_1" => {
                    rotation[1] = value;
                    have_rotation = true;
                }
                "rot_2" => {
                    rotation[2] = value;
                    have_rotation = true;
                }
                "rot_3" => {
                    rotation[3] = value;
                    have_rotation = true;
                }
                _ => {}
            }
        }

        let color = if have_dc {
            [
                clamp_u8(sigmoid(dc[0]) * 255.0),
                clamp_u8(sigmoid(dc[1]) * 255.0),
                clamp_u8(sigmoid(dc[2]) * 255.0),
            ]
        } else if have_rgb {
            [clamp_u8(rgb[0]), clamp_u8(rgb[1]), clamp_u8(rgb[2])]
        } else {
            [220, 220, 220]
        };

        let opacity = sigmoid(opacity_raw).clamp(0.0, 1.0);
        let scale = if have_scale {
            Vec3::new(
                scale_raw[0].exp().max(1e-4),
                scale_raw[1].exp().max(1e-4),
                scale_raw[2].exp().max(1e-4),
            )
        } else {
            Vec3::new(0.05, 0.05, 0.05)
        };

        let rotation = if have_rotation {
            quat_normalize(rotation)
        } else {
            [1.0, 0.0, 0.0, 0.0]
        };

        splats.push(Splat {
            position: p,
            color,
            opacity,
            scale,
            rotation,
        });
    }

    Ok(splats)
}
