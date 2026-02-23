use rand::Rng;
use std::f32::consts::TAU;
use crate::math::{clamp_u8, hsv_to_rgb, Vec3};
use crate::splat::Splat;

// --- Demo splat generators ---

fn random_sphere_point(rng: &mut impl Rng) -> Vec3 {
    let z = rng.random_range(-1.0_f32..1.0_f32);
    let theta = rng.random_range(0.0_f32..TAU);
    let r = (1.0 - z * z).sqrt();
    Vec3::new(r * theta.cos(), z, r * theta.sin())
}

fn generate_torus_knot_splats(count: usize) -> Vec<Splat> {
    let mut rng = rand::rng();
    let mut splats = Vec::with_capacity(count);

    let p = 2.0;
    let q = 3.0;
    // Smaller radii so the scene fits within the FOV from z=5.
    let major = 1.4;
    let minor = 0.38;

    for i in 0..count {
        let t = i as f32 / count.max(1) as f32 * TAU * 2.0;

        // Lay the torus knot in the XZ plane (Y is up) so that when the
        // camera looks along -Z the full loop structure is visible.
        let base = Vec3::new(
            (major + minor * (q * t).cos()) * (p * t).cos(),
            minor * (q * t).sin(),
            (major + minor * (q * t).cos()) * (p * t).sin(),
        );

        let jitter = Vec3::new(
            rng.random_range(-0.04_f32..0.04_f32),
            rng.random_range(-0.04_f32..0.04_f32),
            rng.random_range(-0.04_f32..0.04_f32),
        );

        let hue = ((q * t).sin() * 0.5 + 0.5) * 360.0;
        let color = hsv_to_rgb(hue, 0.80, 0.95);

        // Smaller splats for denser coverage -- reduce scale slightly
        let scale = rng.random_range(0.018_f32..0.042_f32);
        splats.push(Splat {
            position: base + jitter,
            color,
            opacity: rng.random_range(0.68_f32..0.95_f32),
            scale: Vec3::new(scale, scale * rng.random_range(0.9..1.2), scale),
            rotation: [1.0, 0.0, 0.0, 0.0],
        });
    }

    splats
}

fn generate_sphere_cluster_splats(count: usize) -> Vec<Splat> {
    let mut rng = rand::rng();
    let mut splats = Vec::with_capacity(count);

    // Keep clusters close enough to be visible from the default camera (z=5,
    // 60deg FOV).  At depth ~4 the half-width is tan(30deg)*4 ~ 2.3 units.
    let centers = [
        Vec3::new(1.8, 0.3, 0.4),
        Vec3::new(-1.6, -0.2, 0.8),
        Vec3::new(0.3, 1.2, -1.6),
        Vec3::new(-0.5, -1.0, -1.4),
    ];

    let palette = [
        [255, 120, 80],
        [100, 210, 255],
        [160, 255, 130],
        [255, 220, 90],
    ];

    for i in 0..count {
        let cluster = i % centers.len();
        let center = centers[cluster];
        let base_color = palette[cluster];

        let dir = random_sphere_point(&mut rng);
        let radius = rng.random::<f32>().cbrt() * rng.random_range(0.5_f32..1.4_f32);

        let position = center
            + dir * radius
            + Vec3::new(
                rng.random_range(-0.03_f32..0.03_f32),
                rng.random_range(-0.03_f32..0.03_f32),
                rng.random_range(-0.03_f32..0.03_f32),
            );

        let color = [
            clamp_u8(base_color[0] as f32 + rng.random_range(-25.0_f32..25.0_f32)),
            clamp_u8(base_color[1] as f32 + rng.random_range(-25.0_f32..25.0_f32)),
            clamp_u8(base_color[2] as f32 + rng.random_range(-25.0_f32..25.0_f32)),
        ];

        // Smaller splat scale to match denser coverage
        let scale = rng.random_range(0.02_f32..0.06_f32);
        splats.push(Splat {
            position,
            color,
            opacity: rng.random_range(0.60_f32..0.95_f32),
            scale: Vec3::new(scale, scale * rng.random_range(0.8..1.3), scale),
            rotation: [1.0, 0.0, 0.0, 0.0],
        });
    }

    splats
}

pub fn generate_demo_splats() -> Vec<Splat> {
    // 30K torus knot + 15K sphere clusters = 45K total
    let mut splats = generate_torus_knot_splats(30_000);
    splats.extend(generate_sphere_cluster_splats(15_000));
    splats
}
