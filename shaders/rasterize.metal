#include <metal_stdlib>
using namespace metal;

struct ProjectedSplat {
    float screen_x, screen_y, depth;
    float radius_x, radius_y;
    float cov_a, cov_b, cov_c;
    float opacity;
    uint packed_color;
    uint original_index;
};

struct TileData {
    uint width, height;
    float saturation_epsilon;
    float min_gaussian_contrib;
};

float3 invert_2x2_covariance(float3 cov) {
    float det = cov.x * cov.z - cov.y * cov.y;
    if (abs(det) < 1e-8) {
        return float3(0.0);
    }
    float inv_det = 1.0 / det;
    return float3(cov.z * inv_det, -cov.y * inv_det, cov.x * inv_det);
}

float evaluate_2d_gaussian(float dx, float dy, float3 inv_cov) {
    float q = dx * dx * inv_cov.x + 2.0 * dx * dy * inv_cov.y + dy * dy * inv_cov.z;
    if (q > 32.0) { // 4-sigma cutoff
        return 0.0;
    }
    return exp(-0.5 * q);
}

uint8_t blend_component(uint8_t existing, uint8_t new_val, float weight) {
    return uint8_t(clamp(float(existing) + float(new_val) * weight, 0.0f, 255.0f));
}

kernel void rasterize_splats(
    constant ProjectedSplat* splats [[buffer(0)]],
    device uint* framebuffer [[buffer(1)]], // RGBA packed as uint32
    device float* alpha_buffer [[buffer(2)]],
    constant TileData& tile [[buffer(3)]],
    constant uint& splat_count [[buffer(4)]],
    uint2 pixel_pos [[thread_position_in_grid]]
) {
    if (pixel_pos.x >= tile.width || pixel_pos.y >= tile.height) {
        return;
    }

    uint pixel_idx = pixel_pos.y * tile.width + pixel_pos.x;
    float pixel_center_x = pixel_pos.x + 0.5;
    float pixel_center_y = pixel_pos.y + 0.5;

    float existing_alpha = alpha_buffer[pixel_idx];
    if (existing_alpha >= tile.saturation_epsilon) {
        return;
    }

    uint existing_color = framebuffer[pixel_idx];
    uint8_t pixel_r = uint8_t(existing_color & 0xFF);
    uint8_t pixel_g = uint8_t((existing_color >> 8) & 0xFF);
    uint8_t pixel_b = uint8_t((existing_color >> 16) & 0xFF);

    for (uint i = 0; i < splat_count; i++) {
        ProjectedSplat splat = splats[i];

        // Bounding box check
        float dx = pixel_center_x - splat.screen_x;
        float dy = pixel_center_y - splat.screen_y;

        if (abs(dx) > splat.radius_x || abs(dy) > splat.radius_y) {
            continue;
        }

        // Invert covariance
        float3 inv_cov = invert_2x2_covariance(float3(splat.cov_a, splat.cov_b, splat.cov_c));
        if (length(inv_cov) == 0.0) {
            continue;
        }

        // Evaluate Gaussian
        float gaussian = evaluate_2d_gaussian(dx, dy, inv_cov);
        if (gaussian < tile.min_gaussian_contrib) {
            continue;
        }

        float alpha = splat.opacity * gaussian;
        if (alpha <= 0.0) {
            continue;
        }

        float weight = alpha * (1.0 - existing_alpha);
        if (weight < 1e-4) {
            continue;
        }

        // Unpack splat color
        uint splat_color = splat.packed_color;
        uint8_t splat_r = uint8_t(splat_color & 0xFF);
        uint8_t splat_g = uint8_t((splat_color >> 8) & 0xFF);
        uint8_t splat_b = uint8_t((splat_color >> 16) & 0xFF);

        // Match CPU rasterization semantics: additive blend in u8 space.
        pixel_r = blend_component(pixel_r, splat_r, weight);
        pixel_g = blend_component(pixel_g, splat_g, weight);
        pixel_b = blend_component(pixel_b, splat_b, weight);
        existing_alpha += weight;
        existing_alpha = min(existing_alpha, 1.0);

        if (existing_alpha >= tile.saturation_epsilon) {
            break;
        }
    }

    // Pack and write result
    uint packed_result =
        (uint(pixel_r) |
        (uint(pixel_g) << 8) |
        (uint(pixel_b) << 16) |
        (uint(clamp(existing_alpha * 255.0, 0.0, 255.0)) << 24));

    framebuffer[pixel_idx] = packed_result;
    alpha_buffer[pixel_idx] = existing_alpha;
}
