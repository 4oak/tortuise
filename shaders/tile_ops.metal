#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16

struct ProjectedSplat {
    float screen_x, screen_y, depth;
    float radius_x, radius_y;
    float cov_a, cov_b, cov_c;
    float opacity;
    uint packed_color;
    uint original_index;
    uint tile_min;  // packed: (tile_min_y << 16) | tile_min_x
    uint tile_max;  // packed: (tile_max_y << 16) | tile_max_x
};

struct TileConfig {
    uint tile_count_x;
    uint tile_count_y;
    uint screen_width;
    uint screen_height;
};

inline uint2 unpack_tile(uint packed) {
    return uint2(packed & 0xFFFFu, packed >> 16);
}

inline uint float_to_sortable_uint(float f) {
    uint v = as_type<uint>(f);
    return (v & 0x80000000u) ? ~v : (v | 0x80000000u);
}

kernel void count_tile_overlaps(
    constant ProjectedSplat* projected [[buffer(0)]],
    device atomic_uint* tile_counts [[buffer(1)]],
    device atomic_uint* total_overlaps [[buffer(2)]],
    constant uint& valid_count [[buffer(3)]],
    constant TileConfig& tile_config [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= valid_count) {
        return;
    }

    const ProjectedSplat splat = projected[index];
    const uint2 tile_min = unpack_tile(splat.tile_min);
    const uint2 tile_max = unpack_tile(splat.tile_max);

    // Guard against unsigned underflow when tile_min > tile_max (splat
    // bounding box collapsed to zero after clamping).
    if (tile_min.x > tile_max.x || tile_min.y > tile_max.y) {
        return;
    }

    // Number of tiles covered by this splat's AABB in tile space.
    const uint overlap_count = (tile_max.x - tile_min.x + 1u) * (tile_max.y - tile_min.y + 1u);
    atomic_fetch_add_explicit(total_overlaps, overlap_count, memory_order_relaxed);

    // Increment per-tile overlap counts for prefix scan / allocation.
    for (uint ty = tile_min.y; ty <= tile_max.y; ++ty) {
        const uint row_offset = ty * tile_config.tile_count_x;
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx) {
            const uint tile_id = row_offset + tx;
            atomic_fetch_add_explicit(&tile_counts[tile_id], 1u, memory_order_relaxed);
        }
    }
}

/// Clamp the total_overlaps counter to sort_capacity so that the radix sort
/// never addresses more elements than were actually emitted into the sort
/// buffers.  Dispatched as a single thread between emit_tile_keys and the
/// radix sort passes.
kernel void clamp_total_overlaps(
    device uint* total_overlaps [[buffer(0)]],
    constant uint& sort_capacity [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) return;
    if (total_overlaps[0] > sort_capacity) {
        total_overlaps[0] = sort_capacity;
    }
}

kernel void emit_tile_keys(
    constant ProjectedSplat* projected [[buffer(0)]],
    constant uint* tile_offsets [[buffer(1)]],
    device atomic_uint* tile_counters [[buffer(2)]],
    device uint* sort_keys [[buffer(3)]],
    device uint* sort_values [[buffer(4)]],
    constant uint& valid_count [[buffer(5)]],
    constant TileConfig& tile_config [[buffer(6)]],
    device atomic_uint* overflow_flag [[buffer(7)]],
    constant uint& sort_capacity [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= valid_count) {
        return;
    }

    const ProjectedSplat splat = projected[index];
    const uint2 tile_min = unpack_tile(splat.tile_min);
    const uint2 tile_max = unpack_tile(splat.tile_max);

    if (tile_min.x > tile_max.x || tile_min.y > tile_max.y) {
        return;
    }

    // Sort key layout: 10-bit tile_id | 18-bit depth | 4-bit tiebreaker
    //
    // 10 bits supports up to 1023 tiles (a 500x160 terminal with 16x16 tiles
    // = ~320 tiles, plenty of headroom).  18-bit depth gives 262144 depth
    // levels (4x more than the original 16-bit depth).
    //
    // The 4-bit tiebreaker from original_index ensures that splats at
    // identical quantized depth produce distinct sort keys 93.75% of the
    // time.  Combined with the deterministic radix sort scatter (which
    // preserves input order for equal keys by ranking threads by ltid rather
    // than atomic_fetch_add), the remaining collisions only swap splats at
    // near-identical depth -- producing imperceptible visual difference.
    //
    // The atomic_fetch_add for slot assignment in emit_tile_keys remains
    // non-deterministic, but this only affects the input order fed to the
    // radix sort.  Since the sort is keyed on (tile, depth, tiebreaker),
    // the final sorted order is determined by the key, not the slot.
    const uint depth_18 = float_to_sortable_uint(splat.depth) >> 14;  // top 18 of 32 bits
    const uint tiebreaker = splat.original_index & 0xFu;              // low 4 bits of stable ID

    // Emit one key/value pair for each tile this splat overlaps.
    for (uint ty = tile_min.y; ty <= tile_max.y; ++ty) {
        const uint row_offset = ty * tile_config.tile_count_x;
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx) {
            const uint tile_id = row_offset + tx;
            const uint local_offset =
                atomic_fetch_add_explicit(&tile_counters[tile_id], 1u, memory_order_relaxed);
            const uint slot = tile_offsets[tile_id] + local_offset;
            if (slot >= sort_capacity) {
                atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
                return;
            }

            sort_keys[slot] = (tile_id << 22) | (depth_18 << 4) | tiebreaker;
            sort_values[slot] = index;
        }
    }
}
