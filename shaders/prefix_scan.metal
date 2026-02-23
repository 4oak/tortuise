#include <metal_stdlib>
using namespace metal;

// This implementation assumes dispatch with 256 threads per threadgroup.
// On Apple Silicon this corresponds to 8 SIMD groups of width 32.
constant uint kThreadsPerGroup = 256u;
constant uint kSimdWidth = 32u;
constant uint kSimdGroupsPerThreadgroup = kThreadsPerGroup / kSimdWidth;

// Phase 1:
// - Exclusive scan each 256-element block in-place.
// - Emit per-block totals into block_sums[tgid].
kernel void prefix_scan_blocks(
    device uint* data [[buffer(0)]],
    device uint* block_sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint num_blocks = (count + (kThreadsPerGroup - 1u)) / kThreadsPerGroup;
    if (tgid >= num_blocks) {
        return;
    }

    const uint global_index = tgid * kThreadsPerGroup + ltid;
    const uint value = (global_index < count) ? data[global_index] : 0u;

    // 1) Intra-SIMD exclusive scan (32 lanes).
    const uint simd_exclusive = simd_prefix_exclusive_sum(value);

    // Threadgroup scratch for second-level scan across 8 SIMD totals.
    threadgroup uint simd_totals[kSimdGroupsPerThreadgroup];
    threadgroup uint simd_offsets[kSimdGroupsPerThreadgroup];
    threadgroup uint block_total;

    // Last lane in each SIMD writes that SIMD's total sum.
    if (simd_lane == (kSimdWidth - 1u)) {
        simd_totals[simd_group] = simd_exclusive + value;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute full block sum once (sum of 8 SIMD totals).
    if (ltid == 0u) {
        uint total = 0u;
        for (uint i = 0u; i < kSimdGroupsPerThreadgroup; ++i) {
            total += simd_totals[i];
        }
        block_total = total;
    }

    // 2) Scan the 8 SIMD totals to get per-SIMD offsets.
    // Use SIMD 0 for this second-level scan; lanes >= 8 contribute zero.
    if (simd_group == 0u) {
        const uint group_total = (simd_lane < kSimdGroupsPerThreadgroup) ? simd_totals[simd_lane] : 0u;
        const uint group_offset = simd_prefix_exclusive_sum(group_total);

        if (simd_lane < kSimdGroupsPerThreadgroup) {
            simd_offsets[simd_lane] = group_offset;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint scanned_value = simd_offsets[simd_group] + simd_exclusive;

    if (global_index < count) {
        data[global_index] = scanned_value;
    }

    if (ltid == 0u) {
        block_sums[tgid] = block_total;
    }
}

// Phase 2:
// Add scanned block offsets back into each block.
kernel void prefix_scan_add_offsets(
    device uint* data [[buffer(0)]],
    constant uint* block_sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]]
) {
    const uint num_blocks = (count + (kThreadsPerGroup - 1u)) / kThreadsPerGroup;
    if (tgid >= num_blocks) {
        return;
    }

    const uint global_index = tgid * kThreadsPerGroup + ltid;
    if (global_index >= count) {
        return;
    }

    data[global_index] += block_sums[tgid];
}
