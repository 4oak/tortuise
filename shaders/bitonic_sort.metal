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

kernel void bitonic_sort_step(
    device ProjectedSplat* data [[buffer(0)]],
    constant uint& stage [[buffer(1)]],
    constant uint& step [[buffer(2)]],
    constant uint& total_count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= total_count) {
        return;
    }

    uint j = 1u << step;
    uint k = 2u << stage;
    uint ixj = index ^ j;

    // Each pair is handled once by the lower index.
    if (ixj <= index || ixj >= total_count) {
        return;
    }

    bool ascending = ((index & k) == 0);

    ProjectedSplat left = data[index];
    ProjectedSplat right = data[ixj];

    bool should_swap = ascending ? (left.depth > right.depth) : (left.depth < right.depth);

    if (should_swap) {
        data[index] = right;
        data[ixj] = left;
    }
}
