use rayon::prelude::*;

use crate::splat::ProjectedSplat;

pub fn sort_by_depth(projected_splats: &mut [ProjectedSplat]) {
    projected_splats.par_sort_unstable_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}
