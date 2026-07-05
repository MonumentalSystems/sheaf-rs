//! Views: the map between the global grid and per-agent local patches.
//! Ports `sheaf_admm.data.views` (grid-task subset — maze scope).
//!
//! Pinned semantics (PLAN.md §3.5):
//! - centers: first at `patch_size // 2` per axis, step `stride`, row-major
//!   (`[N, 2]` integer (y, x));
//! - maze pre-pad: border of width `patch_size // 2` filled with the WALL
//!   token (= 1) BEFORE extraction; centers shift by the border; then one-hot;
//! - patchify indexes the padded image, so a center is the patch top-left;
//! - edge builder: each undirected edge once, oriented right/down (u before v
//!   in row-major order), 4- or 8-connectivity between centers `stride` apart;
//! - reassembly: overlap-MEAN with `max(counts, 1)` guard;
//! - grid construction is size-generic (rebuilt per batch) so OOD 37x37 /
//!   73x73 mazes work.

use ndarray::{Array2, Array3, Array4, Array5};

/// Maze token ids (data/common.py TOKEN_IDS): 0=empty, 1=wall, 2=start,
/// 3=goal, 5=path. `solved` eval compares only the PATH_TOKEN=5 mask.
pub const TOKEN_EMPTY: i64 = 0;
pub const TOKEN_WALL: i64 = 1;
pub const TOKEN_START: i64 = 2;
pub const TOKEN_GOAL: i64 = 3;
pub const TOKEN_PATH: i64 = 5;

/// Agent centers on a regular grid: `range(patch_size/2, dim, stride)` per
/// axis, row-major -> `[N, 2]` (y, x). Ports `grid_agent_centers`.
pub fn grid_agent_centers(image_hw: (usize, usize), stride: usize, patch_size: usize) -> Array2<i64> {
    todo!()
}

/// 4-/8-conn edges between neighboring centers -> `[E, 2]` (u, v), each
/// undirected edge once, oriented right/down. Ports `build_grid_edge_indices`.
pub fn build_grid_edge_indices(centers: &Array2<i64>, stride: usize, connectivity: usize) -> Vec<[u32; 2]> {
    todo!()
}

/// Wall-token border pre-pad + one-hot + patchify. Ports `prepare_maze_patches`.
/// `tokens: [B, H, W]` int -> patches `[N, B, ps, ps, num_input_classes]` f32.
pub fn prepare_maze_patches(
    tokens: &Array3<i64>,
    centers: &Array2<i64>,
    patch_size: usize,
    num_input_classes: usize,
) -> Array5<f32>
{
    todo!()
}

/// Overlap-mean reassembly of per-agent patch logits into a global logit map.
/// Ports `reassemble_logits(mode="mean")`:
/// `patches [N, B, ps, ps, C]` + `centers [N, 2]` -> `[B, H, W, C]`,
/// accumulating overlapping contributions and dividing by `max(counts, 1)`.
pub fn reassemble_logits(
    patch_logits: &Array5<f32>,
    centers: &Array2<i64>,
    image_hw: (usize, usize),
) -> Array4<f32> {
    todo!()
}
