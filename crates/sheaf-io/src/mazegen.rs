//! Maze generator. Faithful port of `sheaf_admm/data/build_maze.py`:
//! DFS carve on the **odd-index lattice** (cells at odd coordinates, walls
//! knocked out between visited neighbors; strictly-interior checks), start/goal
//! painting, and **BFS minimum-path-length acceptance** filtering.
//!
//! Valid sizes are odd (19x19 in-distribution; OOD suite 37x37 / 73x73). Even
//! sizes are structurally impossible in this distribution. RNG streams differ
//! from numpy — parity tests pin structural invariants, not bits (PLAN.md §5.1).

use ndarray::Array2;

/// A generated maze: token grid (`TOKEN_*` ids from `views`) with painted
/// start/goal, plus the BFS ground-truth shortest path length.
#[derive(Debug, Clone)]
pub struct GeneratedMaze {
    pub tokens: Array2<i64>, // [H, W]
    pub path_len: usize,
}

/// Generate one accepted maze of odd size `size` (H = W), rejecting until the
/// BFS start->goal shortest path length >= `min_path_len`.
/// Deterministic in `seed` (Rust-side RNG; property-tested, not bit-pinned).
pub fn generate_maze(size: usize, seed: u64, min_path_len: usize) -> GeneratedMaze {
    todo!("odd-lattice DFS carve + start/goal painting + BFS acceptance loop")
}

/// BFS shortest path length between the start and goal tokens, `None` if
/// disconnected. Also used by the demo's "no path exists" check.
pub fn bfs_path_len(tokens: &Array2<i64>) -> Option<usize> {
    todo!()
}
