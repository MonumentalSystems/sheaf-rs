//! Maze generator. Faithful port of `sheaf_admm/data/build_maze.py`:
//! DFS carve on the **odd-index lattice** (cells at odd coordinates, walls
//! knocked out between visited neighbors; strictly-interior checks), start/goal
//! painting, and **BFS minimum-path-length acceptance** filtering.
//!
//! Valid sizes are odd (19x19 in-distribution; OOD suite 37x37 / 73x73). Even
//! sizes are structurally impossible in this distribution.
//!
//! Determinism: seeded with a small SplitMix64 PRNG. The RNG **stream differs
//! from numpy's** `np.random.default_rng` — parity is property-level (odd
//! lattice, perfect-maze tree, BFS acceptance), never bit-level (PLAN.md §5.1).

use ndarray::Array2;

use crate::views::{TOKEN_EMPTY, TOKEN_GOAL, TOKEN_PATH, TOKEN_START, TOKEN_WALL};

/// Mirror of Python's `MazeConfig.max_retries` default; the acceptance loop
/// tries `max_retries * 4` carves before giving up (matching `_build_split`).
const MAX_RETRIES: usize = 200;

/// A generated maze: token grids (`TOKEN_*` ids from `views`) with painted
/// start/goal, plus the BFS ground-truth shortest path.
#[derive(Debug, Clone)]
pub struct GeneratedMaze {
    /// Input grid `[H, W]`: walls/empties with start/goal painted (no path).
    pub tokens: Array2<i64>,
    /// Label grid `[H, W]`: as `tokens` but with the shortest path painted
    /// (`TOKEN_PATH`) before start/goal overwrite the path endpoints.
    pub labels: Array2<i64>,
    /// BFS shortest start->goal path length in steps (`len(path) - 1`).
    pub path_len: usize,
}

/// SplitMix64 — tiny, deterministic, good-enough PRNG for maze carving.
/// NOT numpy-compatible (documented above; property parity only).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` via rejection sampling (unbiased).
    fn below(&mut self, n: usize) -> usize {
        assert!(n > 0);
        let n = n as u64;
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < zone {
                return (v % n) as usize;
            }
        }
    }
}

/// Carve a perfect maze by randomized DFS on the odd-index lattice.
/// Ports `_backtrack_maze`: grid starts all-wall; cells live at odd (r, c)
/// with `0 < r < height`, `0 < c < width`; carving knocks out the wall midway
/// between two lattice cells.
fn backtrack_maze(height: usize, width: usize, rng: &mut SplitMix64) -> Array2<i64> {
    let mut grid = Array2::<i64>::from_elem((height, width), TOKEN_WALL);
    let odd_rows: Vec<usize> = (1..height).step_by(2).collect();
    let odd_cols: Vec<usize> = (1..width).step_by(2).collect();
    assert!(
        !odd_rows.is_empty() && !odd_cols.is_empty(),
        "maze must be at least 3x3"
    );
    let start_r = odd_rows[rng.below(odd_rows.len())];
    let start_c = odd_cols[rng.below(odd_cols.len())];
    grid[[start_r, start_c]] = TOKEN_EMPTY;
    let mut stack: Vec<(usize, usize)> = vec![(start_r, start_c)];

    while let Some(&(r, c)) = stack.last() {
        // Unvisited odd-lattice neighbors two cells away, strictly interior
        // (0 < nr < height, 0 < nc < width) and still wall — same predicate,
        // same neighbor order (+2,0), (-2,0), (0,+2), (0,-2) as Python.
        let mut unvisited: Vec<(usize, usize)> = Vec::with_capacity(4);
        for (dr, dc) in [(2i64, 0i64), (-2, 0), (0, 2), (0, -2)] {
            let (nr, nc) = (r as i64 + dr, c as i64 + dc);
            if nr > 0
                && (nr as usize) < height
                && nc > 0
                && (nc as usize) < width
                && nr % 2 == 1
                && nc % 2 == 1
                && grid[[nr as usize, nc as usize]] == TOKEN_WALL
            {
                unvisited.push((nr as usize, nc as usize));
            }
        }
        if unvisited.is_empty() {
            stack.pop();
            continue;
        }
        let (nr, nc) = unvisited[rng.below(unvisited.len())];
        // Knock out the wall midway, then open the new cell.
        grid[[(r + nr) / 2, (c + nc) / 2]] = TOKEN_EMPTY;
        grid[[nr, nc]] = TOKEN_EMPTY;
        stack.push((nr, nc));
    }
    grid
}

/// BFS shortest path over non-wall cells (4-neighborhood). Ports
/// `_shortest_path`; returns the start->goal path (inclusive), empty if
/// unreachable.
fn shortest_path(
    grid: &Array2<i64>,
    start: (usize, usize),
    goal: (usize, usize),
) -> Vec<(usize, usize)> {
    let (h, w) = grid.dim();
    let mut queue: Vec<(usize, usize)> = vec![start];
    let mut parent: Array2<i32> = Array2::from_elem((h, w), -1); // flat parent index
    let mut visited = Array2::<bool>::from_elem((h, w), false);
    visited[start] = true;

    let mut head = 0;
    while head < queue.len() {
        let (r, c) = queue[head];
        if (r, c) == goal {
            break;
        }
        head += 1;
        for (dr, dc) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
            let (nr, nc) = (r as i64 + dr, c as i64 + dc);
            if nr >= 0
                && (nr as usize) < h
                && nc >= 0
                && (nc as usize) < w
                && !visited[[nr as usize, nc as usize]]
                && grid[[nr as usize, nc as usize]] != TOKEN_WALL
            {
                visited[[nr as usize, nc as usize]] = true;
                parent[[nr as usize, nc as usize]] = (r * w + c) as i32;
                queue.push((nr as usize, nc as usize));
            }
        }
    }

    if parent[goal] < 0 && start != goal {
        return Vec::new();
    }
    let mut path = vec![goal];
    let mut cur = goal;
    while cur != start {
        let p = parent[cur] as usize;
        cur = (p / w, p % w);
        path.push(cur);
    }
    path.reverse();
    path
}

/// Sample two distinct empty cells (row-major order over `argwhere(empty)`,
/// like Python's `_pick_positions`, but with our PRNG).
fn pick_positions(grid: &Array2<i64>, rng: &mut SplitMix64) -> ((usize, usize), (usize, usize)) {
    let empties: Vec<(usize, usize)> = grid
        .indexed_iter()
        .filter(|(_, &v)| v == TOKEN_EMPTY)
        .map(|((r, c), _)| (r, c))
        .collect();
    assert!(empties.len() >= 2, "Maze too small for start/goal placement.");
    let start_idx = rng.below(empties.len());
    let mut goal_idx = rng.below(empties.len() - 1);
    if goal_idx >= start_idx {
        goal_idx += 1; // distinct, uniform over the remaining cells
    }
    (empties[start_idx], empties[goal_idx])
}

/// Paint start/goal (and optionally the path) onto a copy of the maze.
/// Ports `_paint_grid` (path first, then start/goal overwrite endpoints).
fn paint_grid(
    base: &Array2<i64>,
    start: (usize, usize),
    goal: (usize, usize),
    path: &[(usize, usize)],
    with_path: bool,
) -> Array2<i64> {
    let mut painted = base.clone();
    if with_path {
        for &(r, c) in path {
            painted[[r, c]] = TOKEN_PATH;
        }
    }
    painted[start] = TOKEN_START;
    painted[goal] = TOKEN_GOAL;
    painted
}

/// Generate one accepted maze of odd `size` (H = W), rejecting until the BFS
/// start->goal shortest path length (in steps) >= `min_path_len`.
/// Deterministic in `seed` (Rust-side RNG; property-tested, not bit-pinned).
///
/// Panics (like Python's `RuntimeError`) if `max_retries * 4` carves fail —
/// only possible with an over-demanding `min_path_len` for the size.
pub fn generate_maze(size: usize, seed: u64, min_path_len: usize) -> GeneratedMaze {
    generate_maze_hw(size, size, seed, min_path_len)
}

/// Rectangular variant (the Python builder supports `19x37` OOD-wide splits).
pub fn generate_maze_hw(
    height: usize,
    width: usize,
    seed: u64,
    min_path_len: usize,
) -> GeneratedMaze {
    assert!(
        height % 2 == 1 && width % 2 == 1,
        "maze sizes must be odd (odd-index lattice); got {height}x{width}"
    );
    let mut rng = SplitMix64::new(seed);
    for _ in 0..MAX_RETRIES * 4 {
        let grid = backtrack_maze(height, width, &mut rng);
        let (start, goal) = pick_positions(&grid, &mut rng);
        let path = shortest_path(&grid, start, goal);
        if path.len() > min_path_len {
            return GeneratedMaze {
                tokens: paint_grid(&grid, start, goal, &path, false),
                labels: paint_grid(&grid, start, goal, &path, true),
                path_len: path.len() - 1,
            };
        }
    }
    panic!("Failed to generate maze with min_path_length={min_path_len}");
}

/// BFS shortest start->goal path length in steps over a painted token grid;
/// `None` if start/goal are missing or disconnected. Used by the demo's
/// "no path exists" check on hand-edited mazes.
pub fn bfs_path_len(tokens: &Array2<i64>) -> Option<usize> {
    let mut start = None;
    let mut goal = None;
    for ((r, c), &v) in tokens.indexed_iter() {
        if v == TOKEN_START {
            start = Some((r, c));
        } else if v == TOKEN_GOAL {
            goal = Some((r, c));
        }
    }
    let (start, goal) = (start?, goal?);
    let path = shortest_path(tokens, start, goal);
    if path.is_empty() {
        None
    } else {
        Some(path.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walls are allowed only where the odd-lattice carve allows: every cell
    /// with both coordinates odd (strictly interior) must be open, and every
    /// cell with both coordinates even must be wall (only odd-odd cells and
    /// the odd-even/even-odd wall slots between them are ever carved).
    fn assert_odd_lattice_structure(grid: &Array2<i64>) {
        let (h, w) = grid.dim();
        // Border is never carved.
        for r in 0..h {
            assert_eq!(grid[[r, 0]], TOKEN_WALL);
            assert_eq!(grid[[r, w - 1]], TOKEN_WALL);
        }
        for c in 0..w {
            assert_eq!(grid[[0, c]], TOKEN_WALL);
            assert_eq!(grid[[h - 1, c]], TOKEN_WALL);
        }
        for r in 0..h {
            for c in 0..w {
                let v = grid[[r, c]];
                let open = v != TOKEN_WALL;
                if r % 2 == 0 && c % 2 == 0 {
                    assert!(!open, "even-even cell ({r},{c}) carved");
                }
                if r % 2 == 1 && c % 2 == 1 && r < h - 1 && c < w - 1 {
                    assert!(open, "odd-odd lattice cell ({r},{c}) not carved");
                }
            }
        }
    }

    /// A perfect maze on the odd lattice: #open cells = lattice cells +
    /// (lattice cells - 1) knocked-out walls (spanning tree).
    fn assert_perfect_maze(grid: &Array2<i64>) {
        let (h, w) = grid.dim();
        let lattice = ((h - 1) / 2) * ((w - 1) / 2);
        let open = grid.iter().filter(|&&v| v != TOKEN_WALL).count();
        assert_eq!(open, 2 * lattice - 1, "open-cell count is not a spanning tree");
    }

    #[test]
    fn mazegen_structural_invariants_all_sizes() {
        // PLAN.md §5.1: every generated maze (19/37/73) is odd-sized, walls
        // only where the carve allows, start/goal BFS-connected with path
        // length >= the acceptance threshold. min_path_len per _OOD_SUITE.
        for &(size, mpl, seed) in &[(19usize, 18usize, 0u64), (37, 36, 1), (73, 72, 2)] {
            let maze = generate_maze(size, seed, mpl);
            assert_eq!(maze.tokens.dim(), (size, size));
            assert!(maze.path_len >= mpl, "path_len {} < {}", maze.path_len, mpl);

            // Exactly one start and one goal painted over empties.
            let starts = maze.tokens.iter().filter(|&&v| v == TOKEN_START).count();
            let goals = maze.tokens.iter().filter(|&&v| v == TOKEN_GOAL).count();
            assert_eq!((starts, goals), (1, 1));
            // Inputs never carry the path token.
            assert!(maze.tokens.iter().all(|&v| v != TOKEN_PATH));

            // Structural invariants hold on the unpainted structure (rebuild
            // it by mapping start/goal/path back to empty).
            let structure = maze.tokens.mapv(|v| if v == TOKEN_WALL { TOKEN_WALL } else { TOKEN_EMPTY });
            assert_odd_lattice_structure(&structure);
            assert_perfect_maze(&structure);

            // BFS re-check on the painted grid matches the recorded length.
            assert_eq!(bfs_path_len(&maze.tokens), Some(maze.path_len));

            // Labels: path painted, endpoints overwritten by start/goal, and
            // path cell count = path_len - 1 (endpoints excluded).
            let path_cells = maze.labels.iter().filter(|&&v| v == TOKEN_PATH).count();
            assert_eq!(path_cells, maze.path_len - 1);
            // Labels agree with tokens everywhere except on the path.
            for ((r, c), &v) in maze.tokens.indexed_iter() {
                let lv = maze.labels[[r, c]];
                assert!(lv == v || (lv == TOKEN_PATH && v == TOKEN_EMPTY));
            }
        }
    }

    #[test]
    fn deterministic_in_seed() {
        let a = generate_maze(19, 42, 18);
        let b = generate_maze(19, 42, 18);
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.labels, b.labels);
        let c = generate_maze(19, 43, 18);
        assert!(a.tokens != c.tokens, "different seeds should differ (w.h.p.)");
    }

    #[test]
    fn bfs_path_len_detects_disconnection() {
        // start | wall | goal in a 3x5: no path.
        let mut grid = Array2::<i64>::from_elem((3, 5), TOKEN_WALL);
        grid[[1, 1]] = TOKEN_START;
        grid[[1, 3]] = TOKEN_GOAL;
        assert_eq!(bfs_path_len(&grid), None);
        grid[[1, 2]] = TOKEN_EMPTY;
        assert_eq!(bfs_path_len(&grid), Some(2));
    }

    #[test]
    #[should_panic(expected = "must be odd")]
    fn even_sizes_are_rejected() {
        let _ = generate_maze(38, 0, 18);
    }
}
