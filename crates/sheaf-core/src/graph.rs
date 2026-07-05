//! The agent graph: edges, node positions, and precomputed slot tables.
//!
//! Ports `sheaf_admm.geometry.restriction_maps.compute_direction_index` and the
//! per-edge gathers. The JAX code recomputes direction indices with `jnp.where`
//! ladders on every geometry build; here they are computed **once** at graph
//! construction with plain `if/else` (they were traceable only for jit).

use ndarray::Array2;

/// Endpoint of an edge (which slot of the `[E, 2, ...]` axis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    U = 0,
    V = 1,
}

/// CSR-style node -> incident-edge table, for node-parallel scatter.
///
/// For node `n`, `entries[offsets[n]..offsets[n+1]]` lists `(edge, endpoint)`.
/// The Laplacian contribution sign is `+` at `U` and `-` at `V` (matching the
/// JAX `at[u].add(contrib_u)` / `at[v].add(-contrib_v)` pair).
#[derive(Debug, Clone)]
pub struct NodeIncidence {
    pub offsets: Vec<u32>,          // [N + 1]
    pub entries: Vec<(u32, Endpoint)>, // edge id, endpoint
}

/// The (static, per-task) agent graph.
///
/// Direction slot ordering is EXACTLY the Python `get_direction_names`:
/// 4-way `(N, E, S, W)`; 8-way `(N, NE, E, SE, S, SW, W, NW)` — indices 0..K.
/// The v-endpoint uses the direction of `(-dy, -dx)`: an N-edge uses `R_N` at
/// u and `R_S` at v. Slot tables are pinned against golden dumps.
#[derive(Debug, Clone)]
pub struct AgentGraph {
    /// `[E]` edges as (u, v) node indices. Each undirected edge appears once,
    /// oriented right/down by the grid edge builder.
    pub edges: Vec<[u32; 2]>,
    /// `[N, 2]` (y, x) agent-center positions (grid tasks). `None` for sudoku.
    pub node_positions: Option<Array2<f32>>,
    /// `[E]` direction slot of the u-endpoint (index into the K base maps).
    pub dir_uv: Vec<u8>,
    /// `[E]` direction slot of the v-endpoint (direction of `(-dy, -dx)`).
    pub dir_vu: Vec<u8>,
    /// Node -> incident edges, for node-parallel scatter (B=1 demo mode).
    pub node_edges: NodeIncidence,
    /// Number of agents N.
    pub num_nodes: usize,
}

impl AgentGraph {
    /// Build a grid-task graph from edges + node positions, precomputing the
    /// directional slot tables (`num_directions` in {4, 8}) and incidence CSR.
    pub fn new_grid(
        edges: Vec<[u32; 2]>,
        node_positions: Array2<f32>,
        num_directions: usize,
    ) -> Self {
        todo!("compute dir_uv/dir_vu via compute_direction_index + build incidence CSR")
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Map a position delta `(dy, dx)` to a direction slot index.
///
/// Literal transcription of `compute_direction_index` (restriction_maps.py):
/// - 4-way: 0=N, 1=E, 2=S, 3=W — **vertical takes priority** on diagonals;
/// - 8-way: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW, tested in the exact
///   nested-`where` order of the Python (NE, NW, N, SE, SW, S, E, W).
pub fn compute_direction_index(dy: f32, dx: f32, num_directions: usize) -> u8 {
    todo!("port the jnp.where ladder as if/else; pin against golden slot tables")
}
