//! The agent graph: edges, node positions, and precomputed slot tables.
//!
//! Ports `sheaf_admm.geometry.restriction_maps.compute_direction_index` and the
//! per-edge gathers. The JAX code recomputes direction indices with `jnp.where`
//! ladders on every geometry build; here they are computed **once** at graph
//! construction with plain `if/else` (they were traceable only for jit).

use ndarray::Array2;

use crate::Scalar;

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
    pub offsets: Vec<u32>,             // [N + 1]
    pub entries: Vec<(u32, Endpoint)>, // edge id, endpoint
}

impl NodeIncidence {
    /// Build the node -> incident-edge CSR by counting sort over edge
    /// endpoints. Per node, entries appear in edge order (u before v when a
    /// self-loop touches the node twice).
    pub fn build(edges: &[[u32; 2]], num_nodes: usize) -> Self {
        let mut offsets = vec![0u32; num_nodes + 1];
        for &[u, v] in edges {
            offsets[u as usize + 1] += 1;
            offsets[v as usize + 1] += 1;
        }
        for n in 0..num_nodes {
            offsets[n + 1] += offsets[n];
        }
        let mut cursor: Vec<u32> = offsets[..num_nodes].to_vec();
        let mut entries = vec![(0u32, Endpoint::U); 2 * edges.len()];
        for (ei, &[u, v]) in edges.iter().enumerate() {
            entries[cursor[u as usize] as usize] = (ei as u32, Endpoint::U);
            cursor[u as usize] += 1;
            entries[cursor[v as usize] as usize] = (ei as u32, Endpoint::V);
            cursor[v as usize] += 1;
        }
        Self { offsets, entries }
    }

    /// The `(edge, endpoint)` pairs incident to node `n`.
    pub fn incident(&self, n: usize) -> &[(u32, Endpoint)] {
        &self.entries[self.offsets[n] as usize..self.offsets[n + 1] as usize]
    }
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
    pub node_positions: Option<Array2<Scalar>>,
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
        node_positions: Array2<Scalar>,
        num_directions: usize,
    ) -> Self {
        assert_eq!(node_positions.ncols(), 2, "node_positions must be [N, 2] (y, x)");
        let num_nodes = node_positions.nrows();
        let mut dir_uv = Vec::with_capacity(edges.len());
        let mut dir_vu = Vec::with_capacity(edges.len());
        for &[u, v] in &edges {
            let (u, v) = (u as usize, v as usize);
            // dy/dx from u to v; positions are (y, x) — matches
            // build_directional_restriction_maps / create_lora_geometry.
            let dy = node_positions[[v, 0]] - node_positions[[u, 0]];
            let dx = node_positions[[v, 1]] - node_positions[[u, 1]];
            dir_uv.push(compute_direction_index(dy, dx, num_directions));
            dir_vu.push(compute_direction_index(-dy, -dx, num_directions));
        }
        let node_edges = NodeIncidence::build(&edges, num_nodes);
        Self {
            edges,
            node_positions: Some(node_positions),
            dir_uv,
            dir_vu,
            node_edges,
            num_nodes,
        }
    }

    /// Build a graph with no positions / directional slot tables (sudoku-style
    /// slot tables live elsewhere; also handy for geometry tests that only
    /// need edges + N).
    pub fn from_edges(edges: Vec<[u32; 2]>, num_nodes: usize) -> Self {
        let node_edges = NodeIncidence::build(&edges, num_nodes);
        Self {
            edges,
            node_positions: None,
            dir_uv: Vec::new(),
            dir_vu: Vec::new(),
            node_edges,
            num_nodes,
        }
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
///
/// North is "up" = smaller y, so `is_north = dy < 0`.
pub fn compute_direction_index(dy: Scalar, dx: Scalar, num_directions: usize) -> u8 {
    let is_north = dy < 0.0;
    let is_south = dy > 0.0;
    let is_east = dx > 0.0;
    let is_west = dx < 0.0;
    match num_directions {
        4 => {
            // jnp.where(is_north, 0, where(is_south, 2, where(is_east, 1, 3)))
            if is_north {
                0
            } else if is_south {
                2
            } else if is_east {
                1
            } else {
                3
            }
        }
        8 => {
            // Exact nested-where order: NE, NW, N, SE, SW, S, E, else W.
            if is_north && is_east {
                1
            } else if is_north && is_west {
                7
            } else if is_north {
                0
            } else if is_south && is_east {
                3
            } else if is_south && is_west {
                5
            } else if is_south {
                4
            } else if is_east {
                2
            } else {
                6
            }
        }
        _ => panic!("num_directions must be 4 or 8, got {num_directions}"),
    }
}
