//! Base restriction-map assembly. Ports `geometry/restriction_maps.py`
//! (directional path — maze scope; 8-way shipped, 4-way kept for parity).
//!
//! The learned parameters are K = `num_directions` base maps `R_name`
//! `[d_e, d_v]`, stacked in the exact `get_direction_names` order:
//! 4-way `(N, E, S, W)`; 8-way `(N, NE, E, SE, S, SW, W, NW)`. Assembly
//! gathers them into the per-edge `[E, 2, d_e, d_v]` layout using the
//! direction slot tables (slot 0 = u-endpoint, 1 = v-endpoint; the
//! **v-endpoint uses the direction of `(-dy, -dx)`** — an N-edge uses `R_N`
//! at u and `R_S` at v).

use ndarray::{s, Array2, Array3, Axis};

use sheaf_core::graph::AgentGraph;
use sheaf_core::tensor::RestrictionMaps;

/// Ordered direction names (must match Python `get_direction_names` — these
/// are also the safetensors key suffixes `rm/R_<name>`).
pub fn direction_names(num_directions: usize) -> &'static [&'static str] {
    match num_directions {
        4 => &["N", "E", "S", "W"],
        8 => &["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        _ => panic!("num_directions must be 4 or 8"),
    }
}

/// Map a position delta `(dy, dx)` to a direction slot index.
///
/// Literal transcription of `compute_direction_index` (restriction_maps.py),
/// with the `jnp.where` ladder as if/else in the exact same test order:
/// - 4-way: 0=N, 1=E, 2=S, 3=W — **vertical takes priority** on diagonals;
/// - 8-way: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW, tested in the
///   nested-`where` order NE, NW, N, SE, SW, S, E, W.
pub fn compute_direction_index(dy: f32, dx: f32, num_directions: usize) -> u8 {
    let is_north = dy < 0.0;
    let is_south = dy > 0.0;
    let is_east = dx > 0.0;
    let is_west = dx < 0.0;
    match num_directions {
        4 => {
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
            if is_north && is_east {
                1 // NE
            } else if is_north && is_west {
                7 // NW
            } else if is_north {
                0 // N
            } else if is_south && is_east {
                3 // SE
            } else if is_south && is_west {
                5 // SW
            } else if is_south {
                4 // S
            } else if is_east {
                2 // E
            } else {
                6 // W
            }
        }
        _ => panic!("num_directions must be 4 or 8"),
    }
}

/// Per-edge direction slot tables `(dir_uv, dir_vu)`, each `[E]`.
///
/// `dy = pos[v].y - pos[u].y`, `dx = pos[v].x - pos[u].x`; the u-endpoint slot
/// is `compute_direction_index(dy, dx)` and the v-endpoint slot uses the
/// negated delta `(-dy, -dx)` (build_directional_restriction_maps in Python).
pub fn direction_slot_tables(
    edges: &[[u32; 2]],
    node_positions: &Array2<f32>,
    num_directions: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut dir_uv = Vec::with_capacity(edges.len());
    let mut dir_vu = Vec::with_capacity(edges.len());
    for &[u, v] in edges {
        let (u, v) = (u as usize, v as usize);
        let dy = node_positions[[v, 0]] - node_positions[[u, 0]];
        let dx = node_positions[[v, 1]] - node_positions[[u, 1]];
        dir_uv.push(compute_direction_index(dy, dx, num_directions));
        dir_vu.push(compute_direction_index(-dy, -dx, num_directions));
    }
    (dir_uv, dir_vu)
}

/// Gather the stacked base maps `r_stack [K, d_e, d_v]` into the per-edge
/// `[E, 2, d_e, d_v]` layout by explicit slot tables (`out[e, 0] =
/// r_stack[dir_uv[e]]`, `out[e, 1] = r_stack[dir_vu[e]]`).
pub fn build_restriction_maps_from_tables(
    r_stack: &Array3<f32>,
    dir_uv: &[u8],
    dir_vu: &[u8],
) -> RestrictionMaps {
    assert_eq!(dir_uv.len(), dir_vu.len(), "slot tables must have equal length");
    let (k, d_e, d_v) = r_stack.dim();
    let e = dir_uv.len();
    let mut out = RestrictionMaps::zeros((e, 2, d_e, d_v));
    for (i, (&du, &dv)) in dir_uv.iter().zip(dir_vu.iter()).enumerate() {
        assert!((du as usize) < k && (dv as usize) < k, "slot index out of range");
        out.slice_mut(s![i, 0, .., ..])
            .assign(&r_stack.index_axis(Axis(0), du as usize));
        out.slice_mut(s![i, 1, .., ..])
            .assign(&r_stack.index_axis(Axis(0), dv as usize));
    }
    out
}

/// Gather the stacked base maps `r_stack [K, d_e, d_v]` into the per-edge
/// `[E, 2, d_e, d_v]` layout by the graph's precomputed direction slot tables.
/// Ports `build_directional_restriction_maps`.
pub fn build_directional_restriction_maps(
    r_stack: &Array3<f32>,
    graph: &AgentGraph,
) -> RestrictionMaps {
    assert_eq!(
        graph.dir_uv.len(),
        graph.num_edges(),
        "graph dir_uv table must cover every edge"
    );
    assert_eq!(
        graph.dir_vu.len(),
        graph.num_edges(),
        "graph dir_vu table must cover every edge"
    );
    build_restriction_maps_from_tables(r_stack, &graph.dir_uv, &graph.dir_vu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use sheaf_core::graph::{Endpoint, NodeIncidence};

    // ---- direction slot tables (PLAN.md: "unit tests pin these tables") ----

    #[test]
    fn eight_way_unit_deltas_pin_the_slot_order() {
        // (dy, dx) -> slot, exact (N, NE, E, SE, S, SW, W, NW) = 0..7 order.
        // dy grows downward (y, x) positions: N means v is ABOVE u (dy < 0).
        let table = [
            ((-1.0, 0.0), 0u8),  // N
            ((-1.0, 1.0), 1),    // NE
            ((0.0, 1.0), 2),     // E
            ((1.0, 1.0), 3),     // SE
            ((1.0, 0.0), 4),     // S
            ((1.0, -1.0), 5),    // SW
            ((0.0, -1.0), 6),    // W
            ((-1.0, -1.0), 7),   // NW
        ];
        for ((dy, dx), slot) in table {
            assert_eq!(compute_direction_index(dy, dx, 8), slot, "delta ({dy},{dx})");
        }
    }

    #[test]
    fn four_way_vertical_priority_on_diagonals() {
        // 4-way: 0=N, 1=E, 2=S, 3=W; any dy != 0 wins over dx.
        assert_eq!(compute_direction_index(-1.0, 0.0, 4), 0); // N
        assert_eq!(compute_direction_index(0.0, 1.0, 4), 1); // E
        assert_eq!(compute_direction_index(1.0, 0.0, 4), 2); // S
        assert_eq!(compute_direction_index(0.0, -1.0, 4), 3); // W
        // Diagonals: vertical takes priority.
        assert_eq!(compute_direction_index(-1.0, 1.0, 4), 0); // NE -> N
        assert_eq!(compute_direction_index(-1.0, -1.0, 4), 0); // NW -> N
        assert_eq!(compute_direction_index(1.0, 1.0, 4), 2); // SE -> S
        assert_eq!(compute_direction_index(1.0, -1.0, 4), 2); // SW -> S
    }

    #[test]
    fn v_endpoint_uses_negated_delta() {
        // A right/down-oriented grid edge builder only ever emits E, S, SE, SW
        // u-slots; the v-slot is always the opposite direction.
        // (u-slot, v-slot) pairs: E<->W, S<->N, SE<->NW, SW<->NE.
        let positions = ndarray::array![
            [0.0f32, 0.0], // node 0 at (y=0, x=0)
            [0.0, 2.0],    // node 1: right of 0
            [2.0, 0.0],    // node 2: below 0
            [2.0, 2.0],    // node 3: below-right of 0
        ];
        let edges = [
            [0u32, 1], // E edge
            [0, 2],    // S edge
            [0, 3],    // SE edge
            [1, 2],    // SW edge (down-left)
        ];
        let (dir_uv, dir_vu) = direction_slot_tables(&edges, &positions, 8);
        assert_eq!(dir_uv, vec![2, 4, 3, 5]); // E, S, SE, SW
        assert_eq!(dir_vu, vec![6, 0, 7, 1]); // W, N, NW, NE
    }

    // ---- assembly ----

    /// Grid graph with real slot tables and a trivially correct incidence CSR.
    fn make_graph(
        edges: Vec<[u32; 2]>,
        node_positions: Array2<f32>,
        num_directions: usize,
    ) -> AgentGraph {
        let n = node_positions.shape()[0];
        let (dir_uv, dir_vu) = direction_slot_tables(&edges, &node_positions, num_directions);
        // Build the node -> (edge, endpoint) incidence CSR.
        let mut per_node: Vec<Vec<(u32, Endpoint)>> = vec![Vec::new(); n];
        for (e, &[u, v]) in edges.iter().enumerate() {
            per_node[u as usize].push((e as u32, Endpoint::U));
            per_node[v as usize].push((e as u32, Endpoint::V));
        }
        let mut offsets = Vec::with_capacity(n + 1);
        let mut entries = Vec::new();
        offsets.push(0u32);
        for list in per_node {
            entries.extend(list);
            offsets.push(entries.len() as u32);
        }
        AgentGraph {
            edges,
            node_positions: Some(node_positions),
            dir_uv,
            dir_vu,
            node_edges: NodeIncidence { offsets, entries },
            num_nodes: n,
        }
    }

    #[test]
    fn assembly_gathers_by_slot_tables() {
        // Base maps R_k = constant k, so the gathered value identifies the slot.
        let (k, d_e, d_v) = (8usize, 2usize, 3usize);
        let r_stack = Array3::from_shape_fn((k, d_e, d_v), |(i, _, _)| i as f32);
        let positions = ndarray::array![[0.0f32, 0.0], [0.0, 2.0], [2.0, 2.0]];
        let edges = vec![[0u32, 1], [1, 2], [0, 2]]; // E, S, SE
        let graph = make_graph(edges, positions, 8);
        let maps = build_directional_restriction_maps(&r_stack, &graph);
        assert_eq!(maps.shape(), &[3, 2, d_e, d_v]);
        // (u-slot, v-slot) per edge: (E,W)=(2,6), (S,N)=(4,0), (SE,NW)=(3,7).
        let expect = [(2.0f32, 6.0), (4.0, 0.0), (3.0, 7.0)];
        for (e, (u_val, v_val)) in expect.iter().enumerate() {
            assert!(maps.slice(s![e, 0, .., ..]).iter().all(|&x| x == *u_val));
            assert!(maps.slice(s![e, 1, .., ..]).iter().all(|&x| x == *v_val));
        }
    }

    #[test]
    fn four_conn_grid_edge_count_and_slots() {
        // 3x3 grid of centers, stride 2, 4-conn right/down edges:
        // 6 right (E) + 6 down (S) = 12 edges.
        let mut positions = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                positions.push([1.0 + 2.0 * y as f32, 1.0 + 2.0 * x as f32]);
            }
        }
        let positions =
            Array2::from_shape_vec((9, 2), positions.into_iter().flatten().collect()).unwrap();
        let mut edges = Vec::new();
        for y in 0..3u32 {
            for x in 0..3u32 {
                let n = y * 3 + x;
                if x + 1 < 3 {
                    edges.push([n, n + 1]); // right
                }
                if y + 1 < 3 {
                    edges.push([n, n + 3]); // down
                }
            }
        }
        assert_eq!(edges.len(), 12);
        let graph = make_graph(edges, positions, 4);
        // Right edges -> (E=1, W=3); down edges -> (S=2, N=0). 4-way order N,E,S,W.
        for (e, &[u, v]) in graph.edges.iter().enumerate() {
            if v == u + 1 {
                assert_eq!((graph.dir_uv[e], graph.dir_vu[e]), (1, 3), "right edge {e}");
            } else {
                assert_eq!((graph.dir_uv[e], graph.dir_vu[e]), (2, 0), "down edge {e}");
            }
        }
    }

    #[test]
    fn direction_names_match_safetensors_suffix_order() {
        assert_eq!(direction_names(4), &["N", "E", "S", "W"]);
        assert_eq!(
            direction_names(8),
            &["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        );
    }
}
