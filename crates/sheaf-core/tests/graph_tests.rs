//! Direction slot-table pins (the off-by-one magnet of PLAN.md §3.1) and
//! incidence-CSR structure tests.

mod common;

use common::*;

use ndarray::Array2;
use sheaf_core::graph::{compute_direction_index, AgentGraph, Endpoint};

#[test]
fn direction_index_4way_table() {
    // 0=N, 1=E, 2=S, 3=W. Positions are (y, x); north = smaller y (dy < 0).
    assert_eq!(compute_direction_index(-1.0, 0.0, 4), 0, "N");
    assert_eq!(compute_direction_index(0.0, 1.0, 4), 1, "E");
    assert_eq!(compute_direction_index(1.0, 0.0, 4), 2, "S");
    assert_eq!(compute_direction_index(0.0, -1.0, 4), 3, "W");
    // Vertical takes priority on diagonals.
    assert_eq!(compute_direction_index(-1.0, 1.0, 4), 0, "NE -> N");
    assert_eq!(compute_direction_index(-1.0, -1.0, 4), 0, "NW -> N");
    assert_eq!(compute_direction_index(1.0, 1.0, 4), 2, "SE -> S");
    assert_eq!(compute_direction_index(1.0, -1.0, 4), 2, "SW -> S");
    // Degenerate zero delta falls through to W (the Python else-branch).
    assert_eq!(compute_direction_index(0.0, 0.0, 4), 3, "zero -> W");
}

#[test]
fn direction_index_8way_table() {
    // 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW.
    assert_eq!(compute_direction_index(-1.0, 0.0, 8), 0, "N");
    assert_eq!(compute_direction_index(-1.0, 1.0, 8), 1, "NE");
    assert_eq!(compute_direction_index(0.0, 1.0, 8), 2, "E");
    assert_eq!(compute_direction_index(1.0, 1.0, 8), 3, "SE");
    assert_eq!(compute_direction_index(1.0, 0.0, 8), 4, "S");
    assert_eq!(compute_direction_index(1.0, -1.0, 8), 5, "SW");
    assert_eq!(compute_direction_index(0.0, -1.0, 8), 6, "W");
    assert_eq!(compute_direction_index(-1.0, -1.0, 8), 7, "NW");
    // Zero delta falls through to W.
    assert_eq!(compute_direction_index(0.0, 0.0, 8), 6, "zero -> W");
}

#[test]
fn opposite_slot_asymmetry() {
    // The v-endpoint uses the direction of (-dy, -dx): an 8-way edge pointing
    // NE at u is SW at v, etc. — every slot pairs with its 180-degree partner.
    let opposite_8 = [4u8, 5, 6, 7, 0, 1, 2, 3];
    for k in 0..8u8 {
        let (dy, dx) = match k {
            0 => (-1.0, 0.0),
            1 => (-1.0, 1.0),
            2 => (0.0, 1.0),
            3 => (1.0, 1.0),
            4 => (1.0, 0.0),
            5 => (1.0, -1.0),
            6 => (0.0, -1.0),
            _ => (-1.0, -1.0),
        };
        assert_eq!(compute_direction_index(dy, dx, 8), k);
        assert_eq!(
            compute_direction_index(-dy, -dx, 8),
            opposite_8[k as usize],
            "8-way opposite of slot {k}"
        );
    }
    let opposite_4 = [2u8, 3, 0, 1];
    for k in 0..4u8 {
        let (dy, dx) = match k {
            0 => (-1.0, 0.0),
            1 => (0.0, 1.0),
            2 => (1.0, 0.0),
            _ => (0.0, -1.0),
        };
        assert_eq!(compute_direction_index(dy, dx, 4), k);
        assert_eq!(
            compute_direction_index(-dy, -dx, 4),
            opposite_4[k as usize],
            "4-way opposite of slot {k}"
        );
    }
}

#[test]
fn grid_slot_tables_pinned() {
    // 2x2 grid, edges oriented right/down (+ diagonals):
    //   [0,1] E, [0,2] S, [1,3] S, [2,3] E, [0,3] SE, [1,2] SW.
    let g = grid_2x2_8way();
    assert_eq!(g.dir_uv, vec![2, 4, 4, 2, 3, 5], "dir_uv");
    assert_eq!(g.dir_vu, vec![6, 0, 0, 6, 7, 1], "dir_vu");
}

#[test]
fn incidence_csr_structure() {
    let g = tiny_graph(); // edges: [0,1],[0,2],[1,3],[2,3],[0,3]
    // Total entries = 2E; each edge appears exactly twice (once per endpoint).
    assert_eq!(g.node_edges.entries.len(), 2 * g.num_edges());
    let mut per_edge = vec![0usize; g.num_edges()];
    for n in 0..g.num_nodes {
        for &(ei, ep) in g.node_edges.incident(n) {
            per_edge[ei as usize] += 1;
            let [u, v] = g.edges[ei as usize];
            match ep {
                Endpoint::U => assert_eq!(u as usize, n, "U entry filed under u"),
                Endpoint::V => assert_eq!(v as usize, n, "V entry filed under v"),
            }
        }
    }
    assert!(per_edge.iter().all(|&c| c == 2), "each edge twice: {per_edge:?}");
    // Node 0 sits on edges 0, 1, 4 (all as u).
    let n0: Vec<u32> = g.node_edges.incident(0).iter().map(|&(e, _)| e).collect();
    assert_eq!(n0, vec![0, 1, 4]);
}

#[test]
fn new_grid_positions_are_y_x() {
    // Two nodes stacked vertically: node 1 is BELOW node 0 (larger y).
    // Edge (0 -> 1) points south: dir_uv = S, dir_vu = N.
    let positions = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
    let g = AgentGraph::new_grid(vec![[0, 1]], positions, 4);
    assert_eq!(g.dir_uv, vec![2], "downward edge is S at u");
    assert_eq!(g.dir_vu, vec![0], "downward edge is N at v");
}
