//! Base restriction-map assembly. Ports `geometry/restriction_maps.py`
//! (directional 8-way path only — maze scope).
//!
//! The learned parameters are K = `num_directions` base maps `R_name`
//! `[d_e, d_v]`, stacked in the exact `get_direction_names` order:
//! 8-way `(N, NE, E, SE, S, SW, W, NW)`. Assembly gathers them into the
//! per-edge `[E, 2, d_e, d_v]` layout using the graph's precomputed
//! `dir_uv` / `dir_vu` slot tables (slot 0 = u-endpoint, 1 = v-endpoint;
//! the v-endpoint uses the direction of `(-dy, -dx)`).

use ndarray::Array3;

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

/// Gather the stacked base maps `r_stack [K, d_e, d_v]` into the per-edge
/// `[E, 2, d_e, d_v]` layout by the graph's direction slot tables.
/// Ports `build_directional_restriction_maps`.
pub fn build_directional_restriction_maps(
    r_stack: &Array3<f32>,
    graph: &AgentGraph,
) -> RestrictionMaps {
    todo!("out[e, 0] = r_stack[dir_uv[e]]; out[e, 1] = r_stack[dir_vu[e]]")
}
