//! LoRA geometry: input-modulated maps `F = R + (alpha/r) A B^T`.
//! Ports `geometry/lora.py`. F is NEVER materialized — the low-rank update is
//! applied factored (two rank-r matvecs).

use std::sync::Arc;

use ndarray::{Array2, Array4, Array5};

use crate::graph::AgentGraph;
use crate::tensor::{BatchVec, EdgeState, NodeState, RestrictionMaps};

use super::SheafGeometry;

/// Sheaf geometry with per-agent LoRA-modulated restriction maps.
///
/// The `*_edge` factors are pre-gathered ONCE at construction (mirrors
/// `create_lora_geometry` / `create_sudoku_lora_geometry`): each edge already
/// holds its endpoints' A/B factors in the relevant direction/slot, so
/// `laplacian_apply` is gather-free.
///
/// Apply:   `F z   = R z + scale * gate * A (B^T z)`
/// Adjoint: `F^T r = R^T r + scale * B (gate * (A^T r))`
///   (the gate multiplies the `A^T r` side, matching lora.py `_adjoint_endpoint`).
/// `scale = lora_alpha / rank`.
pub struct LoraGeometry {
    pub graph: Arc<AgentGraph>,
    /// Base maps R, `[E, 2, d_e, d_v]`.
    pub restriction_maps: RestrictionMaps,
    pub lora_alpha: f32,
    pub a_u_edge: Array4<f32>, // [E, B, d_e, r]
    pub a_v_edge: Array4<f32>, // [E, B, d_e, r]
    pub b_u_edge: Array4<f32>, // [E, B, d_v, r]
    pub b_v_edge: Array4<f32>, // [E, B, d_v, r]
    pub gate_u_edge: Option<Array2<f32>>, // [E, B]
    pub gate_v_edge: Option<Array2<f32>>, // [E, B]
    pub edge_mask: Option<Vec<f32>>,      // [E]; applied once, in edge_residuals
}

impl LoraGeometry {
    /// `scale = lora_alpha / rank` (rank = trailing axis of the A factors).
    pub fn scale(&self) -> f32 {
        self.lora_alpha / self.a_u_edge.shape()[3] as f32
    }

    /// Directional (grid) LoRA geometry: gather per-node factors
    /// `a: [N, B, K, d_e, r]`, `b: [N, B, K, d_v, r]`, `gate: [N, B, K]` into
    /// edge-indexed tensors using the graph's precomputed `dir_uv` / `dir_vu`
    /// slot tables. Ports `create_lora_geometry`.
    pub fn create_directional(
        graph: Arc<AgentGraph>,
        restriction_maps: RestrictionMaps,
        a: &Array5<f32>,
        b: &Array5<f32>,
        gate: Option<&ndarray::Array3<f32>>,
        lora_alpha: f32,
    ) -> Self {
        todo!("gather A[u][:, dir_uv[e]], A[v][:, dir_vu[e]], etc. per edge")
    }
}

impl SheafGeometry for LoraGeometry {
    fn edge_residuals(&self, z: &NodeState) -> EdgeState {
        todo!("(R + scale*gate*A B^T) z_u - (R + scale*gate*A B^T) z_v per edge/batch")
    }

    fn laplacian_apply(&self, z: &NodeState, out: &mut NodeState) {
        todo!("adjoint per endpoint (gate on the A^T r side), scatter-add +u/-v")
    }

    fn energy(&self, z: &NodeState) -> f32 {
        todo!("0.5 * sum(r^2)")
    }

    fn consistency_rms(&self, z: &NodeState) -> BatchVec {
        todo!("sqrt(mean over (E, d_e) of r^2 + 1e-6) per batch element")
    }
}
