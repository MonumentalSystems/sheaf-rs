//! Fixed (input-independent) restriction-map geometry. Ports `geometry/fixed.py`.

use std::sync::Arc;

use crate::graph::AgentGraph;
use crate::tensor::{BatchVec, EdgeState, NodeState, RestrictionMaps};

use super::SheafGeometry;

/// Sheaf geometry with learned but input-independent restriction maps.
pub struct FixedGeometry {
    pub graph: Arc<AgentGraph>,
    /// `[E, 2, d_e, d_v]`; slot 0 = F_{u->e}, slot 1 = F_{v->e}.
    pub restriction_maps: RestrictionMaps,
    /// Optional `[E]` float mask, multiplied onto the residual ONCE
    /// (in `edge_residuals`) — NOT again on the adjoint.
    pub edge_mask: Option<Vec<f32>>,
}

impl FixedGeometry {
    pub fn new(graph: Arc<AgentGraph>, restriction_maps: RestrictionMaps) -> Self {
        Self { graph, restriction_maps, edge_mask: None }
    }
}

impl SheafGeometry for FixedGeometry {
    fn edge_residuals(&self, z: &NodeState) -> EdgeState {
        todo!("per edge e, batch b: r = F_u z_u - F_v z_v (d_e x d_v GEMV per endpoint)")
    }

    fn laplacian_apply(&self, z: &NodeState, out: &mut NodeState) {
        todo!("r = edge_residuals(z); scatter-add F_u^T r at u, -F_v^T r at v")
    }

    fn energy(&self, z: &NodeState) -> f32 {
        todo!("0.5 * sum(r^2)")
    }

    fn consistency_rms(&self, z: &NodeState) -> BatchVec {
        todo!("sqrt(mean over (E, d_e) of r^2 + 1e-6) per batch element")
    }
}
