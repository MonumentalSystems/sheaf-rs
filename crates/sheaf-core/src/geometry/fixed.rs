//! Fixed (input-independent) restriction-map geometry. Ports `geometry/fixed.py`.

use std::sync::Arc;

use ndarray::s;

use crate::graph::AgentGraph;
use crate::tensor::{EdgeState, NodeState, RestrictionMaps};

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
        assert_eq!(
            restriction_maps.shape()[0],
            graph.num_edges(),
            "restriction_maps leading axis must be E"
        );
        Self { graph, restriction_maps, edge_mask: None }
    }
}

impl SheafGeometry for FixedGeometry {
    fn edge_residuals(&self, z: &NodeState) -> EdgeState {
        let (_n, b, _d_v) = z.dim();
        let d_e = self.restriction_maps.shape()[2];
        let e_cnt = self.graph.num_edges();
        let mut r = EdgeState::zeros((e_cnt, b, d_e));
        for (ei, &[u, v]) in self.graph.edges.iter().enumerate() {
            let f_u: ndarray::ArrayView2<f32> =
                self.restriction_maps.slice(s![ei, 0, .., ..]); // [d_e, d_v]
            let f_v: ndarray::ArrayView2<f32> = self.restriction_maps.slice(s![ei, 1, .., ..]);
            for bi in 0..b {
                let z_u: ndarray::ArrayView1<f32> = z.slice(s![u as usize, bi, ..]);
                let z_v: ndarray::ArrayView1<f32> = z.slice(s![v as usize, bi, ..]);
                // r = F_{u->e} z_u - F_{v->e} z_v
                let mut r_eb = f_u.dot(&z_u);
                r_eb -= &f_v.dot(&z_v);
                if let Some(mask) = &self.edge_mask {
                    r_eb *= mask[ei];
                }
                r.slice_mut(s![ei, bi, ..]).assign(&r_eb);
            }
        }
        r
    }

    fn laplacian_apply(&self, z: &NodeState, out: &mut NodeState) {
        assert_eq!(out.dim(), z.dim());
        let r = self.edge_residuals(z); // [E, B, d_e]
        out.fill(0.0);
        let b = r.dim().1;
        // F^T r, scattered back to the two endpoints (orientation sign on v).
        // Plain += accumulation — duplicate node indices accumulate, matching
        // JAX `at[u].add(contrib_u)` / `at[v].add(-contrib_v)`.
        for (ei, &[u, v]) in self.graph.edges.iter().enumerate() {
            let f_u: ndarray::ArrayView2<f32> = self.restriction_maps.slice(s![ei, 0, .., ..]);
            let f_v: ndarray::ArrayView2<f32> = self.restriction_maps.slice(s![ei, 1, .., ..]);
            for bi in 0..b {
                let r_eb: ndarray::ArrayView1<f32> = r.slice(s![ei, bi, ..]);
                let contrib_u = f_u.t().dot(&r_eb); // [d_v]
                let contrib_v = f_v.t().dot(&r_eb);
                let mut out_u = out.slice_mut(s![u as usize, bi, ..]);
                out_u += &contrib_u;
                let mut out_v = out.slice_mut(s![v as usize, bi, ..]);
                out_v -= &contrib_v;
            }
        }
    }
}
