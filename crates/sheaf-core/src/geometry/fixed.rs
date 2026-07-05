//! Fixed (input-independent) restriction-map geometry. Ports `geometry/fixed.py`.

use std::sync::Arc;

use ndarray::{s, Array2, ArrayView2};

use crate::graph::AgentGraph;
use crate::tensor::{EdgeState, NodeState, RestrictionMaps};
use crate::Scalar;

use super::SheafGeometry;

/// Sheaf geometry with learned but input-independent restriction maps.
pub struct FixedGeometry {
    pub graph: Arc<AgentGraph>,
    /// `[E, 2, d_e, d_v]`; slot 0 = F_{u->e}, slot 1 = F_{v->e}.
    pub restriction_maps: RestrictionMaps,
    /// Optional `[E]` float mask, multiplied onto the residual ONCE
    /// (in `edge_residuals`) — NOT again on the adjoint.
    pub edge_mask: Option<Vec<Scalar>>,
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
        // Batch axis is embarrassingly parallel (the Laplacian never mixes B).
        // Each batch element computes its own `[E, d_e]` slab; the per-batch
        // arithmetic is unchanged, so the assembled result is bitwise identical.
        let slabs = crate::par::map_batches(b, |bi| {
            let mut r_b = Array2::<Scalar>::zeros((e_cnt, d_e));
            for (ei, &[u, v]) in self.graph.edges.iter().enumerate() {
                let f_u: ArrayView2<Scalar> = self.restriction_maps.slice(s![ei, 0, .., ..]); // [d_e, d_v]
                let f_v: ArrayView2<Scalar> = self.restriction_maps.slice(s![ei, 1, .., ..]);
                let z_u = z.slice(s![u as usize, bi, ..]);
                let z_v = z.slice(s![v as usize, bi, ..]);
                // r = F_{u->e} z_u - F_{v->e} z_v
                let mut r_eb = f_u.dot(&z_u);
                r_eb -= &f_v.dot(&z_v);
                if let Some(mask) = &self.edge_mask {
                    r_eb *= mask[ei];
                }
                r_b.slice_mut(s![ei, ..]).assign(&r_eb);
            }
            r_b
        });
        let mut r = EdgeState::zeros((e_cnt, b, d_e));
        for (bi, slab) in slabs.iter().enumerate() {
            r.slice_mut(s![.., bi, ..]).assign(slab);
        }
        r
    }

    fn laplacian_apply(&self, z: &NodeState, out: &mut NodeState) {
        assert_eq!(out.dim(), z.dim());
        let r = self.edge_residuals(z); // [E, B, d_e]
        let (n, b, d_v) = z.dim();
        // F^T r, scattered back to the two endpoints (orientation sign on v).
        // Plain += accumulation — duplicate node indices accumulate, matching
        // JAX `at[u].add(contrib_u)` / `at[v].add(-contrib_v)`. Parallel over B
        // (each batch owns a disjoint `[N, d_v]` slab); the per-batch edge-order
        // accumulation is preserved, so results are identical to the serial run.
        let slabs = crate::par::map_batches(b, |bi| {
            let mut out_b = Array2::<Scalar>::zeros((n, d_v));
            for (ei, &[u, v]) in self.graph.edges.iter().enumerate() {
                let f_u: ArrayView2<Scalar> = self.restriction_maps.slice(s![ei, 0, .., ..]);
                let f_v: ArrayView2<Scalar> = self.restriction_maps.slice(s![ei, 1, .., ..]);
                let r_eb = r.slice(s![ei, bi, ..]);
                let contrib_u = f_u.t().dot(&r_eb); // [d_v]
                let contrib_v = f_v.t().dot(&r_eb);
                {
                    let mut out_u = out_b.slice_mut(s![u as usize, ..]);
                    out_u += &contrib_u;
                }
                let mut out_v = out_b.slice_mut(s![v as usize, ..]);
                out_v -= &contrib_v;
            }
            out_b
        });
        out.fill(0.0);
        for (bi, slab) in slabs.iter().enumerate() {
            out.slice_mut(s![.., bi, ..]).assign(slab);
        }
    }
}
