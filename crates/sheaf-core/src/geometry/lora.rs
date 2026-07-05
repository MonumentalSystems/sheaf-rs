//! LoRA geometry: input-modulated maps `F = R + (alpha/r) A B^T`.
//! Ports `geometry/lora.py`. F is NEVER materialized — the low-rank update is
//! applied factored (two rank-r matvecs).

use std::sync::Arc;

use ndarray::{s, Array1, Array2, Array4, Array5, ArrayView1, ArrayView2};

use crate::graph::AgentGraph;
use crate::tensor::{EdgeState, NodeState, RestrictionMaps};
use crate::Scalar;

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
    pub lora_alpha: Scalar,
    pub a_u_edge: Array4<Scalar>, // [E, B, d_e, r]
    pub a_v_edge: Array4<Scalar>, // [E, B, d_e, r]
    pub b_u_edge: Array4<Scalar>, // [E, B, d_v, r]
    pub b_v_edge: Array4<Scalar>, // [E, B, d_v, r]
    pub gate_u_edge: Option<Array2<Scalar>>, // [E, B]
    pub gate_v_edge: Option<Array2<Scalar>>, // [E, B]
    pub edge_mask: Option<Vec<Scalar>>,      // [E]; applied once, in edge_residuals
}

/// Effective map applied to one endpoint: `(R + scale gate A B^T) z_e`.
/// Factored — two rank-r matvecs; the gate multiplies the low-rank term.
fn apply_endpoint(
    r_base: ArrayView2<Scalar>,  // [d_e, d_v]
    a: ArrayView2<Scalar>,       // [d_e, r]
    b: ArrayView2<Scalar>,       // [d_v, r]
    gate: Option<Scalar>,
    z_e: ArrayView1<Scalar>,     // [d_v]
    scale: Scalar,
) -> Array1<Scalar> {
    let rz = r_base.dot(&z_e); // [d_e]
    let btz = b.t().dot(&z_e); // [r]
    let mut abtz = a.dot(&btz); // [d_e]
    if let Some(g) = gate {
        abtz *= g;
    }
    rz + &(abtz * scale)
}

/// Adjoint of `apply_endpoint`: `(R + scale gate A B^T)^T r`.
/// The gate multiplies the `A^T r` side (lora.py `_adjoint_endpoint`).
fn adjoint_endpoint(
    r_base: ArrayView2<Scalar>,  // [d_e, d_v]
    a: ArrayView2<Scalar>,       // [d_e, r]
    b: ArrayView2<Scalar>,       // [d_v, r]
    gate: Option<Scalar>,
    r_e: ArrayView1<Scalar>,     // [d_e]
    scale: Scalar,
) -> Array1<Scalar> {
    let contrib = r_base.t().dot(&r_e); // [d_v]
    let mut atr = a.t().dot(&r_e); // [r]
    if let Some(g) = gate {
        atr *= g;
    }
    let lora = b.dot(&atr); // [d_v]
    contrib + &(lora * scale)
}

impl LoraGeometry {
    /// `scale = lora_alpha / rank` (rank = trailing axis of the A factors).
    pub fn scale(&self) -> Scalar {
        self.lora_alpha / self.a_u_edge.shape()[3] as Scalar
    }

    /// Directional (grid) LoRA geometry: gather per-node factors
    /// `a: [N, B, K, d_e, r]`, `b: [N, B, K, d_v, r]`, `gate: [N, B, K]` into
    /// edge-indexed tensors using the graph's precomputed `dir_uv` / `dir_vu`
    /// slot tables. Ports `create_lora_geometry` (+ `_gather_edge_factors`).
    pub fn create_directional(
        graph: Arc<AgentGraph>,
        restriction_maps: RestrictionMaps,
        a: &Array5<Scalar>,
        b: &Array5<Scalar>,
        gate: Option<&ndarray::Array3<Scalar>>,
        lora_alpha: Scalar,
    ) -> Self {
        let e_cnt = graph.num_edges();
        assert_eq!(
            graph.dir_uv.len(),
            e_cnt,
            "graph has no directional slot tables — build it with AgentGraph::new_grid"
        );
        let (_n, bsz, _k, d_e, rank) = a.dim();
        let d_v = b.dim().3;
        assert_eq!(b.dim().4, rank, "A and B must share the rank axis");

        let mut a_u = Array4::zeros((e_cnt, bsz, d_e, rank));
        let mut a_v = Array4::zeros((e_cnt, bsz, d_e, rank));
        let mut b_u = Array4::zeros((e_cnt, bsz, d_v, rank));
        let mut b_v = Array4::zeros((e_cnt, bsz, d_v, rank));
        let mut g_u = gate.map(|_| Array2::zeros((e_cnt, bsz)));
        let mut g_v = gate.map(|_| Array2::zeros((e_cnt, bsz)));

        for (ei, &[u, v]) in graph.edges.iter().enumerate() {
            let (u, v) = (u as usize, v as usize);
            let su = graph.dir_uv[ei] as usize;
            let sv = graph.dir_vu[ei] as usize;
            a_u.slice_mut(s![ei, .., .., ..]).assign(&a.slice(s![u, .., su, .., ..]));
            a_v.slice_mut(s![ei, .., .., ..]).assign(&a.slice(s![v, .., sv, .., ..]));
            b_u.slice_mut(s![ei, .., .., ..]).assign(&b.slice(s![u, .., su, .., ..]));
            b_v.slice_mut(s![ei, .., .., ..]).assign(&b.slice(s![v, .., sv, .., ..]));
            if let Some(g) = gate {
                let gu = g_u.as_mut().unwrap();
                let gv = g_v.as_mut().unwrap();
                for bi in 0..bsz {
                    gu[[ei, bi]] = g[[u, bi, su]];
                    gv[[ei, bi]] = g[[v, bi, sv]];
                }
            }
        }

        Self {
            graph,
            restriction_maps,
            lora_alpha,
            a_u_edge: a_u,
            a_v_edge: a_v,
            b_u_edge: b_u,
            b_v_edge: b_v,
            gate_u_edge: g_u,
            gate_v_edge: g_v,
            edge_mask: None,
        }
    }
}

impl SheafGeometry for LoraGeometry {
    fn edge_residuals(&self, z: &NodeState) -> EdgeState {
        let (_n, b, _d_v) = z.dim();
        let d_e = self.restriction_maps.shape()[2];
        let e_cnt = self.graph.num_edges();
        let scale = self.scale();
        // Batch axis is embarrassingly parallel; each batch element owns its
        // `[E, d_e]` slab and runs the identical factored per-edge arithmetic.
        let slabs = crate::par::map_batches(b, |bi| {
            let mut r_b = Array2::<Scalar>::zeros((e_cnt, d_e));
            for (ei, &[u, v]) in self.graph.edges.iter().enumerate() {
                let (u, v) = (u as usize, v as usize);
                let r_u = self.restriction_maps.slice(s![ei, 0, .., ..]);
                let r_v = self.restriction_maps.slice(s![ei, 1, .., ..]);
                let fz_u = apply_endpoint(
                    r_u,
                    self.a_u_edge.slice(s![ei, bi, .., ..]),
                    self.b_u_edge.slice(s![ei, bi, .., ..]),
                    self.gate_u_edge.as_ref().map(|g| g[[ei, bi]]),
                    z.slice(s![u, bi, ..]),
                    scale,
                );
                let fz_v = apply_endpoint(
                    r_v,
                    self.a_v_edge.slice(s![ei, bi, .., ..]),
                    self.b_v_edge.slice(s![ei, bi, .., ..]),
                    self.gate_v_edge.as_ref().map(|g| g[[ei, bi]]),
                    z.slice(s![v, bi, ..]),
                    scale,
                );
                let mut r_eb = fz_u - &fz_v;
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
        let scale = self.scale();
        // Adjoint per endpoint, scatter-add +contrib at u, -contrib at v.
        // Parallel over B: each batch owns a disjoint `[N, d_v]` slab and its
        // edge-order accumulation is preserved, so the result is identical.
        let slabs = crate::par::map_batches(b, |bi| {
            let mut out_b = Array2::<Scalar>::zeros((n, d_v));
            for (ei, &[u, v]) in self.graph.edges.iter().enumerate() {
                let (u, v) = (u as usize, v as usize);
                let r_u = self.restriction_maps.slice(s![ei, 0, .., ..]);
                let r_v = self.restriction_maps.slice(s![ei, 1, .., ..]);
                let r_eb = r.slice(s![ei, bi, ..]);
                let contrib_u = adjoint_endpoint(
                    r_u,
                    self.a_u_edge.slice(s![ei, bi, .., ..]),
                    self.b_u_edge.slice(s![ei, bi, .., ..]),
                    self.gate_u_edge.as_ref().map(|g| g[[ei, bi]]),
                    r_eb,
                    scale,
                );
                let contrib_v = adjoint_endpoint(
                    r_v,
                    self.a_v_edge.slice(s![ei, bi, .., ..]),
                    self.b_v_edge.slice(s![ei, bi, .., ..]),
                    self.gate_v_edge.as_ref().map(|g| g[[ei, bi]]),
                    r_eb,
                    scale,
                );
                {
                    let mut out_u = out_b.slice_mut(s![u, ..]);
                    out_u += &contrib_u;
                }
                let mut out_v = out_b.slice_mut(s![v, ..]);
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
