//! Sheaf geometries: the coboundary / Laplacian operators the z-update needs.
//!
//! Ports `sheaf_admm.geometry.{base,fixed,lora}`. The sheaf Laplacian
//! `L_F = delta^T delta` is **never materialized** — both implementations are
//! matrix-free per-edge matvecs with a scatter-add (`+contrib` at u,
//! `-contrib` at v; duplicate node indices accumulate, like JAX `at[].add`).

mod fixed;
mod lora;

pub use fixed::FixedGeometry;
pub use lora::LoraGeometry;

use crate::tensor::{BatchVec, EdgeState, NodeState};
use crate::Scalar;

/// Epsilon UNDER the sqrt in `consistency_rms` (base.py default).
pub const CONSISTENCY_EPS: Scalar = 1e-6;

/// The sheaf-geometry interface (mirrors the Python `SheafGeometry` Protocol).
///
/// `energy` and `consistency_rms` have default implementations in terms of
/// `edge_residuals` — the formulas are identical in both Python geometry
/// classes (fixed.py / lora.py duplicate them verbatim).
pub trait SheafGeometry {
    /// Coboundary `F z`: per-edge disagreement `F_{u->e} z_u - F_{v->e} z_v`.
    /// `z: [N, B, d_v] -> [E, B, d_e]`. `edge_mask` (if any) multiplies the
    /// residual here ONCE — not again on the adjoint.
    fn edge_residuals(&self, z: &NodeState) -> EdgeState;

    /// Sheaf Laplacian matvec `L_F z = F^T F z`, written into `out`
    /// (`[N, B, d_v]`, zeroed by the callee before accumulation).
    fn laplacian_apply(&self, z: &NodeState, out: &mut NodeState);

    /// Scalar sheaf energy `1/2 sum_e ||r_e||^2` (summed over E, B, d_e).
    fn energy(&self, z: &NodeState) -> Scalar {
        let r = self.edge_residuals(z);
        // Final reduction kept serial in flat memory order: bitwise-identical
        // regardless of the `parallel` feature (only `r` is fanned over B).
        0.5 * r.iter().map(|&x| x * x).sum::<Scalar>()
    }

    /// Per-batch RMS disagreement `sqrt(mean_{E,d_e}(r^2) + 1e-6)` -> `[B]`.
    /// The mean reduces over axes (0, 2) of `[E, B, d_e]`; eps is UNDER the
    /// sqrt (keeps the training gradient finite at perfect consensus).
    fn consistency_rms(&self, z: &NodeState) -> BatchVec {
        let r = self.edge_residuals(z);
        let (e, b, d_e) = r.dim();
        let denom = (e * d_e) as Scalar;
        // Batch-parallel (`parallel` feature); each batch's (E, d_e) reduction
        // order is unchanged, so results are identical to the serial path.
        let vals = crate::par::map_batches(b, |bi| {
            let mut acc = 0.0 as Scalar;
            for ei in 0..e {
                for di in 0..d_e {
                    let v = r[[ei, bi, di]];
                    acc += v * v;
                }
            }
            (acc / denom + CONSISTENCY_EPS).sqrt()
        });
        BatchVec::from_vec(vals)
    }
}
