//! Tensor type aliases shared across the workspace.
//!
//! Shape conventions (PLAN.md appendix, hard-coded in tests):
//! - node states x, z, y: `[N, B, d_v]`
//! - edge residuals:      `[E, B, d_e]`
//! - restriction maps:    `[E, 2, d_e, d_v]` (slot 0 = F_{u->e}, slot 1 = F_{v->e})
//! - history:             `[K, N, B, d_v]`; residuals `[K, N, B]`; consistency `[K, B]`

use ndarray::{Array1, Array2, Array3, Array4};

/// Per-agent node state `[N, B, d_v]` (x, z, or y).
pub type NodeState = Array3<f32>;

/// Per-edge state `[E, B, d_e]` (coboundary residuals).
pub type EdgeState = Array3<f32>;

/// Per-batch scalars `[B]`.
pub type BatchVec = Array1<f32>;

/// Per-agent scalars `[N, B]` (residual norms; broadcast rho).
pub type NodeScalars = Array2<f32>;

/// Assembled restriction maps `[E, 2, d_e, d_v]`.
pub type RestrictionMaps = Array4<f32>;
