//! Unrolled conjugate-gradient z-solver. Ports `z_solvers/unrolled_cg.py`.
//!
//! Fixed T iterations (default 5), **no early stopping**. Batched: inner
//! products `bdot` reduce over axes (0, 2) of `[N, B, d_v]` -> per-batch `[B]`
//! scalars, broadcast back as `s[None, :, None]`.
//!
//! DO NOT "improve" any of the following (PLAN.md §3.2 / §3.4):
//! - denominator guard is exactly `1e-8` (1e-12 risks 0/0 NaN near fp32 roundoff);
//! - project mode is deliberately NOT an idempotent projector (warm start
//!   `w0 = z_target - z_prev` keeps kernel components);
//! - prox mode inits at `z_target` (`prox_init = "legacy"`, the shipped
//!   default; the 'warm' detached init is a training-only gradient-boundary
//!   detail, dropped from inference per PLAN.md §4 Tier 1).

use crate::geometry::SheafGeometry;
use crate::tensor::NodeState;

/// CG denominator guard. Exactly 1e-8 — see module docs.
pub const CG_DENOM_EPS: f32 = 1e-8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZMode {
    /// Hard consensus: solve `(L + eps*I) w = L z_target`, return `z_target - w`,
    /// warm start `w0 = z_target - z_prev`. Tikhonov eps default 1e-5.
    Project,
    /// Soft consensus (maze): solve `(gamma*L + rho*I) z = rho * z_target`,
    /// init at `z_target` (legacy).
    Prox,
}

#[derive(Debug, Clone, Copy)]
pub struct UnrolledCgParams {
    pub mode: ZMode,
    /// Soft-consensus weight gamma (maze: 5.0).
    pub gamma: f32,
    /// T, the fixed unroll length (maze: 5).
    pub num_iters: usize,
    /// Project-mode `L + eps*I` regularizer (default 1e-5). Distinct from
    /// `CG_DENOM_EPS`.
    pub tikhonov_eps: f32,
}

/// One z-update: T CG steps against the geometry's matrix-free Laplacian.
pub fn unrolled_cg_solve(
    z_target: &NodeState,
    z_prev: &NodeState,
    geometry: &dyn SheafGeometry,
    params: &UnrolledCgParams,
    rho: f32,
) -> NodeState {
    todo!("batched CG: alpha = rTr/(pTAp + 1e-8); beta = rTr_new/(rTr + 1e-8)")
}
