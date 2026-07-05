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

use ndarray::Axis;

use crate::geometry::SheafGeometry;
use crate::tensor::{BatchVec, NodeState};

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

impl Default for UnrolledCgParams {
    /// Python `UnrolledCGParams` defaults: prox mode, gamma 1.0, T = 5,
    /// Tikhonov 1e-5.
    fn default() -> Self {
        Self { mode: ZMode::Prox, gamma: 1.0, num_iters: 5, tikhonov_eps: 1e-5 }
    }
}

/// `L_F z` through the geometry's matrix-free matvec, as an owned array.
fn lap(geometry: &dyn SheafGeometry, z: &NodeState) -> NodeState {
    let mut out = NodeState::zeros(z.dim());
    geometry.laplacian_apply(z, &mut out);
    out
}

/// Batched inner product: reduce over agents (axis 0) and stalk dim (axis 2),
/// keeping the batch axis -> `[B]`.
fn bdot(a: &NodeState, c: &NodeState) -> BatchVec {
    let (n, b, d) = a.dim();
    let mut out = BatchVec::zeros(b);
    for bi in 0..b {
        let mut acc = 0.0f32;
        for ni in 0..n {
            for di in 0..d {
                acc += a[[ni, bi, di]] * c[[ni, bi, di]];
            }
        }
        out[bi] = acc;
    }
    out
}

/// Per-batch scalar broadcast: `s[None, :, None] * v`.
fn bscale(s: &BatchVec, v: &NodeState) -> NodeState {
    let sb = s.view().insert_axis(Axis(0)).insert_axis(Axis(2)); // [1, B, 1]
    v * &sb
}

/// Batched CG for `A x = b` over node states `[N, B, d]`. `num_iters` is
/// fixed (no early stopping — matching `_batched_cg`).
fn batched_cg<F: Fn(&NodeState) -> NodeState>(
    matvec: &F,
    b: &NodeState,
    x0: NodeState,
    num_iters: usize,
) -> NodeState {
    let mut x = x0;
    let mut r = b - &matvec(&x);
    let mut p = r.clone();
    let mut rtr = bdot(&r, &r);
    for _ in 0..num_iters {
        let ap = matvec(&p);
        let ptap = bdot(&p, &ap);
        let alpha = &rtr / &(ptap + CG_DENOM_EPS);
        x += &bscale(&alpha, &p);
        r -= &bscale(&alpha, &ap);
        let rtr_new = bdot(&r, &r);
        let beta = &rtr_new / &(rtr + CG_DENOM_EPS);
        p = &r + &bscale(&beta, &p);
        rtr = rtr_new;
    }
    x
}

/// One z-update: T CG steps against the geometry's matrix-free Laplacian.
pub fn unrolled_cg_solve(
    z_target: &NodeState,
    z_prev: &NodeState,
    geometry: &dyn SheafGeometry,
    params: &UnrolledCgParams,
    rho: f32,
) -> NodeState {
    match params.mode {
        ZMode::Project => {
            let eps = params.tikhonov_eps;
            let matvec = |x: &NodeState| {
                let mut ax = lap(geometry, x);
                ax.zip_mut_with(x, |a, &xi| *a += eps * xi);
                ax
            };
            let b = lap(geometry, z_target);
            // Warm-start the correction at the ADMM target delta. This is the
            // paper-run inexact hard-consensus path, not a standalone exact
            // Euclidean projector: kernel components in the warm start are not
            // removed by the RHS. Deliberately non-idempotent — do not "fix".
            let w0 = z_target - z_prev;
            let w = batched_cg(&matvec, &b, w0, params.num_iters);
            z_target - &w
        }
        ZMode::Prox => {
            let gamma = params.gamma;
            let matvec = |x: &NodeState| {
                let mut ax = lap(geometry, x);
                ax.zip_mut_with(x, |a, &xi| *a = gamma * *a + rho * xi);
                ax
            };
            let b = z_target.mapv(|v| rho * v);
            // prox_init = "legacy" (shipped default): init at z_target. The
            // 'warm' detached init is training-only and dropped (PLAN.md §4).
            batched_cg(&matvec, &b, z_target.clone(), params.num_iters)
        }
    }
}
