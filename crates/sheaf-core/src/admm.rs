//! The unrolled ADMM loop. Ports `sheaf_admm.admm.run_admm` / `run_admm_history`.
//!
//! One `step()` shared by both drivers:
//! ```text
//! z_prev  = z
//! x       = x_solver(z - y; rho)                       # prox at v = z - y
//! x_relax = if alpha == 1.0 { x } else { alpha*x + (1-alpha)*z_prev }
//! z       = z_solver(x_relax + y, z_prev; geometry, rho)
//! y      += x_relax - z
//! ```
//! Init: `x = z = z_init` (encoder h, or zeros), `y = 0`.
//! `grad_window` / `stop_gradient` / fori-vs-scan: training-only, dropped —
//! forward values are identical (Python test
//! `test_truncated_bptt_forward_matches_full_unroll`). PLAN.md §3.3/§4.

use ndarray::{Array2, Array3, Array4};

use crate::geometry::SheafGeometry;
use crate::solvers::{EncoderOutput, UnrolledCgParams};
use crate::tensor::NodeState;

/// Which x-update to run (selected by config `x_solver`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XSolverKind {
    /// Closed-form diagonal prox (maze / all diagonal objectives).
    DiagonalProx,
    /// Tethered average (objective_mode = "simple").
    Simple,
}

/// ADMM loop parameters. `rho` is the export-baked learned scalar penalty;
/// `alpha` the over-relaxation (shipped 1.0 — the `== 1.0` fast path skips the
/// blend bitwise); `gamma` the soft-consensus weight (maze 5.0); `k` the number
/// of unrolled iterations (K_eval = 100; goldens use K = 12).
#[derive(Debug, Clone, Copy)]
pub struct AdmmParams {
    pub rho: f32,
    pub alpha: f32,
    pub gamma: f32,
    pub k: usize,
}

/// Per-agent ADMM variables, each `[N, B, d_v]`.
#[derive(Debug, Clone)]
pub struct AdmmState {
    /// Local proposal (primal).
    pub x: NodeState,
    /// Consensus iterate.
    pub z: NodeState,
    /// Scaled dual accumulator (u = lambda / rho).
    pub y: NodeState,
}

/// Per-iteration ADMM trajectory (oldest first). Mirrors Python `ADMMHistory`.
#[derive(Debug, Clone)]
pub struct AdmmHistory {
    pub x: Array4<f32>,               // [K, N, B, d_v]
    pub z: Array4<f32>,               // [K, N, B, d_v]
    pub y: Array4<f32>,               // [K, N, B, d_v]
    /// `||x_relaxed - z||_2` over d_v, per agent.
    pub primal_res: Array3<f32>,      // [K, N, B]
    /// `rho * ||z - z_prev||_2` over d_v, per agent.
    pub dual_res: Array3<f32>,        // [K, N, B]
    /// Geometry `consistency_rms(z)` per iteration.
    pub consistency_rms: Array2<f32>, // [K, B]
}

/// Run `params.k` ADMM steps. Returns the final state and the last
/// `loss_window` x-iterates (oldest first, window clamped to k) —
/// `x_window.last() == &state.x`.
#[allow(clippy::too_many_arguments)]
pub fn run_admm(
    enc: &EncoderOutput,
    geometry: &dyn SheafGeometry,
    x_solver: XSolverKind,
    z_params: &UnrolledCgParams,
    params: &AdmmParams,
    z_init: &NodeState,
    loss_window: usize,
) -> (AdmmState, Vec<NodeState>) {
    todo!("shared step() + window collection")
}

/// Run `params.k` ADMM steps recording the full per-iteration trajectory
/// (forward-only diagnostic path; feeds the demo and the golden parity tests).
pub fn run_admm_history(
    enc: &EncoderOutput,
    geometry: &dyn SheafGeometry,
    x_solver: XSolverKind,
    z_params: &UnrolledCgParams,
    params: &AdmmParams,
    z_init: &NodeState,
) -> (AdmmState, AdmmHistory) {
    todo!("same step(); push snapshots + primal/dual residuals + consistency_rms")
}
