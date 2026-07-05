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

use ndarray::{s, Array2, Array3, Array4};

use crate::geometry::SheafGeometry;
use crate::solvers::{
    diagonal_prox_solve, simple_solve, unrolled_cg_solve, EncoderOutput, Objective,
    UnrolledCgParams,
};
use crate::tensor::NodeState;
use crate::Scalar;

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
///
/// Note: the z-update reads gamma from `UnrolledCgParams.gamma` (the direct
/// port of the Python `ZSolverParams`); `AdmmParams.gamma` is the same value
/// carried for convenience and must agree (debug-asserted by the drivers).
#[derive(Debug, Clone, Copy)]
pub struct AdmmParams {
    pub rho: Scalar,
    pub alpha: Scalar,
    pub gamma: Scalar,
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
    pub x: Array4<Scalar>,               // [K, N, B, d_v]
    pub z: Array4<Scalar>,               // [K, N, B, d_v]
    pub y: Array4<Scalar>,               // [K, N, B, d_v]
    /// `||x_relaxed - z||_2` over d_v, per agent.
    pub primal_res: Array3<Scalar>,      // [K, N, B]
    /// `rho * ||z - z_prev||_2` over d_v, per agent.
    pub dual_res: Array3<Scalar>,        // [K, N, B]
    /// Geometry `consistency_rms(z)` per iteration.
    pub consistency_rms: Array2<Scalar>, // [K, B]
}

/// One ADMM step. Returns the new state and `x_relaxed` (needed by the
/// history driver's primal residual).
fn admm_step(
    state: &AdmmState,
    enc: &EncoderOutput,
    geometry: &dyn SheafGeometry,
    x_solver: XSolverKind,
    z_params: &UnrolledCgParams,
    rho: Scalar,
    alpha: Scalar,
) -> (AdmmState, NodeState) {
    let z_prev = &state.z;
    let x = match x_solver {
        XSolverKind::DiagonalProx => diagonal_prox_solve(&state.z, &state.y, rho, &enc.objective),
        XSolverKind::Simple => match &enc.objective {
            Objective::Simple { beta } => simple_solve(&state.z, &state.y, rho, &enc.h, *beta),
            _ => panic!("XSolverKind::Simple requires Objective::Simple"),
        },
    };
    // alpha == 1.0 skips the blend entirely (bitwise fast path, JAX branch).
    let x_relaxed = if alpha == 1.0 {
        x.clone()
    } else {
        let mut xr = x.clone();
        xr.zip_mut_with(z_prev, |xr, &zp| *xr = alpha * *xr + (1.0 - alpha) * zp);
        xr
    };
    let z_target = &x_relaxed + &state.y;
    let z = unrolled_cg_solve(&z_target, z_prev, geometry, z_params, rho);
    let y = &state.y + &(&x_relaxed - &z);
    (AdmmState { x, z, y }, x_relaxed)
}

fn init_state(z_init: &NodeState) -> AdmmState {
    AdmmState {
        x: z_init.clone(),
        z: z_init.clone(),
        y: NodeState::zeros(z_init.dim()),
    }
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
    debug_assert_eq!(
        params.gamma, z_params.gamma,
        "AdmmParams.gamma and UnrolledCgParams.gamma must agree"
    );
    let mut state = init_state(z_init);
    // window = min(loss_window, K); the first K - window steps run without
    // collection (Python: n_pre scan, then the collecting scan).
    let window = loss_window.min(params.k);
    let n_pre = params.k - window;
    let mut x_window = Vec::with_capacity(window);
    for i in 0..params.k {
        let (new_state, _x_relaxed) =
            admm_step(&state, enc, geometry, x_solver, z_params, params.rho, params.alpha);
        state = new_state;
        if i >= n_pre {
            x_window.push(state.x.clone());
        }
    }
    (state, x_window)
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
    debug_assert_eq!(
        params.gamma, z_params.gamma,
        "AdmmParams.gamma and UnrolledCgParams.gamma must agree"
    );
    let (n, b, d_v) = z_init.dim();
    let k = params.k;
    let mut hx = Array4::zeros((k, n, b, d_v));
    let mut hz = Array4::zeros((k, n, b, d_v));
    let mut hy = Array4::zeros((k, n, b, d_v));
    let mut primal_res = Array3::zeros((k, n, b));
    let mut dual_res = Array3::zeros((k, n, b));
    let mut consistency_rms = Array2::zeros((k, b));

    let mut state = init_state(z_init);
    for ki in 0..k {
        let z_prev = state.z.clone();
        let (new_state, x_relaxed) =
            admm_step(&state, enc, geometry, x_solver, z_params, params.rho, params.alpha);
        state = new_state;

        hx.slice_mut(s![ki, .., .., ..]).assign(&state.x);
        hz.slice_mut(s![ki, .., .., ..]).assign(&state.z);
        hy.slice_mut(s![ki, .., .., ..]).assign(&state.y);
        // primal_res = ||x_relaxed - z|| ; dual_res = rho * ||z - z_prev||
        // (L2 norms over the trailing d_v axis, per agent/batch).
        for ni in 0..n {
            for bi in 0..b {
                let mut p_acc = 0.0 as Scalar;
                let mut d_acc = 0.0 as Scalar;
                for di in 0..d_v {
                    let pd = x_relaxed[[ni, bi, di]] - state.z[[ni, bi, di]];
                    p_acc += pd * pd;
                    let dd = state.z[[ni, bi, di]] - z_prev[[ni, bi, di]];
                    d_acc += dd * dd;
                }
                primal_res[[ki, ni, bi]] = p_acc.sqrt();
                dual_res[[ki, ni, bi]] = params.rho * d_acc.sqrt();
            }
        }
        consistency_rms
            .slice_mut(s![ki, ..])
            .assign(&geometry.consistency_rms(&state.z));
    }

    let history = AdmmHistory {
        x: hx,
        z: hz,
        y: hy,
        primal_res,
        dual_res,
        consistency_rms,
    };
    (state, history)
}
