//! ADMM driver tests (PLAN.md §5.1): x_window[last] == state.x, loss_window
//! clamping, run_admm == run_admm_history final state, residual definitions,
//! and a hand-rolled reference step loop (including alpha != 1.0 relaxation).

mod common;

use common::*;

use ndarray::{s, Array3};
use sheaf_core::admm::{run_admm, run_admm_history, AdmmParams, XSolverKind};
use sheaf_core::geometry::{FixedGeometry, SheafGeometry};
use sheaf_core::solvers::{
    diagonal_prox_solve, unrolled_cg_solve, EncoderOutput, Objective, UnrolledCgParams, ZMode,
};
use sheaf_core::tensor::NodeState;

const N: usize = 4;
const B: usize = 2;
const D_V: usize = 4;
const D_E: usize = 2;
const RHO: f32 = 0.25;
const GAMMA: f32 = 5.0;

/// Maze-flavored mini setup: tiny graph, L1Box objective, prox-mode CG T=5.
fn setup(seed: u64) -> (EncoderOutput, FixedGeometry, UnrolledCgParams, NodeState) {
    let mut rng = Rng::new(seed);
    let graph = tiny_graph();
    let e = graph.num_edges();
    let maps = random_maps(&mut rng, e, D_E, D_V, 0.6);
    let geo = FixedGeometry::new(graph, maps);

    let h = rng.array3((N, B, D_V));
    let q_diag = Array3::from_shape_fn((N, B, D_V), |_| rng.f32().abs() + 0.1 + 1e-4);
    let q = rng.array3((N, B, D_V));
    let l1 = Array3::from_shape_fn((N, B, D_V), |_| rng.f32().abs() * 0.05);
    let upper = Array3::from_elem((N, B, D_V), 1.0);
    let enc = EncoderOutput {
        h: h.clone(),
        objective: Objective::L1Box { q_diag, q, l1, upper },
        lora: None,
    };
    let z_params = UnrolledCgParams {
        mode: ZMode::Prox,
        gamma: GAMMA,
        num_iters: 5,
        tikhonov_eps: 1e-5,
    };
    (enc, geo, z_params, h)
}

fn params(k: usize, alpha: f32) -> AdmmParams {
    AdmmParams { rho: RHO, alpha, gamma: GAMMA, k }
}

#[test]
fn x_window_last_equals_final_x() {
    let (enc, geo, z_params, z_init) = setup(101);
    let (state, x_window) =
        run_admm(&enc, &geo, XSolverKind::DiagonalProx, &z_params, &params(6, 1.0), &z_init, 3);
    assert_eq!(x_window.len(), 3);
    assert_close(x_window.last().unwrap(), &state.x, 0.0, 0.0, "x_window[last] == state.x");
}

#[test]
fn loss_window_clamps_to_k() {
    let (enc, geo, z_params, z_init) = setup(103);
    let k = 4;
    let (state, x_window) = run_admm(
        &enc,
        &geo,
        XSolverKind::DiagonalProx,
        &z_params,
        &params(k, 1.0),
        &z_init,
        10, // > K, must clamp
    );
    assert_eq!(x_window.len(), k, "loss_window > K clamps to K");
    assert_close(x_window.last().unwrap(), &state.x, 0.0, 0.0, "clamped window last");
    // Oldest-first: x_window[i] must equal history.x[i].
    let (_, hist) = run_admm_history(
        &enc,
        &geo,
        XSolverKind::DiagonalProx,
        &z_params,
        &params(k, 1.0),
        &z_init,
    );
    for (i, xw) in x_window.iter().enumerate() {
        let hx = hist.x.slice(s![i, .., .., ..]).to_owned();
        assert_close(xw, &hx, 0.0, 0.0, "x_window oldest-first matches history");
    }
}

#[test]
fn history_final_state_matches_run_admm() {
    let (enc, geo, z_params, z_init) = setup(107);
    let p = params(8, 1.0);
    let (state, _) =
        run_admm(&enc, &geo, XSolverKind::DiagonalProx, &z_params, &p, &z_init, 1);
    let (state_h, hist) =
        run_admm_history(&enc, &geo, XSolverKind::DiagonalProx, &z_params, &p, &z_init);
    assert_close(&state.x, &state_h.x, 0.0, 0.0, "final x");
    assert_close(&state.z, &state_h.z, 0.0, 0.0, "final z");
    assert_close(&state.y, &state_h.y, 0.0, 0.0, "final y");
    // Last history snapshot is the final state.
    let hx = hist.x.slice(s![7, .., .., ..]).to_owned();
    assert_close(&hx, &state.x, 0.0, 0.0, "history.x[K-1] == final x");
}

#[test]
fn history_residual_definitions() {
    // With alpha == 1.0, x_relaxed == x, so:
    //   primal_res[k] == ||x[k] - z[k]||, dual_res[k] == rho * ||z[k] - z[k-1]||
    // (z[-1] = z_init), and consistency_rms[k] == geometry.consistency_rms(z[k]).
    let (enc, geo, z_params, z_init) = setup(109);
    let k = 5;
    let (_, hist) = run_admm_history(
        &enc,
        &geo,
        XSolverKind::DiagonalProx,
        &z_params,
        &params(k, 1.0),
        &z_init,
    );
    for ki in 0..k {
        let xk = hist.x.slice(s![ki, .., .., ..]);
        let zk = hist.z.slice(s![ki, .., .., ..]);
        let z_prev = if ki == 0 {
            z_init.view()
        } else {
            hist.z.slice(s![ki - 1, .., .., ..])
        };
        for ni in 0..N {
            for bi in 0..B {
                let mut p_acc = 0.0f32;
                let mut d_acc = 0.0f32;
                for di in 0..D_V {
                    let pd = xk[[ni, bi, di]] - zk[[ni, bi, di]];
                    p_acc += pd * pd;
                    let dd = zk[[ni, bi, di]] - z_prev[[ni, bi, di]];
                    d_acc += dd * dd;
                }
                let p_expect = p_acc.sqrt();
                let d_expect = RHO * d_acc.sqrt();
                assert!(
                    (hist.primal_res[[ki, ni, bi]] - p_expect).abs() <= 1e-6,
                    "primal_res[{ki},{ni},{bi}]"
                );
                assert!(
                    (hist.dual_res[[ki, ni, bi]] - d_expect).abs() <= 1e-6,
                    "dual_res[{ki},{ni},{bi}]"
                );
            }
        }
        let crms = geo.consistency_rms(&zk.to_owned());
        for bi in 0..B {
            assert!(
                (hist.consistency_rms[[ki, bi]] - crms[bi]).abs() <= 1e-6,
                "consistency_rms[{ki},{bi}]"
            );
        }
    }
}

/// Hand-rolled reference of the ADMM recurrence (admm.py step), including the
/// alpha != 1.0 relaxation blend, built from the public solver functions.
fn reference_run(
    enc: &EncoderOutput,
    geo: &dyn SheafGeometry,
    z_params: &UnrolledCgParams,
    z_init: &NodeState,
    k: usize,
    alpha: f32,
) -> (NodeState, NodeState, NodeState) {
    let mut x = z_init.clone();
    let mut z = z_init.clone();
    let mut y = NodeState::zeros(z_init.dim());
    for _ in 0..k {
        let z_prev = z.clone();
        x = diagonal_prox_solve(&z, &y, RHO, &enc.objective);
        let x_relaxed = if alpha == 1.0 {
            x.clone()
        } else {
            let mut xr = x.clone();
            xr.zip_mut_with(&z_prev, |v, &zp| *v = alpha * *v + (1.0 - alpha) * zp);
            xr
        };
        let z_target = &x_relaxed + &y;
        z = unrolled_cg_solve(&z_target, &z_prev, geo, z_params, RHO);
        y = &y + &(&x_relaxed - &z);
    }
    (x, z, y)
}

#[test]
fn matches_reference_loop_alpha_1() {
    let (enc, geo, z_params, z_init) = setup(113);
    let k = 6;
    let (state, _) =
        run_admm(&enc, &geo, XSolverKind::DiagonalProx, &z_params, &params(k, 1.0), &z_init, 1);
    let (rx, rz, ry) = reference_run(&enc, &geo, &z_params, &z_init, k, 1.0);
    assert_close(&state.x, &rx, 0.0, 0.0, "x vs reference (alpha=1)");
    assert_close(&state.z, &rz, 0.0, 0.0, "z vs reference (alpha=1)");
    assert_close(&state.y, &ry, 0.0, 0.0, "y vs reference (alpha=1)");
}

#[test]
fn matches_reference_loop_alpha_relaxed() {
    let (enc, geo, z_params, z_init) = setup(127);
    let k = 6;
    let alpha = 0.5;
    let (state, _) = run_admm(
        &enc,
        &geo,
        XSolverKind::DiagonalProx,
        &z_params,
        &params(k, alpha),
        &z_init,
        1,
    );
    let (rx, rz, ry) = reference_run(&enc, &geo, &z_params, &z_init, k, alpha);
    assert_close(&state.x, &rx, 0.0, 0.0, "x vs reference (alpha=0.5)");
    assert_close(&state.z, &rz, 0.0, 0.0, "z vs reference (alpha=0.5)");
    assert_close(&state.y, &ry, 0.0, 0.0, "y vs reference (alpha=0.5)");
    // And the blend must actually change the trajectory vs alpha = 1.
    let (state1, _) =
        run_admm(&enc, &geo, XSolverKind::DiagonalProx, &z_params, &params(k, 1.0), &z_init, 1);
    let diff: f32 = state.z.iter().zip(state1.z.iter()).map(|(&a, &b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "alpha=0.5 should differ from alpha=1.0");
}

#[test]
fn simple_x_solver_path_runs() {
    let mut rng = Rng::new(131);
    let graph = tiny_graph();
    let e = graph.num_edges();
    let maps = random_maps(&mut rng, e, D_E, D_V, 0.6);
    let geo = FixedGeometry::new(graph, maps);
    let h = rng.array3((N, B, D_V));
    let enc = EncoderOutput {
        h: h.clone(),
        objective: Objective::Simple { beta: 0.8 },
        lora: None,
    };
    let z_params = UnrolledCgParams {
        mode: ZMode::Prox,
        gamma: GAMMA,
        num_iters: 5,
        tikhonov_eps: 1e-5,
    };
    let (state, x_window) =
        run_admm(&enc, &geo, XSolverKind::Simple, &z_params, &params(4, 1.0), &h, 1);
    assert_eq!(x_window.len(), 1);
    assert!(state.x.iter().all(|v| v.is_finite()));
    assert!(state.z.iter().all(|v| v.is_finite()));
}

#[test]
fn prox_z_update_never_increases_sheaf_energy_vs_target() {
    // Prox-mode CG inits at z_target and monotonically decreases the CG
    // quadratic phi(z) = gamma*E(z) + rho/2 ||z - z_target||^2, so every
    // iteration satisfies E(z_k) <= E(z_target_k). With alpha == 1 the dual
    // update gives z_target_k = z_k + y_k, recoverable from the history.
    let (enc, geo, z_params, z_init) = setup(137);
    let k = 20;
    let (_, hist) = run_admm_history(
        &enc,
        &geo,
        XSolverKind::DiagonalProx,
        &z_params,
        &params(k, 1.0),
        &z_init,
    );
    for ki in 0..k {
        let zk = hist.z.slice(s![ki, .., .., ..]).to_owned();
        let z_target = &zk + &hist.y.slice(s![ki, .., .., ..]);
        let e_z = geo.energy(&zk);
        let e_t = geo.energy(&z_target);
        assert!(
            e_z <= e_t * (1.0 + 1e-4) + 1e-6,
            "iter {ki}: E(z) = {e_z} > E(z_target) = {e_t}"
        );
    }
}
