//! Unrolled-CG z-solver tests (PLAN.md §5.1): prox-mode relative residual at
//! T=30 on N*d_v = 24 dims < 1e-4; project mode reduces the sheaf residual
//! below 10% of the input's; the 1e-8 denominator guard keeps degenerate
//! (b = 0) systems NaN-free; the deliberately non-idempotent projector.

mod common;

use common::*;

use std::sync::Arc;

use sheaf_core::geometry::{FixedGeometry, SheafGeometry};
use sheaf_core::graph::AgentGraph;
use sheaf_core::solvers::{unrolled_cg_solve, UnrolledCgParams, ZMode};
use sheaf_core::tensor::NodeState;

const N: usize = 8;
const B: usize = 2;
const D_V: usize = 3; // N * d_v = 24 per batch system
const D_E: usize = 2;
const RHO: f32 = 0.25;
const GAMMA: f32 = 5.0;

/// Ring + chords over 8 nodes, random fixed maps.
fn setup(seed: u64) -> (FixedGeometry, NodeState, NodeState) {
    let mut rng = Rng::new(seed);
    let mut edges: Vec<[u32; 2]> = (0..N as u32).map(|i| [i, (i + 1) % N as u32]).collect();
    edges.push([0, 4]);
    edges.push([2, 6]);
    edges.push([1, 5]);
    let e = edges.len();
    let graph = Arc::new(AgentGraph::from_edges(edges, N));
    let maps = random_maps(&mut rng, e, D_E, D_V, 0.6);
    let geo = FixedGeometry::new(graph, maps);
    let z_target = rng.array3((N, B, D_V));
    let z_prev = rng.array3((N, B, D_V));
    (geo, z_target, z_prev)
}

fn lap(geo: &dyn SheafGeometry, z: &NodeState) -> NodeState {
    let mut out = NodeState::zeros(z.dim());
    geo.laplacian_apply(z, &mut out);
    out
}

/// Per-batch L2 norm over (N, d_v).
fn bnorm(a: &NodeState) -> Vec<f32> {
    let (n, b, d) = a.dim();
    (0..b)
        .map(|bi| {
            let mut acc = 0.0f32;
            for ni in 0..n {
                for di in 0..d {
                    acc += a[[ni, bi, di]] * a[[ni, bi, di]];
                }
            }
            acc.sqrt()
        })
        .collect()
}

#[test]
fn prox_mode_relative_residual_at_t30() {
    let (geo, z_target, z_prev) = setup(53);
    let params = UnrolledCgParams {
        mode: ZMode::Prox,
        gamma: GAMMA,
        num_iters: 30,
        tikhonov_eps: 1e-5,
    };
    let z = unrolled_cg_solve(&z_target, &z_prev, &geo, &params, RHO);
    // Residual of (gamma L + rho I) z = rho z_target.
    let mut az = lap(&geo, &z);
    az.zip_mut_with(&z, |a, &zi| *a = GAMMA * *a + RHO * zi);
    let b = z_target.mapv(|v| RHO * v);
    let resid = &az - &b;
    let rn = bnorm(&resid);
    let bn = bnorm(&b);
    for (bi, (r, n_b)) in rn.iter().zip(bn.iter()).enumerate() {
        let rel = r / n_b;
        assert!(rel < 1e-4, "batch {bi}: relative residual {rel} >= 1e-4");
    }
}

#[test]
fn project_mode_reduces_sheaf_residual_below_10pct() {
    let (geo, z_target, z_prev) = setup(59);
    let params = UnrolledCgParams {
        mode: ZMode::Project,
        gamma: GAMMA, // unused in project mode
        num_iters: 30,
        tikhonov_eps: 1e-5,
    };
    let z = unrolled_cg_solve(&z_target, &z_prev, &geo, &params, RHO);
    let r_in = geo.edge_residuals(&z_target);
    let r_out = geo.edge_residuals(&z);
    let in_sq: f32 = r_in.iter().map(|&v| v * v).sum();
    let out_sq: f32 = r_out.iter().map(|&v| v * v).sum();
    assert!(
        out_sq < 0.1 * in_sq,
        "project mode: residual energy {out_sq} not < 10% of input {in_sq}"
    );
}

#[test]
fn cg_denominator_guard_no_nan_on_zero_system() {
    // b = rho * 0 = 0 and x0 = 0: every pTAp and rTr is 0; the exact 1e-8
    // guard turns 0/0 into 0 and the iterate stays identically zero.
    let (geo, _zt, _zp) = setup(61);
    let z0 = NodeState::zeros((N, B, D_V));
    let params = UnrolledCgParams {
        mode: ZMode::Prox,
        gamma: GAMMA,
        num_iters: 5,
        tikhonov_eps: 1e-5,
    };
    let z = unrolled_cg_solve(&z0, &z0, &geo, &params, RHO);
    for &v in z.iter() {
        assert_eq!(v, 0.0, "zero system must stay exactly zero (no NaN)");
    }
    // Project mode likewise.
    let params_p = UnrolledCgParams { mode: ZMode::Project, ..params };
    let z = unrolled_cg_solve(&z0, &z0, &geo, &params_p, RHO);
    for &v in z.iter() {
        assert!(v.is_finite(), "project mode NaN on zero system");
        assert_eq!(v, 0.0);
    }
}

#[test]
fn project_mode_is_not_idempotent_by_design() {
    // The warm start w0 = z_target - z_prev keeps kernel components: applying
    // the "projector" twice (with the same z_prev) need not be a no-op.
    // This pins the DELIBERATE non-idempotence — do not "fix" it.
    let (geo, z_target, z_prev) = setup(67);
    let params = UnrolledCgParams {
        mode: ZMode::Project,
        gamma: GAMMA,
        num_iters: 5, // shipped T: under-solved, warm-start-visible
        tikhonov_eps: 1e-5,
    };
    let z1 = unrolled_cg_solve(&z_target, &z_prev, &geo, &params, RHO);
    let z2 = unrolled_cg_solve(&z1, &z_prev, &geo, &params, RHO);
    let diff: f32 = z1.iter().zip(z2.iter()).map(|(&a, &b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-4,
        "project mode unexpectedly idempotent (diff {diff}); the legacy warm start semantics changed"
    );
}

#[test]
fn prox_mode_more_iters_reduce_residual() {
    let (geo, z_target, z_prev) = setup(71);
    let resid_at = |t: usize| -> f32 {
        let params = UnrolledCgParams {
            mode: ZMode::Prox,
            gamma: GAMMA,
            num_iters: t,
            tikhonov_eps: 1e-5,
        };
        let z = unrolled_cg_solve(&z_target, &z_prev, &geo, &params, RHO);
        let mut az = lap(&geo, &z);
        az.zip_mut_with(&z, |a, &zi| *a = GAMMA * *a + RHO * zi);
        let b = z_target.mapv(|v| RHO * v);
        bnorm(&(&az - &b)).iter().sum()
    };
    let r5 = resid_at(5);
    let r20 = resid_at(20);
    assert!(
        r20 < r5,
        "CG residual did not shrink with more iterations: T=5 {r5}, T=20 {r20}"
    );
}
