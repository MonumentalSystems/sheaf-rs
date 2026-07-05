//! Closed-form x-solver identity tests (PLAN.md §5.1, atol 1e-5):
//! pure-quadratic stationarity, lasso subgradient optimality, clip-after
//! ordering, non-negativity, Simple's convex combination, soft_threshold kinks.

mod common;

use common::*;

use ndarray::Array3;
use sheaf_core::solvers::{diagonal_prox_solve, simple_solve, soft_threshold, Objective};
use sheaf_core::tensor::NodeState;
use sheaf_core::Scalar;

const N: usize = 3;
const B: usize = 2;
const D: usize = 4;
const RHO: Scalar = 0.25;

/// Strictly positive diagonal curvature (mimics softplus + q_epsilon heads).
fn positive_diag(rng: &mut Rng) -> NodeState {
    Array3::from_shape_fn((N, B, D), |_| rng.scalar().abs() + 0.1 + 1e-4)
}

#[test]
fn soft_threshold_kinks() {
    // On the dead-zone boundary and around it.
    assert_eq!(soft_threshold(0.5, 0.5), 0.0);
    assert_eq!(soft_threshold(-0.5, 0.5), 0.0);
    assert!((soft_threshold(0.6, 0.5) - 0.1).abs() < 1e-7);
    assert!((soft_threshold(-0.6, 0.5) + 0.1).abs() < 1e-7);
    assert_eq!(soft_threshold(0.0, 0.0), 0.0);
    assert_eq!(soft_threshold(0.0, 0.3), 0.0);
    assert_eq!(soft_threshold(2.0, 0.0), 2.0);
    assert_eq!(soft_threshold(-2.0, 0.0), -2.0);
}

#[test]
fn quadratic_stationarity() {
    // x minimizes 1/2 d x^2 + q x + rho/2 (x - v)^2, v = z - y:
    //   q_diag * x + q + rho * (x - (z - y)) == 0   (atol 1e-5)
    let mut rng = Rng::new(31);
    let q_diag = positive_diag(&mut rng);
    let q = rng.array3((N, B, D));
    let z = rng.array3((N, B, D));
    let y = rng.array3((N, B, D));
    let obj = Objective::Quadratic { q_diag: q_diag.clone(), q: q.clone() };
    let x = diagonal_prox_solve(&z, &y, RHO, &obj);
    for (idx, &xi) in x.indexed_iter() {
        let (n, b, d) = idx;
        let v = z[[n, b, d]] - y[[n, b, d]];
        let resid = q_diag[[n, b, d]] * xi + q[[n, b, d]] + RHO * (xi - v);
        assert!(resid.abs() <= 1e-5, "stationarity violated at {idx:?}: {resid}");
    }
}

#[test]
fn lasso_subgradient_optimality() {
    // f(x) = 1/2 d x^2 + q x + l1 |x| + rho/2 (x - v)^2.
    // Where x != 0: d x + q + l1 sign(x) + rho (x - v) == 0.
    // Where x == 0: |q - rho v| <= l1.
    let mut rng = Rng::new(37);
    let q_diag = positive_diag(&mut rng);
    let q = rng.array3((N, B, D));
    let z = rng.array3((N, B, D));
    let y = rng.array3((N, B, D));
    let l1: Scalar = 0.3; // large enough that some coordinates land at 0
    let obj = Objective::Lasso { q_diag: q_diag.clone(), q: q.clone(), l1 };
    let x = diagonal_prox_solve(&z, &y, RHO, &obj);
    let mut zeros = 0;
    let mut nonzeros = 0;
    for (idx, &xi) in x.indexed_iter() {
        let (n, b, d) = idx;
        let v = z[[n, b, d]] - y[[n, b, d]];
        if xi != 0.0 {
            nonzeros += 1;
            let resid = q_diag[[n, b, d]] * xi + q[[n, b, d]] + l1 * xi.signum() + RHO * (xi - v);
            assert!(resid.abs() <= 1e-5, "nonzero stationarity at {idx:?}: {resid}");
        } else {
            zeros += 1;
            let sub = (q[[n, b, d]] - RHO * v).abs();
            assert!(sub <= l1 + 1e-5, "zero subgradient at {idx:?}: |q - rho v| = {sub} > {l1}");
        }
    }
    assert!(zeros > 0, "test vacuous: no coordinate hit the L1 dead zone");
    assert!(nonzeros > 0, "test vacuous: every coordinate hit the dead zone");
}

#[test]
fn l1box_clip_after_soft_threshold_ordering() {
    // Hand-built case where clip-then-threshold differs from
    // threshold-then-clip: q_diag = 1, q = 0, rho = 1 => a = 2, t = v / 2.
    // v = 4 => t = 2; l1 = 1 => threshold t - 0.5 = 1.5; upper = 1 => x = 1.
    // (Clip-before-threshold would give soft_threshold(1, 0.5) = 0.5.)
    let shape = (1, 1, 1);
    let q_diag = Array3::from_elem(shape, 1.0);
    let q = Array3::zeros(shape);
    let l1 = Array3::from_elem(shape, 1.0);
    let upper = Array3::from_elem(shape, 1.0);
    let z = Array3::from_elem(shape, 4.0);
    let y = Array3::zeros(shape);
    let obj = Objective::L1Box { q_diag, q, l1, upper };
    let x = diagonal_prox_solve(&z, &y, 1.0, &obj);
    assert_eq!(x[[0, 0, 0]], 1.0, "clip must run AFTER the soft-threshold");
}

#[test]
fn l1box_respects_box_and_stationarity_in_interior() {
    let mut rng = Rng::new(41);
    let q_diag = positive_diag(&mut rng);
    let q = rng.array3((N, B, D));
    let z = rng.array3((N, B, D));
    let y = rng.array3((N, B, D));
    let l1 = Array3::from_shape_fn((N, B, D), |_| rng.scalar().abs() * 0.2);
    let upper = Array3::from_shape_fn((N, B, D), |_| rng.scalar().abs() * 0.8 + 0.05);
    let obj = Objective::L1Box {
        q_diag: q_diag.clone(),
        q: q.clone(),
        l1: l1.clone(),
        upper: upper.clone(),
    };
    let x = diagonal_prox_solve(&z, &y, RHO, &obj);
    for (idx, &xi) in x.indexed_iter() {
        let (n, b, d) = idx;
        assert!(xi >= 0.0, "lower bound 0 violated at {idx:?}: {xi}");
        assert!(xi <= upper[[n, b, d]], "upper bound violated at {idx:?}");
        // Strict interior + off the L1 kink => plain stationarity holds.
        if xi > 1e-6 && xi < upper[[n, b, d]] - 1e-6 {
            let v = z[[n, b, d]] - y[[n, b, d]];
            let resid =
                q_diag[[n, b, d]] * xi + q[[n, b, d]] + l1[[n, b, d]] + RHO * (xi - v);
            assert!(resid.abs() <= 1e-5, "interior stationarity at {idx:?}: {resid}");
        }
    }
}

#[test]
fn non_negative_projects_to_zero() {
    let mut rng = Rng::new(43);
    let q_diag = positive_diag(&mut rng);
    let q = rng.array3((N, B, D));
    let z = rng.array3((N, B, D));
    let y = rng.array3((N, B, D));
    let obj = Objective::NonNeg { q_diag: q_diag.clone(), q: q.clone() };
    let x = diagonal_prox_solve(&z, &y, RHO, &obj);
    let mut clamped = 0;
    for (idx, &xi) in x.indexed_iter() {
        let (n, b, d) = idx;
        assert!(xi >= 0.0, "NonNeg violated at {idx:?}");
        let v = z[[n, b, d]] - y[[n, b, d]];
        let unconstrained = (RHO * v - q[[n, b, d]]) / (q_diag[[n, b, d]] + RHO);
        if unconstrained < 0.0 {
            clamped += 1;
            assert_eq!(xi, 0.0, "negative unconstrained solution must clamp to 0");
        } else {
            assert!((xi - unconstrained).abs() <= 1e-6);
        }
    }
    assert!(clamped > 0, "test vacuous: nothing clamped");
}

#[test]
fn simple_solver_identity_and_limits() {
    let mut rng = Rng::new(47);
    let z = rng.array3((N, B, D));
    let y = rng.array3((N, B, D));
    let h = rng.array3((N, B, D));
    let beta: Scalar = 0.7;
    let x = simple_solve(&z, &y, RHO, &h, beta);
    // (beta + rho) x == beta h + rho (z - y)
    for (idx, &xi) in x.indexed_iter() {
        let (n, b, d) = idx;
        let lhs = (beta + RHO) * xi;
        let rhs = beta * h[[n, b, d]] + RHO * (z[[n, b, d]] - y[[n, b, d]]);
        assert!((lhs - rhs).abs() <= 1e-5, "simple identity at {idx:?}");
        // Convex combination: x lies between h and v.
        let v = z[[n, b, d]] - y[[n, b, d]];
        let (lo, hi) = if h[[n, b, d]] <= v { (h[[n, b, d]], v) } else { (v, h[[n, b, d]]) };
        assert!(xi >= lo - 1e-6 && xi <= hi + 1e-6, "not a convex combination at {idx:?}");
    }
    // beta = 0 => x = z - y exactly.
    let x0 = simple_solve(&z, &y, RHO, &h, 0.0);
    for (idx, &xi) in x0.indexed_iter() {
        let (n, b, d) = idx;
        assert!((xi - (z[[n, b, d]] - y[[n, b, d]])).abs() <= 1e-6);
    }
}

#[test]
#[should_panic(expected = "use simple_solve")]
fn diagonal_prox_rejects_simple_objective() {
    let z = Array3::zeros((1, 1, 1));
    let y = Array3::zeros((1, 1, 1));
    let obj = Objective::Simple { beta: 1.0 };
    let _ = diagonal_prox_solve(&z, &y, RHO, &obj);
}
