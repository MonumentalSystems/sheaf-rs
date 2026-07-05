//! Closed-form diagonal proximal x-update. Ports `x_solvers/diagonal_prox.py`.
//!
//! One fused elementwise loop over `[N, B, d_v]`:
//! ```text
//! v = z - y
//! a = D + l2 + rho          (l2 = 0 for all shipped configs)
//! t = (rho * v - q) / a
//! x = clip( soft_threshold(t, l1 / a), lo, hi )
//! soft_threshold(x, th) = sign(x) * max(|x| - th, 0)
//! ```
//! Variant selection is purely by what the `Objective` carries:
//! Quadratic (l1=0, no box), Lasso (scalar l1), NonNeg (lo=0), L1Box (per-dim
//! l1, lo=0, per-dim hi — the maze path). Exact — no inner iterations.

use crate::solvers::Objective;
use crate::tensor::NodeState;
use crate::Scalar;

/// `soft_threshold(x, th) = sign(x) * max(|x| - th, 0)`.
///
/// Sign follows `jnp.sign` exactly (`sign(0) = 0`), not `f32::signum`.
#[inline]
pub fn soft_threshold(x: Scalar, threshold: Scalar) -> Scalar {
    let sign = if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    };
    sign * (x.abs() - threshold).max(0.0)
}

/// Solve the diagonal-prox x-update at `v = z - y`.
///
/// `rho` is scalar here (the maze config bakes the learned scalar at export);
/// a `[N, B]` rho would broadcast as `[N, B, 1]` — add if/when a config needs it.
/// Panics if called with `Objective::Simple` (use `simple_solve`).
pub fn diagonal_prox_solve(
    z: &NodeState,
    y: &NodeState,
    rho: Scalar,
    objective: &Objective,
) -> NodeState {
    // (q_diag, q, scalar l1, per-dim l1, lower, per-dim upper).
    // l2 = 0 for every shipped config; lower is a scalar (0 or -inf).
    let (q_diag, q, l1_scalar, l1, lower, upper): (
        &NodeState,
        &NodeState,
        Scalar,
        Option<&NodeState>,
        Scalar,
        Option<&NodeState>,
    ) = match objective {
        Objective::Simple { .. } => {
            panic!("diagonal_prox_solve called with Objective::Simple; use simple_solve")
        }
        Objective::Quadratic { q_diag, q } => {
            (q_diag, q, 0.0, None, Scalar::NEG_INFINITY, None)
        }
        Objective::Lasso { q_diag, q, l1 } => (q_diag, q, *l1, None, Scalar::NEG_INFINITY, None),
        Objective::NonNeg { q_diag, q } => (q_diag, q, 0.0, None, 0.0, None),
        Objective::L1Box { q_diag, q, l1, upper } => (q_diag, q, 0.0, Some(l1), 0.0, Some(upper)),
    };

    let dim = z.dim();
    assert_eq!(q_diag.dim(), dim, "q_diag must be [N, B, d_v]");
    assert_eq!(q.dim(), dim, "q must be [N, B, d_v]");

    let (n, b, d) = dim;
    let mut x = NodeState::zeros(dim);
    for ni in 0..n {
        for bi in 0..b {
            for di in 0..d {
                let idx = [ni, bi, di];
                let v = z[idx] - y[idx];
                let a = q_diag[idx] + rho; // + l2 (= 0)
                let t = (rho * v - q[idx]) / a;
                let l1_here = l1.map_or(l1_scalar, |m| m[idx]);
                let xt = soft_threshold(t, l1_here / a);
                let hi = upper.map_or(Scalar::INFINITY, |m| m[idx]);
                // jnp.clip: min(max(x, lower), upper) — clip AFTER the
                // soft-threshold (ordering is load-bearing, see tests).
                x[idx] = xt.max(lower).min(hi);
            }
        }
    }
    x
}
