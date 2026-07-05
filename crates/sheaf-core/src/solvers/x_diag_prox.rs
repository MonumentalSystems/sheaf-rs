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

/// `soft_threshold(x, th) = sign(x) * max(|x| - th, 0)`.
#[inline]
pub fn soft_threshold(x: f32, threshold: f32) -> f32 {
    todo!()
}

/// Solve the diagonal-prox x-update at `v = z - y`.
///
/// `rho` is scalar here (the maze config bakes the learned scalar at export);
/// a `[N, B]` rho would broadcast as `[N, B, 1]` — add if/when a config needs it.
/// Panics if called with `Objective::Simple` (use `simple_solve`).
pub fn diagonal_prox_solve(
    z: &NodeState,
    y: &NodeState,
    rho: f32,
    objective: &Objective,
) -> NodeState {
    todo!()
}
