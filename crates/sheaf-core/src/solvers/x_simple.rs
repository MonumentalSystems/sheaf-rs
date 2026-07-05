//! Simple (tethered-average) x-update. Ports `x_solvers/simple.py`.
//!
//! `x = (beta * h + rho * (z - y)) / (beta + rho)` — a convex combination of
//! the encoder tether `h` and the consensus target. No shipped maze config
//! uses it, but it is trivial and pins a §5.1 property test.

use crate::tensor::NodeState;
use crate::Scalar;

pub fn simple_solve(z: &NodeState, y: &NodeState, rho: Scalar, h: &NodeState, beta: Scalar) -> NodeState {
    assert_eq!(h.dim(), z.dim(), "h must be [N, B, d_v]");
    let denom = beta + rho;
    let mut x = NodeState::zeros(z.dim());
    ndarray::azip!((x in &mut x, &z in z, &y in y, &h in h) {
        *x = (beta * h + rho * (z - y)) / denom;
    });
    x
}
