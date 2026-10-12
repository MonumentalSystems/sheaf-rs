//! Local (x) and consensus (z) solvers. Ports `sheaf_admm.solvers`.

mod x_diag_prox;
mod x_simple;
mod z_cg;

pub use x_diag_prox::{diagonal_prox_solve, soft_threshold};
pub use x_simple::simple_solve;
pub use z_cg::{unrolled_cg_solve, UnrolledCgParams, ZMode, CG_DENOM_EPS};

use ndarray::{Array3, Array5};

use crate::tensor::NodeState;
use crate::Scalar;

/// Local-objective parameters emitted by the encoder, one variant per
/// `objective_mode` (replaces the Python stringly-typed dict — PLAN.md §3.2).
///
/// Contract: every `q_diag` here already includes the encoder head's
/// `softplus(raw) + q_epsilon` with `q_epsilon = 1e-4`; the floor lives in the
/// sheaf-nn heads (q_diag is input-dependent), not here and not in the exporter.
/// `lower` is hardcoded 0 for `NonNeg` and `L1Box` (matching encoder.py).
pub enum Objective {
    /// `x = (beta*h + rho*(z - y)) / (beta + rho)`.
    Simple {
        /// Strength of the quadratic tether to the encoder state `h`.
        beta: Scalar,
    },
    /// Unconstrained separable quadratic objective.
    Quadratic {
        /// Positive diagonal curvature, shaped `[N, B, d_v]`.
        q_diag: NodeState,
        /// Linear coefficient, shaped `[N, B, d_v]`.
        q: NodeState,
    },
    /// MNIST: scalar l1 from config.
    Lasso {
        /// Positive diagonal curvature, shaped `[N, B, d_v]`.
        q_diag: NodeState,
        /// Linear coefficient, shaped `[N, B, d_v]`.
        q: NodeState,
        /// Scalar L1 regularization strength.
        l1: Scalar,
    },
    /// Sudoku: lower = 0, no upper.
    NonNeg {
        /// Positive diagonal curvature, shaped `[N, B, d_v]`.
        q_diag: NodeState,
        /// Linear coefficient, shaped `[N, B, d_v]`.
        q: NodeState,
    },
    /// Maze: per-dim l1 and box upper, lower = 0.
    L1Box {
        /// Positive diagonal curvature, shaped `[N, B, d_v]`.
        q_diag: NodeState,
        /// Linear coefficient, shaped `[N, B, d_v]`.
        q: NodeState,
        /// Per-coordinate L1 strengths, shaped `[N, B, d_v]`.
        l1: NodeState,
        /// Per-coordinate box upper bounds, shaped `[N, B, d_v]`.
        upper: NodeState,
    },
}

/// Per-node LoRA factors, reshaped by the parent model to `[N, B, K, ., r]`.
pub struct LoraFactors {
    /// Output-side factors, shaped `[N, B, K, d_e, r]`.
    pub a: Array5<Scalar>,
    /// Input-side factors, shaped `[N, B, K, d_v, r]`.
    pub b: Array5<Scalar>,
    /// Optional per-node and direction gates, shaped `[N, B, K]`.
    pub gate: Option<Array3<Scalar>>,
    /// LoRA numerator in the effective scale `lora_alpha / rank`.
    pub lora_alpha: Scalar,
}

/// Everything the ADMM loop consumes from the encoder.
pub struct EncoderOutput {
    /// Consensus seed / tether `h`, `[N, B, d_v]`.
    pub h: NodeState,
    /// Local objective coefficients for the selected task.
    pub objective: Objective,
    /// Optional input-dependent factors used to construct LoRA geometry.
    pub lora: Option<LoraFactors>,
}
