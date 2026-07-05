//! Local (x) and consensus (z) solvers. Ports `sheaf_admm.solvers`.

mod x_diag_prox;
mod x_simple;
mod z_cg;

pub use x_diag_prox::diagonal_prox_solve;
pub use x_simple::simple_solve;
pub use z_cg::{unrolled_cg_solve, UnrolledCgParams, ZMode, CG_DENOM_EPS};

use ndarray::{Array3, Array5};

use crate::tensor::NodeState;

/// Local-objective parameters emitted by the encoder, one variant per
/// `objective_mode` (replaces the Python stringly-typed dict — PLAN.md §3.2).
///
/// Contract: every `q_diag` here already includes the encoder head's
/// `softplus(raw) + q_epsilon` with `q_epsilon = 1e-4`; the floor lives in the
/// sheaf-nn heads (q_diag is input-dependent), not here and not in the exporter.
/// `lower` is hardcoded 0 for `NonNeg` and `L1Box` (matching encoder.py).
pub enum Objective {
    /// `x = (beta*h + rho*(z - y)) / (beta + rho)`.
    Simple { beta: f32 },
    Quadratic { q_diag: NodeState, q: NodeState },
    /// MNIST: scalar l1 from config.
    Lasso { q_diag: NodeState, q: NodeState, l1: f32 },
    /// Sudoku: lower = 0, no upper.
    NonNeg { q_diag: NodeState, q: NodeState },
    /// Maze: per-dim l1 and box upper, lower = 0.
    L1Box {
        q_diag: NodeState,  // [N, B, d_v]
        q: NodeState,       // [N, B, d_v]
        l1: NodeState,      // [N, B, d_v] (softplus head, init 0.01)
        upper: NodeState,   // [N, B, d_v] (softplus head, init 1.0)
    },
}

/// Per-node LoRA factors, reshaped by the parent model to `[N, B, K, ., r]`.
pub struct LoraFactors {
    pub a: Array5<f32>,               // [N, B, K, d_e, r]
    pub b: Array5<f32>,               // [N, B, K, d_v, r]
    pub gate: Option<Array3<f32>>,    // [N, B, K]
    pub lora_alpha: f32,
}

/// Everything the ADMM loop consumes from the encoder.
pub struct EncoderOutput {
    /// Consensus seed / tether `h`, `[N, B, d_v]`.
    pub h: NodeState,
    pub objective: Objective,
    pub lora: Option<LoraFactors>,
}
