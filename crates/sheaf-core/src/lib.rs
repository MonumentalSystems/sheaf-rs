//! Sheaf-ADMM coordination core.
//!
//! Pure `ndarray` port of the JAX reference (`sheaf_admm.admm`,
//! `sheaf_admm.geometry`, `sheaf_admm.solvers`). Everything operates on batched
//! node states `[N, B, d_v]`; the Laplacian never mixes batch elements.
//!
//! Numerics contract (PLAN.md §3.4 — pinned, non-negotiable):
//! - all matmuls true fp32;
//! - CG denominator guard exactly `1e-8`;
//! - project-mode Tikhonov `1e-5`;
//! - `consistency_rms` epsilon `1e-6` UNDER the sqrt;
//! - `q_epsilon = 1e-4` added in the objective heads (sheaf-nn side);
//! - `alpha == 1.0` skips the relaxation blend entirely (bitwise fast path).

// Scaffold phase: stub bodies are `todo!()`. Remove this allow as modules land.
#![allow(unused_variables, dead_code)]

pub mod admm;
pub mod geometry;
pub mod graph;
pub mod solvers;
pub mod tensor;

pub use admm::{run_admm, run_admm_history, AdmmHistory, AdmmParams, AdmmState};
pub use geometry::{FixedGeometry, LoraGeometry, SheafGeometry};
pub use graph::AgentGraph;
pub use solvers::{EncoderOutput, LoraFactors, Objective};
