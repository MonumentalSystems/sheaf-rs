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

/// Crate-wide scalar float type. `f32` by default (the shipped/JAX-fp32 pin and
/// the only wasm build); `f64` under the `f64` feature — a reference-precision
/// build that exists SOLELY to disambiguate roundoff from bugs during parity
/// debugging (PLAN.md §3.4). The whole crate's arithmetic routes through this,
/// so downstream (sheaf-nn/io/demo/web) keeps its default `f32` unchanged.
#[cfg(not(feature = "f64"))]
pub type Scalar = f32;
/// See [`Scalar`] — reference-precision alias under `--features f64`.
#[cfg(feature = "f64")]
pub type Scalar = f64;

pub mod admm;
pub mod geometry;
pub mod graph;
pub(crate) mod par;
pub mod solvers;
pub mod tensor;

pub use admm::{run_admm, run_admm_history, AdmmHistory, AdmmParams, AdmmState, XSolverKind};
pub use geometry::{FixedGeometry, LoraGeometry, SheafGeometry};
pub use graph::AgentGraph;
pub use solvers::{EncoderOutput, LoraFactors, Objective, UnrolledCgParams, ZMode};
