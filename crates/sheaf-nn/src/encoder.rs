//! MLPEncoderV2 (`arch = "mlp_v2"`, the maze encoder). Ports `models/encoder.py`.
//!
//! Trunk: `flatten -> RMSNorm(input_norm) -> Dense(dense, 256) -> gelu_tanh`,
//! then `comm_head` (`comm_dense` -> LayerNorm `comm_norm`) produces `h`, the
//! objective heads read the **trunk** features (not `h`), and the LoRA heads
//! read `LayerNorm(lora_pre_ln)(trunk)`.
//!
//! l1box_diag heads (all read trunk feats, all `[B', d_v]`):
//! - `q_diag = softplus(q_diag_dense(feats)) + 1e-4`   (q_epsilon floor here!)
//! - `q      = q_dense(feats)`
//! - `l1     = softplus(l1_weight_dense(feats))`
//! - `upper  = softplus(upper_bound_dense(feats))`, `lower = 0` hardcoded.
//!
//! LoRA heads: `A = lora_A_dense(feats_ln).reshape(B', K, d_e, r)`,
//! `B = lora_B_dense(feats_ln).reshape(B', K, d_v, r)`; no gate (shipped
//! `lora_use_gate = false`).

use ndarray::Array5;

use sheaf_core::solvers::EncoderOutput;

use crate::layers::{Dense, LayerNorm, RmsNorm};

/// Weights of the maze encoder (loaded by sheaf-io from safetensors; key map
/// in goldens/CONTRACT.md under `MLPEncoderV2_0/...`).
#[derive(Debug, Clone)]
pub struct MlpEncoderV2Params {
    pub input_norm: RmsNorm,          // input_norm/scale [54]
    pub dense: Dense,                 // dense/{kernel [54,256], bias}
    pub comm_dense: Dense,            // comm_head/comm_dense
    pub comm_norm: LayerNorm,         // comm_head/comm_norm
    pub q_diag_dense: Dense,          // objective_heads/q_diag_dense
    pub q_dense: Dense,               // objective_heads/q_dense
    pub l1_weight_dense: Dense,       // objective_heads/l1_weight_dense
    pub upper_bound_dense: Dense,     // objective_heads/upper_bound_dense
    pub lora_pre_ln: LayerNorm,       // lora_pre_ln
    pub lora_a_dense: Dense,          // lora_A_dense [256, K*d_e*r]
    pub lora_b_dense: Dense,          // lora_B_dense [256, K*d_v*r]
}

/// Config the encoder needs at run time.
#[derive(Debug, Clone, Copy)]
pub struct MlpEncoderV2Config {
    pub d_v: usize,          // 10
    pub d_e: usize,          // 5
    pub num_directions: usize, // K = 8
    pub lora_rank: usize,    // 4
    pub lora_alpha: f32,     // 1.0
    pub q_epsilon: f32,      // 1e-4
}

pub struct MlpEncoderV2 {
    pub params: MlpEncoderV2Params,
    pub config: MlpEncoderV2Config,
}

impl MlpEncoderV2 {
    /// Encode patches `[N, B, ph, pw, C]` -> `EncoderOutput` with
    /// `h [N,B,d_v]`, `Objective::L1Box`, and `LoraFactors` (A `[N,B,K,d_e,r]`,
    /// B `[N,B,K,d_v,r]`).
    ///
    /// Flatten to `[N*B, ph*pw*C]`, apply once (encoder shared across agents),
    /// reshape every `[N*B, ...]` output back to `[N, B, ...]`.
    pub fn forward(&self, patches: &Array5<f32>) -> EncoderOutput {
        todo!()
    }
}
