//! Model configuration. Deserializes the exported `goldens/maze/config.json`
//! (see goldens/CONTRACT.md), whose `model` block mirrors the Python
//! `sheaf_admm.models.config.ModelConfig` field names.

use serde::Deserialize;

/// Architectural / solver hyperparameters (maze-relevant subset of the Python
/// `ModelConfig`; unknown JSON keys are rejected so upstream drift fails loudly).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub num_classes: usize,      // 6
    pub d_v: usize,              // 10
    pub d_e: usize,              // 5
    pub encoder_arch: String,    // "mlp_v2"
    pub enc_hidden_dim: usize,   // 256
    pub comm_norm_type: String,  // "layernorm"
    pub objective_mode: String,  // "l1box_diag"
    pub x_solver: String,        // "diagonal_prox"
    pub z_solver: String,        // "unrolled_cg"
    pub z_mode: String,          // "prox"
    pub gamma: f32,              // 5.0
    pub cg_iters: usize,         // 5
    pub tikhonov_eps: f32,       // 1e-5
    pub prox_init: String,       // "legacy"
    pub rm_sharing: String,      // "directional"
    pub rm_init: String,         // "orthonormal"
    pub rm_mode: String,         // "context" (LoRA)
    pub lora_rank: usize,        // 4
    pub lora_alpha: f32,         // 1.0
    pub lora_use_gate: bool,     // false
    pub lora_init_style: String, // "standard"
    pub num_directions: usize,   // 8
    pub relaxation_alpha: f32,   // 1.0
    pub z_init: String,          // "h"
    pub q_epsilon: f32,          // 1e-4
    pub l1_init: f32,            // 0.01
    pub upper_init: f32,         // 1.0
    pub decoder_arch: String,    // "mlp_concat_v2"
    pub dec_hidden_dim: usize,   // 256
}

/// The full exported config.json: model block + task/data geometry + the
/// export-baked scalars (PLAN.md §3.5: rho etc. are collapsed to values at
/// export; Rust never implements the offset-softplus for inference).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportedConfig {
    pub model: ModelConfig,
    pub task: TaskConfig,
    pub baked: BakedScalars,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskConfig {
    pub task: String,          // "maze"
    pub patch_size: usize,     // 3
    pub stride: usize,         // 2
    pub connectivity: usize,   // 8
    pub num_classes: usize,    // 6 (input token classes)
    pub k_eval: usize,         // 100
    pub loss_window: usize,    // 4 (training detail; goldens decode per-iter x)
}

/// Scalars baked from their offset-softplus reparameterization at export time.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BakedScalars {
    /// `softplus(rho_raw + inverse_softplus(rho_init))` evaluated at export.
    pub rho: f32,
}

impl ExportedConfig {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        todo!("serde_json::from_str + sanity asserts (maze scope)")
    }
}
