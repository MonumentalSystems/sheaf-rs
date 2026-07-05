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
    /// Optional exporter provenance flag: `Some(true)` iff the checkpoint was
    /// trained. The current exporter does not write it (the shipped goldens
    /// are seed-0 random init — see goldens/maze/NOTES.md), so consumers must
    /// treat `None` as not-known-trained and label output honestly.
    #[serde(default)]
    pub trained: Option<bool>,
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
    /// Parse + validate an exported `config.json` (goldens/CONTRACT.md).
    /// Unknown keys are a hard error (`deny_unknown_fields`); the sanity
    /// checks pin the maze scope so an exporter drift fails loudly here
    /// instead of as a numeric parity failure.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let cfg: Self = serde_json::from_str(json)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> anyhow::Result<()> {
        use anyhow::ensure;
        let m = &self.model;
        ensure!(m.encoder_arch == "mlp_v2", "unsupported encoder_arch {:?} (maze scope: mlp_v2)", m.encoder_arch);
        ensure!(m.decoder_arch == "mlp_concat_v2", "unsupported decoder_arch {:?}", m.decoder_arch);
        ensure!(m.objective_mode == "l1box_diag", "unsupported objective_mode {:?}", m.objective_mode);
        ensure!(m.x_solver == "diagonal_prox", "unsupported x_solver {:?}", m.x_solver);
        ensure!(m.z_solver == "unrolled_cg", "unsupported z_solver {:?}", m.z_solver);
        ensure!(m.z_mode == "prox" || m.z_mode == "project", "unknown z_mode {:?}", m.z_mode);
        ensure!(m.prox_init == "legacy", "unsupported prox_init {:?} ('warm' is training-only, dropped)", m.prox_init);
        ensure!(m.rm_sharing == "directional", "unsupported rm_sharing {:?} (maze scope)", m.rm_sharing);
        ensure!(m.rm_mode == "context" || m.rm_mode == "fixed", "unknown rm_mode {:?}", m.rm_mode);
        ensure!(m.num_directions == 4 || m.num_directions == 8, "num_directions must be 4 or 8, got {}", m.num_directions);
        ensure!(!m.lora_use_gate, "lora_use_gate is not shipped by any config (descoped)");
        ensure!(m.z_init == "h" || m.z_init == "zeros", "unknown z_init {:?}", m.z_init);
        ensure!(m.comm_norm_type == "layernorm", "unsupported comm_norm_type {:?} (maze scope)", m.comm_norm_type);
        ensure!(m.lora_rank > 0, "lora_rank must be positive");
        ensure!(m.d_v > 0 && m.d_e > 0, "stalk dims must be positive");
        ensure!(m.cg_iters > 0, "cg_iters must be positive");
        ensure!(m.gamma > 0.0, "gamma must be positive");
        ensure!(m.q_epsilon > 0.0, "q_epsilon must be positive");
        ensure!(self.task.task == "maze", "unsupported task {:?} (maze scope)", self.task.task);
        ensure!(self.task.patch_size % 2 == 1, "patch_size must be odd (center-indexed patches)");
        ensure!(self.baked.rho > 0.0 && self.baked.rho.is_finite(), "baked rho must be a positive finite float");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact example from goldens/CONTRACT.md.
    const CONTRACT_JSON: &str = r#"{
      "model": {
        "num_classes": 6, "d_v": 10, "d_e": 5,
        "encoder_arch": "mlp_v2", "enc_hidden_dim": 256, "comm_norm_type": "layernorm",
        "objective_mode": "l1box_diag", "x_solver": "diagonal_prox",
        "z_solver": "unrolled_cg", "z_mode": "prox", "gamma": 5.0,
        "cg_iters": 5, "tikhonov_eps": 1e-5, "prox_init": "legacy",
        "rm_sharing": "directional", "rm_init": "orthonormal", "rm_mode": "context",
        "lora_rank": 4, "lora_alpha": 1.0, "lora_use_gate": false,
        "lora_init_style": "standard", "num_directions": 8,
        "relaxation_alpha": 1.0, "z_init": "h", "q_epsilon": 1e-4,
        "l1_init": 0.01, "upper_init": 1.0,
        "decoder_arch": "mlp_concat_v2", "dec_hidden_dim": 256
      },
      "task": {
        "task": "maze", "patch_size": 3, "stride": 2, "connectivity": 8,
        "num_classes": 6, "k_eval": 100, "loss_window": 4
      },
      "baked": {
        "rho": 0.2751
      }
    }"#;

    #[test]
    fn parses_the_contract_example() {
        let cfg = ExportedConfig::from_json(CONTRACT_JSON).expect("contract JSON must parse");
        assert_eq!(cfg.model.d_v, 10);
        assert_eq!(cfg.model.d_e, 5);
        assert_eq!(cfg.model.num_directions, 8);
        assert_eq!(cfg.model.lora_rank, 4);
        assert_eq!(cfg.model.gamma, 5.0);
        assert_eq!(cfg.model.cg_iters, 5);
        assert_eq!(cfg.model.tikhonov_eps, 1e-5);
        assert_eq!(cfg.model.q_epsilon, 1e-4);
        assert_eq!(cfg.model.relaxation_alpha, 1.0);
        assert_eq!(cfg.task.patch_size, 3);
        assert_eq!(cfg.task.k_eval, 100);
        assert_eq!(cfg.baked.rho, 0.2751);
    }

    #[test]
    fn rejects_unknown_keys() {
        let json = CONTRACT_JSON.replace(r#""rho": 0.2751"#, r#""rho": 0.2751, "eta": 0.5"#);
        assert!(ExportedConfig::from_json(&json).is_err(), "unknown baked key must fail");
        let json = CONTRACT_JSON.replace(r#""d_v": 10,"#, r#""d_v": 10, "mystery": 1,"#);
        assert!(ExportedConfig::from_json(&json).is_err(), "unknown model key must fail");
    }

    #[test]
    fn rejects_out_of_scope_configs() {
        for (from, to) in [
            (r#""objective_mode": "l1box_diag""#, r#""objective_mode": "lasso""#),
            (r#""prox_init": "legacy""#, r#""prox_init": "warm""#),
            (r#""rm_sharing": "directional""#, r#""rm_sharing": "sudoku""#),
            (r#""lora_use_gate": false"#, r#""lora_use_gate": true"#),
            (r#""num_directions": 8"#, r#""num_directions": 6"#),
            (r#""task": "maze""#, r#""task": "sudoku""#),
        ] {
            let json = CONTRACT_JSON.replace(from, to);
            assert_ne!(json, CONTRACT_JSON, "replacement {from} did not apply");
            assert!(ExportedConfig::from_json(&json).is_err(), "must reject {to}");
        }
    }
}
