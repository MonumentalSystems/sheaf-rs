//! Model configuration. Deserializes the exported `config.json`
//! (see goldens/CONTRACT.md), whose `model` block mirrors the Python
//! `sheaf_admm.models.config.ModelConfig` field names.
//!
//! Two task scopes are accepted (Phase A maze + Phase B mnist), dispatched on
//! `task.task`:
//! - **maze**: `mlp_v2` encoder, `l1box_diag` objective, `directional` sharing,
//!   `mlp_concat_v2` decoder (prox-mode CG).
//! - **mnist**: residual `mlp` encoder, `lasso` objective (scalar `l1_weight`),
//!   `global` sharing, `classification` decoder (linear head, `x_only`
//!   readout), hard-consensus `project`-mode CG.
//!
//! Task-specific fields carry serde defaults so a single struct parses both
//! configs; `validate()` is task-dispatched so each scope stays strict (an
//! out-of-scope maze config is still rejected exactly as before).

use serde::Deserialize;

fn default_lora_alpha() -> f32 {
    1.0
}

/// Architectural / solver hyperparameters (subset of the Python `ModelConfig`
/// used by inference; unknown JSON keys are rejected so upstream drift fails
/// loudly). Fields that only one task ships carry serde defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub num_classes: usize,      // maze 6 / mnist 10 (OUTPUT classes)
    pub d_v: usize,              // maze 10 / mnist 32
    pub d_e: usize,              // maze 5 / mnist 24
    pub encoder_arch: String,    // "mlp_v2" (maze) | "mlp" (mnist)
    pub enc_hidden_dim: usize,   // 256
    pub comm_norm_type: String,  // "layernorm"
    pub objective_mode: String,  // "l1box_diag" (maze) | "lasso" (mnist)
    pub x_solver: String,        // "diagonal_prox"
    pub z_solver: String,        // "unrolled_cg"
    pub z_mode: String,          // "prox" (maze) | "project" (mnist)
    #[serde(default)]
    pub gamma: f32, // maze 5.0; unused by project-mode mnist
    pub cg_iters: usize,   // 5
    pub tikhonov_eps: f32, // 1e-5
    #[serde(default)]
    pub prox_init: String, // "legacy" (maze); unused by mnist inference
    pub rm_sharing: String,      // "directional" (maze) | "global" (mnist)
    pub rm_init: String,         // "orthonormal"
    pub rm_mode: String,         // "context" (LoRA)
    pub lora_rank: usize,        // maze 4 / mnist 8
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f32, // 1.0
    #[serde(default)]
    pub lora_use_gate: bool, // false
    pub lora_init_style: String, // "standard" (maze) | "legacy" (mnist); TRAIN-only, no fwd effect
    pub num_directions: usize,   // 8 (LoRA slots are direction-indexed on both tasks)
    pub relaxation_alpha: f32,   // 1.0
    pub z_init: String,          // "h"
    pub q_epsilon: f32,          // 1e-4
    #[serde(default)]
    pub l1_init: f32, // maze 0.01 (per-dim l1box head init)
    #[serde(default)]
    pub upper_init: f32, // maze 1.0 (per-dim box upper init)
    /// mnist `lasso` scalar L1 weight (config, NOT learned). Absent for maze.
    #[serde(default)]
    pub l1_weight: Option<f32>,
    pub decoder_arch: String, // "mlp_concat_v2" (maze) | "classification" (mnist)
    #[serde(default)]
    pub dec_hidden_dim: usize, // maze 256 (mlp_concat_v2 hidden)
    /// mnist classification linear head (`true`). Absent for maze.
    #[serde(default)]
    pub dec_linear_head: Option<bool>,
    /// mnist classification readout: `"x_only"`. Absent for maze.
    #[serde(default)]
    pub dec_readout_mode: Option<String>,
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
    /// trained (`export_weights.py --checkpoint`). Absent in random-init
    /// exports, so consumers must treat `None` as not-known-trained and label
    /// output honestly.
    #[serde(default)]
    pub trained: Option<bool>,
    /// Optional free-form training provenance block written alongside
    /// `trained: true` (dataset, epochs, final metrics). Informational only —
    /// nothing in the inference path reads it, so it stays untyped.
    #[serde(default)]
    pub training: Option<serde_json::Value>,
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

    /// Task-dispatched scope validation (public so loaders can re-check a
    /// config parsed via `load_config`, which skips validation).
    pub fn validate(&self) -> anyhow::Result<()> {
        use anyhow::ensure;
        let m = &self.model;
        // ---- task-agnostic invariants (both scopes share these) ----
        ensure!(m.x_solver == "diagonal_prox", "unsupported x_solver {:?}", m.x_solver);
        ensure!(m.z_solver == "unrolled_cg", "unsupported z_solver {:?}", m.z_solver);
        ensure!(m.z_mode == "prox" || m.z_mode == "project", "unknown z_mode {:?}", m.z_mode);
        ensure!(m.rm_mode == "context" || m.rm_mode == "fixed", "unknown rm_mode {:?}", m.rm_mode);
        ensure!(m.num_directions == 4 || m.num_directions == 8, "num_directions must be 4 or 8, got {}", m.num_directions);
        ensure!(!m.lora_use_gate, "lora_use_gate is not shipped by any config (descoped)");
        ensure!(m.z_init == "h" || m.z_init == "zeros", "unknown z_init {:?}", m.z_init);
        ensure!(m.comm_norm_type == "layernorm", "unsupported comm_norm_type {:?}", m.comm_norm_type);
        ensure!(m.rm_init == "orthonormal", "unsupported rm_init {:?}", m.rm_init);
        ensure!(m.lora_rank > 0, "lora_rank must be positive");
        ensure!(m.d_v > 0 && m.d_e > 0, "stalk dims must be positive");
        ensure!(m.cg_iters > 0, "cg_iters must be positive");
        ensure!(m.q_epsilon > 0.0, "q_epsilon must be positive");
        ensure!(self.task.patch_size % 2 == 1, "patch_size must be odd (center-indexed patches)");
        ensure!(self.baked.rho > 0.0 && self.baked.rho.is_finite(), "baked rho must be a positive finite float");

        // ---- task-dispatched scope guards (each stays strict) ----
        match self.task.task.as_str() {
            "maze" => {
                ensure!(m.encoder_arch == "mlp_v2", "unsupported encoder_arch {:?} (maze scope: mlp_v2)", m.encoder_arch);
                ensure!(m.decoder_arch == "mlp_concat_v2", "unsupported decoder_arch {:?} (maze scope)", m.decoder_arch);
                ensure!(m.objective_mode == "l1box_diag", "unsupported objective_mode {:?} (maze scope)", m.objective_mode);
                ensure!(m.rm_sharing == "directional", "unsupported rm_sharing {:?} (maze scope)", m.rm_sharing);
                ensure!(m.prox_init == "legacy", "unsupported prox_init {:?} ('warm' is training-only, dropped)", m.prox_init);
                ensure!(m.gamma > 0.0, "maze gamma must be positive");
            }
            "mnist" => {
                ensure!(m.encoder_arch == "mlp", "unsupported encoder_arch {:?} (mnist scope: mlp)", m.encoder_arch);
                ensure!(m.decoder_arch == "classification", "unsupported decoder_arch {:?} (mnist scope: classification)", m.decoder_arch);
                ensure!(m.objective_mode == "lasso", "unsupported objective_mode {:?} (mnist scope: lasso)", m.objective_mode);
                ensure!(m.rm_sharing == "global", "unsupported rm_sharing {:?} (mnist scope: global)", m.rm_sharing);
                ensure!(m.z_mode == "project", "mnist uses hard-consensus z_mode=project, got {:?}", m.z_mode);
                ensure!(
                    m.dec_linear_head == Some(true),
                    "mnist classification decoder requires dec_linear_head=true, got {:?}",
                    m.dec_linear_head
                );
                ensure!(
                    m.dec_readout_mode.as_deref() == Some("x_only"),
                    "mnist classification decoder requires dec_readout_mode=\"x_only\", got {:?}",
                    m.dec_readout_mode
                );
                match m.l1_weight {
                    Some(l1) => ensure!(
                        l1.is_finite() && l1 >= 0.0,
                        "mnist lasso l1_weight must be finite and non-negative, got {l1}"
                    ),
                    None => anyhow::bail!("mnist lasso config must carry model.l1_weight (scalar)"),
                }
            }
            other => anyhow::bail!("unsupported task {other:?} (maze|mnist scope)"),
        }
        Ok(())
    }

    /// Scalar lasso L1 weight (mnist); panics on a non-lasso config.
    pub fn l1_weight(&self) -> f32 {
        self.model
            .l1_weight
            .expect("l1_weight() called on a config without model.l1_weight (non-lasso)")
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

    /// Phase B: the shipped mnist config (`configs/experiment/mnist_sheaf.yaml`).
    /// Mirrors the maze CONTRACT layout but with the mnist model block.
    const MNIST_JSON: &str = r#"{
      "model": {
        "num_classes": 10, "d_v": 32, "d_e": 24,
        "encoder_arch": "mlp", "enc_hidden_dim": 256, "comm_norm_type": "layernorm",
        "objective_mode": "lasso", "l1_weight": 0.006337180166370117,
        "x_solver": "diagonal_prox",
        "z_solver": "unrolled_cg", "z_mode": "project",
        "cg_iters": 5, "tikhonov_eps": 1e-5,
        "rm_sharing": "global", "rm_init": "orthonormal", "rm_mode": "context",
        "lora_rank": 8, "lora_alpha": 1.0, "lora_init_style": "legacy",
        "num_directions": 8,
        "relaxation_alpha": 1.0, "z_init": "h", "q_epsilon": 1e-4,
        "decoder_arch": "classification", "dec_linear_head": true,
        "dec_readout_mode": "x_only"
      },
      "task": {
        "task": "mnist", "patch_size": 3, "stride": 3, "connectivity": 8,
        "num_classes": 10, "k_eval": 100, "loss_window": 2
      },
      "baked": { "rho": 0.12 }
    }"#;

    #[test]
    fn parses_the_mnist_config() {
        let cfg = ExportedConfig::from_json(MNIST_JSON).expect("mnist config must parse");
        assert_eq!(cfg.model.d_v, 32);
        assert_eq!(cfg.model.d_e, 24);
        assert_eq!(cfg.model.num_classes, 10);
        assert_eq!(cfg.model.encoder_arch, "mlp");
        assert_eq!(cfg.model.decoder_arch, "classification");
        assert_eq!(cfg.model.objective_mode, "lasso");
        assert_eq!(cfg.model.rm_sharing, "global");
        assert_eq!(cfg.model.z_mode, "project");
        assert_eq!(cfg.model.lora_rank, 8);
        assert_eq!(cfg.model.dec_linear_head, Some(true));
        assert_eq!(cfg.model.dec_readout_mode.as_deref(), Some("x_only"));
        assert_eq!(cfg.l1_weight(), 0.006337180166370117);
        assert_eq!(cfg.task.task, "mnist");
        assert_eq!(cfg.task.stride, 3);
        assert_eq!(cfg.baked.rho, 0.12);
        // gamma / prox_init / l1_init default when the mnist config omits them.
        assert_eq!(cfg.model.gamma, 0.0);
        assert_eq!(cfg.model.lora_alpha, 1.0);
    }

    #[test]
    fn rejects_out_of_scope_mnist_configs() {
        for (from, to) in [
            (r#""encoder_arch": "mlp""#, r#""encoder_arch": "mlp_v2""#),
            (r#""objective_mode": "lasso""#, r#""objective_mode": "l1box_diag""#),
            (r#""rm_sharing": "global""#, r#""rm_sharing": "directional""#),
            (r#""z_mode": "project""#, r#""z_mode": "prox""#),
            (r#""dec_linear_head": true"#, r#""dec_linear_head": false"#),
            (r#""dec_readout_mode": "x_only""#, r#""dec_readout_mode": "concat""#),
        ] {
            let json = MNIST_JSON.replace(from, to);
            assert_ne!(json, MNIST_JSON, "replacement {from} did not apply");
            assert!(ExportedConfig::from_json(&json).is_err(), "must reject {to}");
        }
    }

    #[test]
    fn rejects_mnist_lasso_missing_l1_weight() {
        // Drop the l1_weight key entirely -> lasso objective is unspecified.
        let json = MNIST_JSON.replace(r#""l1_weight": 0.006337180166370117,"#, "");
        assert_ne!(json, MNIST_JSON);
        let err = ExportedConfig::from_json(&json).expect_err("missing l1_weight must fail");
        assert!(err.to_string().contains("l1_weight"), "{err}");
    }
}
