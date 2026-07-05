//! The end-to-end maze model: encode -> geometry -> ADMM -> decode.
//! Ports `models/sheaf_model.py` (maze / `coordinate_history` path).

use std::sync::Arc;

use ndarray::{Array3, Array5};

use sheaf_core::admm::{AdmmHistory, AdmmState};
use sheaf_core::graph::AgentGraph;
use sheaf_core::tensor::RestrictionMaps;

use crate::config::ExportedConfig;
use crate::decoder::ConcatMlpDecoderV2;
use crate::encoder::MlpEncoderV2;

/// Learned base restriction maps, stacked `[K, d_e, d_v]` in
/// `direction_names` order (safetensors keys `rm/R_N`, `rm/R_NE`, ...).
#[derive(Debug, Clone)]
pub struct RmParams {
    pub r_stack: Array3<f32>,
}

/// The full maze Sheaf-ADMM model (weights + baked scalars + config).
pub struct SheafAdmmModel {
    pub config: ExportedConfig,
    pub encoder: MlpEncoderV2,
    pub decoder: ConcatMlpDecoderV2,
    pub rm: RmParams,
    /// Export-baked learned penalty (config.baked.rho).
    pub rho: f32,
}

/// Everything the demo / parity tests read from one forward pass.
pub struct MazeForward {
    pub history: AdmmHistory,
    /// Decoded per-iteration logits `[K, N, B, ph, pw, num_classes]`
    /// (the prediction is read off the **x**-iterate, matching training).
    pub logits_per_iter: ndarray::Array6<f32>,
    pub final_state: AdmmState,
    /// The assembled base maps `[E, 2, d_e, d_v]` (golden cross-check).
    pub base_restriction_maps: RestrictionMaps,
}

impl SheafAdmmModel {
    /// Full forward with per-iteration history (mirrors Python
    /// `coordinate_history`):
    /// 1. encode patches -> `EncoderOutput` (h, L1Box objective, LoRA factors);
    /// 2. assemble base maps + `LoraGeometry::create_directional`;
    /// 3. `z_init = h` (config `z_init = "h"`);
    /// 4. `run_admm_history` for `num_iters`;
    /// 5. decode every `history.x[k]` through the shared decoder.
    ///
    /// `patches: [N, B, ph, pw, C]`; graph built per call (size generalization).
    pub fn forward(
        &self,
        patches: &Array5<f32>,
        graph: Arc<AgentGraph>,
        num_iters: usize,
    ) -> MazeForward {
        todo!()
    }
}
