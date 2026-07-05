//! ConcatMLPDecoderV2 (`arch = "mlp_concat_v2"`, the maze decoder).
//! Ports `models/decoder.py`.
//!
//! `RMSNorm(input_norm)(concat([flatten(patch), x])) -> Dense(dense, 256)
//!  -> gelu_tanh -> Dense(output_dense, ph*pw*num_classes)`,
//! reshaped to `[.., ph, pw, num_classes]`. Note the concat order:
//! **patch first, then x** (encoder feats 54 + d_v 10 = 64 in).

use ndarray::{Array3, Array5};

use crate::layers::{Dense, RmsNorm};

#[derive(Debug, Clone)]
pub struct ConcatMlpDecoderV2Params {
    pub input_norm: RmsNorm, // input_norm/scale [64]
    pub dense: Dense,        // dense/{kernel [64,256], bias}
    pub output_dense: Dense, // output_dense/{kernel [256,54], bias}
}

pub struct ConcatMlpDecoderV2 {
    pub params: ConcatMlpDecoderV2Params,
    /// (ph, pw, num_classes) = (3, 3, 6) for the maze.
    pub output_shape: (usize, usize, usize),
}

impl ConcatMlpDecoderV2 {
    /// Decode one agent-state slab: `x [N, B, d_v]` + `patches [N, B, ph, pw, C]`
    /// -> logits `[N, B, ph, pw, num_classes]`. Shared across agents via the
    /// `[N*B, ...]` flatten contract.
    pub fn forward(
        &self,
        x: &Array3<f32>,
        patches: &Array5<f32>,
    ) -> ndarray::Array5<f32> {
        todo!()
    }
}
