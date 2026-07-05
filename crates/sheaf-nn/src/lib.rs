//! Inference-only neural layers + the maze Sheaf-ADMM model.
//!
//! Ports `sheaf_admm.models.{layers,encoder,decoder,sheaf_model,config}` and
//! `sheaf_admm.geometry.restriction_maps` (base-map assembly).
//!
//! Conventions (PLAN.md §3.4/§3.5 — pinned):
//! - Flax Dense kernels are stored `[in, out]` (`y = x*W + b`); do NOT transpose;
//! - GELU is the tanh approximation (Flax default);
//! - RMSNorm / LayerNorm eps 1e-6; softplus is the stable
//!   `max(x, 0) + log1p(exp(-|x|))` form;
//! - encoders/decoders are shared across agents via the exact
//!   `[N, B, ...] -> [N*B, ...] -> apply -> reshape back` contract;
//! - dropout / training=True branches do not exist here (shipped rate = 0).

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod layers;
pub mod model;
pub mod restriction_maps;

pub use config::{ExportedConfig, ModelConfig};
pub use decoder::{ConcatMlpDecoderV2, ConcatMlpDecoderV2Params};
pub use encoder::{MlpEncoderV2, MlpEncoderV2Config, MlpEncoderV2Params};
pub use model::{MazeForward, RmParams, SheafAdmmModel};
pub use restriction_maps::{
    build_directional_restriction_maps, compute_direction_index, direction_names,
    direction_slot_tables,
};
