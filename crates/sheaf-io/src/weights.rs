//! safetensors -> typed parameter structs.
//!
//! Key naming = flattened Flax path with `/` separators, prefixed by the
//! collection (`params` or `ema_params`) — the exact list for the maze config
//! is pinned in goldens/CONTRACT.md. The name map is resolved ONCE at load
//! into the typed structs; no string lookups after this module returns.
//!
//! Pinned semantics:
//! - Dense kernels are `[in, out]` — do NOT transpose (PLAN.md §3.4);
//! - the loader defaults to **`ema_params`** (paper eval uses the EMA shadow);
//! - tensors may be f32 or f16 (wasm embedding); f16 is widened to f32 on load;
//! - every expected key must be present and no unexpected key may remain
//!   (mirror of the exporter's exhaustive manifest — fail loudly).

use std::path::Path;

use sheaf_nn::config::ExportedConfig;
use sheaf_nn::model::SheafAdmmModel;

/// Which parameter tree to read from the safetensors file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightCollection {
    /// EMA shadow (`ema_params/...`) — the default; paper eval uses this.
    #[default]
    Ema,
    /// Raw trained params (`params/...`).
    Raw,
}

/// Load `config.json` + `weights.safetensors` into a ready-to-run model.
pub fn load_maze_model(
    config_path: &Path,
    weights_path: &Path,
    collection: WeightCollection,
) -> anyhow::Result<SheafAdmmModel> {
    todo!("parse ExportedConfig; open safetensors; build typed params; assert exhaustive keys")
}

/// Load raw config JSON (exposed separately for tools/tests).
pub fn load_config(config_path: &Path) -> anyhow::Result<ExportedConfig> {
    todo!()
}
