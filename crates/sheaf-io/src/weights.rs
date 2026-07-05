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

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, ensure, Context};
use ndarray::{Array1, Array2, Array3};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use sheaf_nn::config::ExportedConfig;
use sheaf_nn::decoder::{
    ClassificationDecoder, ClassificationDecoderParams, ConcatMlpDecoderV2,
    ConcatMlpDecoderV2Params, ReadoutMode, SudokuDecoder, SudokuDecoderParams,
};
use sheaf_nn::encoder::{
    MlpEncoder, MlpEncoderConfig, MlpEncoderParams, MlpEncoderV2, MlpEncoderV2Config,
    MlpEncoderV2Params, SudokuEncoder, SudokuEncoderConfig, SudokuEncoderParams, SudokuLoraHeads,
};
use sheaf_nn::layers::{Dense, LayerNorm, MlpBlock, MlpMixerBlock, RmsNorm, SwiGlu};
use sheaf_nn::model::{MnistSheafModel, RmParams, SheafAdmmModel, SudokuSheafModel};
use sheaf_nn::restriction_maps::direction_names;

/// RMSNorm / LayerNorm epsilon (PLAN.md §3.4 numerics contract).
const NORM_EPS: f32 = 1e-6;

/// Pinned per-collection parameter count for the maze config (CONTRACT.md).
/// Now cross-checked in tests (the loader derives the count from the manifest).
#[allow(dead_code)]
const MAZE_PARAM_COUNT: usize = 181_859;

/// MNIST image channels (grayscale, normalized [0,1] — `data/build_mnist.py`
/// reshapes to `[N, H, W, 1]`). The encoder input width is `ps*ps*C`.
const MNIST_IMAGE_CHANNELS: usize = 1;

/// Which parameter tree to read from the safetensors file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightCollection {
    /// EMA shadow (`ema_params/...`) — the default; paper eval uses this.
    #[default]
    Ema,
    /// Raw trained params (`params/...`).
    Raw,
}

impl WeightCollection {
    fn prefix(self) -> &'static str {
        match self {
            WeightCollection::Ema => "ema_params",
            WeightCollection::Raw => "params",
        }
    }
}

/// Load raw config JSON (exposed separately for tools/tests).
pub fn load_config(config_path: &Path) -> anyhow::Result<ExportedConfig> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading config {}", config_path.display()))?;
    let config: ExportedConfig = serde_json::from_str(&text)
        .with_context(|| format!("parsing config {}", config_path.display()))?;
    Ok(config)
}

/// Exhaustive expected-key manifest (suffix after `<collection>/`) with shapes
/// derived from the config — the loader-side mirror of the exporter's
/// manifest assertion (goldens/CONTRACT.md).
fn expected_keys(config: &ExportedConfig) -> BTreeMap<String, Vec<usize>> {
    let m = &config.model;
    let t = &config.task;
    let in_feats = t.patch_size * t.patch_size * t.num_classes; // 54
    let enc_h = m.enc_hidden_dim; // 256
    let dec_h = m.dec_hidden_dim; // 256
    let dec_in = in_feats + m.d_v; // 64: patch first, then x
    let dec_out = t.patch_size * t.patch_size * m.num_classes; // 54
    let lora_a_out = m.num_directions * m.d_e * m.lora_rank; // 160
    let lora_b_out = m.num_directions * m.d_v * m.lora_rank; // 320

    let mut keys: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let dense = |name: &str, i: usize, o: usize, keys: &mut BTreeMap<String, Vec<usize>>| {
        keys.insert(format!("{name}/kernel"), vec![i, o]);
        keys.insert(format!("{name}/bias"), vec![o]);
    };

    // Encoder.
    keys.insert("MLPEncoderV2_0/input_norm/scale".into(), vec![in_feats]);
    dense("MLPEncoderV2_0/dense", in_feats, enc_h, &mut keys);
    dense("MLPEncoderV2_0/comm_head/comm_dense", enc_h, m.d_v, &mut keys);
    keys.insert("MLPEncoderV2_0/comm_head/comm_norm/scale".into(), vec![m.d_v]);
    keys.insert("MLPEncoderV2_0/comm_head/comm_norm/bias".into(), vec![m.d_v]);
    for head in ["q_diag_dense", "q_dense", "l1_weight_dense", "upper_bound_dense"] {
        dense(
            &format!("MLPEncoderV2_0/objective_heads/{head}"),
            enc_h,
            m.d_v,
            &mut keys,
        );
    }
    keys.insert("MLPEncoderV2_0/lora_pre_ln/scale".into(), vec![enc_h]);
    keys.insert("MLPEncoderV2_0/lora_pre_ln/bias".into(), vec![enc_h]);
    dense("MLPEncoderV2_0/lora_A_dense", enc_h, lora_a_out, &mut keys);
    dense("MLPEncoderV2_0/lora_B_dense", enc_h, lora_b_out, &mut keys);

    // Decoder.
    keys.insert("ConcatMLPDecoderV2_0/input_norm/scale".into(), vec![dec_in]);
    dense("ConcatMLPDecoderV2_0/dense", dec_in, dec_h, &mut keys);
    dense("ConcatMLPDecoderV2_0/output_dense", dec_h, dec_out, &mut keys);

    // Base restriction maps + the raw (unbaked) rho scalar.
    for name in direction_names(m.num_directions) {
        keys.insert(format!("rm/R_{name}"), vec![m.d_e, m.d_v]);
    }
    keys.insert("rho_raw".into(), vec![]);

    keys
}

/// Widen a safetensors view (F32 or F16) to a f32 vec; anything else is a
/// contract violation.
fn to_f32_vec(name: &str, view: &TensorView) -> anyhow::Result<Vec<f32>> {
    let data = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::F16 => Ok(data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        other => bail!("tensor {name:?}: unsupported dtype {other:?} (want F32 or F16)"),
    }
}

/// One collection's tensors, name-mapped and shape-checked against the
/// expected-key manifest. All string lookups happen here, once.
struct Tree<'a> {
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    prefix: &'a str,
}

impl<'a> Tree<'a> {
    fn load(
        st: &SafeTensors,
        prefix: &'a str,
        expected: &BTreeMap<String, Vec<usize>>,
    ) -> anyhow::Result<Self> {
        let mut tensors = BTreeMap::new();
        for (full_name, view) in st.iter() {
            let Some(suffix) = full_name.strip_prefix(prefix).and_then(|s| s.strip_prefix('/'))
            else {
                continue;
            };
            let Some(want_shape) = expected.get(suffix) else {
                bail!("unexpected key {full_name:?} in safetensors (manifest violation)");
            };
            ensure!(
                view.shape() == want_shape.as_slice(),
                "tensor {full_name:?}: expected shape {want_shape:?}, got {:?}",
                view.shape()
            );
            tensors.insert(suffix.to_string(), (view.shape().to_vec(), to_f32_vec(full_name, &view)?));
        }
        for key in expected.keys() {
            ensure!(
                tensors.contains_key(key),
                "missing key {prefix}/{key} in safetensors (manifest violation)"
            );
        }
        // Cross-check the materialized element count against the manifest
        // (the maze count is additionally pinned against CONTRACT.md in tests).
        let count: usize = tensors.values().map(|(_, v)| v.len()).sum();
        let expected_total: usize =
            expected.values().map(|s| s.iter().product::<usize>()).sum();
        ensure!(
            count == expected_total,
            "collection {prefix:?}: {count} params, expected {expected_total}"
        );
        Ok(Tree { tensors, prefix })
    }

    fn take(&mut self, key: &str) -> (Vec<usize>, Vec<f32>) {
        self.tensors
            .remove(key)
            .unwrap_or_else(|| panic!("key {}/{key} vanished after manifest check", self.prefix))
    }

    fn vec1(&mut self, key: &str) -> Array1<f32> {
        let (_, data) = self.take(key);
        Array1::from(data)
    }

    fn mat2(&mut self, key: &str) -> Array2<f32> {
        let (shape, data) = self.take(key);
        // [in, out] — stored row-major, NOT transposed (Flax layout).
        Array2::from_shape_vec((shape[0], shape[1]), data).expect("shape checked at load")
    }

    fn dense(&mut self, name: &str) -> Dense {
        Dense {
            kernel: self.mat2(&format!("{name}/kernel")),
            bias: self.vec1(&format!("{name}/bias")),
        }
    }

    fn rms_norm(&mut self, name: &str) -> RmsNorm {
        RmsNorm {
            scale: self.vec1(&format!("{name}/scale")),
            eps: NORM_EPS,
        }
    }

    fn layer_norm(&mut self, name: &str) -> LayerNorm {
        LayerNorm {
            scale: self.vec1(&format!("{name}/scale")),
            bias: self.vec1(&format!("{name}/bias")),
            eps: NORM_EPS,
        }
    }
}

/// Maze scope guards: this loader only understands the shipped maze config.
fn maze_scope_guards(config: &ExportedConfig) -> anyhow::Result<()> {
    let m = &config.model;
    ensure!(m.encoder_arch == "mlp_v2", "unsupported encoder_arch {:?}", m.encoder_arch);
    ensure!(
        m.decoder_arch == "mlp_concat_v2",
        "unsupported decoder_arch {:?}",
        m.decoder_arch
    );
    ensure!(
        m.rm_sharing == "directional",
        "unsupported rm_sharing {:?}",
        m.rm_sharing
    );
    ensure!(
        m.objective_mode == "l1box_diag",
        "unsupported objective_mode {:?}",
        m.objective_mode
    );
    ensure!(config.task.task == "maze", "unsupported task {:?}", config.task.task);
    Ok(())
}

/// Load `config.json` + `weights.safetensors` into a ready-to-run model.
pub fn load_maze_model(
    config_path: &Path,
    weights_path: &Path,
    collection: WeightCollection,
) -> anyhow::Result<SheafAdmmModel> {
    let config = load_config(config_path)?;
    maze_scope_guards(&config)?;

    let bytes = std::fs::read(weights_path)
        .with_context(|| format!("reading weights {}", weights_path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parsing safetensors {}", weights_path.display()))?;

    let expected = expected_keys(&config);
    // Both collections must satisfy the manifest (the exporter ships both);
    // only the requested one is materialized into the model.
    for prefix in ["params", "ema_params"] {
        Tree::load(&st, prefix, &expected)?;
    }
    // No keys outside the two collections.
    for (name, _) in st.iter() {
        ensure!(
            name.starts_with("params/") || name.starts_with("ema_params/"),
            "unexpected top-level key {name:?} (want params/... or ema_params/...)"
        );
    }

    build_model(config, &st, collection)
}

/// Load a model from in-memory config JSON + safetensors bytes — the wasm
/// embedding path (`include_str!` / `include_bytes!`).
///
/// Unlike [`load_maze_model`], only the **requested** collection must be
/// present and satisfy the manifest: the f16 wasm embedding ships the
/// `ema_params/` tree alone to halve the payload. Keys outside the two known
/// collections are still rejected.
pub fn load_maze_model_from_bytes(
    config_json: &str,
    weights_bytes: &[u8],
    collection: WeightCollection,
) -> anyhow::Result<SheafAdmmModel> {
    let config: ExportedConfig =
        serde_json::from_str(config_json).context("parsing embedded config JSON")?;
    maze_scope_guards(&config)?;
    let st = SafeTensors::deserialize(weights_bytes).context("parsing safetensors bytes")?;
    for (name, _) in st.iter() {
        ensure!(
            name.starts_with("params/") || name.starts_with("ema_params/"),
            "unexpected top-level key {name:?} (want params/... or ema_params/...)"
        );
    }
    build_model(config, &st, collection)
}

/// Materialize the requested collection into typed structs (shared tail of
/// both loaders; the collection is manifest-checked here).
fn build_model(
    config: ExportedConfig,
    st: &SafeTensors,
    collection: WeightCollection,
) -> anyhow::Result<SheafAdmmModel> {
    let m = &config.model;
    let expected = expected_keys(&config);
    let mut tree = Tree::load(st, collection.prefix(), &expected)?;

    let encoder = MlpEncoderV2 {
        params: MlpEncoderV2Params {
            input_norm: tree.rms_norm("MLPEncoderV2_0/input_norm"),
            dense: tree.dense("MLPEncoderV2_0/dense"),
            comm_dense: tree.dense("MLPEncoderV2_0/comm_head/comm_dense"),
            comm_norm: tree.layer_norm("MLPEncoderV2_0/comm_head/comm_norm"),
            q_diag_dense: tree.dense("MLPEncoderV2_0/objective_heads/q_diag_dense"),
            q_dense: tree.dense("MLPEncoderV2_0/objective_heads/q_dense"),
            l1_weight_dense: tree.dense("MLPEncoderV2_0/objective_heads/l1_weight_dense"),
            upper_bound_dense: tree.dense("MLPEncoderV2_0/objective_heads/upper_bound_dense"),
            lora_pre_ln: tree.layer_norm("MLPEncoderV2_0/lora_pre_ln"),
            lora_a_dense: tree.dense("MLPEncoderV2_0/lora_A_dense"),
            lora_b_dense: tree.dense("MLPEncoderV2_0/lora_B_dense"),
        },
        config: MlpEncoderV2Config {
            d_v: m.d_v,
            d_e: m.d_e,
            num_directions: m.num_directions,
            lora_rank: m.lora_rank,
            lora_alpha: m.lora_alpha,
            q_epsilon: m.q_epsilon,
        },
    };

    let decoder = ConcatMlpDecoderV2 {
        params: ConcatMlpDecoderV2Params {
            input_norm: tree.rms_norm("ConcatMLPDecoderV2_0/input_norm"),
            dense: tree.dense("ConcatMLPDecoderV2_0/dense"),
            output_dense: tree.dense("ConcatMLPDecoderV2_0/output_dense"),
        },
        output_shape: (config.task.patch_size, config.task.patch_size, m.num_classes),
    };

    // Base maps stacked [K, d_e, d_v] in direction_names order (= slot order).
    let names = direction_names(m.num_directions);
    let mut r_stack = Array3::<f32>::zeros((names.len(), m.d_e, m.d_v));
    for (k, name) in names.iter().enumerate() {
        let (_, data) = tree.take(&format!("rm/R_{name}"));
        let map = Array2::from_shape_vec((m.d_e, m.d_v), data).expect("shape checked at load");
        r_stack.index_axis_mut(ndarray::Axis(0), k).assign(&map);
    }

    // rho_raw is present in the file but deliberately unused: inference reads
    // the export-baked value (PLAN.md §3.5); drop it from the tree.
    let _ = tree.take("rho_raw");

    let rho = config.baked.rho;
    Ok(SheafAdmmModel {
        config,
        encoder,
        decoder,
        rm: RmParams { r_stack },
        rho,
    })
}

// ===========================================================================
// MNIST loader (Phase B). Residual MLPEncoder, lasso, global sharing,
// classification decoder. Flax module names mirror the maze layout.
// ===========================================================================

/// Exhaustive expected-key manifest for the mnist config. Flax module names are
/// the reconstructed camera-ready layout (`MLPEncoder_0`, `block_0`,
/// `ClassificationDecoder_0`, `rm/R_shared`). If a golden dump disagrees, this
/// is the single place to reconcile the names against `manifest.json`.
fn mnist_expected_keys(config: &ExportedConfig) -> BTreeMap<String, Vec<usize>> {
    let m = &config.model;
    let t = &config.task;
    let in_feats = t.patch_size * t.patch_size * MNIST_IMAGE_CHANNELS; // 9
    let enc_h = m.enc_hidden_dim; // 256
    let lora_a_out = m.num_directions * m.d_e * m.lora_rank; // 8*24*8
    let lora_b_out = m.num_directions * m.d_v * m.lora_rank; // 8*32*8

    let mut keys: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let dense = |name: &str, i: usize, o: usize, keys: &mut BTreeMap<String, Vec<usize>>| {
        keys.insert(format!("{name}/kernel"), vec![i, o]);
        keys.insert(format!("{name}/bias"), vec![o]);
    };

    // Encoder trunk: input_proj -> one width-preserving residual MLPBlock.
    dense("MLPEncoder_0/input_proj", in_feats, enc_h, &mut keys);
    keys.insert("MLPEncoder_0/block_0/norm/scale".into(), vec![enc_h]);
    dense("MLPEncoder_0/block_0/dense1", enc_h, enc_h, &mut keys);
    dense("MLPEncoder_0/block_0/dense2", enc_h, enc_h, &mut keys);
    // comm_head + lasso objective heads (q_diag, q only — l1 is a config scalar).
    dense("MLPEncoder_0/comm_head/comm_dense", enc_h, m.d_v, &mut keys);
    keys.insert("MLPEncoder_0/comm_head/comm_norm/scale".into(), vec![m.d_v]);
    keys.insert("MLPEncoder_0/comm_head/comm_norm/bias".into(), vec![m.d_v]);
    for head in ["q_diag_dense", "q_dense"] {
        dense(&format!("MLPEncoder_0/objective_heads/{head}"), enc_h, m.d_v, &mut keys);
    }
    keys.insert("MLPEncoder_0/lora_pre_ln/scale".into(), vec![enc_h]);
    keys.insert("MLPEncoder_0/lora_pre_ln/bias".into(), vec![enc_h]);
    dense("MLPEncoder_0/lora_A_dense", enc_h, lora_a_out, &mut keys);
    dense("MLPEncoder_0/lora_B_dense", enc_h, lora_b_out, &mut keys);

    // Decoder: bare linear classification head over x (readout_mode = x_only).
    dense("ClassificationDecoder_0/cls_output", m.d_v, m.num_classes, &mut keys);

    // One shared base restriction map + the raw (unbaked) rho scalar.
    keys.insert("rm/R_shared".into(), vec![m.d_e, m.d_v]);
    keys.insert("rho_raw".into(), vec![]);

    keys
}

/// MNIST scope guards: this loader only understands the shipped mnist config.
fn mnist_scope_guards(config: &ExportedConfig) -> anyhow::Result<()> {
    let m = &config.model;
    ensure!(m.encoder_arch == "mlp", "unsupported encoder_arch {:?}", m.encoder_arch);
    ensure!(m.decoder_arch == "classification", "unsupported decoder_arch {:?}", m.decoder_arch);
    ensure!(m.rm_sharing == "global", "unsupported rm_sharing {:?}", m.rm_sharing);
    ensure!(m.objective_mode == "lasso", "unsupported objective_mode {:?}", m.objective_mode);
    ensure!(config.task.task == "mnist", "unsupported task {:?}", config.task.task);
    Ok(())
}

/// Load mnist `config.json` + `weights.safetensors` into a ready-to-run model.
/// The default collection is EMA (paper eval convention).
pub fn load_mnist_model(
    config_path: &Path,
    weights_path: &Path,
    collection: WeightCollection,
) -> anyhow::Result<MnistSheafModel> {
    let config = load_config(config_path)?;
    config.validate()?;
    mnist_scope_guards(&config)?;

    let bytes = std::fs::read(weights_path)
        .with_context(|| format!("reading weights {}", weights_path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parsing safetensors {}", weights_path.display()))?;

    let expected = mnist_expected_keys(&config);
    for prefix in ["params", "ema_params"] {
        Tree::load(&st, prefix, &expected)?;
    }
    for (name, _) in st.iter() {
        ensure!(
            name.starts_with("params/") || name.starts_with("ema_params/"),
            "unexpected top-level key {name:?} (want params/... or ema_params/...)"
        );
    }
    build_mnist_model(config, &st, collection)
}

/// Materialize the requested collection into the typed mnist model.
fn build_mnist_model(
    config: ExportedConfig,
    st: &SafeTensors,
    collection: WeightCollection,
) -> anyhow::Result<MnistSheafModel> {
    let m = &config.model;
    let expected = mnist_expected_keys(&config);
    let mut tree = Tree::load(st, collection.prefix(), &expected)?;

    // Residual trunk: input_proj + one width-preserving MLPBlock (no
    // residual_proj since enc_hidden_dim in == out).
    let block = MlpBlock {
        norm: tree.rms_norm("MLPEncoder_0/block_0/norm"),
        dense1: tree.dense("MLPEncoder_0/block_0/dense1"),
        dense2: tree.dense("MLPEncoder_0/block_0/dense2"),
        residual_proj: None,
    };
    let encoder = MlpEncoder {
        params: MlpEncoderParams {
            input_proj: tree.dense("MLPEncoder_0/input_proj"),
            blocks: vec![block],
            comm_dense: tree.dense("MLPEncoder_0/comm_head/comm_dense"),
            comm_norm: tree.layer_norm("MLPEncoder_0/comm_head/comm_norm"),
            q_diag_dense: tree.dense("MLPEncoder_0/objective_heads/q_diag_dense"),
            q_dense: tree.dense("MLPEncoder_0/objective_heads/q_dense"),
            lora_pre_ln: tree.layer_norm("MLPEncoder_0/lora_pre_ln"),
            lora_a_dense: tree.dense("MLPEncoder_0/lora_A_dense"),
            lora_b_dense: tree.dense("MLPEncoder_0/lora_B_dense"),
        },
        config: MlpEncoderConfig {
            d_v: m.d_v,
            d_e: m.d_e,
            num_directions: m.num_directions,
            lora_rank: m.lora_rank,
            lora_alpha: m.lora_alpha,
            q_epsilon: m.q_epsilon,
            l1_weight: config.l1_weight(),
        },
    };

    let readout_mode = match m.dec_readout_mode.as_deref() {
        Some("x_only") => ReadoutMode::XOnly,
        Some("concat") => ReadoutMode::Concat,
        other => anyhow::bail!("unsupported dec_readout_mode {other:?} (x_only|concat)"),
    };
    let decoder = ClassificationDecoder {
        params: ClassificationDecoderParams {
            cls_output: tree.dense("ClassificationDecoder_0/cls_output"),
        },
        readout_mode,
        num_classes: m.num_classes,
    };

    // The single shared base map R_shared [d_e, d_v].
    let (_, data) = tree.take("rm/R_shared");
    let rm_shared = Array2::from_shape_vec((m.d_e, m.d_v), data).expect("shape checked at load");

    // rho_raw present but unused: inference reads the export-baked value.
    let _ = tree.take("rho_raw");

    let rho = config.baked.rho;
    Ok(MnistSheafModel {
        config,
        encoder,
        decoder,
        rm_shared,
        rho,
    })
}

// ===========================================================================
// Sudoku loader (Phase C). MLP-Mixer SudokuEncoder, non_negative, soft_slice
// sudoku sharing (9 base maps), per-cell SudokuDecoder. Flax module names +
// shapes are pinned against the Python `init` param dump (543,025 / 2,029,233).
// ===========================================================================

/// The Sudoku Mixer uses a fixed `mlp_ratio = 2.0` (SudokuEncoder default; not a
/// config field). token_mlp_dim = int(9 * 2) = 18; channel_mlp_dim = d_model * 2.
const SUDOKU_MLP_RATIO: usize = 2;

/// Exhaustive expected-key manifest for a sudoku config. Flax module names are
/// pinned against the Python param dump (`SudokuEncoder_0`, `mixer_block_{i}`,
/// `token_mlp`/`channel_mlp` SwiGLU `gate_up`/`down`, `SudokuDecoder_0`,
/// `rm/R_indices`). The LoRA heads (`lora_pre_norm`, `lora_A_dense`,
/// `lora_B_dense`) appear only when `rm_mode = "context"`.
fn sudoku_expected_keys(config: &ExportedConfig) -> BTreeMap<String, Vec<usize>> {
    let m = &config.model;
    let d_model = m.enc_d_model; // 128
    let d_v = m.d_v; // 288
    let d_e = m.d_e; // 32
    let cell_dim = d_v / 9; // 32
    let token_mlp_dim = 9 * SUDOKU_MLP_RATIO; // 18
    let channel_mlp_dim = d_model * SUDOKU_MLP_RATIO; // 256
    let r = m.lora_rank; // 4

    let mut keys: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let dense = |name: &str, i: usize, o: usize, keys: &mut BTreeMap<String, Vec<usize>>| {
        keys.insert(format!("{name}/kernel"), vec![i, o]);
        keys.insert(format!("{name}/bias"), vec![o]);
    };

    // ---- SudokuEncoder ----
    dense("SudokuEncoder_0/token_embed", m.num_classes, d_model, &mut keys);
    keys.insert("SudokuEncoder_0/global_pos_embed/embedding".into(), vec![81, d_model]);
    keys.insert("SudokuEncoder_0/pos_embed".into(), vec![1, 9, d_model]);
    for i in 0..m.enc_num_blocks {
        let blk = format!("SudokuEncoder_0/mixer_block_{i}");
        // token_mlp: SwiGLU(hidden=token_mlp_dim, out=T=9) over T=9.
        dense(&format!("{blk}/token_mlp/gate_up"), 9, 2 * token_mlp_dim, &mut keys);
        dense(&format!("{blk}/token_mlp/down"), token_mlp_dim, 9, &mut keys);
        keys.insert(format!("{blk}/token_norm/scale"), vec![d_model]);
        // channel_mlp: SwiGLU(hidden=channel_mlp_dim, out=C=d_model) over C.
        dense(&format!("{blk}/channel_mlp/gate_up"), d_model, 2 * channel_mlp_dim, &mut keys);
        dense(&format!("{blk}/channel_mlp/down"), channel_mlp_dim, d_model, &mut keys);
        keys.insert(format!("{blk}/channel_norm/scale"), vec![d_model]);
    }
    keys.insert("SudokuEncoder_0/pre_flat_norm/scale".into(), vec![d_model]);
    dense("SudokuEncoder_0/cell_proj", d_model, cell_dim, &mut keys);
    keys.insert("SudokuEncoder_0/cell_norm/scale".into(), vec![cell_dim]);
    dense("SudokuEncoder_0/comm_head/comm_dense", d_v, d_v, &mut keys);
    keys.insert("SudokuEncoder_0/comm_head/comm_norm/scale".into(), vec![d_v]);
    keys.insert("SudokuEncoder_0/comm_head/comm_norm/bias".into(), vec![d_v]);
    for head in ["q_diag_dense", "q_dense"] {
        dense(&format!("SudokuEncoder_0/objective_heads/{head}"), d_model, d_v, &mut keys);
    }
    if m.rm_mode == "context" {
        keys.insert("SudokuEncoder_0/lora_pre_norm/scale".into(), vec![d_model]);
        dense("SudokuEncoder_0/lora_A_dense", d_model, 9 * d_e * r, &mut keys);
        dense("SudokuEncoder_0/lora_B_dense", d_model, 9 * d_v * r, &mut keys);
    }

    // ---- SudokuDecoder ----
    let mut blk_in = cell_dim;
    for (i, &dim) in m.dec_hidden_dims.iter().enumerate() {
        let blk = format!("SudokuDecoder_0/block_{i}");
        keys.insert(format!("{blk}/norm/scale"), vec![blk_in]);
        dense(&format!("{blk}/dense1"), blk_in, dim, &mut keys);
        dense(&format!("{blk}/dense2"), dim, dim, &mut keys);
        if blk_in != dim {
            // residual_proj on the residual path when widths differ.
            dense(&format!("{blk}/residual_proj"), blk_in, dim, &mut keys);
        }
        blk_in = dim;
    }
    dense("SudokuDecoder_0/output_dense", blk_in, m.num_classes, &mut keys);

    // ---- 9 soft_slice base maps + raw rho scalar ----
    keys.insert("rm/R_indices".into(), vec![9, d_e, d_v]);
    keys.insert("rho_raw".into(), vec![]);

    keys
}

/// Sudoku scope guards: this loader only understands the shipped sudoku configs.
fn sudoku_scope_guards(config: &ExportedConfig) -> anyhow::Result<()> {
    let m = &config.model;
    ensure!(m.encoder_arch == "sudoku", "unsupported encoder_arch {:?}", m.encoder_arch);
    ensure!(m.decoder_arch == "sudoku", "unsupported decoder_arch {:?}", m.decoder_arch);
    ensure!(m.rm_sharing == "sudoku", "unsupported rm_sharing {:?}", m.rm_sharing);
    ensure!(m.rm_init == "soft_slice", "unsupported rm_init {:?}", m.rm_init);
    ensure!(m.objective_mode == "non_negative", "unsupported objective_mode {:?}", m.objective_mode);
    ensure!(config.task.task == "sudoku", "unsupported task {:?}", config.task.task);
    Ok(())
}

/// Load sudoku `config.json` + `weights.safetensors` into a ready-to-run model.
/// The default collection is EMA (paper eval convention).
pub fn load_sudoku_model(
    config_path: &Path,
    weights_path: &Path,
    collection: WeightCollection,
) -> anyhow::Result<SudokuSheafModel> {
    let config = load_config(config_path)?;
    config.validate()?;
    sudoku_scope_guards(&config)?;

    let bytes = std::fs::read(weights_path)
        .with_context(|| format!("reading weights {}", weights_path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parsing safetensors {}", weights_path.display()))?;

    let expected = sudoku_expected_keys(&config);
    for prefix in ["params", "ema_params"] {
        Tree::load(&st, prefix, &expected)?;
    }
    for (name, _) in st.iter() {
        ensure!(
            name.starts_with("params/") || name.starts_with("ema_params/"),
            "unexpected top-level key {name:?} (want params/... or ema_params/...)"
        );
    }
    build_sudoku_model(config, &st, collection)
}

/// Materialize the requested collection into the typed sudoku model.
fn build_sudoku_model(
    config: ExportedConfig,
    st: &SafeTensors,
    collection: WeightCollection,
) -> anyhow::Result<SudokuSheafModel> {
    let m = &config.model;
    let expected = sudoku_expected_keys(&config);
    let mut tree = Tree::load(st, collection.prefix(), &expected)?;
    let d_model = m.enc_d_model;

    // Mixer blocks (SwiGLU token/channel sub-MLPs, post-norm).
    let mut mixer_blocks = Vec::with_capacity(m.enc_num_blocks);
    for i in 0..m.enc_num_blocks {
        let blk = format!("SudokuEncoder_0/mixer_block_{i}");
        mixer_blocks.push(MlpMixerBlock {
            token_mlp: SwiGlu {
                gate_up: tree.dense(&format!("{blk}/token_mlp/gate_up")),
                down: tree.dense(&format!("{blk}/token_mlp/down")),
            },
            token_norm: tree.rms_norm(&format!("{blk}/token_norm")),
            channel_mlp: SwiGlu {
                gate_up: tree.dense(&format!("{blk}/channel_mlp/gate_up")),
                down: tree.dense(&format!("{blk}/channel_mlp/down")),
            },
            channel_norm: tree.rms_norm(&format!("{blk}/channel_norm")),
        });
    }

    // Embeddings.
    let (gshape, gdata) = tree.take("SudokuEncoder_0/global_pos_embed/embedding");
    let global_pos_embed =
        Array2::from_shape_vec((gshape[0], gshape[1]), gdata).expect("shape checked at load");
    // pos_embed stored [1, 9, d_model]; drop the leading singleton to [9, d_model].
    let (_pshape, pdata) = tree.take("SudokuEncoder_0/pos_embed");
    let pos_embed = Array2::from_shape_vec((9, d_model), pdata).expect("pos_embed [1,9,d_model]");

    // LoRA heads (context only).
    let lora = if m.rm_mode == "context" {
        Some(SudokuLoraHeads {
            lora_pre_norm: tree.rms_norm("SudokuEncoder_0/lora_pre_norm"),
            lora_a_dense: tree.dense("SudokuEncoder_0/lora_A_dense"),
            lora_b_dense: tree.dense("SudokuEncoder_0/lora_B_dense"),
        })
    } else {
        None
    };

    let encoder = SudokuEncoder {
        params: SudokuEncoderParams {
            token_embed: tree.dense("SudokuEncoder_0/token_embed"),
            global_pos_embed,
            pos_embed,
            mixer_blocks,
            pre_flat_norm: tree.rms_norm("SudokuEncoder_0/pre_flat_norm"),
            cell_proj: tree.dense("SudokuEncoder_0/cell_proj"),
            cell_norm: tree.rms_norm("SudokuEncoder_0/cell_norm"),
            comm_dense: tree.dense("SudokuEncoder_0/comm_head/comm_dense"),
            comm_norm: tree.layer_norm("SudokuEncoder_0/comm_head/comm_norm"),
            q_diag_dense: tree.dense("SudokuEncoder_0/objective_heads/q_diag_dense"),
            q_dense: tree.dense("SudokuEncoder_0/objective_heads/q_dense"),
            lora,
        },
        config: SudokuEncoderConfig {
            d_v: m.d_v,
            d_e: m.d_e,
            d_model,
            num_slots: 9,
            lora_rank: m.lora_rank,
            lora_alpha: m.lora_alpha,
            q_epsilon: m.q_epsilon,
        },
    };

    // Decoder: MLPBlock(s) over the 9 cells (residual_proj when widths differ).
    let cell_dim = m.d_v / 9;
    let mut blocks = Vec::with_capacity(m.dec_hidden_dims.len());
    let mut blk_in = cell_dim;
    for (i, &dim) in m.dec_hidden_dims.iter().enumerate() {
        let blk = format!("SudokuDecoder_0/block_{i}");
        let residual_proj = (blk_in != dim).then(|| tree.dense(&format!("{blk}/residual_proj")));
        blocks.push(MlpBlock {
            norm: tree.rms_norm(&format!("{blk}/norm")),
            dense1: tree.dense(&format!("{blk}/dense1")),
            dense2: tree.dense(&format!("{blk}/dense2")),
            residual_proj,
        });
        blk_in = dim;
    }
    let decoder = SudokuDecoder {
        params: SudokuDecoderParams {
            blocks,
            output_dense: tree.dense("SudokuDecoder_0/output_dense"),
        },
        num_classes: m.num_classes,
    };

    // The 9 soft_slice base maps R_indices [9, d_e, d_v].
    let (rshape, rdata) = tree.take("rm/R_indices");
    let r_indices = Array3::from_shape_vec((rshape[0], rshape[1], rshape[2]), rdata)
        .expect("shape checked at load");

    // rho_raw present but unused: inference reads the export-baked value.
    let _ = tree.take("rho_raw");

    let rho = config.baked.rho;
    Ok(SudokuSheafModel {
        config,
        encoder,
        decoder,
        r_indices,
        cell_ids: crate::views::build_sudoku_cell_indices(),
        rho,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const CONFIG_JSON: &str = r#"{
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
      "baked": { "rho": 0.3125 }
    }"#;

    /// Deterministic per-key fill so the test can spot mixed-up tensors.
    fn fill(seed: usize, len: usize) -> Vec<f32> {
        (0..len).map(|i| (seed as f32) + (i as f32) * 1e-3).collect()
    }

    fn write_fixture(dir: &Path, skip_key: Option<&str>, extra_key: Option<&str>) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), CONFIG_JSON).unwrap();

        let config = load_config(&dir.join("config.json")).unwrap();
        let expected = expected_keys(&config);

        let mut buffers: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for prefix in ["params", "ema_params"] {
            for (i, (suffix, shape)) in expected.iter().enumerate() {
                let full = format!("{prefix}/{suffix}");
                if skip_key == Some(full.as_str()) {
                    continue;
                }
                let len: usize = shape.iter().product();
                // Distinct EMA values: offset raw params by +1000.
                let seed = i + if prefix == "ema_params" { 1000 } else { 0 };
                buffers.push((full, shape.clone(), fill(seed, len)));
            }
        }
        if let Some(extra) = extra_key {
            buffers.push((extra.to_string(), vec![2], vec![0.0, 0.0]));
        }

        let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = buffers
            .into_iter()
            .map(|(name, shape, data)| {
                let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                (name, shape, raw)
            })
            .collect();
        let views: HashMap<&str, TensorView> = bytes
            .iter()
            .map(|(name, shape, raw)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), raw).unwrap(),
                )
            })
            .collect();
        let serialized = safetensors::serialize(&views, &None).unwrap();
        std::fs::write(dir.join("weights.safetensors"), serialized).unwrap();
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("sheaf_io_weights_{tag}"))
    }

    #[test]
    fn loads_typed_structs_from_ema_by_default_layout() {
        let dir = temp_dir("ok");
        write_fixture(&dir, None, None);
        let model = load_maze_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::default(), // EMA
        )
        .unwrap();

        // Shapes per goldens/CONTRACT.md.
        let p = &model.encoder.params;
        assert_eq!(p.input_norm.scale.len(), 54);
        assert_eq!(p.dense.kernel.dim(), (54, 256)); // [in, out], untransposed
        assert_eq!(p.comm_dense.kernel.dim(), (256, 10));
        assert_eq!(p.comm_norm.scale.len(), 10);
        assert_eq!(p.lora_a_dense.kernel.dim(), (256, 160));
        assert_eq!(p.lora_b_dense.kernel.dim(), (256, 320));
        assert_eq!(p.q_diag_dense.bias.len(), 10);
        assert_eq!(p.input_norm.eps, 1e-6);
        assert_eq!(p.comm_norm.eps, 1e-6);

        let d = &model.decoder.params;
        assert_eq!(d.input_norm.scale.len(), 64);
        assert_eq!(d.dense.kernel.dim(), (64, 256));
        assert_eq!(d.output_dense.kernel.dim(), (256, 54));
        assert_eq!(model.decoder.output_shape, (3, 3, 6));

        assert_eq!(model.rm.r_stack.dim(), (8, 5, 10));
        assert_eq!(model.rho, 0.3125); // baked, not rho_raw

        // EMA collection selected: fill seeds were offset by +1000, and the
        // kernel is [in, out] row-major so [0, 1] is element index 1.
        let config = load_config(&dir.join("config.json")).unwrap();
        let expected = expected_keys(&config);
        let dense_idx = expected
            .keys()
            .position(|k| k == "MLPEncoderV2_0/dense/kernel")
            .unwrap();
        let want = (dense_idx + 1000) as f32 + 1e-3;
        assert_eq!(p.dense.kernel[[0, 1]], want);

        // Raw collection loads the un-offset values.
        let raw = load_maze_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::Raw,
        )
        .unwrap();
        assert_eq!(raw.encoder.params.dense.kernel[[0, 1]], dense_idx as f32 + 1e-3);

        // r_stack follows the (N, NE, E, SE, S, SW, W, NW) key order.
        let rn_idx = expected.keys().position(|k| k == "rm/R_N").unwrap();
        assert_eq!(model.rm.r_stack[[0, 0, 0]], (rn_idx + 1000) as f32);
        let rnw_idx = expected.keys().position(|k| k == "rm/R_NW").unwrap();
        assert_eq!(model.rm.r_stack[[7, 0, 0]], (rnw_idx + 1000) as f32);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_key_fails_loudly() {
        let dir = temp_dir("missing");
        write_fixture(&dir, Some("ema_params/rm/R_SE"), None);
        let err = load_maze_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::Ema,
        )
        .err()
        .expect("load should fail on a missing key");
        assert!(err.to_string().contains("missing key"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unexpected_key_fails_loudly() {
        let dir = temp_dir("extra");
        write_fixture(&dir, None, Some("ema_params/MLPEncoderV2_0/bogus"));
        let err = load_maze_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::Ema,
        )
        .err()
        .expect("load should fail on an unexpected key");
        assert!(err.to_string().contains("unexpected key"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn f16_is_widened_to_f32() {
        let dir = temp_dir("f16");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), CONFIG_JSON).unwrap();
        let config = load_config(&dir.join("config.json")).unwrap();
        let expected = expected_keys(&config);

        let mut bytes: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for prefix in ["params", "ema_params"] {
            for (suffix, shape) in expected.iter() {
                let len: usize = shape.iter().product();
                let raw: Vec<u8> = (0..len)
                    .flat_map(|i| half::f16::from_f32((i % 7) as f32 * 0.5).to_le_bytes())
                    .collect();
                bytes.push((format!("{prefix}/{suffix}"), shape.clone(), raw));
            }
        }
        let views: HashMap<&str, TensorView> = bytes
            .iter()
            .map(|(name, shape, raw)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F16, shape.clone(), raw).unwrap(),
                )
            })
            .collect();
        std::fs::write(
            dir.join("weights.safetensors"),
            safetensors::serialize(&views, &None).unwrap(),
        )
        .unwrap();

        let model = load_maze_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::Ema,
        )
        .unwrap();
        assert_eq!(model.encoder.params.dense.kernel[[0, 3]], 1.5);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_matches_contract_count() {
        let config: ExportedConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        let expected = expected_keys(&config);
        assert_eq!(expected.len(), 35, "35 arrays per collection (CONTRACT.md table)");
        let total: usize = expected.values().map(|s| s.iter().product::<usize>()).sum();
        assert_eq!(total, MAZE_PARAM_COUNT); // 181,859 (PLAN.md appendix)
    }

    // ---- mnist loader (Phase B) ----

    // Dims shrunk from the shipped mnist config for a fast fixture; scope
    // strings match `configs/experiment/mnist_sheaf.yaml`.
    const MNIST_CONFIG_JSON: &str = r#"{
      "model": {
        "num_classes": 10, "d_v": 4, "d_e": 3,
        "encoder_arch": "mlp", "enc_hidden_dim": 5, "comm_norm_type": "layernorm",
        "objective_mode": "lasso", "l1_weight": 0.006337180166370117,
        "x_solver": "diagonal_prox",
        "z_solver": "unrolled_cg", "z_mode": "project",
        "cg_iters": 5, "tikhonov_eps": 1e-5,
        "rm_sharing": "global", "rm_init": "orthonormal", "rm_mode": "context",
        "lora_rank": 2, "lora_alpha": 1.0, "lora_init_style": "legacy",
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

    fn write_mnist_fixture(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), MNIST_CONFIG_JSON).unwrap();
        let config = load_config(&dir.join("config.json")).unwrap();
        let expected = mnist_expected_keys(&config);

        let mut buffers: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for prefix in ["params", "ema_params"] {
            for (i, (suffix, shape)) in expected.iter().enumerate() {
                let full = format!("{prefix}/{suffix}");
                let len: usize = shape.iter().product();
                let seed = i + if prefix == "ema_params" { 1000 } else { 0 };
                buffers.push((full, shape.clone(), fill(seed, len)));
            }
        }
        let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = buffers
            .into_iter()
            .map(|(name, shape, data)| {
                let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                (name, shape, raw)
            })
            .collect();
        let views: HashMap<&str, TensorView> = bytes
            .iter()
            .map(|(name, shape, raw)| {
                (name.as_str(), TensorView::new(Dtype::F32, shape.clone(), raw).unwrap())
            })
            .collect();
        std::fs::write(
            dir.join("weights.safetensors"),
            safetensors::serialize(&views, &None).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn loads_mnist_typed_structs() {
        let dir = temp_dir("mnist_ok");
        write_mnist_fixture(&dir);
        let model = load_mnist_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::default(), // EMA
        )
        .unwrap();

        // Encoder: residual trunk (in = 3*3*1 = 9), lasso heads (q_diag, q).
        let p = &model.encoder.params;
        assert_eq!(p.input_proj.kernel.dim(), (9, 5));
        assert_eq!(p.blocks.len(), 1);
        assert_eq!(p.blocks[0].dense1.kernel.dim(), (5, 5));
        assert_eq!(p.blocks[0].dense2.kernel.dim(), (5, 5));
        assert!(p.blocks[0].residual_proj.is_none());
        assert_eq!(p.comm_dense.kernel.dim(), (5, 4)); // hidden -> d_v
        assert_eq!(p.q_diag_dense.kernel.dim(), (5, 4));
        assert_eq!(p.lora_a_dense.kernel.dim(), (5, 8 * 3 * 2)); // K*d_e*r
        assert_eq!(p.lora_b_dense.kernel.dim(), (5, 8 * 4 * 2)); // K*d_v*r
        assert_eq!(model.encoder.config.l1_weight, 0.006_337_180_3);

        // Decoder: bare linear head [d_v, num_classes], x_only readout.
        assert_eq!(model.decoder.params.cls_output.kernel.dim(), (4, 10));
        assert_eq!(model.decoder.readout_mode, ReadoutMode::XOnly);
        assert_eq!(model.decoder.num_classes, 10);

        // One shared base map + baked rho.
        assert_eq!(model.rm_shared.dim(), (3, 4)); // [d_e, d_v]
        assert_eq!(model.rho, 0.12);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mnist_loader_runs_a_forward_and_predicts() {
        use ndarray::Array4;
        use sheaf_core::graph::AgentGraph;
        use std::sync::Arc;

        let dir = temp_dir("mnist_fwd");
        write_mnist_fixture(&dir);
        let model = load_mnist_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::Ema,
        )
        .unwrap();

        // Tiny 6x6 image so the grid is small (stride 3 -> centers {1,4} -> 4
        // agents), B=2, C=1. Full end-to-end: patchify -> graph -> ADMM ->
        // classify -> mean-softmax prediction.
        let (h, w, b) = (6usize, 6, 2);
        let images = Array4::<f32>::from_shape_fn((b, h, w, 1), |(bi, y, x, _)| {
            ((bi + y * 6 + x) % 5) as f32 * 0.1
        });
        let centers = crate::views::grid_agent_centers((h, w), 3, 3);
        assert_eq!(centers.nrows(), 4);
        let edges = crate::views::build_grid_edge_indices(&centers, 3, 8);
        let patches = crate::views::patchify_batch(&images, &centers, 3);
        let positions = centers.mapv(|v| v as f32);
        let graph = Arc::new(AgentGraph::new_grid(edges, positions, model.config.model.num_directions));

        let fwd = model.forward(&patches, graph, 6);
        // logits per iter [K, N, B, C]; final [N, B, C]; prediction [B].
        assert_eq!(fwd.logits_per_iter.dim(), (6, 4, b, 10));
        let logits_final = fwd.logits_final();
        assert_eq!(logits_final.dim(), (4, b, 10));
        assert!(logits_final.iter().all(|v| v.is_finite()));
        let pred = fwd.prediction();
        assert_eq!(pred.len(), b);
        assert!(pred.iter().all(|&c| (0..10).contains(&c)));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- sudoku loader (Phase C) ----

    /// The shipped sudoku config at REAL dims (d_v=288, d_e=32, d_model=128,
    /// 2 blocks, dec_hidden_dims=[256]), fixed variant. Used only to pin the
    /// manifest param count (no weights fixture — the sum is derived).
    const SUDOKU_FULL_JSON: &str = r#"{
      "model": {
        "num_classes": 10, "d_v": 288, "d_e": 32,
        "encoder_arch": "sudoku", "enc_d_model": 128, "enc_num_blocks": 2,
        "comm_norm_type": "layernorm",
        "objective_mode": "non_negative", "x_solver": "diagonal_prox",
        "z_solver": "unrolled_cg", "z_mode": "prox", "gamma": 2.0,
        "cg_iters": 5, "tikhonov_eps": 1e-5,
        "rm_sharing": "sudoku", "rm_init": "soft_slice", "rm_mode": "fixed",
        "lora_rank": 4, "lora_alpha": 1.0, "lora_init_style": "standard",
        "num_directions": 9,
        "relaxation_alpha": 1.0, "z_init": "h", "q_epsilon": 1e-4,
        "decoder_arch": "sudoku", "dec_hidden_dims": [256]
      },
      "task": {
        "task": "sudoku", "patch_size": 3, "stride": 3, "connectivity": 8,
        "num_classes": 10, "k_eval": 50, "loss_window": 2
      },
      "baked": { "rho": 0.28 }
    }"#;

    fn manifest_total(config: &ExportedConfig) -> usize {
        sudoku_expected_keys(config)
            .values()
            .map(|s| s.iter().product::<usize>())
            .sum()
    }

    #[test]
    fn sudoku_manifest_matches_param_pins() {
        // PLAN appendix param pins: 543,025 (fixed) / 2,029,233 (LoRA).
        let fixed: ExportedConfig = serde_json::from_str(SUDOKU_FULL_JSON).unwrap();
        assert_eq!(manifest_total(&fixed), 543_025, "fixed sudoku_sheaf param pin");

        let lora_json = SUDOKU_FULL_JSON.replace(r#""rm_mode": "fixed""#, r#""rm_mode": "context""#);
        let lora: ExportedConfig = serde_json::from_str(&lora_json).unwrap();
        assert_eq!(manifest_total(&lora), 2_029_233, "LoRA sudoku_sheaf_lora param pin");
    }

    /// Shrunk sudoku config for a fast load + forward fixture (d_v=18, d_e=2,
    /// d_model=4, 1 block, dec_hidden_dims=[5]); scope strings match the yaml.
    const SUDOKU_TINY_JSON: &str = r#"{
      "model": {
        "num_classes": 10, "d_v": 18, "d_e": 2,
        "encoder_arch": "sudoku", "enc_d_model": 4, "enc_num_blocks": 1,
        "comm_norm_type": "layernorm",
        "objective_mode": "non_negative", "x_solver": "diagonal_prox",
        "z_solver": "unrolled_cg", "z_mode": "prox", "gamma": 2.0,
        "cg_iters": 5, "tikhonov_eps": 1e-5,
        "rm_sharing": "sudoku", "rm_init": "soft_slice", "rm_mode": "context",
        "lora_rank": 1, "lora_alpha": 1.0, "lora_init_style": "standard",
        "num_directions": 9,
        "relaxation_alpha": 1.0, "z_init": "h", "q_epsilon": 1e-4,
        "decoder_arch": "sudoku", "dec_hidden_dims": [5]
      },
      "task": {
        "task": "sudoku", "patch_size": 3, "stride": 3, "connectivity": 8,
        "num_classes": 10, "k_eval": 50, "loss_window": 2
      },
      "baked": { "rho": 0.28 }
    }"#;

    fn write_sudoku_fixture(dir: &Path, config_json: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), config_json).unwrap();
        let config = load_config(&dir.join("config.json")).unwrap();
        let expected = sudoku_expected_keys(&config);

        let mut buffers: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for prefix in ["params", "ema_params"] {
            for (i, (suffix, shape)) in expected.iter().enumerate() {
                let full = format!("{prefix}/{suffix}");
                let len: usize = shape.iter().product();
                let seed = i + if prefix == "ema_params" { 1000 } else { 0 };
                buffers.push((full, shape.clone(), fill(seed, len)));
            }
        }
        let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = buffers
            .into_iter()
            .map(|(name, shape, data)| {
                let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                (name, shape, raw)
            })
            .collect();
        let views: HashMap<&str, TensorView> = bytes
            .iter()
            .map(|(name, shape, raw)| {
                (name.as_str(), TensorView::new(Dtype::F32, shape.clone(), raw).unwrap())
            })
            .collect();
        std::fs::write(
            dir.join("weights.safetensors"),
            safetensors::serialize(&views, &None).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn loads_sudoku_typed_structs() {
        let dir = temp_dir("sudoku_ok");
        write_sudoku_fixture(&dir, SUDOKU_TINY_JSON);
        let model = load_sudoku_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::default(),
        )
        .unwrap();

        let p = &model.encoder.params;
        assert_eq!(p.token_embed.kernel.dim(), (10, 4)); // [num_classes, d_model]
        assert_eq!(p.global_pos_embed.dim(), (81, 4));
        assert_eq!(p.pos_embed.dim(), (9, 4)); // dropped the leading singleton
        assert_eq!(p.mixer_blocks.len(), 1);
        assert_eq!(p.mixer_blocks[0].token_mlp.gate_up.kernel.dim(), (9, 36)); // [T, 2*18]
        assert_eq!(p.mixer_blocks[0].token_mlp.down.kernel.dim(), (18, 9)); // [18, T]
        assert_eq!(p.mixer_blocks[0].channel_mlp.gate_up.kernel.dim(), (4, 16)); // [C, 2*(C*2)]
        assert_eq!(p.mixer_blocks[0].channel_mlp.down.kernel.dim(), (8, 4)); // [C*2, C]
        assert_eq!(p.cell_proj.kernel.dim(), (4, 2)); // d_model -> cell_dim
        assert_eq!(p.comm_dense.kernel.dim(), (18, 18)); // d_v -> d_v
        assert_eq!(p.q_diag_dense.kernel.dim(), (4, 18)); // d_model -> d_v
        let lora = p.lora.as_ref().expect("context config carries LoRA heads");
        assert_eq!(lora.lora_a_dense.kernel.dim(), (4, 9 * 2)); // [d_model, 9*d_e*r], r=1
        assert_eq!(lora.lora_b_dense.kernel.dim(), (4, 9 * 18)); // [d_model, 9*d_v*r], r=1

        // Decoder: block widens cell_dim(2) -> 5, so residual_proj present.
        let d = &model.decoder.params;
        assert_eq!(d.blocks.len(), 1);
        assert_eq!(d.blocks[0].dense1.kernel.dim(), (2, 5));
        assert!(d.blocks[0].residual_proj.is_some(), "2->5 block needs residual_proj");
        assert_eq!(d.output_dense.kernel.dim(), (5, 10));

        assert_eq!(model.r_indices.dim(), (9, 2, 18)); // [9, d_e, d_v]
        assert_eq!(model.cell_ids.dim(), (27, 9));
        assert_eq!(model.rho, 0.28);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sudoku_fixed_config_has_no_lora_heads() {
        let dir = temp_dir("sudoku_fixed");
        let fixed_json = SUDOKU_TINY_JSON.replace(r#""rm_mode": "context""#, r#""rm_mode": "fixed""#);
        write_sudoku_fixture(&dir, &fixed_json);
        let model = load_sudoku_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::Ema,
        )
        .unwrap();
        assert!(model.encoder.params.lora.is_none(), "fixed config must not load LoRA heads");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sudoku_loader_runs_a_forward_nonneg_and_predicts() {
        use ndarray::Array4;
        use std::sync::Arc;

        let dir = temp_dir("sudoku_fwd");
        write_sudoku_fixture(&dir, SUDOKU_TINY_JSON);
        let model = load_sudoku_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            WeightCollection::Ema,
        )
        .unwrap();

        // Patches [N=27, B=2, 9, num_classes=10] one-hot-ish.
        let b = 2usize;
        let patches = Array4::<f32>::from_shape_fn((27, b, 9, 10), |(n, bi, t, c)| {
            if c == (n + bi + t) % 10 { 1.0 } else { 0.0 }
        });
        let graph = Arc::new(crate::views::build_sudoku_graph());
        let fwd = model.forward(&patches, graph, 6);

        // logits_per_iter [K, N, B, 9, C]; final [N, B, 9, C].
        assert_eq!(fwd.logits_per_iter.dim(), (6, 27, b, 9, 10));
        let logits_final = fwd.logits_final();
        assert_eq!(logits_final.dim(), (27, b, 9, 10));
        assert!(logits_final.iter().all(|v| v.is_finite()));

        // NonNeg objective: every x-iterate is clamped at 0 (lower = 0).
        assert!(
            fwd.history.x.iter().all(|&v| v >= 0.0),
            "NonNeg x-update must clamp at 0"
        );

        // Prediction reassembles [27,B,9,C] -> [B,9,9] digits 0..10.
        let pred = crate::views::sudoku_predict(&logits_final);
        assert_eq!(pred.dim(), (b, 9, 9));
        assert!(pred.iter().all(|&d| (0..10).contains(&d)));

        std::fs::remove_dir_all(&dir).ok();
    }
}
