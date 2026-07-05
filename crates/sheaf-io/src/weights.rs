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
use sheaf_nn::decoder::{ConcatMlpDecoderV2, ConcatMlpDecoderV2Params};
use sheaf_nn::encoder::{MlpEncoderV2, MlpEncoderV2Config, MlpEncoderV2Params};
use sheaf_nn::layers::{Dense, LayerNorm, RmsNorm};
use sheaf_nn::model::{RmParams, SheafAdmmModel};
use sheaf_nn::restriction_maps::direction_names;

/// RMSNorm / LayerNorm epsilon (PLAN.md §3.4 numerics contract).
const NORM_EPS: f32 = 1e-6;

/// Pinned per-collection parameter count for the maze config (CONTRACT.md).
const MAZE_PARAM_COUNT: usize = 181_859;

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
        let count: usize = tensors.values().map(|(_, v)| v.len()).sum();
        ensure!(
            count == MAZE_PARAM_COUNT,
            "collection {prefix:?}: {count} params, expected {MAZE_PARAM_COUNT}"
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

/// Load `config.json` + `weights.safetensors` into a ready-to-run model.
pub fn load_maze_model(
    config_path: &Path,
    weights_path: &Path,
    collection: WeightCollection,
) -> anyhow::Result<SheafAdmmModel> {
    let config = load_config(config_path)?;
    let m = &config.model;
    // Maze scope guards: this loader only understands the shipped maze config.
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

    let mut tree = Tree::load(&st, collection.prefix(), &expected)?;

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
}
