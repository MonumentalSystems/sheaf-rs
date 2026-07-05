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
//! `B = lora_B_dense(feats_ln).reshape(B', K, d_v, r)` (row-major reshape —
//! the safetensors contract); no gate (shipped `lora_use_gate = false`).
//!
//! Shared-across-agents contract: the encoder is applied ONCE on
//! `[N*B, ...]` and every `[N*B, ...]` output is reshaped back to `[N, B, ...]`
//! (sheaf_model.py `_encode`; scalars would pass through, but l1box has none).

use ndarray::{Array2, Array3, Array5};

use sheaf_core::solvers::{EncoderOutput, LoraFactors, Objective};

use crate::layers::{gelu_tanh, softplus, Dense, LayerNorm, MlpBlock, RmsNorm};

/// Weights of the maze encoder (loaded by sheaf-io from safetensors; key map
/// in goldens/CONTRACT.md under `MLPEncoderV2_0/...`).
#[derive(Debug, Clone)]
pub struct MlpEncoderV2Params {
    pub input_norm: RmsNorm,      // input_norm/scale [54]
    pub dense: Dense,             // dense/{kernel [54,256], bias}
    pub comm_dense: Dense,        // comm_head/comm_dense
    pub comm_norm: LayerNorm,     // comm_head/comm_norm
    pub q_diag_dense: Dense,      // objective_heads/q_diag_dense
    pub q_dense: Dense,           // objective_heads/q_dense
    pub l1_weight_dense: Dense,   // objective_heads/l1_weight_dense
    pub upper_bound_dense: Dense, // objective_heads/upper_bound_dense
    pub lora_pre_ln: LayerNorm,   // lora_pre_ln
    pub lora_a_dense: Dense,      // lora_A_dense [256, K*d_e*r]
    pub lora_b_dense: Dense,      // lora_B_dense [256, K*d_v*r]
}

/// Config the encoder needs at run time.
#[derive(Debug, Clone, Copy)]
pub struct MlpEncoderV2Config {
    pub d_v: usize,            // 10
    pub d_e: usize,            // 5
    pub num_directions: usize, // K = 8
    pub lora_rank: usize,      // 4
    pub lora_alpha: f32,       // 1.0
    pub q_epsilon: f32,        // 1e-4
}

pub struct MlpEncoderV2 {
    pub params: MlpEncoderV2Params,
    pub config: MlpEncoderV2Config,
}

/// Flatten `[N, B, ...]` (any trailing dims) to `[N*B, feat]`, row-major.
fn flatten_leading2<D: ndarray::Dimension>(
    a: &ndarray::Array<f32, D>,
    nb: usize,
    feat: usize,
) -> Array2<f32> {
    a.as_standard_layout()
        .into_owned()
        .into_shape_with_order((nb, feat))
        .expect("flatten_leading2: shape mismatch")
}

/// Reshape a `[N*B, d]` head output back to `[N, B, d]`.
fn unflatten_nb(a: Array2<f32>, n: usize, b: usize) -> Array3<f32> {
    let d = a.shape()[1];
    a.into_shape_with_order((n, b, d))
        .expect("unflatten_nb: shape mismatch")
}

impl MlpEncoderV2 {
    /// Encode patches `[N, B, ph, pw, C]` -> `EncoderOutput` with
    /// `h [N,B,d_v]`, `Objective::L1Box`, and `LoraFactors` (A `[N,B,K,d_e,r]`,
    /// B `[N,B,K,d_v,r]`).
    ///
    /// Flatten to `[N*B, ph*pw*C]`, apply once (encoder shared across agents),
    /// reshape every `[N*B, ...]` output back to `[N, B, ...]`.
    pub fn forward(&self, patches: &Array5<f32>) -> EncoderOutput {
        let c = &self.config;
        let (n, b, ph, pw, ch) = patches.dim();
        let nb = n * b;
        let flat = flatten_leading2(patches, nb, ph * pw * ch);

        // Trunk: RMSNorm -> Dense(hidden) -> tanh-GELU. (dropout_rate = 0.)
        let feats = self.params.input_norm.forward(&flat);
        let feats = self.params.dense.forward(&feats);
        let feats = feats.mapv(gelu_tanh);

        // comm_head: Dense(d_v) -> LayerNorm (comm_norm_type = "layernorm").
        let h = self.params.comm_norm.forward(&self.params.comm_dense.forward(&feats));

        // Objective heads (l1box_diag) read the trunk feats, not h.
        let q_eps = c.q_epsilon;
        let q_diag = self
            .params
            .q_diag_dense
            .forward(&feats)
            .mapv(|v| softplus(v) + q_eps);
        let q = self.params.q_dense.forward(&feats);
        let l1 = self.params.l1_weight_dense.forward(&feats).mapv(softplus);
        let upper = self.params.upper_bound_dense.forward(&feats).mapv(softplus);

        // LoRA heads read LayerNorm(lora_pre_ln)(trunk feats).
        let feats_ln = self.params.lora_pre_ln.forward(&feats);
        let a_flat = self.params.lora_a_dense.forward(&feats_ln);
        let b_flat = self.params.lora_b_dense.forward(&feats_ln);
        let (k, d_e, d_v, r) = (c.num_directions, c.d_e, c.d_v, c.lora_rank);
        assert_eq!(a_flat.shape()[1], k * d_e * r, "lora_A_dense out dim");
        assert_eq!(b_flat.shape()[1], k * d_v * r, "lora_B_dense out dim");
        // Row-major reshape [N*B, K*d*r] -> [N, B, K, d, r] (CONTRACT.md).
        let a = a_flat
            .into_shape_with_order((n, b, k, d_e, r))
            .expect("lora A reshape");
        let bf = b_flat
            .into_shape_with_order((n, b, k, d_v, r))
            .expect("lora B reshape");

        EncoderOutput {
            h: unflatten_nb(h, n, b),
            objective: Objective::L1Box {
                q_diag: unflatten_nb(q_diag, n, b),
                q: unflatten_nb(q, n, b),
                l1: unflatten_nb(l1, n, b),
                upper: unflatten_nb(upper, n, b),
            },
            lora: Some(LoraFactors {
                a,
                b: bf,
                gate: None, // lora_use_gate = false in every shipped config
                lora_alpha: c.lora_alpha,
            }),
        }
    }
}

// ===========================================================================
// MLPEncoder (`arch = "mlp"`, the residual MNIST encoder).
// ===========================================================================

/// Weights of the residual mnist encoder (`arch = "mlp"`). Trunk is
/// `flatten -> Dense(input_proj) -> MLPBlock x len(hidden_dims)` (mnist ships a
/// single block) `-> comm_head`. Differs from [`MlpEncoderV2`] which is a
/// single `RMSNorm -> Dense -> GELU` trunk with no residual block. The
/// objective heads are the `lasso` subset (`q_diag`, `q`; the L1 weight is a
/// config scalar, not a head), and the LoRA heads read `LayerNorm(lora_pre_ln)`
/// of the block output — identical layout to the maze encoder.
#[derive(Debug, Clone)]
pub struct MlpEncoderParams {
    pub input_proj: Dense,        // input_proj [in, hidden]
    pub blocks: Vec<MlpBlock>,    // block_0 .. (mnist: one width-preserving block)
    pub comm_dense: Dense,        // comm_head/comm_dense
    pub comm_norm: LayerNorm,     // comm_head/comm_norm
    pub q_diag_dense: Dense,      // objective_heads/q_diag_dense
    pub q_dense: Dense,           // objective_heads/q_dense
    pub lora_pre_ln: LayerNorm,   // lora_pre_ln
    pub lora_a_dense: Dense,      // lora_A_dense [hidden, K*d_e*r]
    pub lora_b_dense: Dense,      // lora_B_dense [hidden, K*d_v*r]
}

/// Config the residual encoder needs at run time. `l1_weight` is the `lasso`
/// scalar (from the model config, NOT learned).
#[derive(Debug, Clone, Copy)]
pub struct MlpEncoderConfig {
    pub d_v: usize,            // 32
    pub d_e: usize,            // 24
    pub num_directions: usize, // K = 8
    pub lora_rank: usize,      // 8
    pub lora_alpha: f32,       // 1.0
    pub q_epsilon: f32,        // 1e-4
    pub l1_weight: f32,        // scalar lasso weight (config)
}

pub struct MlpEncoder {
    pub params: MlpEncoderParams,
    pub config: MlpEncoderConfig,
}

impl MlpEncoder {
    /// Encode patches `[N, B, ph, pw, C]` -> `EncoderOutput` with `h [N,B,d_v]`,
    /// `Objective::Lasso { q_diag, q, l1 }`, and directional `LoraFactors`.
    ///
    /// Flatten to `[N*B, ph*pw*C]`, apply once (encoder shared across agents),
    /// reshape every `[N*B, ...]` output back to `[N, B, ...]`.
    pub fn forward(&self, patches: &Array5<f32>) -> EncoderOutput {
        let c = &self.config;
        let (n, b, ph, pw, ch) = patches.dim();
        let nb = n * b;
        let flat = flatten_leading2(patches, nb, ph * pw * ch);

        // Trunk: Dense(input_proj) -> residual MLPBlock(s). (No input RMSNorm,
        // unlike mlp_v2; the first learned map is input_proj.)
        let mut feats = self.params.input_proj.forward(&flat);
        for block in &self.params.blocks {
            feats = block.forward(&feats);
        }

        // comm_head: Dense(d_v) -> LayerNorm (comm_norm_type = "layernorm").
        let h = self.params.comm_norm.forward(&self.params.comm_dense.forward(&feats));

        // Objective heads (lasso) read the trunk `feats`, not h.
        let q_eps = c.q_epsilon;
        let q_diag = self
            .params
            .q_diag_dense
            .forward(&feats)
            .mapv(|v| softplus(v) + q_eps);
        let q = self.params.q_dense.forward(&feats);

        // LoRA heads read LayerNorm(lora_pre_ln)(trunk feats).
        let feats_ln = self.params.lora_pre_ln.forward(&feats);
        let a_flat = self.params.lora_a_dense.forward(&feats_ln);
        let b_flat = self.params.lora_b_dense.forward(&feats_ln);
        let (k, d_e, d_v, r) = (c.num_directions, c.d_e, c.d_v, c.lora_rank);
        assert_eq!(a_flat.shape()[1], k * d_e * r, "lora_A_dense out dim");
        assert_eq!(b_flat.shape()[1], k * d_v * r, "lora_B_dense out dim");
        let a = a_flat
            .into_shape_with_order((n, b, k, d_e, r))
            .expect("lora A reshape");
        let bf = b_flat
            .into_shape_with_order((n, b, k, d_v, r))
            .expect("lora B reshape");

        EncoderOutput {
            h: unflatten_nb(h, n, b),
            objective: Objective::Lasso {
                q_diag: unflatten_nb(q_diag, n, b),
                q: unflatten_nb(q, n, b),
                l1: c.l1_weight, // scalar lasso weight (config, broadcast in the solver)
            },
            lora: Some(LoraFactors {
                a,
                b: bf,
                gate: None, // lora_use_gate = false in every shipped config
                lora_alpha: c.lora_alpha,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, Array5, Axis};

    /// Deterministic pseudo-random weights (tiny LCG; no rand dep).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // top 24 bits -> [-0.5, 0.5)
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        }
        fn mat(&mut self, r: usize, c: usize) -> Array2<f32> {
            Array2::from_shape_fn((r, c), |_| self.next_f32())
        }
        fn vec(&mut self, n: usize) -> Array1<f32> {
            Array1::from_shape_fn(n, |_| self.next_f32())
        }
    }

    /// Tiny maze-shaped encoder: patch (1,1,2) -> in=2, hidden=4, d_v=3,
    /// d_e=2, K=4 directions, rank=1.
    fn tiny_encoder(seed: u64, zero_heads: bool) -> MlpEncoderV2 {
        let mut g = Lcg(seed);
        let (input, hidden, d_v, d_e, k, r) = (2usize, 4usize, 3usize, 2usize, 4usize, 1usize);
        let dense = |g: &mut Lcg, i: usize, o: usize, zero: bool| {
            if zero {
                Dense::new(Array2::zeros((i, o)), Array1::zeros(o))
            } else {
                Dense::new(g.mat(i, o), g.vec(o))
            }
        };
        MlpEncoderV2 {
            params: MlpEncoderV2Params {
                input_norm: RmsNorm::new(Array1::ones(input)),
                dense: dense(&mut g, input, hidden, false),
                comm_dense: dense(&mut g, hidden, d_v, false),
                comm_norm: LayerNorm::new(Array1::ones(d_v), Array1::zeros(d_v)),
                q_diag_dense: dense(&mut g, hidden, d_v, zero_heads),
                q_dense: dense(&mut g, hidden, d_v, zero_heads),
                l1_weight_dense: dense(&mut g, hidden, d_v, zero_heads),
                upper_bound_dense: dense(&mut g, hidden, d_v, zero_heads),
                lora_pre_ln: LayerNorm::new(Array1::ones(hidden), Array1::zeros(hidden)),
                lora_a_dense: dense(&mut g, hidden, k * d_e * r, false),
                lora_b_dense: dense(&mut g, hidden, k * d_v * r, false),
            },
            config: MlpEncoderV2Config {
                d_v,
                d_e,
                num_directions: k,
                lora_rank: r,
                lora_alpha: 1.0,
                q_epsilon: 1e-4,
            },
        }
    }

    fn tiny_patches(n: usize, b: usize) -> Array5<f32> {
        // patch (1,1,2), values distinct per (agent, batch, channel)
        Array5::from_shape_fn((n, b, 1, 1, 2), |(i, j, _, _, c)| {
            0.3 + i as f32 * 0.7 - j as f32 * 0.4 + c as f32 * 1.3
        })
    }

    #[test]
    fn output_shapes_match_contract() {
        let enc = tiny_encoder(7, false);
        let patches = tiny_patches(5, 2);
        let out = enc.forward(&patches);
        assert_eq!(out.h.shape(), &[5, 2, 3]);
        match &out.objective {
            Objective::L1Box { q_diag, q, l1, upper } => {
                assert_eq!(q_diag.shape(), &[5, 2, 3]);
                assert_eq!(q.shape(), &[5, 2, 3]);
                assert_eq!(l1.shape(), &[5, 2, 3]);
                assert_eq!(upper.shape(), &[5, 2, 3]);
                // softplus heads are strictly positive; q_diag >= q_epsilon.
                assert!(q_diag.iter().all(|&v| v >= 1e-4));
                assert!(l1.iter().all(|&v| v > 0.0));
                assert!(upper.iter().all(|&v| v > 0.0));
            }
            _ => panic!("maze encoder must emit Objective::L1Box"),
        }
        let lora = out.lora.expect("rm_mode=context emits LoRA factors");
        assert_eq!(lora.a.shape(), &[5, 2, 4, 2, 1]); // [N,B,K,d_e,r]
        assert_eq!(lora.b.shape(), &[5, 2, 4, 3, 1]); // [N,B,K,d_v,r]
        assert!(lora.gate.is_none());
        assert_eq!(lora.lora_alpha, 1.0);
    }

    #[test]
    fn q_diag_floor_is_softplus_plus_q_epsilon() {
        // Zero head weights -> raw = 0 -> q_diag == softplus(0) + 1e-4 exactly.
        let enc = tiny_encoder(11, true);
        let out = enc.forward(&tiny_patches(2, 1));
        let expected = std::f32::consts::LN_2 + 1e-4;
        match &out.objective {
            Objective::L1Box { q_diag, q, l1, upper } => {
                for &v in q_diag.iter() {
                    assert_abs_diff_eq!(v, expected, epsilon = 1e-7);
                }
                // zero-weight linear head -> q = 0; softplus heads -> ln 2.
                assert!(q.iter().all(|&v| v == 0.0));
                for &v in l1.iter().chain(upper.iter()) {
                    assert_abs_diff_eq!(v, std::f32::consts::LN_2, epsilon = 1e-7);
                }
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn shared_across_agents_reshape_contract() {
        // Applying the encoder to the full [N, B, ...] batch must equal
        // applying it to each agent's [1, B, ...] slab separately.
        let enc = tiny_encoder(23, false);
        let n = 4;
        let patches = tiny_patches(n, 3);
        let full = enc.forward(&patches);
        for i in 0..n {
            let slab = patches
                .index_axis(Axis(0), i)
                .to_owned()
                .insert_axis(Axis(0));
            let single = enc.forward(&slab);
            let h_i = full.h.index_axis(Axis(0), i);
            let h_s = single.h.index_axis(Axis(0), 0);
            for (a, b) in h_i.iter().zip(h_s.iter()) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-6);
            }
            let (fa, sa) = (&full.lora.as_ref().unwrap().a, &single.lora.as_ref().unwrap().a);
            for (a, b) in fa.index_axis(Axis(0), i).iter().zip(sa.index_axis(Axis(0), 0).iter()) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-6);
            }
        }
    }

    /// Parity against the actual Flax `MLPEncoderV2` (values dumped from the
    /// Python reference with patterned weights — see the transcript of
    /// `uv run` against sheaf-admm; hidden=4, comm_dim=3, d_e=2, K=4, r=1,
    /// objective_mode=l1box_diag, comm_norm_type=layernorm).
    #[test]
    fn matches_flax_reference_dump() {
        let kern = |i: usize, o: usize, s: f32, |
         -> Array2<f32> {
            Array2::from_shape_fn((i, o), |(r, c)| (((r * 31 + c * 17) % 13) as f32 / 13.0 - 0.4) * s)
        };
        let bias = |o: usize, off: f32, s: f32| -> Array1<f32> {
            Array1::from_shape_fn(o, |j| (((j * 7) % 5) as f32 / 5.0 - 0.3) * s + off)
        };
        let enc = MlpEncoderV2 {
            params: MlpEncoderV2Params {
                input_norm: RmsNorm::new(ndarray::array![1.0, 1.2]),
                dense: Dense::new(kern(2, 4, 1.0), bias(4, 0.0, 1.0)),
                comm_dense: Dense::new(kern(4, 3, 1.0), bias(3, 0.0, 1.0)),
                comm_norm: LayerNorm::new(
                    ndarray::array![1.0, 1.1, 0.9],
                    ndarray::array![0.0, 0.05, -0.05],
                ),
                q_diag_dense: Dense::new(kern(4, 3, 0.5), bias(3, 0.0, 1.0)),
                q_dense: Dense::new(kern(4, 3, -0.7), bias(3, 0.0, 2.0)),
                l1_weight_dense: Dense::new(kern(4, 3, 0.3), bias(3, -1.0, 1.0)),
                upper_bound_dense: Dense::new(kern(4, 3, 0.2), bias(3, 0.5, 1.0)),
                lora_pre_ln: LayerNorm::new(
                    ndarray::array![1.0, 0.9, 1.1, 1.05],
                    ndarray::array![0.02, -0.02, 0.0, 0.01],
                ),
                lora_a_dense: Dense::new(kern(4, 8, 1.0), bias(8, 0.0, 1.0)),
                lora_b_dense: Dense::new(kern(4, 12, 1.0), bias(12, 0.0, 1.0)),
            },
            config: MlpEncoderV2Config {
                d_v: 3,
                d_e: 2,
                num_directions: 4,
                lora_rank: 1,
                lora_alpha: 1.0,
                q_epsilon: 1e-4,
            },
        };
        // x[b, 0, 0, c] = 0.3 + 0.7*b - 1.1*c, as [N=2, B=1, 1, 1, 2].
        let patches = Array5::from_shape_fn((2, 1, 1, 1, 2), |(n, _, _, _, c)| {
            0.3 + 0.7 * n as f32 - 1.1 * c as f32
        });
        let out = enc.forward(&patches);

        let check = |got: &[f32], expect: &[f32], name: &str| {
            assert_eq!(got.len(), expect.len(), "{name} length");
            for (g, e) in got.iter().zip(expect.iter()) {
                assert_abs_diff_eq!(g, e, epsilon = 1e-5);
            }
        };
        let flat = |a: ndarray::ArrayViewD<f32>| -> Vec<f32> { a.iter().copied().collect() };

        check(
            &flat(out.h.view().into_dyn()),
            &[-0.05034607, -1.2686696, 1.0742228, -0.7055556, -0.72951627, 1.222786],
            "h",
        );
        match &out.objective {
            Objective::L1Box { q_diag, q, l1, upper } => {
                check(
                    &flat(q_diag.view().into_dyn()),
                    &[0.65050924, 0.6507697, 0.99819064, 0.6000957, 0.6950392, 1.0204282],
                    "q_diag",
                );
                check(
                    &flat(q.view().into_dyn()),
                    &[-0.89766204, 0.46157545, 0.94637644, -0.7457681, 0.3349867, 0.8973856],
                    "q",
                );
                check(
                    &flat(l1.view().into_dyn()),
                    &[0.26973206, 0.31002083, 0.48281556, 0.2547201, 0.32477295, 0.49090838],
                    "l1_weight",
                );
                check(
                    &flat(upper.view().into_dyn()),
                    &[0.8457926, 0.98987776, 1.3244853, 0.8212527, 1.0127573, 1.3347793],
                    "upper",
                );
            }
            _ => unreachable!(),
        }
        let lora = out.lora.unwrap();
        check(
            &flat(lora.a.view().into_dyn()),
            &[
                0.72356135, -0.6365961, 0.5467125, 0.13988307, -0.45561427, -0.97954136,
                0.3208648, 1.5615978, 0.50632757, -0.41746548, 0.78691775, -0.44237208,
                -0.23662621, -0.75998324, -0.26153287, 1.3446491,
            ],
            "A",
        );
        check(
            &flat(lora.b.view().into_dyn()),
            &[
                0.72356135, -0.6365961, 0.5467125, 0.13988307, -0.45561427, -0.97954136,
                0.3208648, 1.5615978, -0.79855955, 0.384749, 0.7425795, -0.6175778,
                0.50632757, -0.41746548, 0.78691775, -0.44237208, -0.23662621, -0.75998324,
                -0.26153287, 1.3446491, -0.57914394, 0.62523925, 0.52548826, -0.3983047,
            ],
            "B",
        );
    }

    // ---- MLPEncoder (residual, mnist) ----

    /// Tiny residual encoder: patch (1,1,1) -> in=1, hidden=4, d_v=3, d_e=2,
    /// K=4 directions, rank=1, one width-preserving block.
    fn tiny_mlp_encoder(seed: u64) -> MlpEncoder {
        let mut g = Lcg(seed);
        let (input, hidden, d_v, d_e, k, r) = (1usize, 4usize, 3usize, 2usize, 4usize, 1usize);
        let dense = |g: &mut Lcg, i: usize, o: usize| Dense::new(g.mat(i, o), g.vec(o));
        let block = MlpBlock {
            norm: RmsNorm::new(Array1::ones(hidden)),
            dense1: dense(&mut g, hidden, hidden),
            dense2: dense(&mut g, hidden, hidden),
            residual_proj: None, // width-preserving
        };
        MlpEncoder {
            params: MlpEncoderParams {
                input_proj: dense(&mut g, input, hidden),
                blocks: vec![block],
                comm_dense: dense(&mut g, hidden, d_v),
                comm_norm: LayerNorm::new(Array1::ones(d_v), Array1::zeros(d_v)),
                q_diag_dense: dense(&mut g, hidden, d_v),
                q_dense: dense(&mut g, hidden, d_v),
                lora_pre_ln: LayerNorm::new(Array1::ones(hidden), Array1::zeros(hidden)),
                lora_a_dense: dense(&mut g, hidden, k * d_e * r),
                lora_b_dense: dense(&mut g, hidden, k * d_v * r),
            },
            config: MlpEncoderConfig {
                d_v,
                d_e,
                num_directions: k,
                lora_rank: r,
                lora_alpha: 1.0,
                q_epsilon: 1e-4,
                l1_weight: 0.006337180166370117,
            },
        }
    }

    fn tiny_mlp_patches(n: usize, b: usize) -> Array5<f32> {
        Array5::from_shape_fn((n, b, 1, 1, 1), |(i, j, _, _, _)| {
            0.3 + i as f32 * 0.7 - j as f32 * 0.4
        })
    }

    #[test]
    fn mlp_encoder_output_shapes_and_lasso() {
        let enc = tiny_mlp_encoder(101);
        let out = enc.forward(&tiny_mlp_patches(5, 2));
        assert_eq!(out.h.shape(), &[5, 2, 3]);
        match &out.objective {
            Objective::Lasso { q_diag, q, l1 } => {
                assert_eq!(q_diag.shape(), &[5, 2, 3]);
                assert_eq!(q.shape(), &[5, 2, 3]);
                assert!(q_diag.iter().all(|&v| v >= 1e-4), "q_diag floored at q_epsilon");
                // scalar lasso weight is passed through unchanged.
                assert_abs_diff_eq!(*l1, 0.006337180166370117, epsilon = 0.0);
            }
            _ => panic!("mnist encoder must emit Objective::Lasso"),
        }
        let lora = out.lora.expect("rm_mode=context emits LoRA factors");
        assert_eq!(lora.a.shape(), &[5, 2, 4, 2, 1]); // [N,B,K,d_e,r]
        assert_eq!(lora.b.shape(), &[5, 2, 4, 3, 1]); // [N,B,K,d_v,r]
        assert!(lora.gate.is_none());
    }

    #[test]
    fn mlp_encoder_shared_across_agents() {
        // Whole [N, B, ...] batch must equal per-agent [1, B, ...] application.
        let enc = tiny_mlp_encoder(202);
        let n = 4;
        let patches = tiny_mlp_patches(n, 3);
        let full = enc.forward(&patches);
        for i in 0..n {
            let slab = patches.index_axis(Axis(0), i).to_owned().insert_axis(Axis(0));
            let single = enc.forward(&slab);
            for (a, b) in full
                .h
                .index_axis(Axis(0), i)
                .iter()
                .zip(single.h.index_axis(Axis(0), 0).iter())
            {
                assert_abs_diff_eq!(a, b, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn mlp_encoder_trunk_is_residual_block() {
        // Reproduce the trunk by hand: input_proj then the residual block.
        let enc = tiny_mlp_encoder(303);
        let patches = tiny_mlp_patches(2, 1);
        let flat = Array2::from_shape_fn((2, 1), |(i, _)| 0.3 + i as f32 * 0.7);
        let mut feats = enc.params.input_proj.forward(&flat);
        for block in &enc.params.blocks {
            feats = block.forward(&feats);
        }
        let h_ref = enc.params.comm_norm.forward(&enc.params.comm_dense.forward(&feats));
        let out = enc.forward(&patches);
        for (g, w) in out.h.iter().zip(h_ref.iter()) {
            assert_abs_diff_eq!(g, w, epsilon = 1e-6);
        }
    }

    #[test]
    fn lora_reshape_is_row_major() {
        // Kernel = 0, bias = arange -> every agent's A output is 0..K*d_e*r.
        // Pin a[.., k, i, r] == ((k*d_e + i)*rank + r) (row-major contract).
        let mut enc = tiny_encoder(3, false);
        let k = enc.config.num_directions;
        let (d_e, d_v, rank) = (enc.config.d_e, enc.config.d_v, enc.config.lora_rank);
        enc.params.lora_a_dense = Dense::new(
            Array2::zeros((4, k * d_e * rank)),
            Array1::from_shape_fn(k * d_e * rank, |i| i as f32),
        );
        enc.params.lora_b_dense = Dense::new(
            Array2::zeros((4, k * d_v * rank)),
            Array1::from_shape_fn(k * d_v * rank, |i| 100.0 + i as f32),
        );
        let out = enc.forward(&tiny_patches(2, 2));
        let lora = out.lora.unwrap();
        for n in 0..2 {
            for b in 0..2 {
                for kk in 0..k {
                    for i in 0..d_e {
                        for r in 0..rank {
                            let expect = ((kk * d_e + i) * rank + r) as f32;
                            assert_eq!(lora.a[[n, b, kk, i, r]], expect);
                        }
                    }
                    for j in 0..d_v {
                        for r in 0..rank {
                            let expect = 100.0 + ((kk * d_v + j) * rank + r) as f32;
                            assert_eq!(lora.b[[n, b, kk, j, r]], expect);
                        }
                    }
                }
            }
        }
    }
}
