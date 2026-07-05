//! The end-to-end maze model: encode -> geometry -> ADMM -> decode.
//! Ports `models/sheaf_model.py` (maze / `coordinate_history` path).
//!
//! Reassembly of per-agent logits into the global `[B, H, W, C]` prediction
//! (overlap-mean) lives in `sheaf-io::views`; this model exposes the per-agent
//! logits `[.., N, B, ph, pw, num_classes]` that feed it.

use std::sync::Arc;

use ndarray::{Array3, Array4, Array5, Array6, Axis};

use sheaf_core::admm::{run_admm, run_admm_history, AdmmHistory, AdmmParams, AdmmState, XSolverKind};
use sheaf_core::geometry::{FixedGeometry, LoraGeometry, SheafGeometry};
use sheaf_core::graph::AgentGraph;
use sheaf_core::solvers::{EncoderOutput, UnrolledCgParams, ZMode};
use sheaf_core::tensor::{NodeState, RestrictionMaps};

use crate::config::ExportedConfig;
use crate::decoder::ConcatMlpDecoderV2;
use crate::encoder::MlpEncoderV2;
use crate::restriction_maps::build_directional_restriction_maps;

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
    pub logits_per_iter: Array6<f32>,
    pub final_state: AdmmState,
    /// The assembled base maps `[E, 2, d_e, d_v]` (golden cross-check).
    pub base_restriction_maps: RestrictionMaps,
}

impl MazeForward {
    /// Per-agent logits of the final iterate, `[N, B, ph, pw, num_classes]`
    /// (`logits_per_iter[K-1]`; the maze prediction is the loss-window final
    /// iterate). Feed this to the sheaf-io overlap-mean reassembly.
    pub fn logits_final(&self) -> Array5<f32> {
        let k = self.logits_per_iter.shape()[0];
        assert!(k > 0, "empty history");
        self.logits_per_iter.index_axis(Axis(0), k - 1).to_owned()
    }
}

impl SheafAdmmModel {
    /// Everything `forward` / `forward_window` share: encode, assemble the
    /// base maps, build the (LoRA or fixed) geometry, pick `z_init`, and bake
    /// the solver params. Mirrors sheaf_model.py `_setup_admm`.
    fn setup_admm(
        &self,
        patches: &Array5<f32>,
        graph: &Arc<AgentGraph>,
        num_iters: usize,
    ) -> (
        EncoderOutput,
        Box<dyn SheafGeometry>,
        RestrictionMaps,
        UnrolledCgParams,
        AdmmParams,
        XSolverKind,
        NodeState,
    ) {
        let m = &self.config.model;
        assert_eq!(
            self.rm.r_stack.shape()[0],
            m.num_directions,
            "r_stack must hold one base map per direction"
        );
        assert_eq!(patches.shape()[0], graph.num_nodes, "patches N != graph N");

        let enc_out = self.encoder.forward(patches);
        let base = build_directional_restriction_maps(&self.rm.r_stack, graph);

        let geometry: Box<dyn SheafGeometry> = match m.rm_mode.as_str() {
            "context" => {
                let lora = enc_out
                    .lora
                    .as_ref()
                    .expect("rm_mode=context requires encoder LoRA factors");
                Box::new(LoraGeometry::create_directional(
                    graph.clone(),
                    base.clone(),
                    &lora.a,
                    &lora.b,
                    lora.gate.as_ref(),
                    lora.lora_alpha,
                ))
            }
            "fixed" => Box::new(FixedGeometry::new(graph.clone(), base.clone())),
            other => panic!("unknown rm_mode {other:?} (fixed|context)"),
        };

        // z^0 seed: "h" (encoder embedding, shipped default) | "zeros".
        let z_init = if m.z_init == "h" {
            enc_out.h.clone()
        } else {
            NodeState::zeros(enc_out.h.raw_dim())
        };

        let z_params = UnrolledCgParams {
            mode: match m.z_mode.as_str() {
                "prox" => ZMode::Prox,
                "project" => ZMode::Project,
                other => panic!("unknown z_mode {other:?} (prox|project)"),
            },
            gamma: m.gamma,
            num_iters: m.cg_iters,
            tikhonov_eps: m.tikhonov_eps,
        };
        let admm_params = AdmmParams {
            rho: self.rho, // export-baked softplus(rho_raw + inv_softplus(rho_init))
            alpha: m.relaxation_alpha,
            gamma: m.gamma,
            k: num_iters,
        };
        let x_solver = match m.x_solver.as_str() {
            "diagonal_prox" => XSolverKind::DiagonalProx,
            "simple" => XSolverKind::Simple,
            other => panic!("unknown x_solver {other:?} (diagonal_prox|simple)"),
        };
        (enc_out, geometry, base, z_params, admm_params, x_solver, z_init)
    }

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
        let (enc_out, geometry, base, z_params, admm_params, x_solver, z_init) =
            self.setup_admm(patches, &graph, num_iters);
        let (final_state, history) = run_admm_history(
            &enc_out,
            geometry.as_ref(),
            x_solver,
            &z_params,
            &admm_params,
            &z_init,
        );
        let logits_per_iter = self.decode_x_iterates(&history.x, patches);
        MazeForward {
            history,
            logits_per_iter,
            final_state,
            base_restriction_maps: base,
        }
    }

    /// Training-style forward (mirrors Python `__call__`): run `num_iters`
    /// ADMM steps and decode only the last `loss_window` x-iterates.
    /// Returns the final state and logits `[W, N, B, ph, pw, num_classes]`
    /// (oldest first, W clamped to `num_iters`).
    pub fn forward_window(
        &self,
        patches: &Array5<f32>,
        graph: Arc<AgentGraph>,
        num_iters: usize,
        loss_window: usize,
    ) -> (AdmmState, Array6<f32>) {
        let (enc_out, geometry, _base, z_params, admm_params, x_solver, z_init) =
            self.setup_admm(patches, &graph, num_iters);
        let (final_state, x_window) = run_admm(
            &enc_out,
            geometry.as_ref(),
            x_solver,
            &z_params,
            &admm_params,
            &z_init,
            loss_window,
        );
        let stacked = stack_states(&x_window);
        let logits = self.decode_x_iterates(&stacked, patches);
        (final_state, logits)
    }

    /// Decode a stack of x-iterates `[K, N, B, d_v]` through the shared
    /// decoder -> per-agent logits `[K, N, B, ph, pw, num_classes]`
    /// (sheaf_model.py `_decode_window`, one decoder call per slab).
    pub fn decode_x_iterates(&self, xs: &Array4<f32>, patches: &Array5<f32>) -> Array6<f32> {
        let (k, n, b, _d_v) = xs.dim();
        let (oh, ow, oc) = self.decoder.output_shape;
        let mut out = Array6::zeros((k, n, b, oh, ow, oc));
        for i in 0..k {
            let x_i = xs.index_axis(Axis(0), i).to_owned();
            let logits = self.decoder.forward(&x_i, patches);
            out.index_axis_mut(Axis(0), i).assign(&logits);
        }
        out
    }
}

/// Stack a window of `[N, B, d_v]` states into `[W, N, B, d_v]` (oldest first).
fn stack_states(window: &[NodeState]) -> Array4<f32> {
    assert!(!window.is_empty(), "empty x-window");
    let (n, b, d) = window[0].dim();
    let mut out = Array4::zeros((window.len(), n, b, d));
    for (i, s) in window.iter().enumerate() {
        out.index_axis_mut(Axis(0), i).assign(s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, Array4, Array5};

    use crate::decoder::ConcatMlpDecoderV2Params;
    use crate::layers::{Dense, RmsNorm};

    fn tiny_decoder() -> ConcatMlpDecoderV2 {
        // patch (1,1,2) flat 2 + d_v 2 = 4 in; hidden 3; out (1,1,2) = 2.
        ConcatMlpDecoderV2 {
            params: ConcatMlpDecoderV2Params {
                input_norm: RmsNorm::new(Array1::ones(4)),
                dense: Dense::new(
                    Array2::from_shape_fn((4, 3), |(i, j)| 0.1 * (i as f32 + 1.0) - 0.05 * j as f32),
                    Array1::zeros(3),
                ),
                output_dense: Dense::new(
                    Array2::from_shape_fn((3, 2), |(i, j)| 0.2 * i as f32 + 0.1 * j as f32),
                    Array1::from_vec(vec![0.01, -0.01]),
                ),
            },
            output_shape: (1, 1, 2),
        }
    }

    #[test]
    fn decode_x_iterates_stacks_per_iteration_decodes() {
        // decode_x_iterates([K,N,B,d]) slab k must equal decoder.forward(x[k]).
        let decoder = tiny_decoder();
        let (k, n, b, d_v) = (3, 2, 2, 2);
        let xs = Array4::from_shape_fn((k, n, b, d_v), |(a, i, j, d)| {
            0.1 * a as f32 + 0.3 * i as f32 - 0.2 * j as f32 + 0.7 * d as f32
        });
        let patches = Array5::from_shape_fn((n, b, 1, 1, 2), |(i, j, _, _, c)| {
            0.5 * i as f32 - 0.1 * j as f32 + c as f32
        });
        // Model with only the decoder exercised: build a throwaway wrapper by
        // calling the free logic directly through a minimal model is overkill;
        // replicate via the decoder to pin the stacking semantics.
        let mut expected = Array6::zeros((k, n, b, 1, 1, 2));
        for a in 0..k {
            let x_a = xs.index_axis(Axis(0), a).to_owned();
            expected
                .index_axis_mut(Axis(0), a)
                .assign(&decoder.forward(&x_a, &patches));
        }
        // Same computation through a model instance.
        let model = SheafAdmmModel {
            config: crate::config::ExportedConfig::from_json(TINY_CONFIG).unwrap(),
            encoder: dummy_encoder(),
            decoder: tiny_decoder(),
            rm: RmParams {
                r_stack: Array3::zeros((8, 1, 2)),
            },
            rho: 0.25,
        };
        let got = model.decode_x_iterates(&xs, &patches);
        assert_eq!(got.shape(), &[k, n, b, 1, 1, 2]);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(g, e, epsilon = 1e-6);
        }
        // logits_final()-style read: last slab equals the K-1 decode.
        let last = got.index_axis(Axis(0), k - 1);
        let x_last = xs.index_axis(Axis(0), k - 1).to_owned();
        let dec_last = decoder.forward(&x_last, &patches);
        for (g, e) in last.iter().zip(dec_last.iter()) {
            assert_abs_diff_eq!(g, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn stack_states_is_oldest_first() {
        let mk = |v: f32| NodeState::from_elem((1, 1, 2), v);
        let stacked = stack_states(&[mk(1.0), mk(2.0), mk(3.0)]);
        assert_eq!(stacked.shape(), &[3, 1, 1, 2]);
        assert_eq!(stacked[[0, 0, 0, 0]], 1.0);
        assert_eq!(stacked[[2, 0, 0, 1]], 3.0);
    }

    // A syntactically valid maze config for constructing model instances in
    // tests (dims shrunk; scope strings match the shipped config).
    const TINY_CONFIG: &str = r#"{
      "model": {
        "num_classes": 2, "d_v": 2, "d_e": 1,
        "encoder_arch": "mlp_v2", "enc_hidden_dim": 4, "comm_norm_type": "layernorm",
        "objective_mode": "l1box_diag", "x_solver": "diagonal_prox",
        "z_solver": "unrolled_cg", "z_mode": "prox", "gamma": 5.0,
        "cg_iters": 5, "tikhonov_eps": 1e-5, "prox_init": "legacy",
        "rm_sharing": "directional", "rm_init": "orthonormal", "rm_mode": "context",
        "lora_rank": 1, "lora_alpha": 1.0, "lora_use_gate": false,
        "lora_init_style": "standard", "num_directions": 8,
        "relaxation_alpha": 1.0, "z_init": "h", "q_epsilon": 1e-4,
        "l1_init": 0.01, "upper_init": 1.0,
        "decoder_arch": "mlp_concat_v2", "dec_hidden_dim": 4
      },
      "task": {
        "task": "maze", "patch_size": 3, "stride": 2, "connectivity": 8,
        "num_classes": 2, "k_eval": 100, "loss_window": 4
      },
      "baked": { "rho": 0.25 }
    }"#;

    fn dummy_encoder() -> MlpEncoderV2 {
        use crate::encoder::{MlpEncoderV2Config, MlpEncoderV2Params};
        use crate::layers::LayerNorm;
        let dense = |i: usize, o: usize| Dense::new(Array2::zeros((i, o)), Array1::zeros(o));
        MlpEncoderV2 {
            params: MlpEncoderV2Params {
                input_norm: RmsNorm::new(Array1::ones(2)),
                dense: dense(2, 4),
                comm_dense: dense(4, 2),
                comm_norm: LayerNorm::new(Array1::ones(2), Array1::zeros(2)),
                q_diag_dense: dense(4, 2),
                q_dense: dense(4, 2),
                l1_weight_dense: dense(4, 2),
                upper_bound_dense: dense(4, 2),
                lora_pre_ln: LayerNorm::new(Array1::ones(4), Array1::zeros(4)),
                lora_a_dense: dense(4, 8),  // K*d_e*r = 8*1*1
                lora_b_dense: dense(4, 16), // K*d_v*r = 8*2*1
            },
            config: MlpEncoderV2Config {
                d_v: 2,
                d_e: 1,
                num_directions: 8,
                lora_rank: 1,
                lora_alpha: 1.0,
                q_epsilon: 1e-4,
            },
        }
    }
}
