//! ConcatMLPDecoderV2 (`arch = "mlp_concat_v2"`, the maze decoder).
//! Ports `models/decoder.py`.
//!
//! `RMSNorm(input_norm)(concat([flatten(patch), x])) -> Dense(dense, 256)
//!  -> gelu_tanh -> Dense(output_dense, ph*pw*num_classes)`,
//! reshaped to `[.., ph, pw, num_classes]`. Note the concat order:
//! **patch first, then x** (encoder feats 54 + d_v 10 = 64 in).
//!
//! Shared across agents via the `[N, B, ...] -> [N*B, ...] -> apply ->
//! reshape back` contract (the parent flattens; this mirrors decoder.py which
//! sees a single leading batch axis B' = N*B).

use ndarray::{concatenate, Array2, Array3, Array4, Array5, Axis};

use crate::layers::{gelu_tanh, Dense, MlpBlock, RmsNorm};

#[derive(Debug, Clone)]
pub struct ConcatMlpDecoderV2Params {
    pub input_norm: RmsNorm, // input_norm/scale [64]
    pub dense: Dense,        // dense/{kernel [64,256], bias}
    pub output_dense: Dense, // output_dense/{kernel [256,54], bias}
}

pub struct ConcatMlpDecoderV2 {
    pub params: ConcatMlpDecoderV2Params,
    /// (ph, pw, num_classes) = (3, 3, 6) for the maze. Python derives the
    /// spatial dims from the patch per call (`output_shape = (*patches.shape
    /// [2:-1], num_classes)`); here they are fixed by config and asserted
    /// against the patch at run time.
    pub output_shape: (usize, usize, usize),
}

impl ConcatMlpDecoderV2 {
    /// Decode one agent-state slab: `x [N, B, d_v]` + `patches [N, B, ph, pw, C]`
    /// -> logits `[N, B, ph, pw, num_classes]`. Shared across agents via the
    /// `[N*B, ...]` flatten contract.
    pub fn forward(&self, x: &Array3<f32>, patches: &Array5<f32>) -> Array5<f32> {
        let (n, b, ph, pw, ch) = patches.dim();
        let (nx, bx, d_v) = x.dim();
        assert_eq!((n, b), (nx, bx), "decoder: x/patches agent-batch mismatch");
        let (oh, ow, oc) = self.output_shape;
        assert_eq!(
            (oh, ow),
            (ph, pw),
            "decoder: output spatial dims must match the patch (Python derives them from it)"
        );
        let nb = n * b;

        let patch_flat: Array2<f32> = patches
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((nb, ph * pw * ch))
            .expect("decoder: patch flatten");
        let x_flat: Array2<f32> = x
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((nb, d_v))
            .expect("decoder: x flatten");

        // Concat order is patch FIRST, then x (decoder.py: [patch_flat, x]).
        let h = concatenate(Axis(1), &[patch_flat.view(), x_flat.view()])
            .expect("decoder: concat");
        let h = self.params.input_norm.forward(&h);
        let h = self.params.dense.forward(&h).mapv(gelu_tanh);
        let out = self.params.output_dense.forward(&h);
        assert_eq!(out.shape()[1], oh * ow * oc, "decoder: output_dense width");
        out.into_shape_with_order((n, b, oh, ow, oc))
            .expect("decoder: output reshape")
    }
}

// ===========================================================================
// ClassificationDecoder (`arch = "classification"`, the MNIST head).
// ===========================================================================

/// Per-agent readout: which input the classification head reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadoutMode {
    /// The final agent state `x` alone (the shipped MNIST path).
    XOnly,
    /// `concat([x, flatten(patch)])`.
    Concat,
}

/// Per-agent classification head (`arch = "classification"`). The shipped MNIST
/// config is a bare `Dense(num_classes)` (`linear_head = true`) over `x`
/// (`readout_mode = "x_only"`), producing per-agent logits `[N, B, num_classes]`
/// with NO spatial patch grid (unlike [`ConcatMlpDecoderV2`]). The prediction
/// is the argmax of the MEAN of per-agent softmax over the N agents (see
/// `mnist_mean_softmax_predict`).
#[derive(Debug, Clone)]
pub struct ClassificationDecoderParams {
    /// `cls_output`: Dense(num_classes) — `[readout_dim, num_classes]`.
    pub cls_output: Dense,
}

pub struct ClassificationDecoder {
    pub params: ClassificationDecoderParams,
    pub readout_mode: ReadoutMode,
    /// Number of output classes (= `cls_output` out-dim).
    pub num_classes: usize,
}

impl ClassificationDecoder {
    /// Decode one agent-state slab `x [N, B, d_v]` (+ `patches` for `Concat`)
    /// -> per-agent logits `[N, B, num_classes]`. Shared across agents via the
    /// `[N*B, ...]` flatten contract (only `linear_head = true` is shipped).
    pub fn forward(&self, x: &Array3<f32>, patches: &Array5<f32>) -> Array3<f32> {
        let (n, b, d_v) = x.dim();
        let nb = n * b;
        let x_flat: Array2<f32> = x
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((nb, d_v))
            .expect("classification: x flatten");

        let readout = match self.readout_mode {
            ReadoutMode::XOnly => x_flat,
            ReadoutMode::Concat => {
                let (np, bp, ph, pw, ch) = patches.dim();
                assert_eq!((n, b), (np, bp), "classification: x/patches agent-batch mismatch");
                let patch_flat: Array2<f32> = patches
                    .as_standard_layout()
                    .into_owned()
                    .into_shape_with_order((nb, ph * pw * ch))
                    .expect("classification: patch flatten");
                // concat order is [x, patch] (decoder.py ClassificationDecoder).
                concatenate(Axis(1), &[x_flat.view(), patch_flat.view()])
                    .expect("classification: concat")
            }
        };

        let logits = self.params.cls_output.forward(&readout);
        assert_eq!(logits.shape()[1], self.num_classes, "classification: cls_output width");
        logits
            .into_shape_with_order((n, b, self.num_classes))
            .expect("classification: output reshape")
    }
}

// ===========================================================================
// SudokuDecoder (`arch = "sudoku"`, per-cell digit logits).
// ===========================================================================

/// Weights of the Sudoku decoder. `[B', 9*cell_dim] -> [B', 9, cell_dim]`, then
/// shared [`MlpBlock`]s over the 9 cells (Flax applies Dense over the last axis,
/// sharing across cells), then `Dense(output_channels)`. The shipped config is
/// `dec_hidden_dims = [256]` with `cell_dim = d_v/9 = 32`, so the single block
/// widens 32 -> 256 (its residual path runs through a learned `residual_proj`).
#[derive(Debug, Clone)]
pub struct SudokuDecoderParams {
    pub blocks: Vec<MlpBlock>, // block_0 .. (one 32->256 residual block shipped)
    pub output_dense: Dense,   // output_dense [hidden, output_channels]
}

pub struct SudokuDecoder {
    pub params: SudokuDecoderParams,
    /// Number of output classes (= `output_dense` out-dim), 10 for sudoku.
    pub num_classes: usize,
}

impl SudokuDecoder {
    /// Decode one agent-state slab `x [N, B, d_v]` -> per-cell logits
    /// `[N, B, 9, num_classes]`. Shared across agents AND across the 9 cells via
    /// the `[N*B*9, cell_dim]` flatten contract (Dense over the last axis).
    pub fn forward(&self, x: &Array3<f32>) -> Array4<f32> {
        let (n, b, d_v) = x.dim();
        assert_eq!(d_v % 9, 0, "sudoku decoder d_v must be divisible by 9");
        let cell_dim = d_v / 9;
        let nb = n * b;
        // [N, B, 9*cell_dim] -> [N*B*9, cell_dim] (contiguous cell blocks).
        let mut feats = x
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((nb * 9, cell_dim))
            .expect("sudoku decoder: cell flatten");
        for block in &self.params.blocks {
            feats = block.forward(&feats);
        }
        let logits = self.params.output_dense.forward(&feats);
        assert_eq!(logits.shape()[1], self.num_classes, "sudoku decoder: output width");
        logits
            .into_shape_with_order((n, b, 9, self.num_classes))
            .expect("sudoku decoder: output reshape")
    }
}

/// MNIST prediction (PLAN §5.2 EVAL QUIRK): argmax of the MEAN of per-agent
/// softmax over the N agents — NOT the mean of logits. `logits_final` is the
/// final-iterate per-agent logits `[N, B, num_classes]`; returns `[B]` labels.
pub fn mnist_mean_softmax_predict(logits_final: &Array3<f32>) -> ndarray::Array1<i64> {
    let (n, b, c) = logits_final.dim();
    let mut pred = ndarray::Array1::<i64>::zeros(b);
    for bi in 0..b {
        // Accumulate mean_n softmax(logits[n, bi, :]) over agents.
        let mut agg = vec![0.0f32; c];
        for ni in 0..n {
            // Numerically stable softmax over the class axis.
            let mut max = f32::NEG_INFINITY;
            for ci in 0..c {
                max = max.max(logits_final[[ni, bi, ci]]);
            }
            let mut denom = 0.0f32;
            let mut exps = vec![0.0f32; c];
            for ci in 0..c {
                let e = (logits_final[[ni, bi, ci]] - max).exp();
                exps[ci] = e;
                denom += e;
            }
            for (agg_c, &e) in agg.iter_mut().zip(exps.iter()) {
                *agg_c += e / denom;
            }
        }
        // argmax of the aggregated probabilities (mean over N is monotone in
        // the sum, so dividing by N is unnecessary for the argmax; kept exact
        // to np.argmax first-max tie-break).
        let (mut arg, mut best) = (0usize, f32::NEG_INFINITY);
        for (ci, &val) in agg.iter().enumerate() {
            if val > best {
                best = val;
                arg = ci;
            }
        }
        pred[bi] = arg as i64;
    }
    pred
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, Array3, Array5};

    /// f32 mirror of the decoder math for one row (patch-first concat).
    fn reference_row(patch: &[f32], x: &[f32], dec: &ConcatMlpDecoderV2) -> Vec<f32> {
        let mut v: Vec<f32> = patch.iter().chain(x.iter()).copied().collect();
        let feat = v.len() as f32;
        let ms = v.iter().map(|a| a * a).sum::<f32>() / feat;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for (a, s) in v.iter_mut().zip(dec.params.input_norm.scale.iter()) {
            *a = *a * inv * s;
        }
        let k = &dec.params.dense.kernel;
        let hid: Vec<f32> = (0..k.shape()[1])
            .map(|o| {
                let mut s = dec.params.dense.bias[o];
                for (i, a) in v.iter().enumerate() {
                    s += a * k[[i, o]];
                }
                crate::layers::gelu_tanh(s)
            })
            .collect();
        let k2 = &dec.params.output_dense.kernel;
        (0..k2.shape()[1])
            .map(|o| {
                let mut s = dec.params.output_dense.bias[o];
                for (i, a) in hid.iter().enumerate() {
                    s += a * k2[[i, o]];
                }
                s
            })
            .collect()
    }

    fn tiny_decoder() -> ConcatMlpDecoderV2 {
        // patch (1,2,2)=4 flat + d_v 2 = 6 in; hidden 3; out (1,2,3)=6.
        let in_dim = 6;
        let hidden = 3;
        let out = 6;
        let kernel = Array2::from_shape_fn((in_dim, hidden), |(i, j)| {
            ((i * 31 + j * 17) % 13) as f32 / 13.0 - 0.4
        });
        let kernel2 = Array2::from_shape_fn((hidden, out), |(i, j)| {
            ((i * 7 + j * 11) % 9) as f32 / 9.0 - 0.5
        });
        ConcatMlpDecoderV2 {
            params: ConcatMlpDecoderV2Params {
                input_norm: RmsNorm::new(Array1::from_shape_fn(in_dim, |i| 1.0 + 0.1 * i as f32)),
                dense: Dense::new(kernel, Array1::from_shape_fn(hidden, |i| 0.05 * i as f32)),
                output_dense: Dense::new(kernel2, Array1::from_shape_fn(out, |i| -0.02 * i as f32)),
            },
            output_shape: (1, 2, 3),
        }
    }

    #[test]
    fn shapes_and_values_match_reference() {
        let dec = tiny_decoder();
        let (n, b) = (3, 2);
        let patches = Array5::from_shape_fn((n, b, 1, 2, 2), |(i, j, _, w, c)| {
            0.2 + i as f32 - 0.5 * j as f32 + 0.3 * w as f32 - 0.7 * c as f32
        });
        let x = Array3::from_shape_fn((n, b, 2), |(i, j, d)| {
            -0.1 + 0.4 * i as f32 + 0.2 * j as f32 + 1.1 * d as f32
        });
        let logits = dec.forward(&x, &patches);
        assert_eq!(logits.shape(), &[n, b, 1, 2, 3]);

        // Per-(agent, batch) row must equal the standalone mirror -- this pins
        // both the [N*B] flatten contract and the patch-first concat order.
        for i in 0..n {
            for j in 0..b {
                let patch: Vec<f32> = patches
                    .slice(ndarray::s![i, j, .., .., ..])
                    .iter()
                    .copied()
                    .collect();
                let xv: Vec<f32> = x.slice(ndarray::s![i, j, ..]).iter().copied().collect();
                let expect = reference_row(&patch, &xv, &dec);
                let got: Vec<f32> = logits
                    .slice(ndarray::s![i, j, .., .., ..])
                    .iter()
                    .copied()
                    .collect();
                for (g, e) in got.iter().zip(expect.iter()) {
                    assert_abs_diff_eq!(g, e, epsilon = 1e-5);
                }
            }
        }
    }

    /// Parity against the actual Flax `ConcatMLPDecoderV2` (values dumped from
    /// the Python reference with patterned weights; hidden=3,
    /// output_shape=(1,2,3), patch (1,2,2), d_v=2, B'=2).
    #[test]
    fn matches_flax_reference_dump() {
        let kern = |i: usize, o: usize, s: f32| {
            Array2::from_shape_fn((i, o), |(r, c)| {
                (((r * 31 + c * 17) % 13) as f32 / 13.0 - 0.4) * s
            })
        };
        let bias = |o: usize, off: f32| {
            Array1::from_shape_fn(o, |j| ((j * 7) % 5) as f32 / 5.0 - 0.3 + off)
        };
        let dec = ConcatMlpDecoderV2 {
            params: ConcatMlpDecoderV2Params {
                input_norm: RmsNorm::new(ndarray::array![1.0, 1.1, 0.9, 1.2, 0.8, 1.05]),
                dense: Dense::new(kern(6, 3, 1.0), bias(3, 0.0)),
                output_dense: Dense::new(kern(3, 6, 0.6), bias(6, 0.1)),
            },
            output_shape: (1, 2, 3),
        };
        // N=2, B=1 (the Python dump used a single leading B'=2 axis).
        let patches = Array5::from_shape_fn((2, 1, 1, 2, 2), |(n, _, _, w, c)| {
            0.2 + n as f32 - 0.5 * w as f32 + 0.3 * c as f32
        });
        let x = Array3::from_shape_fn((2, 1, 2), |(n, _, d)| -0.1 + 0.4 * n as f32 + 1.1 * d as f32);
        let logits = dec.forward(&x, &patches);
        assert_eq!(logits.shape(), &[2, 1, 1, 2, 3]);
        let expect = [
            -0.19535841, 0.13287202, 0.61593527, 0.14332421, 0.30102476, -0.17158626,
            -0.08221303, 0.12608501, 0.5638027, 0.06759243, 0.30013758, -0.1960727,
        ];
        for (g, e) in logits.iter().zip(expect.iter()) {
            assert_abs_diff_eq!(g, e, epsilon = 1e-5);
        }
    }

    #[test]
    fn concat_order_is_patch_then_x() {
        // A hidden unit that reads only input slot 4 (the FIRST x element if
        // the order is patch-first) must respond to x and not to the patch.
        let in_dim = 6;
        let mut kernel = Array2::<f32>::zeros((in_dim, 1));
        kernel[[4, 0]] = 1.0;
        let dec = ConcatMlpDecoderV2 {
            params: ConcatMlpDecoderV2Params {
                input_norm: RmsNorm::new(Array1::ones(in_dim)),
                dense: Dense::new(kernel, Array1::zeros(1)),
                output_dense: Dense::new(Array2::ones((1, 6)), Array1::zeros(6)),
            },
            output_shape: (1, 2, 3),
        };
        // Same patch, two different x values with identical norm (|x| equal)
        // -> outputs must differ only through the x slot.
        let patches = Array5::from_elem((1, 1, 1, 2, 2), 1.0);
        let x_pos = Array3::from_shape_vec((1, 1, 2), vec![2.0, 0.0]).unwrap();
        let x_neg = Array3::from_shape_vec((1, 1, 2), vec![-2.0, 0.0]).unwrap();
        let out_pos = dec.forward(&x_pos, &patches);
        let out_neg = dec.forward(&x_neg, &patches);
        // RMS norm is identical for both inputs, so the sign flip of x[0]
        // flips the hidden pre-activation's sign -> different outputs.
        assert!(out_pos[[0, 0, 0, 0, 0]] > out_neg[[0, 0, 0, 0, 0]]);
    }

    // ---- ClassificationDecoder (mnist) ----

    #[test]
    fn classification_x_only_is_linear_over_x() {
        // readout = x, logits = x @ W + b, reshaped to [N, B, num_classes].
        let (d_v, num_classes) = (3usize, 4usize);
        let kernel = Array2::from_shape_fn((d_v, num_classes), |(i, j)| 0.1 * i as f32 - 0.05 * j as f32);
        let bias = Array1::from_shape_fn(num_classes, |j| 0.2 * j as f32);
        let dec = ClassificationDecoder {
            params: ClassificationDecoderParams {
                cls_output: Dense::new(kernel.clone(), bias.clone()),
            },
            readout_mode: ReadoutMode::XOnly,
            num_classes,
        };
        let (n, b) = (3, 2);
        let x = Array3::from_shape_fn((n, b, d_v), |(i, j, d)| 0.3 * i as f32 - 0.2 * j as f32 + 0.5 * d as f32);
        let patches = Array5::<f32>::zeros((n, b, 1, 1, 1)); // ignored by x_only
        let logits = dec.forward(&x, &patches);
        assert_eq!(logits.shape(), &[n, b, num_classes]);
        for i in 0..n {
            for j in 0..b {
                for cc in 0..num_classes {
                    let mut s = bias[cc];
                    for d in 0..d_v {
                        s += x[[i, j, d]] * kernel[[d, cc]];
                    }
                    assert_abs_diff_eq!(logits[[i, j, cc]], s, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn mean_softmax_predict_matches_manual_and_flips_from_logit_mean() {
        // A crisp disagreement between mean-of-logits and mean-of-softmax.
        // agent 0: [0, 0.5] mild class 1; agent 1: [6, 0] saturated class 0.
        // mean-of-logits = [3.0, 0.25] -> argmax 0.
        // softmax: agent0 [0.3775, 0.6225]; agent1 [0.9975, 0.0025].
        // mean-softmax sum = [1.3750, 0.6250] -> argmax 0. (agree)
        // Flip: use TWO mild class-1 agents vs one saturated class-0 agent, but
        // make the saturated one only moderate so softmax mass stays split.
        let logits = Array3::from_shape_vec((3, 1, 2), vec![
            0.0f32, 1.2, // softmax ~ [0.2315, 0.7685]
            0.0, 1.2,    // softmax ~ [0.2315, 0.7685]
            2.5, 0.0,    // softmax ~ [0.9241, 0.0759]
        ])
        .unwrap();
        // mean-of-logits = [2.5/3, 2.4/3] = [0.8333, 0.8000] -> argmax 0.
        // mean-of-softmax sum = [1.3871, 1.6129] -> argmax 1. DISAGREE.
        let pred = mnist_mean_softmax_predict(&logits);
        assert_eq!(pred[0], 1, "prediction must follow the mean of per-agent softmax");

        // Cross-check against a hand-rolled mean-softmax argmax.
        let (n, b, c) = logits.dim();
        let mut agg = vec![0.0f64; c];
        for ni in 0..n {
            let mx = (0..c).map(|ci| logits[[ni, 0, ci]] as f64).fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = (0..c).map(|ci| ((logits[[ni, 0, ci]] as f64) - mx).exp()).collect();
            let denom: f64 = exps.iter().sum();
            for (agg_c, &e) in agg.iter_mut().zip(exps.iter()) {
                *agg_c += e / denom;
            }
        }
        let manual = (0..c).max_by(|&i, &j| agg[i].partial_cmp(&agg[j]).unwrap()).unwrap();
        assert_eq!(pred[0] as usize, manual);
        let _ = b;
    }

    // ---- SudokuDecoder ----

    #[test]
    fn sudoku_decoder_shapes_and_shared_over_cells() {
        // d_v = 18 (cell_dim 2), one width-preserving block, output 10 classes.
        let cell_dim = 2usize;
        let block = MlpBlock {
            norm: RmsNorm::new(Array1::ones(cell_dim)),
            dense1: Dense::new(
                Array2::from_shape_fn((cell_dim, cell_dim), |(i, j)| 0.1 * (i as f32 + 1.0) - 0.05 * j as f32),
                Array1::zeros(cell_dim),
            ),
            dense2: Dense::new(
                Array2::from_shape_fn((cell_dim, cell_dim), |(i, j)| 0.2 * i as f32 - 0.1 * j as f32),
                Array1::from_vec(vec![0.01, -0.02]),
            ),
            residual_proj: None,
        };
        let dec = SudokuDecoder {
            params: SudokuDecoderParams {
                blocks: vec![block],
                output_dense: Dense::new(
                    Array2::from_shape_fn((cell_dim, 10), |(i, j)| 0.03 * i as f32 - 0.01 * j as f32),
                    Array1::from_shape_fn(10, |j| 0.001 * j as f32),
                ),
            },
            num_classes: 10,
        };
        let (n, b) = (3usize, 2);
        let x = Array3::from_shape_fn((n, b, 9 * cell_dim), |(i, j, d)| {
            0.1 * i as f32 - 0.2 * j as f32 + 0.05 * d as f32
        });
        let logits = dec.forward(&x);
        assert_eq!(logits.shape(), &[n, b, 9, 10]);

        // Each cell decodes independently (shared block): the logits for cell k
        // must equal running the block+head on that cell's 2-vector alone.
        for i in 0..n {
            for j in 0..b {
                for k in 0..9 {
                    let cell = Array2::from_shape_fn((1, cell_dim), |(_, d)| x[[i, j, k * cell_dim + d]]);
                    let h = dec.params.blocks[0].forward(&cell);
                    let want = dec.params.output_dense.forward(&h);
                    for cc in 0..10 {
                        assert_abs_diff_eq!(logits[[i, j, k, cc]], want[[0, cc]], epsilon = 1e-6);
                    }
                }
            }
        }
    }
}
