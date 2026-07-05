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

use ndarray::{concatenate, Array2, Array3, Array5, Axis};

use crate::layers::{gelu_tanh, Dense, RmsNorm};

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
}
