//! Shared neural building blocks. Ports `models/layers.py` (inference subset).
//!
//! All layers operate on `[B', feat]` with `B' = N*B` (the parent model owns
//! the `[N, B, ...] <-> [N*B, ...]` reshape contract).
//!
//! Numerics contract (PLAN.md §3.4): RMSNorm/LayerNorm eps `1e-6`; GELU is the
//! tanh approximation (Flax default); softplus is the stable
//! `max(x, 0) + log1p(exp(-|x|))` form; all matmuls true fp32.

use ndarray::{Array1, Array2, Array3, Axis};

/// Flax `nn.Dense`: `y = x * kernel + bias`, kernel stored `[in, out]`.
/// The loader must NOT transpose — this matches the safetensors layout.
#[derive(Debug, Clone)]
pub struct Dense {
    pub kernel: Array2<f32>, // [in, out]
    pub bias: Array1<f32>,   // [out]
}

impl Dense {
    pub fn new(kernel: Array2<f32>, bias: Array1<f32>) -> Self {
        assert_eq!(
            kernel.shape()[1],
            bias.len(),
            "Dense: kernel [in, out] out-dim must match bias"
        );
        Self { kernel, bias }
    }

    /// Input feature width (`kernel.shape[0]`).
    pub fn in_dim(&self) -> usize {
        self.kernel.shape()[0]
    }

    /// Output feature width (`kernel.shape[1]`).
    pub fn out_dim(&self) -> usize {
        self.kernel.shape()[1]
    }

    /// `[B', in] -> [B', out]` in true fp32.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        assert_eq!(
            x.shape()[1],
            self.in_dim(),
            "Dense: input feature dim mismatch"
        );
        x.dot(&self.kernel) + &self.bias
    }
}

/// RMS normalization with learnable per-channel scale:
/// `x / sqrt(mean(x^2, last axis) + 1e-6) * scale`. Ports `RMSNorm`
/// (`rms_norm` + learnable scale, eps UNDER the sqrt).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub scale: Array1<f32>, // [feat]
    pub eps: f32,           // 1e-6
}

impl RmsNorm {
    pub fn new(scale: Array1<f32>) -> Self {
        Self { scale, eps: 1e-6 }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        assert_eq!(x.shape()[1], self.scale.len(), "RmsNorm: feature dim mismatch");
        let feat = x.shape()[1] as f32;
        let mut out = x.clone();
        for mut row in out.axis_iter_mut(Axis(0)) {
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / feat;
            let inv = 1.0 / (mean_sq + self.eps).sqrt();
            for (v, s) in row.iter_mut().zip(self.scale.iter()) {
                *v = *v * inv * s;
            }
        }
        out
    }
}

/// Flax `nn.LayerNorm` (mean/variance over the last axis, eps 1e-6,
/// learnable scale + bias). Used by `comm_head.comm_norm` and `lora_pre_ln`.
///
/// Matches Flax `_compute_stats`: `var = max(0, mean(x^2) - mean(x)^2)`.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub scale: Array1<f32>, // [feat]
    pub bias: Array1<f32>,  // [feat]
    pub eps: f32,           // 1e-6
}

impl LayerNorm {
    pub fn new(scale: Array1<f32>, bias: Array1<f32>) -> Self {
        assert_eq!(scale.len(), bias.len(), "LayerNorm: scale/bias dim mismatch");
        Self { scale, bias, eps: 1e-6 }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        assert_eq!(x.shape()[1], self.scale.len(), "LayerNorm: feature dim mismatch");
        let feat = x.shape()[1] as f32;
        let mut out = x.clone();
        for mut row in out.axis_iter_mut(Axis(0)) {
            let mean = row.iter().sum::<f32>() / feat;
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / feat;
            // Flax clamps at 0: mean2 - mean^2 can go negative in fp roundoff.
            let var = (mean_sq - mean * mean).max(0.0);
            let inv = 1.0 / (var + self.eps).sqrt();
            for ((v, s), b) in row.iter_mut().zip(self.scale.iter()).zip(self.bias.iter()) {
                *v = (*v - mean) * inv * s + b;
            }
        }
        out
    }
}

/// Pre-norm residual MLP block: `x + Dense2(GELU(Dense1(Norm(x))))`.
/// Ports `layers.MLPBlock` (the residual mnist encoder trunk). A learned
/// linear projection sits on the residual path when in/out widths differ
/// (`residual_proj`); the shipped mnist block is width-preserving so it is
/// `None`. `norm` is the block's pre-norm (RMSNorm for `norm_type="rmsnorm"`).
#[derive(Debug, Clone)]
pub struct MlpBlock {
    pub norm: RmsNorm,                 // block/norm (rmsnorm)
    pub dense1: Dense,                 // block/dense1
    pub dense2: Dense,                 // block/dense2 -> out_dim
    pub residual_proj: Option<Dense>,  // block/residual_proj (only when in != out)
}

impl MlpBlock {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let h = self.norm.forward(x);
        let h = self.dense1.forward(&h).mapv(gelu_tanh);
        let h = self.dense2.forward(&h);
        let residual = match &self.residual_proj {
            Some(p) => p.forward(x),
            None => x.clone(),
        };
        residual + h
    }
}

/// GELU, **tanh approximation** (the Flax default `jax.nn.gelu(approximate=True)`):
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`.
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi), f32-rounded
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// Numerically stable softplus: `max(x, 0) + log1p(exp(-|x|))`
/// (the `jax.nn.softplus` = `logaddexp(x, 0)` form).
pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Inverse of softplus for scalar reparameterizations: clamps input >= 1e-7,
/// identity above 20, else `ln(expm1(x))`. Ports `admm.inverse_softplus`.
pub fn inverse_softplus(x: f32) -> f32 {
    let x = x.max(1e-7);
    if x > 20.0 {
        x
    } else {
        x.exp_m1().ln()
    }
}

/// Logistic sigmoid `1 / (1 + exp(-x))` (LoRA gate head; unused by maze).
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// SiLU / swish activation `x * sigmoid(x)` (the `jax.nn.silu` used by SwiGLU).
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Apply an [`RmsNorm`] over the last axis of a `[B, T, C]` tensor (each of the
/// `B*T` rows normalized over `C`). Flax `RMSNorm` normalizes the last axis and
/// shares the scale across all leading axes.
pub fn rms_norm_last_axis(norm: &RmsNorm, x: &Array3<f32>) -> Array3<f32> {
    let (b, t, c) = x.dim();
    let flat = x
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((b * t, c))
        .expect("rms_norm_last_axis: reshape");
    norm.forward(&flat)
        .into_shape_with_order((b, t, c))
        .expect("rms_norm_last_axis: unshape")
}

/// SwiGLU MLP (Shazeer 2020): `down(silu(gate) * up)` with a fused `[gate, up]`
/// projection. Ports `layers.SwiGLU`. `gate_up` splits into `[gate | up]` on the
/// last axis (first half gate, second half up — `jnp.split(.., 2, -1)`).
#[derive(Debug, Clone)]
pub struct SwiGlu {
    pub gate_up: Dense, // [in, 2*hidden]
    pub down: Dense,    // [hidden, out]
}

impl SwiGlu {
    /// `[rows, in] -> [rows, out]`. Flax applies Dense over the last axis, so the
    /// parent flattens any leading `[B, T]` into `rows` before calling.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let gate_up = self.gate_up.forward(x); // [rows, 2*hidden]
        let two_h = gate_up.shape()[1];
        assert_eq!(two_h % 2, 0, "SwiGLU gate_up width must be even");
        let hidden = two_h / 2;
        // First half = gate, second half = up (jnp.split(.., 2, -1)).
        let gate = gate_up.slice(ndarray::s![.., 0..hidden]);
        let up = gate_up.slice(ndarray::s![.., hidden..two_h]);
        let mut h = gate.mapv(silu);
        h *= &up; // silu(gate) * up
        self.down.forward(&h)
    }
}

/// TRM-style **post-norm** MLP-Mixer block over `[B, T, C]`. Ports
/// `layers.MLPMixerBlock` (SwiGLU sub-MLPs — the Sudoku encoder trunk):
/// - token mixing: `x = RMSNorm(x + swapaxes(TokenMLP(swapaxes(x)))) `;
/// - channel mixing: `x = RMSNorm(x + ChannelMLP(x))`.
///
/// The token MLP mixes over the `T` axis (out_dim = `T`); the channel MLP mixes
/// over the `C` axis (out_dim = `C`). Both norms are applied AFTER the residual
/// add (post-norm), unlike the pre-norm [`MlpBlock`].
#[derive(Debug, Clone)]
pub struct MlpMixerBlock {
    pub token_mlp: SwiGlu,   // hidden = token_mlp_dim, out = T
    pub token_norm: RmsNorm, // [C]
    pub channel_mlp: SwiGlu, // hidden = channel_mlp_dim, out = C
    pub channel_norm: RmsNorm, // [C]
}

impl MlpMixerBlock {
    /// `[B, T, C] -> [B, T, C]`.
    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let (b, t, c) = x.dim();

        // ---- token mixing (mix across T) ----
        // y = swapaxes(x, 1, 2) -> [B, C, T]; flatten [B*C, T]; SwiGLU(out=T).
        let xt = x
            .view()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((b * c, t))
            .expect("mixer token swap reshape");
        let yt = self.token_mlp.forward(&xt); // [B*C, T]
        let yt = yt
            .into_shape_with_order((b, c, t))
            .expect("mixer token unshape")
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_owned(); // [B, T, C]
        let x = rms_norm_last_axis(&self.token_norm, &(x + &yt));

        // ---- channel mixing (mix across C) ----
        let x_flat = x
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((b * t, c))
            .expect("mixer channel reshape");
        let yc = self
            .channel_mlp
            .forward(&x_flat)
            .into_shape_with_order((b, t, c))
            .expect("mixer channel unshape");
        rms_norm_last_axis(&self.channel_norm, &(&x + &yc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn dense_matches_manual_matmul() {
        // y = x W + b with W stored [in, out] (Flax layout, NOT transposed).
        let d = Dense::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], array![0.5, -0.5, 0.0]);
        let x = array![[1.0, 1.0], [2.0, -1.0]];
        let y = d.forward(&x);
        assert_eq!(y.shape(), &[2, 3]);
        assert_abs_diff_eq!(y[[0, 0]], 1.0 * 1.0 + 1.0 * 4.0 + 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(y[[0, 1]], 2.0 + 5.0 - 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(y[[1, 2]], 2.0 * 3.0 - 1.0 * 6.0, epsilon = 1e-6);
    }

    #[test]
    fn rms_norm_matches_reference() {
        let norm = RmsNorm::new(array![1.0, 2.0, 0.5]);
        let x = array![[1.0f32, 2.0, 3.0]];
        let y = norm.forward(&x);
        // f64 mirror of x / sqrt(mean(x^2) + 1e-6) * scale
        let ms = (1.0 + 4.0 + 9.0) / 3.0;
        let inv = 1.0 / ((ms + 1e-6f64).sqrt());
        assert_abs_diff_eq!(y[[0, 0]], (1.0 * inv * 1.0) as f32, epsilon = 1e-6);
        assert_abs_diff_eq!(y[[0, 1]], (2.0 * inv * 2.0) as f32, epsilon = 1e-6);
        assert_abs_diff_eq!(y[[0, 2]], (3.0 * inv * 0.5) as f32, epsilon = 1e-6);
    }

    #[test]
    fn layer_norm_matches_reference() {
        let norm = LayerNorm::new(array![1.0, 1.0, 2.0], array![0.0, 0.1, -0.1]);
        let x = array![[1.0f32, 2.0, 6.0]];
        let y = norm.forward(&x);
        // f64 mirror of the Flax stats: var = mean(x^2) - mean(x)^2.
        let mean = 3.0f64;
        let var = (1.0 + 4.0 + 36.0) / 3.0 - 9.0;
        let inv = 1.0 / (var + 1e-6f64).sqrt();
        assert_abs_diff_eq!(y[[0, 0]], ((1.0 - mean) * inv) as f32, epsilon = 1e-5);
        assert_abs_diff_eq!(y[[0, 1]], ((2.0 - mean) * inv + 0.1) as f32, epsilon = 1e-5);
        assert_abs_diff_eq!(y[[0, 2]], ((6.0 - mean) * inv * 2.0 - 0.1) as f32, epsilon = 1e-5);
    }

    #[test]
    fn layer_norm_constant_input_outputs_bias() {
        // var = 0 -> (x - mean) = 0 -> output is exactly the bias.
        let norm = LayerNorm::new(array![1.0, 1.0], array![0.25, -0.75]);
        let y = norm.forward(&array![[3.0f32, 3.0]]);
        assert_abs_diff_eq!(y[[0, 0]], 0.25, epsilon = 1e-7);
        assert_abs_diff_eq!(y[[0, 1]], -0.75, epsilon = 1e-7);
    }

    #[test]
    fn gelu_is_tanh_approximation() {
        assert_eq!(gelu_tanh(0.0), 0.0);
        // f64 mirror of 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3))) at x=1, x=-2.
        let expect = |x: f64| 0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
        assert_abs_diff_eq!(gelu_tanh(1.0), expect(1.0) as f32, epsilon = 1e-6);
        assert_abs_diff_eq!(gelu_tanh(-2.0), expect(-2.0) as f32, epsilon = 1e-6);
        // Large |x| asymptotes.
        assert_abs_diff_eq!(gelu_tanh(10.0), 10.0, epsilon = 1e-5);
        assert_abs_diff_eq!(gelu_tanh(-10.0), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn softplus_is_stable() {
        assert_abs_diff_eq!(softplus(0.0), std::f32::consts::LN_2, epsilon = 1e-7);
        assert_eq!(softplus(1000.0), 1000.0); // no overflow
        assert_eq!(softplus(-1000.0), 0.0); // no underflow to NaN
        assert_abs_diff_eq!(softplus(1.0), 1.313_261_7, epsilon = 1e-6);
    }

    #[test]
    fn inverse_softplus_roundtrip() {
        for &v in &[0.01f32, 0.25, 1.0, 5.0, 19.0] {
            assert_abs_diff_eq!(softplus(inverse_softplus(v)), v, epsilon = 1e-5);
        }
        // Identity above 20; clamp below 1e-7 stays finite.
        assert_eq!(inverse_softplus(25.0), 25.0);
        assert!(inverse_softplus(0.0).is_finite());
    }

    #[test]
    fn mlp_block_is_prenorm_residual() {
        // Width-preserving (in == out) block: no residual_proj, so the output
        // is x + dense2(gelu(dense1(rmsnorm(x)))). Mirror the math by hand.
        let d = 3usize;
        let block = MlpBlock {
            norm: RmsNorm::new(array![1.0, 1.1, 0.9]),
            dense1: Dense::new(
                Array2::from_shape_fn((d, d), |(i, j)| 0.1 * (i as f32 + 1.0) - 0.05 * j as f32),
                array![0.01, -0.02, 0.03],
            ),
            dense2: Dense::new(
                Array2::from_shape_fn((d, d), |(i, j)| 0.2 * i as f32 - 0.1 * j as f32),
                array![-0.01, 0.02, 0.0],
            ),
            residual_proj: None,
        };
        let x = array![[0.5f32, -1.0, 2.0], [1.5, 0.2, -0.3]];
        let got = block.forward(&x);
        let h = block.norm.forward(&x);
        let h = block.dense1.forward(&h).mapv(gelu_tanh);
        let h = block.dense2.forward(&h);
        let want = &x + &h;
        for (g, w) in got.iter().zip(want.iter()) {
            assert_abs_diff_eq!(g, w, epsilon = 1e-6);
        }
    }

    #[test]
    fn mlp_block_residual_proj_when_widths_differ() {
        // in=2, out=3 -> residual path runs through residual_proj.
        let block = MlpBlock {
            norm: RmsNorm::new(array![1.0, 1.0]),
            dense1: Dense::new(Array2::zeros((2, 3)), array![0.0, 0.0, 0.0]),
            dense2: Dense::new(Array2::zeros((3, 3)), array![0.0, 0.0, 0.0]),
            residual_proj: Some(Dense::new(
                Array2::from_shape_fn((2, 3), |(i, j)| (i + j) as f32),
                array![1.0, 2.0, 3.0],
            )),
        };
        let x = array![[1.0f32, 1.0]];
        // dense1/dense2 are zero -> h = gelu(0) then 0, so output == residual_proj(x).
        let got = block.forward(&x);
        let want = block.residual_proj.as_ref().unwrap().forward(&x);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_abs_diff_eq!(g, w, epsilon = 1e-6);
        }
    }

    #[test]
    fn sigmoid_basic() {
        assert_abs_diff_eq!(sigmoid(0.0), 0.5, epsilon = 1e-7);
        assert_abs_diff_eq!(sigmoid(-2.0) + sigmoid(2.0), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sigmoid(100.0), 1.0, epsilon = 1e-7);
    }

    #[test]
    fn silu_is_x_times_sigmoid() {
        assert_eq!(silu(0.0), 0.0);
        assert_abs_diff_eq!(silu(1.0), 1.0 * sigmoid(1.0), epsilon = 1e-7);
        assert_abs_diff_eq!(silu(-2.0), -2.0 * sigmoid(-2.0), epsilon = 1e-7);
        // f64 mirror x/(1+e^-x).
        let expect = |x: f64| x / (1.0 + (-x).exp());
        assert_abs_diff_eq!(silu(3.0), expect(3.0) as f32, epsilon = 1e-6);
    }

    #[test]
    fn swiglu_matches_manual() {
        // in=2, hidden=3, out=2. down(silu(gate) * up), gate=first half.
        let gate_up = Dense::new(
            Array2::from_shape_fn((2, 6), |(i, j)| 0.1 * (i as f32 + 1.0) - 0.05 * j as f32),
            Array1::from_shape_fn(6, |j| 0.02 * j as f32),
        );
        let down = Dense::new(
            Array2::from_shape_fn((3, 2), |(i, j)| 0.2 * i as f32 - 0.1 * j as f32),
            Array1::from_vec(vec![0.01, -0.01]),
        );
        let sw = SwiGlu { gate_up, down };
        let x = array![[0.5f32, -1.0], [1.5, 0.2]];
        let got = sw.forward(&x);
        assert_eq!(got.shape(), &[2, 2]);
        // Manual: gate_up -> split -> silu(gate)*up -> down.
        let gu = sw.gate_up.forward(&x);
        let mut h = Array2::<f32>::zeros((2, 3));
        for r in 0..2 {
            for k in 0..3 {
                h[[r, k]] = silu(gu[[r, k]]) * gu[[r, k + 3]];
            }
        }
        let want = sw.down.forward(&h);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_abs_diff_eq!(g, w, epsilon = 1e-6);
        }
    }

    #[test]
    fn swiglu_gate_up_split_is_first_then_second_half() {
        // gate_up all-zero except gate column 0 (index 0) and up column 0 (index
        // hidden). Only when gate>0 (silu>0) AND up!=0 does hidden channel 0 fire.
        let mut kern = Array2::<f32>::zeros((1, 4)); // in=1, 2*hidden=4 (hidden=2)
        kern[[0, 0]] = 1.0; // gate[0] = x
        kern[[0, 2]] = 1.0; // up[0]   = x  (index hidden=2)
        let gate_up = Dense::new(kern, Array1::zeros(4));
        let down = Dense::new(array![[1.0f32], [0.0]], Array1::zeros(1)); // read hidden[0]
        let sw = SwiGlu { gate_up, down };
        // x=2 -> gate[0]=2, up[0]=2 -> hidden[0]=silu(2)*2 > 0.
        let out = sw.forward(&array![[2.0f32]]);
        assert_abs_diff_eq!(out[[0, 0]], silu(2.0) * 2.0, epsilon = 1e-6);
    }

    #[test]
    fn mixer_block_post_norm_shape_and_reference() {
        // Tiny [B=2, T=3, C=4] block; token hidden=5, channel hidden=6.
        let dense = |i: usize, o: usize, s: f32| {
            Dense::new(
                Array2::from_shape_fn((i, o), |(r, c)| (((r * 5 + c * 3) % 7) as f32 / 7.0 - 0.4) * s),
                Array1::from_shape_fn(o, |j| 0.01 * j as f32),
            )
        };
        let (t, c) = (3usize, 4usize);
        let block = MlpMixerBlock {
            token_mlp: SwiGlu { gate_up: dense(t, 2 * 5, 1.0), down: dense(5, t, 1.0) },
            token_norm: RmsNorm::new(Array1::from_shape_fn(c, |i| 1.0 + 0.1 * i as f32)),
            channel_mlp: SwiGlu { gate_up: dense(c, 2 * 6, 1.0), down: dense(6, c, 1.0) },
            channel_norm: RmsNorm::new(Array1::from_shape_fn(c, |i| 0.9 + 0.05 * i as f32)),
        };
        let x = Array3::from_shape_fn((2, t, c), |(b, tt, cc)| {
            0.3 + 0.2 * b as f32 - 0.1 * tt as f32 + 0.4 * cc as f32
        });
        let got = block.forward(&x);
        assert_eq!(got.shape(), &[2, t, c]);

        // Independent mirror of the post-norm token+channel mixing.
        // Token mixing over T.
        let xt = x
            .view()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((2 * c, t))
            .unwrap();
        let yt = block
            .token_mlp
            .forward(&xt)
            .into_shape_with_order((2, c, t))
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_owned();
        let x1 = rms_norm_last_axis(&block.token_norm, &(&x + &yt));
        let x1f = x1.clone().into_shape_with_order((2 * t, c)).unwrap();
        let yc = block.channel_mlp.forward(&x1f).into_shape_with_order((2, t, c)).unwrap();
        let want = rms_norm_last_axis(&block.channel_norm, &(&x1 + &yc));
        for (g, w) in got.iter().zip(want.iter()) {
            assert_abs_diff_eq!(g, w, epsilon = 1e-6);
        }
    }
}
