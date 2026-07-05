//! Shared neural building blocks. Ports `models/layers.py` (inference subset).
//!
//! All layers operate on `[B', feat]` with `B' = N*B` (the parent model owns
//! the `[N, B, ...] <-> [N*B, ...]` reshape contract).
//!
//! Numerics contract (PLAN.md §3.4): RMSNorm/LayerNorm eps `1e-6`; GELU is the
//! tanh approximation (Flax default); softplus is the stable
//! `max(x, 0) + log1p(exp(-|x|))` form; all matmuls true fp32.

use ndarray::{Array1, Array2, Axis};

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
    fn sigmoid_basic() {
        assert_abs_diff_eq!(sigmoid(0.0), 0.5, epsilon = 1e-7);
        assert_abs_diff_eq!(sigmoid(-2.0) + sigmoid(2.0), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sigmoid(100.0), 1.0, epsilon = 1e-7);
    }
}
