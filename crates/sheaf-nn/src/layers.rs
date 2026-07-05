//! Shared neural building blocks. Ports `models/layers.py` (inference subset).
//!
//! All layers operate on `[B', feat]` with `B' = N*B` (the parent model owns
//! the `[N, B, ...] <-> [N*B, ...]` reshape contract).

use ndarray::{Array1, Array2};

/// Flax `nn.Dense`: `y = x * kernel + bias`, kernel stored `[in, out]`.
/// The loader must NOT transpose — this matches the safetensors layout.
#[derive(Debug, Clone)]
pub struct Dense {
    pub kernel: Array2<f32>, // [in, out]
    pub bias: Array1<f32>,   // [out]
}

impl Dense {
    /// `[B', in] -> [B', out]` in true fp32.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

/// RMS normalization with learnable per-channel scale:
/// `x / sqrt(mean(x^2, last axis) + 1e-6) * scale`. Ports `RMSNorm`.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub scale: Array1<f32>, // [feat]
    pub eps: f32,           // 1e-6
}

impl RmsNorm {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

/// Flax `nn.LayerNorm` (mean/variance over the last axis, eps 1e-6,
/// learnable scale + bias). Used by `comm_head.comm_norm` and `lora_pre_ln`.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub scale: Array1<f32>, // [feat]
    pub bias: Array1<f32>,  // [feat]
    pub eps: f32,           // 1e-6
}

impl LayerNorm {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

/// GELU, **tanh approximation** (the Flax default `jax.nn.gelu(approximate=True)`):
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`.
pub fn gelu_tanh(x: f32) -> f32 {
    todo!()
}

/// Numerically stable softplus: `max(x, 0) + log1p(exp(-|x|))`.
pub fn softplus(x: f32) -> f32 {
    todo!()
}

/// Inverse of softplus for scalar reparameterizations: clamps input >= 1e-7,
/// identity above 20, else `ln(exp(x) - 1)`. Ports `admm.inverse_softplus`.
pub fn inverse_softplus(x: f32) -> f32 {
    todo!()
}

pub fn sigmoid(x: f32) -> f32 {
    todo!()
}
