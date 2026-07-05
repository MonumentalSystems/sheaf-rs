# sheaf-nn

Inference-only neural components for **Sheaf-ADMM** multi-agent coordination:
layers (Dense with Flax `[in, out]` kernels, RMSNorm/LayerNorm, tanh-GELU,
stable softplus), the MLP encoder with objective heads (quadratic / lasso /
non-negative / L1-box), LoRA restriction-map factor heads, 8-way directional
restriction-map assembly, the concat-MLP decoder, and `SheafAdmmModel` wiring
encode → sheaf geometry → unrolled ADMM (`sheaf-core`) → decode.

Parity-faithful port of
[SakanaAI/sheaf-admm](https://github.com/SakanaAI/sheaf-admm)
(JAX/Flax, *Learning Multi-Agent Coordination via Sheaf-ADMM*, ICML 2026,
[arXiv:2605.31005](https://arxiv.org/abs/2605.31005)), verified against
golden traces from the reference implementation. Training stays in the JAX
reference; weights are exported to safetensors and loaded via `sheaf-io`.

Part of the [sheaf-rs](https://github.com/MonumentalSystems/sheaf-rs)
workspace.

Licensed Apache-2.0. Derived from sheaf-admm (Apache-2.0); see the
repository NOTICE for attribution.
