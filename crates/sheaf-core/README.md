# sheaf-core

Core algorithm for **Sheaf-ADMM** multi-agent coordination, in pure Rust
(`ndarray` only): agent graph with directional slot tables, cellular-sheaf
geometry (fixed and LoRA-factored restriction maps, matrix-free sheaf
Laplacian), closed-form diagonal-prox x-solvers, a batched unrolled
conjugate-gradient z-solver (project and prox modes), and the unrolled ADMM
loop with per-iteration history (primal/dual residuals, consistency RMS).

This is a parity-faithful port of the forward path of
[SakanaAI/sheaf-admm](https://github.com/SakanaAI/sheaf-admm)
(JAX/Flax, *Learning Multi-Agent Coordination via Sheaf-ADMM*, ICML 2026,
[arXiv:2605.31005](https://arxiv.org/abs/2605.31005)) — verified
layer-by-layer and per-iteration against golden traces from the reference
implementation.

Part of the [sheaf-rs](https://github.com/MonumentalSystems/sheaf-rs)
workspace: `sheaf-nn` provides the neural encoders/decoders and full model,
`sheaf-io` provides weight/fixture loading and data views.

Licensed Apache-2.0. Derived from sheaf-admm (Apache-2.0); see the
repository NOTICE for attribution.
