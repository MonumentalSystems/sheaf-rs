# sheaf-io

I/O and data preparation for **Sheaf-ADMM** multi-agent coordination:
safetensors weight loading into typed `sheaf-nn` model structs (exhaustive
key manifest, EMA collection by default, Flax `[in, out]` kernels preserved),
`.npz` golden-fixture reading, data views (grid agent centers, 4/8-connected
edge building, wall-token pre-pad one-hot patchify, overlap-mean logit
reassembly), and a faithful port of the reference maze generator
(odd-lattice DFS carve with BFS minimum-path-length acceptance).

Part of the parity-tested Rust port of
[SakanaAI/sheaf-admm](https://github.com/SakanaAI/sheaf-admm)
(JAX/Flax, *Learning Multi-Agent Coordination via Sheaf-ADMM*, ICML 2026,
[arXiv:2605.31005](https://arxiv.org/abs/2605.31005)). See the
[sheaf-rs](https://github.com/MonumentalSystems/sheaf-rs) workspace.

Licensed Apache-2.0. Derived from sheaf-admm (Apache-2.0); see the
repository NOTICE for attribution.
