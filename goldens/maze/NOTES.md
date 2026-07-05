# How these fixtures were produced

Generated on 2026-07-04 from SakanaAI/sheaf-admm at commit
`1e2b5d648361802234348b0b1a7fb3a222128e7d` (the repo's uv env, JAX 0.10.1 /
Flax 0.12.7, CPU, fp32 matmul precision pinned to `highest` by importing
`sheaf_admm`). The model is the shipped `configs/experiment/maze_sheaf.yaml`
config (MLPEncoderV2, l1box_diag, LoRA rank-4 8-way directional maps,
ConcatMLPDecoderV2, prox-mode unrolled CG T=5, gamma=5.0, rho_init=0.25,
d_v=10, d_e=5), **randomly initialized — no training**: `export_weights.py`
mirrors `training.loop.create_train_state` exactly with seed 0
(`init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(0))`,
`model.init({"params": init_rng, "dropout": dropout_rng}, ..., training=False)`),
so `ema_params` is a bytewise copy of `params` (EMA at init, per
`training.optim.init_ema`). The scalar `rho` was baked at export as
`softplus(rho_raw + inverse_softplus(0.25))` on the EMA `rho_raw` (= exactly
0.25, since `rho_raw` inits to zero); no other config overrides were applied.
`dump_goldens.py` then loaded the **ema_params** tree back from
`weights.safetensors` (round-tripping the exact bytes Rust will read), took
the first 2 rows of the `test` split of `datasets/maze_small` in
`iter_test_batches` order as the fixed batch (19x19 mazes, input tokens 1-4,
labels add path token 5), built the agent graph exactly as
`training.tasks.MazeTask.prepare` does (`grid_agent_centers((19,19),2,3)` ->
N=81, `build_grid_edge_indices(..., 8)` -> E=272, `prepare_maze_patches` with
wall pre-pad + one-hot C=6), and ran
`model.apply(..., num_iters=12, node_positions=..., training=False,
method=SheafADMMModel.coordinate_history)` for the K=12 trace. Encoder heads
were dumped by applying the identically-configured `MLPEncoderV2` directly on
the `MLPEncoderV2_0` param subtree (the parent's `_encode` is not
`@nn.compact` and cannot be an apply method) with the parent's
`[N*B] -> [N,B]` reshape. `reassembled_final` is
`reassemble_logits(logits_per_iter[-1], centers, (19,19), 6, mode="mean")` and
`pred_grid` its argmax, matching `MazeTask.evaluate`. All shapes/dtypes were
asserted against `manifest.json` after reload; `trace.npz:rho` was asserted
bitwise-equal to `config.json:baked.rho`.
