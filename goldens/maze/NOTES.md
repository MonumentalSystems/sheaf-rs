# How these fixtures were produced

Generated on 2026-07-04 from SakanaAI/sheaf-admm at commit
`1e2b5d648361802234348b0b1a7fb3a222128e7d` (the repo's uv env, JAX 0.10.1 /
Flax 0.12.7, CPU, fp32 matmul precision pinned to `highest` by importing
`sheaf_admm`). The model is the shipped `configs/experiment/maze_sheaf.yaml`
config (MLPEncoderV2, l1box_diag, LoRA rank-4 8-way directional maps,
ConcatMLPDecoderV2, prox-mode unrolled CG T=5, gamma=5.0, rho_init=0.25,
d_v=10, d_e=5), **TRAINED** — the weights come from a real training run, not
seed-0 init.

## Training run (provenance)

`scripts/train.py experiment=maze_sheaf` (Hydra output
`outputs/2026-07-04/23-02-17/`), 30 epochs on `datasets/maze_small`
(19×19 mazes), seed 42, lr 3e-4, batch 128, K_train=40 (uniform 15..40),
K_eval=100, EMA decay 0.999, CPU. This is a small-dataset CPU sanity run,
**not a paper reproduction**. Final epoch-29 eval (EMA weights, K=100):

| split | solved | cell_acc |
|---|---|---|
| test (19×19) | 99.61% | 100.00% |
| test_ood_2x (37×37) | 53.43% | 96.41% |
| test_ood_2xW (37×37 wide) | 88.45% | 99.63% |
| test_ood_4x (73×73) | 5.43% | 91.66% |
| test_ood_4xW (73×73 wide) | 52.60% | 97.02% |

The raw `checkpoint.pkl` (`{"params", "ema_params", "config"}`, pickled Flax
variables dicts) and the per-epoch `history.json` from that run are committed
alongside these fixtures as provenance.

## Export

`export_weights.py --checkpoint checkpoint.pkl` loaded the trained `params`
and `ema_params` pytrees from the pickle (stripping each tree's top-level
Flax `params` collection key), asserted the pickled Hydra `model` block equal
to the experiment yaml, and dumped both collections (35 arrays / 181,859
params each). The scalar `rho` was baked at export as
`softplus(rho_raw + inverse_softplus(0.25))` on the **EMA** `rho_raw`
(trained value 0.32267 → baked rho **0.33087394**; rho is learnable and moved
from its 0.25 init). No other offset-softplus scalar exists in this config
(`eta_raw` is GD-solver-only). `config.json` carries `"trained": true` plus a
`"training"` provenance block (dataset, epochs, final metrics).

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
`pred_grid` its argmax, matching `MazeTask.evaluate` (with trained weights the
K=12 pred_grid already contains PATH tokens). All shapes/dtypes were asserted
against `manifest.json` after reload; `trace.npz:rho` was asserted
bitwise-equal to `config.json:baked.rho`.
