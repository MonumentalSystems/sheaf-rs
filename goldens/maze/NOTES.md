# How these fixtures were produced

Generated on 2026-07-05 from SakanaAI/sheaf-admm at commit
`1e2b5d648361802234348b0b1a7fb3a222128e7d` (the repo's uv env, JAX 0.10.2 /
Flax, **GPU — NVIDIA GB10**, fp32 matmul precision pinned to `highest` by
importing `sheaf_admm`). The model is the shipped
`configs/experiment/maze_sheaf.yaml` config (MLPEncoderV2, l1box_diag, LoRA
rank-4 8-way directional maps, ConcatMLPDecoderV2, prox-mode unrolled CG T=5,
gamma=5.0, rho_init=0.25, d_v=10, d_e=5), **TRAINED at the paper config** —
these are paper-scale weights, not a small-dataset sanity run.

## Training run (provenance)

`scripts/train.py +experiment=maze_sheaf training.seed=42 data.val_splits=[test]`
(Hydra output `outputs/maze_seed42/`), **50 epochs** on
`datasets/maze_std3_19px_10k` (10k train / 1k test + OOD splits, 19×19
in-distribution), seed 42, lr 3e-4, batch 128, K_train=40 (uniform 15..40),
K_eval=100, EMA decay 0.999, GPU. This is the paper maze configuration and
reaches the paper's Table 1 maze operating point. Eval (EMA weights, K=100):

| split | solved | cell_acc | source |
|---|---|---|---|
| test (19×19, in-dist) | 100.00% | 100.00% | history.json (epoch 49) |
| test_ood_2x (37×37) | 98.30% | 99.95% | eval_seed42.json |
| test_ood_2xW (37×37 wide) | 99.80% | 99.99% | eval_seed42.json |
| test_ood_4x (73×73) | 7.50% | 90.25% | eval_seed42.json |
| test_ood_4xW (73×73 wide) | 99.20% | 99.96% | eval_seed42.json |

(In-distribution eval ran during training with `val_splits=[test]` to keep the
73×73 OOD memory spike off the shared box; the OOD splits were evaluated
afterward on the saved EMA checkpoint with `tools/eval_checkpoint.py` at
K_eval=100 and dumped to `goldens/maze/eval_seed42.json`.)

The raw `checkpoint.pkl` (`{"params", "ema_params", "config"}`, pickled Flax
variables dicts), the per-epoch `history.json`, and `eval_seed42.json` are
committed alongside these fixtures as provenance.

## Export

`export_weights.py --repo <sheaf-admm> --out goldens/maze --checkpoint
checkpoint.pkl` loaded the trained `params` and `ema_params` pytrees, asserted
the pickled Hydra `model` block equal to the experiment yaml, and dumped both
collections (70 tensors total, 181,859 params each). The scalar `rho` was
baked at export as `softplus(rho_raw + inverse_softplus(0.25))` on the **EMA**
`rho_raw` (baked rho **0.80284047**; rho is learnable and moved well above its
0.25 init on this longer run). `config.json` carries `"trained": true` plus a
`"training"` provenance block.

`dump_goldens.py --out goldens/maze --dataset datasets/maze_std3_19px_10k`
then loaded the **ema_params** tree from `weights.safetensors`, took the first
2 rows of the `test` split (19×19 mazes) as the fixed batch, built the agent
graph exactly as `training.tasks.MazeTask.prepare` (N=81, E=272, C=6), and ran
`coordinate_history(num_iters=12)` for the K=12 trace. `reassembled_final` is
the overlap-mean reassembly of the final logits and `pred_grid` its argmax.

`convert_f16.py` down-casts the EMA tensors to f16 for
`crates/sheaf-web/assets/weights_ema_f16.safetensors` (35 tensors, 358.8 KB).

## Parity

Rust `parity_check` replays these fixtures **106/106 GREEN**. All ADMM-state
arrays (x/z/y + residuals + consistency) pass the tight widening schedule
(1e-5 at k=0 → 1e-3 at k=K-1). The per-iteration **decoded `logits`** carry an
f32 decoder-GEMM accumulation-order roundoff (~1e-4, independent of k — see
`logits_final` at max_abs≈9.4e-5), so their absolute tolerance is floored at
`LOGITS_ATOL_FLOOR = 1.5e-4` (states and rtol untouched); `pred_grid` is
bitwise exact.
