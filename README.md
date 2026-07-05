# sheaf-rs

A Rust port of [SakanaAI/sheaf-admm](https://github.com/SakanaAI/sheaf-admm) —
inference-first, parity-tested against golden traces exported from the JAX
reference. See `docs/PLAN.md` for the full design.

## Current scope

**Maze task only** (`configs/experiment/maze_sheaf.yaml`): MLPEncoderV2,
l1box_diag objective, LoRA rank-4 directional geometry (8-way),
ConcatMLPDecoderV2, prox-mode unrolled CG (T=5), d_v=10, d_e=5.
MNIST, sudoku, the MPNN baseline, WASM, and training are out of scope for now.

Training stays in Python; this repo consumes exported weights
(`weights.safetensors`, EMA by default) and golden fixtures whose exact layout
is pinned in `goldens/CONTRACT.md`.

## Layout

- `crates/sheaf-core` — graph, sheaf geometries (fixed + LoRA, matrix-free
  Laplacian), closed-form x-solvers, unrolled CG z-solver, ADMM loop. ndarray only.
- `crates/sheaf-nn` — inference-only layers (Flax `[in, out]` kernels,
  tanh-GELU, RMSNorm eps 1e-6), MLPEncoderV2, ConcatMLPDecoderV2, restriction-map
  assembly, the end-to-end model, config deserialization.
- `crates/sheaf-io` — safetensors → typed params, .npz readers, data views
  (patchify / grid edges / overlap-mean reassembly), maze generator.
- `crates/sheaf-demo` — native binaries (`parity_check` golden replay; demos later).
- `goldens/` — committed golden fixtures + `CONTRACT.md` (the exporter/consumer
  interface).

```
cargo build          # skeleton compiles; solver/model bodies land per PLAN.md M1–M3
```
