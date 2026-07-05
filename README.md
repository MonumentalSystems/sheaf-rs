# sheaf-rs

A Rust port of [SakanaAI/sheaf-admm](https://github.com/SakanaAI/sheaf-admm) —
inference-first, parity-tested layer-by-layer against golden traces exported
from the JAX reference. See [docs/PLAN.md](docs/PLAN.md) for the full design.

## What this is

Sheaf-ADMM treats a grid of local agents (one per 3x3 maze patch) as a
cellular sheaf: each agent encodes its patch into a d_v=10 state, learned
directional restriction maps (LoRA rank-4, 8-way shared) say how neighboring
states must agree on d_e=5 edge stalks, and an ADMM loop (diagonal-prox
x-step for the l1box_diag objective, unrolled-CG T=5 z-step) drives the
agents to a globally consistent state that a shared decoder reads out as
per-cell class logits. This repo re-implements the full inference path in
pure Rust (ndarray, f32) as a literal, parity-faithful transcription of the
Python semantics.

## Quickstart

```sh
# Animated demo: watch the ADMM coordination loop refine per-cell predictions
# on a fresh 19x19 maze (ANSI colors; --no-anim for CI/logs).
cargo run --release -p sheaf-demo --bin maze_demo
cargo run --release -p sheaf-demo --bin maze_demo -- --no-anim
cargo run --release -p sheaf-demo --bin maze_demo -- --maze-from-batch --k 12

# Golden parity replay (layer-by-layer + per-iteration checks; exits non-zero
# on any divergence).
cargo run --release -p sheaf-demo --bin parity_check

cargo test --workspace
```

`maze_demo` flags: `--weights` / `--config` (default `goldens/maze/...`),
`--maze-from-batch` (replay the golden batch input) or `--seed N` (generate a
fresh maze), `--k N` ADMM iterations (default 40), `--fps F`, `--no-anim`.
A captured `--no-anim` run lives in [docs/demo_output.txt](docs/demo_output.txt).

The shipped goldens are **seed-0 random-init weights** (see
`goldens/maze/NOTES.md`), so the demo prints an honest banner: it shows ADMM
coordination dynamics (residuals converging, agents reaching agreement), not
maze solving. The consistency RMS rises for a few iterations off the warm
`z_init = h` seed, then decreases monotonically — exactly matching the Python
trace. Exporting a trained checkpoint with `"trained": true` in `config.json`
suppresses the banner.

## Parity status

`parity_check` replays `goldens/maze/` (exported from the JAX reference at
fp32/`highest` matmul precision) through the Rust pipeline: **106/106 checks
green** — views (centers/edges/patches, bitwise), graph slot tables (exact),
encoder heads + LoRA factors (1e-5), assembled base restriction maps
(bitwise), all K=12 ADMM iterates x/z/y + primal/dual residuals + consistency
+ per-iteration decoded logits (widening 1e-5 -> 1e-3 f32 tolerance schedule),
final overlap-mean reassembly, and the argmax prediction grid (exact).

Caveat: the goldens come from **randomly initialized** (seed 0) weights, so
parity is proven for the full forward path, but the trained-checkpoint
eval-metric acceptance (PLAN.md §5.2, K_eval=100) is pending a trained export.

## Scope / non-goals

**Maze task only** (`configs/experiment/maze_sheaf.yaml`): MLPEncoderV2,
l1box_diag objective, LoRA rank-4 directional geometry (8-way),
ConcatMLPDecoderV2, prox-mode unrolled CG (T=5), d_v=10, d_e=5.
MNIST, sudoku, the MPNN baseline, WASM, and **training** are out of scope —
training stays in Python; this repo consumes exported weights
(`weights.safetensors`, EMA collection by default) and golden fixtures whose
exact layout is pinned in `goldens/CONTRACT.md`.

## Layout

- `crates/sheaf-core` — graph, sheaf geometries (fixed + LoRA, matrix-free
  Laplacian), closed-form x-solvers, unrolled CG z-solver, ADMM loop. ndarray only.
- `crates/sheaf-nn` — inference-only layers (Flax `[in, out]` kernels,
  tanh-GELU, RMSNorm eps 1e-6), MLPEncoderV2, ConcatMLPDecoderV2, restriction-map
  assembly, the end-to-end model, config deserialization.
- `crates/sheaf-io` — safetensors → typed params, .npz readers, data views
  (patchify / grid edges / overlap-mean reassembly), maze generator.
- `crates/sheaf-demo` — native binaries: `maze_demo` (ANSI animation),
  `parity_check` (golden replay for CI).
- `goldens/` — committed golden fixtures + `CONTRACT.md` (the exporter/consumer
  interface).
