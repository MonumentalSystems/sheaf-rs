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

The shipped goldens are **trained weights** (`"trained": true` in
`config.json`; see `goldens/maze/NOTES.md` for full provenance): the demo
solves mazes — the predicted PATH cells trace the actual shortest
start-to-goal route on generated and golden mazes alike. The consistency RMS
rises off the warm `z_init = h` seed and settles into a near-flat plateau
(prox-mode consensus is soft/gamma-weighted) while the primal and dual
residuals keep falling — exactly matching the Python trace.

## Trained weights

`goldens/maze/` carries an EMA checkpoint trained with the upstream
`scripts/train.py experiment=maze_sheaf`: **30 epochs on
`datasets/maze_small`, CPU, seed 42** — a small-dataset sanity run, **not a
paper reproduction**. The learnable scalar rho moved from its 0.25 init to a
baked `softplus(rho_raw + inverse_softplus(0.25))` = **0.33087**. Final
epoch-29 eval (EMA weights, K_eval=100), from the committed
`goldens/maze/history.json`:

| split | solved | cell_acc |
|---|---|---|
| test (19×19, in-dist) | **99.61%** | **100.00%** |
| test_ood_2x (37×37) | 53.43% | 96.41% |
| test_ood_2xW (37×37 wide) | 88.45% | 99.63% |
| test_ood_4x (73×73) | 5.43% | 91.66% |
| test_ood_4xW (73×73 wide) | 52.60% | 97.02% |

The raw `checkpoint.pkl` + `history.json` are committed alongside the
fixtures as provenance; `tools/export_weights.py --checkpoint` converts a
checkpoint to the Rust-consumable safetensors + config.

## Parity status

`parity_check` replays `goldens/maze/` (exported from the JAX reference at
fp32/`highest` matmul precision, trained EMA weights) through the Rust
pipeline: **106/106 checks green** — views (centers/edges/patches, bitwise),
graph slot tables (exact), encoder heads + LoRA factors (1e-5), assembled
base restriction maps (bitwise), all K=12 ADMM iterates x/z/y + primal/dual
residuals + consistency + per-iteration decoded logits (widening 1e-5 -> 1e-3
f32 tolerance schedule), final overlap-mean reassembly, and the argmax
prediction grid (exact).

## Scope / non-goals

**Maze task only** (`configs/experiment/maze_sheaf.yaml`): MLPEncoderV2,
l1box_diag objective, LoRA rank-4 directional geometry (8-way),
ConcatMLPDecoderV2, prox-mode unrolled CG (T=5), d_v=10, d_e=5.
MNIST, sudoku, the MPNN baseline, and **training** are out of scope —
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
- `crates/sheaf-web` — the wasm-bindgen session for the browser demo: embedded
  f16 EMA weights, `generate_maze` + `SheafSession` (solve / per-iteration
  frames / residuals / agent diagnostics). Logic is plain Rust, tested natively.
- `web/` — the static browser demo (vanilla HTML/JS/CSS, no external
  requests); `web/pkg/` is gitignored wasm-bindgen build output.
- `goldens/` — committed golden fixtures + `CONTRACT.md` (the exporter/consumer
  interface).

## Browser demo

**Live demo: <https://monumentalsystems.github.io/sheaf-rs/>** — the maze
solver running as pure Rust compiled to WebAssembly, entirely in your browser
(no server, no external requests). Generate or hand-edit a 19×19 maze (or a
37×37 out-of-distribution one), press Solve, and scrub/play the ADMM
iterations: the solution path crystallizes out of local disagreement while
the residual chart plummets. Hovering the maze inspects the nearest agent's
patch and per-agent consistency. Deployed automatically by
`.github/workflows/pages.yml` on every push to `main`.

Local development:

```sh
# Toolchain pins: wasm32-unknown-unknown + wasm-bindgen-cli 0.2.126
# (must match the wasm-bindgen crate pin in crates/sheaf-web/Cargo.toml).
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.126 --locked

# Build + bindgen into web/pkg/ (gitignored build output; sheaf_web_bg.wasm
# is ~0.93 MB, embedded f16 EMA weights included).
cargo build --target wasm32-unknown-unknown --release -p sheaf-web
wasm-bindgen --target web --out-dir web/pkg \
    target/wasm32-unknown-unknown/release/sheaf_web.wasm

# Serve the static page (any static server works).
python3 -m http.server 8000 --directory web
# open http://localhost:8000/
```

**Mock mode:** open `http://localhost:8000/?mock=1` to run the UI against
`web/mock/sheaf_web.js` — a pure-JS stand-in for the wasm module (same API
contract, BFS "solver", synthetic residuals) — so the UI is developable
without the Rust toolchain. `node web/test/ui_smoke.mjs` smoke-tests the UI
helpers + mock; point it at a real `wasm-bindgen --target nodejs` build with
`SHEAF_WEB_MODULE=/path/to/nodejs-pkg/sheaf_web.js`.

Other WASM notes:

```sh
# Regenerate the embedded f16 EMA weights (only needed after retraining;
# crates/sheaf-web/assets/weights_ema_f16.safetensors is committed).
uv run --directory <sheaf-admm checkout> python tools/convert_f16.py

# Native mirror of the wasm session logic (no wasm toolchain needed).
cargo test -p sheaf-web
```

## Releasing

Pushing a `v*` tag runs `.github/workflows/release.yml`, which gates on the
workspace tests and the golden `parity_check` binary (the release fails if
either fails), then publishes `sheaf-core` -> `sheaf-nn` -> `sheaf-io` to
crates.io in dependency order and attaches the browser-demo bundle to the
GitHub release.

Publishing authenticates via crates.io **Trusted Publishing (OIDC)** using
[`rust-lang/crates-io-auth-action`](https://github.com/rust-lang/crates-io-auth-action);
no long-lived API token is stored in the repo. One-time manual setup, per
crate (`sheaf-core`, `sheaf-nn`, `sheaf-io`): on crates.io open the crate's
*Settings -> Trusted Publishing* and add this GitHub repository with workflow
filename `release.yml` (environment: `crates-io`). After that is configured
for all three crates, the legacy `CARGO_REGISTRY_TOKEN` repository secret can
be deleted.

## License

Apache-2.0 (see [LICENSE](LICENSE)). This project is derived from
[SakanaAI/sheaf-admm](https://github.com/SakanaAI/sheaf-admm), itself
Apache-2.0 — see [NOTICE](NOTICE) for attribution.
