# Roadmap — remaining work (DGX handoff)

Work order for completing the sheaf-rs project. Written for an autonomous
agent with a CUDA machine (DGX). Read `docs/PLAN.md` first — it is the
authoritative design (math mapping §3, numerics contract §3.4, parity
strategy §5, milestones §7, risks §8). This file only says what is left and
in what order.

## Status snapshot (as of v0.2.0)

Done:
- **M1–M3**: workspace, `sheaf-core`/`sheaf-io`/`sheaf-nn`, maze path
  end-to-end, layer-by-layer golden parity vs the JAX reference
  (`parity_check`: 106/106), 90+ workspace tests, CI green.
- **Trained maze weights** (30 epochs, 2k mazes, CPU): 99.61% solved /
  100% cell_acc in-dist, 88.45% ood_2xW. Checkpoint + goldens in
  `goldens/maze/`. **Not a paper reproduction.**
- **M6**: `sheaf-web` WASM crate + browser demo, live at
  https://monumentalsystems.github.io/sheaf-rs/ (Pages deploys from
  `.github/workflows/pages.yml`); demo bundle attached to GitHub releases.
- Releases: crates.io `sheaf-core`/`sheaf-nn`/`sheaf-io` 0.2.0
  (tag-triggered via `.github/workflows/release.yml`).

Remaining: paper-scale training (A), MNIST path (B), Sudoku path (C),
M5 leftovers (D), optional Rust training (E), housekeeping (F).

## Environment setup (once)

```bash
git clone https://github.com/SakanaAI/sheaf-admm   # the JAX reference
cd sheaf-admm && uv sync && uv pip install -U "jax[cuda12]"
# sanity: uv run python -c "import jax; print(jax.devices())"  -> [CudaDevice(...)]
```

The exporter/golden tools in this repo (`tools/export_weights.py`,
`tools/dump_goldens.py`, `tools/convert_f16.py`) run inside that checkout's
uv env. `goldens/maze/NOTES.md` documents exactly how the current fixtures
were produced; replicate that flow for every new config.

**Non-negotiable working conventions** (see PLAN.md §3.4/§5): fp32
everywhere (the reference pins no-TF32 — do NOT enable TF32 on the DGX when
dumping goldens or training for parity targets); never weaken parity
tolerances to make a check pass — find the bug; every new model path gets
layer-by-layer goldens BEFORE debugging end-to-end; eval acceptance is
metrics <0.1% vs Python on a fixed exported batch, replicating the eval
quirks listed in PLAN.md §5.2.

## Phase A — paper-scale maze training

1. Build the full dataset (README of the reference repo: 19×19, 10k/1k,
   `--ood-sizes`).
2. Train `+experiment=maze_sheaf` at paper settings, seeds 42/123/456.
3. Pick the best seed; `tools/export_weights.py --checkpoint ...` (bakes the
   TRAINED rho; writes provenance), `tools/dump_goldens.py`, replace
   `goldens/maze/`, run `parity_check` + full eval acceptance incl. all OOD
   splits at K_eval=100.
4. `tools/convert_f16.py` → refresh `crates/sheaf-web/assets/`, bump the
   README metrics table and the web footer note (it currently says
   "30-epoch small-dataset run"), push → Pages redeploys.

Exit: parity green on paper weights; demo runs paper-quality; README states
per-seed paper-config metrics.

## Phase B — M4a: MNIST path

Reference configs: `mnist_sheaf`. Rust work in `sheaf-nn`/`sheaf-io`:
residual `MLPEncoder` (not V2), linear classification decoder, lasso
objective (l1=0.006337 scalar), `z_mode=project`, shared restriction map,
LoRA r=8 legacy init, d_v=32, d_e=24. Eval quirk: prediction = argmax of the
**mean of per-agent softmax** (NOT mean logits) — PLAN.md §5.2.

1. Train on DGX (build_mnist dataset).
2. Extend the tools for the mnist config (exporter key manifest, goldens
   with K=12 trace on 2 fixed inputs).
3. Implement, add `parity_check --task mnist` (or a second bin), iterate to
   green under the same widening tolerance schedule.
4. `mnist_eval` bin: eval acceptance <0.1% on an exported test batch.

## Phase C — M4b: Sudoku path (the fiddliest — densest goldens)

Reference configs: `sudoku_sheaf`, `sudoku_sheaf_lora`. Rust work: Mixer
encoder (SwiGLU, the √d_model/√2 scale applied AFTER both position
embeddings), d_v=288 with cell k at contiguous stalk block [k·32,(k+1)·32),
9 slot maps via soft_slice, non_negative objective, γ=2, 27 agents
(row/col/box views, [B,27,…] batch-axis convention — do NOT normalize),
the 243-edge multigraph hardcoded as a const table (cross-check vs the
Python builder: each of 81 cells covered exactly 3×), sudoku-LoRA gather.
Param pins to assert: 543,025 (fixed) / 2,029,233 (LoRA).

Same flow as Phase B: train (this is the config that actually needs the
DGX), goldens (extra density: dump every Mixer sublayer), implement, parity,
`sudoku_demo` polish (per-cell digit beliefs sharpening; violation shading).

## Phase D — M5 leftovers (parallel with B/C)

- `viz_export` bin: per-agent [N,K] log-residual heatmap + joint-PCA x/z
  phase portrait (faer SVD) from `AdmmHistory` (PLAN.md §1 viz note).
- `parallel` feature: rayon over the batch axis in sheaf-core (each batch
  element owns its [N,d_v] slab). Keep default single-threaded (wasm).
- `f64` feature build of sheaf-core (roundoff-vs-bug triage, PLAN.md §3.4).
- GIF export for maze_demo (`image`+`gif` crates; plan §6).

## Phase E (optional) — M7: training in Rust

Only if explicitly requested. PLAN.md §4 Tier 3 verbatim: hand-derived
reverse sweep for a frozen-ρ mini-MNIST config, checkpoint+recompute through
CG, on-kink gradient goldens, AdamW+EMA replica. Exit: gradcheck ladder
green; Rust-trained mini config within ~1% of the same config trained in
JAX. Fallback if scope creeps: Burn via the ops facade.

## Phase F — housekeeping

- Migrate crates.io publishing to Trusted Publishing (OIDC) and drop the
  `CARGO_REGISTRY_TOKEN` secret (configure on crates.io per crate, then swap
  the workflow to `rust-lang/crates-io-auth-action`).
- Release procedure per `.github/workflows/release.yml` header comment;
  next feature release: v0.3.0 (Phase A/B), v0.4.0 (Phase C/D).
- Keep `parity_check` as a required release gate — it is the project's
  correctness contract.
