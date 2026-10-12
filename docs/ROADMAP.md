# Roadmap

This is the forward-looking companion to `docs/PLAN.md`. The plan records the
design and numerical contract; this file records what is actually supported,
what is implemented but not yet release-qualified, and what remains.

## v0.3.0 release candidate

The supported artifact is the maze inference path:

- Pure-Rust native and WebAssembly inference with no Python runtime.
- Paper-scale maze weights trained at the Table 1 operating point (seed 42,
  50 epochs, 10,000 training mazes), with checkpoint and training history kept
  as provenance.
- 106/106 JAX-to-Rust parity checks over data views, graph slots, encoder and
  LoRA heads, restriction maps, 12 complete ADMM iterates, residuals, decoded
  logits, reassembly, and the final prediction.
- In-distribution and OOD metrics recorded in the README; these are a
  single-seed result, not a multi-seed reproduction claim.
- Native TUI/GIF demo, `viz_export`, rayon batch parallelism, and the live
  browser demo at <https://monumentalsystems.github.io/sheaf-rs/>.
- Published crates `sheaf-core`, `sheaf-nn`, and `sheaf-io`, now prepared for
  the 0.3.0 release with Rust 1.84 MSRV checks, strict public API docs,
  all-features testing, and LICENSE/NOTICE files inside every crate archive.

The public inference scalar remains fixed to fp32, matching the exported-model
contract. A type-changing `f64` Cargo feature was removed because Cargo
features must be additive and downstream feature unification made that API
non-composable. Precision investigations should use isolated f64 reference
calculations and the exported JAX traces.

## Implemented, not yet release-qualified

MNIST and Sudoku model paths exist in `sheaf-nn` and `sheaf-io`, including the
residual MNIST encoder/classifier, Sudoku MLP-Mixer encoder, soft-slice graph,
LoRA gathering, decoder, and evaluation helpers. They remain scaffolding until
each path has all of the following:

1. Trained exported weights with provenance.
2. Task-specific layer and full-trajectory golden fixtures.
3. A CI parity runner against the pinned JAX reference.
4. Fixed-batch evaluation within 0.1% of the Python metrics.
5. A user-facing example that loads and runs the released artifact.

Until those gates pass, release notes and crate documentation must not present
MNIST or Sudoku as validated model releases.

## Next work

### 1. Multi-seed maze characterization

Train the remaining paper-config seeds (123 and 456), export their provenance,
and report mean and spread alongside the existing seed-42 artifact. Keep the
existing 106-check trace as the deterministic portability contract, and add a
small fixed evaluation bundle for release regression testing.

### 2. Qualify MNIST

Generate weights and goldens for `mnist_sheaf`, add a dedicated parity command,
and verify the evaluation quirk exactly: prediction is the argmax of the mean
of per-agent softmax probabilities, not the mean logits.

### 3. Qualify Sudoku

Generate weights and dense sublayer goldens for `sudoku_sheaf` and
`sudoku_sheaf_lora`. Validate the 27-view convention, 243-edge multigraph,
contiguous cell-stalk blocks, and completion metric on empty input cells.

### 4. Optional Rust training

Only undertake the fenced M7 work when explicitly requested. The proposed
target is a small frozen-rho configuration with an explicit unrolled reverse
sweep and checkpoint/recompute through conjugate gradient. Its acceptance
criterion is gradient and training agreement with the same JAX configuration,
not paper-scale parity.

## Release discipline

- Keep `parity_check` as a mandatory CI and release gate.
- Run clippy, all-target/all-feature tests, strict rustdoc, and package
  validation on every pull request.
- Test the declared MSRV (Rust 1.84) independently from stable Rust.
- Confirm every published `.crate` contains `LICENSE` and `NOTICE`.
- Tag only after the workspace version and intra-workspace dependency versions
  agree. The release workflow publishes in dependency order using crates.io
  Trusted Publishing and performs a dry run immediately before each upload.
- Treat training checkpoints as provenance inputs; Rust consumers load only the
  exported safetensors/config artifacts.
