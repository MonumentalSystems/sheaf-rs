# Changelog

All notable changes to the public Rust crates are recorded here. This project
uses semantic versioning while the crates remain pre-1.0.

## 0.3.0 - Unreleased

### Model artifact

- Replace the small CPU-trained maze artifact with the seed-42 paper-scale
  checkpoint (50 epochs, 10,000 training mazes) and its provenance.
- Refresh the fp32 JAX goldens; the Rust replay passes 106/106 checks across
  views, encoder and LoRA outputs, restriction maps, all traced ADMM states,
  residuals, logits, reassembly, and the final prediction.
- Add the native visualization exporter, GIF output, and embedded f16 browser
  weights used by the live WebAssembly demo.

### Library and release quality

- Add MNIST and Sudoku inference scaffolding while keeping the maze artifact as
  the only release-qualified model path.
- Make Cargo features additive by removing the public-type-changing `f64`
  feature; the inference scalar remains fixed to the fp32 parity contract.
- Document all public APIs in `sheaf-core`, `sheaf-nn`, and `sheaf-io` and make
  warning-free rustdoc a CI and release requirement.
- Declare and test Rust 1.84 as the minimum supported version.
- Include Apache-2.0 `LICENSE` and upstream attribution `NOTICE` files in every
  published crate archive.
- Test all targets and all features, validate packaged crate sources, and run a
  crates.io dry run immediately before each trusted-publishing upload.

## 0.2.0

- Initial crates.io release of `sheaf-core`, `sheaf-nn`, and `sheaf-io`.
- Add the parity-tested maze inference pipeline and browser demo.
