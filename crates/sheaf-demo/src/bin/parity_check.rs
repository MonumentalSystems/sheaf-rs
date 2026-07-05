//! CI-facing golden replay: load goldens/maze/{config.json, weights.safetensors,
//! batch.npz}, run the Rust forward at K = trace K, and assert every array in
//! trace.npz layer-by-layer and per-iteration (widening tolerance schedule —
//! PLAN.md §5.2). Exits non-zero on the first divergence, printing the array
//! name, iteration, and max abs/rel error.

fn main() -> anyhow::Result<()> {
    // Stub: wired up once sheaf-core/nn/io land. See goldens/CONTRACT.md.
    eprintln!("parity_check: not yet implemented (scaffold)");
    std::process::exit(2);
}
