//! Batch-axis parallel dispatch (PLAN.md §3.1, Phase D).
//!
//! The sheaf Laplacian never mixes batch elements — each batch element owns its
//! full `[N, d_v]` slab — so the batch axis B is embarrassingly parallel. Under
//! the `parallel` feature [`map_batches`] fans the per-batch closure across
//! rayon; otherwise it is a plain serial map. Numerics are **identical** either
//! way: every batch element runs the exact same arithmetic in the exact same
//! per-batch reduction order, and rayon's `collect` into a `Vec` is
//! order-preserving. Default builds compile no rayon at all (wasm-safe).

/// Map `f` over batch indices `0..b`, collecting owned per-batch results in
/// batch order. Parallel over B under `--features parallel`, serial otherwise.
#[cfg(feature = "parallel")]
pub(crate) fn map_batches<T, F>(b: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    use rayon::prelude::*;
    (0..b).into_par_iter().map(f).collect()
}

/// Serial fallback (default build; no rayon compiled in).
#[cfg(not(feature = "parallel"))]
pub(crate) fn map_batches<T, F>(b: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T,
{
    (0..b).map(f).collect()
}
