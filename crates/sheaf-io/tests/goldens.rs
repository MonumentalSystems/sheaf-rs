//! Golden-fixture parity for the sheaf-io surface (goldens/CONTRACT.md):
//! views (centers / edges / patches / reassembly) against `batch.npz` +
//! `trace.npz`, and the safetensors loader against `weights.safetensors`.
//!
//! Skips (with a note) if `goldens/maze/` is not present, so `cargo test`
//! works on a fresh checkout before the exporter has run.

use std::path::PathBuf;

use ndarray::{Array2, Array3, Array5, Ix2, Ix3, Ix5};

use sheaf_io::npz::Npz;
use sheaf_io::views::{
    build_grid_edge_indices, grid_agent_centers, prepare_maze_patches, reassemble_logits,
};
use sheaf_io::{load_maze_model, WeightCollection};

fn goldens_dir() -> Option<PathBuf> {
    // crates/sheaf-io -> workspace root.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../goldens/maze")
        .canonicalize()
        .ok()?;
    dir.join("batch.npz").exists().then_some(dir)
}

#[test]
fn views_match_python_dumps() {
    let Some(dir) = goldens_dir() else {
        eprintln!("goldens/maze not present; skipping golden parity test");
        return;
    };
    let mut batch = Npz::open(&dir.join("batch.npz")).unwrap();

    // Centers: exact integer match, row-major (y, x).
    let centers_gold: Array2<i64> = batch
        .i64_shaped("centers", &[81, 2])
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    let centers = grid_agent_centers((19, 19), 2, 3);
    assert_eq!(centers, centers_gold, "grid_agent_centers mismatch");

    // Edges: exact match including ordering.
    let edges_gold: Array2<i64> = batch
        .i64_shaped("edges", &[272, 2])
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    let edges = build_grid_edge_indices(&centers, 2, 8);
    assert_eq!(edges.len(), edges_gold.nrows());
    for (e, row) in edges.iter().zip(edges_gold.rows()) {
        assert_eq!([e[0] as i64, e[1] as i64], [row[0], row[1]]);
    }

    // Patches: wall pre-pad + one-hot is exact 0/1 arithmetic -> bitwise.
    let tokens: Array3<i64> = batch
        .i64_shaped("tokens", &[2, 19, 19])
        .unwrap()
        .into_dimensionality::<Ix3>()
        .unwrap();
    let patches_gold: Array5<f32> = batch
        .f32_shaped("patches", &[81, 2, 3, 3, 6])
        .unwrap()
        .into_dimensionality::<Ix5>()
        .unwrap();
    let patches = prepare_maze_patches(&tokens, &centers, 3, 6);
    assert_eq!(patches, patches_gold, "prepare_maze_patches mismatch");

    // Reassembly: overlap-mean of the dumped final logits vs the dumped
    // reassembled grid (contract tolerance 1e-4 abs).
    let mut trace = Npz::open(&dir.join("trace.npz")).unwrap();
    let logits_final: Array5<f32> = trace
        .f32_shaped("logits_final", &[81, 2, 3, 3, 6])
        .unwrap()
        .into_dimensionality::<Ix5>()
        .unwrap();
    let gold = trace.f32_shaped("reassembled_final", &[2, 19, 19, 6]).unwrap();
    let ours = reassemble_logits(&logits_final, &centers, (19, 19));
    let max_err = gold
        .iter()
        .zip(ours.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "reassemble_logits max abs err {max_err}");
}

#[test]
fn weights_load_from_goldens() {
    let Some(dir) = goldens_dir() else {
        eprintln!("goldens/maze not present; skipping golden weights test");
        return;
    };
    for collection in [WeightCollection::Ema, WeightCollection::Raw] {
        let model = load_maze_model(
            &dir.join("config.json"),
            &dir.join("weights.safetensors"),
            collection,
        )
        .unwrap();
        assert_eq!(model.rm.r_stack.dim(), (8, 5, 10));
        assert!(model.rho > 0.0);
    }

    // rho in the trace must equal the baked config value bitwise.
    let model = load_maze_model(
        &dir.join("config.json"),
        &dir.join("weights.safetensors"),
        WeightCollection::Ema,
    )
    .unwrap();
    let mut trace = Npz::open(&dir.join("trace.npz")).unwrap();
    let rho = trace.f32_shaped("rho", &[]).unwrap();
    assert_eq!(rho.iter().next().copied().unwrap(), model.rho);
}
