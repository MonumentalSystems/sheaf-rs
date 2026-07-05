//! End-to-end smoke test: load the golden maze model (config.json +
//! weights.safetensors via sheaf-io), rebuild the views/graph from the raw
//! tokens in batch.npz, run the full forward (encode -> LoRA geometry ->
//! ADMM K=12 -> decode), and assert shapes + finiteness.
//!
//! Deliberately NOT value parity against trace.npz — that is the next phase
//! (parity_check). This test only pins that the whole pipeline is wired and
//! numerically sane.

use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array2, Array3, Ix1, Ix2, Ix3};

use sheaf_core::graph::AgentGraph;
use sheaf_io::npz::Npz;
use sheaf_io::views::{
    build_grid_edge_indices, grid_agent_centers, prepare_maze_patches, reassemble_logits,
};
use sheaf_io::{load_maze_model, WeightCollection};

fn goldens_dir() -> Option<PathBuf> {
    // crates/sheaf-demo -> workspace root.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../goldens/maze")
        .canonicalize()
        .ok()?;
    dir.join("batch.npz").exists().then_some(dir)
}

#[test]
fn golden_model_forward_smoke() {
    let Some(dir) = goldens_dir() else {
        eprintln!("goldens/maze not present; skipping e2e smoke test");
        return;
    };

    // 1. Weights + config through the sheaf-io loader (EMA tree, paper eval).
    let model = load_maze_model(
        &dir.join("config.json"),
        &dir.join("weights.safetensors"),
        WeightCollection::Ema,
    )
    .expect("golden model must load");
    let t = &model.config.task;
    let m = &model.config.model;
    assert_eq!((t.patch_size, t.stride, t.connectivity), (3, 2, 8));

    // 2. Views from the raw tokens (the same path a real caller takes).
    let mut batch = Npz::open(&dir.join("batch.npz")).unwrap();
    let tokens: Array3<i64> = batch
        .i64_shaped("tokens", &[2, 19, 19])
        .unwrap()
        .into_dimensionality::<Ix3>()
        .unwrap();
    let (b, h, w) = tokens.dim();
    let centers = grid_agent_centers((h, w), t.stride, t.patch_size);
    let n = centers.nrows();
    let edges = build_grid_edge_indices(&centers, t.stride, t.connectivity);
    let patches = prepare_maze_patches(&tokens, &centers, t.patch_size, t.num_classes);
    assert_eq!(patches.dim(), (n, b, t.patch_size, t.patch_size, t.num_classes));

    // 3. Graph with directional slot tables; cross-check them against the
    //    Python dump (reconciles the graph.rs / restriction_maps.rs twins).
    let positions = centers.mapv(|v| v as f32);
    let graph = Arc::new(AgentGraph::new_grid(edges, positions, m.num_directions));
    assert_eq!(graph.num_nodes, 81);
    assert_eq!(graph.num_edges(), 272);
    let dir_uv_gold = batch
        .i64_shaped("dir_uv", &[272])
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    let dir_vu_gold = batch
        .i64_shaped("dir_vu", &[272])
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    for e in 0..graph.num_edges() {
        assert_eq!(graph.dir_uv[e] as i64, dir_uv_gold[e], "dir_uv[{e}]");
        assert_eq!(graph.dir_vu[e] as i64, dir_vu_gold[e], "dir_vu[{e}]");
    }
    let positions_gold: Array2<f32> = batch
        .f32_shaped("node_positions", &[81, 2])
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert_eq!(graph.node_positions.as_ref().unwrap(), &positions_gold);

    // 4. Full forward, K = 12 (the golden trace length).
    let k = 12;
    let fwd = model.forward(&patches, graph, k);

    // Shapes.
    assert_eq!(
        fwd.logits_per_iter.shape(),
        &[k, n, b, t.patch_size, t.patch_size, m.num_classes]
    );
    assert_eq!(fwd.history.x.shape(), &[k, n, b, m.d_v]);
    assert_eq!(fwd.history.z.shape(), &[k, n, b, m.d_v]);
    assert_eq!(fwd.history.y.shape(), &[k, n, b, m.d_v]);
    assert_eq!(fwd.history.primal_res.shape(), &[k, n, b]);
    assert_eq!(fwd.history.dual_res.shape(), &[k, n, b]);
    assert_eq!(fwd.history.consistency_rms.shape(), &[k, b]);
    assert_eq!(
        fwd.base_restriction_maps.shape(),
        &[272, 2, m.d_e, m.d_v],
        "assembled base maps [E, 2, d_e, d_v]"
    );

    // Finiteness (weights are random-init: assert sanity, not solution quality).
    assert!(fwd.logits_per_iter.iter().all(|v| v.is_finite()), "non-finite logits");
    assert!(fwd.history.x.iter().all(|v| v.is_finite()), "non-finite x history");
    assert!(fwd.history.z.iter().all(|v| v.is_finite()), "non-finite z history");
    assert!(fwd.history.y.iter().all(|v| v.is_finite()), "non-finite y history");
    assert!(fwd.final_state.x.iter().all(|v| v.is_finite()));
    // Logits must not be all-zero (the pipeline actually computed something).
    assert!(fwd.logits_per_iter.iter().any(|v| v.abs() > 1e-6), "all-zero logits");

    // 5. Final logits -> overlap-mean global grid, finite and correctly shaped.
    let logits_final = fwd.logits_final();
    assert_eq!(
        logits_final.shape(),
        &[n, b, t.patch_size, t.patch_size, m.num_classes]
    );
    let grid = reassemble_logits(&logits_final, &centers, (h, w));
    assert_eq!(grid.shape(), &[b, h, w, m.num_classes]);
    assert!(grid.iter().all(|v| v.is_finite()), "non-finite reassembled grid");

    // 6. forward_window (training-style read) agrees on shapes too.
    let (final_state, window_logits) = model.forward_window(&patches, // re-forward
        Arc::new(AgentGraph::new_grid(
            build_grid_edge_indices(&centers, t.stride, t.connectivity),
            centers.mapv(|v| v as f32),
            m.num_directions,
        )),
        k,
        t.loss_window,
    );
    assert_eq!(
        window_logits.shape(),
        &[t.loss_window, n, b, t.patch_size, t.patch_size, m.num_classes]
    );
    assert!(window_logits.iter().all(|v| v.is_finite()));
    // Deterministic pipeline: the two drivers must land on the same final x.
    let max_diff = final_state
        .x
        .iter()
        .zip(fwd.final_state.x.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff == 0.0, "forward vs forward_window final x diverged by {max_diff}");
}
