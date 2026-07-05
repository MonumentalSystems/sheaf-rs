//! MNIST golden-parity scaffold (Phase B, "parity_check --task mnist").
//!
//! The maze parity harness lives in the `sheaf-demo` `parity_check` binary,
//! which is outside this phase's write scope. This test is the mnist-side
//! mirror of that consumption pattern, placed in `sheaf-io` so it can be
//! wired into `parity_check` verbatim later. It is GUARDED: if
//! `goldens/mnist/trace.npz` is absent (the default until the orchestrator
//! dumps mnist goldens + trained weights), the test prints a skip line and
//! passes, so the normal `cargo test` run is unaffected.
//!
//! When goldens exist it loads `config.json` + `weights.safetensors`, runs the
//! Rust mnist forward at K = trace K, and checks each intermediate against the
//! Python dump with the same widening tolerance schedule as the maze harness
//! (PLAN.md §5.2). Golden array names mirror the maze trace where they exist;
//! mnist-specific names (no `enc_upper`/`enc_l1_weight` — lasso l1 is scalar;
//! `logits_final` is `[N, B, num_classes]`; `prediction` is `[B]`) are checked
//! only when present, so a partial dump still exercises what it can.
//!
//! NOTE (faithful-transcription flag): the exact mnist golden array names and
//! the safetensors Flax module names (`MLPEncoder_0`, `block_0`,
//! `ClassificationDecoder_0`, `rm/R_shared`) are the Rust side's reconstruction
//! of the camera-ready layout. The orchestrator MUST reconcile them against the
//! actual exporter `manifest.json` when the goldens land; a mismatch surfaces
//! here (or at load time) rather than silently.

use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{ArrayD, ArrayViewD, Axis};

use sheaf_core::graph::AgentGraph;
use sheaf_io::npz::Npz;
use sheaf_io::views::{build_grid_edge_indices, grid_agent_centers, patchify_batch};
use sheaf_io::{load_mnist_model, WeightCollection};
use sheaf_nn::restriction_maps::build_shared_restriction_maps;

/// Widening per-iteration tolerance (identical schedule to the maze harness):
/// 1e-5 at k=0 growing geometrically to 1e-3 at k=K-1.
fn iter_tol(k: usize, k_total: usize) -> f32 {
    if k_total <= 1 {
        return 1e-5;
    }
    let frac = k as f32 / (k_total - 1) as f32;
    1e-5 * 100f32.powf(frac)
}

const ENC_TOL: f32 = 1e-5;

/// Elementwise |got - want| <= atol + rtol*|want|; panics (fails the test) on
/// shape mismatch, NaN, or any out-of-tolerance element.
fn assert_close(name: &str, got: ArrayViewD<f32>, want: ArrayViewD<f32>, tol: f32) {
    assert_eq!(
        got.shape(),
        want.shape(),
        "{name}: shape mismatch got {:?} want {:?}",
        got.shape(),
        want.shape()
    );
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(!g.is_nan() && !w.is_nan(), "{name}: NaN (got {g}, want {w})");
        let err = (g - w).abs();
        worst = worst.max(err);
        let thresh = tol + tol * w.abs();
        assert!(
            err <= thresh,
            "{name}: |{g} - {w}| = {err} > {thresh} (tol {tol:.1e})"
        );
    }
    println!("PASS  {name:<24} max_abs={worst:.2e} (tol {tol:.1e})");
}

/// Fetch a trace array if present (skip individual checks for a partial dump).
fn maybe_f32(npz: &mut Npz, name: &str) -> Option<ArrayD<f32>> {
    let names = npz.names().unwrap_or_default();
    if names.iter().any(|n| n == name) {
        npz.f32(name).ok()
    } else {
        None
    }
}

#[test]
fn mnist_golden_parity_scaffold() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../goldens/mnist");
    if !dir.join("trace.npz").exists() {
        println!(
            "SKIP mnist parity: no goldens at {} (orchestrator drops them later)",
            dir.display()
        );
        return;
    }
    println!("mnist parity: goldens = {}", dir.display());

    let model = load_mnist_model(
        &dir.join("config.json"),
        &dir.join("weights.safetensors"),
        WeightCollection::Ema,
    )
    .expect("load mnist model");
    let t = model.config.task.clone();
    let m = model.config.model.clone();

    let mut batch = Npz::open(&dir.join("batch.npz")).expect("open batch.npz");
    let mut trace = Npz::open(&dir.join("trace.npz")).expect("open trace.npz");

    // Images [B, H, W, C]; labels [B].
    let images = batch.f32("images").expect("batch images");
    let (b, h, w) = (images.shape()[0], images.shape()[1], images.shape()[2]);
    let images4 = images
        .clone()
        .into_dimensionality::<ndarray::Ix4>()
        .expect("images [B,H,W,C]");

    // Views (must match Python exactly — exact where they are pure gathers).
    let centers = grid_agent_centers((h, w), t.stride, t.patch_size);
    if let Some(c_gold) = batch.names().ok().and_then(|n| {
        n.iter()
            .any(|x| x == "centers")
            .then(|| batch.i64("centers").ok())
            .flatten()
    }) {
        assert_eq!(
            centers.mapv(|v| v).into_dyn().shape(),
            c_gold.shape(),
            "centers shape"
        );
    }
    let edges = build_grid_edge_indices(&centers, t.stride, t.connectivity);
    let patches = patchify_batch(&images4, &centers, t.patch_size);
    if let Some(p_gold) = maybe_f32(&mut batch, "patches") {
        assert_close("patches", patches.view().into_dyn(), p_gold.view(), 0.0);
    }

    let positions = centers.mapv(|v| v as f32);
    let graph = Arc::new(AgentGraph::new_grid(edges, positions, m.num_directions));

    // Encoder heads.
    let enc = model.encoder.forward(&patches);
    if let Some(g) = maybe_f32(&mut trace, "enc_h") {
        assert_close("enc_h", enc.h.view().into_dyn(), g.view(), ENC_TOL);
    }
    if let sheaf_core::solvers::Objective::Lasso { q_diag, q, .. } = &enc.objective {
        if let Some(g) = maybe_f32(&mut trace, "enc_q_diag") {
            assert_close("enc_q_diag", q_diag.view().into_dyn(), g.view(), ENC_TOL);
        }
        if let Some(g) = maybe_f32(&mut trace, "enc_q") {
            assert_close("enc_q", q.view().into_dyn(), g.view(), ENC_TOL);
        }
    } else {
        panic!("mnist encoder must emit Objective::Lasso");
    }
    if let Some(lora) = enc.lora.as_ref() {
        if let Some(g) = maybe_f32(&mut trace, "lora_A") {
            assert_close("lora_A", lora.a.view().into_dyn(), g.view(), ENC_TOL);
        }
        if let Some(g) = maybe_f32(&mut trace, "lora_B") {
            assert_close("lora_B", lora.b.view().into_dyn(), g.view(), ENC_TOL);
        }
    }

    // Shared base maps (pure broadcast of the loaded weight — exact).
    let base = build_shared_restriction_maps(&model.rm_shared, graph.num_edges());
    if let Some(g) = maybe_f32(&mut trace, "base_restriction_maps") {
        assert_close("base_restriction_maps", base.view().into_dyn(), g.view(), 0.0);
    }

    // rho (export-baked scalar) — bitwise.
    if let Some(g) = maybe_f32(&mut trace, "rho") {
        let rho_gold = *g.iter().next().expect("rho scalar");
        assert_eq!(model.rho.to_bits(), rho_gold.to_bits(), "rho bitwise");
        println!("PASS  rho                      {} (bitwise)", model.rho);
    }

    // Full forward at K = trace K.
    let x_gold = trace.f32("x").expect("trace x");
    let k_iters = x_gold.shape()[0];
    let fwd = model.forward(&patches, graph.clone(), k_iters);

    let z_gold = maybe_f32(&mut trace, "z");
    let y_gold = maybe_f32(&mut trace, "y");
    let logits_gold = maybe_f32(&mut trace, "logits_per_iter");
    for k in 0..k_iters {
        let tol = iter_tol(k, k_iters);
        assert_close(
            &format!("x[k={k}]"),
            fwd.history.x.index_axis(Axis(0), k).into_dyn(),
            x_gold.index_axis(Axis(0), k),
            tol,
        );
        if let Some(zg) = &z_gold {
            assert_close(
                &format!("z[k={k}]"),
                fwd.history.z.index_axis(Axis(0), k).into_dyn(),
                zg.index_axis(Axis(0), k),
                tol,
            );
        }
        if let Some(yg) = &y_gold {
            assert_close(
                &format!("y[k={k}]"),
                fwd.history.y.index_axis(Axis(0), k).into_dyn(),
                yg.index_axis(Axis(0), k),
                tol,
            );
        }
        if let Some(lg) = &logits_gold {
            assert_close(
                &format!("logits[k={k}]"),
                fwd.logits_per_iter.index_axis(Axis(0), k).into_dyn(),
                lg.index_axis(Axis(0), k),
                tol,
            );
        }
    }

    // Final logits + the mean-of-softmax prediction (PLAN §5.2 eval quirk).
    let final_tol = iter_tol(k_iters - 1, k_iters);
    let logits_final = fwd.logits_final();
    if let Some(g) = maybe_f32(&mut trace, "logits_final") {
        assert_close("logits_final", logits_final.view().into_dyn(), g.view(), final_tol);
    }

    let pred = fwd.prediction();
    assert_eq!(pred.len(), b, "prediction batch size");
    // Exact match against the golden prediction if dumped; else check accuracy
    // against labels as a sanity signal.
    if let Ok(names) = trace.names() {
        if names.iter().any(|n| n == "prediction") {
            let g = trace.i64("prediction").expect("prediction i64");
            let g = g.into_dimensionality::<ndarray::Ix1>().expect("prediction [B]");
            assert_eq!(pred, g, "mnist prediction (mean-softmax argmax) must match");
            println!("PASS  prediction              {b} labels exact");
        }
    }
    if let Ok(labels) = batch.i64("labels") {
        let labels = labels.into_dimensionality::<ndarray::Ix1>().expect("labels [B]");
        let correct = pred.iter().zip(labels.iter()).filter(|(p, l)| p == l).count();
        println!("mnist parity: accuracy {correct}/{b}");
    }

    println!("mnist parity: scaffold complete");
}
