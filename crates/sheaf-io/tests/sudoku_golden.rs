//! Sudoku golden-parity scaffold (Phase C, "parity_check --task sudoku").
//!
//! Mirror of `mnist_golden.rs`: the maze/mnist parity harness lives in the
//! `sheaf-demo` `parity_check` binary (outside this phase's write scope), so
//! this test is the sudoku-side consumption pattern, placed in `sheaf-io` so it
//! can be wired into `parity_check` verbatim later. It is GUARDED: if
//! `goldens/sudoku/trace.npz` is absent (the default until the orchestrator
//! dumps sudoku goldens + trained weights), it prints a skip line and passes, so
//! the normal `cargo test` run is unaffected.
//!
//! When goldens exist it loads `config.json` + `weights.safetensors`, runs the
//! Rust sudoku forward at K = trace K, and checks each intermediate against the
//! Python dump with the same widening tolerance schedule as the maze/mnist
//! harness (PLAN §5.2). Each golden array is checked only when present, so a
//! partial dump still exercises what it can.
//!
//! NOTE (faithful-transcription flags — the orchestrator MUST reconcile these
//! against the real exporter dump when goldens land; a mismatch surfaces here or
//! at load time rather than silently):
//!   1. Safetensors Flax module names for the sudoku encoder/decoder/LoRA and
//!      `rm/R_indices` are pinned in `weights::sudoku_expected_keys` against the
//!      Python `init` param tree (counts verified: 543,025 / 2,029,233).
//!   2. The exact soft_slice slot-map construction (9 base maps) and the 243-edge
//!      ordering/orientation live in `views` (const golden cross-checked against
//!      the generative builder).
//!   3. Golden array names below (`enc_h`, `enc_q_diag`, ... `x`/`z`/`y`,
//!      `logits_per_iter`, `logits_final`) and the sudoku input tensor layout
//!      (`inputs` [B,9,9] -> one-hot(10) -> 27-view slice -> [27,B,9,10]) are the
//!      Rust side's reconstruction and must match the dumper.

use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array4, ArrayD, ArrayViewD, Axis};

use sheaf_io::npz::Npz;
use sheaf_io::views::{build_sudoku_graph, sudoku_metrics, sudoku_predict, sudoku_slice_batch};
use sheaf_io::{load_sudoku_model, WeightCollection};

/// Widening per-iteration tolerance (identical schedule to the maze/mnist
/// harness): 1e-5 at k=0 growing geometrically to 1e-3 at k=K-1.
fn iter_tol(k: usize, k_total: usize) -> f32 {
    if k_total <= 1 {
        return 1e-5;
    }
    let frac = k as f32 / (k_total - 1) as f32;
    1e-5 * 100f32.powf(frac)
}

const ENC_TOL: f32 = 1e-5;

/// Elementwise |got - want| <= atol + rtol*|want|; fails on shape mismatch, NaN,
/// or any out-of-tolerance element.
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
        assert!(err <= thresh, "{name}: |{g} - {w}| = {err} > {thresh} (tol {tol:.1e})");
    }
    println!("PASS  {name:<24} max_abs={worst:.2e} (tol {tol:.1e})");
}

fn maybe_f32(npz: &mut Npz, name: &str) -> Option<ArrayD<f32>> {
    let names = npz.names().unwrap_or_default();
    if names.iter().any(|n| n == name) {
        npz.f32(name).ok()
    } else {
        None
    }
}

/// One-hot encode an integer grid `[B, 9, 9]` -> `[B, 9, 9, num_classes]`.
fn one_hot_grid(inputs: &ArrayD<f32>, num_classes: usize) -> Array4<f32> {
    let b = inputs.shape()[0];
    let flat = inputs.iter().copied().collect::<Vec<_>>();
    let mut out = Array4::<f32>::zeros((b, 9, 9, num_classes));
    for (idx, &v) in flat.iter().enumerate() {
        let bi = idx / 81;
        let rem = idx % 81;
        let (r, c) = (rem / 9, rem % 9);
        let cls = v.round() as i64;
        if (0..num_classes as i64).contains(&cls) {
            out[[bi, r, c, cls as usize]] = 1.0;
        }
    }
    out
}

#[test]
fn sudoku_golden_parity_scaffold() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../goldens/sudoku");
    if !dir.join("trace.npz").exists() {
        println!(
            "SKIP sudoku parity: no goldens at {} (orchestrator drops them later)",
            dir.display()
        );
        return;
    }
    println!("sudoku parity: goldens = {}", dir.display());

    let model = load_sudoku_model(
        &dir.join("config.json"),
        &dir.join("weights.safetensors"),
        WeightCollection::Ema,
    )
    .expect("load sudoku model");
    let m = model.config.model.clone();

    let mut batch = Npz::open(&dir.join("batch.npz")).expect("open batch.npz");
    let mut trace = Npz::open(&dir.join("trace.npz")).expect("open trace.npz");

    // Patches: prefer a dumped `patches` [27,B,9,C]; else build from `inputs`
    // [B,9,9] via one-hot(num_classes) -> 27-view slice -> transpose (1,0,2,3).
    let patches: Array4<f32> = if let Some(p) = maybe_f32(&mut batch, "patches") {
        p.into_dimensionality::<ndarray::Ix4>().expect("patches [27,B,9,C]")
    } else {
        let inputs = batch.f32("inputs").expect("batch inputs");
        let onehot = one_hot_grid(&inputs, m.num_classes); // [B,9,9,C]
        let sliced = sudoku_slice_batch(&onehot); // [B,27,9,C]
        // transpose (1,0,2,3) -> [27,B,9,C]
        let (b, a, s, c) = sliced.dim();
        let mut t = Array4::<f32>::zeros((a, b, s, c));
        for bi in 0..b {
            for ai in 0..a {
                for si in 0..s {
                    for ci in 0..c {
                        t[[ai, bi, si, ci]] = sliced[[bi, ai, si, ci]];
                    }
                }
            }
        }
        t
    };
    if let Some(p_gold) = maybe_f32(&mut batch, "patches") {
        assert_close("patches", patches.view().into_dyn(), p_gold.view(), 0.0);
    }

    let graph = Arc::new(build_sudoku_graph());

    // Encoder heads.
    let enc = model.encoder.forward(&patches, &model.cell_ids);
    if let Some(g) = maybe_f32(&mut trace, "enc_h") {
        assert_close("enc_h", enc.h.view().into_dyn(), g.view(), ENC_TOL);
    }
    if let sheaf_core::solvers::Objective::NonNeg { q_diag, q } = &enc.objective {
        if let Some(g) = maybe_f32(&mut trace, "enc_q_diag") {
            assert_close("enc_q_diag", q_diag.view().into_dyn(), g.view(), ENC_TOL);
        }
        if let Some(g) = maybe_f32(&mut trace, "enc_q") {
            assert_close("enc_q", q.view().into_dyn(), g.view(), ENC_TOL);
        }
    } else {
        panic!("sudoku encoder must emit Objective::NonNeg");
    }
    if let Some(lora) = enc.lora.as_ref() {
        if let Some(g) = maybe_f32(&mut trace, "lora_A") {
            assert_close("lora_A", lora.a.view().into_dyn(), g.view(), ENC_TOL);
        }
        if let Some(g) = maybe_f32(&mut trace, "lora_B") {
            assert_close("lora_B", lora.b.view().into_dyn(), g.view(), ENC_TOL);
        }
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

    // Base maps (pure gather of the loaded R_indices — exact).
    if let Some(g) = maybe_f32(&mut trace, "base_restriction_maps") {
        assert_close(
            "base_restriction_maps",
            fwd.base_restriction_maps.view().into_dyn(),
            g.view(),
            0.0,
        );
    }

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

    // Final logits + reassembled prediction + metrics (completion on empty cells).
    let final_tol = iter_tol(k_iters - 1, k_iters);
    let logits_final = fwd.logits_final();
    if let Some(g) = maybe_f32(&mut trace, "logits_final") {
        assert_close("logits_final", logits_final.view().into_dyn(), g.view(), final_tol);
    }

    let pred = sudoku_predict(&logits_final); // [B, 9, 9]
    if let (Ok(labels), Ok(inputs)) = (batch.i64("labels_grid"), batch.i64("inputs_grid")) {
        let labels = labels.into_dimensionality::<ndarray::Ix3>().expect("labels_grid [B,9,9]");
        let inputs = inputs.into_dimensionality::<ndarray::Ix3>().expect("inputs_grid [B,9,9]");
        let metrics = sudoku_metrics(&pred, &labels, &inputs);
        println!(
            "sudoku parity: cell_acc={:.4} solved={:.4} completion={:.4}",
            metrics.cell_acc, metrics.solved, metrics.completion
        );
    }

    println!("sudoku parity: scaffold complete");
}
