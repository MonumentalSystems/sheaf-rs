//! CI-facing golden replay: load goldens/maze/{config.json, weights.safetensors,
//! batch.npz, trace.npz}, run the Rust forward at K = trace K, and assert every
//! array layer-by-layer and per-iteration (widening tolerance schedule —
//! PLAN.md §5.2). Prints one PASS/FAIL row per array (per iteration for the
//! trajectory) and exits non-zero if anything failed.
//!
//! Check order = module order, so the FIRST failing row names the buggy module:
//!   views (centers/edges/patches) -> graph slot tables -> encoder heads ->
//!   LoRA factors -> assembled base maps -> LoRA edge gather -> rho ->
//!   per-iteration x/z/y + residuals + consistency + decoded logits ->
//!   final logits -> overlap-mean reassembly -> argmax grid (EXACT).
//!
//! Tolerance schedule (do NOT weaken; find the bug instead): encoder-level
//! arrays at rtol=atol=1e-5; ADMM iterate k (0-based, K total) at
//! `1e-5 * 100^(k/(K-1))` — 1e-5 at the first iterate widening to 1e-3 by the
//! last (f32 CG accumulation-order drift compounds over the unroll). Pure
//! gathers / one-hot views / rho are exact-bitwise. pred_grid is exact.
//!
//! Usage: parity_check [GOLDENS_DIR]   (default: <workspace>/goldens/maze)

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use ndarray::{ArrayViewD, Axis, Dimension, Ix3, Ix5};

use sheaf_core::geometry::LoraGeometry;
use sheaf_core::graph::AgentGraph;
use sheaf_io::npz::Npz;
use sheaf_io::views::{
    build_grid_edge_indices, grid_agent_centers, prepare_maze_patches, reassemble_logits,
};
use sheaf_io::{load_maze_model, WeightCollection};

/// Widening per-iteration tolerance: 1e-5 at k=0 growing geometrically to
/// 1e-3 at k=K-1 (PLAN.md §5.2). Used for both rtol and atol.
fn iter_tol(k: usize, k_total: usize) -> f32 {
    if k_total <= 1 {
        return 1e-5;
    }
    let frac = k as f32 / (k_total - 1) as f32;
    1e-5 * 100f32.powf(frac)
}

/// Encoder-level tolerance (single matmul chains, no unroll).
const ENC_TOL: f32 = 1e-5;

struct Report {
    failures: Vec<String>,
    checks: usize,
}

impl Report {
    fn new() -> Self {
        Report { failures: Vec::new(), checks: 0 }
    }

    fn pass(&mut self, name: &str, detail: &str) {
        self.checks += 1;
        println!("PASS  {name:<24} {detail}");
    }

    fn fail(&mut self, name: &str, detail: &str) {
        self.checks += 1;
        self.failures.push(name.to_string());
        println!("FAIL  {name:<24} {detail}");
    }

    /// Elementwise |got - want| <= atol + rtol*|want|. rtol=atol=0 means exact
    /// (bitwise for finite values; NaN anywhere fails).
    fn check_f32(
        &mut self,
        name: &str,
        got: ArrayViewD<f32>,
        want: ArrayViewD<f32>,
        rtol: f32,
        atol: f32,
    ) {
        if got.shape() != want.shape() {
            self.fail(
                name,
                &format!("shape mismatch: got {:?}, want {:?}", got.shape(), want.shape()),
            );
            return;
        }
        let mut violations = 0usize;
        let mut max_abs = 0f32; // max |got - want|
        let mut max_rel = 0f32; // max |got - want| / |want| over |want| > 0
        let mut worst_excess = f32::NEG_INFINITY; // err - thresh, to pick the worst element
        let mut worst: Option<(Vec<usize>, f32, f32)> = None;
        for ((idx, &w), &g) in want.indexed_iter().zip(got.iter()) {
            if g.is_nan() || w.is_nan() {
                self.fail(name, &format!("NaN at {:?}: got {g}, want {w}", idx.slice()));
                return;
            }
            let err = (g - w).abs();
            max_abs = max_abs.max(err);
            if w != 0.0 {
                max_rel = max_rel.max(err / w.abs());
            }
            let thresh = atol + rtol * w.abs();
            if err > thresh {
                violations += 1;
                if err - thresh > worst_excess {
                    worst_excess = err - thresh;
                    worst = Some((idx.slice().to_vec(), g, w));
                }
            }
        }
        let tol_str = if rtol == 0.0 && atol == 0.0 {
            "exact".to_string()
        } else {
            format!("rtol=atol={rtol:.1e}")
        };
        if violations == 0 {
            self.pass(
                name,
                &format!("max_abs={max_abs:.2e} max_rel={max_rel:.2e} ({tol_str})"),
            );
        } else {
            let (idx, g, w) = worst.unwrap();
            self.fail(
                name,
                &format!(
                    "{violations}/{} out of tol ({tol_str}); worst at {idx:?}: got {g:.7e}, want {w:.7e}, |d|={:.2e}; max_abs={max_abs:.2e} max_rel={max_rel:.2e}",
                    want.len(),
                    (g - w).abs(),
                ),
            );
        }
    }

    fn check_i64(&mut self, name: &str, got: ArrayViewD<i64>, want: ArrayViewD<i64>) {
        if got.shape() != want.shape() {
            self.fail(
                name,
                &format!("shape mismatch: got {:?}, want {:?}", got.shape(), want.shape()),
            );
            return;
        }
        let mut violations = 0usize;
        let mut worst: Option<(Vec<usize>, i64, i64)> = None;
        for ((idx, &w), &g) in want.indexed_iter().zip(got.iter()) {
            if g != w {
                violations += 1;
                if worst.is_none() {
                    worst = Some((idx.slice().to_vec(), g, w));
                }
            }
        }
        if violations == 0 {
            self.pass(name, &format!("{} elements exact", want.len()));
        } else {
            let (idx, g, w) = worst.unwrap();
            self.fail(
                name,
                &format!(
                    "{violations}/{} mismatch; first at {idx:?}: got {g}, want {w}",
                    want.len()
                ),
            );
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("parity_check: error: {e:#}");
            ExitCode::from(2)
        }
    }
}

fn run() -> anyhow::Result<ExitCode> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../goldens/maze"));
    anyhow::ensure!(
        dir.join("trace.npz").exists(),
        "goldens dir {} has no trace.npz (pass the dir as the first argument)",
        dir.display()
    );
    println!("parity_check: goldens = {}", dir.display());

    let mut rep = Report::new();

    // ---- model (EMA collection — the tree the goldens were dumped from) ----
    let model = load_maze_model(
        &dir.join("config.json"),
        &dir.join("weights.safetensors"),
        WeightCollection::Ema,
    )?;
    let t = model.config.task.clone();
    let m = model.config.model.clone();

    let mut batch = Npz::open(&dir.join("batch.npz"))?;
    let mut trace = Npz::open(&dir.join("trace.npz"))?;

    // Golden dims (read, not assumed, so a regenerated dump re-pins everything).
    let tokens_gold = batch.i64("tokens")?;
    let (b, h, w) = (tokens_gold.shape()[0], tokens_gold.shape()[1], tokens_gold.shape()[2]);
    let x_gold = trace.f32("x")?;
    let (k_iters, n) = (x_gold.shape()[0], x_gold.shape()[1]);
    let (d_v, d_e, rank, n_dirs) = (m.d_v, m.d_e, m.lora_rank, m.num_directions);
    println!(
        "parity_check: B={b} HxW={h}x{w} N={n} K={k_iters} d_v={d_v} d_e={d_e} r={rank}"
    );

    // ================= 1. views (sheaf-io::views) =================
    let tokens = tokens_gold.clone().into_dimensionality::<Ix3>()?;
    let centers = grid_agent_centers((h, w), t.stride, t.patch_size);
    rep.check_i64("centers", centers.view().into_dyn(), batch.i64("centers")?.view());

    let edges = build_grid_edge_indices(&centers, t.stride, t.connectivity);
    let edges_arr = ndarray::Array2::from_shape_fn((edges.len(), 2), |(e, j)| edges[e][j] as i64);
    rep.check_i64("edges", edges_arr.view().into_dyn(), batch.i64("edges")?.view());

    let positions = centers.mapv(|v| v as f32);
    rep.check_f32(
        "node_positions",
        positions.view().into_dyn(),
        batch.f32("node_positions")?.view(),
        0.0,
        0.0,
    );

    let patches = prepare_maze_patches(&tokens, &centers, t.patch_size, t.num_classes);
    rep.check_f32(
        "patches",
        patches.view().into_dyn(),
        batch.f32("patches")?.view(),
        0.0,
        0.0,
    );

    // ================= 2. graph slot tables (sheaf-core::graph) =================
    let graph = Arc::new(AgentGraph::new_grid(edges, positions, n_dirs));
    let dir_uv = ndarray::Array1::from_iter(graph.dir_uv.iter().map(|&v| v as i64));
    let dir_vu = ndarray::Array1::from_iter(graph.dir_vu.iter().map(|&v| v as i64));
    rep.check_i64("dir_uv", dir_uv.view().into_dyn(), batch.i64("dir_uv")?.view());
    rep.check_i64("dir_vu", dir_vu.view().into_dyn(), batch.i64("dir_vu")?.view());

    // ================= 3. encoder heads (sheaf-nn::encoder) =================
    let enc_out = model.encoder.forward(&patches);
    rep.check_f32(
        "enc_h",
        enc_out.h.view().into_dyn(),
        trace.f32("enc_h")?.view(),
        ENC_TOL,
        ENC_TOL,
    );
    match &enc_out.objective {
        sheaf_core::solvers::Objective::L1Box { q_diag, q, l1, upper } => {
            rep.check_f32(
                "enc_q_diag",
                q_diag.view().into_dyn(),
                trace.f32("enc_q_diag")?.view(),
                ENC_TOL,
                ENC_TOL,
            );
            rep.check_f32("enc_q", q.view().into_dyn(), trace.f32("enc_q")?.view(), ENC_TOL, ENC_TOL);
            rep.check_f32(
                "enc_l1_weight",
                l1.view().into_dyn(),
                trace.f32("enc_l1_weight")?.view(),
                ENC_TOL,
                ENC_TOL,
            );
            rep.check_f32(
                "enc_upper",
                upper.view().into_dyn(),
                trace.f32("enc_upper")?.view(),
                ENC_TOL,
                ENC_TOL,
            );
        }
        _ => rep.fail("enc_objective", "maze encoder must emit Objective::L1Box"),
    }

    // ================= 4. LoRA factors (sheaf-nn::encoder heads) =================
    let lora = enc_out.lora.as_ref().expect("maze config emits LoRA factors");
    let lora_a_gold = trace.f32("lora_A")?;
    let lora_b_gold = trace.f32("lora_B")?;
    rep.check_f32("lora_A", lora.a.view().into_dyn(), lora_a_gold.view(), ENC_TOL, ENC_TOL);
    rep.check_f32("lora_B", lora.b.view().into_dyn(), lora_b_gold.view(), ENC_TOL, ENC_TOL);

    // ================= 5. assembled base maps (sheaf-nn::restriction_maps) =====
    // Pure gather of loaded weights by the slot tables — must be exact.
    let base = sheaf_nn::restriction_maps::build_directional_restriction_maps(
        &model.rm.r_stack,
        &graph,
    );
    rep.check_f32(
        "base_restriction_maps",
        base.view().into_dyn(),
        trace.f32("base_restriction_maps")?.view(),
        0.0,
        0.0,
    );

    // ================= 6. LoRA edge gather (sheaf-core::geometry::lora) ========
    // Built from the GOLDEN per-node factors so a failure here indicts the
    // gather (slot-table direction asymmetry), not the encoder upstream.
    {
        let a_g = lora_a_gold.clone().into_dimensionality::<Ix5>()?;
        let b_g = lora_b_gold.clone().into_dimensionality::<Ix5>()?;
        let geo = LoraGeometry::create_directional(
            graph.clone(),
            base.clone(),
            &a_g,
            &b_g,
            None,
            m.lora_alpha,
        );
        // Reference gather straight off the golden tables: a_u[e,:,..] must be
        // lora_A[u_e, :, dir_uv[e]] and a_v[e,:,..] lora_A[v_e, :, dir_vu[e]].
        let e_cnt = graph.num_edges();
        let mut a_u_ref = ndarray::Array4::zeros((e_cnt, b, d_e, rank));
        let mut a_v_ref = ndarray::Array4::zeros((e_cnt, b, d_e, rank));
        let mut b_u_ref = ndarray::Array4::zeros((e_cnt, b, d_v, rank));
        let mut b_v_ref = ndarray::Array4::zeros((e_cnt, b, d_v, rank));
        for (ei, &[u, v]) in graph.edges.iter().enumerate() {
            let (u, v) = (u as usize, v as usize);
            let (su, sv) = (graph.dir_uv[ei] as usize, graph.dir_vu[ei] as usize);
            a_u_ref
                .index_axis_mut(Axis(0), ei)
                .assign(&a_g.slice(ndarray::s![u, .., su, .., ..]));
            a_v_ref
                .index_axis_mut(Axis(0), ei)
                .assign(&a_g.slice(ndarray::s![v, .., sv, .., ..]));
            b_u_ref
                .index_axis_mut(Axis(0), ei)
                .assign(&b_g.slice(ndarray::s![u, .., su, .., ..]));
            b_v_ref
                .index_axis_mut(Axis(0), ei)
                .assign(&b_g.slice(ndarray::s![v, .., sv, .., ..]));
        }
        rep.check_f32("lora_A_u_edge", geo.a_u_edge.view().into_dyn(), a_u_ref.view().into_dyn(), 0.0, 0.0);
        rep.check_f32("lora_A_v_edge", geo.a_v_edge.view().into_dyn(), a_v_ref.view().into_dyn(), 0.0, 0.0);
        rep.check_f32("lora_B_u_edge", geo.b_u_edge.view().into_dyn(), b_u_ref.view().into_dyn(), 0.0, 0.0);
        rep.check_f32("lora_B_v_edge", geo.b_v_edge.view().into_dyn(), b_v_ref.view().into_dyn(), 0.0, 0.0);
    }

    // ================= 7. rho (export-baked scalar) =================
    let rho_gold = trace.f32("rho")?;
    let rho_gold = *rho_gold.iter().next().expect("rho scalar");
    if model.rho.to_bits() == rho_gold.to_bits() {
        rep.pass("rho", &format!("{} (bitwise)", model.rho));
    } else {
        rep.fail(
            "rho",
            &format!("got {:?} (0x{:08x}), want {rho_gold:?} (0x{:08x})", model.rho, model.rho.to_bits(), rho_gold.to_bits()),
        );
    }

    // ================= 8. full forward: per-iteration trajectory ============
    let fwd = model.forward(&patches, graph.clone(), k_iters);

    let z_gold = trace.f32("z")?;
    let y_gold = trace.f32("y")?;
    let primal_gold = trace.f32("primal_res")?;
    let dual_gold = trace.f32("dual_res")?;
    let cons_gold = trace.f32("consistency")?;
    let logits_gold = trace.f32("logits_per_iter")?;

    for k in 0..k_iters {
        let tol = iter_tol(k, k_iters);
        let pairs: [(&str, ArrayViewD<f32>, ArrayViewD<f32>); 6] = [
            ("x", fwd.history.x.index_axis(Axis(0), k).into_dyn(), x_gold.index_axis(Axis(0), k)),
            ("z", fwd.history.z.index_axis(Axis(0), k).into_dyn(), z_gold.index_axis(Axis(0), k)),
            ("y", fwd.history.y.index_axis(Axis(0), k).into_dyn(), y_gold.index_axis(Axis(0), k)),
            (
                "primal_res",
                fwd.history.primal_res.index_axis(Axis(0), k).into_dyn(),
                primal_gold.index_axis(Axis(0), k),
            ),
            (
                "dual_res",
                fwd.history.dual_res.index_axis(Axis(0), k).into_dyn(),
                dual_gold.index_axis(Axis(0), k),
            ),
            (
                "consistency",
                fwd.history.consistency_rms.index_axis(Axis(0), k).into_dyn(),
                cons_gold.index_axis(Axis(0), k),
            ),
        ];
        for (name, got, want) in pairs {
            rep.check_f32(&format!("{name}[k={k}]"), got, want, tol, tol);
        }
        rep.check_f32(
            &format!("logits[k={k}]"),
            fwd.logits_per_iter.index_axis(Axis(0), k).into_dyn(),
            logits_gold.index_axis(Axis(0), k),
            tol,
            tol,
        );
    }

    // ================= 9. final logits + reassembly + argmax grid ===========
    let final_tol = iter_tol(k_iters - 1, k_iters);
    let logits_final = fwd.logits_final();
    rep.check_f32(
        "logits_final",
        logits_final.view().into_dyn(),
        trace.f32("logits_final")?.view(),
        final_tol,
        final_tol,
    );

    let reassembled = reassemble_logits(&logits_final, &centers, (h, w));
    rep.check_f32(
        "reassembled_final",
        reassembled.view().into_dyn(),
        trace.f32("reassembled_final")?.view(),
        final_tol,
        final_tol,
    );

    // Decoded per-cell argmax — EXACT equality (np.argmax tie-break = first max).
    let mut pred = ndarray::Array3::<i64>::zeros((b, h, w));
    for bi in 0..b {
        for y in 0..h {
            for x in 0..w {
                let (mut arg, mut best) = (0usize, f32::NEG_INFINITY);
                for c in 0..t.num_classes {
                    let v = reassembled[[bi, y, x, c]];
                    if v > best {
                        best = v;
                        arg = c;
                    }
                }
                pred[[bi, y, x]] = arg as i64;
            }
        }
    }
    rep.check_i64("pred_grid", pred.view().into_dyn(), trace.i64("pred_grid")?.view());

    // ================= summary =================
    println!(
        "\nparity_check: {}/{} checks passed",
        rep.checks - rep.failures.len(),
        rep.checks
    );
    if rep.failures.is_empty() {
        println!("parity_check: ALL GREEN");
        Ok(ExitCode::SUCCESS)
    } else {
        println!("parity_check: FAILURES: {}", rep.failures.join(", "));
        Ok(ExitCode::FAILURE)
    }
}
