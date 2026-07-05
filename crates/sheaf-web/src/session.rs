//! The forward-only demo session: embedded f16 EMA weights, one solve at a
//! time, per-iteration artifacts cached for the UI to poll.
//!
//! Pure Rust (no wasm-bindgen) so the same logic runs natively in CI tests;
//! `lib.rs` wraps it for the browser. The wiring transcribes
//! `sheaf-demo/src/bin/maze_demo.rs`: generate -> views -> graph -> forward ->
//! per-iteration overlap-mean reassembly -> argmax frames.

use std::sync::Arc;

use ndarray::{Array3, Axis};

use sheaf_core::geometry::{FixedGeometry, LoraGeometry, SheafGeometry};
use sheaf_core::graph::AgentGraph;
use sheaf_io::views::{
    build_grid_edge_indices, grid_agent_centers, prepare_maze_patches, reassemble_logits,
    TOKEN_GOAL, TOKEN_PATH, TOKEN_START,
};
use sheaf_io::weights::load_maze_model_from_bytes;
use sheaf_io::WeightCollection;
use sheaf_nn::model::SheafAdmmModel;
use sheaf_nn::restriction_maps::build_directional_restriction_maps;

/// Embedded exported config (goldens/CONTRACT.md layout, `"trained": true`).
pub const CONFIG_JSON: &str = include_str!("../../../goldens/maze/config.json");
/// Embedded EMA-only f16 weights (tools/convert_f16.py output, ~359 KB).
pub static WEIGHTS_EMA_F16: &[u8] = include_bytes!("../assets/weights_ema_f16.safetensors");

/// Everything one `solve()` leaves behind for the accessors.
struct SolveCache {
    k: usize,
    /// `[K][H*W]` per-cell argmax class per the token convention.
    frames: Vec<Vec<u8>>,
    /// `[K*3]` rows of (consistency_rms, primal_rms, dual_rms), batch 0.
    residuals: Vec<f32>,
    /// `[K][N]` per-agent RMS edge-residual magnitude.
    agent_cons: Vec<Vec<f32>>,
    /// `[N*3]` rows of (center_y, center_x, patch_size).
    agent_meta: Vec<u32>,
}

/// Session over the embedded (or injected) maze model.
pub struct SessionCore {
    model: SheafAdmmModel,
    cache: Option<SolveCache>,
}

impl SessionCore {
    /// Construct from the embedded config + f16 EMA weights.
    pub fn from_embedded() -> Result<Self, String> {
        Self::from_bytes(CONFIG_JSON, WEIGHTS_EMA_F16)
    }

    /// Construct from caller-provided config JSON + safetensors bytes
    /// (EMA collection; f16 widened to f32 by the sheaf-io loader).
    pub fn from_bytes(config_json: &str, weights: &[u8]) -> Result<Self, String> {
        let model = load_maze_model_from_bytes(config_json, weights, WeightCollection::Ema)
            .map_err(|e| format!("loading embedded weights: {e:#}"))?;
        Ok(SessionCore { model, cache: None })
    }

    /// Run the full forward with history at `K = k` on `B = 1` and cache the
    /// per-iteration frames / residuals / agent diagnostics. Size-generic:
    /// the grid, graph, and patches are rebuilt per call. Returns `k`.
    ///
    /// `k` is capped at 1000 (memory guard on the `[K,N,B,d_v]` history plus
    /// per-iteration frames; the UI uses K <= 100). On any error the previous
    /// solve's cache is left intact — all validation and the whole forward run
    /// happen before `self.cache` is replaced, so `frame()`/`residuals()`
    /// keep serving the last successful solve (the JS mock matches this).
    pub fn solve(
        &mut self,
        tokens: &[u8],
        height: usize,
        width: usize,
        k: usize,
    ) -> Result<usize, String> {
        let t = self.model.config.task.clone();
        let m = &self.model.config.model;

        // ---- input validation (JS-facing: errors, never panics) ----
        if k == 0 {
            return Err("k must be >= 1".into());
        }
        if k > 1000 {
            return Err(format!("k={k} too large (max 1000)"));
        }
        if height < t.patch_size || width < t.patch_size {
            return Err(format!(
                "grid {height}x{width} smaller than patch_size {}",
                t.patch_size
            ));
        }
        if tokens.len() != height * width {
            return Err(format!(
                "tokens length {} != height*width = {}",
                tokens.len(),
                height * width
            ));
        }
        if let Some(bad) = tokens.iter().find(|&&v| (v as usize) >= t.num_classes) {
            return Err(format!(
                "invalid token {bad} (tokens are 0..{})",
                t.num_classes - 1
            ));
        }
        let count = |tok: i64| tokens.iter().filter(|&&v| v as i64 == tok).count();
        if count(TOKEN_START) == 0 || count(TOKEN_GOAL) == 0 {
            return Err("maze must contain a start (3) and a goal (4) cell".into());
        }

        // ---- views + agent graph (transcribed from maze_demo) ----
        let tokens3 =
            Array3::from_shape_fn((1, height, width), |(_, y, x)| tokens[y * width + x] as i64);
        let centers = grid_agent_centers((height, width), t.stride, t.patch_size);
        let n = centers.nrows();
        if n == 0 {
            return Err(format!("grid {height}x{width} yields no agents"));
        }
        let edges = build_grid_edge_indices(&centers, t.stride, t.connectivity);
        let positions = centers.mapv(|v| v as f32);
        let graph = Arc::new(AgentGraph::new_grid(edges, positions, m.num_directions));
        let patches = prepare_maze_patches(&tokens3, &centers, t.patch_size, t.num_classes);

        // ---- one history forward at K = k (frames decode from history.x) ----
        let fwd = self.model.forward(&patches, graph.clone(), k);
        let k_total = fwd.history.x.shape()[0];

        // ---- per-iteration argmax frames + residual rows (batch 0) ----
        let mut frames = Vec::with_capacity(k_total);
        let mut residuals = Vec::with_capacity(k_total * 3);
        for ki in 0..k_total {
            let logits_k = fwd.logits_per_iter.index_axis(Axis(0), ki).to_owned();
            let grid_logits = reassemble_logits(&logits_k, &centers, (height, width));
            let mut frame = vec![0u8; height * width];
            for y in 0..height {
                for x in 0..width {
                    let (mut arg, mut best) = (0usize, f32::NEG_INFINITY);
                    for c in 0..m.num_classes {
                        let v = grid_logits[[0, y, x, c]];
                        if v > best {
                            best = v;
                            arg = c;
                        }
                    }
                    frame[y * width + x] = arg as u8;
                }
            }
            frames.push(frame);

            let cons = fwd.history.consistency_rms[[ki, 0]];
            let rms_b0 = |a: &ndarray::Array3<f32>| {
                let col = a.index_axis(Axis(0), ki);
                let sum: f32 = (0..n).map(|i| col[[i, 0]] * col[[i, 0]]).sum();
                (sum / n as f32).sqrt()
            };
            let primal = rms_b0(&fwd.history.primal_res);
            let dual = rms_b0(&fwd.history.dual_res);
            if !(cons.is_finite() && primal.is_finite() && dual.is_finite()) {
                return Err(format!("non-finite residuals at iteration {ki}"));
            }
            residuals.extend([cons, primal, dual]);
        }

        // ---- per-agent RMS over incident edge residuals, per iteration ----
        // Rebuild the geometry exactly as SheafAdmmModel::setup_admm does
        // (encoder LoRA factors -> directional LoRA geometry).
        let enc_out = self.model.encoder.forward(&patches);
        let base = build_directional_restriction_maps(&self.model.rm.r_stack, &graph);
        let geometry: Box<dyn SheafGeometry> = match m.rm_mode.as_str() {
            "context" => {
                let lora = enc_out
                    .lora
                    .as_ref()
                    .ok_or("rm_mode=context requires encoder LoRA factors")?;
                Box::new(LoraGeometry::create_directional(
                    graph.clone(),
                    base,
                    &lora.a,
                    &lora.b,
                    lora.gate.as_ref(),
                    lora.lora_alpha,
                ))
            }
            _ => Box::new(FixedGeometry::new(graph.clone(), base)),
        };
        let d_e = m.d_e;
        let inc = &graph.node_edges;
        let mut agent_cons = Vec::with_capacity(k_total);
        for ki in 0..k_total {
            let z_k = fwd.history.z.index_axis(Axis(0), ki).to_owned();
            let r = geometry.edge_residuals(&z_k); // [E, B, d_e]
            let mut row = vec![0f32; n];
            for (ni, slot) in row.iter_mut().enumerate() {
                let (lo, hi) = (inc.offsets[ni] as usize, inc.offsets[ni + 1] as usize);
                let mut sum = 0f32;
                for &(ei, _) in &inc.entries[lo..hi] {
                    for c in 0..d_e {
                        let v = r[[ei as usize, 0, c]];
                        sum += v * v;
                    }
                }
                let cnt = ((hi - lo) * d_e).max(1);
                *slot = (sum / cnt as f32).sqrt();
            }
            agent_cons.push(row);
        }

        // ---- agent metadata for the lattice hover overlay ----
        let mut agent_meta = Vec::with_capacity(n * 3);
        for row in centers.rows() {
            agent_meta.push(row[0] as u32);
            agent_meta.push(row[1] as u32);
            agent_meta.push(t.patch_size as u32);
        }

        self.cache = Some(SolveCache {
            k: k_total,
            frames,
            residuals,
            agent_cons,
            agent_meta,
        });
        Ok(k_total)
    }

    fn cache(&self) -> Result<&SolveCache, String> {
        self.cache.as_ref().ok_or_else(|| "no solve() has run yet".into())
    }

    /// Per-cell argmax classes `[height*width]` at `iter` (0-based).
    pub fn frame(&self, iter: usize) -> Result<&[u8], String> {
        let c = self.cache()?;
        c.frames
            .get(iter)
            .map(|f| f.as_slice())
            .ok_or_else(|| format!("iter {iter} out of range (k={})", c.k))
    }

    /// `[k*3]` rows of (consistency_rms, primal_rms, dual_rms), batch 0.
    pub fn residuals(&self) -> Result<&[f32], String> {
        Ok(&self.cache()?.residuals)
    }

    /// `[N]` per-agent RMS edge-residual magnitude at `iter`.
    pub fn agent_consistency(&self, iter: usize) -> Result<&[f32], String> {
        let c = self.cache()?;
        c.agent_cons
            .get(iter)
            .map(|f| f.as_slice())
            .ok_or_else(|| format!("iter {iter} out of range (k={})", c.k))
    }

    /// `[N*3]` rows of (center_y, center_x, patch_size) for the current solve.
    pub fn agent_meta(&self) -> Result<&[u32], String> {
        Ok(&self.cache()?.agent_meta)
    }
}

/// Count PATH cells in a frame (test/demo helper).
pub fn path_cells(frame: &[u8]) -> usize {
    frame.iter().filter(|&&v| v as i64 == TOKEN_PATH).count()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use sheaf_io::mazegen::generate_maze;

    fn tokens_u8(maze: &sheaf_io::mazegen::GeneratedMaze) -> Vec<u8> {
        maze.tokens.iter().map(|&v| v as u8).collect()
    }

    /// The same session logic the wasm build ships, natively: embedded bytes
    /// -> solve -> frames/residuals/meta, on both 19x19 and 37x37.
    #[test]
    fn embedded_session_solves_19x19_and_37x37() {
        let mut s = SessionCore::from_embedded().expect("embedded weights load");

        // 19x19, seed 7 — mirror of the Node smoke test.
        let maze = generate_maze(19, 7, 18);
        let toks = tokens_u8(&maze);
        let k = s.solve(&toks, 19, 19, 40).unwrap();
        assert_eq!(k, 40);

        let f_last = s.frame(39).unwrap();
        assert_eq!(f_last.len(), 19 * 19);
        assert!(
            path_cells(f_last) >= 20,
            "final frame has {} PATH cells (< 20)",
            path_cells(f_last)
        );
        assert!(f_last.iter().all(|&v| v < 6));

        let res = s.residuals().unwrap();
        assert_eq!(res.len(), 40 * 3);
        assert!(res.iter().all(|v| v.is_finite()));
        let (p_first, p_last) = (res[1], res[(40 - 1) * 3 + 1]);
        assert!(p_last < p_first, "primal RMS did not decrease: {p_first} -> {p_last}");

        let ac = s.agent_consistency(39).unwrap();
        assert_eq!(ac.len(), 81);
        assert!(ac.iter().all(|v| v.is_finite() && *v >= 0.0));
        let meta = s.agent_meta().unwrap();
        assert_eq!(meta.len(), 81 * 3);
        assert_eq!(&meta[..3], &[1, 1, 3]); // first center (1,1), patch 3
        assert_eq!(meta[80 * 3], 17); // last center y

        // Size-genericity: same session, 37x37 OOD maze.
        let maze2 = generate_maze(37, 1, 36);
        let toks2 = tokens_u8(&maze2);
        let k2 = s.solve(&toks2, 37, 37, 12).unwrap();
        assert_eq!(k2, 12);
        assert_eq!(s.frame(11).unwrap().len(), 37 * 37);
        assert_eq!(s.agent_meta().unwrap().len(), 18 * 18 * 3); // centers 1..35 step 2
        assert_eq!(s.residuals().unwrap().len(), 12 * 3);
    }

    #[test]
    fn solve_rejects_invalid_inputs() {
        let mut s = SessionCore::from_embedded().unwrap();
        // No start/goal.
        let all_empty = vec![2u8; 19 * 19];
        assert!(s.solve(&all_empty, 19, 19, 4).is_err());
        // Length mismatch.
        assert!(s.solve(&all_empty, 19, 18, 4).is_err());
        // Out-of-range token.
        let mut bad = tokens_u8(&generate_maze(19, 0, 18));
        bad[0] = 6;
        assert!(s.solve(&bad, 19, 19, 4).is_err());
        // k = 0 and k > 1000.
        let good = tokens_u8(&generate_maze(19, 0, 18));
        assert!(s.solve(&good, 19, 19, 0).is_err());
        assert!(s.solve(&good, 19, 19, 1001).is_err());
        // Accessors before any successful solve.
        assert!(s.frame(0).is_err());
        assert!(s.residuals().is_err());
        assert!(s.agent_consistency(0).is_err());
        assert!(s.agent_meta().is_err());
    }

    /// A failed solve() must not clobber the previous solve's cache — the
    /// documented contract shared with the JS mock.
    #[test]
    fn failed_solve_preserves_previous_results() {
        let mut s = SessionCore::from_embedded().unwrap();
        let toks = tokens_u8(&generate_maze(19, 7, 18));
        s.solve(&toks, 19, 19, 6).unwrap();
        let frame_before = s.frame(5).unwrap().to_vec();
        let res_before = s.residuals().unwrap().to_vec();

        // Invalid: no start/goal -> error, previous results still readable.
        let all_empty = vec![2u8; 19 * 19];
        assert!(s.solve(&all_empty, 19, 19, 6).is_err());
        assert_eq!(s.frame(5).unwrap(), frame_before.as_slice());
        assert_eq!(s.residuals().unwrap(), res_before.as_slice());
        assert_eq!(s.agent_meta().unwrap().len(), 81 * 3);
    }

    #[test]
    fn frame_iter_bounds_are_checked() {
        let mut s = SessionCore::from_embedded().unwrap();
        let toks = tokens_u8(&generate_maze(19, 3, 18));
        s.solve(&toks, 19, 19, 5).unwrap();
        assert!(s.frame(4).is_ok());
        assert!(s.frame(5).is_err());
        assert!(s.agent_consistency(5).is_err());
    }
}
