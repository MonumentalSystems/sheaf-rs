//! sheaf-web: the in-browser WASM demo session (PLAN.md §6, milestone M6).
//!
//! Embeds the trained EMA maze weights (f16 safetensors, ~359 KB) and the
//! exported config, and exposes the JS API contract via wasm-bindgen
//! (`--target web`, ES module):
//!
//! - `generate_maze(seed, height, width) -> Uint8Array` (row-major tokens)
//! - `new SheafSession()` / `session.solve(tokens, height, width, k)`
//! - `session.frame(iter)` / `session.residuals()`
//! - `session.agent_consistency(iter)` / `session.agent_meta()`
//!
//! Token convention (sheaf-io `views::TOKEN_*`): 0 pad, 1 wall, 2 empty,
//! 3 start, 4 goal, 5 path.
//!
//! All logic lives in [`session::SessionCore`] (plain Rust, natively tested);
//! this module is the thin wasm-bindgen shim.

pub mod session;

pub use session::SessionCore;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;

    use crate::session::SessionCore;

    /// `bigint|number -> u64` (the JS contract allows both seed forms).
    fn seed_to_u64(seed: &JsValue) -> Result<u64, JsError> {
        if let Some(f) = seed.as_f64() {
            if f.is_finite() && f >= 0.0 && f.fract() == 0.0 && f <= 9_007_199_254_740_992.0 {
                return Ok(f as u64);
            }
            return Err(JsError::new("seed must be a non-negative integer"));
        }
        u64::try_from(seed.clone())
            .map_err(|_| JsError::new("seed must be a non-negative integer or bigint"))
    }

    /// Generate an in-distribution maze (odd-lattice DFS carve + BFS
    /// acceptance, min path length = max(height, width) - 1, matching the
    /// Python `_OOD_SUITE` thresholds). Returns row-major tokens `[h*w]`.
    #[wasm_bindgen]
    pub fn generate_maze(seed: JsValue, height: u32, width: u32) -> Result<Vec<u8>, JsError> {
        let seed = seed_to_u64(&seed)?;
        let (h, w) = (height as usize, width as usize);
        if h < 3 || w < 3 || h % 2 == 0 || w % 2 == 0 {
            return Err(JsError::new(&format!(
                "maze sizes must be odd and >= 3 (odd-index lattice); got {h}x{w}"
            )));
        }
        let min_path = h.max(w) - 1;
        let maze = sheaf_io::mazegen::generate_maze_hw(h, w, seed, min_path);
        Ok(maze.tokens.iter().map(|&v| v as u8).collect())
    }

    /// A solve session over the embedded f16 EMA weights.
    #[wasm_bindgen]
    pub struct SheafSession {
        core: SessionCore,
    }

    #[wasm_bindgen]
    impl SheafSession {
        /// Load the embedded weights; throws (never panics) on bad weights.
        #[wasm_bindgen(constructor)]
        pub fn new() -> Result<SheafSession, JsError> {
            SessionCore::from_embedded()
                .map(|core| SheafSession { core })
                .map_err(|e| JsError::new(&e))
        }

        /// Run the forward with history at `K = k` on `B = 1`; returns `k`.
        /// Size-generic (19x19 and 37x37 both work). Throws on invalid tokens
        /// or `k` out of range (1..=1000, a memory guard on the `[K,N,B,d_v]`
        /// history). A failed solve leaves the previous successful solve's
        /// frames/residuals readable (validation happens before any state is
        /// replaced) — same semantics as the JS mock.
        pub fn solve(
            &mut self,
            tokens: &[u8],
            height: u32,
            width: u32,
            k: u32,
        ) -> Result<u32, JsError> {
            self.core
                .solve(tokens, height as usize, width as usize, k as usize)
                .map(|k| k as u32)
                .map_err(|e| JsError::new(&e))
        }

        /// Per-cell argmax class `[height*width]` at `iter` (overlap-mean
        /// reassembled logits).
        pub fn frame(&self, iter: u32) -> Result<Vec<u8>, JsError> {
            self.core
                .frame(iter as usize)
                .map(<[u8]>::to_vec)
                .map_err(|e| JsError::new(&e))
        }

        /// `[k*3]` rows of (consistency_rms, primal_rms, dual_rms), batch 0.
        pub fn residuals(&self) -> Result<Vec<f32>, JsError> {
            self.core
                .residuals()
                .map(<[f32]>::to_vec)
                .map_err(|e| JsError::new(&e))
        }

        /// `[N]` per-agent RMS edge-residual magnitude at `iter`.
        pub fn agent_consistency(&self, iter: u32) -> Result<Vec<f32>, JsError> {
            self.core
                .agent_consistency(iter as usize)
                .map(<[f32]>::to_vec)
                .map_err(|e| JsError::new(&e))
        }

        /// `[N*3]` rows of (center_y, center_x, patch_size).
        pub fn agent_meta(&self) -> Result<Vec<u32>, JsError> {
            self.core
                .agent_meta()
                .map(<[u32]>::to_vec)
                .map_err(|e| JsError::new(&e))
        }
    }
}
