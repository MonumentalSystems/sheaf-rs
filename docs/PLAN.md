# Porting SakanaAI/sheaf-admm to Rust — Implementation Plan

**Approach: inference-first port with a browser-grade demo.** Train in the existing JAX codebase; export weights + golden traces; implement the full forward path (encoders, sheaf geometry, unrolled ADMM, decoders, reassembly) in pure Rust on boring, stable crates; ship a WASM in-browser demo as the headline deliverable; keep a fenced, optional hand-gradient milestone for training a minimal config in Rust.

---

## 1. Recommended approach & rationale

Every subsystem analysis reaches the same conclusion: **the forward pass is easy — elementwise ops, closed-form proxes, a matrix-free block-sparse Laplacian matvec, and fixed-T CG — while reverse-mode AD through the K×T unroll is the only hard part.** Inference-first ports 100% of what a demo and a reproducibility artifact need while avoiding the one component with real project risk. Two of three judge lenses (feasibility, cost/verifiability) picked this design; the third (demo value) picked hybrid-minimal purely for its WASM demo, which we graft wholesale — the inference-first core is *more* WASM-friendly than hybrid-minimal's own design (no tape, no rayon requirement in the hot path).

Parity is verifiable incrementally because the Python repo already has the perfect oracle: `coordinate_history` emits per-iteration x/z/y, primal/dual residuals, and consistency RMS, and the test suite pins closed-form solver identities and a dense-reference Laplacian. We mirror those tests and add golden `.npz` trajectory fixtures.

### Rejected alternatives

| Design | Effort | Why rejected |
|---|---|---|
| **Burn full-training port** | 11.5–14 wk | Sound autodiff mapping (`detach` = both `stop_gradient` sites; `select`/`select_assign` are the coboundary adjoint pair), but shipped configs train with **full unroll** (`grad_window=None`, maze K=40) and Burn has no scan/remat — the memory fallback is real work, not contingency. Burn 0.x API churn, no QR for orthonormal init, fragile Flax-pickle name mapping, and parity checks that cost a full training run each. Kept as the **pre-negotiated escape hatch** if a full training port is later funded. |
| **candle port** | 13 wk+ | Thinnest backward-op coverage of the mature frameworks (`index_add` backward historically flaky, may need upstream PRs); no fusion/remat ⇒ 3–10× slower steps; its own trainer design omitted EMA (which is on the critical accuracy path) and got the LR schedule and `precision.py` semantics backwards. Highest residual risk for the longest schedule. |
| **hybrid-minimal (hand-rolled tape, maze only)** | 5 wk claimed, 8–10 realistic | Best demo concept (grafted here), but gates demo quality on a bespoke ~1000-LOC AD engine training to ≥95% with no EMA and no weight-import fallback — a silent-VJP-bug away from consuming its entire schedule. Its verification ladder (per-op finite-diff → tiny e2e finite-diff → `jax.grad` goldens) is grafted as the blueprint for our optional training milestone. |

**Scope statement (honest):** training stays in Python for the paper configs. The Rust artifact is a complete, parity-tested inference engine + demo, with hand-derived gradients demonstrated on one small config as a stretch goal. The README states this explicitly, including the "analytic inner gradients only" limitation. **Explicit non-goals, stated in the README so nothing looks forgotten:** (a) the **MPNN/GGNN baseline** (`models/mpnn.py` + `mpnn_model.py` and the three `*_mpnn` experiment configs — roughly half the models-subsystem LOC) is **not ported**: it exists only as a comparison baseline, shares no code with the sheaf path, and its results are reproducible from the Python repo; the mirrored test suite therefore skips the MPNN param-count pins (182,007 / 48,807), which are listed in the README as intentionally out of scope. (b) The Python **viz package** (~625 LOC) is **superseded, not ported wholesale**: the demo's live per-iteration prediction rendering and residual charts replace `prediction_evolution.pdf` and the residual curves; the two renderers with no demo equivalent — the per-agent `[N,K]` log-scale residual heatmap (`coordination_dynamics.pdf`) and the joint-PCA x/z phase portrait (`xz_trajectories.pdf`, a distinctive paper figure) — are added as cheap static exporters in `sheaf-demo` (the PCA is ~10 lines over faer's SVD, already a pinned dep), fed from the same `AdmmHistory`.

---

## 2. Workspace layout

```
sheaf-rs/
├─ Cargo.toml                      # workspace; every dep version pinned
├─ crates/
│  ├─ sheaf-core/                  # ADMM + sheaf linear algebra. ndarray only.
│  │  │                            # wasm-safe: no threads by default; `parallel`
│  │  │                            # feature gates rayon; `f64` feature for the
│  │  │                            # reference-precision build.
│  │  ├─ src/tensor.rs             # aliases: NodeState = Array3<f32> [N,B,d_v],
│  │  │                            #          EdgeState = Array3<f32> [E,B,d_e]
│  │  ├─ src/ops.rs                # thin tensor-ops facade (all gemm/gather/scatter
│  │  │                            # go through here → framework swap touches 1 file)
│  │  ├─ src/graph.rs              # AgentGraph: edges [E,2], node_pos, dir_uv/dir_vu
│  │  │                            # OR map_u/map_v slot ids, node→edge incidence CSR
│  │  ├─ src/geometry/mod.rs       # trait SheafGeometry
│  │  ├─ src/geometry/fixed.rs     # FixedGeometry
│  │  ├─ src/geometry/lora.rs      # LoraGeometry (edge-pre-gathered factors)
│  │  ├─ src/solvers/x_diag_prox.rs
│  │  ├─ src/solvers/x_simple.rs
│  │  ├─ src/solvers/x_dense_quadratic.rs   # test-only; faer Cholesky, feature-gated
│  │  ├─ src/solvers/z_cg.rs       # unrolled batched CG, project + prox modes
│  │  ├─ src/solvers/z_gd.rs       # analytic-gradient Nesterov GD (see §3)
│  │  ├─ src/admm.rs               # run_admm + run_admm_history (one step fn, two drivers)
│  │  └─ src/history.rs            # AdmmHistory {x,z,y,[K,N,B,d_v]; residuals; rms}
│  ├─ sheaf-nn/                    # inference-only layers + the three task models
│  │  ├─ src/layers.rs             # Dense, RMSNorm(eps 1e-6), LayerNorm(eps 1e-6),
│  │  │                            # tanh-GELU, SiLU, SwiGLU, stable softplus, Mixer block
│  │  ├─ src/encoder/{mlp_v2,mlp,sudoku}.rs
│  │  ├─ src/decoder/{concat_mlp_v2,sudoku,classification}.rs
│  │  ├─ src/restriction_maps.rs   # directional/shared/sudoku base-map assembly
│  │  ├─ src/model.rs              # SheafAdmmModel: encode → geometry → admm → decode
│  │  └─ src/config.rs             # ModelConfig from exported config.json
│  ├─ sheaf-io/                    # safetensors loader → typed param structs
│  │  │                            # (name-map resolved once at load; no string
│  │  │                            #  lookups in the hot path); .npz readers;
│  │  │                            # views port (patchify, sudoku 27-view slice,
│  │  │                            # overlap-mean reassembly, grid edge builder);
│  │  │                            # maze generator (src/mazegen.rs): faithful port
│  │  │                            # of build_maze.py — odd-index-lattice DFS carve,
│  │  │                            # start/goal painting, BFS min-path-length
│  │  │                            # acceptance — shared by sheaf-demo AND sheaf-web
│  │  ├─ src/weights.rs
│  │  ├─ src/views.rs
│  │  ├─ src/mazegen.rs
│  │  └─ src/npz.rs
│  ├─ sheaf-grad/                  # OPTIONAL (M7): hand-derived reverse sweep +
│  │                               # AdamW-lite for one small config
│  ├─ sheaf-demo/                  # native bins: maze_demo (ratatui), sudoku_demo,
│  │                               # parity_check (CI-facing golden replay), gif export,
│  │                               # viz_export (residual heatmap + PCA phase portrait
│  │                               # PNG/SVG from AdmmHistory — see §1 viz note)
│  └─ sheaf-web/                   # cdylib, wasm-bindgen: forward-only session over
│                                  # embedded f16 safetensors; demo/ has canvas UI
└─ tools/
   ├─ export_weights.py            # runs in the repo's uv env: checkpoint.pkl →
   │                               # weights.safetensors (+ EMA!) + config.json
   ├─ dump_goldens.py              # golden .npz fixtures via coordinate_history
   └─ export_data.py               # datasets/test batches → .npz bundles
```

**Crate choices (all mature, no 0.x framework churn):** `ndarray` 0.16 (core tensors), `safetensors` 0.5, `ndarray-npy`/`npyz` (fixtures), `faer` 0.22 **feature-gated** — used only for QR (orthonormal init, if ever needed), the test-only dense-quadratic Cholesky, the viz-export SVD/PCA, and optionally the 256-wide encoder GEMMs; `matrixmultiply` as the default GEMM; `rayon` behind `parallel`; `serde`/`serde_json`; `ratatui` + `crossterm`; `image` + `gif`; `wasm-bindgen` + `trunk`; `half` for f16 weight embedding; `approx` for tolerance assertions. **Not used:** burn, candle, dfdx, sprs — the sheaf structure is better served by the explicit edge-block representation than by any sparse-matrix or AD framework.

---

## 3. Mapping the math onto Rust

### 3.1 Sheaf geometry (matrix-free, never materialized)

The sheaf Laplacian L_F = δᵀδ is an (N·d_v)² operator but is **never formed** — same as the JAX code. Representation:

```rust
pub struct AgentGraph {
    pub edges: Vec<[u32; 2]>,            // [E,2], (u,v)
    pub node_positions: Option<Array2<f32>>, // [N,2] (y,x)
    // precomputed at construction (static per task — was runtime jnp.where in JAX):
    pub dir_uv: Vec<u8>, pub dir_vu: Vec<u8>,  // directional sharing slots
    pub map_u: Vec<u8>,  pub map_v: Vec<u8>,   // sudoku shared-cell slots
    pub node_edges: Csr,                 // node → (edge, sign, endpoint) for parallel scatter
}

pub trait SheafGeometry {
    fn edge_residuals(&self, z: &NodeState) -> EdgeState;      // [E,B,d_e]
    fn laplacian_apply(&self, z: &NodeState, out: &mut NodeState);
    fn energy(&self, z: &NodeState) -> f32;                    // ½ Σ ‖r‖²
    fn consistency_rms(&self, z: &NodeState) -> Array1<f32>;   // [B], eps 1e-6 UNDER sqrt
}

pub struct FixedGeometry {
    graph: Arc<AgentGraph>,
    restriction_maps: Array4<f32>,       // [E,2,d_e,d_v]; slot 0 = F_{u→e}, 1 = F_{v→e}
    edge_mask: Option<Vec<f32>>,         // multiply r once — NOT again on the adjoint
}

pub struct LoraGeometry {
    fixed: FixedGeometry,                // base maps R
    // pre-gathered ONCE per forward (mirrors create_lora_geometry / create_sudoku_lora_geometry):
    a_u: Array4<f32>, a_v: Array4<f32>,  // [E,B,d_e,r]
    b_u: Array4<f32>, b_v: Array4<f32>,  // [E,B,d_v,r]
    gate_u: Option<Array2<f32>>, gate_v: Option<Array2<f32>>,  // [E,B]
    scale: f32,                          // lora_alpha / rank
}
```

- **Coboundary** (`edge_residuals`): per edge, per batch element, r = F_u z_u − F_v z_v. Fixed maps: a 5×10-class GEMV per endpoint (register-resident hand loop; d_e≤32, d_v≤288). LoRA maps stay **factored**: `F z = R z + scale · gate · A (Bᵀ z)` — two rank-r matvecs; F is never materialized.
- **Laplacian matvec**: adjoint per endpoint (`Fᵀ r`, LoRA adjoint `Rᵀr + scale·gate·B(Aᵀr)` — gate placement on the Aᵀr side, matching lora.py), then scatter-add **+contrib at u, −contrib at v**. Duplicate node indices accumulate (JAX `at[].add`) — implemented as a plain `+=` loop over edges. Parallelism: default axis is **batch** (each batch element owns its full [N,d_v] slab — the Laplacian never mixes B); node-parallel path via `node_edges` CSR for B=1 demo mode.
- **Slot-selection asymmetry** (a judged off-by-one magnet): the v-endpoint uses the direction of (−dy,−dx) — an N-edge uses R_N at u and R_S at v; sudoku uses map_u/map_v shared-cell slots. All slot indices are computed **once** at graph construction with ordinary `if/else` (they were `jnp.where` ladders only for traceability), with 4-way vertical priority and the exact 8-way ordering (N,NE,E,SE,S,SW,W,NW). Unit tests pin these tables against golden dumps.
- Simplification the JAX code couldn't take: for fixed directional/sudoku sharing we may index the K base maps directly in the matvec via `(dir_uv[e], dir_vu[e])` and skip assembling [E,2,d_e,d_v] — but only **after** the assembled path passes parity, and behind a flag, so parity debugging always has the literal transcription available.

### 3.2 Solvers

```rust
pub struct EncoderOutput {                // replaces the stringly-typed dict
    pub h: NodeState,
    pub objective: Objective,             // enum per objective_mode
    pub lora: Option<LoraFactors>,        // A [N,B,K,d_e,r], B [N,B,K,d_v,r], gate [N,B,K]
}
pub enum Objective {
    Simple    { beta: f32 },
    Quadratic { q_diag: NodeState, q: NodeState },
    Lasso     { q_diag: NodeState, q: NodeState, l1: f32 },          // scalar (config)
    NonNeg    { q_diag: NodeState, q: NodeState },                   // lower = 0 hardcoded
    L1Box     { q_diag: NodeState, q: NodeState, l1: NodeState, upper: NodeState }, // lower = 0
}
```

- **q_diag head contract**: every diagonal-prox objective computes `q_diag = softplus(Dense(feats)) + q_epsilon` with **q_epsilon = 1e-4** (models/config.py) — the `+1e-4` floor lives in the Rust objective heads, not the exporter, since q_diag is input-dependent. All shipped configs (l1box_diag, lasso, non_negative) go through this path.
- **DiagonalProx** (the paper's x-update, exact closed form, one fused loop over [N,B,d_v]): `a = D + l2 + ρ; t = (ρ(z−y) − q)/a; x = clip(soft_threshold(t, l1/a), lo, hi)` with `soft_threshold(x,θ) = sign(x)·max(|x|−θ, 0)`.
- **Simple**: `x = (β·h + ρ(z−y))/(β+ρ)`. **DenseQuadratic**: test-only, faer Cholesky per (n,b), feature-gated, skipped initially.
- **UnrolledCG**: fixed T (default 5, **no early stopping**), inner products `bdot` reduce over axes (0,2) → per-batch `[B]` scalars, broadcast back as `s[None,:,None]`; denominator guard **exactly 1e-8** (the code comment explains 1e-12 risks 0/0 NaN near fp32 roundoff — do not "improve" it). Project mode: matvec `L + εI` with Tikhonov ε=1e-5, `b = L z_target`, warm start `w₀ = z_target − z_prev`, return `z_target − w` — this is deliberately **not** an idempotent projector; do not fix it or paper results change. Prox mode: matvec `γL + ρ·(·)`, `b = ρ z_target`, init at `z_target` (`prox_init='legacy'`, the **shipped default** — the 'warm' detached-init path is a training-only gradient-boundary detail and is dropped from inference, documented as such).
- **GD z-solver** (ablation only; **no shipped config uses it** — implemented last, or descoped): Nesterov SGD (μ=0.9) with **analytic** gradient. Correcting the judged flaw: prox-mode gradient is `γ·L_F z + ρ·(z − z_target)` — the ρ term is required; project mode is `L_F z` alone. Both are compositions of `laplacian_apply` and elementwise ops; no AD needed.
- **ρ broadcasting**: scalar or [N,B] → [N,B,1]; **α == 1.0 skips the relaxation blend entirely** (bitwise-identical fast path, matching the JAX branch).

### 3.3 ADMM loop

One `step()` function, two drivers (`run_admm`, `run_admm_history`) — the simplification the ground truth recommends:

```rust
pub struct AdmmState { pub x: NodeState, pub z: NodeState, pub y: NodeState }
// init: x = z = z_init (h or zeros), y = 0
// step: z_prev = z;
//       x = x_solver.solve(z − y, ρ, enc);            // prox at v = z − y
//       x_rel = if α == 1.0 { x } else { α·x + (1−α)·z_prev };
//       z = z_solver.solve(x_rel + y, z_prev, geom, ρ);
//       y += x_rel − z;
```

`loss_window`: keep the last W x-iterates (oldest first, clamped to K). `grad_window`/`stop_gradient`/`fori_loop`-vs-`scan` split: **training-only, dropped** — forward values are identical (the Python test `test_truncated_bptt_forward_matches_full_unroll` proves it). `run_admm_history` pushes per-iteration snapshots into a `Vec` and computes `primal_res = ‖x_rel − z‖₂` → [K,N,B], `dual_res = ρ·‖z − z_prev‖₂`, `consistency_rms` → [K,B]. K is a plain runtime loop bound — no per-K recompilation, an ergonomic win over the JAX jit cache.

### 3.4 Numerics contract (pinned, non-negotiable)

- All matmuls in **true fp32** — plain Rust f32 already matches, since `precision.py` *pins JAX to fp32/no-TF32* (this is a drift *reducer*, not a drift source — a judge caught candle inverting this). Never enable TF32 paths if a GPU backend is ever added.
- Exact epsilons: CG denom **1e-8**, project Tikhonov **1e-5**, consistency_rms **1e-6 under the sqrt**, RMSNorm/LayerNorm **1e-6**, **q_epsilon 1e-4** added to every softplus q_diag head (§3.2).
- GELU is the **tanh approximation** (Flax default), softplus is the stable `max(x,0) + log1p(exp(−|x|))` form, `inverse_softplus` clamps ≥1e-7 and is identity above 20.
- Flax Dense kernels are stored **[in, out]** (`y = x·W + b`) — the loader must not transpose.
- A `--features f64` build of sheaf-core exists solely to disambiguate roundoff from bugs during parity debugging.

### 3.5 Models & views

- Encoders/decoders are **shared across agents** via the exact `[N,B,…] → [N·B,…] → apply → reshape back` contract (reshape-back only when leading dim == N·B, i.e. scalars pass through).
- SudokuEncoder pitfalls pinned by tests: the `√d_model/√2` scale is applied **after** adding both position embeddings; cell k occupies contiguous stalk block `[k·32,(k+1)·32)` (row-major reshape — load-bearing for the soft_slice slot maps).
- Export-time baking: **scalar** reparameterizations (ρ, η, β = offset-softplus with config-dependent init constants) are collapsed to their values in `export_weights.py` for the **inference** path — the exporter knows each config's init constants, so inference-side Rust never implements the offset-softplus machinery. (M7's training path is the one exception: `sheaf-grad` carries its own unbaked `softplus(δ + inverse_softplus(init))` reparameterization for the scalars it trains — see §4.) **Input-dependent** heads (q_diag incl. the +1e-4 floor, l1_weight, upper) keep their softplus in Rust. Dropout and all `training=True` branches do not exist in Rust (shipped dropout_rate = 0).
- `sheaf-io/views.rs` ports: grid centers (first at `patch_size//2`, step `stride`, row-major), maze wall-token border pre-pad then one-hot, patchify (centers index the **padded** image, so center == top-left; odd patch sizes assumed), 4/8-conn edge builder (each undirected edge once, oriented right/down), overlap-**mean** reassembly with `max(counts,1)`, sudoku 27-view slice/reassemble with the exact box transpose, and the 243-edge sudoku multigraph **hardcoded as a const table** (deterministic for 9×9). Grid construction is **size-generic** (rebuilt per batch) so OOD 2×/4× mazes work. MNIST/sudoku keep their asymmetric batch-axis conventions ([N,B,…] vs [B,27,…]) — normalize nowhere, match the Python.
- `sheaf-io/mazegen.rs` ports `build_maze.py` faithfully: mazes are carved by DFS on the **odd-index lattice** with strictly-interior checks (valid sizes are odd — 19×19 in-distribution; the OOD suite is **37×37 (2×) and 73×73 (4×)**; even sizes like 38×38 are structurally impossible in-distribution), then start/goal painting and **BFS minimum-path-length acceptance** filtering. Both `sheaf-demo` and `sheaf-web` consume this one implementation, so "Generate" produces genuinely in-distribution mazes in both frontends.

---

## 4. Autodiff / training story

**Tier 1 (shipped): gradients avoided, not ported.** Training remains in JAX. Nothing in inference needs a derivative: x-updates are closed-form, CG is plain ops, and even the GD solver's "gradient of energy" is analytic (§3.2). Both `stop_gradient` sites (grad_window boundary; `prox_init='warm'` detach) are training-only and are dropped with a code comment citing this plan.

**Tier 2 (core scope): golden-trace testing replaces the autodiff we don't write.** See §5.

**Tier 3 (optional M7, fenced): hand-derived reverse sweep for one small config.** Honest framing per the judges: the target config is **"mini-MNIST"** — fixed geometry, pure-quadratic diagonal prox (l1=0), prox-mode CG, and **ρ frozen (`rho_learnable=false`)**, which is *not* the shipped MNIST config (lasso l1=0.006337, z_mode=project, LoRA r=8, ρ learnable by default). Freezing ρ is a deliberate scope fence: a learnable ρ threads gradients through the x-prox denominator, the prox-mode CG matvec (ρ·I term) *and* `b = ρ z_target`, and the dual update, plus its offset-softplus reparameterization — all of that is excluded from M7's VJP surface and documented as such. (If ρ training is ever wanted, `sheaf-grad` already carries the unbaked `softplus(δ + inverse_softplus(init))` form for scalars it *does* train, e.g. β if the mini config uses Simple mode — the exporter's baking is an inference-path convenience, not a global invariant.) The exit criterion is therefore **not** paper parity but: (a) whole-model gradients match `jax.grad` on the *same* mini config (with `rho_learnable=false`) run in the Python repo (a tiny script we add), rtol 1e-3 f32; (b) training that mini config in Rust reaches within ~1% of the same config trained in JAX. Design, following hybrid-minimal's grafted ladder:

1. Manual reverse sweep, not a tape framework. **Memory strategy is checkpoint + recompute**: store per-ADMM-iteration checkpoints (z_k, y_k — tiny); the per-CG-step *vectors* (r_t, p_t, w_t) needed by the CG adjoint are **not stored** — during the reverse sweep, each ADMM iteration's T=5 CG steps are **recomputed forward from the (z_k, y_k) checkpoint** (cheap: 5 matvecs) to rebuild the trajectory, then reversed. Storing only the per-step scalars (α_t, β_t) would be insufficient to reconstruct the recurrence; this is an explicit design point, not an optimization. Every adjoint matvec is `laplacian_apply` again (L is self-adjoint); the VJP w.r.t. restriction maps is a per-edge outer-product accumulation. VJP surface for the mini config: restriction maps, encoder parameters via the objective heads (q_diag incl. its softplus+1e-4, q, h), and decoder — ρ excluded per the fence above.
2. **Explicit unrolled differentiation, never implicit** — the model is trained against the under-solved 5-step CG; implicit/IFT gradients would differentiate a different (converged) map.
3. Subgradient conventions **verified against JAX dumps, not assumed**: golden gradient fixtures deliberately place points *on* the soft-threshold dead zone and *at* the box boundaries (measure-zero kinks that random-point gradcheck misses — a judged concern). Kink-sensitive ops (clip, soft_threshold, max) get dedicated on/off-kink tests.
4. Verification ladder (candle's M1 discipline): per-op central finite differences at f64 → tiny end-to-end finite-diff (N=4, d_v=3, K=3) → `jax.grad` goldens → mini training run.
5. Replicate optax exactly for the mini loop: global-norm clip 1.0 **before** AdamW (b1=0.9, b2=0.999, eps=1e-8, decoupled WD on **all** params — no bias/norm mask), linear warmup then **constant** LR (no decay, no cosine), **EMA shadow from init, decay 0.999, eval on the shadow** — EMA is on the critical accuracy path and is not skippable.

If Tier 3's scope creeps, the pre-negotiated fallback is Burn — and the `ops.rs` facade means that migration touches one layer. The README documents the standing limitation: any future solver with a non-analytic inner gradient is blocked without an AD framework.

---

## 5. Parity-testing strategy

Three layers, all runnable in CI (`parity_check` bin replays goldens).

**5.1 Self-referential property tests** (no Python needed — mirror `tests/`, minus the MPNN param-count pins, which are out of scope per §1):
- Dense coboundary reference: build F ∈ R^{E·d_e × N·d_v} explicitly (tiny sizes); `laplacian_apply(z) == FᵀF z` per batch element (atol/rtol **1e-4**); `energy(z) == ½ zᵀLz == ½ Σ r²`; L symmetric PSD via Rayleigh quotients.
- Closed-form x-solver identities (atol **1e-5**): pure-quadratic stationarity `q_diag·x + q + ρ(x−(z−y)) = 0`; L1-only equals soft-threshold; clip-after ordering; Simple's convex combination; q_diag head output ≥ 1e-4 everywhere (the q_epsilon floor).
- CG: prox mode at T=30 on N·d_v=24 dims → relative residual < **1e-4**; project mode reduces Σr² below **10%** of input.
- LoRA-with-B=0 == FixedGeometry exactly (atol/rtol **1e-5**); `consistency_rms(0) == √1e-6`.
- ADMM: `x_window[last] == state.x`; `loss_window > K` clamps to K.
- Data views: 4×4-center grid → 24 4-conn edges; patchify → mean-reassemble round-trips the image (atol 1e-5); sudoku slice/reassemble round-trip; multigraph = 243 edges, each of 81 cells covered exactly 3×.
- Maze generator: every generated maze (19×19, 37×37, 73×73) is odd-sized, has walls only where the odd-lattice carve allows, and the painted start/goal are BFS-connected with path length ≥ the acceptance threshold — cross-checked against a golden dump of Python-generated mazes at a fixed seed *for the structural invariants* (RNG streams differ; we pin properties, not bits).

**5.2 Golden .npz fixtures from Python** (do **not** replicate `jax.random` — dump values instead):
- `tools/export_weights.py`: Flax pickle → safetensors. Exports **both `params` and `ema_params`** (the visualizer and eval use EMA by default — a judged concern; the Rust loader defaults to EMA). Bakes offset-softplus scalars using each config's init constants (inference path only; see §3.5/§4). Asserts an **exhaustive expected-key manifest per config** (Burn's graft) so upstream renames fail loudly, not silently.
- `tools/dump_goldens.py`: for fixed inputs per task config, dumps every encoder head output, assembled restriction maps, per-iteration ADMM state + residuals via `coordinate_history`, and final logits — the same data the demo renders, so parity tests double as demo-correctness tests.
- Rust asserts **layer-by-layer** at every module boundary (catches transposed [in,out] kernels, the misplaced Sudoku scale factor, a dropped q_epsilon) and **per-iteration trajectory** with a **widening tolerance schedule** (start ~1e-5 rel at iter 1; f32 CG accumulation-order drift compounds over 100 iters). Any divergence is triaged with the `f64` build: if f64 agrees, it's roundoff; if not, it's a bug.
- End acceptance: eval metrics (maze solved/cell_acc incl. OOD splits, mnist acc, sudoku cell/solved/completion) match Python eval on a fixed exported batch to **< 0.1%**, replicating the eval quirks exactly: unweighted per-batch metric means; maze `solved` compares only the PATH_TOKEN=5 mask; **MNIST prediction is the argmax of the *mean of per-agent softmax* over the N agents — not mean logits** (averaging logits is near-identical but not identical and would surface as a sub-0.1% flaky parity failure); sudoku completion is scored on empty cells.

**5.3 Gradient goldens (Tier 3 only):** `jax.grad` parameter-gradient dumps on the mini config (`rho_learnable=false`), with kink-placed inputs, rtol 1e-3 f32.

---

## 6. The demo

**Headline: "Watch 81 agents agree on a maze" — a self-contained in-browser WASM page** (grafted from hybrid-minimal; static deploy to GitHub Pages, no server, no Python at runtime). Trained **EMA** weights (~182K params) ship embedded as f16 safetensors (~370 KB) in a <2 MB wasm module.

- **Left pane:** 19×19 maze editor — click walls, drag start/goal, or "Generate" (the shared `sheaf-io` maze generator — odd-lattice DFS carve + BFS acceptance, §3.5 — compiles into the wasm, so generated mazes are genuinely in-distribution); iteration slider + play/pause up to K=100. The UI labels the two input classes distinctly: *generated* mazes are in-distribution; *hand-edited* mazes may be imperfect or unsolvable (a small client-side BFS check warns "no path exists" rather than letting the model silently under-sell itself — watching the agents disagree on an unsolvable maze is itself a nice demo, but it's labeled as such).
- **Center:** the maze with the overlap-reassembled per-cell argmax rendered live per ADMM iteration — the solution path visibly crystallizes out of local disagreement. Agent-lattice overlay; hovering an agent shows its 3×3 local view, its current x-vs-z disagreement, and its 8 sheaf edges colored by residual magnitude.
- **Right:** canvas charts of consistency RMS, primal residual, dual residual vs iteration (log scale, floor at min positive entry), plummeting as the path locks in.
- **OOD button:** one click runs the *same weights* on a **2× (37×37)** maze — the agent graph is a per-call input, so size generalization (the paper's headline property) demos in one click. **4× (73×73)** available with frame-skipping. (These match the Python `_OOD_SUITE` exactly; even-sized mazes don't exist in this distribution.)
- **Efficiency graft:** the animation decodes iterates from **one** `run_admm_history` pass at K=max, not one forward per k — mathematically identical (iterates don't depend on total K; the Python per-k path's `loss_window=1` final-x equals `history.x[k−1]`).
- **Record GIF** button captures client-side.

**Terminal twin + secondary demos (native):**
```
cargo run --release -p sheaf-demo --bin maze_demo -- \
    --weights export/maze_sheaf.safetensors --data export/maze_test.npz
# space = step, a = animate to K=100, n = new maze (shared sheaf-io generator),
# o = cycle OOD splits (37x37 / 73x73), --gif out.gif
cargo run --release -p sheaf-demo --bin sudoku_demo -- --weights export/sudoku_sheaf_lora.safetensors
# 9×9 grid, per-cell digit beliefs sharpening as 27 row/col/box agents reach consensus;
# violation cells shaded until constraints resolve
cargo run --release -p sheaf-demo --bin viz_export -- --history out.npz --pdf-style paper
# per-agent [N,K] log-residual heatmap + joint-PCA x/z phase portrait (faer SVD),
# replacing the Python viz package's coordination_dynamics / xz_trajectories figures
```
Browser demo: `trunk serve` in `crates/sheaf-web/demo/` (dev), `trunk build --release` → static folder.

**Demo-as-integration-test** (candle's graft): in `--ci` mode the demo binaries run **only the shipped `.npz` test-bundle mazes** (which carry ground-truth paths) and assert final path accuracy > 0.9 plus non-increasing tail consistency-RMS. Generated and hand-edited mazes are excluded from the accuracy assertion — generated mazes have a BFS-derivable path but are not part of the pinned acceptance surface, and edited mazes may have no ground truth at all.

Small pre-exported `.npz` test bundles (a few mazes per split, one sudoku batch, one MNIST batch) ship in-repo so the Rust artifact runs without the Python toolchain.

---

## 7. Milestones

| # | Deliverable | Effort | Exit criteria |
|---|---|---|---|
| **M1** | Export + golden harness: `export_weights.py` (params **and ema_params**, baked scalars, key manifest), `dump_goldens.py` via `coordinate_history`, `export_data.py`; workspace skeleton; `sheaf-io` safetensors/npz loading into typed param structs | 1 wk | Round-trip: Python-exported maze weights load into Rust structs; manifest asserts pass for all shipped configs |
| **M2** | `sheaf-core`: graph (+precomputed direction/slot tables), FixedGeometry + LoraGeometry, all four geometry ops, diagonal-prox/simple x-solvers (incl. q_epsilon floor), unrolled CG (project + prox, legacy init), `run_admm`/`run_admm_history` | 1.5 wk | All §5.1 property tests pass; per-iteration golden parity on a tiny fixed-seed problem over 12 iters (~1e-5) |
| **M3** | Maze path end-to-end: `layers.rs` (tanh-GELU, RMSNorm 1e-6, softplus heads), MLPEncoderV2 + l1box_diag + LoRA heads, ConcatMLPDecoderV2, directional map assembly, views port + `mazegen.rs`, reassembly | 1.5 wk | **First demo-able point.** Maze eval matches JAX to <0.1% at K_eval=100 incl. all OOD splits (37×37, 73×73); layer-by-layer goldens pass; mazegen property tests green |
| **M4** | MNIST + Sudoku paths: residual MLPEncoder + classification decoder (lasso, z_mode=project, mean-of-softmax prediction), Sudoku Mixer encoder (SwiGLU, scale-after-pos-embed, contiguous cell blocks), sudoku sharing + sudoku-LoRA gather, SudokuDecoder | 1.5 wk | Same parity exit criteria per config, incl. sudoku_sheaf_lora (flagged as the fiddliest third — extra golden density here) |
| **M5** | Native demos + perf: maze_demo TUI + GIF, sudoku_demo, `viz_export` (heatmap + PCA phase portrait), rayon batch parallelism, optional faer feature, `f64` build, `parity_check` in CI | 1 wk | ≥30 ADMM iters/sec rendered on a laptop for a 4× (73×73) OOD maze; demo `--ci` assertions green on shipped bundles |
| **M6** | WASM demo: forward-only wasm build, f16 embedded EMA weights, canvas UI (editor with solvability check, animation, hover inspection, residual charts, OOD buttons), shared mazegen, GIF capture; README with architecture walkthrough, explicit non-goals (MPNN baseline; viz superseded), + side-by-side JAX/Rust excerpts | 1.5 wk | <2 MB wasm, smooth animation on a 2020 laptop, deployed to Pages |
| **M7** *(optional)* | `sheaf-grad`: hand-derived reverse sweep for the mini-MNIST config (ρ frozen; CG checkpoint+recompute) per §4 Tier 3; unbaked scalar reparameterization where trained; AdamW-lite + EMA loop | +2 wk | Gradcheck ladder green incl. on-kink fixtures; Rust-trained mini config within ~1% of the *same* config trained in JAX (not a paper-parity claim) |

**Core total: 8 weeks** (6 wk of the original design + 2 wk absorbed by the grafted WASM demo and the heavier parity harness). **With M7: 10 weeks.** Descope order if squeezed: GD z-solver (unused by shipped configs) → LoRA gates (`lora_use_gate` defaults False; no shipped config enables them) → dense-quadratic solver → `viz_export` PCA portrait → sudoku_demo polish → M7.

---

## 8. Risk register

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| **Numeric drift over the unroll** (100 ADMM × 5 CG iters compound f32 reduction-order differences) → phantom parity failures burn time | High / Medium | Widening tolerance schedule; `f64` reference build to classify roundoff vs bug; end acceptance on eval **metrics**, never bitwise; all ops via `ops.rs` so accumulation order is controllable |
| **Checkpoint format coupling**: pickled Flax pytree names follow `nn.compact` module naming; upstream refactor silently breaks the name-map | Medium / High | Exhaustive expected-key manifest per config asserted at export; layer-by-layer activation parity catches survivors; exporter lives in *our* repo and pins the upstream commit |
| **Sudoku/LoRA transcription bugs**: Mixer encoder, dir_uv/dir_vu and map_u/map_v slot asymmetry, gate placement — off-by-one magnets | High / Medium | Densest golden coverage on M4; slot tables unit-tested against dumps; LoRA-with-B=0 == Fixed identity test; hardcoded 243-edge const table cross-checked against the Python builder |
| **Semantics visible only in code**: prox_init legacy default, project-mode non-idempotent warm start, over-relaxation ordering, CG eps 1e-8, q_epsilon 1e-4, α==1.0 fast path | Medium / High | Line-by-line transcription of `admm.py`/`unrolled_cg.py` with cross-referenced comments; §3.4 numerics contract enforced by tests; explicitly do **not** "fix" the non-idempotent projector |
| **EMA/weight-selection mistakes** silently degrade demo quality (paper eval always uses EMA shadow) | Medium / High | Exporter emits both trees; Rust loader **defaults to EMA**; parity fixtures generated from EMA weights |
| **Demo generates out-of-distribution inputs**: a naive maze generator (or even sizes like 38×38) silently under-sells the model | Medium / Medium | One shared `mazegen.rs` port of the odd-lattice DFS + BFS-acceptance builder used by both frontends; OOD sizes pinned to the Python suite (37×37/73×73); hand-edited mazes labeled and solvability-checked in the UI; `--ci` accuracy assertions restricted to shipped ground-truth bundles |
| **M7 hand-gradient bugs** (silent VJP errors at kinks; training limps rather than crashes) | High / Medium (fenced) | Scope fenced to one config with ρ frozen; CG reverse via checkpoint+recompute, never scalar-only reconstruction; on-kink golden fixtures; per-op f64 finite-diff first; pre-negotiated fallback = Burn via the `ops.rs` facade, not tape-hacking |
| **Not fully standalone**: paper-config training, dataset builders, and the MPNN baseline stay in Python | Certain / Low | Ship pre-exported `.npz` bundles + embedded demo weights so the Rust artifact runs cold; README states scope honestly (incl. MPNN as an explicit non-goal and viz-package supersession); wasm demo includes the ported in-distribution maze generator |
| **WASM perf/size**: K=100 at 4× OOD (~1300 agents) forward-only in the browser | Low / Low | ~tens of MFLOPs/iter — fine with wasm SIMD; frame-skip on 4×; f16 weights, `opt-level=z`, no rayon in wasm |
| **Ecosystem drift** (faer, ratatui move fast) | Low / Low | Workspace-pinned versions; faer feature-gated; sheaf-core deps limited to ndarray |
| **GD-solver gradient error propagating from the original design** | Resolved | Prox-mode gradient implemented as `γ·L_F z + ρ·(z − z_target)` (§3.2); covered by a property test; solver is descope-eligible anyway since no shipped config uses it |

---

## Appendix: shapes & defaults cheat sheet (hard-code in tests)

States x,z,y `[N,B,d_v]`; ρ scalar or `[N,B]`→`[N,B,1]`; window output `[W,N,B,d_v]` oldest-first; history `[K,N,B,d_v]`, residuals `[K,N,B]`, consistency `[K,B]`; edge residuals `[E,B,d_e]`; restriction maps `[E,2,d_e,d_v]`; LoRA `A_edge [E,B,d_e,r]`, `B_edge [E,B,d_v,r]`, gate `[E,B]`. Defaults: α=1.0, loss_window=1, CG T=5, Tikhonov 1e-5, prox_init=legacy, γ=1.0, mode=prox, q_epsilon=1e-4, rho_learnable=true (frozen only in the M7 mini config), lora_use_gate=false. Shipped configs: **maze** d_v=10, d_e=5, 8-way directional, LoRA r=4 standard-init, l1box_diag, γ=5, ρ_init=0.25, K_eval=100, 181,859 params; mazes 19×19 in-dist, OOD 37×37/73×73; **mnist** d_v=32, d_e=24, shared map, LoRA r=8 legacy, lasso l1=0.006337, z_mode=project, ρ_init=0.12, linear x-only head, prediction = argmax of mean per-agent softmax; **sudoku** d_v=288 (9×32 contiguous), d_e=32, 9 slot maps soft_slice, non_negative, γ=2; param counts to assert: 543,025 (fixed) / 2,029,233 (LoRA). MPNN configs (`*_mpnn`, param pins 182,007 / 48,807): out of scope, README-documented non-goal.