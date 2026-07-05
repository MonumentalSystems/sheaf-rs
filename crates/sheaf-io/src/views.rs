//! Views: the map between the global grid and per-agent local patches.
//! Ports `sheaf_admm.data.views` (grid-task subset — maze scope).
//!
//! Pinned semantics (PLAN.md §3.5):
//! - centers: first at `patch_size // 2` per axis, step `stride`, row-major
//!   (`[N, 2]` integer (y, x));
//! - maze pre-pad: border of width `patch_size // 2` filled with the WALL
//!   token (= 1) BEFORE extraction; centers shift by the border; then one-hot;
//! - patchify indexes the padded image, so a center is the patch top-left;
//! - edge builder: each undirected edge once, oriented right/down (offsets
//!   scanned in the exact Python order), 4- or 8-connectivity between centers
//!   `stride` apart;
//! - reassembly: overlap-MEAN with `max(counts, 1)` guard;
//! - grid construction is size-generic (rebuilt per batch) so OOD 37x37 /
//!   73x73 mazes work.

use std::collections::HashMap;

use ndarray::{Array2, Array3, Array4, Array5};

use sheaf_core::graph::{AgentGraph, NodeIncidence};

/// Maze token ids — the Python `data/common.py` `TOKEN_IDS` convention:
/// slot 0 is the pad/ignore id; inputs carry 1-4; labels add 5 (path).
/// vocab_size = 6. Maze `solved` eval compares only the `TOKEN_PATH = 5` mask.
pub const TOKEN_PAD: i64 = 0;
pub const TOKEN_WALL: i64 = 1;
pub const TOKEN_EMPTY: i64 = 2;
pub const TOKEN_START: i64 = 3;
pub const TOKEN_GOAL: i64 = 4;
pub const TOKEN_PATH: i64 = 5;

/// Agent centers on a regular grid: `range(patch_size // 2, dim, stride)` per
/// axis, row-major -> `[N, 2]` (y, x). Ports `grid_agent_centers`.
pub fn grid_agent_centers(
    image_hw: (usize, usize),
    stride: usize,
    patch_size: usize,
) -> Array2<i64> {
    assert!(stride > 0, "stride must be positive");
    assert!(patch_size > 0, "patch_size must be positive");
    let (h, w) = image_hw;
    let center = patch_size / 2;
    let mut coords: Vec<i64> = Vec::new();
    let mut y = center;
    while y < h {
        let mut x = center;
        while x < w {
            coords.push(y as i64);
            coords.push(x as i64);
            x += stride;
        }
        y += stride;
    }
    Array2::from_shape_vec((coords.len() / 2, 2), coords).expect("centers shape")
}

/// 4-/8-conn edges between neighboring centers -> `[E, 2]` (u, v), each
/// undirected edge once, oriented right/down. Ports `build_grid_edge_indices`.
///
/// Offset order (per center, centers scanned in index order) is exactly the
/// Python one: `(0, s), (s, 0)` for 4-conn, plus `(s, s), (s, -s)` for 8-conn.
pub fn build_grid_edge_indices(
    centers: &Array2<i64>,
    stride: usize,
    connectivity: usize,
) -> Vec<[u32; 2]> {
    assert!(stride > 0, "stride must be positive");
    assert!(
        connectivity == 4 || connectivity == 8,
        "connectivity must be 4 or 8"
    );
    let s = stride as i64;
    let mut center_to_idx: HashMap<(i64, i64), u32> = HashMap::with_capacity(centers.nrows());
    for (i, row) in centers.rows().into_iter().enumerate() {
        center_to_idx.insert((row[0], row[1]), i as u32);
    }
    let mut offsets: Vec<(i64, i64)> = vec![(0, s), (s, 0)];
    if connectivity == 8 {
        offsets.push((s, s));
        offsets.push((s, -s));
    }

    let mut edges: Vec<[u32; 2]> = Vec::new();
    for (i, row) in centers.rows().into_iter().enumerate() {
        let (cy, cx) = (row[0], row[1]);
        for &(dy, dx) in &offsets {
            if let Some(&j) = center_to_idx.get(&(cy + dy, cx + dx)) {
                edges.push([i as u32, j]);
            }
        }
    }
    edges
}

/// Wall-token border pre-pad + one-hot + patchify. Ports `prepare_maze_patches`
/// (which calls `patchify_batch_jax` on the wall-bordered one-hot image).
///
/// `tokens: [B, H, W]` int -> patches `[N, B, ps, ps, num_input_classes]` f32.
///
/// The image is first extended by a border of width `patch_size // 2` filled
/// with `TOKEN_WALL` (so boundary agents see walls, not zeros) and centers
/// shift by the border; then patchify's own zero-pad of `patch_size // 2`
/// makes each shifted center the patch **top-left** on the doubly-padded
/// grid — which keeps the patch centered on the original center.
pub fn prepare_maze_patches(
    tokens: &Array3<i64>,
    centers: &Array2<i64>,
    patch_size: usize,
    num_input_classes: usize,
) -> Array5<f32> {
    assert!(patch_size > 0, "patch_size must be positive");
    let border = patch_size / 2;
    let pad = patch_size / 2; // patchify's own zero pad
    let (b, eh, ew) = tokens.dim();
    let n = centers.nrows();
    let c = num_input_classes;

    // Bordered token grid: [B, eh + 2*border, ew + 2*border], wall fill.
    let (bh, bw) = (eh + 2 * border, ew + 2 * border);
    let mut bordered = Array3::<i64>::from_elem((b, bh, bw), TOKEN_WALL);
    for bi in 0..b {
        for y in 0..eh {
            for x in 0..ew {
                bordered[[bi, y + border, x + border]] = tokens[[bi, y, x]];
            }
        }
    }

    // One-hot + zero-pad by `pad`, fused: the padded one-hot is zero outside
    // the bordered image; inside, channel `tok` is 1.0 (tokens outside 0..C
    // one-hot to all zeros — jax.nn.one_hot semantics).
    let mut patches = Array5::<f32>::zeros((n, b, patch_size, patch_size, c));
    for (a, row) in centers.rows().into_iter().enumerate() {
        // Shifted center = patch top-left in padded coords.
        let (ty, tx) = (row[0] + border as i64, row[1] + border as i64);
        for bi in 0..b {
            for py in 0..patch_size {
                for px in 0..patch_size {
                    // padded coord -> bordered coord (padded = bordered + pad)
                    let by = ty + py as i64 - pad as i64;
                    let bx = tx + px as i64 - pad as i64;
                    if by < 0 || bx < 0 || by >= bh as i64 || bx >= bw as i64 {
                        continue; // patchify's zero padding
                    }
                    let tok = bordered[[bi, by as usize, bx as usize]];
                    if tok >= 0 && (tok as usize) < c {
                        patches[[a, bi, py, px, tok as usize]] = 1.0;
                    }
                }
            }
        }
    }
    patches
}

/// Plain zero-padded patch extraction over a float image batch (the MNIST
/// path). Ports `patchify_batch_jax`: the image is zero-padded by
/// `patch_size // 2` on each spatial side, then a `patch_size` patch is sliced
/// with its top-left at each center on the padded grid (so the patch stays
/// centered on the original center).
///
/// `images: [B, H, W, C]` f32, `centers: [N, 2]` (y, x) -> `[N, B, ps, ps, C]`.
/// Unlike `prepare_maze_patches` there is no wall border and no one-hot: MNIST
/// feeds raw normalized pixels (C = 1).
pub fn patchify_batch(
    images: &Array4<f32>,
    centers: &Array2<i64>,
    patch_size: usize,
) -> Array5<f32> {
    assert!(patch_size > 0, "patch_size must be positive");
    let (b, h, w, c) = images.dim();
    let n = centers.nrows();
    let pad = (patch_size / 2) as i64;

    let mut patches = Array5::<f32>::zeros((n, b, patch_size, patch_size, c));
    for (a, row) in centers.rows().into_iter().enumerate() {
        let (cy, cx) = (row[0], row[1]);
        for py in 0..patch_size {
            for px in 0..patch_size {
                // padded coord (cy+py, cx+px) maps to image coord minus pad.
                let iy = cy + py as i64 - pad;
                let ix = cx + px as i64 - pad;
                if iy < 0 || ix < 0 || iy >= h as i64 || ix >= w as i64 {
                    continue; // zero pad
                }
                for bi in 0..b {
                    for ch in 0..c {
                        patches[[a, bi, py, px, ch]] =
                            images[[bi, iy as usize, ix as usize, ch]];
                    }
                }
            }
        }
    }
    patches
}

/// In-bounds image span and matching local-patch origin for one agent.
/// Ports `_overlap_slices`: returns `((y_start, y_end), (x_start, x_end),
/// patch_y_start, patch_x_start)`.
fn overlap_slices(
    center_yx: (i64, i64),
    patch_size: usize,
    image_hw: (usize, usize),
) -> ((i64, i64), (i64, i64), i64, i64) {
    let (h, w) = (image_hw.0 as i64, image_hw.1 as i64);
    let pad = (patch_size / 2) as i64;
    let (cy, cx) = center_yx;
    let (y0, x0) = (cy - pad, cx - pad);

    let (y_start, x_start) = (y0.max(0), x0.max(0));
    let y_end = (y0 + patch_size as i64).min(h);
    let x_end = (x0 + patch_size as i64).min(w);

    ((y_start, y_end), (x_start, x_end), y_start - y0, x_start - x0)
}

/// Overlap-mean reassembly of per-agent patch logits into a global logit map.
/// Ports `reassemble_logits(mode="mean")`:
/// `patches [N, B, ph, pw, C]` + `centers [N, 2]` -> `[B, H, W, C]`,
/// accumulating overlapping contributions and dividing by `max(counts, 1)`.
pub fn reassemble_logits(
    patch_logits: &Array5<f32>,
    centers: &Array2<i64>,
    image_hw: (usize, usize),
) -> Array4<f32> {
    let (n, b, ph, pw, c) = patch_logits.dim();
    assert_eq!(ph, pw, "patches must be square");
    assert_eq!(
        centers.nrows(),
        n,
        "centers_yx must align with patches on the agent axis"
    );
    let (h, w) = image_hw;

    let mut logits_sum = Array4::<f32>::zeros((b, h, w, c));
    // Counts are batch/channel independent; keep [H, W].
    let mut counts = Array2::<f32>::zeros((h, w));

    for a in 0..n {
        let center = (centers[[a, 0]], centers[[a, 1]]);
        let ((ys, ye), (xs, xe), py0, px0) = overlap_slices(center, ph, (h, w));
        if ys == ye || xs == xe {
            continue;
        }
        for bi in 0..b {
            for y in ys..ye {
                for x in xs..xe {
                    let py = (py0 + (y - ys)) as usize;
                    let px = (px0 + (x - xs)) as usize;
                    for ch in 0..c {
                        logits_sum[[bi, y as usize, x as usize, ch]] +=
                            patch_logits[[a, bi, py, px, ch]];
                    }
                }
            }
        }
        for y in ys..ye {
            for x in xs..xe {
                counts[[y as usize, x as usize]] += 1.0;
            }
        }
    }

    for bi in 0..b {
        for y in 0..h {
            for x in 0..w {
                let denom = counts[[y, x]].max(1.0);
                for ch in 0..c {
                    logits_sum[[bi, y, x, ch]] /= denom;
                }
            }
        }
    }
    logits_sum
}

// =============================================================================
// Sudoku views: 27-agent constraint slices, reassembly, and the 243-edge
// multigraph. Ports `sheaf_admm.data.views` (sudoku subset).
// =============================================================================

/// Number of Sudoku constraint agents (9 rows + 9 cols + 9 boxes).
pub const SUDOKU_NUM_AGENTS: usize = 27;
/// Number of directed sharing edges (81 cells x 3-clique = 243).
pub const SUDOKU_NUM_EDGES: usize = 243;

/// Slice a `[B, 9, 9, C]` grid into the 27 constraint-agent views
/// `[B, 27, 9, C]`. Ports `sudoku_slice_batch_jax`:
/// - agents 0-8 rows: agent `r`, slot `col` -> `grid[r, col]`;
/// - agents 9-17 columns (transpose): agent `9+col`, slot `r` -> `grid[r, col]`;
/// - agents 18-26 the 3x3 boxes: `(3,3,3,3)` reshape then `(0,2,1,3)` transpose,
///   so box `(br, bc)` is agent `18 + br*3 + bc` and within-box slot `ir*3 + ic`
///   maps to `grid[br*3+ir, bc*3+ic]` (row-major within each box).
pub fn sudoku_slice_batch(grids: &Array4<f32>) -> Array4<f32> {
    let (b, h, w, c) = grids.dim();
    assert_eq!((h, w), (9, 9), "sudoku grids must be [B, 9, 9, C]");
    let mut out = Array4::<f32>::zeros((b, SUDOKU_NUM_AGENTS, 9, c));
    for bi in 0..b {
        for ch in 0..c {
            // Rows.
            for r in 0..9 {
                for col in 0..9 {
                    out[[bi, r, col, ch]] = grids[[bi, r, col, ch]];
                }
            }
            // Columns (transpose).
            for col in 0..9 {
                for r in 0..9 {
                    out[[bi, 9 + col, r, ch]] = grids[[bi, r, col, ch]];
                }
            }
            // Boxes ((3,3,3,3) reshape, (0,2,1,3) transpose, row-major within-box).
            for br in 0..3 {
                for bc in 0..3 {
                    for ir in 0..3 {
                        for ic in 0..3 {
                            let agent = 18 + br * 3 + bc;
                            let slot = ir * 3 + ic;
                            out[[bi, agent, slot, ch]] = grids[[bi, br * 3 + ir, bc * 3 + ic, ch]];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Average the three covering views back to a 9x9 grid. Ports
/// `reassemble_sudoku_logits`: `[B, 27, 9, C] -> [B, 9, 9, C]`, the mean of the
/// row, column, and box reconstructions (inverse of [`sudoku_slice_batch`]).
pub fn reassemble_sudoku_logits(views: &Array4<f32>) -> Array4<f32> {
    let (b, a, s, c) = views.dim();
    assert_eq!((a, s), (SUDOKU_NUM_AGENTS, 9), "sudoku views must be [B, 27, 9, C]");
    let mut out = Array4::<f32>::zeros((b, 9, 9, c));
    for bi in 0..b {
        for ch in 0..c {
            for r in 0..9 {
                for col in 0..9 {
                    let from_rows = views[[bi, r, col, ch]];
                    let from_cols = views[[bi, 9 + col, r, ch]];
                    // Box that owns (r, col): br=r/3, bc=col/3, slot=(r%3)*3+(col%3).
                    let agent = 18 + (r / 3) * 3 + (col / 3);
                    let slot = (r % 3) * 3 + (col % 3);
                    let from_boxes = views[[bi, agent, slot, ch]];
                    out[[bi, r, col, ch]] = (from_rows + from_cols + from_boxes) / 3.0;
                }
            }
        }
    }
    out
}

/// Global cell ids (0..80) seen by each of the 27 agents at each of its 9 local
/// slots -> `[27, 9]` i64. Ports `build_sudoku_cell_indices` (a slice of the
/// `arange(81)` grid); feeds the encoder's absolute-position `Embed(81)`.
pub fn build_sudoku_cell_indices() -> Array2<i64> {
    let mut ids = Array2::<i64>::zeros((SUDOKU_NUM_AGENTS, 9));
    for r in 0..9 {
        for col in 0..9 {
            ids[[r, col]] = (r * 9 + col) as i64; // rows
        }
    }
    for col in 0..9 {
        for r in 0..9 {
            ids[[9 + col, r]] = (r * 9 + col) as i64; // columns
        }
    }
    for br in 0..3 {
        for bc in 0..3 {
            for ir in 0..3 {
                for ic in 0..3 {
                    let agent = 18 + br * 3 + bc;
                    let slot = ir * 3 + ic;
                    ids[[agent, slot]] = ((br * 3 + ir) * 9 + (bc * 3 + ic)) as i64;
                }
            }
        }
    }
    ids
}

/// Build the Sudoku constraint multigraph over the 27 agents (generative port of
/// `build_sudoku_multigraph`). Each of the 81 cells is covered by exactly 3
/// agents (its row, column, box); their 3-clique gives 3 edges per cell, 243
/// total. Returns `(edges [E,2], map_u [E], map_v [E])` where `map_*` is the
/// local 0-8 slot holding the shared cell at each endpoint (selects the per-slot
/// restriction map / LoRA factor). Cross-checked against the hardcoded const
/// golden [`SUDOKU_EDGE_U`] etc. in tests.
pub fn build_sudoku_multigraph() -> (Vec<[u32; 2]>, Vec<u8>, Vec<u8>) {
    let cell_ids = build_sudoku_cell_indices();
    // coverage[cell] = list of (agent, local_slot), agents in ascending order
    // (we iterate agents outermost, so each cell's endpoints stay sorted -> u<v).
    let mut coverage: Vec<Vec<(u32, u8)>> = vec![Vec::new(); 81];
    for agent in 0..SUDOKU_NUM_AGENTS {
        for local in 0..9 {
            let cid = cell_ids[[agent, local]] as usize;
            coverage[cid].push((agent as u32, local as u8));
        }
    }
    let mut edges = Vec::with_capacity(SUDOKU_NUM_EDGES);
    let mut map_u = Vec::with_capacity(SUDOKU_NUM_EDGES);
    let mut map_v = Vec::with_capacity(SUDOKU_NUM_EDGES);
    for agents in &coverage {
        for i in 0..agents.len() {
            for j in (i + 1)..agents.len() {
                let (u, ul) = agents[i];
                let (v, vl) = agents[j];
                edges.push([u, v]);
                map_u.push(ul);
                map_v.push(vl);
            }
        }
    }
    (edges, map_u, map_v)
}

/// The Sudoku 27-agent constraint multigraph as an [`AgentGraph`], with the
/// per-endpoint cell-slot tables (`map_u`/`map_v`) carried on `dir_uv`/`dir_vu`
/// (K = 9). This lets the base-map assembly and the LoRA gather reuse the shared
/// directional machinery: `R_indices[map_u], R_indices[map_v]` and the endpoint
/// factor gather select by cell-slot exactly as the Python sudoku path does.
pub fn build_sudoku_graph() -> AgentGraph {
    let (edges, map_u, map_v) = build_sudoku_multigraph();
    let node_edges = NodeIncidence::build(&edges, SUDOKU_NUM_AGENTS);
    AgentGraph {
        edges,
        node_positions: None,
        dir_uv: map_u,
        dir_vu: map_v,
        node_edges,
        num_nodes: SUDOKU_NUM_AGENTS,
    }
}

/// Sudoku prediction: reassemble the per-agent final logits `[N, B, 9, C]`
/// (N = 27) into the `[B, 9, 9]` argmax digit grid. Mirrors
/// `SudokuTask.evaluate`: transpose to `[B, 27, 9, C]`, [`reassemble_sudoku_logits`],
/// then argmax over the class axis (first-max tie-break = `jnp.argmax`).
pub fn sudoku_predict(logits_final: &Array4<f32>) -> Array3<i64> {
    let (n, b, s, c) = logits_final.dim();
    assert_eq!((n, s), (SUDOKU_NUM_AGENTS, 9), "logits must be [27, B, 9, C]");
    // Transpose [N, B, 9, C] -> [B, 27, 9, C].
    let mut views = Array4::<f32>::zeros((b, n, s, c));
    for ni in 0..n {
        for bi in 0..b {
            for si in 0..s {
                for ci in 0..c {
                    views[[bi, ni, si, ci]] = logits_final[[ni, bi, si, ci]];
                }
            }
        }
    }
    let recon = reassemble_sudoku_logits(&views); // [B, 9, 9, C]
    let mut pred = Array3::<i64>::zeros((b, 9, 9));
    for bi in 0..b {
        for r in 0..9 {
            for col in 0..9 {
                let (mut arg, mut best) = (0usize, f32::NEG_INFINITY);
                for ci in 0..c {
                    let v = recon[[bi, r, col, ci]];
                    if v > best {
                        best = v;
                        arg = ci;
                    }
                }
                pred[[bi, r, col]] = arg as i64;
            }
        }
    }
    pred
}

/// Sudoku eval metrics (PLAN §5.2), matching `SudokuTask.evaluate`. All means
/// are unweighted over the batch; **completion is scored on EMPTY cells only**
/// (`inputs == 0`), guarded by `max(sum(empty), 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SudokuMetrics {
    pub cell_acc: f32,
    pub solved: f32,
    pub completion: f32,
}

/// Compute [`SudokuMetrics`] from the predicted grid `[B, 9, 9]`, the label grid
/// `[B, 9, 9]`, and the input (givens) grid `[B, 9, 9]` (empty cells are `== 0`).
pub fn sudoku_metrics(
    pred: &Array3<i64>,
    labels: &Array3<i64>,
    inputs: &Array3<i64>,
) -> SudokuMetrics {
    let (b, h, w) = pred.dim();
    assert_eq!(pred.dim(), labels.dim(), "pred/labels shape mismatch");
    assert_eq!(pred.dim(), inputs.dim(), "pred/inputs shape mismatch");
    let total_cells = (b * h * w) as f32;
    let mut correct = 0.0f32;
    let mut solved = 0.0f32;
    let mut empty_correct = 0.0f32;
    let mut empty_total = 0.0f32;
    for bi in 0..b {
        let mut all = true;
        for r in 0..h {
            for col in 0..w {
                let hit = pred[[bi, r, col]] == labels[[bi, r, col]];
                if hit {
                    correct += 1.0;
                } else {
                    all = false;
                }
                if inputs[[bi, r, col]] == 0 {
                    empty_total += 1.0;
                    if hit {
                        empty_correct += 1.0;
                    }
                }
            }
        }
        if all {
            solved += 1.0;
        }
    }
    SudokuMetrics {
        cell_acc: correct / total_cells,
        solved: solved / b as f32,
        completion: empty_correct / empty_total.max(1.0),
    }
}

/// Hardcoded golden edge/slot tables (the pinned output of the Python
/// `build_sudoku_multigraph(9)`; deterministic for the 9x9 board). The
/// generative [`build_sudoku_multigraph`] is cross-checked against these in
/// tests, so a transcription slip in either surfaces immediately.
#[rustfmt::skip]
pub const SUDOKU_EDGE_U: [u32; SUDOKU_NUM_EDGES] = [
    0,0,9,0,0,10,0,0,11,0,0,12,0,0,13,0,0,14,0,0,15,0,0,16,0,0,17,1,1,9,1,1,10,1,1,11,1,1,12,1,1,13,1,1,14,1,1,15,1,1,16,1,1,17,2,2,9,2,2,10,2,2,11,2,2,12,2,2,13,2,2,14,2,2,15,2,2,16,2,2,17,3,3,9,3,3,10,3,3,11,3,3,12,3,3,13,3,3,14,3,3,15,3,3,16,3,3,17,4,4,9,4,4,10,4,4,11,4,4,12,4,4,13,4,4,14,4,4,15,4,4,16,4,4,17,5,5,9,5,5,10,5,5,11,5,5,12,5,5,13,5,5,14,5,5,15,5,5,16,5,5,17,6,6,9,6,6,10,6,6,11,6,6,12,6,6,13,6,6,14,6,6,15,6,6,16,6,6,17,7,7,9,7,7,10,7,7,11,7,7,12,7,7,13,7,7,14,7,7,15,7,7,16,7,7,17,8,8,9,8,8,10,8,8,11,8,8,12,8,8,13,8,8,14,8,8,15,8,8,16,8,8,17,
];
#[rustfmt::skip]
pub const SUDOKU_EDGE_V: [u32; SUDOKU_NUM_EDGES] = [
    9,18,18,10,18,18,11,18,18,12,19,19,13,19,19,14,19,19,15,20,20,16,20,20,17,20,20,9,18,18,10,18,18,11,18,18,12,19,19,13,19,19,14,19,19,15,20,20,16,20,20,17,20,20,9,18,18,10,18,18,11,18,18,12,19,19,13,19,19,14,19,19,15,20,20,16,20,20,17,20,20,9,21,21,10,21,21,11,21,21,12,22,22,13,22,22,14,22,22,15,23,23,16,23,23,17,23,23,9,21,21,10,21,21,11,21,21,12,22,22,13,22,22,14,22,22,15,23,23,16,23,23,17,23,23,9,21,21,10,21,21,11,21,21,12,22,22,13,22,22,14,22,22,15,23,23,16,23,23,17,23,23,9,24,24,10,24,24,11,24,24,12,25,25,13,25,25,14,25,25,15,26,26,16,26,26,17,26,26,9,24,24,10,24,24,11,24,24,12,25,25,13,25,25,14,25,25,15,26,26,16,26,26,17,26,26,9,24,24,10,24,24,11,24,24,12,25,25,13,25,25,14,25,25,15,26,26,16,26,26,17,26,26,
];
#[rustfmt::skip]
pub const SUDOKU_MAP_U: [u8; SUDOKU_NUM_EDGES] = [
    0,0,0,1,1,0,2,2,0,3,3,0,4,4,0,5,5,0,6,6,0,7,7,0,8,8,0,0,0,1,1,1,1,2,2,1,3,3,1,4,4,1,5,5,1,6,6,1,7,7,1,8,8,1,0,0,2,1,1,2,2,2,2,3,3,2,4,4,2,5,5,2,6,6,2,7,7,2,8,8,2,0,0,3,1,1,3,2,2,3,3,3,3,4,4,3,5,5,3,6,6,3,7,7,3,8,8,3,0,0,4,1,1,4,2,2,4,3,3,4,4,4,4,5,5,4,6,6,4,7,7,4,8,8,4,0,0,5,1,1,5,2,2,5,3,3,5,4,4,5,5,5,5,6,6,5,7,7,5,8,8,5,0,0,6,1,1,6,2,2,6,3,3,6,4,4,6,5,5,6,6,6,6,7,7,6,8,8,6,0,0,7,1,1,7,2,2,7,3,3,7,4,4,7,5,5,7,6,6,7,7,7,7,8,8,7,0,0,8,1,1,8,2,2,8,3,3,8,4,4,8,5,5,8,6,6,8,7,7,8,8,8,8,
];
#[rustfmt::skip]
pub const SUDOKU_MAP_V: [u8; SUDOKU_NUM_EDGES] = [
    0,0,0,0,1,1,0,2,2,0,0,0,0,1,1,0,2,2,0,0,0,0,1,1,0,2,2,1,3,3,1,4,4,1,5,5,1,3,3,1,4,4,1,5,5,1,3,3,1,4,4,1,5,5,2,6,6,2,7,7,2,8,8,2,6,6,2,7,7,2,8,8,2,6,6,2,7,7,2,8,8,3,0,0,3,1,1,3,2,2,3,0,0,3,1,1,3,2,2,3,0,0,3,1,1,3,2,2,4,3,3,4,4,4,4,5,5,4,3,3,4,4,4,4,5,5,4,3,3,4,4,4,4,5,5,5,6,6,5,7,7,5,8,8,5,6,6,5,7,7,5,8,8,5,6,6,5,7,7,5,8,8,6,0,0,6,1,1,6,2,2,6,0,0,6,1,1,6,2,2,6,0,0,6,1,1,6,2,2,7,3,3,7,4,4,7,5,5,7,3,3,7,4,4,7,5,5,7,3,3,7,4,4,7,5,5,8,6,6,8,7,7,8,8,8,8,6,6,8,7,7,8,8,8,8,6,6,8,7,7,8,8,8,
];

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn centers_19x19_ps3_stride2_is_9x9_row_major() {
        let centers = grid_agent_centers((19, 19), 2, 3);
        assert_eq!(centers.dim(), (81, 2));
        // First at patch_size // 2 = 1, step 2 -> 1,3,...,17 per axis.
        assert_eq!(centers.row(0).to_vec(), vec![1, 1]);
        assert_eq!(centers.row(1).to_vec(), vec![1, 3]); // row-major: x fastest
        assert_eq!(centers.row(9).to_vec(), vec![3, 1]);
        assert_eq!(centers.row(80).to_vec(), vec![17, 17]);
    }

    #[test]
    fn four_by_four_center_grid_has_24_four_conn_edges() {
        // PLAN.md §5.1: 4x4-center grid -> 24 4-conn edges.
        let centers = grid_agent_centers((8, 8), 2, 3); // centers 1,3,5,7 per axis
        assert_eq!(centers.nrows(), 16);
        let edges = build_grid_edge_indices(&centers, 2, 4);
        assert_eq!(edges.len(), 24); // 2 * 4 * 3

        // Each undirected edge once, oriented right/down (u < v always here).
        for &[u, v] in &edges {
            assert!(u < v, "edge ({u},{v}) not oriented right/down");
        }
        // First center's edges appear first: (0,1) right then (0,4) down.
        assert_eq!(edges[0], [0, 1]);
        assert_eq!(edges[1], [0, 4]);
    }

    #[test]
    fn maze_contract_grid_sizes() {
        // goldens/CONTRACT.md: 19x19, ps=3, stride=2, conn=8 -> N=81, E=272.
        let centers = grid_agent_centers((19, 19), 2, 3);
        let edges = build_grid_edge_indices(&centers, 2, 8);
        assert_eq!(centers.nrows(), 81);
        assert_eq!(edges.len(), 272);
    }

    #[test]
    fn eight_conn_includes_down_left() {
        let centers = grid_agent_centers((8, 8), 2, 3); // 4x4 centers
        let edges = build_grid_edge_indices(&centers, 2, 8);
        // 4x4 centers: horizontal 12 + vertical 12 + diag 9 + anti-diag 9.
        assert_eq!(edges.len(), 42);
        // Center 1 (y=1,x=3) has a down-left neighbor: center 4 (y=3,x=1).
        assert!(edges.contains(&[1, 4]));
    }

    #[test]
    fn patchify_reassemble_round_trips() {
        // PLAN.md §5.1: patchify -> mean-reassemble round-trips the image.
        // Full coverage (stride <= patch_size) so every pixel is covered.
        let (h, w, b) = (7, 7, 2);
        let mut tokens = Array3::<i64>::zeros((b, h, w));
        for bi in 0..b {
            for y in 0..h {
                for x in 0..w {
                    tokens[[bi, y, x]] = ((bi + 2 * y + 3 * x) % 6) as i64;
                }
            }
        }
        let centers = grid_agent_centers((h, w), 2, 3);
        let patches = prepare_maze_patches(&tokens, &centers, 3, 6);
        assert_eq!(patches.dim(), (centers.nrows(), b, 3, 3, 6));

        let reassembled = reassemble_logits(&patches, &centers, (h, w));
        assert_eq!(reassembled.dim(), (b, h, w, 6));
        // Every covering patch one-hot encodes the true token at each pixel,
        // so averaging identical one-hots reproduces the one-hot image.
        for bi in 0..b {
            for y in 0..h {
                for x in 0..w {
                    let tok = tokens[[bi, y, x]] as usize;
                    for ch in 0..6 {
                        let want = if ch == tok { 1.0 } else { 0.0 };
                        assert_abs_diff_eq!(
                            reassembled[[bi, y, x, ch]],
                            want,
                            epsilon = 1e-5
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn boundary_patches_see_wall_border() {
        // 3x3 all-empty grid, single agent at (1,1) with a 5x5 patch: the
        // patch's outer ring lies beyond the image and must one-hot to the
        // WALL token (border width 5//2 = 2 covers the whole overhang).
        let tokens = Array3::<i64>::from_elem((1, 3, 3), TOKEN_EMPTY);
        let centers = grid_agent_centers((3, 3), 3, 3); // one center at (1,1)
        assert_eq!(centers.nrows(), 1);
        let patches = prepare_maze_patches(&tokens, &centers, 5, 6);
        for py in 0..5usize {
            for px in 0..5usize {
                let inside = (1..4).contains(&py) && (1..4).contains(&px);
                let expect_tok = if inside { TOKEN_EMPTY } else { TOKEN_WALL } as usize;
                for ch in 0..6 {
                    let want = if ch == expect_tok { 1.0 } else { 0.0 };
                    assert_eq!(
                        patches[[0, 0, py, px, ch]],
                        want,
                        "py={py} px={px} ch={ch}"
                    );
                }
            }
        }
    }

    #[test]
    fn mnist_grid_is_81_agents_8conn() {
        // PLAN §3.5: 28x28, patch_size 3, stride 3 -> 9x9 centers = 81 agents.
        let centers = grid_agent_centers((28, 28), 3, 3);
        assert_eq!(centers.nrows(), 81);
        assert_eq!(centers.row(0).to_vec(), vec![1, 1]);
        assert_eq!(centers.row(8).to_vec(), vec![1, 25]); // last of first row
        assert_eq!(centers.row(80).to_vec(), vec![25, 25]);
        // 8-conn edge count for a 9x9 grid: 2*9*8 (H+V) + 2*8*8 (diagonals) = 272.
        let edges = build_grid_edge_indices(&centers, 3, 8);
        assert_eq!(edges.len(), 9 * 8 * 2 + 8 * 8 * 2);
    }

    #[test]
    fn patchify_stride3_tiles_and_round_trips() {
        // 9x9 image, ps=3, stride=3 -> centers {1,4,7}^2 = 9 agents that tile
        // the image with NO overlap and NO gaps. Scattering each patch back to
        // its covered pixels must reproduce the image exactly.
        let (h, w, b, c) = (9usize, 9, 2, 1);
        let images = Array4::<f32>::from_shape_fn((b, h, w, c), |(bi, y, x, _)| {
            (bi * 100 + y * 9 + x) as f32
        });
        let centers = grid_agent_centers((h, w), 3, 3);
        assert_eq!(centers.nrows(), 9);
        let patches = patchify_batch(&images, &centers, 3);
        assert_eq!(patches.dim(), (9, b, 3, 3, c));

        // Scatter patches back; each pixel covered exactly once here.
        let mut recon = Array4::<f32>::from_elem((b, h, w, c), f32::NAN);
        for (a, row) in centers.rows().into_iter().enumerate() {
            let (cy, cx) = (row[0], row[1]);
            for py in 0..3i64 {
                for px in 0..3i64 {
                    let iy = cy + py - 1;
                    let ix = cx + px - 1;
                    if iy < 0 || ix < 0 || iy >= h as i64 || ix >= w as i64 {
                        continue;
                    }
                    for bi in 0..b {
                        recon[[bi, iy as usize, ix as usize, 0]] =
                            patches[[a, bi, py as usize, px as usize, 0]];
                    }
                }
            }
        }
        for (r, im) in recon.iter().zip(images.iter()) {
            assert_eq!(r, im, "every pixel covered exactly once must round-trip");
        }
    }

    #[test]
    fn patchify_zero_pads_border_agents() {
        // A single agent centered at (0,0) on a 3x3 image: its 3x3 patch's top
        // and left rings fall outside and must be zero.
        let images = Array4::<f32>::from_elem((1, 3, 3, 1), 5.0);
        let centers = Array2::from_shape_vec((1, 2), vec![0i64, 0]).unwrap();
        let patches = patchify_batch(&images, &centers, 3);
        // patch (py,px): image coord (py-1, px-1). Row/col 0 are out of bounds.
        for py in 0..3usize {
            for px in 0..3usize {
                let inside = py >= 1 && px >= 1; // image coords (>=0)
                let want = if inside { 5.0 } else { 0.0 };
                assert_eq!(patches[[0, 0, py, px, 0]], want, "py={py} px={px}");
            }
        }
    }

    #[test]
    fn reassembly_counts_guard_uncovered_pixels() {
        // stride > patch_size leaves gaps; uncovered pixels stay 0 (no NaN).
        let centers = grid_agent_centers((9, 9), 4, 3); // centers at 1, 5
        let n = centers.nrows();
        let patches = Array5::<f32>::from_elem((n, 1, 3, 3, 2), 1.0);
        let out = reassemble_logits(&patches, &centers, (9, 9));
        assert!(out.iter().all(|v| v.is_finite()));
        // Pixel (3,3) is not covered by any 3x3 patch centered on {1,5}².
        assert_eq!(out[[0, 3, 3, 0]], 0.0);
        assert_eq!(out[[0, 1, 1, 0]], 1.0);
    }

    // ---- Sudoku views (PLAN §5.1) ----

    #[test]
    fn sudoku_cell_ids_match_python_dump() {
        let ids = build_sudoku_cell_indices();
        assert_eq!(ids.dim(), (27, 9));
        assert_eq!(ids.row(0).to_vec(), (0..9).collect::<Vec<_>>()); // row 0
        assert_eq!(ids.row(9).to_vec(), (0..9).map(|r| r * 9).collect::<Vec<_>>()); // col 0
        assert_eq!(ids.row(18).to_vec(), vec![0, 1, 2, 9, 10, 11, 18, 19, 20]); // box 0
        assert_eq!(ids.row(26).to_vec(), vec![60, 61, 62, 69, 70, 71, 78, 79, 80]); // box 8
    }

    #[test]
    fn sudoku_each_cell_covered_by_exactly_three_agents() {
        // PLAN §5.1: each of the 81 cells is covered exactly 3x (row/col/box).
        let ids = build_sudoku_cell_indices();
        let mut counts = vec![0usize; 81];
        for agent in 0..27 {
            for local in 0..9 {
                counts[ids[[agent, local]] as usize] += 1;
            }
        }
        assert!(counts.iter().all(|&c| c == 3), "every cell covered exactly 3x");
    }

    #[test]
    fn sudoku_multigraph_has_243_edges_and_matches_const_golden() {
        // PLAN §5.1: 243 edges, 27 agents, all oriented u<v; the generative
        // builder must equal the hardcoded Python golden (cross-check).
        let (edges, map_u, map_v) = build_sudoku_multigraph();
        assert_eq!(edges.len(), SUDOKU_NUM_EDGES);
        assert_eq!(map_u.len(), SUDOKU_NUM_EDGES);
        assert_eq!(map_v.len(), SUDOKU_NUM_EDGES);
        let mut max_node = 0u32;
        for (i, &[u, v]) in edges.iter().enumerate() {
            assert!(u < v, "edge {i} ({u},{v}) not oriented u<v");
            max_node = max_node.max(v);
            assert_eq!(u, SUDOKU_EDGE_U[i], "edge {i} u");
            assert_eq!(v, SUDOKU_EDGE_V[i], "edge {i} v");
            assert_eq!(map_u[i], SUDOKU_MAP_U[i], "edge {i} map_u");
            assert_eq!(map_v[i], SUDOKU_MAP_V[i], "edge {i} map_v");
        }
        assert_eq!(max_node + 1, SUDOKU_NUM_AGENTS as u32, "27 agents");
    }

    #[test]
    fn sudoku_multigraph_covers_each_cell_with_a_3clique() {
        // Reconstruct per-cell coverage from (edges, map_u, map_v): each cell's
        // 3 agents must form all 3 undirected pairs exactly once.
        let (edges, map_u, map_v) = build_sudoku_multigraph();
        let cell_ids = build_sudoku_cell_indices();
        // For each edge, both endpoints must reference the SAME global cell id
        // (map_u/map_v are the local slots holding that shared cell).
        let mut cell_edges = vec![0usize; 81];
        for (i, &[u, v]) in edges.iter().enumerate() {
            let cu = cell_ids[[u as usize, map_u[i] as usize]];
            let cv = cell_ids[[v as usize, map_v[i] as usize]];
            assert_eq!(cu, cv, "edge {i} endpoints must share a cell");
            cell_edges[cu as usize] += 1;
        }
        assert!(cell_edges.iter().all(|&c| c == 3), "3 edges (a 3-clique) per cell");
    }

    #[test]
    fn sudoku_graph_carries_slot_tables() {
        let g = build_sudoku_graph();
        assert_eq!(g.num_nodes, 27);
        assert_eq!(g.num_edges(), 243);
        assert_eq!(g.dir_uv.len(), 243);
        assert_eq!(g.dir_vu.len(), 243);
        assert!(g.node_positions.is_none());
        // dir_uv/dir_vu ARE map_u/map_v (all in 0..9).
        assert!(g.dir_uv.iter().all(|&s| s < 9) && g.dir_vu.iter().all(|&s| s < 9));
        assert_eq!(g.dir_uv, SUDOKU_MAP_U.to_vec());
        assert_eq!(g.dir_vu, SUDOKU_MAP_V.to_vec());
    }

    #[test]
    fn sudoku_slice_reassemble_round_trips() {
        // PLAN §5.1: 27-view slice -> reassemble round-trips a 9x9xC grid,
        // including the box transpose (each cell covered by exactly 3 views, so
        // the mean of 3 identical copies reproduces the input).
        let (b, c) = (2usize, 4usize);
        let grid = Array4::from_shape_fn((b, 9, 9, c), |(bi, r, col, ch)| {
            (bi * 1000 + r * 90 + col * 9 + ch) as f32
        });
        let views = sudoku_slice_batch(&grid);
        assert_eq!(views.dim(), (b, 27, 9, c));
        let recon = reassemble_sudoku_logits(&views);
        assert_eq!(recon.dim(), (b, 9, 9, c));
        for (g, r) in grid.iter().zip(recon.iter()) {
            assert_abs_diff_eq!(g, r, epsilon = 1e-3);
        }
    }

    #[test]
    fn sudoku_slice_box_transpose_is_row_major_within_box() {
        // Box agent 18 (top-left 3x3) must see cells (0,0),(0,1),(0,2),(1,0)...
        // in row-major order at slots 0..9 (the (0,2,1,3) transpose contract).
        let grid = Array4::from_shape_fn((1, 9, 9, 1), |(_, r, col, _)| (r * 9 + col) as f32);
        let views = sudoku_slice_batch(&grid);
        let want = [0.0, 1.0, 2.0, 9.0, 10.0, 11.0, 18.0, 19.0, 20.0];
        for (slot, w) in want.iter().enumerate() {
            assert_eq!(views[[0, 18, slot, 0]], *w, "box0 slot {slot}");
        }
    }

    #[test]
    fn sudoku_predict_and_metrics_on_empty_cells() {
        // 1 batch, N=27 agents. Build logits whose reassembled argmax equals a
        // known grid, then check completion is scored on empty cells only.
        let labels = Array3::from_shape_fn((1, 9, 9), |(_, r, col)| ((r * 9 + col) % 9 + 1) as i64);
        // Inputs: half the cells given (nonzero), half empty (0).
        let inputs = Array3::from_shape_fn((1, 9, 9), |(_, r, col)| {
            if (r + col) % 2 == 0 { labels[[0, r, col]] } else { 0 }
        });
        // Perfect logits: one-hot on the label digit for every covering agent.
        let c = 10usize;
        let cell_ids = build_sudoku_cell_indices();
        let mut logits = Array4::<f32>::zeros((27, 1, 9, c));
        for agent in 0..27 {
            for slot in 0..9 {
                let cid = cell_ids[[agent, slot]] as usize;
                let (r, col) = (cid / 9, cid % 9);
                logits[[agent, 0, slot, labels[[0, r, col]] as usize]] = 5.0;
            }
        }
        let pred = sudoku_predict(&logits);
        assert_eq!(pred, labels, "perfect logits must reconstruct the label grid");
        let m = sudoku_metrics(&pred, &labels, &inputs);
        assert_abs_diff_eq!(m.cell_acc, 1.0, epsilon = 0.0);
        assert_abs_diff_eq!(m.solved, 1.0, epsilon = 0.0);
        assert_abs_diff_eq!(m.completion, 1.0, epsilon = 0.0);

        // Now corrupt one EMPTY cell -> completion < 1 but cell_acc still high.
        let mut pred2 = pred.clone();
        // find an empty cell
        'outer: for r in 0..9 {
            for col in 0..9 {
                if inputs[[0, r, col]] == 0 {
                    pred2[[0, r, col]] = (labels[[0, r, col]] % 9) + 1; // wrong, still 1..9
                    break 'outer;
                }
            }
        }
        let m2 = sudoku_metrics(&pred2, &labels, &inputs);
        assert!(m2.completion < 1.0, "corrupting an empty cell must drop completion");
        assert_eq!(m2.solved, 0.0, "any wrong cell -> not solved");
    }
}
