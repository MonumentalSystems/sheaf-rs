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
}
