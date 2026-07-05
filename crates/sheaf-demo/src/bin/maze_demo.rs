//! Maze demo: load exported weights, run the Sheaf-ADMM coordination loop on
//! a 19x19 maze, and animate the per-iteration decoded prediction in the
//! terminal (per-cell argmax classes as colored blocks), with live
//! consistency / primal / dual residuals.
//!
//! Usage:
//!   maze_demo [--weights P] [--config P] [--batch P] [--maze-from-batch]
//!             [--seed N] [--k N] [--fps F] [--no-anim] [--gif OUT.gif]
//!
//! Defaults: goldens/maze/{weights.safetensors, config.json, batch.npz},
//! a fresh generated maze (--seed 0), K=40, 8 fps.
//!
//! --maze-from-batch replays the golden batch input (all rows; row 0 is
//! rendered). --no-anim prints first/middle/last frames plus the residual
//! summary — CI/log friendly, no cursor control. --gif additionally writes
//! the animate-to-K run (the same per-iteration argmax frames the terminal
//! shows, batch row 0) as a looping animated GIF at --fps, last frame held.

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use ndarray::{Array2, Array3, Axis, Ix3};

use sheaf_core::graph::AgentGraph;
use sheaf_io::mazegen::generate_maze;
use sheaf_io::npz::Npz;
use sheaf_io::views::{
    build_grid_edge_indices, grid_agent_centers, prepare_maze_patches, reassemble_logits,
    TOKEN_GOAL, TOKEN_START,
};
use sheaf_io::{load_maze_model, WeightCollection};

struct Args {
    weights: PathBuf,
    config: PathBuf,
    batch: PathBuf,
    maze_from_batch: bool,
    seed: u64,
    k: usize,
    fps: f64,
    no_anim: bool,
    gif: Option<PathBuf>,
}

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../goldens/maze")
}

fn parse_args() -> anyhow::Result<Args> {
    let g = goldens_dir();
    let mut args = Args {
        weights: g.join("weights.safetensors"),
        config: g.join("config.json"),
        batch: g.join("batch.npz"),
        maze_from_batch: false,
        seed: 0,
        k: 40,
        fps: 8.0,
        no_anim: false,
        gif: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = |name: &str| {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))
        };
        match a.as_str() {
            "--weights" => args.weights = PathBuf::from(val("--weights")?),
            "--config" => args.config = PathBuf::from(val("--config")?),
            "--batch" => args.batch = PathBuf::from(val("--batch")?),
            "--maze-from-batch" => args.maze_from_batch = true,
            "--seed" => args.seed = val("--seed")?.parse()?,
            "--k" => args.k = val("--k")?.parse()?,
            "--fps" => args.fps = val("--fps")?.parse()?,
            "--no-anim" => args.no_anim = true,
            "--gif" => args.gif = Some(PathBuf::from(val("--gif")?)),
            "--help" | "-h" => {
                println!(
                    "maze_demo [--weights P] [--config P] [--batch P] [--maze-from-batch] \
                     [--seed N] [--k N] [--fps F] [--no-anim] [--gif OUT.gif]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other:?} (see --help)"),
        }
    }
    anyhow::ensure!(args.k > 0, "--k must be positive");
    anyhow::ensure!(args.fps > 0.0, "--fps must be positive");
    Ok(args)
}

/// Background-color a two-column cell by predicted class token.
/// Palette: walls dark, free light, start green, goal red, PATH bright yellow,
/// pad (class 0) near-black.
fn cell(class: i64, marker: Option<char>) -> String {
    let (bg, fg) = match class {
        1 => (236, 250), // wall: dark gray
        2 => (252, 232), // empty: light gray
        3 => (34, 231),  // start: green
        4 => (160, 231), // goal: red
        5 => (220, 232), // path: bright yellow
        _ => (233, 245), // pad / out-of-range: near-black
    };
    let text = match marker {
        Some(c) => format!("{c} "),
        None => "  ".to_string(),
    };
    format!("\x1b[48;5;{bg}m\x1b[38;5;{fg}m{text}\x1b[0m")
}

/// Render one frame: header status line + the argmax grid (batch row `bi`),
/// with the TRUE start/goal cells lettered S/G for orientation.
fn render_frame(
    k: usize,
    k_total: usize,
    pred: &Array2<i64>,
    tokens: &Array2<i64>,
    cons: f32,
    primal: f32,
    dual: f32,
) -> Vec<String> {
    let (h, w) = pred.dim();
    let mut lines = Vec::with_capacity(h + 1);
    lines.push(format!(
        "iter {:>3}/{k_total}   consistency RMS {cons:>10.6}   primal {primal:>9.6}   dual {dual:>9.6}",
        k + 1
    ));
    for y in 0..h {
        let mut row = String::new();
        for x in 0..w {
            let marker = match tokens[[y, x]] {
                TOKEN_START => Some('S'),
                TOKEN_GOAL => Some('G'),
                _ => None,
            };
            row.push_str(&cell(pred[[y, x]], marker));
        }
        lines.push(row);
    }
    lines
}

/// RMS over all elements.
fn rms(v: ndarray::ArrayView2<f32>) -> f32 {
    let n = v.len() as f32;
    (v.iter().map(|&x| x * x).sum::<f32>() / n).sqrt()
}

/// RGB palette for the GIF frames — same classes as `cell()`:
/// wall #303030, free #d0d0d0, start #00af00, goal #d70000, path #ffd700,
/// pad #121212 (the exact sRGB values of the ANSI-256 codes used above).
fn class_rgb(class: i64) -> [u8; 3] {
    match class {
        1 => [0x30, 0x30, 0x30],
        2 => [0xd0, 0xd0, 0xd0],
        3 => [0x00, 0xaf, 0x00],
        4 => [0xd7, 0x00, 0x00],
        5 => [0xff, 0xd7, 0x00],
        _ => [0x12, 0x12, 0x12],
    }
}

/// Pixels per maze cell in the GIF.
const GIF_CELL: u32 = 12;
/// Height of the iteration progress strip at the bottom.
const GIF_STRIP: u32 = 5;

/// Rasterize one prediction frame: the argmax grid (as in the terminal
/// renderer), the TRUE start/goal cells outlined white, and a bottom
/// progress strip filling with the iteration count.
fn render_gif_frame(
    k: usize,
    k_total: usize,
    pred: &Array2<i64>,
    tokens: &Array2<i64>,
) -> image::RgbaImage {
    let (h, w) = pred.dim();
    let (pw, ph) = (w as u32 * GIF_CELL, h as u32 * GIF_CELL + GIF_STRIP);
    let mut img = image::RgbaImage::new(pw, ph);
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = class_rgb(pred[[y, x]]);
            let outline = matches!(tokens[[y, x]], TOKEN_START | TOKEN_GOAL);
            for py in 0..GIF_CELL {
                for px in 0..GIF_CELL {
                    // 1px white outline marks the true start/goal (the S/G
                    // letters of the terminal renderer).
                    let edge = py == 0 || px == 0 || py == GIF_CELL - 1 || px == GIF_CELL - 1;
                    let rgba = if outline && edge {
                        image::Rgba([0xff, 0xff, 0xff, 0xff])
                    } else {
                        image::Rgba([r, g, b, 0xff])
                    };
                    img.put_pixel(x as u32 * GIF_CELL + px, y as u32 * GIF_CELL + py, rgba);
                }
            }
        }
    }
    // Progress strip: iterations completed, white on near-black.
    let fill = ((k + 1) as f64 / k_total as f64 * pw as f64).round() as u32;
    for py in 0..GIF_STRIP {
        for px in 0..pw {
            let on = px < fill && py > 0;
            let v = if on { 0xe5 } else { 0x12 };
            img.put_pixel(px, h as u32 * GIF_CELL + py, image::Rgba([v, v, v, 0xff]));
        }
    }
    img
}

/// Encode all per-iteration frames as a looping animated GIF at `fps`
/// (the final frame is held ~1.5 s so the converged prediction reads).
fn write_gif(
    path: &std::path::Path,
    preds: &[Array2<i64>],
    tokens: &Array2<i64>,
    fps: f64,
) -> anyhow::Result<()> {
    use image::codecs::gif::{GifEncoder, Repeat};
    let file = std::fs::File::create(path)
        .map_err(|e| anyhow::anyhow!("creating {}: {e}", path.display()))?;
    let mut enc = GifEncoder::new_with_speed(std::io::BufWriter::new(file), 10);
    enc.set_repeat(Repeat::Infinite)?;
    let k_total = preds.len();
    let frame_ms = (1000.0 / fps).round().max(10.0) as u32;
    for (k, pred) in preds.iter().enumerate() {
        let img = render_gif_frame(k, k_total, pred, tokens);
        let ms = if k + 1 == k_total { frame_ms.max(1500) } else { frame_ms };
        let delay = image::Delay::from_numer_denom_ms(ms, 1);
        enc.encode_frame(image::Frame::from_parts(img, 0, 0, delay))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("maze_demo: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let args = parse_args()?;

    let model = load_maze_model(&args.config, &args.weights, WeightCollection::Ema)?;
    let t = model.config.task.clone();

    // Honest labeling: only a config.json carrying `"trained": true` counts as
    // a trained checkpoint. The current exporter writes no flag and its
    // goldens are seed-0 random init (goldens/maze/NOTES.md), so absent ==
    // not-known-trained -> banner.
    if model.config.trained != Some(true) {
        println!("\x1b[1;33m==============================================================\x1b[0m");
        println!("\x1b[1;33m UNTRAINED WEIGHTS — showing ADMM coordination dynamics,\x1b[0m");
        println!("\x1b[1;33m not maze solving\x1b[0m");
        println!("\x1b[1;33m==============================================================\x1b[0m");
    }

    // ---- input maze(s): golden batch or a fresh generated one ----
    let tokens: Array3<i64> = if args.maze_from_batch {
        let mut batch = Npz::open(&args.batch)?;
        batch.i64("tokens")?.into_dimensionality::<Ix3>()?
    } else {
        let maze = generate_maze(19, args.seed, 18);
        println!(
            "generated 19x19 maze (seed {}, shortest path {} steps)",
            args.seed, maze.path_len
        );
        maze.tokens.insert_axis(Axis(0))
    };
    let (b, h, w) = tokens.dim();
    println!(
        "weights: {}   B={b} grid {h}x{w}   K={}",
        args.weights.display(),
        args.k
    );

    // ---- agent graph + patches (MazeTask.prepare) ----
    let centers = grid_agent_centers((h, w), t.stride, t.patch_size);
    let edges = build_grid_edge_indices(&centers, t.stride, t.connectivity);
    let positions = centers.mapv(|v| v as f32);
    let graph = Arc::new(AgentGraph::new_grid(
        edges,
        positions,
        model.config.model.num_directions,
    ));
    let patches = prepare_maze_patches(&tokens, &centers, t.patch_size, t.num_classes);
    println!(
        "agents N={}  edges E={}  ({}-connected, stride {})",
        graph.num_nodes,
        graph.num_edges(),
        t.connectivity,
        t.stride
    );

    // ---- full forward with per-iteration history ----
    let fwd = model.forward(&patches, graph.clone(), args.k);
    let k_total = fwd.history.x.shape()[0];

    // Per-iteration derived frames (batch row 0 rendered) + metrics.
    let tokens0 = tokens.index_axis(Axis(0), 0).to_owned();
    let mut preds: Vec<Array2<i64>> = Vec::with_capacity(k_total);
    let mut metrics: Vec<(f32, f32, f32)> = Vec::with_capacity(k_total); // (cons, primal, dual)
    for k in 0..k_total {
        let logits_k = fwd.logits_per_iter.index_axis(Axis(0), k).to_owned();
        let grid_logits = reassemble_logits(&logits_k, &centers, (h, w)); // [B, H, W, C]
        let mut pred = Array2::<i64>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let (mut arg, mut best) = (0usize, f32::NEG_INFINITY);
                for c in 0..t.num_classes {
                    let v = grid_logits[[0, y, x, c]];
                    if v > best {
                        best = v;
                        arg = c;
                    }
                }
                pred[[y, x]] = arg as i64;
            }
        }
        preds.push(pred);
        let cons = fwd.history.consistency_rms.index_axis(Axis(0), k);
        let cons = cons.iter().sum::<f32>() / cons.len() as f32; // mean over B
        let primal = rms(fwd.history.primal_res.index_axis(Axis(0), k));
        let dual = rms(fwd.history.dual_res.index_axis(Axis(0), k));
        anyhow::ensure!(
            cons.is_finite() && primal.is_finite() && dual.is_finite(),
            "non-finite residuals at iteration {k} (cons={cons}, primal={primal}, dual={dual})"
        );
        metrics.push((cons, primal, dual));
    }

    // ---- optional GIF export (image + gif crates; PLAN.md §6) ----
    if let Some(path) = &args.gif {
        write_gif(path, &preds, &tokens0, args.fps)?;
        let bytes = std::fs::metadata(path)?.len();
        println!(
            "wrote {} ({} frames, {}x{} px, {bytes} bytes)",
            path.display(),
            preds.len(),
            w as u32 * GIF_CELL,
            h as u32 * GIF_CELL + GIF_STRIP,
        );
    }

    println!(
        "legend: {}wall {}free {}start {}goal {}PATH   S/G = true start/goal\n",
        cell(1, None),
        cell(2, None),
        cell(3, None),
        cell(4, None),
        cell(5, None),
    );

    if args.no_anim {
        // First / middle / last frames only.
        let picks = [0, (k_total - 1) / 2, k_total - 1];
        let mut shown = std::collections::BTreeSet::new();
        for &k in &picks {
            if !shown.insert(k) {
                continue;
            }
            let (cons, primal, dual) = metrics[k];
            for line in render_frame(k, k_total, &preds[k], &tokens0, cons, primal, dual) {
                println!("{line}");
            }
            println!();
        }
    } else {
        // ANSI animation: draw a frame, sleep, rewind the cursor, redraw.
        let delay = Duration::from_secs_f64(1.0 / args.fps);
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        write!(out, "\x1b[?25l")?; // hide cursor
        for k in 0..k_total {
            let (cons, primal, dual) = metrics[k];
            let lines = render_frame(k, k_total, &preds[k], &tokens0, cons, primal, dual);
            let frame_height = lines.len();
            for line in &lines {
                writeln!(out, "\x1b[2K{line}")?;
            }
            out.flush()?;
            if k + 1 < k_total {
                std::thread::sleep(delay);
                write!(out, "\x1b[{frame_height}A")?;
            }
        }
        write!(out, "\x1b[?25h")?; // show cursor
        writeln!(out)?;
    }

    // ---- residual summary every 5 iterations (plus the last) ----
    println!("residuals (batch mean; primal/dual are RMS over agents):");
    println!("{:>6}  {:>15}  {:>12}  {:>12}", "iter", "consistency_rms", "primal_rms", "dual_rms");
    let rows: std::collections::BTreeSet<usize> =
        (0..k_total).step_by(5).chain([k_total - 1]).collect();
    for k in rows {
        let (cons, primal, dual) = metrics[k];
        println!("{:>6}  {:>15.6}  {:>12.6}  {:>12.6}", k, cons, primal, dual);
    }
    // Consistency rises for a few iterations off the warm z_init = h seed
    // (matches the Python trace). With random-init weights it then decreases
    // as ADMM drives agreement; the trained maze model instead settles into a
    // near-flat plateau — prox-mode consensus is soft (gamma-weighted), so the
    // converged solution holds a steady nonzero edge disagreement while the
    // primal/dual residuals keep falling.
    let (peak_k, &(c_peak, ..)) = metrics
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
        .expect("non-empty history");
    let (c_last, ..) = metrics[k_total - 1];
    let tail_min = metrics[k_total / 2..]
        .iter()
        .map(|m| m.0)
        .fold(f32::INFINITY, f32::min);
    println!(
        "\nconsistency RMS: peak {c_peak:.6} (iter {peak_k}) -> {c_last:.6} (iter {}): {}",
        k_total - 1,
        if c_last < c_peak || k_total == 1 {
            "DECREASING — ADMM drives the agents toward agreement"
        } else if c_last <= tail_min * 1.05 {
            "PLATEAU — soft (gamma-weighted) consensus holds steady disagreement"
        } else {
            "did NOT decrease (unexpected)"
        }
    );
    Ok(())
}
