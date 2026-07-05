//! Maze demo: load exported weights, run the Sheaf-ADMM coordination loop on
//! a 19x19 maze, and animate the per-iteration decoded prediction in the
//! terminal (per-cell argmax classes as colored blocks), with live
//! consistency / primal / dual residuals.
//!
//! Usage:
//!   maze_demo [--weights P] [--config P] [--batch P] [--maze-from-batch]
//!             [--seed N] [--k N] [--fps F] [--no-anim]
//!
//! Defaults: goldens/maze/{weights.safetensors, config.json, batch.npz},
//! a fresh generated maze (--seed 0), K=40, 8 fps.
//!
//! --maze-from-batch replays the golden batch input (all rows; row 0 is
//! rendered). --no-anim prints first/middle/last frames plus the residual
//! summary — CI/log friendly, no cursor control.

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
            "--help" | "-h" => {
                println!(
                    "maze_demo [--weights P] [--config P] [--batch P] [--maze-from-batch] \
                     [--seed N] [--k N] [--fps F] [--no-anim]"
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
    // (matches the Python trace), then must decrease as ADMM drives agreement.
    let (peak_k, &(c_peak, ..)) = metrics
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
        .expect("non-empty history");
    let (c_last, ..) = metrics[k_total - 1];
    println!(
        "\nconsistency RMS: peak {c_peak:.6} (iter {peak_k}) -> {c_last:.6} (iter {}): {}",
        k_total - 1,
        if c_last < c_peak || k_total == 1 {
            "DECREASING — ADMM drives the agents toward agreement"
        } else {
            "did NOT decrease (unexpected)"
        }
    );
    Ok(())
}
