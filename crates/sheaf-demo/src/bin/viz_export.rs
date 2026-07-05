//! viz_export: static paper-figure exporters fed from an `AdmmHistory` dump.
//!
//! Renders the two Python-viz figures the live demos do NOT cover (PLAN.md §1
//! viz note, §6):
//!   * `coordination_dynamics.svg` — per-agent `[N, K]` PRIMAL and DUAL
//!     residual heatmaps on a log color scale (floored at the smallest
//!     positive entry), plus the mean/max residual + consistency-RMS
//!     convergence curves on a log y-axis. Replaces the Python
//!     `coordination_dynamics.pdf`.
//!   * `xz_trajectories.svg` — per-agent 2-D phase portrait of the local
//!     proposal `x_i^k` (blue) vs the consensus iterate `z_i^k` (red),
//!     projected with a single PCA basis fit jointly over ALL agents' x and z
//!     across all iterations (faer SVD), so every panel shares one coordinate
//!     frame. Replaces the Python `xz_trajectories.pdf`.
//!
//! Input is a .npz `AdmmHistory` dump (goldens/maze/trace.npz is one — the
//! golden trace is a superset of the history). Expected keys:
//!   x           f32 [K, N, B, d_v]   (required)
//!   z           f32 [K, N, B, d_v]   (required)
//!   primal_res  f32 [K, N, B]        (required)
//!   dual_res    f32 [K, N, B]        (required)
//!   consistency f32 [K, B]           (optional; adds the green curve)
//!   centers     i64 [N, 2]           (optional; grid panel layout, else a
//!                                     near-square index grid is used)
//! One batch row (`--batch`, default 0) is rendered, matching the Python
//! `Trajectory` (one example per figure).
//!
//! Usage:
//!   viz_export --history goldens/maze/trace.npz [--out-dir D] [--batch I]
//!              [--pdf-style paper|bare]
//!
//! `--pdf-style paper` (default) adds the figure suptitles; `bare` omits them.
//! Output is SVG (self-contained, no font rasterization needed).

use std::path::PathBuf;
use std::process::ExitCode;

use ndarray::{s, Array1, Array2, Array3, Axis, Ix2, Ix3, Ix4};

use sheaf_io::npz::Npz;

struct Args {
    history: PathBuf,
    out_dir: PathBuf,
    batch: usize,
    pdf_style: String,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut history: Option<PathBuf> = None;
    let mut args = Args {
        history: PathBuf::new(),
        out_dir: PathBuf::from("."),
        batch: 0,
        pdf_style: "paper".to_string(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = |name: &str| {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))
        };
        match a.as_str() {
            "--history" => history = Some(PathBuf::from(val("--history")?)),
            "--out-dir" => args.out_dir = PathBuf::from(val("--out-dir")?),
            "--batch" => args.batch = val("--batch")?.parse()?,
            "--pdf-style" => args.pdf_style = val("--pdf-style")?,
            "--help" | "-h" => {
                println!(
                    "viz_export --history H.npz [--out-dir D] [--batch I] [--pdf-style paper|bare]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other:?} (see --help)"),
        }
    }
    args.history = history.ok_or_else(|| anyhow::anyhow!("--history is required (see --help)"))?;
    anyhow::ensure!(
        matches!(args.pdf_style.as_str(), "paper" | "bare"),
        "--pdf-style must be `paper` or `bare`, got {:?}",
        args.pdf_style
    );
    Ok(args)
}

// ---------------------------------------------------------------------------
// Colormaps (matplotlib viridis / magma, 9 linearly-interpolated anchors).
// ---------------------------------------------------------------------------

const VIRIDIS: [(u8, u8, u8); 9] = [
    (68, 1, 84),
    (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (53, 183, 121),
    (110, 206, 88),
    (253, 231, 37),
];

const MAGMA: [(u8, u8, u8); 9] = [
    (0, 0, 4),
    (20, 14, 54),
    (59, 15, 112),
    (100, 26, 128),
    (140, 41, 129),
    (183, 55, 121),
    (222, 73, 104),
    (247, 112, 92),
    (252, 253, 191),
];

/// Sample a 9-anchor colormap at `t` in [0, 1] (clamped, linear interpolation).
fn colormap(anchors: &[(u8, u8, u8); 9], t: f32) -> String {
    let t = t.clamp(0.0, 1.0) * 8.0;
    let i = (t.floor() as usize).min(7);
    let f = t - i as f32;
    let (r0, g0, b0) = anchors[i];
    let (r1, g1, b1) = anchors[i + 1];
    let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
    format!("#{:02x}{:02x}{:02x}", lerp(r0, r1), lerp(g0, g1), lerp(b0, b1))
}

/// Log color/axis normalization: vmin = smallest positive entry, vmax = max
/// (matches the Python `LogNorm` floor — zeros stay finite at t = 0).
struct LogNorm {
    ln_min: f32,
    ln_max: f32,
    vmin: f32,
    vmax: f32,
}

impl LogNorm {
    fn from_values<'a>(values: impl Iterator<Item = &'a f32>) -> Self {
        let mut vmin = f32::INFINITY;
        let mut vmax = f32::NEG_INFINITY;
        for &v in values {
            if v > 0.0 && v < vmin {
                vmin = v;
            }
            if v > vmax {
                vmax = v;
            }
        }
        if !vmin.is_finite() {
            vmin = 1e-12;
        }
        let vmax = vmax.max(vmin * 10.0);
        LogNorm { ln_min: vmin.ln(), ln_max: vmax.ln(), vmin, vmax }
    }

    /// Map `v` to [0, 1] on the log scale (values <= vmin clamp to 0).
    fn t(&self, v: f32) -> f32 {
        let v = v.max(self.vmin);
        ((v.ln() - self.ln_min) / (self.ln_max - self.ln_min)).clamp(0.0, 1.0)
    }

    /// Decade tick values 10^p with vmin <= 10^p <= vmax, as (t, exponent).
    fn decade_ticks(&self) -> Vec<(f32, i32)> {
        let lo = (self.vmin.log10()).ceil() as i32;
        let hi = (self.vmax.log10()).floor() as i32;
        (lo..=hi)
            .map(|p| {
                let v = 10f32.powi(p);
                (self.t(v), p)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Minimal SVG canvas.
// ---------------------------------------------------------------------------

struct Svg {
    body: String,
    w: f64,
    h: f64,
}

fn f(v: f64) -> String {
    // Compact fixed-point coordinates keep the file size sane.
    format!("{v:.2}")
}

impl Svg {
    fn new(w: f64, h: f64) -> Self {
        let mut body = String::new();
        body.push_str(&format!(
            "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"white\"/>\n",
            f(w),
            f(h)
        ));
        Svg { body, w, h }
    }

    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str) {
        self.body.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{fill}\"/>\n",
            f(x),
            f(y),
            f(w),
            f(h)
        ));
    }

    fn frame(&mut self, x: f64, y: f64, w: f64, h: f64, stroke: &str) {
        self.body.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"0.8\"/>\n",
            f(x),
            f(y),
            f(w),
            f(h)
        ));
    }

    #[allow(clippy::too_many_arguments)] // SVG line primitive: geometry + style params
    fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, width: f64, dash: &str) {
        let dash_attr = if dash.is_empty() {
            String::new()
        } else {
            format!(" stroke-dasharray=\"{dash}\"")
        };
        self.body.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"{width}\"{dash_attr}/>\n",
            f(x1),
            f(y1),
            f(x2),
            f(y2)
        ));
    }

    fn polyline(&mut self, pts: &[(f64, f64)], stroke: &str, width: f64, dash: &str, opacity: f64) {
        if pts.is_empty() {
            return;
        }
        let mut s = String::with_capacity(pts.len() * 12);
        for (i, (x, y)) in pts.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("{},{}", f(*x), f(*y)));
        }
        let dash_attr = if dash.is_empty() {
            String::new()
        } else {
            format!(" stroke-dasharray=\"{dash}\"")
        };
        self.body.push_str(&format!(
            "<polyline points=\"{s}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"{width}\" stroke-opacity=\"{opacity}\"{dash_attr}/>\n"
        ));
    }

    fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: &str, stroke: &str) {
        let stroke_attr = if stroke.is_empty() {
            String::new()
        } else {
            format!(" stroke=\"{stroke}\" stroke-width=\"1\"")
        };
        self.body.push_str(&format!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{fill}\"{stroke_attr}/>\n",
            f(cx),
            f(cy),
            f(r)
        ));
    }

    /// `anchor`: start | middle | end. `rotate`: degrees around (x, y), 0 = none.
    #[allow(clippy::too_many_arguments)] // SVG text primitive: placement + style params
    fn text(&mut self, x: f64, y: f64, size: f64, anchor: &str, fill: &str, rotate: f64, s: &str) {
        let esc = s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
        let transform = if rotate != 0.0 {
            format!(" transform=\"rotate({rotate} {} {})\"", f(x), f(y))
        } else {
            String::new()
        };
        self.body.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"{size}\" text-anchor=\"{anchor}\" fill=\"{fill}\"{transform}>{esc}</text>\n",
            f(x),
            f(y)
        ));
    }

    fn finish(self) -> String {
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n{}</svg>\n",
            f(self.w),
            f(self.h),
            f(self.w),
            f(self.h),
            self.body
        )
    }
}

// ---------------------------------------------------------------------------
// Joint PCA over faer's SVD (PLAN.md §1: "~10 lines over faer's SVD").
// ---------------------------------------------------------------------------

/// Fit a 2-component PCA on `[M, d]` points: mean `[d]` + components `[d, 2]`.
/// The principal axes are the top right-singular vectors of the centered data
/// = the top singular vectors of its `[d, d]` Gram matrix (symmetric PSD, so
/// faer's SVD returns them sorted by explained variance).
fn pca_basis(points: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    let (m, d) = points.dim();
    assert!(d >= 2 && m >= 2, "PCA needs at least 2 points in >= 2 dims");
    let mean = points.mean_axis(Axis(0)).expect("non-empty points");
    // Gram of the centered data, accumulated in f64 for stability.
    let mut gram = vec![0f64; d * d];
    for row in points.rows() {
        let c: Vec<f64> = row.iter().zip(mean.iter()).map(|(&v, &mu)| (v - mu) as f64).collect();
        for i in 0..d {
            for j in 0..d {
                gram[i * d + j] += c[i] * c[j];
            }
        }
    }
    let g = faer::Mat::<f64>::from_fn(d, d, |i, j| gram[i * d + j]);
    let svd = g.svd().expect("SVD of a finite Gram matrix converges");
    let u = svd.U();
    let comps = Array2::from_shape_fn((d, 2), |(i, c)| u[(i, c)] as f32);
    (mean, comps)
}

/// Project x/z `[K, N, d]` to 2-D `[K, N, 2]` with one shared basis
/// (d == 2 passes through; d == 1 pads a zero column; else joint PCA).
fn project_xz(x: &Array3<f32>, z: &Array3<f32>) -> (Array3<f32>, Array3<f32>) {
    let (k, n, d) = x.dim();
    match d {
        2 => (x.clone(), z.clone()),
        1 => {
            let pad = |a: &Array3<f32>| {
                Array3::from_shape_fn((k, n, 2), |(ki, ni, c)| if c == 0 { a[[ki, ni, 0]] } else { 0.0 })
            };
            (pad(x), pad(z))
        }
        _ => {
            let mut stacked = Array2::zeros((2 * k * n, d));
            stacked
                .slice_mut(s![..k * n, ..])
                .assign(&x.to_shape((k * n, d)).expect("contiguous"));
            stacked
                .slice_mut(s![(k * n).., ..])
                .assign(&z.to_shape((k * n, d)).expect("contiguous"));
            let (mean, comps) = pca_basis(&stacked);
            let proj = |a: &Array3<f32>| {
                Array3::from_shape_fn((k, n, 2), |(ki, ni, c)| {
                    (0..d).map(|di| (a[[ki, ni, di]] - mean[di]) * comps[[di, c]]).sum()
                })
            };
            (proj(x), proj(z))
        }
    }
}

// ---------------------------------------------------------------------------
// Figure 1: coordination dynamics (heatmaps + convergence curves).
// ---------------------------------------------------------------------------

/// One `[N, K]` log-scale heatmap panel (agents on y, origin at the bottom,
/// matching the Python `imshow(origin="lower")`), with a decade colorbar.
#[allow(clippy::too_many_arguments)]
fn draw_heatmap(
    svg: &mut Svg,
    x0: f64,
    y0: f64,
    pw: f64,
    ph: f64,
    data: &Array2<f32>, // [K, N]
    title: &str,
    cmap: &[(u8, u8, u8); 9],
) {
    let (k, n) = data.dim();
    let norm = LogNorm::from_values(data.iter());
    let cw = pw / k as f64;
    let ch = ph / n as f64;
    for ki in 0..k {
        for ni in 0..n {
            let color = colormap(cmap, norm.t(data[[ki, ni]]));
            let cx = x0 + ki as f64 * cw;
            let cy = y0 + ph - (ni + 1) as f64 * ch;
            // +0.5px overdraw hides antialiasing seams between cells.
            svg.rect(cx, cy, cw + 0.5, ch + 0.5, &color);
        }
    }
    svg.frame(x0, y0, pw, ph, "#374151");
    svg.text(x0 + pw / 2.0, y0 - 8.0, 12.0, "middle", "#111827", 0.0, title);
    svg.text(x0 + pw / 2.0, y0 + ph + 28.0, 11.0, "middle", "#111827", 0.0, "ADMM iteration k");
    svg.text(x0 - 30.0, y0 + ph / 2.0, 11.0, "middle", "#111827", -90.0, "agent i");
    // x ticks (iteration, 1-based like the Python figure).
    let step = (k as f64 / 5.0).ceil().max(1.0) as usize;
    for ki in (0..k).step_by(step) {
        let tx = x0 + (ki as f64 + 0.5) * cw;
        svg.line(tx, y0 + ph, tx, y0 + ph + 3.0, "#374151", 0.8, "");
        svg.text(tx, y0 + ph + 14.0, 9.0, "middle", "#374151", 0.0, &format!("{}", ki + 1));
    }
    // y ticks (agent index).
    let ystep = (n as f64 / 5.0).ceil().max(1.0) as usize;
    for ni in (0..n).step_by(ystep) {
        let ty = y0 + ph - (ni as f64 + 0.5) * ch;
        svg.line(x0 - 3.0, ty, x0, ty, "#374151", 0.8, "");
        svg.text(x0 - 6.0, ty + 3.0, 9.0, "end", "#374151", 0.0, &format!("{ni}"));
    }
    // Colorbar: 64-step vertical gradient + decade tick labels.
    let bx = x0 + pw + 8.0;
    let bw = 12.0;
    let steps = 64;
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        let cy = y0 + ph - (i + 1) as f64 * ph / steps as f64;
        svg.rect(bx, cy, bw, ph / steps as f64 + 0.5, &colormap(cmap, t));
    }
    svg.frame(bx, y0, bw, ph, "#374151");
    for (t, p) in norm.decade_ticks() {
        let ty = y0 + ph - t as f64 * ph;
        svg.line(bx + bw, ty, bx + bw + 3.0, ty, "#374151", 0.8, "");
        svg.text(bx + bw + 5.0, ty + 3.0, 8.5, "start", "#374151", 0.0, &format!("1e{p}"));
    }
}

/// Convergence panel: mean/max primal and dual residual over agents, plus the
/// consistency RMS (when present), on a log y-axis.
#[allow(clippy::too_many_arguments)]
fn draw_convergence(
    svg: &mut Svg,
    x0: f64,
    y0: f64,
    pw: f64,
    ph: f64,
    primal: &Array2<f32>, // [K, N]
    dual: &Array2<f32>,   // [K, N]
    cons: Option<&Array1<f32>>, // [K]
) {
    let k = primal.shape()[0];
    let mean_max = |a: &Array2<f32>| {
        let mean: Vec<f32> = a.rows().into_iter().map(|r| r.sum() / r.len() as f32).collect();
        let max: Vec<f32> = a
            .rows()
            .into_iter()
            .map(|r| r.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)))
            .collect();
        (mean, max)
    };
    let (p_mean, p_max) = mean_max(primal);
    let (d_mean, d_max) = mean_max(dual);
    let mut series: Vec<(&str, &str, &str, Vec<f32>)> = vec![
        ("#1d4ed8", "", "primal (mean)", p_mean),
        ("#1d4ed8", "4 3", "primal (max)", p_max),
        ("#dc2626", "", "dual (mean)", d_mean),
        ("#dc2626", "4 3", "dual (max)", d_max),
    ];
    if let Some(c) = cons {
        series.push(("#16a34a", "", "consistency RMS", c.to_vec()));
    }
    let norm = LogNorm::from_values(series.iter().flat_map(|(.., v)| v.iter()));
    let sx = |ki: usize| x0 + (ki as f64 + 0.5) / k as f64 * pw;
    let sy = |v: f32| y0 + ph - norm.t(v) as f64 * ph;
    // Grid + y decade ticks.
    for (t, p) in norm.decade_ticks() {
        let ty = y0 + ph - t as f64 * ph;
        svg.line(x0, ty, x0 + pw, ty, "#e5e7eb", 0.7, "");
        svg.text(x0 - 5.0, ty + 3.0, 8.5, "end", "#374151", 0.0, &format!("1e{p}"));
    }
    for (color, dash, _label, values) in &series {
        let pts: Vec<(f64, f64)> = values.iter().enumerate().map(|(ki, &v)| (sx(ki), sy(v))).collect();
        let opacity = if dash.is_empty() { 1.0 } else { 0.6 };
        svg.polyline(&pts, color, 1.4, dash, opacity);
    }
    svg.frame(x0, y0, pw, ph, "#374151");
    svg.text(x0 + pw / 2.0, y0 - 8.0, 12.0, "middle", "#111827", 0.0, "convergence");
    svg.text(x0 + pw / 2.0, y0 + ph + 28.0, 11.0, "middle", "#111827", 0.0, "ADMM iteration k");
    svg.text(x0 - 42.0, y0 + ph / 2.0, 11.0, "middle", "#111827", -90.0, "residual (log)");
    let step = (k as f64 / 5.0).ceil().max(1.0) as usize;
    for ki in (0..k).step_by(step) {
        let tx = sx(ki);
        svg.line(tx, y0 + ph, tx, y0 + ph + 3.0, "#374151", 0.8, "");
        svg.text(tx, y0 + ph + 14.0, 9.0, "middle", "#374151", 0.0, &format!("{}", ki + 1));
    }
    // Legend (top right, inside the panel).
    let lx = x0 + pw - 118.0;
    let mut ly = y0 + 12.0;
    for (color, dash, label, _values) in &series {
        svg.line(lx, ly - 3.0, lx + 18.0, ly - 3.0, color, 1.6, dash);
        svg.text(lx + 22.0, ly, 8.5, "start", "#111827", 0.0, label);
        ly += 11.0;
    }
}

fn render_coordination(
    primal: &Array2<f32>, // [K, N]
    dual: &Array2<f32>,   // [K, N]
    cons: Option<&Array1<f32>>,
    suptitle: Option<&str>,
) -> String {
    let (pw, ph) = (330.0, 280.0);
    let (ml, mt, mb) = (58.0, if suptitle.is_some() { 52.0 } else { 32.0 }, 44.0);
    let panel_stride = ml + pw + 52.0; // colorbar + labels
    let w = 3.0 * panel_stride;
    let h = mt + ph + mb;
    let mut svg = Svg::new(w, h);
    if let Some(t) = suptitle {
        svg.text(w / 2.0, 22.0, 14.0, "middle", "#111827", 0.0, t);
    }
    draw_heatmap(&mut svg, ml, mt, pw, ph, primal, "primal residual  ||x_i - z_i||", &VIRIDIS);
    draw_heatmap(
        &mut svg,
        panel_stride + ml,
        mt,
        pw,
        ph,
        dual,
        "dual residual  rho ||z_i - z_prev_i||",
        &MAGMA,
    );
    draw_convergence(&mut svg, 2.0 * panel_stride + ml, mt, pw, ph, primal, dual, cons);
    svg.finish()
}

// ---------------------------------------------------------------------------
// Figure 2: per-agent x/z phase portraits (joint PCA frame).
// ---------------------------------------------------------------------------

/// Panel slots: `centers` [N, 2] (y, x) lays panels out on the agent grid
/// (the Python grid layout); otherwise a near-square index grid.
fn grid_layout(n: usize, centers: Option<&Array2<i64>>) -> (usize, usize, Vec<(usize, usize)>) {
    if let Some(c) = centers {
        let mut ys: Vec<i64> = c.column(0).iter().copied().collect();
        let mut xs: Vec<i64> = c.column(1).iter().copied().collect();
        ys.sort_unstable();
        ys.dedup();
        xs.sort_unstable();
        xs.dedup();
        let row_of = |v: i64| ys.binary_search(&v).expect("center y in table");
        let col_of = |v: i64| xs.binary_search(&v).expect("center x in table");
        let slots = (0..n).map(|a| (row_of(c[[a, 0]]), col_of(c[[a, 1]]))).collect();
        return (ys.len(), xs.len(), slots);
    }
    let ncols = (n as f64).sqrt().ceil() as usize;
    let nrows = n.div_ceil(ncols);
    (nrows, ncols, (0..n).map(|a| (a / ncols, a % ncols)).collect())
}

fn render_xz(
    x2: &Array3<f32>, // [K, N, 2]
    z2: &Array3<f32>, // [K, N, 2]
    centers: Option<&Array2<i64>>,
    suptitle: Option<&str>,
) -> String {
    let (k, n, _) = x2.dim();
    let (nrows, ncols, slots) = grid_layout(n, centers);
    let (pw, ph) = (62.0, 62.0);
    let (gx, gy) = (6.0, 16.0); // gy leaves room for the per-panel title
    let ml = 14.0;
    let mt = if suptitle.is_some() { 40.0 } else { 20.0 };
    let mb = 30.0; // legend row
    let w = 2.0 * ml + ncols as f64 * pw + (ncols - 1) as f64 * gx;
    let h = mt + nrows as f64 * (ph + gy) - gy + mb;
    let mut svg = Svg::new(w, h);
    if let Some(t) = suptitle {
        svg.text(w / 2.0, 22.0, 14.0, "middle", "#111827", 0.0, t);
    }

    // Shared axis limits across panels (5% pad), so trajectories compare.
    let mut lo = [f32::INFINITY; 2];
    let mut hi = [f32::NEG_INFINITY; 2];
    for a in [x2, z2] {
        for p in a.rows() {
            for c in 0..2 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
    }
    for c in 0..2 {
        let pad = 0.05 * (hi[c] - lo[c]).max(1e-6);
        lo[c] -= pad;
        hi[c] += pad;
    }

    let draw_markers = k <= 40; // per-vertex dots get muddy on long unrolls
    for (a, &(r, col)) in slots.iter().enumerate() {
        let px = ml + col as f64 * (pw + gx);
        let py = mt + r as f64 * (ph + gy);
        svg.frame(px, py, pw, ph, "#d1d5db");
        svg.text(px + pw / 2.0, py - 3.0, 6.5, "middle", "#6b7280", 0.0, &format!("agent {a}"));
        let to_px = |v: f32, c: usize| (v - lo[c]) / (hi[c] - lo[c]);
        let map = |ki: usize, arr: &Array3<f32>| {
            (
                px + to_px(arr[[ki, a, 0]], 0) as f64 * pw,
                py + ph - to_px(arr[[ki, a, 1]], 1) as f64 * ph,
            )
        };
        for (arr, color) in [(x2, "#1d4ed8"), (z2, "#dc2626")] {
            let pts: Vec<(f64, f64)> = (0..k).map(|ki| map(ki, arr)).collect();
            svg.polyline(&pts, color, 0.9, "", 1.0);
            if draw_markers {
                for &(cx, cy) in &pts {
                    svg.circle(cx, cy, 1.1, color, "");
                }
            }
        }
        // Consensus path start (open) and end (filled), like the paper figure.
        let (sx0, sy0) = map(0, z2);
        let (sx1, sy1) = map(k - 1, z2);
        svg.circle(sx0, sy0, 2.6, "none", "#dc2626");
        svg.circle(sx1, sy1, 2.2, "#111827", "");
    }

    // Legend, bottom center.
    let ly = h - 12.0;
    let lx = w / 2.0 - 90.0;
    svg.line(lx, ly - 3.0, lx + 18.0, ly - 3.0, "#1d4ed8", 1.6, "");
    svg.circle(lx + 9.0, ly - 3.0, 1.6, "#1d4ed8", "");
    svg.text(lx + 22.0, ly, 9.0, "start", "#111827", 0.0, "x (local)");
    svg.line(lx + 90.0, ly - 3.0, lx + 108.0, ly - 3.0, "#dc2626", 1.6, "");
    svg.circle(lx + 99.0, ly - 3.0, 1.6, "#dc2626", "");
    svg.text(lx + 112.0, ly, 9.0, "start", "#111827", 0.0, "z (consensus)");
    svg.finish()
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("viz_export: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let args = parse_args()?;
    let mut npz = Npz::open(&args.history)?;
    let names = npz.names()?;

    let x = npz.f32("x")?.into_dimensionality::<Ix4>()?;
    let z = npz.f32("z")?.into_dimensionality::<Ix4>()?;
    let primal = npz.f32("primal_res")?.into_dimensionality::<Ix3>()?;
    let dual = npz.f32("dual_res")?.into_dimensionality::<Ix3>()?;
    let (k, n, b, d_v) = x.dim();
    anyhow::ensure!(z.dim() == x.dim(), "z shape {:?} != x shape {:?}", z.dim(), x.dim());
    anyhow::ensure!(
        primal.dim() == (k, n, b) && dual.dim() == (k, n, b),
        "primal/dual shapes {:?}/{:?} do not match x [K,N,B,*] = [{k},{n},{b},{d_v}]",
        primal.dim(),
        dual.dim()
    );
    anyhow::ensure!(args.batch < b, "--batch {} out of range (B = {b})", args.batch);
    let bi = args.batch;
    println!(
        "viz_export: {}  K={k} N={n} B={b} d_v={d_v}  (batch row {bi})",
        args.history.display()
    );

    let cons: Option<Array1<f32>> = if names.iter().any(|s| s == "consistency") {
        let c = npz.f32("consistency")?.into_dimensionality::<Ix2>()?;
        anyhow::ensure!(c.dim() == (k, b), "consistency shape {:?}, want [{k},{b}]", c.dim());
        Some(c.column(bi).to_owned())
    } else {
        None
    };
    let centers: Option<Array2<i64>> = if names.iter().any(|s| s == "centers") {
        let c = npz.i64("centers")?.into_dimensionality::<Ix2>()?;
        anyhow::ensure!(c.dim() == (n, 2), "centers shape {:?}, want [{n},2]", c.dim());
        Some(c)
    } else {
        None
    };

    // Batch-select down to the Python Trajectory shapes.
    let primal_kn = primal.slice(s![.., .., bi]).to_owned(); // [K, N]
    let dual_kn = dual.slice(s![.., .., bi]).to_owned(); // [K, N]
    let x_knd = x.slice(s![.., .., bi, ..]).to_owned(); // [K, N, d_v]
    let z_knd = z.slice(s![.., .., bi, ..]).to_owned();

    let paper = args.pdf_style == "paper";
    std::fs::create_dir_all(&args.out_dir)?;

    let coord_title = format!("Coordination dynamics (N={n} agents, K={k} iterations)");
    let coord = render_coordination(
        &primal_kn,
        &dual_kn,
        cons.as_ref(),
        paper.then_some(coord_title.as_str()),
    );
    let coord_path = args.out_dir.join("coordination_dynamics.svg");
    std::fs::write(&coord_path, &coord)?;
    println!("wrote {} ({} bytes)", coord_path.display(), coord.len());

    let (x2, z2) = project_xz(&x_knd, &z_knd);
    let xz_title = format!("x/z phase portraits, joint 2-D PCA frame (d_v={d_v})");
    let xz = render_xz(&x2, &z2, centers.as_ref(), paper.then_some(xz_title.as_str()));
    let xz_path = args.out_dir.join("xz_trajectories.svg");
    std::fs::write(&xz_path, &xz)?;
    println!("wrote {} ({} bytes)", xz_path.display(), xz.len());
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn colormap_endpoints_and_interior() {
        assert_eq!(colormap(&VIRIDIS, 0.0), "#440154");
        assert_eq!(colormap(&VIRIDIS, 1.0), "#fde725");
        assert_eq!(colormap(&MAGMA, 0.0), "#000004");
        assert_eq!(colormap(&MAGMA, 1.0), "#fcfdbf");
        // Out-of-range clamps rather than panicking.
        assert_eq!(colormap(&VIRIDIS, -1.0), colormap(&VIRIDIS, 0.0));
        assert_eq!(colormap(&VIRIDIS, 2.0), colormap(&VIRIDIS, 1.0));
    }

    #[test]
    fn log_norm_floors_zeros_at_min_positive() {
        let vals = [0.0f32, 1e-3, 1e-1, 10.0];
        let norm = LogNorm::from_values(vals.iter());
        assert_eq!(norm.t(0.0), 0.0); // floored at vmin = 1e-3
        assert_eq!(norm.t(1e-3), 0.0);
        assert_eq!(norm.t(10.0), 1.0);
        let mid = norm.t(0.1);
        assert!(mid > 0.0 && mid < 1.0);
        // Decades 1e-3..1e1 inclusive.
        let ticks = norm.decade_ticks();
        assert_eq!(ticks.len(), 5);
        assert_eq!(ticks[0].1, -3);
        assert_eq!(ticks[4].1, 1);
    }

    #[test]
    fn pca_recovers_a_planar_embedding() {
        // Points on a 2-D lattice embedded in 5-D by a fixed linear map:
        // pairwise distances must survive the joint PCA projection.
        let (k, n, d) = (6, 4, 5);
        let embed = |u: f32, v: f32| {
            [u + 0.5 * v, -u + v, 2.0 * u, 0.3 * v, u - 0.7 * v]
        };
        let x = Array3::from_shape_fn((k, n, d), |(ki, ni, di)| {
            embed(ki as f32, ni as f32)[di]
        });
        let z = Array3::from_shape_fn((k, n, d), |(ki, ni, di)| {
            embed(ki as f32 + 0.25, ni as f32 - 0.25)[di]
        });
        let (x2, z2) = project_xz(&x, &z);
        assert_eq!(x2.dim(), (k, n, 2));
        // Distances in the embedded plane vs. in the projection.
        let d5 = |a: &Array3<f32>, i: (usize, usize), b: &Array3<f32>, j: (usize, usize)| {
            (0..d)
                .map(|c| (a[[i.0, i.1, c]] - b[[j.0, j.1, c]]).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        let d2 = |a: &Array3<f32>, i: (usize, usize), b: &Array3<f32>, j: (usize, usize)| {
            (0..2)
                .map(|c| (a[[i.0, i.1, c]] - b[[j.0, j.1, c]]).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        for (i, j) in [((0, 0), (5, 3)), ((2, 1), (3, 2)), ((1, 3), (4, 0))] {
            let want = d5(&x, i, &x, j);
            let got = d2(&x2, i, &x2, j);
            assert!((want - got).abs() < 1e-3 * want.max(1.0), "{want} vs {got}");
            let want = d5(&x, i, &z, j);
            let got = d2(&x2, i, &z2, j);
            assert!((want - got).abs() < 1e-3 * want.max(1.0), "{want} vs {got}");
        }
    }

    #[test]
    fn grid_layout_uses_centers_when_present() {
        // 2x3 grid of centers, shuffled agent order.
        let centers = ndarray::array![[9i64, 4], [1, 4], [1, 0], [9, 0], [1, 8], [9, 8]];
        let (nrows, ncols, slots) = grid_layout(6, Some(&centers));
        assert_eq!((nrows, ncols), (2, 3));
        assert_eq!(slots, vec![(1, 1), (0, 1), (0, 0), (1, 0), (0, 2), (1, 2)]);
        // Fallback: near-square index grid.
        let (nrows, ncols, slots) = grid_layout(5, None);
        assert_eq!((nrows, ncols), (2, 3));
        assert_eq!(slots[4], (1, 1));
    }
}
