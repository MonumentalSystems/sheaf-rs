// sheaf-rs web demo — vanilla ES module, no frameworks, no external requests.
// Imports the wasm module per the JS/WASM API contract from './pkg/sheaf_web.js'
// (or './mock/sheaf_web.js' with ?mock=1).
//
// Pure helpers are exported at the top and are DOM-free so they run under Node
// (web/test/ui_smoke.mjs); the UI itself only boots when `document` exists.

// ---------------------------------------------------------------------------
// Pure helpers (exported, Node-testable)
// ---------------------------------------------------------------------------

/** Token classes (row-major Uint8Array grids). */
export const T = { PAD: 0, WALL: 1, EMPTY: 2, START: 3, GOAL: 4, PATH: 5 };

/** Per-class cell colors (dark theme; path = the accent amber). */
export const CLASS_COLORS = {
  [T.PAD]: "#101114",
  [T.WALL]: "#2c3038",
  [T.EMPTY]: "#cfd2c9",
  [T.START]: "#26a269",
  [T.GOAL]: "#e05252",
  [T.PATH]: "#f5b83d",
};

/** Residual series (order matches session.residuals() rows). Palette
 *  validated (dataviz six checks) against the dark surface #16181d. */
export const SERIES = [
  { key: "consistency", label: "consistency", color: "#3987e5" },
  { key: "primal", label: "primal", color: "#199e70" },
  { key: "dual", label: "dual", color: "#c98500" },
];

/** K per maze size: 100 in-distribution (19), 60 for 37x37 OOD. */
export function defaultK(h) {
  return h <= 19 ? 100 : 60;
}

/** Module URL for the given query string ('?mock=1' -> the JS mock). */
export function wasmModuleUrl(search) {
  return new URLSearchParams(search).get("mock") === "1"
    ? "./mock/sheaf_web.js"
    : "./pkg/sheaf_web.js";
}

/** First index of token `tok`, or -1. */
export function findToken(tokens, tok) {
  for (let i = 0; i < tokens.length; i++) if (tokens[i] === tok) return i;
  return -1;
}

/** BFS start->goal through non-wall cells. {solvable, dist, path|null}. */
export function bfsPath(tokens, h, w) {
  const from = findToken(tokens, T.START);
  const to = findToken(tokens, T.GOAL);
  if (from < 0 || to < 0) return { solvable: false, dist: -1, path: null };
  const prev = new Int32Array(h * w).fill(-1);
  const seen = new Uint8Array(h * w);
  const queue = new Int32Array(h * w);
  let head = 0, tail = 0;
  queue[tail++] = from; seen[from] = 1;
  while (head < tail) {
    const cur = queue[head++];
    if (cur === to) break;
    const cy = (cur / w) | 0, cx = cur % w;
    const nbrs = [cur - w, cur + w, cur - 1, cur + 1];
    const ok = [cy > 0, cy < h - 1, cx > 0, cx < w - 1];
    for (let d = 0; d < 4; d++) {
      const n = nbrs[d];
      if (!ok[d] || seen[n] || tokens[n] === T.WALL || tokens[n] === T.PAD) continue;
      seen[n] = 1; prev[n] = cur; queue[tail++] = n;
    }
  }
  if (!seen[to]) return { solvable: false, dist: -1, path: null };
  const rev = [];
  for (let c = to; c !== -1; c = prev[c]) rev.push(c);
  rev.reverse();
  return { solvable: true, dist: rev.length - 1, path: Int32Array.from(rev) };
}

export function bfsSolvable(tokens, h, w) {
  return bfsPath(tokens, h, w).solvable;
}

/** Pointer px -> integer cell coords. */
export function cellAt(px, py, cellSize) {
  return { x: Math.floor(px / cellSize), y: Math.floor(py / cellSize) };
}

/** Nearest agent index for a cell (meta = Uint32Array rows of cy,cx,patch). */
export function nearestAgent(meta, y, x) {
  let best = -1, bestD = Infinity;
  for (let a = 0; a * 3 < meta.length; a++) {
    const dy = meta[a * 3] - y, dx = meta[a * 3 + 1] - x;
    const d = dy * dy + dx * dx;
    if (d < bestD) { bestD = d; best = a; }
  }
  return best;
}

/** Do the predicted PATH cells of a frame connect start to goal?
 *  Walkable set: predicted PATH plus the true start/goal endpoints. */
export function predictedPathConnects(frame, tokens, h, w) {
  const from = findToken(tokens, T.START);
  const to = findToken(tokens, T.GOAL);
  if (from < 0 || to < 0) return false;
  const walk = new Uint8Array(h * w);
  for (let i = 0; i < frame.length; i++) if (frame[i] === T.PATH) walk[i] = 1;
  walk[from] = 1; walk[to] = 1;
  const seen = new Uint8Array(h * w);
  const queue = new Int32Array(h * w);
  let head = 0, tail = 0;
  queue[tail++] = from; seen[from] = 1;
  while (head < tail) {
    const cur = queue[head++];
    if (cur === to) return true;
    const cy = (cur / w) | 0, cx = cur % w;
    const nbrs = [cur - w, cur + w, cur - 1, cur + 1];
    const ok = [cy > 0, cy < h - 1, cx > 0, cx < w - 1];
    for (let d = 0; d < 4; d++) {
      const n = nbrs[d];
      if (ok[d] && !seen[n] && walk[n]) { seen[n] = 1; queue[tail++] = n; }
    }
  }
  return false;
}

/** Log-scale y mapping: value -> pixel (top=0). Clamps below the floor. */
export function logYPixel(v, yMin, yMax, heightPx) {
  const lv = Math.log10(Math.max(v, yMin));
  const lo = Math.log10(yMin), hi = Math.log10(yMax);
  const f = hi > lo ? (lv - lo) / (hi - lo) : 0.5;
  return (1 - f) * heightPx;
}

/** [min positive, max] over a residuals Float32Array (k*3), first `upto` iters. */
export function residualRange(res, upto) {
  let lo = Infinity, hi = -Infinity;
  const n = Math.min(res.length, upto * 3);
  for (let i = 0; i < n; i++) {
    const v = res[i];
    if (v > 0 && v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!isFinite(lo)) { lo = 1e-6; hi = 1; }
  return [lo, hi];
}

// ---------------------------------------------------------------------------
// UI (browser only)
// ---------------------------------------------------------------------------

if (typeof document !== "undefined") {
  boot().catch((e) => {
    const el = document.getElementById("banner");
    if (el) {
      el.hidden = false;
      el.textContent =
        "failed to load the wasm module (" + e.message + ") — build crates/sheaf-web " +
        "into web/pkg/, or append ?mock=1 to run the UI against the pure-JS mock.";
    }
    console.error(e);
  });
}

async function boot() {
  const mod = await import(wasmModuleUrl(location.search));
  await mod.default(); // wasm-bindgen init (no-op in the mock)

  let session;
  try {
    session = new mod.SheafSession();
  } catch (e) {
    throw new Error("SheafSession failed to construct: " + (e && e.message ? e.message : e));
  }

  const $ = (id) => document.getElementById(id);
  const maze = $("maze"), chart = $("chart");
  const mctx = maze.getContext("2d"), cctx = chart.getContext("2d");
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));

  const st = {
    h: 19, w: 19, seed: 42,
    tokens: null,          // Uint8Array [h*w]
    generated: true,       // false once hand-edited
    k: 0, iter: 0,
    solved: false, stale: false,
    playing: false,
    frames: new Map(),     // iter -> Uint8Array
    agentCons: new Map(),  // iter -> Float32Array
    residuals: null, meta: null,
    editMode: false, overlayOn: true,
    hoverAgent: -1, dragTok: 0, dragging: false,
    cell: 30,
  };

  // ---- maze canvas sizing -------------------------------------------------
  function sizeMaze() {
    const maxPx = Math.min(600, maze.parentElement.clientWidth || 600);
    st.cell = Math.max(6, Math.floor(maxPx / st.w));
    const cssW = st.cell * st.w, cssH = st.cell * st.h;
    maze.style.width = cssW + "px";
    maze.style.height = cssH + "px";
    maze.width = Math.round(cssW * dpr);
    maze.height = Math.round(cssH * dpr);
    mctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    mctx.imageSmoothingEnabled = false;
  }

  function sizeChart() {
    const cssW = chart.parentElement.clientWidth || 420;
    const cssH = 300;
    chart.style.width = cssW + "px";
    chart.style.height = cssH + "px";
    chart.width = Math.round(cssW * dpr);
    chart.height = Math.round(cssH * dpr);
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ---- drawing ------------------------------------------------------------
  function displayGrid() {
    if (st.solved && !st.stale) return getFrame(st.iter);
    return st.tokens;
  }

  function getFrame(i) {
    let f = st.frames.get(i);
    if (!f) { f = session.frame(i); st.frames.set(i, f); }
    return f;
  }

  function getAgentCons(i) {
    let a = st.agentCons.get(i);
    if (!a) { a = session.agent_consistency(i); st.agentCons.set(i, a); }
    return a;
  }

  function drawMaze() {
    const grid = displayGrid();
    if (!grid) return;
    const c = st.cell;
    for (let y = 0; y < st.h; y++) {
      for (let x = 0; x < st.w; x++) {
        mctx.fillStyle = CLASS_COLORS[grid[y * st.w + x]] || CLASS_COLORS[T.PAD];
        mctx.fillRect(x * c, y * c, c, c);
      }
    }
    // true S/G markers from the input tokens, always on top
    mctx.font = `bold ${Math.max(9, c - 6)}px system-ui, sans-serif`;
    mctx.textAlign = "center";
    mctx.textBaseline = "middle";
    for (let i = 0; i < st.tokens.length; i++) {
      const t = st.tokens[i];
      if (t !== T.START && t !== T.GOAL) continue;
      const y = (i / st.w) | 0, x = i % st.w;
      mctx.fillStyle = CLASS_COLORS[t];
      mctx.fillRect(x * c, y * c, c, c);
      mctx.fillStyle = "#0d0f12";
      mctx.fillText(t === T.START ? "S" : "G", x * c + c / 2, y * c + c / 2 + 0.5);
    }
    // agent-lattice hover overlay
    if (st.overlayOn && st.solved && !st.stale && st.hoverAgent >= 0 && st.meta) {
      const a = st.hoverAgent;
      const cy = st.meta[a * 3], cx = st.meta[a * 3 + 1], p = st.meta[a * 3 + 2];
      const r = (p / 2) | 0;
      mctx.strokeStyle = "#f5b83d";
      mctx.lineWidth = 2;
      mctx.strokeRect((cx - r) * c + 1, (cy - r) * c + 1, p * c - 2, p * c - 2);
      mctx.fillStyle = "#f5b83d";
      mctx.fillRect(cx * c + c / 2 - 2, cy * c + c / 2 - 2, 4, 4);
    }
  }

  function fmt(v) {
    return v.toExponential(2);
  }

  function drawChart() {
    const W = chart.width / dpr, H = chart.height / dpr;
    cctx.clearRect(0, 0, W, H);
    const padL = 44, padR = 10, padT = 8, padB = 22;
    const iw = W - padL - padR, ih = H - padT - padB;
    cctx.fillStyle = "#16181d";
    cctx.fillRect(0, 0, W, H);
    if (!st.solved || st.stale || !st.residuals) {
      cctx.fillStyle = "#898781";
      cctx.font = "13px system-ui, sans-serif";
      cctx.textAlign = "center";
      cctx.fillText("solve to see residuals", W / 2, H / 2);
      return;
    }
    const res = st.residuals, upto = st.iter + 1;
    let [lo, hi] = residualRange(res, st.k); // fixed scale over the full run
    lo *= 0.8; hi *= 1.25;

    // decade gridlines + tick labels
    cctx.font = "10px system-ui, sans-serif";
    cctx.textAlign = "right";
    cctx.textBaseline = "middle";
    for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) {
      const y = padT + logYPixel(Math.pow(10, e), lo, hi, ih);
      cctx.strokeStyle = "#2c2c2a";
      cctx.lineWidth = 1;
      cctx.beginPath();
      cctx.moveTo(padL, y + 0.5);
      cctx.lineTo(W - padR, y + 0.5);
      cctx.stroke();
      cctx.fillStyle = "#898781";
      cctx.fillText("1e" + e, padL - 6, y);
    }
    // x ticks
    cctx.textAlign = "center";
    cctx.textBaseline = "top";
    const xStep = st.k > 60 ? 20 : 10;
    for (let i = 0; i < st.k; i += xStep) {
      const x = padL + (st.k > 1 ? (i / (st.k - 1)) * iw : 0);
      cctx.fillStyle = "#898781";
      cctx.fillText(String(i), x, H - padB + 6);
    }
    // series lines up to the current iteration
    for (let s = 0; s < 3; s++) {
      cctx.strokeStyle = SERIES[s].color;
      cctx.lineWidth = 2;
      cctx.lineJoin = "round";
      cctx.beginPath();
      for (let i = 0; i < upto; i++) {
        const x = padL + (st.k > 1 ? (i / (st.k - 1)) * iw : 0);
        const y = padT + logYPixel(res[i * 3 + s], lo, hi, ih);
        if (i === 0) cctx.moveTo(x, y);
        else cctx.lineTo(x, y);
      }
      cctx.stroke();
      // direct label at the line end
      if (upto > 1) {
        const i = upto - 1;
        const x = padL + (i / (st.k - 1)) * iw;
        const y = padT + logYPixel(res[i * 3 + s], lo, hi, ih);
        cctx.fillStyle = SERIES[s].color;
        cctx.beginPath();
        cctx.arc(x, y, 2.5, 0, Math.PI * 2);
        cctx.fill();
      }
    }
    // legend readout (HTML, tabular numerals)
    const i = st.iter;
    for (let s = 0; s < 3; s++) {
      $("v-" + SERIES[s].key).textContent = fmt(res[i * 3 + s]);
    }
  }

  function updateStatus() {
    const dist = st.tokens ? bfsPath(st.tokens, st.h, st.w).dist : -1;
    const kind = st.generated ? "in-distribution" : "hand-edited";
    const ood = st.h > 19 ? " · out-of-distribution size" : "";
    $("status").textContent =
      `${st.h}×${st.w} · seed ${st.seed} · ${kind}${ood}` +
      (dist >= 0 ? ` · shortest path ${dist}` : "") +
      (st.solved && !st.stale ? ` · iter ${st.iter + 1}/${st.k}` : "");
    $("warning").hidden = !st.tokens || bfsSolvable(st.tokens, st.h, st.w);
    $("stale").hidden = !(st.solved && st.stale);
    $("iterLabel").textContent = st.solved && !st.stale ? `${st.iter + 1}/${st.k}` : "—";
  }

  function redraw() {
    drawMaze();
    drawChart();
    updateStatus();
  }

  // ---- actions ------------------------------------------------------------
  function invalidateSolve() {
    st.stale = true;
    st.playing = false;
    $("play").textContent = "play";
  }

  function generate(seed) {
    st.seed = seed >>> 0;
    $("seed").value = String(st.seed);
    st.tokens = mod.generate_maze(BigInt(st.seed), st.h, st.w);
    st.generated = true;
    st.solved = false;
    invalidateSolve();
    st.frames.clear(); st.agentCons.clear();
    sizeMaze();
    redraw();
  }

  function doSolve() {
    if (!st.tokens) return;
    try {
      const k = defaultK(st.h);
      st.k = Number(session.solve(st.tokens, st.h, st.w, k));
      st.frames.clear();
      st.agentCons.clear();
      st.residuals = session.residuals();
      st.meta = session.agent_meta();
      st.solved = true;
      st.stale = false;
      st.iter = 0;
      const slider = $("iter");
      slider.max = String(st.k - 1);
      slider.value = "0";
      slider.disabled = false;
      setPlaying(true);
      $("banner").hidden = true;
      redraw();
    } catch (e) {
      const el = $("banner");
      el.hidden = false;
      el.textContent = "solve failed: " + (e && e.message ? e.message : e);
    }
  }

  function setIter(i) {
    st.iter = Math.min(Math.max(0, i), st.k - 1);
    $("iter").value = String(st.iter);
    redraw();
  }

  function setPlaying(p) {
    if (!st.solved || st.stale) p = false;
    st.playing = p;
    $("play").textContent = p ? "pause" : "play";
    if (p) requestAnimationFrame(tick);
  }

  let lastStep = 0;
  const FRAME_MS = 1000 / 15;
  function tick(now) {
    if (!st.playing) return;
    if (now - lastStep >= FRAME_MS) {
      lastStep = now;
      if (st.iter + 1 >= st.k) {
        setPlaying(false);
      } else {
        setIter(st.iter + 1);
      }
    }
    if (st.playing) requestAnimationFrame(tick);
  }

  // ---- maze editing -------------------------------------------------------
  function pointerCell(ev) {
    const r = maze.getBoundingClientRect();
    const { x, y } = cellAt(ev.clientX - r.left, ev.clientY - r.top, st.cell);
    if (x < 0 || y < 0 || x >= st.w || y >= st.h) return -1;
    return y * st.w + x;
  }

  maze.addEventListener("pointerdown", (ev) => {
    if (!st.editMode || !st.tokens) return;
    const i = pointerCell(ev);
    if (i < 0) return;
    const t = st.tokens[i];
    if (t === T.START || t === T.GOAL) {
      st.dragging = true;
      st.dragTok = t;
      maze.setPointerCapture(ev.pointerId);
      return;
    }
    if (t === T.WALL || t === T.EMPTY) {
      st.tokens[i] = t === T.WALL ? T.EMPTY : T.WALL;
      st.generated = false;
      invalidateSolve();
      redraw();
    }
  });

  maze.addEventListener("pointermove", (ev) => {
    // hover overlay tracking
    const i = pointerCell(ev);
    if (st.solved && !st.stale && st.overlayOn && st.meta && i >= 0) {
      const a = nearestAgent(st.meta, (i / st.w) | 0, i % st.w);
      if (a !== st.hoverAgent) {
        st.hoverAgent = a;
        const v = getAgentCons(st.iter)[a];
        $("agentReadout").textContent =
          `agent ${a} · center (${st.meta[a * 3]}, ${st.meta[a * 3 + 1]}) · ` +
          `patch ${st.meta[a * 3 + 2]}×${st.meta[a * 3 + 2]} · consistency ${fmt(v)}`;
        drawMaze();
      }
    }
    // S/G drag
    if (st.dragging && st.editMode && i >= 0) {
      const t = st.tokens[i];
      if (t === T.EMPTY) {
        const old = findToken(st.tokens, st.dragTok);
        if (old >= 0) st.tokens[old] = T.EMPTY;
        st.tokens[i] = st.dragTok;
        st.generated = false;
        invalidateSolve();
        redraw();
      }
    }
  });

  maze.addEventListener("pointerup", () => { st.dragging = false; });
  maze.addEventListener("pointerleave", () => {
    if (st.hoverAgent !== -1) {
      st.hoverAgent = -1;
      $("agentReadout").textContent = "hover the maze to inspect an agent";
      drawMaze();
    }
  });

  // ---- controls -----------------------------------------------------------
  $("generate").addEventListener("click", () => {
    generate(parseInt($("seed").value, 10) || 0);
  });
  $("random").addEventListener("click", () => {
    generate((Math.random() * 0xffffffff) >>> 0);
  });
  document.querySelectorAll('input[name="size"]').forEach((r) => {
    r.addEventListener("change", () => {
      st.h = st.w = parseInt(r.value, 10);
      generate(st.seed);
    });
  });
  $("editMode").addEventListener("change", (ev) => {
    st.editMode = ev.target.checked;
    maze.classList.toggle("editing", st.editMode);
  });
  $("overlay").addEventListener("change", (ev) => {
    st.overlayOn = ev.target.checked;
    st.hoverAgent = -1;
    drawMaze();
  });
  $("solve").addEventListener("click", doSolve);
  $("play").addEventListener("click", () => setPlaying(!st.playing));
  $("iter").addEventListener("input", (ev) => {
    setPlaying(false);
    setIter(parseInt(ev.target.value, 10) || 0);
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.target && /^(INPUT|TEXTAREA|SELECT)$/.test(ev.target.tagName) &&
        ev.target.type !== "range" && ev.target.type !== "checkbox") return;
    if (ev.key === " ") {
      ev.preventDefault();
      setPlaying(!st.playing);
    } else if (ev.key === "ArrowRight") {
      ev.preventDefault();
      setPlaying(false);
      if (st.solved && !st.stale) setIter(st.iter + 1);
    } else if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      setPlaying(false);
      if (st.solved && !st.stale) setIter(st.iter - 1);
    }
  });

  window.addEventListener("resize", () => { sizeMaze(); sizeChart(); redraw(); });

  // ---- boot ---------------------------------------------------------------
  // series legend swatch colors
  for (const s of SERIES) {
    const sw = $("sw-" + s.key);
    if (sw) sw.style.background = s.color;
  }
  sizeChart();
  generate(st.seed);
}
