// Pure-JS mock of the sheaf_web wasm module (same JS/WASM API contract).
// Used with `?mock=1` so the UI is developable/testable without the wasm pkg.
//
// Contract:
//   init()                         default export, async no-op here
//   generate_maze(seed, h, w)      -> Uint8Array [h*w] tokens
//   new SheafSession()
//   session.solve(tokens, h, w, k) -> number (k). Throws Error on bad input.
//   session.frame(iter)            -> Uint8Array [h*w] per-cell argmax class
//   session.residuals()            -> Float32Array [k*3]
//   session.agent_consistency(it)  -> Float32Array [N]
//   session.agent_meta()           -> Uint32Array [N*3] (cy, cx, patch_size)
//
// Tokens: 0 pad, 1 wall, 2 empty, 3 start, 4 goal, 5 path.
// The "solver" here is BFS: frames reveal the true shortest path progressively
// with decaying fake disagreement, so the UI animation is representative.
//
// Failed-solve semantics (matches the wasm session): solve() validates before
// mutating any state, so a throwing solve() leaves the previous successful
// solve's frames/residuals/meta readable.

const PAD = 0, WALL = 1, EMPTY = 2, START = 3, GOAL = 4, PATH = 5;

function mulberry32(a) {
  a >>>= 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function h32(a) {
  a |= 0;
  a = Math.imul(a ^ (a >>> 16), 2246822507);
  a = Math.imul(a ^ (a >>> 13), 3266489909);
  return (a ^= a >>> 16) >>> 0;
}

function hashTokens(tokens) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < tokens.length; i++) {
    h ^= tokens[i];
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** BFS through non-wall cells; returns {dist, path: Int32Array|null}. */
function bfs(tokens, h, w, from, to) {
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
      if (!ok[d] || seen[n] || tokens[n] === WALL || tokens[n] === PAD) continue;
      seen[n] = 1; prev[n] = cur; queue[tail++] = n;
    }
  }
  if (!seen[to]) return { dist: -1, path: null };
  const rev = [];
  for (let c = to; c !== -1; c = prev[c]) rev.push(c);
  rev.reverse();
  return { dist: rev.length - 1, path: Int32Array.from(rev) };
}

/** Faithful-in-spirit port of the odd-index-lattice DFS carve + BFS-distance
 *  start/goal placement (double-BFS picks a near-diameter pair). */
export function generate_maze(seed, height, width) {
  const h = Number(height), w = Number(width);
  if (h % 2 === 0 || w % 2 === 0 || h < 5 || w < 5) {
    throw new Error(`odd sizes only (got ${h}x${w})`);
  }
  const s = typeof seed === "bigint" ? Number(seed & 0xffffffffn) : seed >>> 0;
  const rng = mulberry32(h32(s ^ Math.imul(h, 73856093) ^ Math.imul(w, 19349663)));
  const grid = new Uint8Array(h * w).fill(WALL);

  // odd-lattice cells strictly interior
  const oddCells = [];
  for (let y = 1; y < h - 1; y += 2)
    for (let x = 1; x < w - 1; x += 2) oddCells.push(y * w + x);

  const start0 = oddCells[(rng() * oddCells.length) | 0];
  grid[start0] = EMPTY;
  const stack = [start0];
  while (stack.length) {
    const cur = stack[stack.length - 1];
    const cy = (cur / w) | 0, cx = cur % w;
    const cand = [];
    for (const [dy, dx] of [[-2, 0], [2, 0], [0, -2], [0, 2]]) {
      const ny = cy + dy, nx = cx + dx;
      if (ny >= 1 && ny < h - 1 && nx >= 1 && nx < w - 1 && grid[ny * w + nx] === WALL) {
        cand.push([ny, nx]);
      }
    }
    if (cand.length === 0) { stack.pop(); continue; }
    const [ny, nx] = cand[(rng() * cand.length) | 0];
    grid[((cy + ny) / 2) * w + (cx + nx) / 2] = EMPTY;
    grid[ny * w + nx] = EMPTY;
    stack.push(ny * w + nx);
  }

  // start/goal: double BFS from a random empty cell -> far pair (BFS acceptance)
  const empties = [];
  for (let i = 0; i < grid.length; i++) if (grid[i] === EMPTY) empties.push(i);
  const seedCell = empties[(rng() * empties.length) | 0];
  const farthest = (from) => {
    let best = from, bestD = 0;
    for (const e of empties) {
      const { dist } = bfs(grid, h, w, from, e);
      if (dist > bestD) { bestD = dist; best = e; }
    }
    return best;
  };
  const a = farthest(seedCell);
  const b = farthest(a);
  grid[a] = START;
  grid[b] = GOAL;
  return grid;
}

export class SheafSession {
  constructor() {
    this._solved = false;
  }

  solve(tokens, height, width, k) {
    const h = Number(height), w = Number(width);
    k = Number(k);
    if (!(tokens instanceof Uint8Array)) throw new Error("tokens must be a Uint8Array");
    if (tokens.length !== h * w) throw new Error(`tokens length ${tokens.length} != ${h}x${w}`);
    if (h % 2 === 0 || w % 2 === 0) throw new Error("odd sizes only");
    if (!(k >= 1)) throw new Error("k must be >= 1");
    let si = -1, gi = -1;
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i] === START) { if (si >= 0) throw new Error("multiple start cells"); si = i; }
      if (tokens[i] === GOAL) { if (gi >= 0) throw new Error("multiple goal cells"); gi = i; }
    }
    if (si < 0 || gi < 0) throw new Error("maze needs exactly one start and one goal");

    this.h = h; this.w = w; this.k = k;
    this.tokens = Uint8Array.from(tokens);
    this.hash = hashTokens(tokens);
    const { path } = bfs(tokens, h, w, si, gi);
    this.path = path; // null if unsolvable (interior cells incl. endpoints)
    this._solved = true;

    // agent lattice: 3x3 patches, stride 2, first center at 1 (81 agents at 19x19)
    const patch = 3, stride = 2;
    const cys = [], cxs = [];
    for (let c = (patch / 2) | 0; c + ((patch / 2) | 0) < h; c += stride) cys.push(c);
    for (let c = (patch / 2) | 0; c + ((patch / 2) | 0) < w; c += stride) cxs.push(c);
    const meta = new Uint32Array(cys.length * cxs.length * 3);
    let m = 0;
    for (const cy of cys) for (const cx of cxs) { meta[m++] = cy; meta[m++] = cx; meta[m++] = patch; }
    this.meta = meta;
    return k;
  }

  _t(iter) { return this.k > 1 ? iter / (this.k - 1) : 1; }

  _requireSolved() {
    if (!this._solved) throw new Error("call solve() first");
  }

  frame(iter) {
    this._requireSolved();
    iter = Math.min(Math.max(0, Number(iter)), this.k - 1);
    const { tokens, h, w, path, hash } = this;
    const t = this._t(iter);
    const out = Uint8Array.from(tokens);
    const solvable = path !== null;
    // fake disagreement: sprinkle PATH on empty cells, decaying (or not, if unsolvable)
    const bucket = (iter / 3) | 0;
    const noiseP = solvable ? (t >= 0.95 ? 0 : 0.3 * Math.exp(-5 * t)) : 0.22;
    for (let i = 0; i < out.length; i++) {
      if (tokens[i] !== EMPTY) continue;
      if (h32(Math.imul(i, 2654435761) ^ hash ^ Math.imul(bucket, 40503)) / 4294967296 < noiseP) {
        out[i] = PATH;
      }
    }
    // reveal the true path progressively from start to goal
    if (solvable) {
      const n = path.length;
      for (let j = 0; j < n; j++) {
        const jitter = (h32(path[j] ^ hash) / 4294967296) * 0.1;
        const tau = 0.15 + 0.55 * (n > 1 ? j / (n - 1) : 0) + jitter;
        if (t >= tau && tokens[path[j]] === EMPTY) out[path[j]] = PATH;
      }
    }
    return out;
  }

  residuals() {
    this._requireSolved();
    const out = new Float32Array(this.k * 3);
    const solvable = this.path !== null;
    for (let i = 0; i < this.k; i++) {
      const t = this._t(i);
      const j = (n) => 1 + 0.08 * Math.sin(i * 1.7 + n);
      const consFloor = solvable ? 0.015 : 0.12;
      out[i * 3 + 0] = (0.5 * Math.exp(-3.2 * t) + consFloor) * j(0);
      out[i * 3 + 1] = (0.8 * Math.exp(-4.0 * t) + 0.004) * j(1);
      out[i * 3 + 2] = (0.3 * Math.exp(-2.5 * t) + 0.008) * j(2);
    }
    return out;
  }

  agent_consistency(iter) {
    this._requireSolved();
    iter = Math.min(Math.max(0, Number(iter)), this.k - 1);
    const t = this._t(iter);
    const n = this.meta.length / 3;
    const out = new Float32Array(n);
    const solvable = this.path !== null;
    const onPath = new Uint8Array(this.h * this.w);
    if (this.path) for (const c of this.path) onPath[c] = 1;
    for (let a = 0; a < n; a++) {
      const cy = this.meta[a * 3], cx = this.meta[a * 3 + 1], p = this.meta[a * 3 + 2];
      let near = 0;
      const r = (p / 2) | 0;
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++)
          if (onPath[(cy + dy) * this.w + (cx + dx)]) near = 1;
      const base = 0.05 + 0.25 * (h32(a ^ this.hash) / 4294967296) + 0.2 * near;
      out[a] = solvable
        ? base * Math.exp(-4 * t) + 0.001
        : base * (0.6 + 0.15 * Math.sin(iter * 0.9 + a));
    }
    return out;
  }

  agent_meta() {
    this._requireSolved();
    return Uint32Array.from(this.meta);
  }
}

export default async function init() {
  return {};
}
