// Node smoke test for the web UI's pure helpers + the wasm API contract.
//
//   node web/test/ui_smoke.mjs                      # against the JS mock
//   SHEAF_WEB_MODULE=/path/to/nodejs-pkg/sheaf_web.js \
//     node web/test/ui_smoke.mjs                    # against the real wasm
//                                                   # (wasm-bindgen --target nodejs)
//
// Exercises the full generate -> solve -> frame flow (same API contract for
// mock and wasm) and the DOM-free helpers app.js exports. Assertions that
// depend on the mock's fake solver internals, or on model accuracy that is
// only guaranteed in-distribution, are gated accordingly.

import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import {
  T, CLASS_COLORS, SERIES,
  defaultK, wasmModuleUrl, findToken,
  bfsPath, bfsSolvable, cellAt, nearestAgent,
  predictedPathConnects, logYPixel, residualRange,
} from "../app.js";

const modulePath = process.env.SHEAF_WEB_MODULE;
const isMock = !modulePath;
const mod = await import(
  isMock ? "../mock/sheaf_web.js" : pathToFileURL(modulePath).href
);
const { generate_maze, SheafSession } = mod;
console.log(isMock ? "module: ../mock/sheaf_web.js" : `module: ${modulePath}`);

let n = 0;
function ok(cond, msg) {
  n++;
  assert.ok(cond, msg);
  console.log(`  ok ${n}: ${msg}`);
}

// ---- static helpers --------------------------------------------------------
ok(defaultK(19) === 100 && defaultK(37) === 60, "defaultK: 100 @19, 60 @37");
ok(wasmModuleUrl("?mock=1") === "./mock/sheaf_web.js", "?mock=1 -> mock module URL");
ok(wasmModuleUrl("") === "./pkg/sheaf_web.js", "default -> pkg module URL");
ok(Object.keys(CLASS_COLORS).length === 6, "six token classes colored");
ok(SERIES.length === 3, "three residual series");
{
  const c = cellAt(95, 31, 30);
  ok(c.x === 3 && c.y === 1, "cellAt pixel->cell mapping");
}
{
  ok(Math.abs(logYPixel(1e-3, 1e-3, 1e-1, 100) - 100) < 1e-9 &&
     Math.abs(logYPixel(1e-1, 1e-3, 1e-1, 100)) < 1e-9 &&
     Math.abs(logYPixel(1e-2, 1e-3, 1e-1, 100) - 50) < 1e-9,
     "logYPixel maps decades linearly");
}

// ---- module init -----------------------------------------------------------
// The mock's default export is an async no-op init(); the nodejs-target wasm
// build initializes on import and has no default export.
if (typeof mod.default === "function") await mod.default();
ok(true, "module init() resolves");

// ---- generate -> solve -> frame, both sizes --------------------------------
for (const size of [19, 37]) {
  console.log(`-- ${size}x${size} --`);
  const tokens = generate_maze(BigInt(7), size, size);
  ok(tokens instanceof Uint8Array && tokens.length === size * size,
     `generate_maze returns Uint8Array [${size * size}]`);

  // structural invariants: border all wall, exactly one S and one G, odd lattice
  let border = true;
  for (let i = 0; i < size; i++) {
    border &&= tokens[i] === T.WALL && tokens[(size - 1) * size + i] === T.WALL &&
               tokens[i * size] === T.WALL && tokens[i * size + size - 1] === T.WALL;
  }
  ok(border, "border cells are all wall");
  const counts = new Array(6).fill(0);
  for (const t of tokens) counts[t]++;
  ok(counts[T.START] === 1 && counts[T.GOAL] === 1, "exactly one start and one goal");
  ok(counts[T.PAD] === 0 && counts[T.PATH] === 0, "no pad/path tokens in a fresh maze");
  // odd-lattice carve: every even-even interior coordinate is wall
  let lattice = true;
  for (let y = 2; y < size - 1; y += 2)
    for (let x = 2; x < size - 1; x += 2)
      lattice &&= tokens[y * size + x] === T.WALL;
  ok(lattice, "even-even lattice sites are wall (odd-lattice carve)");

  const p = bfsPath(tokens, size, size);
  ok(p.solvable && p.dist >= size - 1,
     `generated maze BFS-solvable with path >= ${size - 1} (got ${p.dist})`);

  // solve
  const sess = new SheafSession();
  const k = sess.solve(tokens, size, size, defaultK(size));
  ok(k === defaultK(size), `solve returns k=${k}`);

  const res = sess.residuals();
  ok(res instanceof Float32Array && res.length === k * 3, "residuals [k*3]");
  ok([...res].every((v) => v > 0 && Number.isFinite(v)), "residuals positive/finite");
  const [lo, hi] = residualRange(res, k);
  ok(lo > 0 && hi > lo, "residualRange gives a usable log-scale span");

  const meta = sess.agent_meta();
  const nAgents = size === 19 ? 81 : 324; // ((size-1)/2)^2 for patch 3, stride 2
  ok(meta instanceof Uint32Array && meta.length === nAgents * 3,
     `agent_meta [${nAgents}*3]`);
  ok(meta[2] === 3 && meta[0] === 1 && meta[1] === 1,
     "first agent centered (1,1), patch 3");
  const cons = sess.agent_consistency(0);
  ok(cons instanceof Float32Array && cons.length === nAgents,
     `agent_consistency [${nAgents}]`);
  ok(nearestAgent(meta, 1, 1) === 0, "nearestAgent finds agent 0 at (1,1)");

  // frames
  const f0 = sess.frame(0);
  const fLast = sess.frame(k - 1);
  ok(f0 instanceof Uint8Array && f0.length === size * size &&
     fLast.length === size * size, "frames are [h*w]");
  ok([...fLast].every((v) => v <= 5), "frame classes within token range");
  // Accuracy-dependent assertions: guaranteed in-distribution (19x19, where
  // the trained weights hit 99.6% solved / 100% cell_acc) and by the mock's
  // BFS solver; at 37x37 OOD the real model solves ~53%, so skip there.
  if (isMock || size === 19) {
    ok(findToken(fLast, T.START) === findToken(tokens, T.START) &&
       findToken(fLast, T.GOAL) === findToken(tokens, T.GOAL),
       "start/goal preserved in frames");
    // final frame's predicted PATH agrees with BFS (connects start to goal,
    // and every PATH cell sits on a true empty cell)
    let onEmpty = true;
    for (let i = 0; i < fLast.length; i++) {
      if (fLast[i] === T.PATH && tokens[i] !== T.EMPTY) onEmpty = false;
    }
    ok(onEmpty, "predicted PATH cells lie on empty cells");
    ok(predictedPathConnects(fLast, tokens, size, size),
       "final frame PATH connects start to goal (BFS agreement)");
  }
  if (isMock) {
    ok(!predictedPathConnects(f0, tokens, size, size) ||
       sess.path.length <= 3,
       "iter-0 frame has not yet crystallized the path");
  }
}

// ---- hand-edited unsolvable maze -------------------------------------------
console.log("-- unsolvable edit --");
{
  const size = 19;
  const tokens = generate_maze(BigInt(3), size, size);
  // wall off the goal completely
  const gi = findToken(tokens, T.GOAL);
  const gy = (gi / size) | 0, gx = gi % size;
  for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
    const y = gy + dy, x = gx + dx;
    if (y >= 0 && y < size && x >= 0 && x < size) tokens[y * size + x] = T.WALL;
  }
  ok(!bfsSolvable(tokens, size, size), "walled-off goal detected as unsolvable");
  const sess = new SheafSession();
  const k = sess.solve(tokens, size, size, 100); // solve still runs
  ok(k === 100, "solve runs on an unsolvable maze (agents disagree, no throw)");
  if (isMock) {
    ok(!predictedPathConnects(sess.frame(k - 1), tokens, size, size),
       "no BFS-connected prediction on an unsolvable maze");
  }
}

// ---- error contract ----------------------------------------------------------
console.log("-- error contract --");
{
  const sess = new SheafSession();
  const blank = new Uint8Array(19 * 19).fill(T.EMPTY);
  assert.throws(() => sess.solve(blank, 19, 19, 100), /start/);
  ok(true, "solve throws on tokens without start/goal");
  assert.throws(() => sess.solve(new Uint8Array(10), 19, 19, 100), /length/);
  ok(true, "solve throws on wrong-length tokens");
  assert.throws(() => generate_maze(1n, 20, 20), /odd/);
  ok(true, "generate_maze rejects even sizes");

  // Failed-solve semantics (shared contract, mock and wasm): a throwing
  // solve() leaves the previous successful solve's results readable.
  const tokens = generate_maze(11n, 19, 19);
  const k = sess.solve(tokens, 19, 19, 8);
  const frameBefore = sess.frame(k - 1);
  const resBefore = sess.residuals();
  assert.throws(() => sess.solve(blank, 19, 19, 8), /start/);
  assert.deepEqual(sess.frame(k - 1), frameBefore);
  assert.deepEqual(sess.residuals(), resBefore);
  ok(true, "failed solve preserves the previous solve's frames/residuals");
}

console.log(`\nui_smoke: all ${n} assertions passed`);
