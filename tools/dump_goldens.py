#!/usr/bin/env python
"""Dump maze golden fixtures (batch.npz + trace.npz) per goldens/CONTRACT.md.

Run *after* export_weights.py, from the upstream sheaf-admm checkout's uv env:

    uv run python /rjs/AI/sheaf-rs/tools/dump_goldens.py \
        --repo /Users/rjs/.claude/jobs/707f1939/tmp/sheaf-admm \
        --out  /rjs/AI/sheaf-rs/goldens/maze

The weights are NOT re-initialized here: the ema_params tree is loaded back
from weights.safetensors (round-trip through the exact bytes Rust will read)
and unflattened into the Flax pytree. The batch is the first 2 rows of the
``test`` split of datasets/maze_small (sequential iter_test_batches order).
The trace is produced by ``SheafADMMModel.coordinate_history`` with
num_iters=K=12, plus the encoder heads (via the model's ``_encode`` method,
same params) and the eval-time overlap-mean reassembly + argmax grid.
Completes manifest.json with the batch.npz / trace.npz sections.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

K_ITERS = 12
BATCH = 2


def unflatten(flat: dict[str, np.ndarray]) -> dict:
    """'/'-joined keys -> nested dict (inverse of export_weights.flatten_params)."""
    tree: dict = {}
    for key, value in flat.items():
        parts = key.split("/")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return tree


def f32(x) -> np.ndarray:
    a = np.asarray(x)
    assert a.dtype == np.float32, f"expected f32, got {a.dtype}"
    return a


def i64(x) -> np.ndarray:
    return np.asarray(x).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="puzzle dataset dir (default: <repo>/datasets/maze_small)")
    args = ap.parse_args()
    dataset_dir = args.dataset or (args.repo / "datasets" / "maze_small")

    sys.path.insert(0, str(args.repo / "src"))

    import jax.numpy as jnp  # noqa: E402
    import sheaf_admm  # noqa: F401,E402  pins fp32 matmul precision on import
    from safetensors.numpy import load_file  # noqa: E402
    from sheaf_admm.data import PuzzleDataset  # noqa: E402
    from sheaf_admm.data import views as V  # noqa: E402
    from sheaf_admm.geometry.restriction_maps import compute_direction_index  # noqa: E402
    from sheaf_admm.models import model_config_from_dict  # noqa: E402
    from sheaf_admm.models.sheaf_model import SheafADMMModel  # noqa: E402

    # --- model from the experiment yaml (the exporter's source of truth: the
    # contract's config.json "model" object deliberately omits rho_init because
    # rho is baked, so it cannot rebuild the Python ModelConfig by itself);
    # weights from the exported safetensors ---
    import yaml  # noqa: E402

    exp = yaml.safe_load((args.repo / "configs/experiment/maze_sheaf.yaml").read_text())
    config = json.loads((args.out / "config.json").read_text())
    cfg = model_config_from_dict(exp["model"])
    model = SheafADMMModel(config=cfg)

    tensors = load_file(str(args.out / "weights.safetensors"))
    ema_flat = {k[len("ema_params/"):]: v for k, v in tensors.items()
                if k.startswith("ema_params/")}
    variables = {"params": unflatten(ema_flat)}

    tk = config["task"]
    ps, stride, conn, C = tk["patch_size"], tk["stride"], tk["connectivity"], tk["num_classes"]

    # --- fixed batch: first BATCH rows of the test split, sequential order ---
    ds = PuzzleDataset(dataset_dir, "test")
    _set, raw = next(ds.iter_test_batches(BATCH))
    H, W = int(raw["height"]), int(raw["width"])
    tokens = i64(raw["inputs"]).reshape(BATCH, H, W)
    labels = i64(raw["labels"]).reshape(BATCH, H, W)

    # --- graph / views (mirrors training.tasks.MazeTask.prepare) ---
    centers = V.grid_agent_centers((H, W), stride=stride, patch_size=ps)  # [N,2] i64
    edges_np = V.build_grid_edge_indices(centers, stride, conn)  # [E,2] i32
    npos = V.node_positions(centers)  # [N,2] f32
    edges = jnp.asarray(edges_np)
    patches = V.prepare_maze_patches(jnp.asarray(tokens), centers, ps, C)  # [N,B,ps,ps,C] f32

    u, v = edges_np[:, 0], edges_np[:, 1]
    dy = np.asarray(npos)[v, 0] - np.asarray(npos)[u, 0]
    dx = np.asarray(npos)[v, 1] - np.asarray(npos)[u, 1]
    dir_uv = i64(compute_direction_index(jnp.asarray(dy), jnp.asarray(dx), cfg.num_directions))
    dir_vu = i64(compute_direction_index(jnp.asarray(-dy), jnp.asarray(-dx), cfg.num_directions))

    batch_arrays = {
        "tokens": tokens,
        "labels": labels,
        "centers": i64(centers),
        "node_positions": f32(np.asarray(npos)),
        "edges": i64(edges_np),
        "dir_uv": dir_uv,
        "dir_vu": dir_vu,
        "patches": f32(np.asarray(patches)),
    }

    # --- encoder heads: apply the encoder module directly on its param subtree ---
    # (SheafADMMModel._encode is not @nn.compact, so it cannot be an apply
    # method; this builds the identical MLPEncoderV2 with the identical kwargs
    # and the "MLPEncoderV2_0" params, then does the parent's [N*B]->[N,B]
    # reshape for array outputs — bit-identical to what coordinate_history sees.)
    from sheaf_admm.models.encoder import create_encoder  # noqa: E402

    encoder = create_encoder(
        "mlp_v2",
        comm_dim=cfg.d_v,
        edge_stalk_dim=cfg.d_e,
        comm_norm_type=cfg.comm_norm_type,
        objective_mode=cfg.objective_mode,
        q_epsilon=cfg.q_epsilon,
        l1_weight=cfg.l1_weight,
        l1_init=cfg.l1_init,
        upper_init=cfg.upper_init,
        beta_init=cfg.beta_init,
        rm_mode=cfg.rm_mode,
        lora_rank=cfg.lora_rank,
        lora_use_gate=cfg.lora_use_gate,
        lora_init_style=cfg.lora_init_style,
        num_directions=cfg.num_directions,
        hidden_dim=cfg.enc_hidden_dim,
        dropout_rate=cfg.dropout_rate,
    )
    N_agents, B_sz = patches.shape[:2]
    enc_flat = encoder.apply(
        {"params": variables["params"]["MLPEncoderV2_0"]},
        patches.reshape((N_agents * B_sz, *patches.shape[2:])),
        training=False,
    )
    enc_out = dict(enc_flat)
    for k in ("h", "q_diag", "q", "l1_weight", "upper", "A", "B", "gate"):
        if k in enc_out and jnp.ndim(enc_out[k]) >= 1 and enc_out[k].shape[0] == N_agents * B_sz:
            enc_out[k] = enc_out[k].reshape((N_agents, B_sz, *enc_out[k].shape[1:]))

    # --- full trace via coordinate_history (the paper-eval EMA weights) ---
    history, logits_per_iter, _final_state, geometry, rho = model.apply(
        variables,
        patches,
        edges,
        num_iters=K_ITERS,
        node_positions=npos,
        training=False,
        method=SheafADMMModel.coordinate_history,
    )

    rho_np = f32(np.asarray(rho))
    baked = np.float32(config["baked"]["rho"])
    assert rho_np == baked, f"trace rho {rho_np!r} != config.baked.rho {baked!r} (must be bitwise)"

    logits_final = np.asarray(logits_per_iter[-1])  # [N,B,ps,ps,C]
    reassembled = V.reassemble_logits(logits_final, centers, (H, W), C, mode="mean")
    pred_grid = i64(np.argmax(reassembled, axis=-1))  # [B,H,W] decoded per-cell argmax

    trace_arrays = {
        "enc_h": f32(np.asarray(enc_out["h"])),
        "enc_q_diag": f32(np.asarray(enc_out["q_diag"])),
        "enc_q": f32(np.asarray(enc_out["q"])),
        "enc_l1_weight": f32(np.asarray(enc_out["l1_weight"])),
        "enc_upper": f32(np.asarray(enc_out["upper"])),
        "lora_A": f32(np.asarray(enc_out["A"])),
        "lora_B": f32(np.asarray(enc_out["B"])),
        "base_restriction_maps": f32(np.asarray(geometry.restriction_maps)),
        "rho": rho_np,
        "x": f32(np.asarray(history.x)),
        "z": f32(np.asarray(history.z)),
        "y": f32(np.asarray(history.y)),
        "primal_res": f32(np.asarray(history.primal_res)),
        "dual_res": f32(np.asarray(history.dual_res)),
        "consistency": f32(np.asarray(history.consistency_rms)),
        "logits_per_iter": f32(np.asarray(logits_per_iter)),
        "logits_final": f32(logits_final),
        "reassembled_final": f32(reassembled),
        "pred_grid": pred_grid,
    }

    # --- shape assertions against the contract's fixed sizes ---
    N, E, d_v, d_e, r, D = 81, 272, cfg.d_v, cfg.d_e, cfg.lora_rank, cfg.num_directions
    expect = {
        "tokens": (BATCH, H, W), "labels": (BATCH, H, W), "centers": (N, 2),
        "node_positions": (N, 2), "edges": (E, 2), "dir_uv": (E,), "dir_vu": (E,),
        "patches": (N, BATCH, ps, ps, C),
        "enc_h": (N, BATCH, d_v), "enc_q_diag": (N, BATCH, d_v), "enc_q": (N, BATCH, d_v),
        "enc_l1_weight": (N, BATCH, d_v), "enc_upper": (N, BATCH, d_v),
        "lora_A": (N, BATCH, D, d_e, r), "lora_B": (N, BATCH, D, d_v, r),
        "base_restriction_maps": (E, 2, d_e, d_v), "rho": (),
        "x": (K_ITERS, N, BATCH, d_v), "z": (K_ITERS, N, BATCH, d_v),
        "y": (K_ITERS, N, BATCH, d_v),
        "primal_res": (K_ITERS, N, BATCH), "dual_res": (K_ITERS, N, BATCH),
        "consistency": (K_ITERS, BATCH),
        "logits_per_iter": (K_ITERS, N, BATCH, ps, ps, C),
        "logits_final": (N, BATCH, ps, ps, C),
        "reassembled_final": (BATCH, H, W, C), "pred_grid": (BATCH, H, W),
    }
    for name, arrs in (("batch", batch_arrays), ("trace", trace_arrays)):
        for k, a in arrs.items():
            assert a.shape == expect[k], f"{name}.{k}: shape {a.shape} != {expect[k]}"

    np.savez(args.out / "batch.npz", **batch_arrays)
    np.savez(args.out / "trace.npz", **trace_arrays)

    # --- complete manifest.json ---
    def dt(a: np.ndarray) -> str:
        return {"float32": "f32", "int64": "i64"}[str(a.dtype)]

    manifest = json.loads((args.out / "manifest.json").read_text())
    manifest["batch.npz"] = {
        k: {"shape": list(a.shape), "dtype": dt(a)} for k, a in batch_arrays.items()
    }
    manifest["trace.npz"] = {
        k: {"shape": list(a.shape), "dtype": dt(a)} for k, a in trace_arrays.items()
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- verify the files load back and match the manifest ---
    for fname, section in (("batch.npz", "batch.npz"), ("trace.npz", "trace.npz")):
        loaded = np.load(args.out / fname)
        assert sorted(loaded.files) == sorted(manifest[section]), f"{fname} key mismatch"
        for k in loaded.files:
            spec = manifest[section][k]
            a = loaded[k]
            assert list(a.shape) == spec["shape"], f"{fname}:{k} shape"
            assert dt(a) == spec["dtype"], f"{fname}:{k} dtype"

    print(f"[goldens] batch.npz: {len(batch_arrays)} arrays; trace.npz: {len(trace_arrays)} arrays")
    print(f"[goldens] consistency[0]={trace_arrays['consistency'][0]}, "
          f"[K-1]={trace_arrays['consistency'][-1]}")
    print(f"[goldens] pred_grid tokens: {np.unique(pred_grid)}")
    print(f"[goldens] wrote {args.out}/batch.npz, trace.npz; manifest.json completed")


if __name__ == "__main__":
    main()
