#!/usr/bin/env python
"""Export maze_sheaf weights + config for the Rust port (goldens/CONTRACT.md).

Run from the upstream sheaf-admm checkout (uv env):

    uv run python /rjs/AI/sheaf-rs/tools/export_weights.py \
        --repo /Users/rjs/.claude/jobs/707f1939/tmp/sheaf-admm \
        --out  /rjs/AI/sheaf-rs/goldens/maze

Two modes:

* default (no ``--checkpoint``): the model is *initialized* (jax.random seed 0,
  mirroring ``create_train_state``: ``init_rng, dropout_rng =
  split(PRNGKey(0))``) and both the ``params`` and ``ema_params`` trees are
  dumped (EMA at init is a copy of params, exactly as ``init_ema`` produces).
* ``--checkpoint path/to/checkpoint.pkl``: the TRAINED ``params`` and
  ``ema_params`` pytrees are loaded from the pickle written by
  ``scripts/train.py`` (``{"params", "ema_params", "config"}``; each tree is
  the Flax variables dict, i.e. wrapped in a top-level ``params`` collection
  key that is stripped here). The pickled Hydra config's ``model`` block is
  asserted equal to the experiment yaml's so the goldens can't silently mix
  architectures. ``config.json`` gains ``"trained": true`` and a
  ``"training"`` provenance block (dataset, epochs, final metrics from the
  sibling ``history.json``).

In both modes the offset-softplus scalar ``rho`` is baked to its plain value
in config.json (PLAN.md section 3.5): ``softplus(rho_raw +
inverse_softplus(rho_init))`` evaluated on the **EMA** ``rho_raw`` — NOT
assumed to be rho_init, since ``rho_learnable`` defaults true and training
moves it. (``eta_raw`` is the only other offset-softplus scalar in
``SheafADMMModel`` and exists only for the GD z-solver, which no shipped
config uses — the maze checkpoint contains exactly one scalar, ``rho_raw``,
and the key manifest would fail loudly if that changed. ``l1_weight`` /
``upper`` / ``q_diag`` are input-dependent softplus heads, computed in Rust.)
The raw ``rho_raw`` stays in the safetensors for completeness but Rust
ignores it.

Writes into --out:
    config.json          model + task + baked scalars (contract shape)
    weights.safetensors  params/... and ema_params/... , '/'-joined Flax paths
    manifest.json        partial: generator + safetensors sections
                         (dump_goldens.py completes batch.npz / trace.npz)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


# The exhaustive per-collection key list from goldens/CONTRACT.md. Any
# missing or extra key is an export failure (upstream renames must fail loudly).
EXPECTED_KEYS = [
    "ConcatMLPDecoderV2_0/dense/bias",
    "ConcatMLPDecoderV2_0/dense/kernel",
    "ConcatMLPDecoderV2_0/input_norm/scale",
    "ConcatMLPDecoderV2_0/output_dense/bias",
    "ConcatMLPDecoderV2_0/output_dense/kernel",
    "MLPEncoderV2_0/comm_head/comm_dense/bias",
    "MLPEncoderV2_0/comm_head/comm_dense/kernel",
    "MLPEncoderV2_0/comm_head/comm_norm/bias",
    "MLPEncoderV2_0/comm_head/comm_norm/scale",
    "MLPEncoderV2_0/dense/bias",
    "MLPEncoderV2_0/dense/kernel",
    "MLPEncoderV2_0/input_norm/scale",
    "MLPEncoderV2_0/lora_A_dense/bias",
    "MLPEncoderV2_0/lora_A_dense/kernel",
    "MLPEncoderV2_0/lora_B_dense/bias",
    "MLPEncoderV2_0/lora_B_dense/kernel",
    "MLPEncoderV2_0/lora_pre_ln/bias",
    "MLPEncoderV2_0/lora_pre_ln/scale",
    "MLPEncoderV2_0/objective_heads/l1_weight_dense/bias",
    "MLPEncoderV2_0/objective_heads/l1_weight_dense/kernel",
    "MLPEncoderV2_0/objective_heads/q_dense/bias",
    "MLPEncoderV2_0/objective_heads/q_dense/kernel",
    "MLPEncoderV2_0/objective_heads/q_diag_dense/bias",
    "MLPEncoderV2_0/objective_heads/q_diag_dense/kernel",
    "MLPEncoderV2_0/objective_heads/upper_bound_dense/bias",
    "MLPEncoderV2_0/objective_heads/upper_bound_dense/kernel",
    "rho_raw",
    "rm/R_E",
    "rm/R_N",
    "rm/R_NE",
    "rm/R_NW",
    "rm/R_S",
    "rm/R_SE",
    "rm/R_SW",
    "rm/R_W",
]
EXPECTED_PARAM_COUNT = 181_859


def flatten_params(tree) -> dict[str, np.ndarray]:
    """Flatten a Flax param pytree to {'/'-joined path: f32 ndarray}."""
    import jax

    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    out = {}
    for path, leaf in flat:
        keys = []
        for p in path:
            if hasattr(p, "key"):
                keys.append(str(p.key))
            elif hasattr(p, "name"):
                keys.append(str(p.name))
            else:
                keys.append(str(p))
        arr = np.asarray(leaf)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        out["/".join(keys)] = arr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="upstream sheaf-admm checkout")
    ap.add_argument("--out", type=Path, required=True, help="goldens/maze output dir")
    ap.add_argument("--seed", type=int, default=0,
                    help="init seed (ignored when --checkpoint is given)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="trained checkpoint.pkl from scripts/train.py; "
                         "loads its params/ema_params instead of seed-init")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo / "src"))

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import sheaf_admm  # noqa: F401,E402  pins fp32 matmul precision on import
    from safetensors.numpy import save_file  # noqa: E402
    from sheaf_admm.admm import inverse_softplus  # noqa: E402
    from sheaf_admm.data import views as V  # noqa: E402
    from sheaf_admm.models import model_config_from_dict  # noqa: E402
    from sheaf_admm.models.sheaf_model import SheafADMMModel  # noqa: E402

    exp = yaml.safe_load((args.repo / "configs/experiment/maze_sheaf.yaml").read_text())
    model_dict = exp["model"]
    task_cfg = exp["task_cfg"]
    cfg = model_config_from_dict(model_dict)
    model = SheafADMMModel(config=cfg)

    # --- a shape-correct sample batch for init (values are irrelevant to init) ---
    H = W = 19
    B = 2
    centers = V.grid_agent_centers((H, W), task_cfg["stride"], task_cfg["patch_size"])
    edges = jnp.asarray(
        V.build_grid_edge_indices(centers, task_cfg["stride"], task_cfg["connectivity"])
    )
    npos = V.node_positions(centers)
    tokens = jnp.ones((B, H, W), dtype=jnp.int32)  # all-wall dummy maze
    patches = V.prepare_maze_patches(
        tokens, centers, task_cfg["patch_size"], task_cfg["num_classes"]
    )

    training_provenance = None
    if args.checkpoint is not None:
        # --- trained pytrees from scripts/train.py's checkpoint.pkl ---
        import pickle

        with open(args.checkpoint, "rb") as f:
            ckpt = pickle.load(f)

        def unwrap(tree):
            # train.py pickles state.params, the Flax *variables* dict
            # ({"params": {...}}); strip the collection key if present.
            if isinstance(tree, dict) and set(tree) == {"params"}:
                return tree["params"]
            return tree

        params = unwrap(ckpt["params"])
        ema_params = unwrap(ckpt["ema_params"])

        # The checkpoint must be the same architecture the yaml describes.
        ckpt_model = dict(ckpt["config"]["model"])
        if ckpt_model != dict(model_dict):
            diff = {
                k: (model_dict.get(k), ckpt_model.get(k))
                for k in set(model_dict) | set(ckpt_model)
                if model_dict.get(k) != ckpt_model.get(k)
            }
            raise SystemExit(f"checkpoint model config != {args.repo}/configs yaml: {diff}")

        # Provenance: dataset/epochs from the pickled Hydra config, final
        # metrics from the sibling history.json written by the same run.
        tr = ckpt["config"]["training"]
        history_path = args.checkpoint.parent / "history.json"
        final_metrics = None
        if history_path.exists():
            history = json.loads(history_path.read_text())
            if history:
                final_metrics = {
                    k: v for k, v in history[-1].items() if k not in ("epoch", "loss")
                }
        training_provenance = {
            "checkpoint": args.checkpoint.name,
            "dataset": ckpt["config"]["data"]["dir"],
            "epochs": tr["epochs"],
            "seed": tr["seed"],
            "K_train": tr["K_train"],
            "K_eval": tr["K_eval"],
            "ema_decay": tr["ema_decay"],
            "final_metrics": final_metrics,
        }
    else:
        # --- init exactly as training.loop.create_train_state does (seed 0) ---
        init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(args.seed))
        variables = model.init(
            {"params": init_rng, "dropout": dropout_rng},
            patches,
            edges,
            num_iters=2,
            loss_window=1,
            node_positions=npos,
            training=False,
        )
        params = variables["params"]
        # EMA at init: a copy of the params (training.optim.init_ema).
        ema_params = jax.tree_util.tree_map(jnp.copy, params)

    flat = flatten_params(params)
    flat_ema = flatten_params(ema_params)
    got = sorted(flat)
    if got != sorted(EXPECTED_KEYS):
        missing = sorted(set(EXPECTED_KEYS) - set(got))
        extra = sorted(set(got) - set(EXPECTED_KEYS))
        raise SystemExit(f"param key mismatch: missing={missing} extra={extra}")
    n_params = sum(int(a.size) for a in flat.values())
    if n_params != EXPECTED_PARAM_COUNT:
        raise SystemExit(f"param count {n_params} != expected {EXPECTED_PARAM_COUNT}")

    tensors = {f"params/{k}": v for k, v in flat.items()}
    tensors.update({f"ema_params/{k}": v for k, v in flat_ema.items()})

    # --- bake the offset-softplus rho on the EMA tree (PLAN.md section 3.5) ---
    rho_raw = jnp.asarray(flat_ema["rho_raw"], dtype=jnp.float32)
    rho_baked = np.float32(jax.nn.softplus(rho_raw + inverse_softplus(cfg.rho_init)))

    config_json = {
        "model": {
            "num_classes": cfg.num_classes,
            "d_v": cfg.d_v,
            "d_e": cfg.d_e,
            "encoder_arch": cfg.encoder_arch,
            "enc_hidden_dim": cfg.enc_hidden_dim,
            "comm_norm_type": cfg.comm_norm_type,
            "objective_mode": cfg.objective_mode,
            "x_solver": cfg.x_solver,
            "z_solver": cfg.z_solver,
            "z_mode": cfg.z_mode,
            "gamma": cfg.gamma,
            "cg_iters": cfg.cg_iters,
            "tikhonov_eps": cfg.tikhonov_eps,
            "prox_init": cfg.prox_init,
            "rm_sharing": cfg.rm_sharing,
            "rm_init": cfg.rm_init,
            "rm_mode": cfg.rm_mode,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_use_gate": cfg.lora_use_gate,
            "lora_init_style": cfg.lora_init_style,
            "num_directions": cfg.num_directions,
            "relaxation_alpha": cfg.relaxation_alpha,
            "z_init": cfg.z_init,
            "q_epsilon": cfg.q_epsilon,
            "l1_init": cfg.l1_init,
            "upper_init": cfg.upper_init,
            "decoder_arch": cfg.decoder_arch,
            "dec_hidden_dim": cfg.dec_hidden_dim,
        },
        "task": {
            "task": exp["task"],
            "patch_size": task_cfg["patch_size"],
            "stride": task_cfg["stride"],
            "connectivity": task_cfg["connectivity"],
            "num_classes": task_cfg["num_classes"],
            "k_eval": exp["training"]["K_eval"],
            "loss_window": exp["training"]["loss_window"],
        },
        "baked": {"rho": float(rho_baked)},
    }
    if training_provenance is not None:
        config_json["trained"] = True
        config_json["training"] = training_provenance

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "weights.safetensors"))
    (args.out / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")

    # --- partial manifest (generator + safetensors); dump_goldens.py completes it ---
    try:
        sha = subprocess.run(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    manifest = {
        "config": "maze_sheaf",
        "generator": {
            "upstream_commit": sha,
            "seed": None if args.checkpoint is not None else args.seed,
            "trained": args.checkpoint is not None,
            "checkpoint": args.checkpoint.name if args.checkpoint is not None else None,
            "B": 2,
            "K": 12,
            "N": int(centers.shape[0]),
            "E": int(edges.shape[0]),
            "weights": "ema_params",
        },
        "safetensors": {
            k: {"shape": list(v.shape), "dtype": "F32"} for k, v in sorted(tensors.items())
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- verify round trip ---
    from safetensors.numpy import load_file

    back = load_file(str(args.out / "weights.safetensors"))
    assert sorted(back) == sorted(tensors), "safetensors round-trip key mismatch"
    for k, v in tensors.items():
        assert back[k].dtype == np.float32 and back[k].shape == v.shape
        assert np.array_equal(back[k], v), f"round-trip value mismatch at {k}"

    print(f"[export] {len(tensors)} tensors ({n_params:,} params x2), rho={rho_baked!r}")
    print(f"[export] wrote {args.out}/weights.safetensors, config.json, manifest.json")


if __name__ == "__main__":
    main()
