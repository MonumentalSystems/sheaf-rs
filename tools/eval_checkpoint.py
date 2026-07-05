"""Evaluate a trained checkpoint (EMA weights) on chosen splits at a given K.

Standalone finale helper for the sheaf-rs port: used to (a) pick the best maze
seed and (b) read off the paper OOD metrics for the README / web footer, without
re-running training. Mirrors scripts/train.py's eval path exactly (same
create_train_state -> evaluate on ema_params), but loads params/ema_params from
a checkpoint.pkl and lets you cap batches so the 73x73 OOD splits stay within
the shared box's memory.

    uv run --no-sync python tools/eval_checkpoint.py \
        --repo /home/ms/sheaf-admm \
        --checkpoint /home/ms/sheaf-admm/outputs/maze_seed42/checkpoint.pkl \
        --dataset /home/ms/sheaf-admm/datasets/maze_std3_19px_10k \
        --splits test,test_ood_2x,test_ood_2xW,test_ood_4x,test_ood_4xW \
        --k 100 --batch 64
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


def _puzzle_batch(batch, np):
    out = {"inputs": np.asarray(batch["inputs"]), "labels": np.asarray(batch["labels"])}
    for key in ("height", "width"):
        if key in batch:
            out[key] = batch[key]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=str, default="test")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches per split (memory/time); None = full split")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="optional: write {split: metrics} JSON here")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo / "src"))
    import numpy as np  # noqa: E402
    import sheaf_admm as _sa  # noqa: F401,E402  pins fp32 precision on import
    from sheaf_admm.data import ImageDataset, PuzzleDataset  # noqa: E402
    from sheaf_admm.models import model_config_from_dict  # noqa: E402
    from sheaf_admm.training import (  # noqa: E402
        build_model, create_train_state, evaluate, make_task,
    )

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    t = cfg["training"]
    model_type = cfg["model_type"]

    task = make_task(cfg["task"], **cfg.get("task_cfg", {}))
    model_cfg = model_config_from_dict(cfg["model"])
    model = build_model(model_cfg, model_type)
    graph_readout = model_cfg.mpnn_graph_readout
    loader = cfg["data"]["loader"]

    def batches_for(split):
        if loader == "puzzle":
            ds = PuzzleDataset(args.dataset, split)
            for _set, b in ds.iter_test_batches(args.batch):
                yield _puzzle_batch(b, np)
        else:
            ds = ImageDataset(args.dataset, split)
            for b in ds.iter_batches(args.batch, shuffle=False):
                yield {"images": np.asarray(b["images"]), "labels": np.asarray(b["labels"])}

    # a shape-correct sample batch to init the state skeleton (values irrelevant)
    sample_fwd, _, _ = task.prepare(next(batches_for(args.splits.split(",")[0])))
    state = create_train_state(
        model, sample_fwd, model_type=model_type,
        lr=t["lr"], weight_decay=t["weight_decay"], warmup_steps=t["warmup_steps"],
        grad_clip=t["grad_clip"], ema_decay=t["ema_decay"], k_init=t["K_train"],
        loss_window=t["loss_window"], seed=t["seed"],
    )
    # swap in the trained pytrees; evaluate() reads ema_params by default
    state = state.replace(params=ckpt["params"], ema_params=ckpt["ema_params"])

    results = {}
    for split in args.splits.split(","):
        m = evaluate(state, task, batches_for(split), model_type=model_type,
                     graph_readout=graph_readout, k_eval=args.k,
                     max_batches=args.max_batches)
        results[split] = {k: float(v) for k, v in m.items()}
        print(f"{split:>18}: " + "  ".join(f"{k}={v * 100:.2f}%" for k, v in m.items()))

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
