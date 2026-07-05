#!/usr/bin/env python
"""Convert goldens/maze/weights.safetensors to the wasm-embedded f16 file.

Keeps ONLY the `ema_params/` collection (paper eval uses the EMA shadow;
the wasm session loads nothing else), casts f32 -> f16, and writes
crates/sheaf-web/assets/weights_ema_f16.safetensors.

Run inside the Python reference env (safetensors + numpy):
    uv run --directory <sheaf-admm checkout> \
        python /rjs/AI/sheaf-rs/tools/convert_f16.py
"""

import os

import numpy as np
from safetensors.numpy import load_file, save_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "goldens", "maze", "weights.safetensors")
DST = os.path.join(REPO, "crates", "sheaf-web", "assets", "weights_ema_f16.safetensors")


def main() -> None:
    tensors = load_file(SRC)
    ema = {k: v for k, v in tensors.items() if k.startswith("ema_params/")}
    if not ema:
        raise SystemExit(f"no ema_params/ keys in {SRC}")
    dropped = len(tensors) - len(ema)

    out = {}
    for k, v in ema.items():
        if v.dtype != np.float32:
            raise SystemExit(f"{k}: expected float32, got {v.dtype}")
        out[k] = v.astype(np.float16)

    n_params = sum(v.size for v in out.values())
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file(out, DST)
    size = os.path.getsize(DST)
    print(f"kept {len(out)} ema_params tensors ({n_params} params), dropped {dropped} others")
    print(f"wrote {DST}: {size} bytes ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
