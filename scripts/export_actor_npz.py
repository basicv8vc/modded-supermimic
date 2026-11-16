#!/usr/bin/env python
"""Export an rsl_rl actor to the evaluator's canonical ``.npz`` checkpoint.

Run this **on the training machine** (needs torch + the mjlab env), then copy the
resulting ``.npz`` to the evaluation host:

    python scripts/export_actor_npz.py \
        --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt \
        --out checkpoints/g1_tracking_model_30000.npz \
        --mjlab-commit 0fb8a681136be94ffc636a3dd423cabb97d91f10

The exported file is a plain weight dump: ``w0/b0 ... wn/bn`` plus the
observation-normalizer statistics and a ``meta_json`` block that the evaluator
checks against its own joint order and action scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="rsl_rl model_<it>.pt")
    parser.add_argument("--out", required=True, help="output .npz")
    parser.add_argument("--mjlab-commit", default="0fb8a681136be94ffc636a3dd423cabb97d91f10")
    parser.add_argument("--run", default="", help="run id / wandb path for provenance")
    parser.add_argument("--iteration", type=int, default=-1)
    args = parser.parse_args(argv)

    import torch  # noqa: PLC0415

    from benchmark.evaluator import constants as C  # noqa: PLC0415
    from benchmark.evaluator.spec import build_model  # noqa: PLC0415

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    prefixes = {
        key.split(".")[0]
        for key in state
        if key.endswith(".0.weight")
    }
    if "actor" in prefixes:
        prefix = "actor"
    elif len(prefixes) == 1:
        prefix = prefixes.pop()
    else:
        raise SystemExit(f"cannot pick the actor prefix from {sorted(prefixes)}")

    layers = []
    index = 0
    while f"{prefix}.{2 * index}.weight" in state:
        layers.append(
            (
                state[f"{prefix}.{2 * index}.weight"].detach().cpu().numpy().astype(np.float32),
                state[f"{prefix}.{2 * index}.bias"].detach().cpu().numpy().astype(np.float32),
            )
        )
        index += 1
    if not layers:
        raise SystemExit("no dense layers found")

    payload = {}
    for i, (weight, bias) in enumerate(layers):
        payload[f"w{i}"] = weight
        payload[f"b{i}"] = bias

    for mean_key in ("obs_normalizer._mean", "obs_normalizer.mean"):
        if mean_key in state:
            payload["obs_mean"] = state[mean_key].detach().cpu().numpy().astype(np.float32)
            break
    for var_key in ("obs_normalizer._var", "obs_normalizer.var"):
        if var_key in state:
            payload["obs_var"] = state[var_key].detach().cpu().numpy().astype(np.float32)
            break

    info = build_model()
    meta = {
        "joint_names": list(C.JOINT_ORDER),
        "action_scale": [float(x) for x in info.action_scale],
        "obs_dim": int(layers[0][0].shape[1]),
        "action_dim": int(layers[-1][0].shape[0]),
        "hidden_dims": [int(w.shape[0]) for w, _ in layers[:-1]],
        "activation": C.ACTOR_ACTIVATION,
        "mjlab_commit": args.mjlab_commit,
        "source": {
            "checkpoint": str(args.checkpoint),
            "run": args.run,
            "iteration": args.iteration,
        },
    }
    payload["meta_json"] = np.array(json.dumps(meta))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(f"wrote {out}")
    print(json.dumps({k: v for k, v in meta.items() if k != "joint_names"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
